"""Cumulative billed spend across workers, with in-flight headroom.

Added for the hosted pilot.

Every ATTEMPT -- first try or retry -- reserves its worst-case cost before it
is sent: an upper bound on its prompt tokens from the request's UTF-8 bytes
(rule "bytes_upper", see RESERVATION_RULES), plus the full ``max_tokens`` of
output, at the pinned endpoint's price. A call
whose reservation would take committed spend plus everything in flight past
the cap is refused with ``BudgetExhausted`` (never turned into a planner
fallback: the run stops). When the attempt ends its reservation is settled:

* ``billed``      -- the provider returned the billed cost (OpenRouter usage
                     accounting); that amount is committed;
* ``estimated``   -- token counts but no cost came back; tokens x price;
* ``unverified``  -- the attempt failed after it may have reached a provider
                     (timeout, 5xx, connection error); the worst case stays
                     committed, conservatively, until reconciled against the
                     account;
* ``not_billed``  -- refused before generation (HTTP 429 / 402); zero.

With ``path`` the state is a JSON file under an exclusive file lock, so
threads AND processes (concurrent episodes) share one total. Without it the
ledger is in-memory (tests, one process). The ledger is an accounting guard,
not the bill: reconcile ``totals()`` against the provider's account usage.
"""

import fcntl
import json
import os
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Dict, Optional

from planning.stops import StopRun

BILLED = "billed"
ESTIMATED = "estimated"
UNVERIFIED = "unverified"
NOT_BILLED = "not_billed"
OUTCOMES = (BILLED, ESTIMATED, UNVERIFIED, NOT_BILLED)

# A conservative floor on characters per token for the worst-case prompt
# estimate (English + JSON runs ~3.5-4). Used by the
# historical reservation rule "chars_half" only: typical, NOT a bound.
CHARS_PER_TOKEN_FLOOR = 2.0

# Reservation rules for an attempt's prompt tokens:
#   "bytes_upper" (default) -- at most one token per UTF-8 byte of the request
#       text (byte-level BPE and SentencePiece-with-byte-fallback tokenizers
#       never emit a token covering less than one byte) plus a fixed
#       REQUEST_OVERHEAD_TOKENS for the chat template and schema framing. With
#       completion <= max_tokens (checked after the fact from billed usage),
#       the reservation is an upper bound on the attempt's cost at the pinned
#       price. Declared assumptions, not a provider guarantee.
#   "chars_half" -- the historical rule (size / CHARS_PER_TOKEN_FLOOR + 1).
REQUEST_OVERHEAD_TOKENS = 256
RESERVATION_RULES = ("bytes_upper", "chars_half")


class BudgetExhausted(StopRun):
    """The next attempt's worst case would exceed the cap. Stops the run."""


@dataclass(frozen=True)
class Price:
    """USD per token at the pinned endpoint, and where the numbers came from."""

    prompt_usd_per_token: float
    completion_usd_per_token: float
    source: str = ""


class SpendLedger:
    def __init__(self, cap_usd: float, price: Price, path: Optional[str] = None,
                 reservation_rule: str = "bytes_upper"):
        if reservation_rule not in RESERVATION_RULES:
            raise ValueError(f"unknown reservation rule {reservation_rule!r}")
        self.cap_usd = float(cap_usd)
        self.price = price
        self.path = path
        self.reservation_rule = reservation_rule
        self._lock = threading.Lock()
        self._state = self._empty()
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with self._locked() as state:
                if state.get("cap_usd") not in (None, self.cap_usd):
                    raise ValueError(f"ledger {path} has cap {state['cap_usd']}, not {self.cap_usd}")
                state["cap_usd"] = self.cap_usd
                # The rule in force for the next reservations (recorded).
                state["reservation_rule"] = self.reservation_rule
                # One ledger file may be shared by several pinned endpoints
                # (a multi-candidate study): each records its own price.
                state.setdefault("prices", {})[price.source or "default"] = asdict(price)

    def _empty(self) -> Dict:
        return {"cap_usd": self.cap_usd, "prices": {self.price.source or "default": asdict(self.price)},
                "committed": {o: 0.0 for o in OUTCOMES}, "attempts": {o: 0 for o in OUTCOMES},
                "inflight": {}, "refused_reservations": 0}

    @contextmanager
    def _locked(self):
        with self._lock:
            if not self.path:
                yield self._state
                return
            with open(self.path + ".lock", "a+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                try:
                    try:
                        with open(self.path, "r", encoding="utf-8") as f:
                            state = json.load(f)
                    except (OSError, ValueError):
                        state = self._empty()
                    yield state
                    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(self.path)),
                                               suffix=".tmp")
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(state, f, indent=1)
                    os.replace(tmp, self.path)
                finally:
                    fcntl.flock(lockf, fcntl.LOCK_UN)

    def raise_cap(self, new_cap_usd: float, note: str) -> None:
        """Raise the cap to a newly AUTHORIZED limit, recorded with a note
        (e.g. passing a reconciled checkpoint). Never lowers it below what is
        committed; the change history stays in the file."""
        new_cap_usd = float(new_cap_usd)
        with self._locked() as state:
            if new_cap_usd < sum(state["committed"].values()):
                raise ValueError("new cap is below committed spend")
            state.setdefault("cap_history", []).append(
                {"from": state["cap_usd"], "to": new_cap_usd, "note": str(note)})
            state["cap_usd"] = new_cap_usd
        self.cap_usd = new_cap_usd

    def worst_case_usd(self, prompt_chars: int, max_tokens: int) -> float:
        """``prompt_chars``: the request text's size, measured in UTF-8 bytes
        by the call path (equal to characters for ASCII)."""
        if self.reservation_rule == "bytes_upper":
            prompt_tokens = int(prompt_chars) + REQUEST_OVERHEAD_TOKENS
        else:
            prompt_tokens = int(prompt_chars / CHARS_PER_TOKEN_FLOOR) + 1
        return (prompt_tokens * self.price.prompt_usd_per_token
                + int(max_tokens) * self.price.completion_usd_per_token)

    def cost_of(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (int(prompt_tokens) * self.price.prompt_usd_per_token
                + int(completion_tokens) * self.price.completion_usd_per_token)

    def reserve(self, prompt_chars: int, max_tokens: int) -> str:
        amount = self.worst_case_usd(prompt_chars, max_tokens)
        rid = uuid.uuid4().hex
        with self._locked() as state:
            committed = sum(state["committed"].values())
            inflight = sum(state["inflight"].values())
            if committed + inflight + amount > self.cap_usd:
                state["refused_reservations"] += 1
                raise BudgetExhausted(
                    f"cap ${self.cap_usd:.4f}: committed ${committed:.4f} + in flight "
                    f"${inflight:.4f} + this attempt's worst case ${amount:.6f} exceeds it")
            state["inflight"][rid] = amount
        return rid

    def settle(self, rid: str, outcome: str, billed_usd: Optional[float] = None) -> float:
        """Close a reservation; returns the amount committed for it."""
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}")
        with self._locked() as state:
            reserved = state["inflight"].pop(rid, None)
            if reserved is None:
                raise KeyError(f"no open reservation {rid}")
            if outcome == UNVERIFIED:
                amount = reserved
            elif outcome == NOT_BILLED:
                amount = 0.0
            else:
                if billed_usd is None:
                    raise ValueError(f"outcome {outcome!r} needs an amount")
                amount = float(billed_usd)
            state["committed"][outcome] += amount
            state["attempts"][outcome] += 1
        return amount

    def totals(self) -> Dict:
        with self._locked() as state:
            snap = json.loads(json.dumps(state))
        snap["committed_usd"] = sum(snap["committed"].values())
        snap["inflight_usd"] = sum(snap["inflight"].values())
        snap["n_inflight"] = len(snap["inflight"])
        del snap["inflight"]
        return snap
