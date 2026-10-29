"""Consensus-Based Auction Algorithm, single assignment (Choi, Brunet & How 2009).

 Network-agnostic: a
``CBAAAgent`` produces and consumes ``AuctionState`` values and knows nothing
about how they travel. The tests drive it over an in-memory graph; the
testbed drives the same object over the real ``Network`` as ``AUCTION_MESSAGE``
envelopes. One algorithm, two drivers.

Per agent i: assignment ``x`` (one task or None), known winning bids ``y[t]``,
known winners ``z[t]``.

1. **Select.** If unassigned, take the best task it can outbid.
2. **Consensus.** Per task, keep the best (bid, winner) seen, where "best" is
   higher bid, **ties broken by lower agent id** -- a total order every agent
   evaluates identically, so it is a tie-break and not a race.
3. **Release.** If some other agent now holds ``x``, drop it and go to 1.

With static bids this converges to the same assignment as the centralized
sequential greedy algorithm on any connected graph; the tests check exactly
that, against an independent oracle, on random bid matrices.

**Epochs, per the rule that an auction may span planning epochs (asynchronous, spanning epochs).** An agent resets its
auction at each planning epoch but keeps executing its previous commitment
until its own state has been unchanged for ``stable_rounds`` consecutive rounds
-- locally computable, no agent needs to know global convergence. Messages from
a superseded epoch are ignored here as well as by the transport's
``is_stale_allocation``: this object must be correct under any driver.

**Auction traffic can never create a belief**. An entry for a
task the agent did not generate from its own view is used for nothing -- not for
selection, not for consensus -- and is counted in ``foreign_entries_ignored`` so
the size of the channel is measured rather than assumed away.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

NEG_INF = float("-inf")


def _beats(bid_a, winner_a, bid_b, winner_b) -> bool:
    """Does (bid_a, winner_a) outrank (bid_b, winner_b)? Higher bid wins; equal
    bids go to the lower agent id. An unheld task (winner None) loses to any
    held one at equal bid, which only arises at -inf."""
    if bid_a != bid_b:
        return bid_a > bid_b
    if winner_b is None:
        return winner_a is not None
    if winner_a is None:
        return False
    return winner_a < winner_b


@dataclass(frozen=True)
class AuctionState:
    """What one agent broadcasts: its view of every task's winning bid.

    Frozen, and built by copying, so a receiver can never mutate a sender's
    state through a shared reference -- the double buffer is enforced by the
    type, not by the driver's good behaviour.
    """

    epoch: int
    sender: int
    entries: Tuple[Tuple[str, float, Optional[int]], ...]  # (task, y, z)


class CBAAAgent:
    def __init__(self, agent_id: int, stable_rounds: int):
        self.agent_id = int(agent_id)
        self.stable_rounds = int(stable_rounds)
        self.epoch: Optional[int] = None
        self.bids: Dict[str, float] = {}
        self.order: List[str] = []
        self.y: Dict[str, float] = {}
        self.z: Dict[str, Optional[int]] = {}
        self.x: Optional[str] = None
        self.commitment: Optional[str] = None
        self.rounds_unchanged = 0
        self.stale_messages_ignored = 0
        self.foreign_entries_ignored = 0
        self.rounds_to_stable: Optional[int] = None
        # Commitment changes after this epoch's first local stability: news from
        # far away overturned a result the agent had already acted on. That is
        # assignment churn, one of Exp 4's metrics, and it is counted rather
        # than prevented -- preventing it would mean executing a task another
        # agent holds.
        self.recommitments = 0
        self._rounds_this_epoch = 0

    # -- epoch lifecycle ---------------------------------------------------

    def start_epoch(self, epoch: int, bids: Mapping[str, float], order: Sequence[str]):
        """Begin a new auction. ``bids`` are this agent's own, computed from its
        own view; ``order`` is the canonical task order used to break ties
        between two tasks one agent values equally.

        The previous commitment is **kept**: the agent goes on executing it
        until this auction has settled locally.
        """
        if set(order) != set(bids):
            raise ValueError("order must list exactly the tasks that were bid on")
        self.epoch = int(epoch)
        self.bids = {t: float(b) for t, b in bids.items()}
        self.order = list(order)
        self.y = {t: NEG_INF for t in self.order}
        self.z = {t: None for t in self.order}
        self.x = None
        self.rounds_unchanged = 0
        self.rounds_to_stable = None
        self.recommitments = 0
        self._rounds_this_epoch = 0
        self.select()

    # -- the algorithm -----------------------------------------------------

    def select(self):
        if self.x is not None:
            return
        best = None
        for t in self.order:  # canonical order: first of equal bids wins
            if not _beats(self.bids[t], self.agent_id, self.y[t], self.z[t]):
                continue
            if best is None or self.bids[t] > self.bids[best]:
                best = t
        if best is not None:
            self.x = best
            self.y[best] = self.bids[best]
            self.z[best] = self.agent_id

    def outgoing(self) -> AuctionState:
        return AuctionState(
            epoch=int(self.epoch),
            sender=self.agent_id,
            entries=tuple((t, self.y[t], self.z[t]) for t in self.order),
        )

    def receive(self, messages: Iterable[AuctionState]):
        """Consensus over whatever arrived this round, then re-select.

        Order-independent by construction: per task it keeps a maximum under a
        total order, and max is commutative.
        """
        before = self._signature()
        for msg in messages:
            if msg.epoch != self.epoch:
                # Superseded (or not yet started) epoch: never allowed to touch
                # this epoch's state. Also enforced by the transport; enforced
                # here too so the object is correct under any driver.
                self.stale_messages_ignored += 1
                continue
            for task, bid, winner in msg.entries:
                if task not in self.y:
                    # The rule that auction traffic never creates a belief: a task this agent did not generate from its own
                    # belief. Using it for anything would let the auction tell
                    # a `local`-mode agent that a contact exists.
                    self.foreign_entries_ignored += 1
                    continue
                if _beats(bid, winner, self.y[task], self.z[task]):
                    self.y[task], self.z[task] = float(bid), winner
        if self.x is not None and self.z[self.x] != self.agent_id:
            self.x = None  # outbid: release
        self.select()

        self._rounds_this_epoch += 1
        if self._signature() == before:
            self.rounds_unchanged += 1
        else:
            self.rounds_unchanged = 0
        if self.stable:
            # The rule that an auction may span planning epochs: adopt the auction's result only once it has settled locally --
            # and re-adopt it whenever it settles again. Local stability is a
            # heuristic, not a guarantee: measured, CBAA settled after MORE than
            # diameter rounds in most random trials (release cascades, up to
            # 4x D), so a far agent's bid can land after this agent went quiet.
            # Committing only on the first stable round would leave the agent
            # executing a task someone else now holds.
            if self.rounds_to_stable is None:
                self.rounds_to_stable = self._rounds_this_epoch
            elif self.commitment != self.x:
                self.recommitments += 1
            self.commitment = self.x

    def _signature(self):
        return (self.x, tuple(self.y[t] for t in self.order), tuple(self.z[t] for t in self.order))

    # -- read-outs ---------------------------------------------------------

    @property
    def stable(self) -> bool:
        return self.rounds_unchanged >= self.stable_rounds

    @property
    def assignment(self) -> Optional[str]:
        """What the auction currently says -- which may still change."""
        return self.x


def conflicts(agents: Iterable[CBAAAgent]) -> Dict[str, List[int]]:
    """Tasks claimed by more than one agent. **Logged, never repaired** (plan
    Sec 23): under a partition, components can legitimately disagree, and
    papering over that would hide exactly the regime the paper cares about."""
    claims: Dict[str, List[int]] = {}
    for a in agents:
        if a.x is not None:
            claims.setdefault(a.x, []).append(a.agent_id)
    return {t: sorted(ids) for t, ids in claims.items() if len(ids) > 1}


def commitment_conflicts(agents: Iterable[CBAAAgent]) -> Dict[str, List[int]]:
    """Tasks more than one agent is COMMITTED to -- what is in force, which is
    not what ``conflicts`` samples. Under the rule that an auction may span planning epochs an agent executes its commitment
    and keeps it until its auction settles again, so claims (``x``) and
    commitments differ whenever an epoch has not settled or has changed since:
    The auction-timing check found claims conflicting while nothing was
    committed at a 0.5 s period, and commitments conflicting MORE than claims
    at R_comm = 15. Logged,
    never repaired, as ``conflicts`` is."""
    claims: Dict[str, List[int]] = {}
    for a in agents:
        if a.commitment is not None:
            claims.setdefault(a.commitment, []).append(a.agent_id)
    return {t: sorted(ids) for t, ids in claims.items() if len(ids) > 1}


def run_rounds(agents: Sequence[CBAAAgent], neighbors: Mapping[int, Iterable[int]],
               rounds: int, order=None):
    """In-memory driver: synchronous rounds over a fixed graph.

    **Double-buffered, as requires.** Every agent's outgoing state is
    captured before any agent receives, so no agent can see a same-round update
    from an agent the loop happened to process first. ``order`` exists so the
    tests can prove that, by permuting it.
    """
    by_id = {a.agent_id: a for a in agents}
    sequence = list(order) if order is not None else sorted(by_id)
    for _ in range(int(rounds)):
        outbox = {i: by_id[i].outgoing() for i in sorted(by_id)}
        for i in sequence:
            by_id[i].receive(outbox[j] for j in sorted(neighbors.get(i, ())) if j in outbox)
