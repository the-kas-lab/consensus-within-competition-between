"""Strategy weights: each blue sets the bid formula's
preference coefficients within declared ranges, instead of choosing one of the
fixed profiles.

Bid version "v2.3" is used unchanged for every FEATURE: urgency, opportunity,
breach forecast, station coverage, information, continuity relief, belief
age, travel distances, the leaving and switching masks. The strategy fixes
none of them. Only the coefficients that
multiply them change:

    DEFEND:k     = w_station + w_coverage*m_k - w_travel*d/L - penalty + bonus
    RECON:r      = w_scout*I_r - w_scout_travel*d/L - penalty + bonus
    NEUTRALIZE:c = points*p_c*(w_threat + w_engage*u*o) - w_travel*d/L
                   - w_attrition/(1+n) - w_belief_age*age - penalty + bonus
    penalty      = w_leave*leave*m_post + w_switch*switch + w_continuity*cont

**The default is exact, not approximate.** Each bid is computed as the v2.3
balanced bid (``bid_terms_v2`` with the default modifiers, untouched) plus
one delta per coefficient, ``(w - w_default) * feature``. A coefficient at its
default contributes exactly 0.0, so the all-default answer reproduces the
historical bid bit for bit, and the auction's tie-breaks with it.

**Budgets.** ``w_threat + w_engage +
w_station + w_coverage + w_scout <= 3.2`` (the default total), and the
priorities of the locally eligible contacts sum to at most their number.
Sums are taken in decimal (``Decimal(repr(x))``), so the default total 3.2 is
exactly within budget. A violated budget resets that block to its defaults;
nothing is clamped or renormalised.

**Validation drops, it never repairs.** A missing, non-numeric, boolean,
non-finite or out-of-range coefficient falls back to its default; a priority
for a contact the agent does not believe in, an invalid or duplicate one, or
an unlisted / excess preferred task is dropped. Every drop is an event.
"""

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from coordination.bidding import PREFERRED_BONUS, V22_PROFILES, BidModifiers, bid_terms_v2
from coordination.tasks import DEFEND, NEUTRALIZE, RECON

STRATEGY_SPEC_VERSION = "sw.2"
# A violated budget block is scaled down in proportion to the budget.
# Modifier sources that answer with StrategyWeights, and the bid version each
# one states (CBAARuntime gives them the default terms and checks the version).
STRATEGY_SOURCES = {"m8b.9": "v2.3"}
MAX_PREFERRED = 3
PRIORITY_RANGE = (0.0, 2.0)

_B = V22_PROFILES["balanced"]


@dataclass(frozen=True)
class WeightField:
    name: str
    default: float
    lo: float
    hi: float
    term: str  # what it multiplies, stated to the model (objective text)


# One source for the scorer, the validator, the prompt table, the schema and
# the execution identity. Defaults are read from the v2.3 balanced profile, so
# they cannot drift from it.
WEIGHT_FIELDS: Tuple[WeightField, ...] = (
    WeightField("w_threat", _B.threat_value, 0.0, 3.0,
                "NEUTRALIZE: value every believed red carries, before priority"),
    WeightField("w_engage", 1.0, 0.0, 3.0,
                "NEUTRALIZE: x urgency x opportunity"),
    WeightField("w_station", _B.station_base, 0.0, 1.5,
                "DEFEND: value of holding any station"),
    WeightField("w_coverage", _B.w_cov, 0.0, 3.0,
                "DEFEND: x the station's coverage"),
    WeightField("w_scout", _B.w_info, 0.0, 3.0,
                "RECON: x information"),
    WeightField("w_attrition", _B.w_att, 0.0, 2.0,
                "NEUTRALIZE: cost x 1/(1 + teammates), because contact disables you"),
    WeightField("w_belief_age", _B.w_unc, 0.0, 1.0,
                "NEUTRALIZE: cost x the belief-age term"),
    WeightField("w_travel", _B.w_eta, 0.0, 0.6,
                "NEUTRALIZE and DEFEND: cost per 25 units of distance"),
    WeightField("w_scout_travel", _B.w_eta_recon, 0.0, 0.3,
                "RECON: cost per 25 units of distance"),
    WeightField("w_leave", _B.w_leave, 0.0, 1.5,
                "any task that leaves your current station: cost x that station's coverage"),
    WeightField("w_continuity", _B.pursuit_continuity, 0.0, 1.0,
                "any task other than the contact you pursue: cost x the continuity factor"),
    WeightField("w_switch", _B.switch_cost, 0.0, 0.5,
                "any task other than your current task: cost"),
)
FIELD = {f.name: f for f in WEIGHT_FIELDS}
DEFAULT_WEIGHTS: Dict[str, float] = {f.name: float(f.default) for f in WEIGHT_FIELDS}
BUDGET_FIELDS = ("w_threat", "w_engage", "w_station", "w_coverage", "w_scout")
VALUE_BUDGET = Decimal("3.2")
assert sum(Decimal(repr(DEFAULT_WEIGHTS[k])) for k in BUDGET_FIELDS) == VALUE_BUDGET


def _dec(x) -> Decimal:
    return Decimal(repr(float(x)))


@dataclass(frozen=True)
class StrategyWeights:
    """One blue's validated strategy for one epoch. ``priorities`` holds only
    contacts given a non-default value; every other eligible contact is 1."""

    weights: Tuple[Tuple[str, float], ...] = tuple(sorted(DEFAULT_WEIGHTS.items()))
    priorities: Tuple[Tuple[int, float], ...] = ()
    preferred_tasks: Tuple[str, ...] = ()
    source: str = "default"  # "llm", "rule:<policy>", "default"

    @staticmethod
    def build(weights=None, priorities=None, preferred_tasks=None, source="default"):
        w = dict(DEFAULT_WEIGHTS)
        w.update({k: float(v) for k, v in (weights or {}).items()})
        if set(w) != set(DEFAULT_WEIGHTS):
            raise ValueError(f"unknown weight(s) {sorted(set(w) - set(DEFAULT_WEIGHTS))}")
        for k, v in w.items():
            f = FIELD[k]
            if not (math.isfinite(v) and f.lo <= v <= f.hi):
                raise ValueError(f"{k}={v!r} outside [{f.lo}, {f.hi}]")
        if sum(_dec(w[k]) for k in BUDGET_FIELDS) > VALUE_BUDGET:
            raise ValueError("value-weight budget exceeded")
        p = {int(c): float(v) for c, v in (priorities or {}).items()}
        for c, v in p.items():
            if not (math.isfinite(v) and PRIORITY_RANGE[0] <= v <= PRIORITY_RANGE[1]):
                raise ValueError(f"priority for contact {c} = {v!r} outside {PRIORITY_RANGE}")
        return StrategyWeights(weights=tuple(sorted(w.items())),
                               priorities=tuple(sorted(p.items())),
                               preferred_tasks=tuple(sorted(str(t) for t in (preferred_tasks or ()))),
                               source=source)

    @property
    def w(self) -> Dict[str, float]:
        return dict(self.weights)

    def priority(self, contact_id) -> float:
        return dict(self.priorities).get(int(contact_id), 1.0)

    def is_default(self) -> bool:
        return (self.w == DEFAULT_WEIGHTS and all(v == 1.0 for _, v in self.priorities)
                and not self.preferred_tasks)

    def as_dict(self) -> Dict:
        return {"weights": self.w, "priorities": [[c, v] for c, v in self.priorities],
                "preferred_tasks": list(self.preferred_tasks), "source": self.source}


def features(terms: Mapping, kind: str) -> Dict[str, float]:
    """The unweighted features of one task, recovered from its v2.3 balanced
    terms (each weighted term divided by its nonzero default weight)."""
    f = {"leave": float(terms.get("leave", 0.0)) / _B.w_leave,
         "switch": 1.0 if terms.get("switch", 0.0) else 0.0,
         "continuity": float(terms.get("continuity", 0.0)) / _B.pursuit_continuity}
    if kind == DEFEND:
        f.update(coverage=float(terms["coverage"]), dist=float(terms["travel"]) / _B.w_eta)
    elif kind == RECON:
        f.update(information=float(terms["information"]),
                 dist=float(terms["travel"]) / _B.w_eta_recon)
    elif kind == NEUTRALIZE:
        f.update(urgency=float(terms["urgency"]), opportunity=float(terms["opportunity"]),
                 dist=float(terms["travel"]) / _B.w_eta,
                 attrition=float(terms["attrition"]) / _B.w_att,
                 age=float(terms["uncertainty"]) / _B.w_unc)
    else:
        raise ValueError(f"strategy weights are defined for the three-task model, not {kind}")
    return f


def _kind(task_id: str) -> str:
    return task_id.split(":", 1)[0]


def strategy_bid(terms: Mapping, task_id: str, sw: StrategyWeights, points: float) -> float:
    """One task's bid under ``sw``: the v2.3 balanced bid plus exact deltas."""
    kind = _kind(task_id)
    f = features(terms, kind)
    w = sw.w
    d = DEFAULT_WEIGHTS
    delta = (-(w["w_leave"] - d["w_leave"]) * f["leave"]
             - (w["w_switch"] - d["w_switch"]) * f["switch"]
             - (w["w_continuity"] - d["w_continuity"]) * f["continuity"])
    if kind == DEFEND:
        delta += ((w["w_station"] - d["w_station"]) + (w["w_coverage"] - d["w_coverage"]) * f["coverage"]
                  - (w["w_travel"] - d["w_travel"]) * f["dist"])
    elif kind == RECON:
        delta += ((w["w_scout"] - d["w_scout"]) * f["information"]
                  - (w["w_scout_travel"] - d["w_scout_travel"]) * f["dist"])
    else:
        p = sw.priority(int(task_id.split(":")[1]))
        # Same operand order as _neutralize_v22's value, so the default is exact.
        value = points * (p * (w["w_threat"] + w["w_engage"] * f["urgency"] * f["opportunity"]))
        delta += ((value - float(terms["value"]))
                  - (w["w_travel"] - d["w_travel"]) * f["dist"]
                  - (w["w_attrition"] - d["w_attrition"]) * f["attrition"]
                  - (w["w_belief_age"] - d["w_belief_age"]) * f["age"])
    b = float(terms["bid"]) + delta
    if task_id in sw.preferred_tasks:
        b += PREFERRED_BONUS
    if not math.isfinite(b):
        raise ValueError(f"non-finite strategy bid for {task_id}: {b!r}")
    return b


def strategy_terms(view, tasks: Sequence, ctx, current_task_id, sw: StrategyWeights) -> Dict[str, Dict]:
    """``bid_terms_v2`` under the default modifiers (the shared features), with
    each task's ``bid`` replaced by its strategy bid; the default bid is kept
    as ``bid_default``."""
    if ctx.v2 is None or ctx.v2.version != "v2.3":
        raise ValueError("strategy weights are defined on bid version v2.3 only")
    terms = bid_terms_v2(view, tasks, ctx, current_task_id, BidModifiers())
    out = {}
    for t in tasks:
        x = dict(terms[t.task_id])
        x["bid_default"] = x["bid"]
        x["bid"] = strategy_bid(terms[t.task_id], t.task_id, sw, ctx.neutralize_points)
        out[t.task_id] = x
    return out


# -- validation ---------------------------------------------------------------

def _is_number(x) -> bool:
    return isinstance(x, (int, float, Decimal)) and not isinstance(x, bool) and math.isfinite(float(x))


def _scale_down(values: Dict, budget: Decimal) -> Dict:
    """Scale ``values`` by budget / sum, floored to 1e-6 so the decimal sum
    stays within the budget."""
    total = sum(_dec(v) for v in values.values())
    f = budget / total
    return {k: math.floor(float(_dec(v) * f) * 1e6) / 1e6 for k, v in values.items()}


def validate_answer(answer: Mapping, believed: Sequence[int], listed: Sequence[str],
                    source: str = "llm") -> Tuple[StrategyWeights, List[Dict]]:
    """A parsed answer ``{"weights": {...}, "contact_priorities": [...],
    "preferred_tasks": [...]}`` -> the strategy applied and the events. Never
    raises on content; a structurally unusable answer is the caller's total
    failure."""
    events: List[Dict] = []
    raw = answer.get("weights")
    raw = raw if isinstance(raw, Mapping) else {}
    if not isinstance(answer.get("weights"), Mapping):
        events.append({"kind": "weights_missing", "applied": "defaults"})
    w = dict(DEFAULT_WEIGHTS)
    for f in WEIGHT_FIELDS:
        if f.name not in raw:
            events.append({"kind": "weight_missing", "field": f.name, "applied": f.default})
            continue
        v = raw[f.name]
        if not _is_number(v) or not (f.lo <= float(v) <= f.hi):
            events.append({"kind": "weight_invalid", "field": f.name, "value": repr(v),
                           "applied": f.default})
            continue
        w[f.name] = float(v)
    extra = sorted(set(raw) - set(FIELD))
    if extra:
        events.append({"kind": "ignored_weight_fields", "value": extra})
    if sum(_dec(w[k]) for k in BUDGET_FIELDS) > VALUE_BUDGET:
        raw_block = {k: w[k] for k in BUDGET_FIELDS}
        w.update(_scale_down(raw_block, VALUE_BUDGET))
        events.append({"kind": "value_budget_scaled", "value": raw_block,
                       "applied": {k: w[k] for k in BUDGET_FIELDS}})

    eligible = sorted({int(c) for c in believed})
    pri: Dict[int, float] = {}
    entries = answer.get("contact_priorities")
    for e in entries if isinstance(entries, list) else []:
        cid = e.get("contact_id") if isinstance(e, Mapping) else None
        val = e.get("priority") if isinstance(e, Mapping) else None
        if not isinstance(cid, int) or isinstance(cid, bool) or cid not in eligible:
            events.append({"kind": "unbelieved_contact", "value": repr(cid)})
        elif cid in pri:
            events.append({"kind": "duplicate_priority", "value": cid})
        elif not _is_number(val) or not (PRIORITY_RANGE[0] <= float(val) <= PRIORITY_RANGE[1]):
            events.append({"kind": "invalid_priority", "value": repr(val), "contact_id": cid})
        else:
            pri[cid] = float(val)
    if eligible and sum(_dec(pri.get(c, 1.0)) for c in eligible) > Decimal(len(eligible)):
        full = {c: pri.get(c, 1.0) for c in eligible}
        pri = _scale_down(full, Decimal(len(eligible)))
        events.append({"kind": "priority_budget_scaled", "value": full, "applied": pri})

    preferred: List[str] = []
    prefs = answer.get("preferred_tasks")
    for t in prefs if isinstance(prefs, list) else []:
        if not isinstance(t, str) or t not in listed:
            events.append({"kind": "unlisted_task", "value": repr(t)})
        elif t in preferred:
            continue
        elif len(preferred) >= MAX_PREFERRED:
            events.append({"kind": "too_many_preferred", "value": t})
        else:
            preferred.append(t)
    return StrategyWeights.build(w, {c: v for c, v in pri.items() if v != 1.0}, preferred,
                                 source=source), events


def spec_identity() -> Dict:
    """What the execution identity records about this interface."""
    fallback = "per field to default; violated budget block scaled down to the budget"
    return {"version": STRATEGY_SPEC_VERSION, "bid_version": "v2.3",
            "fields": [[f.name, f.default, f.lo, f.hi] for f in WEIGHT_FIELDS],
            "value_budget": str(VALUE_BUDGET), "budget_fields": list(BUDGET_FIELDS),
            "priority_range": list(PRIORITY_RANGE), "priority_budget": "sum <= eligible count",
            "preferred_bonus": PREFERRED_BONUS, "max_preferred": MAX_PREFERRED,
            "t0_s": _B.t0, "fallback": fallback}
