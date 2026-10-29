"""The bid weights the consensus team's models wrote, from the cached answers (no simulation, no model call).

For every validated bid-weight answer of the valid episodes: the weights and pursuit multipliers that were
applied (after validation and budget scaling, coordination.strategy_weights.validate_answer), and whether a
budget was scaled. Teams:

  W      consensus team at high authority, low-cost model (all 10 cells, 300 episodes)
  W_ref4 the same team, ideal network with four intruders only (30 episodes; matched to the near-frontier run)
  WSOL   the same team with the near-frontier model, valid records only (30 episodes)

Answers are split by whether the defender believed in any intruder (with contact / before contact). Only
quantities of the answers are reported.

    python3 -m analysis.bid_weights --data DATA [--out OUT]       ->  OUT/bid_weights.json
"""
import glob
import json
import os

import numpy as np

from analysis import common
from coordination.strategy_weights import BUDGET_FIELDS, DEFAULT_WEIGHTS, WEIGHT_FIELDS, validate_answer

slug = common.slug


def cache_dirs(data):
    out = {"W": [], "W_ref4": [], "WSOL": []}
    run = common.HIGH_AUTHORITY_LOW_COST[0]
    base = os.path.join(common.run_dir(data, run), "cache", common.identity(data, run))
    for r in common.records(data, [run]):
        cd = os.path.join(base, slug(r["episode_id"]))
        out["W"].append(cd)
        if r["reds"] == 4 and r["condition"] == "REF" and not r["attacked"]:
            out["W_ref4"].append(cd)
    for run in common.HIGH_AUTHORITY_NEAR_FRONTIER:
        base = os.path.join(common.run_dir(data, run), "cache", common.identity(data, run))
        out["WSOL"] += [os.path.join(base, slug(r["episode_id"]) + "__near_frontier")
                        for r in common.valid_records(data, [run])]
    return out


def answers(dirs):
    rows = []
    for cd in dirs:
        files = glob.glob(os.path.join(cd, "*.json"))
        assert files, f"empty cache dir {cd}"
        for f in files:
            try:
                rec = json.load(open(f))
                ans = json.loads(rec.get("content") or "")
            except (ValueError, TypeError):
                continue
            prov = rec.get("provenance") or {}
            if rec.get("kind") != "response" or not str(prov.get("prompt_builder_version", "")).startswith("m8b.9"):
                continue
            if not isinstance(ans, dict) or not isinstance(ans.get("weights"), dict):
                continue
            vc = prov.get("validation_context") or {}
            bel = vc.get("believed_contacts", [])
            s, ev = validate_answer(ans, bel, vc.get("task_ids", []))
            kinds = {e["kind"] for e in ev}
            pri = dict(s.priorities)
            rows.append({"contact": bool(bel), "w": dict(s.w),
                         "value_scaled": "value_budget_scaled" in kinds,
                         "priority_scaled": "priority_budget_scaled" in kinds,
                         "mult": [pri.get(int(c), 1.0) for c in bel]})
    return rows


def q(xs):
    return [round(float(x), 3) for x in np.percentile(xs, [10, 25, 50, 75, 90])] if xs else None


def summary(rows):
    out = {}
    for ph, key in ((True, "with_contact"), (False, "before_contact")):
        s = [r for r in rows if r["contact"] == ph]
        m = [x for r in s for x in r["mult"]]
        out[key] = {
            "answers": len(s),
            "weights_p10_25_50_75_90": {f.name: q([r["w"][f.name] for r in s]) for f in WEIGHT_FIELDS},
            "pursuit_value_threat_plus_engage": q([r["w"]["w_threat"] + r["w"]["w_engage"] for r in s]),
            "holding_station_plus_coverage": q([r["w"]["w_station"] + r["w"]["w_coverage"] for r in s]),
            "value_budget_scaled": sum(r["value_scaled"] for r in s),
            "priority_budget_scaled": sum(r["priority_scaled"] for r in s),
            "multipliers": len(m),
            "multipliers_zero": sum(x == 0 for x in m),
            "multipliers_below_1": sum(x < 1 - 1e-9 for x in m),
            "multipliers_above_1": sum(x > 1 + 1e-9 for x in m),
            "answers_with_a_zero_multiplier": sum(any(x == 0 for x in r["mult"]) for r in s),
        }
    return out


def main(data):
    dirs = cache_dirs(data)
    res = {"note": __doc__.split("\n\n")[0], "defaults": DEFAULT_WEIGHTS, "budget_fields": list(BUDGET_FIELDS),
           "ranges": {f.name: [f.lo, f.hi] for f in WEIGHT_FIELDS},
           "episodes": {k: len(v) for k, v in dirs.items()}}
    for k, v in dirs.items():
        res[k] = summary(answers(v))
    return res


if __name__ == "__main__":
    a = common.parser(__doc__.split("\n")[0]).parse_args()
    common.dump(main(a.data), os.path.join(common.out_dir(a), "bid_weights.json"))
