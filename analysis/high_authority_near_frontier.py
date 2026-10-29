"""The consensus team at high authority with the near-frontier model writing the defenders' bid weights (W-Sol),
on the ideal network with four intruders, one pair per seed (seeds 4001-4030).

  S1  W-Sol - W (same team with the low-cost model): capability within the interface
      exact McNemar p, Newcombe hybrid score interval (method 10), seed-level t interval as sensitivity.
Secondary, unadjusted: W-Sol - B (the main study's consensus team, low authority), W-Sol - commander team with
the near-frontier model; breach-free duration (t interval); kills and defenders disabled.

Valid W-Sol records only: the valid records of the first run plus the second run, which repeats the seeds whose
first episodes failed under a provider's rate limit; each is checked against its own run's identity.

Also the share of active-intruder time before any defender had seen the intruder ("unseen") and the share of
known intruders that no defender was pursuing ("uncovered"), W-Sol against W and B, from the coverage
re-simulations.

    python3 -m analysis.high_authority_near_frontier --data DATA [--out OUT]
        ->  OUT/high_authority_near_frontier.json, OUT/high_authority_near_frontier_secondary.json

The second file reads OUT/coverage_exposure_*.json (analysis.coverage_replay) for the main study, the
high-authority low-cost study and this study.
"""
import json
import os

import numpy as np
from scipy.stats import binomtest

from analysis import common, matrix_stats
from scripts.run_competing_teams import record_problem

SEEDS = list(common.SEEDS)


def ref4(recs, keep=lambda r: True):
    out = {}
    for r in recs:
        if r["reds"] == 4 and r["condition"] == "REF" and not r["attacked"] and r["seed"] in SEEDS and keep(r):
            assert r["seed"] not in out
            out[r["seed"]] = r
    assert sorted(out) == SEEDS, len(out)
    return out


def valid_w_sol(data):
    recs, used = [], {}
    bad = []
    for run in common.HIGH_AUTHORITY_NEAR_FRONTIER:
        rr = common.records(data, [run])
        ident = common.identity(data, run)
        invalid = {eid for eid, _ in common.problems(rr)}
        for r in rr:
            e = {k: r.get(k) for k in ("arm", "reds", "family", "condition", "attacked", "seed", "rep", "design")}
            p = record_problem(e, [x for x in rr if x["episode_id"] == r["episode_id"]], ident)
            if r["episode_id"] in invalid:
                continue
            if p:
                bad.append((r["episode_id"], p))
                continue
            recs.append(r)
            used.setdefault(run[1], []).append(r["seed"])
    if bad:
        raise SystemExit(f"invalid W-Sol records: {bad}")
    return recs, used


def main(data):
    recs, used = valid_w_sol(data)
    ident = {run[1]: common.identity(data, run) for run in common.HIGH_AUTHORITY_NEAR_FRONTIER}
    arms = {"W-Sol": ref4(recs)}
    arms["W"] = ref4(common.records(data, common.HIGH_AUTHORITY_LOW_COST))
    arms["B"] = ref4(common.records(data, common.MAIN), lambda r: r["arm"] == "cbaa")
    arms["Sol commander"] = ref4(common.records(data, common.NEAR_FRONTIER_COMMANDER))

    br = lambda r: int(bool(r["outcome"]["touchdown"]))  # noqa: E731
    dur = lambda r: (r["outcome"]["touchdown_step"] if r["outcome"]["touchdown"] else r["horizon_steps"]) * 0.1  # noqa: E731

    def row(x, y):
        a, b = arms[x], arms[y]
        e = sum(br(a[s]) and br(b[s]) for s in SEEDS)
        f = sum(br(a[s]) and not br(b[s]) for s in SEEDS)
        g = sum(br(b[s]) and not br(a[s]) for s in SEEDS)
        h = len(SEEDS) - e - f - g
        ds = [br(a[s]) - br(b[s]) for s in SEEDS]
        return {"first": f"{sum(br(a[s]) for s in SEEDS)}/30", "second": f"{sum(br(b[s]) for s in SEEDS)}/30",
                "estimate": round((f - g) / 30, 4), "first_only": f, "second_only": g,
                "newcombe10": [round(z, 4) for z in matrix_stats.newcombe10(e, f, g, h)],
                "exact_mcnemar_p": 1.0 if f + g == 0 else round(float(binomtest(f, f + g, 0.5).pvalue), 5),
                "t_seed": matrix_stats.cluster_interval(ds),
                "breach_free_s": matrix_stats.cluster_interval([dur(a[s]) - dur(b[s]) for s in SEEDS])}

    out = {"identity": ident, "seeds_by_run": {k: sorted(v) for k, v in used.items()}, "levels": {}, "rows": {}}
    for k, v in arms.items():
        out["levels"][k] = {"breaches": f"{sum(br(v[s]) for s in SEEDS)}/30",
                            "kills_mean": round(float(np.mean([v[s]["outcome"]["num_neutralized"] for s in SEEDS])), 2),
                            "blues_disabled_mean": round(float(np.mean([v[s]["outcome"]["n_blues_disabled"]
                                                                        for s in SEEDS])), 2)}
    out["rows"]["S1 W-Sol - W"] = row("W-Sol", "W")
    for rid, x, y in (("W-Sol - B", "W-Sol", "B"), ("W-Sol - Sol commander", "W-Sol", "Sol commander")):
        out["rows"][f"secondary {rid}"] = row(x, y)
    return out


def secondary(data, out_dir):
    valid = {r["episode_id"] for r in valid_w_sol(data)[0]}

    def load(pop):
        return json.load(open(os.path.join(out_dir, f"coverage_exposure_{pop}.json")))

    def ref4cov(d, arm=None, only=None):
        out = {}
        for eid, e in d["episodes"].items():
            if only is not None and eid not in only:
                continue
            if e["reds"] == 4 and e["condition"] == "REF" and not e["attacked"] and (arm is None or e["arm"] == arm):
                out[e["seed"]] = e["coverage"]
        return out

    d = load("high_authority_near_frontier")
    assert d["outcome_mismatches"] == []
    arms = {"W-Sol": ref4cov(d, only=valid),
            "W": ref4cov(load("high_authority_low_cost")),
            "B": ref4cov(load("main_study"), "cbaa")}
    out = {"unseen_share_pooled": {}, "uncovered_known_pooled": {}, "rows": {}}
    for k, v in arms.items():
        assert sorted(v) == SEEDS, k
        out["unseen_share_pooled"][k] = round(sum(v[s]["unseen"] for s in SEEDS) / sum(v[s]["active"] for s in SEEDS), 4)
        out["uncovered_known_pooled"][k] = round(sum(v[s]["known_team_no_exec"] for s in SEEDS)
                                                 / sum(v[s]["known_team"] for s in SEEDS), 4)
    per = {k: [v[s]["unseen"] / v[s]["active"] for s in SEEDS] for k, v in arms.items()}
    for y in ("W", "B"):
        out["rows"][f"unseen W-Sol - {y}"] = matrix_stats.cluster_interval([a - b for a, b in zip(per["W-Sol"], per[y])])
    return out


if __name__ == "__main__":
    a = common.parser(__doc__.split("\n")[0]).parse_args()
    o = common.out_dir(a)
    common.dump(main(a.data), os.path.join(o, "high_authority_near_frontier.json"))
    if os.path.exists(os.path.join(o, "coverage_exposure_high_authority_near_frontier.json")):
        common.dump(secondary(a.data, o), os.path.join(o, "high_authority_near_frontier_secondary.json"))
