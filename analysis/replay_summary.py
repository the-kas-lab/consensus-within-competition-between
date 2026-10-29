"""Summaries of main_replays.json: connectivity of the defenders' network, how the no-breach episodes ended at
the horizon, and how often the safety filter changed the defenders' commands.

    python3 -m analysis.replay_summary --out OUT        reads OUT/main_replays.json, writes OUT/main_replays_summary.json
"""
import json
import os
from collections import Counter

import numpy as np

from analysis import common

ARMS, CONDS = ("commander", "cbaa", "independent"), ("REF", "LI", "LB", "REF+attack", "MH")


def summarize(d):
    cond = common.cond
    E = [x for x in d["episodes"] if x["reproduced"]]
    out = {"n_episodes": d["n"], "n_reproduced": d["n_reproduced"], "comm": {}, "horizon": {}, "filter": {}}
    for a in ARMS:
        for c in CONDS:
            for r in (2, 4, "both"):
                s = [x for x in E if x["arm"] == a and cond(x) == c and (r == "both" or x["reds"] == r)]
                cf = np.array([x["comm"]["connected_fraction"] for x in s])
                out["comm"][f"{a}|{c}|r{r}"] = {
                    "n": len(s), "connected_fraction_mean": round(float(cf.mean()), 3),
                    "p10_p50_p90": [round(float(v), 3) for v in np.percentile(cf, [10, 50, 90])],
                    "mean_lcc_fraction": round(float(np.mean([x["comm"]["mean_lcc_fraction"] for x in s])), 3),
                    "mean_components": round(float(np.mean([x["comm"]["mean_n_components"] for x in s])), 2),
                    "mean_partition_s": round(float(np.mean([x["comm"]["mean_partition_steps"] for x in s])) * 0.1, 1),
                    "episodes_connected_lt_5pct": int((cf < 0.05).sum())}
                # horizon endings: no breach, >=1 red active, ran to the horizon
                hz = [x for x in s if x["live"]]
                cls = Counter(l["class"] for x in hz for l in x["live"])
                out["horizon"][f"{a}|{c}|r{r}"] = {
                    "episodes": len(s), "no_breach": sum(not x["outcome"]["touchdown"] for x in s),
                    "no_breach_horizon_red_active": len(hz), "live_reds": dict(cls),
                    "episodes_with_standoff": sum(any(l["class"] == "stand-off" for l in x["live"]) for x in hz)}
        for r in (2, 4, "both"):
            for c in CONDS + ("all",):
                s = [x for x in E if x["arm"] == a and (c == "all" or cond(x) == c) and (r == "both" or x["reds"] == r)]
                f = Counter()
                for x in s:
                    f.update(x["filt"])
                g = lambda k: f.get(k, 0)
                pc = g("pursuer_installed|calls")
                oc = g("other_role|calls")
                out["filter"][f"{a}|{c}|r{r}"] = {
                    "pursuer_calls": pc,
                    "pursuer_barrier_share": round(g("pursuer_installed|barrier") / pc, 4) if pc else None,
                    "pursuer_barrier_red_share": round(g("pursuer_installed|barrier|red") / pc, 4) if pc else None,
                    "pursuer_barrier_teammate_wreck_share": round(g("pursuer_installed|barrier|teammate_wreck") / pc, 4) if pc else None,
                    "pursuer_box_only_share": round(g("pursuer_installed|box_only") / pc, 4) if pc else None,
                    "other_role_calls": oc,
                    "other_barrier_red_share": round(g("other_role|barrier|red") / oc, 4) if oc else None,
                    "other_barrier_share": round(g("other_role|barrier") / oc, 4) if oc else None,
                    "red_rows_unbelieved_share": round((g("pursuer_installed|barrier_red|unbelieved") + g("other_role|barrier_red|unbelieved") + g("neutralize_no_target|barrier_red|unbelieved"))
                                                       / max(1, g("pursuer_installed|barrier|red") + g("other_role|barrier|red") + g("neutralize_no_target|barrier|red")), 4),
                    "paths": {k.split("|")[1]: v for k, v in f.items() if k.startswith("path|")}}

    return out


if __name__ == "__main__":
    ap = common.parser(__doc__.split("\n")[0])
    a = ap.parse_args()
    o = common.out_dir(a)
    common.dump(summarize(json.load(open(os.path.join(o, "main_replays.json")))),
                os.path.join(o, "main_replays_summary.json"))
