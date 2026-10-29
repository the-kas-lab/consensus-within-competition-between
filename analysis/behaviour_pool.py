"""Behaviour counts of the main study pooled over the 30 seeds.

  * defenders' executed roles before and after the team's first contact (defender-steps), summed from the
    per-episode counts of ``analysis.behaviour_replays``;
  * the commander's answered task kinds, and the intruders' answers (by phase and action), recounted from the
    cached model answers (no simulation, no model call).

    python3 -m analysis.behaviour_pool --data DATA [--out OUT]       ->  OUT/behaviour_pool.json

Reads OUT/behaviour_replays.json.
"""
import glob
import json
import os
from collections import Counter

from analysis import common

ARMS = ("cbaa", "independent", "commander")


def cache_answers(base, r):
    """(commander answered task kinds, intruder answers) Counters of one episode's cache folder."""
    ans, red = Counter(), Counter()
    for f in glob.glob(os.path.join(base, common.slug(r["episode_id"]), "*.json")):
        if f.endswith("namespace.json"):
            continue
        d = json.load(open(f))
        vc = (d.get("provenance") or {}).get("validation_context") or {}
        try:
            c = json.loads(d.get("content") or "")
        except ValueError:
            continue
        k = vc.get("kind")
        if k == "commander":
            for t in (c.get("task_by_agent") or {}).values():
                ans[f"{str(t).split(':')[0]}"] += 1
            ans["epochs"] += 1
        elif k == "intruder":
            red["answers"] += 1
            red[f"{vc.get('phase')}:{c.get('action')}"] += 1
            if vc.get("phase") == "approach" and c.get("action") == "attack":
                red[f"approach_gate:{c.get('target_id')}"] += 1
    return ans, red


def main(data, out_dir):
    beh = json.load(open(os.path.join(out_dir, "behaviour_replays.json")))
    assert beh["n"] == beh["n_reproduced"] == 900
    cells = {f"{a}|r{n}|ALL": {"episodes": 0, "active": Counter(), "answered": Counter(), "red": Counter()}
             for a in ARMS for n in (2, 4)}
    for x in beh["episodes"]:
        c = cells[f"{x['arm']}|r{x['reds']}|ALL"]
        c["episodes"] += 1
        c["active"].update(x["active"])
    for run in common.MAIN:
        base = os.path.join(common.run_dir(data, run), "cache", common.identity(data, run))
        for r in common.records(data, [run]):
            ans, red = cache_answers(base, r)
            c = cells[f"{r['arm']}|r{r['reds']}|ALL"]
            c["answered"].update(ans)
            c["red"].update(red)
    out = {"cells": {}}
    for k, c in cells.items():
        out["cells"][k] = {
            "episodes": c["episodes"],
            "exact_counts": {"active_blue_steps": dict(sorted(c["active"].items())),
                             "commander_answered_kinds": dict(sorted(c["answered"].items()))},
            "red_answers": dict(c["red"]),
            "red_attack_share": (c["red"]["approach:attack"] + c["red"]["interior:attack"]) / c["red"]["answers"]}
    return out


if __name__ == "__main__":
    a = common.parser(__doc__.split("\n")[0]).parse_args()
    o = common.out_dir(a)
    common.dump(main(a.data, o), os.path.join(o, "behaviour_pool.json"))
