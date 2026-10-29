"""Counts about the runs themselves, read from the records and the cached answers.

  * main study: valid episodes, model calls (cached answers with a recorded request and usage), retries
    (failed attempts before the answer that was kept), permanently failed calls, episodes per intruder count;
  * commitment conflicts of the consensus and independent teams: sampled planning-epoch boundaries and how many
    held two defenders on one task, per team and condition, and per seed for the conditions the text compares;
  * near-frontier commander study: valid episodes; near-frontier high-authority study: invalid first-run episodes.

    python3 -m analysis.run_facts --data DATA [--out OUT]       ->  OUT/run_facts.json
"""
import json
import os

from analysis import common
from analysis.common import cond


def model_calls(data, runs):
    calls = retries = 0
    for run in runs:
        for f in common.cache_files(data, run):
            d = json.load(open(f))
            prov, usage = d.get("provenance") or {}, d.get("usage") or {}
            if prov.get("request_body") and usage:
                calls += 1
                retries += len((prov.get("attempts") or [])[:-1])
    return calls, retries


def conflicts(recs):
    """Sampled boundaries and conflicted boundaries per (team, condition), and per (team, MH or other, seed)."""
    cell, seed = {}, {}
    for x in recs:
        if x["arm"] == "commander":
            continue
        al = x["outcome"]["allocation"]
        assert al["epochs"] == -(-x["steps_run"] // 10)
        assert al["epochs_with_commitment_conflicts"] <= al["epochs"] - 1
        for key, d in (((x["arm"], cond(x)), cell),
                       ((x["arm"], "MH" if (x["condition"] == "MH" and not x["attacked"]) else "other", x["seed"]), seed)):
            a = d.setdefault(key, [0, 0])
            a[0] += al["epochs_with_commitment_conflicts"]
            a[1] += al["epochs"] - 1
    return cell, seed


def main(data):
    recs = common.records(data, common.MAIN)
    assert len(recs) == 900 and len({r["episode_id"] for r in recs}) == 900
    assert not common.problems(recs), common.problems(recs)[:3]
    calls, retries = model_calls(data, common.MAIN)
    tf = sum(int((r.get(side) or {}).get("transport_failures") or 0) for r in recs
             for side in ("defender", "intruder", "commander"))
    cell, seed = conflicts(recs)
    nf = common.HIGH_AUTHORITY_NEAR_FRONTIER[0]
    return {
        "main_valid": len(recs), "main_calls": calls, "main_retries": retries, "main_transport_failures": tf,
        "episodes_per_size": {str(n): sum(r["reds"] == n for r in recs) for n in (2, 4)},
        "conflicts_by_cell": {f"{a}|{c}": v for (a, c), v in cell.items()},
        "conflicts_by_seed": {f"{a}|{g}|{s}": v for (a, g, s), v in seed.items()},
        "near_frontier_commander_valid": len(common.valid_records(data, common.NEAR_FRONTIER_COMMANDER)),
        "near_frontier_first_run_invalid": len({eid for eid, _ in common.problems(common.records(data, [nf]))}),
    }


if __name__ == "__main__":
    a = common.parser(__doc__.split("\n")[0]).parse_args()
    common.dump(main(a.data), os.path.join(common.out_dir(a), "run_facts.json"))
