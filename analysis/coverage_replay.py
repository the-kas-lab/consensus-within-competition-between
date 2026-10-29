"""Coverage and exposure of every episode of a study, measured on its re-simulation.

Per recorded step and per intruder still active, from the finished simulation:

  known_team   at least one active defender holds a belief about the intruder (beliefs never expire by age,
               so any held belief is current);
  known_b0     defender 0 (the commander, in the commander team) holds one;
  exec         at least one active defender's effective target is that intruder under the pursuit controller;
  assigned     at least one active defender holds that intruder as its target in the pursue role (after
               delivery, in the commander team).

Counted per episode in intruder-steps: active, unseen (active, not known_team), known_team, known_team and
not exec, known_team and not assigned, and the same for defender 0; and the known_team counts split by the
intruder's current role. In the episodes with falsified reports the attack log gives the exposure: stored
false steps, fresh false steps and commitment steps per defender, the receivers that ever accepted an altered
record, and the deliveries and acceptances by defender 0.

    python3 -m analysis.coverage_replay main_study --data DATA [--out OUT] [--workers 8]

Populations: main_study, high_authority_low_cost, near_frontier_commander, high_authority_near_frontier.
Writes OUT/coverage_exposure_<population>.json.
"""
import os
from collections import Counter

from analysis import common, replay

POPULATIONS = {"main_study": common.MAIN, "high_authority_low_cost": common.HIGH_AUTHORITY_LOW_COST,
               "near_frontier_commander": common.NEAR_FRONTIER_COMMANDER,
               "high_authority_near_frontier": common.HIGH_AUTHORITY_NEAR_FRONTIER}


def coverage(sim):
    c = Counter()
    for st in sim.recorder.steps:
        dis = st["blue_disabled"]
        active_blues = [b for b in range(len(dis)) if not dis[b]]
        beliefs = st.get("beliefs") or {}
        eff = st.get("effective_targets") or {}
        tgt = st.get("target_red_ids") or {}
        roles = st.get("blue_roles") or {}
        for r, status in enumerate(st["red_outcomes"]):
            if status != "active":
                continue
            c["active"] += 1
            kt = any(str(r) in (beliefs.get(str(b)) or {}) for b in active_blues)
            k0 = (0 in active_blues) and str(r) in (beliefs.get("0") or {})
            ex = any((eff.get(str(b)) or {}).get("contact_id") == r
                     and (eff.get(str(b)) or {}).get("controller") == "neutralize"
                     for b in active_blues)
            asg = any(tgt.get(str(b)) == r and roles.get(str(b)) == "neutralize" for b in active_blues)
            if not kt:
                c["unseen"] += 1
                continue
            c["known_team"] += 1
            c["known_team_no_exec"] += not ex
            role = str((st.get("red_roles") or {}).get(str(r), "none")).lower()
            c[f"known_team_role_{role}"] += 1
            c[f"known_team_no_exec_role_{role}"] += not ex
            c["known_team_no_assigned"] += not asg
            c["known_team_assigned_no_exec"] += asg and not ex
            if k0:
                c["known_b0"] += 1
                c["known_b0_no_exec"] += not ex
                c["known_b0_no_assigned"] += not asg
    return dict(c)


def exposure(sim):
    log = sim.attack_log
    if log is None:
        return None
    acc = log["accepted"]
    return {"sender": log["sender"], "target_red": log["target_red"],
            "target_selected_step": log["target_selected_step"],
            "reports_altered": log["reports_altered"],
            "deliveries": len(log["deliveries"]), "accepted": len(acc),
            "receivers_accepting": sorted({int(a["receiver"]) for a in acc}),
            "deliveries_to_b0": sum(1 for d in log["deliveries"] if d["receiver"] == 0),
            "accepted_by_b0": sum(1 for a in acc if a["receiver"] == 0),
            "stored_false_steps": {str(k): v for k, v in log["stored_false_steps"].items()},
            "fresh_false_steps": {str(k): v for k, v in log["fresh_false_steps"].items()},
            "commitment_steps": {str(k): v for k, v in log["commitment_steps"].items()}}


def extract(entry, sim):
    return {"coverage": coverage(sim), "exposure": exposure(sim)}


def assemble(population, items):
    """The output of one population from (stored record, replay result) pairs."""
    episodes, mism = {}, []
    for stored, x in items:
        if not x["matches"]:
            mism.append(x["episode_id"])
        episodes[x["episode_id"]] = {
            "episode_id": x["episode_id"], "outcome": x["outcome"], **x["result"],
            **{k: stored[k] for k in ("arm", "reds", "condition", "attacked", "seed")}}
    return {"population": population, "n": len(episodes), "outcome_mismatches": mism, "episodes": episodes}


def main(data, population, out_path, workers=4, only=None):
    items = []
    for run in POPULATIONS[population]:
        stored = {r["episode_id"]: r for r in common.records(data, [run])}
        items += [(stored[x["episode_id"]], x) for x in replay.replay_run(data, run, extract, workers, only)]
    res = assemble(population, items)
    common.dump(res, out_path)
    return res["outcome_mismatches"]


if __name__ == "__main__":
    ap = common.parser(__doc__.split("\n")[0])
    ap.add_argument("population", choices=sorted(POPULATIONS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--episode", action="append", default=None, help="only this episode id")
    a = ap.parse_args()
    bad = main(a.data, a.population, os.path.join(common.out_dir(a), f"coverage_exposure_{a.population}.json"),
               a.workers, set(a.episode) if a.episode else None)
    print(f"{len(bad)} replayed episodes differ from their records")
