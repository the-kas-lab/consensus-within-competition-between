"""Measures of the commander team's orders, taken while its 300 episodes are re-simulated from the cache.

Duplicate commitments. At every planning-epoch boundary after an episode's first (sim_step > 0), before the
epoch's planning call, take the order in force for every live defender (the newest accepted assignment
command) and count the boundary as conflicted if two or more live defenders hold the same task. This matches
how the auction runtime logs commitment conflicts; the study records hold no such log for the commander team.
Defenders that were never given an order hold no commitment and are skipped.

Missed orders. Of the commander's orders that never reached their defender, how many would have changed that
defender's task? The per-order log (Simulation.command_log) is not stored in the records, so it is read from
each replayed episode. Per order addressed to another defender (via == "network"):

  applied        applied_at is not None (the addressee accepted it);
  held           the task the addressee was following when the order was generated: the task of its newest
                 applied order with applied_at <= generated_at; with no order yet, "DEFEND:<own id>", since a
                 defender without an order defends its own station;
  changed        the order's task differs from held;
  kind_changed   the task kind (the prefix before the first ':') differs;
  addressee_lost the addressee was disabled within 10 steps (one planning interval) of generation, so a
                 missed order could not have mattered for long.

Per defender-step (live defenders other than the commander, steps after the first order) a step is stale if
the commander's newest order for that defender names a different task than the one the defender holds.
The replayed ledger counts must also equal the stored outcome.commands.

    python3 -m analysis.commander_replays commitments|missed_orders --data DATA [--out OUT] [--workers 8]

Writes OUT/commander_commitments.json and OUT/commander_missed_orders.json.
"""
import os
from collections import Counter

from analysis import common, replay

LEDGER = ("generated", "self_applied", "no_link", "n_delivered_commands_applied")
LOST_WINDOW = 10


# -- duplicate commitments ---------------------------------------------------------------------

class _Sampler:
    def __init__(self, tb, orig, samples):
        self.tb, self.orig, self.samples = tb, orig, samples

    def close(self):
        self.tb.PlannerObservation = self.orig


def commitments_setup(e, sim):
    import simulation.testbed as tb

    samples = []
    orig = tb.PlannerObservation

    def observed(*a, **k):
        step = int(k["sim_step"])
        if step > 0:
            held = {}
            for agent, cmd in sim.agent_commitments.items():
                if not sim.blue_disabled[agent] and cmd.task_id is not None:
                    held.setdefault(cmd.task_id, []).append(int(agent))
            dup = {t: sorted(ids) for t, ids in held.items() if len(ids) > 1}
            samples.append({"step": step, "dup": dup,
                            "live": int(sum(not d for d in sim.blue_disabled)),
                            "held": int(sum(len(v) for v in held.values()))})
        return orig(*a, **k)

    tb.PlannerObservation = observed
    return _Sampler(tb, orig, samples)


def commitments_extract(e, sim, st):
    samples = st.samples
    kinds = Counter(t.split(":")[0] for s in samples for t in s["dup"])
    return {"boundaries": len(samples),
            "boundaries_with_duplicates": sum(1 for s in samples if s["dup"]),
            "duplicate_task_kinds": dict(kinds),
            "boundaries_without_any_commitment": sum(1 for s in samples if s["held"] == 0)}


# -- missed orders -----------------------------------------------------------------------------

def kind(task_id):
    return None if task_id is None else str(task_id).split(":")[0]


def analyse(log, disabled_at, steps_run, commander):
    """Pure post-processing of one episode's command log (see the module docstring)."""
    net = [e for e in log if e["via"] == "network"]
    by_agent = {}
    for e in net:
        by_agent.setdefault(e["agent"], []).append(e)
    out = {"orders": len(net), "applied": 0, "applied_changed": 0, "missed": 0, "missed_changed": 0,
           "missed_kind_changed": 0, "missed_same": 0, "missed_addressee_lost": 0,
           "missed_changed_by_kind": {}, "defender_steps": 0, "stale_steps": 0}
    for a, es in by_agent.items():
        applied = sorted((e for e in es if e["applied_at"] is not None), key=lambda e: e["applied_at"])
        for e in es:
            held = None
            for x in applied:                      # newest applied order before this one was generated
                if x["applied_at"] <= e["generated_at"] and (held is None or x["epoch"] > held["epoch"]):
                    held = x
            held_task = held["task_id"] if held else f"DEFEND:{a}"
            changed = e["task_id"] != held_task
            if e["applied_at"] is not None:
                out["applied"] += 1
                out["applied_changed"] += int(changed)
                continue
            out["missed"] += 1
            d = disabled_at.get(a)
            if d is not None and d <= e["generated_at"] + LOST_WINDOW:
                out["missed_addressee_lost"] += 1
                continue
            if changed:
                out["missed_changed"] += 1
                k = f"{kind(held_task)}->{kind(e['task_id'])}"
                out["missed_changed_by_kind"][k] = out["missed_changed_by_kind"].get(k, 0) + 1
                out["missed_kind_changed"] += int(kind(e["task_id"]) != kind(held_task))
            else:
                out["missed_same"] += 1
        # stale defender-steps: commander's newest order for a names another task than a holds
        gen = sorted(es, key=lambda e: e["generated_at"])
        if not gen:
            continue
        end = min(steps_run, disabled_at.get(a, steps_run))
        gi, ai, intent, held = 0, 0, None, f"DEFEND:{a}"
        for t in range(gen[0]["generated_at"], end):
            while gi < len(gen) and gen[gi]["generated_at"] <= t:
                intent = gen[gi]["task_id"]
                gi += 1
            while ai < len(applied) and applied[ai]["applied_at"] <= t:
                held = applied[ai]["task_id"]
                ai += 1
            out["defender_steps"] += 1
            out["stale_steps"] += int(intent != held)
    return out




def missed_extract(e, sim):
    steps_run = len(sim.history_blues) - 1
    disabled_at = {}
    for ev in sim.resolution_log:
        if ev["team"] == "blue" and ev["event"] == "disabled":
            disabled_at.setdefault(int(ev["agent"]), int(ev["step"]))
    log = [dict(x) for x in sim.command_log]
    res = analyse(log, disabled_at, steps_run, sim.commander_id)
    cm = sim.command_metrics()
    return {"ledger": {k: cm[k] for k in LEDGER},
            "task_ids_seen": sorted({str(x["task_id"]) for x in log})[:12], **res}


def _ledger_matches(stored, x):
    return {"reproduced": x["matches"] and all(x["result"]["ledger"][k] == stored["outcome"]["commands"][k]
                                               for k in LEDGER)}


def main(which, data, out_path, workers=4, limit=None):
    if which == "commitments":
        return replay.collect(data, common.MAIN, commitments_extract, workers, limit,
                              setup=commitments_setup, arm="commander", out_path=out_path,
                              note="commander team re-simulated from the cache; duplicate commitments "
                                   "sampled as the auction runtime does")
    return replay.collect(data, common.MAIN, missed_extract, workers, limit, arm="commander",
                          out_path=out_path, extra=_ledger_matches,
                          note="commander team re-simulated from the cache; missed orders that would "
                               "have changed the addressee's task")


if __name__ == "__main__":
    ap = common.parser(__doc__.split("\n")[0])
    ap.add_argument("which", choices=("commitments", "missed_orders"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    name = {"commitments": "commander_commitments", "missed_orders": "commander_missed_orders"}[a.which]
    res = main(a.which, a.data, os.path.join(common.out_dir(a), name + ".json"), a.workers, a.limit)
    print(f"{res['n']} episodes, {res['n_reproduced']} reproduced")
