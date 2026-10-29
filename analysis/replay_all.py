"""Every re-simulation measure of the analysis, taken in one pass over each study's episodes.

Re-simulating an episode from its cached answers takes tens of seconds, so the measures that need a finished
simulation are all read off the same run (each only observes; none changes the episode):

  main study (900 episodes)   coverage and exposure, connectivity / time-limit endings / safety-filter
                              measures, role counts, and for the commander team the duplicate commitments and
                              the missed orders
  high authority, low-cost    coverage and exposure, time-limit endings
  high authority, near-frontier (valid episodes for the endings)   the same
  near-frontier commander     coverage and exposure, time-limit endings

    python3 -m analysis.replay_all --data DATA [--out OUT] [--workers 8] [--study NAME ...] [--limit N]

Writes OUT/coverage_exposure_<study>.json, main_replays.json, behaviour_replays.json,
commander_commitments.json, commander_missed_orders.json and endings_<group>.json. The same measures can be
taken one at a time with the module of each (analysis.coverage_replay, analysis.main_replays, ...).
"""
import os

from analysis import (behaviour_replays, common, commander_replays, coverage_replay, endings_replays,
                      main_replays, replay)


class _Measure:
    """A named measure: optional ``setup(entry, sim)`` (returns a state), ``extract``, and where it applies."""

    def __init__(self, name, extract, setup=None, applies=lambda e: True):
        self.name, self.extract, self.setup, self.applies = name, extract, setup, applies


MEASURES = {
    "coverage": _Measure("coverage", coverage_replay.extract),
    "main": _Measure("main", main_replays.extract, main_replays.setup),
    "behaviour": _Measure("behaviour", behaviour_replays.extract),
    "commitments": _Measure("commitments", commander_replays.commitments_extract,
                            commander_replays.commitments_setup, lambda e: e["arm"] == "commander"),
    "missed_orders": _Measure("missed_orders", commander_replays.missed_extract,
                              applies=lambda e: e["arm"] == "commander"),
    "endings": _Measure("endings", endings_replays.extract, endings_replays.setup),
}
# study -> runs -> measures. The main study's "endings" are the "main" measure (it includes them).
PLAN = {
    "main_study": [(run, ("coverage", "main", "behaviour", "commitments", "missed_orders")) for run in common.MAIN],
    "high_authority_low_cost": [(common.HIGH_AUTHORITY_LOW_COST[0], ("coverage", "endings"))],
    "high_authority_near_frontier": [(run, ("coverage", "endings")) for run in common.HIGH_AUTHORITY_NEAR_FRONTIER],
    "near_frontier_commander": [(run, ("coverage", "endings")) for run in common.NEAR_FRONTIER_COMMANDER],
}


class _States:
    def __init__(self, states):
        self.states = states

    def close(self):
        for st in reversed(list(self.states.values())):
            if st is not None and hasattr(st, "close"):
                st.close()


def _names(entry, measures):
    return [m for m in measures if MEASURES[m].applies(entry)]


class make_setup:
    """Picklable ``setup(entry, sim)`` for a group of measures."""

    def __init__(self, measures):
        self.measures = measures

    def __call__(self, entry, sim):
        return _States({m: (MEASURES[m].setup(entry, sim) if MEASURES[m].setup else None)
                        for m in _names(entry, self.measures)})


class make_extract:
    """Picklable ``extract(entry, sim, state)`` for a group of measures."""

    def __init__(self, measures):
        self.measures = measures

    def __call__(self, entry, sim, st):
        return {m: (MEASURES[m].extract(entry, sim, st.states[m]) if MEASURES[m].setup
                    else MEASURES[m].extract(entry, sim)) for m in _names(entry, self.measures)}


def run_study(data, out_dir, study, workers=4, limit=None):
    results = {}   # measure -> list of (stored, x, result)
    for run, measures in PLAN[study]:
        stored = {r["episode_id"]: r for r in common.records(data, [run])}
        ids = sorted(stored)[:limit] if limit else None
        xs = replay.replay_run(data, run, make_extract(measures), workers, only=set(ids) if ids else None,
                               setup=make_setup(measures))
        for x in xs:
            for m, res in x["result"].items():
                results.setdefault((run, m), []).append((stored[x["episode_id"]], x, res))
    return results


def write(data, out_dir, study, results):
    def items(measure, runs):
        return [(s, x, r) for run in runs for (s, x, r) in results.get((run, measure), [])]

    def entries(measure, runs, extra=None, only=None):
        # ``extra`` reads the measure's own result from x["result"]
        return [replay.episode_entry(s, x, r, (lambda st, xx, r=r: extra(st, dict(xx, result=r))) if extra else None)
                for s, x, r in items(measure, runs) if only is None or s["episode_id"] in only]

    if study == "main_study":
        runs = common.MAIN
        common.dump(coverage_replay.assemble(study, [(s, dict(x, result=r)) for s, x, r in items("coverage", runs)]),
                    os.path.join(out_dir, "coverage_exposure_main_study.json"))
        for m, name, note in (("main", "main_replays", "main study, re-simulated from the cache; see the module docstring"),
                              ("behaviour", "behaviour_replays", "main study, re-simulated from the cache; per-episode active role counts"),
                              ("commitments", "commander_commitments", "commander team re-simulated from the cache; duplicate commitments sampled as the auction runtime does"),
                              ("missed_orders", "commander_missed_orders", "commander team re-simulated from the cache; missed orders that would have changed the addressee's task")):
            extra = commander_replays._ledger_matches if m == "missed_orders" else None
            common.dump(replay.summary(entries(m, runs, extra), note), os.path.join(out_dir, name + ".json"))
        return
    if study == "high_authority_low_cost":
        runs = common.HIGH_AUTHORITY_LOW_COST
        common.dump(coverage_replay.assemble(study, [(s, dict(x, result=r)) for s, x, r in items("coverage", runs)]),
                    os.path.join(out_dir, "coverage_exposure_high_authority_low_cost.json"))
        common.dump(replay.summary(entries("endings", runs), "time-limit endings, group W, re-simulated from the cache"),
                    os.path.join(out_dir, "endings_W.json"))
        return
    if study == "high_authority_near_frontier":
        runs = common.HIGH_AUTHORITY_NEAR_FRONTIER
        common.dump(coverage_replay.assemble(study, [(s, dict(x, result=r)) for s, x, r in items("coverage", runs)]),
                    os.path.join(out_dir, "coverage_exposure_high_authority_near_frontier.json"))
        valid = {r["episode_id"] for r in common.valid_records(data, runs)}
        common.dump(replay.summary(entries("endings", runs, only=valid), "time-limit endings, group WSOL, re-simulated from the cache"),
                    os.path.join(out_dir, "endings_WSOL.json"))
        return
    if study == "near_frontier_commander":
        runs = common.NEAR_FRONTIER_COMMANDER
        common.dump(coverage_replay.assemble(study, [(s, dict(x, result=r)) for s, x, r in items("coverage", runs)]),
                    os.path.join(out_dir, "coverage_exposure_near_frontier_commander.json"))
        for group, run in (("SOLCMD_STD", common.NEAR_FRONTIER_COMMANDER[0]), ("SOLCMD_EXT", common.NEAR_FRONTIER_COMMANDER[1])):
            common.dump(replay.summary(entries("endings", [run]), f"time-limit endings, group {group}, re-simulated from the cache"),
                        os.path.join(out_dir, f"endings_{group}.json"))


if __name__ == "__main__":
    ap = common.parser(__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--study", action="append", choices=sorted(PLAN), default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N episodes (by id) of each run")
    a = ap.parse_args()
    out = common.out_dir(a)
    for study in a.study or list(PLAN):
        res = run_study(a.data, out, study, a.workers, a.limit)
        write(a.data, out, study, res)
        n = sum(len(v) for k, v in res.items() if k[1] in ("coverage", "endings", "main"))
        bad = [s["episode_id"] for v in res.values() for s, x, _ in v if not x["matches"]]
        print(f"{study}: {len(set(s['episode_id'] for v in res.values() for s, _, _ in v))} episodes re-simulated, "
              f"{len(set(bad))} differ from their records", flush=True)
