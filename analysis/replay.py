"""Re-simulate stored episodes from their cached model answers and hand each finished simulation to a function.

An episode is rebuilt from the run manifest (experiments/studies.py), run with every model answer read from
the run's cache (no model is called; a missing answer stops the episode) and compared with its stored record.
``extract(entry, sim)`` then reads whatever the analysis needs from the finished simulation.
"""
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Optional

from analysis import common

CHECK = ("touchdown", "touchdown_step", "touchdown_red_index", "red_outcomes", "num_neutralized")


def _drop_keys():
    for k in [k for k in os.environ if k.startswith("OPENROUTER_API_KEY")]:
        os.environ.pop(k)


def build(entry: Dict, run_dir: str, identity: str):
    """The simulation of one episode, wired to replay its cached answers."""
    from experiments.run import _llm_args
    from scripts import run_competing_teams as rct

    llm_args = _llm_args(run_dir, identity, replay=True)
    slug = entry["episode_id"].replace("|", "_").replace("+", "_")
    llm_args["cache_dir"] = os.path.join(llm_args["cache_dir"], identity, slug)
    llm_args["cache_per_episode"] = True
    sim, _ = rct.build(entry, "profile", llm_args)
    return sim


def _job(payload):
    run, data, entry, extract, setup, bird_drone_dir = payload
    _drop_keys()
    from scripts.episode_builder import _Quiet

    sim = build(entry, common.run_dir(data, run), common.identity(data, run))
    state = setup(entry, sim) if setup else None
    try:
        with _Quiet():
            sim.run()
    finally:
        if state is not None and hasattr(state, "close"):
            state.close()
    o = sim.episode_outcome()
    return {"episode_id": entry["episode_id"], "outcome": {k: o[k] for k in CHECK},
            "steps_run": len(sim.history_blues) - 1,
            "result": extract(entry, sim, state) if setup else extract(entry, sim)}


def manifest(run, bird_drone_dir: Optional[str] = None) -> List[Dict]:
    from experiments import studies

    r = next(x for x in studies.RUNS if (x.study, x.name) == tuple(run))
    return r.entries(bird_drone_dir) if r.needs_bird_drones else r.entries()


def replay_run(data: str, run, extract: Callable, workers: int = 4, only=None,
               bird_drone_dir: Optional[str] = None, setup: Optional[Callable] = None) -> List[Dict]:
    """Replay every stored episode of ``run``; returns one dict per episode with the extracted result and
    ``matches`` (replayed outcome equal to the stored record on the check fields and the steps run).
    ``setup(entry, sim)`` runs before the episode (to install read-only instrumentation) and returns a state
    passed on as ``extract(entry, sim, state)``; a state with a ``close()`` method is closed after the run."""
    stored = {r["episode_id"]: r for r in common.records(data, [run])}
    entries = [e for e in manifest(run, bird_drone_dir) if e["episode_id"] in stored]
    if only is not None:
        entries = [e for e in entries if e["episode_id"] in only]
    out = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for x in pool.map(_job, [(run, data, e, extract, setup, bird_drone_dir) for e in entries]):
            s = stored[x["episode_id"]]
            x["matches"] = (all(x["outcome"][k] == s["outcome"][k] for k in CHECK)
                            and x["steps_run"] == s["steps_run"])
            out.append(x)
    return out


def episode_entry(stored: Dict, x: Dict, result: Dict, extra: Optional[Callable] = None) -> Dict:
    """One output entry: the replay's outcome and the extracted result next to the stored record's keys."""
    ep = {"episode_id": x["episode_id"], "outcome": x["outcome"], "steps_run": x["steps_run"], **result,
          **{k: stored[k] for k in ("arm", "reds", "condition", "attacked", "seed")},
          "reproduced": x["matches"]}
    if extra:
        ep.update(extra(stored, x))
    return ep


def summary(episodes: List[Dict], note: str = "") -> Dict:
    return {"note": note, "n": len(episodes), "n_reproduced": sum(e["reproduced"] for e in episodes),
            "mismatches": [e["episode_id"] for e in episodes if not e["reproduced"]], "episodes": episodes}


def collect(data: str, runs, extract: Callable, workers: int = 4, limit: Optional[int] = None,
            setup: Optional[Callable] = None, arm: Optional[str] = None, out_path: Optional[str] = None,
            note: str = "", extra: Optional[Callable] = None, only=None) -> Dict:
    """Replay the stored episodes of ``runs`` (optionally one arm only) and gather one entry per episode:
    the extracted result, the arm, intruder count, condition, attack flag and seed of its record, and
    ``reproduced``. ``extra(stored_record, replay_result)`` may add fields. ``only`` restricts to those
    episode ids; ``limit`` keeps the first ``limit`` episodes (by id) of each run."""
    episodes = []
    for run in runs:
        stored = {r["episode_id"]: r for r in common.records(data, [run])
                  if arm in (None, r["arm"]) and (only is None or r["episode_id"] in only)}
        ids = sorted(stored)[:limit] if limit else sorted(stored)
        for x in replay_run(data, run, extract, workers, only=set(ids), setup=setup):
            episodes.append(episode_entry(stored[x["episode_id"]], x, x["result"], extra))
    res = summary(episodes, note)
    if out_path:
        common.dump(res, out_path)
    return res
