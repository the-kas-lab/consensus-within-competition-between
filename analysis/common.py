"""Where the data are, and how the analysis scripts read them."""

import argparse
import glob
import json
import os
from typing import Dict, Iterable, List

ARMS = ("cbaa", "independent", "commander")
REDS = (2, 4)
# Network conditions of the main study; ``REF+attack`` is the ideal network with falsified reports.
CONDS = ("REF", "MH", "LI", "LB", "REF+attack")
BASE = ("REF", "MH", "LI", "LB")
SEEDS = tuple(range(4001, 4031))

MAIN = [("main_study", "seeds_4001_4010"), ("main_study", "seeds_4011_4030")]
HIGH_AUTHORITY_LOW_COST = [("high_authority_low_cost", "seeds_4001_4030")]
HIGH_AUTHORITY_NEAR_FRONTIER = [("high_authority_near_frontier", "first_run"),
                                ("high_authority_near_frontier", "second_run")]
NEAR_FRONTIER_COMMANDER = [("near_frontier_commander", "seeds_4001_4010"),
                           ("near_frontier_commander", "seeds_4011_4030")]
AIRSPACE = [("airspace", "all_arms")]
DISGUISE = [("disguise", "all_arms")]


def cond(r: Dict) -> str:
    """Network condition of a record: the base condition, plus ``+attack`` for falsified reports."""
    return r["condition"] + ("+attack" if r["attacked"] else "")


def run_dir(data: str, run) -> str:
    return os.path.join(data, *run)


def records(data: str, runs: Iterable) -> List[Dict]:
    """The stored episode records of the given runs, in file order."""
    out = []
    for run in runs:
        with open(os.path.join(run_dir(data, run), "records.jsonl")) as fh:
            out += [json.loads(line) for line in fh if line.strip()]
    return out


def identity(data: str, run) -> str:
    with open(os.path.join(run_dir(data, run), "records.jsonl.identity.json")) as fh:
        return json.load(fh)["identity_hash"]


def cache_files(data: str, run) -> List[str]:
    """Every cached model answer of a run (the per-episode folders under cache/<identity>/)."""
    base = os.path.join(run_dir(data, run), "cache", identity(data, run))
    return sorted(f for f in glob.glob(os.path.join(base, "*", "*.json"))
                  if os.path.basename(f) != "namespace.json")


def dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)


def parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--data", required=True, help="the data deposit (layout: experiments/studies.py)")
    ap.add_argument("--out", default=None, help="results directory (default <data>/analysis)")
    return ap


def out_dir(args) -> str:
    return args.out or os.path.join(args.data, "analysis")


def slug(episode_id: str) -> str:
    return episode_id.replace("|", "_").replace("+", "_")


def problems(recs: Iterable[Dict]) -> List:
    """(episode id, reason) for every record that is not a valid episode: not in a terminal state, the
    solver failed, the team's model made no calls, or a model call failed for good (transport failure)."""
    from scripts import run_competing_teams as rct

    out = []
    for r in recs:
        eid = r["episode_id"]
        tp = rct.terminal_problem(r)
        if tp:
            out.append((eid, f"terminal: {tp}"))
        if r.get("solver_failed"):
            out.append((eid, "solver_failed"))
        d, i, c = r.get("defender") or {}, r.get("intruder") or {}, r.get("commander") or {}
        if r["arm"] != "commander" and not d.get("calls"):
            out.append((eid, "defender made no calls"))
        if d.get("transport_failures") or (d.get("events_by_kind") or {}).get("call_failure"):
            out.append((eid, "defender transport failure"))
        if r["arm"] == "commander" and not c.get("calls"):
            out.append((eid, "commander made no calls"))
        if c.get("transport_failures"):
            out.append((eid, "commander transport failure"))
        if i.get("transport_failures") or (i.get("invalid_by_reason") or {}).get("call_failure"):
            out.append((eid, "intruder transport failure"))
    return out


def valid_records(data: str, runs: Iterable) -> List[Dict]:
    """The stored records of ``runs`` that are valid episodes (see ``problems``)."""
    out = []
    for run in runs:
        recs = records(data, [run])
        bad = {eid for eid, _ in problems(recs)}
        out += [r for r in recs if r["episode_id"] not in bad]
    return out
