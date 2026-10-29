"""Replay or run a reported study.

    # Re-simulate every stored episode of a study from its cached model answers.
    # Free: no model is called, a missing answer stops the episode.
    python3 -m experiments.run main_study --data /path/to/data --replay

    # Run a study live (needs OPENROUTER_API_KEY; costs money, so a spend cap is required).
    python3 -m experiments.run airspace --data /path/to/new_data --live --cap-usd 20

Studies: main_study, high_authority_low_cost, high_authority_near_frontier,
near_frontier_commander, airspace, disguise (see experiments/studies.py).

A replayed episode must equal its stored record on the breach, the breach step, the
breaching intruder, every intruder's outcome, the number of intruders destroyed and
the steps run. The disguise study needs the four bird-drone photographs
(``--bird-drone-dir``); the other studies need only the licensed photographs in
assets/contact_photos.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import studies  # noqa: E402
from scripts import run_competing_teams as rct  # noqa: E402

MATCH_FIELDS = ("touchdown", "touchdown_step", "touchdown_red_index", "red_outcomes",
                "num_neutralized")
KEY_PREFIX = "OPENROUTER_API_KEY"


def _drop_keys():
    for k in [k for k in os.environ if k.startswith(KEY_PREFIX)]:
        os.environ.pop(k)


def _llm_args(run_dir, identity, replay, cap_usd=None, cap_usd_near_frontier=None):
    args = {"profile_obj": studies.profile("low-cost"), "replay": replay,
            "cache_dir": os.path.join(run_dir, "cache"), "agent_workers": 4,
            "identity_hash": identity, "cache_per_episode": True,
            "near_frontier": {"profile_obj": studies.profile("near-frontier"),
                              "cap_usd": cap_usd_near_frontier,
                              "ledger": os.path.join(run_dir, "ledger_near_frontier.json")}}
    if not replay:
        args.update(cap_usd=cap_usd, ledger=os.path.join(run_dir, "ledger.json"))
    else:
        args["agent_workers"] = 1
    return args


def _job(payload):
    entry, llm_args = payload
    return rct.episode_record(entry, "profile", llm_args)


def _matches(rec, stored):
    bad = [k for k in MATCH_FIELDS if rec["outcome"][k] != stored["outcome"][k]]
    if rec["steps_run"] != stored["steps_run"]:
        bad.append("steps_run")
    return bad


def replay_run(run, data_dir, out_dir, bird_drone_dir, workers, limit, only):
    run_dir = os.path.join(data_dir, run.subdir)
    ident = studies.stored_identity(run_dir)
    stored = {r["episode_id"]: r for r in studies.stored_records(run_dir)}
    if ident is None or not stored:
        print(f"{run.subdir}: no stored records in {run_dir}; skipped")
        return 0, 0
    entries = run.entries(bird_drone_dir) if run.needs_bird_drones else run.entries()
    entries = [e for e in entries if e["episode_id"] in stored]
    if only:
        entries = [e for e in entries if e["episode_id"] in only]
    if limit:
        entries = entries[:limit]
    llm_args = _llm_args(run_dir, ident["identity_hash"], replay=True)
    os.makedirs(os.path.join(out_dir, run.study), exist_ok=True)
    sink = os.path.join(out_dir, run.subdir.replace(os.sep, "_") + ".jsonl")
    n_ok, n = 0, 0
    with open(sink, "w") as fh, ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_job, (e, llm_args)): e for e in entries}
        for f in as_completed(futs):
            e = futs[f]
            rec = f.result()
            bad = _matches(rec, stored[e["episode_id"]])
            n += 1
            n_ok += not bad
            fh.write(json.dumps({"episode_id": e["episode_id"], "matched": not bad,
                                 "differs_on": bad, "outcome": rec["outcome"]}) + "\n")
            print(f"  {'match   ' if not bad else 'MISMATCH'} {e['episode_id']} {bad or ''}", flush=True)
    print(f"{run.subdir}: {n_ok} of {n} re-simulated episodes equal their records", flush=True)
    return n_ok, n


def live_run(run, data_dir, bird_drone_dir, workers, limit, only, cap_usd, cap_usd_near_frontier):
    if not os.environ.get(KEY_PREFIX):
        raise SystemExit(f"{KEY_PREFIX} is not set")
    run_dir = os.path.join(data_dir, run.subdir)
    os.makedirs(run_dir, exist_ok=True)
    entries = run.entries(bird_drone_dir) if run.needs_bird_drones else run.entries()
    arms = sorted({e["arm"] for e in entries})
    modes = {e.get("image_mode") for e in entries if e.get("image_mode")}
    ident = rct.execution_identity("profile", studies.profile("low-cost"), run.subdir,
                                   designs=[studies.DESIGN], arms=arms,
                                   decoys=any(e.get("decoys") for e in entries), image_modes=modes)
    ident["near_frontier_profile"] = studies.profile("near-frontier").as_dict()
    ident.pop("identity_hash")
    import hashlib

    ident["identity_hash"] = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
    id_path = os.path.join(run_dir, "records.jsonl.identity.json")
    held = studies.stored_identity(run_dir)
    if held is not None and held["identity_hash"] != ident["identity_hash"]:
        raise SystemExit(f"{run_dir} was produced under identity {held['identity_hash']}, not "
                         f"{ident['identity_hash']}: refusing to resume a different configuration")
    json.dump(ident, open(id_path, "w"), indent=1, sort_keys=True)
    done = {r["episode_id"] for r in studies.stored_records(run_dir)}
    todo = [e for e in entries if e["episode_id"] not in done]
    if only:
        todo = [e for e in todo if e["episode_id"] in only]
    if limit:
        todo = todo[:limit]
    llm_args = _llm_args(run_dir, ident["identity_hash"], replay=False, cap_usd=cap_usd,
                         cap_usd_near_frontier=cap_usd_near_frontier or cap_usd)
    print(f"{run.subdir}: {len(entries)} episodes, {len(done)} recorded, {len(todo)} to run", flush=True)
    t0 = time.monotonic()
    with open(os.path.join(run_dir, "records.jsonl"), "a") as sink, \
            ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_job, (e, llm_args)): e for e in todo}
        for f in as_completed(futs):
            rec = f.result()  # a spend-cap or replay stop ends the batch
            rec["identity_hash"] = ident["identity_hash"]
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            print(f"  {rec['episode_id']}: {rec['wall_s']} s", flush=True)
    print(f"batch wall {time.monotonic() - t0:.1f} s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("study", choices=studies.STUDIES)
    ap.add_argument("--data", required=True, help="data directory (layout: experiments/studies.py)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replay", action="store_true", help="re-simulate from cached answers (free)")
    mode.add_argument("--live", action="store_true", help="call the models (paid)")
    ap.add_argument("--out", default=None, help="replay results directory (default <data>/replays)")
    ap.add_argument("--bird-drone-dir", default=None, help="the four bird-drone photographs")
    ap.add_argument("--workers", type=int, default=4, help="episodes in parallel")
    ap.add_argument("--limit", type=int, default=None, help="at most this many episodes per run")
    ap.add_argument("--episode", action="append", default=[], help="only this episode id")
    ap.add_argument("--cap-usd", type=float, default=None, help="spend cap of a live run")
    ap.add_argument("--cap-usd-near-frontier", type=float, default=None,
                    help="spend cap of the near-frontier model (default: --cap-usd)")
    args = ap.parse_args(argv)
    if args.live and args.cap_usd is None:
        raise SystemExit("a live run spends money: pass --cap-usd")
    if args.replay:
        _drop_keys()  # replays never call a model
    runs = studies.runs_of(args.study)
    if any(r.needs_bird_drones for r in runs) and not args.bird_drone_dir:
        raise SystemExit(f"the {args.study} study needs --bird-drone-dir (see the README)")
    out_dir = args.out or os.path.join(args.data, "replays")
    total_ok = total = 0
    for run in runs:
        if args.replay:
            ok, n = replay_run(run, args.data, out_dir, args.bird_drone_dir, args.workers,
                               args.limit, set(args.episode))
            total_ok, total = total_ok + ok, total + n
        else:
            live_run(run, args.data, args.bird_drone_dir, args.workers, args.limit,
                     set(args.episode), args.cap_usd, args.cap_usd_near_frontier)
    if args.replay and total_ok != total:
        raise SystemExit(f"{total - total_ok} re-simulated episodes differ from their records")


if __name__ == "__main__":
    main()
