"""Exp 4: independent selection vs CBAA, rule-based, at ideal communication.

Specification:

    arm 1  independent   allocation="cbaa", cbaa_transport="none", slots=1
    arm 2  CBAA, 1 slot  cbaa_transport="ideal", slots=1
    arm 3  CBAA, 2 slots cbaa_transport="ideal", slots=2

All arms: belief "shared", profile "balanced", station_heading="outward",
4 blues / 2 reds, 400 steps, planning 1.0 s, the safety-parameter choice CBF. Seeds 3801-3900, each in
both families and all three arms. Sequential; no interim looks; every episode
kept.

Two phases, deliberately separate:

    run      one JSONL line per episode, appended as it finishes, so a crash
             loses nothing and nothing is recomputed
    analyze  the pre-declared contrasts from that file, and nothing else

    PYTHONPATH=$PWD python3 -m scripts.arm_variants run --out
    PYTHONPATH=$PWD python3 -m scripts.arm_variants analyze --episodes \
        --out

Intervals are 95% POINTWISE paired-bootstrap intervals with NO familywise error
control (19, Amendment 1.4). Free: rule-based throughout, no network calls.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.episode_builder import _Quiet, build_episode  # noqa: E402
from scripts.auction_smoke import (  # noqa: E402
    attach_bearing_log,
    attach_ledger,
    measure,
    measure_posture,
)

SPEC = ""
FAMILIES = ("symmetric", "multi_bearing")
ARMS = {
    1: dict(cbaa_transport="none", cbaa_slots=1),
    2: dict(cbaa_transport="ideal", cbaa_slots=1),
    3: dict(cbaa_transport="ideal", cbaa_slots=2),
}
BASE_CFG = dict(
    blues=4, reds=2, steps=400, planning_interval_seconds=1.0,
    rule="nearest",  # built by run_episode, superseded by allocation="cbaa"
    neutralize_fraction=0.5, recon_fraction=0.5,
    allocation="cbaa", cbaa_profile="balanced", station_heading="outward",
)
BELIEF_MODE = "shared"
SEEDS = tuple(range(3801, 3901))
CONTRASTS = {"C1": (2, 1), "C2": (3, 2), "C3": (3, 1)}  # (A, B): A - B
OUTCOMES = ("breach", "attrition_rate", "blue_score")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260923


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def coordination_extras(sim, ledger):
    """Spec Sec 5's coordination measures that the smoke's ``measure`` lacks:
    unassigned believed threats at planning epochs, and commitment changes per
    agent."""
    interval = sim.planning_interval_steps
    by_step = {}
    for entry in ledger:
        by_step.setdefault(entry["step"], []).append(entry)
    current, previous = {}, {}
    unassigned = planning_epochs = 0
    changes = Counter()
    for st in sim.recorder.steps:
        for entry in by_step.get(st["step"], []):
            current = dict(entry["commitments"])
        for agent, task in current.items():
            if agent in previous and previous[agent] != task and task is not None:
                changes[agent] += 1
        previous = dict(current)
        if st["step"] % interval:
            continue
        planning_epochs += 1
        believed = {int(c) for per in st["beliefs"].values() for c in per}
        active = {c for c in believed if st["red_outcomes"][c] == "active"}
        covered = {t[1] for t in current.values()
                   if t is not None and t[0] in ("NEUTRALIZE", "SCREEN")}
        unassigned += bool(active - covered)
    return {
        "planning_epochs": planning_epochs,
        "epochs_with_unassigned_believed_threat": unassigned,
        "commitment_changes_per_agent": {str(a): n for a, n in sorted(changes.items())},
    }


def run_episode_record(seed, family, arm):
    cfg = dict(BASE_CFG, family=family, **ARMS[arm])
    sim = build_episode(seed, BELIEF_MODE, cfg)
    ledger = attach_ledger(sim)
    bearing_log = attach_bearing_log(sim)
    with _Quiet():
        sim.run()
    m = measure(sim, ledger)
    outcome = sim.episode_outcome()
    return {
        "seed": seed, "family": family, "arm": arm,
        "breach": bool(outcome["touchdown"]),
        "attrition_rate": float(outcome["attrition_rate"]),
        "blue_score": float(outcome["blue_score"]),
        "num_neutralized": int(outcome["num_neutralized"]),
        "num_escaped": int(outcome["num_escaped"]),
        "flagged": bool(outcome["solver_failed"] or sim.collided),
        "solver_failed": bool(outcome["solver_failed"]),
        "collided": bool(sim.collided),
        "n_steps": m["n_steps"],
        "detected_any": m["first_detection_step"] is not None,
        "first_belief_step": m["first_belief_step"],
        "fraction_agent_steps_believing": (
            m["agent_steps_believing_a_contact"] / max(1, m["active_agent_steps"])
        ),
        "allocation": m["allocation"],
        "rounds_to_stable": m["rounds_to_stable"],
        "commitment_mix": m["commitment_mix"],
        "executed": m["executed"],
        "not_executed": m["not_executed"],
        "not_executed_causes": m["not_executed_causes"],
        "posture": measure_posture(sim, bearing_log),
        **coordination_extras(sim, ledger),
    }


def cmd_run(args):
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r["seed"], r["family"], r["arm"]) for r in map(json.loads, f)}
    seeds = SEEDS if args.seeds is None else tuple(args.seeds)
    t0 = time.time()
    n = 0
    with open(args.out, "a") as f:
        for seed in seeds:
            for family in FAMILIES:
                for arm in ARMS:
                    if (seed, family, arm) in done:
                        continue
                    rec = run_episode_record(seed, family, arm)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    n += 1
                    print(f"seed {seed} {family:13s} arm {arm}  steps {rec['n_steps']}  "
                          f"flagged {rec['flagged']}  [{time.time() - t0:.0f}s, {n} run]",
                          flush=True)


# ---------------------------------------------------------------------------
# Analysis -- the pre-declared contrasts only
# ---------------------------------------------------------------------------


def paired_interval(a, b, rng):
    """Mean of a - b over seeds with a 95% pointwise paired-bootstrap interval."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    idx = rng.integers(0, len(d), size=(BOOTSTRAP_RESAMPLES, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"estimate": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(d)),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def analyze(records):
    by = {(r["seed"], r["family"], r["arm"]): r for r in records}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = {"spec": SPEC, "interval": "95% pointwise paired bootstrap, no familywise control",
           "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
           "families": {}}
    n_intervals = 0
    for family in FAMILIES:
        seeds = sorted({s for (s, f, _) in by if f == family
                        and all((s, family, a) in by for a in ARMS)})
        fam = {"n_seeds": len(seeds), "per_arm": {}, "contrasts": {}}
        for arm in ARMS:
            rs = [by[(s, family, arm)] for s in seeds]
            fam["per_arm"][arm] = {
                "breach_rate": float(np.mean([r["breach"] for r in rs])),
                "attrition_rate": float(np.mean([r["attrition_rate"] for r in rs])),
                "blue_score": float(np.mean([r["blue_score"] for r in rs])),
                "flagged": sum(r["flagged"] for r in rs),
                # Detection and coordination: RESULTS, reported beside the
                # outcomes, never a validity gate (19, Amendment 1.3).
                "episodes_with_any_detection": float(np.mean([r["detected_any"] for r in rs])),
                "fraction_agent_steps_believing": float(np.mean(
                    [r["fraction_agent_steps_believing"] for r in rs])),
                "conflicted_task_epochs": float(np.mean(
                    [r["allocation"]["conflicted_task_epochs"] for r in rs])),
                "epochs_with_unassigned_believed_threat": float(np.mean(
                    [r["epochs_with_unassigned_believed_threat"] for r in rs])),
                "mean_rounds_to_stable": float(np.mean(
                    [x for r in rs for x in r["rounds_to_stable"]] or [np.nan])),
                "unsettled_at_epoch_end": sum(r["allocation"]["unsettled_at_epoch_end"] for r in rs),
                "recommitments_technical": sum(r["allocation"]["recommitments"] for r in rs),
                "commitment_changes_per_agent_episode": float(np.mean(
                    [sum(r["commitment_changes_per_agent"].values()) / 4.0 for r in rs])),
                "posture_outside_tolerance": sum(r["posture"]["outside_tolerance"] for r in rs),
                "posture_tracking_violations": sum(r["posture"]["tracking_violations"] for r in rs),
            }
        for name, (a, b) in CONTRASTS.items():
            c = {}
            for outcome in OUTCOMES:
                xa = [float(by[(s, family, a)][outcome]) for s in seeds]
                xb = [float(by[(s, family, b)][outcome]) for s in seeds]
                c[outcome] = paired_interval(xa, xb, rng)
                n_intervals += 1
                unflagged = [s for s in seeds
                             if not by[(s, family, a)]["flagged"] and not by[(s, family, b)]["flagged"]]
                if len(unflagged) < len(seeds):
                    c[outcome + "_excluding_flagged"] = paired_interval(
                        [float(by[(s, family, a)][outcome]) for s in unflagged],
                        [float(by[(s, family, b)][outcome]) for s in unflagged], rng)
            c["breach_discordant"] = {
                f"arm{a}_only": sum(by[(s, family, a)]["breach"] and not by[(s, family, b)]["breach"] for s in seeds),
                f"arm{b}_only": sum(by[(s, family, b)]["breach"] and not by[(s, family, a)]["breach"] for s in seeds),
            }
            # Descriptive: the detection difference, reported next to the outcome.
            c["detection_difference"] = paired_interval(
                [float(by[(s, family, a)]["detected_any"]) for s in seeds],
                [float(by[(s, family, b)]["detected_any"]) for s in seeds], rng)
            fam["contrasts"][name] = c
        out["families"][family] = fam
    out["n_outcome_intervals"] = n_intervals
    return out


def cmd_analyze(args):
    with open(args.episodes) as f:
        records = [json.loads(line) for line in f]
    result = analyze(records)
    text = json.dumps(result, indent=1)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="override ONLY for checking the runner on spent seeds")
    a = sub.add_parser("analyze")
    a.add_argument("--episodes", required=True)
    a.add_argument("--out", default=None)
    args = ap.parse_args()
    {"run": cmd_run, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
