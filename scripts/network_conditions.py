"""Exp 5: does the coordination effect survive restricted information and delivery?

Design, frozen with its amendments before any episode:
 This script implements it and
adds nothing to it.

    arm I  independent  allocation="cbaa", cbaa_transport="none"
    arm C  CBAA         cbaa_transport="network"

Both arms: ``networked`` belief over the SAME Network (range, loss model and
counter-based draws), comms every 0.1 s, one slot, profile "balanced", the station-heading rule
outward posture, the communication-round schedule station fallback, 4 blues / 2 reds, 400 steps, planning
1.0 s, the safety-parameter choice CBF. Six communication conditions (REF, MH, IP, LI, LB, IPL) plus
the NX anchor (``local`` belief, transport ``none``, one arm): "no
target-observation or auction-message exchange", not zero communication
(Amendment 3).

Loss is a per-LINK, per-ROUND erasure shared by every message on the link, in
both directions (Amendment 1). LI and LB share a NOMINAL stationary mean of
0.3. The realised erasure is measured and reported, never tuned (Amendment 2).

Subcommands:

    calibrate  23 Sec 5: arms I and C, both families, R in {8, 10, 12, 15, 20}
               (then 6 and 4 if the declared rule asks), spent seeds 3705-3714.
               Records communication and allocation measures ONLY: no mission
               outcome is recorded at all, so none can be read.
    select     applies 23 Sec 5's rule to a calibration file.
    identity   runs spent-seed episodes sequentially and in a process pool and
               checks the records are identical, before any parallel batch.
    run        the batch: one JSONL line per episode, appended as it finishes,
               resumable, every episode kept.
    analyze    the pre-declared contrasts from that file, and nothing else.

Free: rule-based throughout, no network calls.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.episode_builder import _Quiet, build_episode  # noqa: E402
from scripts.arm_variants import BASE_CFG as EXP4_BASE_CFG  # noqa: E402
from scripts.arm_variants import coordination_extras  # noqa: E402
from scripts.auction_smoke import (  # noqa: E402
    attach_bearing_log,
    attach_ledger,
    attach_timing_log,
    executed,
    measure,
    measure_fallback,
    measure_posture,
    measure_timing,
)

SPEC = ""
FAMILIES = ("symmetric", "multi_bearing")
INF = float("inf")

BASE_CFG = dict(
    EXP4_BASE_CFG,                  # 4v2, 400 steps, planning 1 s, balanced, the station-heading rule outward
    comm_interval_seconds=0.1,      # the communication-round schedule(e): fixed
    cbaa_slots=1,                   # 21 C5
    cbaa_fallback="station",        # the communication-round schedule, every arm
)
ARMS = {
    "I": dict(cbaa_transport="none"),
    "C": dict(cbaa_transport="network"),
}
IID_03 = {"model": "iid", "p": 0.3}
GE_03 = {"model": "gilbert_elliott", "p_gb": 0.15, "p_bg": 0.30,
         "loss_good": 0.0, "loss_bad": 0.9}

# FROZEN by 23 Amendments 4-5 and the range-calibration rule, on calibration
# records only (no outcome recorded): R_MH by 23 Sec 5's rule as declared
# (connected 0.801, diameter 1.97); R_IP by the range-calibration rule's re-calibration inside the
# connectivity cliff (nothing lies between 0.025 at R = 13.5 and 0.745 at
# R = 14, so the closest to 0.50 other than R_MH was taken, as the range-calibration rule declares).
R_MH = 15.0
R_IP = 14.0


def conditions():
    return {
        "REF": dict(r_comm=INF, comm_loss_model=None),
        "MH": dict(r_comm=R_MH, comm_loss_model=None),
        "IP": dict(r_comm=R_IP, comm_loss_model=None),
        "LI": dict(r_comm=INF, comm_loss_model=IID_03),
        "LB": dict(r_comm=INF, comm_loss_model=GE_03),
        "IPL": dict(r_comm=R_IP, comm_loss_model=IID_03),
    }


CONTRAST_CONDITIONS = ("REF", "MH", "IP", "LI", "LB", "IPL")
S_CONDITIONS = ("MH", "IP", "LI", "LB", "IPL")
ANCHOR = "NX"
BATCH_SEEDS = tuple(range(3901, 4001))
CALIBRATION_SEEDS = tuple(range(3705, 3715))  # spent
CALIBRATION_RANGES = (8.0, 10.0, 12.0, 15.0, 20.0)
CALIBRATION_FALLBACK_RANGES = (6.0, 4.0)
OUTCOMES = ("breach", "attrition_rate", "blue_score")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260928


# ---------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------


def episode_cfg(family, condition, arm, r_comm=None, loss=None):
    """``(belief_mode, cfg)``. ``condition`` is a key of ``conditions()``, the
    anchor ``NX``, or ``"CAL"`` with an explicit range (calibration)."""
    if condition == ANCHOR:
        return "local", dict(BASE_CFG, family=family, cbaa_transport="none",
                             r_comm=INF, comm_loss_model=None)
    if condition == "CAL":
        spec = dict(r_comm=float(r_comm), comm_loss_model=loss)
    else:
        spec = dict(conditions()[condition])
    if spec["r_comm"] is None:
        raise SystemExit(f"condition {condition} needs R_MH / R_IP, which are frozen only "
                         f"by 23's calibration amendment")
    return "networked", dict(BASE_CFG, family=family, **ARMS[arm], **spec)


def attach_channel_log(sim):
    """Amendment 2's channel-level realised erasure: at every round, for every
    link in that round's graph, the loss model's own verdict for that link and
    round. The model is a pure function of (seed, link, round), and it is the
    episode's own instance, so evaluating it here changes no draw. Installed on
    ``_record_connectivity``, which runs right after ``begin_round``."""
    log = {"link_rounds": 0, "erased": 0, "accepted": 0}
    original = sim._record_connectivity

    def count_accepted(net):
        send = net.send

        def counted(envelope):
            accepted = send(envelope)
            log["accepted"] += accepted
            return accepted

        net.send = counted

    def record():
        net = sim.network
        if not getattr(net, "_exp5_counted", False):
            # The Network exists only once the episode runs; this is its
            # first round, before anything is sent.
            count_accepted(net)
            net._exp5_counted = True
        for edge in net.edges:
            i, j = sorted(edge)
            if i in net.disabled or j in net.disabled:
                continue
            log["link_rounds"] += 1
            log["erased"] += not net.loss_model.delivers(net.seed, i, j, net.comm_tick)
        return original()

    sim._record_connectivity = record
    return log


def resolved_target_holding(sim, ledger):
    """23 Sec 9.4, the communication-round schedule's "resolved target behaviour", counted directly.

    Active agent-steps whose commitment in force is a contact task
    (NEUTRALIZE / SCREEN) on a contact that had ALREADY resolved by the
    previous recorded step: ``agent_steps``. Steps where the contact resolved
    during that very step are counted apart (``at_resolution_step``), since the
    agent was engaging a live contact when the step began. ``executed`` counts
    those steps (either kind) where the controller still executed the task,
    e.g. screening a resolved contact it still believes in. The smoke's
    execution measure cannot show those: it counts them as executed. So
    ``agent_steps + at_resolution_step - executed`` equals its
    ``target_resolved`` non-execution count, which the tests check.
    """
    by_step = {}
    for entry in ledger:
        by_step.setdefault(entry["step"], []).append(entry)
    current, prev = {}, None
    out = {"agent_steps": 0, "at_resolution_step": 0, "executed": 0}
    by_kind = Counter()
    for st in sim.recorder.steps:
        for entry in by_step.get(st["step"], []):
            current = dict(entry["commitments"])
        for agent, dead in enumerate(st["blue_disabled"]):
            task = current.get(agent)
            if dead or task is None or task[0] not in ("NEUTRALIZE", "SCREEN"):
                continue
            if st["red_outcomes"][task[1]] == "active":
                continue
            earlier = prev is not None and prev["red_outcomes"][task[1]] != "active"
            if earlier:
                out["agent_steps"] += 1
                by_kind[task[0]] += 1
            else:
                out["at_resolution_step"] += 1
            out["executed"] += executed(task, st["effective_targets"].get(str(agent)))
        prev = st
    out["by_kind"] = dict(by_kind)
    return out


def belief_age_at_planning(sim):
    interval = sim.planning_interval_steps
    ages = [age for s in sim.recorder.steps if s["step"] % interval == 0
            for per in s["beliefs"].values() for age in per.values()]
    return {"mean": float(np.mean(ages)) if ages else None,
            "p90": float(np.percentile(ages, 90)) if ages else None,
            "n": len(ages)}


def run_episode_record(seed, family, condition, arm, r_comm=None, loss=None,
                       with_outcomes=True, return_sim=False):
    """One episode's record. ``return_sim=True`` also returns the finished
    Simulation, so an inspection can read the trajectories of exactly the
    episode the record describes."""
    belief_mode, cfg = episode_cfg(family, condition, arm, r_comm, loss)
    sim = build_episode(seed, belief_mode, cfg)
    ledger = attach_ledger(sim)
    bearing_log = attach_bearing_log(sim)
    timing_log = attach_timing_log(sim)
    channel = attach_channel_log(sim) if belief_mode == "networked" else None
    t0 = time.monotonic()
    with _Quiet():
        sim.run()
    wall = time.monotonic() - t0
    m = measure(sim, ledger)
    timing = measure_timing(sim, timing_log, ledger)
    stats = dict(sim.network.stats) if sim.network is not None else None
    rec = {
        "seed": seed, "family": family, "condition": condition, "arm": arm,
        "belief_mode": belief_mode,
        "r_comm": None if math.isinf(cfg["r_comm"]) else cfg["r_comm"],
        "loss": cfg.get("comm_loss_model"),
        "n_steps": m["n_steps"],
        "flagged": bool(m["operational"]["solver_failed"] or m["operational"]["collided"]),
        "solver_failed": bool(m["operational"]["solver_failed"]),
        "collided": bool(m["operational"]["collided"]),
        # Realised communication (8): reported per arm, never assumed equal.
        "comm": sim.comm_metrics(),
        "network_stats": stats,
        "channel": None if channel is None else dict(
            channel, erased_fraction=(channel["erased"] / channel["link_rounds"]
                                      if channel["link_rounds"] else None)),
        "belief_age_at_planning": belief_age_at_planning(sim),
        # Allocation (8).
        "allocation": m["allocation"],
        "rounds_to_stable": m["rounds_to_stable"],
        "unsettled_at_epoch_end": timing["unsettled_at_epoch_end"],
        "longest_unsettled_run_per_agent": timing["longest_unsettled_run_per_agent"],
        "x_conflicts_by_kind": timing["x_conflicts_by_kind"],
        "commitment_conflicts_by_kind": timing["commitment_conflicts_by_kind"],
        "stale_dropped_by_transport": timing["stale_dropped_by_transport"],
        "stale_dropped_by_agents": timing["stale_dropped_by_agents"],
        "foreign_entries_ignored": timing["foreign_entries_ignored"],
        "relay_applicable": timing["relay_applicable"],
        "relay": timing["relay"],
        # Execution (8).
        "commitment_mix": m["commitment_mix"],
        "executed": m["executed"],
        "not_executed": m["not_executed"],
        "not_executed_causes": m["not_executed_causes"],
        "midepoch_change_propagation": m["midepoch_change_propagation"],
        "resolved_target_holding": resolved_target_holding(sim, ledger),
        "fallback_agent_steps": timing["agent_steps_no_commitment"],
        "fallback": measure_fallback(sim, ledger, bearing_log),
        "posture": measure_posture(sim, bearing_log),
        # Detection (8): a result, never a gate.
        "detected_any": m["first_detection_step"] is not None,
        "first_belief_step": m["first_belief_step"],
        "fraction_agent_steps_believing": (
            m["agent_steps_believing_a_contact"] / max(1, m["active_agent_steps"])),
        "wall_s": round(wall, 2),
    }
    if channel is not None:
        # Envelope-level drop fraction (Amendment 2 ii): among (envelope,
        # receiver) attempts over an existing link, the share the channel
        # erased. It weights by traffic, so it differs between arms.
        attempts = stats["dropped_loss"] + channel["accepted"]
        rec["envelope_drop_fraction"] = stats["dropped_loss"] / attempts if attempts else None
    if with_outcomes:
        outcome = sim.episode_outcome()
        rec.update({
            "breach": bool(outcome["touchdown"]),
            "attrition_rate": float(outcome["attrition_rate"]),
            "blue_score": float(outcome["blue_score"]),
            "num_neutralized": int(outcome["num_neutralized"]),
            "num_escaped": int(outcome["num_escaped"]),
            **coordination_extras(sim, ledger),
        })
    return (rec, sim) if return_sim else rec


def _job(args):
    return run_episode_record(*args[:4], **args[4])


def _run_jobs(jobs, workers, sink):
    """Run jobs sequentially (workers == 1) or in a process pool. Results are
    handed to ``sink`` as they finish; records carry their own keys, so order
    does not matter."""
    if workers <= 1:
        for job in jobs:
            sink(_job(job))
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_job, job) for job in jobs]
        for f in as_completed(futures):
            sink(f.result())


# ---------------------------------------------------------------------------
# Calibration (23 Sec 5)
# ---------------------------------------------------------------------------


def _cal_jobs(ranges, seeds):
    return [(seed, family, "CAL", arm, dict(r_comm=r, with_outcomes=False))
            for r in ranges for family in FAMILIES for arm in ARMS for seed in seeds]


def cmd_calibrate(args):
    seeds = CALIBRATION_SEEDS if args.seeds is None else tuple(args.seeds)
    done = _load_keys(args.out, ("seed", "family", "r_comm", "arm"))
    t0 = time.time()
    with open(args.out, "a") as f:
        def sink(rec):
            assert "breach" not in rec, "calibration must not record outcomes"
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"R={rec['r_comm']:>5} {rec['family']:13s} {rec['arm']} seed {rec['seed']}  "
                  f"connected {rec['comm']['connected_fraction']:.3f}  [{time.time() - t0:.0f}s]",
                  flush=True)

        ranges = CALIBRATION_RANGES
        jobs = [j for j in _cal_jobs(ranges, seeds) if (j[0], j[1], j[4]["r_comm"], j[3]) not in done]
        _run_jobs(jobs, args.workers, sink)
        sel = select(_load(args.out))
        if sel["fallback_needed"]:
            print("no calibrated range below 0.65: adding R = 6 and 4, as declared", flush=True)
            jobs = [j for j in _cal_jobs(CALIBRATION_FALLBACK_RANGES, seeds)
                    if (j[0], j[1], j[4]["r_comm"], j[3]) not in done]
            _run_jobs(jobs, args.workers, sink)
    print(json.dumps(select(_load(args.out)), indent=1))


def select(records):
    """23 Sec 5's rule, applied mechanically.

    Statistic: connected fraction, mean over both arms and both families (each
    cell weighted equally: the mean of the four per-(family, arm) means).
    R_MH: closest to 0.90 among ranges whose mean diameter while connected is
    >= 1.5. R_IP: closest to 0.50. Ties to the larger range. If no calibrated
    range is below 0.65, R = 6 and 4 are added and the rule re-applied.
    """
    by = {}
    for r in records:
        by.setdefault(r["r_comm"], {}).setdefault((r["family"], r["arm"]), []).append(r)
    table = {}
    for rng in sorted(by):
        cells = by[rng]
        per_cell = {}
        for (family, arm), rs in sorted(cells.items()):
            diam = [x["comm"]["mean_diameter_when_connected"] for x in rs
                    if x["comm"]["mean_diameter_when_connected"] == x["comm"]["mean_diameter_when_connected"]]
            per_cell[f"{family}/{arm}"] = {
                "n": len(rs),
                "connected_fraction": float(np.mean([x["comm"]["connected_fraction"] for x in rs])),
                "mean_partition_steps": float(np.mean([x["comm"]["mean_partition_steps"] for x in rs])),
                "mean_diameter_when_connected": float(np.mean(diam)) if diam else None,
                "unsettled_agent_epochs": int(sum(x["unsettled_at_epoch_end"] for x in rs)),
                "commitment_conflicted_task_epochs": int(sum(
                    x["allocation"]["commitment_conflicted_task_epochs"] for x in rs)),
                "claim_conflicted_task_epochs": int(sum(
                    x["allocation"]["conflicted_task_epochs"] for x in rs)),
            }
        cf = float(np.mean([c["connected_fraction"] for c in per_cell.values()]))
        diams = [c["mean_diameter_when_connected"] for c in per_cell.values()
                 if c["mean_diameter_when_connected"] is not None]
        table[rng] = {
            "connected_fraction": cf,
            "mean_diameter_when_connected": float(np.mean(diams)) if diams else None,
            "mean_partition_steps": float(np.mean([c["mean_partition_steps"] for c in per_cell.values()])),
            "per_cell": per_cell,
        }
    calibrated = [r for r in table if r in CALIBRATION_RANGES + CALIBRATION_FALLBACK_RANGES]
    fallback_needed = (all(table[r]["connected_fraction"] >= 0.65 for r in calibrated)
                       and not any(r in table for r in CALIBRATION_FALLBACK_RANGES))

    def closest(target, eligible):
        if not eligible:
            return None
        # Closest first, then the larger range on a tie.
        # Rounded so that a tie in exact arithmetic is a tie in floating point
        # too (|0.45 - 0.5| and |0.55 - 0.5| differ in the last bit).
        return sorted(eligible, key=lambda r: (round(abs(table[r]["connected_fraction"] - target), 9),
                                               -r))[0]

    mh_eligible = [r for r in calibrated if (table[r]["mean_diameter_when_connected"] or 0) >= 1.5]
    return {
        "rule": "23 Sec 5",
        "table": {str(r): table[r] for r in sorted(table)},
        "R_MH": closest(0.90, mh_eligible),
        "R_MH_eligible": sorted(mh_eligible),
        "R_IP": closest(0.50, calibrated),
        "fallback_needed": fallback_needed,
    }


def cmd_select(args):
    print(json.dumps(select(_load(args.calibration)), indent=1))


# ---------------------------------------------------------------------------
# Sequential / parallel identity
# ---------------------------------------------------------------------------


def _canonical(rec):
    rec = dict(rec)
    rec.pop("wall_s", None)
    return json.dumps(rec, sort_keys=True)


def cmd_identity(args):
    """The same episodes, sequential in this process and in a pool, must give
    identical records (wall time aside)."""
    seeds = tuple(args.seeds)
    jobs = [(seed, family, "CAL", arm, dict(r_comm=r, loss=loss))
            for seed in seeds for family in FAMILIES for arm in ARMS
            for r, loss in ((12.0, IID_03), (INF, GE_03))]
    seq, par = {}, {}

    def key(rec):
        return (rec["seed"], rec["family"], rec["arm"], rec["r_comm"], json.dumps(rec["loss"]))

    t0 = time.time()
    _run_jobs(jobs, 1, lambda r: seq.__setitem__(key(r), _canonical(r)))
    t1 = time.time()
    _run_jobs(jobs, args.workers, lambda r: par.__setitem__(key(r), _canonical(r)))
    t2 = time.time()
    same = set(seq) == set(par) and all(seq[k] == par[k] for k in seq)
    out = {"episodes": len(jobs), "seeds": list(seeds), "workers": args.workers,
           "identical": same, "sequential_s": round(t1 - t0, 1), "parallel_s": round(t2 - t1, 1),
           "differing": [list(map(str, k)) for k in seq if seq[k] != par.get(k)]}
    print(json.dumps(out, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    if not same:
        raise SystemExit("SEQUENTIAL AND PARALLEL RECORDS DIFFER")


# ---------------------------------------------------------------------------
# The batch
# ---------------------------------------------------------------------------


def batch_jobs(seeds):
    jobs = []
    for seed in seeds:
        for family in FAMILIES:
            for condition in CONTRAST_CONDITIONS:
                for arm in ARMS:
                    jobs.append((seed, family, condition, arm, {}))
            jobs.append((seed, family, ANCHOR, "N", {}))
    return jobs


def cmd_run(args):
    if R_MH is None or R_IP is None:
        raise SystemExit("R_MH / R_IP are not frozen yet: run calibrate, apply 23 Sec 5, "
                         "and record the amendment first")
    seeds = BATCH_SEEDS if args.seeds is None else tuple(args.seeds)
    done = _load_keys(args.out, ("seed", "family", "condition", "arm"))
    jobs = [j for j in batch_jobs(seeds) if (j[0], j[1], j[2], j[3]) not in done]
    t0 = time.time()
    n = [0]
    with open(args.out, "a") as f:
        def sink(rec):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n[0] += 1
            print(f"seed {rec['seed']} {rec['family']:13s} {rec['condition']:3s} {rec['arm']}  "
                  f"steps {rec['n_steps']}  flagged {rec['flagged']}  "
                  f"[{time.time() - t0:.0f}s, {n[0]}/{len(jobs)}]", flush=True)

        _run_jobs(jobs, args.workers, sink)


# ---------------------------------------------------------------------------
# Analysis -- the pre-declared contrasts only (23 Sec 7)
# ---------------------------------------------------------------------------


def _interval(d, rng):
    d = np.asarray(d, dtype=float)
    idx = rng.integers(0, len(d), size=(BOOTSTRAP_RESAMPLES, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"estimate": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(d)),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def _mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return float(np.mean(xs)) if xs else None


def cell_summary(rs):
    def total(field):
        out = Counter()
        for r in rs:
            out.update(r[field])
        return dict(sorted(out.items()))

    comm = [r["comm"] for r in rs if r["comm"]]
    return {
        "n": len(rs),
        "breach_rate": _mean([r["breach"] for r in rs]),
        "attrition_rate": _mean([r["attrition_rate"] for r in rs]),
        "blue_score": _mean([r["blue_score"] for r in rs]),
        "flagged": sum(r["flagged"] for r in rs),
        "comm": None if not comm else {
            "connected_fraction": _mean([c["connected_fraction"] for c in comm]),
            "mean_lcc_fraction": _mean([c["mean_lcc_fraction"] for c in comm]),
            "n_partitions": _mean([c["n_partitions"] for c in comm]),
            "mean_partition_steps": _mean([c["mean_partition_steps"] for c in comm]),
            "mean_diameter_when_connected": _mean([c["mean_diameter_when_connected"] for c in comm]),
        },
        "channel_erased_fraction": _mean([(r["channel"] or {}).get("erased_fraction") for r in rs]),
        "channel_erased_fraction_range": (
            [min(v), max(v)] if (v := [r["channel"]["erased_fraction"] for r in rs
                                       if r["channel"] and r["channel"]["erased_fraction"] is not None])
            else None),
        "envelope_drop_fraction": _mean([r.get("envelope_drop_fraction") for r in rs]),
        "belief_age_at_planning_mean": _mean([r["belief_age_at_planning"]["mean"] for r in rs]),
        "unsettled_agent_epochs": sum(r["unsettled_at_epoch_end"] for r in rs),
        "longest_unsettled_run_max": max((n for r in rs for n in r["longest_unsettled_run_per_agent"].values()),
                                         default=0),
        "mean_rounds_to_stable": _mean([x for r in rs for x in r["rounds_to_stable"]]),
        "claim_conflicts_by_kind": total("x_conflicts_by_kind"),
        "commitment_conflicts_by_kind": total("commitment_conflicts_by_kind"),
        "recommitments_technical": sum(r["allocation"]["recommitments"] for r in rs),
        "foreign_entries_ignored": sum(r["foreign_entries_ignored"] for r in rs),
        "stale_dropped_transport": sum(r["stale_dropped_by_transport"] for r in rs),
        "stale_dropped_agents": sum(r["stale_dropped_by_agents"] for r in rs),
        "relay": total("relay"),
        "not_executed_causes": total("not_executed_causes"),
        "midepoch_change_propagation": total("midepoch_change_propagation"),
        "resolved_target_holding_agent_steps": sum(r["resolved_target_holding"]["agent_steps"] for r in rs),
        "fallback_agent_steps": sum(r["fallback_agent_steps"] for r in rs),
        "episodes_with_any_detection": _mean([r["detected_any"] for r in rs]),
        "fraction_agent_steps_believing": _mean([r["fraction_agent_steps_believing"] for r in rs]),
        "first_belief_step_median": (float(np.median(f)) if (f := [r["first_belief_step"] for r in rs
                                                                  if r["first_belief_step"] is not None])
                                     else None),
        "posture_outside_tolerance": sum(r["posture"]["outside_tolerance"] for r in rs),
    }


def analyze(records):
    by = {(r["seed"], r["family"], r["condition"], r["arm"]): r for r in records}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = {"spec": SPEC, "frozen": {"R_MH": R_MH, "R_IP": R_IP},
           "interval": "95% POINTWISE paired bootstrap over seeds, no familywise control",
           "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
           "families": {}}
    counts = Counter()
    for family in FAMILIES:
        cells = [(c, a) for c in CONTRAST_CONDITIONS for a in ARMS]
        seeds = sorted({s for (s, f, _, _) in by if f == family
                        and all((s, family, c, a) in by for c, a in cells)})
        fam = {"n_seeds": len(seeds), "cells": {}, "anchor": None,
               "C1": {}, "S": {}, "D": {}}

        def x(s, c, a, o):
            return float(by[(s, family, c, a)][o])

        for c, a in cells:
            fam["cells"][f"{c}/{a}"] = cell_summary([by[(s, family, c, a)] for s in seeds])
        nx = [by[(s, family, ANCHOR, "N")] for s in seeds if (s, family, ANCHOR, "N") in by]
        fam["anchor"] = cell_summary(nx) if nx else None

        clean = [s for s in seeds if not any(by[(s, family, c, a)]["flagged"] for c, a in cells)]
        for c in CONTRAST_CONDITIONS:
            res = {}
            for o in OUTCOMES:
                d = [x(s, c, "C", o) - x(s, c, "I", o) for s in seeds]
                res[o] = _interval(d, rng)
                counts["C1_breach" if o == "breach" else "secondary"] += 1
                if len(clean) < len(seeds):
                    res[o + "_excluding_flagged"] = _interval(
                        [x(s, c, "C", o) - x(s, c, "I", o) for s in clean], rng)
            res["breach_discordant"] = {
                "C_only": sum(by[(s, family, c, "C")]["breach"] and not by[(s, family, c, "I")]["breach"]
                              for s in seeds),
                "I_only": sum(by[(s, family, c, "I")]["breach"] and not by[(s, family, c, "C")]["breach"]
                              for s in seeds),
            }
            fam["C1"][c] = res
        for c in S_CONDITIONS:
            res = {}
            for o in OUTCOMES:
                # Per seed: (C - I under c) - (C - I under REF). One resampling
                # of seeds carries all four cells jointly.
                d = [(x(s, c, "C", o) - x(s, c, "I", o)) - (x(s, "REF", "C", o) - x(s, "REF", "I", o))
                     for s in seeds]
                res[o] = _interval(d, rng)
                counts["S_breach_PRIMARY" if o == "breach" else "secondary"] += 1
                if len(clean) < len(seeds):
                    res[o + "_excluding_flagged"] = _interval(
                        [(x(s, c, "C", o) - x(s, c, "I", o)) - (x(s, "REF", "C", o) - x(s, "REF", "I", o))
                         for s in clean], rng)
            fam["S"][c] = res
            # D(c): descriptive POINT estimates only; no interval is declared.
            fam["D"][c] = {a: {o: float(np.mean([x(s, c, a, o) - x(s, "REF", a, o) for s in seeds]))
                               for o in OUTCOMES} for a in ARMS}
        fam["n_seeds_unflagged_all_cells"] = len(clean)
        out["families"][family] = fam
    out["n_intervals"] = dict(counts)
    return out


def cmd_analyze(args):
    result = analyze(_load(args.episodes))
    text = json.dumps(result, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)


# ---------------------------------------------------------------------------


def _load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_keys(path, fields):
    return {tuple(r[k] for k in fields) for r in _load(path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--out", required=True)
    c.add_argument("--workers", type=int, default=1)
    c.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="override ONLY for checking the runner")
    s = sub.add_parser("select")
    s.add_argument("--calibration", required=True)
    i = sub.add_parser("identity")
    i.add_argument("--seeds", type=int, nargs="+", default=[3705, 3706])
    i.add_argument("--workers", type=int, default=4)
    i.add_argument("--out", default=None)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="override ONLY for checking the runner on spent seeds")
    a = sub.add_parser("analyze")
    a.add_argument("--episodes", required=True)
    a.add_argument("--out", default=None)
    args = ap.parse_args()
    {"calibrate": cmd_calibrate, "select": cmd_select, "identity": cmd_identity,
     "run": cmd_run, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
