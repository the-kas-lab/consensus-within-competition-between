"""The auction step 6: the rule-based CBAA smoke at full research config.

Declaration, written before any episode ran on these seeds:
, Part 1. The run answers **operational**
questions only:

1. do auctions produce commitments the controller actually executes?
2. how often does an epoch reset before an agent settles?
3. do mid-epoch recommitments reach the controller at the step they land?

**Mission outcomes are saved and not read.** Ten seeds per cell resolve no
outcome contrast, and the bid weights have been frozen since c5457e6. Local
stability is a commitment heuristic, not proof of global convergence, which is
why recommitments and epoch-boundary conflicts are reported alongside it.

Run as a package:

    PYTHONPATH=$PWD python3 -m scripts.auction_smoke --out

Free: rule-based throughout, no network calls.

``--grid timing`` runs the auction-timing check instead
(): the same episode and the same
measurements, at ``networked`` belief with per-cell range, communication period
and transport, plus timing instrumentation attached from outside the testbed.
Its declaration, including a prediction derived from the code, was written
before any episode.

    PYTHONPATH=$PWD python3 -m scripts.auction_smoke --grid timing --station-heading outward \
        --first-seed 3705 --seeds 10 --out
"""

import argparse
import json
import os
import sys
import math
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planning.roles import BlueRole  # noqa: E402
from scripts.episode_builder import _Quiet, build_episode  # noqa: E402

FAMILIES = ("symmetric", "multi_bearing")

# The Exp 4 arms, so the smoke
# exercises exactly what Exp 4 will run.
CELLS = {
    "independent": dict(cbaa_transport="none", cbaa_slots=1),
    "cbaa_s1": dict(cbaa_transport="ideal", cbaa_slots=1),
    "cbaa_s2": dict(cbaa_transport="ideal", cbaa_slots=2),
}

BASE_CFG = dict(
    blues=4,
    reds=2,
    steps=400,
    planning_interval_seconds=1.0,
    rule="nearest",  # built by run_episode, superseded by allocation="cbaa"
    neutralize_fraction=0.5,
    recon_fraction=0.5,
    allocation="cbaa",
    cbaa_profile="balanced",
)

BELIEF_MODE = "shared"


# ---------------------------------------------------------------------------
# The assignment ledger
# ---------------------------------------------------------------------------


def commitment_of(planned, agent):
    """The task one agent is committed to, read back from the PlannerOutput
    that ``coordination.runtime.output_for`` built.

    Read from the output the testbed was handed, before the testbed filters it.
    ``defend_targets`` in the recorder drops a screened contact once it is
    resolved, so a resolved SCREEN would otherwise be indistinguishable from
    "not yet settled". Slot is not recoverable (``target_red_ids`` carries the
    contact only), and nothing here needs it.
    """
    role = planned.roles.get(agent)
    if role == BlueRole.NEUTRALIZE:
        return ("NEUTRALIZE", int(planned.target_red_ids[agent]))
    if role == BlueRole.RECON:
        return ("RECON", int(planned.recon_target_regions[agent]))
    target = planned.defend_targets.get(agent)
    if target is None:
        return None  # DEFEND with no override: no commitment yet
    kind, ident = target
    return ("SCREEN" if kind == "contact" else "DEFEND", int(ident))


def attach_ledger(sim):
    """Record every assignment the testbed puts into force, with its step and
    whether it was a mid-epoch recommitment (``resolve_only`` set).

    Wraps the instance attribute from outside, so the testbed is not modified.
    ``_apply_assignment`` is the right seam: it is the one path through which
    both the planning-step output and ``_apply_cbaa``'s mid-epoch changes enter
    force, and ``_put_assignment_in_force`` was split out of it precisely so a
    wrapper sees exactly one call per application.
    """
    ledger = []
    original = sim._apply_assignment

    def recording(planned, step, believed_targets, resolve_only=None):
        runtime = getattr(sim, "cbaa_runtime", None)
        agents = getattr(runtime, "agents", {}) if runtime is not None else {}

        def committed(a):
            # The communication-round schedule's station fallback puts an explicit DEFEND(station i) into
            # force for an agent that has no commitment. It is a fallback, not
            # a commitment, so the ledger says None, exactly as it does for
            # the legacy DEFEND (which commitment_of already reads as None).
            if a in agents and agents[a].commitment is None:
                return None
            return commitment_of(planned, a)

        ledger.append({
            "step": int(step),
            "mid_epoch": resolve_only is not None,
            "changed": sorted(int(a) for a in (resolve_only or ())),
            "commitments": {int(a): committed(a) for a in planned.roles},
        })
        return original(planned, step, believed_targets, resolve_only=resolve_only)

    sim._apply_assignment = recording
    return ledger


def attach_bearing_log(sim):
    """Per solve: which agents had their own detection-driven bearing target,
    and which heading target the solver actually gave each agent. Wrapped from outside, like the ledger."""
    log = []
    original = sim.solver.solve_decentralized

    def recording(*args, **kwargs):
        out = original(*args, **kwargs)
        log.append((
            sorted(int(a) for a in (kwargs.get("bearing_target_override") or {})),
            {int(a): v for a, v in getattr(sim.solver, "bearing_sources", {}).items()},
        ))
        return out

    sim.solver.solve_decentralized = recording
    return log


# 2 s. Not 1 s, as first declared: the heading state is unwrapped and the
# target atan2-wrapped, so a turn can take the long way round -- up to ~2*pi
# rad, which alone needs >= 1 s at omega_max = 2*pi rad/s, plus the asymptotic
# tail. Found on a dry run on spent seed 2001 (a 5.3 rad turn at 7.9 deg on its
# 10th step, 1.0 deg on its 13th) before any re-smoke episode.
POSTURE_TRANSIENT_STEPS = 20
POSTURE_TOLERANCE_DEG = 5.0


def measure_posture(sim, bearing_log):
    """18 Part 3 criteria 1-3: posture acquired and kept, tracking overrides it,
    and only DEFEND(station) gets it. Headings are compared wrapped; the heading
    state itself is not wrapped by the dynamics."""
    center = np.asarray(sim.defense_center, dtype=float).ravel()[:2]
    steps = sim.recorder.steps
    run, errors = {}, []
    tracking_violations = eligibility_violations = 0
    tracking_while_station = 0
    for s, (override, sources) in enumerate(bearing_log):
        if s >= len(steps):
            break
        st = steps[s]
        for agent, source in sources.items():
            if st["blue_disabled"][agent]:
                continue
            target = st["defend_targets"].get(str(agent))
            station = (target is not None and target[0] == "station"
                       and st["blue_roles"].get(str(agent)) == "defend")
            if agent in override:
                tracking_violations += source != "detection"
                tracking_while_station += station
            elif source == "station_outward" and not station:
                eligibility_violations += 1
            # The transient restarts whenever the posture starts OR the station
            # changes: a move from station k to k' is a new turn, and first
            # measured, counting it as post-transient flagged a 7.9 deg turn
            # in progress as a posture failure.
            key = (source, target[1] if station else None)
            if source == "station_outward" and run.get(agent, (None, 0))[0] == key:
                run[agent] = (key, run[agent][1] + 1)
            else:
                run[agent] = (key, 1 if source == "station_outward" else 0)
            if run[agent][1] >= POSTURE_TRANSIENT_STEPS and s + 1 < len(sim.history_blues):
                pos = np.asarray(sim.solver.defend_stations[int(target[1])], dtype=float)
                outward = math.atan2(*(pos - center)[::-1])
                heading = float(np.asarray(sim.history_blues[s + 1]).ravel()[agent * sim.blue_nx + 4])
                errors.append(abs((math.degrees(heading - outward) + 180.0) % 360.0 - 180.0))
    return {
        "eligible_agent_steps": len(errors),
        "max_error_deg": max(errors) if errors else None,
        "outside_tolerance": sum(e > POSTURE_TOLERANCE_DEG for e in errors),
        "tracking_violations": tracking_violations,
        "tracking_while_committed_to_station": tracking_while_station,
        "eligibility_violations": eligibility_violations,
    }


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def executed(task, eff):
    """Does this step's effective target execute ``task``?"""
    if task is None or eff is None:
        return False
    kind, ident = task
    if kind == "NEUTRALIZE":
        return eff["source"] == "assigned" and eff["contact_id"] == ident
    if kind == "SCREEN":
        return eff["source"] == "screen" and eff["contact_id"] == ident
    if kind == "DEFEND":
        return eff["source"] == "station"
    if kind == "RECON":
        return eff["source"] == "recon_waypoint"
    raise ValueError(kind)


def cause(task, eff, red_outcomes):
    """Why a committed task was not executed, for the breakdown."""
    kind, ident = task
    if kind in ("NEUTRALIZE", "SCREEN") and red_outcomes[ident] != "active":
        return "target_resolved"
    if eff is None:
        return "no_effective_record"
    if eff["source"] == "nearest_believed":
        same = eff["contact_id"] == ident if kind in ("NEUTRALIZE", "SCREEN") else False
        return "fallback_nearest_believed" + ("_same_contact" if same else "")
    if eff["source"] == "station" and kind in ("NEUTRALIZE", "SCREEN"):
        return "no_belief_held_station"
    return f"other:{eff['source']}"


def measure(sim, ledger):
    steps = sim.recorder.steps
    by_step = {}
    for entry in ledger:
        by_step.setdefault(entry["step"], []).append(entry)

    mix = Counter()
    exec_ok = Counter()
    exec_fail = Counter()
    fail_causes = Counter()
    recorded_vs_effective = Counter()
    current = {}
    for st in steps:
        s = st["step"]
        for entry in by_step.get(s, []):  # every application at this step, in order
            current = dict(entry["commitments"])
        effs = st["effective_targets"]
        for agent, dead in enumerate(st["blue_disabled"]):
            if dead:
                continue
            task = current.get(agent)
            kind = "none" if task is None else task[0]
            mix[kind] += 1
            if task is None:
                continue
            eff = effs.get(str(agent))
            if executed(task, eff):
                exec_ok[kind] += 1
            else:
                exec_fail[kind] += 1
                fail_causes[f"{kind}:{cause(task, eff, st['red_outcomes'])}"] += 1

        # Two recorded-vs-effective checks, on the recorder alone.
        for a, c in st["target_red_ids"].items():
            if c is None or st["blue_roles"].get(a) != BlueRole.NEUTRALIZE.value:
                continue
            if st["blue_disabled"][int(a)] or st["red_outcomes"][c] != "active":
                continue
            eff = effs.get(a)
            ok = eff is not None and eff["source"] == "assigned" and eff["contact_id"] == c
            recorded_vs_effective["target_red_ids_agree" if ok else "target_red_ids_differ"] += 1
        for a, (kind, ident) in st["defend_targets"].items():
            if kind != "contact" or st["blue_disabled"][int(a)]:
                continue
            eff = effs.get(a)
            ok = eff is not None and eff["source"] == "screen" and eff["contact_id"] == ident
            recorded_vs_effective["defend_targets_agree" if ok else "defend_targets_differ"] += 1

    # Recommitments: every agent a mid-epoch application changed, checked on the
    # SAME step's effective target (the local round and _apply_cbaa run before
    # the solve the recorder logs).
    step_index = {st["step"]: st for st in steps}
    propagation = Counter()
    for entry in ledger:
        if not entry["mid_epoch"]:
            continue
        st = step_index.get(entry["step"])
        if st is None:
            propagation["step_not_recorded"] += 1
            continue
        for agent in entry["changed"]:
            task = entry["commitments"].get(agent)
            if task is None:
                propagation["changed_to_none"] += 1
                continue
            eff = st["effective_targets"].get(str(agent))
            if executed(task, eff):
                propagation[f"{task[0]}:reached"] += 1
            else:
                propagation[f"{task[0]}:not_reached:{cause(task, eff, st['red_outcomes'])}"] += 1

    # Does the team ever see anything? Added after the dry run on spent seed
    # 2001: under `independent` every blue settled on a DEFEND station facing
    # inward and no red was ever detected, so no contact task could exist. An
    # exercise measure, not an outcome -- it says whether the allocator had
    # anything to allocate.
    active_steps = [
        (st, a) for st in steps for a, dead in enumerate(st["blue_disabled"]) if not dead
    ]
    first_detection = next((st["step"] for st in steps if st["detected_red_union"]), None)
    first_belief = next((st["step"] for st in steps if any(st["beliefs"].values())), None)
    believing = sum(1 for st, a in active_steps if st["beliefs"].get(str(a)))

    outcome = sim.episode_outcome()
    alloc = outcome["allocation"]
    return {
        "first_detection_step": first_detection,
        "first_belief_step": first_belief,
        "agent_steps_believing_a_contact": believing,
        "active_agent_steps": len(active_steps),
        "allocation": alloc,
        "rounds_to_stable": list(sim.cbaa_runtime.stats["rounds_to_stable"]),
        "commitment_mix": dict(mix),
        "executed": dict(exec_ok),
        "not_executed": dict(exec_fail),
        "not_executed_causes": dict(fail_causes),
        "recorded_vs_effective": dict(recorded_vs_effective),
        "midepoch_change_propagation": dict(propagation),
        "n_mid_epoch_applications": sum(1 for e in ledger if e["mid_epoch"]),
        "n_steps": len(steps),
        "operational": {
            "solver_failed": outcome["solver_failed"],
            "collided": bool(sim.collided),
        },
        # Saved, NOT read. See the module docstring.
        "outcome_not_read": {
            k: outcome[k]
            for k in ("touchdown", "num_neutralized", "num_escaped", "blue_score",
                      "attrition_rate", "red_outcomes")
        },
    }


# ---------------------------------------------------------------------------
# Auction timing under slow and finite-range communication
#
# ---------------------------------------------------------------------------

# networked belief throughout, lossless; slots 1. The timing cells isolate the
# communication period at complete connectivity; the range cells run at the one
# period where the auction can settle, 0.1 s.
# The 0.5 s cells are rejected by the communication-round schedule schedule guard (no agent can ever
# reach local stability there) and run only with its diagnostic override: they
# exist to show that failure, which 22 Part 2 did, on 9a7ccb1, before the guard.
TIMING_CELLS = {
    "rinf_p0.1_net": dict(belief_mode="networked", r_comm=float("inf"),
                          comm_interval_seconds=0.1, cbaa_transport="network", cbaa_slots=1),
    "rinf_p0.1_none": dict(belief_mode="networked", r_comm=float("inf"),
                           comm_interval_seconds=0.1, cbaa_transport="none", cbaa_slots=1),
    "rinf_p0.5_net": dict(belief_mode="networked", r_comm=float("inf"),
                          comm_interval_seconds=0.5, cbaa_transport="network", cbaa_slots=1,
                          cbaa_allow_unsettleable=True),
    "rinf_p0.5_none": dict(belief_mode="networked", r_comm=float("inf"),
                           comm_interval_seconds=0.5, cbaa_transport="none", cbaa_slots=1,
                           cbaa_allow_unsettleable=True),
    "r15_p0.1_net": dict(belief_mode="networked", r_comm=15.0,
                         comm_interval_seconds=0.1, cbaa_transport="network", cbaa_slots=1),
    "r20_p0.1_net": dict(belief_mode="networked", r_comm=20.0,
                         comm_interval_seconds=0.1, cbaa_transport="network", cbaa_slots=1),
    "r25_p0.1_net": dict(belief_mode="networked", r_comm=25.0,
                         comm_interval_seconds=0.1, cbaa_transport="network", cbaa_slots=1),
}

CONTACT_TASK_KINDS = ("NEUTRALIZE", "SCREEN")


def attach_timing_log(sim):
    """Wrappers from outside the testbed, like the ledger. Nothing in
    ``simulation/`` or ``coordination/`` changes, so this is default-off by
    construction: a run that does not call it is the run it always was.

    * ``CBAARuntime.start_epoch``: each live agent's settle state is read
      BEFORE the reset, exactly where the runtime samples its own
      ``unsettled_at_epoch_end`` (the two must agree), along with the
      commitments in force at the boundary. After the reset, each agent's task
      set and bids for the new epoch.
    * ``Simulation._record_connectivity``: called right after
      ``Network.begin_round``, so it sees the graph that round's sends use.
      Only exists under ``networked``.
    * ``solve_decentralized``: once per step, after that step's round and any
      epoch start, a snapshot of every agent's commitment and local stability.
    """
    runtime = sim.cbaa_runtime
    log = {"boundaries": [], "epochs": [], "rounds": [], "snapshots": []}

    original_start = runtime.start_epoch

    def start_epoch(step, views, *args, **kwargs):
        if runtime.epoch >= 0:
            live = sorted(int(i) for i in views)
            log["boundaries"].append({
                "step": int(step),
                "epoch": int(runtime.epoch),
                "agents": {
                    i: {
                        "settled": runtime.agents[i].rounds_to_stable is not None,
                        "rounds_to_stable": runtime.agents[i].rounds_to_stable,
                        "rounds": runtime.agents[i]._rounds_this_epoch,
                        "x": runtime.agents[i].x,
                        "commitment": runtime.agents[i].commitment,
                    }
                    for i in live
                },
            })
        out = original_start(step, views, *args, **kwargs)
        log["epochs"].append({
            "step": int(step),
            "epoch": int(runtime.epoch),
            "tasks": {int(i): set(runtime.agents[i].order) for i in views},
            "bids": {int(i): dict(runtime.agents[i].bids) for i in views},
        })
        return out

    runtime.start_epoch = start_epoch

    original_record = sim._record_connectivity

    def record_connectivity():
        net = sim.network
        log["rounds"].append({
            "step": int(net.comm_tick) * int(sim.comm_interval_steps),
            "edges": sorted(tuple(sorted(e)) for e in net.edges),
            "disabled": sorted(net.disabled),
        })
        return original_record()

    sim._record_connectivity = record_connectivity

    original_solve = sim.solver.solve_decentralized

    def solve(*args, **kwargs):
        # Same step alignment as attach_bearing_log: one solve per step.
        agents = getattr(runtime, "agents", {})
        log["snapshots"].append({
            i: (a.commitment, bool(a.stable), a.x) for i, a in agents.items()
        })
        return original_solve(*args, **kwargs)

    sim.solver.solve_decentralized = solve
    return log


def reachable(src, dst, send_rounds, relays):
    """Could ``src``'s auction entry reach ``dst`` over ``send_rounds``?

    One hop per round, as the Network's double buffer enforces: an agent that
    learns the entry in round t sends it in round t, and it lands at t+1. Only
    agents in ``relays`` store an entry, and therefore only they re-send it --
    under the rule that auction traffic never creates a belief an agent that did not generate the task discards its entries and
    leaves it out of its own ``outgoing()`` (17 C3). ``dst`` must be in
    ``relays`` to hear anything; a believer always is.

    ``send_rounds`` is a sequence of edge collections, one per round, each edge
    a 2-tuple, in the order the rounds ran.
    """
    known = {src}
    for edges in send_rounds:
        heard = set()
        for a, b in edges:
            if a in known and b in relays:
                heard.add(b)
            if b in known and a in relays:
                heard.add(a)
        known |= heard  # computed from the pre-round set: one hop per round
        if dst in known:
            return True
    return dst in known


def _winner_first(claimants, bids, task):
    """CBAA's own order: higher bid first, equal bids to the lower id."""
    return sorted(claimants, key=lambda i: (-bids[i].get(task, float("-inf")), i))


def classify_relay(epoch_rec, send_rounds, live, conflicts, claim_since=None):
    """17 C3, per boundary conflict on a contact task, plus the structural
    opportunities for it whether or not a conflict resulted.

    ``epoch_rec``: that epoch's ``tasks`` and ``bids``. ``send_rounds``:
    ``(step, edges)`` for the rounds whose sends were accepted in the epoch.
    ``conflicts``: ``{task: [claimant ids]}`` sampled at the epoch's end.
    ``claim_since``: ``{agent: step}``, the step from which the agent has held
    its end-of-epoch claim without a break. **The winner's claim can only
    travel from then on.** First written without it, the classifier let a
    claim travel from the epoch's start, and a dry run on spent seed 2001
    flagged 40 "impossible" conflicts at 0.5 s: two agents released in the
    epoch's last round and re-selected the same task, a claim neither had
    ever sent.
    """
    tasks, bids = epoch_rec["tasks"], epoch_rec["bids"]
    live = set(live)
    claim_since = claim_since or {}
    all_rounds = [edges for _, edges in send_rounds]
    counts = Counter()
    for task, claimants in conflicts.items():
        if task.split(":")[0] not in CONTACT_TASK_KINDS:
            continue
        ordered = _winner_first(claimants, bids, task)
        winner = ordered[0]
        since = claim_since.get(winner)
        claim_rounds = [edges for step, edges in send_rounds if since is None or step >= since]
        believers = {i for i in live if task in tasks.get(i, ())}
        for loser in ordered[1:]:
            via_believers = reachable(winner, loser, claim_rounds, believers)
            via_all = reachable(winner, loser, claim_rounds, live)
            if via_believers:
                counts["path_via_believers_still_conflicted"] += 1  # should not happen
            elif via_all:
                counts["relay_failure"] += 1
            else:
                counts["no_path_in_time"] += 1
    contact_tasks = {t for i in live for t in tasks.get(i, ())
                     if t.split(":")[0] in CONTACT_TASK_KINDS}
    for task in sorted(contact_tasks):
        believers = sorted(i for i in live if task in tasks.get(i, ()))
        for n, i in enumerate(believers):
            for j in believers[n + 1:]:
                counts["believer_pairs"] += 1
                b = (reachable(i, j, all_rounds, set(believers))
                     or reachable(j, i, all_rounds, set(believers)))
                a = (reachable(i, j, all_rounds, live)
                     or reachable(j, i, all_rounds, live))
                if a and not b:
                    counts["structural_relay_opportunity"] += 1
                if not a:
                    counts["believer_pair_no_path"] += 1
    return counts


def _duplicates(assignments):
    claims = {}
    for agent, task in assignments.items():
        if task is not None:
            claims.setdefault(task, []).append(agent)
    return {t: sorted(a) for t, a in claims.items() if len(a) > 1}


def _kind_counts(conflicts):
    return Counter(t.split(":")[0] for t in conflicts)


def measure_timing(sim, log, ledger):
    runtime = sim.cbaa_runtime
    steps = sim.recorder.steps

    # 1. Resets before settling, per agent-epoch, sampled where the runtime
    #    samples. The wrapper's count must equal the runtime's.
    per_agent = {}
    rounds_per_epoch = Counter()
    for b in log["boundaries"]:
        for i, rec in b["agents"].items():
            per_agent.setdefault(i, []).append(rec["settled"])
            rounds_per_epoch[rec["rounds"]] += 1
    unsettled = sum(not s for seq in per_agent.values() for s in seq)
    longest = {}
    for i, seq in per_agent.items():
        run = best = 0
        for settled in seq:
            run = 0 if settled else run + 1
            best = max(best, run)
        longest[i] = best

    # 2. No commitment at all, and time since the auction last settled, per
    #    active agent-step.
    none_steps = never_settled_steps = active_steps = 0
    since = Counter()
    last_stable = {}
    for s, snap in enumerate(log["snapshots"]):
        if s >= len(steps):
            break
        disabled = steps[s]["blue_disabled"]
        for i, (commitment, stable, _x) in snap.items():
            if stable:
                last_stable[i] = s
            if disabled[i]:
                continue
            active_steps += 1
            none_steps += commitment is None
            if i in last_stable:
                since[s - last_stable[i]] += 1
            else:
                never_settled_steps += 1
    ledger_none = 0
    current = {}
    by_step = {}
    for entry in ledger:
        by_step.setdefault(entry["step"], []).append(entry)
    for st in steps:
        for entry in by_step.get(st["step"], []):
            current = dict(entry["commitments"])
        ledger_none += sum(1 for a, dead in enumerate(st["blue_disabled"])
                           if not dead and current.get(a) is None)

    # 4 (P3). Which effective-target sources the solver actually used.
    sources = Counter(
        eff["source"]
        for st in steps
        for a, eff in st["effective_targets"].items()
        if not st["blue_disabled"][int(a)]
    )

    # 5. Boundary conflicts on x (the runtime's own log) and on commitments in
    #    force, and 17 C3's relay failure.
    x_conflicts = Counter()
    for entry in runtime.conflict_log:
        x_conflicts.update(_kind_counts(entry["conflicts"]))
    commitment_conflicts = Counter()
    for b in log["boundaries"]:
        commitment_conflicts.update(_kind_counts(
            _duplicates({i: r["commitment"] for i, r in b["agents"].items()})))

    relay = Counter()
    interval = sim.comm_interval_steps
    rounds_by_step = {r["step"]: r for r in log["rounds"]}
    epochs_by_number = {e["epoch"]: e for e in log["epochs"]}
    boundary_by_epoch = {b["epoch"]: b for b in log["boundaries"]}
    # Only auction traffic that actually travels can be relayed. Under "none"
    # the Network still carries BELIEF traffic, so its graph exists, but no
    # auction entry crosses it. First written without this gate, the dry run
    # on spent seed 2001 classified 55-68 "none" conflicts as the impossible
    # case, reading the belief graph as an auction graph.
    relay_applicable = bool(rounds_by_step) and runtime.transport == "network"
    if relay_applicable:
        for entry in runtime.conflict_log:
            k = entry["epoch"]
            epoch_rec = epochs_by_number.get(k)
            if epoch_rec is None:
                continue
            start, end = epoch_rec["step"], entry["step"]
            # Sends accepted in epoch k: after its start (the round AT the start
            # step carries epoch k-1 states), delivered by the sampling step.
            send_rounds = [
                (t, rounds_by_step[t]["edges"])
                for t in range(start + 1, end - interval + 1)
                if t in rounds_by_step
            ]
            boundary = boundary_by_epoch[k]
            live = sorted(boundary["agents"])
            # When each claimant's end-of-epoch claim began: walk back from the
            # boundary sample (x before the reset) over the per-step snapshots,
            # which are taken after each step's round and any epoch start.
            claim_since = {}
            for i in live:
                final, held_from = boundary["agents"][i]["x"], end
                for s in range(end - 1, start - 1, -1):
                    if s >= len(log["snapshots"]) or log["snapshots"][s].get(i, (None,) * 3)[2] != final:
                        break
                    held_from = s
                claim_since[i] = held_from
            relay.update(classify_relay(epoch_rec, send_rounds, live, entry["conflicts"],
                                        claim_since=claim_since))

    agents = runtime.agents.values()
    return {
        "agent_epochs_sampled": sum(len(seq) for seq in per_agent.values()),
        "unsettled_at_epoch_end": unsettled,
        "runtime_unsettled_at_epoch_end": runtime.stats["unsettled_at_epoch_end"],
        "longest_unsettled_run_per_agent": {str(i): n for i, n in sorted(longest.items())},
        "epochs_sampled_per_agent": {str(i): len(seq) for i, seq in sorted(per_agent.items())},
        "rounds_per_epoch": {str(k): v for k, v in sorted(rounds_per_epoch.items())},
        "active_agent_steps": active_steps,
        "agent_steps_no_commitment": none_steps,
        "ledger_agent_steps_no_commitment": ledger_none,
        "agent_steps_never_settled": never_settled_steps,
        "steps_since_last_settled": {str(k): v for k, v in sorted(since.items())},
        "stale_dropped_by_transport": runtime.stats["stale_dropped_by_transport"],
        "stale_dropped_by_agents": sum(a.stale_messages_ignored for a in agents),
        "foreign_entries_ignored": sum(a.foreign_entries_ignored for a in agents),
        "effective_sources": dict(sorted(sources.items())),
        "x_conflicts_by_kind": dict(sorted(x_conflicts.items())),
        "commitment_conflicts_by_kind": dict(sorted(commitment_conflicts.items())),
        "relay_applicable": relay_applicable,
        "relay": dict(sorted(relay.items())),
        "comm": sim.comm_metrics(),
    }


def aggregate_timing(records):
    def total(field):
        out = Counter()
        for r in records:
            out.update(r["timing"][field])
        return dict(sorted(out.items(), key=lambda kv: (len(kv[0]), kv[0])))

    t = [r["timing"] for r in records]
    since = Counter()
    for x in t:
        since.update({int(k): v for k, v in x["steps_since_last_settled"].items()})
    settled_steps = sum(since.values())

    def quantile(q):
        if not settled_steps:
            return None
        acc = 0
        for k in sorted(since):
            acc += since[k]
            if acc >= q * settled_steps:
                return k
        return max(since)

    comm = [x["comm"] for x in t if x["comm"]]
    return {
        "agent_epochs_sampled": sum(x["agent_epochs_sampled"] for x in t),
        "unsettled_at_epoch_end": sum(x["unsettled_at_epoch_end"] for x in t),
        "runtime_unsettled_agrees": all(
            x["unsettled_at_epoch_end"] == x["runtime_unsettled_at_epoch_end"] for x in t),
        "longest_unsettled_run_max": max(
            (n for x in t for n in x["longest_unsettled_run_per_agent"].values()), default=0),
        "episodes_with_an_agent_never_settling": sum(
            any(x["longest_unsettled_run_per_agent"][a] == x["epochs_sampled_per_agent"][a]
                and x["epochs_sampled_per_agent"][a] > 0
                for a in x["longest_unsettled_run_per_agent"])
            for x in t),
        "rounds_per_epoch": total("rounds_per_epoch"),
        "active_agent_steps": sum(x["active_agent_steps"] for x in t),
        "agent_steps_no_commitment": sum(x["agent_steps_no_commitment"] for x in t),
        "ledger_agrees_on_no_commitment": all(
            x["agent_steps_no_commitment"] == x["ledger_agent_steps_no_commitment"] for x in t),
        "agent_steps_never_settled": sum(x["agent_steps_never_settled"] for x in t),
        "steps_since_last_settled_median": quantile(0.5),
        "steps_since_last_settled_p95": quantile(0.95),
        "steps_since_last_settled_max": max(since) if since else None,
        "stale_dropped_by_transport": sum(x["stale_dropped_by_transport"] for x in t),
        "stale_dropped_by_agents": sum(x["stale_dropped_by_agents"] for x in t),
        "foreign_entries_ignored": sum(x["foreign_entries_ignored"] for x in t),
        "effective_sources": total("effective_sources"),
        "x_conflicts_by_kind": total("x_conflicts_by_kind"),
        "commitment_conflicts_by_kind": total("commitment_conflicts_by_kind"),
        "relay": total("relay"),
        "connected_fraction": (sum(c["connected_fraction"] for c in comm) / len(comm)) if comm else None,
        "mean_partition_steps": (sum(c["mean_partition_steps"] for c in comm) / len(comm)) if comm else None,
    }


def measure_fallback(sim, ledger, bearing_log):
    """What an agent with NO commitment actually executed.

    Per active agent-step whose ledger commitment is None: the recorded DEFEND
    target, the effective-target source, and the solver's heading source.
    Under ``cbaa_fallback="station"`` every such step should hold the agent's
    own station through the override path (recorded ``["station", i]``, source
    ``station``), with the station-heading rule posture unless its own sensor is tracking. Under
    ``"legacy"`` the recorded target is absent and the source may be
    ``nearest_believed``.
    """
    by_step = {}
    for entry in ledger:
        by_step.setdefault(entry["step"], []).append(entry)
    current = {}
    counts = Counter()
    for s, st in enumerate(sim.recorder.steps):
        for entry in by_step.get(st["step"], []):
            current = dict(entry["commitments"])
        sources = bearing_log[s][1] if s < len(bearing_log) else {}
        for a, dead in enumerate(st["blue_disabled"]):
            if dead or current.get(a) is not None:
                continue
            counts["uncommitted_agent_steps"] += 1
            target = st["defend_targets"].get(str(a))
            own = target is not None and tuple(target) == ("station", a)
            counts["recorded_own_station" if own else f"recorded:{target}"] += 1
            eff = st["effective_targets"].get(str(a)) or {}
            counts[f"source:{eff.get('source')}"] += 1
            counts[f"heading:{sources.get(a)}"] += 1
    return dict(sorted(counts.items()))


def cell_config(cell_spec, family, station_heading):
    """``(belief_mode, cfg)`` for one cell.

    A cell may carry its own ``belief_mode``, ``r_comm`` and
    ``comm_interval_seconds`` (the auction-timing grid,
). Absent, they fall back to
    this smoke's ``BELIEF_MODE`` and to ``build_episode``'s own defaults, so
    the Exp 4 cells build exactly the cfg they always did -- the command-delivery conditions
    ``cfg.get`` pattern. ``tests/test_auction_timing_check.py`` pins that.
    """
    spec = dict(cell_spec)
    belief_mode = spec.pop("belief_mode", BELIEF_MODE)
    return belief_mode, dict(BASE_CFG, family=family, station_heading=station_heading, **spec)


def run_one(seed, family, cell, station_heading="inherited", cells=None, timing=False,
            fallback="legacy"):
    belief_mode, cfg = cell_config((cells or CELLS)[cell], family, station_heading)
    if fallback != "legacy":
        # Only when asked, so the default cfg stays exactly what it always was.
        cfg["cbaa_fallback"] = fallback
    sim = build_episode(seed, belief_mode, cfg)
    ledger = attach_ledger(sim)
    bearing_log = attach_bearing_log(sim)
    timing_log = attach_timing_log(sim) if timing else None
    with _Quiet():
        sim.run()
    out = measure(sim, ledger)
    out["posture"] = measure_posture(sim, bearing_log)
    if timing:
        out["timing"] = measure_timing(sim, timing_log, ledger)
    out["fallback"] = measure_fallback(sim, ledger, bearing_log)
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(records):
    def total(field):
        out = Counter()
        for r in records:
            out.update(r[field])
        return dict(sorted(out.items()))

    rts = [x for r in records for x in r["rounds_to_stable"]]
    unsettled = sum(r["allocation"]["unsettled_at_epoch_end"] for r in records)
    agent_epochs = len(rts) + unsettled
    executed_ = total("executed")
    not_executed = total("not_executed")
    firsts = [r["first_belief_step"] for r in records if r["first_belief_step"] is not None]
    return {
        "episodes": len(records),
        "episodes_with_any_detection": sum(
            r["first_detection_step"] is not None for r in records
        ),
        "first_belief_steps": [r["first_belief_step"] for r in records],
        "median_first_belief_step": (sorted(firsts)[len(firsts) // 2] if firsts else None),
        "fraction_agent_steps_believing": (
            sum(r["agent_steps_believing_a_contact"] for r in records)
            / max(1, sum(r["active_agent_steps"] for r in records))
        ),
        "agent_epochs_sampled": agent_epochs,
        "unsettled_at_epoch_end": unsettled,
        "rounds_to_stable_distribution": dict(sorted(Counter(rts).items())),
        "mean_rounds_to_stable": (sum(rts) / len(rts)) if rts else None,
        "max_rounds_to_stable": max(rts) if rts else None,
        "recommitments": sum(r["allocation"]["recommitments"] for r in records),
        "epochs_with_conflicts": sum(r["allocation"]["epochs_with_conflicts"] for r in records),
        "conflicted_task_epochs": sum(r["allocation"]["conflicted_task_epochs"] for r in records),
        "foreign_entries_ignored": sum(r["allocation"]["foreign_entries_ignored"] for r in records),
        "stale_messages_dropped": sum(r["allocation"]["stale_messages_dropped"] for r in records),
        "auction_messages_sent": sum(r["allocation"]["auction_messages_sent"] for r in records),
        "commitment_mix": total("commitment_mix"),
        "executed": executed_,
        "not_executed": not_executed,
        "executability": {
            k: executed_.get(k, 0) / (executed_.get(k, 0) + not_executed.get(k, 0))
            for k in sorted(set(executed_) | set(not_executed))
        },
        "not_executed_causes": total("not_executed_causes"),
        "recorded_vs_effective": total("recorded_vs_effective"),
        "midepoch_change_propagation": total("midepoch_change_propagation"),
        "n_mid_epoch_applications": sum(r["n_mid_epoch_applications"] for r in records),
        "posture": {
            k: (max((r["posture"][k] for r in records if r["posture"][k] is not None), default=None)
                if k == "max_error_deg" else sum(r["posture"][k] for r in records))
            for k in records[0]["posture"]
        },
        "solver_failures": sum(r["operational"]["solver_failed"] for r in records),
        "collisions": sum(r["operational"]["collided"] for r in records),
        "mean_episode_steps": sum(r["n_steps"] for r in records) / len(records),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first-seed", type=int, default=3705)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--station-heading", default="inherited", choices=["inherited", "outward"],
                    help="the station-heading rule posture; the first smoke (3705-3714) ran 'inherited', "
                         "the re-smoke (3715-3724) 'outward'")
    ap.add_argument("--grid", default="exp4", choices=["exp4", "timing"],
                    help="exp4: the Exp 4 arms at shared belief (18). timing: the "
                         "auction-timing check's networked cells (22), with its "
                         "timing instrumentation attached")
    ap.add_argument("--cells", default=None,
                    help="comma-separated subset of the grid's cells (dry runs)")
    ap.add_argument("--fallback", default="legacy", choices=["legacy", "station"],
                    help="the communication-round schedule: what an agent with no commitment executes (default: "
                         "Simulation's own, the legacy implicit DEFEND)")
    args = ap.parse_args()

    timing = args.grid == "timing"
    grid = TIMING_CELLS if timing else CELLS
    cells = args.cells.split(",") if args.cells else list(grid)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    result = {
        "design": ("" if timing
                   else ""),
        # The timing grid's belief mode and communication parameters are per cell.
        "config": (dict(BASE_CFG, station_heading=args.station_heading) if timing else
                   dict(BASE_CFG, belief_mode=BELIEF_MODE, station_heading=args.station_heading)),
        "cells": {c: grid[c] for c in cells},
        "fallback": args.fallback,
        "seeds": seeds,
        "warning": "operational smoke; mission outcomes are saved under "
                   "outcome_not_read and must not be read as a finding",
        "families": {},
    }
    t0 = time.time()
    for family in FAMILIES:
        result["families"][family] = {}
        for cell in cells:
            records = []
            for seed in seeds:
                rec = run_one(seed, family, cell, args.station_heading, cells=grid, timing=timing,
                              fallback=args.fallback)
                rec["seed"] = seed
                records.append(rec)
                print(f"{family:14s} {cell:12s} seed {seed}  "
                      f"steps {rec['n_steps']}  "
                      f"unsettled {rec['allocation']['unsettled_at_epoch_end']}  "
                      f"recommit {rec['allocation']['recommitments']}  "
                      f"[{time.time() - t0:.0f}s]", flush=True)
            result["families"][family][cell] = {
                "summary": aggregate(records),
                "episodes": records,
            }
            if timing:
                result["families"][family][cell]["timing"] = aggregate_timing(records)
    result["wall_s"] = time.time() - t0

    for family in FAMILIES:
        for cell in cells:
            print(f"\n== {family} / {cell}")
            print(json.dumps(result["families"][family][cell]["summary"], indent=1))
            if timing:
                print(json.dumps(result["families"][family][cell]["timing"], indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
