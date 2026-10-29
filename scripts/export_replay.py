"""Export an episode as a replay bundle for the static viewer in ``viewer/``.

Why this exists
---------------
``EpisodeRecorder`` holds everything a replay needs *per step*, but nothing
saves it: no runner calls ``EpisodeRecorder.save``, and the Exp 4 JSONL keeps
no trajectories. Episodes are deterministic in (seed, config), so this script
re-runs one from its seed and config and writes the recorder's data out with
the three things a replay also needs and the recorder does not hold:

* **static geometry** -- the defended shape, guard polygon, DEFEND stations and
  coverage cells. They are read off the live ``Simulation``, never re-derived,
  so a drawn station is the point the controller actually steers to (the same
  rule ``coordination.tasks.TaskContext.from_simulation`` follows);
* **a terminal frame.** ``record_step`` runs before integration, so the state
  that ended the episode -- the breach, the last kill -- is only in
  ``sim.history_*[-1]``. Without it a viewer cannot show the event it exists to
  jump to;
* **read-only instrumentation**: the auction smoke's ledger and bearing log
  (the full CBAA commitment and the solver's heading source), attached from
  outside exactly as Exp 4 attaches them, plus each communication round's link
  set. It is *not* recorder data and the bundle says so;
* **what the run kept outside the recorder** (format 2): the contact kinds of a
  non-hostile episode,
  the commander's command log with each order classified, and -- only under the
  falsified-report attack -- the believed positions that differ from the ones
  derived from the truth, each tied to a falsified record its holder accepted.

The export never changes simulation behaviour: nothing under ``simulation/`` is
touched, and ``tests/test_export_replay.py`` pins the wrapped episode to the
unwrapped one. Before writing, the bundle is checked against the live episode
(history, beliefs, coverage, neutralization pairs); a mismatch raises instead of
writing a bundle that shows something the episode did not do.

Inventory and design (personal notes).

    PYTHONPATH=$PWD python3 -m scripts.export_replay --preset exp4 --arm 2 \\
        --family symmetric --seed 3715 --out
    # two arms, one bundle -> the viewer opens in compare mode
    PYTHONPATH=$PWD python3 -m scripts.export_replay --preset exp4 --arm 1 --arm 2 \\
        --family multi_bearing --seed 3719 --out

Free: rule-based planners, no network calls.
"""

import argparse
import datetime
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.coverage import CoverageTracker  # noqa: E402
from scripts.episode_builder import _Quiet, build_episode  # noqa: E402
from scripts.auction_smoke import attach_bearing_log, attach_ledger  # noqa: E402
from simulation.recorder import INTERVENTION_TOL, _git_commit  # noqa: E402

FORMAT = "pev-replay"
FORMAT_VERSION = 2
# An order counts as never received only if its addressee was not lost within
# one planning interval of it being generated (the study's missed-order
# analysis uses the same window): a missed order to a defender that was about
# to be lost could not have mattered for long.
ORDER_LOST_WINDOW = 10

# Exp 4 draws from 3801-3900, and every seed from 3725 up is unspent. A demo
# export must never be the thing that spends one.
FIRST_UNSPENT_SEED = 3725

PRESETS = ("exp4", "exp0")


# ---------------------------------------------------------------------------
# Configuration: through the shared definitions, never a copy of them
# ---------------------------------------------------------------------------


def preset_config(preset, family, arm=None, belief=None, overrides=None):
    """``(belief_mode, cfg, label)`` for ``build_episode``.

    ``exp4`` is ``scripts.arm_variants``'s frozen BASE_CFG plus one of its ARMS, so a
    replay of arm 2 is the episode Exp 4 runs, not a lookalike. ``exp0`` is the
    Exp 0 pilot's command-line defaults.
    """
    if preset == "exp4":
        from scripts.arm_variants import ARMS, BASE_CFG, BELIEF_MODE

        if arm not in ARMS:
            raise ValueError(f"--preset exp4 needs --arm in {sorted(ARMS)}, got {arm!r}")
        if belief not in (None, BELIEF_MODE):
            raise ValueError(f"Exp 4 fixes belief_mode={BELIEF_MODE!r}")
        cfg = dict(BASE_CFG, family=family, **ARMS[arm])
        belief_mode = BELIEF_MODE
        names = {1: "independent", 2: "CBAA, 1 slot", 3: "CBAA, 2 slots"}
        label = f"arm {arm} · {names.get(arm, '')}".strip(" ·")
    elif preset == "exp0":
        cfg = dict(
            blues=4, reds=2, steps=600, planning_interval_seconds=5.0,
            neutralize_fraction=0.5, recon_fraction=0.5, family=family, rule="nearest",
        )
        belief_mode = belief or "shared"
        label = f"Exp 0 · {belief_mode}"
    else:
        raise ValueError(f"unknown preset {preset!r}; expected one of {PRESETS}")
    for key, value in (overrides or {}).items():
        cfg[key] = value
    return belief_mode, cfg, label


def _parse_override(text):
    key, _, raw = text.partition("=")
    if not key or not _:
        raise argparse.ArgumentTypeError(f"--set expects key=value, got {text!r}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key, value


# ---------------------------------------------------------------------------
# Running, with read-only instrumentation
# ---------------------------------------------------------------------------


def attach_believed_log(sim):
    """``{step: {blue: {contact: predicted state}}}`` -- what the controller was
    handed. Wrapped from outside like the smoke's ledger. Used only to check the
    bundle's belief derivation; it is not shipped.

    ``_believed_targets`` can run more than once in a step (a networked round
    and a mid-epoch CBAA application call it too), and beliefs do not move
    within a step after the update, so the last call wins -- which is the
    solver's."""
    log = {}
    original = sim._believed_targets

    def recording(step):
        out = original(step)
        log[int(step)] = {
            int(i): {int(c): np.asarray(s, dtype=float).copy() for c, s in per.items()}
            for i, per in out.items()
        }
        return out

    sim._believed_targets = recording
    return log


def attach_link_log(sim):
    """``{step: [[a, b], ...]}`` -- the link set each communication round froze
    (``Network.begin_round``), between defenders active in that round. Read
    after ``_exchange_beliefs_over_network``, the testbed's only caller of
    ``begin_round``; the network object itself is built inside ``run()``."""
    log = {}
    original = sim._exchange_beliefs_over_network

    def recording(step, *args, **kwargs):
        out = original(step, *args, **kwargs)
        net = sim.network
        if net is not None and step % sim.comm_interval_steps == 0:
            down = set(net.disabled)
            log[int(step)] = sorted(
                sorted(int(x) for x in edge) for edge in net.edges if not (set(edge) & down)
            )
        return out

    sim._exchange_beliefs_over_network = recording
    return log


def attach_capture(sim):
    """Every read-only wrapper the bundle uses, attached before ``sim.run()``.
    Callers that build their own ``Simulation`` (e.g. an LLM episode re-run
    from cached answers) use this, so they capture exactly what ``export``
    captures."""
    return {
        "ledger": attach_ledger(sim),
        "bearing_log": attach_bearing_log(sim),
        "believed": attach_believed_log(sim),
        "links": attach_link_log(sim),
    }


def run_episode(seed, belief_mode, cfg, instrument=True):
    """Build through ``build_episode``, attach the wrappers, run quietly."""
    sim = build_episode(seed, belief_mode, cfg)
    capture = attach_capture(sim) if instrument else None
    with _Quiet():
        sim.run()
    return sim, capture


# ---------------------------------------------------------------------------
# Bundle construction
# ---------------------------------------------------------------------------


def _floats(arr):
    """Nested lists of Python floats. Full repr precision, so the check against
    the recorder is equality, never a tolerance."""
    return np.asarray(arr, dtype=float).tolist()


def terminal_reason(sim):
    if sim.touchdown:
        return "breach"
    if sim.collided:
        return "collision"
    if sim.solver_failed:
        return "solver_failure"
    if sim.defenders_won:
        return "all_reds_resolved"
    return "horizon"


def derive_coverage(steps, blue_frames, geometry, num_blues, blue_nx, extra_update=None):
    """Per frame, the step at which each coverage cell was last swept.

    Replays ``CoverageTracker.update`` over the recorded pre-step states with
    the recorded ``blue_disabled`` -- the same inputs ``_compute_detected_indices``
    used -- so it reproduces the live tracker exactly. That is checked, not
    assumed (``check_bundle``). ``extra_update`` is (frame, disabled) for a
    solver-failure ending, where detection ran on a step that was never
    recorded.
    """
    tracker = CoverageTracker(geometry["arena"]["size"], grid_dim=geometry["coverage"]["grid_dim"])
    half, rng = geometry["fov"]["half_angle_rad"], geometry["fov"]["range"]
    out = []

    def update(t, disabled):
        state = blue_frames[t]
        observers = [
            (i, state[i * blue_nx : i * blue_nx + 2]) for i in range(num_blues) if not disabled[i]
        ]
        bearings = {i: float(state[i * blue_nx + 4]) for i in range(num_blues)}
        tracker.update(t, observers, bearings, half, rng)

    for t, st in enumerate(steps):
        update(t, st["blue_disabled"])
        out.append(tracker.last_seen_step.tolist())
    if extra_update is not None:
        update(*extra_update)
    out.append(tracker.last_seen_step.tolist())  # the terminal frame
    return out, tracker


def derive_neutralizations(steps, terminal, red_frames, blue_frames, num_blues, num_reds,
                           blue_nx, red_nx, tolerance):
    """Who neutralized whom, and at which frame.

    Redoes ``_resolve_active_reds``' pairing on the frame the outcome first
    appears in: reds in index order, each against the closest blue still active
    at that point in the pass. A bird or civilian drone destroyed by contact (event kind ``decoy_destroyed``)
    takes its blue the same way, in the same pass (``_resolve_decoy``). The set
    of blues this disables is checked against the recorded ``blue_disabled``
    flips.
    """
    n = len(steps)
    before_out = ["active"] * num_reds
    before_dis = [False] * num_blues
    events = []
    for k in range(n + 1):
        after_out = steps[k]["red_outcomes"] if k < n else terminal["red_outcomes"]
        after_dis = steps[k]["blue_disabled"] if k < n else terminal["blue_disabled"]
        disabled = list(before_dis)
        for j in range(num_reds):
            if before_out[j] == "active" and after_out[j] in ("neutralized", "decoy_destroyed"):
                p = np.asarray(red_frames[k][j * red_nx : j * red_nx + 2])
                active = [i for i in range(num_blues) if not disabled[i]]
                dists = {
                    i: float(np.linalg.norm(np.asarray(blue_frames[k][i * blue_nx : i * blue_nx + 2]) - p))
                    for i in active
                }
                blue = min(dists, key=dists.get)
                if dists[blue] > tolerance:
                    raise AssertionError(
                        f"frame {k}: red {j} neutralized but the nearest active blue is "
                        f"{dists[blue]:.3f} away (> {tolerance})"
                    )
                disabled[blue] = True
                events.append({"frame": k, "red": j, "blue": blue, "distance": dists[blue],
                               "pos": [float(p[0]), float(p[1])], "outcome": after_out[j]})
        newly = {i for i in range(num_blues) if after_dis[i] and not before_dis[i]}
        paired = {e["blue"] for e in events if e["frame"] == k}
        if newly != paired:
            raise AssertionError(
                f"frame {k}: blues disabled {sorted(newly)} but neutralization pairing "
                f"gives {sorted(paired)}"
            )
        before_out, before_dis = list(after_out), list(after_dis)
    return events


def derive_events(steps, terminal, neutralizations, red_frames, red_nx, num_reds, reason, sim):
    """The timeline's jump targets. ``frame`` is the first frame showing it."""
    events = []
    n = len(steps)
    first = next((t for t, st in enumerate(steps) if st["detected_red_union"]), None)
    if first is not None:
        events.append({"frame": first, "kind": "first_detection",
                       "reds": steps[first]["detected_red_union"]})
    for e in neutralizations:
        kind = "neutralization" if e["outcome"] == "neutralized" else "decoy_destroyed"
        events.append({"frame": e["frame"], "kind": kind, "red": e["red"],
                       "blue": e["blue"], "pos": e["pos"]})
    before = ["active"] * num_reds
    for k in range(n + 1):
        after = steps[k]["red_outcomes"] if k < n else terminal["red_outcomes"]
        for j in range(num_reds):
            if before[j] == "active" and after[j] in ("escaped", "decoy_left"):
                p = red_frames[k][j * red_nx : j * red_nx + 2]
                events.append({"frame": k, "kind": "escape" if after[j] == "escaped" else "decoy_left",
                               "red": j, "pos": [float(p[0]), float(p[1])]})
        before = list(after)
    if reason == "breach":
        j = int(sim.touchdown_red_index)
        p = red_frames[n][j * red_nx : j * red_nx + 2]
        events.append({"frame": n, "kind": "breach", "red": j, "pos": [float(p[0]), float(p[1])]})
    elif reason == "collision":
        events.append({"frame": n, "kind": "collision", "blues": list(sim.collision_agents)})
    elif reason == "solver_failure":
        events.append({"frame": n, "kind": "solver_failure"})
    events.sort(key=lambda e: (e["frame"], e["kind"]))
    return events


def instrumentation(capture, n_steps, sim):
    """Per-step commitments and heading sources from the wrappers.

    Commitments: the ledger holds every application; the one in force at step
    t is the last applied at or before t (applications precede the solve, and
    so the record, within a step)."""
    by_step = {}
    for entry in capture["ledger"]:
        by_step.setdefault(entry["step"], []).append(entry)
    current = {}
    commitments = []
    for t in range(n_steps):
        for entry in by_step.get(t, []):
            current = dict(entry["commitments"])
        commitments.append({
            str(a): (None if c is None else [c[0], int(c[1])]) for a, c in sorted(current.items())
        })
    bearing = []
    for t in range(n_steps):
        override, sources = capture["bearing_log"][t]
        bearing.append({
            "tracking": [int(a) for a in override],
            "sources": {str(a): s for a, s in sorted(sources.items())},
        })
    runtime = getattr(sim, "cbaa_runtime", None)
    return {
        "note": (
            "Captured by read-only wrappers (the auction smoke's ledger and bearing log), "
            "attached from outside the testbed. NOT part of the EpisodeRecorder schema."
        ),
        "commitments": commitments,
        "applications": [
            {"step": e["step"], "mid_epoch": e["mid_epoch"], "changed": e["changed"]}
            for e in capture["ledger"]
        ],
        "bearing": bearing,
        "conflict_log": (
            [] if runtime is None
            else [{"step": c["step"], "epoch": c["epoch"], "conflicts": sorted(map(str, c["conflicts"]))}
                  for c in runtime.conflict_log]
        ),
    }


def _header(sim, source, outcome, reason):
    config = sim.recorder.config
    allocation = config.get("allocation")
    transport = None
    if sim.cbaa_runtime is not None:
        transport = sim.cbaa_runtime.transport
    return {
        "seed": int(sim.seed),
        "family": sim.scenario_family,
        "belief_mode": sim.belief_mode,
        "allocation": allocation or "role planner",
        "planner": config.get("role_planner", {}).get("__type__"),
        "transport": transport,
        "slots": sim.cbaa_runtime.slots if sim.cbaa_runtime is not None else None,
        "station_heading": sim.station_heading,
        "planning_interval_s": sim.planning_interval_seconds,
        "label": source["label"],
        "reason": reason,
        "num_neutralized": outcome["num_neutralized"],
        "num_escaped": outcome["num_escaped"],
        "n_blues_disabled": outcome["n_blues_disabled"],
        "breach": outcome["touchdown"],
    }


def episode_bundle(sim, capture, source):
    """One episode of the bundle, from a finished ``Simulation``."""
    if sim.recorder is None:
        raise ValueError("the episode was run with record_episode=False; nothing to export")
    if sim.contact_substitution_enabled:
        # The legacy Exp II roster keeps birds and civilians outside the red
        # state; non-hostile contacts are red-state rows and are exported.
        raise ValueError("Exp II episodes (contact substitution) are out of scope")
    rec = sim.recorder
    manifest = rec.manifest()
    arrays = rec.arrays()
    steps = manifest.pop("steps")
    n = len(steps)
    nb, nr, bnx, rnx = sim.num_blues, sim.num_reds, sim.blue_nx, sim.red_nx

    blue_frames = np.vstack([arrays["blue_state"].reshape(n, -1), np.asarray(sim.history_blues)[-1][None, :]]) \
        if n else np.asarray(sim.history_blues)
    red_frames = np.vstack([arrays["red_state"].reshape(n, -1), np.asarray(sim.history_red)[-1][None, :]]) \
        if n else np.asarray(sim.history_red)

    outcome = sim.episode_outcome()
    reason = terminal_reason(sim)
    terminal = {
        # Every red-state row, non-hostile contacts included: the outcome dict keeps the real
        # reds only, and the recorder's per-step rows hold them all.
        "red_outcomes": list(sim.red_outcomes),
        "blue_disabled": list(outcome["blue_disabled"]),
        "reason": reason,
        "collision": (
            None if not sim.collided
            else {"step": sim.collision_step, "agents": list(sim.collision_agents), "kind": sim.collision_kind}
        ),
    }

    tracker = sim.coverage_tracker
    geometry = {
        "arena": {"min": float(sim.arena_min), "max": float(sim.arena_max), "size": float(sim.arena_size)},
        "defended_shape": _floats(sim.defended_shape_points),
        "defense_polygon": _floats(sim.defense_polygon),
        "defense_center": _floats(sim.defense_center),
        "stations": {str(k): _floats(v) for k, v in sorted(sim.solver.defend_stations.items())},
        "coverage": {
            "grid_dim": int(tracker.grid_dim),
            "cell_size": float(tracker.cell_size),
            "centers": _floats(tracker.centers),
        },
        "fov": {"half_angle_rad": float(sim.fov_half_angle_rad), "range": float(sim.fov_range)},
        "contact_tolerance": float(sim.contact_tolerance),
        "v_max": {"blue": float(sim.blue_v_max), "red": float(sim.red_v_max)},
        "planning_interval_steps": int(sim.planning_interval_steps),
        "comm_interval_steps": int(sim.comm_interval_steps),
        "belief_validity_age_steps": sim.belief_validity_age_steps,
        "intervention_tol": INTERVENTION_TOL,
    }

    extra = None
    if reason == "solver_failure":
        extra = (n, terminal["blue_disabled"])
    coverage, _ = derive_coverage(steps, blue_frames, geometry, nb, bnx, extra_update=extra)
    neutralizations = derive_neutralizations(
        steps, terminal, red_frames, blue_frames, nb, nr, bnx, rnx, sim.contact_tolerance,
    )
    events = derive_events(steps, terminal, neutralizations, red_frames, rnx, nr, reason, sim)

    slack = arrays["cbf_slack"]
    width = arrays["cbf_slack_width"]
    slack_max, qp_failed = [], []
    for t in range(n):
        row = slack[t, : int(width[t])] if slack.size else np.zeros(0)
        finite = row[np.isfinite(row)]
        slack_max.append(float(finite.max()) if finite.size else None)
        qp_failed.append(bool(row.size and not np.isfinite(row).all()))

    ep = {
        "label": source["label"],
        "source": source,
        "header": _header(sim, source, outcome, reason),
        "recorder": {k: manifest[k] for k in (
            "schema_version", "recorded_at", "git_commit", "seed", "rng_mode", "config",
            "config_hash", "outcome", "safety", "n_recorded_steps",
        )},
        "dims": {
            "num_blues": nb, "num_reds": nr, "blue_nx": bnx, "red_nx": rnx,
            "blue_nu": int(sim.blue.nu_single), "dt": float(sim.dt),
            "n_steps": n, "n_frames": n + 1, "sim_steps": int(sim.sim_steps),
        },
        "geometry": geometry,
        "frames": {"blue_state": _floats(blue_frames), "red_state": _floats(red_frames)},
        "controls": {
            "u_requested": _floats(arrays["u_requested"]),
            "u_executed": _floats(arrays["u_executed"]),
            "cbf_slack_max": slack_max,
            "cbf_qp_failed": qp_failed,
        },
        "steps": steps,
        "terminal": terminal,
        "derived": {
            "coverage_last_seen": coverage,
            "neutralizations": neutralizations,
            "events": events,
        },
    }
    ep["contacts"] = contacts_block(sim)
    ep["network"] = network_block(sim, capture)
    ep["command"] = command_block(sim, n)
    if capture is not None:
        ep["instrumentation"] = instrumentation(capture, n, sim)
        held = held_belief_overrides(ep, sim, capture)
        if held:
            ep["beliefs_held"] = held
    return ep


def contacts_block(sim):
    """What each red-state row is. non-hostile contacts are rows num_real_reds..;
    ``images`` is the photograph each contact was shown as, by row, relative to
    the repository root (file names repeat across stimulus sets, so a bare name
    would be ambiguous)."""
    spec = sim.decoy_spec or {}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def rel(p):
        p = os.path.abspath(str(p))
        return os.path.relpath(p, root) if p.startswith(root + os.sep) else os.path.basename(p)

    return {
        "kinds": list(sim.red_kinds),
        "num_intruders": int(sim.num_real_reds),
        "images": {str(int(j)): rel(p) for j, p in sorted(sim.contact_image_paths.items())},
        "decoy_motion": spec.get("motion"),
        "decoy_events": [dict(e) for e in sim.decoy_events],
    }


def network_block(sim, capture):
    loss = sim.comm_loss_model
    return {
        "belief_mode": sim.belief_mode,
        "r_comm": float(sim.r_comm),
        "loss_model": None if loss is None else (loss.describe() if hasattr(loss, "describe") else repr(loss)),
        "comm_interval_steps": int(sim.comm_interval_steps),
        "links": None if capture is None or "links" not in capture
        else {str(t): edges for t, edges in sorted(capture["links"].items())},
    }


def _disabled_at(sim):
    """First resolution step at which each blue was disabled. From the public
    resolution log when the run keeps one (the analysis reads the same events),
    else from the first recorded step that shows it."""
    out = {}
    log = getattr(sim, "resolution_log", None) or []
    for ev in log:
        if ev["team"] == "blue" and ev["event"] == "disabled":
            out.setdefault(int(ev["agent"]), int(ev["step"]))
    if not log and sim.recorder is not None:
        for t, st in enumerate(sim.recorder.steps):
            for i, d in enumerate(st["blue_disabled"]):
                if d:
                    out.setdefault(i, t)
    return out


def classify_orders(log, disabled_at, lost_window=ORDER_LOST_WINDOW):
    """Each command-log row's fate, by the definition the study's missed-order
    analysis uses:

    * ``self``    -- the planning defender's own order (never crosses a link);
    * ``applied`` -- accepted by its addressee;
    * ``addressee_lost`` -- never applied, but the addressee was disabled within
      ``lost_window`` steps of generation;
    * ``missed_changed`` -- never applied, and it names a different task than
      the one the addressee held when it was generated (its newest applied
      network order, else ``DEFEND:<own id>``): an order that would have changed
      what the defender did;
    * ``missed_same`` -- never applied, and it repeats the held task.
    """
    applied_by_agent = {}
    for e in log:
        if e["via"] == "network" and e["applied_at"] is not None:
            applied_by_agent.setdefault(e["agent"], []).append(e)
    fates = []
    for e in log:
        if e["via"] != "network":
            fates.append("self")
            continue
        if e["applied_at"] is not None:
            fates.append("applied")
            continue
        a = e["agent"]
        d = disabled_at.get(a)
        if d is not None and d <= e["generated_at"] + lost_window:
            fates.append("addressee_lost")
            continue
        held = None
        for x in applied_by_agent.get(a, []):
            if x["applied_at"] <= e["generated_at"] and (held is None or x["epoch"] > held["epoch"]):
                held = x
        held_task = held["task_id"] if held else f"DEFEND:{a}"
        fates.append("missed_changed" if e["task_id"] != held_task else "missed_same")
    return fates


def command_block(sim, n_steps):
    """The commander's orders, or None for a team without one.
    ``source`` per planning epoch is who actually planned: the commander, or
    after its loss the lowest-indexed active defender (``planning_source``),
    read off the order it applied to itself."""
    log = [dict(e) for e in getattr(sim, "command_log", None) or []]
    if not log:
        return None
    keep = ("epoch", "agent", "via", "task_id", "defend_target", "generated_at", "sent_at",
            "received_at", "applied_at", "hops", "outcome")
    rows = [{k: _plain(e.get(k)) for k in keep} for e in log]
    fates = classify_orders(rows, _disabled_at(sim))
    for row, fate in zip(rows, fates):
        row["fate"] = fate
    source = {}
    for row in rows:
        if row["via"] == "self":
            source[str(row["generated_at"])] = row["agent"]
    return {
        "commander_id": int(sim.commander_id),
        "commander_lost_step": sim.commander_lost_step,
        "failure_stops_planning": bool(getattr(sim, "commander_failure_stops_planning", False)),
        "source_by_step": source,
        "orders": rows,
        "lost_window": ORDER_LOST_WINDOW,
    }


def _plain(v):
    if isinstance(v, (tuple, list)):
        return [_plain(x) for x in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def held_belief_overrides(ep, sim, capture):
    """The believed positions the controller was handed that differ from the
    ones derived from the truth (``believed_position``), by step, blue and
    contact. Empty unless the falsified-report attack altered a record: each
    entry must be the attack's target red, held by a blue that had accepted an
    altered record of it by then, or this raises."""
    validity = ep["geometry"]["belief_validity_age_steps"]
    attack = getattr(sim, "attack_log", None)
    accepted = {}
    for rec in (attack or {}).get("accepted", []):
        key = (int(rec["receiver"]), int(rec["red"]))
        accepted[key] = min(accepted.get(key, rec["step"]), rec["step"])
    out = {}
    for t, st in enumerate(ep["steps"]):
        live = capture["believed"].get(t, {})
        for agent, per in st["beliefs"].items():
            got = live.get(int(agent), {})
            for c, age in per.items():
                c = int(c)
                if (validity is not None and age > validity) or c not in got:
                    continue
                held = got[c][0:2]
                if np.array_equal(believed_position(ep, t, c, age), held):
                    continue
                first = accepted.get((int(agent), c))
                if attack is None or first is None or first > t:
                    raise AssertionError(
                        f"step {t} blue {agent} contact {c}: held belief differs from the truth-derived "
                        "one without an accepted falsified record"
                    )
                out.setdefault(str(t), {}).setdefault(str(agent), {})[str(c)] = _floats(held)
    return out


# ---------------------------------------------------------------------------
# Self-checks: a bundle that disagrees with its episode is never written
# ---------------------------------------------------------------------------


def believed_position(ep, t, contact, age):
    """The position the controller acts on, derived from the recorder alone:
    the red's recorded state at ``t - age``, extrapolated at constant velocity
    exactly as ``TargetBelief.predicted_position`` does (same expression order,
    so the floats match)."""
    rnx = ep["dims"]["red_nx"]
    row = np.asarray(ep["frames"]["red_state"][t - age][contact * rnx : contact * rnx + 4])
    return row[0:2] + row[2:4] * (age * float(ep["dims"]["dt"]))


def check_bundle(ep, sim, capture):
    """Raise on any disagreement between the bundle and the live episode."""
    blue = np.asarray(ep["frames"]["blue_state"])
    red = np.asarray(ep["frames"]["red_state"])
    if not np.array_equal(blue, np.asarray(sim.history_blues).reshape(blue.shape)):
        raise AssertionError("bundle blue frames != sim.history_blues")
    if not np.array_equal(red, np.asarray(sim.history_red).reshape(red.shape)):
        raise AssertionError("bundle red frames != sim.history_red")

    final = ep["derived"]["coverage_last_seen"][-1]
    if final != sim.coverage_tracker.last_seen_step.tolist():
        raise AssertionError(
            f"derived coverage {final} != live tracker {sim.coverage_tracker.last_seen_step.tolist()}"
        )

    if capture is not None:
        validity = ep["geometry"]["belief_validity_age_steps"]
        for t, st in enumerate(ep["steps"]):
            live = capture["believed"].get(t)
            if live is None:
                raise AssertionError(f"no believed-targets capture at step {t}")
            for agent, per in st["beliefs"].items():
                fresh = {
                    int(c): a for c, a in per.items() if validity is None or a <= validity
                }
                got = live.get(int(agent), {})
                if set(fresh) != set(got):
                    raise AssertionError(
                        f"step {t} blue {agent}: recorded beliefs {sorted(fresh)} "
                        f"!= controller's {sorted(got)}"
                    )
                held = ep.get("beliefs_held", {}).get(str(t), {}).get(str(agent), {})
                for c, age in fresh.items():
                    derived = (np.asarray(held[str(c)]) if str(c) in held
                               else believed_position(ep, t, c, age))
                    if not np.array_equal(derived, got[c][0:2]):
                        raise AssertionError(
                            f"step {t} blue {agent} contact {c} age {age}: derived "
                            f"{derived} != controller's {got[c][0:2]}"
                        )
    return True


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def bundle_document(episodes, argv=None):
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "exporter": {"script": "scripts/export_replay.py", "git_commit": _git_commit(),
                     "argv": list(argv or [])},
        "episodes": episodes,
    }


def _json_safe(value):
    """Non-finite floats as the strings "inf", "-inf", "nan". The recorded
    config legitimately holds ``r_comm = inf``, and bare ``Infinity``/``NaN``
    are not JSON: ``JSON.parse`` would reject the whole bundle."""
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def write_bundle(doc, path):
    """``.js`` wraps the JSON in ``PEV_REPLAY(...)`` so the viewer can load it
    with a script tag from ``file://``, where fetch is refused. Anything else
    is plain JSON."""
    text = json.dumps(_json_safe(doc), separators=(",", ":"), allow_nan=False)
    parent = os.path.dirname(os.fspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        if os.fspath(path).endswith(".js"):
            f.write("PEV_REPLAY(" + text + ");\n")
        else:
            f.write(text + "\n")
    return path


def export(seed, preset, family, arms=(None,), belief=None, overrides=None, check=True):
    """Run and bundle one episode per arm. Returns the list of episode dicts."""
    episodes = []
    for arm in arms:
        belief_mode, cfg, label = preset_config(preset, family, arm, belief, overrides)
        sim, capture = run_episode(seed, belief_mode, cfg, instrument=True)
        source = {"preset": preset, "arm": arm, "seed": int(seed), "family": family,
                  "belief_mode": belief_mode, "cfg": cfg, "label": label}
        ep = episode_bundle(sim, capture, source)
        if check:
            check_bundle(ep, sim, capture)
        episodes.append(ep)
    return episodes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preset", choices=PRESETS, default="exp4")
    ap.add_argument("--arm", type=int, action="append", default=None,
                    help="Exp 4 arm (1, 2, 3). Repeat for a compare bundle.")
    ap.add_argument("--family", default="symmetric", choices=("symmetric", "multi_bearing"))
    ap.add_argument("--belief", default=None, choices=("shared", "local", "networked"),
                    help="exp0 only; Exp 4 fixes 'shared'")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--set", dest="overrides", type=_parse_override, action="append", default=[],
                    metavar="KEY=VALUE", help="override one build_episode cfg key (JSON value)")
    ap.add_argument("--out", required=True, help=".replay.js (file:// loadable) or .json")
    ap.add_argument("--allow-unspent-seed", action="store_true",
                    help=f"permit seed >= {FIRST_UNSPENT_SEED}; this SPENDS it")
    args = ap.parse_args(argv)

    if args.seed >= FIRST_UNSPENT_SEED and not args.allow_unspent_seed:
        ap.error(f"seed {args.seed} is not spent (>= {FIRST_UNSPENT_SEED}, and 3801-3900 are "
                 "Exp 4's). Pass --allow-unspent-seed only if you mean to spend it.")
    arms = tuple(args.arm) if args.arm else ((2,) if args.preset == "exp4" else (None,))
    episodes = export(args.seed, args.preset, args.family, arms, args.belief, dict(args.overrides))
    doc = bundle_document(episodes, argv=sys.argv[1:] if argv is None else argv)
    write_bundle(doc, args.out)
    size = os.path.getsize(args.out) / 1e6
    for ep in episodes:
        h = ep["header"]
        print(f"{ep['label']:>22s}  seed {h['seed']} {h['family']}  {ep['dims']['n_steps']} steps  "
              f"end: {h['reason']}  checks: passed")
    print(f"wrote {args.out} ({size:.2f} MB, {len(episodes)} episode(s))")


if __name__ == "__main__":
    main()
