"""Red sensor control ("track_or_velocity") and heading observability in the
prompts. Rule planners and recorded prompts only: no model calls."""

import json

import numpy as np
import pytest

from conftest import build_simulation, capture_fd_output


def _episode(red_sensor_control, reds=4, seed=3904):
    from scripts import experiment_policy as ep
    from scripts import run_competing_teams as rct

    saved = dict(ep.NEW_EXPERIMENT_COMMON)
    ep.NEW_EXPERIMENT_COMMON["red_sensor_control"] = red_sensor_control
    try:
        e = {"arm": "cbaa", "reds": reds, "family": "multi_bearing", "condition": "REF",
             "attacked": False, "seed": seed}
        e["episode_id"] = rct.episode_id(e)
        sim, _ = rct.build(e, "rule")
    finally:
        ep.NEW_EXPERIMENT_COMMON.clear()
        ep.NEW_EXPERIMENT_COMMON.update(saved)
    with capture_fd_output():
        sim.run()
    return sim


def test_fixed_is_the_unrecorded_default_and_values_are_checked():
    sim = build_simulation()
    assert sim.red_sensor_control == "fixed" and "red_sensor_control" not in sim._init_args
    with pytest.raises(ValueError, match="red_sensor_control"):
        build_simulation(red_sensor_control="scan")


def test_red_headings_turn_within_the_yaw_limit_and_follow_the_rule():
    fixed = _episode("fixed")
    hr = np.asarray(fixed.history_red)
    # historical: no heading objective; measured drift <= 0.02 deg per episode
    assert all(np.ptp(np.unwrap(hr[:, j * fixed.red_nx + 4])) < np.radians(0.1)
               for j in range(fixed.num_reds))
    sim = _episode("track_or_velocity")
    hr = np.asarray(sim.history_red)
    nx, dt = sim.red_nx, sim.dt
    omega_max = sim.red.omega_max
    turned = 0.0
    for j in range(sim.num_reds):
        th = np.unwrap(hr[:, j * nx + 4])
        assert np.max(np.abs(np.diff(th))) <= omega_max * dt + 1e-9
        turned = max(turned, np.ptp(th))
    assert turned > 0.1  # reds now actually turn
    modes = {r["mode"] for r in sim.red_sensor_log}
    assert {"track", "velocity"} <= modes
    # every "track" entry names a blue the red's own cone contained at that step
    from perception import sensing
    for r in sim.red_sensor_log[:400]:
        if r["mode"] != "track":
            continue
        t, j, b = r["step"], r["red"], r["blue"]
        p, th = hr[t][j * nx: j * nx + 2], hr[t][j * nx + 4]
        q = np.asarray(sim.history_blues)[t][b * sim.blue_nx: b * sim.blue_nx + 2]
        assert sensing.is_detected(p, th, q, sim.fov_half_angle_rad, sim.fov_range)


def test_velocity_mode_converges_to_the_direction_of_motion():
    sim = _episode("track_or_velocity", reds=2)
    hr = np.asarray(sim.history_red)
    nx = sim.red_nx
    errs = [abs(r["heading_error"]) for r in sim.red_sensor_log if r["mode"] == "velocity"]
    # after the first turn, velocity-facing errors are small (36 deg/step limit)
    assert np.median(errs) < np.radians(5)


def test_heading_prompts_show_observed_headings_without_labels():
    from planning.intruder_llm import observation_from_dict, system_prompt, user_message

    obs_ctx = {"kind": "intruder", "agent_id": 0, "now": 50, "dt": 0.1,
               "own_state": [10.0, 10.0, 1.0, 0.0, 0.0], "phase": "interior",
               "commitment": {"role": "attack", "gate_id": 2, "evade_blue_id": None},
               "teammates": {"1": [20.0, 10.0, 1.0, 0.0, 1.5708]}, "teammate_intents": {},
               "believed_blues": {"3": [[30.0, 30.0, 0.0, -1.0, 3.14159], 40]},
               "gates": [[2, [40.0, 40.0]]], "defense_center": [50.0, 50.0]}
    obs = observation_from_dict(obs_ctx)
    u = user_message(obs)
    assert "heading_deg=0" in u and "heading_deg=90" in u
    assert "observed_heading_deg=180" in u and "seen 10 steps ago" in u
    assert "your_current_action: attack\n" in u + "\n"  # passed gate no longer printed (interior)
    for banned in ("cone", "covers", "engaged", "pursu", "looking at"):
        assert banned not in u.lower()
    assert "60-degree wedge" in system_prompt("approach", 400, 0.1)


def test_observed_heading_is_the_sighting_heading_not_extrapolated():
    from perception.belief.records import Provenance, TargetBelief

    b = TargetBelief(1, np.array([0.0, 0.0, 1.0, 0.0, 0.7]),
                     Provenance(origin_agent=0, observed_at=10, received_at=12, relay_agent=2))
    st = b.predicted_state(30, 0.1)
    assert st[4] == pytest.approx(0.7) and st[0] == pytest.approx(2.0)
    relayed = b.provenance.relayed_by(3, 15)
    assert relayed.observed_at == 10


def test_bridge_and_commander_prompts_state_the_sensor_geometry():
    from coordination.commander_llm import _system_prompt
    from coordination.llm_bridge import build_bridge_system_prompt, prompt_version
    from planning.prompt_builder import PromptContext

    ctx = PromptContext(arena_size=100.0, horizon_steps=400, dt=0.1)
    s = build_bridge_system_prompt(ctx, "three_task")
    assert "60-degree wedge" in s and "Respond only with a valid JSON object" in s
    assert prompt_version("three_task") == "m8b.7+three_task"
    assert "60-degree wedge" in _system_prompt(400, 0.1)


def test_episodes_must_use_the_sensor_geometry_the_prompts_state():
    from types import SimpleNamespace

    from scripts.inference_profile import INTERFACE_VERSIONS, check_heading_prompt_geometry

    assert INTERFACE_VERSIONS == {"bridge_version": "m8b.7", "task_set": "three_task",
                                  "intruder_prompt": "intruder.4",
                                  "commander_prompt": "commander.10"}
    ok = SimpleNamespace(fov_half_angle_rad=np.deg2rad(30.0), fov_range=35.0)
    check_heading_prompt_geometry(ok)
    with pytest.raises(ValueError, match="refusing"):
        check_heading_prompt_geometry(SimpleNamespace(fov_half_angle_rad=np.deg2rad(45.0),
                                                      fov_range=35.0))
    with pytest.raises(ValueError, match="refusing"):
        check_heading_prompt_geometry(SimpleNamespace(fov_half_angle_rad=np.deg2rad(30.0),
                                                      fov_range=50.0))
