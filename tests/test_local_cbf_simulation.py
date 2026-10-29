"""safety_filter="local" in Simulation: option validation, what each agent's
filter is told about its neighbors, and an end-to-end smoke. The joint
default is covered by the golden episode."""

import numpy as np
import pytest

from conftest import build_simulation, run_quiet
from control.local_cbf import COOPERATIVE, FULL
from planning.roles import BlueRole

LOCAL = dict(safety_filter="local", safety_state_sync="frozen", local_cbf_gamma=0.1)


def test_joint_is_the_default_and_is_not_recorded():
    sim = build_simulation()
    assert sim.safety_filter == "joint"
    for key in ("safety_filter", "local_cbf_gamma", "local_cbf_red_red"):
        assert key not in sim._init_args
    assert build_simulation(**LOCAL)._init_args["local_cbf_gamma"] == 0.1


@pytest.mark.parametrize("overrides, match", [
    (dict(safety_filter="local", local_cbf_gamma=0.1), "frozen"),
    (dict(safety_filter="local", safety_state_sync="frozen"), "local_cbf_gamma"),
    (dict(LOCAL, enable_convex_hull_containment=True), "convex-hull"),
    (dict(safety_filter="per_agent"), "safety_filter"),
])
def test_local_mode_refuses_incomplete_configurations(overrides, match):
    with pytest.raises(ValueError, match=match):
        build_simulation(**overrides)


def _sim_with_snapshot(**overrides):
    sim = build_simulation(num_blues=3, num_reds=2, enable_red_cbf=True, **LOCAL, **overrides)
    sim.setup_agents()
    # Distinct, nonzero velocities so "zero for an immobile agent" is visible.
    for i in range(sim.num_blues):
        sim.x_current[i * sim.blue_nx + 2 : i * sim.blue_nx + 4] = [0.3 + i, -0.2]
    for j in range(sim.num_reds):
        sim.red_state[j * sim.red_nx + 2 : j * sim.red_nx + 4] = [-0.5, 0.1 * (j + 1)]
    return sim


def _by_key(neighbors):
    return {nb.key: nb for nb in neighbors}


def test_blue_neighbors_cooperative_teammates_and_constant_velocity_reds():
    sim = _sim_with_snapshot()
    snap = sim._local_safety_snapshot(sim.x_current, sim.red_state)
    nbs = _by_key(sim._local_blue_neighbors(0, snap))
    assert set(nbs) == {("blue", 1), ("blue", 2), ("red", 0), ("red", 1)}
    assert nbs[("blue", 1)].share == COOPERATIVE
    assert nbs[("red", 0)].share == FULL
    np.testing.assert_array_equal(nbs[("red", 1)].velocity, [-0.5, 0.2])


def test_disabled_blue_is_an_immobile_full_responsibility_obstacle_for_blues_only():
    sim = _sim_with_snapshot()
    sim.blue_disabled[2] = True
    snap = sim._local_safety_snapshot(sim.x_current, sim.red_state)
    nb = _by_key(sim._local_blue_neighbors(0, snap))[("blue", 2)]
    assert nb.share == FULL
    np.testing.assert_array_equal(nb.velocity, [0.0, 0.0])
    assert ("blue", 2) not in _by_key(sim._local_red_neighbors(0, snap))


def test_resolved_reds_leave_every_filter():
    sim = _sim_with_snapshot()
    sim.red_outcomes[1] = "neutralized"
    snap = sim._local_safety_snapshot(sim.x_current, sim.red_state)
    assert ("red", 1) not in _by_key(sim._local_blue_neighbors(0, snap))
    assert ("red", 1) not in _by_key(sim._local_red_neighbors(0, snap))


def test_only_the_installed_pursuit_pair_is_exempt_and_the_target_still_avoids():
    sim = _sim_with_snapshot()
    sim.role_assignment = {0: BlueRole.NEUTRALIZE, 1: BlueRole.NEUTRALIZE, 2: BlueRole.DEFEND}
    sim.blue_target_red_idx = {0: 1, 1: 0}
    sim.target_override = {0: np.zeros(4)}  # blue 1's pursuit was never installed
    snap = sim._local_safety_snapshot(sim.x_current, sim.red_state)
    assert snap["pursued"] == {0: 1}
    b0 = _by_key(sim._local_blue_neighbors(0, snap))
    assert ("red", 1) not in b0 and ("red", 0) in b0
    assert ("blue", 1) in b0  # teammate avoidance is kept
    assert {("red", 0), ("red", 1)} <= set(_by_key(sim._local_blue_neighbors(1, snap)))
    assert ("blue", 0) in _by_key(sim._local_red_neighbors(1, snap))


def test_red_teammates_are_cooperative_and_can_be_switched_off():
    sim = _sim_with_snapshot()
    snap = sim._local_safety_snapshot(sim.x_current, sim.red_state)
    assert _by_key(sim._local_red_neighbors(0, snap))[("red", 1)].share == COOPERATIVE
    sim = _sim_with_snapshot(local_cbf_red_red=False)
    snap = sim._local_safety_snapshot(sim.x_current, sim.red_state)
    assert ("red", 1) not in _by_key(sim._local_red_neighbors(0, snap))


def test_local_episode_logs_every_active_agent_every_step():
    sim = build_simulation(num_blues=3, num_reds=2, enable_red_cbf=True, sim_steps=25, **LOCAL)
    run_quiet(sim)
    assert sim.local_cbf_log
    for entry in sim.local_cbf_log:
        assert entry["version"] == "lcbf.1"
        active_blues = {i for i in range(3) if not sim.blue_disabled_history[entry["step"]][i]}
        assert set(entry["blue"]) == active_blues
        for rec in list(entry["blue"].values()) + list(entry["red"].values()):
            assert rec["path"] in ("unconstrained", "hard", "relaxed", "braking")


def test_red_red_separation_is_counted_without_changing_the_episode():
    from conftest import fingerprint
    a = run_quiet(build_simulation(num_blues=2, num_reds=3, sim_steps=30))
    b = run_quiet(build_simulation(num_blues=2, num_reds=3, sim_steps=30))
    assert fingerprint(a) == fingerprint(b)
    rec = a.red_red_separation
    hist = np.asarray(a.history_red)
    # Reds that stay active throughout: the recorded minimum is over steps 1..T.
    if all(o == "active" for o in a.red_outcomes):
        nx = a.red_nx
        best = min(np.linalg.norm(hist[t][j*nx:j*nx+2] - hist[t][k*nx:k*nx+2])
                   for t in range(1, len(hist)) for j in range(3) for k in range(j + 1, 3))
        assert rec["min_distance"] == pytest.approx(best)
    assert rec["pair_steps_at_contact"] == len(rec["contact_events"])


def test_red_blue_rows_switch_off_and_are_recorded_only_when_off():
    """safety.2: red constraints = active red
    teammates (cooperative halves) + control bounds; no blue rows. The blue
    side, including the installed-pursuit exemption, is unchanged."""
    assert "local_cbf_red_blue" not in _sim_with_snapshot()._init_args
    sim = _sim_with_snapshot(local_cbf_red_blue=False)
    assert sim._init_args["local_cbf_red_blue"] is False
    sim.role_assignment = {0: BlueRole.NEUTRALIZE, 1: BlueRole.DEFEND, 2: BlueRole.DEFEND}
    sim.blue_target_red_idx = {0: 1}
    sim.target_override = {0: np.zeros(4)}
    snap = sim._local_safety_snapshot(sim.x_current, sim.red_state)
    for j in range(sim.num_reds):
        nbs = _by_key(sim._local_red_neighbors(j, snap))
        assert set(nbs) == {("red", 1 - j)}
        assert nbs[("red", 1 - j)].share == COOPERATIVE
    b0 = _by_key(sim._local_blue_neighbors(0, snap))
    assert set(b0) == {("blue", 1), ("blue", 2), ("red", 0)}  # red 1 exempt for its pursuer only
    assert set(_by_key(sim._local_blue_neighbors(1, snap))) == {("blue", 0), ("blue", 2), ("red", 0), ("red", 1)}
    # Resolved reds still leave every filter; a lone red keeps only its bounds.
    sim.red_outcomes[1] = "neutralized"
    snap = sim._local_safety_snapshot(sim.x_current, sim.red_state)
    assert sim._local_red_neighbors(0, snap) == []


def test_red_blue_switch_requires_the_local_filter():
    with pytest.raises(ValueError, match="local_cbf_red_blue"):
        build_simulation(local_cbf_red_blue=False)


def test_red_filter_without_rows_applies_control_bounds_only():
    from control.local_cbf import LocalCBFConfig, filter_agent
    cfg = LocalCBFConfig(gamma=0.05, dt=0.1, a_max=2.0, v_max=2.0)
    r = filter_agent([0, 0], [1.95, 0.0], [5.0, -5.0], [], cfg)
    assert r.path == "unconstrained" and r.num_rows == 0
    np.testing.assert_allclose(r.u, [0.5, -2.0])  # v_max bounds x, a_max bounds y
