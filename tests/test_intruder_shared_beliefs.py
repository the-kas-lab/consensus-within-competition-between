"""intruder_info="shared_beliefs": per-red first-hand sightings merged over an
ideal intruder channel, and EVADE driven by constant-velocity predictions
from those beliefs instead of the blues' actual MPC plans."""

import numpy as np
import pytest

from conftest import build_simulation, run_quiet


def test_legacy_is_the_default_and_is_not_recorded():
    sim = build_simulation()
    assert sim.intruder_info == "legacy"
    assert "intruder_info" not in sim._init_args
    with pytest.raises(ValueError, match="intruder_info"):
        build_simulation(intruder_info="omniscient")


def _placed(sim, team, idx, pos, vel=(0.0, 0.0), theta=None):
    nx = sim.blue_nx if team == "blue" else sim.red_nx
    x = sim.x_current if team == "blue" else sim.red_state
    x[idx * nx : idx * nx + 4] = [*pos, *vel]
    if theta is not None:
        x[idx * nx + 4] = theta


def test_ideal_channel_merges_first_hand_sightings_by_latest_observation():
    sim = build_simulation(num_blues=3, num_reds=2, intruder_info="shared_beliefs")
    sim.setup_agents()
    from perception.belief.store import LocalBeliefStore
    sim.red_belief_stores = {j: LocalBeliefStore(j) for j in range(sim.num_reds)}
    r = 0.5 * sim.fov_range
    # Red 0 looks east and sees blue 0; red 1 looks north and sees blue 1.
    _placed(sim, "red", 0, (10.0, 10.0), theta=0.0)
    _placed(sim, "red", 1, (60.0, 10.0), theta=np.pi / 2)
    _placed(sim, "blue", 0, (10.0 + r, 10.0), vel=(0.3, 0.0))
    _placed(sim, "blue", 1, (60.0, 10.0 + r), vel=(0.0, -0.4))
    _placed(sim, "blue", 2, (90.0, 90.0))  # seen by nobody
    sim._update_red_beliefs(step=0)
    # Step 7: only red 1 sees blue 0, which has moved into its cone.
    _placed(sim, "blue", 0, (60.0, 10.0 + 0.8 * r), vel=(0.0, 0.2))
    _placed(sim, "blue", 1, (95.0, 50.0))
    blue0_at_7 = sim.x_current[0 : sim.blue_nx].copy()
    sim._update_red_beliefs(step=7)

    for j in range(sim.num_reds):
        t = sim.red_belief_stores[j].targets
        assert set(t) == {0, 1}  # blue 2 was never seen
        assert t[0].provenance.observed_at == 7  # latest sighting wins
        np.testing.assert_array_equal(t[0].state, blue0_at_7)
        assert t[1].provenance.observed_at == 0  # timestamp preserved, not refreshed
    believed = sim._red_believed_blues(step=10)
    assert believed[0].keys() == believed[1].keys() == {0, 1}
    np.testing.assert_allclose(believed[0][1][0:2],
                               [60.0, 10.0 + r - 0.4 * 10 * sim.dt])  # aged 10 steps
    # A resolved red stops observing and stops receiving.
    sim.red_outcomes[1] = "neutralized"
    _placed(sim, "blue", 2, (10.0 + r, 10.0))
    sim._update_red_beliefs(step=12)
    assert 2 in sim.red_belief_stores[0].targets
    assert 2 not in sim.red_belief_stores[1].targets
    assert set(sim._red_believed_blues(step=12)) == {0}


def test_disabled_blues_are_never_sighted():
    sim = build_simulation(num_blues=2, num_reds=1, intruder_info="shared_beliefs")
    sim.setup_agents()
    from perception.belief.store import LocalBeliefStore
    sim.red_belief_stores = {0: LocalBeliefStore(0)}
    _placed(sim, "red", 0, (10.0, 10.0), theta=0.0)
    _placed(sim, "blue", 1, (10.0 + 0.5 * sim.fov_range, 10.0))
    sim.blue_disabled[1] = True
    sim._update_red_beliefs(step=3)
    assert 1 not in sim.red_belief_stores[0].targets


def test_believed_trajectories_are_constant_velocity_and_mask_unbelieved_blues():
    sim = build_simulation(num_blues=3, num_reds=1)
    solver = sim.solver
    nx = sim.blue_nx
    trajs, ids = solver._believed_blue_trajectories({2: np.array([1.0, 2.0, 0.5, -1.0, 0.0])}, nx)
    assert ids == [2]
    k = np.arange(solver.N + 1) * solver.dt
    np.testing.assert_allclose(trajs[2 * nx], 1.0 + 0.5 * k)
    np.testing.assert_allclose(trajs[2 * nx + 1], 2.0 - 1.0 * k)
    assert not trajs[0:nx].any()


def test_red_controllers_receive_belief_predictions_not_blue_plans():
    sim = build_simulation(num_blues=3, num_reds=2, sim_steps=40, intruder_info="shared_beliefs")
    seen = []
    for role, ctrl in sim.solver._red_role_controllers.items():
        original = ctrl.plan

        def spy(red_state, blue_trajs, detected_blue_indices=None, _orig=original, **kw):
            seen.append((np.array(blue_trajs, copy=True), list(detected_blue_indices or [])))
            return _orig(red_state, blue_trajs, detected_blue_indices=detected_blue_indices, **kw)

        ctrl.plan = spy
    # Force a known belief so the path is exercised whatever the reds see.
    fixed = np.array([30.0, 40.0, 0.5, -0.25, 0.0])
    sim._red_believed_blues = lambda step: {
        j: {1: fixed} for j in range(sim.num_reds) if sim.red_outcomes[j] == "active"
    }
    run_quiet(sim)
    assert seen
    nx = sim.blue_nx
    k = np.arange(sim.solver.N + 1) * sim.dt
    for trajs, ids in seen:
        assert ids == [1]
        np.testing.assert_allclose(trajs[nx], 30.0 + 0.5 * k)
        np.testing.assert_allclose(trajs[nx + 1], 40.0 - 0.25 * k)
        for i in range(sim.num_blues):
            row = trajs[i * nx : i * nx + 2]
            if i not in ids:
                assert not row.any()
            else:  # straight line at constant velocity, never a curved MPC plan
                second = np.diff(row, n=2, axis=1)
                np.testing.assert_allclose(second, 0.0, atol=1e-9)
