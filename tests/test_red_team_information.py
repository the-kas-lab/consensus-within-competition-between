"""What the red team actually knows — the intruder-team rule.

The intruder-team rule keeps red's information unchanged for Paper 1 and holds it **constant across
every experimental condition**. That is the whole point: red information is a
control variable, so a measured difference is attributable to the blue
manipulation. If any of it ever starts varying with `belief_mode`, every Exp 0
and the communication conditions comparison silently loses its attribution — and nothing would have caught
that before this file.

These also pin the *description*. The phrase "centralized and omniscient" was
carried in `episode_builder.py` and two result documents until the intruder-team rule audited it, and
it is wrong: reds are FOV-gated. The tests below are the reason nobody has to
take that audit on trust.
"""

import numpy as np
import pytest

from conftest import build_simulation, run_quiet
from planning.planner import RuleBasedRedRolePlanner
from planning.roles import RedRole


def _run(**kwargs):
    kwargs.setdefault("num_blues", 3)
    kwargs.setdefault("num_reds", 2)
    kwargs.setdefault("sim_steps", 120)
    return run_quiet(build_simulation(**kwargs))


# ---------------------------------------------------------------------------
# 1. Reds are FOV-gated, not omniscient
# ---------------------------------------------------------------------------


def test_reds_do_not_see_every_blue_all_the_time():
    """If reds were omniscient this is the test that would fail.

    `detected_blue_indices` comes from `sensing.team_detected_indices` over red
    observers with red bearings and the *same* FOV half-angle and range as blue
    (`testbed.py:958-962`). Replayed over every recorded step of an episode,
    there must be steps at which some active blue is not in it.

    Replayed from the recorder rather than sampled at termination: a single
    final-state check would pass or fail on where the agents happened to stop.
    """
    from perception import sensing

    sim = _run(seed=3001, sim_steps=200)
    arrays = sim.recorder.arrays()
    blue_hist, red_hist = arrays["blue_state"], arrays["red_state"]

    steps_checked = 0
    steps_with_a_gap = 0
    for index, record in enumerate(sim.recorder.steps):
        blue_state, red_state = blue_hist[index], red_hist[index]
        active_reds = [
            j for j, outcome in enumerate(record["red_outcomes"]) if outcome == "active"
        ]
        active_blues = [i for i, dead in enumerate(record["blue_disabled"]) if not dead]
        if not active_reds or not active_blues:
            continue
        detected = sensing.team_detected_indices(
            [(j, red_state[j * sim.red_nx: j * sim.red_nx + 2]) for j in active_reds],
            {j: red_state[j * sim.red_nx + 4] for j in active_reds},
            [(i, blue_state[i * sim.blue_nx: i * sim.blue_nx + 2]) for i in active_blues],
            sim.fov_half_angle_rad,
            sim.fov_range,
        )
        steps_checked += 1
        steps_with_a_gap += len(detected) < len(active_blues)

    assert steps_checked > 0, "no step had both an active red and an active blue"
    assert steps_with_a_gap > 0, (
        f"the red team detected every active blue at all {steps_checked} steps "
        "— reds are behaving omnisciently, contradicting the intruder-team rule's audit"
    )


def test_red_planner_ignores_undetected_blues():
    """The gate is real at the decision layer, not only in the sensing call.

    Two identical observations differing only in `detected_blue_indices`: with
    the near blue hidden, the red must ATTACK rather than EVADE, even though its
    exact position is present in `x_current`.
    """
    from planning.planner import PlannerObservation

    planner = RuleBasedRedRolePlanner(min_evade_distance=15.0)
    # One blue sitting right on top of the red, one far away.
    x_current = np.array([10.0, 10.0, 0, 0, 0,   90.0, 90.0, 0, 0, 0], float)
    red_states = np.array([11.0, 11.0, 0, 0, 0], float)
    common = dict(x_current=x_current, red_states=red_states, num_blues=2,
                  num_reds=1, sim_step=0, arena_size=100.0,
                  defense_center=np.array([20.0, 20.0]), disabled_blues=[])

    seen = planner.plan(PlannerObservation(detected_blue_indices=[0, 1], **common))
    hidden = planner.plan(PlannerObservation(detected_blue_indices=[1], **common))

    assert seen[0] == RedRole.EVADE, "a detected blue at range 1.4 did not trigger EVADE"
    assert hidden[0] == RedRole.ATTACK, (
        "the red evaded a blue it had not detected — the FOV gate is not applied "
        "at the planner, so reds ARE effectively omniscient"
    )


def test_red_evade_controller_zeroes_undetected_blues():
    """And at the control layer.

    `detected_weight_param` is 1.0 for detected blues and 0.0 otherwise
    (`red_controllers.py:72-77`), so an undetected blue contributes nothing to
    the avoidance cost even though its full planned trajectory was handed in.

    The controller is taken **off a live Simulation's solver** rather than
    constructed here: hand-building it would duplicate the testbed's
    construction and would keep passing after that construction changed.
    """
    from planning.roles import RedRole as _RedRole

    sim = build_simulation(num_blues=3, num_reds=2, seed=3001)
    sim.setup_agents()
    ctrl = sim.solver._red_role_controllers[_RedRole.EVADE]

    plans = np.zeros((sim.blue_nx * sim.num_blues, ctrl.N + 1))
    red_state = np.zeros(sim.red_nx)

    ctrl._set_extra_values(red_state, plans, detected_blue_indices=[1])
    weights = np.asarray(ctrl.opti.value(ctrl.detected_weight_param)).ravel()
    np.testing.assert_allclose(weights, [0.0, 1.0, 0.0])

    # None means "no gate supplied", which must fall back to all-visible --
    # otherwise a caller that forgot the argument would silently blind the red.
    ctrl._set_extra_values(red_state, plans, detected_blue_indices=None)
    weights = np.asarray(ctrl.opti.value(ctrl.detected_weight_param)).ravel()
    np.testing.assert_allclose(weights, [1.0, 1.0, 1.0])


def test_red_evade_receives_blue_planned_trajectories_not_just_positions():
    """The asymmetry the intruder-team rule declares, asserted rather than described.

    `EvadeController` is parameterised on `blue_traj_param`, shaped
    (blue_nx * n_blues, N+1) -- the blues' planned states across the whole MPC
    horizon, not a single current position. Blue has no counterpart: it
    extrapolates a red's position at constant velocity from its own belief
    (`records.py:94-100`).

    If this ever narrows to one column, the asymmetry has changed and the
    declaration in the intruder-team rule needs rewriting.
    """
    from planning.roles import RedRole as _RedRole

    sim = build_simulation(num_blues=3, num_reds=2, seed=3001)
    sim.setup_agents()
    ctrl = sim.solver._red_role_controllers[_RedRole.EVADE]
    rows, cols = ctrl.blue_traj_param.shape
    assert rows == sim.blue_nx * sim.num_blues
    assert cols == ctrl.N + 1 > 1, (
        "the evade controller no longer sees blue plans over a horizon"
    )


# ---------------------------------------------------------------------------
# 2. Red's information RULE is invariant to belief_mode
# ---------------------------------------------------------------------------
#
# The precise claim. What must be
# invariant is the *function* mapping physical state to red's information --
# NOT what red actually receives. Blue's information changes blue's decisions,
# which changes blue's trajectories, which changes which blues fall in a red's
# sensor cone. That variation is a downstream response to the treatment (the
# loop 06 C5 describes), not a second manipulation.
#
# So these tests hold the physical state IDENTICAL across belief modes and
# compare red's actual inputs. An earlier version asserted only that the red
# planner had the right type and that no red belief store existed, which would
# have passed through any number of mode-dependent leaks.


def _red_inputs_at(sim, blue_state, red_state, disabled):
    """Red's actual planner and controller inputs at an injected physical state.

    Drives the *shipped* paths -- `_compute_detected_indices` for sensing and
    `RuleBasedRedRolePlanner.plan` for the decision -- rather than
    recomputing either here.
    """
    from planning.planner import PlannerObservation

    sim.x_current = np.array(blue_state, dtype=float).copy()
    sim.red_state = np.array(red_state, dtype=float).copy()
    sim.blue_disabled = list(disabled)

    detected_red, detected_blue, detected_contacts, per_blue = (
        sim._compute_detected_indices(0)
    )
    obs = PlannerObservation(
        x_current=sim.x_current,
        red_states=sim.red_state,
        num_blues=sim.num_blues,
        num_reds=sim.num_reds,
        sim_step=0,
        arena_size=sim.arena_size,
        defense_center=sim.defense_center,
        disabled_blues=[i for i in range(sim.num_blues) if sim.blue_disabled[i]],
        detected_red_indices=sorted(detected_red),
        detected_blue_indices=sorted(detected_blue),
    )
    roles = sim.red_role_planner.plan(obs)
    return {
        "detected_blue_indices": sorted(detected_blue),
        "red_roles": {j: r.value for j, r in sorted(roles.items())},
    }


def _fixture_state(sim, rng):
    """A physical state with blues spread widely enough that some are outside
    red's FOV -- otherwise every mode trivially agrees on 'all detected' and the
    comparison has no power."""
    blue = np.zeros(sim.blue_nx * sim.num_blues)
    red = np.zeros(sim.red_nx * sim.num_reds)
    for i in range(sim.num_blues):
        blue[i * sim.blue_nx: i * sim.blue_nx + 2] = rng.uniform(5, 95, 2)
        blue[i * sim.blue_nx + 4] = rng.uniform(-np.pi, np.pi)
    for j in range(sim.num_reds):
        red[j * sim.red_nx: j * sim.red_nx + 2] = rng.uniform(5, 95, 2)
        red[j * sim.red_nx + 4] = rng.uniform(-np.pi, np.pi)
    return blue, red


def test_red_information_rule_is_identical_across_belief_modes():
    """The load-bearing property of the intruder-team rule, tested as a function.

    Four sims, one per belief mode, driven through **the same 40 injected
    physical states**. Red's detections and red's role decisions must agree
    exactly, every time. If any belief-mode-dependent information reached the
    red team -- a store consulted, a detection filtered, a role switched -- this
    fails.
    """
    modes = ["ground_truth", "shared", "local", "networked"]
    sims = {}
    for mode in modes:
        sim = build_simulation(num_blues=4, num_reds=2, seed=3007,
                               belief_mode=mode, r_comm=float("inf"))
        sim.setup_agents()
        sims[mode] = sim

    reference = sims["ground_truth"]
    saw_a_partial_detection = False
    for trial in range(40):
        rng = np.random.default_rng(9000 + trial)
        blue, red = _fixture_state(reference, rng)
        # Vary liveness too -- a disabled blue is dropped from red's targets.
        disabled = [bool(rng.integers(0, 2)) if i else False
                    for i in range(reference.num_blues)]

        results = {
            mode: _red_inputs_at(sims[mode], blue, red, disabled) for mode in modes
        }
        baseline = results["ground_truth"]
        n_live = sum(1 for d in disabled if not d)
        if 0 < len(baseline["detected_blue_indices"]) < n_live:
            saw_a_partial_detection = True
        for mode in modes[1:]:
            assert results[mode] == baseline, (
                f"red's inputs differ between ground_truth and {mode} at trial "
                f"{trial}: {baseline} vs {results[mode]}"
            )

    assert saw_a_partial_detection, (
        "no trial produced a partial detection, so the comparison never "
        "exercised the FOV gate and has no power"
    )


def test_red_evade_controller_inputs_are_identical_across_belief_modes():
    """The same invariance at the control layer.

    `EvadeController` receives the blues' planned trajectories and a per-blue
    detection weight. Both are compared here for identical injected state,
    because the planner test above cannot see the controller's gate.
    """
    from planning.roles import RedRole as _RedRole

    modes = ["ground_truth", "shared", "local", "networked"]
    captured = {}
    for mode in modes:
        sim = build_simulation(num_blues=4, num_reds=2, seed=3007,
                               belief_mode=mode, r_comm=float("inf"))
        sim.setup_agents()
        rng = np.random.default_rng(4242)
        blue, red = _fixture_state(sim, rng)
        inputs = _red_inputs_at(sim, blue, red, [False] * sim.num_blues)

        ctrl = sim.solver._red_role_controllers[_RedRole.EVADE]
        plans = np.tile(blue[:, None], (1, ctrl.N + 1))
        ctrl._set_extra_values(
            red[: sim.red_nx], plans,
            detected_blue_indices=inputs["detected_blue_indices"],
        )
        captured[mode] = (
            np.asarray(ctrl.opti.value(ctrl.detected_weight_param)).ravel().copy(),
            np.asarray(ctrl.opti.value(ctrl.blue_traj_param)).copy(),
        )

    base_w, base_traj = captured["ground_truth"]
    for mode in modes[1:]:
        w, traj = captured[mode]
        np.testing.assert_array_equal(w, base_w)
        np.testing.assert_allclose(traj, base_traj)


@pytest.mark.parametrize("mode", ["ground_truth", "shared", "local", "networked"])
def test_no_red_belief_state_exists_in_any_mode(mode):
    """Structural backstop for the tests above: reds hold no belief at all, so
    there is no red-side store for a future change to start gating."""
    # A short episode, not setup_agents() alone: belief stores are episode-scoped
    # and are built inside run().
    sim = _run(seed=3007, belief_mode=mode, sim_steps=40)
    assert sim.belief_mode == mode
    assert isinstance(sim.red_role_planner, RuleBasedRedRolePlanner)
    assert not hasattr(sim, "red_belief_stores")
    assert set(sim.belief_stores) == set(range(sim.num_blues))
