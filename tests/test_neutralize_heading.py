"""The pursuer sensor-heading rule: where a NEUTRALIZE agent points its sensor.

 The pursuer sensor-heading rule (diagnosis in 31 §2). Under
``neutralize_heading="pursued_contact"`` an agent flying the NEUTRALIZE
controller points at its effective contact's believed position. That contact
ranks ahead of any other own detection and of the grace hold, and the turn takes
the nearest equivalent angle. ``"inherited"`` (the default) is today's behaviour:
own detection only.

**No episode runs here.** Simulations are constructed and ``setup_agents()``'d,
never ``run()``. The solver is driven directly with fabricated beliefs (the shape
test_defend_target_override.py and test_information_isolation.py use), and the
belief-mode tests build those beliefs with the testbed's own ``_update_beliefs``
and ``_believed_targets``.

Controls are compared with ``atol=1e-9``, as in test_defend_target_override.py.
"""

import ast
import copy
import inspect
import math
import textwrap

import numpy as np
import pytest

from conftest import build_simulation, capture_fd_output
from perception.belief import LocalBeliefStore
from planning.roles import BlueRole
from scripts import arm_variants, network_conditions
from scripts.episode_builder import build_episode
from simulation.recorder import config_hash, describe_config
from simulation.testbed import NEUTRALIZE_HEADING_MODES, Simulation

N, D, R = BlueRole.NEUTRALIZE, BlueRole.DEFEND, BlueRole.RECON
PURSUED = "pursued_contact"


def _prepared(heading=PURSUED, num_blues=2, num_reds=2, **kw):
    sim = build_simulation(num_blues=num_blues, num_reds=num_reds,
                           neutralize_heading=heading, **kw)
    sim.setup_agents()
    sim.solver.set_fixed_adjacency_from_state(sim.x_current, sim.red_state)
    return sim


def _place_blue(sim, i, x, y, theta=0.0):
    nx = sim.blue_nx
    sim.x_current[i * nx : i * nx + 5] = [x, y, 0.0, 0.0, theta]


def _reds(sim, *positions):
    return np.concatenate([np.array([x, y, 0.0, 0.0, 0.0])[: sim.red_nx] for x, y in positions])


def _s(x, y, vx=0.0, vy=0.0):
    return np.array([x, y, vx, vy, 0.0])


def _solve(sim, roles, believed, red_state=None, **kw):
    sim.solver.reset_warm_starts()
    with capture_fd_output():
        u, _, _, _ = sim.solver.solve_decentralized(
            x_current=sim.x_current,
            red_states=_reds(sim, (90, 10), (10, 90)) if red_state is None else red_state,
            role_assignment=roles,
            believed_targets=believed,
            **kw,
        )
    return np.asarray(u).copy(), dict(sim.solver.bearing_sources), dict(sim.solver.effective_targets)


def _heading_target(sim, controller):
    """The heading-target parameter the last solve of ``controller`` used, and
    whether the tracking term was active."""
    return (float(controller.opti.value(controller.bearing_target_param)),
            float(controller.opti.value(controller.bearing_active_param)))


def _bearing(sim, i, point):
    p = sim.x_current[i * sim.blue_nx : i * sim.blue_nx + 2]
    return math.atan2(point[1] - p[1], point[0] - p[0])


# --- the historical default -------------------------------------------------------

def test_inherited_is_the_default_and_leaves_a_blind_pursuer_without_a_heading_target():
    sim = _prepared(heading="inherited")
    assert build_simulation().neutralize_heading == "inherited" == NEUTRALIZE_HEADING_MODES[0]
    assert sim.solver.neutralize_heading == "inherited"
    _place_blue(sim, 0, 50, 50)
    believed = {0: {1: _s(40, 40)}, 1: {}}
    omitted = _solve(sim, {0: N, 1: D}, believed, target_override={0: _s(40, 40)[0:4]},
                     target_override_ids={0: 1})
    passed = _solve(sim, {0: N, 1: D}, believed, target_override={0: _s(40, 40)[0:4]},
                    target_override_ids={0: 1}, own_detected_reds={0: set()},
                    pursued_belief_keys={0: 1})
    assert omitted[1][0] is None, "the 3870 condition: pursuing with no heading target"
    assert np.allclose(omitted[0], passed[0], atol=1e-9) and omitted[1] == passed[1]
    assert _heading_target(sim, sim.solver.neutralize_controller)[1] == 0.0


def test_a_solver_built_outside_simulation_defaults_to_inherited():
    """Simulation always sets the mode, so this pins the solver's OWN default --
    what a directly constructed solver (verification scripts) runs with."""
    assert build_simulation(neutralize_heading=PURSUED)._build_solver().neutralize_heading == "inherited"


def test_an_unknown_heading_mode_is_refused():
    with pytest.raises(ValueError, match="neutralize_heading"):
        build_simulation(neutralize_heading="scan")


# --- pursued_contact: belief, not truth --------------------------------------------

def test_a_blind_pursuer_points_at_its_believed_contact_not_at_the_truth():
    sim = _prepared()
    _place_blue(sim, 0, 50, 50, theta=0.0)
    believed_point = (40.0, 40.0)               # behind and to the left
    truth = _reds(sim, (90, 10), (55, 70))      # red 1 is really elsewhere
    u, sources, eff = _solve(
        sim, {0: N, 1: D}, {0: {1: _s(*believed_point)}, 1: {}}, red_state=truth,
        target_override={0: _s(*believed_point)[0:4]}, target_override_ids={0: 1},
        pursued_belief_keys={0: 1}, own_detected_reds={},
    )
    assert eff[0]["source"] == "assigned" and eff[0]["contact_id"] == 1
    assert sources[0] == "pursued_belief"
    target, active = _heading_target(sim, sim.solver.neutralize_controller)
    assert active == 1.0
    assert target == pytest.approx(_bearing(sim, 0, believed_point))
    assert abs(target - _bearing(sim, 0, (55, 70))) > 0.5, "pointed at the true red"


def test_the_heading_follows_the_live_belief_while_translation_keeps_the_frozen_aim():
    """The pursuer sensor-heading rule and the pursuer aim-point rule are separable: the heading reads the agent's current belief
    about the contact even when the aim point handed to translation is an older
    snapshot."""
    sim = _prepared()
    _place_blue(sim, 0, 50, 50)
    _, sources, _ = _solve(
        sim, {0: N, 1: D}, {0: {1: _s(50, 70)}, 1: {}},
        target_override={0: _s(70, 50)[0:4]}, target_override_ids={0: 1},
        pursued_belief_keys={0: 1},
    )
    assert sources[0] == "pursued_belief"
    assert _heading_target(sim, sim.solver.neutralize_controller)[0] == pytest.approx(math.pi / 2)


@pytest.mark.parametrize("belief_mode", ["shared", "local"])
def test_the_policy_reads_beliefs_built_by_the_testbed_in_both_modes(belief_mode):
    """Shared: blue 0 has never seen red 1; blue 1 did. Local: blue 0 saw it
    itself 20 steps ago and has not since. Either way blue 0's heading target
    is its OWN extrapolated belief, built by the testbed's belief machinery."""
    sim = _prepared(belief_mode=belief_mode)
    sim.belief_stores = {
        i: LocalBeliefStore(i, n_regions=len(sim.coverage_tracker.centers))
        for i in range(sim.num_blues)
    }
    sim.red_state[1 * sim.red_nx : 1 * sim.red_nx + 4] = [30.0, 60.0, 1.0, -0.5]
    observer = 1 if belief_mode == "shared" else 0
    bearings = sim._compute_bearings(sim.x_current, sim.num_blues, sim.blue_nx)
    sim._update_beliefs(10, {observer: {1}}, blue_observers=None, blue_bearings=bearings)
    sim._update_beliefs(30, {}, blue_observers=None, blue_bearings=bearings)
    believed = sim._believed_targets(30)
    assert 1 in believed[0], "blue 0 holds no belief about red 1"
    expected = np.array([30.0 + 20 * 0.1 * 1.0, 60.0 - 20 * 0.1 * 0.5])
    assert np.allclose(np.asarray(believed[0][1])[0:2], expected)
    _place_blue(sim, 0, 50, 50, theta=0.0)
    _, sources, eff = _solve(sim, {0: N, 1: D}, believed, red_state=sim.red_state.copy(),
                             own_detected_reds={})
    assert eff[0]["contact_id"] == 1
    assert sources[0] == "pursued_belief"
    assert _heading_target(sim, sim.solver.neutralize_controller)[0] == pytest.approx(
        _bearing(sim, 0, expected))


# --- priority -------------------------------------------------------------------------

def test_direct_detection_of_the_pursued_contact_is_labelled_and_used():
    sim = _prepared()
    _place_blue(sim, 0, 50, 50)
    seen = (60.0, 52.0)
    _, sources, _ = _solve(
        sim, {0: N, 1: D}, {0: {1: _s(*seen)}, 1: {}},
        target_override={0: _s(*seen)[0:4]}, target_override_ids={0: 1},
        pursued_belief_keys={0: 1}, own_detected_reds={0: {1}},
        bearing_target_override={0: np.array(seen)},
    )
    assert sources[0] == "pursued_detection"
    assert _heading_target(sim, sim.solver.neutralize_controller)[0] == pytest.approx(
        _bearing(sim, 0, seen))


def test_the_pursued_contact_outranks_another_detected_red_and_the_grace_hold():
    """Own sensor sees red 0 (a bystander, or a held grace position); the agent
    pursues red 1. The pursued contact wins."""
    sim = _prepared()
    _place_blue(sim, 0, 50, 50)
    bystander, pursued = (60.0, 50.0), (50.0, 30.0)
    _, sources, _ = _solve(
        sim, {0: N, 1: D}, {0: {0: _s(*bystander), 1: _s(*pursued)}, 1: {}},
        target_override={0: _s(*pursued)[0:4]}, target_override_ids={0: 1},
        pursued_belief_keys={0: 1}, own_detected_reds={0: {0}},
        bearing_target_override={0: np.array(bystander)},
    )
    assert sources[0] == "pursued_belief"
    assert _heading_target(sim, sim.solver.neutralize_controller)[0] == pytest.approx(-math.pi / 2)


def test_after_retargeting_the_heading_follows_the_effective_contact():
    """No override (its red resolved, say): the solver pursues the nearest
    believed contact, and the heading follows THAT contact."""
    sim = _prepared()
    _place_blue(sim, 0, 50, 50)
    near, far = (50.0, 60.0), (90.0, 90.0)
    _, sources, eff = _solve(sim, {0: N, 1: D}, {0: {0: _s(*far), 1: _s(*near)}, 1: {}})
    assert eff[0] == {"source": "nearest_believed", "contact_id": 1, "controller": "neutralize"}
    assert sources[0] == "pursued_belief"
    assert _heading_target(sim, sim.solver.neutralize_controller)[0] == pytest.approx(math.pi / 2)


def test_a_contact_roster_id_without_a_belief_key_points_at_the_aim_point():
    """Contact id 1 (roster space) collides with belief key 1, a DIFFERENT red;
    without a belief key the policy must not read that belief."""
    sim = _prepared()
    _place_blue(sim, 0, 50, 50)
    aim = (50.0, 40.0)
    _, sources, _ = _solve(sim, {0: N, 1: D}, {0: {1: _s(50.0, 90.0)}, 1: {}},
                           target_override={0: _s(*aim)[0:4]}, target_override_ids={0: 1})
    assert sources[0] == "pursued_aim"
    assert _heading_target(sim, sim.solver.neutralize_controller)[0] == pytest.approx(-math.pi / 2)


# --- what does not change ----------------------------------------------------------------

@pytest.mark.parametrize("detected", [False, True])
def test_a_neutralize_agent_believing_nothing_keeps_the_historical_heading(detected):
    """Station hold through the DEFEND controller: detection (or nothing) exactly
    as before, with the historical turn convention."""
    outs = {}
    for heading in NEUTRALIZE_HEADING_MODES:
        sim = _prepared(heading=heading)
        _place_blue(sim, 0, 50, 50, theta=3.0)
        kw = {"bearing_target_override": {0: np.array([40.0, 49.0])}} if detected else {}
        outs[heading] = _solve(sim, {0: N, 1: D}, {0: {}, 1: {}}, **kw)
        assert outs[heading][2][0]["controller"] == "defend"
        assert outs[heading][1][0] == ("detection" if detected else None)
    assert np.allclose(outs["inherited"][0], outs[PURSUED][0], atol=1e-9)


@pytest.mark.parametrize("role", [D, R])
def test_defend_and_recon_are_identical_under_either_heading_mode(role):
    believed = {0: {1: _s(40, 40)}, 1: {0: _s(70, 70)}}
    kw = dict(bearing_target_override={1: np.array([70.0, 70.0])}, own_detected_reds={1: {0}},
              recon_target_override={0: np.array([20.0, 20.0])})
    outs = []
    for heading in NEUTRALIZE_HEADING_MODES:
        sim = _prepared(heading=heading)
        _place_blue(sim, 0, 50, 50, theta=3.0)
        _place_blue(sim, 1, 60, 60, theta=-3.0)
        outs.append(_solve(sim, {0: role, 1: role}, believed, **kw))
    assert np.allclose(outs[0][0], outs[1][0], atol=1e-9)
    assert outs[0][1] == outs[1][1] and outs[0][2] == outs[1][2]


# --- turn direction -----------------------------------------------------------------------

def test_the_pursued_heading_turns_the_short_way_across_the_branch_cut():
    """Heading 3.0 rad; contact at bearing -3.0 rad: 0.28 rad away the short way.
    The historical detection path aims at -3.0 itself (a 6.0 rad turn)."""
    contact = (50.0 + 10 * math.cos(-3.0), 50.0 + 10 * math.sin(-3.0))
    sim = _prepared()
    _place_blue(sim, 0, 50, 50, theta=3.0)
    u, sources, _ = _solve(sim, {0: N, 1: D}, {0: {1: _s(*contact)}, 1: {}},
                           target_override={0: _s(*contact)[0:4]}, target_override_ids={0: 1},
                           pursued_belief_keys={0: 1})
    target, _ = _heading_target(sim, sim.solver.neutralize_controller)
    assert target == pytest.approx(3.0 + (2 * math.pi - 6.0))
    assert u[2] > 0, "turned the long way"

    old = _prepared(heading="inherited")
    _place_blue(old, 0, 50, 50, theta=3.0)
    u_old, src_old, _ = _solve(old, {0: N, 1: D}, {0: {1: _s(*contact)}, 1: {}},
                               target_override={0: _s(*contact)[0:4]},
                               target_override_ids={0: 1},
                               bearing_target_override={0: np.array(contact)})
    assert src_old[0] == "detection"
    assert _heading_target(old, old.solver.neutralize_controller)[0] == pytest.approx(-3.0)
    assert u_old[2] < 0, "the historical convention is expected to take the long way here"


# --- no negative evidence -------------------------------------------------------------------

def test_a_believed_point_in_view_with_nothing_detected_deletes_nothing():
    sim = _prepared()
    _place_blue(sim, 0, 50, 50, theta=0.0)
    in_view = (60.0, 50.0)  # dead ahead, well inside range
    believed = {0: {1: _s(*in_view)}, 1: {}}
    before = copy.deepcopy(believed)
    _solve(sim, {0: N, 1: D}, believed, own_detected_reds={0: set()})
    assert set(believed[0]) == {1}
    assert np.array_equal(believed[0][1], before[0][1])


# --- recording, runners, wiring ---------------------------------------------------------------

def _config(sim):
    return describe_config(sim._init_args)


def test_the_option_is_recorded_only_when_it_is_not_the_default():
    implicit = build_simulation()
    explicit = build_simulation(neutralize_heading="inherited")
    assert "neutralize_heading" not in _config(implicit)
    assert config_hash(_config(implicit)) == config_hash(_config(explicit))
    sim = build_simulation(neutralize_heading=PURSUED)
    assert sim.solver.neutralize_heading == PURSUED
    assert _config(sim)["neutralize_heading"] == PURSUED
    assert {k: v for k, v in _config(sim).items() if k != "neutralize_heading"} == _config(implicit)


def test_runners_are_unchanged_without_the_key_and_pass_it_through_with_it():
    exp4 = dict(arm_variants.BASE_CFG, family="multi_bearing", **arm_variants.ARMS[2])
    belief, exp5 = network_conditions.episode_cfg("multi_bearing", "NX", "I")
    for mode_of_belief, cfg in ((arm_variants.BELIEF_MODE, exp4), (belief, exp5)):
        assert "neutralize_heading" not in cfg
        hist = build_episode(2001, mode_of_belief, cfg)
        assert hist.solver.neutralize_heading == "inherited"
        assert "neutralize_heading" not in _config(hist)
        new = build_episode(2001, mode_of_belief, dict(cfg, neutralize_heading=PURSUED))
        assert new.solver.neutralize_heading == PURSUED
        assert {k: v for k, v in _config(new).items() if k != "neutralize_heading"} == _config(hist)


def test_run_hands_the_solver_its_own_detections_and_the_belief_keys():
    """Static wiring check of run()'s one solve_decentralized call (observing
    it needs an episode, and none may run here)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(Simulation.run)))
    (call,) = [n for n in ast.walk(tree)
               if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "solve_decentralized"]
    kw = {k.arg: k.value for k in call.keywords}
    assert isinstance(kw["own_detected_reds"], ast.Name)
    assert kw["own_detected_reds"].id == "per_blue_detected_reds"
    keys = kw["pursued_belief_keys"]
    assert isinstance(keys, ast.DictComp)
    assert "target_override_applied" in ast.unparse(keys) and "key is not None" in ast.unparse(keys)


def test_the_pursued_belief_keys_run_would_pass_name_only_belief_sourced_overrides():
    """The comprehension at the call site, evaluated on a live Simulation's
    state after a real application: belief-sourced entries only."""
    from planning.planner import PlannerOutput

    sim = _prepared(num_blues=2, num_reds=2)
    sim.red_outcomes = ["active", "active"]
    sim._build_contact_roster()
    contact = next(c for c, (kind, _) in enumerate(sim.contact_roster) if kind == "red")
    plan = PlannerOutput(roles={0: N, 1: N}, target_red_ids={0: 1},
                         target_contact_ids={1: contact})
    sim._apply_assignment(plan, 20, {0: {1: _s(10, 10)}, 1: {}})
    keys = {b: k for b, (_, k) in sim.target_override_applied.items() if k is not None}
    assert keys == {0: 1}
