"""The pursuer aim-point rule: the NEUTRALIZE aim point between assignment applications.

 The pursuer aim-point rule (diagnosis in 31). Three modes:
``frozen`` (historical: the believed state at the application step, used as if
it were current), ``aligned`` (that same state advanced at its own velocity to
the solve step, so no newer information enters) and ``live`` (the agent's current
belief about the contact resolved at application).

**No episode runs here.** Simulations are constructed and ``setup_agents()``'d,
never ``run()``. The method is driven directly. The encounter tests close the
loop with a SCRIPTED red (no red controller, no testbed loop): each step, the
shipped ``_apply_assignment`` / ``_neutralize_aim_points`` pick the aim point,
and the shipped ``NeutralizeController`` is integrated with ``Agent.step_single``.
The red's belief is fed in as its exact state (an age-0 belief). That isolates
the aim point. It says nothing about how a reacting red would respond.
"""

import ast
import inspect
import textwrap

import numpy as np
import pytest

from conftest import build_simulation, capture_fd_output
from perception.belief import LocalBeliefStore
from planning.planner import PlannerOutput
from planning.roles import BlueRole
from scripts import arm_variants, network_conditions
from scripts.intercept_reachability import first_reachable_step, reachable_gaps
from scripts.episode_builder import build_episode
from simulation.recorder import config_hash, describe_config
from simulation.testbed import NEUTRALIZE_AIM_MODES, Simulation

N = BlueRole.NEUTRALIZE
DT = 0.1


def _prepared(mode="frozen", **kw):
    params = dict(num_blues=2, num_reds=2, neutralize_aim=mode)
    params.update(kw)
    sim = build_simulation(**params)
    sim.setup_agents()
    sim.red_outcomes = ["active"] * sim.num_reds
    return sim


def _plan(**targets):
    """_plan(b0=1) -> both agents NEUTRALIZE, blue 0 assigned red 1."""
    ids = {int(k[1:]): v for k, v in targets.items()}
    return PlannerOutput(roles={b: N for b in ids}, target_red_ids=ids)


def _state(x, y, vx=0.0, vy=0.0):
    return np.array([x, y, vx, vy, 0.0])


# --- the historical default -----------------------------------------------------

def test_frozen_is_the_default_and_hands_the_solver_the_frozen_dict_itself():
    sim = _prepared(mode="frozen")
    default = build_simulation(num_blues=2, num_reds=2)
    assert default.neutralize_aim == "frozen" == NEUTRALIZE_AIM_MODES[0]
    sim._apply_assignment(_plan(b0=1), 10, {0: {1: _state(10, 10, 1, -2)}})
    moved = {0: {1: _state(90, 90, 0, 0)}}
    out = sim._neutralize_aim_points(17, moved)
    assert out is sim.target_override, "frozen must pass the historical dict through untouched"
    assert np.allclose(out[0], [10, 10, 1, -2])
    assert sim.neutralize_aim_sources == {}


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="neutralize_aim"):
        build_simulation(neutralize_aim="predictive")


# --- aligned: the time-indexing correction ---------------------------------------

def test_aligned_advances_the_snapshot_by_its_own_velocity_and_ignores_newer_beliefs():
    sim = _prepared(mode="aligned")
    sim._apply_assignment(_plan(b0=1), 10, {0: {1: _state(10, 10, 1, -2)}})
    newer = {0: {1: _state(90, 90, 0, 0)}}  # must not be read
    at_application = sim._neutralize_aim_points(10, newer)
    assert np.allclose(at_application[0], [10, 10, 1, -2])
    later = sim._neutralize_aim_points(17, newer)
    assert np.allclose(later[0], [10 + 0.7, 10 - 1.4, 1, -2])
    assert sim.neutralize_aim_sources == {0: "aligned"}
    assert np.allclose(sim.target_override[0], [10, 10, 1, -2]), "the frozen record was mutated"


def test_aligned_is_the_application_time_belief_predicted_to_now():
    """'No newer information': the aligned aim point equals what the SAME belief
    record, as it stood at application, predicts for the solve step."""
    store = LocalBeliefStore(0, n_regions=1)
    store.observe_target(1, _state(40, 60, -1.5, 0.5), 4)
    record = store.targets[1]
    sim = _prepared(mode="aligned")
    sim._apply_assignment(_plan(b0=1), 10, {0: {1: record.predicted_state(10, DT)}})
    out = sim._neutralize_aim_points(19, {0: {}})
    assert np.allclose(out[0], np.asarray(record.predicted_state(19, DT))[0:4])


# --- live: the policy change --------------------------------------------------

def test_live_reads_the_current_belief_about_the_contact_resolved_at_application():
    sim = _prepared(mode="live")
    sim._apply_assignment(_plan(b0=1), 10, {0: {1: _state(10, 10, 1, -2)}})
    now = {0: {0: _state(5, 5), 1: _state(12, 7, 0.5, 0.5)}}
    out = sim._neutralize_aim_points(14, now)
    assert np.allclose(out[0], [12, 7, 0.5, 0.5]), "live must follow contact 1, not contact 0"
    assert sim.neutralize_aim_sources == {0: "belief"}
    assert sim.target_override_ids[0] == 1


def test_live_without_a_belief_entry_falls_back_to_the_aligned_snapshot():
    sim = _prepared(mode="live")
    sim._apply_assignment(_plan(b0=1), 10, {0: {1: _state(10, 10, 1, -2)}})
    out = sim._neutralize_aim_points(13, {0: {}})
    assert np.allclose(out[0], [10.3, 9.4, 1, -2])
    assert sim.neutralize_aim_sources == {0: "aligned"}


@pytest.mark.parametrize("mode", ["aligned", "live"])
def test_a_contact_roster_entry_is_not_a_belief_and_passes_through(mode):
    """Experiment II resolves contact-space ids, not belief keys; neither mode may
    advance or replace those entries."""
    sim = _prepared(mode=mode)
    sim._build_contact_roster()
    contact = next(c for c, (kind, _) in enumerate(sim.contact_roster) if kind == "red")
    plan = PlannerOutput(roles={0: N}, target_contact_ids={0: contact})
    sim._apply_assignment(plan, 10, {0: {}})
    assert sim.target_override_applied[0] == (10, None)
    before = np.array(sim.target_override[0], dtype=float)
    out = sim._neutralize_aim_points(15, {0: {contact: _state(1, 1)}})
    assert np.allclose(out[0], before)
    assert sim.neutralize_aim_sources == {0: "snapshot"}


# --- the command-delivery conditions rule: restricted re-resolution keeps its meaning -----------------------

@pytest.mark.parametrize("mode", ["aligned", "live"])
def test_resolve_only_keeps_each_untouched_agents_own_application_step(mode):
    sim = _prepared(mode=mode)
    plan = _plan(b0=0, b1=1)
    sim._apply_assignment(plan, 10, {0: {0: _state(10, 10, 1, 0)}, 1: {1: _state(20, 20, 0, 1)}})
    moved = {0: {0: _state(99, 99, 1, 0)}, 1: {1: _state(77, 77, 0, 1)}}
    sim._apply_assignment(plan, 11, moved, resolve_only={1})
    assert sim.target_override_applied == {0: (10, 0), 1: (11, 1)}
    if mode == "aligned":
        out = sim._neutralize_aim_points(13, moved)
        assert np.allclose(out[0], [10.3, 10, 1, 0]), "blue 0 took a fresher aim point"
        assert np.allclose(out[1], [77, 77.2, 0, 1])


# --- recording: only when non-default ------------------------------

def _config(sim):
    return describe_config(sim._init_args)


def test_the_option_is_recorded_only_when_it_is_not_the_default():
    implicit = build_simulation()
    explicit = build_simulation(neutralize_aim="frozen")
    assert "neutralize_aim" not in _config(implicit)
    assert config_hash(_config(implicit)) == config_hash(_config(explicit))
    for mode in ("aligned", "live"):
        sim = build_simulation(neutralize_aim=mode)
        assert _config(sim)["neutralize_aim"] == mode
        assert {k: v for k, v in _config(sim).items() if k != "neutralize_aim"} == _config(implicit)


def test_runners_are_unchanged_without_the_key_and_pass_it_through_with_it():
    exp4 = dict(arm_variants.BASE_CFG, family="multi_bearing", **arm_variants.ARMS[2])
    belief, exp5 = network_conditions.episode_cfg("multi_bearing", "REF", "C")
    for mode_of_belief, cfg in ((arm_variants.BELIEF_MODE, exp4), (belief, exp5)):
        assert "neutralize_aim" not in cfg
        hist = build_episode(2001, mode_of_belief, cfg)
        assert hist.neutralize_aim == "frozen" and "neutralize_aim" not in _config(hist)
        new = build_episode(2001, mode_of_belief, dict(cfg, neutralize_aim="live"))
        assert new.neutralize_aim == "live"
        assert {k: v for k, v in _config(new).items() if k != "neutralize_aim"} == _config(hist)


# --- the call site -----------------------------------------------------------------

def test_run_hands_the_solver_the_aim_points_not_the_raw_frozen_dict():
    """Static wiring check (an episode would be needed to observe it, and none
    may run here): run()'s solve_decentralized call passes
    ``target_override=self._neutralize_aim_points(step, believed_targets)``."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(Simulation.run)))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "solve_decentralized"
    ]
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    value = kw["target_override"]
    assert isinstance(value, ast.Call) and value.func.attr == "_neutralize_aim_points"
    assert [a.id for a in value.args] == ["step", "believed_targets"]


# --- controlled encounters (scripted red, no episode) --------------------------------

def _red_path(kind, n):
    rows = []
    if kind == "crossing":  # constant velocity across the pursuer's front
        p, v = np.array([8.0, -4.0]), np.array([-0.5, 1.0])
        rows = [np.r_[p + t * DT * v, v] for t in range(n + 1)]
    elif kind == "juke":  # reverses its acceleration every 10 steps, at the epoch steps
        p, v = np.array([6.0, 0.0]), np.array([0.0, 2.0])
        for t in range(n + 1):
            rows.append(np.r_[p.copy(), v.copy()])
            a = np.array([0.0, -2.0]) if (t // 10) % 2 == 0 else np.array([0.0, 2.0])
            v = np.clip(v + a * DT, -2.0, 2.0)
            p = p + DT * v
    elif kind == "flee":  # straight away along x at the axis limit
        p, v = np.array([3.0, 0.0]), np.array([2.0, 0.0])
        rows = [np.r_[p + t * DT * v, v] for t in range(n + 1)]
    return np.array(rows)


def _encounter(mode, kind, blue0, n=80):
    """(min distance, first step within 0.4 or None, reachable first step or None,
    minimum reachable gap). Assignment re-applied every 10 steps, as at 1 s epochs."""
    sim = build_simulation(num_blues=1, num_reds=1, neutralize_aim=mode)
    sim.setup_agents()
    sim.red_outcomes = ["active"]
    controller = sim.solver.neutralize_controller
    red = _red_path(kind, n)
    blue = np.array(blue0, dtype=float)
    plan = PlannerOutput(roles={0: N}, target_red_ids={0: 0})
    tol = sim.contact_tolerance
    dmin, hit = np.inf, None
    for t in range(n):
        believed = {0: {0: np.r_[red[t], 0.0]}}
        if t % sim.planning_interval_steps == 0:
            sim._apply_assignment(plan, t, believed)
        aim = sim._neutralize_aim_points(t, believed)[0]
        with capture_fd_output():
            u0, _, _, ok = controller.plan(blue, np.r_[aim, 0.0], None, agent_id=0)
        assert ok
        blue = sim.blue.step_single(blue, u0)
        d = float(np.linalg.norm(blue[0:2] - red[t + 1, 0:2]))
        dmin = min(dmin, d)
        if hit is None and d <= tol:
            hit = t + 1
    gaps = reachable_gaps(np.asarray(blue0, float), red[1:, 0:2], DT, sim.blue_a_max, sim.blue_v_max)
    return dmin, hit, first_reachable_step(gaps, tol), min(gaps)


def test_reachable_crossing_frozen_trails_while_aligned_and_live_reach_contact():
    """A constant-velocity red: the CV model is exact, so the only error left in
    the frozen aim point is its time index. Frozen trails the red and never
    comes within the threshold; aligned and live reach it within two steps of
    the box-limit minimum time."""
    results = {m: _encounter(m, "crossing", [0, 0, 0, 0, 0]) for m in NEUTRALIZE_AIM_MODES}
    reachable_at = results["frozen"][2]
    assert reachable_at is not None
    assert results["frozen"][1] is None and results["frozen"][0] > 0.4
    for mode in ("aligned", "live"):
        assert results[mode][1] is not None and results[mode][1] <= reachable_at + 2


def test_reachable_juke_live_reaches_contact_before_frozen():
    """A red that reverses at the snapshot steps (the 3870 pattern, scripted):
    every frozen snapshot carries the previous manoeuvre's velocity."""
    results = {m: _encounter(m, "juke", [0, 0, 0, 0, 0]) for m in NEUTRALIZE_AIM_MODES}
    assert results["live"][2] is not None
    assert results["live"][1] is not None and results["frozen"][1] is not None
    assert results["live"][1] < results["frozen"][1]
    assert results["aligned"][1] < results["frozen"][1]


def test_physically_infeasible_flee_is_not_rescued_by_any_mode():
    """A red fleeing along x at the axis limit, 3 units ahead of a pursuer at the
    same speed: no admissible motion closes, and no aim-point mode claims to."""
    for mode in NEUTRALIZE_AIM_MODES:
        dmin, hit, reachable_at, floor = _encounter(mode, "flee", [0, 0, 2, 0, 0])
        assert reachable_at is None and floor > 2.9
        assert hit is None and dmin > 0.4
        # OSQP satisfies the velocity box to its tolerance, not exactly; over 80
        # steps that is worth a few hundredths, never a closure.
        assert dmin >= floor - 0.05


def test_reachability_bound_is_exact_on_one_axis():
    """The helper's per-axis extremes are the greedy bang-bang ones."""
    gaps = reachable_gaps(np.array([0.0, 0.0, 0.0, 0.0]), [[0.02, 0.0], [10.0, 0.0]], 0.1, 2.0, 2.0)
    assert gaps[0] == pytest.approx(0.0)  # 0.1*0.2 = 0.02 reachable in one step
    assert gaps[1] == pytest.approx(10.0 - 0.06)  # v = 0.2 then 0.4: p = 0.02 + 0.04
