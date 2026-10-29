"""The auction step 1: tasks are generated from one agent's belief, and only from it.

 The geometry every test uses is
taken off a live Simulation's solver, not hand-built: a SCREEN or DEFEND
location that drifted from what the controllers actually steer to would make
every ETA downstream describe a journey no agent makes.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import build_simulation  # noqa: E402
from coordination.tasks import (  # noqa: E402
    DEFEND,
    NEUTRALIZE,
    RECON,
    SCREEN,
    TaskContext,
    closest_point_on_polygon,
    generate_tasks,
)
from perception.belief import LocalBeliefStore  # noqa: E402
from planning.roles import BlueRole  # noqa: E402


@pytest.fixture(scope="module")
def live():
    sim = build_simulation(num_blues=4, num_reds=2)
    sim.setup_agents()
    return sim


def _view(sim, agent_id=0, now=40, contacts=None, n_regions=0, observed_cells=()):
    store = LocalBeliefStore(agent_id, n_regions=n_regions)
    for contact_id, (state, observed_at) in (contacts or {}).items():
        store.observe_target(contact_id, np.asarray(state, dtype=float), observed_at)
    for cell in observed_cells:
        store.observe_coverage(cell, now)
    return store.snapshot(
        now=now,
        dt=0.1,
        self_state=np.array([50.0, 50.0, 0.0, 0.0, 0.0]),
        coverage_centers=sim.coverage_tracker.centers if n_regions else None,
    )


def _ids(tasks, kind=None):
    return [t.task_id for t in tasks if kind is None or t.kind == kind]


# --- information isolation ------------------------------------------------

def test_a_contact_the_agent_does_not_believe_in_generates_no_task(live):
    """Fog of war at the task layer. There is no task for contact 7 because the
    agent has no belief about contact 7 -- the only way a task can exist."""
    ctx = TaskContext.from_simulation(live)
    tasks = generate_tasks(_view(live, contacts={3: ([20, 20, 0, 0, 0], 30)}), ctx)

    assert "NEUTRALIZE:3:0" in _ids(tasks)
    assert "SCREEN:3" in _ids(tasks)
    assert not any(t.contact_id == 7 for t in tasks)


def test_the_task_location_states_the_belief_not_the_truth(live):
    """The L1/L2-shaped test. The agent saw contact 0 at (20, 20) thirty steps
    ago. Wherever it truly is now, the task is placed where the agent believes
    it is -- the generator has no access to anything else."""
    ctx = TaskContext.from_simulation(live)
    view = _view(live, now=40, contacts={0: ([20.0, 20.0, 0.0, 0.0, 0.0], 10)})
    (task,) = [t for t in generate_tasks(view, ctx) if t.task_id == "NEUTRALIZE:0:0"]
    assert task.location == pytest.approx((20.0, 20.0))


def test_neutralize_location_is_the_extrapolated_belief_not_the_raw_sighting(live):
    """Where the agent believes the contact is *now* -- the same prediction the
    solver steers to (``_believed_targets``) -- not where it was last seen."""
    ctx = TaskContext.from_simulation(live)
    view = _view(live, now=40, contacts={0: ([20.0, 20.0, 1.0, 0.0, 0.0], 10)})
    (task,) = [t for t in generate_tasks(view, ctx) if t.task_id == "NEUTRALIZE:0:0"]
    # 20 + 1.0 * (30 steps * 0.1) = 23
    assert task.location == pytest.approx((23.0, 20.0))


def test_max_age_withholds_a_stale_contact_exactly_as_the_prompt_builder_does(live):
    """The rule planner, the prompt builder and the task generator must reason
    over the same contact set at identical settings, or an LLM and an auction
    fed 'the same belief' would not be."""
    from planning.prompt_builder import PromptContext, build_user_message

    ctx = TaskContext.from_simulation(live)
    view = _view(live, now=100, contacts={0: ([20, 20, 0, 0, 0], 55),
                                          1: ([60, 60, 0, 0, 0], 95)})
    for max_age in (None, 50, 10):
        from_tasks = {t.contact_id for t in generate_tasks(view, ctx, max_age=max_age)
                      if t.contact_id is not None}
        prompt = build_user_message(view, PromptContext(arena_size=100.0), max_age=max_age)
        from_prompt = {c for c in (0, 1) if f"contact_id={c} " in prompt}
        assert from_tasks == from_prompt, f"contact sets differ at max_age={max_age}"


# --- the consensus precondition --------------------------------------------

def test_agents_with_identical_beliefs_generate_identical_task_ids(live):
    """CBAA's consensus is keyed by task id. Two agents believing the same thing
    must name the same tasks the same way, or they would auction past each
    other. Agent position must not enter the id."""
    ctx = TaskContext.from_simulation(live)
    contacts = {0: ([20, 20, 0, 0, 0], 30), 1: ([70, 30, 0, 0, 0], 35)}
    a = generate_tasks(_view(live, agent_id=0, contacts=contacts, n_regions=16), ctx)
    b = generate_tasks(_view(live, agent_id=3, contacts=contacts, n_regions=16), ctx)
    assert _ids(a) == _ids(b)
    assert [t.location for t in a] == [t.location for t in b]


def test_generation_is_pure_and_in_canonical_order(live):
    ctx = TaskContext.from_simulation(live)
    contacts = {1: ([70, 30, 0, 0, 0], 35), 0: ([20, 20, 0, 0, 0], 30)}
    first = generate_tasks(_view(live, contacts=contacts, n_regions=16), ctx)
    second = generate_tasks(_view(live, contacts=contacts, n_regions=16), ctx)
    assert first == second
    kinds = [t.kind for t in first]
    assert kinds == sorted(kinds, key=[NEUTRALIZE, SCREEN, DEFEND, RECON].index)


# --- slots: the duplicate-pursuit arm -------------------------------------

def test_slots_two_gives_two_distinct_pursuit_tasks_per_contact(live):
    """How Exp 4 permits two pursuers without changing CBAA: two *tasks*, so the
    one-agent-per-task invariant still holds. SCREEN is not duplicated."""
    ctx = TaskContext.from_simulation(live, slots=2)
    tasks = generate_tasks(_view(live, contacts={4: ([20, 20, 0, 0, 0], 30)}), ctx)

    assert _ids(tasks, NEUTRALIZE) == ["NEUTRALIZE:4:0", "NEUTRALIZE:4:1"]
    assert _ids(tasks, SCREEN) == ["SCREEN:4"]


# --- geometry pinned to the controllers -----------------------------------

def test_screen_location_is_where_the_defend_controller_steers(live):
    """SCREEN is realised by the existing DefendController with a chosen
    contact. Its location must be the controller's own interposition
    point, computed by the controller taken off the live solver."""
    ctx = TaskContext.from_simulation(live)
    controller = live.solver._blue_role_controllers[BlueRole.DEFEND]
    for believed in ([20.0, 20.0], [5.0, 90.0], [95.0, 10.0], [60.0, 40.0]):
        view = _view(live, contacts={0: ([*believed, 0, 0, 0], 40)})
        (screen,) = [t for t in generate_tasks(view, ctx) if t.kind == SCREEN]
        expected = controller._closest_point_on_polygon(np.asarray(believed))
        assert screen.location == pytest.approx(tuple(expected))


def test_closest_point_helper_matches_the_controller_everywhere(live):
    """Pinned over a grid, not four hand-picked points: one diverging branch of
    the segment projection is enough to break every SCREEN ETA near it."""
    controller = live.solver._blue_role_controllers[BlueRole.DEFEND]
    polygon = TaskContext.from_simulation(live).polygon_array()
    for x in np.linspace(0, 100, 11):
        for y in np.linspace(0, 100, 11):
            p = np.array([x, y])
            assert closest_point_on_polygon(p, polygon) == pytest.approx(
                controller._closest_point_on_polygon(p)
            )


def test_defend_tasks_are_the_solvers_own_stations(live):
    ctx = TaskContext.from_simulation(live)
    tasks = generate_tasks(_view(live), ctx)
    stations = {t.station_id: t.location for t in tasks if t.kind == DEFEND}
    expected = {k: tuple(v) for k, v in live.solver.defend_stations.items()}
    assert set(stations) == set(expected)
    for k in expected:
        assert stations[k] == pytest.approx(expected[k])


def test_a_station_needs_no_contact_to_exist(live):
    """'I know nothing' must still be a well-defined posture."""
    ctx = TaskContext.from_simulation(live)
    tasks = generate_tasks(_view(live, contacts={}), ctx)
    assert _ids(tasks, NEUTRALIZE) == [] and _ids(tasks, SCREEN) == []
    assert len(_ids(tasks, DEFEND)) == len(live.solver.defend_stations)


# --- recon comes from the agent's own coverage belief ---------------------

def test_recon_tasks_come_from_the_agents_own_coverage_belief(live):
    ctx = TaskContext.from_simulation(live)
    with_cov = generate_tasks(_view(live, n_regions=16), ctx)
    without = generate_tasks(_view(live, n_regions=0), ctx)
    assert len(_ids(with_cov, RECON)) == 16
    assert _ids(without, RECON) == [], "no coverage belief, no recon task"


def test_context_off_a_live_simulation_is_frozen(live):
    import dataclasses

    ctx = TaskContext.from_simulation(live)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.slots = 3
    assert isinstance(ctx.defense_polygon, tuple)
    assert isinstance(ctx.stations, tuple)


def test_context_needs_no_setup_agents_call():
    """The solver, and so its geometry, exists from construction
    (testbed.py:480). A context taken before setup_agents() must equal one
    taken after, or tests and harnesses would silently disagree by call order."""
    fresh = build_simulation(num_blues=4, num_reds=2)
    before = TaskContext.from_simulation(fresh)
    fresh.setup_agents()
    assert TaskContext.from_simulation(fresh) == before


def test_context_refuses_an_object_with_no_solver():
    sim = build_simulation()
    sim.solver = None
    with pytest.raises(ValueError, match="no solver"):
        TaskContext.from_simulation(sim)
