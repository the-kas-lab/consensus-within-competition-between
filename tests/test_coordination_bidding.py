"""The auction step 2: bids are deterministic, belief-only, and bounded at the LLM seam.

 Nothing here tests that a
bid *wins* -- weights are fixed before any outcome is observed, and tuning them
on seeds would be a researcher degree of freedom. The tests pin properties: what
a bid may read, how each term moves, and what the LLM will be able to touch.
"""

import dataclasses

import coordination.bidding as bidding_mod
import inspect
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import build_simulation  # noqa: E402
from coordination.bidding import (  # noqa: E402
    PREFERRED_BONUS,
    PRIORITY_MULTIPLIER,
    PROFILES,
    BidContext,
    BidModifiers,
    bid,
    bids,
    urgency,
)
from coordination.tasks import TaskContext, generate_tasks, tasks_by_id  # noqa: E402
from perception.belief import LocalBeliefStore  # noqa: E402

MATES3 = {1: np.zeros(5), 2: np.zeros(5), 3: np.zeros(5)}


@pytest.fixture(scope="module")
def ctx():
    sim = build_simulation(num_blues=4, num_reds=2)
    tc = TaskContext.from_simulation(sim)
    return sim, tc, BidContext.from_simulation(sim, tc)


def _view(sim, me=(55.0, 55.0), now=40, contacts=None, n_regions=0, mates=None, fresh_cells=()):
    store = LocalBeliefStore(0, n_regions=n_regions)
    for cid, (state, t) in (contacts or {}).items():
        store.observe_target(cid, np.asarray(state, dtype=float), t)
    for cell in fresh_cells:
        store.observe_coverage(cell, now)
    return store.snapshot(
        now=now, dt=0.1, self_state=np.array([*me, 0.0, 0.0, 0.0]),
        coverage_centers=sim.coverage_tracker.centers if n_regions else None,
        teammates=MATES3 if mates is None else mates,
    )


def _b(ctx, view, task_id, **kw):
    sim, tc, bc = ctx
    return bid(view, tasks_by_id(generate_tasks(view, tc))[task_id], bc, **kw)


# --- what a bid may read ----------------------------------------------------

def test_a_bid_takes_no_input_that_could_carry_ground_truth():
    """The signature is the audit. A bid sees one agent's view, one task, static
    constants, the agent's own current task and bounded modifiers -- no
    Simulation, no red state, no other agent's assignment."""
    params = list(inspect.signature(bid).parameters)
    assert params == ["view", "task", "ctx", "current_task_id", "modifiers"]
    fields = {f.name for f in dataclasses.fields(BidContext)}
    assert fields == {"arena_size", "neutralize_points", "red_v_max", "defense_polygon", "v2"}
    # v2 carries STATIC geometry only: the approach-prior
    # lattice, station positions and posture headings, the horizon and speed.
    from coordination.bidding import V2Context

    # v2.2 adds static fields only: the version, the approach
    # funnel built from the polygon and the PUBLIC gates, both teams' per-axis
    # bounds and the contact radius.
    assert {f.name for f in dataclasses.fields(V2Context)} == {
        "field", "stations", "horizon_steps", "dt", "blue_speed",
        "version", "funnel", "blue_v_max", "blue_a_max", "red_v_max", "contact_radius"}
    params_v2 = list(inspect.signature(bidding_mod.bid_terms_v2).parameters)
    assert params_v2 == ["view", "tasks", "ctx", "current_task_id", "modifiers"]


def test_the_bid_reflects_the_belief_not_the_truth(ctx):
    """L1/L2-shaped. Two views that differ ONLY in where the agent believes the
    contact is must bid differently -- so the bid is reading the belief -- and
    nothing else about the world is available to it."""
    sim = ctx[0]
    near = _view(sim, contacts={0: ([72, 72, 0, 0, 0], 38)})
    far = _view(sim, contacts={0: ([10, 10, 0, 0, 0], 38)})
    assert _b(ctx, near, "NEUTRALIZE:0:0") > _b(ctx, far, "NEUTRALIZE:0:0")


def test_the_bid_values_where_the_contact_is_believed_to_be_now(ctx):
    """Added after a mutation survived. The test above uses stationary
    contacts, where the raw sighting and the extrapolated belief coincide -- so
    a bid computing urgency from the stale sighting passed everything.

    A contact seen 30 steps ago at (40, 40) moving (10, 10) per second is
    believed to be at (70, 70) now. Its bid must equal that of a stationary
    contact at (70, 70) with the same age -- which pins the bid to the
    prediction exactly, the same point the solver steers to."""
    sim = ctx[0]
    moving = _view(sim, now=40, contacts={0: ([40, 40, 10, 10, 0], 10)})
    parked = _view(sim, now=40, contacts={0: ([70, 70, 0, 0, 0], 10)})
    for task_id in ("NEUTRALIZE:0:0", "SCREEN:0"):
        assert _b(ctx, moving, task_id) == pytest.approx(_b(ctx, parked, task_id))


def test_bids_are_pure(ctx):
    sim, tc, bc = ctx
    args = dict(contacts={0: ([60, 70, 0.5, 0.5, 0], 30)}, n_regions=16)
    v1, v2 = _view(sim, **args), _view(sim, **args)
    assert bids(v1, generate_tasks(v1, tc), bc) == bids(v2, generate_tasks(v2, tc), bc)


# --- how each term moves ---------------------------------------------------

def test_a_contact_nearer_the_perimeter_is_worth_more(ctx):
    sim = ctx[0]
    # Same distance from the agent, different distance from the defended region.
    toward = _view(sim, me=(60, 60), contacts={0: ([70, 70, 0, 0, 0], 38)})
    away = _view(sim, me=(60, 60), contacts={0: ([50, 50, 0, 0, 0], 38)})
    assert _b(ctx, toward, "NEUTRALIZE:0:0") > _b(ctx, away, "NEUTRALIZE:0:0")


def test_a_nearer_agent_bids_higher_for_the_same_task(ctx):
    sim = ctx[0]
    c = {0: ([40, 40, 0, 0, 0], 38)}
    assert _b(ctx, _view(sim, me=(45, 45), contacts=c), "NEUTRALIZE:0:0") > \
        _b(ctx, _view(sim, me=(90, 10), contacts=c), "NEUTRALIZE:0:0")


def test_an_older_belief_is_worth_less(ctx):
    """Same believed position now (stationary contact), different sighting age."""
    sim = ctx[0]
    fresh = _view(sim, now=100, contacts={0: ([60, 60, 0, 0, 0], 98)})
    stale = _view(sim, now=100, contacts={0: ([60, 60, 0, 0, 0], 10)})
    assert _b(ctx, fresh, "NEUTRALIZE:0:0") > _b(ctx, stale, "NEUTRALIZE:0:0")


def test_attrition_cost_grows_as_the_team_shrinks(ctx):
    """A3. The last defender should be the most reluctant to trade itself.
    Applies to NEUTRALIZE only: screening costs no one."""
    sim = ctx[0]
    c = {0: ([60, 60, 0, 0, 0], 38)}
    full = _view(sim, contacts=c, mates=MATES3)
    alone = _view(sim, contacts=c, mates={})
    assert _b(ctx, full, "NEUTRALIZE:0:0") > _b(ctx, alone, "NEUTRALIZE:0:0")
    assert _b(ctx, full, "SCREEN:0") == pytest.approx(_b(ctx, alone, "SCREEN:0"))


def test_switching_away_from_the_current_task_costs_something(ctx):
    sim = ctx[0]
    v = _view(sim, contacts={0: ([60, 60, 0, 0, 0], 38)})
    stay = _b(ctx, v, "NEUTRALIZE:0:0", current_task_id="NEUTRALIZE:0:0")
    leave = _b(ctx, v, "NEUTRALIZE:0:0", current_task_id="RECON:3")
    leave_post = _b(ctx, v, "NEUTRALIZE:0:0", current_task_id="SCREEN:0")
    assert stay > leave > leave_post, "leaving a defensive post costs more than leaving a patrol"


def test_staler_cells_are_worth_more_to_recon(ctx):
    sim = ctx[0]
    swept = _view(sim, n_regions=16, fresh_cells=(5,))
    never = _view(sim, n_regions=16)
    assert _b(ctx, never, "RECON:5") > _b(ctx, swept, "RECON:5")


# --- constraint 1: no deconfliction term -----------------------------------

def test_there_is_no_penalty_for_two_agents_on_one_contact(ctx):
    """Constraint 1 from the allocation diagnostic. Both pursuit slots on one
    contact are worth exactly the same to one agent -- and a bid has no input
    through which another agent's choice could reach it (see the signature
    test above)."""
    sim = ctx[0]
    tc2 = TaskContext.from_simulation(sim, slots=2)
    v = _view(sim, contacts={0: ([60, 60, 0, 0, 0], 38)})
    by = tasks_by_id(generate_tasks(v, tc2))
    assert bid(v, by["NEUTRALIZE:0:0"], ctx[2]) == bid(v, by["NEUTRALIZE:0:1"], ctx[2])


def test_screening_is_valued_comparably_to_pursuit_not_as_a_leftover(ctx):
    """Constraint 2: SCREEN must be a first-class task. For a threatening
    contact it outranks every station and every recon cell."""
    sim, tc, bc = ctx
    v = _view(sim, me=(62, 62), contacts={0: ([70, 70, 0, 0, 0], 38)}, n_regions=16)
    b = bids(v, generate_tasks(v, tc), bc)
    others = [x for k, x in b.items() if k.startswith(("DEFEND", "RECON"))]
    assert b["SCREEN:0"] > max(others)


# --- qualitative validity, never outcomes ---------------

def test_a_close_threat_outranks_a_far_stale_cell(ctx):
    sim, tc, bc = ctx
    v = _view(sim, me=(60, 60), contacts={0: ([70, 70, 0, 0, 0], 38)}, n_regions=16)
    b = bids(v, generate_tasks(v, tc), bc)
    best = max(b, key=b.get)
    assert best.startswith(("NEUTRALIZE:0", "SCREEN:0"))


def test_with_nothing_believed_the_agent_still_has_a_posture(ctx):
    sim, tc, bc = ctx
    v = _view(sim, contacts={})
    b = bids(v, generate_tasks(v, tc), bc)
    assert b and all(k.startswith("DEFEND") for k in b)


# --- the LLM seam: bounded by construction ---------------------------------

def test_modifiers_accept_only_named_profiles_and_three_labels():
    with pytest.raises(ValueError, match="scoring_profile"):
        BidModifiers(scoring_profile="yolo")
    with pytest.raises(ValueError, match="threat priority"):
        BidModifiers.build(threat_priorities={0: "CRITICAL"})
    assert set(PRIORITY_MULTIPLIER) == {"LOW", "MEDIUM", "HIGH"}


def test_a_threat_priority_can_only_scale_urgency_by_a_fixed_step(ctx):
    sim = ctx[0]
    v = _view(sim, contacts={0: ([60, 60, 0, 0, 0], 38)})
    low = _b(ctx, v, "NEUTRALIZE:0:0", modifiers=BidModifiers.build(threat_priorities={0: "LOW"}))
    med = _b(ctx, v, "NEUTRALIZE:0:0", modifiers=BidModifiers.build(threat_priorities={0: "MEDIUM"}))
    high = _b(ctx, v, "NEUTRALIZE:0:0", modifiers=BidModifiers.build(threat_priorities={0: "HIGH"}))
    assert low < med < high
    assert med == pytest.approx(_b(ctx, v, "NEUTRALIZE:0:0")), "MEDIUM is the no-modifier default"


def test_a_preferred_task_gets_exactly_the_fixed_bonus(ctx):
    sim = ctx[0]
    v = _view(sim, n_regions=16)
    base = _b(ctx, v, "RECON:3")
    pref = _b(ctx, v, "RECON:3", modifiers=BidModifiers.build(preferred_tasks=["RECON:3"]))
    assert pref - base == pytest.approx(PREFERRED_BONUS)


def test_profiles_move_the_ranking_in_their_declared_direction(ctx):
    sim = ctx[0]
    v = _view(sim, contacts={0: ([60, 60, 0, 0, 0], 38)}, n_regions=16)

    def gap(profile):  # pursuit minus screening, same contact
        m = BidModifiers(scoring_profile=profile)
        return _b(ctx, v, "NEUTRALIZE:0:0", modifiers=m) - _b(ctx, v, "SCREEN:0", modifiers=m)

    assert gap("aggressive") > gap("balanced") > gap("conservative")
    rb = _b(ctx, v, "RECON:5", modifiers=BidModifiers(scoring_profile="balanced"))
    rs = _b(ctx, v, "RECON:5", modifiers=BidModifiers(scoring_profile="scouting"))
    assert rs > rb


def test_profiles_are_the_declared_four():
    """Adding a profile after seeing outcomes is a researcher degree of freedom
    and must be a visible diff to this test, not a quiet addition."""
    assert sorted(PROFILES) == ["aggressive", "balanced", "conservative", "scouting"]


def test_urgency_is_bounded_and_monotone(ctx):
    bc = ctx[2]
    values = [urgency((x, x), bc) for x in (5, 20, 40, 60, 70)]
    assert all(0 < u <= 1 for u in values)
    assert values == sorted(values), "closer to the defended region must be more urgent"
