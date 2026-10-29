"""command-delivery conditions A/B/C/D — the contract

The defect these exist to prevent. Before the command-delivery conditions the centralized planner branch
built its `PlannerObservation` from `x_current`, `self.red_state` and this
step's `detected_red_indices` -- **ground-truth positions, no memory, no
staleness** -- and never read `belief_stores`. `belief_mode` did not reach it at
all. So a 2x2 built by toggling `per_agent_planning` would move the information
axis and the decision axis together, and report the sum as a decision effect.

Condition A is therefore a thing that had to be *built*, not configured, and
these tests are what say it was built correctly.
"""

import numpy as np
import pytest

from conftest import build_simulation, capture_fd_output, fingerprint, run_quiet
from planning.roles import BlueRole


def _run(**kwargs):
    kwargs.setdefault("num_blues", 4)
    kwargs.setdefault("num_reds", 2)
    kwargs.setdefault("sim_steps", 200)
    return run_quiet(build_simulation(**kwargs))


def _epoch_assignments(sim):
    interval = sim.planning_interval_steps
    return {
        s["step"]: (dict(s["blue_roles"]), dict(s["target_red_ids"]))
        for s in sim.recorder.steps
        if s["step"] % interval == 0
    }


# ---------------------------------------------------------------------------
# The planner branch (first, because everything else is worthless if it fails)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["ground_truth", "shared", "local", "networked"])
@pytest.mark.parametrize("per_agent", [False, True])
def test_m6_options_default_off_leaves_every_mode_byte_identical(mode, per_agent):
    """New subsystems land default-off. With `centralized_from_belief` unset,
    the centralized branch must still take the pre-the command-delivery conditions ground-truth path."""
    a = fingerprint(_run(seed=7, sim_steps=120, belief_mode=mode,
                         per_agent_planning=per_agent))
    b = fingerprint(_run(seed=7, sim_steps=120, belief_mode=mode,
                         per_agent_planning=per_agent,
                         centralized_from_belief=False, commander_id=0,
                         commander_failure_stops_planning=False))
    assert a == b


# ---------------------------------------------------------------------------
# A and C plan identically under shared belief
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [3101, 3102, 3103])
def test_condition_A_and_C_produce_identical_assignments(seed):
    """The row is defined by its information.

    Under `shared` every active store holds the same merged contents, so the
    commander's pooled snapshot (A) and each agent's own snapshot (C) are the
    same belief. `plan_team` and `plan_one` share a body, so A must reproduce
    what C assembles -- every epoch, every agent.

    This is the M3 equivalence gate applied to the new A path. If it fails, a
    difference measured between rows is a decomposition artifact, not
    information.
    """
    a = _run(seed=seed, belief_mode="shared", per_agent_planning=False,
             centralized_from_belief=True)
    c = _run(seed=seed, belief_mode="shared", per_agent_planning=True)

    assert _epoch_assignments(a) == _epoch_assignments(c)
    # And therefore the whole episode, not only the decisions.
    assert fingerprint(a) == fingerprint(c)


# ---------------------------------------------------------------------------
# A plans from belief, not from ground truth
# ---------------------------------------------------------------------------


def test_condition_A_assigns_targets_nobody_currently_detects():
    """Memory, asserted directly rather than as a fingerprint inequality.

    The old centralized path was memoryless by construction:
    `detected_red_indices` is *this step's* FOV union, so a contact that left
    every cone vanished from the observation and could not be assigned. Under A
    it persists in the belief store, keeps ageing, and remains assignable.

    So the positive statement is the test: A must, at some epoch, assign a
    NEUTRALIZE target that **no blue is detecting at that step**. The legacy
    path structurally cannot do that.

    An earlier version of this test asserted instead that A and the legacy path
    produce different episodes. That premise was wrong: under `shared` with good
    detection, belief and ground truth largely coincide, and on seed 3101 the
    two paths produced identical assignments. Equality there is correct
    behaviour, not a missing feature.

    The full 400-step horizon is required, not an abbreviation. Reds spawn at
    standoff and are not detected at all until roughly step 150 on these seeds,
    so a 200-step episode contains no interval over which anything could be
    remembered -- the earlier draft failed for exactly that reason.
    """
    sim = _run(seed=3101, sim_steps=400, belief_mode="shared",
               per_agent_planning=False, centralized_from_belief=True)
    interval = sim.planning_interval_steps
    held_undetected, remembered = 0, 0
    for record in sim.recorder.steps:
        if record["step"] % interval:
            continue
        detected_now = set(record["detected_red_union"])
        believed = {int(c) for c in record["beliefs"].get(str(sim.commander_id), {})}
        if believed - detected_now:
            held_undetected += 1
        for agent, target in record["target_red_ids"].items():
            if target is None:
                continue
            if record["blue_roles"].get(agent) != BlueRole.NEUTRALIZE.value:
                continue
            if target not in detected_now:
                remembered += 1

    # Asserted separately so a future failure says WHICH thing broke: no
    # opportunity to remember, or memory not being used.
    assert held_undetected > 0, (
        "the commander never believed in a contact nobody was detecting, so "
        "this episode cannot distinguish belief from ground truth"
    )
    assert remembered > 0, (
        "condition A never assigned a target that was undetected at that step, "
        "so nothing distinguishes it from the memoryless ground-truth path"
    )


def test_condition_A_never_sees_a_contact_no_blue_has_observed():
    """The other half of 'plans from belief': no ground-truth leak.

    Under `local`-style isolation the commander can only assign a target it
    believes in. A target chosen at an epoch must appear in the commander's
    own belief at that epoch.
    """
    sim = _run(seed=3104, belief_mode="networked", r_comm=0.0,
               per_agent_planning=False, centralized_from_belief=True)
    interval = sim.planning_interval_steps
    checked = 0
    for record in sim.recorder.steps:
        if record["step"] % interval:
            continue
        believed = set(record["beliefs"].get(str(sim.commander_id), {}))
        for agent, target in record["target_red_ids"].items():
            if target is None:
                continue
            if record["blue_roles"].get(agent) != BlueRole.NEUTRALIZE.value:
                continue
            checked += 1
            assert str(target) in believed, (
                f"step {record['step']}: commander assigned red {target}, which "
                f"it does not believe in ({sorted(believed)})"
            )
    # With r_comm=0 the commander only ever sees what it personally detects, so
    # there must be *some* assignment to have checked.
    assert checked > 0, "no NEUTRALIZE assignment was made; the test is vacuous"


# ---------------------------------------------------------------------------
# B reads the commander's store and no other
# ---------------------------------------------------------------------------


def test_condition_B_plans_from_the_commanders_store_only():
    """Changing which blue is commander must change the episode under
    `networked`, where the stores genuinely differ.

    If commander identity were ignored, B would be reading something other than
    `belief_stores[commander_id]`.
    """
    cfg = dict(seed=3105, belief_mode="networked", r_comm=15.0,
               comm_interval_seconds=1.0, per_agent_planning=False,
               centralized_from_belief=True)
    first = _run(commander_id=0, **cfg)
    second = _run(commander_id=2, **cfg)
    assert fingerprint(first) != fingerprint(second), (
        "commander identity did not affect the episode under networked belief"
    )


def test_commander_identity_is_irrelevant_under_shared_belief():
    """The complement, and the reason A is well defined.

    Under `shared` every active store is identical, so *any* commander gives
    the pooled belief. If this fails, `shared` is not merging what it claims.
    """
    cfg = dict(seed=3105, belief_mode="shared", per_agent_planning=False,
               centralized_from_belief=True)
    assert fingerprint(_run(commander_id=0, **cfg)) == fingerprint(
        _run(commander_id=3, **cfg)
    )


# ---------------------------------------------------------------------------
# Commander loss, identically in A and B
# ---------------------------------------------------------------------------


def test_commander_loss_freezes_commitments_and_leaves_safety_running():
    """Driven deterministically by disabling the commander mid-episode rather
    than waiting for a seed where it happens -- a test that skips when nobody
    dies is a test that does not test.
    """
    sim = build_simulation(num_blues=4, num_reds=2, seed=3106, sim_steps=200,
                           belief_mode="shared", per_agent_planning=False,
                           centralized_from_belief=True,
                           commander_failure_stops_planning=True)
    original = sim._plan_centralized_from_belief
    kill_at = 60

    def planning_with_a_scheduled_death(step):
        if step >= kill_at:
            sim.blue_disabled[sim.commander_id] = True
        return original(step)

    sim._plan_centralized_from_belief = planning_with_a_scheduled_death
    run_quiet(sim)

    assert sim.commander_lost_step is not None
    assert sim.commander_lost_step >= kill_at
    assert sim.episode_outcome()["commander_lost"] is True

    interval = sim.planning_interval_steps
    after = [
        (dict(s["blue_roles"]), dict(s["target_red_ids"]))
        for s in sim.recorder.steps
        if s["step"] % interval == 0 and s["step"] > sim.commander_lost_step
    ]
    assert after, "the episode ended at the moment the commander died"
    assert all(a == after[0] for a in after), (
        "assignments changed after the commander was lost; commitments are not "
        "frozen"
    )

    # The safety supervisor is never gated by planning.
    assert any(s["cbf_active"] for s in sim.recorder.steps)


def test_commander_failure_rule_is_available_to_A_and_B_alike():
    """B - A must not confound communication with planner loss, so the failure
    rule is a property of the *centralized* path, not of the networked one."""
    for mode, extra in (("shared", {}), ("networked", {"r_comm": 15.0})):
        sim = _run(seed=3107, belief_mode=mode, per_agent_planning=False,
                   centralized_from_belief=True,
                   commander_failure_stops_planning=True, **extra)
        assert "commander_lost" in sim.episode_outcome()


def test_without_the_failure_rule_planning_survives_the_commander():
    """The default. `commander_failure_stops_planning=False` falls back to the
    lowest-indexed active blue so the condition still produces assignments --
    used for configurations where commander mortality is not the question."""
    sim = build_simulation(num_blues=4, num_reds=2, seed=3106, sim_steps=120,
                           belief_mode="shared", per_agent_planning=False,
                           centralized_from_belief=True,
                           commander_failure_stops_planning=False)
    original = sim._plan_centralized_from_belief

    def planning_with_a_dead_commander(step):
        sim.blue_disabled[sim.commander_id] = True
        return original(step)

    sim._plan_centralized_from_belief = planning_with_a_dead_commander
    run_quiet(sim)
    assert sim.commander_lost_step is None, "the failure rule fired while off"
    assert sim.planner_output.roles, "no assignment survived the commander"


# ---------------------------------------------------------------------------
# Contract Sec 4 — "hold the commitment" is an ASSIGNMENT-level statement
#
# Required before condition B's results can be read. `decentralized_solver.py`
# gives a NEUTRALIZE agent with no `target_override` entry the nearest contact
# it BELIEVES in, and `target_override` is only set when the assigned red is
# still active. So once a frozen target goes inactive the recorded assignment
# stays frozen while the controller silently retargets -- and the recorded
# `target_red_ids` then describes a commitment nobody is flying.
#
# The behaviour is deliberately NOT changed (it is a sensible fallback, and
# moving the motion layer during an information experiment would confound it).
# It is made observable instead.
# ---------------------------------------------------------------------------


def _run_with_a_scheduled_commander_death(seed, kill_at, sim_steps=400):
    """Condition A, mortality on, the commander disabled at a chosen step.

    Driven deterministically rather than by waiting for a seed where the
    commander happens to die -- same reasoning as
    test_commander_loss_freezes_commitments_and_leaves_safety_running.
    """
    sim = build_simulation(num_blues=4, num_reds=2, seed=seed, sim_steps=sim_steps,
                           belief_mode="shared", per_agent_planning=False,
                           centralized_from_belief=True,
                           commander_failure_stops_planning=True)
    original = sim._plan_centralized_from_belief

    def planning_with_a_scheduled_death(step):
        if step >= kill_at:
            sim.blue_disabled[sim.commander_id] = True
        return original(step)

    sim._plan_centralized_from_belief = planning_with_a_scheduled_death
    return run_quiet(sim)


def _neutralize_rows(sim):
    """(step, agent, assigned_target, effective_entry) for every NEUTRALIZE
    agent carrying a target, at every step."""
    for record in sim.recorder.steps:
        for agent, assigned in record["target_red_ids"].items():
            if assigned is None:
                continue
            if record["blue_roles"].get(agent) != BlueRole.NEUTRALIZE.value:
                continue
            yield record["step"], agent, assigned, record["effective_targets"][agent]


def test_effective_target_agrees_with_the_assignment_while_the_override_holds():
    """The log must not invent divergence where there is none.

    Whenever the solver reports that it steered by the assignment, the contact
    it names has to be the one the planner recorded. If this can fail, a
    divergence count is measuring the instrument.
    """
    sim = _run_with_a_scheduled_commander_death(seed=3206, kill_at=160)
    checked = 0
    for step, agent, assigned, effective in _neutralize_rows(sim):
        if effective["source"] != "assigned":
            continue
        checked += 1
        assert effective["contact_id"] == assigned, (
            f"step {step}: blue {agent} recorded as steering by its assignment "
            f"({assigned}) but the log names contact {effective['contact_id']}"
        )
        assert effective["controller"] == BlueRole.NEUTRALIZE.value
    assert checked > 0, "no agent ever steered by its assignment; test is vacuous"


def test_a_frozen_assignment_is_distinguishable_from_actual_pursuit():
    """The positive statement, and the reason this log exists.

    After the commander is lost the assignment record freezes. Once a frozen
    target is resolved the override is dropped and the controller falls back to
    the agent's nearest believed contact -- which on this seed is a DIFFERENT
    contact from the one still on record. Without the effective-target log the
    two are indistinguishable in the recorder, and a commander-loss analysis
    would read "commitments held" as "the team kept chasing the same contacts".

    Seed 3206 with the commander disabled at step 160 is chosen because it
    exhibits the divergence, not to make it likely: the assertion is that the
    instrument RESOLVES the case, so the case has to be present.
    """
    sim = _run_with_a_scheduled_commander_death(seed=3206, kill_at=160)
    assert sim.commander_lost_step is not None

    frozen_but_retargeted = [
        (step, agent, assigned, effective["contact_id"])
        for step, agent, assigned, effective in _neutralize_rows(sim)
        if effective["source"] == "nearest_believed"
        and effective["contact_id"] != assigned
    ]
    assert frozen_but_retargeted, (
        "no step where a recorded assignment and the contact actually pursued "
        "disagreed, so this episode cannot demonstrate the distinction"
    )
    # Every one of them is after the commander was lost, which is the regime
    # the contract's Sec 4 warning is about.
    assert all(step > sim.commander_lost_step for step, *_ in frozen_but_retargeted)

    # And the assignment record itself did NOT move -- both halves of the
    # claim, asserted together, because either alone is misleading.
    interval = sim.planning_interval_steps
    after = [
        dict(s["target_red_ids"])
        for s in sim.recorder.steps
        if s["step"] % interval == 0 and s["step"] > sim.commander_lost_step
    ]
    assert after and all(a == after[0] for a in after)


def test_recon_is_never_logged_as_pursuing_a_contact():
    """RECON ignores `target_state` entirely -- its cost depends only on its own
    state and its waypoint. The solver still computes a nearest believed contact
    for it, and logging that as a target would manufacture pursuit that the
    motion layer never performs."""
    sim = _run_with_a_scheduled_commander_death(seed=3206, kill_at=160)
    seen = 0
    for record in sim.recorder.steps:
        for agent, role in record["blue_roles"].items():
            if role != BlueRole.RECON.value:
                continue
            seen += 1
            entry = record["effective_targets"][agent]
            assert entry == {
                "source": "recon_waypoint",
                "contact_id": None,
                "controller": BlueRole.RECON.value,
            }, f"step {record['step']}: blue {agent} RECON logged as {entry}"
    assert seen > 0, "no RECON agent in the episode; test is vacuous"


def test_an_unrealizable_neutralize_is_logged_as_holding_station():
    """The third branch: assigned to intercept while believing in no contact.

    The solver swaps in the DEFEND controller and sends the agent to its
    station. The log has to say DEFEND, because reporting the *assignment* there
    would describe an interception that never happened.

    Taken off a live Simulation rather than a hand-built solver, so what is
    exercised is the object the episode actually uses.
    """
    sim = build_simulation(num_blues=4, num_reds=2, seed=3206, sim_steps=1,
                           belief_mode="shared", per_agent_planning=False,
                           centralized_from_belief=True)
    sim.setup_agents()
    sim.solver.reset_warm_starts()
    sim.solver.set_fixed_adjacency_from_state(sim.x_current, sim.red_state)
    solver = sim.solver
    with capture_fd_output():
        solver.solve_decentralized(
            x_current=sim.x_current,
            red_states=sim.red_state,
            role_assignment={i: BlueRole.NEUTRALIZE for i in range(sim.num_blues)},
            target_override={},
            # Nobody believes in anything: the role is unrealizable for all four.
            believed_targets={i: {} for i in range(sim.num_blues)},
        )
    assert solver.effective_targets == {
        i: {"source": "station", "contact_id": None,
            "controller": BlueRole.DEFEND.value}
        for i in range(sim.num_blues)
    }
