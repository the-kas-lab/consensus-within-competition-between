"""M3: per-agent role planning, and the information channel it is allowed.

The milestone's own gate lives here. the decentralization rule settles that the rule planner
is invoked *per agent* on that agent's ``LocalWorldView``, with teammate state
as a declared common-knowledge channel, and commitment 5 makes one property a
precondition on reading any Experiment 0 number:

    under ``belief_mode="shared"``, per-agent planning must reproduce the
    centralized assignment exactly.

Until that holds, information and decision decomposition are confounded and a
difference between the arms means nothing. Tests accumulate here as M3's five
build steps land; step 1 (the ``teammates`` channel) is what is covered so far.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.belief import LocalBeliefStore  # noqa: E402


def _store(agent_id=0, n_regions=0):
    return LocalBeliefStore(agent_id, n_regions=n_regions)


def test_teammates_default_to_empty_not_none():
    """A view built without teammates is still usable -- an agent alone on the
    field ranks a team of one, it does not crash."""
    view = _store(0).snapshot(now=0, dt=0.1, self_state=np.zeros(5))
    assert view.teammates == {}
    assert list(view.active_blue_states()) == [0]


def test_view_carries_teammates_and_excludes_self():
    view = _store(1).snapshot(
        now=3,
        dt=0.1,
        self_state=np.array([1.0, 1.0, 0.0, 0.0, 0.0]),
        teammates={
            0: np.array([5.0, 5.0, 0.0, 0.0, 0.0]),
            1: np.array([9.9, 9.9, 0.0, 0.0, 0.0]),  # self, must be dropped
            2: np.array([7.0, 7.0, 0.0, 0.0, 0.0]),
        },
    )
    assert sorted(view.teammates) == [0, 2]
    np.testing.assert_allclose(view.self_state[0:2], [1.0, 1.0])


def test_active_blue_states_includes_self_and_is_ordered():
    """Ordering is not cosmetic: the per-agent ranking is built from this, and
    an unstable order would make the equivalence gate flaky rather than false."""
    view = _store(2).snapshot(
        now=0,
        dt=0.1,
        self_state=np.array([2.0, 2.0, 0.0, 0.0, 0.0]),
        teammates={
            3: np.array([3.0, 3.0, 0.0, 0.0, 0.0]),
            0: np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            1: np.array([1.0, 1.0, 0.0, 0.0, 0.0]),
        },
    )
    states = view.active_blue_states()
    assert list(states) == [0, 1, 2, 3]
    for i, s in states.items():
        assert s[0] == float(i), "each agent's own state must land under its own id"


def test_teammate_states_are_frozen_at_snapshot_time():
    """'s frozen-epoch requirement applies to the teammate channel too.

    If the view aliased the caller's arrays, agent 1 planning after agent 0 in
    the same epoch would see agent 0's *post-step* position -- the exact
    within-epoch leak the snapshot exists to prevent.
    """
    live = np.array([5.0, 5.0, 0.0, 0.0, 0.0])
    view = _store(0).snapshot(
        now=0, dt=0.1, self_state=np.zeros(5), teammates={1: live}
    )
    live[0] = 999.0
    assert view.teammates[1][0] == 5.0


def test_teammates_are_not_stored_as_belief():
    """The channel is declared common knowledge, not something the store holds.

    Guards against a later refactor quietly turning teammate state into a
    belief record -- which would be M4's job, and would silently change what
    Experiment 0 brackets.
    """
    store = _store(0)
    store.snapshot(
        now=0, dt=0.1, self_state=np.zeros(5), teammates={1: np.ones(5)}
    )
    assert store.targets == {}
    assert not hasattr(store, "teammates")


# ---------------------------------------------------------------------------
# Step 2: PlannerOutput -- roles and targets leave the planner in one object.
# ---------------------------------------------------------------------------


def _obs(n_blues=4, n_reds=2, detected=None, coverage=None):
    from planning.planner import PlannerObservation

    blues = np.array(
        [[10.0 * i, 10.0 * i, 0.0, 0.0, 0.0] for i in range(n_blues)], float
    ).ravel()
    reds = np.array([[15.0 + 40.0 * j, 15.0, 0.0, 0.0] for j in range(n_reds)], float).ravel()
    return PlannerObservation(
        x_current=blues,
        red_states=reds,
        num_blues=n_blues,
        num_reds=n_reds,
        sim_step=0,
        arena_size=100.0,
        detected_red_indices=list(range(n_reds)) if detected is None else detected,
        coverage_regions=coverage,
    )


def _coverage(n=4):
    return [
        {"region_id": k, "steps_since_seen": -1 if k == 0 else 10 * k}
        for k in range(n)
    ]


def test_plan_full_matches_plan_plus_the_side_channel_for_the_rule_planner():
    """The shim must be a pure reading of what plan() already decided.

    Compared against a *separately constructed* planner running plan(), so a
    plan_full() that mutated state or re-ran the ranking would show up here.
    """
    from planning.planner import RuleBasedBlueRolePlanner

    obs = _obs(coverage=_coverage())
    kw = dict(neutralize_fraction=0.5, min_neutralizers=1, recon_fraction=0.5, min_recon=1)

    reference = RuleBasedBlueRolePlanner(**kw)
    roles = reference.plan(obs)

    out = RuleBasedBlueRolePlanner(**kw).plan_full(obs)

    assert out.roles == roles
    assert out.target_red_ids == reference.last_target_red_ids
    assert out.recon_target_regions == reference.last_recon_target_regions
    assert out.target_contact_ids == {}, "Exp I planner must not populate contact ids"


def test_plan_full_reads_the_contact_namespace_for_a_vision_planner():
    """Exp II planners populate last_targets, not last_target_red_ids. The two
    namespaces must not be crossed -- a contact_id is not a red index."""
    from planning.planner import RuleBasedVisionRolePlanner

    obs = _obs(coverage=_coverage())
    obs.detected_contact_ids = [0, 1]
    obs.contact_states = {
        0: np.array([15.0, 15.0, 0.0, 0.0]),
        1: np.array([55.0, 15.0, 0.0, 0.0]),
    }

    reference = RuleBasedVisionRolePlanner(recon_fraction=0.5, min_recon=1)
    roles = reference.plan(obs)
    out = RuleBasedVisionRolePlanner(recon_fraction=0.5, min_recon=1).plan_full(obs)

    assert out.roles == roles
    assert out.target_contact_ids == reference.last_targets
    assert out.recon_target_regions == reference.last_recon_target_regions
    assert out.target_red_ids == {}, "Exp II planner must not populate red indices"


def test_plan_full_returns_empty_targets_for_a_planner_with_no_side_channel():
    """A planner that sets no last_* attributes at all -- which is every planner
    a collaborator might write against the documented BlueRolePlanner
    interface -- must still work, and yield empty targets rather than raising."""
    from planning.planner import BlueRolePlanner
    from planning.roles import BlueRole

    class Minimal(BlueRolePlanner):
        def plan(self, obs):
            return {i: BlueRole.DEFEND for i in range(obs.num_blues)}

    out = Minimal().plan_full(_obs())
    assert out.roles == {i: BlueRole.DEFEND for i in range(4)}
    assert out.target_red_ids == {}
    assert out.target_contact_ids == {}
    assert out.recon_target_regions == {}


def test_plan_full_snapshots_the_side_channel_rather_than_aliasing_it():
    """The output is one epoch's decision. If it aliased the planner's live
    dicts, a later plan() would retroactively rewrite an earlier epoch's
    recorded targets -- and the recorder reads this object every step."""
    from planning.planner import RuleBasedBlueRolePlanner

    planner = RuleBasedBlueRolePlanner(neutralize_fraction=0.5, min_neutralizers=1)
    first = planner.plan_full(_obs(n_reds=2))
    before = dict(first.target_red_ids)

    planner.plan_full(_obs(n_reds=2, detected=[]))  # nothing detected: clears targets
    assert first.target_red_ids == before


def test_llm_planners_inherit_the_shim_without_overriding_plan():
    """The LLM planners are untouched by this refactor by design -- they reach
    PlannerOutput through the inherited shim. If someone gives one its own
    plan_full() later, that is a deliberate act, not an accident of MRO."""
    from planning.planner import (
        BlueRolePlanner,
        LLMRolePlanner,
        LLMVisionRolePlanner,
        RuleBasedBlueRolePlanner,
        RuleBasedVisionRolePlanner,
    )

    for cls in (
        LLMRolePlanner,
        LLMVisionRolePlanner,
        RuleBasedBlueRolePlanner,
        RuleBasedVisionRolePlanner,
    ):
        assert issubclass(cls, BlueRolePlanner)
        assert "plan_full" not in cls.__dict__, f"{cls.__name__} overrides the shim"
        assert cls.plan_full is BlueRolePlanner.plan_full


def test_duck_typed_wrapper_planner_still_works():
    """scripts/ passes plain wrapper classes that implement plan() and nothing
    else -- ReasoningLoggingPlanner in run_vision_experiment.py and
    run_red_count_grid.py. Requiring a BlueRolePlanner subclass at the
    Simulation boundary would break those shipped scripts.
    """
    from planning.planner import RuleBasedVisionRolePlanner, plan_full_from

    class Wrapper:  # deliberately not a BlueRolePlanner
        def __init__(self, inner):
            self.inner = inner
            self.last_targets = {}

        def plan(self, obs):
            roles = self.inner.plan(obs)
            self.last_targets = getattr(self.inner, "last_targets", {})
            return roles

    obs = _obs(coverage=_coverage())
    obs.detected_contact_ids = [0, 1]
    obs.contact_states = {
        0: np.array([15.0, 15.0, 0.0, 0.0]),
        1: np.array([55.0, 15.0, 0.0, 0.0]),
    }

    wrapper = Wrapper(RuleBasedVisionRolePlanner(recon_fraction=0.5, min_recon=1))
    out = plan_full_from(wrapper, obs)

    assert out.roles
    assert out.target_contact_ids == wrapper.last_targets


def test_wrapper_scrape_does_not_reach_through_to_the_inner_planner():
    """Behavior preservation, precisely.

    ReasoningLoggingPlanner re-exposes last_targets but NOT
    last_recon_target_regions, so its recon assignments were already being
    dropped under the old getattr-at-a-distance convention. Reading through to
    the inner planner would look like a bug fix and would silently change what a
    shipped experiment does.
    """
    from planning.planner import RuleBasedVisionRolePlanner, plan_full_from

    class Wrapper:
        def __init__(self, inner):
            self.inner = inner
            self.last_targets = {}

        def plan(self, obs):
            roles = self.inner.plan(obs)
            self.last_targets = getattr(self.inner, "last_targets", {})
            return roles

    obs = _obs(coverage=_coverage())
    obs.detected_contact_ids = [0, 1]
    obs.contact_states = {
        0: np.array([15.0, 15.0, 0.0, 0.0]),
        1: np.array([55.0, 15.0, 0.0, 0.0]),
    }

    inner = RuleBasedVisionRolePlanner(recon_fraction=0.5, min_recon=1)
    wrapper = Wrapper(inner)
    out = plan_full_from(wrapper, obs)

    assert inner.last_recon_target_regions, "inner must have chosen recon regions"
    assert out.recon_target_regions == {}, (
        "the wrapper does not re-expose them, and neither must we"
    )


def test_simulation_runs_with_a_duck_typed_wrapper_planner():
    """End-to-end: the Simulation boundary itself, not just plan_full_from.

    This is the path scripts/run_vision_experiment.py and
    scripts/run_red_count_grid.py actually take, and it is the one that would
    have raised AttributeError had the planning block called
    ``self.role_planner.plan_full(obs)`` directly.
    """
    from conftest import build_simulation, fingerprint, run_quiet
    from planning.planner import RuleBasedBlueRolePlanner

    class Wrapper:  # mirrors scripts/'s ReasoningLoggingPlanner
        def __init__(self, inner):
            self.inner = inner
            self.log = []
            self.last_target_red_ids = {}
            self.last_recon_target_regions = {}

        def plan(self, obs):
            roles = self.inner.plan(obs)
            self.last_target_red_ids = getattr(self.inner, "last_target_red_ids", {})
            self.last_recon_target_regions = getattr(
                self.inner, "last_recon_target_regions", {}
            )
            self.log.append((obs.sim_step, dict(roles)))
            return roles

    wrapped = run_quiet(
        build_simulation(
            role_planner=Wrapper(
                RuleBasedBlueRolePlanner(recon_fraction=0.5, min_recon=1)
            )
        )
    )
    plain = run_quiet(build_simulation())

    assert wrapped.role_planner.log, "the wrapper must actually have been called"
    assert fingerprint(wrapped) == fingerprint(plain), (
        "a transparent wrapper must not change the episode"
    )


# ---------------------------------------------------------------------------
# Step 3: plan_one(view) -- the per-agent decision, and the equivalence gate.
# ---------------------------------------------------------------------------

BLUES_4 = {
    0: np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    1: np.array([10.0, 10.0, 0.0, 0.0, 0.0]),
    2: np.array([60.0, 60.0, 0.0, 0.0, 0.0]),
    3: np.array([70.0, 70.0, 0.0, 0.0, 0.0]),
}
CONTACTS_2 = {
    0: np.array([15.0, 15.0, 0.0, 0.0]),
    1: np.array([65.0, 65.0, 0.0, 0.0]),
}


def _views(blue_states, per_agent_contacts, now=0, dt=0.1, observed_at=0):
    """One LocalWorldView per blue.

    ``per_agent_contacts`` is {agent_id: {contact_id: state}} -- pass every
    agent the same dict to model the shared arm, different ones for the local
    arm.
    """
    views = {}
    for agent_id, self_state in blue_states.items():
        store = LocalBeliefStore(agent_id)
        for cid, state in per_agent_contacts.get(agent_id, {}).items():
            store.observe_target(cid, state, observed_at)
        views[agent_id] = store.snapshot(
            now=now,
            dt=dt,
            self_state=self_state,
            teammates={i: s for i, s in blue_states.items() if i != agent_id},
        )
    return views


def _assemble(planner, views, max_age=None):
    """Run plan_one per agent and merge the slots into a team assignment."""
    from planning.planner import PlannerOutput

    merged = PlannerOutput()
    for agent_id in sorted(views):
        out = planner.plan_one(views[agent_id], max_age=max_age)
        assert set(out.roles) <= {agent_id}, "an agent may only assign itself"
        merged.roles.update(out.roles)
        merged.target_red_ids.update(out.target_red_ids)
        merged.recon_target_regions.update(out.recon_target_regions)
    return merged


def _centralized_on_the_same_belief(planner, view, blue_states, n_reds):
    """The team planner, fed the belief instead of ground truth.

    This is the right reference for the gate. Comparing against the team
    planner on *ground truth* would conflate the decomposition with M2's
    belief change, which is a different question and already tested elsewhere.
    """
    from planning.planner import PlannerObservation, plan_full_from

    believed = view.fresh_targets(None)
    red_states = np.zeros((n_reds, 4), dtype=float)
    for cid, belief in believed.items():
        red_states[cid] = np.asarray(
            belief.predicted_state(view.now, view.dt), dtype=float
        ).ravel()[0:4]

    obs = PlannerObservation(
        x_current=np.concatenate([blue_states[i] for i in sorted(blue_states)]),
        red_states=red_states.ravel(),
        num_blues=len(blue_states),
        num_reds=n_reds,
        sim_step=view.now,
        arena_size=100.0,
        detected_red_indices=sorted(believed),
        coverage_regions=view.coverage_regions(),
    )
    return plan_full_from(planner, obs)


def test_shared_beliefs_reproduce_the_centralized_assignment():
    """**The M3 gate**.

    Under shared beliefs every agent holds the same contacts, so every agent
    computes the same team ranking and keeps its own slot. The assembled
    assignment must equal what the same planner produces centrally on that
    belief -- roles, intercept targets and recon regions alike.

    Until this passes, no Experiment 0 number may be read: a difference between
    the arms would be unattributable, because information and decision
    decomposition would be confounded.
    """
    from planning.planner import RuleBasedBlueRolePlanner

    planner = RuleBasedBlueRolePlanner(neutralize_fraction=0.5, min_neutralizers=1)
    shared = {i: CONTACTS_2 for i in BLUES_4}
    views = _views(BLUES_4, shared)

    assembled = _assemble(planner, views)
    central = _centralized_on_the_same_belief(
        planner, views[0], BLUES_4, n_reds=len(CONTACTS_2)
    )

    assert assembled.roles == central.roles
    assert assembled.target_red_ids == central.target_red_ids
    assert assembled.recon_target_regions == central.recon_target_regions


def test_the_gate_holds_when_the_shared_belief_is_stale():
    """Equivalence must not depend on beliefs being fresh.

    A shared belief that is 30 steps old is extrapolated to a *predicted*
    position, which is not where the contact actually is. The decomposition has
    to be faithful to the belief it is given, whatever its age -- otherwise the
    gate would silently only hold in the regime where it does not matter.
    """
    from planning.planner import RuleBasedBlueRolePlanner

    planner = RuleBasedBlueRolePlanner(neutralize_fraction=0.5, min_neutralizers=1)
    moving = {
        0: np.array([15.0, 15.0, 2.0, 1.0]),
        1: np.array([65.0, 65.0, -1.0, 3.0]),
    }
    views = _views(BLUES_4, {i: moving for i in BLUES_4}, now=30, observed_at=0)

    assert all(b.age(30) == 30 for b in views[0].targets.values())

    assembled = _assemble(planner, views)
    central = _centralized_on_the_same_belief(planner, views[0], BLUES_4, n_reds=2)

    assert assembled.roles == central.roles
    assert assembled.target_red_ids == central.target_red_ids


def test_per_agent_planning_does_not_collapse_to_all_neutralize():
    """Regression guard for the decentralization rule defect A.

    Handing the rule planner a view of one agent makes n_intercept
    unconditionally 1, collapsing it to "NEUTRALIZE iff I believe in any
    contact" -- 4 blues / 2 contacts goes from two interceptors to four. Since
    NEUTRALIZE is mutual destruction and the shared arm gives more agents a
    non-empty belief, that would invert the bracket outright.
    """
    from planning.planner import RuleBasedBlueRolePlanner
    from planning.roles import BlueRole

    planner = RuleBasedBlueRolePlanner(neutralize_fraction=0.5, min_neutralizers=1)
    views = _views(BLUES_4, {i: CONTACTS_2 for i in BLUES_4})
    assembled = _assemble(planner, views)

    n_neutralize = sum(1 for r in assembled.roles.values() if r == BlueRole.NEUTRALIZE)
    assert n_neutralize == 2, f"expected the top half to intercept, got {assembled.roles}"


def test_local_beliefs_diverge_from_the_centralized_assignment():
    """The paired control, so the gate cannot pass vacuously.

    If per-agent planning could never differ from the centralized assignment,
    the gate would be asserting a tautology and Experiment 0 would have nothing
    to measure. Here each agent believes in a *different* contact, which is
    exactly the belief disagreement the local arm is built to create.
    """
    from planning.planner import RuleBasedBlueRolePlanner

    planner = RuleBasedBlueRolePlanner(neutralize_fraction=0.5, min_neutralizers=1)
    split = {
        0: {0: CONTACTS_2[0]},
        1: {0: CONTACTS_2[0]},
        2: {1: CONTACTS_2[1]},
        3: {1: CONTACTS_2[1]},
    }
    views = _views(BLUES_4, split)
    assembled = _assemble(planner, views)

    shared_views = _views(BLUES_4, {i: CONTACTS_2 for i in BLUES_4})
    central = _centralized_on_the_same_belief(
        planner, shared_views[0], BLUES_4, n_reds=2
    )

    assert (assembled.roles, assembled.target_red_ids) != (
        central.roles,
        central.target_red_ids,
    ), "divergent beliefs must be able to produce a different assignment"


def test_local_beliefs_can_double_assign_one_contact():
    """A coordination failure the team planner cannot produce, and which
    per-agent planning must expose rather than repair.

    Every agent believes only in contact 0, so every agent that intercepts
    picks contact 0 -- nobody is covering contact 1. This is the raw material
    for the duplicate-NEUTRALIZE metric.
    """
    from planning.planner import RuleBasedBlueRolePlanner

    planner = RuleBasedBlueRolePlanner(neutralize_fraction=0.5, min_neutralizers=1)
    views = _views(BLUES_4, {i: {0: CONTACTS_2[0]} for i in BLUES_4})
    assembled = _assemble(planner, views)

    assert set(assembled.target_red_ids.values()) == {0}
    assert len(assembled.target_red_ids) >= 2, "more than one agent on the same contact"


def test_plan_one_with_no_believed_contact_defends():
    """Matches the team planner's fog-of-war branch exactly: an agent that
    believes in nothing holds the perimeter rather than inventing a target."""
    from planning.planner import RuleBasedBlueRolePlanner
    from planning.roles import BlueRole

    planner = RuleBasedBlueRolePlanner(neutralize_fraction=0.5, min_neutralizers=1)
    views = _views(BLUES_4, {})
    out = planner.plan_one(views[0])

    assert out.roles == {0: BlueRole.DEFEND}
    assert out.target_red_ids == {}


def test_plan_one_returns_only_the_acting_agents_slot():
    """Information isolation at the output, not just the input. An agent that
    returned roles for its teammates would be dictating to them, which is a
    centralized decision wearing a per-agent interface."""
    from planning.planner import RuleBasedBlueRolePlanner

    planner = RuleBasedBlueRolePlanner(neutralize_fraction=0.5, min_neutralizers=1)
    views = _views(BLUES_4, {i: CONTACTS_2 for i in BLUES_4})

    for agent_id in BLUES_4:
        out = planner.plan_one(views[agent_id])
        assert set(out.roles) == {agent_id}
        assert set(out.target_red_ids) <= {agent_id}
        assert set(out.recon_target_regions) <= {agent_id}


def test_plan_one_respects_a_belief_validity_age():
    """max_age is the consumer-side staleness limit the store deliberately does
    not apply itself. Past it, the agent believes in nothing and defends."""
    from planning.planner import RuleBasedBlueRolePlanner
    from planning.roles import BlueRole

    planner = RuleBasedBlueRolePlanner(neutralize_fraction=0.5, min_neutralizers=1)
    views = _views(BLUES_4, {i: CONTACTS_2 for i in BLUES_4}, now=50, observed_at=0)

    fresh_enough = planner.plan_one(views[1], max_age=100)
    too_old = planner.plan_one(views[1], max_age=10)

    assert fresh_enough.roles[1] == BlueRole.NEUTRALIZE
    assert too_old.roles[1] == BlueRole.DEFEND


def test_the_gate_holds_when_a_blue_is_disabled():
    """The two paths determine the active set by *different mechanisms*, and
    this is where they could genuinely disagree.

    The team planner filters on obs.disabled_blues. The per-agent path has no
    such field -- a disabled blue is simply absent from the teammate channel
. Since the role
    split is a fraction of the active count, disagreeing about who is active
    changes n_intercept and therefore the whole assignment.
    """
    from planning.planner import PlannerObservation, RuleBasedBlueRolePlanner, plan_full_from

    planner = RuleBasedBlueRolePlanner(neutralize_fraction=0.5, min_neutralizers=1)
    alive = {i: s for i, s in BLUES_4.items() if i != 2}  # blue 2 is disabled

    views = _views(alive, {i: CONTACTS_2 for i in alive})
    assembled = _assemble(planner, views)

    red_states = np.zeros((2, 4), dtype=float)
    for cid, state in CONTACTS_2.items():
        red_states[cid] = state
    obs = PlannerObservation(
        x_current=np.concatenate([BLUES_4[i] for i in sorted(BLUES_4)]),
        red_states=red_states.ravel(),
        num_blues=4,
        num_reds=2,
        sim_step=0,
        arena_size=100.0,
        disabled_blues=[2],
        detected_red_indices=[0, 1],
    )
    central = plan_full_from(planner, obs)

    assert 2 not in assembled.roles, "a disabled blue must not be assigned"
    assert assembled.roles == central.roles
    assert assembled.target_red_ids == central.target_red_ids


def test_the_gate_holds_for_an_odd_team_and_three_contacts():
    """Exercises the ceil() in n_intercept and a contact set larger than the
    team, so the gate is not only checked on the even split where a rounding
    disagreement between the two paths would cancel out."""
    from planning.planner import RuleBasedBlueRolePlanner

    planner = RuleBasedBlueRolePlanner(neutralize_fraction=0.5, min_neutralizers=1)
    blues = {i: BLUES_4[i] for i in (0, 1, 2)}
    contacts = {
        0: np.array([15.0, 15.0, 0.0, 0.0]),
        1: np.array([65.0, 65.0, 0.0, 0.0]),
        2: np.array([40.0, 5.0, 0.0, 0.0]),
    }
    views = _views(blues, {i: contacts for i in blues})

    assembled = _assemble(planner, views)
    central = _centralized_on_the_same_belief(planner, views[0], blues, n_reds=3)

    assert len(assembled.roles) == 3
    assert assembled.roles == central.roles
    assert assembled.target_red_ids == central.target_red_ids


# ---------------------------------------------------------------------------
# Step 4: RECON staleness comes from the agent's own CoverageBelief.
# ---------------------------------------------------------------------------

def _tracker_2x2():
    from perception.coverage import CoverageTracker

    return CoverageTracker(arena_size=100.0, grid_dim=2)


# Taken from the tracker rather than hardcoded: region_id -> center is arena
# geometry owned by CoverageTracker, and a test that guessed the ordering would
# be asserting against its own fixture instead of against the code.
CENTERS_4 = np.asarray(_tracker_2x2().centers, dtype=float)


def _views_with_coverage(blue_states, per_agent_contacts, per_agent_sweeps, now=100):
    """Views whose coverage belief is populated per agent.

    ``per_agent_sweeps`` is {agent_id: {region_id: swept_at_step}}; a region
    absent from an agent's dict is one that agent believes was never swept.
    """
    views = {}
    for agent_id, self_state in blue_states.items():
        store = LocalBeliefStore(agent_id, n_regions=len(CENTERS_4))
        for cid, state in per_agent_contacts.get(agent_id, {}).items():
            store.observe_target(cid, state, 0)
        for region_id, swept_at in per_agent_sweeps.get(agent_id, {}).items():
            store.observe_coverage(region_id, swept_at)
        views[agent_id] = store.snapshot(
            now=now,
            dt=0.1,
            self_state=self_state,
            coverage_centers=CENTERS_4,
            teammates={i: s for i, s in blue_states.items() if i != agent_id},
        )
    return views


def test_coverage_belief_summarize_matches_the_tracker_shape():
    """plan_one feeds view.coverage_regions() to the same
    _select_recon_assignments the team path uses, so the two summaries must be
    structurally interchangeable -- not merely similar."""
    tracker = _tracker_2x2()
    tracker.last_seen_step[0] = 40
    tracker.last_seen_step[3] = 90

    views = _views_with_coverage(
        {0: BLUES_4[0]}, {}, {0: {0: 40, 3: 90}}, now=100
    )
    belief_summary = views[0].coverage_regions()
    tracker_summary = tracker.summarize(100)

    assert [r["region_id"] for r in belief_summary] == [
        r["region_id"] for r in tracker_summary
    ]
    assert [r["steps_since_seen"] for r in belief_summary] == [
        r["steps_since_seen"] for r in tracker_summary
    ]
    assert set(belief_summary[0]) == set(tracker_summary[0])


def test_recon_target_follows_the_agents_own_coverage_belief():
    """The same agent, same position, same everything except what it believes
    about coverage, must choose a different region.

    Before this, recon_target_override came from the global CoverageTracker for
    every agent -- a free team oracle. Note the rule assigns the k-th
    DEFEND candidate the k-th stalest region, so agent 1 takes the runner-up,
    not the stalest; the test holds the rule fixed and varies only the belief.
    """
    from planning.planner import RuleBasedBlueRolePlanner
    from planning.roles import BlueRole

    planner = RuleBasedBlueRolePlanner(
        neutralize_fraction=0.0, min_neutralizers=0, recon_fraction=1.0, min_recon=1
    )
    blues = {0: BLUES_4[0], 1: BLUES_4[1]}

    # Agent 0 believes 3 is stalest (ranked 3,1,2,0 -- it takes 3).
    # Agent 1 believes 0 is stalest (ranked 0,2,3,1 -- as runner-up it takes 2).
    own_beliefs = {
        0: {0: 95, 1: 90, 2: 90, 3: 10},
        1: {0: 10, 1: 95, 2: 90, 3: 90},
    }
    views = _views_with_coverage(blues, {}, own_beliefs, now=100)

    first = planner.plan_one(views[0])
    second = planner.plan_one(views[1])

    assert first.roles[0] == BlueRole.RECON
    assert second.roles[1] == BlueRole.RECON
    assert first.recon_target_regions[0] == 3
    assert second.recon_target_regions[1] == 2

    # The control: give agent 1 agent 0's belief instead, changing nothing else.
    # It is the belief that moved the choice, not agent 1's rank in the order.
    shared = _views_with_coverage(blues, {}, {0: own_beliefs[0], 1: own_beliefs[0]}, now=100)
    assert planner.plan_one(shared[1]).recon_target_regions[1] == 1


def test_identical_coverage_beliefs_give_identical_recon_assignments():
    """The coverage half of the equivalence gate: with a shared coverage
    belief the per-agent recon assignment matches the centralized one, so
    divergence in the local arm is attributable to information."""
    from planning.planner import RuleBasedBlueRolePlanner

    planner = RuleBasedBlueRolePlanner(
        neutralize_fraction=0.5, min_neutralizers=1, recon_fraction=1.0, min_recon=1
    )
    sweeps = {0: 95, 1: 20, 2: 90, 3: 60}
    views = _views_with_coverage(
        BLUES_4, {i: CONTACTS_2 for i in BLUES_4}, {i: sweeps for i in BLUES_4}, now=100
    )

    assembled = _assemble(planner, views)
    central = _centralized_on_the_same_belief(planner, views[0], BLUES_4, n_reds=2)

    assert assembled.recon_target_regions
    assert assembled.recon_target_regions == central.recon_target_regions
    assert assembled.roles == central.roles


def test_an_agent_with_no_coverage_belief_gets_no_recon():
    """n_regions=0 disables coverage for an agent. It must degrade to DEFEND
    rather than raising or inventing a region."""
    from planning.planner import RuleBasedBlueRolePlanner
    from planning.roles import BlueRole

    planner = RuleBasedBlueRolePlanner(
        neutralize_fraction=0.0, min_neutralizers=0, recon_fraction=1.0, min_recon=1
    )
    views = _views(BLUES_4, {})  # built with n_regions=0, so coverage is None

    out = planner.plan_one(views[0])
    assert out.roles[0] == BlueRole.DEFEND
    assert out.recon_target_regions == {}


def test_region_centers_are_static_geometry_not_information():
    """A region id resolves to the same point for every agent regardless of
    belief. Centers are arena geometry, so resolving them through the global
    CoverageTracker is not a ground-truth leak -- only *staleness* is
    information, and that comes from the agent's own belief."""
    tracker = _tracker_2x2()
    views = _views_with_coverage(
        {0: BLUES_4[0], 1: BLUES_4[1]},
        {},
        {0: {0: 95}, 1: {3: 95}},  # deliberately different beliefs
        now=100,
    )

    for region_id in range(len(CENTERS_4)):
        from_agent_0 = views[0].coverage_regions()[region_id]["center"]
        from_agent_1 = views[1].coverage_regions()[region_id]["center"]
        np.testing.assert_allclose(from_agent_0, from_agent_1)
        np.testing.assert_allclose(from_agent_0, tracker.region_center(region_id))


# ---------------------------------------------------------------------------
# Step 5: Simulation dispatches per agent, on frozen epoch snapshots.
# ---------------------------------------------------------------------------


class _CapturingPlanner:
    """Transparent wrapper that records every view Simulation hands to plan_one.

    Lets the episode-level tests check the *wiring* -- that Simulation builds
    views with the right teammates, coverage centers, active set and max_age --
    rather than re-testing the planner, which the unit tests above already do.
    """

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def plan(self, obs):
        return self.inner.plan(obs)

    def plan_one(self, view, max_age=None):
        out = self.inner.plan_one(view, max_age=max_age)
        self.calls.append((view.now, view, out))
        return out

    def epochs(self):
        """[(now, {agent_id: view}, assembled PlannerOutput), ...]"""
        from planning.planner import PlannerOutput

        grouped = {}
        for now, view, out in self.calls:
            views, merged = grouped.setdefault(now, ({}, PlannerOutput()))
            views[view.agent_id] = view
            merged.roles.update(out.roles)
            merged.target_red_ids.update(out.target_red_ids)
            merged.recon_target_regions.update(out.recon_target_regions)
        return [(now, *grouped[now]) for now in sorted(grouped)]


# conftest's default fixture (2 blues / 1 red / 40 steps) never detects the red,
# so every belief stays empty and both Experiment 0 arms trivially agree. A gate
# test on that fixture would pass vacuously. This is the smallest configuration
# measured to actually produce detections, and therefore non-empty beliefs.
DETECTING = dict(num_blues=3, num_reds=2, sim_steps=150, seed=7)


def _run_per_agent(**overrides):
    from conftest import build_simulation, run_quiet
    from planning.planner import RuleBasedBlueRolePlanner

    inner = overrides.pop(
        "inner", RuleBasedBlueRolePlanner(recon_fraction=0.5, min_recon=1)
    )
    params = dict(DETECTING)
    params.update(overrides)
    planner = _CapturingPlanner(inner)
    sim = build_simulation(role_planner=planner, per_agent_planning=True, **params)
    return run_quiet(sim), planner


def test_simulation_per_agent_planning_is_off_by_default():
    """The byte-identity gate. A new subsystem lands default-off, and
    belief_mode='ground_truth' must keep reproducing the golden baselines."""
    from conftest import build_simulation

    assert build_simulation().per_agent_planning is False


def test_per_agent_planning_requires_a_planner_that_supports_it():
    """Fail at construction, not 400 steps into a sweep. Silently falling back
    to the team planner would produce an episode that looks decentralized in
    the config and is not -- worse than a crash."""
    import pytest
    from conftest import build_simulation
    from planning.planner import BlueRolePlanner
    from planning.roles import BlueRole

    class TeamOnly(BlueRolePlanner):  # no plan_one -- like the LLM planners
        def plan(self, obs):
            return {i: BlueRole.DEFEND for i in range(obs.num_blues)}

    with pytest.raises(TypeError, match="plan_one"):
        build_simulation(role_planner=TeamOnly(), per_agent_planning=True)


def test_simulation_builds_views_with_the_declared_channels():
    """Wiring check: teammates present and active-only, coverage centers
    threaded through, self excluded."""
    sim, planner = _run_per_agent(belief_mode="shared")
    epochs = planner.epochs()
    assert epochs, "planning must actually have run per agent"

    for now, views, _ in epochs:
        for agent_id, view in views.items():
            assert view.agent_id == agent_id
            assert agent_id not in view.teammates
            assert set(view.teammates) == set(views) - {agent_id}
            assert view.coverage_centers is not None
            assert view.coverage_regions() is not None


def test_simulation_shared_beliefs_reproduce_the_centralized_assignment():
    """**The M3 gate, at episode level**.

    The unit tests establish that the decomposition is faithful. This checks
    Simulation's wiring does not break it: at every planning epoch of a real
    episode, the assembled per-agent assignment equals what the same planner
    produces centrally on the same shared belief.

    The reference is deliberately the team planner fed *the belief*, not the
    team planner as Simulation runs it by default -- that one reads the
    detection union and ground-truth positions, so comparing against it would
    conflate the decomposition with the belief layer itself.
    """
    sim, planner = _run_per_agent(belief_mode="shared")
    assert not any(sim.blue_disabled), (
        "index mapping below assumes no blue was disabled"
    )

    epochs = planner.epochs()
    assert epochs

    # Guard against a vacuous pass: if no agent ever believed in a contact,
    # every epoch takes the fog-of-war branch and the gate proves nothing.
    assert any(
        v.known_contact_ids() for _, views, _ in epochs for v in views.values()
    ), "no contact was ever believed -- the gate would pass vacuously"

    for now, views, assembled in epochs:
        any_view = views[min(views)]
        # Shared mode: every agent must hold the same contacts, or the gate is
        # not even applicable.
        contact_sets = {frozenset(v.known_contact_ids()) for v in views.values()}
        assert len(contact_sets) == 1, f"beliefs diverged at step {now}"

        blue_states = any_view.active_blue_states()
        central = _centralized_on_the_same_belief(
            planner.inner, any_view, blue_states, n_reds=sim.num_reds
        )
        assert assembled.roles == central.roles, f"roles differ at step {now}"
        assert assembled.target_red_ids == central.target_red_ids, (
            f"targets differ at step {now}"
        )


def test_epoch_snapshots_are_frozen_against_planning_order():
    """: all of epoch k's decisions read beliefs captured at the same
    logical instant. Replaying a real epoch's views in reverse must give the
    same answers -- if views aliased live state, order would matter.
    """
    sim, planner = _run_per_agent(belief_mode="shared")
    for now, views, assembled in planner.epochs():
        forward = {}
        for agent_id in sorted(views):
            forward.update(planner.inner.plan_one(views[agent_id]).roles)
        backward = {}
        for agent_id in sorted(views, reverse=True):
            backward.update(planner.inner.plan_one(views[agent_id]).roles)
        assert forward == backward == assembled.roles


def test_local_and_shared_arms_make_different_decisions():
    """What M3 was for.

    Before this step the two arms differed only in what the controllers acted
    on -- role assignment was centralized and fed by the team-wide detection
    union in both. If the arms still produced identical role histories, the
    bracket would have nothing to measure at the decision level.
    """
    shared_sim, shared_planner = _run_per_agent(belief_mode="shared")
    local_sim, _ = _run_per_agent(belief_mode="local")

    assert any(
        v.known_contact_ids()
        for _, views, _ in shared_planner.epochs()
        for v in views.values()
    ), "no contact was ever believed -- the arms would agree for the wrong reason"

    shared_roles = [dict(d) for d in shared_sim.role_assignment_history]
    local_roles = [dict(d) for d in local_sim.role_assignment_history]
    assert shared_roles != local_roles, (
        "the arms must be able to differ in their decisions, not only in control"
    )


def test_ground_truth_mode_is_not_an_experimental_condition_for_per_agent_planning():
    """Documents a real asymmetry rather than asserting it away.

    belief_mode='ground_truth' gives every agent *every* red at age 0,
    including undetected ones, because it exists to reproduce the pre-M2
    nearest-red heuristic which scanned all N_red without a detection filter.
    The team planner, by contrast, is fed the detection union. So with
    per-agent planning the two paths legitimately disagree in this mode -- the
    per-agent planner sees through the fog of war.

    That is harmless because ground_truth is a regression baseline, not an arm
    of Experiment 0. It is pinned here so nobody reads the disagreement as a
    gate failure, and so nobody runs an arm in this mode by accident.
    """
    from conftest import build_simulation, run_quiet

    gt_sim, _ = _run_per_agent(belief_mode="ground_truth")
    central = run_quiet(build_simulation(**DETECTING))

    gt_roles = [dict(d) for d in gt_sim.role_assignment_history]
    central_roles = [dict(d) for d in central.role_assignment_history]
    assert gt_roles != central_roles, (
        "if these ever match, ground_truth belief has become detection-gated "
        "and this test should be revisited along with _update_beliefs"
    )
