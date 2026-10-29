"""`DeduplicatingBlueRolePlanner`: the second arm of the step-3 comparison.

Three things need pinning, and the first matters most:

1. **The base rule did not move.** Extracting ``_select_intercept_targets`` out
   of ``_assign_team_roles`` is a refactor of shared code, and every result to
   date plus the M3 equivalence gate rests on the original behaviour. The golden
   episode baselines cover this end-to-end; what is asserted here is the direct
   claim that the extracted method returns exactly what the inline code did.
2. **Only the target changes.** The role split -- who intercepts, who defends,
   who recons -- must be identical between the two planners, or an outcome
   difference could not be attributed to allocation.
3. **The dedup rule actually deduplicates**, and degrades sensibly when there
   are more interceptors than contacts.
"""

import numpy as np
import pytest

from conftest import build_simulation, run_quiet
from planning.planner import (
    DeduplicatingBlueRolePlanner,
    RuleBasedBlueRolePlanner,
)
from planning.roles import BlueRole


def _positions(*xy):
    return {i: np.array(p, dtype=float) for i, p in enumerate(xy)}


def _contacts(*xy):
    return {i: np.array(p, dtype=float) for i, p in enumerate(xy)}


# ---------------------------------------------------------------------------
# 1. the base rule is unchanged by the refactor
# ---------------------------------------------------------------------------


def test_base_rule_still_assigns_each_interceptor_its_nearest_contact():
    """The pre-refactor behaviour, stated directly: no deconfliction, every
    interceptor takes its own nearest contact even when that collides."""
    planner = RuleBasedBlueRolePlanner(neutralize_fraction=1.0, min_neutralizers=2)
    blues = _positions((0.0, 0.0), (1.0, 0.0))
    contacts = _contacts((2.0, 0.0), (50.0, 0.0))
    roles, targets, _ = planner._assign_team_roles(
        [0, 1], blues, contacts, coverage_regions=None
    )
    assert roles == {0: BlueRole.NEUTRALIZE, 1: BlueRole.NEUTRALIZE}
    # Both are nearest to contact 0, and the base rule lets them collide.
    assert targets == {0: 0, 1: 0}


def test_base_rule_produces_duplicates_under_perfect_information():
    """The property the Exp 0 pilot rests on: duplication follows from the rule,
    not from missing information, so it survives perfect shared belief."""
    planner = RuleBasedBlueRolePlanner(neutralize_fraction=1.0, min_neutralizers=3)
    blues = _positions((0.0, 0.0), (0.5, 0.0), (1.0, 0.0))
    contacts = _contacts((2.0, 0.0), (80.0, 0.0))
    _, targets, _ = planner._assign_team_roles(
        [0, 1, 2], blues, contacts, coverage_regions=None
    )
    assert len(set(targets.values())) < len(targets)


# ---------------------------------------------------------------------------
# 2. only the target changes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "blues,contacts,active",
    [
        (((0.0, 0.0), (1.0, 0.0)), ((2.0, 0.0), (50.0, 0.0)), [0, 1]),
        (((0.0, 0.0), (9.0, 9.0), (3.0, 1.0)), ((2.0, 0.0), (8.0, 8.0)), [0, 1, 2]),
        (((0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)), ((5.0, 5.0),), [0, 1, 2, 3]),
    ],
)
def test_role_split_is_identical_between_the_two_planners(blues, contacts, active):
    """The whole comparison depends on this. If the dedup condition also changed
    *who* intercepts, an outcome difference would confound allocation with team
    composition."""
    kwargs = dict(neutralize_fraction=0.5, min_neutralizers=1, recon_fraction=0.5, min_recon=1)
    base = RuleBasedBlueRolePlanner(**kwargs)
    dedup = DeduplicatingBlueRolePlanner(**kwargs)
    bp, cp = _positions(*blues), _contacts(*contacts)
    roles_a, _, recon_a = base._assign_team_roles(active, bp, cp, coverage_regions=None)
    roles_b, _, recon_b = dedup._assign_team_roles(active, bp, cp, coverage_regions=None)
    assert roles_a == roles_b
    assert recon_a == recon_b


# ---------------------------------------------------------------------------
# 3. the dedup rule deduplicates
# ---------------------------------------------------------------------------


def test_dedup_gives_each_interceptor_a_distinct_contact():
    planner = DeduplicatingBlueRolePlanner(neutralize_fraction=1.0, min_neutralizers=2)
    blues = _positions((0.0, 0.0), (1.0, 0.0))
    contacts = _contacts((2.0, 0.0), (50.0, 0.0))
    _, targets, _ = planner._assign_team_roles([0, 1], blues, contacts, coverage_regions=None)
    assert sorted(targets.values()) == [0, 1], "interceptors must not collide"
    # The globally closest pair is committed first: blue 1 is nearer contact 0.
    assert targets[1] == 0 and targets[0] == 1


def test_dedup_commits_the_globally_closest_pair_first():
    """Greedy is over *pairs*, not over agents in rank order -- otherwise the
    nearest agent would grab a contact a much closer agent needed."""
    planner = DeduplicatingBlueRolePlanner(neutralize_fraction=1.0, min_neutralizers=2)
    blues = _positions((0.0, 0.0), (10.0, 0.0))
    contacts = _contacts((11.0, 0.0), (3.0, 0.0))
    _, targets, _ = planner._assign_team_roles([0, 1], blues, contacts, coverage_regions=None)
    # closest pair overall is blue1->contact0 at distance 1
    assert targets[1] == 0
    assert targets[0] == 1


def test_dedup_surplus_interceptors_fall_back_rather_than_idle():
    """More pursuers than contacts. The surplus must still be sent somewhere --
    idling it would make this a 'fewer interceptors' condition as well, which
    is a second manipulation."""
    planner = DeduplicatingBlueRolePlanner(neutralize_fraction=1.0, min_neutralizers=3)
    blues = _positions((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
    contacts = _contacts((5.0, 0.0),)
    _, targets, _ = planner._assign_team_roles([0, 1, 2], blues, contacts, coverage_regions=None)
    assert set(targets) == {0, 1, 2}, "every interceptor keeps a target"
    assert set(targets.values()) == {0}


def test_dedup_is_deterministic_under_ties():
    """A planner that shuffled under ties would break the paired design."""
    planner = DeduplicatingBlueRolePlanner(neutralize_fraction=1.0, min_neutralizers=2)
    blues = _positions((0.0, 0.0), (0.0, 0.0))
    contacts = _contacts((1.0, 0.0), (-1.0, 0.0))
    runs = [
        planner._assign_team_roles([0, 1], blues, contacts, coverage_regions=None)[1]
        for _ in range(8)
    ]
    assert all(r == runs[0] for r in runs)


def test_dedup_with_no_contacts_assigns_nothing():
    planner = DeduplicatingBlueRolePlanner(neutralize_fraction=1.0, min_neutralizers=1)
    roles, targets, _ = planner._assign_team_roles(
        [0, 1], _positions((0.0, 0.0), (1.0, 0.0)), {}, coverage_regions=None
    )
    assert targets == {}
    assert set(roles.values()) == {BlueRole.DEFEND}


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


def test_dedup_runs_a_full_episode_in_both_belief_modes():
    """The probe has to survive the real pipeline, including the side-channel
    target handoff and the CBF, not just the decision rule in isolation."""
    for mode in ("shared", "local"):
        sim = run_quiet(
            build_simulation(
                num_blues=4,
                num_reds=2,
                seed=1001,
                sim_steps=60,
                belief_mode=mode,
                per_agent_planning=True,
                role_planner=DeduplicatingBlueRolePlanner(
                    neutralize_fraction=0.5, min_neutralizers=1,
                    recon_fraction=0.5, min_recon=1,
                ),
            )
        )
        assert sim.recorder.steps, f"{mode}: episode recorded nothing"


def test_shared_beliefs_still_reproduce_a_consistent_allocation_under_dedup():
    """Under shared belief every agent allocates over an identical contact set,
    so no two interceptors should collide. This is the dedup condition's
    analogue of the M3 equivalence property; the original rule keeps its own
    equivalence test, which is deliberately left attached to it.
    """
    sim = run_quiet(
        build_simulation(
            num_blues=4,
            num_reds=2,
            seed=1001,
            sim_steps=200,
            belief_mode="shared",
            per_agent_planning=True,
            role_planner=DeduplicatingBlueRolePlanner(
                neutralize_fraction=0.5, min_neutralizers=1,
                recon_fraction=0.5, min_recon=1,
            ),
        )
    )
    interval = sim.planning_interval_steps
    for step in (s for s in sim.recorder.steps if s["step"] % interval == 0):
        assigned = [
            t
            for a, t in step["target_red_ids"].items()
            if t is not None
            and step["blue_roles"].get(str(a)) == BlueRole.NEUTRALIZE.value
        ]
        # Collisions are only allowed when interceptors outnumber contacts.
        believed = {int(c) for per in step["beliefs"].values() for c in per}
        if len(assigned) <= len(believed):
            assert len(assigned) == len(set(assigned)), (
                f"step {step['step']}: shared-belief duplicate {assigned} "
                f"against believed {believed}"
            )
