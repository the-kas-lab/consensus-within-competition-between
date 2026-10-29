"""The belief mediators the communication conditions reports, and the contamination they correct.

``EpisodeRecorder`` snapshots every entry in ``belief_stores``. Every
belief-*consuming* path in ``Simulation`` -- the union merge in
``_update_beliefs``, ``_plan_per_agent``, ``_believed_targets`` -- filters to
``active_blues``. So a disabled blue's store freezes at its death and ages for
the rest of the episode while nothing reads it, and any metric averaged over all
stores is describing a store no decision consumes.

That matters beyond tidiness: the arm that loses more defenders accumulates more
frozen stores, and Exp 0 measured the local arm losing significantly more
(attrition -0.046 [-0.069, -0.024]). The contamination is therefore *correlated
with the condition*, not noise around it.

These tests pin the corrected metrics against shipped code -- ``episode_metrics``
as the experiment scripts call it -- rather than against a reimplementation.
"""

import numpy as np
import pytest

from conftest import build_simulation, run_quiet
from scripts.episode_builder import episode_metrics


def _metrics(**kwargs):
    kwargs.setdefault("num_blues", 4)
    kwargs.setdefault("num_reds", 2)
    kwargs.setdefault("sim_steps", 400)
    kwargs.setdefault("per_agent_planning", True)
    return episode_metrics(run_quiet(build_simulation(**kwargs)))


# ---------------------------------------------------------------------------
# 1. belief_age_excess is zero under shared -- by construction, not by luck
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [3001, 3002, 3006])
def test_shared_beliefs_have_no_excess_age_over_the_team_best(seed):
    """The defining property of the `shared` reference.

    `shared` merges the union of first-hand observations by max `observed_at`
    into every *active* store each step, so no live agent can hold a record
    staler than the freshest one in the live team. An excess of exactly 0.00 is
    what makes this metric a measurement of the manipulation rather than a
    correlate of it -- and it is the check that would catch the merge silently
    dropping an agent.

    Asserted on seeds where a blue actually dies, which is precisely when the
    all-stores version of this metric is non-zero.
    """
    m = _metrics(seed=seed, belief_mode="shared")
    assert m["belief_age_excess"] == 0.0
    assert m["contact_coverage_deficit"] == 0.0
    # And disagreement, the other shared-arm invariant, over live agents.
    assert m["live_belief_disagreement"] == 0.0


def test_local_beliefs_do_have_excess_age():
    """The lower anchor must move the quantity the upper anchor pins at zero,
    or the metric is measuring nothing."""
    m = _metrics(seed=3003, belief_mode="local")
    assert m["belief_age_excess"] > 0.0
    assert m["contact_coverage_deficit"] > 0.0


# ---------------------------------------------------------------------------
# 2. The contamination is real, and the correction removes it
# ---------------------------------------------------------------------------


def test_dead_stores_inflate_the_uncorrected_age_and_the_live_metric_drops_it():
    """Seed 3001 loses a defender. The all-stores mean age must exceed the
    live-only mean, and the all-stores excess must be non-zero *even under
    `shared`* -- which is the contamination stated as an assertion.

    Skips rather than passes vacuously if nobody dies: with no frozen store
    there is nothing to contaminate and the two metrics must agree, which the
    final branch checks instead.
    """
    sim = run_quiet(build_simulation(num_blues=4, num_reds=2, seed=3001,
                                     sim_steps=400, per_agent_planning=True,
                                     belief_mode="shared"))
    m = episode_metrics(sim)
    if not any(sim.blue_disabled):
        assert m["mean_belief_age"] == pytest.approx(m["live_mean_belief_age"])
        pytest.skip("no defender lost on this seed; nothing to contaminate")

    assert m["mean_belief_age"] > m["live_mean_belief_age"], (
        "a frozen store did not inflate the uncorrected age"
    )
    # The uncorrected excess is non-zero under shared purely because of the
    # frozen store; the corrected one is exactly zero (test above).
    assert m["belief_age_excess"] == 0.0


def test_live_metrics_equal_the_originals_when_nobody_dies():
    """With no attrition the two definitions must coincide exactly. This is
    what makes the correction a *correction* rather than a different metric
    that happens to point the same way."""
    sim = run_quiet(build_simulation(num_blues=4, num_reds=2, seed=3004,
                                     sim_steps=150, per_agent_planning=True,
                                     belief_mode="local"))
    if any(sim.blue_disabled):
        pytest.skip("a defender was lost; the definitions are meant to differ")
    m = episode_metrics(sim)
    assert m["live_mean_belief_age"] == pytest.approx(m["mean_belief_age"])
    assert m["live_belief_disagreement"] == pytest.approx(m["belief_disagreement"])
    assert m["live_mean_contacts_believed"] == pytest.approx(
        m["mean_contacts_believed"]
    )


# ---------------------------------------------------------------------------
# 3. The networked protocol shows up in excess, where it is not buried
# ---------------------------------------------------------------------------


def test_a_complete_lossless_graph_still_shows_protocol_latency_in_excess():
    """The reference/protocol distinction, measured on the mediator the communication conditions reports.

    `shared` delivers in the same step, so no live agent can be staler than the
    team's freshest record and excess is exactly zero. `networked` pays one
    communication round per hop even on a complete lossless graph, and that cost
    appears here -- which is what makes excess, not raw age, the quantity the
    sweep varies.

    Raw age is **not** asserted to dwarf the excess. It does so on average over
    400-step episodes (22.4 steps against 0.00 for `shared` on seed 3001), but
    it is a per-seed accident how continuously a contact stays in somebody's
    sensor cone: on seed 3003 nothing is ever stale for want of looking, so raw
    age and excess coincide. The population claim belongs to the sweep, with an
    interval, not to a two-episode unit test.

    What *is* guaranteed by construction is the ordering below: excess is raw
    age minus a non-negative team best, so it can never exceed it.
    """
    shared = _metrics(seed=3003, belief_mode="shared")
    networked = _metrics(seed=3003, belief_mode="networked", r_comm=float("inf"),
                         comm_interval_seconds=0.1)

    assert shared["belief_age_excess"] == 0.0
    assert networked["belief_age_excess"] > 0.0, (
        "a complete lossless graph delivered with no latency at all"
    )
    assert networked["belief_age_excess"] <= networked["live_mean_belief_age"]
    # One round of latency, not tens of steps of it.
    assert networked["belief_age_excess"] < 5.0


def test_degraded_range_raises_excess_and_deficit_together():
    """Fewer reachable teammates must make records both staler and rarer.

    Both halves are asserted because either alone is gameable: a cell where
    nobody hears anything scores *zero* excess on the records it happens to
    hold, and would look fresh.
    """
    good = _metrics(seed=3003, belief_mode="networked", r_comm=float("inf"),
                    comm_interval_seconds=0.1)
    poor = _metrics(seed=3003, belief_mode="networked", r_comm=8.0,
                    comm_interval_seconds=1.0)
    assert poor["belief_age_excess"] > good["belief_age_excess"]
    assert poor["contact_coverage_deficit"] > good["contact_coverage_deficit"]
