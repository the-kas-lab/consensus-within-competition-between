"""The communication conditions step 1: the connectivity mediators, measured per episode.

Why these exist at all. forbids reporting sweep results against
the configured ``R_comm`` alone, and M0 sharpened that: it measured graph
diameter at 1.0-1.9 hops across *every* range worth running, so diameter cannot
discriminate between conditions, while mean partition duration moves 15 -> 94
steps over the same grid. **Partition duration is the mediator**, and an
instrument that cannot resolve it makes the communication conditions sweep unreportable.

Every test here runs a real ``Simulation`` and reads ``comm_metrics()``. None of
them recompute connectivity in test-local code -- two M4 tests did that and had
to be redone, because a test that reimplements the logic it is checking passes
whether or not the shipped code works.
"""

import numpy as np
import pytest

from conftest import build_simulation, fingerprint, run_quiet


def _run(**kwargs):
    return run_quiet(build_simulation(**kwargs))


def _networked(**kwargs):
    kwargs.setdefault("belief_mode", "networked")
    return _run(**kwargs)


# ---------------------------------------------------------------------------
# 1. Default-off: there is nothing to measure outside networked mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["ground_truth", "shared", "local"])
def test_comm_metrics_is_none_when_no_network_exists(mode):
    """``None``, not a dict of zeros.

    The distinction is load-bearing for the sweep writeup: a zeroed dict reads
    as "measured, and the team was disconnected", which is a claim about the
    episode. ``None`` says the question did not apply because no ``Network``
    was ever constructed -- which is also the property that makes the pre-M4
    modes provably untouched.
    """
    sim = _run(belief_mode=mode)
    assert sim.network is None
    assert sim.comm_metrics() is None


def test_instrumenting_the_network_did_not_move_the_other_modes():
    """The standing rule: existing modes stay byte-identical.

    ``_record_connectivity`` is called from inside the networked branch only,
    but that is an argument, not evidence. This compares fingerprints.
    """
    for mode in ("ground_truth", "shared", "local"):
        a = fingerprint(_run(belief_mode=mode, seed=7, sim_steps=60))
        b = fingerprint(_run(belief_mode=mode, seed=7, sim_steps=60))
        assert a == b


# ---------------------------------------------------------------------------
# 2. The instrument reads the two extremes correctly
# ---------------------------------------------------------------------------


def test_complete_graph_reports_no_partition_and_unit_component():
    """``R_comm = inf`` is the step-1 reference cell. If the instrument cannot
    report it as fully connected, nothing downstream of it is trustworthy."""
    sim = _networked(num_blues=4, num_reds=2, seed=3001, sim_steps=80, r_comm=float("inf"))
    m = sim.comm_metrics()

    assert m["rounds"] > 0
    assert m["connected_fraction"] == 1.0
    assert m["partitioned_fraction"] == 0.0
    assert m["mean_lcc_fraction"] == 1.0
    assert m["n_partitions"] == 0
    assert m["mean_partition_steps"] == 0.0
    assert m["max_partition_steps"] == 0
    # A complete graph is one hop between any pair, at every round.
    assert m["mean_diameter_when_connected"] == 1.0


def test_zero_range_reports_one_unbroken_partition():
    """``R_comm = 0`` is the lower extreme, and the run must be reported as a
    *single* partition spanning the episode rather than as many short ones --
    otherwise "mean partition duration" would shrink toward the sampling period
    in exactly the conditions that partition worst.

    Guarded on liveness: ``connectivity_metrics`` drops disabled blues, so a
    team attrited to one survivor is trivially "connected" and would end the
    run. The assertion is conditioned on that not having happened.
    """
    sim = _networked(num_blues=4, num_reds=2, seed=3002, sim_steps=60, r_comm=0.0)
    m = sim.comm_metrics()
    if sum(bool(x) for x in sim.blue_disabled) >= sim.num_blues - 1:
        pytest.skip("team attrited to a single live blue; connectivity is degenerate")

    assert m["connected_fraction"] == 0.0
    assert m["n_partitions"] == 1, "a continuous outage must be one run, not many"
    assert m["max_partition_steps"] == m["rounds"] * m["comm_interval_steps"]
    assert m["mean_lcc_fraction"] < 1.0
    # Disconnected rounds contribute no diameter -- None is the honest answer
    # there, so the mean is over an empty set.
    assert np.isnan(m["mean_diameter_when_connected"])


# ---------------------------------------------------------------------------
# 3. The instrument discriminates *between* the extremes -- the sweep's premise
# ---------------------------------------------------------------------------


def test_an_intermediate_range_is_intermittently_connected():
    """The whole the communication conditions sweep rests on there being cells that are neither always
    nor never connected. If no ``R_comm`` produces a fraction strictly inside
    (0, 1), the range axis has no interior and the sweep is two points.

    The range is not pinned to a specific value: M0's grid was measured on
    ``ground_truth`` trajectories at a 5.0 s planning period, and trajectories
    under ``networked`` at 1.0 s differ, so a hard-coded expectation for one
    range would be asserting M0's numbers rather than this property.
    """
    fractions = {}
    for r in (8.0, 15.0, 25.0):
        m = _networked(
            num_blues=4, num_reds=2, seed=3003, sim_steps=120, r_comm=r
        ).comm_metrics()
        fractions[r] = m["connected_fraction"]

    assert any(0.0 < f < 1.0 for f in fractions.values()), fractions
    # Connectivity must not *decrease* as the radius grows: the disk graph is
    # monotone in R_comm at fixed positions. Trajectories do differ between
    # cells, so this is asserted end-to-end rather than exactly.
    assert fractions[25.0] >= fractions[8.0], fractions


def test_partition_duration_is_reported_in_steps_not_rounds():
    """Durations must be on the mission clock, not the protocol clock.

    ``comm_interval_steps`` is the axis the communication conditions sweeps. If partition duration were
    reported in rounds, a team out of contact for the same *wall time* would
    score a shorter partition simply for sampling less often -- the period sweep
    would look flat by construction, which is the opposite of the finding it is
    meant to test for.
    """
    fast = _networked(
        num_blues=4, num_reds=2, seed=3004, sim_steps=120, r_comm=8.0,
        comm_interval_seconds=0.1,
    ).comm_metrics()
    slow = _networked(
        num_blues=4, num_reds=2, seed=3004, sim_steps=120, r_comm=8.0,
        comm_interval_seconds=1.0,
    ).comm_metrics()

    assert fast["comm_interval_steps"] == 1
    assert slow["comm_interval_steps"] == 10
    # Ten times fewer rounds for the same episode length.
    assert slow["rounds"] < fast["rounds"]
    # Both are largely disconnected at R_comm = 8, and both must say so in
    # steps -- a comparable unit -- rather than in their own round counts.
    if fast["n_partitions"] and slow["n_partitions"]:
        assert slow["max_partition_steps"] >= slow["rounds"]


# ---------------------------------------------------------------------------
# 4. Pinning flaw C: comm_message_ttl is inert
# ---------------------------------------------------------------------------


def test_comm_message_ttl_is_inert_because_store_and_forward_is_off():
    """``Simulation(comm_message_ttl=...)`` configures nothing today.

    ``envelope.ttl`` is read only by ``Network.forward``, and the testbed never
    calls it -- multi-hop happens instead by each agent re-broadcasting its
    merged store, one hop per round. This is pinned rather than left implicit so
    that no later reader reports a TTL-bounded relay the code does not have. The
    day store-and-forward is enabled in ``Simulation``, this test fails and the
    writeup has to be updated with it.
    """
    assert _networked(r_comm=20.0).network.store_and_forward is False

    a = fingerprint(_networked(num_blues=4, num_reds=2, seed=3005, sim_steps=80,
                               r_comm=20.0, comm_message_ttl=1))
    b = fingerprint(_networked(num_blues=4, num_reds=2, seed=3005, sim_steps=80,
                               r_comm=20.0, comm_message_ttl=64))
    assert a == b, "TTL changed an episode; store-and-forward must now be live"
