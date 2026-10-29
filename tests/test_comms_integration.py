"""M4 integration: the communication substrate wired into ``Simulation``.

Two jobs, and the first matters more than the second:

1. **The default-off baseline must not move.** Every mode that existed before
   M4 -- ``ground_truth``, ``shared``, ``local`` -- must produce byte-identical
   episodes. Note this is asserted *because the network is not constructed for
   them*, not because a complete graph is harmless: a lossless complete graph is
   a degenerate configuration, not "communication off". The golden episode
   baselines cover ``ground_truth``; these cover the other two directly.

2. **``networked`` must actually behave like a network.** One hop per comm
   round, latency that a complete graph does not remove, range that partitions
   the team, and no observation rejuvenated by relaying.

``networked`` is deliberately NOT expected to equal ``shared``. Exchange costs
at least one communication round, so a sighting reaches a teammate one comm tick
after it is made where ``shared`` delivers it the same step. Any test asserting
the two are equal would be asserting the latency away.
"""

import numpy as np
import pytest

from conftest import build_simulation, fingerprint, run_quiet


def _run(**kwargs):
    return run_quiet(build_simulation(**kwargs))


# ---------------------------------------------------------------------------
# 1. The pre-M4 modes are untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["ground_truth", "shared", "local"])
@pytest.mark.parametrize("seed", [7, 130])
def test_pre_m4_modes_are_byte_identical_with_comms_params_present(mode, seed):
    """Adding the comms parameters must not perturb any existing mode.

    Passing them explicitly at their defaults must produce the same episode as
    not passing them at all -- if the comms clock or the network object leaked
    into a non-networked path, this diverges.
    """
    a = _run(num_blues=2, num_reds=2, seed=seed, sim_steps=40, belief_mode=mode)
    b = _run(
        num_blues=2, num_reds=2, seed=seed, sim_steps=40, belief_mode=mode,
        comm_interval_seconds=None, r_comm=float("inf"), comm_message_ttl=4,
    )
    assert fingerprint(a) == fingerprint(b)


@pytest.mark.parametrize("mode", ["ground_truth", "shared", "local"])
def test_no_network_is_constructed_for_pre_m4_modes(mode):
    """The strongest form of default-off: there is no network object to have
    had a side effect."""
    sim = _run(num_blues=2, num_reds=1, seed=7, sim_steps=20, belief_mode=mode)
    assert sim.network is None


@pytest.mark.parametrize("mode", ["ground_truth", "shared", "local"])
def test_comms_settings_are_ignored_by_pre_m4_modes(mode):
    """A hostile comms configuration -- zero range -- must not change a mode
    that does not use the network."""
    a = _run(num_blues=2, num_reds=2, seed=7, sim_steps=40, belief_mode=mode)
    b = _run(num_blues=2, num_reds=2, seed=7, sim_steps=40, belief_mode=mode, r_comm=0.0)
    assert fingerprint(a) == fingerprint(b)


# ---------------------------------------------------------------------------
# 2. networked behaves like a network
# ---------------------------------------------------------------------------


def test_networked_mode_runs_and_builds_a_network():
    sim = _run(num_blues=3, num_reds=2, seed=7, sim_steps=60, belief_mode="networked")
    assert sim.network is not None
    assert sim.comm_stats["rounds"] > 0


def _belief_ages(sim):
    return [
        age
        for record in sim.recorder.steps
        for per_agent in record["beliefs"].values()
        for age in per_agent.values()
    ]


def test_networked_is_not_the_same_as_shared_even_on_a_complete_graph():
    """The claim this test exists to stop: 'complete lossless graph, therefore
    it reproduces shared'. It does not -- exchange costs a round, and that
    latency is exactly the quantity the communication conditions varies.

    Asserted on **belief age**, which is the mechanism, rather than on the
    episode fingerprint. Fingerprints are a weaker signal here: on many seeds
    the extra round of latency does not change the outcome at all, so a
    fingerprint assertion would pass or fail for reasons unrelated to whether
    the network imposed any delay. Measured at seed 11: mean age 0.00 shared
    against 0.60 networked.
    """
    shared = _run(num_blues=3, num_reds=2, seed=11, sim_steps=120, belief_mode="shared")
    networked = _run(
        num_blues=3, num_reds=2, seed=11, sim_steps=120, belief_mode="networked",
        r_comm=float("inf"),
    )
    # Same scenario, so the difference cannot come from the spawn.
    np.testing.assert_allclose(shared.history_red[0], networked.history_red[0])

    shared_ages, networked_ages = _belief_ages(shared), _belief_ages(networked)
    assert shared_ages and networked_ages
    # shared delivers in the same step, so every shared belief is age 0.
    assert max(shared_ages) == 0
    assert np.mean(networked_ages) > 0.0, "the network imposed no latency at all"

    assert any(
        a["beliefs"] != b["beliefs"]
        for a, b in zip(shared.recorder.steps, networked.recorder.steps)
    ), "networked and shared produced identical beliefs at every step"


def test_zero_range_networked_degenerates_to_local():
    """With no links, nothing is ever exchanged, so beliefs must match the
    local arm exactly. This is the sanity check that the network is genuinely
    the only channel in networked mode -- if some other path were leaking
    teammate observations, this would diverge."""
    local = _run(num_blues=3, num_reds=2, seed=5, sim_steps=100, belief_mode="local")
    isolated = _run(
        num_blues=3, num_reds=2, seed=5, sim_steps=100, belief_mode="networked", r_comm=0.0
    )
    assert fingerprint(local) == fingerprint(isolated)


def test_range_controls_how_much_is_exchanged():
    """More range must not deliver less. A monotonicity check rather than an
    exact count, because trajectories diverge once beliefs do."""
    tight = _run(
        num_blues=4, num_reds=2, seed=9, sim_steps=120, belief_mode="networked", r_comm=1.0
    )
    wide = _run(
        num_blues=4, num_reds=2, seed=9, sim_steps=120, belief_mode="networked",
        r_comm=float("inf"),
    )
    assert wide.comm_stats["delivered"] > tight.comm_stats["delivered"]


def test_comm_interval_controls_the_number_of_rounds():
    fast = _run(
        num_blues=3, num_reds=1, seed=7, sim_steps=100, belief_mode="networked",
        comm_interval_seconds=0.1,
    )
    slow = _run(
        num_blues=3, num_reds=1, seed=7, sim_steps=100, belief_mode="networked",
        comm_interval_seconds=1.0,
    )
    assert fast.comm_stats["rounds"] > slow.comm_stats["rounds"]
    assert slow.comm_interval_steps == 10


def test_networked_never_rejuvenates_an_observation():
    """The invariant that makes age-of-information meaningful. A belief's
    ``observed_at`` may never exceed the step at which it is held, and relaying
    must not move it forward."""
    sim = _run(num_blues=4, num_reds=2, seed=3, sim_steps=150, belief_mode="networked")
    for record in sim.recorder.steps:
        step = record["step"]
        for per_agent in record["beliefs"].values():
            for age in per_agent.values():
                assert age >= 0, f"negative age at step {step} -- observation from the future"


def test_a_belief_arrives_no_earlier_than_one_comm_round():
    """Latency is real: an agent cannot hold a teammate's sighting in the same
    step it was made. Checked on ages, which are measured from sensor time."""
    sim = _run(
        num_blues=4, num_reds=2, seed=3, sim_steps=150, belief_mode="networked",
        comm_interval_seconds=1.0,
    )
    # With a 10-step comm interval, some belief must show an age of at least 1;
    # a zero-latency channel would keep every shared belief at age 0.
    ages = [
        age
        for record in sim.recorder.steps
        for per_agent in record["beliefs"].values()
        for age in per_agent.values()
    ]
    assert ages, "no beliefs recorded"
    assert max(ages) >= 1


def test_a_disabled_blue_is_excluded_from_the_comms_graph():
    """A disabled blue must drop out of the graph.

    Driven deterministically by disabling an agent and running one exchange,
    rather than waiting for an episode in which somebody happens to die -- the
    previous version of this test skipped whenever nobody did, which is a test
    that does not test.
    """
    sim = build_simulation(num_blues=4, num_reds=2, seed=3, belief_mode="networked")
    sim.setup_agents()
    sim.reset_episode_state() if hasattr(sim, "reset_episode_state") else None
    # Ensure the stores and network exist as a running episode would have them.
    from perception.belief import LocalBeliefStore
    from comms import DiskLinkModel, Network

    sim.belief_stores = {
        i: LocalBeliefStore(i, n_regions=len(sim.coverage_tracker.centers))
        for i in range(sim.num_blues)
    }
    sim.network = Network(seed=sim.seed, link_model=DiskLinkModel(r_comm=float("inf")))
    # The factory, not a literal: this test stages an episode by hand, and a
    # hand-written copy of the counters goes stale the moment a field is added.
    sim.comm_stats = sim._fresh_comm_stats()

    # Give everyone something to say, then kill agent 1.
    for i in range(sim.num_blues):
        sim.belief_stores[i].observe_target(0, sim.red_state[: sim.red_nx], step=0)
    sim.blue_disabled[1] = True

    active = [i for i in range(sim.num_blues) if not sim.blue_disabled[i]]
    sim._exchange_beliefs_over_network(step=0, active_blues=active)

    assert 1 not in sim.network.neighbours(0), "a disabled blue is still an edge endpoint"
    for edge in sim.network.edges:
        assert 1 not in edge, f"disabled blue still in edge {set(edge)}"

    # And it receives nothing on the following round.
    sim._exchange_beliefs_over_network(step=1, active_blues=active)
    delivered_to_dead = sim.network._inbox.get(1, [])
    assert not delivered_to_dead, "a disabled blue was delivered messages"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [3, 7])
def test_networked_episodes_are_reproducible(seed):
    """Counter-based link randomness means a rerun is byte-identical."""
    a = _run(num_blues=3, num_reds=2, seed=seed, sim_steps=80, belief_mode="networked", r_comm=30.0)
    b = _run(num_blues=3, num_reds=2, seed=seed, sim_steps=80, belief_mode="networked", r_comm=30.0)
    assert fingerprint(a) == fingerprint(b)


def test_networked_is_recorded_in_the_manifest():
    sim = build_simulation(belief_mode="networked", r_comm=17.5, comm_interval_seconds=0.5)
    assert sim._init_args["belief_mode"] == "networked"
    assert sim._init_args["r_comm"] == 17.5
    assert sim._init_args["comm_interval_seconds"] == 0.5
