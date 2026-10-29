"""Determinism guarantees the whole statistical design rests on.

 requires matched scenario seeds across conditions and paired
clean/attack episodes from identical initial state. None of that means anything
unless an episode is a pure function of (seed, config) -- these tests pin that
down *before* the belief/comms/attack layers start adding new randomness.

Two RNG modes are exercised throughout (see simulation/rng.py):

* ``legacy_global_rng=True`` (default) -- spawn/patrol/inert stay on the
  process-global ``np.random``, so episodes remain byte-identical to
  pre-refactor runs. This mode is *known* to leak construction order between
  episodes; that is preserved deliberately, not fixed.
* ``legacy_global_rng=False`` -- every purpose gets an independent Generator.
  This is the mode that must hold the stronger isolation property.

See
"""

import numpy as np
import pytest

from conftest import build_simulation, fingerprint, run_quiet

RNG_MODES = [
    pytest.param(True, id="legacy_global"),
    pytest.param(False, id="isolated_streams"),
]


@pytest.mark.parametrize("legacy", RNG_MODES)
def test_same_seed_same_episode(legacy):
    """Same seed + same config => identical trajectory, roles and outcome."""
    a = fingerprint(run_quiet(build_simulation(seed=11, legacy_global_rng=legacy)))
    b = fingerprint(run_quiet(build_simulation(seed=11, legacy_global_rng=legacy)))
    assert a == b


@pytest.mark.parametrize("legacy", RNG_MODES)
def test_different_seed_different_episode(legacy):
    """Guards against the opposite failure: a fingerprint so coarse, or a sim so
    seed-insensitive, that test_same_seed_same_episode passes vacuously."""
    a = fingerprint(run_quiet(build_simulation(seed=11, legacy_global_rng=legacy)))
    b = fingerprint(run_quiet(build_simulation(seed=12, legacy_global_rng=legacy)))
    assert a["blue_traj"] != b["blue_traj"]


def test_construction_order_isolated_streams():
    """Two Simulations constructed back-to-back, then run in the reverse order,
    must still match their independently-run counterparts.

    This is the property that breaks when randomness flows through a global
    stream: construction seeds that stream, so interleaving construction and
    execution reorders every draw the *other* episode makes afterwards.
    """
    expected_a = fingerprint(run_quiet(build_simulation(seed=21, legacy_global_rng=False)))
    expected_b = fingerprint(run_quiet(build_simulation(seed=22, legacy_global_rng=False)))

    sim_a = build_simulation(seed=21, legacy_global_rng=False)
    sim_b = build_simulation(seed=22, legacy_global_rng=False)
    got_b = fingerprint(run_quiet(sim_b))
    got_a = fingerprint(run_quiet(sim_a))

    assert got_a == expected_a, "episode A perturbed by B's construction"
    assert got_b == expected_b, "episode B perturbed by A's construction"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known and deliberately preserved: legacy_global_rng=True keeps spawn/"
        "patrol/inert on the process-global np.random so episodes stay "
        "byte-identical to pre-refactor runs. Constructing a second Simulation "
        "re-seeds that stream and corrupts the first episode's patrol waypoints. "
        "Fixed by legacy_global_rng=False -- see "
        "test_construction_order_isolated_streams. strict=True so this flips to "
        "a failure the moment legacy mode is removed."
    ),
)
def test_construction_order_legacy_leaks():
    expected_a = fingerprint(run_quiet(build_simulation(seed=21, legacy_global_rng=True)))
    sim_a = build_simulation(seed=21, legacy_global_rng=True)
    build_simulation(seed=22, legacy_global_rng=True)  # re-seeds the global stream
    assert fingerprint(run_quiet(sim_a)) == expected_a


def test_new_streams_never_touch_the_legacy_stream():
    """The core guarantee the stream split buys: a NEW subsystem drawing from
    its own stream cannot perturb an existing episode's draws.

    This is what makes's paired clean/attack episodes valid -- adding
    an attack or a packet-loss draw must not renumber the patrol waypoints.
    """
    from simulation.rng import RngStreams

    baseline = fingerprint(run_quiet(build_simulation(seed=33, legacy_global_rng=True)))

    sim = build_simulation(seed=33, legacy_global_rng=True)
    # Simulate a future subsystem burning a lot of randomness mid-setup.
    scratch = RngStreams(seed=999)
    for _ in range(1000):
        scratch.comm.random()
        scratch.attack.random()
        scratch.scenario.random()

    assert fingerprint(run_quiet(sim)) == baseline


def test_streams_are_mutually_independent():
    from simulation.rng import RngStreams

    a = RngStreams(seed=4, legacy_global=False)
    b = RngStreams(seed=4, legacy_global=False)

    # Draining one stream must not change what another produces.
    for _ in range(100):
        a.comm.random()
    assert a.attack.random() == b.attack.random()
    assert a.scenario.random() == b.scenario.random()


def test_unknown_stream_name_raises():
    from simulation.rng import RngStreams

    with pytest.raises(AttributeError, match="No RNG stream named"):
        RngStreams(seed=1).telemetry
