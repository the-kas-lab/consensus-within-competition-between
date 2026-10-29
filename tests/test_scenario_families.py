"""Scenario families: ``symmetric`` and ``multi_bearing``.

Two things need pinning here, and they pull in opposite directions:

* ``multi_bearing`` must actually *do something* -- a bearing family that does
  not widen the approach arc is a no-op dressed up as an experimental
  condition, and the whole point of the scenario-geometry rule is to widen the information bracket.
* ``symmetric`` must not move **at all**. It is retained precisely so that
  "sharing only helps under condition X" stays a statement the paper can make,
  which requires the old and new numbers to be comparable. The golden episode
  baselines cover the default path; what is asserted below is the stronger
  claim that merely *adding* the family left the default draw sequence
  untouched, byte for byte.

The geometry these tests encode, measured before the family was
written: the defended region sits 17.5 units from the NE arena walls against a
red standoff radius of ~90.5, so only the bearing arc 206deg-244deg can hold a
red. ``multi_bearing`` widens within that arc; it is not omnidirectional, and
these tests are written to fail if anyone later assumes it is.

**On cost.** Constructing a ``Simulation`` builds a CasADi solver and takes
~0.26 s, so a test that wants thousands of spawn draws cannot get them that
way. The distributional contracts of ``_randomize_red_bearings`` (containment,
separation, independence, radius preservation) are therefore exercised as unit
tests against one shared instance, and full construction is reserved for the
handful of assertions that genuinely need the integrated path.
"""

import numpy as np
import pytest

from conftest import build_simulation
from simulation.testbed import SCENARIO_FAMILIES, Simulation

# Blue's sensor reach. Two reds further apart than this cannot both be seen by
# one blue, which is the disagreement the scenario-geometry rule is trying to manufacture.
FOV_RANGE = 35.0
MIN_RED_SPAWN_DIST = 4.0


# ---------------------------------------------------------------------------
# Shared fixtures -- one construction each, reused
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim_mb():
    return build_simulation(num_reds=2, seed=7, scenario_family="multi_bearing")


@pytest.fixture(scope="module")
def symmetric_box_draws(sim_mb):
    """Positions drawn from exactly the box ``symmetric`` spawns reds in.

    Reproduced here rather than harvested from ``setup_agents`` so that a few
    thousand of them cost nothing. The box is ``arena_size - defense_center``
    +- ``0.10 * arena_size`` (testbed.setup_agents).
    """
    center = sim_mb.arena_size - np.asarray(sim_mb.defense_center, dtype=float)
    half = 0.10 * sim_mb.arena_size
    rng = np.random.default_rng(20260921)
    return rng.uniform(center - half, center + half, size=(2000, 2))


def _red_positions(sim):
    sim.setup_agents()
    return sim.red_state.reshape(sim.num_reds, -1)[:, :2].copy()


def _bearings_deg(sim, positions):
    radial = np.asarray(positions, dtype=float) - np.asarray(
        sim.defense_center, dtype=float
    )
    return np.degrees(np.arctan2(radial[:, 1], radial[:, 0])) % 360.0


def _radii(sim, positions):
    return np.linalg.norm(
        np.asarray(positions, dtype=float) - np.asarray(sim.defense_center, dtype=float),
        axis=1,
    )


# ---------------------------------------------------------------------------
# symmetric is untouched
# ---------------------------------------------------------------------------


def test_symmetric_is_the_default():
    assert build_simulation().scenario_family == "symmetric"


def test_known_families_are_exactly_these():
    assert SCENARIO_FAMILIES == ("symmetric", "multi_bearing")


@pytest.mark.parametrize("seed", [7, 130])
def test_symmetric_spawns_are_unchanged_by_the_new_family(seed):
    """Adding ``multi_bearing`` must not perturb ``symmetric``'s draw sequence.

    Passing the family explicitly and leaving it at its default must produce
    identical spawns -- if the new code drew from any stream on the default
    path, even once and even discarding the value, this would diverge.
    """
    default = _red_positions(build_simulation(num_reds=2, seed=seed))
    explicit = _red_positions(
        build_simulation(num_reds=2, seed=seed, scenario_family="symmetric")
    )
    np.testing.assert_array_equal(default, explicit)


def test_symmetric_ignores_the_offset_parameter():
    """The knob belongs to one family. Setting it must not reach the other --
    otherwise every existing result silently depends on its default."""
    base = _red_positions(build_simulation(num_reds=2, seed=7))
    wide = _red_positions(
        build_simulation(num_reds=2, seed=7, multi_bearing_max_offset_deg=45.0)
    )
    np.testing.assert_array_equal(base, wide)


def test_symmetric_approach_arc_is_narrow(sim_mb, symmetric_box_draws):
    """Records the measurement that motivates the scenario-geometry rule at all: roughly a single
    corridor, ~18deg wide at the extremes. If this widens, ``multi_bearing``'s
    reason for existing has changed and the pilot needs rethinking, not just
    rerunning."""
    bearings = _bearings_deg(sim_mb, symmetric_box_draws)
    assert 5.0 < (bearings.max() - bearings.min()) < 25.0


# ---------------------------------------------------------------------------
# multi_bearing does what it claims -- distributional contracts
# ---------------------------------------------------------------------------


def test_multi_bearing_widens_the_approach_arc(sim_mb, symmetric_box_draws):
    """The exit criterion for the family: measurably wider than ``symmetric``,
    while staying inside the arc the geometry actually permits."""
    sym = _bearings_deg(sim_mb, symmetric_box_draws)
    pairs = symmetric_box_draws.reshape(-1, 2, 2)
    rotated = np.concatenate(
        [sim_mb._randomize_red_bearings(list(p), MIN_RED_SPAWN_DIST) for p in pairs]
    )
    mb = _bearings_deg(sim_mb, rotated)
    sym_spread = sym.max() - sym.min()
    mb_spread = mb.max() - mb.min()
    assert mb_spread > 1.5 * sym_spread, (
        f"multi_bearing spread {mb_spread:.1f} deg vs symmetric {sym_spread:.1f} deg"
    )
    # Stays inside the measured feasible arc; if this trips, the geometry
    # analysis recorded in 03-open-decisions.md the scenario-geometry rule is stale.
    assert 200.0 < mb.min() and mb.max() < 250.0


def test_multi_bearing_preserves_standoff_radius_exactly(sim_mb, symmetric_box_draws):
    """The manipulation is bearing and *only* bearing. Radius is carried over
    untouched from the ``symmetric`` draw, so any change in outcomes cannot be
    attributed to reds starting closer or further away."""
    pairs = symmetric_box_draws.reshape(-1, 2, 2)
    for p in pairs[:200]:
        rotated = sim_mb._randomize_red_bearings(list(p), MIN_RED_SPAWN_DIST)
        np.testing.assert_allclose(
            _radii(sim_mb, p), _radii(sim_mb, rotated), rtol=1e-12
        )


def test_multi_bearing_separates_reds_beyond_sensor_range(sim_mb, symmetric_box_draws):
    """The mechanism the scenario-geometry rule names: blue A sees red 1 while blue B sees red 2. That
    requires the reds to be further apart than one blue can see, at least
    sometimes. A family that never achieves it would be testing nothing."""
    pairs = symmetric_box_draws.reshape(-1, 2, 2)
    seps = [
        float(np.linalg.norm(np.subtract(*sim_mb._randomize_red_bearings(list(p), MIN_RED_SPAWN_DIST))))
        for p in pairs
    ]
    assert max(seps) > FOV_RANGE, f"widest red-red separation only {max(seps):.1f}"


def test_multi_bearing_spawns_inside_the_arena(sim_mb, symmetric_box_draws):
    """Rejection sampling's contract. A red spawned outside the arena would be
    resolved as 'escaped' on step 0 and hand blue a free point."""
    pairs = symmetric_box_draws.reshape(-1, 2, 2)
    for p in pairs:
        r = np.asarray(sim_mb._randomize_red_bearings(list(p), MIN_RED_SPAWN_DIST))
        assert np.all(r >= sim_mb.arena_min) and np.all(r <= sim_mb.arena_max)


def test_multi_bearing_respects_minimum_red_separation(sim_mb, symmetric_box_draws):
    """Rotation can bring two reds together; the family must re-enforce the
    separation ``symmetric`` guarantees, not just inherit it."""
    triples = symmetric_box_draws[: (len(symmetric_box_draws) // 3) * 3].reshape(-1, 3, 2)
    for p in triples:
        r = np.asarray(sim_mb._randomize_red_bearings(list(p), MIN_RED_SPAWN_DIST))
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                assert np.linalg.norm(r[i] - r[j]) >= MIN_RED_SPAWN_DIST


def test_multi_bearing_offsets_are_independent_per_red(sim_mb, symmetric_box_draws):
    """A single per-episode rotation would move both reds together and produce
    shared ignorance rather than disagreement. Assert the bearings genuinely
    decouple: the *change* in bearing must differ between the two reds."""
    pairs = symmetric_box_draws.reshape(-1, 2, 2)
    deltas = []
    for p in pairs[:400]:
        r = sim_mb._randomize_red_bearings(list(p), MIN_RED_SPAWN_DIST)
        d = _bearings_deg(sim_mb, r) - _bearings_deg(sim_mb, p)
        deltas.append(abs(d[0] - d[1]))
    assert max(deltas) > 1.0, "per-red bearing offsets look shared, not independent"
    assert np.mean(deltas) > 0.1


# ---------------------------------------------------------------------------
# multi_bearing through the integrated spawn path
# ---------------------------------------------------------------------------


def test_multi_bearing_changes_the_spawn():
    # Built-then-spawned one at a time; see the note on the construction-order
    # RNG leak in test_multi_bearing_preserves_radius_through_setup_agents.
    sym = _red_positions(build_simulation(num_reds=2, seed=7, scenario_family="symmetric"))
    mb = _red_positions(build_simulation(num_reds=2, seed=7, scenario_family="multi_bearing"))
    assert not np.allclose(sym, mb)


def test_multi_bearing_preserves_radius_through_setup_agents():
    """The unit test above pins the method; this pins that ``setup_agents``
    actually calls it on the positions, before headings are derived.

    Each simulation is built and spawned before the next is built. In legacy
    RNG mode the ``spawn`` stream *is* the process-global ``np.random``, which
    ``__init__`` re-seeds -- so building both up front and spawning afterwards
    would have the second episode draw from a stream the first had already
    advanced. That is the construction-order leak simulation/rng.py documents,
    and it silently invalidates any paired comparison written the naive way.
    """
    sym = build_simulation(num_reds=2, seed=42, scenario_family="symmetric")
    sym_radii = _radii(sym, _red_positions(sym))
    mb = build_simulation(num_reds=2, seed=42, scenario_family="multi_bearing")
    mb_radii = _radii(mb, _red_positions(mb))
    np.testing.assert_allclose(sym_radii, mb_radii, rtol=1e-12)


def test_red_spawn_heading_follows_the_rotated_position():
    """Reds spawn facing the defended region. If the rotation were applied
    after the state was built, every red would face the nominal 225deg axis
    instead -- a second, unintended manipulation."""
    mb = build_simulation(num_reds=2, seed=42, scenario_family="multi_bearing")
    positions = _red_positions(mb)
    state = mb.red_state.reshape(mb.num_reds, -1)
    assert state.shape[1] == 5, "heading is always on (testbed.py)"
    for pos, row in zip(positions, state):
        to_target = np.asarray(mb.defense_center, dtype=float) - pos
        assert row[4] == pytest.approx(float(np.arctan2(to_target[1], to_target[0])))


def test_zero_offset_degenerates_to_symmetric():
    """Continuity check: the family is a widening of ``symmetric``, so a zero
    offset must land exactly back on it rather than on some third geometry."""
    sym = _red_positions(build_simulation(num_reds=2, seed=7))
    mb0 = _red_positions(
        build_simulation(
            num_reds=2,
            seed=7,
            scenario_family="multi_bearing",
            multi_bearing_max_offset_deg=0.0,
        )
    )
    np.testing.assert_allclose(sym, mb0, atol=1e-12)


# ---------------------------------------------------------------------------
# configuration is strict
# ---------------------------------------------------------------------------


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="Unknown scenario_family"):
        build_simulation(scenario_family="omnidirectional")


def test_negative_offset_is_rejected():
    with pytest.raises(ValueError, match="multi_bearing_max_offset_deg"):
        build_simulation(multi_bearing_max_offset_deg=-1.0)


def test_exhausted_rejection_budget_fails_loudly(sim_mb, symmetric_box_draws):
    """Better to raise than to silently narrow the realized distribution: a
    family that quietly stopped randomizing would look like a null result.

    Driven through the *separation* constraint, because that is the branch that
    can actually exhaust. A zero offset always satisfies the arena constraint
    (it is the ``symmetric`` position, which is inside the arena by
    construction), so widening ``multi_bearing_max_offset_deg`` alone never
    makes placement impossible -- it only makes it slower. An unsatisfiable
    separation does.
    """
    positions = list(symmetric_box_draws[:2])
    with pytest.raises(RuntimeError, match="could not place red 1"):
        sim_mb._randomize_red_bearings(positions, min_red_spawn_dist=500.0)


def test_family_is_recorded_in_the_manifest(sim_mb):
    """The episode manifest is how a result is traced back to its geometry."""
    assert sim_mb._init_args["scenario_family"] == "multi_bearing"
    assert "multi_bearing_max_offset_deg" in sim_mb._init_args


# ---------------------------------------------------------------------------
# Attrition accounting
# ---------------------------------------------------------------------------


def test_attrition_is_reported_in_the_episode_outcome():
    """Attrition is a first-class outcome, not something each analysis script
    re-derives from the ``blue_disabled`` mask."""
    sim = build_simulation(num_blues=2, num_reds=1, seed=7)
    sim.setup_agents()
    outcome = sim.episode_outcome()
    assert outcome["n_blues_disabled"] == 0
    assert outcome["attrition_rate"] == 0.0


def test_attrition_equals_neutralizations_by_construction():
    """The identity that Correction C2 recorded as an empirical 30/30
    regularity is actually structural: ``_resolve_red_neutralized`` is the only
    site that sets ``blue_disabled``, and it sets one entry per kill.

    Asserted directly rather than trusted, because if a second disable site is
    ever added (collision damage, fuel, a blue-blue contact rule) this identity
    breaks and every report that treats kills and losses as one number becomes
    wrong. Fail here rather than in the analysis.
    """
    sim = build_simulation(num_blues=3, num_reds=2, seed=11)
    sim.setup_agents()
    sim._resolve_red_neutralized(step=5, red_idx=0, blue_idx=1, distance=0.1)
    outcome = sim.episode_outcome()
    assert outcome["num_neutralized"] == outcome["n_blues_disabled"] == 1
    assert outcome["attrition_rate"] == pytest.approx(1 / 3)

    sim._resolve_red_neutralized(step=6, red_idx=1, blue_idx=2, distance=0.1)
    outcome = sim.episode_outcome()
    assert outcome["num_neutralized"] == outcome["n_blues_disabled"] == 2
    assert outcome["attrition_rate"] == pytest.approx(2 / 3)


def test_attrition_survives_a_full_episode():
    """The identity must hold through the real resolution path, not just when
    the helper is called by hand."""
    from conftest import run_quiet

    for seed in (3, 11):
        sim = run_quiet(build_simulation(num_blues=3, num_reds=2, seed=seed, sim_steps=60))
        outcome = sim.episode_outcome()
        assert outcome["n_blues_disabled"] == outcome["num_neutralized"]
        assert outcome["attrition_rate"] == pytest.approx(
            outcome["n_blues_disabled"] / sim.num_blues
        )
