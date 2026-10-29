"""'s information-isolation tests: an agent acts only on what it knows.

The two leaks these exist to hold closed are catalogued in
:

* **L1** -- ``solve_decentralized`` computed the nearest red from ground-truth
  ``red_states`` for DEFEND and RECON too, so agents closed on reds nobody had
  detected. Fog of war applied to the planner but not to behavior. This would
  have made Experiment 0 measure an artifact.
* **L2** -- ``target_override`` was filled with the red's true state *after* the
  planner chose an id, so an interceptor committed to a stale or falsified
  belief still tracked the real red perfectly. Without this closed, a
  false-location attack changes the allocation and nothing downstream, and plan
  Sec 19's "trajectory divergence" reads ~0 for reasons unrelated to the defense.

The decisive tests here are the two that drive the solver directly with
fabricated beliefs: they prove ground truth cannot influence the control, rather
than observing that it happens not to.
"""

import numpy as np
import pytest

from conftest import build_simulation, fingerprint, run_quiet


def _prepared(**overrides):
    """A Simulation with agents and solver built, ready to drive by hand."""
    sim = build_simulation(**overrides)
    sim.setup_agents()
    sim.solver.reset_warm_starts()
    sim.solver.set_fixed_adjacency_from_state(sim.x_current, sim.red_state)
    return sim


def _solve(sim, red_state, believed, roles=None):
    from planning.roles import BlueRole

    roles = roles or {i: BlueRole.DEFEND for i in range(sim.num_blues)}
    u, _, _, _ = sim.solver.solve_decentralized(
        x_current=sim.x_current,
        red_states=red_state,
        role_assignment=roles,
        believed_targets=believed,
    )
    return u


# --- L1 -------------------------------------------------------------------

def test_defend_ignores_ground_truth_when_belief_is_empty():
    """The decisive L1 test.

    Same blues, same empty beliefs, two wildly different *actual* red positions.
    If ground truth can still reach the controller, the two solves differ.
    """
    sim = _prepared(num_blues=4, num_reds=1)
    empty = {i: {} for i in range(sim.num_blues)}

    red_near = np.array([80.0, 80.0, 0.0, 0.0, 0.0])[: sim.red_nx]
    red_far = np.array([5.0, 5.0, 0.0, 0.0, 0.0])[: sim.red_nx]

    u_near = _solve(sim, red_near, empty)
    u_far = _solve(sim, red_far, empty)

    assert np.allclose(u_near, u_far, atol=1e-9), (
        "DEFEND control depends on the true red position despite an empty "
        "belief -- ground truth is leaking into belief."
    )


def test_defend_does_follow_a_believed_contact():
    """The control for the test above: with a belief, the target must matter.

    Without this, test_defend_ignores_ground_truth_when_belief_is_empty would
    pass vacuously if DEFEND had simply stopped responding to contacts at all.
    """
    sim = _prepared(num_blues=4, num_reds=1)
    here = {i: {0: np.array([70.0, 95.0, 0.0, 0.0, 0.0])[: sim.red_nx]} for i in range(4)}
    there = {i: {0: np.array([95.0, 70.0, 0.0, 0.0, 0.0])[: sim.red_nx]} for i in range(4)}
    true_red = sim.red_state[: sim.red_nx]

    assert not np.allclose(_solve(sim, true_red, here), _solve(sim, true_red, there)), (
        "DEFEND ignores its believed contact entirely"
    )


def test_empty_belief_sends_defenders_to_distinct_stations():
    """Stations must be per-agent, or 'hold the perimeter' degenerates into
    every defender converging on one point."""
    sim = _prepared(num_blues=4, num_reds=1)
    stations = sim.solver.defend_stations

    assert len(stations) == 4
    pts = np.array([stations[i] for i in range(4)])
    assert len({tuple(np.round(p, 6)) for p in pts}) == 4, f"stations coincide: {pts}"


# --- L2 -------------------------------------------------------------------

def test_neutralize_flies_to_the_believed_position_not_the_true_one():
    """The decisive L2 test, and the one the semantic-attack work depends on."""
    from planning.roles import BlueRole

    sim = _prepared(num_blues=4, num_reds=1)
    roles = {0: BlueRole.NEUTRALIZE, 1: BlueRole.DEFEND, 2: BlueRole.DEFEND, 3: BlueRole.DEFEND}
    true_red = np.array([50.0, 50.0, 0.0, 0.0, 0.0])[: sim.red_nx]

    honest = {i: {0: true_red.copy()} for i in range(4)}
    falsified = {i: {0: np.array([10.0, 90.0, 0.0, 0.0, 0.0])[: sim.red_nx]} for i in range(4)}

    u_honest = _solve(sim, true_red, honest, roles)
    u_false = _solve(sim, true_red, falsified, roles)

    nu = sim.blue.nu_single
    assert not np.allclose(u_honest[:nu], u_false[:nu], atol=1e-6), (
        "A falsified belief did not move the interceptor -- ground truth is leaking into pursuit, "
        "and any semantic-attack result would be measuring nothing."
    )


# --- belief store isolation ----------------------------------------------

def test_local_mode_belief_contains_only_own_observations():
    """At zero communication an agent's belief must be a function only of its
    own observation history."""
    sim = run_quiet(
        build_simulation(seed=130, num_blues=4, num_reds=2, sim_steps=200, belief_mode="local")
    )
    seen_by = {i: set() for i in range(sim.num_blues)}
    for step in sim.recorder.steps:
        for k, reds in step["per_blue_detected_reds"].items():
            seen_by[int(k)].update(reds)

    for i, store in sim.belief_stores.items():
        assert set(store.targets) <= seen_by[i], (
            f"blue {i} believes in contacts {set(store.targets) - seen_by[i]} it "
            "never personally detected, at zero communication"
        )


def _propagation_events(sim):
    """Steps at which some agent believed in a contact it had not itself
    detected *up to that point*.

    Comparing against cumulative end-of-episode observations is far too weak:
    over a long enough episode every agent eventually sees every red on its own,
    so the difference vanishes regardless of whether sharing works.
    """
    seen_so_far = {i: set() for i in range(sim.num_blues)}
    events = 0
    for step in sim.recorder.steps:
        for k, reds in step["per_blue_detected_reds"].items():
            seen_so_far[int(k)].update(reds)
        for agent, per in step["beliefs"].items():
            believed = {int(c) for c in per}
            if believed - seen_so_far[int(agent)]:
                events += 1
    return events


def test_shared_mode_propagates_beyond_own_observations():
    """The control: sharing must actually share, or 'local' is trivially
    isolated because nothing ever propagates anywhere."""
    shared = run_quiet(
        build_simulation(seed=130, num_blues=4, num_reds=2, sim_steps=200, belief_mode="shared")
    )
    assert _propagation_events(shared) > 0, (
        "no agent ever believed in a contact it had not personally detected -- "
        "sharing is not propagating"
    )


def test_local_mode_never_propagates():
    """The mirror of the above, and the actual zero-communication guarantee."""
    local = run_quiet(
        build_simulation(seed=130, num_blues=4, num_reds=2, sim_steps=200, belief_mode="local")
    )
    assert _propagation_events(local) == 0, (
        "an agent believed in a contact it never detected, at zero communication"
    )


def test_belief_modes_produce_different_episodes():
    """Information structure must change behavior, or Experiment 0 cannot
    possibly measure anything."""
    gt = fingerprint(run_quiet(build_simulation(seed=130, num_blues=4, num_reds=2,
                                                sim_steps=200, belief_mode="ground_truth")))
    shared = fingerprint(run_quiet(build_simulation(seed=130, num_blues=4, num_reds=2,
                                                    sim_steps=200, belief_mode="shared")))
    local = fingerprint(run_quiet(build_simulation(seed=130, num_blues=4, num_reds=2,
                                                   sim_steps=200, belief_mode="local")))
    assert gt["blue_traj"] != shared["blue_traj"], "shared belief == omniscience"
    assert shared["blue_traj"] != local["blue_traj"], "sharing has no effect on behavior"


def test_relaying_never_rejuvenates_an_observation():
    """: a sighting relayed later is still as old as the sighting."""
    from perception.belief import Provenance

    p = Provenance(origin_agent=0, observed_at=5, received_at=5)
    relayed = p.relayed_by(relay_agent=1, received_at=20)
    assert relayed.observed_at == 5
    assert relayed.origin_agent == 0
    assert relayed.received_at == 20


def test_merge_is_idempotent_and_order_independent():
    """Duplicate delivery must be harmless and reordering must not regress a
    track -- the lattice properties asserts of the comms layer."""
    import numpy as np

    from perception.belief.merge import merge_target_dicts
    from perception.belief.records import Provenance, TargetBelief

    def belief(cid, at):
        return TargetBelief(cid, np.array([float(at), 0.0, 0.0, 0.0]),
                            Provenance(origin_agent=0, observed_at=at, received_at=at))

    old, new = {1: belief(1, 5)}, {1: belief(1, 9)}
    assert merge_target_dicts(old, new)[1].provenance.observed_at == 9
    assert merge_target_dicts(new, old)[1].provenance.observed_at == 9, "reorder regressed"
    twice = merge_target_dicts(merge_target_dicts(old, new), new)
    assert twice[1].provenance.observed_at == 9, "duplicate delivery not idempotent"


def test_invalid_belief_mode_raises():
    with pytest.raises(ValueError, match="Unsupported belief_mode"):
        build_simulation(belief_mode="telepathy")
