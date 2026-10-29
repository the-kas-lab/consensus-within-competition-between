"""A relayed observation is predicted to now exactly once, by the belief layer,
and the live aim point and pursued-contact heading consume that
prediction without adding any communication-delay compensation of their own.

Relays preserve the sensor timestamp; pursuit and
heading consume the already-predicted belief; no hidden delay parameter, no
true state. The common simulation clock is an explicit timing assumption: every
agent shares one integer step, so ``now - observed_at`` is exact.

**No episode runs here.** A networked Simulation is constructed and
``setup_agents()``'d, never ``run()``. Its belief stores and Network are built
the way ``run()`` builds them, and the shipped communication round
(``_exchange_beliefs_over_network``) is called step by step on a three-agent
chain, so a sighting reaches agent 2 only through agent 1.
"""

import numpy as np
import pytest

from comms import DiskLinkModel, Network
from conftest import build_simulation, capture_fd_output
from perception.belief import LocalBeliefStore
from planning.planner import PlannerOutput
from planning.roles import BlueRole

DT = 0.1
SIGHTING = np.array([40.0, 60.0, 1.5, -0.5, 0.3])  # red 1, seen by agent 0 at step 10


def _chain():
    """Agents 0 - 1 - 2 in a line, 8 units apart, r_comm 10: 0 and 2 cannot hear
    each other, so 2 learns only by relay through 1."""
    sim = build_simulation(num_blues=3, num_reds=2, belief_mode="networked",
                           r_comm=10.0, comm_interval_seconds=0.1,
                           neutralize_aim="live", neutralize_heading="pursued_contact")
    sim.setup_agents()
    for i, x in enumerate((20.0, 28.0, 36.0)):
        sim.x_current[i * sim.blue_nx : i * sim.blue_nx + 5] = [x, 20.0, 0.0, 0.0, 0.0]
    sim.belief_stores = {i: LocalBeliefStore(i, n_regions=len(sim.coverage_tracker.centers))
                         for i in range(sim.num_blues)}
    sim.network = Network(seed=sim.seed, link_model=DiskLinkModel(r_comm=sim.r_comm),
                          loss_model=None)
    sim.comm_stats = sim._fresh_comm_stats()
    sim._reset_command_state()
    sim.solver.set_fixed_adjacency_from_state(sim.x_current, sim.red_state)
    return sim


def _relay_to_agent_2(sim):
    """Agent 0 sees red 1 at step 10; rounds at 10, 11, 12 move it two hops."""
    sim.belief_stores[0].observe_target(1, SIGHTING, 10)
    arrival = None
    for step in (10, 11, 12):
        with capture_fd_output():
            sim._exchange_beliefs_over_network(step, [0, 1, 2])
        if arrival is None and 1 in sim.belief_stores[2].targets:
            arrival = step
    return arrival


def _expected(now):
    out = SIGHTING.copy()
    out[0:2] = SIGHTING[0:2] + (now - 10) * DT * SIGHTING[2:4]
    return out


def test_a_two_hop_relay_keeps_the_original_sensor_timestamp():
    sim = _chain()
    arrival = _relay_to_agent_2(sim)
    assert arrival == 12, "the chain should take two rounds to reach agent 2"
    record = sim.belief_stores[2].targets[1]
    assert record.provenance.observed_at == 10
    assert record.provenance.origin_agent == 0
    assert np.array_equal(record.state, SIGHTING), "relaying altered the observed state"


@pytest.mark.parametrize("now", [12, 17])
def test_live_aim_and_pursued_heading_use_the_prediction_to_now_exactly_once(now):
    sim = _chain()
    _relay_to_agent_2(sim)
    sim.red_outcomes = ["active"] * sim.num_reds
    plan = PlannerOutput(roles={2: BlueRole.NEUTRALIZE}, target_red_ids={2: 1})
    sim._apply_assignment(plan, 12, sim._believed_targets(12))

    believed = sim._believed_targets(now)
    expected = _expected(now)
    assert np.allclose(believed[2][1], expected, atol=1e-12), "belief not predicted from observed_at"

    aim = sim._neutralize_aim_points(now, believed)[2]
    assert sim.neutralize_aim_sources == {2: "belief"}
    assert np.allclose(aim, expected[0:4], atol=1e-12)

    calls = []
    controller = sim.solver.neutralize_controller
    original = controller.plan

    def spy(own_state, red_state, *args, **kwargs):
        calls.append((np.asarray(red_state, float).copy(), kwargs.get("bearing_target")))
        return original(own_state, red_state, *args, **kwargs)

    controller.plan = spy
    with capture_fd_output():
        sim.solver.solve_decentralized(
            x_current=sim.x_current, red_states=sim.red_state,
            role_assignment={0: BlueRole.DEFEND, 1: BlueRole.DEFEND, 2: BlueRole.NEUTRALIZE},
            target_override={2: aim}, target_override_ids=dict(sim.target_override_ids),
            believed_targets=believed, own_detected_reds={},
            pursued_belief_keys={b: k for b, (_, k) in sim.target_override_applied.items()
                                 if k is not None},
        )
    (red_state, bearing), = calls
    assert np.allclose(red_state[0:4], expected[0:4], atol=1e-12), "aim predicted twice or not at all"
    assert np.allclose(bearing, expected[0:2], atol=1e-12), "heading predicted twice or not at all"
    assert sim.solver.bearing_sources[2] == "pursued_belief"


def test_the_relay_delay_is_not_compensated_a_second_time():
    """The two rounds of relay latency are already inside `now - observed_at`.
    Using `received_at` or the arrival round instead would aim short by
    exactly that latency; adding it again would aim long."""
    sim = _chain()
    _relay_to_agent_2(sim)
    now = 14
    got = np.asarray(sim._believed_targets(now)[2][1])[0:2]
    short = SIGHTING[0:2] + (now - 12) * DT * SIGHTING[2:4]
    long_ = SIGHTING[0:2] + (now - 10 + 2) * DT * SIGHTING[2:4]
    assert np.allclose(got, _expected(now)[0:2], atol=1e-12)
    assert not np.allclose(got, short) and not np.allclose(got, long_)
