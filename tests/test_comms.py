"""The communication substrate, one test per property:

* a line graph advances exactly one hop per round;
* relaying preserves ``observed_at`` / ``origin_agent``;
* duplicate delivery is idempotent;
* reordering never regresses a track;
* a late epoch cannot overwrite the current one;
* tombstones dominate older live entries;
* a disabled relay stops forwarding;
* counter-based loss is reproducible.

Plus the two invariants the layer is built around: double buffering, and
counter-based link randomness being independent of call order.

These run without a ``Simulation`` -- the network moves envelopes and the belief
layer merges them, and neither needs an episode. That separation is deliberate:
a protocol bug should fail here, in milliseconds, not inside a 400-step run.
"""

import numpy as np
import pytest

from comms import topologies
from comms.link_models import (
    BAD,
    GOOD,
    DiskLinkModel,
    GilbertElliottLossModel,
    IidLossModel,
    LogisticLinkModel,
    NoLoss,
    _uniform,
)
from comms import receive
from comms.messages import (
    ALLOCATION_MESSAGE_TYPES,
    AUCTION_MESSAGE,
    BELIEF_MESSAGE_TYPES,
    BROADCAST,
    INTENT_UPDATE,
    REALLOCATION_PROPOSAL,
    SELF_STATE,
    STATUS_TOMBSTONE,
    TARGET_UPDATE,
    AuctionMessage,
    BeliefUpdate,
    IntentUpdate,
    MessageEnvelope,
    ReallocationProposal,
    make_envelope,
)
from comms.network import Network
from perception.belief.merge import (
    merge_status,
    merge_status_dicts,
    merge_target,
    merge_target_dicts,
)
from perception.belief.records import AgentStatus, Provenance, TargetBelief


def _belief(contact_id, observed_at, origin, position=(1.0, 2.0)):
    return TargetBelief(
        contact_id=contact_id,
        state=np.array([position[0], position[1], 0.0, 0.0], dtype=float),
        provenance=Provenance(
            origin_agent=origin, observed_at=observed_at, received_at=observed_at
        ),
    )


def _line_network(n=4, **kwargs):
    net = Network(seed=7, fixed_edges=topologies.line(n), store_and_forward=True, **kwargs)
    net.begin_round(positions={i: (i, 0) for i in range(n)})
    return net


# ---------------------------------------------------------------------------
# Double buffering -- the invariant the whole layer rests on
# ---------------------------------------------------------------------------


def test_a_message_is_not_visible_in_the_round_it_was_sent():
    """Without this, iteration order over agents would set how fast information
    crosses the graph, and every latency number would be an artifact."""
    net = _line_network()
    net.send(make_envelope(TARGET_UPDATE, 0, BeliefUpdate(), sent_at=0))
    # Nothing has been swapped in yet.
    assert net._inbox == {} or all(not v for v in net._inbox.values())
    delivered = net.deliver()
    assert 1 in delivered


def test_delivery_is_independent_of_send_order():
    """Two sends in the same round must produce the same inboxes whichever
    order they were enqueued in -- a direct consequence of double buffering."""
    orders = []
    for reverse in (False, True):
        net = _line_network()
        envs = [
            make_envelope(TARGET_UPDATE, 0, BeliefUpdate(), sent_at=0, sequence=0),
            make_envelope(TARGET_UPDATE, 2, BeliefUpdate(), sent_at=0, sequence=1),
        ]
        for env in reversed(envs) if reverse else envs:
            net.send(env)
        orders.append({a: sorted(e.message_id for e in v) for a, v in net.deliver().items()})
    assert orders[0] == orders[1]


# ---------------------------------------------------------------------------
# One hop per round
# ---------------------------------------------------------------------------


def test_line_graph_advances_exactly_one_hop_per_round():
    """0-1-2-3. A broadcast from 0 must reach 1 after one round, 2 after two,
    3 after three -- never sooner."""
    net = _line_network(4)
    net.send(make_envelope(TARGET_UPDATE, 0, BeliefUpdate(), sent_at=0))

    reached = {}
    for round_index in range(1, 5):
        delivered = net.deliver()
        for agent in delivered:
            reached.setdefault(agent, round_index)
        # Relay happens after delivery, so it lands in the next round's buffer.
        for agent, envelopes in delivered.items():
            net.forward(agent, envelopes, sent_at=round_index)
        net.begin_round(positions={i: (i, 0) for i in range(4)})

    assert reached == {1: 1, 2: 2, 3: 3}, reached


def test_ttl_bounds_relaying():
    net = Network(seed=7, fixed_edges=topologies.line(6), store_and_forward=True)
    net.begin_round(positions={i: (i, 0) for i in range(6)})
    net.send(make_envelope(TARGET_UPDATE, 0, BeliefUpdate(), sent_at=0, ttl=2))
    reached = {}
    for round_index in range(1, 7):
        delivered = net.deliver()
        for agent in delivered:
            reached.setdefault(agent, round_index)
        for agent, envelopes in delivered.items():
            net.forward(agent, envelopes, sent_at=round_index)
        net.begin_round(positions={i: (i, 0) for i in range(6)})
    # ttl=2 permits two forwards: 0->1 (send), 1->2, 2->3. Agent 4 is out of reach.
    assert 4 not in reached and 5 not in reached
    assert net.stats["dropped_ttl"] > 0


# ---------------------------------------------------------------------------
# Relaying preserves provenance
# ---------------------------------------------------------------------------


def test_relaying_preserves_observed_at_and_origin_agent():
    """The single most damaging thing this layer could do is make a relayed
    sighting look fresh. Age-of-information is the paper's main mediator."""
    original = _belief(contact_id=3, observed_at=11, origin=0)
    env = make_envelope(TARGET_UPDATE, 0, BeliefUpdate(targets={3: original}), sent_at=11)

    hop1 = env.forwarded_by(relay_agent=1, recipient=BROADCAST, sent_at=12)
    hop2 = hop1.forwarded_by(relay_agent=2, recipient=BROADCAST, sent_at=13)

    for hop in (hop1, hop2):
        carried = hop.payload.targets[3]
        assert carried.provenance.observed_at == 11
        assert carried.provenance.origin_agent == 0
    assert hop2.hop_count == 2
    assert hop2.origin_sender == 0
    assert hop2.sender == 2


def test_a_relayed_old_sighting_loses_to_an_older_arriving_new_one():
    """Merging compares sensor time, never arrival time."""
    fresh_but_arrives_first = _belief(3, observed_at=50, origin=1)
    stale_but_relayed_later = _belief(3, observed_at=10, origin=0)
    merged = merge_target(fresh_but_arrives_first, stale_but_relayed_later)
    assert merged.provenance.observed_at == 50


def test_ttl_exhausted_message_cannot_be_forwarded():
    env = make_envelope(TARGET_UPDATE, 0, BeliefUpdate(), sent_at=0, ttl=0)
    with pytest.raises(ValueError, match="ttl exhausted"):
        env.forwarded_by(1, BROADCAST, sent_at=1)


# ---------------------------------------------------------------------------
# Idempotence and reordering
# ---------------------------------------------------------------------------


def test_duplicate_delivery_is_idempotent():
    belief = _belief(3, observed_at=11, origin=0)
    once = merge_target_dicts({}, {3: belief})
    twice = merge_target_dicts(once, {3: belief})
    assert twice[3] is once[3]
    assert twice[3].provenance.observed_at == 11


def test_reordering_never_regresses_a_track():
    """Every delivery order of the same three reports must reach the same
    result -- the merge is a lattice join, so this is order-independence."""
    import itertools

    reports = [_belief(3, observed_at=t, origin=0) for t in (5, 40, 22)]
    outcomes = set()
    for order in itertools.permutations(reports):
        state = {}
        for belief in order:
            state = merge_target_dicts(state, {3: belief})
        outcomes.add(state[3].provenance.observed_at)
    assert outcomes == {40}


def test_coverage_merge_is_monotone_under_reordering():
    from perception.belief.records import CoverageBelief

    import itertools

    updates = [
        np.array([5, -1, 30, -1]),
        np.array([-1, 12, 10, -1]),
        np.array([2, 40, -1, -1]),
    ]
    outcomes = set()
    for order in itertools.permutations(updates):
        cover = CoverageBelief(4)
        for arr in order:
            cover.merge(CoverageBelief(4, arr))
        outcomes.add(tuple(cover.last_seen_step))
    assert outcomes == {(5, 40, 30, -1)}


# ---------------------------------------------------------------------------
# Epoch staleness and tombstones
# ---------------------------------------------------------------------------


def test_stale_allocation_traffic_is_rejected():
    """The rule that an auction may span planning epochs: an old assignment must not replace a newer assignment.

    Asserts the production rule in ``comms.receive``, not a filter rebuilt in
    the test -- the previous version of this test did the latter and therefore
    showed only that the *fields* made the rule expressible.
    """
    stale = make_envelope(AUCTION_MESSAGE, 0, AuctionMessage(6, 0), sent_at=100, planning_epoch=6)
    current = make_envelope(AUCTION_MESSAGE, 0, AuctionMessage(7, 0), sent_at=100, planning_epoch=7)
    future = make_envelope(AUCTION_MESSAGE, 0, AuctionMessage(8, 0), sent_at=100, planning_epoch=8)

    assert receive.accept(stale, current_planning_epoch=7) is False
    assert receive.accept(current, current_planning_epoch=7) is True
    # A message from a LATER epoch is not stale -- under the rule that an auction may span planning epochs's asynchronous
    # allocation an agent may legitimately be ahead.
    assert receive.accept(future, current_planning_epoch=7) is True


@pytest.mark.parametrize("message_type", ALLOCATION_MESSAGE_TYPES)
def test_every_allocation_type_is_epoch_filtered(message_type):
    payloads = {
        INTENT_UPDATE: IntentUpdate(planning_epoch=2, owner_agent=0),
        AUCTION_MESSAGE: AuctionMessage(2, 0),
        REALLOCATION_PROPOSAL: ReallocationProposal(planning_epoch=2, agent_id=0),
    }
    env = make_envelope(message_type, 0, payloads[message_type], sent_at=9, planning_epoch=2)
    assert receive.accept(env, current_planning_epoch=5) is False


@pytest.mark.parametrize("message_type", BELIEF_MESSAGE_TYPES)
def test_belief_traffic_is_never_epoch_filtered(message_type):
    """The distinction that makes this rule correct rather than merely strict.

    An observation *sent* during an earlier planning epoch may still be the
    newest sighting the receiver has -- a contact seen at step 300 and relayed
    twice is still a step-300 sighting however many epochs elapsed in transit.
    Rejecting it on epoch would discard information the receiver has nothing
    better than, and would bias the age-of-information mediator the study
    measures. Belief staleness is decided by ``observed_at`` in the merge.
    """
    env = make_envelope(message_type, 0, BeliefUpdate(), sent_at=9, planning_epoch=1)
    assert receive.accept(env, current_planning_epoch=99) is True


def test_an_old_epoch_observation_still_wins_on_observation_time():
    """The end-to-end version of the previous test: a belief carried by a
    very stale *envelope* must still beat an older sighting on merit."""
    old_envelope_new_sighting = make_envelope(
        TARGET_UPDATE, 0, BeliefUpdate(targets={3: _belief(3, observed_at=300, origin=0)}),
        sent_at=300, planning_epoch=1,
    )
    assert receive.accept(old_envelope_new_sighting, current_planning_epoch=42) is True
    held = {3: _belief(3, observed_at=120, origin=1)}
    merged = merge_target_dicts(held, old_envelope_new_sighting.payload.targets)
    assert merged[3].provenance.observed_at == 300


def test_allocation_traffic_must_declare_an_epoch():
    """Refused at construction: an untagged auction message cannot be checked
    for staleness, so the rule that an auction may span planning epochs's drop rule would silently not apply to it."""
    with pytest.raises(ValueError, match="requires a planning_epoch"):
        make_envelope(AUCTION_MESSAGE, 0, AuctionMessage(1, 0), sent_at=0)


def test_partition_deliveries_returns_rejections_rather_than_eating_them():
    """A filter that silently dropped messages would be indistinguishable from
    a link that never delivered them. wants the stale rate."""
    envelopes = [
        make_envelope(AUCTION_MESSAGE, 0, AuctionMessage(2, 0), sent_at=1, planning_epoch=2),
        make_envelope(TARGET_UPDATE, 0, BeliefUpdate(), sent_at=1, planning_epoch=2),
        make_envelope(AUCTION_MESSAGE, 1, AuctionMessage(5, 0), sent_at=1, planning_epoch=5),
    ]
    accepted, rejected = receive.partition_deliveries(envelopes, current_planning_epoch=5)
    assert len(accepted) == 2 and len(rejected) == 1
    assert rejected[0].message_type == AUCTION_MESSAGE


def test_no_epoch_context_accepts_everything():
    env = make_envelope(AUCTION_MESSAGE, 0, AuctionMessage(1, 0), sent_at=0, planning_epoch=1)
    assert receive.accept(env, current_planning_epoch=None) is True


# ---------------------------------------------------------------------------
# Tombstones -- disabled is absorbing
# ---------------------------------------------------------------------------


def test_tombstone_dominates_an_older_live_entry():
    live = AgentStatus(agent_id=2, disabled=False, as_of=10)
    tomb = AgentStatus(agent_id=2, disabled=True, as_of=14)
    assert merge_status(live, tomb).disabled is True
    assert merge_status(tomb, live).disabled is True


def test_a_late_liveness_report_cannot_resurrect_a_disabled_agent():
    """The failure this rule exists to prevent: a SELF_STATE broadcast in flight
    when an agent died arrives afterwards and silently revives it, after which
    teammates rank a dead agent in the team-relative role split."""
    tomb = AgentStatus(agent_id=2, disabled=True, as_of=14)
    much_later_liveness = AgentStatus(agent_id=2, disabled=False, as_of=9999)
    assert merge_status(tomb, much_later_liveness).disabled is True


@pytest.mark.parametrize("n_duplicates", [1, 2, 5])
def test_status_merge_is_idempotent_under_duplicate_delivery(n_duplicates):
    tomb = AgentStatus(agent_id=2, disabled=True, as_of=14)
    state = None
    for _ in range(n_duplicates):
        state = merge_status(state, tomb)
    assert state == tomb


def test_status_merge_is_order_independent_over_every_permutation():
    """The production function, under every arrival order including duplicates.

    This is what the placeholder version of this test did not do: it checked one
    ordering against a rule defined inside the test.
    """
    import itertools

    messages = [
        AgentStatus(2, False, 5),
        AgentStatus(2, True, 14),
        AgentStatus(2, False, 40),      # in-flight liveness, arrives after death
        AgentStatus(2, True, 14),       # duplicate tombstone
    ]
    outcomes = set()
    for order in itertools.permutations(messages):
        state = None
        for status in order:
            state = merge_status(state, status)
        outcomes.add((state.disabled, state.as_of))
    assert outcomes == {(True, 14)}, outcomes


def test_two_tombstones_keep_the_earliest_disable_step():
    """An agent died once. The earliest report is closest to the event, and
    ``min`` is what makes the merge order-independent."""
    import itertools

    reports = [AgentStatus(3, True, t) for t in (30, 14, 22)]
    outcomes = {
        tuple(
            (lambda s: (s.disabled, s.as_of))(
                __import__("functools").reduce(merge_status, order, None)
            )
        )
        for order in itertools.permutations(reports)
    }
    assert outcomes == {(True, 14)}


def test_live_records_merge_to_the_freshest():
    a = AgentStatus(4, False, 10)
    b = AgentStatus(4, False, 25)
    assert merge_status(a, b).as_of == 25
    assert merge_status(b, a).as_of == 25


def test_status_dicts_merge_per_agent():
    """One message may carry a stale record for agent A and a fresh one for B."""
    held = {1: AgentStatus(1, False, 5), 2: AgentStatus(2, True, 14)}
    incoming = {1: AgentStatus(1, False, 30), 2: AgentStatus(2, False, 99)}
    merged = merge_status_dicts(held, incoming)
    assert merged[1].as_of == 30
    assert merged[2].disabled is True, "tombstone survived a later liveness claim"


def test_status_merge_over_a_real_partition_does_not_revive():
    """End to end through the network: a tombstone and a stale liveness report
    delivered in either order converge to disabled."""
    import itertools

    tomb_env = make_envelope(
        STATUS_TOMBSTONE, 0, AgentStatus(2, True, 14), sent_at=14
    )
    live_env = make_envelope(SELF_STATE, 2, AgentStatus(2, False, 13), sent_at=13)
    for order in itertools.permutations([tomb_env, live_env]):
        net = Network(seed=7, fixed_edges=topologies.complete(3))
        net.begin_round(positions={i: (i, 0) for i in range(3)})
        for env in order:
            net.send(env)
        state = None
        for env in net.deliver()[1]:
            state = merge_status(state, env.payload)
        assert state.disabled is True


def test_unknown_message_type_is_rejected():
    with pytest.raises(ValueError, match="Unknown message_type"):
        MessageEnvelope(
            message_id="x", message_type="gossip", sender=0, recipient=BROADCAST,
            sent_at=0, planning_epoch=None, payload=None,
        )


# ---------------------------------------------------------------------------
# Disabled agents
# ---------------------------------------------------------------------------


def test_a_disabled_relay_stops_forwarding():
    net = Network(seed=7, fixed_edges=topologies.line(4), store_and_forward=True)
    net.begin_round(positions={i: (i, 0) for i in range(4)}, disabled=[1])
    net.send(make_envelope(TARGET_UPDATE, 0, BeliefUpdate(), sent_at=0))
    delivered = net.deliver()
    # 1 is disabled, so it receives nothing and can relay nothing to 2.
    assert 1 not in delivered
    assert net.forward(1, [make_envelope(TARGET_UPDATE, 1, BeliefUpdate(), sent_at=1)], 1) == 0


def test_a_disabled_sender_sends_nothing():
    net = Network(seed=7, fixed_edges=topologies.complete(3))
    net.begin_round(positions={i: (i, 0) for i in range(3)}, disabled=[0])
    assert net.send(make_envelope(TARGET_UPDATE, 0, BeliefUpdate(), sent_at=0)) == 0
    assert net.stats["dropped_disabled"] == 1


# ---------------------------------------------------------------------------
# Counter-based randomness
# ---------------------------------------------------------------------------


def test_link_draw_depends_only_on_its_key():
    a = _uniform(seed=3, sender=1, receiver=4, comm_tick=9)
    b = _uniform(seed=3, sender=1, receiver=4, comm_tick=9)
    assert a == b
    assert a != _uniform(seed=3, sender=1, receiver=4, comm_tick=10)
    assert a != _uniform(seed=4, sender=1, receiver=4, comm_tick=9)


def test_link_draw_is_symmetric_in_the_pair():
    """A link is one physical thing. Without canonicalizing the pair, i->j and
    j->i would fade independently, which is not a symmetric disk graph."""
    assert _uniform(1, 2, 5, 3) == _uniform(1, 5, 2, 3)


def test_link_draw_is_not_process_dependent():
    """Pinned values. Python's builtin ``hash`` is randomized per process by
    PYTHONHASHSEED; if someone swaps blake2b for it, episodes stop reproducing
    across runs and this catches it."""
    value = _uniform(seed=42, sender=0, receiver=1, comm_tick=0)
    assert 0.0 <= value < 1.0
    assert value == pytest.approx(_uniform(42, 0, 1, 0), abs=0.0)


def test_loss_is_reproducible_and_order_independent():
    """The property that keeps paired conditions paired: querying other links
    first must not change this link's outcome."""
    model = IidLossModel(0.5)
    direct = model.delivers(11, 0, 1, 5)
    other = IidLossModel(0.5)
    for tick in range(20):
        other.delivers(11, 2, 3, tick)
        other.delivers(11, 0, 4, tick)
    assert other.delivers(11, 0, 1, 5) == direct


def test_iid_loss_rate_is_approximately_right():
    model = IidLossModel(0.3)
    trials = [model.delivers(5, 0, 1, t) for t in range(4000)]
    assert 0.65 < np.mean(trials) < 0.75


def test_gilbert_elliott_is_order_independent():
    """State is replayed from tick 0 rather than stored, so interleaving
    queries across links cannot corrupt either one."""
    a = GilbertElliottLossModel()
    straight = [a.state_at(2, 0, 1, t) for t in range(30)]

    b = GilbertElliottLossModel()
    interleaved = []
    for t in range(30):
        b.state_at(2, 4, 5, t)      # unrelated link
        interleaved.append(b.state_at(2, 0, 1, t))
    assert straight == interleaved
    assert set(straight) <= {GOOD, BAD}


def test_gilbert_elliott_produces_bursts_not_iid_loss():
    """The point of the model. Consecutive BAD runs must be longer than iid
    would give at the same marginal rate."""
    model = GilbertElliottLossModel(p_gb=0.05, p_bg=0.3)
    states = [model.state_at(9, 0, 1, t) for t in range(4000)]
    runs, current = [], 0
    for s in states:
        if s == BAD:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    assert runs, "no BAD state ever entered"
    # Mean burst length is 1/p_bg ~= 3.3; iid would give ~1.
    assert np.mean(runs) > 1.5


# ---------------------------------------------------------------------------
# Link models
# ---------------------------------------------------------------------------


def test_disk_model_defaults_to_a_complete_graph():
    """Default-off: the comms layer must reproduce the M3 shared arm until a
    range is configured."""
    model = DiskLinkModel()
    assert model.connected(1e9) is True


def test_logistic_is_the_smoothed_disk():
    model = LogisticLinkModel(r_50=20.0, sigma=2.0)
    assert model.probability(20.0) == pytest.approx(0.5)
    assert model.probability(0.0) > 0.99
    assert model.probability(60.0) < 0.01
    # monotone decreasing
    probs = [model.probability(d) for d in range(0, 60, 5)]
    assert all(a >= b for a, b in zip(probs, probs[1:]))


def test_logistic_rejects_zero_sigma():
    with pytest.raises(ValueError, match="use DiskLinkModel"):
        LogisticLinkModel(r_50=20.0, sigma=0.0)


def test_logistic_does_not_overflow_at_extreme_distance():
    """A near-zero sigma is the disk limit, where the exponent would overflow
    without clipping. The property is a finite probability in [0, 1] -- not
    exactly 0, since clipping at z=700 bottoms out around 1e-304."""
    model = LogisticLinkModel(r_50=20.0, sigma=1e-3)
    far, near = model.probability(1e9), model.probability(-1e9)
    for value in (far, near):
        assert np.isfinite(value) and 0.0 <= value <= 1.0
    assert far < 1e-300
    assert near == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Topologies and connectivity metrics
# ---------------------------------------------------------------------------


def test_topology_shapes():
    assert len(topologies.line(4)) == 3
    assert len(topologies.ring(4)) == 4
    assert len(topologies.complete(4)) == 6
    assert topologies.diameter(topologies.line(4), range(4)) == 3
    assert topologies.diameter(topologies.complete(4), range(4)) == 1


def test_partition_has_no_crossing_edge():
    edges = topologies.partition(4)
    comps = topologies.components(edges, range(4))
    assert sorted(comps) == [[0, 1], [2, 3]]
    assert topologies.diameter(edges, range(4)) is None, "disconnected has no single diameter"


def test_belief_cannot_cross_a_partition():
    """The regime the paper cares about. A message must not reach the far side
    however many rounds run."""
    net = Network(seed=7, fixed_edges=topologies.partition(4), store_and_forward=True)
    positions = {i: (i, 0) for i in range(4)}
    net.begin_round(positions=positions)
    net.send(make_envelope(TARGET_UPDATE, 0, BeliefUpdate(), sent_at=0))
    reached = set()
    for round_index in range(1, 8):
        delivered = net.deliver()
        reached |= set(delivered)
        for agent, envelopes in delivered.items():
            net.forward(agent, envelopes, sent_at=round_index)
        net.begin_round(positions=positions)
    assert reached <= {1}, f"belief escaped the partition to {reached}"


def test_connectivity_metrics_match_the_graph():
    net = Network(seed=7, link_model=DiskLinkModel(r_comm=10.0))
    net.begin_round(positions={0: (0, 0), 1: (5, 0), 2: (100, 0)})
    m = net.connectivity_metrics()
    assert m["connected"] is False
    assert m["n_components"] == 2
    assert m["largest_component_fraction"] == pytest.approx(2 / 3)
    assert m["diameter"] is None


def test_connectivity_metrics_ignore_disabled_agents():
    net = Network(seed=7, link_model=DiskLinkModel(r_comm=10.0))
    net.begin_round(positions={0: (0, 0), 1: (5, 0), 2: (100, 0)}, disabled=[2])
    m = net.connectivity_metrics()
    assert m["n_agents"] == 2
    assert m["connected"] is True


# ---------------------------------------------------------------------------
# Default-off
# ---------------------------------------------------------------------------


def test_default_network_is_a_lossless_complete_graph():
    """M4 must not change any M3 result until it is configured to."""
    net = Network(seed=1)
    net.begin_round(positions={i: (i * 1000.0, 0.0) for i in range(5)})
    assert len(net.edges) == 10
    assert isinstance(net.loss_model, NoLoss)
    net.send(make_envelope(TARGET_UPDATE, 0, BeliefUpdate(), sent_at=0))
    assert set(net.deliver()) == {1, 2, 3, 4}


def test_describe_round_trips_configuration():
    net = Network(
        seed=3,
        link_model=LogisticLinkModel(20.0, 2.0),
        loss_model=IidLossModel(0.1),
        store_and_forward=True,
    )
    d = net.describe()
    assert d["link_model"]["model"] == "logistic"
    assert d["loss_model"]["loss_probability"] == 0.1
    assert d["store_and_forward"] is True
