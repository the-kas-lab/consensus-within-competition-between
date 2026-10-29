"""The auction step 3: CBAA is correct. "CBAA" plus this design's own.



Correctness is checked against an **independent oracle**: with static bids,
single-assignment CBAA converges to the same assignment as the centralized
sequential greedy algorithm (SGA) on any connected graph. The oracle is written
here, deliberately simply, and the shipped auction is run against it on random
bid matrices over complete, line and ring graphs.

Order-invariance is never
accepted as the only evidence of isolation: the double buffer is also tested
directly, as "information travels one hop per round".
"""

import itertools
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coordination.cbaa import AuctionState, CBAAAgent, conflicts, run_rounds  # noqa: E402


# --- graphs -----------------------------------------------------------------

def complete(n):
    return {i: [j for j in range(n) if j != i] for i in range(n)}


def line(n):
    return {i: [j for j in (i - 1, i + 1) if 0 <= j < n] for i in range(n)}


def ring(n):
    return {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}


def diameter(graph):
    n = len(graph)
    worst = 0
    for s in range(n):
        dist, frontier = {s: 0}, [s]
        while frontier:
            nxt = []
            for u in frontier:
                for v in graph[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        worst = max(worst, max(dist.values()))
    return worst


# --- helpers ----------------------------------------------------------------

def make_agents(bid_matrix, stable_rounds=2, epoch=0):
    agents = []
    for agent_id in sorted(bid_matrix):
        a = CBAAAgent(agent_id, stable_rounds=stable_rounds)
        order = sorted(bid_matrix[agent_id])
        a.start_epoch(epoch, bid_matrix[agent_id], order)
        agents.append(a)
    return agents


def sga(bid_matrix):
    """Sequential greedy: repeatedly take the globally best remaining
    (agent, task) pair. The oracle, written independently of the shipped code."""
    agents = set(bid_matrix)
    tasks = set().union(*(bid_matrix[a] for a in agents))
    out = {}
    while agents and tasks:
        best = max(
            ((a, t) for a in agents for t in tasks if t in bid_matrix[a]),
            key=lambda at: (bid_matrix[at[0]][at[1]], -at[0]),
        )
        out[best[0]] = best[1]
        agents.discard(best[0])
        tasks.discard(best[1])
    return out


def random_bids(rng, n_agents, n_tasks):
    tasks = [f"T{k}" for k in range(n_tasks)]
    return {a: {t: rng.random() for t in tasks} for a in range(n_agents)}


# --- the auction converges on a graph ---

def test_two_agents_one_task_one_winner():
    agents = make_agents({0: {"T": 0.4}, 1: {"T": 0.9}})
    run_rounds(agents, complete(2), rounds=5)
    assert [a.assignment for a in agents] == [None, "T"]
    assert conflicts(agents) == {}


# --- agreement ---

def test_equal_bids_go_to_the_lower_id_at_every_agent_every_run():
    """A tie-break, not a race: the same answer whatever order the agents are
    processed in, and every agent agrees who won."""
    bids = {0: {"T": 0.5}, 1: {"T": 0.5}, 2: {"T": 0.5}}
    for order in itertools.permutations(range(3)):
        agents = make_agents(bids)
        run_rounds(agents, complete(3), rounds=6, order=order)
        assert [a.assignment for a in agents] == ["T", None, None]
        assert {a.z["T"] for a in agents} == {0}


def test_an_agent_breaks_ties_between_its_own_tasks_in_canonical_order():
    agents = make_agents({0: {"A": 0.5, "B": 0.5}})
    assert agents[0].assignment == "A"


# --- correctness against the oracle ---correctness against the oracle ------------------------------

@pytest.mark.parametrize("graph_name", ["complete", "line", "ring"])
def test_converges_to_sequential_greedy_on_connected_graphs(graph_name):
    """The strongest check available. 60 random bid matrices per graph, 3-6
    agents, as many or more tasks than agents: every run must be conflict-free,
    every agent must agree on every winner, and the assignment must equal SGA's.

    Round budget is the literature's N * D bound plus the stability window --
    the design's first draft stated D + stable_rounds, which is optimistic
    under release cascades (see test below)."""
    rng = random.Random(3702)
    build = {"complete": complete, "line": line, "ring": ring}[graph_name]
    for _ in range(60):
        n = rng.randint(3, 6)
        bids = random_bids(rng, n, rng.randint(n, n + 4))
        graph = build(n)
        agents = make_agents(bids)
        run_rounds(agents, graph, rounds=n * diameter(graph) + 3)

        assert conflicts(agents) == {}
        assert {a.agent_id: a.assignment for a in agents if a.assignment} == sga(bids)
        # Every agent holds the same picture of who won what.
        for t in bids[0]:
            assert len({(a.y[t], a.z[t]) for a in agents}) == 1, f"disagreement on {t}"


def test_every_agent_gets_a_task_when_there_are_enough():
    rng = random.Random(3703)
    bids = random_bids(rng, 5, 8)
    agents = make_agents(bids)
    run_rounds(agents, ring(5), rounds=5 * diameter(ring(5)) + 3)
    assert all(a.assignment is not None for a in agents)


# --- stale messages ---

def test_disconnected_components_may_disagree_and_it_is_logged_not_repaired():
    """ Two components, each of two agents over the same two
    tasks. Each converges internally to a complete assignment -- so the whole
    assignment is duplicated across the partition, T and U both. Reported, and
    still there after any number of further rounds."""
    bids = {0: {"T": 0.9, "U": 0.1}, 1: {"T": 0.2, "U": 0.3},
            2: {"T": 0.8, "U": 0.1}, 3: {"T": 0.1, "U": 0.4}}
    graph = {0: [1], 1: [0], 2: [3], 3: [2]}
    agents = make_agents(bids)
    run_rounds(agents, graph, rounds=10)
    assert conflicts(agents) == {"T": [0, 2], "U": [1, 3]}
    run_rounds(agents, graph, rounds=50)
    assert conflicts(agents) == {"T": [0, 2], "U": [1, 3]}, (
        "a conflict must never be silently repaired"
    )


# --- one hop per round ---

def test_a_stale_epoch_message_cannot_touch_the_new_epoch():
    """The rule that an auction may span planning epochs. A bid from epoch k arriving during k+1 -- a high one, the kind that
    would otherwise steal a task -- changes nothing and is counted."""
    agent = CBAAAgent(0, stable_rounds=2)
    agent.start_epoch(5, {"T": 0.3}, ["T"])
    stale = AuctionState(epoch=4, sender=1, entries=(("T", 0.99, 1),))
    agent.receive([stale])
    assert agent.assignment == "T"
    assert agent.y["T"] == 0.3 and agent.z["T"] == 0
    assert agent.stale_messages_ignored == 1


# --- no belief from auction traffic ---the rule that auction traffic never creates a belief ----------------------------------------------------

def test_auction_traffic_cannot_create_a_task():
    """The rule that auction traffic never creates a belief. Agent 0 believes in contact 7 and holds its pursuit task.
    Agent 1 does not believe in contact 7. However many rounds of agent 0's
    traffic arrive, agent 1 never acquires a task, a bid, a winner or an
    assignment for it -- and the channel it would have been is counted."""
    agents = make_agents({0: {"NEUTRALIZE:7:0": 0.9, "DEFEND:0": 0.1},
                          1: {"DEFEND:0": 0.2, "DEFEND:1": 0.1}})
    run_rounds(agents, complete(2), rounds=8)
    one = agents[1]
    assert "NEUTRALIZE:7:0" not in one.y and "NEUTRALIZE:7:0" not in one.bids
    assert "NEUTRALIZE:7:0" not in one.order
    assert one.assignment != "NEUTRALIZE:7:0"
    assert one.foreign_entries_ignored > 0


# --- commitment ---

def test_zero_communication_is_independent_greedy():
    """The natural no-coordination bound, reached without a special case: with
    no messages every agent takes its own best task, conflicts and all."""
    bids = {0: {"T": 0.9, "U": 0.1}, 1: {"T": 0.8, "U": 0.7}}
    agents = make_agents(bids)
    run_rounds(agents, {0: [], 1: []}, rounds=5)
    assert [a.assignment for a in agents] == ["T", "T"]
    assert conflicts(agents) == {"T": [0, 1]}


# --- the double buffer, tested directly and by order ----------------------

def test_information_travels_one_hop_per_round():
    """The direct test of the double buffer, not merely an order test. On the
    line 0-1-2, after ONE round agent 2 must not know agent 0's bid, whatever
    order the driver processes agents in. A driver that let agent 1 relay
    within the round would leak it when 1 is processed before 2."""
    bids = {0: {"T": 0.9}, 1: {"T": 0.1}, 2: {"T": 0.5}}
    for order in itertools.permutations(range(3)):
        agents = make_agents(bids)
        run_rounds(agents, line(3), rounds=1, order=order)
        assert agents[2].z["T"] != 0, f"two hops in one round with order {order}"
    agents = make_agents(bids)
    run_rounds(agents, line(3), rounds=2)
    assert agents[2].z["T"] == 0, "and it must arrive in exactly two"


def test_the_result_does_not_depend_on_processing_order():
    rng = random.Random(3704)
    bids = random_bids(rng, 4, 6)
    results = set()
    for order in itertools.permutations(range(4)):
        agents = make_agents(bids)
        run_rounds(agents, ring(4), rounds=15, order=order)
        results.add(tuple(a.assignment for a in agents))
    assert len(results) == 1


def test_a_broadcast_state_is_a_copy_not_a_reference():
    a = make_agents({0: {"T": 0.5}})[0]
    sent = a.outgoing()
    a.y["T"] = 0.99
    assert sent.entries == (("T", 0.5, 0),)


# --- the rule that an auction may span planning epochs: commitment is held until the auction settles locally ---------------

def test_the_previous_commitment_is_held_until_the_new_auction_is_stable():
    agent = CBAAAgent(0, stable_rounds=3)
    agent.start_epoch(0, {"A": 0.9, "B": 0.1}, ["A", "B"])
    for _ in range(3):
        agent.receive([])
    assert agent.commitment == "A"

    # New epoch prefers B. The agent keeps executing A until 3 quiet rounds.
    agent.start_epoch(1, {"A": 0.1, "B": 0.9}, ["A", "B"])
    assert agent.assignment == "B"
    for expected in ("A", "A", "B"):
        agent.receive([])
        assert agent.commitment == expected
    assert agent.rounds_to_stable == 3


def test_order_must_list_exactly_the_bid_tasks():
    with pytest.raises(ValueError, match="order"):
        CBAAAgent(0, 2).start_epoch(0, {"A": 0.1}, ["A", "B"])


def test_a_late_outbid_after_local_stability_updates_the_commitment():
    """Found by measuring convergence rather than assuming it: in 251-307 of 400
    random trials per graph CBAA took MORE than diameter rounds to settle
    (release cascades, up to 4x D). Local stability is therefore a heuristic --
    news from far away can arrive after an agent has gone quiet.

    Line 0-1-2-3, stable_rounds=1. Agent 0 claims T and goes quiet before agent
    3's higher bid, three hops away, has reached it. When that bid lands, agent 0
    must release T and, once quiet again, COMMIT to its new task. Committing only
    on the first stable round would leave it executing T -- a task another agent
    now holds -- for the rest of the epoch."""
    bids = {0: {"T": 0.5, "U": 0.4}, 1: {"T": 0.0, "U": 0.0},
            2: {"T": 0.0, "U": 0.0}, 3: {"T": 0.9, "U": 0.0}}
    agents = make_agents(bids, stable_rounds=1)
    zero = agents[0]
    run_rounds(agents, line(4), rounds=1)
    assert zero.commitment == "T", "precondition: stable before the news arrives"

    run_rounds(agents, line(4), rounds=12)
    assert agents[3].assignment == "T"
    assert zero.assignment == "U"
    assert zero.commitment == "U", "the commitment must follow a late outbid"
    assert zero.recommitments >= 1, "and the churn must be counted"
