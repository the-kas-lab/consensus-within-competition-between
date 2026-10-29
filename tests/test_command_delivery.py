"""The command-delivery conditions contract Sec 5.5 — the deterministic delivery checks for condition B.

Not a sweep. The next evidence the command-delivery conditions needs is that the delivery path has the
semantics the contract specifies, and that is settled by a small set of staged
cases rather than by episodes averaged over seeds.

Every check reads the **generation, receipt and application step of individual
commands** off `sim.command_log`, because the limitation in contract Sec 3 --
B - A mixes inbound information latency with outbound assignment latency, and
this design cannot separate them -- is only *observable* if the outbound half is
timestamped. A mean latency would not show that a particular late command was
applied, which is the thing most at risk of being silently wrong.

The five cases, in contract order:

1. a command arrives no earlier than one comm round after the epoch that
   produced it;
2. delayed commands are applied, not rejected (Sec 3a);
3. an older command arriving after a newer one is rejected, and the newer one
   survives;
4. a disconnected recipient receives nothing and holds its last accepted
   command;
5. commander loss with a command in transit: it still lands and is applied
   (Sec 3b).
"""

import pytest

from conftest import build_simulation, fingerprint, run_quiet
from comms.messages import ASSIGNMENT_COMMAND, AssignmentCommand, make_envelope
from planning.roles import BlueRole


def _delivery_sim(comm_interval_seconds=0.1, r_comm=float("inf"), seed=3302,
                  sim_steps=200, **extra):
    """Condition B with assignment delivery on. Everything else is the communication conditions
    configuration: `symmetric`, 4 blues / 2 reds, planning interval 1.0 s."""
    return build_simulation(
        num_blues=4, num_reds=2, seed=seed, sim_steps=sim_steps,
        belief_mode="networked", r_comm=r_comm,
        comm_interval_seconds=comm_interval_seconds,
        centralized_from_belief=True,
        deliver_assignments_over_network=True,
        commander_failure_stops_planning=True,
        **extra,
    )


def _run_killing_the_commander_at(sim, kill_at):
    """Disable the commander at a chosen step, checked every step.

    `_update_beliefs` is the hook because it runs every step, unlike the
    planning path -- and the point of case 5 is a death that falls BETWEEN two
    planning epochs, while a command is already on the wire.
    """
    original = sim._update_beliefs

    def update_with_a_scheduled_death(step, *args, **kwargs):
        if step >= kill_at:
            sim.blue_disabled[sim.commander_id] = True
        return original(step, *args, **kwargs)

    sim._update_beliefs = update_with_a_scheduled_death
    return run_quiet(sim)


def _network_commands(sim):
    return [e for e in sim.command_log if e["via"] == "network"]


# ---------------------------------------------------------------------------
# 0. Default-off, first, because everything below is worthless if it leaks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["ground_truth", "shared", "local", "networked"])
def test_delivery_off_leaves_every_existing_mode_byte_identical(mode):
    def run(**extra):
        return fingerprint(run_quiet(build_simulation(
            num_blues=4, num_reds=2, seed=7, sim_steps=120, belief_mode=mode,
            **extra,
        )))

    assert run() == run(deliver_assignments_over_network=False,
                        command_max_age_steps=None)


def test_no_command_is_issued_unless_delivery_is_on():
    """The ledger is empty in every other configuration, including condition A,
    which delivers immediately and has nothing to send."""
    sim = run_quiet(build_simulation(
        num_blues=4, num_reds=2, seed=7, sim_steps=60, belief_mode="shared",
        centralized_from_belief=True))
    assert sim.command_log == []
    assert sim.command_metrics()["generated"] == 0


def test_delivery_requires_a_commander_and_a_network():
    """Refused at construction rather than silently doing nothing. C and D
    compute their own assignments and have nothing to deliver; the non-networked
    belief modes build no network for a command to travel on."""
    with pytest.raises(ValueError, match="centralized_from_belief"):
        build_simulation(num_blues=4, num_reds=2, belief_mode="networked",
                         deliver_assignments_over_network=True)
    with pytest.raises(ValueError, match="networked"):
        build_simulation(num_blues=4, num_reds=2, belief_mode="shared",
                         centralized_from_belief=True,
                         deliver_assignments_over_network=True)


# ---------------------------------------------------------------------------
# 1. No command arrives sooner than one communication round
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("comm_interval_seconds", [0.1, 0.5, 1.0])
def test_direct_delivery_costs_exactly_one_communication_round(comm_interval_seconds):
    """**The DIRECT case only**, and it is parametrised to say so.

    `R_comm` is infinite here, so every subordinate is a one-hop neighbour and
    every generation step has a round open. Those two conditions are what make
    the cost one round; neither is general. Generation-to-receipt is, in
    general, *wait for the next open round + one round per hop traversed*, and
    `test_a_command_crosses_a_chain_one_hop_per_round` is the multi-hop half.

    The equality is on **generation to receipt**, not on handoff to receipt.
    Handoff to receipt is one round whatever else is wrong -- the double buffer
    guarantees it -- so asserting that would pass even if a command sat a whole
    interval waiting to be handed over. Planning runs *after* the step's
    communication round, so a command that waited for the next round would cost
    two intervals, and B would carry an outbound latency the contract never
    declared. Every comm period here divides the 10-step planning interval, so
    every generation step has a round open and one round is the exact cost.
    """
    sim = run_quiet(_delivery_sim(comm_interval_seconds=comm_interval_seconds))
    per_round = sim.comm_interval_steps
    assert sim.planning_interval_steps % per_round == 0, (
        "this check assumes a round is open at every planning step"
    )
    delivered = [e for e in _network_commands(sim) if e["received_at"] is not None]
    assert delivered, "no command crossed the network; the check is vacuous"
    for entry in delivered:
        # The premise of the one-round claim, asserted rather than assumed: on
        # a complete graph nothing should ever arrive by a longer path.
        assert entry["hops"] == 1, entry
        assert entry["received_at"] - entry["generated_at"] == per_round, entry
        assert entry["received_at"] - entry["sent_at"] == per_round, entry
        # Applied on receipt, not deferred to the next planning epoch -- which,
        # once the commander is gone, may never come.
        assert entry["applied_at"] == entry["received_at"], entry


def test_the_commander_does_not_radio_itself():
    """It is the source. Routing its own assignment through the network would
    model a link an agent has with itself and would delay condition B's
    commander relative to condition A's for no physical reason."""
    sim = run_quiet(_delivery_sim())
    own = [e for e in sim.command_log if e["agent"] == sim.commander_id]
    assert own and all(e["via"] == "self" for e in own)
    assert all(e["applied_at"] == e["generated_at"] for e in own)


# ---------------------------------------------------------------------------
# 2. A delayed command is APPLIED, not rejected — contract Sec 3a
# ---------------------------------------------------------------------------


def test_a_command_older_than_the_current_epoch_is_still_applied():
    """The defect the contract settled in advance, tested end to end.

    `comms/receive.py`'s epoch filter rejects allocation traffic older than the
    epoch the caller supplies. Had condition B reused it with the wall-clock
    planning epoch, every command slower than one planning interval would be
    rejected *because* it was slow, and B would report "commands never arrive"
    as a communication result.

    A 5 s communication period against a 1 s planning interval guarantees the
    case: a command is generated at epoch k and lands several epochs later.
    """
    sim = run_quiet(_delivery_sim(comm_interval_seconds=5.0))
    per_planning = sim.planning_interval_steps

    late = [
        e for e in _network_commands(sim)
        if e["received_at"] is not None
        and e["epoch"] < e["received_at"] // per_planning
    ]
    assert late, (
        "no command arrived after its own epoch had passed, so this episode "
        "cannot distinguish 'late' from 'rejected'"
    )
    assert all(e["outcome"] == "accepted" for e in late), (
        "a delayed command was rejected: the ordering rule is reading the wall "
        "clock rather than the receiver's accepted-command history"
    )
    # And the delay is real and reported, not a rounding artifact.
    assert max(e["received_at"] - e["generated_at"] for e in late) >= per_planning


def test_the_age_limit_is_off_by_default_and_binds_only_when_declared():
    """A separate, separately-declared parameter -- never the ordering rule.
    With it off, "arrived late" stays distinguishable from "never arrived",
    which is the distinction condition B exists to measure."""
    baseline = run_quiet(_delivery_sim(comm_interval_seconds=5.0))
    assert baseline.command_metrics()["rejected_too_old"] == 0

    limited = run_quiet(_delivery_sim(comm_interval_seconds=5.0,
                                      command_max_age_steps=5))
    assert limited.command_metrics()["rejected_too_old"] > 0, (
        "an age limit shorter than the communication period rejected nothing"
    )
    assert fingerprint(baseline) != fingerprint(limited)


# ---------------------------------------------------------------------------
# 3. Reordering — the newer command survives
# ---------------------------------------------------------------------------


def test_an_older_command_arriving_after_a_newer_one_is_rejected():
    """Staged directly against a live Simulation's receive path, because the
    network delivers in send order and will not reorder on its own. What is
    exercised is the shipped acceptance rule and the shipped ledger, not logic
    defined in this test.
    """
    sim = _delivery_sim()
    sim.setup_agents()

    def deliver(epoch, role, target):
        envelope = make_envelope(
            ASSIGNMENT_COMMAND, sim.commander_id,
            AssignmentCommand(planning_epoch=epoch, commander_id=sim.commander_id,
                              agent_id=1, issued_at=epoch * 10, role=role,
                              target_red_id=target),
            sent_at=epoch * 10 + 1, recipient=1, planning_epoch=epoch,
        )
        return sim._receive_command(1, envelope, step=epoch * 10 + 2)

    assert deliver(7, BlueRole.NEUTRALIZE, 1) is True
    assert sim.agent_commitments[1].planning_epoch == 7

    # The straggler from epoch 5, arriving after epoch 7 is in force.
    assert deliver(5, BlueRole.RECON, 0) is False
    assert sim.agent_commitments[1].planning_epoch == 7, (
        "an older command displaced a newer one already in force"
    )
    assert sim.agent_commitments[1].role == BlueRole.NEUTRALIZE
    assert sim.command_metrics()["rejected_superseded"] == 1

    # And the assignment actually put into force is the surviving one.
    assert sim._commitment_output().roles[1] == BlueRole.NEUTRALIZE
    assert sim._commitment_output().target_red_ids[1] == 1


def test_a_duplicate_of_the_command_in_force_does_not_count_as_an_arrival():
    sim = _delivery_sim()
    sim.setup_agents()
    envelope = make_envelope(
        ASSIGNMENT_COMMAND, sim.commander_id,
        AssignmentCommand(planning_epoch=3, commander_id=sim.commander_id,
                          agent_id=2, issued_at=30, role=BlueRole.DEFEND),
        sent_at=31, recipient=2, planning_epoch=3,
    )
    assert sim._receive_command(2, envelope, step=32) is True
    assert sim._receive_command(2, envelope, step=40) is False
    assert sim.command_metrics()["accepted"] == 1
    assert sim.command_metrics()["rejected_superseded"] == 1


# ---------------------------------------------------------------------------
# 4. A disconnected recipient hears nothing and holds what it has
# ---------------------------------------------------------------------------


def test_a_disconnected_team_receives_no_command_and_holds_its_commitment():
    """`r_comm = 0` isolates every blue. The commander still plans and still
    applies its own assignment; nobody else is ever told anything, and nobody
    idles waiting -- they hold the initial assignment.

    This is the lower anchor for condition B's outbound half, and it is what
    says a delivery failure is visible rather than silent.
    """
    sim = run_quiet(_delivery_sim(r_comm=0.0))
    metrics = sim.command_metrics()
    assert metrics["generated"] > 0
    assert metrics["delivered"] == 0
    assert metrics["no_link"] > 0
    assert metrics["handed_to_network"] == 0

    # The commander is the only agent ever under orders.
    assert set(sim.agent_commitments) == {sim.commander_id}
    assert set(sim.last_accepted_command_epoch) == {sim.commander_id}
    assert set(sim.planner_output.roles) <= {sim.commander_id}

    # Every undelivered command is recorded as such, with no receipt or
    # application step -- a drop that is counted, not a message that vanished.
    for entry in _network_commands(sim):
        assert entry["outcome"] == "no_link"
        assert entry["received_at"] is None and entry["applied_at"] is None


def test_a_partially_connected_team_is_told_in_part():
    """The complement: at a range where the graph comes and goes, some commands
    land and some do not, and which is which is on the record."""
    sim = run_quiet(_delivery_sim(r_comm=15.0, comm_interval_seconds=1.0))
    outcomes = {e["outcome"] for e in _network_commands(sim)}
    assert "accepted" in outcomes and "no_link" in outcomes


# ---------------------------------------------------------------------------
# 5. Commander loss with a command in transit — contract Sec 3b
# ---------------------------------------------------------------------------


def test_a_command_already_in_flight_lands_after_the_commander_dies():
    """Death stops GENERATION. It does not cancel what is already travelling:
    the network does not know the sender died.

    So survivors retain their last *received* commitment, which may still
    update for a round or two after the commander is gone. That is more precise
    than freezing the team at the commander's final planning epoch, and it is
    the property this check pins.

    Staged with a 2 s communication period against a 1 s planning interval, so
    that the death at step 101 falls between a command's handoff and its
    delivery rather than at a planning epoch.
    """
    sim = _run_killing_the_commander_at(
        _delivery_sim(comm_interval_seconds=2.0), kill_at=101
    )
    lost = sim.commander_lost_step
    assert lost is not None

    in_flight = [
        e for e in _network_commands(sim)
        if e["sent_at"] is not None and e["sent_at"] < lost
        and e["applied_at"] is not None and e["applied_at"] > lost
    ]
    assert in_flight, (
        "no command was handed to the network before the commander died and "
        "applied after; this episode cannot demonstrate Sec 3b"
    )
    for entry in in_flight:
        assert entry["outcome"] == "accepted"
        assert entry["generated_at"] < lost

    # No NEW command is generated after the loss. Both halves together are the
    # claim; either alone reads as the other failing.
    assert not [e for e in sim.command_log if e["generated_at"] > lost]


def test_a_command_not_yet_transmitted_dies_with_its_commander():
    """The complement of the case above, and the boundary between them.

    A command generated at a planning step with no communication round open is
    still in the commander's hands. If the commander is destroyed before the
    next round it is never transmitted -- `Network.send` refuses a disabled
    sender -- and that is the physically honest reading of "in flight". It is
    counted rather than silent, because a command that was never sent and a
    command that was sent and rejected are different findings.
    """
    sim = _run_killing_the_commander_at(
        _delivery_sim(comm_interval_seconds=3.0), kill_at=101
    )
    never_sent = [
        e for e in sim.command_log
        if e["outcome"] == "not_sent_commander_disabled"
    ]
    assert never_sent, "no command was caught between generation and handoff"
    for entry in never_sent:
        assert entry["sent_at"] is None
        assert entry["received_at"] is None and entry["applied_at"] is None
    assert sim.command_metrics()["not_sent_sender_disabled"] == len(never_sent)


def test_the_safety_filter_still_runs_after_the_commander_is_lost():
    """Planning is gated by commander loss. The safety supervisor never is --
    it runs every step on ground truth, in every condition."""
    sim = _run_killing_the_commander_at(
        _delivery_sim(comm_interval_seconds=2.0), kill_at=101
    )
    after = [s for s in sim.recorder.steps if s["step"] > sim.commander_lost_step]
    assert after and any(s["cbf_active"] for s in after)


# ---------------------------------------------------------------------------
# The confound the delivery path could have introduced, and did not
# ---------------------------------------------------------------------------


def test_a_landing_command_does_not_refresh_anybody_elses_aim_point():
    """B - A must isolate information, not move the motion layer.

    A resolved `target_override` is a believed position extrapolated to the step
    it was resolved on, and between planning epochs it stays frozen at that
    value -- that is true under condition A and must stay true under B. If a
    command landing for blue 1 re-resolved the whole team, every other blue
    would get a fresher aim point than its condition-A counterpart ever sees,
    and B - A would carry a motion-layer advantage created by the delivery path.

    Driven through the shipped `_apply_assignment` on a live Simulation, with
    the belief moved between the two calls so that a re-resolution would be
    visible.
    """
    import numpy as np
    from planning.planner import PlannerOutput

    sim = _delivery_sim()
    sim.setup_agents()
    for red in range(sim.num_reds):
        sim.red_outcomes[red] = "active"

    plan = PlannerOutput(
        roles={0: BlueRole.NEUTRALIZE, 1: BlueRole.NEUTRALIZE},
        target_red_ids={0: 0, 1: 1},
    )
    at_epoch = {
        0: {0: np.array([10.0, 10.0, 0.0, 0.0])},
        1: {1: np.array([20.0, 20.0, 0.0, 0.0])},
    }
    sim._apply_assignment(plan, step=10, believed_targets=at_epoch)
    frozen = np.array(sim.target_override[0], dtype=float)

    # One step later the beliefs have moved for BOTH agents, and a command
    # lands for blue 1 only.
    moved = {
        0: {0: np.array([99.0, 99.0, 0.0, 0.0])},
        1: {1: np.array([77.0, 77.0, 0.0, 0.0])},
    }
    sim._apply_assignment(plan, step=11, believed_targets=moved, resolve_only={1})

    assert np.allclose(sim.target_override[0], frozen), (
        "blue 0's aim point was refreshed by a command addressed to blue 1"
    )
    assert np.allclose(sim.target_override[1], [77.0, 77.0, 0.0, 0.0]), (
        "blue 1 did not act on the order it just accepted"
    )
    # The recorded output is still the whole team's, not the restricted copy
    # used to bound the re-resolution.
    assert sim.planner_output.target_red_ids == {0: 0, 1: 1}
    assert sim.role_assignment == plan.roles


def test_the_unrestricted_path_does_refresh_everyone():
    """The complement, so the test above is known to be measuring the
    restriction and not an absence of movement."""
    import numpy as np
    from planning.planner import PlannerOutput

    sim = _delivery_sim()
    sim.setup_agents()
    for red in range(sim.num_reds):
        sim.red_outcomes[red] = "active"

    plan = PlannerOutput(roles={0: BlueRole.NEUTRALIZE}, target_red_ids={0: 0})
    sim._apply_assignment(plan, 10, {0: {0: np.array([10.0, 10.0, 0.0, 0.0])}})
    sim._apply_assignment(plan, 11, {0: {0: np.array([99.0, 99.0, 0.0, 0.0])}})
    assert np.allclose(sim.target_override[0], [99.0, 99.0, 0.0, 0.0])


def test_in_a_real_episode_only_the_receiving_agents_are_re_resolved():
    """The same property at the call site, not just in the method.

    The test above drives `_apply_assignment` directly, so it pins the
    restriction but not the wiring: passing `resolve_only=None` from the
    delivery path would still satisfy it. This one runs a real episode with a
    0.1 s communication period -- so commands land one step after each planning
    epoch, on steps that are NOT planning steps -- and watches what the override
    actually does there.
    """
    import numpy as np

    sim = _delivery_sim(comm_interval_seconds=0.1, sim_steps=120)
    seen = []
    original = sim._apply_assignment

    def watched(planned, step, believed_targets, resolve_only=None):
        before = {a: np.array(v, float).copy()
                  for a, v in sim.target_override.items()}
        original(planned, step, believed_targets, resolve_only=resolve_only)
        after = {a: np.array(v, float).copy()
                 for a, v in sim.target_override.items()}
        seen.append((step, resolve_only, before, after))

    sim._apply_assignment = watched
    run_quiet(sim)

    off_epoch = [
        row for row in seen if row[0] % sim.planning_interval_steps != 0
    ]
    assert off_epoch, (
        "nothing was applied between planning epochs, so this episode cannot "
        "show whether the re-resolution is restricted"
    )
    # Asserted before the contents, so removing the restriction fails HERE and
    # says so, rather than emptying the sample and failing as 'vacuous'.
    assert all(row[1] is not None for row in off_epoch), (
        "an off-epoch application re-resolved the whole team: the delivery "
        "path is not passing resolve_only, so condition B gets a fresher aim "
        "point than condition A"
    )
    moved_somewhere = False
    for step, resolve_only, before, after in off_epoch:
        for agent, old in before.items():
            if agent in resolve_only:
                continue
            assert agent in after and np.allclose(after[agent], old), (
                f"step {step}: blue {agent}'s aim point moved on a command "
                f"addressed to {sorted(resolve_only)}"
            )
        moved_somewhere |= any(
            agent in after and not np.allclose(after[agent], before[agent])
            for agent in resolve_only if agent in before
        ) or bool(set(after) - set(before))
    # Not asserted as a requirement -- a receiving agent's aim point need not
    # move every time -- but reported so a wholly inert delivery path cannot
    # pass this check by doing nothing at all. The delivery checks above are
    # what establish that commands are applied.
    assert sim.command_metrics()["accepted"] > 0


# ---------------------------------------------------------------------------
# Multi-hop: a command must reach the whole connected component, not only the
# commander's immediate neighbours
# ---------------------------------------------------------------------------
#
# `Network.send` intersects an ADDRESSED envelope with the sender's immediate
# neighbours, so a point-to-point command could never leave the commander's own
# neighbourhood -- while beliefs reach the whole component by re-broadcast. On a
# connected chain 0-1-2 the commander would then learn blue 2's observations
# through belief gossip but be unable to give blue 2 an order. Condition B would
# have measured multi-hop inbound information against direct-only outbound
# control: a protocol restriction beyond the declared inbound/outbound latency
# mix, and worst in exactly the finite-range regimes the communication conditions selected.
#
# Commands are therefore broadcast, addressed in the payload, and carried one
# hop per round by intermediaries. The chain below is the decisive check; it
# needs no episode batch.


def _chain_simulation(n=3, **extra):
    """A live Simulation whose network is a fixed chain 0-1-2-...

    The topology is replaced rather than searched for in a seed: a staged graph
    is what makes "two rounds, not one" an assertion instead of an observation.
    Everything else -- generation, relay, acceptance, the ledger -- is the
    shipped code the episode loop runs.
    """
    from comms import Network
    from comms import topologies
    from comms.link_models import DiskLinkModel

    sim = build_simulation(
        num_blues=n, num_reds=2, seed=3303, sim_steps=50,
        belief_mode="networked", r_comm=float("inf"),
        comm_interval_seconds=0.1,
        centralized_from_belief=True,
        deliver_assignments_over_network=True,
        commander_failure_stops_planning=True,
        **extra,
    )
    from perception.belief import LocalBeliefStore

    sim.setup_agents()
    # The per-episode state `run` builds before its first round. Assembled
    # through the shipped factory rather than a literal, so a counter added
    # later cannot silently go missing here -- `_fresh_comm_stats` exists for
    # exactly this reason.
    sim.belief_stores = {
        i: LocalBeliefStore(i, n_regions=len(sim.coverage_tracker.centers))
        for i in range(sim.num_blues)
    }
    sim.comm_stats = sim._fresh_comm_stats()
    sim.network = Network(
        seed=sim.seed,
        link_model=DiskLinkModel(r_comm=sim.r_comm),
        fixed_edges=topologies.line(n),
    )
    return sim


def _order_for(sim, addressee, role=BlueRole.NEUTRALIZE, target=0, step=0):
    """Issue one epoch's commands through the shipped generation path."""
    from planning.planner import PlannerOutput

    plan = PlannerOutput(
        roles={i: role for i in range(sim.num_blues)},
        target_red_ids={addressee: target},
    )
    sim._issue_commands(plan, step, believed_targets={}, generated=True)


def _round(sim, step):
    sim._exchange_beliefs_over_network(
        step, [i for i in range(sim.num_blues) if not sim.blue_disabled[i]]
    )


def test_a_command_crosses_a_chain_one_hop_per_round():
    """0-1-2. Blue 2 is two hops from the commander.

    It must receive on the second round and not the first. Both halves are
    asserted, because "eventually arrives" would also be satisfied by a flood
    that crossed the whole graph in one tick.

    The one-hop-per-round half is *enforced* by `Network`'s double buffer and
    pinned there (tests/test_comms.py); what this check adds is that command
    traffic actually goes through it, rather than that the property holds. The
    load-bearing assertion here is the third round -- without a relay, blue 2
    is never reached at all.
    """
    sim = _chain_simulation()
    _order_for(sim, addressee=2, step=0)

    _round(sim, 0)   # commander hands it over
    assert 2 not in sim.agent_commitments

    _round(sim, 1)   # blue 1 receives it and carries it
    assert 2 not in sim.agent_commitments, (
        "blue 2 was reached in one round; a command crossed two hops in one "
        "tick and the double buffer is not holding"
    )
    assert sim.command_stats["relayed"] > 0, "blue 1 did not carry the order"

    _round(sim, 2)   # blue 2 receives it
    assert 2 in sim.agent_commitments, (
        "blue 2 never received its order: commands are not being relayed, so "
        "outbound control is limited to the commander's direct neighbours"
    )
    assert sim.agent_commitments[2].target_red_id == 0

    entry = [e for e in sim.command_log if e["agent"] == 2][-1]
    assert (entry["generated_at"], entry["sent_at"]) == (0, 0)
    assert entry["received_at"] == 2 and entry["applied_at"] == 2
    assert entry["hops"] == 2, entry


def test_an_intermediary_carries_an_order_without_executing_it():
    """Blue 1 relays blue 2's order. It must not read it into its own
    commitment, and the acceptance rule must be evaluated against blue 2's
    history rather than blue 1's."""
    sim = _chain_simulation()
    _order_for(sim, addressee=2, role=BlueRole.NEUTRALIZE, target=1, step=0)
    for step in (0, 1, 2):
        _round(sim, step)

    assert sim.agent_commitments[2].target_red_id == 1
    # Blue 1 has its own order from the same epoch, and it is its OWN.
    assert sim.agent_commitments[1].agent_id == 1
    assert sim.agent_commitments[1].target_red_id is None
    assert sim.last_accepted_command_epoch[1] == sim.last_accepted_command_epoch[2]


def test_relaying_preserves_origin_issue_time_and_epoch():
    """A relay must not rejuvenate an order, exactly as it must not rejuvenate
    an observation. `forwarded_by` rewrites sender, hop_count and ttl and never
    touches the payload, and this is what says so."""
    sim = _chain_simulation()
    _order_for(sim, addressee=2, step=0)
    for step in (0, 1, 2):
        _round(sim, step)

    carried = sim.agent_commitments[2]
    assert carried.commander_id == sim.commander_id
    assert carried.issued_at == 0
    assert carried.planning_epoch == 0


def test_each_agent_carries_a_given_command_only_once():
    """What terminates the flood. Without it a command would circulate forever
    on any graph with a cycle, and the relay counters would grow without
    bound."""
    sim = _chain_simulation()
    _order_for(sim, addressee=2, step=0)
    for step in range(8):
        _round(sim, step)
    # 3 nodes, one command per agent per epoch, each relayed at most once by
    # each non-addressee: bounded, and small.
    assert sim.command_stats["relayed"] <= sim.num_blues * sim.num_blues
    assert sim.command_stats["relay_suppressed_duplicate"] > 0


def test_a_surviving_relay_keeps_carrying_an_order_after_the_commander_dies():
    """Contract Sec 3b at range, not only for direct neighbours.

    The sender of a relayed envelope is the relay, not the commander, so
    `Network.send`'s disabled-sender rule does not stop it. An order that
    reached blue 1 before the commander died still reaches blue 2 afterwards.
    """
    sim = _chain_simulation()
    _order_for(sim, addressee=2, step=0)
    _round(sim, 0)                 # handed over
    _round(sim, 1)                 # blue 1 holds it and re-broadcasts

    sim.blue_disabled[sim.commander_id] = True   # commander destroyed

    _round(sim, 2)
    assert 2 in sim.agent_commitments, (
        "the order died with the commander even though a live relay was "
        "already carrying it"
    )
    assert sim.agent_commitments[2].commander_id == sim.commander_id


def test_a_command_hop_cap_binds_only_when_declared():
    """`command_max_hops` is a separately-declared cap, off by default. With it
    set to 1 on a chain, blue 2 is unreachable and that is visible rather than
    silent."""
    capped = _chain_simulation(command_max_hops=1)
    _order_for(capped, addressee=2, step=0)
    for step in (0, 1, 2, 3):
        _round(capped, step)
    assert 2 not in capped.agent_commitments
    assert capped.command_stats["relay_dropped"] > 0

    uncapped = _chain_simulation()
    _order_for(uncapped, addressee=2, step=0)
    for step in (0, 1, 2, 3):
        _round(uncapped, step)
    assert 2 in uncapped.agent_commitments


def test_comm_message_ttl_still_does_not_reach_commands():
    """Flaw C's pin, extended to condition B.

    `comm_message_ttl` is inert -- read only by `Network.forward`, which
    production code never calls. The command relay is bounded by its OWN
    parameter, so a reader cannot conclude that TTL governs command reach.
    """
    for ttl in (1, 64):
        sim = _chain_simulation(comm_message_ttl=ttl)
        assert sim.network.store_and_forward is False
        _order_for(sim, addressee=2, step=0)
        for step in (0, 1, 2):
            _round(sim, step)
        assert 2 in sim.agent_commitments, (
            f"comm_message_ttl={ttl} changed how far a command travelled"
        )
