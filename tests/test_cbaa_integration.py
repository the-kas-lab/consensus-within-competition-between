"""The auction step 5: CBAA inside a live episode, default-off, per the rule that an auction may span planning epochs and the rule that auction traffic never creates a belief.

 Every test drives a live
Simulation. The fixture was chosen by measurement: under CBAA a 150-step episode
never believes in a contact (agents hold stations facing outward and the reds are
still far away), so every contact-task test on it would pass vacuously. At 300
steps contacts are believed from step ~166 and both SCREEN and NEUTRALIZE are
committed -- asserted below as a guard, not assumed.
"""

from collections import Counter

import pytest

from conftest import build_simulation, run_quiet

EXERCISED = dict(num_blues=3, num_reds=2, sim_steps=300, seed=7)


def _run(**overrides):
    params = dict(EXERCISED)
    params.update(overrides)
    return run_quiet(build_simulation(**params))


@pytest.fixture(scope="module")
def shared_run():
    return _run(belief_mode="shared", allocation="cbaa")


def _sources(sim):
    return Counter(
        e["source"] for st in sim.recorder.steps for e in st["effective_targets"].values()
    )


# --- default-off ---------------------------------------------------------------

def test_no_allocation_builds_no_auction():
    sim = build_simulation()
    assert sim.allocation is None and sim.cbaa_runtime is None


def test_a_rule_episode_reports_an_empty_allocation_and_no_defend_targets():
    sim = run_quiet(build_simulation())
    assert sim.episode_outcome()["allocation"] == {}
    assert all(st["defend_targets"] == {} for st in sim.recorder.steps)


@pytest.mark.parametrize("bad, match", [
    (dict(allocation="cbba"), "unknown allocation"),
    (dict(allocation="cbaa", belief_mode="ground_truth"), "belief_mode"),
    (dict(allocation="cbaa", belief_mode="shared", centralized_from_belief=True), "commander"),
])
def test_invalid_configurations_fail_at_construction(bad, match):
    with pytest.raises(ValueError, match=match):
        build_simulation(**bad)


@pytest.mark.parametrize("mode, transport", [
    ("shared", "ideal"), ("networked", "network"), ("local", "none"),
])
def test_the_transport_follows_the_belief_mode_by_default(mode, transport):
    assert build_simulation(belief_mode=mode, allocation="cbaa").cbaa_runtime.transport == transport


# --- non-vacuity guard ------------------------------------------------------------

def test_the_fixture_exercises_every_task_kind(shared_run):
    """Guard for everything below. If the fixture stops believing in contacts,
    the contact-task tests would pass by seeing nothing."""
    src = _sources(shared_run)
    assert src["screen"] > 0, "SCREEN never reached the controller"
    assert src["assigned"] > 0, "NEUTRALIZE never reached the controller"
    assert src["station"] > 0


# --- commitments reach the controller (the call site) --------------------------

def test_every_allocated_screen_is_what_the_controller_screened(shared_run):
    """The wiring test. At every step where an agent is allocated SCREEN(c) and
    still believes in c, the solver must have steered it at c as a screen --
    read from the effective-target log, which records what the controller
    actually did, not from the allocation."""
    checked = 0
    for st in shared_run.recorder.steps:
        for agent, (kind, ident) in st["defend_targets"].items():
            if kind != "contact":
                continue
            eff = st["effective_targets"][agent]
            if str(ident) not in st["beliefs"].get(agent, {}):
                continue  # falls back by design when the belief is gone
            assert eff["source"] == "screen" and eff["contact_id"] == ident, (
                f"step {st['step']} agent {agent}: allocated SCREEN({ident}), "
                f"controller did {eff}"
            )
            checked += 1
    assert checked > 50, "too few screen steps to mean anything"


def test_an_allocated_station_is_held_by_the_controller(shared_run):
    for st in shared_run.recorder.steps:
        for agent, (kind, _) in st["defend_targets"].items():
            if kind == "station":
                assert st["effective_targets"][agent]["source"] == "station"


# --- the rule that an auction may span planning epochs: asynchronous, spanning epochs ------------------------------------------

def test_commitments_change_between_planning_epochs(shared_run):
    """The rule that an auction may span planning epochs
    settles, not at the epoch boundary. So some assignment changes must land at
    steps that are not planning steps -- otherwise the auction is being run as
    virtual rounds inside one epoch, which is the rule that an auction may span planning epochs's REJECTED option."""
    interval = shared_run.planning_interval_steps
    history = shared_run.role_assignment_history
    off_epoch = [
        t for t in range(1, len(history))
        if history[t] != history[t - 1] and (t - 1) % interval != 0
    ]
    assert off_epoch, "every role change happened at a planning step"


def test_before_the_first_auction_settles_every_agent_defends(shared_run):
    assert all(
        r.value == "defend" for r in shared_run.role_assignment_history[1].values()
    )


# --- conflict freedom where it should hold, and logging where it cannot ---------

def test_ideal_comms_with_shared_beliefs_is_conflict_free_at_every_epoch(shared_run):
    """The auction analogue of M3's equivalence gate: identical beliefs over a
    complete, lossless graph must reach consensus by every epoch boundary."""
    log = shared_run.cbaa_runtime.conflict_log
    assert log, "no epoch boundary was sampled"
    assert all(entry["conflicts"] == {} for entry in log)


def test_no_communication_is_independent_greedy():
    sim = _run(belief_mode="local", allocation="cbaa")
    rt = sim.cbaa_runtime
    assert sim.episode_outcome()["allocation"]["auction_messages_sent"] == 0
    for i, agent in rt.agents.items():
        if sim.blue_disabled[i] or not agent.bids:
            continue
        assert agent.assignment == max(agent.order, key=lambda t: (agent.bids[t], -agent.order.index(t)))


# --- the rule that auction traffic never creates a belief at episode level --------------------------------------------------------

def test_no_agent_bids_on_a_contact_absent_from_its_own_belief_store(monkeypatch):
    """The rule that auction traffic never creates a belief, at the call site. Under a finite range beliefs diverge, so
    auction traffic names contacts some receivers do not believe in. Checked
    against each agent's ACTUAL belief store at the moment its epoch starts --
    not against the view handed to the auction, because a call-site bug that
    handed an agent someone else's view would pass a view-based check."""
    from coordination import runtime as rt

    sim = build_simulation(**EXERCISED, belief_mode="networked", r_comm=20.0, allocation="cbaa")
    original = rt.CBAARuntime.start_epoch
    violations, checked = [], [0]

    def spy(self, step, views, max_age=None, modifiers=None):
        original(self, step, views, max_age, modifiers)
        for i in views:
            own = set(sim.belief_stores[i].targets)
            for task in self.agents[i].order:
                if task.startswith(("NEUTRALIZE", "SCREEN")):
                    checked[0] += 1
                    if int(task.split(":")[1]) not in own:
                        violations.append((step, i, task))

    monkeypatch.setattr(rt.CBAARuntime, "start_epoch", spy)
    run_quiet(sim)
    assert checked[0] > 0, "no contact task was ever generated -- vacuous"
    assert violations == []
    assert sim.episode_outcome()["allocation"]["foreign_entries_ignored"] > 0, (
        "beliefs never diverged enough for the channel to exist -- the test "
        "would prove nothing about refusing it"
    )


# --- transport timing ------------------------------------------------------------

def test_ideal_transport_matches_the_network_round_for_round_at_infinite_range():
    """'ideal' is meant to differ from a networked R=inf auction in graph and
    losses only, never in timing. Same message count and the same one-round
    epoch handover (the same stale drops) is the check that it does."""
    a = _run(sim_steps=150, belief_mode="shared", allocation="cbaa").episode_outcome()["allocation"]
    b = _run(sim_steps=150, belief_mode="networked", allocation="cbaa").episode_outcome()["allocation"]
    assert a["auction_messages_sent"] == b["auction_messages_sent"] > 0
    assert a["stale_messages_dropped"] == b["stale_messages_dropped"] > 0


def test_allocation_metrics_are_surfaced(shared_run):
    m = shared_run.episode_outcome()["allocation"]
    for key in ("transport", "epochs", "rounds", "mean_rounds_to_stable", "recommitments",
                "epochs_with_conflicts", "stale_messages_dropped", "foreign_entries_ignored"):
        assert key in m
    assert m["epochs"] == 30 and m["transport"] == "ideal"
