"""The task-model commander (coordination/commander.py) and the meaning of a
command through delivery: planner output -> command -> delivery -> accepted
commitment -> controller target. Rule-based, free; spent seeds 3705-3706."""

import numpy as np
import pytest

from conftest import build_simulation, capture_fd_output
from coordination.commander import (
    CAPACITY_CONFLICT, CONFLICTING_ENTRIES, MISSING, NOT_ACTIVE, UNKNOWN_TASK,
    UNUSABLE_RESPONSE, CommanderTaskPlanner, NearestCommanderDecider, own_station_task,
    validate_assignment,
)
from coordination.runtime import output_for
from coordination.tasks import generate_tasks
from planning.roles import BlueRole
from scripts import network_conditions
from scripts.experiment_policy import (
    NEW_EXPERIMENT_FALSIFIED_REPORTS, NEW_EXPERIMENT_INTRUDER_INFO, NEW_EXPERIMENT_PURSUIT,
    NEW_EXPERIMENT_SAFETY,
)
from scripts.episode_builder import build_episode

CAT = ["NEUTRALIZE:3:0", "NEUTRALIZE:3:1", "DEFEND:0", "DEFEND:1", "DEFEND:2",
       "DEFEND:3", "RECON:5"]


def _reasons(report):
    return {a: r for a, r, _ in report["failures"]}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_valid_assignments_are_retained_including_two_slots_of_one_contact():
    a, rep = validate_assignment(
        [(0, "NEUTRALIZE:3:0"), (1, "NEUTRALIZE:3:1"), (2, "RECON:5"), (3, "DEFEND:3")],
        CAT, [0, 1, 2, 3])
    assert a == {0: "NEUTRALIZE:3:0", 1: "NEUTRALIZE:3:1", 2: "RECON:5", 3: "DEFEND:3"}
    assert rep["fallback"] == {} and rep["failures"] == []


def test_missing_unknown_and_conflicting_entries_fall_back_per_agent():
    a, rep = validate_assignment(
        [(0, "NEUTRALIZE:3:0"), (1, "NEUTRALIZE:9:0"), (2, "RECON:5"), (2, "DEFEND:2"),
         (3, "DEFEND:3"), (3, "DEFEND:3")],  # identical repeat: not ambiguous
        CAT, [0, 1, 2, 3, 4])
    assert _reasons(rep) == {1: UNKNOWN_TASK, 2: CONFLICTING_ENTRIES, 4: MISSING}
    assert rep["retained"] == {0: "NEUTRALIZE:3:0", 3: "DEFEND:3"}
    assert a[1] == own_station_task(1) and a[2] == own_station_task(2) and a[4] == "DEFEND:4"


@pytest.mark.parametrize("task", ["NEUTRALIZE:3:0", "DEFEND:2", "RECON:5"])
def test_every_task_id_has_capacity_one_and_the_lowest_agent_keeps_it(task):
    a, rep = validate_assignment([(2, task), (0, task), (1, task)], CAT, [0, 1, 2])
    assert rep["retained"] == {0: task}
    assert _reasons(rep) == {1: CAPACITY_CONFLICT, 2: CAPACITY_CONFLICT}
    assert a[1] == "DEFEND:1" and a[2] == "DEFEND:2"


def test_a_fallback_station_overlapping_a_valid_assignment_is_recorded_not_repaired():
    a, rep = validate_assignment([(0, "DEFEND:1")], CAT, [0, 1])
    assert a == {0: "DEFEND:1", 1: "DEFEND:1"}
    assert rep["fallback_conflicts"] == [(1, "DEFEND:1", 0)]


def test_an_unusable_response_falls_back_for_every_active_agent():
    a, rep = validate_assignment(None, CAT, [0, 1, 2])
    assert a == {0: "DEFEND:0", 1: "DEFEND:1", 2: "DEFEND:2"}
    assert set(_reasons(rep).values()) == {UNUSABLE_RESPONSE}
    assert rep["response"] is None


def test_entries_for_inactive_agents_are_reported_and_ignored():
    a, rep = validate_assignment([(5, "DEFEND:0"), (0, "DEFEND:0")], CAT, [0])
    assert a == {0: "DEFEND:0"} and (5, NOT_ACTIVE, "DEFEND:0") in rep["failures"]


def test_the_response_failures_retained_and_fallbacks_are_logged_separately():
    raw = [(0, "NEUTRALIZE:3:0"), (1, "BOGUS")]
    _, rep = validate_assignment(raw, CAT, [0, 1])
    assert rep["response"] == raw
    assert set(rep) >= {"response", "failures", "retained", "fallback", "fallback_conflicts"}


# ---------------------------------------------------------------------------
# Episodes through the commander path
# ---------------------------------------------------------------------------


class ScriptedDecider:
    """Fixed per-agent choices where the catalogue offers them; own station
    otherwise. Records every catalogue and view it was shown."""

    def __init__(self, choose):
        self.choose = choose
        self.seen = []

    def decide(self, view, catalogue, positions):
        self.seen.append((view, [t.task_id for t in catalogue]))
        ids = {t.task_id for t in catalogue}
        out = []
        for agent in sorted(positions):
            want = self.choose(agent, catalogue)
            out.append((agent, want if want in ids else own_station_task(agent)))
        return out


def _commander_cfg(decider, family="multi_bearing", **extra):
    belief, cfg = network_conditions.episode_cfg(family, "REF", "C")
    cfg = {k: v for k, v in cfg.items() if not k.startswith("cbaa_") and k != "allocation"}
    cfg.update(NEW_EXPERIMENT_PURSUIT, **NEW_EXPERIMENT_SAFETY, **NEW_EXPERIMENT_INTRUDER_INFO,
               per_agent_planning=False, centralized_from_belief=True,
               deliver_assignments_over_network=True, cbaa_task_set="three_task",
               commander_planner=CommanderTaskPlanner(decider))
    cfg.update(extra)
    return belief, cfg


def _run(decider, seed=3705, **extra):
    belief, cfg = _commander_cfg(decider, **extra)
    sim = build_episode(seed, belief, cfg)
    with capture_fd_output():
        sim.run()
    return sim


def test_the_simulator_declares_three_task_and_disables_implicit_screening():
    belief, cfg = _commander_cfg(NearestCommanderDecider())
    sim = build_episode(3705, belief, cfg)
    assert sim.cbaa_task_set == "three_task" and sim.solver.defend_screening is False
    assert sim.commander_task_model is True


def test_a_task_model_commander_must_be_declared_with_its_task_set():
    belief, cfg = _commander_cfg(NearestCommanderDecider(), cbaa_task_set="four_task")
    with pytest.raises(ValueError, match="declare cbaa_task_set"):
        build_episode(3705, belief, cfg)


def test_legacy_role_planners_still_cannot_claim_three_task():
    with pytest.raises(ValueError, match="needs allocation='cbaa'"):
        build_simulation(cbaa_task_set="three_task")


def test_the_catalogue_comes_from_the_commanders_own_store_by_the_cbaa_generator():
    decider = ScriptedDecider(lambda a, cat: None)
    sim = _run(decider)
    assert decider.seen
    for view, ids in decider.seen:
        assert view.agent_id == sim.commander_id
        assert ids == [t.task_id for t in generate_tasks(view, sim.role_planner.task_ctx)]
        assert not any(i.startswith("SCREEN") for i in ids)


def _directed_targets(sim, agent):
    seen = []
    ctrl = sim.solver.defend_controller
    orig = ctrl.plan

    def spy(own_state, target_state, other_states, agent_id=None, directed_target=None, **kw):
        if agent_id == agent:
            seen.append(None if directed_target is None else np.array(directed_target, float))
        return orig(own_state, target_state, other_states, agent_id=agent_id,
                    directed_target=directed_target, **kw)

    ctrl.plan = spy
    return seen


def test_a_commanded_station_other_than_its_own_reaches_the_controller_after_delivery():
    decider = ScriptedDecider(lambda a, cat: "DEFEND:3" if a == 1 else f"DEFEND:{a if a != 3 else 1}")
    belief, cfg = _commander_cfg(decider)
    sim = build_episode(3705, belief, cfg)
    seen = _directed_targets(sim, 1)
    with capture_fd_output():
        sim.run()
    entries = [e for e in sim.command_log if e["agent"] == 1]
    assert entries and all(e["task_id"] == "DEFEND:3" for e in entries)
    assert all(e["via"] == "network" for e in entries)
    assert sim.agent_commitments[1].defend_target == ("station", 3)
    assert sim.agent_commitments[1].task_id == "DEFEND:3"
    station3 = np.asarray(sim.solver.defend_stations[3], float)
    first_applied = min(e["applied_at"] for e in entries if e["applied_at"] is not None)
    assert seen and np.allclose(seen[-1], station3)
    assert any(np.allclose(t, station3) for t in seen[first_applied:])


def test_neutralize_contact_and_slot_survive_delivery_and_logs():
    def pick(agent, cat):
        neut = [t.task_id for t in cat if t.task_id.startswith("NEUTRALIZE")]
        if agent == 2 and neut:
            return neut[0]
        return f"DEFEND:{agent}"
    sim = _run(ScriptedDecider(pick))
    neut = [e for e in sim.command_log if e["task_id"] and e["task_id"].startswith("NEUTRALIZE")]
    assert neut and all(e["agent"] == 2 for e in neut)
    accepted = [e for e in neut if e["outcome"] == "accepted"]
    assert accepted
    c, slot = accepted[-1]["task_id"].split(":")[1:]
    assert int(slot) == 0
    final = sim.agent_commitments[2]
    if final.task_id.startswith("NEUTRALIZE"):
        assert final.target_red_id == int(c)


def test_two_slots_of_one_contact_are_commanded_and_delivered_when_configured():
    def pick(agent, cat):
        neut = sorted(t.task_id for t in cat if t.task_id.startswith("NEUTRALIZE:"))
        first = neut[0].rsplit(":", 1)[0] if neut else None
        if first and agent in (1, 2):
            return f"{first}:{agent - 1}"
        return f"DEFEND:{agent}"
    sim = _run(ScriptedDecider(pick), cbaa_slots=2)
    by_agent = {}
    for e in sim.command_log:
        if e["task_id"] and e["task_id"].startswith("NEUTRALIZE") and e["outcome"] == "accepted":
            by_agent.setdefault(e["agent"], set()).add(e["task_id"].rsplit(":", 1)[1])
    assert by_agent.get(1) == {"0"} and by_agent.get(2) == {"1"}


def test_live_pursuit_and_pursued_heading_use_the_executing_defenders_own_belief():
    belief, cfg = _commander_cfg(NearestCommanderDecider())
    sim = build_episode(3705, belief, cfg)
    checked = []
    orig = sim.solver.solve_decentralized

    def spy(*a, **k):
        out = orig(*a, **k)
        aims = k["target_override"]
        believed = k["believed_targets"]
        for i, role in sim.role_assignment.items():
            if role != BlueRole.NEUTRALIZE or i not in aims:
                continue
            red = sim.target_override_ids[i]
            assert sim.neutralize_aim_sources[i] == "belief"
            np.testing.assert_allclose(aims[i][0:2], believed[i][red][0:2])
            assert sim.solver.bearing_sources[i] in ("pursued_detection", "pursued_belief")
            checked.append(i)
        return out

    sim.solver.solve_decentralized = spy
    with capture_fd_output():
        sim.run()
    assert checked


def test_commanded_neutralize_never_executes_the_nearest_believed_fallback():
    belief, cfg = _commander_cfg(NearestCommanderDecider())
    for seed in (3705, 3706):
        sim = build_episode(seed, belief, cfg)
        sources = []
        orig = sim.solver.solve_decentralized

        def spy(*a, **k):
            out = orig(*a, **k)
            # A blue disabled by its own contact keeps its last role and the
            # solver still computes a nominal for it, but it is held in place
            # (testbed: disabled blues' states are restored every step). Only
            # active agents execute anything; with the red-side blue rows
            # removed (safety.2) contacts now happen on these seeds.
            sources.extend(e["source"] for i, e in sim.solver.effective_targets.items()
                           if e["controller"] == "neutralize" and not sim.blue_disabled[i])
            return out

        sim.solver.solve_decentralized = spy
        with capture_fd_output():
            sim.run()
        assert sources and set(sources) == {"assigned"}


def test_undelivered_commands_leave_the_previous_commitment_in_force():
    sim = _run(NearestCommanderDecider(), r_comm=14.0)
    undelivered = [e for e in sim.command_log if e["via"] == "network" and e["outcome"] is None]
    assert undelivered, "need some undelivered commands for this check"
    for agent, command in sim.agent_commitments.items():
        accepted = [e["epoch"] for e in sim.command_log
                    if e["agent"] == agent and e["outcome"] == "accepted"]
        assert command.planning_epoch == max(accepted)


def test_a_commanded_contact_without_a_local_belief_is_pending_on_station_then_executes():
    sim = _run(NearestCommanderDecider(), steps=20)
    step = 20
    ledger_before = dict(sim.agent_commitments)
    red_state = sim.red_state[0:4].copy()
    plan = output_for({0: "DEFEND:0", 1: "NEUTRALIZE:0:0", 2: "DEFEND:2", 3: "DEFEND:3"},
                      fallback="station", task_set="three_task")
    sim._apply_assignment(plan, step, {})
    assert sim.role_assignment[1] == BlueRole.DEFEND
    assert sim.defend_target_override[1] == ("station", 1)
    assert 1 not in sim.target_override
    assert sim.commander_pending == {1: ("NEUTRALIZE:0:0", 0)}
    assert sim.commander_pending_log[-1]["event"] == "pending_no_local_belief"
    assert sim.agent_commitments == ledger_before  # the commanded task is kept
    sim._activate_ready_pending(step + 1, {1: {0: red_state}})
    assert sim.role_assignment[1] == BlueRole.NEUTRALIZE
    assert sim.target_override_ids[1] == 0
    np.testing.assert_allclose(sim.target_override[1][0:2], red_state[0:2])
    assert sim.commander_pending == {}


def test_a_resolved_commanded_contact_executes_station_not_another_contact():
    sim = _run(NearestCommanderDecider(), steps=20)
    sim.red_outcomes[0] = "escaped"
    plan = output_for({0: "DEFEND:0", 1: "NEUTRALIZE:0:0", 2: "DEFEND:2", 3: "DEFEND:3"},
                      fallback="station", task_set="three_task")
    sim._apply_assignment(plan, 20, {1: {0: sim.red_state[0:4].copy()}})
    assert sim.role_assignment[1] == BlueRole.DEFEND
    assert sim.defend_target_override[1] == ("station", 1)
    assert sim.commander_pending_log[-1]["event"] == "commanded_contact_resolved"


def test_falsified_reports_reach_the_commander_through_its_network_store():
    sim = _run(NearestCommanderDecider(), falsified_reports=NEW_EXPERIMENT_FALSIFIED_REPORTS)
    log = sim.attack_log
    assert log["sender"] != sim.commander_id
    assert log["target_red"] is not None
    assert any(d["receiver"] == sim.commander_id for d in log["deliveries"])


def test_the_commander_arm_draws_the_same_attacker_as_the_allocation_arms():
    senders = []
    belief, cfg = _commander_cfg(NearestCommanderDecider(),
                                 falsified_reports=NEW_EXPERIMENT_FALSIFIED_REPORTS)
    sim = build_episode(3705, belief, cfg)
    sim._reset_falsified_reports()
    senders.append(sim.attack_log["sender"])
    for arm in ("C", "I"):
        b, c = network_conditions.episode_cfg("multi_bearing", "REF", arm)
        s = build_episode(3705, b, dict(c, falsified_reports=NEW_EXPERIMENT_FALSIFIED_REPORTS))
        s._reset_falsified_reports()
        senders.append(s.attack_log["sender"])
    assert len(set(senders)) == 1
