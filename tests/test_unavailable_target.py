"""unavailable_target="station": an unavailable NEUTRALIZE target is executed the
same way in every defender arm. Rule planners only; free."""

import pytest

from conftest import capture_fd_output
from coordination.commander import CommanderTaskPlanner, NearestCommanderDecider
from coordination.runtime import output_for
from planning.intruder import GateRuleIntruderPlanner
from planning.roles import BlueRole
from scripts import network_conditions
from scripts.experiment_policy import (
    NEW_EXPERIMENT_EXECUTION, NEW_EXPERIMENT_INTRUDER_INFO, NEW_EXPERIMENT_PURSUIT,
    NEW_EXPERIMENT_SAFETY, NEW_EXPERIMENT_TASKS,
)
from scripts.episode_builder import build_episode

PLAN = {0: "DEFEND:0", 1: "NEUTRALIZE:0:0", 2: "DEFEND:2", 3: "DEFEND:3"}


def _sim(arm, unavailable="station", steps=20):
    common = dict(NEW_EXPERIMENT_PURSUIT, **NEW_EXPERIMENT_SAFETY, **NEW_EXPERIMENT_INTRUDER_INFO,
                  intruder_team="gates", red_planner=GateRuleIntruderPlanner(), steps=steps,
                  unavailable_target=unavailable)
    if arm in ("cbaa", "independent"):
        belief, cfg = network_conditions.episode_cfg("multi_bearing", "REF", "C" if arm == "cbaa" else "I")
        sim = build_episode(3705, belief, dict(cfg, **common, **NEW_EXPERIMENT_TASKS))
    else:
        belief, cfg = network_conditions.episode_cfg("multi_bearing", "REF", "C")
        cfg = {k: v for k, v in cfg.items() if not k.startswith("cbaa_") and k != "allocation"}
        cfg.update(common, per_agent_planning=False, centralized_from_belief=True,
                   deliver_assignments_over_network=True, cbaa_task_set="three_task",
                   commander_planner=CommanderTaskPlanner(NearestCommanderDecider()))
        sim = build_episode(3705, belief, cfg)
    with capture_fd_output():
        sim.run()
    return sim


def _plan():
    return output_for(PLAN, fallback="station", task_set="three_task")


@pytest.mark.parametrize("arm", ["cbaa", "independent", "commander"])
def test_a_resolved_committed_contact_holds_station_in_every_arm(arm):
    sim = _sim(arm)
    sim.red_outcomes[0] = "escaped"
    sim._apply_assignment(_plan(), 20, {1: {0: sim.red_state[0:4].copy()},
                                        2: {1: sim.red_state[5:9].copy()}})
    assert sim.role_assignment[1] == BlueRole.DEFEND
    assert sim.defend_target_override[1] == ("station", 1)
    assert 1 not in sim.target_override
    events = sim.commander_pending_log if arm == "commander" else sim.unavailable_target_log
    assert events[-1]["event"] in ("commanded_contact_resolved", "committed_contact_resolved")


@pytest.mark.parametrize("arm", ["cbaa", "independent", "commander"])
def test_a_committed_contact_without_a_local_belief_holds_station_in_every_arm(arm):
    sim = _sim(arm)
    sim._apply_assignment(_plan(), 20, {1: {}})
    assert sim.role_assignment[1] == BlueRole.DEFEND
    assert sim.defend_target_override[1] == ("station", 1)


def test_the_historical_default_keeps_neutralize_without_an_aim_point():
    sim = _sim("cbaa", unavailable="nearest_believed")
    sim.red_outcomes[0] = "escaped"
    sim._apply_assignment(_plan(), 20, {1: {0: sim.red_state[0:4].copy()}})
    assert sim.role_assignment[1] == BlueRole.NEUTRALIZE
    assert 1 not in sim.target_override  # the solver then falls back to nearest believed
    assert sim.unavailable_target_log == []


def test_an_available_target_is_untouched_and_logging_is_once_per_change():
    sim = _sim("cbaa")
    believed = {1: {0: sim.red_state[0:4].copy()}}
    sim._apply_assignment(_plan(), 20, believed)
    assert sim.role_assignment[1] == BlueRole.NEUTRALIZE and sim.target_override_ids[1] == 0
    sim._apply_assignment(_plan(), 30, {1: {}})
    sim._apply_assignment(_plan(), 40, {1: {}})
    assert [e["event"] for e in sim.unavailable_target_log] == ["committed_contact_not_believed"]


def test_the_default_leaves_the_config_record_unchanged():
    a, b = _sim("cbaa", unavailable="nearest_believed", steps=5), _sim("cbaa", steps=5)
    assert "unavailable_target" not in a._init_args
    assert b._init_args["unavailable_target"] == "station"
    assert NEW_EXPERIMENT_EXECUTION == {"unavailable_target": "station"}
    with pytest.raises(ValueError, match="unavailable_target"):
        _sim("cbaa", unavailable="chase", steps=5)


@pytest.mark.parametrize("arm", ["cbaa", "commander"])
def test_a_restricted_reapplication_keeps_other_agents_neutralize(arm):
    """Between epochs only some agents are re-resolved (resolve_only); an
    untouched NEUTRALIZE agent with an available target must keep it."""
    sim = _sim(arm)
    believed = {1: {0: sim.red_state[0:4].copy()}}
    sim._apply_assignment(_plan(), 20, believed)
    assert sim.role_assignment[1] == BlueRole.NEUTRALIZE
    sim._apply_assignment(_plan(), 25, believed, resolve_only={2})
    assert sim.role_assignment[1] == BlueRole.NEUTRALIZE
    assert sim.target_override_ids[1] == 0
    assert sim.unavailable_target_log == []
