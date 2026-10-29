"""The per-intruder LLM planner (planning/intruder_llm.py) and the commander's
LLM decider (coordination/commander_llm.py), with FAKE clients only: no
model, no network, no cost. Interface and failure semantics, not quality."""

import json
import threading
import time

import numpy as np
import pytest

from conftest import capture_fd_output, fingerprint
from control.gates import APPROACH, INTERIOR
from coordination.commander import CommanderTaskPlanner
from coordination.commander_llm import LLMCommanderDecider, commander_geometry
from coordination.tasks import TaskContext
from planning.intruder import ATTACK, EVADE, IntruderDecision, IntruderObservation
from planning.intruder_llm import LLMIntruderTeam, _schema, system_prompt, user_message
from planning.local_llm import ReplayMiss
from planning.response_cache import ResponseCache
from scripts import network_conditions
from scripts.experiment_policy import (
    DESIGN_POLICY, NEW_EXPERIMENT_INTRUDER_INFO, NEW_EXPERIMENT_PURSUIT, NEW_EXPERIMENT_SAFETY,
    NEW_EXPERIMENT_TASKS,
)
from scripts.episode_builder import build_episode


class _Msg:
    def __init__(self, content):
        self.message = type("m", (), {"content": content})()


class FakeClient:
    """policy(system, user, schema) -> dict | str | None | Exception."""

    def __init__(self, policy, jitter=False):
        self.policy, self.jitter = policy, jitter
        self.calls, self._lock = [], threading.Lock()
        outer = self

        class _Completions:
            def create(self, **kw):
                system, user = kw["messages"][0]["content"], kw["messages"][1]["content"]
                schema = kw["response_format"]["json_schema"]["schema"]
                with outer._lock:
                    outer.calls.append((system, user, schema))
                if outer.jitter:
                    time.sleep((hash(user) % 7) * 0.002)
                ans = outer.policy(system, user, schema)
                if isinstance(ans, Exception):
                    raise ans
                content = ans if ans is None or isinstance(ans, str) else json.dumps(ans)
                return type("r", (), {"choices": [_Msg(content)]})()

        self.chat = type("chat", (), {})()
        self.chat.completions = _Completions()


def _obs(phase=APPROACH, believed=None, agent=0):
    return IntruderObservation(
        agent_id=agent, now=50, dt=0.1, own_state=(40.0, 60.0, 1.0, 0.5, 0.0), phase=phase,
        commitment=None, teammates={1: (45.0, 55.0, 0.0, 1.0, 0.0)},
        teammate_intents={1: (IntruderDecision(ATTACK, gate_id=2), 40)},
        believed_blues=believed or {3: ((60.0, 70.0, 0.0, 0.0, 0.0), 44)},
        gates=((0, (69.4, 82.0)), (1, (70.2, 75.2)), (2, (74.0, 68.2)), (3, (82.0, 68.6))),
        defense_center=(82.0, 82.0))


# ---------------------------------------------------------------------------
# Intruder planner
# ---------------------------------------------------------------------------


def test_the_schema_and_prompt_are_phase_specific():
    assert "gate_id" in _schema(APPROACH)["properties"]["target_id"]["description"]
    assert "gate_id" not in _schema(INTERIOR)["properties"]["target_id"]["description"]
    assert "gate" in system_prompt(APPROACH, 400, 0.1) and "no gates" in system_prompt(INTERIOR, 400, 0.1)
    assert "phase: interior" in user_message(_obs(INTERIOR))
    assert "gate 0 at" in user_message(_obs(APPROACH))
    assert "gate 0 at" not in user_message(_obs(INTERIOR))


def test_the_prompt_holds_only_this_intruders_observation():
    um = user_message(_obs())
    assert "you: red 0" in um and "red 1 pos" in um
    assert "chose attack gate 2 at step 40" in um       # previous-epoch intent
    assert "blue 3 predicted" in um and "seen 6 steps ago" in um
    assert "blue 0" not in um and "blue 1 " not in um    # unseen blues absent


def _team(policy, **kw):
    team = LLMIntruderTeam(client=FakeClient(policy), cache=ResponseCache(enabled=False), **kw)
    team.set_episode_horizon(400, 0.1)
    return team


def test_answers_become_decisions_and_failures_become_named_reasons():
    team = _team(lambda s, u, sc: {"reasoning": "r", "action": "attack", "target_id": 1})
    assert team.decide_all([_obs()])[0] == IntruderDecision(ATTACK, gate_id=1, reasoning="r")
    team = _team(lambda s, u, sc: {"reasoning": "r", "action": "EVADE", "target_id": 3})
    assert team.decide_all([_obs(INTERIOR)])[0] == IntruderDecision(EVADE, evade_blue_id=3,
                                                                     reasoning="r")
    assert _team(lambda s, u, sc: "not json").decide_all([_obs()])[0].startswith("schema_failure")
    assert _team(lambda s, u, sc: None).decide_all([_obs()])[0] == "empty_response"
    assert _team(lambda s, u, sc: RuntimeError("down")).decide_all([_obs()])[0].startswith(
        "call_failure")


def test_each_intruder_has_its_own_planner_and_concurrency_changes_nothing():
    def policy(s, u, sc):
        me = int(u.split("you: red ")[1].split()[0])
        return {"reasoning": "r", "action": "attack", "target_id": me % 4}
    obs = [_obs(agent=a) for a in range(4)]
    serial = LLMIntruderTeam(client=FakeClient(policy), cache=ResponseCache(enabled=False))
    pooled = LLMIntruderTeam(client=FakeClient(policy, jitter=True),
                             cache=ResponseCache(enabled=False), max_workers=4)
    serial.set_episode_horizon(400, 0.1)
    pooled.set_episode_horizon(400, 0.1)
    assert serial.decide_all(obs) == pooled.decide_all(obs)
    assert len({id(p) for p in pooled.planners.values()}) == 4


def _gated_cfg(red_planner, **extra):
    belief, cfg = network_conditions.episode_cfg("multi_bearing", "REF", "C")
    cfg = dict(cfg, **NEW_EXPERIMENT_PURSUIT, **NEW_EXPERIMENT_SAFETY,
               **NEW_EXPERIMENT_INTRUDER_INFO, **NEW_EXPERIMENT_TASKS,
               intruder_team="gates", red_planner=red_planner, steps=60, **extra)
    return belief, cfg


def _gated_sim(red_planner):
    sim = build_episode(3706, *_gated_cfg(red_planner))
    red_planner.set_episode_horizon(sim.sim_steps, sim.dt)
    return sim


def _toward_gate_policy(s, u, sc):
    approach = "gate_id to attack" in sc["properties"]["target_id"]["description"]
    return {"reasoning": "go", "action": "attack", "target_id": 2 if approach else -1}


def test_an_llm_intruder_episode_runs_and_replays_from_the_cache_without_calls(tmp_path):
    cache = ResponseCache(str(tmp_path))
    live = LLMIntruderTeam(client=FakeClient(_toward_gate_policy), cache=cache, max_workers=2)
    a = _gated_sim(live)
    with capture_fd_output():
        a.run()
    entries = [e for e in a.intruder_log if "decision" in e]
    assert entries and all(e["fallback"] is None for e in entries)
    assert live.diagnostics()["calls"] > 0
    replay = LLMIntruderTeam(cache=ResponseCache(str(tmp_path)), replay_only=True)
    b = _gated_sim(replay)
    with capture_fd_output():
        b.run()
    assert fingerprint(a) == fingerprint(b) and replay.diagnostics()["calls"] == 0


def test_a_replay_miss_stops_the_run_rather_than_falling_back(tmp_path):
    replay = LLMIntruderTeam(cache=ResponseCache(str(tmp_path)), replay_only=True)
    sim = _gated_sim(replay)
    with pytest.raises(ReplayMiss), capture_fd_output():
        sim.run()


def test_an_invalid_gate_from_the_model_falls_back_and_is_logged():
    bad = LLMIntruderTeam(client=FakeClient(
        lambda s, u, sc: {"reasoning": "r", "action": "attack", "target_id": 9}),
        cache=ResponseCache(enabled=False))
    sim = _gated_sim(bad)
    with capture_fd_output():
        sim.run()
    e = next(e for e in sim.intruder_log if "decision" in e)
    assert e["invalid_reason"] == "approach_attack_needs_a_known_gate"
    assert e["fallback"] == "attack_gate" and e["decision"].gate_id in range(4)


# ---------------------------------------------------------------------------
# Commander decider
# ---------------------------------------------------------------------------


def _commander_cfg(decider, **extra):
    belief, cfg = network_conditions.episode_cfg("multi_bearing", "REF", "C")
    cfg = {k: v for k, v in cfg.items() if not k.startswith("cbaa_") and k != "allocation"}
    cfg.update(NEW_EXPERIMENT_PURSUIT, **NEW_EXPERIMENT_SAFETY, **NEW_EXPERIMENT_INTRUDER_INFO,
               **DESIGN_POLICY, per_agent_planning=False, centralized_from_belief=True,
               deliver_assignments_over_network=True, cbaa_task_set="three_task",
               commander_planner=CommanderTaskPlanner(decider), steps=60, **extra)
    return belief, cfg


def _commander_sim(decider):
    """The episode, with the static geometry the commander's prompt states."""
    sim = build_episode(3705, *_commander_cfg(decider))
    decider.geometry = commander_geometry(
        sim, TaskContext.from_simulation(sim, slots=sim.commander_slots, task_set="three_task"))
    return sim


def _agents(sc):
    return [int(a) for a in sc["properties"]["task_by_agent"]["required"]]


def _own_stations(s, u, sc):
    return {"reasoning": "hold", "task_by_agent": {str(a): f"DEFEND:{a}" for a in _agents(sc)}}


def test_the_commander_schema_requires_exactly_one_task_per_active_agent():
    client = FakeClient(_own_stations)
    sim = _commander_sim(LLMCommanderDecider(client=client, cache=ResponseCache(enabled=False)))
    with capture_fd_output():
        sim.run()
    assert client.calls
    for (system, _, schema), report in zip(client.calls, sim.role_planner.reports):
        tba = schema["properties"]["task_by_agent"]
        assert schema["required"] == ["reasoning", "task_by_agent"]
        assert sorted(tba["required"]) == sorted(tba["properties"]) == sorted(
            str(a) for a in report["retained"])
        assert tba["additionalProperties"] is False
        for prop in tba["properties"].values():
            assert prop["enum"] == report["catalogue"]
            assert not any(t.startswith("SCREEN") for t in prop["enum"])
        assert "task_by_agent" in system
        assert report["fallback"] == {} and report["failures"] == []
    # Reasoning comes first under schema order AND under alphabetical order.
    assert sorted(schema["properties"]) == list(schema["properties"])


def test_a_failed_commander_call_falls_back_for_every_agent_and_is_logged():
    sim = _commander_sim(LLMCommanderDecider(
        client=FakeClient(lambda s, u, sc: RuntimeError("down")),
        cache=ResponseCache(enabled=False)))
    with capture_fd_output():
        sim.run()
    r = sim.role_planner.reports[0]
    assert r["response"] is None and r["error"] and r["retained"] == {}
    assert r["fallback"] == {a: f"DEFEND:{a}" for a in r["fallback"]}


def test_duplicate_claims_from_the_model_keep_the_lowest_agent():
    def dup(s, u, sc):
        return {"reasoning": "all to 3", "task_by_agent": {str(a): "DEFEND:3" for a in _agents(sc)}}
    sim = _commander_sim(LLMCommanderDecider(
        client=FakeClient(dup), cache=ResponseCache(enabled=False)))
    with capture_fd_output():
        sim.run()
    r = sim.role_planner.reports[0]
    low = min(r["retained"])
    assert r["retained"] == {low: "DEFEND:3"}
    assert all(why == "capacity_conflict" for _, why, _ in r["failures"])


def test_a_commander_replay_miss_stops_the_run(tmp_path):
    sim = _commander_sim(LLMCommanderDecider(cache=ResponseCache(str(tmp_path)), replay_only=True))
    with pytest.raises(ReplayMiss), capture_fd_output():
        sim.run()


# -- the commander answer is validated against the catalogue and the active agents ----------


def test_the_validator_rules_and_version_in_the_key():
    from planning.semantic_validation import validate

    dec = LLMCommanderDecider()
    ctx = {"kind": "commander", "commander": 0, "catalogue": ["DEFEND:0", "DEFEND:1", "DEFEND:2",
                                                              "NEUTRALIZE:5:0"],
           "active_agents": [0, 1, 2], "prompt_version": "commander.10"}
    # duplicate task: lowest agent keeps it, the other falls back -> partial
    dup = json.dumps({"reasoning": "x", "task_by_agent": {"0": "NEUTRALIZE:5:0",
                                                          "1": "NEUTRALIZE:5:0", "2": "DEFEND:2"}})
    v = validate(dup, ctx)
    assert v["status"] == "partial" and v["applied"] == {"0": "NEUTRALIZE:5:0", "1": "DEFEND:1",
                                                         "2": "DEFEND:2"}
    assert v["drops"] == ["capacity_conflict"]
    # a missing agent (schema not enforced by a provider) -> that agent's station
    v = validate(json.dumps({"reasoning": "x", "task_by_agent": {"0": "DEFEND:0", "1": "DEFEND:1"}}), ctx)
    assert v["status"] == "partial" and v["drops"] == ["missing"]
    # an unknown task id -> station, logged
    v = validate(json.dumps({"reasoning": "x", "task_by_agent": {"0": "RECON:9", "1": "DEFEND:1",
                                                                 "2": "DEFEND:2"}}), ctx)
    assert v["drops"] == ["unknown_task"]
    # unusable: non-integer key, malformed JSON, empty -> every agent falls back
    for bad in (json.dumps({"reasoning": "x", "task_by_agent": {"zero": "DEFEND:0"}}), "{", ""):
        v = validate(bad, ctx)
        assert v["status"] == "fallback" and v["n_fallback"] == 3
    # an array-shaped answer is not an answer
    v = validate(json.dumps({"reasoning": "x", "assignments": []}), ctx)
    assert v["status"] == "fallback"
    assert dec._PROMPT_VERSION == "commander.10"
