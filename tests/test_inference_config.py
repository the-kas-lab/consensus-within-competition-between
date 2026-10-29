"""Backend request configuration, spend ledger, stop-the-run errors and offline
semantic validation (planning/request_config.py, planning/spend.py,
planning/stops.py, planning/semantic_validation.py). FAKE clients only: no
model, no network, no cost."""

import json
import os
import subprocess
import sys
import threading

import pytest

from conftest import capture_fd_output
from coordination.commander import CommanderTaskPlanner
from coordination.commander_llm import LLMCommanderDecider, commander_geometry
from coordination.llm_bridge import LLMModifierSource
from planning.intruder_llm import LLMIntruderTeam, observation_from_dict, observation_to_dict
from planning.local_llm import DecodeConfig, LocalLLMPlanner
from planning.prompt_builder import PromptContext
from planning.request_config import OPENROUTER, VLLM, ProviderRouting, RequestConfig, backend_of
from planning.response_cache import ResponseCache, cache_key
from planning.semantic_validation import validate
from planning.spend import BudgetExhausted, Price, SpendLedger
from planning.stops import StopRun
from planning.local_llm import ReplayMiss
from scripts import network_conditions
from coordination.tasks import TaskContext
from scripts.experiment_policy import (
    DESIGN_POLICY,
    NEW_EXPERIMENT_INTRUDER_INFO, NEW_EXPERIMENT_PURSUIT, NEW_EXPERIMENT_SAFETY,
    NEW_EXPERIMENT_TASKS,
)
from scripts.episode_builder import build_episode
from test_llm_teams import FakeClient, _obs

OR_URL = "https://openrouter.ai/api/v1"
PIN = ProviderRouting(("siliconflow/fp8",))


# ---------------------------------------------------------------------------
# request configuration
# ---------------------------------------------------------------------------


def test_openrouter_refuses_vllm_thinking_flags_and_unpinned_routing():
    rc = RequestConfig(OPENROUTER, routing=PIN, usage_accounting=True)
    with pytest.raises(ValueError, match="enable_thinking"):
        rc.check(DecodeConfig(enable_thinking=False), OR_URL)
    with pytest.raises(ValueError, match="pin a provider"):
        RequestConfig(OPENROUTER).check(DecodeConfig(), OR_URL)
    with pytest.raises(ValueError, match="OpenRouter request fields"):
        RequestConfig(VLLM, routing=PIN)
    with pytest.raises(ValueError, match="endpoint"):
        rc.check(DecodeConfig(), "http://localhost:8000/v1")
    body = rc.extra_body(DecodeConfig())
    assert body == {"provider": {"order": ["siliconflow/fp8"], "allow_fallbacks": False,
                                 "require_parameters": True},
                    "usage": {"include": True}}
    assert "chat_template_kwargs" not in json.dumps(body)
    # vLLM keeps exactly the DecodeConfig body (the thinking switch).
    d = DecodeConfig(enable_thinking=False)
    assert RequestConfig(VLLM).extra_body(d) == d.extra_body()
    assert backend_of(OR_URL) == OPENROUTER and backend_of("http://localhost:8000/v1") == VLLM


def test_a_planner_refuses_a_thinking_flag_bound_for_openrouter():
    with pytest.raises(ValueError, match="enable_thinking"):
        LocalLLMPlanner(0, None, endpoint=OR_URL, decode=DecodeConfig(enable_thinking=False),
                        request_config=RequestConfig(OPENROUTER, routing=PIN))


def _answer_client(answer, usage=None):
    calls = []

    class _C:
        def create(self, **kw):
            calls.append(kw)
            msg = type("m", (), {"content": json.dumps(answer)})()
            choice = type("c", (), {"message": msg, "finish_reason": "stop"})()
            r = type("r", (), {"choices": [choice], "usage": usage, "model": "m", "id": "x"})()
            r.model_dump = lambda: {"id": "x", "provider": "SiliconFlow",
                                    "choices": [{"native_finish_reason": "stop"}]}
            return r

    client = type("client", (), {})()
    client.chat = type("chat", (), {})()
    client.chat.completions = _C()
    return client, calls


def _planner(client, cache, rc=None, schema=None, ledger=None, endpoint=OR_URL):
    p = LocalLLMPlanner(0, None, model="qwen/x", client=client, cache=cache, endpoint=endpoint,
                        request_config=rc, ledger=ledger, request_timeout_s=30)
    p._SCHEMA = schema or {"type": "object", "properties": {"a": {"type": "integer"}}}
    return p


def test_request_config_and_schema_are_in_the_key_and_absent_keys_are_unchanged():
    client, _ = _answer_client({"a": 1})
    cache = ResponseCache()
    legacy = _planner(client, cache, endpoint="http://localhost:8000/v1")
    legacy._fetch("s", "u")
    assert legacy.last_key == cache_key(
        prompt_builder_version=legacy._PROMPT_VERSION, system_prompt="s", user_message="u",
        model="qwen/x", endpoint="http://localhost:8000/v1",
        decode_config=legacy.decode.as_dict())
    rc = RequestConfig(OPENROUTER, routing=PIN, usage_accounting=True)
    a = _planner(client, cache, rc)
    a._fetch("s", "u")
    b = _planner(client, cache, rc, schema={"type": "object", "properties": {"b": {}}})
    b._fetch("s", "u")
    c = _planner(client, cache, RequestConfig(OPENROUTER, routing=ProviderRouting(("nebius/fp8",)),
                                              usage_accounting=True))
    c._fetch("s", "u")
    assert len({a.last_key, b.last_key, c.last_key}) == 3
    assert cache.stats()["hits"] == 0


def test_the_raw_request_response_and_validation_context_are_recorded():
    class U:
        prompt_tokens, completion_tokens, cost = 100, 20, 0.0000123

    client, calls = _answer_client({"a": 1}, usage=U())
    cache = ResponseCache()
    rc = RequestConfig(OPENROUTER, routing=PIN, usage_accounting=True)
    p = _planner(client, cache, rc)
    p._fetch("sys", "usr", validation_context={"kind": "x", "n": 1})
    rec = cache.get_record(p.last_key)
    prov = rec["provenance"]
    sent = dict(calls[0])
    assert sent.pop("timeout") == 30
    assert prov["request_body"] == json.loads(json.dumps(sent))  # as sent, timeout aside
    assert prov["request_body"]["extra_body"]["provider"]["allow_fallbacks"] is False
    assert prov["raw_response"]["provider"] == "SiliconFlow"
    assert prov["request_config"] == rc.as_dict()
    assert prov["validation_context"] == {"kind": "x", "n": 1}
    assert prov["finish_reason"] == "stop"
    assert rec["usage"]["cost_usd"] == pytest.approx(0.0000123)


# ---------------------------------------------------------------------------
# spend ledger
# ---------------------------------------------------------------------------

PRICE = Price(1e-7, 3e-7)


def test_ledger_reserves_worst_case_settles_billed_and_refuses_past_the_cap():
    led = SpendLedger(0.01, PRICE)
    worst = led.worst_case_usd(1000, 2000)
    # default "bytes_upper": one token per byte + the request overhead
    assert worst == pytest.approx((1000 + 256) * 1e-7 + 2000 * 3e-7)
    assert SpendLedger(0.01, PRICE, reservation_rule="chars_half").worst_case_usd(
        1000, 2000) == pytest.approx(501 * 1e-7 + 2000 * 3e-7)
    with pytest.raises(ValueError):
        SpendLedger(0.01, PRICE, reservation_rule="guess")
    rid = led.reserve(1000, 2000)
    assert led.totals()["inflight_usd"] == pytest.approx(worst)
    assert led.settle(rid, "billed", 0.0001) == 0.0001
    t = led.totals()
    assert t["committed_usd"] == pytest.approx(0.0001) and t["n_inflight"] == 0
    tiny = SpendLedger(worst * 1.5, PRICE)
    tiny.reserve(1000, 2000)
    with pytest.raises(BudgetExhausted):
        tiny.reserve(1000, 2000)  # in-flight headroom counts against the cap
    assert tiny.totals()["refused_reservations"] == 1
    assert issubclass(BudgetExhausted, StopRun) and issubclass(ReplayMiss, StopRun)


def test_ledger_counts_retries_and_keeps_failed_attempts_at_their_worst_case():
    from openai import RateLimitError

    class R:
        status_code = 429
        headers = {}
        request = None

    state = {"n": 0}

    class U:
        prompt_tokens, completion_tokens, cost = 10, 5, 0.00002

    client, _ = _answer_client({"a": 1}, usage=U())
    real = client.chat.completions.create

    def flaky(**kw):
        state["n"] += 1
        if state["n"] == 1:
            raise RateLimitError("429", response=type("r", (), {"status_code": 429, "headers": {},
                                                               "request": None})(), body=None)
        return real(**kw)

    client.chat.completions.create = flaky
    led = SpendLedger(1.0, PRICE)
    p = _planner(client, ResponseCache(), RequestConfig(OPENROUTER, routing=PIN,
                                                        usage_accounting=True), ledger=led)
    p.max_rate_limit_retries = 2
    import planning.local_llm as ll
    orig_sleep = ll.time.sleep
    ll.time.sleep = lambda s: None
    try:
        p._fetch("s", "u")
    finally:
        ll.time.sleep = orig_sleep
    t = led.totals()
    assert t["attempts"]["not_billed"] == 1 and t["attempts"]["billed"] == 1
    assert t["committed_usd"] == pytest.approx(0.00002)
    rec = p.cache.get_record(p.last_key)
    assert [a["outcome"] for a in rec["provenance"]["attempts"]] == ["not_billed", "billed"]

    def broken(**kw):
        raise ConnectionError("reset")

    client.chat.completions.create = broken
    q = _planner(client, ResponseCache(), RequestConfig(OPENROUTER, routing=PIN,
                                                        usage_accounting=True), ledger=led)
    with pytest.raises(Exception):
        q._fetch("s2", "u2")
    t2 = led.totals()
    assert t2["attempts"]["unverified"] == 1
    assert t2["committed"]["unverified"] > 0  # conservatively kept at its worst case


def test_ledger_is_shared_across_threads_and_processes(tmp_path):
    path = str(tmp_path / "ledger.json")
    led = SpendLedger(10.0, PRICE, path=path)

    def work():
        for _ in range(20):
            led.settle(led.reserve(100, 10), "billed", 0.001)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from planning.spend import SpendLedger, Price\n"
            "led = SpendLedger(10.0, Price(1e-7, 3e-7), path=%r)\n"
            "for _ in range(20): led.settle(led.reserve(100, 10), 'billed', 0.001)\n"
            % (os.getcwd(), path))
    procs = [subprocess.Popen([sys.executable, "-c", code]) for _ in range(3)]
    assert all(p.wait() == 0 for p in procs)
    t = SpendLedger(10.0, PRICE, path=path).totals()
    assert t["attempts"]["billed"] == 140
    assert t["committed_usd"] == pytest.approx(0.14)


# ---------------------------------------------------------------------------
# BudgetExhausted stops the run; it is never a fallback
# ---------------------------------------------------------------------------


def _exhausted_ledger():
    return SpendLedger(1e-12, PRICE)


def _commander_sim(dec, steps):
    """A commander-team episode driven by ``dec``, with the geometry its prompt states."""
    belief, cfg = network_conditions.episode_cfg("multi_bearing", "REF", "C")
    cfg = {k: v for k, v in cfg.items() if not k.startswith("cbaa_") and k != "allocation"}
    planner = CommanderTaskPlanner(dec)
    cfg.update(NEW_EXPERIMENT_PURSUIT, **NEW_EXPERIMENT_SAFETY, **NEW_EXPERIMENT_INTRUDER_INFO,
               **DESIGN_POLICY, per_agent_planning=False, centralized_from_belief=True,
               deliver_assignments_over_network=True, cbaa_task_set="three_task",
               commander_planner=planner, steps=steps)
    sim = build_episode(3705, belief, cfg)
    dec.geometry = commander_geometry(
        sim, TaskContext.from_simulation(sim, slots=sim.commander_slots, task_set="three_task"))
    return sim, planner


def test_the_spend_cap_stops_intruder_commander_and_defender_paths_without_a_fallback():
    rc = RequestConfig(OPENROUTER, routing=PIN, usage_accounting=True)
    client = FakeClient(lambda s, u, sc: {"reasoning": "r", "action": "attack", "target_id": 1})
    team = LLMIntruderTeam(client=client, endpoint=OR_URL, request_config=rc,
                           ledger=_exhausted_ledger())
    team.set_episode_horizon(400, 0.1)
    with pytest.raises(BudgetExhausted):
        team.decide_all([_obs()])
    assert client.calls == []  # nothing was sent

    dec = LLMCommanderDecider(client=client, endpoint=OR_URL, request_config=rc,
                              ledger=_exhausted_ledger())
    sim, _ = _commander_sim(dec, 30)
    with pytest.raises(BudgetExhausted), capture_fd_output():
        sim.run()

    src = LLMModifierSource(PromptContext(arena_size=0.0), client=client, base_url=OR_URL,
                            task_set="three_task", request_config=rc,
                            ledger=_exhausted_ledger())
    belief, cfg = network_conditions.episode_cfg("multi_bearing", "REF", "C")
    cfg = dict(cfg, **NEW_EXPERIMENT_PURSUIT, **NEW_EXPERIMENT_SAFETY, **NEW_EXPERIMENT_TASKS,
               **DESIGN_POLICY, cbaa_modifier_source=src, steps=30)
    sim = build_episode(3705, belief, cfg)
    src.context = PromptContext.from_simulation(sim)
    with pytest.raises(BudgetExhausted), capture_fd_output():
        sim.run()
    assert src.fallback_events == []


# ---------------------------------------------------------------------------
# offline semantic validation == what the simulator applied live
# ---------------------------------------------------------------------------


def _h(text):
    return sum(ord(ch) for ch in text) % 7


def _bridge_policy(system, user, schema):
    k = _h(user)
    if k == 0:
        return "{not json"
    if k == 1:
        return ""
    if k == 2:
        return {"reasoning": "r", "scoring_profile": "reckless", "threat_priorities": [],
                "preferred_tasks": []}
    if k == 3:
        return {"reasoning": "r", "scoring_profile": "aggressive",
                "threat_priorities": [{"contact_id": 99, "priority": "HIGH"},
                                      {"contact_id": 0, "priority": "HIGH"},
                                      {"contact_id": 0, "priority": "LOW"}],
                "preferred_tasks": ["DEFEND:0", "SCREEN:0", "DEFEND:1", "DEFEND:2", "DEFEND:3"]}
    return {"reasoning": "r", "scoring_profile": "balanced", "threat_priorities": [],
            "preferred_tasks": []}


def _intruder_policy(system, user, schema):
    k = _h(user)
    approach = "gate_id to attack" in schema["properties"]["target_id"]["description"]
    if k == 0:
        return "nope"
    if k == 1:
        return {"reasoning": "r", "action": "evade", "target_id": 77}
    if k == 2 and approach:
        return {"reasoning": "r", "action": "attack", "target_id": 99}
    if k == 3 and not approach:
        return {"reasoning": "r", "action": "attack", "target_id": 1}
    return {"reasoning": "r", "action": "attack", "target_id": 1 if approach else -1}


def _commander_policy(system, user, schema):
    agents = sorted(int(a) for a in schema["properties"]["task_by_agent"]["required"])
    k = _h(user)
    if k == 0:
        return "]["
    if k == 1:  # duplicate claim, an unknown task, one agent missing
        return {"reasoning": "r", "task_by_agent": {str(agents[0]): "DEFEND:0",
                                                    str(agents[1]): "DEFEND:0",
                                                    str(agents[2]): "FOO:1"}}
    return {"reasoning": "r", "task_by_agent": {str(a): f"DEFEND:{a}" for a in agents}}


def _records(cache):
    return [r for r in cache._memory.values() if r["kind"] == "response"]


def test_offline_validation_reproduces_the_live_defender_and_intruder_outcomes():
    rc = RequestConfig(VLLM)
    red_cache, blue_cache = ResponseCache(), ResponseCache()
    team = LLMIntruderTeam(client=FakeClient(_intruder_policy), cache=red_cache,
                           request_config=rc)
    src = LLMModifierSource(PromptContext(arena_size=0.0), client=FakeClient(_bridge_policy),
                            cache=blue_cache, task_set="three_task", request_config=rc)
    belief, cfg = network_conditions.episode_cfg("multi_bearing", "REF", "C")
    cfg = dict(cfg, **NEW_EXPERIMENT_PURSUIT, **NEW_EXPERIMENT_SAFETY,
               **NEW_EXPERIMENT_INTRUDER_INFO, **NEW_EXPERIMENT_TASKS, **DESIGN_POLICY,
               intruder_team="gates", red_planner=team, cbaa_modifier_source=src, steps=120)
    sim = build_episode(3706, belief, cfg)
    src.context = PromptContext.from_simulation(sim)
    team.set_episode_horizon(sim.sim_steps, sim.dt)
    with capture_fd_output():
        sim.run()

    live_events = {}
    for e in src.fallback_events:
        live_events.setdefault((e["sim_step"], e["agent_id"]), []).append(e["kind"])
    live_mods = {(e["step"], e["agent_id"]): e["modifiers"] for e in src.modifier_log}
    statuses = set()
    for rec in _records(blue_cache):
        ctx = rec["provenance"]["validation_context"]
        got = validate(rec["content"], ctx)
        key = (ctx["now"], ctx["agent_id"])
        assert got["drops"] == live_events.get(key, []), key
        assert got["applied"] == live_mods[key], key
        statuses.add(got["status"])
    assert statuses == {"accepted", "partial", "fallback"}

    live_intr = {(e["step"], e["red"]): e for e in sim.intruder_log if "invalid_reason" in e}
    seen = set()
    for rec in _records(red_cache):
        ctx = rec["provenance"]["validation_context"]
        got = validate(rec["content"], ctx)
        live = live_intr[(ctx["now"], ctx["agent_id"])]
        if live["invalid_reason"] is None:
            assert got["status"] == "accepted"
        else:
            assert got["status"] == "fallback"
            assert got["invalid_reason"] == live["invalid_reason"]
            assert got["events"][0]["detail"] == (live["planner_reason"] or live["invalid_reason"])
            assert got["fallback"] == live["fallback"]
        d = live["decision"]
        assert got["applied"] == {"role": d.role, "gate_id": d.gate_id,
                                  "evade_blue_id": d.evade_blue_id}
        seen.add(got["status"])
    assert seen == {"accepted", "fallback"}


def test_offline_validation_reproduces_the_live_commander_outcomes():
    rc = RequestConfig(VLLM)
    cache = ResponseCache()
    dec = LLMCommanderDecider(client=FakeClient(_commander_policy), cache=cache,
                              request_config=rc)
    sim, planner = _commander_sim(dec, 120)
    with capture_fd_output():
        sim.run()
    by_now = {r["now"]: r for r in planner.reports}
    seen = set()
    for rec in _records(cache):
        ctx = rec["provenance"]["validation_context"]
        got = validate(rec["content"], ctx)
        live = by_now[ctx["now"]]
        assert [(f["agent"], f["kind"], f["detail"]) for f in got["events"]] == \
            [tuple(f) for f in live["failures"]]
        assert got["n_fallback"] == len(live["fallback"])
        seen.add(got["status"])
    assert seen == {"accepted", "partial", "fallback"}


def test_an_intruder_observation_round_trips_through_plain_data():
    obs = _obs()
    assert observation_from_dict(json.loads(json.dumps(observation_to_dict(obs)))) == obs


# ---------------------------------------------------------------------------
# intruder.2 (opt-in): one required integer target, no nullable field
# ---------------------------------------------------------------------------


def test_connection_errors_are_retried_and_recorded_timeouts_are_not():
    from openai import APIConnectionError, APITimeoutError

    import planning.local_llm as ll

    req = None  # the errors only store it
    client, _ = _answer_client({"a": 1})
    real = client.chat.completions.create
    state = {"n": 0}

    def flaky(**kw):
        state["n"] += 1
        if state["n"] == 1:
            raise APIConnectionError(request=req)
        return real(**kw)

    client.chat.completions.create = flaky
    orig_sleep = ll.time.sleep
    ll.time.sleep = lambda s: None
    try:
        p = _planner(client, ResponseCache(), RequestConfig(VLLM), endpoint="http://localhost:8000/v1")
        p._fetch("s", "u")
        rec = p.cache.get_record(p.last_key)
        assert rec["kind"] == "response"
        assert [a["outcome"] for a in rec["provenance"]["attempts"]] == ["unverified", "ok"]

        def timeout(**kw):
            raise APITimeoutError(request=req)

        client.chat.completions.create = timeout
        q = _planner(client, ResponseCache(), RequestConfig(VLLM), endpoint="http://localhost:8000/v1")
        with pytest.raises(Exception):
            q._fetch("s3", "u3")
        rec = q.cache.get_record(q.last_key)
        assert rec["kind"] == "transport_error" and rec["error_type"] == "APITimeoutError"
        assert len(rec["provenance"]["attempts"]) == 1
    finally:
        ll.time.sleep = orig_sleep
