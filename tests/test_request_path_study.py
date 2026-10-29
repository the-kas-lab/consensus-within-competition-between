"""Request-path additions for the multi-candidate model-selection study
: OpenRouter's reasoning object, the frozen retry policy, attempt
identity, billed token categories, and a ledger shared by several pinned
endpoints. FAKE clients only: no network, no cost."""

import json

import pytest

from planning.local_llm import DecodeConfig, LocalLLMPlanner, TransportFailure
from planning.request_config import OPENROUTER, VLLM, ProviderRouting, RequestConfig, RetryPolicy
from planning.response_cache import ResponseCache
from planning.spend import BudgetExhausted, Price, SpendLedger

OR_URL = "https://openrouter.ai/api/v1"
PIN = ProviderRouting(("xiaomi/fp8",))


def test_reasoning_object_is_canonical_openrouter_only_and_keyed_only_when_set():
    d = DecodeConfig(reasoning='{"enabled": false}')
    assert d.as_dict()["reasoning"] == '{"enabled": false}'
    assert "reasoning" not in DecodeConfig().as_dict()
    assert DecodeConfig(reasoning='{"b": 1, "a": 2}').reasoning == '{"a": 2, "b": 1}'
    with pytest.raises(ValueError):
        DecodeConfig(reasoning='{"effort": "low"}', reasoning_effort="low")
    with pytest.raises(ValueError):
        DecodeConfig(reasoning="[]")
    rc = RequestConfig(OPENROUTER, routing=PIN, usage_accounting=True)
    assert rc.extra_body(d)["reasoning"] == {"enabled": False}
    with pytest.raises(ValueError, match="refused for vLLM"):
        RequestConfig(VLLM).check(d, "http://localhost:8000/v1")


class _Usage:
    prompt_tokens, completion_tokens, cost = 100, 50, 0.00002
    completion_tokens_details = {"reasoning_tokens": 30}
    prompt_tokens_details = {"cached_tokens": 64, "cache_write_tokens": 0}


def _client(script):
    """``script``: a list of exceptions or 'ok' / 'empty' / 'abort' (a 200 whose
    choice finished "error" after a fragment), consumed per attempt."""
    calls = []

    class _C:
        def create(self, **kw):
            calls.append(kw)
            step = script.pop(0)
            if isinstance(step, Exception):
                raise step
            choices = [] if step == "empty" else [type("c", (), {
                "message": type("m", (), {"content": '{"a' if step == "abort" else '{"a": 1}'})(),
                "finish_reason": "error" if step == "abort" else "stop"})()]
            r = type("r", (), {"choices": choices, "usage": _Usage(), "model": "m", "id": "x",
                               "model_extra": {"error": {"code": 502}} if step == "empty" else {}})()
            r.model_dump = lambda: {"id": "x"}
            return r

    client = type("client", (), {})()
    client.chat = type("chat", (), {})()
    client.chat.completions = _C()
    return client, calls


def _status_error(code):
    from openai import APIStatusError

    resp = type("resp", (), {"status_code": code, "request": None, "headers": {}})()
    return APIStatusError("err", response=resp, body=None)


def _timeout():
    from openai import APITimeoutError

    return APITimeoutError(request=None)


def _planner(client, retry, ledger=None):
    rc = RequestConfig(OPENROUTER, routing=PIN, usage_accounting=True, retry=retry)
    p = LocalLLMPlanner(0, None, model="x/y", client=client, cache=ResponseCache(),
                        endpoint=OR_URL, request_config=rc, ledger=ledger, request_timeout_s=5)
    p._SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}}
    return p


def test_the_retry_policy_retries_provider_faults_and_records_every_attempt(monkeypatch):
    import time as _t

    monkeypatch.setattr(_t, "sleep", lambda s: None)
    client, calls = _client([_status_error(502), "empty", _timeout(), "ok"])
    p = _planner(client, RetryPolicy(server_error=1, empty_choices=1, timeout=1))
    assert p._fetch("s", "u") == '{"a": 1}'
    rec = p.cache._memory[p.last_key]
    kinds = [a.get("retry_kind") for a in rec["provenance"]["attempts"]]
    assert kinds == ["server_error", "empty_choices", "timeout", None]
    ids = [a["attempt_id"] for a in rec["provenance"]["attempts"]]
    assert len(set(ids)) == 4 and all(a["started_at"] for a in rec["provenance"]["attempts"])
    assert rec["usage"]["reasoning_tokens"] == 30 and rec["usage"]["cached_tokens"] == 64
    assert rec["provenance"]["request_config"]["retry"]["timeout"] == 1


def test_without_a_policy_provider_errors_are_not_retried_and_become_failures(monkeypatch):
    import time as _t

    monkeypatch.setattr(_t, "sleep", lambda s: None)
    client, calls = _client([_status_error(503), "ok"])
    p = _planner(client, None)
    with pytest.raises(TransportFailure):
        p._fetch("s", "u")
    assert len(calls) == 1
    # And a policy that exhausts its retries fails the same, recorded way.
    client, calls = _client([_status_error(503), _status_error(503)])
    with pytest.raises(TransportFailure):
        _planner(client, RetryPolicy(server_error=1))._fetch("s", "u")
    assert len(calls) == 2


def test_a_provider_abort_is_retried_under_the_policy_and_fails_when_spent(monkeypatch):
    import time as _t

    monkeypatch.setattr(_t, "sleep", lambda s: None)
    client, calls = _client(["abort", "ok"])
    p = _planner(client, RetryPolicy(provider_abort=2))
    assert p._fetch("s", "u") == '{"a": 1}'
    rec = p.cache._memory[p.last_key]
    assert [a.get("retry_kind") for a in rec["provenance"]["attempts"]] == ["provider_abort", None]
    assert rec["provenance"]["request_config"]["retry"]["provider_abort"] == 2
    # Spent retries: a transport failure, never an answer.
    client, calls = _client(["abort", "abort"])
    with pytest.raises(TransportFailure):
        _planner(client, RetryPolicy(provider_abort=1))._fetch("s", "u")
    assert len(calls) == 2
    # Policy without it (0, the default): the fragment is returned as before
    # and the key's retry block is unchanged.
    client, calls = _client(["abort"])
    p = _planner(client, RetryPolicy())
    assert p._fetch("s", "u") == '{"a'
    assert "provider_abort" not in RetryPolicy().limits()


def test_a_ledger_file_is_shared_by_endpoints_with_different_prices(tmp_path):
    path = str(tmp_path / "ledger.json")
    a = SpendLedger(1.0, Price(1e-7, 1e-6, source="a"), path=path)
    b = SpendLedger(1.0, Price(5e-7, 5e-6, source="b"), path=path)
    ra, rb = a.reserve(1000, 100), b.reserve(1000, 100)
    assert b.worst_case_usd(1000, 100) > a.worst_case_usd(1000, 100)
    a.settle(ra, "billed", 0.01)
    b.settle(rb, "billed", 0.02)
    t = a.totals()
    assert set(t["prices"]) == {"a", "b"} and t["committed_usd"] == pytest.approx(0.03)
    with pytest.raises(ValueError):
        a.raise_cap(0.001, "below committed")
    a.raise_cap(2.0, "checkpoint reconciled")
    assert SpendLedger(2.0, Price(1e-7, 1e-6, source="a"), path=path).totals()["cap_history"][0]["to"] == 2.0
    with pytest.raises(BudgetExhausted):
        SpendLedger(2.0, Price(1.0, 1.0, source="c"), path=path).reserve(10, 10)


def test_temperature_none_is_omitted_from_the_request():
    client, calls = _client(["ok"])
    rc = RequestConfig(OPENROUTER, routing=PIN, usage_accounting=True)
    p = LocalLLMPlanner(0, None, model="x/y", client=client, cache=ResponseCache(), endpoint=OR_URL,
                        request_config=rc, decode=DecodeConfig(temperature=None))
    p._SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}}
    p._fetch("s", "u")
    assert "temperature" not in calls[0] and calls[0]["max_tokens"] == 2000
    client, calls = _client(["ok"])
    p = LocalLLMPlanner(0, None, model="x/y", client=client, cache=ResponseCache(), endpoint=OR_URL,
                        request_config=rc)
    p._SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}}
    p._fetch("s", "u")
    assert calls[0]["temperature"] == 0.0
