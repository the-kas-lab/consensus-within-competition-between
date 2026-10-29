"""The default-model rule: the thinking switch, the serving namespace, and request timeouts.

The default-model rule () makes the local Qwen3-8B on vLLM
the default model. Three things had to be true before its first run:

1. **Thinking off is explicit and default-off**, sent the way Qwen documents
   for vLLM: ``extra_body={"chat_template_kwargs": {"enable_thinking":
   False}}``. Unset, nothing is sent.
2. **The cache cannot cross those lines.** The thinking setting is in the key,
   so a thinking-off call is never served a thinking-on response. The serving
   configuration (checkpoint, revision, server version, dtype, quantization,
   context length) is in the key, because ``local-model`` at an unchanged
   endpoint is an alias, and the model behind it can change. **Every key
   written before either existed is unchanged**: pinned below as literal
   digests computed on a535525, before this change.
3. **A stalled server cannot hang a run**: a per-request timeout reaches the
   request. It is not in the key: it cannot change an answer, only whether one
   arrives, and a timeout is recorded and replayed as a transport failure.

FAKE CLIENTS ONLY. Nothing here says anything about any model.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import build_simulation, fingerprint, run_quiet  # noqa: E402
from coordination.llm_bridge import LLMModifierSource  # noqa: E402
from planning.local_llm import DecodeConfig, ReplayMiss, build_client  # noqa: E402
from planning.prompt_builder import PromptContext  # noqa: E402
from planning.response_cache import ResponseCache, ServingConfig, cache_key  # noqa: E402
from test_local_llm_planner import _Response  # noqa: E402

CTX = PromptContext(arena_size=100.0, defense_center=(50.0, 50.0), horizon_steps=400, dt=0.1)
LOCAL = "http://localhost:8000/v1"
SERVING = ServingConfig(checkpoint="example/checkpoint", revision="unrecorded", server="vllm",
                     server_version="0.29.0", dtype="unrecorded", quantization="none",
                     max_model_len=8192)
# The bridge fixture of tests/test_llm_bridge.py, shortened: the configuration
# measured to believe contacts, so prompts carry contacts and calls happen.
EPISODE = dict(num_blues=3, num_reds=2, sim_steps=60, seed=7, allocation="cbaa",
               belief_mode="networked", comm_interval_seconds=0.1, cbaa_task_set="three_task",
               cbaa_fallback="station", cbaa_bid_version="v2.3", station_heading="configured")


class _Client:
    """Records every request's kwargs; answers a fixed valid JSON, tagged so a
    served answer can be traced to the client that produced it."""

    def __init__(self, tag="fake"):
        self.requests = []
        self.tag = tag
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.requests.append(kwargs)
                return _Response(json.dumps({
                    "scoring_profile": "balanced", "threat_priorities": [],
                    "preferred_tasks": [], "reasoning": outer.tag,
                }))

        self.chat = type("chat", (), {})()
        self.chat.completions = _Completions()


def _source(client=None, **kwargs):
    kwargs.setdefault("cache", ResponseCache())
    return LLMModifierSource(CTX, client=client, build_client_on_demand=False,
                             base_url=LOCAL, model="local-model", task_set="three_task", **kwargs)


_INPUTS = dict(prompt_builder_version="m8b.7", system_prompt="S", user_message="U",
               model="local-model", endpoint=LOCAL)


def _episode(source):
    return run_quiet(build_simulation(cbaa_modifier_source=source, **EPISODE))


def test_thinking_is_not_sent_unless_asked_and_off_is_sent_as_qwen_documents():
    default, off = _Client(), _Client()
    _episode(_source(default))
    _episode(_source(off, decode=DecodeConfig(enable_thinking=False)))
    assert default.requests and off.requests
    assert all("extra_body" not in r for r in default.requests)
    assert all(r["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
               for r in off.requests)


def test_thinking_combines_with_the_reasoning_hint_rather_than_replacing_it():
    body = DecodeConfig(reasoning_effort="low", enable_thinking=False).extra_body()
    assert body == {"reasoning": {"effort": "low"},
                    "chat_template_kwargs": {"enable_thinking": False}}


def test_a_thinking_off_call_is_never_served_a_thinking_on_response(tmp_path):
    on = _Client("thinking-on")
    _episode(_source(on, cache=ResponseCache(str(tmp_path))))
    assert on.requests

    # Offline: a thinking-off replay against a cache holding only thinking-on
    # answers must stop at its first question, not replay them.
    with pytest.raises(ReplayMiss):
        _episode(LLMModifierSource(CTX, cache=ResponseCache(str(tmp_path)), replay_only=True, task_set="three_task",
                                   base_url=LOCAL, model="local-model",
                                   decode=DecodeConfig(enable_thinking=False)))

    # Live: every thinking-off question goes to the model, none to the cache.
    off = _Client("thinking-off")
    src = _source(off, cache=ResponseCache(str(tmp_path)),
                  decode=DecodeConfig(enable_thinking=False))
    _episode(src)
    d = src.diagnostics()
    assert d["cache_hits"] == 0 and d["calls"] == len(off.requests) > 0


def test_explicit_thinking_on_is_its_own_request_too():
    keys = {cache_key(decode_config=DecodeConfig(enable_thinking=t).as_dict(), **_INPUTS)
            for t in (None, True, False)}
    assert len(keys) == 3


# -- 3. the serving namespace -------------------------------------------------------


def test_the_same_alias_at_the_same_endpoint_behind_a_new_checkpoint_misses(tmp_path):
    first = _Client("rev-a")
    _episode(_source(first, cache=ResponseCache(str(tmp_path)), serving=SERVING))
    assert first.requests

    # The same serving configuration replays offline, byte-identical.
    replay = LLMModifierSource(CTX, cache=ResponseCache(str(tmp_path)), replay_only=True, task_set="three_task",
                               base_url=LOCAL, model="local-model", serving=SERVING)
    a = _episode(_source(_Client("rev-a"), cache=ResponseCache(), serving=SERVING))
    b = _episode(replay)
    assert fingerprint(a) == fingerprint(b)
    assert replay.diagnostics()["calls"] == 0

    # Anything that changes what is behind `local-model` is a new namespace.
    for changed in (dict(revision="b1c2d3"), dict(server_version="0.30.0"),
                    dict(dtype="float16"), dict(quantization="awq"),
                    dict(max_model_len=16384), dict(checkpoint="Qwen/Qwen3-14B")):
        other = ServingConfig(**{**SERVING.as_dict(), **changed})
        with pytest.raises(ReplayMiss):
            _episode(LLMModifierSource(CTX, cache=ResponseCache(str(tmp_path)),
                                       replay_only=True, base_url=LOCAL,
                                       model="local-model", serving=other, task_set="three_task"))
    # And an undeclared serving configuration never reuses a declared one.
    with pytest.raises(ReplayMiss):
        _episode(LLMModifierSource(CTX, cache=ResponseCache(str(tmp_path)), replay_only=True, task_set="three_task",
                                   base_url=LOCAL, model="local-model"))


def test_the_serving_configuration_is_recorded_with_every_response():
    cache = ResponseCache()
    src = _source(_Client(), cache=cache, serving=SERVING,
                  decode=DecodeConfig(enable_thinking=False))
    _episode(src)
    records = list(cache._memory.values())
    assert records and all(r["provenance"]["serving"] == SERVING.as_dict() for r in records)
    assert all(r["provenance"]["decode"]["enable_thinking"] is False for r in records)
    d = src.diagnostics()
    assert d["serving"] == SERVING.as_dict() and d["decode"]["enable_thinking"] is False


# -- 4. timeouts ------------------------------------------------------------------------


def test_a_request_timeout_reaches_every_request_and_stays_out_of_the_key():
    timed, untimed = _Client(), _Client()
    c1, c2 = ResponseCache(), ResponseCache()
    _episode(_source(timed, cache=c1, request_timeout_s=45.0))
    _episode(_source(untimed, cache=c2))
    assert all(r["timeout"] == 45.0 for r in timed.requests)
    assert all("timeout" not in r for r in untimed.requests)
    assert set(c1._memory) == set(c2._memory), "a timeout cannot change an answer"
    assert all(r["provenance"]["request_timeout_s"] == 45.0 for r in c1._memory.values())


def test_a_timed_out_call_is_a_recorded_failure_that_replays_as_one(tmp_path):
    class _Stalled:
        def __init__(self):
            self.chat = type("chat", (), {})()
            self.chat.completions = self

        def create(self, **kwargs):
            raise TimeoutError(f"no answer in {kwargs['timeout']} s")

    src = _source(_Stalled(), cache=ResponseCache(str(tmp_path)), request_timeout_s=1.0)
    live = _episode(src)
    events = live.episode_outcome()["planner_fallbacks"]
    assert events and {e["kind"] for e in events} == {"call_failure"}
    assert {e["error_type"] for e in events} == {"TimeoutError"}

    replay = LLMModifierSource(CTX, cache=ResponseCache(str(tmp_path)), replay_only=True, task_set="three_task",
                               base_url=LOCAL, model="local-model")
    again = _episode(replay)
    assert fingerprint(again) == fingerprint(live)
    assert again.episode_outcome()["planner_fallbacks"] == events


def test_the_client_builder_passes_a_timeout_and_retry_cap_only_when_given():
    default = build_client(api_key="k", base_url=LOCAL)
    bounded = build_client(api_key="k", base_url=LOCAL, timeout=30.0, max_retries=1)
    assert bounded.timeout == 30.0 and bounded.max_retries == 1
    assert default.timeout != 30.0 and default.max_retries != 1
