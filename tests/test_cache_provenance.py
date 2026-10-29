"""Cache provenance, failure replay, and an explicitly offline replay mode.

Repairs required before any reported LLM batch:

* records must say what was asked, of which endpoint and model, and who answered;
* endpoint identity must be in the key, or OpenRouter and a local vLLM serving
  the same model id share entries -- and reported runs must be OpenRouter's;
* a transport exception must leave a replayable record. Before this, an episode
  that lost a call replayed as a DIFFERENT episode, because the replay asked the
  network again;
* replay must be a mode that refuses the network, not a run that happens to hit.

Every test that claims a property of a replay drives a live Simulation.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import build_simulation, fingerprint  # noqa: E402
from planning.local_llm import PerAgentLLMPlanner, ReplayMiss  # noqa: E402
from planning.planner import _OPENROUTER_BASE_URL  # noqa: E402
from planning.response_cache import (  # noqa: E402
    TRANSPORT_ERROR,
    ResponseCache,
    cache_key,
    legacy_cache_key_v1,
)
from planning.roles import BlueRole  # noqa: E402
from test_local_llm_planner import (  # noqa: E402
    CTX,
    DETECTING,
    _FakeClient,
    _RaisingClient,
    _Response,
    _view,
)

OPENROUTER = _OPENROUTER_BASE_URL
VLLM = "http://localhost:8000/v1"


def _team(client=None, cache=None, base_url=OPENROUTER, **kwargs):
    return PerAgentLLMPlanner(
        CTX,
        client=client,
        cache=cache if cache is not None else ResponseCache(),
        base_url=base_url,
        build_client_on_demand=False,
        **kwargs,
    )


def _episode(team):
    """A live Simulation driven by ``team``; returns the finished sim."""
    from conftest import run_quiet

    sim = build_simulation(per_agent_planning=True, role_planner=team, **DETECTING)
    return run_quiet(sim)


class _Flaky(_FakeClient):
    """Raises on every third call -- a transport failure mid-episode, the case
    that used to leave no record."""

    def __init__(self):
        super().__init__()
        self.n = 0

    def answer(self, user_message):
        self.n += 1
        if self.n % 3 == 0:
            raise ConnectionError("upstream reset")
        return super().answer(user_message)


# --- the key ---------------------------------------------------------------

def test_the_key_demands_an_endpoint():
    """No default: a caller that forgets who it asked must fail, not share."""
    with pytest.raises(TypeError):
        cache_key(prompt_builder_version="1", system_prompt="s", user_message="u", model="m")


def test_a_trailing_slash_does_not_split_one_endpoint_into_two():
    """The OpenAI client reports its base URL with a trailing slash."""
    common = dict(prompt_builder_version="1", system_prompt="s", user_message="u", model="m")
    assert cache_key(endpoint=OPENROUTER, **common) == cache_key(endpoint=OPENROUTER + "/", **common)


def test_openrouter_and_local_vllm_never_share_an_entry(tmp_path):
    cache_dir = str(tmp_path)
    view = _view()
    _team(client=_FakeClient(), cache=ResponseCache(cache_dir)).plan_one(view)

    local = _team(client=_FakeClient(), cache=ResponseCache(cache_dir), base_url=VLLM)
    local.plan_one(view)
    assert local.diagnostics()["calls"] == 1, "a local-vLLM run was served OpenRouter's answer"
    assert local.diagnostics()["endpoints_served"] == {VLLM: 1}


# --- provenance --------------------------------------------------------------

def test_a_record_says_what_was_asked_of_whom_and_who_answered(tmp_path):
    class _Answering(_FakeClient):
        def __init__(self):
            super().__init__()
            outer = self

            class _C:
                def create(self, **kwargs):
                    r = _Response(outer.answer(kwargs["messages"][1]["content"]))
                    r.model = "qwen/qwen3-30b-a3b-instruct-2507"
                    r.id = "gen-123"
                    r.provider = "SomeUpstream"
                    return r

            self.chat.completions = _C()

    team = _team(client=_Answering(), cache=ResponseCache(str(tmp_path)))
    team.plan_one(_view())

    (name,) = os.listdir(str(tmp_path))
    with open(os.path.join(str(tmp_path), name)) as f:
        record = json.load(f)
    prov = record["provenance"]
    assert record["kind"] == "response"
    assert prov["model"] == team.model
    assert prov["endpoint"] == OPENROUTER
    assert prov["key_schema"] == 2
    assert prov["prompt_builder_version"]
    assert prov["decode"]["max_tokens"] == team.decode.max_tokens
    assert prov["response_model"] == "qwen/qwen3-30b-a3b-instruct-2507"
    assert prov["response_id"] == "gen-123"
    assert prov["provider"] == "SomeUpstream"
    assert prov["request"]["user_message"] and prov["request"]["system_prompt"]
    assert prov["recorded_at"]


# --- failures are records -----------------------------------------------------

def test_a_transport_exception_is_recorded_not_just_survived(tmp_path):
    class _Broken(_FakeClient):
        def answer(self, user_message):
            raise ConnectionError("upstream reset")

    team = _team(client=_Broken(), cache=ResponseCache(str(tmp_path)))
    out = team.plan_one(_view())

    assert out.roles == {0: BlueRole.DEFEND}
    assert team.fallback_events[0]["error_type"] == "ConnectionError"
    (name,) = os.listdir(str(tmp_path))
    with open(os.path.join(str(tmp_path), name)) as f:
        record = json.load(f)
    assert record["kind"] == TRANSPORT_ERROR
    assert record["error_type"] == "ConnectionError"
    assert record["provenance"]["endpoint"] == OPENROUTER


def test_an_episode_with_transport_failures_replays_offline_as_the_same_episode(tmp_path):
    """Failures reproduce: a recorded transport failure replays as a failure.

    Live: every third call raises. Replay: replay_only, no client. The replay
    must make no call, raise no ReplayMiss, and produce the identical episode
    and the identical fallback record.
    """
    cache_dir = str(tmp_path)
    live_team = _team(client=_Flaky(), cache=ResponseCache(cache_dir))
    live = _episode(live_team)
    live_events = live.episode_outcome()["planner_fallbacks"]
    assert any(e["kind"] == "call_failure" for e in live_events), "fixture must fail some calls"

    replay_team = _team(cache=ResponseCache(cache_dir), replay_only=True)
    replay = _episode(replay_team)

    assert fingerprint(replay) == fingerprint(live)
    assert replay.episode_outcome()["planner_fallbacks"] == live_events
    d = replay_team.diagnostics()
    assert d["calls"] == 0 and d["mode"] == "replay_only"
    assert d["transport_failures"] == live_team.diagnostics()["transport_failures"] > 0


# --- replay is a mode that refuses the network ----------------------------------

def test_replay_only_refuses_a_client():
    with pytest.raises(ValueError, match="refuses a client"):
        _team(client=_RaisingClient(), replay_only=True)


def test_replay_only_builds_no_client_even_when_asked_to():
    team = PerAgentLLMPlanner(CTX, replay_only=True, build_client_on_demand=True)
    assert team.client is None
    assert team.planner_for(0).client is None


def test_a_replay_miss_stops_the_episode_instead_of_becoming_a_fallback():
    """A replay that silently turned a miss into DEFEND would be a different
    episode passed off as the recorded one. Asserted through a live Simulation,
    so the call site cannot swallow what the method raises."""
    with pytest.raises(ReplayMiss):
        _episode(_team(replay_only=True))


# --- migration of schema-1 entries -------------------------------

def _write_v1_entry(cache_dir, view, team):
    from planning.prompt_builder import (
        PROMPT_BUILDER_VERSION,
        build_system_prompt,
        build_user_message,
    )

    key = legacy_cache_key_v1(
        prompt_builder_version=PROMPT_BUILDER_VERSION,
        system_prompt=build_system_prompt(CTX),
        user_message=build_user_message(view, CTX),
        model=team.model,
        decode_config=team.decode.as_dict(),
    )
    content = json.dumps({"role": "defend", "target_contact_id": -1,
                          "target_region_id": -1, "reasoning": "v1"})
    with open(os.path.join(cache_dir, f"{key}.json"), "w") as f:
        json.dump({"content": content, "usage": {"prompt_tokens": 5}}, f)


def test_a_schema_1_entry_is_migrated_only_when_asked_and_only_for_openrouter(tmp_path):
    cache_dir = str(tmp_path)
    view = _view()
    _write_v1_entry(cache_dir, view, _team(replay_only=True))

    with pytest.raises(ReplayMiss):
        _team(cache=ResponseCache(cache_dir), replay_only=True).plan_one(view)
    with pytest.raises(ReplayMiss):
        _team(cache=ResponseCache(cache_dir), replay_only=True, base_url=VLLM,
              migrate_legacy_v1=True).plan_one(view)

    migrating = _team(cache=ResponseCache(cache_dir), replay_only=True, migrate_legacy_v1=True)
    assert migrating.plan_one(view).roles == {0: BlueRole.DEFEND}
    assert migrating.diagnostics()["migrated_v1"] == 1
    usage = migrating.diagnostics()["usage"]
    assert usage["calls"] == 0 and usage["prompt_tokens"] == 0, "a migration spent nothing"

    # Now addressable under schema 2 without the migration flag, with the
    # endpoint marked as asserted rather than observed.
    after = _team(cache=ResponseCache(cache_dir), replay_only=True)
    assert after.plan_one(view).roles == {0: BlueRole.DEFEND}
    records = [json.load(open(os.path.join(cache_dir, n))) for n in os.listdir(cache_dir)]
    migrated = [r for r in records if (r.get("provenance") or {}).get("migrated_from_key_schema") == 1]
    assert len(migrated) == 1
    assert migrated[0]["provenance"]["endpoint_basis"].startswith("asserted")
