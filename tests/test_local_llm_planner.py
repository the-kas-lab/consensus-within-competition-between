"""The LLM planner's exit gate: per-agent LLM planners see only what they are allowed to see,
and what they produce is reproducible and auditable.

The tests check
*validity, not performance* -- nothing here grades a model's
tactical judgement, and no test in this file may be read as evidence about one.

Every test drives shipped code. The episode-level tests take views off a live
``Simulation`` rather than hand-building them, because the failure that matters
is in the wiring: an the command-delivery conditions confound once passed a method-level test while the call
site was broken.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import build_simulation, fingerprint, run_quiet  # noqa: E402
from perception.belief import LocalBeliefStore  # noqa: E402
from planning.local_llm import (  # noqa: E402
    DecodeConfig,
    LocalLLMPlanner,
    PerAgentLLMPlanner,
    build_client,
)
from planning.prompt_builder import PromptContext  # noqa: E402
from planning.response_cache import ResponseCache  # noqa: E402
from planning.roles import BlueRole  # noqa: E402

CTX = PromptContext(arena_size=100.0, defense_center=(50.0, 50.0))


# --- a client that answers without a network ------------------------------


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Usage:
    prompt_tokens = 1100
    completion_tokens = 90


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]
        self.usage = _Usage()


class _FakeCompletions:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        messages = kwargs["messages"]
        system = messages[0]["content"]
        user = messages[1]["content"]
        self.owner.prompts.append((system, user))
        return _Response(self.owner.answer(user))


# "no override" has to be distinguishable from "override with None", because
# None is the real API's way of saying a reasoning model produced no answer --
# the failure a reasoning model that returns nothing produces.
_UNSET = object()


class _FakeClient:
    """Answers from the prompt alone, deterministically.

    The policy is deliberately simple and *belief-driven*: pursue the first
    contact the prompt actually lists, else defend. That keeps the episode
    non-trivial while making every answer a function of the prompt, so a leak
    in the prompt would show up as a change in behaviour.
    """

    def __init__(self, answer=_UNSET):
        self.chat = type("chat", (), {})()
        self.chat.completions = _FakeCompletions(self)
        self.prompts = []
        self._answer = answer

    def answer(self, user_message):
        if self._answer is not _UNSET:
            return self._answer
        contact_ids = [
            int(line.split("contact_id=")[1].split()[0])
            for line in user_message.splitlines()
            if "contact_id=" in line and "believed_position" in line
        ]
        if contact_ids:
            return json.dumps(
                {
                    "role": "neutralize",
                    "target_contact_id": min(contact_ids),
                    "target_region_id": -1,
                    "reasoning": "closest believed contact",
                }
            )
        return json.dumps(
            {
                "role": "defend",
                "target_contact_id": -1,
                "target_region_id": -1,
                "reasoning": "nothing believed",
            }
        )


class _RaisingClient:
    """Any use at all is a test failure, which is how 'free' is asserted."""

    def __init__(self):
        self.chat = type("chat", (), {})()
        self.chat.completions = self

    def create(self, **kwargs):
        raise AssertionError("a cached episode must make no network call")


def _planner(client=None, cache=None, **kwargs):
    return PerAgentLLMPlanner(
        CTX,
        client=client if client is not None else _FakeClient(),
        cache=cache if cache is not None else ResponseCache(),
        build_client_on_demand=False,
        **kwargs,
    )


def _view(agent_id=0, now=10, contacts=None, teammates=None, n_regions=0):
    store = LocalBeliefStore(agent_id, n_regions=n_regions)
    for contact_id, (state, observed_at) in (contacts or {}).items():
        store.observe_target(contact_id, np.asarray(state, dtype=float), observed_at)
    if n_regions:
        store.observe_coverage(0, now)
    return store.snapshot(
        now=now,
        dt=0.1,
        self_state=np.array([10.0, 10.0, 1.0, 0.0, 0.0]),
        coverage_centers=np.zeros((n_regions, 2)) if n_regions else None,
        teammates=teammates or {},
    )


# --- the side channel is gone on this path --------------------------------

def test_the_planner_exposes_no_last_star_side_channel():
    """The debt the LLM planner comes due on.

    ``getattr(planner, "last_target_red_ids")`` names nothing once there is a
    planner per agent. These classes do not merely avoid the convention -- they
    do not implement it, so a caller that reaches for it fails loudly instead of
    silently reading an empty dict and calling it 'no targets'.
    """
    team = _planner()
    one = team.planner_for(0)

    for attribute in ("last_target_red_ids", "last_targets", "last_recon_target_regions"):
        assert not hasattr(team, attribute)
        assert not hasattr(one, attribute)


def test_targets_come_back_in_the_output():
    view = _view(contacts={2: ([20.0, 20.0, 0.0, 0.0, 0.0], 10)})
    out = _planner().plan_one(view)

    assert out.roles == {0: BlueRole.NEUTRALIZE}
    assert out.target_red_ids == {0: 2}


def test_plan_one_returns_only_the_acting_agents_slot():
    """An agent that decided for anyone else would be exercising an authority
    the decentralized condition does not give it."""
    view = _view(
        agent_id=1,
        contacts={0: ([20.0, 20.0, 0.0, 0.0, 0.0], 10)},
        teammates={0: np.array([5.0, 5.0, 0.0, 0.0, 0.0]),
                   2: np.array([9.0, 9.0, 0.0, 0.0, 0.0])},
    )
    out = _planner().plan_one(view)

    assert set(out.roles) == {1}
    assert set(out.target_red_ids) <= {1}


# --- fallbacks: silent to the run, loud to the record ---------------------

def test_pursuing_an_unbelieved_contact_falls_back_to_defend():
    """Leak L1 trying to re-enter through the planner's mouth. An agent must not
    be able to pursue something it does not believe in, whatever the model says.
    """
    client = _FakeClient(answer=json.dumps({
        "role": "neutralize", "target_contact_id": 99,
        "target_region_id": -1, "reasoning": "a contact it cannot see",
    }))
    team = _planner(client=client)
    out = team.plan_one(_view(contacts={2: ([20.0, 20.0, 0.0, 0.0, 0.0], 10)}))

    assert out.roles == {0: BlueRole.DEFEND}
    assert out.target_red_ids == {}
    assert [e["kind"] for e in team.fallback_events] == ["invalid_target_contact_id"]
    assert team.fallback_events[0]["invalid_value"] == 99


def test_recon_to_a_region_that_does_not_exist_falls_back_to_defend():
    client = _FakeClient(answer=json.dumps({
        "role": "recon", "target_contact_id": -1,
        "target_region_id": 77, "reasoning": "nowhere",
    }))
    team = _planner(client=client)
    out = team.plan_one(_view(n_regions=4))

    assert out.roles == {0: BlueRole.DEFEND}
    assert out.recon_target_regions == {}
    assert [e["kind"] for e in team.fallback_events] == ["invalid_target_region_id"]


def test_a_schema_invalid_response_falls_back_and_is_recorded():
    team = _planner(client=_FakeClient(answer="this is not json"))
    out = team.plan_one(_view())

    assert out.roles == {0: BlueRole.DEFEND}
    assert [e["kind"] for e in team.fallback_events] == ["schema_failure"]


def test_a_failed_call_falls_back_and_is_recorded():
    class _Broken(_FakeClient):
        def answer(self, user_message):
            raise RuntimeError("connection reset")

    team = _planner(client=_Broken())
    out = team.plan_one(_view())

    assert out.roles == {0: BlueRole.DEFEND}
    assert team.fallback_events[0]["kind"] == "call_failure"
    assert team.fallback_events[0]["error_type"] == "RuntimeError"


def test_one_agents_failure_does_not_defend_the_whole_team():
    """The behavioural improvement per-agent planning buys, asserted rather
    than assumed. The team planner's fallback defends everyone; here a bad
    response costs one agent one cycle."""
    class _BadForAgentOne(_FakeClient):
        def answer(self, user_message):
            if "you are blue agent_id=1" in user_message:
                return "not json"
            return super().answer(user_message)

    team = _planner(client=_BadForAgentOne())
    contacts = {0: ([20.0, 20.0, 0.0, 0.0, 0.0], 10)}
    good = team.plan_one(_view(agent_id=0, contacts=contacts))
    bad = team.plan_one(_view(agent_id=1, contacts=contacts))

    assert good.roles == {0: BlueRole.NEUTRALIZE}
    assert bad.roles == {1: BlueRole.DEFEND}
    assert len(team.fallback_events) == 1


# --- independent context --------------------------------------------------

def test_each_blue_gets_its_own_planner_object():
    team = _planner()
    team.plan_one(_view(agent_id=0))
    team.plan_one(_view(agent_id=1))

    assert set(team.planners) == {0, 1}
    assert team.planners[0] is not team.planners[1]


def test_the_cache_is_shared_but_the_context_is_not():
    """One cache across agents is correct -- two agents asking the identical
    question deserve the identical answer, and paying twice for it would be a
    waste, not an isolation property. Decision state is what must not be
    shared."""
    team = _planner()
    team.plan_one(_view(agent_id=0))
    team.plan_one(_view(agent_id=1))

    assert team.planners[0].cache is team.planners[1].cache
    assert team.planners[0].fallback_events is not team.planners[1].fallback_events


# --- provider swap --------------------------------------------------------

def test_the_base_url_is_a_call_site_change_not_a_code_change():
    """The declared end state is a local vLLM server. A local server needs no
    credential, so an empty key is accepted for a non-default base URL."""
    client = build_client(api_key=None, base_url="http://localhost:8000/v1")
    assert "localhost:8000" in str(client.base_url)


def test_the_default_provider_still_demands_a_key(monkeypatch):
    """Silently sending no key to OpenRouter would fail later and less clearly.
    monkeypatched rather than skipped, so the test runs in an environment that
    happens to have the key set -- which is every environment that can run the
    smoke test."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OpenRouter API key"):
        build_client(api_key=None, base_url=None)


# ---------------------------------------------------------------------------
# Episode level: the exit gate. Everything below drives a live Simulation.
# ---------------------------------------------------------------------------

# conftest's default fixture (2 blues / 1 red / 40 steps) never detects the red,
# so every belief stays empty, every answer is DEFEND, and a gate test on it
# would pass vacuously. This is the configuration measured to produce
# detections, and it is the same one the M3 per-agent tests use.
DETECTING = dict(num_blues=3, num_reds=2, sim_steps=150, seed=3701)


class _CapturingTeam(PerAgentLLMPlanner):
    """Records the frozen views Simulation hands to plan_one.

    Checks the *wiring* -- that the views reaching the LLM path are the frozen
    ones -- rather than re-testing the planner, which the unit tests above
    already do.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def plan_one(self, view, max_age=None):
        out = super().plan_one(view, max_age=max_age)
        self.calls.append((int(view.now), view, out))
        return out

    def epochs(self):
        grouped = {}
        for now, view, _ in self.calls:
            grouped.setdefault(now, {})[int(view.agent_id)] = view
        return [(now, grouped[now]) for now in sorted(grouped)]


def _run_llm_episode(client=None, cache=None, **overrides):
    params = dict(DETECTING)
    params.update(overrides)
    team = _CapturingTeam(
        CTX,
        client=client if client is not None else _FakeClient(),
        cache=cache if cache is not None else ResponseCache(),
        build_client_on_demand=False,
    )
    sim = build_simulation(role_planner=team, per_agent_planning=True, **params)
    return run_quiet(sim), team


def test_the_simulation_gate_accepts_a_per_agent_llm_planner():
    """``testbed.py:399`` raises TypeError for a planner with no plan_one, and
    that gate is deliberate. the LLM planner's job was to satisfy it, never to remove it."""
    team = _planner()
    sim = build_simulation(role_planner=team, per_agent_planning=True)
    assert sim.per_agent_planning is True


def test_an_llm_episode_actually_plans_per_agent_on_believed_contacts():
    """Guards every episode-level test below against a vacuous pass: if no
    agent ever believed in a contact, every answer is DEFEND and the gate
    proves nothing."""
    sim, team = _run_llm_episode()

    epochs = team.epochs()
    assert epochs, "planning must have run per agent"
    assert any(
        view.known_contact_ids() for _, views in epochs for view in views.values()
    ), "no contact was ever believed -- the tests below would pass vacuously"
    assert any(
        role == BlueRole.NEUTRALIZE
        for step in sim.role_assignment_history
        for role in step.values()
    ), "the planner never engaged -- the LLM path is not being exercised"


def test_exit_2_epoch_k_calls_cannot_see_epoch_k_outputs():
    """Information isolation.

    Replay one real epoch's frozen views forward and in reverse. Every agent's
    **prompt string** must be byte-identical between the two orders, and so
    must every output.

    The assertion is at the prompt level deliberately: two different prompts
    could coincidentally yield the same role, and an output-level test would
    pass anyway. The cache is disabled for this test so every call reaches the
    recording client -- a replay served from cache would record no prompt and
    the test would pass by seeing nothing.

    **Mutation-checked, and it revealed this test's own limit.** Against a
    mutation that made the N planners share decision state and fed it into the
    prompt, *this test still passed*: by replay time the shared state had
    already converged, so it no longer varied with order. The test below is
    what caught it. Order-invariance alone is not sufficient evidence of
    isolation, and neither test is redundant.
    """
    client = _FakeClient()
    sim, team = _run_llm_episode(client=client, cache=ResponseCache(enabled=False))

    epochs = team.epochs()
    assert epochs

    def replay(views, order):
        client.prompts.clear()
        prompts, outputs = {}, {}
        for agent_id in order:
            out = team.plan_one(views[agent_id])
            prompts[agent_id] = client.prompts[-1]
            outputs[agent_id] = (dict(out.roles), dict(out.target_red_ids))
        return prompts, outputs

    for now, views in epochs:
        ascending = sorted(views)
        forward_prompts, forward_out = replay(views, ascending)
        backward_prompts, backward_out = replay(views, list(reversed(ascending)))

        assert forward_prompts == backward_prompts, f"prompts moved with order at {now}"
        assert forward_out == backward_out, f"decisions moved with order at {now}"


def test_exit_2b_no_agents_prompt_mentions_another_agents_decision():
    """The ground-truth leak the isolation test exists to catch, stated directly rather than only
    as an order-invariance.

    An epoch's prompts must carry no role word at all: the moment one agent's
    assignment appears in another's prompt, blue 1 is reading blue 0's
    same-epoch output and the decentralized condition is fiction.
    """
    client = _FakeClient()
    _run_llm_episode(client=client, cache=ResponseCache(enabled=False))

    for _system, user in client.prompts:
        lowered = user.lower()
        for word in ("neutralize", "defend", "recon", "assigned", "role="):
            assert word not in lowered, f"{word!r} leaked into a per-agent prompt"


def test_exit_3_a_repeated_episode_is_free_and_byte_identical(tmp_path):
    """Replay.

    Run the episode once against a client, then again against a client that
    raises on any use. The second run must complete from the cache alone and
    produce a byte-identical episode -- compared with conftest's own
    ``fingerprint``, which hashes trajectories, roles and outcome, rather than
    with a summary written for this test.
    """
    cache_dir = str(tmp_path / "cache")

    first, first_team = _run_llm_episode(cache=ResponseCache(path=cache_dir))
    assert first_team.diagnostics()["calls"] > 0

    second, second_team = _run_llm_episode(
        client=_RaisingClient(), cache=ResponseCache(path=cache_dir)
    )

    assert fingerprint(first) == fingerprint(second)
    diagnostics = second_team.diagnostics()
    assert diagnostics["calls"] == 0, "a replayed episode must call nothing"
    assert diagnostics["cache_hits"] == first_team.diagnostics()["calls"]
    assert diagnostics["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "calls": 0,
    }, "a replay spent nothing and must report nothing"


def test_exit_4_fallbacks_reach_the_episode_outcome():
    """Audit trail.

    LLM failure is silent to the run by design -- and that design's cost is a
    complete, plausible-looking episode that is really the rule fallback
    wearing an LLM's name. The episode must finish, and the record must say so.
    """
    sim, team = _run_llm_episode(client=_FakeClient(answer="not json at all"))

    assert sim.episode_outcome()["n_trajectory_samples"] > 1, "the episode must finish"

    events = sim.episode_outcome()["planner_fallbacks"]
    assert events, "a broken model produced an episode with no recorded fallback"
    assert {e["kind"] for e in events} == {"schema_failure"}
    assert all(
        role == BlueRole.DEFEND
        for step in sim.role_assignment_history
        for role in step.values()
    )

    diagnostics = sim.episode_outcome()["planner_diagnostics"]
    assert diagnostics["fallbacks"] == len(events)
    assert diagnostics["model"] == team.model
    assert diagnostics["prompt_builder_version"]


def test_a_rule_planner_episode_reports_no_fallbacks_and_no_diagnostics():
    """Additive, so no existing result's schema moves: every rule episode
    carries the keys and carries them empty."""
    sim = run_quiet(build_simulation())
    outcome = sim.episode_outcome()

    assert outcome["planner_fallbacks"] == []
    assert outcome["planner_diagnostics"] == {}


# ---------------------------------------------------------------------------
# Regressions found by the first live smoke run, not by the tests above.
# ---------------------------------------------------------------------------

def test_an_empty_response_is_recorded_as_such_not_as_bad_json():
    """The failure the first live run actually produced, 8 times out of 8.

    ``openai/gpt-oss-20b`` is a reasoning model: it spends its token budget
    thinking and emits the answer last, so a budget that is too small does not
    truncate the answer, it prevents it -- ``content`` comes back ``None``.
    'Returned nothing' and 'returned invalid JSON' have different causes and
    different fixes, so they are different kinds in the record.
    """
    team = _planner(client=_FakeClient(answer=None))
    out = team.plan_one(_view())

    assert out.roles == {0: BlueRole.DEFEND}
    assert [e["kind"] for e in team.fallback_events] == ["empty_response"]


def test_an_empty_response_is_cached_so_the_replay_reproduces_it(tmp_path):
    """A bug the exit-3 test could not see, because its episode never failed.

    Cached as ``None``, an empty response is indistinguishable from 'no entry'
    on read -- so the replay would go back to the network, and a failed episode
    would not reproduce. A reproducibility mechanism has to reproduce the bad
    runs too, which are the ones worth re-examining.
    """
    cache_dir = str(tmp_path)
    first = _planner(client=_FakeClient(answer=None), cache=ResponseCache(path=cache_dir))
    first.plan_one(_view())

    replay = _planner(client=_RaisingClient(), cache=ResponseCache(path=cache_dir))
    out = replay.plan_one(_view())

    assert out.roles == {0: BlueRole.DEFEND}
    assert [e["kind"] for e in replay.fallback_events] == ["empty_response"]
    assert replay.diagnostics()["calls"] == 0, "the replay must not call out"


def test_reasoning_effort_is_opt_in_and_reaches_the_request():
    """An OpenRouter routing hint, off by default, so the local-vLLM path sends
    nothing a plain OpenAI-compatible server would not understand."""
    recorder = {}

    class _Recording(_FakeClient):
        def __init__(self):
            super().__init__()
            outer = self

            class _C:
                def create(self, **kwargs):
                    recorder.update(kwargs)
                    return _Response(outer.answer(kwargs["messages"][1]["content"]))

            self.chat.completions = _C()

    _planner(client=_Recording()).plan_one(_view())
    assert "extra_body" not in recorder

    recorder.clear()
    _planner(
        client=_Recording(), decode=DecodeConfig(reasoning_effort="low")
    ).plan_one(_view())
    assert recorder["extra_body"] == {"reasoning": {"effort": "low"}}


def test_the_decode_config_is_part_of_the_cache_key():
    """Two decode configs are two different questions. Sharing an entry would
    serve a high-effort answer to a low-effort request."""
    cache = ResponseCache()
    view = _view(contacts={0: ([20.0, 20.0, 0.0, 0.0, 0.0], 10)})

    _planner(cache=cache, decode=DecodeConfig(max_tokens=2000)).plan_one(view)
    second = _planner(cache=cache, decode=DecodeConfig(max_tokens=512))
    second.plan_one(view)

    assert second.diagnostics()["calls"] == 1, "a different decode config must miss"


def test_an_uppercase_role_is_accepted_because_it_is_unambiguous():
    """4 of 36 responses in the first valid smoke run said "RECON".

    Partly caused by a prompt that spells the roles in capitals, now fixed --
    but "RECON" unambiguously means recon, and discarding a well-judged
    decision over its casing would inflate the fallback rate with something
    that is not a planning failure.
    """
    client = _FakeClient(answer=json.dumps({
        "role": "RECON", "target_region_id": 0, "reasoning": "stale region",
    }))
    team = _planner(client=client)
    out = team.plan_one(_view(n_regions=4))

    assert out.roles == {0: BlueRole.RECON}
    assert out.recon_target_regions == {0: 0}
    assert team.fallback_events == []


def test_an_omitted_irrelevant_target_defaults_rather_than_failing():
    """An agent that is not neutralizing has no contact id to give."""
    client = _FakeClient(answer=json.dumps({
        "role": "defend", "reasoning": "holding the perimeter",
    }))
    team = _planner(client=client)

    assert team.plan_one(_view()).roles == {0: BlueRole.DEFEND}
    assert team.fallback_events == []


def test_a_missing_target_is_never_guessed_for_a_role_that_needs_one():
    """Where the line is drawn. Normalising syntax keeps a good decision;
    inventing a target would manufacture one. -1 is not a believed contact, so
    this still falls back and is still recorded."""
    client = _FakeClient(answer=json.dumps({
        "role": "neutralize", "reasoning": "engaging something unnamed",
    }))
    team = _planner(client=client)
    out = team.plan_one(_view(contacts={0: ([20.0, 20.0, 0.0, 0.0, 0.0], 10)}))

    assert out.roles == {0: BlueRole.DEFEND}
    assert out.target_red_ids == {}
    assert [e["kind"] for e in team.fallback_events] == ["invalid_target_contact_id"]
    assert team.fallback_events[0]["invalid_value"] == -1


def test_a_role_that_is_not_a_role_still_fails():
    """Tolerance must not become credulity."""
    team = _planner(client=_FakeClient(answer=json.dumps({
        "role": "retreat", "target_contact_id": -1,
        "target_region_id": -1, "reasoning": "leaving",
    })))

    assert team.plan_one(_view()).roles == {0: BlueRole.DEFEND}
    assert [e["kind"] for e in team.fallback_events] == ["schema_failure"]
