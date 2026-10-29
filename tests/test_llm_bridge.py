"""The auction LLM bridge: the LLM chooses bounded bid modifiers, never a bid.

Contract established with FAKE clients only -- no network, no spend. The LLM planner
smoke's 60 valid role responses (none of them NEUTRALIZE) say nothing about
this bridge; its contract is pinned here.

Two layers:

* the contract, on hand-built views: what is kept, what is dropped, what is
  recorded, and that total failure means exactly the default modifiers;
* the wiring, on live Simulations (the CBAA fixture, 3 blues / 2 reds / 300
  steps / seed 7, the configuration measured to believe contacts): modifiers
  reach the bids, total failure reproduces rule CBAA byte-for-byte, concurrent
  calls change nothing, every prompt is a pure function of its own frozen
  (view, tasks), and priorities are validated against BELIEF on moving reds.

"""

import json
import os
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import build_simulation, fingerprint, run_quiet  # noqa: E402
from coordination.bidding import BidContext, BidModifiers, bids  # noqa: E402
from coordination.llm_bridge import (  # noqa: E402
    MAX_PREFERRED,
    LLMModifierSource,
    build_bridge_system_prompt,
    build_bridge_user_message,
)
from coordination.tasks import TaskContext, generate_tasks  # noqa: E402
from planning.local_llm import ReplayMiss  # noqa: E402
from planning.prompt_builder import PromptContext  # noqa: E402
from planning.response_cache import ResponseCache  # noqa: E402
from test_local_llm_planner import _Response, _view  # noqa: E402

CTX = PromptContext(arena_size=100.0, defense_center=(50.0, 50.0), horizon_steps=400, dt=0.1)
TASK_CTX = TaskContext(
    defense_polygon=((40.0, 40.0), (60.0, 40.0), (60.0, 60.0), (40.0, 60.0)),
    stations=((0, (60.0, 50.0)), (1, (40.0, 50.0))),
)
EXERCISED = dict(num_blues=3, num_reds=2, sim_steps=300, seed=7, allocation="cbaa",
                 cbaa_task_set="three_task", cbaa_fallback="station", cbaa_bid_version="v2.3",
                 station_heading="configured")


class _Client:
    """A fake OpenAI client whose answer is a function of the prompt alone.

    ``policy(user_message) -> dict | str | None``; a dict is JSON-encoded, a
    str is returned verbatim, None means the model returned no content, and an
    exception propagates as a transport failure. Thread-safe prompt capture,
    because the concurrency test drives it from a pool.
    """

    def __init__(self, policy):
        self.policy = policy
        self.prompts = []
        self._lock = threading.Lock()
        outer = self

        class _Completions:
            def create(self, **kwargs):
                user = kwargs["messages"][1]["content"]
                with outer._lock:
                    outer.prompts.append((kwargs["messages"][0]["content"], user))
                answer = outer.policy(user)
                return _Response(answer if (answer is None or isinstance(answer, str))
                                 else json.dumps(answer))

        self.chat = type("chat", (), {})()
        self.chat.completions = _Completions()


def _answer(profile="balanced", priorities=(), preferred=(), **extra):
    out = {
        "scoring_profile": profile,
        "threat_priorities": [{"contact_id": c, "priority": p} for c, p in priorities],
        "preferred_tasks": list(preferred),
        "reasoning": "fake",
    }
    out.update(extra)
    return out


def _listed(user_message, prefix):
    return [line.split()[0] for line in user_message.splitlines()
            if line.strip().startswith(prefix)]


def _contacts(user_message):
    return [int(line.split("contact_id=")[1].split()[0])
            for line in user_message.splitlines() if "believed_position" in line]


def _source(policy, cache=None, **kwargs):
    return LLMModifierSource(
        CTX, client=_Client(policy), cache=cache if cache is not None else ResponseCache(),
        build_client_on_demand=False, **kwargs,
    )


def _wired(policy, **kwargs):
    """A source for a live episode: the three-task model the episode runs."""
    return _source(policy, task_set="three_task", **kwargs)


def _one(policy, view, **kwargs):
    src = _source(policy, **kwargs)
    tasks = generate_tasks(view, TASK_CTX)
    agent = int(view.agent_id)
    terms = {t.task_id: {"value": 1.0, "bid": 1.0} for t in tasks}  # the prompt shows each task's terms
    mods = src.modifiers_for_epoch(int(view.now), {agent: (view, tasks)},
                                   default_profile="balanced", bid_terms={agent: terms},
                                   current_tasks={}, remaining_s=40.0)[agent]
    return src, tasks, mods


MOVING = {0: ([20.0, 20.0, 1.0, 0.5, 0.0], 6), 3: ([80.0, 30.0, -1.0, 0.0, 0.0], 9)}


# --- the contract --------------------------------------------------------------

def test_a_valid_answer_becomes_exactly_those_bounded_modifiers():
    view = _view(contacts=MOVING)
    src, tasks, mods = _one(lambda u: _answer(
        "conservative", [(0, "HIGH"), (3, "LOW")], ["SCREEN:0", "DEFEND:1"]), view)
    assert mods == BidModifiers.build("conservative", {0: "HIGH", 3: "LOW"},
                                      ["SCREEN:0", "DEFEND:1"])
    assert src.fallback_events == []


def test_an_unknown_profile_is_dropped_to_the_default_and_recorded_the_rest_kept():
    view = _view(contacts=MOVING)
    src, _, mods = _one(lambda u: _answer("reckless", [(0, "HIGH")]), view)
    assert mods.scoring_profile == "balanced"
    assert mods.threat_priorities == ((0, "HIGH"),), "one bad field must not discard the rest"
    assert [e["kind"] for e in src.fallback_events] == ["unknown_profile"]


def test_a_priority_for_an_unbelieved_contact_is_dropped_and_recorded():
    view = _view(contacts=MOVING)
    src, _, mods = _one(lambda u: _answer(priorities=[(0, "HIGH"), (1, "HIGH")]), view)
    assert dict(mods.threat_priorities) == {0: "HIGH"}
    assert [(e["kind"], e["value"]) for e in src.fallback_events] == [("unbelieved_contact", 1)]


def test_an_unlisted_task_is_dropped_and_preferences_are_capped():
    view = _view(contacts=MOVING)
    listed = ["NEUTRALIZE:0:0", "SCREEN:0", "NEUTRALIZE:3:0", "SCREEN:3", "DEFEND:0"]
    src, _, mods = _one(lambda u: _answer(preferred=["RECON:99"] + listed), view)
    assert list(mods.preferred_tasks) == sorted(listed[:MAX_PREFERRED])
    kinds = [e["kind"] for e in src.fallback_events]
    assert kinds.count("unlisted_task") == 1
    assert kinds.count("too_many_preferred") == len(listed) - MAX_PREFERRED


def test_no_number_the_llm_writes_can_reach_a_bid():
    """A numeric profile, a numeric priority and a numeric task id are labels
    that do not exist -- dropped, not coerced. And the plan's
    commitment_horizon_s float is ignored and recorded."""
    view = _view(contacts=MOVING)
    src, _, mods = _one(lambda u: {
        "scoring_profile": 0.9,
        "threat_priorities": [{"contact_id": 0, "priority": 1.7}],
        "preferred_tasks": [2.5],
        "reasoning": "numbers",
        "commitment_horizon_s": 12.5,
    }, view)
    assert mods == BidModifiers.build("balanced")
    kinds = {e["kind"] for e in src.fallback_events}
    assert kinds == {"unknown_profile", "invalid_priority", "unlisted_task", "ignored_fields"}


def test_label_syntax_is_normalised_but_semantics_never_are():
    view = _view(contacts=MOVING)
    src, _, mods = _one(lambda u: _answer(" Scouting ", [(0, "high"), (3, "EXTREME")]), view)
    assert mods.scoring_profile == "scouting"
    assert dict(mods.threat_priorities) == {0: "HIGH"}
    assert [e["kind"] for e in src.fallback_events] == ["invalid_priority"]


@pytest.mark.parametrize("policy, kind", [
    (lambda u: "not json", "schema_failure"),
    (lambda u: None, "empty_response"),
    (lambda u: (_ for _ in ()).throw(ConnectionError("reset")), "call_failure"),
])
def test_total_failure_yields_default_modifiers_and_is_recorded(policy, kind):
    view = _view(contacts=MOVING)
    src, _, mods = _one(policy, view)
    assert mods is None, "total failure must hand the runtime None -> default modifiers"
    (event,) = src.fallback_events
    assert event["kind"] == kind and event["applied"] == "default_modifiers"
    assert src.diagnostics()["defaulted_after_total_failure"] == 1


def test_the_bridge_is_not_a_role_planner():
    with pytest.raises(TypeError):
        _source(lambda u: _answer()).plan_one(_view())


# --- the wiring, on live episodes ----------------------------------------------

def _run(source=None, **overrides):
    params = dict(EXERCISED)
    params.update(overrides)
    return run_quiet(build_simulation(cbaa_modifier_source=source, **params))


def _believing_policy(u):
    """Belief-driven, so behaviour moves with the prompt: HIGH on the first
    believed contact, prefer pursuing it, profile conservative."""
    contacts = _contacts(u)
    if not contacts:
        return _answer("scouting")
    c = min(contacts)
    return _answer("conservative", [(c, "HIGH")], [f"NEUTRALIZE:{c}:0"])


@pytest.fixture(scope="module")
def bridged():
    src = _wired(_believing_policy)
    captured = []
    original = src.modifiers_for_epoch

    def spy(step, requests, default_profile, **kw):
        out = original(step, requests, default_profile, **kw)
        captured.append((step, requests, out, kw))
        return out

    src.modifiers_for_epoch = spy
    sim = _run(src, belief_mode="shared")
    return sim, src, captured


def test_modifiers_reach_the_bids_the_auction_uses(monkeypatch):
    """Every agent's epoch-k bids, as CBAAAgent.start_epoch received them, must
    equal the scorer applied to that agent's frozen (view, tasks), its current
    commitment, and the modifiers the bridge returned for it -- and must differ
    from default-modifier bids somewhere, or the test is vacuous."""
    from coordination import cbaa

    used = {}
    original = cbaa.CBAAAgent.start_epoch

    def record(self, epoch, bid_map, order):
        # Before the call: start_epoch keeps the commitment, and the runtime
        # priced the switch cost against exactly this value.
        used[(epoch, self.agent_id)] = (dict(bid_map), self.commitment)
        return original(self, epoch, bid_map, order)

    monkeypatch.setattr(cbaa.CBAAAgent, "start_epoch", record)
    src = _wired(_believing_policy)
    log = []
    orig = src.modifiers_for_epoch

    def spy(step, requests, default_profile, **kw):
        out = orig(step, requests, default_profile, **kw)
        log.append((requests, out))
        return out

    src.modifiers_for_epoch = spy
    sim = _run(src, belief_mode="shared")
    bid_ctx = BidContext.from_simulation(sim, sim.cbaa_runtime.task_ctx)

    differs = 0
    for epoch, (requests, out) in enumerate(log):
        for agent, (view, tasks) in requests.items():
            got, current = used[(epoch, agent)]
            mods = out[agent] or BidModifiers(scoring_profile="balanced")
            assert _close(got, bids(view, tasks, bid_ctx, current, mods)), (epoch, agent)
            if not _close(got, bids(view, tasks, bid_ctx, current)):
                differs += 1
    assert differs > 0, "modifiers never changed a bid: the test would pass vacuously"


def _close(a, b):
    return a.keys() == b.keys() and all(abs(a[k] - b[k]) <= 1e-9 for k in a)


def test_total_failure_every_epoch_reproduces_rule_cbaa_exactly():
    """'Total failure -> default modifiers (= rule CBAA)', as an identity on a
    live episode, with every failure on the record."""
    rule = _run(None, belief_mode="shared")
    src = _wired(lambda u: "not json")
    failed = _run(src, belief_mode="shared")
    assert fingerprint(failed) == fingerprint(rule)
    events = failed.episode_outcome()["planner_fallbacks"]
    assert events and {e["kind"] for e in events} == {"schema_failure"}
    assert len(events) == len(src.modifier_log)


def test_every_prompt_is_a_pure_function_of_its_own_frozen_view_and_tasks(bridged):
    """Not an order test.
    Each prompt the client received is rebuilt from that agent's own frozen
    (view, tasks) alone and must match byte for byte."""
    sim, src, captured = bridged
    for _, requests, _, _ in captured:
        for agent, (view, _) in requests.items():
            assert int(view.agent_id) == int(agent), "an agent was handed another agent's view"
    received = [u for _, u in src.client.prompts]
    rebuilt = [
        build_bridge_user_message(view, tasks, src.context, kw["bid_terms"][agent],
                                  kw["current_tasks"].get(agent), kw["remaining_s"])
        for step, requests, _, kw in captured
        for agent, (view, tasks) in sorted(requests.items())
    ]
    assert received == rebuilt
    assert any(_contacts(u) for u in received), "fixture must believe some contact"
    systems = {s for s, _ in src.client.prompts}
    assert systems == {build_bridge_system_prompt(src.context, "three_task", "cbaa")}


def test_concurrent_calls_change_nothing(tmp_path):
    seq = _wired(_believing_policy)
    par = _wired(_believing_policy, max_workers=4)
    a = _run(seq, belief_mode="shared")
    b = _run(par, belief_mode="shared")
    assert fingerprint(a) == fingerprint(b)
    assert seq.modifier_log == par.modifier_log
    assert sorted(seq.client.prompts) == sorted(par.client.prompts)
    assert par.diagnostics()["max_workers"] == 4


def test_priorities_are_validated_against_belief_not_truth_on_moving_reds():
    """The fake model names EVERY red. Per epoch per agent, what is kept must be
    exactly that agent's believed set, and what is dropped exactly the rest --
    judged against the agent's own frozen view, never against which reds exist.

    ``networked`` at R=20, not ``local``: measured, under ``local`` with no
    auction traffic the stationed agents almost never detect anything, so the check would be vacuous. At R=20 the
    fixture has 87 agent-epochs believing nothing while both reds are active --
    where a truth-based check would keep both priorities -- and 2 believing
    only one red."""
    everyone = lambda u: _answer(priorities=[(0, "HIGH"), (1, "HIGH")])  # noqa: E731
    src = _wired(everyone)
    captured = []
    original = src.modifiers_for_epoch

    def spy(step, requests, default_profile, **kw):
        out = original(step, requests, default_profile, **kw)
        captured.append((step, requests, out))
        return out

    src.modifiers_for_epoch = spy
    sim = _run(src, belief_mode="networked", r_comm=20.0)
    dropped = {(e["sim_step"], e["agent_id"], e["value"])
               for e in src.fallback_events if e["kind"] == "unbelieved_contact"}
    partial = 0
    for step, requests, out in captured:
        for agent, (view, _) in requests.items():
            believed = set(view.fresh_targets(sim.belief_validity_age_steps))
            assert set(dict(out[agent].threat_priorities)) == believed & {0, 1}
            assert {c for (s, a, c) in dropped if s == step and a == agent} == {0, 1} - believed
            if believed and believed != {0, 1}:
                partial += 1
    assert partial > 0, "fixture never split beliefs; the test would be vacuous"
    # The reds MOVE: a stationary contact once hid a raw-sighting bug twice.
    reds = np.asarray(sim.history_red)
    assert np.ptp(reds[:, 0]) > 1.0


def test_a_bridged_episode_replays_offline_as_the_same_episode(tmp_path):
    live_src = _wired(_believing_policy, cache=ResponseCache(str(tmp_path)))
    live = _run(live_src, belief_mode="shared")
    replay_src = LLMModifierSource(CTX, cache=ResponseCache(str(tmp_path)), replay_only=True,
                                   base_url="unspecified", task_set="three_task")
    # The fake client has no base_url, so its entries are keyed "unspecified";
    # a replay must name the same endpoint or it is asking someone else.
    replay = _run(replay_src, belief_mode="shared")
    assert fingerprint(replay) == fingerprint(live)
    assert replay_src.diagnostics()["calls"] == 0
    assert replay_src.modifier_log == live_src.modifier_log


def test_a_replay_miss_stops_a_bridged_episode():
    with pytest.raises(ReplayMiss):
        _run(LLMModifierSource(CTX, replay_only=True, task_set="three_task"), belief_mode="shared")


def test_a_source_without_cbaa_is_refused():
    with pytest.raises(ValueError, match="allocation='cbaa'"):
        build_simulation(cbaa_modifier_source=_source(lambda u: _answer()))
