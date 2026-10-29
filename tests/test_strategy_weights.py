"""Strategy weights: the scorer, validation and budgets, the bridge, the runtime
dispatch and the runner arm (the consensus team at high authority).
Same hand-built static scene as test_bid_threat_forecast (spent seed 3901). No model
calls; the bridge is driven by a fake client."""

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_bid_threat_forecast import _belief, _red, _team, _views  # noqa: E402

from coordination import bidding  # noqa: E402
from coordination.bidding import BidModifiers, bid_terms_v2  # noqa: E402
from coordination.strategy_weights import (DEFAULT_WEIGHTS, WEIGHT_FIELDS,  # noqa: E402
                                           StrategyWeights, strategy_terms, validate_answer)
from coordination.tasks import generate_tasks  # noqa: E402


@pytest.fixture(scope="module")
def scene():
    from coordination.tasks import TaskContext
    from scripts import run_competing_teams as rct

    e = {"arm": "cbaa", "reds": 4, "family": "multi_bearing", "condition": "REF",
         "attacked": False, "seed": 3901, "design": "v2.3mb"}
    e["episode_id"] = rct.episode_id(e)
    sim, _ = rct.build(e, "rule")
    task_ctx = TaskContext.from_simulation(sim, task_set="three_task")
    ctx = bidding.BidContext.from_simulation(sim, task_ctx)
    assert ctx.v2.version == "v2.3"
    return sim, task_ctx, ctx


def _states(sim, ctx, task_ctx):
    """A few views with believed contacts and commitments of every kind."""
    out = []
    for status in ("approaching", "moving_away", "crossing"):
        for dist in (10, 50):
            team = _team(ctx, scout=(dist == 50))
            p1, v1 = _red(sim, ctx, 220, dist, status, 2.0)
            p2, v2 = _red(sim, ctx, 260, dist + 20, "approaching", 1.5)
            targets = {1: _belief(1, p1, v1), 2: _belief(2, p2, v2, age=12)}
            first = next(iter(team))
            team[first] = (team[first][0], "NEUTRALIZE:1:0")
            for i, v in _views(sim, ctx, team, targets).items():
                out.append((v, generate_tasks(v, task_ctx), team[i][1]))
    return out


def test_default_strategy_is_bit_identical(scene):
    sim, task_ctx, ctx = scene
    sw = StrategyWeights.build()
    n = 0
    for v, tasks, commit in _states(sim, ctx, task_ctx):
        old = bid_terms_v2(v, tasks, ctx, commit, BidModifiers())
        new = strategy_terms(v, tasks, ctx, commit, sw)
        for t in tasks:
            assert new[t.task_id]["bid"] == old[t.task_id]["bid"]  # exact, not approx
            n += 1
    assert n > 100


def test_strategy_bid_matches_direct_formula(scene):
    sim, task_ctx, ctx = scene
    w = dict(DEFAULT_WEIGHTS, w_threat=0.4, w_engage=2.0, w_station=0.3, w_coverage=0.4,
             w_scout=0.1, w_attrition=1.7, w_belief_age=0.9, w_travel=0.5, w_scout_travel=0.2,
             w_leave=1.1, w_continuity=0.8, w_switch=0.3)
    for v, tasks, commit in _states(sim, ctx, task_ctx)[:8]:
        pri = {c: 0.5 for c in v.fresh_targets(None)}
        sw = StrategyWeights.build(w, pri, [tasks[-1].task_id])
        got = strategy_terms(v, tasks, ctx, commit, sw)
        ref = bid_terms_v2(v, tasks, ctx, commit, BidModifiers())
        for t in tasks:
            x = ref[t.task_id]
            pen = (w["w_leave"] * x.get("leave", 0) / 0.25 + w["w_switch"] * (1 if x.get("switch") else 0)
                   + w["w_continuity"] * x.get("continuity", 0) / 0.15)
            if t.kind == "DEFEND":
                b = w["w_station"] + w["w_coverage"] * x["coverage"] - w["w_travel"] * x["travel"] / 0.15
            elif t.kind == "RECON":
                b = w["w_scout"] * x["information"] - w["w_scout_travel"] * x["travel"] / 0.05
            else:
                b = (0.5 * (w["w_threat"] + w["w_engage"] * x["urgency"] * x["opportunity"])
                     - w["w_travel"] * x["travel"] / 0.15 - w["w_attrition"] * x["attrition"] / 0.5
                     - w["w_belief_age"] * x["uncertainty"] / 0.1)
            b -= pen
            if t.task_id == tasks[-1].task_id:
                b += 0.2
            assert got[t.task_id]["bid"] == pytest.approx(b, abs=1e-12)


@pytest.mark.parametrize("bad", [True, "0.3", float("nan"), float("inf"), -0.1, 99, None])
def test_invalid_field_falls_back_alone(bad):
    w = dict(DEFAULT_WEIGHTS, w_travel=bad, w_attrition=1.5)
    sw, ev = validate_answer({"weights": w, "contact_priorities": [], "preferred_tasks": []}, [], [])
    assert [(e["kind"], e["field"]) for e in ev] == [("weight_invalid", "w_travel")]
    assert sw.w["w_travel"] == DEFAULT_WEIGHTS["w_travel"] and sw.w["w_attrition"] == 1.5


def test_priority_and_preference_drops():
    ans = {"weights": dict(DEFAULT_WEIGHTS),
           "contact_priorities": [{"contact_id": 9, "priority": 1}, {"contact_id": 1, "priority": 0},
                                  {"contact_id": 1, "priority": 2}, {"contact_id": 2, "priority": 3}],
           "preferred_tasks": ["DEFEND:0", "DEFEND:0", "X", "DEFEND:1", "RECON:2", "RECON:3"]}
    sw, ev = validate_answer(ans, [1, 2], ["DEFEND:0", "DEFEND:1", "RECON:2", "RECON:3"])
    kinds = [e["kind"] for e in ev]
    assert kinds == ["unbelieved_contact", "duplicate_priority", "invalid_priority",
                     "unlisted_task", "too_many_preferred"]
    assert dict(sw.priorities) == {1: 0.0}
    assert sw.preferred_tasks == ("DEFEND:0", "DEFEND:1", "RECON:2")


def test_missing_weights_each_default():
    sw, ev = validate_answer({"weights": {"w_threat": 0.0}}, [], [])
    assert sw.w["w_threat"] == 0.0
    assert sum(e["kind"] == "weight_missing" for e in ev) == len(WEIGHT_FIELDS) - 1


def test_m8b8_prompt_states_every_weight_and_no_tactic(scene):
    from coordination.strategy_bridge import build_system_prompt
    from planning.prompt_builder import PromptContext

    sim, _, _ = scene
    text = build_system_prompt(PromptContext.from_simulation(sim))
    for f in WEIGHT_FIELDS:
        assert f"- {f.name}: " in text and f"Default {f.default:g}, allowed {f.lo:g} to {f.hi:g}" in text
    assert "must not exceed 3.2" in text
    low = text.lower()
    for word in ("perimeter hold", "all-out", "forward interception", "scout-heavy", "risk-averse",
                 "aggressive", "conservative", "division of labo", "sacrific", "strategy such as",
                 "for example", "e.g.", "we recommend", "you should", "it is best"):
        assert word not in low, word
    assert "nothing numeric" not in low


class _FakeClient:
    def __init__(self, content):
        self.content = content
        self.base_url = "https://fake.invalid/v1"
        self.chat = self
        self.completions = self

    def create(self, **kw):
        class _M:
            pass
        m, c, r = _M(), _M(), _M()
        m.content, m.refusal, m.reasoning, m.reasoning_details = self.content, None, None, None
        c.message, c.finish_reason, c.native_finish_reason = m, "stop", "stop"
        r.choices, r.usage, r.id, r.model, r.provider = [c], None, "x", "fake", "fake"
        r.model_dump = lambda: {"id": "x"}
        return r


def _bridge_case(scene, content, tmp_path):
    from coordination.strategy_bridge import StrategyWeightSource
    from planning.prompt_builder import PromptContext
    from planning.response_cache import ResponseCache

    sim, task_ctx, ctx = scene
    v, tasks, commit = _states(sim, ctx, task_ctx)[0]
    src = StrategyWeightSource(PromptContext.from_simulation(sim), client=_FakeClient(content),
                               cache=ResponseCache(path=str(tmp_path)), build_client_on_demand=False)
    terms = bid_terms_v2(v, tasks, ctx, commit, BidModifiers())
    out = src.modifiers_for_epoch(100, {v.agent_id: (v, tasks)}, "balanced",
                                  bid_terms={v.agent_id: terms}, current_tasks={v.agent_id: commit},
                                  remaining_s=30.0)
    return src, out[v.agent_id], v, tasks


def test_bridge_valid_answer(scene, tmp_path):
    w = dict(DEFAULT_WEIGHTS, w_threat=0.2, w_station=1.0, w_coverage=0.55)  # value sum 3.2
    ans = json.dumps({"reasoning": "r", "weights": w, "contact_priorities": [{"contact_id": 1, "priority": 0.5}],
                      "preferred_tasks": []})
    src, sw, v, tasks = _bridge_case(scene, ans, tmp_path)
    assert isinstance(sw, StrategyWeights) and sw.w == w and dict(sw.priorities) == {1: 0.5}
    assert src.diagnostics()["defaulted_after_total_failure"] == 0
    sysmsg, user = src.planners[v.agent_id].last_prompt
    for t in tasks:
        assert f'task_id "{t.task_id}"' in user
    assert "-> default bid" in user


def test_bridge_total_failure_is_none(scene, tmp_path):
    src, sw, _, _ = _bridge_case(scene, "not json", tmp_path)
    assert sw is None
    assert [e["kind"] for e in src.fallback_events] == ["schema_failure"]
    assert src.diagnostics()["defaulted_after_total_failure"] == 1


def test_runner_identity_unchanged_without_new_arms():
    from scripts.run_competing_teams import execution_identity

    a = execution_identity("rule", designs=["v2.3mb"])
    b = execution_identity("rule", designs=["v2.3mb"], arms=["cbaa", "independent", "commander"])
    c = execution_identity("rule", designs=["v2.3mb"], arms=["cbaa", "cbaa_sw2"])
    assert a == b and "strategy_arms" not in a
    assert c["strategy_arms"]["arms"] == ["cbaa_sw2"] and c["identity_hash"] != a["identity_hash"]


def test_runner_refuses_unknown_arm_and_wrong_design():
    from scripts import run_competing_teams as rct

    e = {"arm": "cbaa_typo", "reds": 2, "family": "multi_bearing", "condition": "REF",
         "attacked": False, "seed": 3901, "design": "v2.3mb"}
    e["episode_id"] = rct.episode_id(e)
    with pytest.raises(ValueError, match="unknown arm"):
        rct.build(e, "rule")
    with pytest.raises(ValueError, match="needs an inference profile"):
        rct.build(dict(e, arm="cbaa_sw2"), "profile", {"profile_obj": None})


# -- sw.2 / m8b.9: declared proportional scaling ------------------

def test_sw2_scales_value_block_within_budget():
    from decimal import Decimal

    over = dict(DEFAULT_WEIGHTS, w_station=1.0, w_scout=1.5)  # value sum 4.25
    sw, ev = validate_answer({"weights": over, "contact_priorities": [], "preferred_tasks": []},
                             [], [])
    assert [e["kind"] for e in ev] == ["value_budget_scaled"]
    vals = [sw.w[k] for k in ("w_threat", "w_engage", "w_station", "w_coverage", "w_scout")]
    assert sum(Decimal(repr(v)) for v in vals) <= Decimal("3.2")
    assert sum(vals) == pytest.approx(3.2, abs=1e-5)
    assert sw.w["w_scout"] / sw.w["w_station"] == pytest.approx(1.5, rel=1e-4)  # proportions kept
    assert sw.w["w_travel"] == DEFAULT_WEIGHTS["w_travel"]  # unbudgeted fields untouched


def test_sw2_scales_priorities_including_omitted_defaults():
    from decimal import Decimal

    pri = [{"contact_id": 1, "priority": 1.5}, {"contact_id": 2, "priority": 1.5}]
    sw, ev = validate_answer({"weights": dict(DEFAULT_WEIGHTS), "contact_priorities": pri,
                              "preferred_tasks": []}, [1, 2, 3], [])
    assert [e["kind"] for e in ev] == ["priority_budget_scaled"]
    p = {c: sw.priority(c) for c in (1, 2, 3)}
    assert sum(Decimal(repr(v)) for v in p.values()) <= Decimal(3)
    assert p[1] == p[2] and p[1] / p[3] == pytest.approx(1.5, rel=1e-4)


def test_prompt_states_scaling_and_budget_line(scene):
    from coordination.strategy_bridge import build_system_prompt, build_user_message
    from planning.prompt_builder import PromptContext

    sim, task_ctx, ctx = scene
    pc = PromptContext.from_simulation(sim)
    system = build_system_prompt(pc)
    assert "scaled down in proportion" in system
    v, tasks, commit = _states(sim, ctx, task_ctx)[0]
    terms = bid_terms_v2(v, tasks, ctx, commit, BidModifiers())
    user = build_user_message(v, tasks, pc, terms, commit, 30.0)
    assert "the priorities of your 2 believed contacts sum to at most 2" in user


def test_bridge_scales_over_budget_answer(scene, tmp_path):
    from coordination.strategy_bridge import StrategyWeightSource
    from planning.prompt_builder import PromptContext
    from planning.response_cache import ResponseCache

    sim, task_ctx, ctx = scene
    v, tasks, commit = _states(sim, ctx, task_ctx)[0]
    ans = json.dumps({"reasoning": "r", "weights": dict(DEFAULT_WEIGHTS, w_threat=2.0),
                      "contact_priorities": [{"contact_id": 1, "priority": 2.0},
                                             {"contact_id": 2, "priority": 1.0}],
                      "preferred_tasks": []})
    src = StrategyWeightSource(PromptContext.from_simulation(sim), client=_FakeClient(ans),
                               cache=ResponseCache(path=str(tmp_path)), build_client_on_demand=False)
    terms = bid_terms_v2(v, tasks, ctx, commit, BidModifiers())
    sw = src.modifiers_for_epoch(100, {v.agent_id: (v, tasks)}, "balanced",
                                 bid_terms={v.agent_id: terms}, current_tasks={v.agent_id: commit},
                                 remaining_s=30.0)[v.agent_id]
    assert sorted(e["kind"] for e in src.fallback_events) == ["priority_budget_scaled", "value_budget_scaled"]
    assert sw.w["w_threat"] > DEFAULT_WEIGHTS["w_threat"]  # intent kept, block scaled
    assert src.diagnostics()["prompt_builder_version"] == "m8b.9+three_task"
    assert src.planners[v.agent_id]._PROMPT_VERSION == "m8b.9+three_task"

