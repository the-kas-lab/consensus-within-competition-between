"""The three-task rule: a three-task allocation mode (SCREEN disabled), the four-task model kept.

A POLICY change,
not a correction and not a demonstrated improvement. What must hold:

* three-task mode generates only NEUTRALIZE, DEFEND and RECON; four-task mode
  is unchanged;
* every CBAA arm, the independent one included, uses the configured task set;
* no DEFEND can screen in three-task mode, allocated or implicitly: it holds an
  explicit station, and a SCREEN commitment or override is refused;
* the remaining task mappings and engagement authorization are unchanged;
* the LLM bridge advertises and accepts only the configured tasks, under a
  cache key of its own;
* the mode is recorded, and only when it is not the historical default.

**Unit and constructor level only.** Simulations are built and
``setup_agents()``'d, and the solver is called once at a time, as in
test_defend_target_override.py. No test here runs an episode.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import build_simulation  # noqa: E402
from coordination.bidding import BidContext  # noqa: E402
from coordination.cbaa import CBAAAgent  # noqa: E402
from coordination.runtime import CBAARuntime, output_for  # noqa: E402
from coordination.tasks import (  # noqa: E402
    DEFAULT_TASK_SET,
    DEFEND,
    FOUR_TASK,
    NEUTRALIZE,
    RECON,
    SCREEN,
    THREE_TASK,
    TaskContext,
    generate_tasks,
)
from planning.roles import BlueRole  # noqa: E402
from test_coordination_tasks import _view  # noqa: E402

CONTACTS = {0: ([30.0, 20.0, 0.0, 0.0, 0.0], 38), 1: ([85.0, 70.0, 0.0, 0.0, 0.0], 39)}


@pytest.fixture(scope="module")
def live():
    sim = build_simulation(num_blues=4, num_reds=2)
    sim.setup_agents()
    return sim


def _ctx(sim, task_set=DEFAULT_TASK_SET, slots=1):
    return TaskContext.from_simulation(sim, slots=slots, task_set=task_set)


def _tasks(sim, task_set, slots=1):
    view = _view(sim, contacts=CONTACTS, n_regions=len(sim.coverage_tracker.centers))
    return generate_tasks(view, _ctx(sim, task_set, slots))


# --- 1. the task set -------------------------------------------------------------


def test_the_default_task_set_is_the_historical_four():
    assert DEFAULT_TASK_SET == FOUR_TASK
    ctx = TaskContext(defense_polygon=((0, 0), (1, 0), (0, 1)), stations=())
    assert ctx.task_set == FOUR_TASK


def test_an_unknown_task_set_is_refused(live):
    with pytest.raises(ValueError, match="unknown task set"):
        _ctx(live, "two_task")


def test_three_task_generates_no_screen_and_four_task_still_does(live):
    four = _tasks(live, FOUR_TASK)
    three = _tasks(live, THREE_TASK)
    assert {t.kind for t in four} == {NEUTRALIZE, SCREEN, DEFEND, RECON}
    assert {t.kind for t in three} == {NEUTRALIZE, DEFEND, RECON}
    assert [t.task_id for t in four if t.kind == SCREEN] == ["SCREEN:0", "SCREEN:1"]


@pytest.mark.parametrize("slots", [1, 2])
def test_removing_screen_changes_nothing_else(live, slots):
    """Same ids, locations and canonical order for every remaining task, and
    the slot count is not raised to compensate (requirement 6)."""
    four = [t for t in _tasks(live, FOUR_TASK, slots) if t.kind != SCREEN]
    three = _tasks(live, THREE_TASK, slots)
    assert three == four
    assert sum(t.kind == NEUTRALIZE for t in three) == 2 * slots


# --- 1. the runtime: output_for, the fallback, the arms ------------------------------

COMMITTED = {0: "NEUTRALIZE:1:0", 1: "DEFEND:2", 2: "RECON:5", 3: None}


def test_remaining_mappings_are_identical_across_task_sets():
    four = output_for(COMMITTED, fallback="station", task_set=FOUR_TASK)
    three = output_for(COMMITTED, fallback="station", task_set=THREE_TASK)
    for field in ("roles", "target_red_ids", "defend_targets", "recon_target_regions"):
        assert getattr(three, field) == getattr(four, field), field
    assert three.roles == {0: BlueRole.NEUTRALIZE, 1: BlueRole.DEFEND,
                           2: BlueRole.RECON, 3: BlueRole.DEFEND}
    assert three.defend_targets == {1: ("station", 2), 3: ("station", 3)}


def test_engagement_authorization_is_unchanged():
    """The testbed authorizes engagement for exactly the agents whose role is
    NEUTRALIZE. The roles are identical, so the bit is."""
    def authorized(out):
        return [out.roles.get(i) == BlueRole.NEUTRALIZE for i in range(4)]
    four = output_for(COMMITTED, fallback="station", task_set=FOUR_TASK)
    three = output_for(COMMITTED, fallback="station", task_set=THREE_TASK)
    assert authorized(three) == authorized(four) == [True, False, False, False]


def test_a_screen_commitment_is_refused_without_screen_and_kept_with_it():
    assert output_for({0: "SCREEN:3"}, task_set=FOUR_TASK).defend_targets == {0: ("contact", 3)}
    with pytest.raises(ValueError, match="has no SCREEN"):
        output_for({0: "SCREEN:3"}, fallback="station", task_set=THREE_TASK)


def test_the_legacy_fallback_is_refused_without_screen():
    """legacy = DEFEND with no override = the implicit screen of the nearest
    believed contact."""
    with pytest.raises(ValueError, match="implicit screening"):
        output_for({0: None}, fallback="legacy", task_set=THREE_TASK)
    with pytest.raises(ValueError, match="implicit screening"):
        CBAARuntime(fallback="legacy", task_set=THREE_TASK)
    with pytest.raises(ValueError, match="implicit screening"):
        CBAARuntime(task_set=THREE_TASK)  # legacy is the runtime's default
    assert output_for({0: None}, task_set=FOUR_TASK).defend_targets == {}


def test_the_runtime_renders_through_its_own_task_set():
    """At the call site: CBAARuntime.output must pass its task set on, or a
    SCREEN commitment would be rendered as a screen."""
    rt = CBAARuntime(fallback="station", task_set=THREE_TASK)
    rt.agents = {0: CBAAAgent(0, stable_rounds=1)}
    rt.agents[0].commitment = "SCREEN:1"
    with pytest.raises(ValueError, match="has no SCREEN"):
        rt.output([0])


def _reset(rt, sim, task_set):
    ctx = _ctx(sim, task_set)
    rt.reset(sim.num_blues, ctx, BidContext.from_simulation(sim, ctx))
    return rt


def test_a_mismatched_task_context_is_refused(live):
    rt = CBAARuntime(fallback="station", task_set=THREE_TASK)
    with pytest.raises(ValueError, match="configured with 'three_task'"):
        _reset(rt, live, FOUR_TASK)


def test_the_runtime_refuses_a_generated_kind_outside_its_set(live):
    """Belt and braces behind reset's check: had a four-task context reached a
    three-task auction, start_epoch would stop it before any bid."""
    rt = _reset(CBAARuntime(fallback="station", task_set=THREE_TASK), live, THREE_TASK)
    rt.task_ctx = _ctx(live, FOUR_TASK)
    views = {i: _view(live, agent_id=i, contacts=CONTACTS) for i in range(live.num_blues)}
    with pytest.raises(ValueError, match="outside task set"):
        rt.start_epoch(40, views)


@pytest.mark.parametrize("transport", ["none", "ideal", "network"])
def test_every_transport_bids_on_the_same_three_tasks(live, transport):
    """The independent arm is this runtime with transport "none": no path lets
    an arm generate a different task set."""
    rt = _reset(CBAARuntime(fallback="station", task_set=THREE_TASK, transport=transport),
                live, THREE_TASK)
    views = {i: _view(live, agent_id=i, contacts=CONTACTS) for i in range(live.num_blues)}
    rt.start_epoch(40, views)
    kinds = {t.split(":")[0] for a in rt.agents.values() for t in a.order}
    assert kinds == {NEUTRALIZE, DEFEND}  # no coverage regions in these views
    assert all("SCREEN:0" not in a.bids for a in rt.agents.values())


def test_metrics_name_the_task_set_only_when_it_is_not_the_default(live):
    four = _reset(CBAARuntime(fallback="station"), live, FOUR_TASK).metrics()
    three = _reset(CBAARuntime(fallback="station", task_set=THREE_TASK), live, THREE_TASK).metrics()
    assert "task_set" not in four
    assert three["task_set"] == THREE_TASK
    assert {k: v for k, v in three.items() if k != "task_set"} == four


# --- 2. the solver: no DEFEND can screen when screening is disabled -------------------

from test_defend_target_override import _belief, _prepared, _solve  # noqa: E402

D, N, R = BlueRole.DEFEND, BlueRole.NEUTRALIZE, BlueRole.RECON


def _no_screen(sim):
    sim.solver.defend_screening = False
    return sim


def test_screening_is_on_by_default():
    assert _prepared().solver.defend_screening is True


def test_defend_with_no_override_holds_its_own_station_while_a_contact_is_believed():
    """I1, and I4/I5/I6, which all arrive here as DEFEND with no override. The
    precondition is the historical behaviour this mode must not reproduce."""
    sim = _prepared()
    believed = _belief(sim, c0=(30, 30))
    legacy_u, legacy_eff = _solve(sim, believed)
    assert legacy_eff[0]["source"] == "nearest_believed", "precondition: legacy screens"
    own = {0: ("station", 0), 1: ("station", 1)}
    explicit_u, explicit_eff = _solve(sim, believed, defend_target_override=own)

    _no_screen(sim)
    u, eff = _solve(sim, believed)
    assert eff[0] == {"source": "station", "contact_id": None, "controller": "defend"}
    # Both agents: each must hold ITS OWN station, not some station.
    assert np.allclose(u, explicit_u, atol=1e-9), "must be the explicit own-station hold"
    assert not np.allclose(u[0:3], legacy_u[0:3], atol=1e-6), "must not be the legacy screen"
    assert eff == explicit_eff


def test_an_explicit_station_is_the_same_hold_in_both_modes():
    sim = _prepared()
    believed = _belief(sim, c0=(30, 30))
    override = {0: ("station", 1), 1: ("station", 0)}
    four = _solve(sim, believed, defend_target_override=override)
    three = _solve(_no_screen(sim), believed, defend_target_override=override)
    assert np.allclose(four[0], three[0], atol=1e-9) and four[1] == three[1]


@pytest.mark.parametrize("believed_ids", [(0,), ()])
def test_a_screen_override_is_refused_whether_or_not_its_contact_is_believed(believed_ids):
    """I2: historically an unbelieved screen target fell back to the nearest
    believed contact. With screening disabled any contact override raises."""
    sim = _no_screen(_prepared())
    believed = _belief(sim, c0=(30, 30)) if believed_ids else {0: {}, 1: {}}
    with pytest.raises(ValueError, match="screening is disabled"):
        _solve(sim, believed, defend_target_override={0: ("contact", 0)})


def test_an_unknown_station_is_refused_not_screened():
    """I3: historically an unknown station id fell through to the nearest
    believed contact."""
    sim = _prepared()
    believed = _belief(sim, c0=(30, 30))
    _, eff = _solve(sim, believed, defend_target_override={0: ("station", 99)})
    assert eff[0]["source"] == "nearest_believed", "precondition: the historical fall-through"
    with pytest.raises(ValueError, match="no station 99"):
        _solve(_no_screen(sim), believed, defend_target_override={0: ("station", 99)})


def test_an_agent_missing_from_the_role_assignment_holds_its_station():
    """I6: role_assignment.get(i, DEFEND) makes any missing agent a DEFEND with
    no override."""
    sim = _no_screen(_prepared())
    believed = {0: {}, 1: _belief(sim, c0=(30, 30))[0]}
    _, eff = _solve(sim, believed, roles={0: D})
    assert eff[1]["source"] == "station" and eff[1]["contact_id"] is None


def test_the_outward_posture_reaches_the_implicit_station_too():
    sim = _no_screen(_prepared())
    sim.solver.station_heading = "outward"
    _solve(sim, _belief(sim, c0=(30, 30)))
    assert sim.solver.bearing_sources[0] == "station_outward"


def test_neutralize_and_recon_are_realised_identically_in_both_modes():
    """Requirement 6: the other mappings are untouched, including NEUTRALIZE's
    own fallbacks (pursue the nearest believed contact without an override;
    hold station while believing nothing)."""
    sim = _prepared()
    believed = _belief(sim, c0=(30, 30), c1=(70, 70))
    cases = [
        dict(roles={0: N, 1: R}),
        dict(roles={0: N, 1: R}, target_override={0: believed[0][1]}, target_override_ids={0: 1}),
        dict(roles={0: R, 1: N}),  # agent 1 believes nothing
    ]
    for kw in cases:
        sim.solver.defend_screening = True
        four = _solve(sim, believed, **kw)
        sim.solver.defend_screening = False
        three = _solve(sim, believed, **kw)
        assert np.allclose(four[0], three[0], atol=1e-9), kw
        assert four[1] == three[1], kw


# --- 3. the LLM bridge: advertises, accepts and caches only the configured tasks -------

from coordination.bidding import BidModifiers  # noqa: E402
from coordination.llm_bridge import (  # noqa: E402
    LLMModifierSource,
    LocalModifierPlanner,
    build_bridge_system_prompt,
    build_bridge_user_message,
)
from planning.local_llm import ReplayMiss  # noqa: E402
from planning.response_cache import ResponseCache, cache_key  # noqa: E402
from test_llm_bridge import CTX, MOVING, TASK_CTX, _answer, _Client  # noqa: E402
from test_local_llm_planner import _view as _bare_view  # noqa: E402

TASK_CTX_3 = TaskContext(defense_polygon=TASK_CTX.defense_polygon, stations=TASK_CTX.stations,
                         task_set=THREE_TASK)


def _bridge(policy, task_set, cache=None, **kw):
    return LLMModifierSource(CTX, client=_Client(policy),
                             cache=cache if cache is not None else ResponseCache(),
                             build_client_on_demand=False, task_set=task_set, **kw)


def _epoch(src, view, task_ctx):
    tasks = generate_tasks(view, task_ctx)
    agent = int(view.agent_id)
    terms = {t.task_id: {"value": 1.0, "bid": 1.0} for t in tasks}  # the prompt shows each task's terms
    mods = src.modifiers_for_epoch(int(view.now), {agent: (view, tasks)},
                                   default_profile="balanced", bid_terms={agent: terms},
                                   current_tasks={}, remaining_s=40.0)[agent]
    return tasks, mods


def test_the_three_task_prompt_does_not_advertise_screening():
    three = build_bridge_system_prompt(CTX, THREE_TASK)
    four = build_bridge_system_prompt(CTX, FOUR_TASK)
    assert "screen" not in three.lower()
    assert "SCREEN:c holds the perimeter" in four
    assert "DEFEND:k holds perimeter station k" in three and "NEUTRALIZE:c:s" in three
    assert build_bridge_system_prompt(CTX) == four, "four-task is the default"


def test_the_prompts_actually_sent_carry_no_screen():
    """At the call site: what the client received, from the source's own
    per-agent planner, not what a builder would return if called directly."""
    src = _bridge(lambda u: _answer(), THREE_TASK)
    _epoch(src, _bare_view(contacts=MOVING), TASK_CTX_3)
    ((system, user),) = src.planner_for(0).client.prompts
    assert "screen" not in system.lower() and "SCREEN" not in user
    assert '"NEUTRALIZE:0:0"' in user and '"DEFEND:1"' in user


def test_a_screen_preference_is_dropped_as_unlisted_and_the_rest_kept():
    src = _bridge(lambda u: _answer("conservative", [(0, "HIGH")],
                                    ["SCREEN:0", "DEFEND:1", "NEUTRALIZE:3:0"]), THREE_TASK)
    _, mods = _epoch(src, _bare_view(contacts=MOVING), TASK_CTX_3)
    assert mods == BidModifiers.build("conservative", {0: "HIGH"}, ["DEFEND:1", "NEUTRALIZE:3:0"])
    assert [(e["kind"], e["value"]) for e in src.fallback_events] == [("unlisted_task", "'SCREEN:0'")]


def test_the_same_preference_is_kept_in_four_task():
    src = _bridge(lambda u: _answer(preferred=["SCREEN:0"]), FOUR_TASK)
    _, mods = _epoch(src, _bare_view(contacts=MOVING), TASK_CTX)
    assert mods.preferred_tasks == ("SCREEN:0",) and src.fallback_events == []


def test_the_task_set_is_in_the_key_and_the_provenance():
    four = _bridge(lambda u: _answer(), FOUR_TASK)
    three = _bridge(lambda u: _answer(), THREE_TASK)
    assert four.planner_for(0)._PROMPT_VERSION == "m8b.7", "historical keys unchanged"
    assert three.planner_for(0)._PROMPT_VERSION == "m8b.7+three_task"
    cache = ResponseCache()
    src = _bridge(lambda u: _answer(), THREE_TASK, cache=cache)
    _epoch(src, _bare_view(contacts=MOVING), TASK_CTX_3)
    (record,) = cache._memory.values()
    prov = record["provenance"]
    assert prov["prompt_builder_version"] == "m8b.7+three_task"
    assert set(cache._memory) == {cache_key(
        prompt_builder_version="m8b.7+three_task",
        system_prompt=prov["request"]["system_prompt"],
        user_message=prov["request"]["user_message"],
        model=prov["model"], endpoint=prov["endpoint"], decode_config=prov["decode"])}


def test_a_four_task_cache_never_answers_a_three_task_prompt():
    """A contact-free view gives both task sets the SAME task list, so only the
    system prompt and the version separate the two entries."""
    view = _bare_view()
    assert [t.task_id for t in generate_tasks(view, TASK_CTX)] == \
        [t.task_id for t in generate_tasks(view, TASK_CTX_3)], "precondition"
    cache = ResponseCache()
    _epoch(_bridge(lambda u: _answer(), FOUR_TASK, cache=cache), view, TASK_CTX)
    replay = LLMModifierSource(CTX, cache=cache, replay_only=True, task_set=THREE_TASK)
    with pytest.raises(ReplayMiss):
        _epoch(replay, view, TASK_CTX_3)


def test_a_bridge_handed_screen_tasks_refuses_before_any_call():
    src = _bridge(lambda u: _answer(), THREE_TASK)
    with pytest.raises(ValueError, match="outside this bridge's task set"):
        _epoch(src, _bare_view(contacts=MOVING), TASK_CTX)
    assert src.planner_for(0).client.prompts == []


def test_diagnostics_name_the_task_set_only_when_it_is_not_the_default():
    four, three = _bridge(lambda u: _answer(), FOUR_TASK), _bridge(lambda u: _answer(), THREE_TASK)
    assert "task_set" not in four.diagnostics()
    assert four.diagnostics()["prompt_builder_version"] == "m8b.7"
    assert three.diagnostics()["task_set"] == THREE_TASK
    assert three.diagnostics()["prompt_builder_version"] == "m8b.7+three_task"


def test_the_user_message_is_the_task_list_it_is_given():
    """No task-set logic in the user message: it prints what generate_tasks
    produced, so the task set reaches it only through the tasks."""
    view = _bare_view(contacts=MOVING)
    def msg(task_ctx):
        tasks = generate_tasks(view, task_ctx)
        terms = {t.task_id: {"value": 1.0, "bid": 1.0} for t in tasks}
        return build_bridge_user_message(view, tasks, CTX, terms, None, 40.0)

    three, four = msg(TASK_CTX_3), msg(TASK_CTX)
    assert [l for l in four.splitlines() if "SCREEN" not in l] == three.splitlines()


# --- 4. the Simulation and the runners: one recorded option, one task set ------------

from scripts import arm_variants, network_conditions  # noqa: E402
from scripts.episode_builder import build_episode  # noqa: E402
from simulation.recorder import config_hash, describe_config  # noqa: E402

CBAA = dict(num_blues=4, num_reds=2, allocation="cbaa", belief_mode="shared")
THREE = dict(CBAA, cbaa_fallback="station", cbaa_task_set=THREE_TASK)


def _config(sim):
    """What the recorder records: run() hands describe_config this same dict."""
    return describe_config(sim._init_args)


def test_the_default_simulation_is_four_task_and_records_nothing_new():
    implicit = build_simulation(**dict(CBAA, cbaa_fallback="station"))
    explicit = build_simulation(**dict(CBAA, cbaa_fallback="station", cbaa_task_set=FOUR_TASK))
    for sim in (implicit, explicit):
        assert sim.cbaa_runtime.task_set == FOUR_TASK
        assert sim.solver.defend_screening is True
        assert "cbaa_task_set" not in _config(sim)
    assert config_hash(_config(implicit)) == config_hash(_config(explicit))
    rule = build_simulation()
    assert rule.cbaa_runtime is None and rule.solver.defend_screening is True
    assert "cbaa_task_set" not in _config(rule)


def test_three_task_reaches_the_runtime_the_solver_and_the_record():
    sim = build_simulation(**THREE)
    four = build_simulation(**dict(CBAA, cbaa_fallback="station"))
    assert sim.cbaa_runtime.task_set == THREE_TASK
    assert sim.solver.defend_screening is False
    assert _config(sim)["cbaa_task_set"] == THREE_TASK
    assert config_hash(_config(sim)) != config_hash(_config(four))
    assert {k: v for k, v in _config(sim).items() if k != "cbaa_task_set"} == _config(four)


def test_the_auction_the_run_builds_generates_no_screen():
    """The context run() actually builds (through _reset_allocation), taken off
    the live Simulation: its geometry, its task set."""
    sim = build_simulation(**THREE)
    sim.setup_agents()
    sim._reset_allocation()
    ctx = sim.cbaa_runtime.task_ctx
    assert ctx.task_set == THREE_TASK
    view = _view(sim, contacts=CONTACTS, n_regions=len(sim.coverage_tracker.centers))
    assert {t.kind for t in generate_tasks(view, ctx)} == {NEUTRALIZE, DEFEND, RECON}


@pytest.mark.parametrize("belief_mode, transport", [
    ("shared", "none"), ("shared", "ideal"), ("networked", "network"), ("networked", "none"),
])
def test_every_arm_gets_the_configured_task_set(belief_mode, transport):
    sim = build_simulation(**dict(THREE, belief_mode=belief_mode, cbaa_transport=transport,
                                  comm_interval_seconds=0.1))
    sim.setup_agents()
    sim._reset_allocation()
    assert sim.cbaa_runtime.transport == transport
    assert sim.cbaa_runtime.task_set == sim.cbaa_runtime.task_ctx.task_set == THREE_TASK
    assert sim.solver.defend_screening is False


@pytest.mark.parametrize("overrides, match", [
    (dict(cbaa_task_set=THREE_TASK), "needs allocation='cbaa'"),
    (dict(CBAA, cbaa_task_set=THREE_TASK), "implicit screening"),  # legacy fallback default
    (dict(THREE, cbaa_fallback="legacy"), "implicit screening"),
    (dict(CBAA, cbaa_fallback="station", cbaa_task_set="two_task"), "unknown task set"),
    (dict(cbaa_task_set="two_task"), "unknown task set"),
])
def test_invalid_combinations_fail_at_construction(overrides, match):
    with pytest.raises(ValueError, match=match):
        build_simulation(**overrides)


@pytest.mark.parametrize("sim_set, bridge_set", [(THREE_TASK, FOUR_TASK), (FOUR_TASK, THREE_TASK)])
def test_a_bridge_for_another_task_set_is_refused(sim_set, bridge_set):
    source = _bridge(lambda u: _answer(), bridge_set)
    with pytest.raises(ValueError, match="was built for task set"):
        build_simulation(**dict(CBAA, cbaa_fallback="station", cbaa_task_set=sim_set,
                                cbaa_modifier_source=source))


def test_a_matching_bridge_is_accepted():
    source = _bridge(lambda u: _answer(), THREE_TASK)
    sim = build_simulation(**dict(THREE, cbaa_modifier_source=source))
    assert sim.cbaa_runtime.modifier_source is source


def test_runners_are_unchanged_without_the_key_and_select_three_task_with_it():
    """Exp 5's own configs (both arms, a finite-range condition, the NX anchor)
    build exactly the four-task Simulations they always did; adding the one key
    selects the new mode. Exp 4 used the legacy fallback, so the key alone is
    refused there rather than silently changing the fallback."""
    for condition, arm in (("REF", "I"), ("REF", "C"), ("MH", "C"), ("NX", "I")):
        belief, cfg = network_conditions.episode_cfg("multi_bearing", condition, arm)
        assert "cbaa_task_set" not in cfg
        hist = build_episode(2001, belief, cfg)
        assert hist.cbaa_runtime.task_set == FOUR_TASK and hist.solver.defend_screening
        assert "cbaa_task_set" not in _config(hist)
        new = build_episode(2001, belief, dict(cfg, cbaa_task_set=THREE_TASK))
        assert new.cbaa_runtime.task_set == THREE_TASK and not new.solver.defend_screening
        assert new.cbaa_runtime.transport == hist.cbaa_runtime.transport
        assert {k: v for k, v in _config(new).items() if k != "cbaa_task_set"} == _config(hist)
    exp4 = dict(arm_variants.BASE_CFG, family="symmetric", **arm_variants.ARMS[1])
    assert "cbaa_task_set" not in _config(build_episode(2001, arm_variants.BELIEF_MODE, exp4))
    with pytest.raises(ValueError, match="implicit screening"):
        build_episode(2001, arm_variants.BELIEF_MODE, dict(exp4, cbaa_task_set=THREE_TASK))
