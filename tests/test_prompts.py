"""The objective-only prompts of the reported runs ("v2.3mb"): the blue and red briefs,
the commander prompt, the defender bridge prompt (consensus and independent teams) and the
intruder prompt. The same objective facts in the blue and red prompts; the removed
prescriptions absent. Same static scene helpers as test_bid_threat_forecast (spent seed 3901).
No model calls."""

import os
import sys
from dataclasses import replace

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_bid_threat_forecast import NOW, _belief, _red, _team, _views  # noqa: E402

from coordination.tasks import NEUTRALIZE, generate_tasks  # noqa: E402

# the briefs, verbatim (the 400 / 0.1 / 40 values are the run's).
BLUE_BRIEF_V1 = (
    "Mission (the same for every blue team): your team's objective is to prevent any red from "
    "entering the defended region before the episode ends. The episode lasts 400 steps (dt = "
    "0.1 s, 40 s). Any entry by a red is a breach and ends the episode. A blue that touches a red "
    "destroys it and is permanently disabled. A red that leaves the arena is removed and can no "
    "longer breach. The episode also ends, without a breach, once every red has been destroyed or "
    "has left the arena. Reds can change direction at any time, and reds your team has not "
    "detected may exist.")
RED_BRIEF_V1 = (
    "Your team's objective is for at least one red to enter the defended region before the "
    "episode ends. The episode lasts 400 steps (dt = 0.1 s, 40 s). Any entry by a red is a "
    "breach: your team succeeds and the episode ends. A red that leaves the arena is removed and "
    "can no longer breach. The episode also ends, without a breach, once every red has been "
    "destroyed or has left the arena.")
SECTOR_10 = (
    "Approach sector and sensing: intruders are expected from bearings 160 to 300 degrees as "
    "seen from the defended centre (any distance). A blue holding a station looks along that "
    "station's fixed heading; the approach-sector ground each station's sensor wedge covers is "
    "listed per station. A blue on RECON flies toward its cell with its sensor pointed ahead and "
    "holds there. Ground is watched only while some blue's sensor wedge covers it: a scout adds "
    "coverage only where no other blue is watching at that time, and a station's ground stays "
    "watched after its blue leaves only if another blue's sensor covers it.")
FORECAST_NOTE = (" Every forecast and coverage figure is computed from your picture, which can be "
                 "stale or incomplete.")
ALLOCATION_NOTE = (" Every number below is an allocation value that only ranks tasks for "
                   "allocation; none of them is the mission's objective.")

# The prescriptions removed.
REMOVED_COMMANDER = (
    "sweeping ground nobody watches", "earlier warning", "its station's ground is then unwatched",
    "further from the perimeter", "Seek useful information", "preserving defensive coverage",
    "engage believed reds as the mission policy", "preferring the blue that can reach",
    "Mission policy", "How to apply", "must be engaged", "is still pursued",
    "at least one blue scouts", "Keep a blue on its pursuit", "EVERY listed red",
    "most threatening", "Covering every station must not", "keeps a pursuer",
    "Mission trade-off")
REMOVED_DEFENDER = ("the kill is worth", "points)", "Mission policy", "must be engaged",
                    "is still pursued", "at least one blue scouts", "Keep a blue on its pursuit")
# Objective facts each team's LLM is told.
BLUE_FACTS = ("objective is to prevent any red from entering the defended region before the "
              "episode ends", "The episode lasts 400 steps (dt = 0.1 s, 40 s).",
              "Any entry by a red is a breach", "A blue that touches a red destroys it and is "
              "permanently disabled.", "A red that leaves the arena is removed and can no longer "
              "breach.", "The episode also ends, without a breach, once every red has been "
              "destroyed or has left the arena.")
RED_FACTS = ("objective is for at least one red to enter the defended region before the episode "
             "ends", "The episode lasts 400 steps (dt = 0.1 s, 40 s).",
             "Any entry by a red is a breach", "a blue that touches you destroys you (and "
             "itself).", "A red that leaves the arena is removed and can no longer breach.",
             "The episode also ends, without a breach, once every red has been destroyed or has "
             "left the arena.")


@pytest.fixture(scope="module")
def scene():
    from coordination import bidding
    from coordination.tasks import TaskContext
    from scripts import run_competing_teams as rct

    e = {"arm": "cbaa", "reds": 4, "family": "multi_bearing", "condition": "REF",
         "attacked": False, "seed": 3901, "design": "v2.3mb"}
    e["episode_id"] = rct.episode_id(e)
    sim, _ = rct.build(e, "rule")
    assert sim.cbaa_bid_version == "v2.3" and (sim.sim_steps, sim.dt) == (400, 0.1)
    task_ctx = TaskContext.from_simulation(sim, task_set="three_task")
    return sim, task_ctx, bidding.BidContext.from_simulation(sim, task_ctx)


# -- briefs ------------------------------------------------------------------------------------


def test_briefs_are_the_frozen_texts_with_the_runs_episode_length():
    from coordination.mission_policy import blue_brief_v1, red_brief_v1

    assert blue_brief_v1(400, 0.1) == BLUE_BRIEF_V1
    assert red_brief_v1(400, 0.1) == RED_BRIEF_V1
    assert "300 steps (dt = 0.1 s, 30 s)" in blue_brief_v1(300, 0.1)
    assert "600 steps (dt = 0.05 s, 30 s)" in red_brief_v1(600, 0.05)
    for f in (blue_brief_v1, red_brief_v1):
        with pytest.raises(ValueError, match="episode length"):
            f(None, 0.1)


def test_brief_parity_across_the_blue_and_red_prompts(scene):
    """The same objective facts reach every LLM of both teams: breach objective,
    episode length, breach ends the episode, contact, escape removal, all-resolved
    end. The red contact fact is the intruder prompt's game sentence, so it is checked
    at the prompt level."""
    from coordination.commander_llm import _system_prompt
    from coordination.llm_bridge import build_bridge_system_prompt
    from planning.intruder_llm import system_prompt
    from planning.prompt_builder import PromptContext

    sim = scene[0]
    pc = PromptContext.from_simulation(sim)
    blue = [_system_prompt(400, 0.1)] + [
        build_bridge_system_prompt(pc, "three_task", sel)
        for sel in ("cbaa", "independent")]
    red = [system_prompt(ph, 400, 0.1) for ph in ("approach", "interior")]
    assert len(BLUE_FACTS) == len(RED_FACTS) == 6
    for p in blue:
        assert p.count(BLUE_BRIEF_V1) == 1
        for fact in BLUE_FACTS:
            assert fact in p, fact
    for p in red:
        assert p.count(RED_BRIEF_V1) == 1
        for fact in RED_FACTS:
            assert fact in p, fact
    # Mirror facts already present before v2.3mb: unseen opponents may exist.
    assert all("reds your team has not detected may exist" in p for p in blue)
    assert all("blues you have not seen may exist" in p for p in red)


# -- commander prompt -----------------------------------------------------------------------


def test_commander_prompt_states_the_brief_and_has_no_prescription_left():
    from coordination.commander_llm import _system_prompt

    c = _system_prompt(400, 0.1)
    assert c.count(BLUE_BRIEF_V1) == 1 and SECTOR_10 in c and FORECAST_NOTE in c
    for frag in REMOVED_COMMANDER:
        assert frag not in c, frag
    for word in ("must", "should", "prefer"):
        assert word not in c.lower(), word


def _capture_commander(geo, view, catalogue, positions, history):
    from coordination.commander_llm import LLMCommanderDecider
    from planning.response_cache import ResponseCache

    d = LLMCommanderDecider(cache=ResponseCache(enabled=False), client=object())
    d.geometry = geo
    d.observe_history(history)
    got = {}

    def fetch(system, user, validation_context=None):
        got.update(system=system, user=user, ctx=validation_context, schema=d._SCHEMA)
        raise RuntimeError("captured")

    d._fetch = fetch
    with pytest.raises(RuntimeError, match="captured"):
        d.decide(view, catalogue, positions)
    return got


def test_commander_user_message_schema_and_context(scene):
    from coordination.commander_llm import _system_prompt, commander_geometry

    sim, task_ctx, ctx = scene
    geo = commander_geometry(sim, task_ctx)
    team = _team(ctx)
    reds = {0: _belief(0, *_red(sim, ctx, 220, 50, "approaching", 2.0)),
            1: _belief(1, *_red(sim, ctx, 260, 30, "moving_away", 2.0))}
    vw = _views(sim, ctx, team, reds)[0]
    pos = {i: s[0:2] for i, (s, _) in team.items()}
    cat = generate_tasks(vw, task_ctx)
    hist = (90, {2: "NEUTRALIZE:1:0", 1: "RECON:5"})
    g = _capture_commander(geo, vw, cat, pos, hist)
    assert g["system"] == _system_prompt(400, 0.1)
    assert "last issued assignment NEUTRALIZE:1:0 (issued at step 90)" in g["user"]
    assert list(g["schema"]["properties"]) == ["reasoning", "task_by_agent"]
    assert g["ctx"]["prompt_version"] == "commander.10"
    # The brief's episode length is the run's (the forecast context's horizon).
    fc = geo["forecast_ctx"]
    short = dict(geo, forecast_ctx=replace(fc, v2=replace(fc.v2, horizon_steps=300)))
    g = _capture_commander(short, vw, cat, pos, hist)
    assert "The episode lasts 300 steps (dt = 0.1 s, 30 s)." in g["system"]


# -- defender bridge prompt ------------------------------------------------------------------


@pytest.mark.parametrize("sel", ["cbaa", "independent"])
def test_defender_prompt_states_the_brief_and_the_allocation_values(scene, sel):
    from coordination.llm_bridge import build_bridge_system_prompt
    from planning.prompt_builder import PromptContext

    pc = PromptContext.from_simulation(scene[0])
    assert (pc.horizon_steps, pc.dt) == (400, 0.1)
    s = build_bridge_system_prompt(pc, "three_task", sel)
    assert s.count(BLUE_BRIEF_V1) == 1 and ALLOCATION_NOTE in s
    assert " (you are permanently disabled on contact)." in s
    for frag in REMOVED_DEFENDER:
        assert frag not in s, frag
    # the arm's allocation sentence: the independent arm hears of no auction
    assert ("auction" in s.lower()) == (sel == "cbaa")


def test_defender_user_message_schema_and_version(scene):
    from coordination import bidding
    from coordination.llm_bridge import (_MODIFIER_SCHEMA, LocalModifierPlanner,
                                         build_bridge_system_prompt, prompt_version)
    from planning.prompt_builder import PromptContext

    sim, task_ctx, ctx = scene
    pc = PromptContext.from_simulation(sim)
    assert prompt_version("three_task") == "m8b.7+three_task"
    assert list(_MODIFIER_SCHEMA["properties"])[0] == "reasoning"
    p = LocalModifierPlanner(0, pc, task_set="three_task", selection="independent", client=object())
    assert p._PROMPT_VERSION == "m8b.7+three_task" and p._SCHEMA == _MODIFIER_SCHEMA
    with pytest.raises(ValueError, match="episode length"):
        build_bridge_system_prompt(PromptContext(arena_size=100.0), "three_task")


# -- intruder.4 --------------------------------------------------------------------------------


def _obs(phase):
    from planning.intruder_llm import observation_from_dict

    return observation_from_dict({
        "kind": "intruder", "agent_id": 0, "now": 50, "dt": 0.1,
        "own_state": [10.0, 10.0, 1.0, 0.0, 0.0], "phase": phase,
        "commitment": {"role": "attack", "gate_id": 2, "evade_blue_id": None},
        "teammates": {"1": [20.0, 10.0, 1.0, 0.0, 1.5708]}, "teammate_intents": {},
        "believed_blues": {"3": [[30.0, 30.0, 0.0, -1.0, 3.14159], 40]},
        "gates": [[2, [40.0, 40.0]]], "defense_center": [50.0, 50.0]})


@pytest.mark.parametrize("phase", ["approach", "interior"])
def test_intruder_prompt_states_the_red_brief_and_needs_the_horizon(phase):
    from planning.intruder_llm import system_prompt

    s = system_prompt(phase, 400, 0.1)
    game = ("Your team tries to breach a defended region; blue defenders try to intercept you, "
            "and a blue that touches you destroys you (and itself).")
    assert s.count(game + " " + RED_BRIEF_V1) == 1
    with pytest.raises(ValueError, match="episode length"):
        system_prompt(phase, None, 0.1)


def test_intruder_user_message_and_context_and_the_horizon_check():
    from planning.intruder_llm import LLMIntruderTeam, user_message
    from planning.response_cache import ResponseCache

    team = LLMIntruderTeam(cache=ResponseCache(enabled=False), client=object())
    p = team.planner_for(0)  # created before the horizon is set
    with pytest.raises(ValueError, match="set_episode_horizon"):
        p.decide(_obs("approach"))
    team.set_episode_horizon(400, 0.1)
    p = team.planner_for(0)
    got = {}

    def fetch(system, user, validation_context=None):
        got.update(system=system, user=user, ctx=validation_context, schema=p._SCHEMA)
        raise RuntimeError("captured")

    p._fetch = fetch
    assert p.decide(_obs("approach")) == "call_failure:RuntimeError"
    assert RED_BRIEF_V1 in got["system"] and got["user"] == user_message(_obs("approach"))
    assert got["ctx"]["prompt_version"] == "intruder.4"
    team = LLMIntruderTeam(client=object())
    team.set_episode_horizon(400, 0.05)
    with pytest.raises(ValueError, match="dt"):
        team.planner_for(0).decide(_obs("approach"))


# -- design v2.3mb -----------------------------------------------------------------------------


def test_the_design_uses_the_frozen_interface_versions():
    from scripts.experiment_policy import DESIGN_POLICY, DESIGNS
    from scripts.inference_profile import INTERFACE_VERSIONS

    assert DESIGNS["v2.3mb"] == DESIGN_POLICY
    assert (INTERFACE_VERSIONS["bridge_version"], INTERFACE_VERSIONS["commander_prompt"],
            INTERFACE_VERSIONS["intruder_prompt"]) == ("m8b.7", "commander.10", "intruder.4")


@pytest.mark.parametrize("arm", ["cbaa", "independent", "commander"])
def test_the_design_builds_with_the_profile_and_rule_backends(arm):
    from scripts import run_competing_teams as rct
    from scripts.inference_profile import InferenceProfile

    prof = InferenceProfile(profile_id="p", backend="vllm", model="m",
                            max_tokens={"defender": 100, "commander": 100, "intruder": 100})
    e = {"arm": arm, "reds": 2, "family": "multi_bearing", "condition": "REF",
         "attacked": False, "seed": 3901, "design": "v2.3mb"}
    e["episode_id"] = rct.episode_id(e)
    sim, planners = rct.build(e, "profile", {"profile_obj": prof, "replay": True,
                                             "cache_dir": None, "agent_workers": 1})
    assert sim.cbaa_bid_version == "v2.3"
    red = planners["intruders"]
    assert red.episode_horizon == (sim.sim_steps, sim.dt) == (400, 0.1)
    if arm == "commander":
        assert planners["commander"]._PROMPT_VERSION == "commander.10"
        assert "forecast_ctx" in planners["commander"].geometry
    else:
        src = planners["defenders"]
        assert src.bridge_version == "m8b.7" and src.selection == arm
        assert (src.context.horizon_steps, src.context.dt) == (400, 0.1)
    sim, planners = rct.build(e, "rule")
    assert "commander" not in planners and "intruders" not in planners
