"""Semantic validation of a recorded LLM response, offline, with the real validators.

Added: JSON parse validity is not
acceptance. Each response is judged by the SAME code the simulator applies to
it live, against the validation context recorded with the request
(``LocalLLMPlanner._fetch(validation_context=...)``):

* defender  -- ``LocalModifierPlanner.interpret`` (profile, believed-contact
  priorities, listed preferred tasks, cap) on a view reduced to the fields
  ``_validate`` reads;
* intruder  -- ``_OneIntruder.interpret`` then the simulator's
  ``validate_decision`` and, on failure, ``fallback_decision`` on the
  rebuilt ``IntruderObservation``;
* commander -- ``LLMCommanderDecider.interpret`` then
  ``validate_assignment`` (catalogue, one task per agent, capacity one).

Statuses are kept apart, never merged into one "valid" count:

* ``accepted``        -- applied as answered;
* ``partial``         -- applied with drops (defender: some fields dropped,
                         rest kept; commander: some agents fell back to their
                         own station, the rest kept);
* ``fallback``        -- nothing of the answer applied (defender: default
                         modifiers; intruder: kept commitment / declared
                         ATTACK; commander: every agent's station fallback).
"""

from typing import Dict

ACCEPTED = "accepted"
PARTIAL = "partial"
FALLBACK = "fallback"


class _View:
    """The fields ``LocalModifierPlanner._validate`` and ``_event`` read."""

    def __init__(self, ctx):
        self.agent_id = int(ctx["agent_id"])
        self.now = int(ctx["now"])
        self._believed = {int(c): None for c in ctx["believed_contacts"]}

    def fresh_targets(self, max_age=None):
        return dict(self._believed)  # already filtered at capture time


class _Task:
    def __init__(self, task_id):
        self.task_id = task_id
        self.kind = task_id.split(":", 1)[0]


def _defender(content, ctx, task_set="three_task"):
    from coordination.llm_bridge import LocalModifierPlanner
    from planning.prompt_builder import PromptContext

    planner = LocalModifierPlanner(int(ctx["agent_id"]), PromptContext(arena_size=0.0),
                                   task_set=task_set)
    view = _View(ctx)
    tasks = [_Task(t) for t in ctx["task_ids"]]
    default = ctx.get("default_profile") or "balanced"
    mods = planner.interpret(content, view, tasks, default)
    events = [{k: v for k, v in e.items() if k not in ("sim_step", "agent_id")}
              for e in planner.fallback_events]
    if mods is None:
        status = FALLBACK
    else:
        status = PARTIAL if events else ACCEPTED
    applied = None if mods is None else {
        "scoring_profile": mods.scoring_profile,
        "threat_priorities": [list(p) for p in mods.threat_priorities],
        "preferred_tasks": list(mods.preferred_tasks)}
    return {"status": status, "events": events, "applied": applied,
            "drops": [e["kind"] for e in events]}


def _intruder(content, ctx):
    from planning.intruder import fallback_decision, validate_decision
    from planning.intruder_llm import _OneIntruder, observation_from_dict

    obs = observation_from_dict(ctx)
    planner = _OneIntruder(obs.agent_id, context=None)
    out = planner.interpret(content, obs.phase)
    # Exactly the simulator's verdict (a parse failure is "not_a_decision"),
    # plus the planner's own named reason when it gave one.
    invalid = validate_decision(out, obs)
    if invalid is None:
        return {"status": ACCEPTED, "events": [], "drops": [], "invalid_reason": None,
                "applied": {"role": out.role, "gate_id": out.gate_id,
                            "evade_blue_id": out.evade_blue_id}}
    detail = out if isinstance(out, str) else invalid
    decision, fb = fallback_decision(obs)
    return {"status": FALLBACK, "events": [{"kind": detail.split(":")[0], "detail": detail}],
            "drops": [detail.split(":")[0]], "invalid_reason": invalid, "fallback": fb,
            "applied": {"role": decision.role, "gate_id": decision.gate_id,
                        "evade_blue_id": decision.evade_blue_id}}


def _commander(content, ctx):
    from coordination.commander import validate_assignment
    from coordination.commander_llm import LLMCommanderDecider

    error = None
    try:
        raw = LLMCommanderDecider(int(ctx.get("commander", 0))).interpret(content)
    except Exception as exc:  # noqa: BLE001 -- the live path's unusable response
        raw, error = None, f"{type(exc).__name__}"
    assignment, report = validate_assignment(raw, ctx["catalogue"], ctx["active_agents"])
    failures = [{"kind": r, "agent": a, "detail": d} for a, r, d in report["failures"]]
    if raw is None:
        status = FALLBACK
    elif report["fallback"]:
        status = PARTIAL
    else:
        status = ACCEPTED
    return {"status": status, "events": failures, "drops": [f["kind"] for f in failures],
            "error": error, "applied": {str(a): t for a, t in sorted(assignment.items())},
            "n_agents": len(ctx["active_agents"]), "n_fallback": len(report["fallback"]),
            "fallback_conflicts": len(report["fallback_conflicts"])}


def validate(content: str, ctx: Dict, **defender_kwargs) -> Dict:
    """Judge one recorded response against its recorded validation context."""
    kind = ctx["kind"]
    if kind == "defender":
        return _defender(content, ctx, **defender_kwargs)
    if kind == "intruder":
        return _intruder(content, ctx)
    if kind == "commander":
        return _commander(content, ctx)
    raise ValueError(f"unknown validation context kind {kind!r}")

