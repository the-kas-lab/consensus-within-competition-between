"""Bridge "m8b.9": each blue's LLM writes the coefficients of
its own bid formula (coordination.strategy_weights), not a profile name.

What it shares with the defender bridge (llm_bridge): the role sentence, the objective-only blue brief
(``blue_brief_v1``), the sensor geometry, the incomplete-belief sentence, the
allocation-value sentence, the frozen (view, tasks) snapshot per agent per
epoch, concurrent per-agent calls, and the cached call path with provenance.

What changes: the answer is ``reasoning`` (first), ``weights`` (all twelve,
numbers), ``contact_priorities`` (a number per believed contact) and
``preferred_tasks``; the system prompt states the formula, every weight's
default and allowed range and both budgets, generated from
``strategy_weights.WEIGHT_FIELDS``; each candidate task line shows the
task's unweighted features and its bid under the default weights. No tactic is
named or suggested (test-enforced). No memory of earlier answers and no
teammate weights.

A total failure (no answer, unparseable, transport failure, no ``weights``
object) returns None, which the runtime replaces with the default weights --
exactly rule-based CBAA -- and records.
"""

import json
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

from coordination.strategy_weights import (BUDGET_FIELDS, DEFAULT_WEIGHTS, MAX_PREFERRED,
                                           PRIORITY_RANGE, VALUE_BUDGET, WEIGHT_FIELDS,
                                           StrategyWeights, features, spec_identity,
                                           validate_answer)
from coordination.tasks import DEFEND, NEUTRALIZE, RECON, THREE_TASK
from planning.local_llm import LocalLLMPlanner, PerAgentLLMPlanner, TransportFailure
from planning.prompt_builder import PromptContext, _vec, build_user_message
from planning.stops import StopRun

STRATEGY_BRIDGE_VERSION = "m8b.9"  # the label in every cache key; do not change
PROMPT_VERSION = f"{STRATEGY_BRIDGE_VERSION}+{THREE_TASK}"


def prompt_version() -> str:
    return PROMPT_VERSION


SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "weights": {
            "type": "object",
            "properties": {f.name: {"type": "number"} for f in WEIGHT_FIELDS},
            "required": [f.name for f in WEIGHT_FIELDS],
            "additionalProperties": False,
        },
        "contact_priorities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"contact_id": {"type": "integer"}, "priority": {"type": "number"}},
                "required": ["contact_id", "priority"],
                "additionalProperties": False,
            },
        },
        "preferred_tasks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reasoning", "weights", "contact_priorities", "preferred_tasks"],
    "additionalProperties": False,
}


def _weight_table() -> str:
    return "\n".join(f"- {f.name}: {f.term}. Default {f.default:g}, allowed {f.lo:g} to {f.hi:g}."
                     for f in WEIGHT_FIELDS)


_BUDGET_SCALE = ("If the five budgeted weights sum to more than the budget, all five are scaled down "
                 "in proportion until they sum to the budget; likewise, if the priorities of the "
                 "contacts you believe in sum to more than their number, all of them are scaled down "
                 "in proportion until they sum to their number. So raising one budgeted weight (or one "
                 "priority) without lowering the others lowers the others. ")


def _mechanics() -> str:
    from coordination import bidding as bd
    from coordination import coverage_value as cv
    from coordination.bidding import PREFERRED_BONUS, V22_PROFILES

    B = V22_PROFILES["balanced"]
    lo, hi = cv.PRIOR_SECTOR_DEG
    budget = " + ".join(BUDGET_FIELDS)
    return (
        "How bids are formed (the highest bid is your task unless the auction gives it to a "
        "teammate who bids more for it). Every blue computes its bids with the same formula from "
        "its own beliefs and its own weights. You choose the weights YOUR bids use; teammates "
        "choose theirs. Every number below is an allocation value that only ranks tasks for "
        "allocation; none of them is the mission's objective.\n"
        "- DEFEND:k bid = w_station + w_coverage x coverage - w_travel x distance/25 - penalties\n"
        "- RECON:r bid = w_scout x information - w_scout_travel x distance/25 - penalties\n"
        "- NEUTRALIZE:c bid = priority(c) x (w_threat + w_engage x urgency x opportunity) "
        "- w_travel x distance/25 - w_attrition x 1/(1 + teammates) - w_belief_age x age term "
        "- penalties\n"
        "- penalties = w_leave x leaving + w_switch x switching + w_continuity x continuity; "
        f"then plus {PREFERRED_BONUS:g} for each of your preferred tasks.\n"
        "Features (each candidate task line below lists its own):\n"
        f"- Approach sector: intruders are expected from bearings {lo:g} to {hi:g} degrees as "
        "seen from the defended centre (any distance). Only ground in that sector counts below.\n"
        "- coverage: how much approach-sector ground station k's sensor wedge (at the station's "
        "fixed heading) adds beyond what your teammates' sensors watch now (1.0 = one full "
        "sensor's worth).\n"
        "- information: approach-sector ground your sensor sweeps on the way to region r and "
        "there, counted by how stale each region is in YOUR coverage record and by how soon you "
        "reach it (an optimistic travel forecast from your current velocity); ground a blue "
        "already watches counts 0.\n"
        "- urgency: the larger of 1/(1 + forecast time to breach / "
        f"{B.t0:g} s) and 1/(1 + distance to the region / 25). Forecast time to breach: from the "
        "contact's believed velocity, the time to close on the approach funnel (the region "
        "together with the four public approach gates) at its current closing speed, plus the "
        "last leg into the region; none when it is not closing (moving away, crossing or "
        "stationary).\n"
        "- opportunity: from an OPTIMISTIC kinematic interception forecast (from your current "
        "velocity, with the per-axis speed and acceleration limits of 2; it ignores collision "
        "avoidance and the contact's reaction): 1 if you can reach the contact's forecast path "
        "before its forecast breach and within the episode, falling linearly to "
        f"{B.o_min:g} over {B.t1:g} s later than that, {B.o_min:g} with no interception; "
        f"averaged over +-{cv.VELOCITY_UNCERTAINTY:g} u/s of uncertainty in the contact's "
        "velocity.\n"
        "- age term: the belief's age x the reds' top speed / 25.\n"
        "- leaving: the coverage of the station you would leave (0 if you hold no station or "
        "the task is that station).\n"
        "- switching: 1 for every task other than your current task, else 0.\n"
        "- continuity: while you pursue a contact you still believe in, 1 for every other task "
        "(RECON, DEFEND); for ANOTHER CONTACT it is (1 - R x H x S), down to 0: R = how fast the "
        "contact you pursue moves away from the approach funnel (0 if it is not moving away, 1 at "
        f"{bd.V23_RECEDE_SPEED:g} u/s or faster); H = 1 if the other contact's forecast breach "
        f"comes at least {bd.V23_HORIZON_RAMP_S:g} s before the episode ends, falling to 0 at "
        "the end (0 with no forecast breach); S = how far your optimistic interception of the "
        "other contact comes before its forecast breach (0 for no lead, 1 for "
        f"{bd.V23_MARGIN_RAMP_S:g} s or more). 0 when you pursue no contact.\n"
        "Weights (what each multiplies, default, allowed range):\n"
        + _weight_table() + "\n"
        f"- Budget: {budget} must not exceed {VALUE_BUDGET} (the defaults sum to exactly "
        f"{VALUE_BUDGET}).\n"
        f"- priority(c): a number from {PRIORITY_RANGE[0]:g} to {PRIORITY_RANGE[1]:g} per contact "
        "you believe in, default 1 for any contact you leave out; the priorities of all the "
        "contacts you believe in must not sum to more than their number.\n"
        "A value outside its allowed range is replaced by its default. "
        + _BUDGET_SCALE
        + "A low "
        "weight or priority lowers a bid; it does not remove the task, and a blue that is "
        "outbid takes its best remaining task, even one with a negative bid. Your candidate task "
        "list shows each task's features and its bid under the default weights.")


def build_system_prompt(ctx: PromptContext) -> str:
    from coordination.llm_bridge import _SELECTION_SENTENCE
    from coordination.mission_policy import blue_brief_v1
    from planning.prompt_builder import SENSOR_GEOMETRY_TEXT

    if ctx.horizon_steps is None or ctx.dt is None:
        raise ValueError("the prompt states the episode length; build the PromptContext with "
                         "PromptContext.from_simulation")
    selection = _SELECTION_SENTENCE["cbaa"].replace(
        "You choose how YOUR bids are weighted.", "You choose the weights YOUR bids use.")
    return f"""You are ONE defending agent (BLUE) in a 2D pursuit-evasion simulation. Blue defends a region against RED intruders. {selection}

Answer with four things, in this order:

1. reasoning -- FIRST: at most two sentences about the actual picture and what your weights trade off.

2. weights -- all {len(WEIGHT_FIELDS)} weights listed below, each a number within its allowed range.

3. contact_priorities -- for contacts YOU believe in (listed below), a number from {PRIORITY_RANGE[0]:g} to {PRIORITY_RANGE[1]:g}. 1 is the default for any contact you leave out. You may only name contact_id values that are listed below.

4. preferred_tasks -- at most {MAX_PREFERRED} task ids, each copied exactly as one of the quoted task_id strings in YOUR candidate task list below (the text inside the quotes only, without the location). Leave it empty if nothing stands out.

Task kinds: NEUTRALIZE:c:s closes to contact c (you are permanently disabled on contact). DEFEND:k holds perimeter station k. RECON:r patrols toward coverage region r.

What you know is what YOU believe, and it is incomplete. Contacts you cannot see are not listed, and a short list does NOT mean the region is safe. Other agents are answering at the same moment on their own beliefs.

{_mechanics()}

{blue_brief_v1(ctx.horizon_steps, ctx.dt)}

{SENSOR_GEOMETRY_TEXT}

Respond only with a valid JSON object matching the required schema, with "reasoning" as its first field."""


_STATUS = {"approaching": "approaching", "moving_away": "moving away", "crossing": "crossing",
           "stationary": "stationary", "inside": "inside the region"}


def _task_line(task, t, remaining_s) -> str:
    f = features(t, task.kind)
    if task.kind == DEFEND:
        head = f"coverage {f['coverage']:.2f}"
    elif task.kind == RECON:
        head = f"information {f['information']:.2f}"
    else:
        tau, ti = t["time_to_breach"], t["intercept_time"]
        ic = ("none" if not math.isfinite(ti) else
              f"{ti:.1f} s (after the episode ends)" if ti > remaining_s else f"{ti:.1f} s")
        head = (f"urgency {f['urgency']:.2f} x opportunity {f['opportunity']:.2f} "
                f"({_STATUS.get(t['breach_category'], t['breach_category'])}, forecast breach "
                f"{'none' if not math.isfinite(tau) else f'{tau:.1f} s'}, interception forecast {ic}), "
                f"1/(1+teammates) {f['attrition']:.2f}, age term {f['age']:.2f}")
    parts = [head, f"distance/25 {f['dist']:.2f}", f"leaving {f['leave']:.2f}",
             f"switching {f['switch']:.0f}", f"continuity {f['continuity']:.2f}"]
    return ", ".join(parts) + f" -> default bid {t['bid']:.2f}"


def build_user_message(view, tasks: Sequence, ctx: PromptContext, bid_terms, current_task,
                       remaining_s, max_age=None) -> str:
    from planning.prompt_builder import build_user_message as base

    head = (f"your current task: {current_task or 'none (holding your own station)'}\n"
            f"episode time left: {remaining_s:.1f} s\n")
    n = len({int(t.contact_id) for t in tasks if t.kind == NEUTRALIZE})
    head += (f"your budgets: {' + '.join(BUDGET_FIELDS)} sum to at most {VALUE_BUDGET}; "
             + (f"the priorities of your {n} believed contact{'s' if n != 1 else ''} sum to at most {n}\n"
                if n else "no believed contacts, so no priorities\n"))
    lines = [base(view, ctx, max_age=max_age, with_headings=True),
             head + "your candidate tasks (task_id, where it is, features, bid under the default weights):"]
    for task in tasks:
        lines.append(f'  task_id "{task.task_id}"    at {_vec(task.location)}    '
                     f'{_task_line(task, bid_terms[task.task_id], remaining_s)}')
    return "\n".join(lines)


class LocalStrategyPlanner(LocalLLMPlanner):
    """One blue's strategy-weight bridge on LocalLLMPlanner's cached call path."""

    _SCHEMA_NAME = "strategy_weights"
    _SCHEMA = SCHEMA
    _PROMPT_VERSION = PROMPT_VERSION

    bridge_version = STRATEGY_BRIDGE_VERSION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_prompt: Optional[Tuple[str, str]] = None

    def plan_one(self, view, max_age=None):
        raise TypeError("a strategy-weight bridge chooses weights, not roles")

    def _event(self, view, kind, **fields):
        event = {"sim_step": int(view.now), "agent_id": int(view.agent_id), "kind": kind}
        event.update(fields)
        self.fallback_events.append(event)

    def strategy_for(self, view, tasks: Sequence, bid_terms, current_task, remaining_s,
                     max_age=None) -> Optional[StrategyWeights]:
        stray = sorted({t.kind for t in tasks} - {NEUTRALIZE, DEFEND, RECON})
        if stray:
            raise ValueError(f"agent {view.agent_id}: task kinds {stray} outside the three-task model")
        system_prompt = build_system_prompt(self.context)
        user_message = build_user_message(view, tasks, self.context, bid_terms, current_task,
                                          remaining_s, max_age=max_age)
        believed = sorted(int(c) for c in view.fresh_targets(max_age))
        system_prompt, images = self._image_request(system_prompt, believed)
        self.last_prompt = (system_prompt, user_message)
        context = {"kind": "strategy_defender", "agent_id": int(view.agent_id), "now": int(view.now),
                   "believed_contacts": believed, "task_ids": [t.task_id for t in tasks]}
        try:
            content = self._fetch(system_prompt, user_message, validation_context=context,
                                  **({"images": images} if images else {}))
        except StopRun:
            raise
        except TransportFailure as exc:
            self._event(view, "call_failure", error_type=exc.error_type, applied="default_weights")
            return None
        except Exception as exc:  # noqa: BLE001
            self._event(view, "call_failure", error_type=type(exc).__name__,
                        applied="default_weights")
            return None
        return self.interpret(content, view, tasks, believed)

    def interpret(self, content, view, tasks, believed) -> Optional[StrategyWeights]:
        if not (content or "").strip():
            self._event(view, "empty_response", applied="default_weights")
            return None
        try:
            answer = json.loads(content)
        except json.JSONDecodeError as exc:
            self._event(view, "schema_failure", error_type=type(exc).__name__,
                        applied="default_weights")
            return None
        if not isinstance(answer, dict) or not isinstance(answer.get("weights"), dict):
            self._event(view, "schema_failure", error_type="no_weights_object",
                        applied="default_weights")
            return None
        reasoning = answer.get("reasoning")
        self.last_reasoning = reasoning if isinstance(reasoning, str) else ""
        sw, events = validate_answer(answer, believed, [t.task_id for t in tasks], source="llm")
        for e in events:
            kind = e.pop("kind")
            self._event(view, kind, **e)
        return sw


class StrategyWeightSource(PerAgentLLMPlanner):
    """What CBAARuntime calls once per epoch."""

    _LOCAL_CLASS = LocalStrategyPlanner
    bridge_version = STRATEGY_BRIDGE_VERSION
    task_set = THREE_TASK
    selection = "cbaa"

    def __init__(self, context: PromptContext, *args, max_workers: int = 1, **kwargs):
        super().__init__(context, *args, **kwargs)
        self.max_workers = max(1, int(max_workers))
        self.bridge_version = type(self).bridge_version  # an instance attribute, so the recorded configuration names it
        self.modifier_log: List[Dict] = []

    def plan_one(self, view, max_age=None):
        raise TypeError("StrategyWeightSource chooses bid weights, not roles")

    def modifiers_for_epoch(self, step: int, requests: Dict[int, Tuple[object, Sequence]],
                            default_profile: str, max_age=None, bid_terms=None,
                            current_tasks=None, remaining_s=None) -> Dict[int, Optional[StrategyWeights]]:
        if bid_terms is None or remaining_s is None:
            raise ValueError("the prompt shows each task's features; the runtime must supply the "
                             "default bid terms and the remaining time (bid version v2.3)")
        agents = sorted(requests)
        planners = {i: self.planner_for(i) for i in agents}

        def call(i):
            view, tasks = requests[i]
            return planners[i].strategy_for(view, tasks, bid_terms[i], (current_tasks or {}).get(i),
                                            remaining_s, max_age=max_age)

        if self.max_workers == 1 or len(agents) <= 1:
            results = {i: call(i) for i in agents}
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(agents))) as pool:
                futures = {i: pool.submit(call, i) for i in agents}
                results = {i: futures[i].result() for i in agents}
        for i in agents:
            sw = results[i]
            self.modifier_log.append({"step": int(step), "agent_id": int(i),
                                      "strategy": None if sw is None else sw.as_dict(),
                                      "reasoning": planners[i].last_reasoning if sw is not None else None})
        return results

    def diagnostics(self) -> Dict:
        d = super().diagnostics()
        d["bridge"] = True
        d["prompt_builder_version"] = prompt_version()
        d["task_set"] = self.task_set
        d["strategy_spec"] = spec_identity()
        d["max_workers"] = self.max_workers
        d["epochs_x_agents"] = len(self.modifier_log)
        d["defaulted_after_total_failure"] = sum(1 for e in self.modifier_log if e["strategy"] is None)
        return d
