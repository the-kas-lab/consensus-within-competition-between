"""The LLM-to-auction bridge: an LLM chooses bounded bid modifiers, never a bid.

's division of labour, at the seam the auction fixed
(). Each blue's LLM reads its own
frozen ``LocalWorldView`` plus its own candidate task list and answers with:

    scoring_profile     one of bidding.PROFILES' names
    threat_priorities   [{contact_id, priority: LOW | MEDIUM | HIGH}]
    preferred_tasks     up to MAX_PREFERRED task ids from its candidate list

Every one of those is a label or a choice from a list shown to it. **No number
the LLM writes reaches a bid**: the profile selects a fixed weight vector, a
priority selects one of three fixed multipliers, a preference adds one fixed
bonus. The deterministic scorer computes every bid.'s
``commitment_horizon_s`` float is dropped: nothing in the frozen
scorer could consume it.

**Validation drops, it never repairs.** An unknown profile falls back to the
run's default profile; a priority for a contact the agent does not believe in,
an invalid label, a duplicate, a task not in its list, or preferences beyond
the cap are each dropped -- and every drop is a recorded fallback event. The
rest of the answer is kept. Syntax is normalised (``"high"`` means HIGH, as the LLM planner
accepts ``"RECON"``); semantics never are. **A total failure** (no answer,
unparseable answer, transport failure) yields ``None`` for that agent, which
the runtime replaces with the default modifiers -- exactly rule-based CBAA --
and records.

**Frozen epoch snapshots, as in the LLM planner.** The runtime builds every agent's view and
task list before any call; each call receives only its own agent's pair; the
prompt is a pure function of ``(view, tasks, PromptContext)``. So per-agent
calls within one epoch **may run concurrently** (``max_workers``): no call can
observe another's answer, and the result does not depend on completion order.
Live episodes are slow (~10 min per 400 steps sequentially), which is why.

**The task set**: under ``"three_task"`` SCREEN is not a
task, so the prompt neither lists it among the task kinds nor mentions
screening in a profile description, validation drops ``SCREEN:c`` as an
unlisted task like any other, and the prompt version in the cache key is
``"<version>+three_task"``. ``"four_task"``, the default, sends exactly the
historical prompts under the historical key.

Calls go through ``LocalLLMPlanner``'s path unchanged: cached by content with
the endpoint in the key, provenance recorded, transport failures recorded and
replayed, ``replay_only`` refusing the network.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from coordination.bidding import PRIORITY_MULTIPLIER, PROFILES, BidModifiers
from coordination.tasks import DEFAULT_TASK_SET, TASK_SETS, check_task_set, screen_available
from planning.local_llm import (
    LocalLLMPlanner,
    PerAgentLLMPlanner,
    ReplayMiss,
    TransportFailure,
)
from planning.prompt_builder import PromptContext, _vec, build_user_message
from planning.stops import StopRun

BRIDGE_PROMPT_VERSION = "m8b.7"  # the label in every cache key; do not change
MAX_PREFERRED = 3

# ``reasoning`` is the first property and the first required field: under a strict
# json_schema grammar a model that is not asked for it first tends to omit it and
# pad whitespace to the token cap.
_MODIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "scoring_profile": {"type": "string", "enum": sorted(PROFILES)},
        "threat_priorities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "contact_id": {"type": "integer"},
                    "priority": {"type": "string", "enum": sorted(PRIORITY_MULTIPLIER)},
                },
                "required": ["contact_id", "priority"],
                "additionalProperties": False,
            },
        },
        "preferred_tasks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reasoning", "scoring_profile", "threat_priorities", "preferred_tasks"],
    "additionalProperties": False,
}
# How a decentralized arm turns bids into tasks; the prompt states the one in use.
SELECTIONS = ("cbaa", "independent")
# The bid version the prompt states.
BID_VERSION = "v2.3"


def prompt_version(task_set=DEFAULT_TASK_SET) -> str:
    """The version string that enters the cache key and the provenance:
    tagged with the task set unless it is the default, so a cache cannot
    answer a prompt from another task model."""
    check_task_set(task_set)
    return BRIDGE_PROMPT_VERSION if task_set == DEFAULT_TASK_SET else f"{BRIDGE_PROMPT_VERSION}+{task_set}"


def _task_kinds(ctx, task_set) -> str:
    screen = (" SCREEN:c holds the perimeter between contact c and the defended centre."
              if screen_available(task_set) else "")
    return (f"Task kinds: NEUTRALIZE:c:s closes to contact c (you are permanently disabled on "
            f"contact; the kill is worth {ctx.neutralize_points} points).{screen} DEFEND:k holds "
            f"perimeter station k. RECON:r patrols toward coverage region r.")


class _Priority(BaseModel):
    model_config = ConfigDict(extra="ignore")
    contact_id: int
    priority: str

    @field_validator("priority", mode="before")
    @classmethod
    def _label(cls, value):
        # Syntax only: case and whitespace. A number is not a label and stays
        # invalid, so it is dropped downstream rather than coerced.
        return value.strip().upper() if isinstance(value, str) else repr(value)


class _ModifierAnswer(BaseModel):
    """Lenient on shape so each element can be judged on its own: one bad
    priority must not throw away a good profile choice."""

    model_config = ConfigDict(extra="allow")
    scoring_profile: object = None
    threat_priorities: List[_Priority] = []
    preferred_tasks: List[object] = []
    reasoning: str = ""


def _base_system_prompt(ctx: PromptContext, profiles: str, kinds: str) -> str:
    return f"""You are ONE defending agent (BLUE) in a 2D pursuit-evasion simulation. Blue defends a region against RED intruders. Your team allocates tasks with an auction: each agent bids on its own candidate tasks, and conflicts are resolved automatically after you answer. You do NOT choose your task and you do NOT write any bid. You choose how YOUR bids are weighted.

Answer with four things, in this order, and nothing numeric beyond contact ids:

1. reasoning -- FIRST: one sentence about the actual picture (distances, ages, what you are trading).

2. scoring_profile -- exactly one of:
{profiles}

3. threat_priorities -- for contacts YOU believe in (listed below), a priority of LOW, MEDIUM or HIGH. MEDIUM is the default for any contact you leave out. You may only name contact_id values that are listed below.

4. preferred_tasks -- at most {MAX_PREFERRED} task ids, each copied exactly as one of the quoted task_id strings in YOUR candidate task list below (the text inside the quotes only, without the location). Each receives a fixed bonus. Leave it empty if nothing stands out.

{kinds}

What you know is what YOU believe, and it is incomplete. Contacts you cannot see are not listed, and a short list does NOT mean the region is safe. Other agents are answering at the same moment on their own beliefs.

Respond only with a valid JSON object matching the required schema, with "reasoning" as its first field."""


def build_bridge_system_prompt(ctx: PromptContext, task_set: str = DEFAULT_TASK_SET,
                               selection: str = "cbaa") -> str:
    """Depends on the context, task set and the arm's selection rule:
    identical for every agent and epoch."""
    from coordination.mission_policy import blue_brief_v1
    from planning.prompt_builder import SENSOR_GEOMETRY_TEXT

    check_task_set(task_set)
    if selection not in SELECTIONS:
        raise ValueError(f"unknown selection {selection!r}; expected one of {SELECTIONS}")
    profiles = "\n".join(f"- {name}: {_profile_effect_v2(name)}" for name in sorted(PROFILES))
    base = _base_system_prompt(ctx, profiles, _task_kinds(ctx, task_set))
    assert _AUCTION_SENTENCE in base
    base = base.replace(_AUCTION_SENTENCE, _SELECTION_SENTENCE[selection], 1)
    anchor = "\n\nRespond only with a valid JSON object"
    mechanics = _bid_mechanics_v22(selection, conditional_continuity=True)
    if ctx.horizon_steps is None or ctx.dt is None:
        raise ValueError("the prompt states the episode length; build the PromptContext with "
                         "PromptContext.from_simulation")
    assert base.count(_KILL_POINTS.format(ctx.neutralize_points)) == 1
    base = base.replace(_KILL_POINTS.format(ctx.neutralize_points), "", 1)
    assert mechanics.count(_TERMS_SHOWN) == 1
    mechanics = mechanics.replace(_TERMS_SHOWN, _TERMS_SHOWN + _ALLOCATION_VALUES, 1)
    extra = mechanics + "\n\n" + blue_brief_v1(ctx.horizon_steps, ctx.dt) + "\n\n" + SENSOR_GEOMETRY_TEXT
    return base.replace(anchor, "\n\n" + extra + anchor, 1)


def _profile_effect_v2(name) -> str:
    from coordination.bidding import V2_PROFILES

    W, B = V2_PROFILES[name], V2_PROFILES["balanced"]
    if name == "balanced":
        return (f"the reference weights (coverage {B.w_cov}, leaving {B.w_leave}, information "
                f"{B.w_info}, attrition {B.w_att}, urgency time scale {B.t0:g} s)")
    diffs = []
    for key, label in (("w_cov", "coverage"), ("w_leave", "leaving"), ("w_info", "information"),
                       ("w_att", "attrition"), ("t0", "urgency time scale")):
        a, b = getattr(W, key), getattr(B, key)
        if a != b:
            diffs.append(f"{label} {b:g} -> {a:g}{' s' if key == 't0' else ''}")
    return "as balanced except " + ", ".join(diffs)


_AUCTION_SENTENCE = ("Your team allocates tasks with an auction: each agent bids on its own "
                     "candidate tasks, and conflicts are resolved automatically after you answer. "
                     "You do NOT choose your task and you do NOT write any bid. You choose how "
                     "YOUR bids are weighted.")
_SELECTION_SENTENCE = {
    "cbaa": ("Your team allocates tasks with an auction: each blue bids on its own candidate "
             "tasks; the highest bid for a task wins it, and a blue that is outbid takes its "
             "best remaining task. You do NOT choose your task and you do NOT write any bid. "
             "You choose how YOUR bids are weighted."),
    "independent": ("Each blue computes bids for its own candidate tasks and takes its own "
                    "highest-bid task. There is no coordination between blues: teammates may "
                    "take the same task. You do NOT choose your task and you do NOT write any "
                    "bid. You choose how YOUR bids are weighted."),
}


_KILL_POINTS = "; the kill is worth {} points"
_TERMS_SHOWN = "Your candidate task list shows each task's terms under the balanced profile."
_ALLOCATION_VALUES = (" Every number below is an allocation value that only ranks tasks for "
                      "allocation; none of them is the mission's objective.")


def _bid_mechanics_v22(selection, conditional_continuity=False) -> str:
    """Generated from the same parameter objects as the bids (m8b.5; m8b.6
    with ``conditional_continuity``, bid version v2.3)."""
    from coordination import coverage_value as cv
    from coordination import bidding as bd
    from coordination.bidding import PREFERRED_BONUS, PRIORITY_MULTIPLIER, V22_PROFILES

    B = V22_PROFILES["balanced"]
    lo, hi = cv.PRIOR_SECTOR_DEG
    win = ("the highest bid is your task unless the auction gives it to a teammate who bids "
           "more for it" if selection == "cbaa" else "your highest bid is your task")
    return (
        f"How your answer changes the bids ({win}). Bid = value - travel - leaving - switching "
        "- continuity (NEUTRALIZE also minus attrition and belief-age uncertainty), plus "
        f"{PREFERRED_BONUS} for a preferred task. Your candidate task list shows each task's "
        "terms under the balanced profile.\n"
        f"- Approach sector: intruders are expected from bearings {lo:g} to {hi:g} degrees as "
        "seen from the defended centre (any distance). Only ground in that sector counts below.\n"
        f"- DEFEND:k value = {B.station_base} + coverage weight x coverage, where coverage is how "
        "much approach-sector ground station k's sensor wedge (at the station's fixed heading) "
        "adds beyond what your teammates' sensors watch now (1.0 = one full sensor's worth).\n"
        "- RECON:r value = information weight x information: approach-sector ground your sensor "
        "sweeps on the way to region r and there, counted by how stale each region is in YOUR "
        "coverage record and by how soon you reach it (an optimistic travel forecast from your "
        "current velocity); ground a blue already watches counts 0. "
        f"Travel {B.w_eta_recon} per 25 units.\n"
        f"- NEUTRALIZE:c value = 1.0 x (threat value {B.threat_value:g} + priority x urgency x "
        "opportunity). The threat value is the same for every believed red, whatever its "
        "heading, priority or reachability. Urgency = the larger of 1/(1 + forecast time to "
        f"breach / {B.t0:g} s) and 1/(1 + distance to the region / 25). Forecast time to breach: "
        "from the contact's believed velocity, the time to close on the approach funnel (the "
        "region together with the four public approach gates) at its current closing speed, "
        "plus the last leg into the region; none when it is not closing (moving away, crossing "
        "or stationary). Opportunity comes from an OPTIMISTIC kinematic interception forecast "
        "(from your current velocity, with the per-axis speed and acceleration limits of 2; it "
        "ignores collision avoidance and the contact's reaction): 1 if you can reach the "
        "contact's forecast path before its forecast breach and within the episode, falling "
        f"linearly to {B.o_min:g} over {B.t1:g} s later than that, {B.o_min:g} with no "
        f"interception; averaged over +-{cv.VELOCITY_UNCERTAINTY:g} u/s of uncertainty in the "
        f"contact's velocity. Travel {B.w_eta} per 25 units of current distance; attrition = "
        "attrition weight / (1 + teammates), because contact disables you.\n"
        "- Leaving your current station costs leaving weight x that station's coverage. Changing "
        f"task costs {B.switch_cost}. While you pursue a contact you still believe in, every other "
        f"task (another contact, RECON, DEFEND) costs an extra {B.pursuit_continuity:g} "
        "(continuity)."
        + ((" For ANOTHER CONTACT only, that extra cost is multiplied by (1 - R x H x S), "
            "down to 0: R = how fast the contact you pursue moves away from the approach funnel "
            f"(0 if it is not moving away, 1 at {bd.V23_RECEDE_SPEED:g} u/s or faster); H = 1 "
            f"if the other contact's forecast breach comes at least {bd.V23_HORIZON_RAMP_S:g} s "
            "before the episode ends, falling to 0 at the end (0 with no forecast breach); "
            "S = how far your optimistic interception of the other contact comes before its "
            f"forecast breach (0 for no lead, 1 for {bd.V23_MARGIN_RAMP_S:g} s or more), "
            "averaged over the same velocity uncertainty. RECON and DEFEND always pay the full "
            "extra cost. Each task line shows the continuity actually applied.")
           if conditional_continuity else "")
        + "\n"
        "- Priorities multiply only urgency x opportunity, never the threat value: "
        + ", ".join(f"{k} {v:g}" for k, v in sorted(PRIORITY_MULTIPLIER.items(), key=lambda x: x[1]))
        + ".")


_STATUS_TEXT = {"approaching": "approaching", "moving_away": "moving away",
                "crossing": "crossing", "stationary": "stationary", "inside": "inside the region"}


def _fmt_terms_v22(t, remaining_s) -> str:
    import math

    if "threat_value" not in t:
        return _fmt_terms(t, extra=("continuity",))
    tau, ti = t["time_to_breach"], t["intercept_time"]
    if not math.isfinite(ti):
        ic = "none"
    elif ti > remaining_s:
        ic = f"{ti:.1f} s (after the episode ends)"
    else:
        ic = f"{ti:.1f} s"
    head = (f"value {t['value']:.2f} (threat {t['threat_value']:.2f} + urgency {t['urgency']:.2f} "
            f"x opportunity {t['opportunity']:.2f}; {_STATUS_TEXT.get(t['breach_category'], t['breach_category'])}, "
            f"forecast breach {'none' if not math.isfinite(tau) else f'{tau:.1f} s'}, "
            f"interception forecast {ic})")
    parts = [head]
    for k in ("travel", "leave", "switch", "continuity", "attrition", "uncertainty"):
        if t.get(k):
            parts.append(f"{'leaving' if k == 'leave' else k} {t[k]:.2f}")
    return ", ".join(parts) + f" -> bid {t['bid']:.2f}"


def _fmt_terms(t, extra=()) -> str:
    import math

    parts = [f"value {t['value']:.2f}"]
    if "coverage" in t:
        parts[0] += f" (coverage {t['coverage']:.2f})"
    if "information" in t:
        parts[0] += f" (information {t['information']:.2f})"
    if "time_to_breach" in t:
        tau = t["time_to_breach"]
        ti = t["intercept_time"]
        parts[0] += (f" (time to breach {'none' if not math.isfinite(tau) else f'{tau:.1f} s'}, "
                     f"urgency {t['urgency']:.2f}, intercept "
                     f"{'none' if not math.isfinite(ti) else f'{ti:.1f} s'}, "
                     f"opportunity {t['opportunity']:.2f})")
    for k in ("travel", "leave", "switch") + tuple(extra) + ("attrition", "uncertainty"):
        if t.get(k):
            parts.append(f"{'leaving' if k == 'leave' else k} {t[k]:.2f}")
    return ", ".join(parts) + f" -> bid {t['bid']:.2f}"


def build_bridge_user_message(view, tasks: Sequence, ctx: PromptContext, bid_terms,
                              current_task, remaining_s, max_age=None) -> str:
    """The agent's situation plus its candidate tasks, each with its bid terms
    under the balanced profile.

    A pure function of its arguments. Task locations are derived from the
    view and static geometry by ``generate_tasks``; nothing else enters. Each
    id is printed quoted and apart from its location.
    """
    if bid_terms is None or remaining_s is None:
        raise ValueError("the prompt shows each task's bid terms; the runtime must supply them "
                         "(cbaa_bid_version='v2.3') and the remaining episode time")
    lines = [build_user_message(view, ctx, max_age=max_age, with_headings=True),
             f"your current task: {current_task or 'none (holding your own station)'}\n"
             f"episode time left: {remaining_s:.1f} s\n"
             "your candidate tasks (task_id, where it is, terms under the balanced profile):"]
    for task in tasks:
        lines.append(f'  task_id "{task.task_id}"    at {_vec(task.location)}    '
                     f'{_fmt_terms_v22(bid_terms[task.task_id], remaining_s)}')
    return "\n".join(lines)


def defender_validation_context(view, tasks: Sequence, default_profile: str, max_age=None) -> Dict:
    """Everything ``LocalModifierPlanner._validate`` reads, as plain data:
    the believed contacts (the same ``fresh_targets(max_age)`` call), the
    candidate task ids and the run's default profile."""
    return {"kind": "defender", "agent_id": int(view.agent_id), "now": int(view.now),
            "believed_contacts": sorted(int(c) for c in view.fresh_targets(max_age)),
            "task_ids": [t.task_id for t in tasks], "default_profile": default_profile}


class LocalModifierPlanner(LocalLLMPlanner):
    """One blue's bridge. Reuses LocalLLMPlanner's cached call path entirely."""

    _SCHEMA_NAME = "bid_modifiers"
    _SCHEMA = _MODIFIER_SCHEMA
    _PROMPT_VERSION = BRIDGE_PROMPT_VERSION
    bridge_version = BRIDGE_PROMPT_VERSION

    def __init__(self, *args, task_set: str = DEFAULT_TASK_SET, selection: str = "cbaa", **kwargs):
        super().__init__(*args, **kwargs)
        self.last_prompt: Optional[Tuple[str, str]] = None
        # Instance attribute shadows the class one that LocalLLMPlanner's
        # cached call path reads: the version in the key.
        self.task_set = check_task_set(task_set)
        if selection not in SELECTIONS:
            raise ValueError(f"unknown selection {selection!r}; expected one of {SELECTIONS}")
        self.selection = selection
        self._PROMPT_VERSION = prompt_version(self.task_set)

    def plan_one(self, view, max_age=None):
        raise TypeError("a bid-modifier bridge chooses modifiers, not roles; use modifiers_for")

    def _event(self, view, kind, **fields):
        event = {"sim_step": int(view.now), "agent_id": int(view.agent_id), "kind": kind}
        event.update(fields)
        self.fallback_events.append(event)

    def modifiers_for(self, view, tasks: Sequence, default_profile: str,
                      max_age=None, bid_terms=None, current_task=None,
                      remaining_s=None) -> Optional[BidModifiers]:
        """This agent's modifiers, or None on total failure (the runtime then
        applies the default modifiers, i.e. rule-based CBAA)."""
        stray = sorted({t.kind for t in tasks} - TASK_SETS[self.task_set])
        if stray:
            # The runtime and this bridge disagree about the task model, which
            # Simulation refuses at construction; never prompt across it.
            raise ValueError(f"agent {view.agent_id}: tasks of kind {stray} are outside "
                             f"this bridge's task set {self.task_set!r}")
        system_prompt = build_bridge_system_prompt(self.context, self.task_set, self.selection)
        user_message = build_bridge_user_message(view, tasks, self.context, bid_terms,
                                                 current_task, remaining_s, max_age=max_age)
        system_prompt, images = self._image_request(system_prompt, view.fresh_targets(max_age))
        self.last_prompt = (system_prompt, user_message)
        context = defender_validation_context(view, tasks, default_profile, max_age)
        try:
            content = self._fetch(system_prompt, user_message, validation_context=context,
                                  **({"images": images} if images else {}))
        except StopRun:
            raise
        except TransportFailure as exc:
            self._event(view, "call_failure", error_type=exc.error_type,
                        applied="default_modifiers")
            return None
        except Exception as exc:  # noqa: BLE001 -- e.g. no client and no entry
            self._event(view, "call_failure", error_type=type(exc).__name__,
                        applied="default_modifiers")
            return None
        return self.interpret(content, view, tasks, default_profile, max_age)

    def interpret(self, content, view, tasks: Sequence, default_profile: str,
                  max_age=None) -> Optional[BidModifiers]:
        """Parse and validate one response: the modifiers applied, or None on
        total failure. Every drop is an event. The live path and the offline
        semantic validator (planning/semantic_validation.py) both call this."""
        if not (content or "").strip():
            self._event(view, "empty_response", applied="default_modifiers")
            return None
        try:
            answer = _ModifierAnswer.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError) as exc:
            self._event(view, "schema_failure", error_type=type(exc).__name__,
                        applied="default_modifiers")
            return None
        self.last_reasoning = answer.reasoning
        return self._validate(view, tasks, answer, default_profile, max_age)

    def _validate(self, view, tasks, answer, default_profile, max_age) -> BidModifiers:
        profile = answer.scoring_profile
        profile = profile.strip().lower() if isinstance(profile, str) else profile
        if profile not in PROFILES:
            self._event(view, "unknown_profile", value=repr(answer.scoring_profile),
                        applied=default_profile)
            profile = default_profile

        # Validated against THIS agent's belief -- the same fresh_targets call,
        # with the same staleness limit, as the task generator and the prompt.
        believed = set(int(c) for c in view.fresh_targets(max_age))
        priorities: Dict[int, str] = {}
        for entry in answer.threat_priorities:
            cid = int(entry.contact_id)
            if cid not in believed:
                self._event(view, "unbelieved_contact", value=cid)
            elif entry.priority not in PRIORITY_MULTIPLIER:
                self._event(view, "invalid_priority", value=entry.priority, contact_id=cid)
            elif cid in priorities:
                self._event(view, "duplicate_priority", value=cid)
            else:
                priorities[cid] = entry.priority

        listed = [t.task_id for t in tasks]
        preferred: List[str] = []
        for task_id in answer.preferred_tasks:
            if not isinstance(task_id, str) or task_id not in listed:
                self._event(view, "unlisted_task", value=repr(task_id))
            elif task_id in preferred:
                continue
            elif len(preferred) >= MAX_PREFERRED:
                self._event(view, "too_many_preferred", value=task_id)
            else:
                preferred.append(task_id)

        extra = sorted((answer.model_extra or {}).keys())
        if extra:
            # e.g.'s commitment_horizon_s. Ignored, never read.
            self._event(view, "ignored_fields", value=extra)

        return BidModifiers.build(
            scoring_profile=profile,
            threat_priorities=priorities,
            preferred_tasks=preferred,
        )


class LLMModifierSource(PerAgentLLMPlanner):
    """What ``CBAARuntime`` calls once per epoch. One ``LocalModifierPlanner``
    per blue; no shared decision state; diagnostics aggregated after the calls.

    ``max_workers > 1`` issues one epoch's per-agent calls concurrently. Safe
    because every view and task list is frozen before the first call, each
    planner object is used by exactly one call per epoch, and the cache is
    thread-safe. Results are assembled by agent id, never by completion order.
    """

    _LOCAL_CLASS = LocalModifierPlanner

    bridge_version = BRIDGE_PROMPT_VERSION

    def __init__(self, context: PromptContext, *args, max_workers: int = 1,
                 task_set: str = DEFAULT_TASK_SET, selection: str = "cbaa", **kwargs):
        super().__init__(context, *args, **kwargs)
        self.max_workers = max(1, int(max_workers))
        self.bridge_version = type(self).bridge_version  # an instance attribute, so the recorded configuration names it
        # The task set must match the Simulation's (checked there).
        self.task_set = check_task_set(task_set)
        # The arm's selection rule, stated in the prompt ("cbaa" or "independent").
        if selection not in SELECTIONS:
            raise ValueError(f"unknown selection {selection!r}; expected one of {SELECTIONS}")
        self.selection = selection
        # (step, agent) -> the modifiers applied, or None for "default after
        # total failure". The audit trail of what the LLM actually changed.
        self.modifier_log: List[Dict] = []

    def _local_kwargs(self):
        return {"task_set": self.task_set, "selection": self.selection}

    def plan_one(self, view, max_age=None):
        raise TypeError("LLMModifierSource is a bid-modifier bridge, not a role planner")

    def modifiers_for_epoch(self, step: int, requests: Dict[int, Tuple[object, Sequence]],
                            default_profile: str, max_age=None, bid_terms=None,
                            current_tasks=None, remaining_s=None) -> Dict[int, Optional[BidModifiers]]:
        agents = sorted(requests)
        planners = {i: self.planner_for(i) for i in agents}  # created serially

        def call(i):
            view, tasks = requests[i]
            if bid_terms is None:
                return planners[i].modifiers_for(view, tasks, default_profile, max_age=max_age)
            return planners[i].modifiers_for(view, tasks, default_profile, max_age=max_age,
                                             bid_terms=bid_terms[i],
                                             current_task=(current_tasks or {}).get(i),
                                             remaining_s=remaining_s)

        if self.max_workers == 1 or len(agents) <= 1:
            results = {i: call(i) for i in agents}
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(agents))) as pool:
                futures = {i: pool.submit(call, i) for i in agents}
                results = {i: futures[i].result() for i in agents}  # re-raises StopRun

        for i in agents:
            mods = results[i]
            self.modifier_log.append({
                "step": int(step),
                "agent_id": int(i),
                "modifiers": None if mods is None else {
                    "scoring_profile": mods.scoring_profile,
                    "threat_priorities": [list(p) for p in mods.threat_priorities],
                    "preferred_tasks": list(mods.preferred_tasks),
                },
            })
        return results

    def diagnostics(self) -> Dict:
        d = super().diagnostics()
        d["bridge"] = True
        d["prompt_builder_version"] = prompt_version(self.task_set)
        if self.task_set != DEFAULT_TASK_SET:
            d["task_set"] = self.task_set
        d["max_workers"] = self.max_workers
        d["epochs_x_agents"] = len(self.modifier_log)
        d["defaulted_after_total_failure"] = sum(
            1 for e in self.modifier_log if e["modifiers"] is None
        )
        return d
