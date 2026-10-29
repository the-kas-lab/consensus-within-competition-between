"""LLM decider for the task-model commander (coordination/commander.py).

One call per planning epoch, on the commander's own catalogue (built by CBAA's
task generator from the commander's own network-delivered belief view). The
schema restricts task ids to that catalogue and agent ids to the active
agents; it does NOT forbid duplicates or missing agents -- those are the
validator's job (per-agent DEFEND(own station) fallback, capacity one per
task id, lowest agent id keeps), so every repair is logged rather than hidden
in decoding. Any call or schema failure raises, which the commander records
as an unusable response: fallback for every active agent.
"""

import json
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel

from coordination.tasks import DEFEND, NEUTRALIZE, RECON
from planning.local_llm import LocalLLMPlanner

COMMANDER_PROMPT_VERSION = "commander.10"  # the label in every cache key; do not change


def _system_prompt(horizon_steps, dt):
    """The commander's system prompt: the game and task description, the sensor
    geometry, the approach sector, how the forecasts are computed, the
    objective-only mission brief (which states the episode length
    ``horizon_steps`` * ``dt``) and the validity rule."""
    from coordination import coverage_value as cv
    from coordination.bidding import V22_PROFILES
    from coordination.mission_policy import blue_brief_v1
    from planning.prompt_builder import SENSOR_GEOMETRY_TEXT

    B = V22_PROFILES["balanced"]
    lo, hi = cv.PRIOR_SECTOR_DEG
    game = (
        "You are the commander of the BLUE defender team in a 2D perimeter-defense game. "
        "Red intruders try to breach the defended region; a blue that touches a red "
        "destroys it and is itself disabled.\n"
        "Each planning step you assign every active blue exactly ONE task from the task "
        "list. Task kinds:\n"
        "- NEUTRALIZE:<contact>:<slot> -- pursue and intercept that believed red contact. "
        "Each slot can go to one blue only.\n"
        "- DEFEND:<station> -- hold that defensive station on the perimeter. One blue per "
        "station.\n"
        "- RECON:<cell> -- sweep that arena cell to find reds. One blue per cell.\n"
        "Every task id may be assigned to at most one blue. Your picture of the reds comes "
        "from reports your team delivered to you; reds you have not heard about may exist. ")
    sector = (
        f"Approach sector and sensing: intruders are expected from bearings {lo:g} to {hi:g} "
        "degrees as seen from the defended centre (any distance). A blue holding a station looks "
        "along that station's fixed heading; the approach-sector ground each station's sensor "
        "wedge covers is listed per station. A blue on RECON flies toward its cell with its "
        "sensor pointed ahead and holds there. Ground is watched only while some blue's sensor "
        "wedge covers it: a scout adds coverage only where no other blue is watching at that "
        "time, and a station's ground stays watched after its blue leaves only if another blue's "
        "sensor covers it.\n")
    forecasts = (
        "Forecasts shown with the reports (from your picture, labelled as forecasts): each "
        "contact's status (approaching / moving away / crossing / stationary, from its reported "
        "velocity relative to the approach funnel = the region together with the four public "
        "approach gates), its forecast time to breach (closing time to the funnel plus the last "
        "leg; none when not closing) and urgency = the larger of 1/(1 + forecast time / "
        f"{B.t0:g} s) and 1/(1 + distance to the region / 25); for each blue and contact an "
        "OPTIMISTIC kinematic interception forecast: it assumes the contact keeps its reported "
        "velocity and uses only the blue's speed and acceleration limits of 2 per axis from its "
        "current velocity (no collision avoidance). A contact that reacts can make the real "
        "interception earlier, later or impossible. Every forecast and coverage figure is "
        "computed from your picture, which can be stale or incomplete.\n"
        "Each blue's line shows the LAST ASSIGNMENT YOU ISSUED to it and the step you issued "
        "it. Issuing is not delivery: with delayed or lost communication the blue may not have "
        "received or started it. \"no assignment issued yet\" only means you have not assigned "
        "that blue anything; it does not tell you what the blue is doing.\n")
    valid = (
        "Valid answers:\n"
        "- Give every active blue exactly one task, and give each full task id to at most one "
        "blue. Different slots of one contact (NEUTRALIZE:c:0, NEUTRALIZE:c:1) are different "
        "tasks; the same task id given to two blues is invalid and the second blue is sent back "
        "to its station.\n")
    answer = ("Answer with JSON matching the schema: task_by_agent maps every active blue's id "
              "to its one task id. Give your reasoning first.")
    return (game + SENSOR_GEOMETRY_TEXT + "\n" + sector + forecasts
            + blue_brief_v1(horizon_steps, dt) + "\n" + valid + answer)


def commander_forecasts(view, positions, forecast_ctx):
    """The v2.2 forecast terms for the commander, from its own view: per
    contact (status, forecast breach, urgency) and per (blue, contact) the
    optimistic interception forecast -- the functions the v2.2 bids use."""
    from coordination.bidding import V22_PROFILES, _neutralize_v22

    W = V22_PROFILES["balanced"]
    v2 = forecast_ctx.v2
    remaining_s = max(0.0, (v2.horizon_steps - int(view.now)) * v2.dt)
    states = {int(view.agent_id): np.asarray(view.self_state, dtype=float).ravel()}
    for j, st in (view.teammates or {}).items():
        states[int(j)] = np.asarray(st, dtype=float).ravel()
    out = {}
    for c, b in sorted(view.fresh_targets(None).items()):
        p = b.predicted_position(view.now, view.dt)
        v = np.asarray(b.velocity, dtype=float)
        per = {a: _neutralize_v22(p, v, states[a][0:2], states[a][2:4], forecast_ctx, W,
                                  remaining_s, 1.0) for a in sorted(positions) if a in states}
        any_t = next(iter(per.values()), None)
        out[int(c)] = {"per_blue": per, "remaining_s": remaining_s,
                       "status": None if any_t is None else any_t["breach_category"],
                       "time_to_breach": None if any_t is None else any_t["time_to_breach"],
                       "urgency": None if any_t is None else any_t["urgency"]}
    return out


def commander_geometry(sim, task_ctx):
    """Static geometry commander.4 shows: each station's position, fixed
    heading and approach-sector coverage, and each coverage region's
    approach-sector share. From the episode's construction, never its state."""
    from coordination import coverage_value as cv

    f = cv.approach_field(sim.defense_center, task_ctx.defense_polygon, sim.arena_min,
                          sim.arena_max, sim.fov_half_angle_rad, sim.fov_range)
    heads = sim.station_heading_angles(task_ctx.stations)
    poses = {int(k): (np.asarray(p, float), heads[int(k)]) for k, p in task_ctx.stations}
    stations = {k: {"heading": float(h),
                    "coverage": f.mass(f.wedge(p, h)),
                    "unique": f.marginal((p, h), f.union(q for j, q in poses.items() if j != k))}
                for k, (p, h) in poses.items()}
    centers = np.array([[(i + .5) * sim.arena_size / 4, (j + .5) * sim.arena_size / 4]
                        for i in range(4) for j in range(4)])
    idx = f.cell_index(centers)
    cell_area = (sim.arena_size / 4) ** 2
    share = {r: float(np.count_nonzero(idx == r)) * f.cell_area / cell_area for r in range(16)}
    geo = {"stations": stations, "region_sector_share": share}
    if getattr(sim, "cbaa_bid_version", "v1") in ("v2.2", "v2.3"):
        from coordination.bidding import BidContext

        geo["forecast_ctx"] = BidContext.from_simulation(sim, task_ctx)
    return geo


class _Answer(BaseModel):
    reasoning: str
    task_by_agent: Dict[str, str]


def _schema(task_ids, agents):
    """The answer is an object keyed by agent id with every active agent
    required, so the schema itself admits exactly one task per active agent.
    Task-id capacity (duplicates across agents) is the validator's job."""
    ids = [str(a) for a in sorted(int(a) for a in agents)]
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "task_by_agent": {
                "type": "object",
                "properties": {a: {"type": "string", "enum": list(task_ids)} for a in ids},
                "required": ids,
                "additionalProperties": False,
            },
        },
        "required": ["reasoning", "task_by_agent"],
        "additionalProperties": False,
    }


def _blue_headings(view):
    """{agent: heading} from the commander's own view: its own state and the
    declared teammate channel (the same states its positions come from)."""
    out = {}
    own = np.asarray(view.self_state, dtype=float).ravel()
    if own.size >= 5:
        out[int(view.agent_id)] = float(own[4])
    for j, st in (view.teammates or {}).items():
        st = np.asarray(st, dtype=float).ravel()
        if st.size >= 5:
            out[int(j)] = float(st[4])
    return out


def _fmt_s(x, remaining_s=None):
    import math

    if x is None or not math.isfinite(x):
        return "none"
    if remaining_s is not None and x > remaining_s:
        return f"{x:.1f} s (after the episode ends)"
    return f"{x:.1f} s"


def user_message(view, catalogue, positions, geometry, forecasts, history) -> str:
    from planning.prompt_builder import heading_deg

    headings = _blue_headings(view)
    lines = [f"time_step: {view.now} (dt={view.dt} s)", "blues (active):"]
    if forecasts:
        rem = next(iter(forecasts.values()))["remaining_s"]
        lines[0] += f", episode time left {rem:.1f} s"
    for a in sorted(positions):
        p = positions[a]
        h = f" heading_deg={heading_deg(headings[a]):.0f}" if a in headings else ""
        step, issued = history if history else (None, {})
        cur = (f" last issued assignment {issued[a]} (issued at step {step})"
               if a in issued else " no assignment issued yet")
        lines.append(f"  blue {a} pos=[{p[0]:.1f}, {p[1]:.1f}]{h}{cur}")
    believed = view.fresh_targets(None)
    lines.append("red_contacts (as reported to you):" + ("" if believed else " none"))
    for c in sorted(believed):
        b = believed[c]
        pp = b.predicted_position(view.now, view.dt)
        st = np.asarray(b.state, dtype=float).ravel()
        h = f" observed_heading_deg={heading_deg(st[4]):.0f}" if st.size >= 5 else ""
        lines.append(f"  contact {c} predicted pos=[{pp[0]:.1f}, {pp[1]:.1f}] "
                     f"vel=[{b.velocity[0]:.2f}, {b.velocity[1]:.2f}]{h} "
                     f"seen {b.age(view.now)} steps ago")
        if int(c) in forecasts:
            fc = forecasts[int(c)]
            status = (fc["status"] or "").replace("_", " ")
            lines.append(f"    forecast: {status}, breach {_fmt_s(fc['time_to_breach'])}, "
                         f"urgency {fc['urgency']:.2f}; optimistic interception: "
                         + ", ".join(f"blue {a} {_fmt_s(t['intercept_time'], fc['remaining_s'])}"
                                     for a, t in sorted(fc["per_blue"].items())))
    lines.append("tasks:")
    for t in catalogue:
        extra = ""
        if t.kind == DEFEND:
            st = geometry["stations"][int(t.task_id.split(":")[1])]
            extra = (f" sensor heading_deg={heading_deg(st['heading']):.0f} approach coverage "
                     f"{st['coverage']:.2f} ({st['unique']:.2f} not covered by any other station)")
        lines.append(f"  {t.task_id} at [{t.location[0]:.1f}, {t.location[1]:.1f}]{extra}")
    lines.append("coverage (YOUR record: steps since each region was last seen, -1 = never; "
                 "share = fraction of the region in the approach sector):")
    for r in view.coverage_regions() or []:
        rid = int(r["region_id"])
        lines.append(f"  region {rid} center=[{r['center'][0]:.1f}, {r['center'][1]:.1f}] "
                     f"steps_since_seen={int(r['steps_since_seen'])} "
                     f"share={geometry['region_sector_share'][rid]:.2f}")
    return "\n".join(lines)


class LLMCommanderDecider(LocalLLMPlanner):
    _SCHEMA_NAME = "commander_assignment"
    _PROMPT_VERSION = COMMANDER_PROMPT_VERSION

    def __init__(self, commander_id: int = 0, **local_kwargs):
        super().__init__(commander_id, context=None, **local_kwargs)

    def observe_history(self, history):
        """(step, {agent: task}) the commander last issued, or None; set by
        CommanderTaskPlanner before each decision."""
        self.history = history

    def decide(self, view, catalogue, positions):
        allowed = {NEUTRALIZE, DEFEND, RECON}
        stray = sorted({t.kind for t in catalogue} - allowed)
        if stray:
            raise ValueError(f"catalogue holds {stray}, outside the three-task model")
        geo = getattr(self, "geometry", None)
        if geo is None:
            raise ValueError("the commander prompt needs the episode's static geometry "
                             "(commander_geometry)")
        if "forecast_ctx" not in geo:
            raise ValueError("the commander prompt states the forecasts; the run must use "
                             "cbaa_bid_version='v2.3'")
        self._SCHEMA = _schema([t.task_id for t in catalogue], positions)
        # The brief's episode length: the run's own horizon and dt.
        v2c = geo["forecast_ctx"].v2
        system = _system_prompt(v2c.horizon_steps, v2c.dt)
        um = user_message(view, catalogue, positions, geo,
                          commander_forecasts(view, positions, geo["forecast_ctx"]),
                          getattr(self, "history", None) or ())
        # One image per contact the catalogue lists (its own beliefs).
        system, images = self._image_request(
            system, {t.contact_id for t in catalogue if t.contact_id is not None})
        self.last_prompt = (system, um)
        context = {"kind": "commander", "now": int(view.now), "commander": int(view.agent_id),
                   "catalogue": [t.task_id for t in catalogue],
                   "active_agents": sorted(int(a) for a in positions),
                   "prompt_version": COMMANDER_PROMPT_VERSION}
        content = self._fetch(system, um, validation_context=context,
                             **({"images": images} if images else {}))
        # TransportFailure / ReplayMiss / BudgetExhausted raise
        return self.interpret(content)

    def interpret(self, content):
        """Parse one response into ``(agent, task_id)`` pairs; raise if it is
        unusable. Capacity, catalogue and per-agent checks are
        ``validate_assignment``'s. Shared by the live path and the offline
        semantic validator (planning/semantic_validation.py)."""
        if not (content or "").strip():
            raise ValueError("empty_response")
        ans = _Answer.model_validate_json(content)  # ValidationError raises
        self.last_reasoning = ans.reasoning
        # A key that is not an integer agent id makes the response unusable
        # (the strict schema admits none); an integer that is not an active
        # agent is the validator's NOT_ACTIVE failure.
        return [(int(a), t) for a, t in ans.task_by_agent.items()]
