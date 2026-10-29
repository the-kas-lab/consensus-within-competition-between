"""One LLM per intruder (intruder_team="gates").

Each intruder has its own planner instance and its own stateless calls (an
isolated context: no conversation history, no shared object), built from that
intruder's frozen ``IntruderObservation`` only. The prompt and the output
schema are PHASE-SPECIFIC: in the approach phase the model may choose a gate;
after the one-way transition to the interior phase the schema has no gate
field at all, so the interface never advertises a choice the controller would
ignore. Outputs are still validated by ``planning.intruder.validate_decision``
in the simulator, and every failure falls back there and is logged.

The call path (cache, provenance, failure replay, decode settings) is
``LocalLLMPlanner``'s, as for the defender bridge. All observations exist
before any call is made; calls may run concurrently and their completion order
cannot change any input.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import numpy as np
from pydantic import BaseModel, ValidationError, field_validator

from control.gates import APPROACH, INTERIOR
from planning.intruder import ATTACK, EVADE, IntruderDecision, IntruderObservation
from planning.local_llm import LocalLLMPlanner, TransportFailure
from planning.stops import StopRun

INTRUDER_PROMPT_VERSION = "intruder.4"  # the label in every cache key; do not change
_GAME_SENTENCE = ("a blue that touches you destroys you (and itself).")
NO_TARGET = -1


def _schema(phase: str) -> dict:
    """One required integer target: the gate to attack or the blue to evade
    (-1 for an interior attack). No nullable field."""
    what = ("the gate_id to attack, or the blue id to evade" if phase == APPROACH
            else "the blue id to evade, or -1 for attack")
    return {"type": "object",
            "properties": {"reasoning": {"type": "string"},
                           "action": {"type": "string", "enum": [ATTACK, EVADE]},
                           "target_id": {"type": "integer", "description": what}},
            "required": ["reasoning", "action", "target_id"],
            "additionalProperties": False}


class _Answer(BaseModel):
    reasoning: str
    action: str
    target_id: int

    @field_validator("action", mode="before")
    @classmethod
    def _lower(cls, v):
        return v.strip().lower() if isinstance(v, str) else v


def _base_prompt(phase: str) -> str:
    common = (
        "You are ONE red intruder drone in a 2D perimeter-defense game. Your team "
        "tries to breach a defended region; blue defenders try to intercept you, and "
        "a blue that touches you destroys you (and itself). You decide only for "
        "yourself; each red teammate decides for itself.\n"
        "You know: your own state, your teammates' states, your team's shared "
        "sightings of blue defenders (each with when it was seen and its predicted "
        "position now; blues you have not seen may exist), and what each teammate "
        "chose at the previous planning step.\n"
        "Actions:\n"
        "- evade: move away from ONE blue you currently believe in "
        "(evade_blue_id). Evading does not change your phase.\n"
    )
    if phase == APPROACH:
        return common + (
            "- attack: fly to ONE of the approach gates (gate_id) -- public waypoints "
            "outside the defended region. On reaching your chosen gate you switch "
            "permanently to the interior attack. You may change gate at any planning "
            "step before you reach one.\n"
            "Set gate_id for attack and evade_blue_id for evade; set the other to "
            "null. Answer with JSON matching the schema; give your reasoning first."
        )
    return common + (
        "- attack: you have passed your gate and are in the INTERIOR phase: attack "
        "means advance on the defended region's centre. There are no gates to "
        "choose any more.\n"
        "Set evade_blue_id for evade, null for attack. Answer with JSON matching "
        "the schema; give your reasoning first."
    )


def system_prompt(phase: str, horizon_steps, dt) -> str:
    """The system prompt: the base text with the answer instructions rewritten
    for ``target_id``, the sensor geometry, and the objective-only red brief
    (which states the episode length ``horizon_steps`` * ``dt``)."""
    from coordination.mission_policy import red_brief_v1
    from planning.prompt_builder import SENSOR_GEOMETRY_TEXT

    base = _base_prompt(phase)
    if phase == APPROACH:
        old = ("Set gate_id for attack and evade_blue_id for evade; set the other to "
               "null. Answer with JSON matching the schema; give your reasoning first.")
        new = ("target_id is the gate_id you attack, or the blue id you evade. Answer "
               "with JSON matching the schema; give your reasoning first.")
    else:
        old = ("Set evade_blue_id for evade, null for attack. Answer with JSON matching "
               "the schema; give your reasoning first.")
        new = ("target_id is the blue id you evade, or -1 for attack. Answer with JSON "
               "matching the schema; give your reasoning first.")
    assert old in base
    base = base.replace(old, new).replace("(evade_blue_id)", "(target_id)").replace(
        "(gate_id)", "(target_id)")
    assert "target_id is" in base
    base = base.replace("target_id is", SENSOR_GEOMETRY_TEXT + "\ntarget_id is", 1)
    assert base.count(_GAME_SENTENCE) == 1
    return base.replace(_GAME_SENTENCE,
                        _GAME_SENTENCE + " " + red_brief_v1(horizon_steps, dt), 1)


def _fmt_state(s, heading: str = None):
    """``heading``: None (no heading shown), "current" or "observed"."""
    s = np.asarray(s, float)
    out = f"pos=[{s[0]:.1f}, {s[1]:.1f}] vel=[{s[2]:.2f}, {s[3]:.2f}]"
    if heading is not None and s.size >= 5:
        from planning.prompt_builder import heading_deg

        label = "heading_deg" if heading == "current" else "observed_heading_deg"
        out += f" {label}={heading_deg(s[4]):.0f}"
    return out


def user_message(obs: IntruderObservation) -> str:
    cur, obsd = "current", "observed"
    lines = [f"time_step: {obs.now} (dt={obs.dt} s)",
             f"you: red {obs.agent_id} {_fmt_state(obs.own_state, cur)}",
             f"phase: {obs.phase}",
             f"defended_region_centre: [{obs.defense_center[0]:.1f}, {obs.defense_center[1]:.1f}]"]
    c = obs.commitment
    show_gate = c is not None and c.gate_id is not None and obs.phase != INTERIOR
    lines.append("your_current_action: " + (
        "none" if c is None else
        f"{c.role}" + (f" gate {c.gate_id}" if show_gate else "")
        + (f" blue {c.evade_blue_id}" if c.evade_blue_id is not None else "")))
    if obs.phase == APPROACH:
        lines.append("gates:")
        p = np.asarray(obs.own_state, float)[0:2]
        for gid, pos in obs.gates:
            lines.append(f"  gate {gid} at [{pos[0]:.1f}, {pos[1]:.1f}] "
                         f"distance {np.linalg.norm(p - np.asarray(pos)):.1f}")
    lines.append("teammates:" + ("" if obs.teammates else " none active"))
    for k in sorted(obs.teammates):
        intent = obs.teammate_intents.get(k)
        said = "no previous choice" if intent is None else (
            f"chose {intent[0].role}"
            + (f" gate {intent[0].gate_id}" if intent[0].gate_id is not None else "")
            + (f" blue {intent[0].evade_blue_id}" if intent[0].evade_blue_id is not None else "")
            + f" at step {intent[1]}")
        lines.append(f"  red {k} {_fmt_state(obs.teammates[k], cur)}; {said}")
    lines.append("blue_sightings (team-shared; others may exist unseen):"
                 + ("" if obs.believed_blues else " none"))
    for b in sorted(obs.believed_blues):
        state, seen = obs.believed_blues[b]
        lines.append(f"  blue {b} predicted {_fmt_state(state, obsd)} seen {obs.now - seen} steps ago")
    return "\n".join(lines)


class _OneIntruder(LocalLLMPlanner):
    _SCHEMA_NAME = "intruder_decision"
    _PROMPT_VERSION = INTRUDER_PROMPT_VERSION

    def decide(self, obs: IntruderObservation):
        """An IntruderDecision, or a string naming why there is none (logged by
        the simulator as an invalid output, which then falls back)."""
        self._SCHEMA = _schema(obs.phase)  # per instance, per call
        horizon = getattr(self, "episode_horizon", None)
        if horizon is None:
            raise ValueError("the intruder prompt states the episode length; call "
                             "LLMIntruderTeam.set_episode_horizon first")
        if abs(float(horizon[1]) - float(obs.dt)) > 1e-12:
            raise ValueError(f"episode horizon dt {horizon[1]} != observation dt {obs.dt}")
        sp = system_prompt(obs.phase, *horizon)
        um = user_message(obs)
        self.last_prompt = (sp, um)
        try:
            content = self._fetch(sp, um, validation_context=dict(
                observation_to_dict(obs), prompt_version=INTRUDER_PROMPT_VERSION))
        except StopRun:
            raise
        except TransportFailure as exc:
            return f"call_failure:{exc.error_type}"
        except Exception as exc:  # noqa: BLE001 -- e.g. no client and no entry
            return f"call_failure:{type(exc).__name__}"
        return self.interpret(content, obs.phase)

    def interpret(self, content, phase=None):
        """Parse one response: an IntruderDecision, or the reason there is none.
        Semantic validity against the observation is the simulator's
        ``validate_decision``. Shared by the live path and the offline
        semantic validator (planning/semantic_validation.py)."""
        if not (content or "").strip():
            return "empty_response"
        try:
            ans = _Answer.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError) as exc:
            return f"schema_failure:{type(exc).__name__}"
        self.last_reasoning = ans.reasoning
        target = int(ans.target_id)
        if ans.action == EVADE:
            return IntruderDecision(EVADE, evade_blue_id=target, reasoning=ans.reasoning)
        if ans.action == ATTACK and phase == APPROACH:
            return IntruderDecision(ATTACK, gate_id=target, reasoning=ans.reasoning)
        if ans.action == ATTACK:
            # Interior attack takes no target; anything but -1 is named.
            if target != NO_TARGET:
                return "interior_attack_with_target"
            return IntruderDecision(ATTACK, reasoning=ans.reasoning)
        return IntruderDecision(ans.action, reasoning=ans.reasoning)


def _decision_to_dict(d):
    return None if d is None else {"role": d.role, "gate_id": d.gate_id,
                                   "evade_blue_id": d.evade_blue_id}


def _decision_from_dict(d):
    return None if d is None else IntruderDecision(d["role"], gate_id=d["gate_id"],
                                                   evade_blue_id=d["evade_blue_id"])


def observation_to_dict(obs: IntruderObservation) -> dict:
    """The whole frozen observation as plain data (reasoning text dropped),
    enough to rebuild it for ``validate_decision`` / ``fallback_decision``."""
    return {
        "kind": "intruder", "agent_id": int(obs.agent_id), "now": int(obs.now),
        "dt": float(obs.dt), "own_state": [float(v) for v in obs.own_state],
        "phase": obs.phase, "commitment": _decision_to_dict(obs.commitment),
        "teammates": {str(k): [float(v) for v in s] for k, s in obs.teammates.items()},
        "teammate_intents": {str(k): [_decision_to_dict(d), int(t)]
                             for k, (d, t) in obs.teammate_intents.items()},
        "believed_blues": {str(b): [[float(v) for v in s], int(t)]
                           for b, (s, t) in obs.believed_blues.items()},
        "gates": [[int(g), [float(p[0]), float(p[1])]] for g, p in obs.gates],
        "defense_center": [float(obs.defense_center[0]), float(obs.defense_center[1])],
    }


def observation_from_dict(d: dict) -> IntruderObservation:
    return IntruderObservation(
        agent_id=int(d["agent_id"]), now=int(d["now"]), dt=float(d["dt"]),
        own_state=tuple(d["own_state"]), phase=d["phase"],
        commitment=_decision_from_dict(d["commitment"]),
        teammates={int(k): tuple(s) for k, s in d["teammates"].items()},
        teammate_intents={int(k): (_decision_from_dict(v[0]), int(v[1]))
                          for k, v in d["teammate_intents"].items()},
        believed_blues={int(b): (tuple(v[0]), int(v[1])) for b, v in d["believed_blues"].items()},
        gates=tuple((int(g), (float(p[0]), float(p[1]))) for g, p in d["gates"]),
        defense_center=(float(d["defense_center"][0]), float(d["defense_center"][1])),
    )


class LLMIntruderTeam:
    """Holds one planner per intruder; ``decide_all`` calls them (optionally
    concurrently) on observations that already exist."""

    def __init__(self, max_workers: int = 1, **local_kwargs):
        self.max_workers = int(max_workers)
        self.local_kwargs = local_kwargs
        self.planners: Dict[int, _OneIntruder] = {}
        self.decision_log = []
        self.episode_horizon = None

    def set_episode_horizon(self, horizon_steps: int, dt: float):
        """The run's length, stated in the prompt (set by the runner from the
        built Simulation)."""
        self.episode_horizon = (int(horizon_steps), float(dt))
        for p in self.planners.values():
            p.episode_horizon = self.episode_horizon

    def planner_for(self, agent_id: int) -> _OneIntruder:
        if agent_id not in self.planners:
            self.planners[agent_id] = _OneIntruder(agent_id, context=None, **self.local_kwargs)
            self.planners[agent_id].episode_horizon = self.episode_horizon
        return self.planners[agent_id]

    def decide_all(self, observations):
        planners = {o.agent_id: self.planner_for(o.agent_id) for o in observations}
        if self.max_workers == 1 or len(observations) <= 1:
            out = {o.agent_id: planners[o.agent_id].decide(o) for o in observations}
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(observations))) as pool:
                futures = {o.agent_id: pool.submit(planners[o.agent_id].decide, o)
                           for o in observations}
                out = {a: f.result() for a, f in futures.items()}
        return out

    def diagnostics(self) -> dict:
        return {"calls": sum(p.calls for p in self.planners.values()),
                "cache_hits": sum(p.cache_hits for p in self.planners.values()),
                "transport_failures": sum(p.transport_failures for p in self.planners.values())}
