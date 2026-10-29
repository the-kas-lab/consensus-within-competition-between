"""Per-intruder decisions for the competing-teams experiments.

Each intruder decides from its own frozen ``IntruderObservation``: its own
state and phase, accurate teammate states, the team's shared tactical
beliefs about blues (ideal intruder channel, ``intruder_info=
"shared_beliefs"``), teammates' intents from the PREVIOUS completed planning
epoch, the public gate geometry, and its own current commitment. All
observations are built before any decision is requested, so the order in
which decisions complete cannot expose another intruder's new choice.

Decisions:

* ``ATTACK(gate_id)`` in the approach phase -- head for that gate;
* ``ATTACK()`` in the interior phase -- head for the interior objective (no
  gate: after the one-way transition a gate is not a valid output);
* ``EVADE(blue_id)`` in either phase, for a blue the intruder currently
  believes in. EVADE interrupts motion; it never resets the phase.

Invalid output: keep the current commitment if it is still valid, else the
declared deterministic ATTACK fallback (``fallback_decision``). Every
fallback is logged by the simulator.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from control.gates import APPROACH, INTERIOR

ATTACK = "attack"
EVADE = "evade"


@dataclass(frozen=True)
class IntruderDecision:
    role: str
    gate_id: Optional[int] = None
    evade_blue_id: Optional[int] = None
    reasoning: str = ""


@dataclass(frozen=True)
class IntruderObservation:
    agent_id: int
    now: int
    dt: float
    own_state: Tuple[float, ...]
    phase: str
    commitment: Optional[IntruderDecision]
    teammates: Dict[int, Tuple[float, ...]]
    # {teammate: (decision, issued_at)} -- the previous completed epoch only.
    teammate_intents: Dict[int, Tuple[IntruderDecision, int]]
    # {blue_id: (state predicted to now, observed_at)} -- shared intruder beliefs.
    believed_blues: Dict[int, Tuple[Tuple[float, ...], int]]
    gates: Tuple[Tuple[int, Tuple[float, float]], ...]
    defense_center: Tuple[float, float]
    extra: dict = field(default_factory=dict)


def validate_decision(decision, obs: IntruderObservation) -> Optional[str]:
    """None if valid for this observation, else the reason it is not."""
    if not isinstance(decision, IntruderDecision):
        return "not_a_decision"
    gate_ids = {g for g, _ in obs.gates}
    if decision.role == ATTACK:
        if decision.evade_blue_id is not None:
            return "attack_with_evade_target"
        if obs.phase == APPROACH:
            if decision.gate_id not in gate_ids:
                return "approach_attack_needs_a_known_gate"
        elif decision.gate_id is not None:
            return "interior_attack_takes_no_gate"
        return None
    if decision.role == EVADE:
        if decision.gate_id is not None:
            return "evade_with_gate"
        if decision.evade_blue_id not in obs.believed_blues:
            return "evade_target_not_believed"
        return None
    return "unknown_role"


def nearest_gate(obs: IntruderObservation) -> int:
    p = np.asarray(obs.own_state, float)[0:2]
    return min(obs.gates, key=lambda g: (float(np.linalg.norm(p - np.asarray(g[1]))), g[0]))[0]


def fallback_decision(obs: IntruderObservation) -> Tuple[IntruderDecision, str]:
    """Keep a still-valid commitment; otherwise the declared ATTACK: the
    currently selected gate (else the nearest) in the approach phase, the
    interior objective after it."""
    if obs.commitment is not None and validate_decision(obs.commitment, obs) is None:
        return obs.commitment, "kept_commitment"
    if obs.phase == INTERIOR:
        return IntruderDecision(ATTACK), "attack_interior"
    current = obs.commitment.gate_id if obs.commitment is not None else None
    gate = current if current in {g for g, _ in obs.gates} else nearest_gate(obs)
    return IntruderDecision(ATTACK, gate_id=gate), "attack_gate"


class GateRuleIntruderPlanner:
    """Deterministic rule intruder for the free rule backend (plumbing, not a
    result of the paper): EVADE the
    nearest believed blue closer than ``evade_radius``; otherwise ATTACK --
    keeping the current gate in the approach phase, else the nearest one."""

    def __init__(self, evade_radius: float = 6.0):
        self.evade_radius = float(evade_radius)

    def decide(self, obs: IntruderObservation) -> IntruderDecision:
        p = np.asarray(obs.own_state, float)[0:2]
        near = sorted(
            (float(np.linalg.norm(p - np.asarray(s, float)[0:2])), b)
            for b, (s, _) in obs.believed_blues.items()
        )
        if near and near[0][0] < self.evade_radius:
            return IntruderDecision(EVADE, evade_blue_id=near[0][1])
        if obs.phase == INTERIOR:
            return IntruderDecision(ATTACK)
        current = obs.commitment.gate_id if obs.commitment is not None else None
        return IntruderDecision(ATTACK, gate_id=current if current is not None else nearest_gate(obs))

    def decide_all(self, observations):
        return {o.agent_id: self.decide(o) for o in observations}
