"""Intruder gates (control/gates.py) and per-intruder decisions
(planning/intruder.py) under intruder_team="gates". Rule-based, free."""

import numpy as np
import pytest

from conftest import build_simulation, capture_fd_output, fingerprint
from control.gates import (
    APPROACH, GATE_ARRIVAL_RADIUS, GATE_BEARINGS_DEG, GATE_OUTWARD_OFFSET, INTERIOR,
    advance_phase, build_gates,
)
from planning.intruder import (
    ATTACK, EVADE, GateRuleIntruderPlanner, IntruderDecision, IntruderObservation,
    fallback_decision, validate_decision,
)
from planning.roles import RedRole
from scripts import network_conditions
from scripts.experiment_policy import (
    NEW_EXPERIMENT_INTRUDER_INFO, NEW_EXPERIMENT_PURSUIT, NEW_EXPERIMENT_SAFETY,
    NEW_EXPERIMENT_TASKS,
)
from scripts.episode_builder import build_episode

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def geometry():
    sim = build_simulation(num_blues=4, num_reds=2)
    return sim, build_gates(sim.defense_center, sim.defended_shape_points,
                            sim.arena_min, sim.arena_max)


def _dist_to_polygon(p, poly):
    poly = np.asarray(poly, float)
    best = np.inf
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        e = b - a
        t = np.clip(np.dot(p - a, e) / np.dot(e, e), 0, 1)
        best = min(best, np.linalg.norm(a + t * e - p))
    return best


def test_gates_sit_on_their_bearing_ccw_from_positive_x_at_the_declared_offset(geometry):
    sim, gates = geometry
    c = np.asarray(sim.defense_center, float)
    assert [g.bearing_deg for g in gates] == list(GATE_BEARINGS_DEG)
    for g in gates:
        u = np.array([np.cos(np.deg2rad(g.bearing_deg)), np.sin(np.deg2rad(g.bearing_deg))])
        np.testing.assert_allclose(np.asarray(g.position) - np.asarray(g.boundary_point),
                                   GATE_OUTWARD_OFFSET * u, atol=1e-9)
        d = np.asarray(g.boundary_point) - c
        np.testing.assert_allclose(d / np.linalg.norm(d), u, atol=1e-9)
        assert _dist_to_polygon(np.asarray(g.boundary_point), sim.defended_shape_points) < 1e-9
    # 180 deg is due west of the centre, 270 deg due south.
    assert gates[0].position[1] == pytest.approx(c[1]) and gates[0].position[0] < c[0]
    assert gates[3].position[0] == pytest.approx(c[0]) and gates[3].position[1] < c[1]


def test_gates_and_arrival_disks_stay_outside_the_region_and_inside_the_arena(geometry):
    sim, gates = geometry
    for g in gates:
        p = np.asarray(g.position)
        assert not sim._point_in_polygon(p, sim.defended_shape_points)
        assert _dist_to_polygon(p, sim.defended_shape_points) > GATE_ARRIVAL_RADIUS
        assert sim.arena_min + GATE_ARRIVAL_RADIUS <= p.min()
        assert p.max() <= sim.arena_max - GATE_ARRIVAL_RADIUS


def test_geometry_violations_are_refused(geometry):
    sim, _ = geometry
    with pytest.raises(ValueError, match="defended polygon"):
        build_gates(sim.defense_center, sim.defended_shape_points, sim.arena_min,
                    sim.arena_max, outward_offset=1.0)
    with pytest.raises(ValueError, match="arena"):
        build_gates(sim.defense_center, sim.defended_shape_points, sim.arena_min,
                    sim.arena_max, outward_offset=80.0)


def test_the_phase_advances_once_only_at_the_selected_gate(geometry):
    _, gates = geometry
    at0, at1 = np.asarray(gates[0].position), np.asarray(gates[1].position)
    assert advance_phase(APPROACH, 1, at0, gates) == APPROACH   # wrong gate
    assert advance_phase(APPROACH, None, at0, gates) == APPROACH
    assert advance_phase(APPROACH, 0, at0 + [1.9, 0], gates) == INTERIOR
    assert advance_phase(APPROACH, 0, at0 + [2.1, 0], gates) == APPROACH
    assert advance_phase(INTERIOR, 1, at0 + [50, 50], gates) == INTERIOR  # never back
    assert advance_phase(INTERIOR, 1, at1, gates) == INTERIOR


# ---------------------------------------------------------------------------
# Validation and fallback
# ---------------------------------------------------------------------------


def _obs(phase=APPROACH, commitment=None, believed=None):
    return IntruderObservation(
        agent_id=0, now=10, dt=0.1, own_state=(40.0, 60.0, 0.0, 0.0, 0.0), phase=phase,
        commitment=commitment, teammates={}, teammate_intents={},
        believed_blues=believed or {}, gates=((0, (69.4, 82.0)), (1, (70.2, 75.2))),
        defense_center=(82.0, 82.0))


@pytest.mark.parametrize("decision, phase, believed, reason", [
    (IntruderDecision(ATTACK, gate_id=1), APPROACH, None, None),
    (IntruderDecision(ATTACK), APPROACH, None, "approach_attack_needs_a_known_gate"),
    (IntruderDecision(ATTACK, gate_id=7), APPROACH, None, "approach_attack_needs_a_known_gate"),
    (IntruderDecision(ATTACK), INTERIOR, None, None),
    (IntruderDecision(ATTACK, gate_id=0), INTERIOR, None, "interior_attack_takes_no_gate"),
    (IntruderDecision(EVADE, evade_blue_id=2), APPROACH, {2: ((0,) * 5, 3)}, None),
    (IntruderDecision(EVADE, evade_blue_id=3), INTERIOR, {2: ((0,) * 5, 3)}, "evade_target_not_believed"),
    (IntruderDecision(EVADE, evade_blue_id=2, gate_id=0), APPROACH, {2: ((0,) * 5, 3)}, "evade_with_gate"),
    (IntruderDecision("recon"), APPROACH, None, "unknown_role"),
    ("ATTACK gate 1", APPROACH, None, "not_a_decision"),
])
def test_decisions_are_validated_against_the_phase(decision, phase, believed, reason):
    assert validate_decision(decision, _obs(phase, believed=believed)) == reason


def test_fallback_keeps_a_valid_commitment_else_attacks_by_phase():
    keep = IntruderDecision(ATTACK, gate_id=1)
    assert fallback_decision(_obs(commitment=keep)) == (keep, "kept_commitment")
    stale = IntruderDecision(EVADE, evade_blue_id=9)  # no longer believed
    d, why = fallback_decision(_obs(commitment=stale))
    assert why == "attack_gate" and d.gate_id in (0, 1)
    d, why = fallback_decision(_obs(INTERIOR, commitment=keep))  # gate invalid inside
    assert (d, why) == (IntruderDecision(ATTACK), "attack_interior")


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


def _cfg(planner, **extra):
    belief, cfg = network_conditions.episode_cfg("multi_bearing", "REF", "C")
    cfg = dict(cfg, **NEW_EXPERIMENT_PURSUIT, **NEW_EXPERIMENT_SAFETY,
               **NEW_EXPERIMENT_INTRUDER_INFO, **NEW_EXPERIMENT_TASKS,
               intruder_team="gates", red_planner=planner, **extra)
    return belief, cfg


def _run(planner, seed=3706, **extra):
    belief, cfg = _cfg(planner, **extra)
    sim = build_episode(seed, belief, cfg)
    with capture_fd_output():
        sim.run()
    return sim


def test_default_is_no_intruder_team_and_requirements_are_enforced():
    sim = build_simulation()
    assert sim.intruder_team is None and "intruder_team" not in sim._init_args
    with pytest.raises(ValueError, match="shared_beliefs"):
        build_simulation(intruder_team="gates", red_role_planner=GateRuleIntruderPlanner())
    with pytest.raises(ValueError, match="decide_all"):
        build_simulation(intruder_team="gates", intruder_info="shared_beliefs")


class Recording:
    """Wraps a planner; records what it was shown and decided, per epoch."""

    def __init__(self, inner, reverse=False):
        self.inner, self.reverse, self.epochs = inner, reverse, []

    def decide_all(self, observations):
        order = list(reversed(observations)) if self.reverse else list(observations)
        out = {o.agent_id: self.inner.decide(o) for o in order}
        self.epochs.append((observations, out))
        return out


def test_teammate_intents_are_the_previous_epochs_decisions_only():
    rec = Recording(GateRuleIntruderPlanner())
    _run(rec)
    assert len(rec.epochs) > 2
    for (prev_obs, prev_out), (obs, _) in zip(rec.epochs, rec.epochs[1:]):
        for o in obs:
            for k, (decision, issued_at) in o.teammate_intents.items():
                assert decision == prev_out[k] and issued_at == prev_obs[0].now
    first_obs, _ = rec.epochs[0]
    assert all(o.teammate_intents == {} for o in first_obs)


def test_decisions_do_not_depend_on_completion_order():
    a = _run(Recording(GateRuleIntruderPlanner()))
    b = _run(Recording(GateRuleIntruderPlanner(), reverse=True))
    assert fingerprint(a) == fingerprint(b)


def test_phase_is_exposed_and_attack_targets_follow_it():
    rec = Recording(GateRuleIntruderPlanner())
    sim = _run(rec)
    transitions = [e for e in sim.intruder_log if e.get("event") == "phase"]
    assert transitions, "seed 3706 reaches a gate"
    t = transitions[0]
    assert (t["from"], t["to"]) == (APPROACH, INTERIOR)
    for obs, out in rec.epochs:
        for o in obs:
            d = out[o.agent_id]
            if o.phase == INTERIOR and d.role == ATTACK:
                assert d.gate_id is None
            if o.agent_id == t["red"] and o.now > t["step"]:
                assert o.phase == INTERIOR  # never reset, EVADE included
    assert t["red"] not in sim.gate_attack_controller.goals  # interior: the centre


def test_approach_goals_are_the_selected_gates():
    sim = _run(GateRuleIntruderPlanner(), steps=30)
    for j, goal in sim.gate_attack_controller.goals.items():
        assert sim.red_phase[j] == APPROACH
        assert goal == sim.gates[sim._selected_gate[j]].position


def test_evade_sees_only_the_chosen_blue():
    sim_cfg = _cfg(GateRuleIntruderPlanner())
    sim = build_episode(3705, *sim_cfg)
    seen = []
    orig = sim.solver.solve_decentralized

    def spy(*a, **k):
        for j, d in sim.red_decisions.items():
            if d.role == EVADE and sim.red_outcomes[j] == "active":
                seen.append(set(k["red_believed_blues"].get(j, {})) <= {d.evade_blue_id})
        return orig(*a, **k)

    sim.solver.solve_decentralized = spy
    with capture_fd_output():
        sim.run()
    assert seen and all(seen)


class Garbage:
    def decide_all(self, observations):
        return {o.agent_id: "go west" for o in observations}


def test_invalid_outputs_fall_back_and_are_logged():
    sim = _run(Garbage(), steps=40)
    entries = [e for e in sim.intruder_log if "decision" in e]
    assert entries
    for e in entries:
        assert e["invalid_reason"] == "not_a_decision" and e["fallback"] is not None
        assert e["decision"].role == ATTACK
    assert all(r == RedRole.ATTACK for r in sim.red_role_assignment.values())
