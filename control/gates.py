"""Intruder approach gates: public waypoints outside the defended region.

Each gate lies on the ray from the defended-region centre at a fixed bearing
(degrees, counterclockwise from the positive x-axis), at the point where that
ray leaves the ACTUAL defended polygon (the shape the breach test uses), plus
an explicit outward offset. An intruder that has selected a gate advances,
permanently, to the interior-attack phase on first entering that gate's
arrival disk. Construction refuses geometry where a gate or its arrival disk
touches the defended polygon or leaves the arena.

Development constants (proposed, reviewed; not frozen): bearings 180/210/240/
270 deg, outward offset 6 units, arrival radius 2 units.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

GATE_BEARINGS_DEG = (180.0, 210.0, 240.0, 270.0)
GATE_OUTWARD_OFFSET = 6.0
GATE_ARRIVAL_RADIUS = 2.0

APPROACH = "approach"
INTERIOR = "interior"


@dataclass(frozen=True)
class Gate:
    gate_id: int
    bearing_deg: float
    position: Tuple[float, float]
    boundary_point: Tuple[float, float]
    arrival_radius: float


def _ray_exit(center, direction, polygon):
    """Farthest intersection of the ray center + t*direction (t >= 0) with the
    polygon's edges: where the ray finally leaves the polygon."""
    best = None
    closed = np.vstack([polygon, polygon[0]])
    for k in range(len(polygon)):
        a, b = closed[k], closed[k + 1]
        e = b - a
        m = np.array([[direction[0], -e[0]], [direction[1], -e[1]]])
        det = np.linalg.det(m)
        if abs(det) < 1e-12:
            continue
        t, s = np.linalg.solve(m, a - center)
        if t >= 0 and -1e-12 <= s <= 1 + 1e-12 and (best is None or t > best):
            best = t
    if best is None:
        raise ValueError("gate ray never leaves the defended polygon")
    return center + best * direction


def _distance_to_boundary(point, polygon):
    closed = np.vstack([polygon, polygon[0]])
    best = np.inf
    for k in range(len(polygon)):
        a, b = closed[k], closed[k + 1]
        e = b - a
        alpha = np.clip(np.dot(point - a, e) / max(float(np.dot(e, e)), 1e-12), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(a + alpha * e - point)))
    return best


def _inside(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    for k in range(n):
        x1, y1 = polygon[k]
        x2, y2 = polygon[(k + 1) % n]
        if (y1 > y) != (y2 > y):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xc:
                inside = not inside
    return inside


def build_gates(center, polygon, arena_min, arena_max,
                bearings_deg: Sequence[float] = GATE_BEARINGS_DEG,
                outward_offset: float = GATE_OUTWARD_OFFSET,
                arrival_radius: float = GATE_ARRIVAL_RADIUS) -> Tuple[Gate, ...]:
    center = np.asarray(center, dtype=float).reshape(2)
    polygon = np.asarray(polygon, dtype=float).reshape(-1, 2)
    gates = []
    for gate_id, bearing in enumerate(bearings_deg):
        u = np.array([np.cos(np.deg2rad(bearing)), np.sin(np.deg2rad(bearing))])
        boundary = _ray_exit(center, u, polygon)
        position = boundary + float(outward_offset) * u
        if _inside(position, polygon) or _distance_to_boundary(position, polygon) <= arrival_radius:
            raise ValueError(f"gate {gate_id}: arrival disk reaches the defended polygon")
        lo, hi = float(arena_min) + arrival_radius, float(arena_max) - arrival_radius
        if not (lo <= position[0] <= hi and lo <= position[1] <= hi):
            raise ValueError(f"gate {gate_id}: arrival disk leaves the arena")
        gates.append(Gate(gate_id, float(bearing), (float(position[0]), float(position[1])),
                          (float(boundary[0]), float(boundary[1])), float(arrival_radius)))
    return tuple(gates)


def advance_phase(phase, selected_gate, position, gates):
    """One step of the one-way phase machine: APPROACH -> INTERIOR on first
    entry into the SELECTED gate's arrival disk. Never goes back."""
    if phase == INTERIOR or selected_gate is None:
        return phase
    g = gates[selected_gate]
    if np.linalg.norm(np.asarray(position, float)[0:2] - np.asarray(g.position)) <= g.arrival_radius:
        return INTERIOR
    return phase
