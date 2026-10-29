"""Verification script for the breach predicate, Simulation._point_in_polygon.

Free (no API calls). Plain-assert style, no test framework, same convention as
verify_bearing_fov.py / verify_contact_substitution.py.

    python3 -m verification.verify_point_in_polygon

Why this exists. The previous predicate guarded its ray-crossing division with
``max(y_prev - y_curr, 1e-9)``, which also replaced every NEGATIVE denominator
(every edge whose y increases) with 1e-9. Such an edge then counted as a
crossing always or never, whatever the point's x. On the defended shape that
made a band OUTSIDE the region (to the west/south-west, reaching the arena
edge) read as inside, so a red far from the region "breached" and ended the
episode. Much of the true region read as outside, so real entries could go
unrecorded. Breach (touchdown) is the episode's terminal outcome, so both
errors change results: breach rates, and every attrition / score / episode-
length figure that depends on when an episode stops.

Checks:
1. The original defect's two errors on a plain triangle: an exterior point
   read inside and an interior point read outside.
2. Agreement with an independent reference (matplotlib.path.Path) on random
   points away from the boundary, for oblique edges, both vertex orders,
   concave polygons, and the actual defended shape of a live Simulation.
3. The boundary, decided explicitly: the region is CLOSED (vertices and edge
   points are inside), and points 1e-6 outside an edge are outside.
4. The call site: on a live Simulation, sweeping a red over a grid around the
   defended region, _resolve_active_reds ends the episode exactly where the
   reference says the point is inside.
"""

import numpy as np
from matplotlib.path import Path

from simulation.testbed import Simulation

inside = Simulation._point_in_polygon
TRIANGLE = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]


def _edge_distance(p, poly):
    v = np.asarray(poly, float)
    p = np.asarray(p, float)
    best = np.inf
    for a, b in zip(np.roll(v, 1, axis=0), v):
        d = b - a
        t = np.clip(np.dot(p - a, d) / max(np.dot(d, d), 1e-300), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(p - (a + t * d))))
    return best


def _star(n=7, r_out=10.0, r_in=4.0, center=(50.0, 50.0), phase=0.3):
    return [(center[0] + (r_out if k % 2 == 0 else r_in) * np.cos(phase + np.pi * k / n),
             center[1] + (r_out if k % 2 == 0 else r_in) * np.sin(phase + np.pi * k / n))
            for k in range(2 * n)]


def _sim():
    sim = Simulation(solver_mode="decentralized", num_blues=2, num_reds=2, sim_steps=5,
                     horizon=10, seed=42, enable_blue_blue_cbf=True, enable_blue_red_cbf=True)
    sim.setup_agents()
    return sim


def check_original_defect():
    assert inside((-5.0, 3.0), TRIANGLE) is False, "exterior point read inside"
    assert inside((5.0, 3.0), TRIANGLE) is True, "interior point read outside"
    print("PASS: triangle exterior/interior points classified correctly (both original errors gone)")


def check_reference(polygons):
    rng = np.random.default_rng(20260928)
    for name, poly in polygons.items():
        ref = Path(np.asarray(poly, float))
        lo, hi = np.min(poly, axis=0) - 5, np.max(poly, axis=0) + 5
        n = n_in = 0
        for p in rng.uniform(lo, hi, size=(4000, 2)):
            if _edge_distance(p, poly) < 1e-6:
                continue
            expected = bool(ref.contains_point(p))
            assert inside(p, poly) is expected, (name, p.tolist())
            assert inside(p, poly[::-1]) is expected, ("reversed", name, p.tolist())
            n += 1
            n_in += expected
        assert n > 3500 and 100 < n_in < n - 100, name
        print(f"PASS: {name}: {n} points agree with matplotlib.path.Path, both vertex orders")


def check_boundary(polygons):
    for name, poly in polygons.items():
        v = np.asarray(poly, float)
        for a, b in zip(np.roll(v, 1, axis=0), v):
            assert inside(tuple(a), v) and inside(tuple((a + b) / 2), v), name
            assert inside(tuple(a + 0.3 * (b - a)), v), name
    v = np.asarray(TRIANGLE, float)
    for a, b in zip(np.roll(v, 1, axis=0), v):
        d = b - a
        normal = np.array([d[1], -d[0]]) / np.linalg.norm(d)
        mid = (a + b) / 2
        assert inside(tuple(mid + 1e-6 * normal), v) is False
        assert inside(tuple(mid - 1e-6 * normal), v) is True
    print("PASS: closed region: vertices and edge points inside, 1e-6 outside an edge is outside")


def check_call_site():
    sim = _sim()
    poly = np.asarray(sim.defended_shape_points, float)
    ref = Path(poly)
    xb = np.asarray(sim.x_current, float).ravel()
    blues = [xb[i * sim.blue_nx: i * sim.blue_nx + 2] for i in range(sim.num_blues)]
    sim.red_state[1 * sim.red_nx: 1 * sim.red_nx + 2] = (5.0, 5.0)
    n = n_in = 0
    for x in np.arange(poly[:, 0].min() - 25, poly[:, 0].max() + 6, 0.7):
        for y in np.arange(poly[:, 1].min() - 25, poly[:, 1].max() + 6, 0.7):
            p = np.array([x, y])
            if not (0.5 < x < sim.arena_size - 0.5 and 0.5 < y < sim.arena_size - 0.5):
                continue
            if _edge_distance(p, poly) < 0.05 or min(np.linalg.norm(p - b) for b in blues) < 1.0:
                continue
            sim.touchdown, sim.touchdown_step, sim.touchdown_red_index = False, None, None
            sim.red_outcomes[0] = "active"
            sim.red_state[0: 2] = p
            ended = sim._resolve_active_reds(1)
            expected = bool(ref.contains_point(p))
            assert ended == expected, (p.tolist(), ended, expected)
            n += 1
            n_in += expected
    assert n > 1000 and n_in > 100
    print(f"PASS: call site: {n} red positions around the defended region, "
          f"episode ends exactly where the reference says inside ({n_in} of them)")


if __name__ == "__main__":
    shape = np.asarray(_sim().defended_shape_points, float)
    polygons = {
        "triangle": TRIANGLE,
        "oblique_quad": [(1.0, 2.0), (9.0, 0.5), (11.0, 8.0), (3.0, 11.0)],
        "concave_L": [(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6)],
        "concave_star": _star(),
        "defended_shape": [tuple(p) for p in shape],
    }
    check_original_defect()
    check_reference(polygons)
    check_boundary(polygons)
    check_call_site()
    print("ALL CHECKS PASSED")
