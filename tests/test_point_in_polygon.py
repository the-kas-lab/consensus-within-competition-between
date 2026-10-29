"""The breach predicate: is a point in the defended polygon? the breach-test rule.

``Simulation._point_in_polygon`` decides breach, the primary outcome of every
experiment. Until it clamped its signed ray-crossing denominator
with ``max(y_prev - y_curr, 1e-9)``. Every edge whose y increases then counted
as a crossing always or never, whatever the point's x. Exterior points read as
inside (false breaches ended episodes) and interior points as outside.

Pinned here:

1. The original defect's two errors, on a plain triangle: an exterior point
   and an interior point.
2. Agreement with an INDEPENDENT reference (matplotlib's ``Path``) on points
   away from the boundary, over oblique edges, both vertex orders, concave
   polygons, and the actual defended shapes of both scenario families.
3. The boundary, decided explicitly and tested separately: the region is
   CLOSED, so vertices and edge points are inside, and points just outside
   are not.
4. The call site: on a live Simulation, an exterior red does not end the
   episode and an interior one does. The recorded Exp 5 false breach (seed
   3905, `multi_bearing`, REF/I, red 0 at (58.76, 83.35), 23 units from the
   centre) no longer terminates the episode at step 293.

"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib.path import Path  # noqa: E402

from simulation.testbed import Simulation  # noqa: E402

inside = Simulation._point_in_polygon

TRIANGLE = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]  # counter-clockwise, oblique edges


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
    pts = []
    for k in range(2 * n):
        r = r_out if k % 2 == 0 else r_in
        a = phase + np.pi * k / n
        pts.append((center[0] + r * np.cos(a), center[1] + r * np.sin(a)))
    return pts


def _defended_shapes():
    from scripts.episode_builder import build_episode
    from scripts.network_conditions import episode_cfg

    shapes = {}
    for family in ("symmetric", "multi_bearing"):
        for seed in (3905, 3950):
            belief, cfg = episode_cfg(family, "REF", "I")
            shapes[(family, seed)] = np.asarray(
                build_episode(seed, belief, cfg).defended_shape_points, float)
    return shapes


POLYGONS = {
    "triangle_ccw": TRIANGLE,
    "triangle_cw": TRIANGLE[::-1],
    "oblique_quad": [(1.0, 2.0), (9.0, 0.5), (11.0, 8.0), (3.0, 11.0)],
    "concave_L": [(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6)],
    "concave_star": _star(),
    "concave_star_cw": _star()[::-1],
}


# -- 1. the original defect ---------------------------------------------------------


def test_the_original_defect_is_gone_on_a_triangle():
    # With the clamp, the edge (10,0)->(5,10) (y increasing, x decreasing)
    # never counted: an exterior point to the left read INSIDE, and an interior
    # point read OUTSIDE.
    assert inside((-5.0, 3.0), TRIANGLE) is False
    assert inside((5.0, 3.0), TRIANGLE) is True


# -- 2. an independent reference, away from the boundary ---------------------------


@pytest.mark.parametrize("name", sorted(POLYGONS))
def test_agrees_with_an_independent_reference(name):
    poly = POLYGONS[name]
    ref = Path(np.asarray(poly, float))
    lo, hi = np.min(poly, axis=0) - 5, np.max(poly, axis=0) + 5
    rng = np.random.default_rng(20260928)
    checked = n_in = 0
    for p in rng.uniform(lo, hi, size=(4000, 2)):
        if _edge_distance(p, poly) < 1e-6:
            continue
        expected = bool(ref.contains_point(p))
        assert inside(p, poly) is expected, (name, p.tolist())
        checked += 1
        n_in += expected
    assert checked > 3500 and 100 < n_in < checked - 100, "both sides must be exercised"


def test_vertex_order_does_not_matter():
    rng = np.random.default_rng(1)
    for poly in POLYGONS.values():
        for p in rng.uniform(-5, 65, size=(500, 2)):
            assert inside(p, poly) == inside(p, poly[::-1])


def test_agrees_on_the_actual_defended_shapes():
    rng = np.random.default_rng(3905)
    for key, poly in _defended_shapes().items():
        ref = Path(poly)
        pts = np.vstack([rng.uniform(0, 100, size=(3000, 2)),          # whole arena
                         rng.uniform(poly.min(0) - 3, poly.max(0) + 3, size=(3000, 2))])
        n_in = 0
        for p in pts:
            if _edge_distance(p, poly) < 1e-6:
                continue
            expected = bool(ref.contains_point(p))
            assert inside(p, poly) is expected, (key, p.tolist())
            n_in += expected
        assert n_in > 300, key
        # The recorded false breach, and the band it came from, are outside.
        assert inside((58.76, 83.35), poly) is False
        assert not any(inside((x, 85.0), poly) for x in np.arange(0.0, 70.0, 0.5))


# -- 3. the boundary, decided explicitly -----------------------------------------------


@pytest.mark.parametrize("name", sorted(POLYGONS))
def test_the_region_is_closed(name):
    v = np.asarray(POLYGONS[name], float)
    for a, b in zip(np.roll(v, 1, axis=0), v):
        assert inside(tuple(a), v)                          # vertex
        assert inside(tuple((a + b) / 2), v)                # edge midpoint
        assert inside(tuple(a + 0.3 * (b - a)), v)          # oblique edge point


def test_just_outside_the_boundary_is_outside():
    # Outward normal of each counter-clockwise triangle edge, 1e-6 away.
    v = np.asarray(TRIANGLE, float)
    for a, b in zip(np.roll(v, 1, axis=0), v):
        d = b - a
        normal = np.array([d[1], -d[0]]) / np.linalg.norm(d)
        mid = (a + b) / 2
        assert inside(tuple(mid + 1e-6 * normal), v) is False
        assert inside(tuple(mid - 1e-6 * normal), v) is True


# -- 4. the call site, on a live Simulation -----------------------------------------------


def _episode(steps):
    from scripts.episode_builder import _Quiet, build_episode
    from scripts.network_conditions import episode_cfg

    belief, cfg = episode_cfg("multi_bearing", "REF", "I")
    cfg = dict(cfg, steps=steps)
    sim = build_episode(3905, belief, cfg)
    with _Quiet():
        sim.run()
    return sim


def _place_red(sim, j, xy):
    sim.red_state[j * sim.red_nx: j * sim.red_nx + 2] = np.asarray(xy, float)


def test_an_exterior_red_does_not_end_the_episode_and_an_interior_one_does():
    sim = _episode(2)
    far = (5.0, 5.0)
    _place_red(sim, 1, far)
    _place_red(sim, 0, (58.76, 83.35))  # the recorded false breach
    assert sim._resolve_active_reds(10) is False and not sim.touchdown
    center = np.asarray(sim.defense_center, float).ravel()[:2]
    _place_red(sim, 0, center + np.array([0.0, 2.0]))
    assert sim._resolve_active_reds(11) is True and sim.touchdown
    assert sim.touchdown_red_index == 0


def test_the_breach_check_matches_the_reference_over_the_whole_neighbourhood():
    """At the call site, not only at the method: every grid point around the
    defended region, placed as red 0 and resolved by ``_resolve_active_reds``,
    ends the episode exactly when the reference says it is inside. The shape
    is nearly symmetric about y = x, so a handful of points cannot tell a
    swapped call from a correct one; the sweep can."""
    sim = _episode(2)
    poly = np.asarray(sim.defended_shape_points, float)
    ref = Path(poly)
    xb = np.asarray(sim.x_current, float).ravel()
    blues = [xb[i * sim.blue_nx: i * sim.blue_nx + 2] for i in range(sim.num_blues)]
    _place_red(sim, 1, (5.0, 5.0))
    checked = mismatched = n_in = 0
    for x in np.arange(poly[:, 0].min() - 6, poly[:, 0].max() + 6, 0.7):
        for y in np.arange(poly[:, 1].min() - 6, poly[:, 1].max() + 6, 0.7):
            p = np.array([x, y])
            if _edge_distance(p, poly) < 0.05 or min(np.linalg.norm(p - b) for b in blues) < 1.0:
                continue
            sim.touchdown, sim.touchdown_step, sim.touchdown_red_index = False, None, None
            sim.red_outcomes[0] = "active"
            _place_red(sim, 0, p)
            ended = sim._resolve_active_reds(12)
            expected = bool(ref.contains_point(p))
            mismatched += ended != expected
            n_in += expected
            checked += 1
    assert checked > 500 and n_in > 100
    assert mismatched == 0


def test_the_recorded_false_breach_no_longer_terminates_the_episode():
    """Exp 5, seed 3905, `multi_bearing`, REF/I ended at step 293 on red 0 at
    (58.76, 83.35), 23 units from the centre, never inside. Corrected, it runs
    past 293, and any touchdown it does record is truly inside."""
    sim = _episode(400)
    assert len(sim.recorder.steps) > 294
    if sim.touchdown:
        j = sim.touchdown_red_index
        p = np.asarray(sim.red_state, float)[j * sim.red_nx: j * sim.red_nx + 2]
        assert Path(np.asarray(sim.defended_shape_points, float)).contains_point(p) or \
            _edge_distance(p, sim.defended_shape_points) < 1e-6
