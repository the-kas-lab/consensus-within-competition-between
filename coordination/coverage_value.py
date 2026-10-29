"""Coverage, information and threat-timing terms for bid version "v2".

Added. Opt-in: nothing here is used unless
a run selects ``cbaa_bid_version="v2"`` or ``station_heading="configured"``.

* **Approach prior** (USER-APPROVED): bearing from the defended centre in
  ``PRIOR_SECTOR_DEG`` carries weight 1, elsewhere 0. No radial assumption:
  density is uniform over arena AREA outside the defended polygon, sampled on
  a ``LATTICE``-unit point lattice. Static scenario knowledge.
* **Coverage** uses the simulator's own wedge (origin, half-angle, range).
  ``mass`` is prior-weighted lattice area; ``W_FULL`` (one full wedge's area)
  normalises it, so 1.0 = "one sensor's worth of fully relevant ground".
* **Information proxy** for RECON: the prior-weighted, staleness-weighted area
  swept by the wedges the directed RECON controller points along its route.
  Staleness comes from the agent's own coverage record, a CELL-CENTRE
  visibility sample used as a proxy for the whole cell. Not calibrated
  expected information gain.
* **Threat timing**: constant-velocity time-to-breach against the actual
  defended polygon and an interception time at a guaranteed blue speed --
  forecasts from beliefs, never known arrival times.
"""

import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

PRIOR_SECTOR_DEG = (160.0, 300.0)
LATTICE = 1.0
SWEEP_STEP = 5.0
# Each velocity component is bounded by v_max = 2 (control/base.py), so 2.0 is
# the speed guaranteed in EVERY direction; Euclidean speed can reach 2*sqrt(2).
GUARANTEED_SPEED = 2.0


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _bearing_in_sector(theta, sector_deg):
    lo, hi = (math.radians(x) for x in sector_deg)
    t = np.mod(theta, 2.0 * np.pi)
    if lo <= hi:
        return (t >= lo) & (t <= hi)
    return (t >= lo) | (t <= hi)


def _points_in_polygon(points, polygon):
    from matplotlib.path import Path

    return Path(np.asarray(polygon, dtype=float)).contains_points(points)


class ApproachField:
    """The prior-weighted lattice for one scenario geometry (static)."""

    def __init__(self, center, polygon, arena_min, arena_max, half_angle_rad, fov_range,
                 sector_deg=PRIOR_SECTOR_DEG, lattice=LATTICE):
        self.center = np.asarray(center, dtype=float).reshape(2)
        self.polygon = np.asarray(polygon, dtype=float).reshape(-1, 2)
        self.half_angle = float(half_angle_rad)
        self.range = float(fov_range)
        self.sector_deg = tuple(float(x) for x in sector_deg)
        self.lattice = float(lattice)
        g = np.arange(float(arena_min) + lattice / 2.0, float(arena_max), lattice)
        xx, yy = np.meshgrid(g, g, indexing="ij")
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        rel = pts - self.center
        in_sector = _bearing_in_sector(np.arctan2(rel[:, 1], rel[:, 0]), self.sector_deg)
        outside = ~_points_in_polygon(pts, self.polygon) if len(self.polygon) >= 3 else np.ones(len(pts), bool)
        keep = in_sector & outside
        # Only weighted points can carry mass, so only they are kept.
        self.points = pts[keep]
        self.cell_area = lattice * lattice
        self.w_full = self.half_angle * self.range ** 2  # area of one wedge
        self._cell_index = {}

    # -- geometry ------------------------------------------------------------

    def wedge(self, origin, heading) -> np.ndarray:
        d = self.points - np.asarray(origin, dtype=float).reshape(2)
        dist = np.hypot(d[:, 0], d[:, 1])
        ang = _wrap(np.arctan2(d[:, 1], d[:, 0]) - float(heading))
        return (dist <= self.range) & (np.abs(ang) <= self.half_angle)

    def union(self, poses: Iterable[Tuple[np.ndarray, float]]) -> np.ndarray:
        m = np.zeros(len(self.points), dtype=bool)
        for origin, heading in poses:
            m |= self.wedge(origin, heading)
        return m

    def mass(self, mask, weights=None) -> float:
        """Normalised mass of ``mask`` (in units of one full wedge)."""
        if weights is None:
            return float(np.count_nonzero(mask)) * self.cell_area / self.w_full
        return float(np.sum(weights[mask])) * self.cell_area / self.w_full

    def total_mass(self) -> float:
        return len(self.points) * self.cell_area / self.w_full

    # -- cells -----------------------------------------------------------------

    def cell_index(self, centers) -> np.ndarray:
        """Index of the coverage cell (nearest centre) holding each point."""
        c = np.asarray(centers, dtype=float).reshape(-1, 2)
        key = c.tobytes()
        idx = self._cell_index.get(key)
        if idx is None:
            d2 = ((self.points[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
            idx = self._cell_index[key] = np.argmin(d2, axis=1)
        return idx

    # -- terms -------------------------------------------------------------------

    def marginal(self, pose, others_mask) -> float:
        """Coverage a pose adds beyond ``others_mask``."""
        return self.mass(self.wedge(*pose) & ~others_mask)

    def swept(self, start, dest, max_dist, fallback_heading) -> np.ndarray:
        """Route distance at which each point is FIRST inside the wedge a
        directed RECON agent points along the straight route start -> dest
        (heading = travel direction, the controller's bearing-to-waypoint),
        sampled every SWEEP_STEP and truncated at ``max_dist``; inf where never.
        At the destination itself the agent keeps ``fallback_heading``."""
        start = np.asarray(start, dtype=float).reshape(2)
        dest = np.asarray(dest, dtype=float).reshape(2)
        first = np.full(len(self.points), np.inf)
        u = dest - start
        length = float(np.hypot(*u))
        if length < 1e-6:
            first[self.wedge(start, fallback_heading)] = 0.0
            return first
        heading = math.atan2(u[1], u[0])
        reach = max(0.0, min(length, float(max_dist)))
        for x in list(np.arange(0.0, reach, SWEEP_STEP)) + [reach]:
            m = self.wedge(start + (x / length) * u, heading) & ~np.isfinite(first)
            first[m] = x
        return first

    def information(self, first_swept, watched_mask, staleness_by_cell, centers,
                    discount_dist) -> float:
        """Prior x staleness x timeliness mass of the swept points not already
        watched: a point first swept after travelling x counts
        exp(-x / discount_dist) (information that arrives later is worth less).
        """
        stale = np.asarray(staleness_by_cell, dtype=float)[self.cell_index(centers)]
        ok = np.isfinite(first_swept) & ~watched_mask
        w = np.zeros(len(self.points))
        w[ok] = stale[ok] * np.exp(-first_swept[ok] / float(discount_dist))
        return float(np.sum(w)) * self.cell_area / self.w_full


_FIELDS: Dict[tuple, ApproachField] = {}


def approach_field(center, polygon, arena_min, arena_max, half_angle_rad, fov_range,
                   sector_deg=PRIOR_SECTOR_DEG) -> ApproachField:
    """Cached per geometry (static for a scenario)."""
    key = (tuple(np.round(np.asarray(center, float), 9)),
           tuple(map(tuple, np.round(np.asarray(polygon, float), 9))),
           float(arena_min), float(arena_max), round(float(half_angle_rad), 12),
           float(fov_range), tuple(sector_deg))
    f = _FIELDS.get(key)
    if f is None:
        f = _FIELDS[key] = ApproachField(center, polygon, arena_min, arena_max, half_angle_rad,
                                         fov_range, sector_deg)
    return f


def configured_station_headings(field: ApproachField, stations: Sequence[Tuple[int, Sequence[float]]],
                                step_deg: float = 5.0, sweeps: int = 10) -> Dict[int, float]:
    """Static station headings maximising the prior-weighted UNION coverage of
    all stations' wedges, each from its own station origin. Deterministic
    coordinate ascent from the outward headings over a ``step_deg`` grid; ties
    -> the smaller change from outward."""
    ids = [int(k) for k, _ in stations]
    pos = {int(k): np.asarray(p, dtype=float).reshape(2) for k, p in stations}
    outward = {k: math.atan2(*(pos[k] - field.center)[::-1]) for k in ids}
    grid = [math.radians(d) for d in np.arange(0.0, 360.0, step_deg)]
    head = dict(outward)
    for _ in range(sweeps):
        changed = False
        for k in ids:
            others = field.union((pos[j], head[j]) for j in ids if j != k)
            best, best_val, best_dev = head[k], None, None
            for h in grid + [outward[k]]:
                val = round(field.mass(field.wedge(pos[k], h) | others), 9)
                dev = abs(float(_wrap(h - outward[k])))
                if best_val is None or val > best_val or (val == best_val and dev < best_dev - 1e-12):
                    best, best_val, best_dev = h, val, dev
            if abs(float(_wrap(best - head[k]))) > 1e-9:
                head[k], changed = best, True
        if not changed:
            break
    return {k: float(np.mod(head[k], 2.0 * np.pi)) for k in ids}


# ---------------------------------------------------------------------------
# threat timing
# ---------------------------------------------------------------------------

STATIONARY_SPEED = 0.05


def time_to_breach(position, velocity, polygon) -> Tuple[float, str]:
    """Constant-velocity forecast: (tau seconds, category). Categories:
    "inside" (tau 0), "breach" (finite), "stationary", "non_intersecting"
    (incl. receding) -- the last two with tau = inf."""
    p = np.asarray(position, dtype=float).reshape(2)
    v = np.asarray(velocity, dtype=float).reshape(2)
    poly = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if len(poly) >= 3 and _points_in_polygon(p.reshape(1, 2), poly)[0]:
        return 0.0, "inside"
    if float(np.hypot(*v)) < STATIONARY_SPEED:
        return math.inf, "stationary"
    best = math.inf
    closed = np.vstack([poly, poly[0]])
    for k in range(len(poly)):
        a, b = closed[k], closed[k + 1]
        e = b - a
        den = v[0] * (-e[1]) - v[1] * (-e[0])
        if abs(den) < 1e-12:
            continue
        rhs = a - p
        t = (rhs[0] * (-e[1]) - rhs[1] * (-e[0])) / den
        s = (v[0] * rhs[1] - v[1] * rhs[0]) / den
        if t >= 0.0 and -1e-9 <= s <= 1.0 + 1e-9:
            best = min(best, float(t))
    return (best, "breach") if math.isfinite(best) else (math.inf, "non_intersecting")


def intercept_time(position, velocity, blue_position, speed=GUARANTEED_SPEED) -> Optional[float]:
    """Earliest t >= 0 with |p + v t - b| = speed * t, or None."""
    p = np.asarray(position, dtype=float).reshape(2)
    v = np.asarray(velocity, dtype=float).reshape(2)
    r = p - np.asarray(blue_position, dtype=float).reshape(2)
    a = float(v @ v) - speed ** 2
    b = 2.0 * float(r @ v)
    c = float(r @ r)
    if c < 1e-12:
        return 0.0
    if abs(a) < 1e-12:
        if b >= 0.0:
            return None
        return -c / b
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    roots = sorted(t for t in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)) if t >= 0.0)
    return roots[0] if roots else None


# ---------------------------------------------------------------------------
# bid version "v2.2":
# gate-aware breach forecast against the approach funnel, and optimistic
# kinematic reachability (interception, RECON travel) from the per-axis
# bounds of the discrete integrator. Forecasts from beliefs; none of them
# accounts for the safety filter, the controllers' actual paths, or the
# target's reaction.
# ---------------------------------------------------------------------------

APPROACHING = "approaching"
MOVING_AWAY = "moving_away"
CROSSING = "crossing"
STATIONARY = "stationary"
INSIDE = "inside"
_DISK_SAMPLES = 32


def convex_hull(points) -> np.ndarray:
    """Counter-clockwise convex hull (Andrew's monotone chain)."""
    pts = sorted(set((float(x), float(y)) for x, y in np.asarray(points, float).reshape(-1, 2)))
    if len(pts) <= 2:
        return np.asarray(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def approach_funnel(polygon, gate_positions, arrival_radius) -> np.ndarray:
    """Convex hull of the defended polygon and the public gates' arrival
    disks (static public geometry; blues are not told which gate a red
    chose)."""
    pts = [np.asarray(polygon, float).reshape(-1, 2)]
    ang = np.linspace(0.0, 2.0 * np.pi, _DISK_SAMPLES, endpoint=False)
    ring = np.column_stack([np.cos(ang), np.sin(ang)]) * float(arrival_radius)
    for g in gate_positions:
        pts.append(np.asarray(g, float).reshape(1, 2) + ring)
    return convex_hull(np.vstack(pts))


def _nearest_on_boundary(point, polygon):
    from coordination.tasks import closest_point_on_polygon

    q = closest_point_on_polygon(point, polygon)
    return q, float(np.linalg.norm(np.asarray(point, float) - q))


def funnel_forecast(position, velocity, funnel, polygon, red_speed_euclid,
                    stationary_speed=STATIONARY_SPEED) -> Dict[str, float]:
    """Gate-aware breach forecast.

    Outside the funnel: closing speed c = v . n (n: unit vector to the
    nearest funnel point e); course time tau = d_funnel / c + d(e, polygon) /
    red_speed_euclid when c > stationary_speed, else inf. Inside the funnel
    (past the gates) the polygon replaces the funnel. Continuous in v.
    """
    p = np.asarray(position, dtype=float).reshape(2)
    v = np.asarray(velocity, dtype=float).reshape(2)
    poly = np.asarray(polygon, dtype=float).reshape(-1, 2)
    out = {"time_to_breach": math.inf, "closing_speed": 0.0, "distance": 0.0, "status": STATIONARY}
    if _points_in_polygon(p.reshape(1, 2), poly)[0]:
        out.update(time_to_breach=0.0, status=INSIDE)
        return out
    in_funnel = bool(_points_in_polygon(p.reshape(1, 2), funnel)[0])
    target = poly if in_funnel else funnel
    e, d = _nearest_on_boundary(p, target)
    n = (e - p) / d if d > 1e-9 else np.zeros(2)
    c = float(v @ n)
    rest = 0.0 if in_funnel else _nearest_on_boundary(e, poly)[1] / float(red_speed_euclid)
    out.update(closing_speed=c, distance=d)
    if float(np.hypot(*v)) < stationary_speed:
        return out
    if c > stationary_speed:
        out.update(time_to_breach=d / c + rest, status=APPROACHING)
    elif c < -stationary_speed:
        out["status"] = MOVING_AWAY
    else:
        out["status"] = CROSSING
    return out


def axis_reach(p0, v0, v_max, a_max, dt, k_max) -> Tuple[np.ndarray, np.ndarray]:
    """Exact reachable interval of one position axis after k = 0..k_max steps
    of v_{k+1} = v_k + dt a_k, p_{k+1} = p_k + dt v_{k+1}, |a| <= a_max,
    |v| <= v_max (the simulator's and the MPC's integrator; v0 clamped to the
    bound). Extreme acceleration maximises every later velocity at once, so
    the bounds are attained."""
    v0 = float(np.clip(v0, -v_max, v_max))
    j = np.arange(1, int(k_max) + 1, dtype=float)
    hi = np.minimum(v0 + j * dt * a_max, v_max)
    lo = np.maximum(v0 - j * dt * a_max, -v_max)
    p0 = float(p0)
    return (np.concatenate([[p0], p0 + dt * np.cumsum(lo)]),
            np.concatenate([[p0], p0 + dt * np.cumsum(hi)]))


def reach_intercept(blue_pos, blue_vel, target_pos, target_vel, v_max, a_max, dt, k_max,
                    radius) -> Optional[Tuple[float, np.ndarray]]:
    """OPTIMISTIC KINEMATIC INTERCEPTION FORECAST: the
    earliest t = k dt, k in [0, k_max], at which the constant-velocity target
    forecast q_k comes within ``radius`` of the blue's exact per-axis
    reachable rectangle. Returns (t, q_k) or None. Not a guarantee that the
    pursuit controller intercepts: it ignores the safety filter, the
    controller's path and the target's reaction."""
    b = np.asarray(blue_pos, float).reshape(2)
    vb = np.asarray(blue_vel, float).reshape(2)
    q = np.asarray(target_pos, float).reshape(2)
    vq = np.asarray(target_vel, float).reshape(2)
    k_max = max(0, int(k_max))
    k = np.arange(k_max + 1, dtype=float)
    qx, qy = q[0] + vq[0] * k * dt, q[1] + vq[1] * k * dt
    lox, hix = axis_reach(b[0], vb[0], v_max, a_max, dt, k_max)
    loy, hiy = axis_reach(b[1], vb[1], v_max, a_max, dt, k_max)
    dx = np.maximum(lox - qx, 0.0) + np.maximum(qx - hix, 0.0)
    dy = np.maximum(loy - qy, 0.0) + np.maximum(qy - hiy, 0.0)
    ok = np.flatnonzero(np.hypot(dx, dy) <= radius)
    if ok.size == 0:
        return None
    i = int(ok[0])
    return float(i * dt), np.array([qx[i], qy[i]])


def reach_intercept_times(blue_pos, blue_vel, target_pos, target_vels, v_max, a_max, dt, k_max,
                          radius) -> np.ndarray:
    """``reach_intercept`` for many constant target velocities at once
    (rows of ``target_vels``): the forecast interception time per row, inf
    where there is none within k_max steps."""
    b = np.asarray(blue_pos, float).reshape(2)
    vb = np.asarray(blue_vel, float).reshape(2)
    q = np.asarray(target_pos, float).reshape(2)
    vq = np.asarray(target_vels, float).reshape(-1, 2)
    k_max = max(0, int(k_max))
    k = np.arange(k_max + 1, dtype=float)[None, :]
    qx = q[0] + vq[:, :1] * k * dt
    qy = q[1] + vq[:, 1:] * k * dt
    lox, hix = axis_reach(b[0], vb[0], v_max, a_max, dt, k_max)
    loy, hiy = axis_reach(b[1], vb[1], v_max, a_max, dt, k_max)
    dx = np.maximum(lox[None, :] - qx, 0.0) + np.maximum(qx - hix[None, :], 0.0)
    dy = np.maximum(loy[None, :] - qy, 0.0) + np.maximum(qy - hiy[None, :], 0.0)
    ok = np.hypot(dx, dy) <= radius
    first = np.argmax(ok, axis=1).astype(float) * dt
    return np.where(ok.any(axis=1), first, np.inf)


# v2.2 revision 1: the believed velocity is an estimate
# and the red re-plans, so opportunity is the MEAN over a declared box of
# velocity uncertainty around it (per axis, a 7 x 7 grid, +-0.3 u/s, clamped
# to the red's per-axis bound). Smooths the forecast's jumps when an
# interception window opens or closes (tail chases, near-misses).
VELOCITY_UNCERTAINTY = 0.3
VELOCITY_SAMPLES = 7


def velocity_box(velocity, red_v_max, half_width=VELOCITY_UNCERTAINTY, n=VELOCITY_SAMPLES):
    v = np.asarray(velocity, float).reshape(2)
    g = np.linspace(-half_width, half_width, n)
    vx, vy = np.meshgrid(v[0] + g, v[1] + g, indexing="ij")
    return np.clip(np.column_stack([vx.ravel(), vy.ravel()]), -red_v_max, red_v_max)


def route_reach(velocity, direction, v_max, a_max, horizon_s, dt) -> float:
    """OPTIMISTIC TRAVEL FORECAST along a straight route: the farthest
    distance reachable along unit ``direction`` within ``horizon_s`` while
    staying on the route, from the along-route component of the current
    velocity (cross-route motion ignored), with the per-axis bounds expressed
    along the route (speed v_max / max|u_i|, acceleration a_max / max|u_i|)."""
    u = np.asarray(direction, float).reshape(2)
    s = float(max(abs(u[0]), abs(u[1])))
    if s < 1e-12 or horizon_s <= 0.0:
        return 0.0
    smax, acc = v_max / s, a_max / s
    w0 = float(np.clip(np.asarray(velocity, float).reshape(2) @ u, -smax, smax))
    j = np.arange(1, int(round(horizon_s / dt)) + 1, dtype=float)
    return float(max(0.0, dt * np.sum(np.minimum(w0 + j * dt * acc, smax))))


def opportunity_v22(t_i, tau_ref, t_rem, t1, o_min) -> float:
    """Monotone, continuous opportunity: o_min + (1 -
    o_min) g1 g2. g1 = 1 until tau_ref, then linear to 0 over t1. g2 = 1
    within the remaining episode t_rem, then linear to 0 over t1 beyond it
    (the interception search runs to t_rem + t1, so O reaches o_min exactly
    where the forecast stops finding one). o_min when there is no
    interception forecast. Non-increasing in t_i; never below O(None)."""
    if t_i is None:
        return float(o_min)
    g1 = 1.0 if not math.isfinite(tau_ref) else float(np.clip(1.0 - (t_i - tau_ref) / t1, 0.0, 1.0))
    g2 = float(np.clip(1.0 - (t_i - t_rem) / t1, 0.0, 1.0))
    return float(o_min + (1.0 - o_min) * g1 * g2)
