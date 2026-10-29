"""Numerical tests of the discrete-time control barrier function definition under the simulator's limits.

Pure numpy except where the simulator's own safety filter is called; no episode runs, no model is called.

    python3 -m analysis.barrier_tests --out OUT       ->  OUT/barrier_tests.json

Definition tested (Cheng et al. 2020, Def. 1; Agrawal and Sreenath 2017, Def. 3): h is a discrete-time control
barrier function if there is a gamma in (0, 1] such that for every state x with h(x) >= 0 some admissible input
u gives h(x+) >= (1 - gamma) h(x).

Plant (agents/dynamics.py, semi-implicit Euler, dt = 0.1): v+ = v + dt a, p+ = p + dt v+. Admissible per agent:
|a_x|, |a_y| <= 2 and |v+_x|, |v+_y| <= 2 (per-axis boxes).

A. Exact one-step test for h = |r|^2 - D^2 on a head-on pair (both agents act, or one acts against an immobile
   neighbour). h+ is a convex quadratic in the accelerations, so its maximum over the box input set is attained
   at a vertex: checking the vertices is exact, not a grid approximation.
B. The head-on counterexample (gap 3.1, closing 4) through the simulator's own filter_agent.
C. Random two-agent states, both agents acting: the fraction of states with h >= 0 for which no grid input
   satisfies the condition, for |r|^2 - D^2 and for two braking-aware barriers (Wang, Ames and Egerstedt eq. 8;
   Cheng et al. 2020 eq. 5). For the braking barriers the grid is an approximation (h+ is not convex in u), so a
   reported failure is real but a pass is "no failure found on this grid".
D. Closed loop with the simulator's filter_agent: two defenders head-on from 20 units at full closing speed,
   nominal input full thrust at each other, for several gamma. Records the trajectory.
E. The braking barriers again, only for states right at their boundary.
"""
import argparse
import json
import os

import numpy as np

DT, A, V, D = 0.1, 2.0, 2.0, 3.0


def axis_limits(v):
    return np.maximum(-A, (-V - v) / DT), np.minimum(A, (V - v) / DT)


# A. exact head-on map
def best_next_h_headon(gap, w, mode):
    """Head-on: i at (0,0) moving +x at w/2, j at (gap,0) moving -x at w/2 (mode 'coop').
    mode 'one': i moves at w, j is held at constant velocity 0 and does not act.
    Returns max over admissible inputs of |r+|^2 - D^2 (exact: vertex enumeration)."""
    if mode == "coop":
        vi, vj = np.array([w / 2, 0.0]), np.array([-w / 2, 0.0])
        actors = [(vi, 1.0), (vj, -1.0)]
    else:
        vi, vj = np.array([w, 0.0]), np.array([0.0, 0.0])
        actors = [(vi, 1.0)]
    r = np.array([-gap, 0.0])            # r = p_i - p_j
    c = r + DT * (vi - vj)
    # each actor contributes sign * a to delta-u; vertices of each actor's box
    vert = []
    for v, sgn in actors:
        lo, hi = axis_limits(v)
        vert.append([sgn * np.array([x, y]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
    best = -np.inf
    if len(vert) == 1:
        for du in vert[0]:
            rp = c + DT ** 2 * du
            best = max(best, rp @ rp - D ** 2)
    else:
        for a in vert[0]:
            for b in vert[1]:
                rp = c + DT ** 2 * (a + b)
                best = max(best, rp @ rp - D ** 2)
    return best


def headon_map():
    gaps = np.round(np.arange(3.02, 20.0001, 0.02), 3)
    speeds = np.round(np.arange(0.0, 4.0001, 0.05), 3)
    out = {}
    for mode, wmax in (("coop", 4.0), ("one", 2.0)):
        sp = speeds[speeds <= wmax + 1e-9]
        best = [[round(float(best_next_h_headon(g, w, mode)), 5) for g in gaps] for w in sp]
        out[mode] = {"gaps": gaps.tolist(), "speeds": sp.tolist(), "best_h_next": best}
    return out


# B. the head-on counterexample
def counterexample():
    from control.local_cbf import COOPERATIVE, LocalCBFConfig, Neighbor, filter_agent
    cfg = LocalCBFConfig(gamma=0.05, dt=DT, a_max=A, v_max=V)
    p0, p1 = np.array([0.0, 0.0]), np.array([3.1, 0.0])
    v0, v1 = np.array([2.0, 0.0]), np.array([-2.0, 0.0])
    r0 = filter_agent(p0, v0, np.array([2.0, 0.0]), [Neighbor(1, p1, v1, D, COOPERATIVE)], cfg)
    r1 = filter_agent(p1, v1, np.array([-2.0, 0.0]), [Neighbor(0, p0, v0, D, COOPERATIVE)], cfg)
    h = 3.1 ** 2 - D ** 2
    return {"gap": 3.1, "closing": 4.0, "h": round(h, 4), "required": round(0.95 * h, 4),
            "best_h_next_exact": round(float(best_next_h_headon(3.1, 4.0, "coop")), 4),
            "filter_paths": [r0.path, r1.path], "filter_reasons": [r0.reason, r1.reason]}


# C. random states, three barriers
def h_dist(r, w):
    return np.einsum("...i,...i->...", r, r) - D ** 2


def h_wang(r, w, a_rel=4.0):
    n = np.linalg.norm(r, axis=-1)
    return np.sqrt(np.maximum(2 * a_rel * (n - D), 0.0)) + np.einsum("...i,...i->...", r, w) / n


def h_cheng(r, w, a_rel=4.0):
    n = np.linalg.norm(r, axis=-1)
    return np.sqrt(np.maximum(a_rel * (n - D), 0.0)) + np.einsum("...i,...i->...", r, w) / n


def random_states(n, rng):
    gap = rng.uniform(D, D + 8.0, n)
    ang = rng.uniform(0, 2 * np.pi, n)
    r = np.stack([gap * np.cos(ang), gap * np.sin(ang)], 1)     # r = p_i - p_j
    vi = rng.uniform(-V, V, (n, 2))
    vj = rng.uniform(-V, V, (n, 2))
    return r, vi, vj


def grid_inputs(v, k):
    lo, hi = axis_limits(v)
    t = np.linspace(0, 1, k)
    ax = lo[:, 0:1] + (hi[:, 0:1] - lo[:, 0:1]) * t
    ay = lo[:, 1:2] + (hi[:, 1:2] - lo[:, 1:2]) * t
    return np.stack(np.broadcast_arrays(ax[:, :, None], ay[:, None, :]), -1).reshape(len(v), k * k, 2)


def random_test(n=20000, k=9, seed=7):
    rng = np.random.default_rng(seed)
    r, vi, vj = random_states(n, rng)
    ui, uj = grid_inputs(vi, k), grid_inputs(vj, k)               # (n, k2, 2)
    res = {}
    for name, h in (("dist_sq", h_dist), ("wang_braking", h_wang), ("cheng_braking", h_cheng)):
        h0 = h(r, vi - vj)
        safe = h0 >= 0
        rows = {}
        for gamma in (0.05, 0.2, 1.0):
            fail = 0
            worst = 0.0
            for s in range(0, n, 500):
                sl = slice(s, s + 500)
                vi1 = vi[sl, None, None, :] + DT * ui[sl, :, None, :]
                vj1 = vj[sl, None, None, :] + DT * uj[sl, None, :, :]
                r1 = r[sl, None, None, :] + DT * (vi1 - vj1)
                h1 = h(r1, vi1 - vj1).reshape(len(r[sl]), -1)
                need = (1 - gamma) * h0[sl]
                best = h1.max(1)
                bad = safe[sl] & (best < need - 1e-9)
                fail += int(bad.sum())
                if bad.any():
                    worst = max(worst, float((need - best)[bad].max()))
            rows[str(gamma)] = {"no_input_found": fail, "worst_shortfall": round(worst, 4)}
        res[name] = {"safe_states": int(safe.sum()), "by_gamma": rows}
    return {"samples": n, "grid_per_agent": f"{k}x{k}", "gap_range": [D, D + 8.0], "results": res}


def boundary_test(n=20000, k=15, seed=11):
    """E. The braking barriers again, but only states right at their boundary
    (0 <= h <= 0.3, gap < 5.5), where a failure would be most likely."""
    rng = np.random.default_rng(seed)
    res = {}
    for name, h in (("wang_braking", h_wang), ("cheng_braking", h_cheng)):
        keep, have = [], 0
        while have < n:
            r, vi, vj = random_states(400000, rng)
            m = np.linalg.norm(r, axis=1) < 5.5
            r, vi, vj = r[m], vi[m], vj[m]
            hv = h(r, vi - vj)
            m = (hv >= 0) & (hv <= 0.3)
            keep.append((r[m], vi[m], vj[m]))
            have += int(m.sum())
        r, vi, vj = (np.concatenate([x[i] for x in keep])[:n] for i in range(3))
        ui, uj = grid_inputs(vi, k), grid_inputs(vj, k)
        h0 = h(r, vi - vj)
        rows = {}
        for gamma in (0.05, 1.0):
            fail = 0
            for s in range(0, n, 200):
                sl = slice(s, s + 200)
                vi1 = vi[sl, None, None, :] + DT * ui[sl, :, None, :]
                vj1 = vj[sl, None, None, :] + DT * uj[sl, None, :, :]
                r1 = r[sl, None, None, :] + DT * (vi1 - vj1)
                best = h(r1, vi1 - vj1).reshape(len(r[sl]), -1).max(1)
                fail += int((best < (1 - gamma) * h0[sl] - 1e-9).sum())
            rows[str(gamma)] = {"no_input_found": fail}
        res[name] = {"states": int(n), "grid_per_agent": f"{k}x{k}", "by_gamma": rows}
    return res


# D. closed loop, two agents
def closed_loop(gamma, start_gap=20.0, steps=120, mode="coop"):
    """mode 'coop': two blues, each filtered with the cooperative half.
    mode 'one': a blue flying at 2 units/s at an immobile wreck (FULL row); only the blue acts."""
    from control.local_cbf import COOPERATIVE, FULL, LocalCBFConfig, Neighbor, filter_agent
    cfg = LocalCBFConfig(gamma=gamma, dt=DT, a_max=A, v_max=V)
    p = [np.array([0.0, 0.0]), np.array([start_gap, 0.0])]
    v = [np.array([2.0, 0.0]), np.array([-2.0, 0.0]) if mode == "coop" else np.zeros(2)]
    traj = []
    for k in range(steps):
        us, paths = [], []
        for i in ((0, 1) if mode == "coop" else (0,)):
            j = 1 - i
            res = filter_agent(p[i], v[i], np.array([A if i == 0 else -A, 0.0]),
                               [Neighbor(j, p[j], v[j], D, COOPERATIVE if mode == "coop" else FULL)], cfg)
            us.append(res.u)
            paths.append(res.path)
        if mode == "one":
            us.append(np.zeros(2))
        gap = float(np.linalg.norm(p[1] - p[0]))
        traj.append([k, round(gap, 4), round(float(v[0][0] - v[1][0]), 4), paths[0]])
        for i in (0, 1):
            v[i] = v[i] + DT * us[i]
            p[i] = p[i] + DT * v[i]
    return traj


def main():
    return {"A_headon_map": headon_map(), "B_counterexample": counterexample(),
            "C_random": random_test(), "E_boundary": boundary_test(),
            "D_closed_loop": {m: {str(g): closed_loop(g, mode=m) for g in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)}
                              for m in ("coop", "one")}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    out_dir = ap.parse_args().out
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "barrier_tests.json"), "w") as f:
        json.dump(main(), f, separators=(",", ":"))
