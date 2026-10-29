"""Exact hindsight reachability of an intercept, from saved states (no episode).

Question answered: given a pursuer's state at step f and a target's positions at
later steps, is there ANY admissible pursuer motion that comes within ``tol`` of
the target at some step? This separates "the controller missed a reachable
intercept" from "no controller could have made it".

Exact for the testbed's blue dynamics (agents/dynamics.Agent.step_single:
``v += a*dt; p += v*dt``) under the per-axis box limits the MPC enforces
(``|a_x|, |a_y| <= a_max`` and ``|v_x|, |v_y| <= v_max`` after each step). The
two axes are independent, so the reachable position set after n steps is a box,
and its bounds per axis are attained by full acceleration one way:
``v_k = min(v_max, v_{k-1} + a_max*dt)`` is pointwise maximal, so position is too.

**The target's path is taken as given (open loop).** A red that reacts to the
pursuer could move differently, so "reachable" here does not mean another
controller would have succeeded. "Unreachable" does: against that path, nothing
admissible gets within ``tol``.

    python3 -m scripts.intercept_reachability REPLAY.js --episode 1 --blue 0 --red 1 \
        --frames 300,310,320,330
"""

import argparse
import json
import math

import numpy as np


def _axis_extremes(p, v, n, dt, a_max, v_max):
    """Per-step (lowest, highest) reachable position on one axis, n steps."""
    lo, hi = [], []
    p_lo = p_hi = float(p)
    v_lo = v_hi = float(v)
    for _ in range(n):
        v_lo = max(-v_max, v_lo - a_max * dt)
        v_hi = min(v_max, v_hi + a_max * dt)
        p_lo += dt * v_lo
        p_hi += dt * v_hi
        lo.append(p_lo)
        hi.append(p_hi)
    return lo, hi


def reachable_gaps(pursuer_state, target_positions, dt, a_max, v_max):
    """Minimum achievable pursuer-target distance at each of the next
    ``len(target_positions)`` steps. ``pursuer_state`` is ``[px, py, vx, vy, ...]``;
    ``target_positions[k]`` is the target's position k+1 steps later."""
    target_positions = np.asarray(target_positions, dtype=float).reshape(-1, 2)
    n = len(target_positions)
    x_lo, x_hi = _axis_extremes(pursuer_state[0], pursuer_state[2], n, dt, a_max, v_max)
    y_lo, y_hi = _axis_extremes(pursuer_state[1], pursuer_state[3], n, dt, a_max, v_max)
    gaps = []
    for k, (tx, ty) in enumerate(target_positions):
        dx = max(x_lo[k] - tx, 0.0, tx - x_hi[k])
        dy = max(y_lo[k] - ty, 0.0, ty - y_hi[k])
        gaps.append(math.hypot(dx, dy))
    return gaps


def first_reachable_step(gaps, tol):
    """1-based offset of the first step whose gap is within ``tol``, else None."""
    return next((k + 1 for k, g in enumerate(gaps) if g <= tol), None)


def _load_replay(path):
    text = open(path).read()
    return json.loads(text[text.index("(") + 1 : text.rindex(")")])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("replay")
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--blue", type=int, required=True)
    ap.add_argument("--red", type=int, required=True)
    ap.add_argument("--frames", required=True, help="comma-separated start frames")
    args = ap.parse_args()

    episode = _load_replay(args.replay)["episodes"][args.episode]
    cfg = episode["recorder"]["config"]
    dims = episode["dims"]
    tol = float(episode["geometry"]["contact_tolerance"])
    blue = np.asarray(episode["frames"]["blue_state"], dtype=float)
    red = np.asarray(episode["frames"]["red_state"], dtype=float)
    bnx, rnx = dims["blue_nx"], dims["red_nx"]
    last = len(blue) - 1
    for f in (int(x) for x in args.frames.split(",")):
        b = blue[f, args.blue * bnx : args.blue * bnx + 4]
        future = red[f + 1 : last + 1, args.red * rnx : args.red * rnx + 2]
        gaps = reachable_gaps(b, future, dims["dt"], cfg["blue_a_max"], cfg["blue_v_max"])
        first = first_reachable_step(gaps, tol)
        best = int(np.argmin(gaps))
        now = float(np.linalg.norm(b[0:2] - red[f, args.red * rnx : args.red * rnx + 2]))
        print(
            f"frame {f}: distance {now:.3f}; first frame with a reachable point within "
            f"{tol}: {None if first is None else f + first}; minimum achievable gap "
            f"{gaps[best]:.3f} at frame {f + best + 1}"
        )


if __name__ == "__main__":
    main()
