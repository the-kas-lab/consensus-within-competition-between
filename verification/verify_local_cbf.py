"""Development checks for the per-agent CBF (control/local_cbf.py).

Free, no episode, no network. Agents fly the testbed's double integrator
(semi-implicit Euler) toward goals that force conflicts; every agent runs its
own filter each step from one shared pre-step snapshot. Reported per gamma and
scenario family, kept separate on purpose:

  * path counts      -- hard / relaxed / braking agent-steps (and why)
  * max residual     -- max(b - g.u) on accepted hard steps (solver accuracy)
  * actual separation-- min true distance, and pair-steps below D

Every case is reported. Each family is ALSO reported restricted to starts
that pass a radial braking screen: for every pair, closing speed w along the
line of centres and gap |r| - D satisfy |r| - D >= w^2 / (2 a), with a = 2 a_max
when both brake (same team) and a = a_max when one side holds velocity (the
cross-team prediction). The screen is a one-dimensional heuristic. Failing it
does NOT show that no controller could avoid a violation (agents can steer
laterally), and passing it does not show joint feasibility with several
neighbours.

These are acceptance checks for an implementation and a basis for choosing
gamma on development cases. They are not a proof of recursive feasibility.

    python3 -m verification.verify_local_cbf [--gammas 0.05 0.1 0.2] [--json out.json]
"""

import argparse
import itertools
import json
from collections import Counter

import numpy as np

from control.local_cbf import COOPERATIVE, FULL, LocalCBFConfig, Neighbor, filter_agent

DT, VMAX, AMAX, D, CONTACT = 0.1, 2.0, 2.0, 3.0, 0.4
STEPS = 400


def nominal(p, v, goal):
    return np.clip(4.0 * (goal - p) - 2.0 * v, -AMAX, AMAX)


def simulate(agents, cfg, cross_team):
    """agents: list of dicts with p, v, goal, team. Same-team pairs share
    responsibility; cross-team pairs use the constant-velocity heuristic
    (FULL, other side's velocity as observed)."""
    P = np.array([a["p"] for a in agents], float)
    V = np.array([a["v"] for a in agents], float)
    goals = np.array([a["goal"] for a in agents], float)
    teams = [a["team"] for a in agents]
    n = len(agents)
    paths, reasons = Counter(), Counter()
    max_res, dmin, below = 0.0, np.inf, 0
    for _ in range(STEPS):
        U = np.zeros((n, 2))
        for i in range(n):
            nbs = []
            for j in range(n):
                if j == i:
                    continue
                same = teams[i] == teams[j]
                if not same and not cross_team:
                    continue
                nbs.append(Neighbor(j, P[j], V[j], D, COOPERATIVE if same else FULL))
            r = filter_agent(P[i], V[i], nominal(P[i], V[i], goals[i]), nbs, cfg)
            U[i] = r.u
            paths[r.path] += 1
            if r.reason:
                reasons[r.reason] += 1
            if r.path == "hard":
                max_res = max(max_res, r.barrier_residual)
        V = V + DT * U
        P = P + DT * V
        for i, j in itertools.combinations(range(n), 2):
            d = float(np.linalg.norm(P[i] - P[j]))
            dmin = min(dmin, d)
            below += d < D - 1e-9
    return paths, reasons, max_res, dmin, below


def passes_radial_braking_screen(agents, cross_team):
    for a, b in itertools.combinations(agents, 2):
        same = a["team"] == b["team"]
        if not same and not cross_team:
            continue
        r = np.asarray(a["p"], float) - np.asarray(b["p"], float)
        v = np.asarray(a["v"], float) - np.asarray(b["v"], float)
        dist = np.linalg.norm(r)
        closing = max(0.0, -(r @ v) / dist)
        decel = 2 * AMAX if same else AMAX
        if dist - D < closing ** 2 / (2 * decel):
            return False
    return True


def pair_cases():
    for speed, sep in itertools.product([0.5, 1.0, 2.0], [4.0, 8.0, 20.0]):
        # Overtaking: same heading, the follower twice as fast.
        yield f"overtake s={speed} r0={sep}", [
            dict(p=np.array([0.0, 0.0]), v=np.array([2 * speed if speed < 1.5 else 2.0, 0.0]),
                 goal=np.array([80.0, 0.0]), team=0),
            dict(p=np.array([sep, 0.3]), v=np.array([speed / 2, 0.0]),
                 goal=np.array([sep + 10.0, 0.3]), team=0),
        ]
    for speed, sep, angle in itertools.product([0.5, 1.0, 2.0], [4.0, 8.0, 20.0], [0, 45, 90, 135]):
        a = np.deg2rad(angle)
        # Two agents aimed through a common point at the given crossing angle.
        d1 = np.array([1.0, 0.0])
        d2 = -np.array([np.cos(a), np.sin(a)])  # angle 0 = head-on
        p1 = -d1 * sep / 2
        p2 = -d2 * sep / 2 + np.array([0.0, 0.25])  # small offset breaks exact symmetry
        yield f"pair s={speed} r0={sep} ang={angle}", [
            dict(p=p1, v=d1 * speed, goal=p1 + d1 * 40, team=0),
            dict(p=p2, v=d2 * speed, goal=p2 + d2 * 40, team=0),
        ]


def ring_cases():
    for n, radius in itertools.product([3, 4, 6, 8], [8.0, 14.0]):
        th = 2 * np.pi * np.arange(n) / n + 0.07
        pts = radius * np.c_[np.cos(th), np.sin(th)]
        yield f"swap n={n} R={radius}", [
            dict(p=pts[k], v=-pts[k] / radius * 1.5, goal=-pts[k], team=0) for k in range(n)
        ]


def cross_cases():
    for speed, angle in itertools.product([1.0, 2.0], [0, 45, 90, 135]):
        a = np.deg2rad(angle)
        d2 = -np.array([np.cos(a), np.sin(a)])
        p1, p2 = np.array([-10.0, 0.0]), -d2 * 10 + np.array([0.0, 0.25])
        yield f"cross s={speed} ang={angle}", [
            dict(p=p1, v=np.array([speed, 0.0]), goal=p1 + np.array([40.0, 0]), team=0),
            dict(p=p2, v=d2 * speed, goal=p2 + d2 * 40, team=1),
        ]


FAMILIES = {
    "pair (same team)": (pair_cases, False),
    "antipodal swap (same team)": (ring_cases, False),
    "cross-team (CV heuristic)": (cross_cases, True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows = []
    print(f"dt={DT} v_max={VMAX} a_max={AMAX} D={D} contact={CONTACT} steps={STEPS}")
    print(f"{'family':28s} {'subset':10s} {'gamma':>5s} {'cases':>5s} {'hard%':>7s} {'relax':>6s} "
          f"{'brake':>6s} {'maxres':>8s} {'dmin':>7s} {'<D steps':>8s} {'cases<D':>7s} {'at contact':>10s}")
    for fam, (gen, cross) in FAMILIES.items():
        for g in args.gammas:
            cfg = LocalCBFConfig(gamma=g, dt=DT, a_max=AMAX, v_max=VMAX)
            runs = []
            for name, agents in gen():
                paths, rsn, res, dm, bl = simulate(agents, cfg, cross)
                runs.append(dict(case=name, screen=passes_radial_braking_screen(agents, cross),
                                 paths=paths, reasons=rsn, res=res, dmin=dm, below=bl))
            for subset, sel in (("all", runs), ("screened", [r for r in runs if r["screen"]])):
                tot, reasons = Counter(), Counter()
                for r in sel:
                    tot.update(r["paths"]); reasons.update(r["reasons"])
                steps = sum(tot.values())
                worst = min(sel, key=lambda r: r["dmin"]) if sel else None
                row = dict(family=fam, subset=subset, gamma=g, cases=len(sel), paths=dict(tot),
                           reasons=dict(reasons),
                           max_hard_residual=max((r["res"] for r in sel), default=0.0),
                           dmin=worst["dmin"] if worst else float("nan"),
                           worst_dmin_case=worst["case"] if worst else None,
                           pair_steps_below_D=sum(r["below"] for r in sel),
                           cases_below_D=sum(r["below"] > 0 for r in sel),
                           cases_at_contact=sum(r["dmin"] <= CONTACT for r in sel),
                           failing_screen=[r["case"] for r in runs if not r["screen"]]
                           if subset == "all" else None)
                hard = (tot["hard"] + tot["unconstrained"]) / steps if steps else float("nan")
                print(f"{fam:28s} {subset:10s} {g:5.2f} {row['cases']:5d} {100 * hard:6.2f}% "
                      f"{tot['relaxed']:6d} {tot['braking']:6d} {row['max_hard_residual']:8.1e} "
                      f"{row['dmin']:7.3f} {row['pair_steps_below_D']:8d} {row['cases_below_D']:7d} "
                      f"{row['cases_at_contact']:10d}")
                rows.append(row)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(dt=DT, v_max=VMAX, a_max=AMAX, D=D, steps=STEPS, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
