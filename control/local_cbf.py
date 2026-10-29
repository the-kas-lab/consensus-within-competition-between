"""Per-agent (local) discrete-time CBF safety filter.

Each agent solves its own QP over its own translational acceleration
u_i in R^2. No agent optimizes, or assumes it knows, another agent's
filtered control. Heading control is not touched here.

Barrier (sampled-step, matched to Agent.step_single's semi-implicit Euler,
p+ = p + dt*v + dt^2*u). For a pair (i, j) with r = p_i - p_j,
v = v_i - v_j, c = r + dt*v and h = |r|^2 - D^2:

    |c + dt^2 (u_i - u_j)|^2 - D^2 >= |c|^2 - D^2 + 2 dt^2 c.(u_i - u_j)

so  g.(u_i - u_j) >= b,  g = 2 dt^2 c,  b = (1 - gamma) h - (|c|^2 - D^2)
is sufficient for h+ >= (1 - gamma) h, 0 < gamma <= 1.

How each pair's condition is allocated (``Neighbor.share``):

* ``COOPERATIVE`` -- a same-team, mobile teammate running this same filter.
  Agent i enforces g.u_i >= b/2 and j enforces the mirrored half; if both
  hold, the pair condition holds. Nothing is exchanged.
* ``FULL`` -- everything else: an opposing agent (predicted at constant
  velocity, u_j = 0, a declared heuristic with no cross-team guarantee) or an
  immobile agent (velocity and acceleration zero). Agent i enforces g.u_i >= b.

Solve order: hard QP; accepted only after the solver status AND a direct
residual check. Otherwise a relaxed QP with one nonnegative slack per pair
(acceleration and next-step velocity limits stay hard). Otherwise bounded
braking. Every step reports which path ran and why.

What an accepted hard solve establishes (numerically, not exactly). Acceptance
requires every row to hold to within ``residual_tol`` (tau):
g.u_i >= share * b - tau, and the limits to within tau. Because
h+ >= |c|^2 - D^2 + g.(u_i - u_j) (the dropped term is nonnegative):

* FULL row, neighbour behaving as predicted:   h+ >= (1 - gamma) h - tau
* COOPERATIVE pair, BOTH halves accepted hard: h+ >= (1 - gamma) h - 2 tau
  (the two agents' allowances add)

conditional on the model, the synchronized snapshot, and floating-point
evaluation of the rows. A relaxed or braking step carries no such condition.
Nothing here gives recursive feasibility, continuous-time separation, or any
cross-team guarantee (the constant-velocity prediction is a heuristic).
"""

from dataclasses import dataclass, field
from typing import Hashable, List, Optional

import numpy as np
import osqp
import scipy.sparse as sparse

LOCAL_CBF_VERSION = "lcbf.1"

COOPERATIVE = 0.5
FULL = 1.0


@dataclass(frozen=True)
class LocalCBFConfig:
    gamma: float
    dt: float
    a_max: float
    v_max: float
    relax_weight: float = 1e5
    residual_tol: float = 1e-6

    def __post_init__(self):
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], not {self.gamma!r}")


@dataclass(frozen=True)
class Neighbor:
    key: Hashable
    position: np.ndarray
    velocity: np.ndarray
    d_safe: float
    share: float


@dataclass
class LocalCBFResult:
    u: np.ndarray
    path: str  # "unconstrained" | "hard" | "relaxed" | "braking"
    reason: Optional[str] = None  # why the hard solve was not used
    hard_status: Optional[str] = None
    relaxed_status: Optional[str] = None
    slacks: dict = field(default_factory=dict)
    barrier_residual: float = 0.0  # max(b - g.u) over rows, after the chosen path
    velocity_box_empty: bool = False
    num_rows: int = 0


def pair_row(p_i, v_i, p_j, v_j, d_safe, gamma, dt):
    """(g, b) for the pair condition g.(u_i - u_j) >= b."""
    r = np.asarray(p_i, float) - np.asarray(p_j, float)
    v = np.asarray(v_i, float) - np.asarray(v_j, float)
    c = r + dt * v
    h = r @ r - d_safe ** 2
    return 2.0 * dt ** 2 * c, (1.0 - gamma) * h - (c @ c - d_safe ** 2)


def _limits(v_own, cfg):
    """Componentwise feasible acceleration interval from |u| <= a_max and
    |v + dt u| <= v_max."""
    lo = np.maximum(-cfg.a_max, (-cfg.v_max - v_own) / cfg.dt)
    hi = np.minimum(cfg.a_max, (cfg.v_max - v_own) / cfg.dt)
    return lo, hi


def braking(v_own, cfg):
    """Bounded braking toward zero velocity, projected onto the admissible
    acceleration/next-velocity box when it is nonempty; otherwise acceleration
    limits only (the velocity violation is reported by the caller)."""
    target = -np.asarray(v_own, float) / cfg.dt
    lo, hi = _limits(v_own, cfg)
    if np.all(lo <= hi):
        return np.clip(target, lo, hi), False
    return np.clip(target, -cfg.a_max, cfg.a_max), True


def _solve(P, q, A, l, u):
    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, verbose=False, polish=True,
               eps_abs=1e-9, eps_rel=1e-9, max_iter=20000)
    return prob.solve()


def filter_agent(p_own, v_own, u_nom, neighbors: List[Neighbor], cfg: LocalCBFConfig):
    p_own = np.asarray(p_own, float).reshape(2)
    v_own = np.asarray(v_own, float).reshape(2)
    u_nom = np.asarray(u_nom, float).reshape(2)

    G, B = [], []
    for nb in neighbors:
        g, b = pair_row(p_own, v_own, nb.position, nb.velocity, nb.d_safe, cfg.gamma, cfg.dt)
        G.append(g)
        B.append(nb.share * b)
    G = np.array(G).reshape(-1, 2)
    B = np.array(B)
    m = len(B)
    lo, hi = _limits(v_own, cfg)
    box_empty = bool(np.any(lo > hi))

    def residual(u):
        return float(np.max(B - G @ u)) if m else 0.0

    def box_ok(u):
        tol = cfg.residual_tol
        return (np.all(np.abs(u) <= cfg.a_max + tol)
                and np.all(np.abs(v_own + cfg.dt * u) <= cfg.v_max + tol))

    if box_empty:
        u, _ = braking(v_own, cfg)
        return LocalCBFResult(u, "braking", reason="velocity_box_empty",
                              barrier_residual=residual(u), velocity_box_empty=True,
                              num_rows=m)

    if m == 0:
        u = np.clip(u_nom, lo, hi)
        return LocalCBFResult(u, "unconstrained", num_rows=0)

    # The objective is separable, so clipping the nominal to the admissible
    # box is the box-only optimum; if that point also satisfies every barrier
    # row it is the full QP's optimum, exactly, with no solver call.
    u_box = np.clip(u_nom, lo, hi)
    if residual(u_box) <= 0.0:
        return LocalCBFResult(u_box, "hard", hard_status="box_projection",
                              barrier_residual=residual(u_box), num_rows=m)

    box_A = sparse.vstack([sparse.eye(2), cfg.dt * sparse.eye(2)])
    box_l = np.r_[-cfg.a_max, -cfg.a_max, -cfg.v_max - v_own]
    box_u = np.r_[cfg.a_max, cfg.a_max, cfg.v_max - v_own]

    # 1. hard
    res = _solve(sparse.eye(2, format="csc"), -u_nom,
                 sparse.vstack([box_A, sparse.csc_matrix(G)]).tocsc(),
                 np.r_[box_l, B], np.r_[box_u, np.full(m, np.inf)])
    hard_status = res.info.status
    if hard_status == "solved" and res.x is not None:
        u = np.asarray(res.x[:2], float)
        if box_ok(u) and residual(u) <= cfg.residual_tol:
            return LocalCBFResult(u, "hard", hard_status=hard_status,
                                  barrier_residual=residual(u), num_rows=m)
        reason = "numerical"
    elif "infeasible" in hard_status and "dual" not in hard_status:
        reason = "infeasible"
    else:
        reason = "numerical"

    # 2. relaxed: one slack per pair, limits stay hard
    P = sparse.diags([1.0, 1.0] + [cfg.relax_weight] * m).tocsc()
    q = np.r_[-u_nom, np.zeros(m)]
    A = sparse.vstack([
        sparse.hstack([box_A, sparse.csc_matrix((4, m))]),
        sparse.hstack([sparse.csc_matrix(G), sparse.eye(m)]),
        sparse.hstack([sparse.csc_matrix((m, 2)), sparse.eye(m)]),
    ]).tocsc()
    l = np.r_[box_l, B, np.zeros(m)]
    u_ = np.r_[box_u, np.full(m, np.inf), np.full(m, np.inf)]
    res = _solve(P, q, A, l, u_)
    relaxed_status = res.info.status
    if relaxed_status == "solved" and res.x is not None:
        u = np.asarray(res.x[:2], float)
        s = np.maximum(B - G @ u, 0.0)  # realised violation, not the solver's slack variable
        if box_ok(u):
            return LocalCBFResult(
                u, "relaxed", reason=reason, hard_status=hard_status,
                relaxed_status=relaxed_status,
                slacks={nb.key: float(si) for nb, si in zip(neighbors, s)},
                barrier_residual=residual(u), num_rows=m)

    # 3. bounded braking
    u, empty = braking(v_own, cfg)
    return LocalCBFResult(u, "braking", reason=reason, hard_status=hard_status,
                          relaxed_status=relaxed_status, barrier_residual=residual(u),
                          velocity_box_empty=empty, num_rows=m)
