"""control.local_cbf: the per-agent sampled-step barrier filter.

The sufficiency tests integrate the real Agent dynamics (semi-implicit
Euler) after filtering and check h+ >= (1 - gamma) h on every pair whose
filters reported an accepted hard solve. They test the stated condition only:
no recursive feasibility, no continuous-time claim."""

import numpy as np
import pytest

from agents.dynamics import Agent
from control.local_cbf import (
    COOPERATIVE, FULL, LocalCBFConfig, Neighbor, braking, filter_agent,
)

DT, VMAX, AMAX, D = 0.1, 2.0, 2.0, 3.0
CFG = LocalCBFConfig(gamma=0.1, dt=DT, a_max=AMAX, v_max=VMAX)
AGENT = Agent(Agent.BLUE, dt=DT, count=1, v_max=VMAX, a_max=AMAX)


def _step(p, v, u):
    x = AGENT.step_single(np.r_[p, v], u)
    return x[0:2], x[2:4]


def _h(p_i, p_j):
    r = p_i - p_j
    return r @ r - D ** 2


def _random_pair(rng):
    p_i = rng.uniform(-6, 6, 2)
    p_j = rng.uniform(-6, 6, 2)
    v_i = rng.uniform(-VMAX, VMAX, 2)
    v_j = rng.uniform(-VMAX, VMAX, 2)
    return p_i, v_i, p_j, v_j


def test_cooperative_halves_imply_the_pair_condition():
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(3000):
        p_i, v_i, p_j, v_j = _random_pair(rng)
        if _h(p_i, p_j) < 0:
            continue
        r_i = filter_agent(p_i, v_i, rng.uniform(-AMAX, AMAX, 2),
                           [Neighbor("j", p_j, v_j, D, COOPERATIVE)], CFG)
        r_j = filter_agent(p_j, v_j, rng.uniform(-AMAX, AMAX, 2),
                           [Neighbor("i", p_i, v_i, D, COOPERATIVE)], CFG)
        if r_i.path != "hard" or r_j.path != "hard":
            continue
        pn_i, _ = _step(p_i, v_i, r_i.u)
        pn_j, _ = _step(p_j, v_j, r_j.u)
        assert _h(pn_i, pn_j) >= (1 - CFG.gamma) * _h(p_i, p_j) - 1e-7
        checked += 1
    assert checked > 500


def test_full_responsibility_against_a_constant_velocity_neighbor():
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(3000):
        p_i, v_i, p_j, v_j = _random_pair(rng)
        if _h(p_i, p_j) < 0:
            continue
        r = filter_agent(p_i, v_i, rng.uniform(-AMAX, AMAX, 2),
                         [Neighbor("j", p_j, v_j, D, FULL)], CFG)
        if r.path != "hard":
            continue
        pn_i, _ = _step(p_i, v_i, r.u)
        pn_j, _ = _step(p_j, v_j, np.zeros(2))  # the declared prediction
        assert _h(pn_i, pn_j) >= (1 - CFG.gamma) * _h(p_i, p_j) - 1e-7
        checked += 1
    assert checked > 500


def test_accepted_hard_solutions_respect_limits():
    rng = np.random.default_rng(2)
    for _ in range(500):
        p_i, v_i, p_j, v_j = _random_pair(rng)
        r = filter_agent(p_i, v_i, rng.uniform(-5, 5, 2),
                         [Neighbor("j", p_j, v_j, D, FULL)], CFG)
        if r.path in ("hard", "relaxed"):
            assert np.all(np.abs(r.u) <= AMAX + 1e-6)
            assert np.all(np.abs(v_i + DT * r.u) <= VMAX + 1e-6)


def test_infeasible_hard_falls_to_relaxed_and_reports_the_slack():
    # Head-on at full speed, already near D: no admissible braking suffices.
    r = filter_agent([0.0, 0.0], [2.0, 0.0], [2.0, 0.0],
                     [Neighbor("j", np.array([3.3, 0.0]), np.array([-2.0, 0.0]), D, FULL)], CFG)
    assert r.path == "relaxed"
    assert r.reason == "infeasible"
    assert r.slacks["j"] > 0
    assert r.barrier_residual == pytest.approx(r.slacks["j"])


def test_empty_velocity_box_brakes_and_reports_it():
    r = filter_agent([0.0, 0.0], [3.0, 0.0], [0.0, 0.0], [], CFG)
    assert r.path == "braking" and r.velocity_box_empty
    np.testing.assert_allclose(r.u, [-AMAX, 0.0])


def test_braking_is_projected_onto_the_admissible_box():
    u, empty = braking(np.array([0.05, -1.0]), CFG)
    assert not empty
    np.testing.assert_allclose(u, [-0.5, AMAX])


def test_no_neighbors_passes_the_nominal_within_limits():
    r = filter_agent([0, 0], [1.9, 0], [3.0, -1.0], [], CFG)
    assert r.path == "unconstrained"
    np.testing.assert_allclose(r.u, [1.0, -1.0])


@pytest.mark.parametrize("gamma", [0.0, -0.1, 1.5])
def test_gamma_outside_the_unit_interval_is_refused(gamma):
    with pytest.raises(ValueError, match="gamma"):
        LocalCBFConfig(gamma=gamma, dt=DT, a_max=AMAX, v_max=VMAX)
