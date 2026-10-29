"""Bid version "v2.3" (conditional pursuit continuity) and m8b.6. Same static scene
and helpers as test_bid_threat_forecast (spent seed 3901, views built by hand). No model
calls."""

import math
import os
import sys
from dataclasses import replace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_bid_threat_forecast import (NOW, _belief, _cbaa, _engaged, _grid, _red, _team,  # noqa: E402
                          _views)

from coordination import bidding  # noqa: E402
from coordination.bidding import V22_PROFILES  # noqa: E402
from coordination.tasks import NEUTRALIZE, generate_tasks  # noqa: E402

H_P = V22_PROFILES["balanced"].pursuit_continuity


@pytest.fixture(scope="module")
def scene():
    from coordination.tasks import TaskContext
    from scripts import run_competing_teams as rct

    e = {"arm": "cbaa", "reds": 4, "family": "multi_bearing", "condition": "REF",
         "attacked": False, "seed": 3901, "design": "v2.3"}
    e["episode_id"] = rct.episode_id(e)
    sim, _ = rct.build(e, "rule")
    assert sim.cbaa_bid_version == "v2.3"
    task_ctx = TaskContext.from_simulation(sim, task_set="three_task")
    ctx = bidding.BidContext.from_simulation(sim, task_ctx)
    assert ctx.v2.version == "v2.3"
    ctx22 = replace(ctx, v2=replace(ctx.v2, version="v2.2"))
    return sim, task_ctx, ctx, ctx22


def _terms(sim, ctx, task_ctx, team, targets, agent):
    vw = _views(sim, ctx, team, targets)[agent]
    return bidding.bid_terms_v2(vw, generate_tasks(vw, task_ctx), ctx, team[agent][1])


def _cv_red(ctx, bearing, dist, speed=2.0):
    """ Part B: a red on a constant-velocity path to the nearest polygon point."""
    from control.gates import _ray_exit

    c = ctx.v2.field.center
    poly = np.asarray(ctx.defense_polygon)
    u = np.array([math.cos(math.radians(bearing)), math.sin(math.radians(bearing))])
    p = c + (float(np.linalg.norm(_ray_exit(c, u, poly) - c)) + dist) * u
    d = bidding.closest_point_on_polygon(p, poly) - p
    v = speed * d / np.linalg.norm(d)
    return p, v * min(1.0, 2.0 / max(abs(v[0]), abs(v[1])))


def _counterexample(sim, ctx, a_speed=1.0, frac=0.6, g=2):
    """: scout away (team variant B), g pursuing receding A,
    B on a CV path that only g's forecast reaches before its deadline."""
    pa, va = _red(sim, ctx, 160, 10, "moving_away", a_speed)
    pb, vb = _cv_red(ctx, 170, 10)
    team = _team(ctx, scout=True)
    st = team[g][0]
    u = pa - st[:2]
    p = st[:2] + frac * u
    vel = 2.0 * u / np.linalg.norm(u)
    team[g] = (np.array([p[0], p[1], vel[0], vel[1], math.atan2(u[1], u[0])]), "NEUTRALIZE:0:0")
    return team, {0: _belief(0, pa, va), 1: _belief(1, pb, vb)}


# -- frozen constants, v2.2 untouched -------------------------------------------------


def test_v23_constants_are_the_frozen_values_and_v22_terms_are_unchanged(scene):
    sim, task_ctx, ctx, ctx22 = scene
    assert (bidding.V23_RECEDE_SPEED, bidding.V23_HORIZON_RAMP_S, bidding.V23_MARGIN_RAMP_S) == (
        0.5, 5.0, 2.0)
    assert bidding.v2_profiles("v2.3") is bidding.V22_PROFILES
    team, tg = _counterexample(sim, ctx)
    t22 = _terms(sim, ctx22, task_ctx, team, tg, 2)
    assert all("continuity_relief" not in x and "own_margin_score" not in x for x in t22.values())
    t23 = _terms(sim, ctx, task_ctx, team, tg, 2)
    # v2.3 changes only the continuity on the other contact's NEUTRALIZE tasks
    for k, x in t22.items():
        y = t23[k]
        same = {kk for kk in x if kk not in ("continuity", "bid")}
        assert {kk: x[kk] for kk in same} == {kk: y[kk] for kk in same}, k
        if not k.startswith("NEUTRALIZE:1:"):
            assert y["bid"] == x["bid"], k


# -- V1: the counterexample --------------------------------------------------------------


@pytest.mark.parametrize("a_speed", [1.0, 2.0])
def test_v1_counterexample_urgent_red_goes_to_the_pursuer_and_the_receding_red_stays_engaged(
        scene, a_speed):
    sim, task_ctx, ctx, ctx22 = scene
    team, tg = _counterexample(sim, ctx, a_speed)
    old = _cbaa(sim, ctx22, task_ctx, team, tg)
    assert old[2].startswith("NEUTRALIZE:0:") and old[3].startswith("NEUTRALIZE:1:")  # 2.3
    new = _cbaa(sim, ctx, task_ctx, team, tg)
    assert new[2].startswith("NEUTRALIZE:1:")
    assert _engaged(new, 0)
    t = _terms(sim, ctx, task_ctx, team, tg, 2)
    assert t["NEUTRALIZE:1:0"]["continuity_relief"] == pytest.approx(1.0)
    assert t["NEUTRALIZE:1:0"].get("continuity", 0.0) == pytest.approx(0.0)


# -- V3: ordinary continuity is kept ----------------------------------------------------------


def test_v3_recon_defend_and_ordinary_retargeting_keep_the_full_continuity_cost(scene):
    sim, task_ctx, ctx, _ = scene
    pb, vb = _cv_red(ctx, 170, 10)
    for status in ("approaching", "crossing", "stationary"):
        pa, va = _red(sim, ctx, 260, 30, status, 2.0 if status != "stationary" else 0.0)
        team = _team(ctx)
        team[2] = (team[2][0], "NEUTRALIZE:0:0")
        t = _terms(sim, ctx, task_ctx, team, {0: _belief(0, pa, va), 1: _belief(1, pb, vb)}, 2)
        for k, x in t.items():
            if not k.startswith("NEUTRALIZE:0:"):
                assert x["continuity"] == H_P, (status, k)  # zero velocity included: R = 0
    # receding A and an urgent, reachable B: RECON / DEFEND still pay the full cost
    team, tg = _counterexample(sim, ctx)
    t = _terms(sim, ctx, task_ctx, team, tg, 2)
    assert all(x["continuity"] == H_P for k, x in t.items() if not k.startswith(NEUTRALIZE))


def test_v3_near_threshold_band_gets_at_most_a_tenth_of_the_relief(scene):
    """: R uses the closing speed directly, so a pursued red
    labelled stationary or crossing with a small NEGATIVE closing speed
    (-0.05 < c < 0) gets R = -c / 0.5 <= 0.1: the 0.15 cost drops by at most
    0.015. Zero velocity gets exactly none."""
    from coordination import coverage_value as cv

    sim, task_ctx, ctx, _ = scene
    team, tg = _counterexample(sim, ctx)
    a = tg[0]
    pa = a.predicted_position(NOW, sim.dt)

    def closing(v):
        return cv.funnel_forecast(pa, np.asarray(v, float), ctx.v2.funnel, ctx.defense_polygon,
                                  2 * math.sqrt(2))

    n = np.array([closing([1.0, 0.0])["closing_speed"], closing([0.0, 1.0])["closing_speed"]])
    perp = np.array([-n[1], n[0]]) / np.linalg.norm(n)
    n = n / np.linalg.norm(n)
    cases = [(np.zeros(2), "stationary", 0.0),
             (-0.04 * n, "stationary", 0.04),        # |v| < 0.05: labelled stationary
             (-0.04 * n + 1.0 * perp, "crossing", 0.04),
             (-0.0499 * n + 1.0 * perp, "crossing", 0.0499)]
    for v, label, c_away in cases:
        fc = closing(v)
        assert fc["status"] == label and fc["closing_speed"] == pytest.approx(-c_away, abs=1e-9)
        t = _terms(sim, ctx, task_ctx, team, {0: _belief(0, pa, v), 1: tg[1]}, 2)
        b = t["NEUTRALIZE:1:0"]
        # H = S = 1 in this geometry, so relief = R = c_away / 0.5
        assert b["continuity_relief"] == pytest.approx(c_away / bidding.V23_RECEDE_SPEED, abs=1e-9)
        assert H_P - b["continuity"] <= 0.015 + 1e-12
        assert all(x["continuity"] == H_P for k, x in t.items() if not k.startswith(NEUTRALIZE))


def test_v3_no_relief_without_a_within_episode_breach_or_an_own_margin(scene):
    sim, task_ctx, ctx, _ = scene
    team, tg = _counterexample(sim, ctx)
    pb, vb = _red(sim, ctx, 200, 30, "moving_away", 2.0)  # B not closing: no forecast breach
    t = _terms(sim, ctx, task_ctx, team, {0: tg[0], 1: _belief(1, pb, vb)}, 2)
    assert t["NEUTRALIZE:1:0"]["continuity"] == H_P
    # B's forecast breach after the episode ends (now = 398: 0.2 s left)
    late = 398
    vw = _views(sim, ctx, team, {c: _belief(c, b.state[:2], b.velocity, now=late)
                                 for c, b in tg.items()}, now=late)[2]
    t = bidding.bid_terms_v2(vw, generate_tasks(vw, task_ctx), ctx, "NEUTRALIZE:0:0")
    assert t["NEUTRALIZE:1:0"]["continuity"] == H_P
    # a pursuer that cannot reach B before its forecast breach: S = 0
    far = dict(team)
    far[2] = (np.array([10.0, 10.0, 0.0, 0.0, 0.0]), "NEUTRALIZE:0:0")
    t = _terms(sim, ctx, task_ctx, far, tg, 2)
    assert t["NEUTRALIZE:1:0"]["own_margin_score"] == 0.0
    assert t["NEUTRALIZE:1:0"]["continuity"] == H_P


# -- V4: unchanged properties ---------------------------------------------------------------


def test_v4_p3_p5_p6_hold_under_v23(scene):
    sim, task_ctx, ctx, _ = scene
    p, v = _red(sim, ctx, 220, 50, "approaching", 2.0)
    team = _team(ctx)
    team[2] = (team[2][0], "NEUTRALIZE:0:0")
    assert all(x.get("continuity", 0.0) == 0.0 for x in _terms(sim, ctx, task_ctx, team, {}, 2)
               .values())  # pursued contact absent / resolved
    vw = _views(sim, ctx, team, {0: _belief(0, p, v)})[2]
    t = bidding.bid_terms_v2(vw, generate_tasks(vw, replace(task_ctx, slots=2)), ctx,
                             "NEUTRALIZE:0:0")
    assert t["NEUTRALIZE:0:1"].get("continuity", 0.0) == 0.0  # slot change is free


def test_v4_bids_without_a_pursuer_are_identical_to_v22(scene):
    sim, task_ctx, ctx, ctx22 = scene
    for scout in (False, True):
        team = _team(ctx, scout)
        for _, p, v in _grid(sim, ctx):
            tg = {0: _belief(0, p, v)}
            for i, vw in _views(sim, ctx, team, tg).items():
                tasks = generate_tasks(vw, task_ctx)
                assert (bidding.bids(vw, tasks, ctx, current_task_id=team[i][1])
                        == bidding.bids(vw, tasks, ctx22, current_task_id=team[i][1]))


def test_v4_p2_prime_a_pursuers_neutralize_bids_move_at_most_0_15(scene):
    sim, task_ctx, ctx, _ = scene
    worst = 0.0
    for a_speed in (1.0, 2.0):
        team, tg = _counterexample(sim, ctx, a_speed)
        base = {k: x["bid"] for k, x in _terms(sim, ctx, task_ctx, team, tg, 2).items()
                if k.startswith(NEUTRALIZE)}
        for cid in (0, 1):
            b = tg[cid]
            v0 = np.asarray(b.velocity, float)
            sp, hd = float(np.hypot(*v0)), math.atan2(v0[1], v0[0])
            pert = [v0 + np.array(d) for d in ((0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1))]
            pert += [sp * np.array([math.cos(hd + s), math.sin(hd + s)])
                     for s in (math.radians(2), -math.radians(2))]
            for v in pert:
                tg2 = dict(tg)
                tg2[cid] = _belief(cid, b.state[:2], np.clip(v, -2, 2))
                t = _terms(sim, ctx, task_ctx, team, tg2, 2)
                worst = max(worst, max(abs(t[k]["bid"] - base[k]) for k in base))
    assert worst <= 0.15, worst


def test_v4_p7_cbaa_and_independent_compute_identical_v23_bids():
    from scripts import run_competing_teams as rct
    from scripts.export_replay import _Quiet

    got = {}
    for arm in ("cbaa", "independent"):
        e = {"arm": arm, "reds": 2, "family": "multi_bearing", "condition": "REF",
             "attacked": False, "seed": 3901, "design": "v2.3"}
        e["episode_id"] = rct.episode_id(e)
        sim, _ = rct.build(e, "rule")
        assert sim.cbaa_bid_version == "v2.3"
        captured = []
        orig = sim.cbaa_runtime.start_epoch

        def se(step, views, max_age=None, modifiers=None, _rt=sim.cbaa_runtime, _o=orig, _c=captured):
            _o(step, views, max_age=max_age, modifiers=modifiers)
            if not _c:
                _c.append({i: dict(_rt.agents[i].bids) for i in views})

        sim.cbaa_runtime.start_epoch = se
        sim.sim_steps = 1
        with _Quiet():
            sim.run()
        got[arm] = (sim.cbaa_runtime.transport, captured[0])
    assert got["cbaa"][0] != got["independent"][0]
    assert got["cbaa"][1] == got["independent"][1]


# -- the commander's inputs are unchanged ---------------------------------------------------


# -- m8b.6 ----------------------------------------------------------------------------------


def test_runtime_logs_relief_without_changing_the_bids(scene):
    """: the v2.3 runtime records agent-epochs whose bids carried
    continuity relief; the bids it auctions equal bidding.bids()."""
    sim, task_ctx, ctx, _ = scene
    rt = sim.cbaa_runtime
    rt.reset(4, task_ctx, ctx)
    team, tg = _counterexample(sim, ctx)
    views = _views(sim, ctx, team, tg)
    for i, (_, com) in team.items():
        rt.agents[i].commitment = com
    rt.start_epoch(NOW, views, max_age=None)
    for i, vw in views.items():
        tasks = generate_tasks(vw, task_ctx)
        assert rt.agents[i].bids == bidding.bids(vw, tasks, ctx, current_task_id=team[i][1])
    assert rt.relief_log == [(NOW, 2, "NEUTRALIZE:0:0", {"NEUTRALIZE:1:0": 1.0})]
