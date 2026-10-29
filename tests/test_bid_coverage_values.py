"""Bid version "v2" and the configured station posture.

Functional criteria frozen before any v2 verification episode. Static
geometry from a BUILT (never run) multi_bearing episode on spent seed 3901;
views are constructed by hand so each criterion is checked in isolation.
No model calls."""

import math

import numpy as np
import pytest

from coordination import bidding, coverage_value as cv
from coordination.bidding import PREFERRED_BONUS, PRIORITY_MULTIPLIER, BidModifiers, V2_PROFILES
from coordination.tasks import TaskContext, generate_tasks
from perception.belief.records import CoverageBelief, Provenance, TargetBelief
from perception.belief.view import LocalWorldView


@pytest.fixture(scope="module")
def scene():
    from scripts import run_competing_teams as rct

    e = {"arm": "cbaa", "reds": 4, "family": "multi_bearing", "condition": "REF",
         "attacked": False, "seed": 3901, "design": "v2"}
    e["episode_id"] = rct.episode_id(e)
    sim, _ = rct.build(e, "rule")
    task_ctx = TaskContext.from_simulation(sim, task_set="three_task")
    ctx = bidding.BidContext.from_simulation(sim, task_ctx)
    return sim, task_ctx, ctx


def _state(pos, heading, vel=(0.0, 0.0)):
    return np.array([pos[0], pos[1], vel[0], vel[1], heading], dtype=float)


def _view(sim, ctx, agent, now=10, poses=None, seen=None, targets=None):
    """Agent ``agent`` and teammates at ``poses`` {id: (pos, heading)};
    coverage record ``seen`` {region: last_seen_step} (default never)."""
    centers = np.array([[(i + .5) * 25, (j + .5) * 25] for i in range(4) for j in range(4)])
    cov = CoverageBelief(n_regions=16)
    for r, st in (seen or {}).items():
        cov.observe(r, st)
    me = _state(*poses[agent])
    mates = {j: _state(*p) for j, p in poses.items() if j != agent}
    return LocalWorldView(now=now, dt=sim.dt, agent_id=agent, self_state=me,
                          targets=targets or {}, coverage=cov, coverage_centers=centers,
                          teammates=mates)


def _stations(ctx):
    return {k: (np.asarray(p), h) for k, p, h in ctx.v2.stations}


def _on_stations(ctx):
    return {k: (p, h) for k, (p, h) in _stations(ctx).items()}


def _terms(view, sim, task_ctx, ctx, commitment, mods=None):
    tasks = generate_tasks(view, task_ctx, max_age=None)
    return bidding.bid_terms_v2(view, tasks, ctx, commitment, mods)


# -- F8 posture --------------------------------------------------------------


def test_f8_configured_headings_from_station_origins_cover_more_of_the_prior(scene):
    sim, task_ctx, ctx = scene
    f = ctx.v2.field
    st = _stations(ctx)
    conf = f.mass(f.union(st.values()))
    outward = f.mass(f.union((p, math.atan2(*(p - f.center)[::-1])) for p, _ in st.values()))
    assert conf > outward
    # every configured wedge points into the approach sector (positive coverage)
    for p, h in st.values():
        assert f.mass(f.wedge(p, h)) > 0.5
    # the solver holds them (configured posture), and they are deterministic
    again = cv.configured_station_headings(f, [(k, p) for k, (p, _) in st.items()])
    assert again == sim.configured_station_headings


# -- F1 / F2 / F3 coverage and recon --------------------------------------------


def test_f1_useful_recon_outranks_the_most_redundant_post_not_the_most_essential(scene):
    sim, task_ctx, ctx = scene
    poses = _on_stations(ctx)
    f = ctx.v2.field
    marg = {}
    for k, pose in poses.items():
        others = f.union(p for j, p in poses.items() if j != k)
        marg[k] = f.marginal(pose, others)
    low, high = min(marg, key=marg.get), max(marg, key=marg.get)
    for agent, expect_recon in ((low, True), (high, False)):
        v = _view(sim, ctx, agent, poses=poses)
        t = _terms(v, sim, task_ctx, ctx, f"DEFEND:{agent}")
        best_recon = max(x["bid"] for k, x in t.items() if k.startswith("RECON"))
        assert (best_recon > t[f"DEFEND:{agent}"]["bid"]) is expect_recon, (agent, marg)


def test_f2_refreshed_or_watched_recon_never_wins_by_proximity(scene):
    sim, task_ctx, ctx = scene
    poses = _on_stations(ctx)
    seen_now = {r: 10 for r in range(16)}  # every region refreshed this step
    for agent in poses:
        v = _view(sim, ctx, agent, now=10, poses=poses, seen=seen_now)
        t = _terms(v, sim, task_ctx, ctx, f"DEFEND:{agent}")
        assert all(x["information"] == 0.0 for k, x in t.items() if k.startswith("RECON"))
        if t[f"DEFEND:{agent}"]["coverage"] > 0:
            assert max(x["bid"] for k, x in t.items() if k.startswith("RECON")) < t[f"DEFEND:{agent}"]["bid"]


def test_f3_departure_cost_is_w_leave_times_the_posts_marginal_coverage(scene):
    sim, task_ctx, ctx = scene
    poses = _on_stations(ctx)
    W = V2_PROFILES["balanced"]
    leaves = {}
    for agent in poses:
        v = _view(sim, ctx, agent, poses=poses)
        t = _terms(v, sim, task_ctx, ctx, f"DEFEND:{agent}")
        m = t[f"DEFEND:{agent}"]["coverage"]
        for k, x in t.items():
            if k != f"DEFEND:{agent}":
                assert x["leave"] == pytest.approx(W.w_leave * m)
        assert t[f"DEFEND:{agent}"]["leave"] == 0.0
        leaves[agent] = (m, t["RECON:5"]["leave"])
    ordered = sorted(leaves.values())
    assert all(a[1] <= b[1] for a, b in zip(ordered, ordered[1:]))  # monotone in coverage
    # committed to RECON: nothing is being left
    v = _view(sim, ctx, 0, poses=poses)
    assert all(x["leave"] == 0.0 for x in _terms(v, sim, task_ctx, ctx, "RECON:5").values())


# -- F4 modifiers -----------------------------------------------------------------


def test_f4_profiles_priorities_and_preferences_have_their_documented_effects(scene):
    sim, task_ctx, ctx = scene
    poses = _on_stations(ctx)
    v = _view(sim, ctx, 1, poses=poses)
    base = _terms(v, sim, task_ctx, ctx, "DEFEND:1")
    for prof in ("aggressive", "conservative", "scouting"):
        t = _terms(v, sim, task_ctx, ctx, "DEFEND:1", BidModifiers(scoring_profile=prof))
        W, B = V2_PROFILES[prof], V2_PROFILES["balanced"]
        for k, x in t.items():
            if k.startswith("RECON"):
                assert x["value"] == pytest.approx(base[k]["value"] * W.w_info / B.w_info)
            if k.startswith("DEFEND"):
                assert x["value"] == pytest.approx(W.station_base + W.w_cov * base[k]["coverage"])
    pref = _terms(v, sim, task_ctx, ctx, "DEFEND:1", BidModifiers.build(preferred_tasks=["RECON:5"]))
    assert pref["RECON:5"]["bid"] == pytest.approx(base["RECON:5"]["bid"] + PREFERRED_BONUS)
    # priorities scale the NEUTRALIZE value exactly
    tgt = {0: _belief(0, (50.0, 50.0), (1.4, 1.4), now=10)}
    v = _view(sim, ctx, 1, poses=poses, targets=tgt)
    t0 = _terms(v, sim, task_ctx, ctx, "DEFEND:1")
    for lab in ("LOW", "HIGH"):
        t = _terms(v, sim, task_ctx, ctx, "DEFEND:1", BidModifiers.build(threat_priorities={0: lab}))
        assert t["NEUTRALIZE:0:0"]["value"] == pytest.approx(
            t0["NEUTRALIZE:0:0"]["value"] * PRIORITY_MULTIPLIER[lab])


# -- F5 engagement ------------------------------------------------------------------


def _belief(cid, pos, vel, now, age=0):
    st = np.array([pos[0] - age * 0.1 * vel[0], pos[1] - age * 0.1 * vel[1], vel[0], vel[1],
                   math.atan2(vel[1], vel[0])])
    return TargetBelief(cid, st, Provenance(origin_agent=0, observed_at=now - age,
                                            received_at=now - age, relay_agent=None,
                                            source_sequence=0))


def test_f5_reachable_breach_course_threat_justifies_leaving_a_redundant_post(scene):
    sim, task_ctx, ctx = scene
    poses = _on_stations(ctx)
    f = ctx.v2.field
    marg = {k: f.marginal(p, f.union(q for j, q in poses.items() if j != k)) for k, p in poses.items()}
    low = min(marg, key=marg.get)
    c = f.center
    u = np.array([math.cos(math.radians(225)), math.sin(math.radians(225))])
    p = c + 36.0 * u
    approaching = {0: _belief(0, p, -2.0 * u, now=10)}
    receding = {0: _belief(0, p, 2.0 * u, now=10)}
    stationary = {0: _belief(0, c + 50.0 * u, (0.0, 0.0), now=10)}
    seen_all = {r: 10 for r in range(16)}  # no recon distraction: isolate the engagement term

    def wins(tg):
        v = _view(sim, ctx, low, poses=poses, targets=tg, seen=seen_all)
        t = _terms(v, sim, task_ctx, ctx, f"DEFEND:{low}")
        return t["NEUTRALIZE:0:0"]["bid"] > t[f"DEFEND:{low}"]["bid"], t["NEUTRALIZE:0:0"]

    ok, terms = wins(approaching)
    assert terms["breach_category"] == "breach" and terms["opportunity"] == 1.0
    assert ok
    ok, terms = wins(receding)
    assert terms["breach_category"] == "non_intersecting" and not ok
    ok, terms = wins(stationary)
    assert terms["breach_category"] == "stationary" and not ok


def test_time_to_breach_and_intercept_forecasts():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert cv.time_to_breach((5, -10), (0, 2), square) == (5.0, "breach")
    assert cv.time_to_breach((5, -10), (0, -2), square)[1] == "non_intersecting"
    assert cv.time_to_breach((5, -10), (0, 0), square)[1] == "stationary"
    assert cv.time_to_breach((5, 5), (0, 0), square) == (0.0, "inside")
    assert cv.intercept_time((10, 0), (-2, 0), (0, 0), 2.0) == pytest.approx(2.5)
    assert cv.intercept_time((10, 0), (3, 0), (0, 0), 2.0) is None  # faster, receding


# -- F6 identical scorer in both decentralized arms ------------------------------------


def test_f6_cbaa_and_independent_compute_identical_bids(scene):
    from scripts import run_competing_teams as rct
    from scripts.export_replay import _Quiet

    got = {}
    for arm in ("cbaa", "independent"):
        e = {"arm": arm, "reds": 2, "family": "multi_bearing", "condition": "REF",
             "attacked": False, "seed": 3901, "design": "v2"}
        e["episode_id"] = rct.episode_id(e)
        sim, _ = rct.build(e, "rule")
        captured = []
        orig = sim.cbaa_runtime.start_epoch

        def se(step, views, max_age=None, modifiers=None, _rt=sim.cbaa_runtime, _o=orig, _c=captured):
            _o(step, views, max_age=max_age, modifiers=modifiers)
            if not _c:  # first epoch: identical states in both arms
                _c.append({i: dict(_rt.agents[i].bids) for i in views})

        sim.cbaa_runtime.start_epoch = se
        sim.sim_steps = 1
        with _Quiet():
            sim.run()
        got[arm] = (sim.cbaa_runtime.transport, captured[0])
    assert got["cbaa"][0] != got["independent"][0]
    assert got["cbaa"][1] == got["independent"][1]


# -- historical defaults ---------------------------------------------------------------


def test_v1_is_the_default_and_v2_needs_a_posture(scene):
    from simulation.testbed import Simulation

    sim, task_ctx, ctx = scene
    assert ctx.v2 is not None
    assert bidding.BidContext(100.0, 1.0, 2.0, ((0, 0), (1, 0), (1, 1))).v2 is None
    with pytest.raises(ValueError, match="station posture"):
        Simulation(allocation="cbaa", belief_mode="shared", cbaa_task_set="three_task",
                   cbaa_fallback="station",
                   cbaa_bid_version="v2", station_heading="inherited")


# -- prompts m8b.4 / commander.4 ---------------------------------------------------------

