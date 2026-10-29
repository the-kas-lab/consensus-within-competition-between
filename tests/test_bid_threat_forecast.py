"""Bid version "v2.2", m8b.5 / commander.5 and the rule mission-policy
commander.

Static geometry from a BUILT (never run, except where a test says so)
multi_bearing episode on spent seed 3901; views are constructed by hand so each
criterion is checked in isolation. No model calls."""

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest

from coordination import bidding, coverage_value as cv
from coordination.bidding import PRIORITY_MULTIPLIER, BidModifiers, V22_PROFILES
from coordination.cbaa import CBAAAgent, run_rounds
from coordination.tasks import NEUTRALIZE, TaskContext, generate_tasks
from perception.belief.records import CoverageBelief, Provenance, TargetBelief
from perception.belief.view import LocalWorldView

NOW = 100


@pytest.fixture(scope="module")
def scene():
    from scripts import run_competing_teams as rct

    e = {"arm": "cbaa", "reds": 4, "family": "multi_bearing", "condition": "REF",
         "attacked": False, "seed": 3901, "design": "v2.2"}
    e["episode_id"] = rct.episode_id(e)
    sim, _ = rct.build(e, "rule")
    task_ctx = TaskContext.from_simulation(sim, task_set="three_task")
    ctx = bidding.BidContext.from_simulation(sim, task_ctx)
    return sim, task_ctx, ctx


def _belief(cid, pos, vel, now=NOW, age=0):
    st = np.array([pos[0] - age * 0.1 * vel[0], pos[1] - age * 0.1 * vel[1], vel[0], vel[1],
                   math.atan2(vel[1], vel[0])])
    return TargetBelief(cid, st, Provenance(origin_agent=0, observed_at=now - age,
                                            received_at=now - age, relay_agent=None,
                                            source_sequence=0))


def _coverage(ctx, now=NOW):
    """Realistic stale coverage (sec 10.8): regions whose centre a station
    wedge holds are seen now, all others never (no scouting)."""
    f = ctx.v2.field
    centers = np.array([[(i + .5) * 25, (j + .5) * 25] for i in range(4) for j in range(4)])
    cov = CoverageBelief(n_regions=16)
    for r, c in enumerate(centers):
        for _, p, h in ctx.v2.stations:
            d = c - np.asarray(p)
            ang = (math.atan2(d[1], d[0]) - h + math.pi) % (2 * math.pi) - math.pi
            if np.hypot(*d) <= f.range and abs(ang) <= f.half_angle:
                cov.observe(r, now)
    return cov, centers


def _team(ctx, scout=False):
    """{agent: (state, commitment)}: four blues on configured stations, or the
    least-covering one out on RECON 30 units along 225 deg."""
    f = ctx.v2.field
    st = {k: (np.asarray(p), h) for k, p, h in ctx.v2.stations}
    out = {k: (np.array([p[0], p[1], 0.0, 0.0, h]), f"DEFEND:{k}") for k, (p, h) in st.items()}
    if scout:
        marg = {k: f.marginal(st[k], f.union(q for j, q in st.items() if j != k)) for k in st}
        k = min(marg, key=marg.get)
        u = np.array([math.cos(math.radians(225)), math.sin(math.radians(225))])
        p = st[k][0] + 30.0 * u
        out[k] = (np.array([p[0], p[1], 2 * u[0], 2 * u[1], math.radians(225)]), "RECON:0")
    return out


def _views(sim, ctx, team, targets, now=NOW):
    cov, centers = _coverage(ctx, now)
    return {i: LocalWorldView(now=now, dt=sim.dt, agent_id=i, self_state=s, targets=targets,
                              coverage=cov, coverage_centers=centers,
                              teammates={j: t for j, (t, _) in team.items() if j != i})
            for i, (s, _) in team.items()}


def _cbaa(sim, ctx, task_ctx, team, targets, mods=None):
    agents = []
    for i, v in _views(sim, ctx, team, targets).items():
        tasks = generate_tasks(v, task_ctx, max_age=None)
        b = bidding.bids(v, tasks, ctx, current_task_id=team[i][1], modifiers=mods)
        a = CBAAAgent(i, stable_rounds=3)
        a.start_epoch(0, b, [t.task_id for t in tasks])
        agents.append(a)
    ids = [a.agent_id for a in agents]
    run_rounds(agents, {i: [j for j in ids if j != i] for i in ids}, rounds=12)
    return {a.agent_id: a.x for a in agents}


def _engaged(assign, cid):
    return any(x is not None and x.startswith(f"{NEUTRALIZE}:{cid}:") for x in assign.values())


def _red(sim, ctx, bearing, dist, status, speed):
    from control.gates import _ray_exit

    c = ctx.v2.field.center
    u = np.array([math.cos(math.radians(bearing)), math.sin(math.radians(bearing))])
    r0 = float(np.linalg.norm(_ray_exit(c, u, np.asarray(ctx.defense_polygon)) - c))
    p = c + (r0 + dist) * u
    gates = np.array(bidding.public_gates(sim)[0])
    g = gates[np.argmin(np.linalg.norm(gates - p, axis=1))]
    to_g = (g - p) / np.linalg.norm(g - p)
    d = {"approaching": to_g, "moving_away": -to_g, "crossing": np.array([-to_g[1], to_g[0]]),
         "stationary": np.zeros(2)}[status]
    v = speed * d
    m = max(abs(v[0]), abs(v[1]), 1e-12)
    return p, (v * (2.0 / m) if m > 2.0 else v)


def _grid(sim, ctx, bearings=(180, 220, 260), dists=(10, 50, 90)):
    for b, d, (status, sp) in itertools.product(
            bearings, dists, (("approaching", 2.0), ("moving_away", 2.0), ("moving_away", 1.0),
                              ("crossing", 2.0), ("stationary", 0.0))):
        p, v = _red(sim, ctx, b, d, status, sp)
        if 1 <= p[0] <= 99 and 1 <= p[1] <= 99:
            yield (b, d, status, sp), p, v


# -- frozen parameters ---------------------------------------------------------------


def test_v22_parameters_are_the_frozen_values_and_v2_is_unchanged():
    for name, w in V22_PROFILES.items():
        assert w.threat_value == 1.2 and w.pursuit_continuity == 0.15
        assert replace(w, threat_value=0.0, pursuit_continuity=0.0) == bidding.V2_PROFILES[name]
        assert bidding.V2_PROFILES[name].threat_value == 0.0


# -- I-exact, O-mono, forecast ---------------------------------------------------------


def test_axis_reach_equals_extreme_rollouts_and_contains_random_ones():
    rng = np.random.default_rng(0)
    for v0 in (-2.0, -0.7, 0.0, 1.3, 2.0):
        lo, hi = cv.axis_reach(5.0, v0, 2.0, 2.0, 0.1, 60)
        for sign, bound in ((1, hi), (-1, lo)):
            p, v = 5.0, v0
            for k in range(1, 61):
                v = float(np.clip(v + 0.1 * sign * 2.0, -2.0, 2.0))
                p += 0.1 * v
                assert p == pytest.approx(bound[k])
        for _ in range(50):
            p, v = 5.0, v0
            for k in range(1, 61):
                a = rng.uniform(-2.0, 2.0)
                v = float(np.clip(v + 0.1 * a, -2.0, 2.0))
                p += 0.1 * v
                assert lo[k] - 1e-9 <= p <= hi[k] + 1e-9


def test_reach_intercept_times_match_the_single_target_forecast():
    vels = np.array([[-1.0, 0.0], [0.0, 0.0], [2.0, 0.0], [-2.0, 1.0]])
    many = cv.reach_intercept_times((0, 0), (0, 0), (10, 0), vels, 2.0, 2.0, 0.1, 400, 0.4)
    for v, t in zip(vels, many):
        one = cv.reach_intercept((0, 0), (0, 0), (10, 0), v, 2.0, 2.0, 0.1, 400, 0.4)
        assert (one is None and not math.isfinite(t)) or one[0] == pytest.approx(t)
    assert not math.isfinite(many[2])  # same per-axis bound, running straight away


def test_opportunity_is_monotone_continuous_and_none_is_the_floor():
    for tau in (5.0, 20.0, math.inf):
        ts = np.linspace(0.0, 60.0, 601)
        o = [cv.opportunity_v22(t, tau, 30.0, 10.0, 0.25) for t in ts]
        assert all(a >= b - 1e-12 for a, b in zip(o, o[1:]))
        assert min(o) >= cv.opportunity_v22(None, tau, 30.0, 10.0, 0.25) - 1e-12
        assert max(abs(a - b) for a, b in zip(o, o[1:])) <= 0.75 * 0.1 / 10.0 + 1e-9
    assert cv.opportunity_v22(40.0, math.inf, 30.0, 10.0, 0.25) == pytest.approx(0.25)
    assert cv.opportunity_v22(29.0, math.inf, 30.0, 10.0, 0.25) == 1.0  # inside the episode


def test_funnel_forecast_is_gate_aware_and_continuous_in_heading(scene):
    sim, task_ctx, ctx = scene
    gates = np.array(bidding.public_gates(sim)[0])
    poly = ctx.defense_polygon
    p = gates[2] + np.array([-40.0, -40.0])
    to_g = (gates[2] - p) / np.linalg.norm(gates[2] - p)
    # aimed at a gate: v2's polygon ray misses, the funnel forecast does not
    assert cv.time_to_breach(p, 2 * to_g, poly)[1] in ("non_intersecting", "breach")
    fc = cv.funnel_forecast(p, 2 * to_g, ctx.v2.funnel, poly, 2 * math.sqrt(2))
    assert fc["status"] == cv.APPROACHING and 25.0 < fc["time_to_breach"] < 35.0
    assert cv.funnel_forecast(p, -2 * to_g, ctx.v2.funnel, poly, 2.83)["status"] == cv.MOVING_AWAY
    # the urgency the bid uses, max(1/(1 + tau/60), proximity floor), changes
    # little per degree of heading: the time term's step at the closing-speed
    # threshold (c_min 0.05 -> about 0.05) lies below the proximity floor
    from coordination.tasks import closest_point_on_polygon

    floor = 1.0 / (1.0 + float(np.linalg.norm(p - closest_point_on_polygon(p, poly))) / 25.0)
    us = []
    for deg in range(-90, 91):
        a = math.radians(deg)
        v = 2 * np.array([to_g[0] * math.cos(a) - to_g[1] * math.sin(a),
                          to_g[0] * math.sin(a) + to_g[1] * math.cos(a)])
        tau = cv.funnel_forecast(p, v, ctx.v2.funnel, poly, 2.83)["time_to_breach"]
        us.append(max(floor, 0.0 if not math.isfinite(tau) else 1.0 / (1.0 + tau / 60.0)))
    assert max(abs(a - b) for a, b in zip(us, us[1:])) < 0.03


# -- U1 (strong incentive) -------------------------------------------------------------


@pytest.mark.parametrize("scout", [False, True])
def test_u1a_one_believed_red_is_engaged_whatever_its_course_or_priority(scene, scout):
    sim, task_ctx, ctx = scene
    team = _team(ctx, scout=scout)
    for key, p, v in _grid(sim, ctx):
        for prio in ("LOW", "MEDIUM", "HIGH"):
            mods = BidModifiers.build(threat_priorities={0: prio})
            assert _engaged(_cbaa(sim, ctx, task_ctx, team, {0: _belief(0, p, v)}, mods), 0), \
                (key, prio, scout)


def test_u1b_two_believed_reds_are_both_engaged(scene):
    sim, task_ctx, ctx = scene
    cases = list(_grid(sim, ctx, bearings=(180, 260), dists=(10, 90)))
    for (ka, pa, va), (kb, pb, vb) in itertools.combinations(cases, 2):
        if np.linalg.norm(pa - pb) < 1.0:
            continue
        asg = _cbaa(sim, ctx, task_ctx, _team(ctx),
                    {0: _belief(0, pa, va), 1: _belief(1, pb, vb)})
        assert _engaged(asg, 0) and _engaged(asg, 1), (ka, kb)


def test_p1_reachable_approaching_red_beats_the_most_valuable_recon(scene):
    sim, task_ctx, ctx = scene
    team = _team(ctx)
    p, v = _red(sim, ctx, 220, 50, "approaching", 2.0)
    views = _views(sim, ctx, team, {0: _belief(0, p, v)})
    best = None
    for i, vw in views.items():
        t = bidding.bid_terms_v2(vw, generate_tasks(vw, task_ctx), ctx, team[i][1])
        if best is None or t["NEUTRALIZE:0:0"]["bid"] > best[1]["NEUTRALIZE:0:0"]["bid"]:
            best = (i, t)
    t = best[1]
    assert math.isfinite(t["NEUTRALIZE:0:0"]["intercept_time"])
    assert t["NEUTRALIZE:0:0"]["bid"] > max(x["bid"] for k, x in t.items() if k.startswith("RECON"))


def test_priority_scales_only_urgency_times_opportunity(scene):
    sim, task_ctx, ctx = scene
    team = _team(ctx)
    p, v = _red(sim, ctx, 220, 50, "approaching", 2.0)
    vw = _views(sim, ctx, team, {0: _belief(0, p, v)})[3]
    tasks = generate_tasks(vw, task_ctx)
    base = bidding.bid_terms_v2(vw, tasks, ctx, "DEFEND:3")["NEUTRALIZE:0:0"]
    for lab, mult in PRIORITY_MULTIPLIER.items():
        t = bidding.bid_terms_v2(vw, tasks, ctx, "DEFEND:3",
                                 BidModifiers.build(threat_priorities={0: lab}))["NEUTRALIZE:0:0"]
        assert t["value"] == pytest.approx(1.2 + mult * base["urgency"] * base["opportunity"])


# -- P2 smoothness ------------------------------------------------------------------------


def test_p2_small_velocity_changes_move_a_neutralize_bid_by_at_most_0_15(scene):
    sim, task_ctx, ctx = scene
    team = _team(ctx)

    def nb(agent, p, v):
        vw = _views(sim, ctx, team, {0: _belief(0, p, v)})[agent]
        neu = [t for t in generate_tasks(vw, task_ctx) if t.kind == NEUTRALIZE]
        return bidding.bids(vw, neu, ctx, current_task_id=team[agent][1])["NEUTRALIZE:0:0"]

    worst = 0.0
    for _, p, v in _grid(sim, ctx):
        for agent in (0, 2, 3):
            base = nb(agent, p, v)
            for d in ((0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)):
                worst = max(worst, abs(nb(agent, p, np.clip(v + np.array(d), -2, 2)) - base))
    assert worst <= 0.15, worst


# -- continuity P3'-P6 ---------------------------------------------------------------------


def _pursuit_terms(sim, ctx, task_ctx, targets, commitment="NEUTRALIZE:0:0", agent=2, slots=1):
    team = _team(ctx)
    team[agent] = (team[agent][0], commitment)
    vw = _views(sim, ctx, team, targets)[agent]
    tc = replace(task_ctx, slots=slots)
    return bidding.bid_terms_v2(vw, generate_tasks(vw, tc), ctx, commitment)


def test_p3_p5_continuity_only_while_the_pursued_contact_is_believed(scene):
    sim, task_ctx, ctx = scene
    p, v = _red(sim, ctx, 220, 50, "approaching", 2.0)
    t = _pursuit_terms(sim, ctx, task_ctx, {0: _belief(0, p, v)})
    assert t["NEUTRALIZE:0:0"].get("continuity", 0.0) == 0.0
    assert all(x["continuity"] == 0.15 for k, x in t.items() if not k.startswith("NEUTRALIZE:0:"))
    # resolved / absent: no task for it, so no continuity cost anywhere
    t = _pursuit_terms(sim, ctx, task_ctx, {})
    assert all(x.get("continuity", 0.0) == 0.0 for x in t.values())
    with pytest.raises(ValueError, match="catalogue"):
        bidding.bid(None, None, ctx)


def test_p3_a_moving_away_red_stays_its_pursuers_best_task_when_alone(scene):
    sim, task_ctx, ctx = scene
    for dist in (30, 70):
        p, v = _red(sim, ctx, 220, dist, "moving_away", 2.0)
        t = _pursuit_terms(sim, ctx, task_ctx, {0: _belief(0, p, v)})
        assert max(t, key=lambda k: t[k]["bid"]) == "NEUTRALIZE:0:0", dist


def test_p4_team_engages_a_more_urgent_second_red(scene):
    sim, task_ctx, ctx = scene
    pa, va = _red(sim, ctx, 260, 50, "moving_away", 2.0)
    for b, d in ((180, 10), (220, 30), (200, 50)):
        pb, vb = _red(sim, ctx, b, d, "approaching", 2.0)
        team = _team(ctx)
        team[2] = (team[2][0], "NEUTRALIZE:0:0")
        asg = _cbaa(sim, ctx, task_ctx, team, {0: _belief(0, pa, va), 1: _belief(1, pb, vb)})
        assert _engaged(asg, 0) and _engaged(asg, 1), (b, d)


def test_p6_slot_change_of_the_same_contact_pays_no_continuity(scene):
    sim, task_ctx, ctx = scene
    p, v = _red(sim, ctx, 220, 50, "approaching", 2.0)
    t = _pursuit_terms(sim, ctx, task_ctx, {0: _belief(0, p, v)}, slots=2)
    assert t["NEUTRALIZE:0:1"].get("continuity", 0.0) == 0.0
    assert t["NEUTRALIZE:0:1"]["switch"] == V22_PROFILES["balanced"].switch_cost


# -- F1 / F3 under v2.2, P7 = F6 ------------------------------------------------------------


def test_f1_f3_hold_under_v22(scene):
    sim, task_ctx, ctx = scene
    f = ctx.v2.field
    st = {k: (np.asarray(p), h) for k, p, h in ctx.v2.stations}
    marg = {k: f.marginal(st[k], f.union(q for j, q in st.items() if j != k)) for k in st}
    low, high = min(marg, key=marg.get), max(marg, key=marg.get)
    team = _team(ctx)
    views = _views(sim, ctx, team, {}, now=10)
    for agent, expect in ((low, True), (high, False)):
        cov = CoverageBelief(n_regions=16)  # episode start: nothing seen
        vw = replace(views[agent], coverage=cov)
        t = bidding.bid_terms_v2(vw, generate_tasks(vw, task_ctx), ctx, f"DEFEND:{agent}")
        assert (max(x["bid"] for k, x in t.items() if k.startswith("RECON"))
                > t[f"DEFEND:{agent}"]["bid"]) is expect
        for k, x in t.items():
            if k != f"DEFEND:{agent}":
                assert x["leave"] == pytest.approx(V22_PROFILES["balanced"].w_leave * marg[agent])


def test_p7_cbaa_and_independent_compute_identical_v22_bids():
    from scripts import run_competing_teams as rct
    from scripts.export_replay import _Quiet

    got = {}
    for arm in ("cbaa", "independent"):
        e = {"arm": arm, "reds": 2, "family": "multi_bearing", "condition": "REF",
             "attacked": False, "seed": 3901, "design": "v2.2"}
        e["episode_id"] = rct.episode_id(e)
        sim, _ = rct.build(e, "rule")
        assert sim.cbaa_bid_version == "v2.2"
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


# -- commander team -----------------------------------------------------------------------


def _commander_geo(sim, task_ctx):
    from coordination.commander_llm import commander_geometry

    return commander_geometry(sim, task_ctx)


# -- prompts m8b.5 / commander.5 ---------------------------------------------------------------

