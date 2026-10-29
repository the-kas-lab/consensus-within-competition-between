"""Deterministic bids: what a task is worth to one agent, from its belief alone.

 extended by A3:

    b_i(t) = w_value * Value(t) - w_eta * ETA_i(t) - w_unc * Uncertainty(t)
             - SwitchCost_i(t) - w_att * AttritionCost_i(t)

Every term is defined and
computed from agent i's frozen ``LocalWorldView`` plus static geometry. A bid is
a pure function of ``(view, task, context, current_task_id, modifiers)``;
nothing else is in scope.

Three things this deliberately does NOT do, each for a stated reason:

* **No term for "another agent is on this contact".** Two blues pursuing one red
  disable exactly one blue (``testbed.py:2099-2109``), and the only measured
  deconfliction rule made the mission worse
. CBAA resolves
  conflicts over *tasks*; nothing here discourages two tasks on one contact.
* **No value for a red escaping.** the escape-scoring rule keeps ``escape_points`` in the
  game score, but a bid that valued it would reward herding reds out of the
  arena -- a scoring artifact turned into behaviour. Contact value is breach
  prevention, scaled by ``neutralize_points``.
* **No tuning on outcomes.** Profile weights are fixed here, before any outcome
  is observed. Tests check that allocations are qualitatively valid, never that
  they win.
"""

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from coordination.tasks import DEFEND, NEUTRALIZE, RECON, SCREEN, closest_point_on_polygon


@dataclass(frozen=True)
class ScoringProfile:
    """One named, fixed weight vector. All quantities are dimensionless:
    distances and uncertainty radii are divided by ``L`` (a quarter of the
    arena), so no weight silently depends on arena size."""

    name: str
    w_value: float = 1.0
    w_eta: float = 0.25
    w_unc: float = 0.1
    w_att: float = 0.5
    screen_weight: float = 0.8
    station_weight: float = 0.15
    recon_weight: float = 0.3
    recon_cap_steps: int = 200
    switch_cost: float = 0.05
    # Leaving a defensive post costs more than leaving a patrol:'s
    # departure cost, merged into the switch cost because a principled one needs
    # M11's coverage floor, which does not exist yet.
    departure_extra: float = 0.05


# Fixed before any outcome was observed (design Sec 3a). Changing a number here
# after seeing results is a researcher degree of freedom, and must be recorded
# as one.
PROFILES: Dict[str, ScoringProfile] = {
    "balanced": ScoringProfile("balanced"),
    "aggressive": ScoringProfile("aggressive", w_att=0.2, screen_weight=0.6),
    "conservative": ScoringProfile(
        "conservative", w_att=0.9, screen_weight=1.0, station_weight=0.25
    ),
    "scouting": ScoringProfile("scouting", recon_weight=0.6),
}

# The only three levels a threat priority can take. A label, never a number:
# forbids the LLM from inventing a floating-point score.
PRIORITY_MULTIPLIER = {"LOW": 0.5, "MEDIUM": 1.0, "HIGH": 2.0}
PREFERRED_BONUS = 0.2


@dataclass(frozen=True)
class BidModifiers:
    """The seam the LLM plugs into at the next step (design Sec 4).

    Bounded by construction: a profile name from ``PROFILES``, a three-level
    label per contact, and a fixed bonus per preferred task. Validated at
    construction so a bad label fails where it was produced, not three calls
    later inside an auction.
    """

    scoring_profile: str = "balanced"
    threat_priorities: Tuple[Tuple[int, str], ...] = ()
    preferred_tasks: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.scoring_profile not in PROFILES:
            raise ValueError(
                f"unknown scoring_profile {self.scoring_profile!r}; "
                f"choose one of {sorted(PROFILES)}"
            )
        for contact_id, label in self.threat_priorities:
            if label not in PRIORITY_MULTIPLIER:
                raise ValueError(
                    f"threat priority for contact {contact_id} must be one of "
                    f"{sorted(PRIORITY_MULTIPLIER)}, got {label!r}"
                )

    @staticmethod
    def build(scoring_profile="balanced", threat_priorities=None, preferred_tasks=None):
        return BidModifiers(
            scoring_profile=scoring_profile,
            threat_priorities=tuple(sorted((int(k), str(v)) for k, v in (threat_priorities or {}).items())),
            preferred_tasks=tuple(sorted(str(t) for t in (preferred_tasks or ()))),
        )

    @property
    def profile(self) -> ScoringProfile:
        return PROFILES[self.scoring_profile]

    def priority(self, contact_id) -> float:
        return PRIORITY_MULTIPLIER[dict(self.threat_priorities).get(int(contact_id), "MEDIUM")]


@dataclass(frozen=True)
class BidContext:
    """Static constants a bid needs. Built once, off a live Simulation.

    ``v2`` carries the static geometry bid version "v2"
    needs; None -- the default -- is bid version "v1", byte-identical to
    before."""

    arena_size: float
    neutralize_points: float
    red_v_max: float
    defense_polygon: Tuple[Tuple[float, float], ...]
    v2: Optional["V2Context"] = field(default=None, compare=False)

    @property
    def length_scale(self) -> float:
        return self.arena_size / 4.0

    @staticmethod
    def from_simulation(sim, task_context):
        v2 = None
        if getattr(sim, "cbaa_bid_version", "v1") in V2_VERSIONS:
            v2 = V2Context.from_simulation(sim, task_context)
        return BidContext(
            arena_size=float(sim.arena_size),
            neutralize_points=float(sim.neutralize_points),
            red_v_max=float(sim.red.v_max),
            defense_polygon=task_context.defense_polygon,
            v2=v2,
        )


def urgency(believed_position, ctx: BidContext) -> float:
    """``1 / (1 + d / L)``, d = distance from the believed position to the
    defended perimeter. Bounded in (0, 1], monotone in closeness. A contact
    inside the region has already breached and ended the episode, so the
    boundary distance is the only case that arises."""
    polygon = np.asarray(ctx.defense_polygon, dtype=float).reshape(-1, 2)
    p = np.asarray(believed_position, dtype=float).reshape(2)
    d = float(np.linalg.norm(p - closest_point_on_polygon(p, polygon)))
    return 1.0 / (1.0 + d / ctx.length_scale)


def bid(view, task, ctx: BidContext, current_task_id: Optional[str] = None,
        modifiers: Optional[BidModifiers] = None) -> float:
    """What ``task`` is worth to the agent whose view this is."""
    if ctx.v2 is not None:
        if ctx.v2.version in V22_FAMILY:
            # The continuity cost depends on whether the pursued contact is in
            # the catalogue, which one task in isolation cannot show.
            raise ValueError(f"bid version {ctx.v2.version} bids need the agent's task catalogue; "
                             "use bids()")
        return bid_terms_v2(view, [task], ctx, current_task_id, modifiers)[task.task_id]["bid"]
    mods = modifiers or BidModifiers()
    prof = mods.profile
    L = ctx.length_scale
    me = np.asarray(view.self_state, dtype=float).ravel()[0:2]

    value = 0.0
    uncertainty = 0.0
    attrition = 0.0
    if task.kind in (NEUTRALIZE, SCREEN):
        belief = view.targets[task.contact_id]
        predicted = belief.predicted_position(view.now, view.dt)
        u = urgency(predicted, ctx) * mods.priority(task.contact_id)
        value = ctx.neutralize_points * u * (1.0 if task.kind == NEUTRALIZE else prof.screen_weight)
        uncertainty = belief.age(view.now) * view.dt * ctx.red_v_max / L
        if task.kind == NEUTRALIZE:
            # A3: winning this removes the winner from the team. Scaled by how
            # few remain, read from the declared teammate channel -- no
            # new information.
            attrition = 1.0 / (1 + len(view.teammates))
    elif task.kind == DEFEND:
        value = prof.station_weight
    elif task.kind == RECON:
        staleness = _cell_staleness(view, task.cell_id, prof.recon_cap_steps)
        value = prof.recon_weight * staleness

    eta = float(np.linalg.norm(me - np.asarray(task.location, dtype=float))) / L

    switch = 0.0
    if current_task_id is not None and task.task_id != current_task_id:
        switch = prof.switch_cost
        if current_task_id.split(":", 1)[0] in (SCREEN, DEFEND):
            switch += prof.departure_extra

    b = (prof.w_value * value - prof.w_eta * eta - prof.w_unc * uncertainty
         - switch - prof.w_att * attrition)
    if task.task_id in mods.preferred_tasks:
        b += PREFERRED_BONUS
    return float(b)


def _cell_staleness(view, cell_id, cap) -> float:
    """Normalised staleness in [0, 1]; never-swept (-1) counts as maximally
    stale, since that is the case RECON most exists for."""
    for region in view.coverage_regions() or []:
        if int(region["region_id"]) == int(cell_id):
            steps = int(region["steps_since_seen"])
            return 1.0 if steps < 0 else min(steps, cap) / float(cap)
    return 0.0


def bids(view, tasks: Sequence, ctx: BidContext, current_task_id=None,
         modifiers: Optional[BidModifiers] = None) -> Dict[str, float]:
    if ctx.v2 is not None:
        return {k: v["bid"] for k, v in bid_terms_v2(view, tasks, ctx, current_task_id,
                                                      modifiers).items()}
    return {t.task_id: bid(view, t, ctx, current_task_id, modifiers) for t in tasks}


# ---------------------------------------------------------------------------
# Bid version "v2" (opt-in): coverage-valued stations,
# a shared departure cost, an information-opportunity proxy for RECON, and
# belief-derived time-to-breach / interception urgency for NEUTRALIZE.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V2Weights:
    """One named v2 weight vector. Values fixed by static geometry checks
    before any v2 episode, never on outcomes."""

    name: str
    station_base: float = 0.05   # readiness value of holding any station
    w_cov: float = 0.5           # x marginal prior-weighted coverage of a station
    w_leave: float = 0.25        # x marginal coverage of the post being left
    w_info: float = 0.45         # x information-opportunity proxy (RECON)
    info_discount_dist: float = 40.0  # units along the route (timeliness)
    w_eta_recon: float = 0.05    # per L of route (RECON travel)
    w_eta: float = 0.15          # per L to the interception point (pursuit / stations)
    w_unc: float = 0.1
    w_att: float = 0.5           # one-for-one attrition, retained
    t0: float = 60.0             # s: urgency 1 / (1 + tau / t0)
    # kappa: a contact keeps (kappa x) the v1 proximity urgency 1/(1 + d/L)
    # whatever its forecast; a forecast breach can only raise urgency. v2.1
    # 0.5 -> 1.0 after the verification runs showed pursuits
    # dropped the moment an evading red's instantaneous velocity stopped
    # pointing at the region.
    proximity_floor: float = 1.0
    t1: float = 10.0             # s: opportunity decays over t1 past the forecast breach
    o_min: float = 0.25          # opportunity when no interception solution exists
    switch_cost: float = 0.05
    staleness_cap_steps: int = 200
    # v2.2 only: the threat value
    # every believed red carries whatever its heading, priority, urgency or
    # opportunity (U1), and the continuity cost on every alternative to a
    # still-valid pursuit. Values: V22_PROFILES.
    threat_value: float = 0.0
    pursuit_continuity: float = 0.0


V2_VERSIONS = ("v2", "v2.2", "v2.3")
# v2.3 = v2.2 with conditional pursuit continuity; both use the
# v2.2 forecasts, weights and catalogue rules.
V22_FAMILY = ("v2.2", "v2.3")

V2_PROFILES: Dict[str, V2Weights] = {
    "balanced": V2Weights("balanced"),
    "aggressive": V2Weights("aggressive", w_att=0.2, t0=90.0),
    "conservative": V2Weights("conservative", w_cov=0.8, w_leave=0.4, w_att=0.9),
    "scouting": V2Weights("scouting", w_info=0.75),
}
assert set(V2_PROFILES) == set(PROFILES)

# Bid version "v2.2": v2.1's weights plus the threat value and
# pursuit continuity, the same for every profile. Set by the
# procedure (controlled cases and bid traces on spent seeds 3901-3902 only),
# frozen in sec 10.10 before the verification run.
V22_THREAT_VALUE = 1.2
V22_PURSUIT_CONTINUITY = 0.15
V22_PROFILES: Dict[str, V2Weights] = {
    k: replace(w, threat_value=V22_THREAT_VALUE, pursuit_continuity=V22_PURSUIT_CONTINUITY)
    for k, w in V2_PROFILES.items()
}


# Bid version "v2.3": the extra
# continuity cost of switching from a pursued contact a to another contact b
# is reduced to h_p x (1 - R(a) H(b) S(b)): R = a is receding (its funnel
# closing speed below -V23_RECEDE_SPEED reaches 1), H = b's forecast breach
# lies within the remaining episode (ramp V23_HORIZON_RAMP_S), S = the
# agent's own optimistic interception margin on b over the velocity box
# (ramp V23_MARGIN_RAMP_S). Declared from existing scales, not fitted.
V23_RECEDE_SPEED = 0.5
V23_HORIZON_RAMP_S = 5.0
V23_MARGIN_RAMP_S = 2.0


def v2_profiles(version: str) -> Dict[str, V2Weights]:
    return V22_PROFILES if version in V22_FAMILY else V2_PROFILES


@dataclass(frozen=True)
class V2Context:
    """Static geometry for v2 bids: the approach-prior lattice, each station's
    position and the heading the run's posture gives it, the episode horizon
    and the sensor wedge. Built once per episode."""

    field: object  # coordination.coverage_value.ApproachField
    stations: Tuple[Tuple[int, Tuple[float, float], float], ...]  # (id, pos, heading)
    horizon_steps: int
    dt: float
    blue_speed: float
    # v2.2: the version, the approach funnel (convex hull of
    # the polygon and the public gates' arrival disks), both teams' per-axis
    # bounds and the contact radius. Unused under "v2".
    version: str = "v2"
    funnel: Optional[np.ndarray] = field(default=None, compare=False)
    blue_v_max: float = 2.0
    blue_a_max: float = 2.0
    red_v_max: float = 2.0
    contact_radius: float = 0.4

    @staticmethod
    def from_simulation(sim, task_context):
        from coordination import coverage_value as cv

        f = cv.approach_field(sim.defense_center, task_context.defense_polygon, sim.arena_min,
                              sim.arena_max, sim.fov_half_angle_rad, sim.fov_range)
        headings = sim.station_heading_angles(task_context.stations)
        version = getattr(sim, "cbaa_bid_version", "v2")
        funnel = None
        if version in V22_FAMILY:
            funnel = cv.approach_funnel(task_context.defense_polygon, *public_gates(sim))
        return V2Context(field=f,
                         stations=tuple((int(k), (float(p[0]), float(p[1])), float(headings[int(k)]))
                                        for k, p in task_context.stations),
                         horizon_steps=int(sim.sim_steps), dt=float(sim.dt),
                         blue_speed=cv.GUARANTEED_SPEED, version=version, funnel=funnel,
                         blue_v_max=float(sim.blue_v_max), blue_a_max=float(sim.blue_a_max),
                         red_v_max=float(sim.red_v_max),
                         contact_radius=float(sim.contact_tolerance))


def public_gates(sim):
    """(gate positions, arrival radius): the episode's gates when it has an
    intruder team, else the same public construction (control.gates)."""
    from control.gates import build_gates

    gates = getattr(sim, "gates", None) or build_gates(
        sim.defense_center, sim.defended_shape_points, sim.arena_min, sim.arena_max)
    return [g.position for g in gates], float(gates[0].arrival_radius)


def _pose(state):
    s = np.asarray(state, dtype=float).ravel()
    return s[0:2].copy(), (float(s[4]) if s.size >= 5 else 0.0)


def bid_terms_v2(view, tasks: Sequence, ctx: BidContext, current_task_id=None,
                 modifiers: Optional[BidModifiers] = None) -> Dict[str, Dict[str, float]]:
    """Every task's v2 bid AND its terms (the prompt shows the terms). One
    frozen view in, nothing else: teammates' CURRENT geometric poses (not their
    intent), static station geometry, the approach prior, this agent's own
    commitment and coverage record."""
    from coordination import coverage_value as cv

    v2 = ctx.v2
    v22 = v2.version in V22_FAMILY
    v23 = v2.version == "v2.3"
    mods = modifiers or BidModifiers()
    W = v2_profiles(v2.version)[mods.scoring_profile]
    f = v2.field
    L = ctx.length_scale
    me, my_heading = _pose(view.self_state)
    my_vel = np.asarray(view.self_state, dtype=float).ravel()[2:4]
    mates = {int(j): _pose(s) for j, s in (view.teammates or {}).items()}
    others = f.union(mates.values())
    watched = others | f.wedge(me, my_heading)
    stations = {k: (np.asarray(p), h) for k, p, h in v2.stations}

    def marginal_station(k):
        return f.marginal(stations[k], others)

    # The post being left: the DEFEND commitment's station, else (no
    # commitment) the agent's own fallback station; none when committed to
    # RECON / NEUTRALIZE.
    post = None
    if current_task_id is None:
        post = int(view.agent_id) if int(view.agent_id) in stations else None
    elif current_task_id.split(":", 1)[0] == DEFEND:
        post = int(current_task_id.split(":")[1])
    m_post = marginal_station(post) if post is not None else 0.0

    remaining_s = max(0.0, (v2.horizon_steps - int(view.now)) * v2.dt)
    reach = v2.blue_speed * remaining_s
    # v2.2 continuity: only while the pursued contact still
    # has a task in this catalogue (believed, fresh, unresolved).
    pursued = None
    if v22 and current_task_id is not None and current_task_id.split(":", 1)[0] == NEUTRALIZE:
        c = int(current_task_id.split(":")[1])
        if any(t.kind == NEUTRALIZE and int(t.contact_id) == c for t in tasks):
            pursued = c
    recede = 0.0
    if v23 and pursued is not None:
        # R(a): the pursued contact recedes from the approach funnel (own belief).
        from coordination import coverage_value as cv

        ba = view.targets[pursued]
        ca = cv.funnel_forecast(ba.predicted_position(view.now, view.dt), np.asarray(ba.velocity, float),
                                v2.funnel, ctx.defense_polygon,
                                math.sqrt(2.0) * v2.red_v_max)["closing_speed"]
        recede = float(np.clip(-ca / V23_RECEDE_SPEED, 0.0, 1.0))
    staleness = None
    if any(t.kind == RECON for t in tasks):
        regions = view.coverage_regions() or []
        centers = np.array([r["center"] for r in regions], dtype=float)
        cap = float(W.staleness_cap_steps)
        staleness = np.array([1.0 if int(r["steps_since_seen"]) < 0
                              else min(int(r["steps_since_seen"]), cap) / cap for r in regions])

    out = {}
    for t in tasks:
        terms = {"value": 0.0, "travel": 0.0, "uncertainty": 0.0, "attrition": 0.0,
                 "leave": 0.0, "switch": 0.0, "preferred": 0.0}
        leaving = not (t.kind == DEFEND and post is not None and int(t.station_id) == post)
        if t.kind == SCREEN:
            raise ValueError("bid version v2 is defined for the three-task model only (no SCREEN)")
        if t.kind == DEFEND:
            k = int(t.station_id)
            m = marginal_station(k)
            terms["value"] = W.station_base + W.w_cov * m
            terms["coverage"] = m
            d = float(np.hypot(*(np.asarray(t.location) - me)))
            terms["travel"] = W.w_eta * d / L
        elif t.kind == RECON:
            if v22:
                # optimistic travel forecast along the straight route (sec 10.3)
                u_r = np.asarray(t.location, dtype=float) - me
                n_r = float(np.hypot(*u_r))
                reach = (cv.route_reach(my_vel, u_r / n_r, v2.blue_v_max, v2.blue_a_max,
                                        remaining_s, v2.dt) if n_r > 1e-9 else 0.0)
            first = f.swept(me, t.location, reach, my_heading)
            info = f.information(first, watched, staleness, centers, W.info_discount_dist)
            terms["value"] = W.w_info * info
            terms["information"] = info
            d = float(np.hypot(*(np.asarray(t.location) - me)))
            terms["travel"] = W.w_eta_recon * d / L
        elif t.kind == NEUTRALIZE:
            belief = view.targets[t.contact_id]
            p = belief.predicted_position(view.now, view.dt)
            v = np.asarray(belief.velocity, dtype=float)
            if v22:
                terms.update(_neutralize_v22(p, v, me, my_vel, ctx, W, remaining_s,
                                             mods.priority(t.contact_id),
                                             margin_ramp=V23_MARGIN_RAMP_S if v23 else None))
                terms["uncertainty"] = W.w_unc * belief.age(view.now) * view.dt * ctx.red_v_max / L
                terms["attrition"] = W.w_att * (1.0 / (1 + len(view.teammates)))
            else:
                _neutralize_v2(terms, p, v, me, ctx, W, v2, L, mods.priority(t.contact_id))
                terms["uncertainty"] = W.w_unc * belief.age(view.now) * view.dt * ctx.red_v_max / L
                terms["attrition"] = W.w_att * (1.0 / (1 + len(view.teammates)))
        if leaving:
            terms["leave"] = W.w_leave * m_post
        if current_task_id is not None and t.task_id != current_task_id:
            terms["switch"] = W.switch_cost
        if pursued is not None and not (t.kind == NEUTRALIZE and int(t.contact_id) == pursued):
            terms["continuity"] = W.pursuit_continuity
            if v23 and t.kind == NEUTRALIZE:
                tau = terms["time_to_breach"]
                h = (float(np.clip((remaining_s - tau) / V23_HORIZON_RAMP_S, 0.0, 1.0))
                     if math.isfinite(tau) else 0.0)
                relief = recede * h * terms["own_margin_score"]
                terms["continuity_relief"] = relief
                terms["continuity"] = W.pursuit_continuity * (1.0 - relief)
        if t.task_id in mods.preferred_tasks:
            terms["preferred"] = PREFERRED_BONUS
        terms["bid"] = float(terms["value"] - terms["travel"] - terms["uncertainty"]
                             - terms["attrition"] - terms["leave"] - terms["switch"]
                             - terms.get("continuity", 0.0) + terms["preferred"])
        out[t.task_id] = terms
    return out


def _neutralize_v2(terms, p, v, me, ctx, W, v2, L, priority):
    """Bid version v2 / v2.1 NEUTRALIZE value and travel, unchanged."""
    from coordination import coverage_value as cv

    tau, cat = cv.time_to_breach(p, v, ctx.defense_polygon)
    d_poly = float(np.linalg.norm(np.asarray(p) - closest_point_on_polygon(
        p, np.asarray(ctx.defense_polygon))))
    u_time = 1.0 / (1.0 + tau / W.t0) if math.isfinite(tau) else 0.0
    u = max(u_time, W.proximity_floor / (1.0 + d_poly / L))
    ti = cv.intercept_time(p, v, me, v2.blue_speed)
    if ti is None:
        o = W.o_min
        d_int = float(np.hypot(*(np.asarray(p) - me)))
    else:
        o = 1.0 if ti <= tau else max(0.0, 1.0 - (ti - tau) / W.t1)
        d_int = v2.blue_speed * ti
    terms["value"] = ctx.neutralize_points * u * o * priority
    terms.update(time_to_breach=tau, breach_category=cat, urgency=u, opportunity=o,
                 intercept_time=(math.inf if ti is None else ti))
    terms["travel"] = W.w_eta * d_int / L


def _neutralize_v22(p, v, me, my_vel, ctx, W, remaining_s, priority,
                    margin_ramp=None) -> Dict[str, float]:
    """Bid version v2.2 NEUTRALIZE value and travel:
    points x (threat value + priority x urgency x opportunity). Urgency from
    the gate-aware funnel forecast with the v2.1 proximity floor; opportunity
    from the optimistic kinematic interception forecast, monotone in the
    forecast interception time; the threat value is untouched by heading,
    priority, urgency and opportunity."""
    from coordination import coverage_value as cv

    v2 = ctx.v2
    L = ctx.length_scale
    fc = cv.funnel_forecast(p, v, v2.funnel, ctx.defense_polygon, math.sqrt(2.0) * v2.red_v_max)
    tau = fc["time_to_breach"]
    d_poly = float(np.linalg.norm(np.asarray(p) - closest_point_on_polygon(
        p, np.asarray(ctx.defense_polygon))))
    u_time = 1.0 / (1.0 + tau / W.t0) if math.isfinite(tau) else 0.0
    u = max(u_time, W.proximity_floor / (1.0 + d_poly / L))
    k_max = int(round(remaining_s / v2.dt)) + int(round(W.t1 / v2.dt))
    # Optimistic kinematic interception forecast for every velocity in the
    # declared uncertainty box; opportunity is the mean.
    box = cv.velocity_box(v, v2.red_v_max)
    times = cv.reach_intercept_times(me, my_vel, p, box, v2.blue_v_max, v2.blue_a_max, v2.dt,
                                     k_max, v2.contact_radius)
    o = float(np.mean([cv.opportunity_v22(None if not math.isfinite(x) else float(x), tau,
                                          remaining_s, W.t1, W.o_min) for x in times]))
    ti = float(times[len(times) // 2])  # the believed velocity itself (box centre)
    ti = None if not math.isfinite(ti) else ti
    # Travel to the contact's current forecast position: continuous in its
    # velocity (sec 10.12; the interception point jumps when a window closes).
    d_int = float(np.hypot(*(np.asarray(p) - me)))
    out = {
        "value": ctx.neutralize_points * (W.threat_value + priority * u * o),
        "threat_value": ctx.neutralize_points * W.threat_value,
        "time_to_breach": tau, "breach_category": fc["status"], "closing_speed": fc["closing_speed"],
        "urgency": u, "opportunity": o, "intercept_time": (math.inf if ti is None else ti),
        "travel": W.w_eta * d_int / L,
    }
    if margin_ramp is not None:
        # v2.3 S(b): own optimistic interception margin on
        # the forecast breach, ramped over margin_ramp and averaged over the
        # same velocity box as O; 0 without a finite forecast breach. Only
        # v2.3 bids ask for it, so v2.2 callers get the same dict as before.
        out["own_margin_score"] = (float(np.mean(np.clip((tau - np.asarray(times)) / margin_ramp,
                                                         0.0, 1.0)))
                                   if math.isfinite(tau) else 0.0)
    return out
