/* Pure geometry and episode semantics for the replay viewer.
 *
 * No DOM, no canvas: everything here is a function of a bundle episode and a
 * (fractional) frame, so the renderer and the Node/Chrome checks call exactly
 * the same code. Where a quantity mirrors a Python computation, the expression
 * order is kept identical so the floats agree bit for bit (belief prediction,
 * closest point on the guard polygon).
 *
 * World coordinates are the simulator's: x right, y up, angles atan2-style
 * (counter-clockwise from +x). The screen transform flips y.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else (root.PEV = root.PEV || {}).geom = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const TAU = Math.PI * 2;
  const lerp = (a, b, f) => a + (b - a) * f;
  const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

  /* ---- time ------------------------------------------------------------ */

  // Frame k is the world at the start of control step k; frame n is terminal.
  // Discrete fields (roles, targets, beliefs) belong to step floor(f), clamped
  // to the last recorded step once the playhead reaches the terminal frame.
  function split(ep, f) {
    const last = ep.dims.n_frames - 1;
    const fc = Math.max(0, Math.min(last, f));
    const k = Math.min(Math.floor(fc), last);
    return { k, k1: Math.min(k + 1, last), a: fc - k, step: Math.min(k, ep.dims.n_steps - 1) };
  }

  /* ---- poses ------------------------------------------------------------- */

  // Linear between frames: the semi-implicit Euler step moves an agent along a
  // straight segment per step, so linear is the honest interpolant. theta is
  // unwrapped state, so interpolating it linearly never takes the long way.
  function pose(rows, idx, nx, f, ep) {
    const { k, k1, a } = split(ep, f);
    const r0 = rows[k], r1 = rows[k1], o = idx * nx;
    return {
      x: lerp(r0[o], r1[o], a),
      y: lerp(r0[o + 1], r1[o + 1], a),
      vx: lerp(r0[o + 2], r1[o + 2], a),
      vy: lerp(r0[o + 3], r1[o + 3], a),
      theta: lerp(r0[o + 4], r1[o + 4], a),
    };
  }
  const bluePose = (ep, i, f) => pose(ep.frames.blue_state, i, ep.dims.blue_nx, f, ep);
  const redPose = (ep, j, f) => pose(ep.frames.red_state, j, ep.dims.red_nx, f, ep);

  // Status at an INTEGER frame: steps[k] for recorded frames, terminal after.
  function statusAt(ep, k) {
    const n = ep.dims.n_steps;
    if (k >= n) return { red_outcomes: ep.terminal.red_outcomes, blue_disabled: ep.terminal.blue_disabled };
    return ep.steps[k];
  }

  /* ---- sensor ------------------------------------------------------------ */

  // The FOV test is sensing.is_detected: |wrap(bearing_to_target - theta)| <= half
  // within range. The wedge spans [theta - half, theta + half] out to range.
  function fovWedge(p, fov) {
    return { x: p.x, y: p.y, a0: p.theta - fov.half_angle_rad, a1: p.theta + fov.half_angle_rad, r: fov.range };
  }
  function inFov(p, fov, tx, ty) {
    const dx = tx - p.x, dy = ty - p.y, d = Math.hypot(dx, dy);
    if (d > fov.range) return false;
    if (d <= 1e-9) return true;
    return Math.abs(wrap(Math.atan2(dy, dx) - p.theta)) <= fov.half_angle_rad;
  }

  /* ---- polygons ---------------------------------------------------------- */

  // coordination.tasks.closest_point_on_polygon / DefendController, same order.
  function closestPointOnPolygon(px, py, poly) {
    let best = poly[0], bestD = Infinity;
    for (let k = 0; k < poly.length; k++) {
      const s = poly[k], e = poly[(k + 1) % poly.length];
      const sx = e[0] - s[0], sy = e[1] - s[1];
      const denom = sx * sx + sy * sy;
      let cx, cy;
      if (denom <= 1e-12) { cx = s[0]; cy = s[1]; }
      else {
        let al = ((px - s[0]) * sx + (py - s[1]) * sy) / denom;
        al = Math.min(1, Math.max(0, al));
        cx = s[0] + al * sx; cy = s[1] + al * sy;
      }
      const d = (cx - px) ** 2 + (cy - py) ** 2;
      if (d < bestD) { bestD = d; best = [cx, cy]; }
    }
    return best;
  }
  function pointInPolygon(x, y, poly) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi || 1e-9) + xi) inside = !inside;
    }
    return inside;
  }

  /* ---- beliefs ----------------------------------------------------------- */

  // The recorder stores ages only. A belief of age a at step t was observed at
  // t - a, of the red's true recorded state then; the controller acts on its
  // constant-velocity prediction (TargetBelief.predicted_position). The export
  // checks this derivation against the controller's inputs, exactly.
  function belief(ep, t, contact, age) {
    const nx = ep.dims.red_nx, row = ep.frames.red_state[t - age], o = contact * nx;
    const h = age * ep.dims.dt;
    return {
      contact, age,
      obs: [row[o], row[o + 1]],
      pred: [row[o] + row[o + 2] * h, row[o + 1] + row[o + 3] * h],
      // perception.belief.records.TargetBelief.uncertainty_radius
      radius: ep.geometry.v_max.red * age * ep.dims.dt,
    };
  }
  function beliefsOf(ep, t, agent) {
    const per = ep.steps[t].beliefs[String(agent)] || {};
    return Object.keys(per).map((c) => belief(ep, t, +c, per[c]));
  }
  // The team's freshest belief per contact among ACTIVE agents.
  function teamBeliefs(ep, t) {
    const st = ep.steps[t], best = {};
    for (const a of Object.keys(st.beliefs)) {
      if (st.blue_disabled[+a]) continue;
      const per = st.beliefs[a];
      for (const c of Object.keys(per)) {
        if (best[c] === undefined || per[c] < best[c].age) best[c] = { age: per[c], holder: +a };
      }
    }
    return Object.keys(best).map((c) => Object.assign(belief(ep, t, +c, best[c].age), { holder: best[c].holder }));
  }

  /* ---- tasks: assigned vs effective --------------------------------------- */

  // What the allocator/planner gave this agent. The commitment ledger (full
  // task, incl. SCREEN of a contact that has since resolved) is preferred when
  // the bundle carries it; otherwise the recorder's fields.
  function assignedTask(ep, t, i) {
    const st = ep.steps[t], key = String(i);
    const ins = ep.instrumentation;
    if (ins && ins.commitments) {
      const c = ins.commitments[t][key];
      if (!c) return { kind: "DEFAULT", id: null, source: "ledger" };
      const kind = c[0] === "DEFEND" ? "STATION" : c[0];
      return { kind, id: c[1], source: "ledger" };
    }
    const role = st.blue_roles[key];
    if (role === "neutralize") return { kind: "NEUTRALIZE", id: st.target_red_ids[key] ?? null, source: "recorder" };
    if (role === "recon") return { kind: "RECON", id: st.recon_target_regions[key] ?? null, source: "recorder" };
    const d = st.defend_targets[key];
    if (d) return { kind: d[0] === "contact" ? "SCREEN" : "STATION", id: d[1], source: "recorder" };
    return { kind: "DEFAULT", id: null, source: "recorder" };
  }

  // What the controller actually did, from effective_targets, as the same task
  // vocabulary. nearest_believed is realized by whichever controller ran: the
  // DEFEND controller screens it, NEUTRALIZE pursues it.
  function effectiveTask(ep, t, i) {
    const st = ep.steps[t], key = String(i), e = st.effective_targets[key];
    if (!e) return { kind: "NONE", id: null, source: null, controller: null };
    const out = { source: e.source, controller: e.controller, id: e.contact_id };
    if (e.source === "assigned") out.kind = "NEUTRALIZE";
    else if (e.source === "screen") out.kind = "SCREEN";
    else if (e.source === "nearest_believed") out.kind = e.controller === "neutralize" ? "NEUTRALIZE" : "SCREEN";
    else if (e.source === "station") {
      out.kind = "STATION";
      const d = st.defend_targets[key];
      out.id = st.blue_roles[key] === "defend" && d && d[0] === "station" ? d[1] : i;
    } else if (e.source === "recon_waypoint") {
      out.kind = "RECON";
      out.id = st.recon_target_regions[key] ?? null;
    } else out.kind = "NONE";
    return out;
  }

  // Divergence: an explicit task the controller is not executing. Mirrors
  // scripts/run_m8_smoke.executed() and cause() exactly -- the project's own
  // definition, which checks the SOURCE as well as the contact: a NEUTRALIZE
  // agent steered at the same contact via nearest_believed (its aim was never
  // resolved from its belief) is NOT executing its assignment.
  // DEFAULT (no explicit task) never diverges: the agent's own choice IS its task.
  function executes(as, e) {
    if (!e) return false;
    if (as.kind === "NEUTRALIZE") return e.source === "assigned" && e.contact_id === as.id;
    if (as.kind === "SCREEN") return e.source === "screen" && e.contact_id === as.id;
    if (as.kind === "STATION") return e.source === "station";
    if (as.kind === "RECON") return e.source === "recon_waypoint";
    return true;
  }
  function divergence(ep, t, i) {
    const st = ep.steps[t], as = assignedTask(ep, t, i);
    if (as.kind === "DEFAULT" || st.blue_disabled[i]) return null;
    const e = st.effective_targets[String(i)];
    if (executes(as, e)) return null;
    const contactTask = as.kind === "NEUTRALIZE" || as.kind === "SCREEN";
    let why, code;
    if (contactTask && st.red_outcomes[as.id] !== "active") { code = "target_resolved"; why = `R${as.id} already ${st.red_outcomes[as.id]}`; }
    else if (!e) { code = "no_effective_record"; why = "no effective-target record"; }
    else if (e.source === "nearest_believed") {
      const same = contactTask && e.contact_id === as.id;
      code = "fallback_nearest_believed" + (same ? "_same_contact" : "");
      why = same ? `fell back to its nearest belief, which is R${as.id} itself (aim never resolved)`
        : `fell back to its nearest believed contact R${e.contact_id}`;
    } else if (e.source === "station" && contactTask) { code = "no_belief_held_station"; why = `does not believe R${as.id}; holding station`; }
    else { code = `other:${e.source}`; why = `controller ran ${e.source}`; }
    return { assigned: as, effective: effectiveTask(ep, t, i), code, why };
  }

  // The point the controller aims at, at integer step t.
  //  NEUTRALIZE: the agent's CURRENT belief of the contact (the MPC's internal
  //    aim state is frozen at assignment and is not recorded -- see design note).
  //  SCREEN / DEFEND-screening: the guard-polygon point closest to the belief,
  //    recomputed every step exactly as DefendController does; `via` is the belief.
  //  STATION: the station.  RECON: the assigned cell centre, else unknown.
  function aimPoint(ep, t, i) {
    const ef = effectiveTask(ep, t, i), g = ep.geometry;
    const b = (c) => {
      const age = (ep.steps[t].beliefs[String(i)] || {})[c];
      if (age === undefined) return null;
      return belief(ep, t, c, age).pred;
    };
    if (ef.kind === "NEUTRALIZE") {
      const p = b(ef.id);
      return p ? { kind: ef.kind, id: ef.id, point: p } : null;
    }
    if (ef.kind === "SCREEN") {
      const p = b(ef.id);
      if (!p) return null;
      return { kind: ef.kind, id: ef.id, point: closestPointOnPolygon(p[0], p[1], g.defense_polygon), via: p };
    }
    if (ef.kind === "STATION") {
      const s = g.stations[String(ef.id)];
      return s ? { kind: ef.kind, id: ef.id, point: s } : null;
    }
    if (ef.kind === "RECON") {
      if (ef.id == null) return { kind: ef.kind, id: null, point: null };
      return { kind: ef.kind, id: ef.id, point: g.coverage.centers[ef.id] };
    }
    return null;
  }

  // Where the ASSIGNED target is, for drawing it when it diverges.
  function assignedPoint(ep, t, i) {
    const as = assignedTask(ep, t, i), g = ep.geometry;
    if (as.kind === "NEUTRALIZE" || as.kind === "SCREEN") {
      if (as.id == null) return null;
      const age = (ep.steps[t].beliefs[String(i)] || {})[as.id];
      const p = age !== undefined ? belief(ep, t, as.id, age).pred
        : [ep.frames.red_state[t][as.id * ep.dims.red_nx], ep.frames.red_state[t][as.id * ep.dims.red_nx + 1]];
      return as.kind === "SCREEN" ? closestPointOnPolygon(p[0], p[1], g.defense_polygon) : p;
    }
    if (as.kind === "STATION") return g.stations[String(as.id)] || null;
    if (as.kind === "RECON" && as.id != null) return g.coverage.centers[as.id];
    return null;
  }

  /* ---- safety filter ----------------------------------------------------- */

  // ||u_req - u_exe|| over (ax, ay) -- the recorder's own intervention measure.
  function intervention(ep, t, i) {
    const nu = ep.dims.blue_nu, r = ep.controls.u_requested[t], e = ep.controls.u_executed[t];
    const dx = e[i * nu] - r[i * nu], dy = e[i * nu + 1] - r[i * nu + 1];
    return { dx, dy, mag: Math.hypot(dx, dy) };
  }

  /* ---- screen transform -------------------------------------------------- */

  // view = {cx, cy, scale} (world centre, CSS px per world unit), canvas W x H CSS px.
  const toScreen = (v, W, H, x, y) => [W / 2 + (x - v.cx) * v.scale, H / 2 - (y - v.cy) * v.scale];
  const toWorld = (v, W, H, sx, sy) => [v.cx + (sx - W / 2) / v.scale, v.cy - (sy - H / 2) / v.scale];
  // A world angle drawn on a y-flipped screen.
  const screenAngle = (a) => -a;

  return {
    TAU, lerp, wrap, split, bluePose, redPose, statusAt, fovWedge, inFov,
    closestPointOnPolygon, pointInPolygon, belief, beliefsOf, teamBeliefs,
    assignedTask, effectiveTask, divergence, aimPoint, assignedPoint, intervention,
    toScreen, toWorld, screenAngle,
  };
});
