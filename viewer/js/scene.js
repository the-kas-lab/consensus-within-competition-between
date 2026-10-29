/* The one drawing of an episode: SVG markup, in points, in the paper's grammar.
 *
 * Everything the viewer shows on the arena, every still, every SI figure panel
 * and every video frame is a string this file returns -- the UI puts it in the
 * page, the exporter prints or screenshots that same page. There is no second
 * renderer that could drift.
 *
 * Units: one SVG user unit is one point. Type, strokes and marks have fixed
 * sizes in points (tokens.js); only the world-to-paper zoom changes between a
 * 42 mm figure panel and a full-screen arena. Pure: no DOM, so the Node checks
 * run exactly this code.
 *
 * Time: frame k is the world at the start of control step k. The last recorded
 * step comes before that step's resolution, so it is drawn with the stored
 * outcome (the episode's terminal red_outcomes and blue_disabled) and labelled
 * with its own time, as the paper reports the step (a breach "at 34.1 s").
 */
(function (root, factory) {
  const api = factory(
    root.PEV ? root.PEV.geom : require("./geometry.js"),
    root.PEV ? root.PEV.tokens : require("./tokens.js"),
    root.PEV ? root.PEV.font : require("./fontmetrics.js"),
  );
  if (typeof module === "object" && module.exports) module.exports = api;
  else (root.PEV = root.PEV || {}).scene = api;
})(typeof self !== "undefined" ? self : this, function (G, T, F) {
  "use strict";

  const MM = 72 / 25.4;                 // points per millimetre
  const TRAIL = 40;                     // steps of track: the last 4 s at dt = 0.1
  const f1 = (v) => (Math.round(v * 1000) / 1000).toString();
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // Layers. FIGURE is what an SI figure draws; the rest are for exploring and
  // are off unless asked for.
  const FIGURE_LAYERS = {
    wedges: true, region: true, tracks: true, pursuit: true, commander: true,
    links: false, orders: false, falseBeliefs: true, time: true, scaleBar: false,
    labels: false, beliefs: false, tasks: false, sight: false, coverage: false, cbf: false, photos: false,
  };

  /* ---- episode state at one frame ---------------------------------------- */

  const lastStep = (ep) => ep.dims.n_steps - 1;
  const kindOf = (ep, j) => (ep.contacts ? ep.contacts.kinds[j] : "drone");

  // Discrete state at integer step k (0..n-1): outcomes, disabled flags.
  function status(ep, k) {
    if (k >= lastStep(ep)) return { red_outcomes: ep.terminal.red_outcomes, blue_disabled: ep.terminal.blue_disabled };
    return ep.steps[k];
  }

  // Positions at a fractional frame, never past the last recorded step.
  function poses(ep, f) {
    const g = Math.max(0, Math.min(lastStep(ep), f));
    const b = [], r = [];
    for (let i = 0; i < ep.dims.num_blues; i++) b.push(G.bluePose(ep, i, g));
    for (let j = 0; j < ep.dims.num_reds; j++) r.push(G.redPose(ep, j, g));
    return { b, r, f: g, k: Math.floor(g + 1e-9) };
  }

  // {defender: contact} for every active defender NEUTRALIZE-steered at an
  // active contact: the controller's effective target, as in the SI figures.
  function pursuits(ep, k, st) {
    const s = ep.steps[k], out = [];
    for (const [a, e] of Object.entries(s.effective_targets || {})) {
      const i = +a;
      if (st.blue_disabled[i] || e.contact_id == null || s.blue_roles[a] !== "neutralize") continue;
      if (st.red_outcomes[e.contact_id] !== "active") continue;
      out.push([i, e.contact_id]);
    }
    return out;
  }

  // Who planned at step k: the source of the newest planning epoch so far (its
  // order to itself), else the configured commander. Null for a team without one.
  function planner(ep, k) {
    const c = ep.command;
    if (!c) return null;
    let best = -1, who = c.commander_id;
    for (const [t, a] of Object.entries(c.source_by_step)) if (+t <= k && +t > best) { best = +t; who = a; }
    return who;
  }

  // The radio links of the newest communication round at or before step k,
  // between defenders still active at k.
  function links(ep, k, st) {
    const L = ep.network && ep.network.links;
    if (!L) return [];
    let best = -1;
    for (const t of Object.keys(L)) if (+t <= k && +t > best) best = +t;
    if (best < 0) return [];
    return L[String(best)].filter(([a, b]) => !st.blue_disabled[a] && !st.blue_disabled[b]);
  }

  // Where a task id points: an intruder, bird or civilian-drone row, a station, a cell.
  function taskTarget(ep, order, P) {
    const id = String(order.task_id || ""), parts = id.split(":");
    if (parts[0] === "NEUTRALIZE" || parts[0] === "SCREEN") {
      const j = +parts[1];
      return Number.isFinite(j) && P.r[j] ? { contact: j, point: [P.r[j].x, P.r[j].y] } : null;
    }
    const d = order.defend_target;
    if (parts[0] === "DEFEND" && d) {
      if (d[0] === "contact" && P.r[d[1]]) return { contact: d[1], point: [P.r[d[1]].x, P.r[d[1]].y] };
      const s = ep.geometry.stations[String(d[1])];
      return s ? { contact: null, point: s } : null;
    }
    if (parts[0] === "RECON" && Number.isFinite(+parts[1])) return { contact: null, point: ep.geometry.coverage.centers[+parts[1]] };
    return null;
  }

  // Orders never received at step k (the analysis definition, classified by the
  // exporter as "missed_changed"): for each active defender, its NEWEST order --
  // by any route, so an order a new commander gives itself supersedes the dead
  // commander's -- if that order never arrived and would have changed its task.
  function lostOrders(ep, k, st, P, pursuing) {
    const c = ep.command;
    if (!c) return [];
    const newest = {};
    for (const o of c.orders) {
      if (o.generated_at > k) continue;
      const cur = newest[o.agent];
      if (!cur || o.generated_at > cur.generated_at || (o.generated_at === cur.generated_at && o.epoch > cur.epoch)) newest[o.agent] = o;
    }
    const out = [];
    for (const [a, o] of Object.entries(newest)) {
      const i = +a;
      if (st.blue_disabled[i] || o.fate !== "missed_changed") continue;
      const tg = taskTarget(ep, o, P);
      if (!tg) continue;
      if (tg.contact != null && (st.red_outcomes[tg.contact] !== "active" || pursuing[i] === tg.contact)) continue;
      out.push({ agent: i, order: o, point: tg.point, contact: tg.contact });
    }
    return out;
  }

  // Believed positions the defender held that differ from the truth-derived ones:
  // under the falsified-report attack only (exporter: beliefs_held).
  function falseBeliefs(ep, k, st) {
    const h = ep.beliefs_held && ep.beliefs_held[String(k)];
    if (!h) return [];
    const out = [];
    for (const [a, per] of Object.entries(h)) {
      if (st.blue_disabled[+a]) continue;
      for (const [c, p] of Object.entries(per)) if (st.red_outcomes[+c] === "active") out.push({ agent: +a, contact: +c, point: p });
    }
    return out;
  }

  function frameState(ep, f) {
    const P = poses(ep, f), k = P.k, st = status(ep, k);
    const pur = pursuits(ep, k, st), pursuing = {};
    for (const [a, j] of pur) pursuing[a] = j;
    return {
      k, f: P.f, P, st, final: k >= lastStep(ep), pursuits: pur,
      planner: planner(ep, k), links: links(ep, k, st), lost: lostOrders(ep, k, st, P, pursuing),
      falseBeliefs: falseBeliefs(ep, k, st),
    };
  }

  /* ---- SVG primitives ------------------------------------------------------ */

  function line(x1, y1, x2, y2, color, dash, w) {
    return `<line x1="${f1(x1)}" y1="${f1(y1)}" x2="${f1(x2)}" y2="${f1(y2)}" stroke="${color}" stroke-width="${w || T.LW_MIN}"` +
      (dash ? ` stroke-dasharray="${dash.join(" ")}"` : "") + ` stroke-linecap="butt" fill="none"/>`;
  }
  function polyline(pts, color, w) {
    if (pts.length < 2) return "";
    return `<polyline points="${pts.map((p) => f1(p[0]) + "," + f1(p[1])).join(" ")}" stroke="${color}" stroke-width="${w || T.LW_MIN}" fill="none" stroke-linejoin="round" stroke-linecap="butt"/>`;
  }
  function text(x, y, s, o) {
    o = o || {};
    const anchor = o.ha === "center" ? "middle" : o.ha === "right" ? "end" : "start";
    // va: top / center / bottom / baseline, from the font's cap height and descent
    const size = o.size || T.FS, cap = F.CAP * size;
    const by = o.va === "top" ? y + cap + 0.06 * size : o.va === "center" ? y + cap / 2 : o.va === "bottom" ? y - F.DESC * size : y;
    return `<text x="${f1(x)}" y="${f1(by)}" font-family="${T.FONT}" font-size="${size}"` +
      (o.bold ? ' font-weight="bold"' : "") + ` text-anchor="${anchor}" fill="${T.TEXT}">${esc(s)}</text>`;
  }
  // matplotlib-sized markers: ms is a circle's diameter, a triangle's height
  function marker(kind, x, y, ms, face, edge) {
    const h = ms / 2, w = T.LW_RULE, fill = face || "none";
    const st = `fill="${fill}" stroke="${edge}" stroke-width="${w}" stroke-linejoin="miter"`;
    if (kind === "o") return `<circle cx="${f1(x)}" cy="${f1(y)}" r="${f1(h)}" ${st}/>`;
    if (kind === "^") return `<path d="M${f1(x)},${f1(y - h)}L${f1(x - h)},${f1(y + h)}L${f1(x + h)},${f1(y + h)}Z" ${st}/>`;
    if (kind === "v") return `<path d="M${f1(x)},${f1(y + h)}L${f1(x - h)},${f1(y - h)}L${f1(x + h)},${f1(y - h)}Z" ${st}/>`;
    if (kind === "s") return `<rect x="${f1(x - h)}" y="${f1(y - h)}" width="${f1(ms)}" height="${f1(ms)}" ${st}/>`;
    if (kind === "x") {
      return `<path d="M${f1(x - h)},${f1(y - h)}L${f1(x + h)},${f1(y + h)}M${f1(x - h)},${f1(y + h)}L${f1(x + h)},${f1(y - h)}" stroke="${edge}" stroke-width="${w}" fill="none"/>`;
    }
    return "";
  }
  const CONTACT_MARK = { drone: ["^", 0.5], bird: ["v", 0], civilian: ["s", -0.5] };

  // Short names in the paper's terms: D1.. defenders, I1.. intruders, then the
  // birds and civilian drones by kind, numbered within their kind.
  function contactName(ep, j) {
    const kind = kindOf(ep, j);
    if (kind === "drone") return `I${j + 1}`;
    const same = ep.contacts.kinds.slice(0, j + 1).filter((x) => x === kind).length;
    return kind === "bird" ? `Bird ${same}` : `Civilian ${same}`;
  }

  /* ---- world -> paper -------------------------------------------------------- */

  // A panel box {X, Y, W, H} in points and a world window {x0, y0, w, h}.
  function transform(box, win) {
    const k = Math.min(box.W / win.w, box.H / win.h);
    const ox = box.X + (box.W - win.w * k) / 2, oy = box.Y + (box.H - win.h * k) / 2;
    return {
      box, win, k,
      p: (x, y) => [ox + (x - win.x0) * k, oy + (win.h - (y - win.y0)) * k],
      w: (sx, sy) => [win.x0 + (sx - ox) / k, win.y0 + win.h - (sy - oy) / k],
    };
  }

  /* ---- one panel ----------------------------------------------------------- */

  // opts: {box, win, layers, selected, first (scale bar), probe (array), id, photoRoot}
  function panel(ep, f, opts) {
    const L = Object.assign({}, FIGURE_LAYERS, opts.layers || {});
    const S = frameState(ep, f), P = S.P, st = S.st, g = ep.geometry;
    const tr = transform(opts.box, opts.win), p = tr.p, k = S.k;
    const clip = `clip-${opts.id || "p"}`;
    const out = [];
    const probe = opts.probe;
    const active = (i) => !st.blue_disabled[i];
    const live = (j) => st.red_outcomes[j] === "active";
    const B = opts.box;

    out.push(`<defs><clipPath id="${clip}"><rect x="${f1(B.X)}" y="${f1(B.Y)}" width="${f1(B.W)}" height="${f1(B.H)}"/></clipPath></defs>`);
    out.push(`<rect x="${f1(B.X)}" y="${f1(B.Y)}" width="${f1(B.W)}" height="${f1(B.H)}" fill="${T.WHITE}"/>`);
    out.push(`<g clip-path="url(#${clip})">`);

    if (L.coverage && ep.derived && ep.derived.coverage_last_seen) {
      // never swept by any defender's sensor: the ground-truth sweep record, not a belief
      const cov = ep.derived.coverage_last_seen[Math.min(k, ep.derived.coverage_last_seen.length - 1)], cs = g.coverage.cell_size;
      g.coverage.centers.forEach((c, idx) => {
        if (cov[idx] >= 0) return;
        const [x0, y0] = p(c[0] - cs / 2, c[1] + cs / 2);
        out.push(`<rect x="${f1(x0)}" y="${f1(y0)}" width="${f1(cs * tr.k)}" height="${f1(cs * tr.k)}" fill="${T.GRID}"/>`);
      });
    }

    // sensor wedges
    if (L.wedges) {
      for (let i = 0; i < ep.dims.num_blues; i++) {
        if (!active(i)) continue;
        const w = G.fovWedge(P.b[i], g.fov), [cx, cy] = p(w.x, w.y), r = w.r * tr.k;
        const a0 = -w.a0, a1 = -w.a1;           // screen angles (y flipped)
        const x0 = cx + r * Math.cos(a0), y0 = cy + r * Math.sin(a0), x1 = cx + r * Math.cos(a1), y1 = cy + r * Math.sin(a1);
        const large = w.a1 - w.a0 > Math.PI ? 1 : 0;
        out.push(`<path d="M${f1(cx)},${f1(cy)}L${f1(x0)},${f1(y0)}A${f1(r)},${f1(r)} 0 ${large} 0 ${f1(x1)},${f1(y1)}Z" fill="${T.DEF_FILL}" stroke="none"/>`);
        if (probe) probe.push({ team: "blue", id: i, cone: { apex: [+f1(cx), +f1(cy)], r: +f1(r), a0: w.a0, a1: w.a1 } });
      }
    }

    // defended region
    if (L.region) {
      const pts = g.defended_shape.map((q) => p(q[0], q[1]));
      out.push(`<path d="M${pts.map((q) => f1(q[0]) + "," + f1(q[1])).join("L")}Z" fill="${T.FILL}" stroke="${T.TEXT}" stroke-width="${T.LW_RULE}" stroke-linejoin="round"/>`);
    }

    // radio links (limited range)
    if (L.links) for (const [a, b] of S.links) {
      const [x1, y1] = p(P.b[a].x, P.b[a].y), [x2, y2] = p(P.b[b].x, P.b[b].y);
      out.push(line(x1, y1, x2, y2, T.ZERO));
    }

    // tracks: the last 4 s, up to the drawn position
    if (L.tracks) {
      const lo = Math.max(0, k - TRAIL);
      const track = (rows, idx, nx, head) => {
        const pts = [];
        for (let q = lo; q <= k; q++) pts.push(p(rows[q][idx * nx], rows[q][idx * nx + 1]));
        pts.push(p(head.x, head.y));
        return pts;
      };
      for (let i = 0; i < ep.dims.num_blues; i++) {
        if (active(i)) out.push(polyline(track(ep.frames.blue_state, i, ep.dims.blue_nx, P.b[i]), T.DEF));
      }
      for (let j = 0; j < ep.dims.num_reds; j++) {
        if (live(j)) out.push(polyline(track(ep.frames.red_state, j, ep.dims.red_nx, P.r[j]), kindOf(ep, j) === "drone" ? T.INT : T.ZERO));
      }
    }

    // sight lines: each defender's own sensor holds the contact
    if (L.sight) {
      const s0 = ep.steps[Math.min(k, lastStep(ep))];
      for (const [a, reds] of Object.entries(s0.per_blue_detected_reds || {})) {
        if (!active(+a)) continue;
        const [x1, y1] = p(P.b[+a].x, P.b[+a].y);
        for (const j of reds) { const [x2, y2] = p(P.r[j].x, P.r[j].y); out.push(line(x1, y1, x2, y2, T.DEF, [1, 2])); }
      }
    }

    // tasks other than pursuit: where the controller aims (explore only)
    if (L.tasks) {
      const t = Math.min(k, lastStep(ep));
      for (let i = 0; i < ep.dims.num_blues; i++) {
        if (!active(i)) continue;
        const aim = G.aimPoint(ep, t, i);
        if (!aim || !aim.point || aim.kind === "NEUTRALIZE") continue;
        const [x1, y1] = p(P.b[i].x, P.b[i].y), [x2, y2] = p(aim.point[0], aim.point[1]);
        const dash = { SCREEN: [6, 2], STATION: [1, 2], RECON: [5, 2, 1, 2] }[aim.kind];
        out.push(line(x1, y1, x2, y2, T.DEF, dash));
      }
    }

    // pursuits: defender to the contact its controller is steered at
    if (L.pursuit) for (const [a, j] of S.pursuits) {
      const [x1, y1] = p(P.b[a].x, P.b[a].y), [x2, y2] = p(P.r[j].x, P.r[j].y);
      out.push(line(x1, y1, x2, y2, T.DEF, T.DASH_PURSUIT, T.LW_RULE));
    }

    // orders never received
    if (L.orders) for (const o of S.lost) {
      const [x1, y1] = p(P.b[o.agent].x, P.b[o.agent].y), [x2, y2] = p(o.point[0], o.point[1]);
      out.push(line(x1, y1, x2, y2, T.TEXT, T.DASH_ORDER, T.LW_RULE));
    }

    // false beliefs: where a defender believes an intruder is, on a falsified report
    if (L.falseBeliefs) for (const b of S.falseBeliefs) {
      const [x1, y1] = p(P.b[b.agent].x, P.b[b.agent].y), [x2, y2] = p(b.point[0], b.point[1]);
      out.push(line(x1, y1, x2, y2, T.INT, T.DASH_ORDER));
      out.push(marker("^", x2, y2, T.MS + 0.5, T.WHITE, T.INT));
    }

    // beliefs (explore): the team's freshest, or the selected defender's own
    if (L.beliefs && k <= lastStep(ep)) {
      const t = Math.min(k, lastStep(ep)), sel = opts.selected && opts.selected.team === "blue" ? opts.selected.id : null;
      const bs = sel != null ? G.beliefsOf(ep, t, sel) : G.teamBeliefs(ep, t);
      for (const b of bs) {
        if (b.age < 1 || !live(b.contact)) continue;
        const [x, y] = p(b.pred[0], b.pred[1]);
        out.push(marker(CONTACT_MARK[kindOf(ep, b.contact)][0], x, y, T.MS + 0.5, T.WHITE, kindOf(ep, b.contact) === "drone" ? T.INT : T.ZERO));
        out.push(text(x + 4, y, `${(b.age * ep.dims.dt).toFixed(1)} s old`, { va: "center" }));
      }
    }

    // safety filter (explore): direction of the push, when it intervened
    if (L.cbf && k < ep.dims.n_steps) {
      for (let i = 0; i < ep.dims.num_blues; i++) {
        if (!active(i)) continue;
        const d = G.intervention(ep, k, i);
        if (d.mag <= g.intervention_tol) continue;
        const [x, y] = p(P.b[i].x, P.b[i].y), len = Math.min(14, 4 + d.mag * 3);
        out.push(line(x, y, x + (d.dx / d.mag) * len, y - (d.dy / d.mag) * len, T.ZERO));
      }
    }

    // commander
    if (L.commander && S.planner != null && active(S.planner)) {
      const [x, y] = p(P.b[S.planner].x, P.b[S.planner].y);
      out.push(marker("o", x, y, T.MS_COMMANDER, null, T.TEXT));
    }

    // defenders
    for (let i = 0; i < ep.dims.num_blues; i++) {
      const [x, y] = p(P.b[i].x, P.b[i].y);
      out.push(active(i) ? marker("o", x, y, T.MS, T.DEF, T.DEF) : marker("x", x, y, T.MS, null, T.ZERO));
      if (probe) probe.push({ team: "blue", id: i, x: +f1(x), y: +f1(y), theta: P.b[i].theta, disabled: !active(i) });
    }
    // contacts: intruders, birds, civilian drones
    for (let j = 0; j < ep.dims.num_reds; j++) {
      const o = st.red_outcomes[j], kind = kindOf(ep, j), [m, dm] = CONTACT_MARK[kind] || CONTACT_MARK.drone;
      const [x, y] = p(P.r[j].x, P.r[j].y);
      if (o === "active") out.push(marker(m, x, y, T.MS + dm, kind === "drone" ? T.INT : T.ZERO, kind === "drone" ? T.INT : T.ZERO));
      else if (o === "neutralized" || o === "decoy_destroyed") out.push(marker(m, x, y, T.MS + dm, T.WHITE, T.ZERO));
      if (probe) probe.push({ team: "red", id: j, x: +f1(x), y: +f1(y), theta: P.r[j].theta, outcome: o });
    }

    // selection
    if (opts.selected) {
      const s = opts.selected, q = s.team === "blue" ? P.b[s.id] : P.r[s.id];
      if (q) { const [x, y] = p(q.x, q.y); out.push(`<circle cx="${f1(x)}" cy="${f1(y)}" r="7" fill="none" stroke="${T.TEXT}" stroke-width="${T.LW_RULE}" stroke-dasharray="2 1.5"/>`); }
    }

    // photographs each LLM saw (explore / opt-in)
    if (L.photos && ep.contacts && opts.photoRoot != null) {
      for (const [j, path] of Object.entries(ep.contacts.images || {})) {
        if (!live(+j)) continue;
        const [x, y] = p(P.r[+j].x, P.r[+j].y), s = 22;
        out.push(`<image href="${esc(opts.photoRoot + path)}" x="${f1(x + 4)}" y="${f1(y - s - 4)}" width="${s}" height="${s}" preserveAspectRatio="xMidYMid slice"/>`);
      }
    }
    out.push("</g>");

    // labels (explore)
    if (L.labels) out.push(labels(ep, P, st, p, B));

    // panel frame, time, scale bar
    out.push(`<rect x="${f1(B.X)}" y="${f1(B.Y)}" width="${f1(B.W)}" height="${f1(B.H)}" fill="none" stroke="${T.ZERO}" stroke-width="${T.LW_RULE}"/>`);
    if (L.time) out.push(text(B.X + B.W * 0.04, B.Y + B.H * 0.04, `${(k * ep.dims.dt).toFixed(1)} s`, { va: "top" }));
    if (L.scaleBar) out.push(scaleBar(ep, P, tr, B));
    return { svg: out.join(""), state: S, transform: tr };
  }

  // 10 units, in the emptier bottom corner (the SI figures' rule)
  function scaleBar(ep, P, tr, B) {
    const pts = P.b.concat(P.r).map((q) => tr.p(q.x, q.y));
    const left = pts.filter(([x, y]) => x < B.X + B.W * 0.4 && y > B.Y + B.H * 0.8).length;
    const right = pts.filter(([x, y]) => x > B.X + B.W * 0.6 && y > B.Y + B.H * 0.8).length;
    const len = 10 * tr.k;
    const x = left <= right ? B.X + B.W * 0.05 : B.X + B.W * 0.95 - len, y = B.Y + B.H * 0.94;
    return line(x, y, x + len, y, T.TEXT, null, T.LW_RULE) + text(x + len / 2, y - B.H * 0.025, "10 units", { ha: "center", va: "bottom" });
  }

  // id labels, placed at the first free offset (explore only)
  function labels(ep, P, st, p, B) {
    const placed = [], out = [], size = T.FS;
    const items = [];
    P.b.forEach((q, i) => items.push([p(q.x, q.y), `D${i + 1}`]));
    P.r.forEach((q, j) => {
      const o = st.red_outcomes[j];
      if (o === "escaped" || o === "decoy_left") return;
      items.push([p(q.x, q.y), contactName(ep, j)]);
    });
    for (const [[x, y]] of items) placed.push([x - 3, y - 3, x + 3, y + 3]);
    const hit = (b) => b[0] < B.X || b[1] < B.Y || b[2] > B.X + B.W || b[3] > B.Y + B.H ||
      placed.some((o) => b[0] < o[2] && b[2] > o[0] && b[1] < o[3] && b[3] > o[1]);
    for (const [[x, y], s] of items) {
      const w = F.width(s, size), h = size * 0.8;
      for (const [dx, dy] of [[4, -4], [4, 4 + h], [-4 - w, -4], [-4 - w, 4 + h], [-w / 2, -6], [-w / 2, 7 + h]]) {
        const box = [x + dx, y + dy - h, x + dx + w, y + dy];
        if (!hit(box)) { placed.push(box); out.push(text(x + dx, y + dy, s)); break; }
      }
    }
    return out.join("");
  }

  /* ---- views ---------------------------------------------------------------- */

  // The whole arena, with a margin.
  function arenaWindow(ep) {
    const a = ep.geometry.arena;
    return { x0: a.min, y0: a.min, w: a.size, h: a.size };
  }

  // One square window for a figure: the region and every active agent and its
  // track at the shown steps (the SI figures' crop rule).
  function cropWindow(ep, steps) {
    const xs = [], ys = [];
    const add = (x, y) => { xs.push(x); ys.push(y); };
    ep.geometry.defended_shape.forEach((q) => add(q[0], q[1]));
    const bn = ep.dims.blue_nx, rn = ep.dims.red_nx;
    for (const k of steps) {
      const sk = ep.steps[k];
      for (let q = Math.max(0, k - TRAIL); q <= k; q++) {
        const B = ep.frames.blue_state[q], R = ep.frames.red_state[q];
        for (let i = 0; i < ep.dims.num_blues; i++) if (!sk.blue_disabled[i] || q === k) add(B[i * bn], B[i * bn + 1]);
        for (let j = 0; j < ep.dims.num_reds; j++) if (sk.red_outcomes[j] === "active" || q === k) add(R[j * rn], R[j * rn + 1]);
      }
    }
    const size = ep.geometry.arena.size;
    const lo = [Math.min(...xs), Math.min(...ys)], hi = [Math.max(...xs), Math.max(...ys)];
    const side = Math.min(size, Math.max(hi[0] - lo[0], hi[1] - lo[1]) + 8);
    const c = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2];
    const clip = (v) => Math.min(Math.max(v - side / 2, 0), size - side);
    return { x0: clip(c[0]), y0: clip(c[1]), w: side, h: side };
  }

  /* ---- the key ------------------------------------------------------------ */

  // Key entries for what a drawing shows, in the SI figures' order.
  function keyEntries(eps, L) {
    L = Object.assign({}, FIGURE_LAYERS, L || {});
    const kinds = new Set(), any = (fn) => eps.some(fn);
    eps.forEach((ep) => (ep.contacts ? ep.contacts.kinds : []).forEach((x) => kinds.add(x)));
    const decoys = kinds.has("bird") || kinds.has("civilian");
    const lostDecoy = any((ep) => ep.contacts && ep.contacts.decoy_events.some((e) => e.event === "destroyed"));
    const e = [["o-def", "Defender"]];
    if (L.commander && any((ep) => ep.command)) e.push(["o-cmd", "Commander"]);
    e.push(["^-int", "Intruder"]);
    if (decoys) { if (kinds.has("bird")) e.push(["v-dec", "Bird"]); if (kinds.has("civilian")) e.push(["s-dec", "Civilian drone"]); }
    if (L.tracks) e.push(["l-track", "Last 4 s of track"]);
    if (L.pursuit) e.push(["l-pursuit", "Pursuit"]);
    if (L.orders && any((ep) => ep.command && ep.command.orders.some((o) => o.fate === "missed_changed"))) e.push(["l-order", "Order never received"]);
    if (L.links && any((ep) => ep.network && ep.network.links && Number.isFinite(+ep.network.r_comm))) e.push(["l-link", "Radio link"]);
    if (L.falseBeliefs && any((ep) => ep.beliefs_held)) e.push(["^-false", "False belief"]);
    e.push(["^-dead", "Intruder destroyed"]);
    if (lostDecoy) e.push(["s-dead", "Civilian drone destroyed"]);
    e.push(["x-dis", "Defender disabled"]);
    if (L.wedges) e.push(["p-wedge", "Sensor wedge"]);
    if (L.region) e.push(["p-region", "Defended region"]);
    return e;
  }

  const HANDLE = { len: 1.8 * T.FS, pad: 0.5 * T.FS, col: 1.2 * T.FS, row: 0.3 * T.FS };

  function handle(code, x, y) {
    const w = HANDLE.len, cx = x + w / 2, ms = T.MS;
    switch (code) {
      case "o-def": return marker("o", cx, y, ms, T.DEF, T.DEF);
      case "o-cmd": return marker("o", cx, y, T.MS_COMMANDER, null, T.TEXT);
      case "^-int": return marker("^", cx, y, ms + 0.5, T.INT, T.INT);
      case "v-dec": return marker("v", cx, y, ms, T.ZERO, T.ZERO);
      case "s-dec": return marker("s", cx, y, ms - 0.5, T.ZERO, T.ZERO);
      case "l-track": return line(x, y, x + w, y, T.DEF);
      case "l-pursuit": return line(x, y, x + w, y, T.DEF, T.DASH_PURSUIT, T.LW_RULE);
      case "l-order": return line(x, y, x + w, y, T.TEXT, T.DASH_ORDER, T.LW_RULE);
      case "l-link": return line(x, y, x + w, y, T.ZERO);
      case "^-false": return line(x, y, x + w * 0.55, y, T.INT, T.DASH_ORDER) + marker("^", x + w * 0.75, y, ms + 0.5, T.WHITE, T.INT);
      case "^-dead": return marker("^", cx, y, ms + 0.5, T.WHITE, T.ZERO);
      case "s-dead": return marker("s", cx, y, ms - 0.5, T.WHITE, T.ZERO);
      case "x-dis": return marker("x", cx, y, ms, null, T.ZERO);
      case "p-wedge": return `<rect x="${f1(x)}" y="${f1(y - 0.35 * T.FS)}" width="${f1(w)}" height="${f1(0.7 * T.FS)}" fill="${T.DEF_FILL}"/>`;
      case "p-region": return `<rect x="${f1(x)}" y="${f1(y - 0.35 * T.FS)}" width="${f1(w)}" height="${f1(0.7 * T.FS)}" fill="${T.FILL}" stroke="${T.TEXT}" stroke-width="${T.LW_RULE}"/>`;
      default: return "";
    }
  }

  // A frameless key, column-major like matplotlib's, lower-left at (x, yBottom).
  function key(entries, x, yBottom, rows) {
    const ncol = Math.ceil(entries.length / rows), out = [];
    const rowH = T.FS + HANDLE.row;
    let cx = x;
    for (let c = 0; c < ncol; c++) {
      const col = entries.slice(c * rows, (c + 1) * rows);
      const w = Math.max(...col.map(([, s]) => F.width(s, T.FS)));
      col.forEach(([code, s], r) => {
        const yc = yBottom - (rows - 1 - r) * rowH - T.FS / 2;
        out.push(handle(code, cx, yc));
        out.push(text(cx + HANDLE.len + HANDLE.pad, yc, s, { va: "center" }));
      });
      cx += HANDLE.len + HANDLE.pad + w + HANDLE.col;
    }
    return { svg: out.join(""), width: cx - HANDLE.col - x, height: rows * rowH - HANDLE.row };
  }

  /* ---- documents ------------------------------------------------------------ */

  function doc(wPt, hPt, body, units) {
    const wh = units === "mm" ? `width="${f1(wPt / MM)}mm" height="${f1(hPt / MM)}mm"` : `width="${f1(wPt)}pt" height="${f1(hPt)}pt"`;
    return `<svg xmlns="http://www.w3.org/2000/svg" ${wh} viewBox="0 0 ${f1(wPt)} ${f1(hPt)}">` +
      `<rect x="0" y="0" width="${f1(wPt)}" height="${f1(hPt)}" fill="${T.WHITE}"/>${body}</svg>`;
  }

  // An SI figure: one episode, panels at given steps, each with a letter and a
  // heading, one shared square window, and the key. The layout of
  // fig_episodes.py, in millimetres: panel 42, gap 3.5, left 1.5, heading 4.5,
  // key band 9.
  // spec.key_right: the key in one column right of the last panel, top-aligned
  // with it, and no band underneath (a single panel set into a larger figure).
  function figure(ep, spec) {
    const P = spec.panel_mm || 42, GAP = spec.gap_mm || 3.5, LEFT = spec.left_mm || 1.5, HEAD = 4.5;
    const KEY = spec.key_right ? -0.5 : 9.0;
    const Wmm = spec.width_mm || 180, n = spec.steps.length, Hmm = HEAD + P + KEY + 2.0;
    const L = Object.assign({}, FIGURE_LAYERS, spec.layers || {});
    const win = spec.window || cropWindow(ep, spec.steps);
    const out = [], probes = [];
    spec.steps.forEach((k, idx) => {
      const x = LEFT + idx * (P + GAP), top = Hmm - (KEY + 1.0 + P);  // panel top, mm from the top edge
      const box = { X: x * MM, Y: top * MM, W: P * MM, H: P * MM };
      const probe = spec.probe ? [] : null;
      const r = panel(ep, k, { box, win, layers: Object.assign({}, L, { scaleBar: idx === 0 && !(spec.layers && spec.layers.scaleBar === false) }), id: `p${idx}`, probe });
      out.push(r.svg);
      if (probe) probes.push({ step: k, drawn: probe, box, win });
      const ty = (Hmm - (KEY + 1.0 + P + HEAD)) * MM;
      out.push(text(x * MM, ty, String.fromCharCode(97 + idx), { va: "top", bold: true }));
      if (spec.headings && spec.headings[idx]) out.push(text((x + 3.0) * MM, ty, spec.headings[idx], { va: "top" }));
    });
    const ent = keyEntries([ep], L);
    const kx = LEFT + n * (P + GAP), top = (Hmm - (KEY + 1.0 + P)) * MM;
    const kk = spec.key_right ? key(ent, kx * MM, top + ent.length * (T.FS + HANDLE.row) - HANDLE.row, ent.length)
      : key(ent, LEFT * MM, (Hmm - 0.5) * MM, 2);
    out.push(kk.svg);
    return { svg: doc(Wmm * MM, Hmm * MM, out.join(""), "mm"), width_mm: Wmm, height_mm: Hmm, probes, keyWidth_mm: kk.width / MM };
  }

  // A video frame or a still: a heading line, the panel, the key underneath.
  // spec: {width_mm, title, subtitle, window, layers, rows}
  function frameDoc(ep, f, spec) {
    const Wmm = spec.width_mm || 120, pad = 2.5, head = spec.title ? (spec.subtitle ? 9.5 : 6) : 0;
    const L = Object.assign({}, FIGURE_LAYERS, { scaleBar: true }, spec.layers || {});
    const ent = keyEntries([ep], L), rows = spec.rows || Math.ceil(ent.length / 4);
    const keyH = rows * (T.FS + HANDLE.row) / MM + 1.5;
    const side = Wmm - 2 * pad, Hmm = pad + head + side + keyH + pad;
    const box = { X: pad * MM, Y: (pad + head) * MM, W: side * MM, H: side * MM };
    const out = [];
    if (spec.title) out.push(text(pad * MM, pad * MM, spec.title, { va: "top" }));
    if (spec.subtitle) out.push(text(pad * MM, (pad + 3.5) * MM, spec.subtitle, { va: "top" }));
    const r = panel(ep, f, { box, win: spec.window || arenaWindow(ep), layers: L, id: "v", photoRoot: spec.photoRoot });
    out.push(r.svg);
    out.push(key(ent, pad * MM, (Hmm - pad) * MM, rows).svg);
    return { svg: doc(Wmm * MM, Hmm * MM, out.join(""), "mm"), width_mm: Wmm, height_mm: Hmm };
  }

  return {
    MM, TRAIL, FIGURE_LAYERS, frameState, status, panel, transform, arenaWindow, cropWindow,
    keyEntries, key, figure, frameDoc, doc, lastStep, contactName,
  };
});
