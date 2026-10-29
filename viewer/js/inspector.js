/* The inspector: the selected agent's task, what the controller actually did,
 * its heading source, what it sees and believes (with age and error against
 * truth), and safety-filter activity. With nothing selected: the legend.
 * Rebuilt only when the step or the selection changes. */
(function (PEV) {
  "use strict";
  const G = PEV.geom;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const deg = (a) => ((G.wrap(a) * 180) / Math.PI + 360) % 360;
  const TASK_WORD = { NEUTRALIZE: "Pursue", SCREEN: "Screen", STATION: "Hold station", RECON: "Scout", DEFAULT: "Default posture", NONE: "—" };
  let EP = null;                      // the episode being described, for contact names
  const D = (i) => `D${i + 1}`;
  const C = (j) => (EP && PEV.scene ? PEV.scene.contactName(EP, j) : `I${j + 1}`);

  function taskText(task) {
    if (!task) return "—";
    const id = task.id;
    if (task.kind === "NEUTRALIZE" || task.kind === "SCREEN") return `${TASK_WORD[task.kind]} <b>${C(id)}</b>`;
    if (task.kind === "STATION") return `${TASK_WORD.STATION} <b>S${id}</b>`;
    if (task.kind === "RECON") return id == null ? "Scout · patrol point not recorded" : `Scout cell <b>c${id}</b>`;
    return TASK_WORD[task.kind] || task.kind;
  }

  // A command-log task id in words.
  function orderText(o) {
    const p = String(o.task_id || "").split(":");
    if (p[0] === "NEUTRALIZE") return `pursue <b>${C(+p[1])}</b>`;
    if (p[0] === "DEFEND") {
      const d = o.defend_target;
      return d && d[0] === "contact" ? `screen <b>${C(d[1])}</b>` : `hold station <b>S${d ? d[1] : p[1]}</b>`;
    }
    if (p[0] === "RECON") return `scout cell <b>c${p[1]}</b>`;
    return esc(o.task_id);
  }
  const FATE = {
    self: "its own order", applied: "arrived", missed_changed: "never arrived, and would have changed its task",
    missed_same: "never arrived (a repeat of its task)", addressee_lost: "never arrived (lost within 1 s)",
  };

  function ordersBlock(ep, t, i) {
    const c = ep.command;
    if (!c) return "";
    const mine = c.orders.filter((o) => o.agent === i && o.generated_at <= t);
    if (!mine.length) return "";
    const o = mine.reduce((a, b) => (b.generated_at > a.generated_at ? b : a));
    const lost = mine.filter((x) => x.fate === "missed_changed").length;
    return `<h3 class="cap">Commander's orders</h3><dl class="kv">
      <dt>Newest</dt><dd>${orderText(o)} <span class="dim">· from ${o.via === "self" ? "itself, as commander" : "the commander"} at ${(o.generated_at * ep.dims.dt).toFixed(1)} s</span></dd>
      <dt>Fate</dt><dd>${FATE[o.fate] || o.fate}${o.applied_at != null && o.via !== "self" ? ` at ${(o.applied_at * ep.dims.dt).toFixed(1)} s` : ""}</dd>
      <dt>So far</dt><dd>${mine.length} order${mine.length > 1 ? "s" : ""}; ${lost} never arrived that would have changed its task</dd></dl>`;
  }

  const SOURCE_WORD = {
    assigned: "the assigned contact", screen: "the allocated screen", nearest_believed: "its own nearest believed contact",
    station: "a station", recon_waypoint: "a recon waypoint",
  };

  function glyphSvg(kind, team) {
    const c = team === "red" ? "var(--red)" : "var(--blue)";
    if (kind === "LOST") return `<svg width="22" height="22" viewBox="-11 -11 22 22"><path d="M-4,-4L4,4M4,-4L-4,4" stroke="#8F8B85" stroke-width="1.4"/></svg>`;
    if (kind === "bird") return `<svg width="22" height="22" viewBox="-11 -11 22 22"><path d="M0,6L6,-6L-6,-6Z" fill="#8F8B85"/></svg>`;
    if (kind === "civilian") return `<svg width="22" height="22" viewBox="-11 -11 22 22"><rect x="-5" y="-5" width="10" height="10" fill="#8F8B85"/></svg>`;
    if (team === "red") return `<svg width="22" height="22" viewBox="-11 -11 22 22"><path d="M0,-6L6,6L-6,6Z" fill="${c}"/></svg>`;
    return `<svg width="22" height="22" viewBox="-11 -11 22 22"><circle r="5.5" fill="${c}"/></svg>`;
  }

  // The key IS the scene's key (scene.keyEntries / scene.key), one entry a row.
  function legend(eps) {
    const L = (PEV.app && PEV.app.state.layers) || {};
    const ent = PEV.scene.keyEntries(eps, L);
    const k = PEV.scene.key(ent, 2, ent.length * 9.1 + 2, ent.length);
    const h = (ent.length * 9.1 + 4) / 0.75;
    return `
      <p class="insp-hint">Click an agent, or press <kbd>1</kbd>–<kbd>9</kbd>. The arena is the exported drawing: a still, an SI figure panel and a video frame are this picture.</p>
      <h3 class="cap">Key</h3>
      <svg width="${(k.width + 6) / 0.75}" height="${h}" viewBox="0 0 ${k.width + 6} ${ent.length * 9.1 + 4}">${k.svg}</svg>
      <p class="insp-foot">D1, D2… are defenders; I1, I2… intruders. Believed positions are derived from recorded ages and checked at export against the controller's inputs; under falsified reports the false ones are recorded as held. Bids, auction messages and per-message losses are not recorded and are not drawn.</p>`;
  }

  function blueBlock(ep, f, i) {
    EP = ep;
    const sp = G.split(ep, f), t = sp.step, st = ep.steps[t];
    const stat = PEV.scene.status(ep, sp.k);
    const p = G.bluePose(ep, i, f);
    if (stat.blue_disabled[i]) {
      const n = ep.derived.neutralizations.find((x) => x.blue === i);
      return `<div class="insp-head">${glyphSvg("LOST")}<h2>${D(i)}</h2><span class="pill muted">lost</span></div>
        <dl class="kv"><dt>Status</dt><dd>${n ? `Destroyed ${C(n.red)} by contact at ${(n.frame * ep.dims.dt).toFixed(1)} s, and was disabled with it.` : "Disabled."}</dd></dl>`;
    }
    const as = G.assignedTask(ep, t, i), ef = G.effectiveTask(ep, t, i), div = G.divergence(ep, t, i);
    const kind = as.kind === "DEFAULT" ? "DEFAULT" : as.kind;
    const ins = ep.instrumentation;
    const bsrc = ins && ins.bearing ? ins.bearing[t].sources[String(i)] : undefined;
    const tracking = ins && ins.bearing ? ins.bearing[t].tracking.includes(i) : false;
    const headSrc = bsrc === "detection" ? "tracking a detection" : bsrc === "station_outward" ? "station, facing outward"
      : bsrc === null ? "role default (no heading target)" : "not captured";
    const sees = st.per_blue_detected_reds[String(i)] || [];
    const seeTxt = sees.length ? sees.map((j) => {
      const q = G.redPose(ep, j, f); return `<b>${C(j)}</b> at ${Math.hypot(q.x - p.x, q.y - p.y).toFixed(1)} units`;
    }).join(", ") : `nothing in cone${tracking ? " · holding last bearing" : ""}`;
    const beliefs = G.beliefsOf(ep, t, i).sort((a, b) => a.contact - b.contact);
    const maxAge = Math.max(50, ...beliefs.map((b) => b.age));
    const brows = beliefs.map((b) => {
      const truth = G.redPose(ep, b.contact, f);
      const err = Math.hypot(b.pred[0] - truth.x, b.pred[1] - truth.y);
      const resolved = stat.red_outcomes[b.contact] !== "active";
      return `<div class="brow${resolved ? " muted" : ""}"><span class="bid">${C(b.contact)}</span>
        <span class="bbar"><i style="width:${(100 * b.age / maxAge).toFixed(1)}%"></i></span>
        <span class="bage">${(b.age * ep.dims.dt).toFixed(1)} s</span>
        <span class="berr">${resolved ? stat.red_outcomes[b.contact] : `±${err.toFixed(1)} units`}</span></div>`;
    }).join("") || `<div class="brow muted">no contact believed</div>`;
    const d = t < ep.dims.n_steps ? G.intervention(ep, t, i) : { mag: 0 };
    const execLine = ef.kind === "NONE" ? "—" :
      `${taskText(ef)} <span class="dim">· via ${SOURCE_WORD[ef.source] || ef.source}, ${ef.controller} controller</span>`;
    return `
      <div class="insp-head">${glyphSvg(kind)}${ep.command && PEV.scene.frameState(ep, f).planner === i ? '<span class="pill">commander</span>' : ""}<h2>${D(i)}</h2><span class="pill">${PEV.model.TAG[kind]}</span></div>
      <dl class="kv">
        <dt>Task</dt><dd>${as.kind === "DEFAULT" ? "No explicit task · default posture" : taskText(as)}
          <span class="dim">· ${as.source === "ledger" ? "commitment ledger" : "recorder"}</span></dd>
        <dt>Executing</dt><dd>${execLine}</dd>
        ${div ? `<dt class="warn">≠</dt><dd class="warn">Assigned and executed differ: ${esc(div.why)}.</dd>` : ""}
        <dt>Heading</dt><dd>${deg(p.theta).toFixed(0)}° · ${headSrc}</dd>
        <dt>Speed</dt><dd>${Math.hypot(p.vx, p.vy).toFixed(2)} units/s</dd>
        <dt>Sees</dt><dd>${seeTxt}</dd>
      </dl>
      ${ordersBlock(ep, t, i)}
      <h3 class="cap">Believes <span class="dim">age · error of the predicted position</span></h3>
      <div class="beliefs">${brows}</div>
      <h3 class="cap">Safety filter</h3>
      <dl class="kv"><dt>|Δu| now</dt><dd>${d.mag > ep.geometry.intervention_tol ? `<b>${d.mag.toFixed(2)}</b>` : d.mag.toFixed(3)}</dd>
      <dt>Episode</dt><dd>intervened on ${(100 * ep.interventionFraction[i]).toFixed(0)}% of active steps (&gt; 0.01)</dd></dl>`;
  }

  function kindWord(ep, j) {
    const k = ep.contacts && ep.contacts.kinds[j];
    return k === "bird" ? "bird" : k === "civilian" ? "civilian drone" : null;
  }

  function redBlock(ep, f, j) {
    EP = ep;
    const sp = G.split(ep, f), t = sp.step, st = ep.steps[t], stat = PEV.scene.status(ep, sp.k);
    const p = G.redPose(ep, j, f), out = stat.red_outcomes[j], role = st.red_roles[String(j)];
    const c = ep.geometry.defense_center;
    const seenBy = Object.entries(st.per_blue_detected_reds).filter(([, r]) => r.includes(j)).map(([b]) => D(+b));
    const believers = Object.entries(st.beliefs).filter(([b, per]) => per[j] !== undefined && !st.blue_disabled[+b])
      .map(([b, per]) => `${D(+b)} <span class="dim">${(per[j] * ep.dims.dt).toFixed(1)} s</span>`);
    const n = ep.derived.neutralizations.find((x) => x.red === j);
    return `
      <div class="insp-head">${glyphSvg(ep.contacts && ep.contacts.kinds[j] !== "drone" ? ep.contacts.kinds[j] : role, "red")}<h2>${C(j)}</h2><span class="pill red">${kindWord(ep, j) ? (out === "active" ? kindWord(ep, j) : out.replace("decoy_", "")) : out === "active" ? (role || "—").toUpperCase() : out}</span></div>
      <dl class="kv">
        <dt>Outcome</dt><dd>${out === "active" ? "active" : out === "neutralized" && n ? `destroyed by ${D(n.blue)} at ${(n.frame * ep.dims.dt).toFixed(1)} s` : out}</dd>
        <dt>Seen by</dt><dd>${seenBy.length ? seenBy.join(", ") : "no defender's sensor"}</dd>
        <dt>Believed by</dt><dd>${believers.length ? believers.join(" · ") : "nobody"}</dd>
        <dt>To centre</dt><dd>${Math.hypot(p.x - c[0], p.y - c[1]).toFixed(1)} units</dd>
        <dt>Heading</dt><dd>${deg(p.theta).toFixed(0)}° · ${Math.hypot(p.vx, p.vy).toFixed(2)} units/s</dd>
        ${ep.contacts && ep.contacts.images[String(j)] ? `<dt>Shown as</dt><dd><img class="photo" src="../${esc(ep.contacts.images[String(j)])}" alt="the photograph the LLMs saw for this contact"></dd>` : ""}
      </dl>`;
  }

  function render(el, eps, f, selected) {
    if (!selected) { el.innerHTML = legend(eps); return; }
    const blocks = eps.map((ep, k) => {
      const inner = selected.team === "blue"
        ? (selected.id < ep.nb ? blueBlock(ep, f, selected.id) : "")
        : (selected.id < ep.nr ? redBlock(ep, f, selected.id) : "");
      return `<section class="insp-ep">${eps.length > 1 ? `<div class="insp-tag">${["A", "B"][k]} · ${esc(ep.label)}</div>` : ""}${inner}</section>`;
    });
    el.innerHTML = blocks.join("");
  }

  PEV.inspector = { render };
})(window.PEV = window.PEV || {});
