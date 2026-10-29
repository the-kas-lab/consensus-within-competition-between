/* The timeline: time axis, event rail, one task ribbon per defender, contact lanes,
 * and a CBF track. Drag anywhere to scrub; click an event mark to jump.
 *
 * Ribbon fill = the task the allocator gave (pattern, not only colour):
 *   NEUTRALIZE solid · SCREEN hatched · station/DEFEND tint · RECON dotted.
 *   An agent with NO explicit task (rule planner's default posture) is drawn
 *   as an outline in the style of what it actually did.
 * Marks above the ribbon: the assigned task is not what the controller ran.
 * Line below the ribbon: the agent's own sensor holds a red. */
(function (PEV) {
  "use strict";
  const MONO = '"SF Mono", ui-monospace, "JetBrains Mono", Menlo, Consolas, monospace';
  const GUTTER = 64, RIGHT = 14;
  const FULL = { axis: 18, rail: 22, head: 16, blue: 13, red: 9, cbf: 20, gapB: 4, gapR: 3, group: 10 };
  // Compare mode stacks two episodes: tighter rows keep the arenas the hero.
  const COMPACT = { axis: 18, rail: 20, head: 15, blue: 9, red: 6, cbf: 14, gapB: 3, gapR: 2, group: 6 };
  let ROW = FULL;

  class Timeline {
    constructor(canvas, onSeek, onHover) {
      this.canvas = canvas; this.ctx = canvas.getContext("2d");
      this.onSeek = onSeek; this.onHover = onHover;
      this.eps = []; this.hits = []; this.dragging = false;
      canvas.addEventListener("pointerdown", (e) => this.down(e));
      canvas.addEventListener("pointermove", (e) => this.move(e));
      canvas.addEventListener("pointerup", (e) => this.up(e));
      canvas.addEventListener("pointerleave", () => this.onHover(null));
    }

    setEpisodes(eps) {
      this.eps = eps;
      ROW = eps.length > 1 ? COMPACT : FULL;
      this.frames = Math.max(...eps.map((e) => e.lastFrame));
      let h = ROW.axis + ROW.rail;
      for (const ep of eps) {
        h += ROW.group + (eps.length > 1 ? ROW.head : 0);
        h += ep.nb * (ROW.blue + ROW.gapB) + 4 + ep.nr * (ROW.red + ROW.gapR) + 4 + ROW.cbf;
      }
      this.height = h + 8;
      this.canvas.style.height = this.height + "px";
    }

    resize() {
      const r = this.canvas.getBoundingClientRect();
      this.dpr = window.devicePixelRatio || 1;
      this.W = Math.max(1, Math.round(r.width)); this.H = this.height || Math.round(r.height);
      this.canvas.width = Math.round(this.W * this.dpr); this.canvas.height = Math.round(this.H * this.dpr);
    }

    x(frame) { return GUTTER + (frame / this.frames) * (this.W - GUTTER - RIGHT); }
    frameAt(px) { return Math.max(0, Math.min(this.frames, ((px - GUTTER) / (this.W - GUTTER - RIGHT)) * this.frames)); }

    /* -------------------------------------------------------------------- */

    draw(f, P, selected) {
      const ctx = this.ctx;
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      ctx.clearRect(0, 0, this.W, this.H);
      if (!this.eps.length) return;
      this.hits = [];
      const ep0 = this.eps[0], dt = ep0.dims.dt;

      // axis: seconds, with planning epochs as fine ticks
      let y = 2;
      ctx.font = `9.5px ${MONO}`; ctx.fillStyle = P.ink3; ctx.textBaseline = "top"; ctx.textAlign = "center";
      const secs = this.frames * dt, stepS = secs > 60 ? 10 : 5;
      ctx.strokeStyle = P.hair; ctx.lineWidth = 1; ctx.beginPath();
      const pis = ep0.geometry.planning_interval_steps;
      for (let k = 0; k <= this.frames; k += pis) { const x = Math.round(this.x(k)) + 0.5; ctx.moveTo(x, y + 13); ctx.lineTo(x, y + 16); }
      ctx.stroke();
      for (let s = 0; s <= secs + 1e-9; s += stepS) {
        const x = this.x(s / dt);
        ctx.fillText(s === 0 ? "0 s" : `${s}`, x, y);
      }
      y += ROW.axis;

      // event rail: every episode's events on one rail, marked by episode in compare
      const railY = y + ROW.rail / 2;
      ctx.strokeStyle = P.hair; ctx.beginPath(); ctx.moveTo(GUTTER, railY + 0.5); ctx.lineTo(this.W - RIGHT, railY + 0.5); ctx.stroke();
      if (this.eps.length > 1) {
        ctx.font = `8.5px ${MONO}`; ctx.fillStyle = P.ink3; ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText("A", GUTTER - 9, railY - 5); ctx.fillText("B", GUTTER - 9, railY + 5);
        ctx.font = `9.5px ${MONO}`; ctx.fillStyle = P.ink3; ctx.textAlign = "right";
        ctx.fillText("events", GUTTER - 14, railY);
      } else this.caption("events", railY, P);
      this.eps.forEach((ep, ei) => {
        for (const e of ep.events) {
          const x = this.x(e.frame), yy = railY + (this.eps.length > 1 ? (ei ? 5 : -5) * (ROW === COMPACT ? 1 : 1.2) : 0);
          this.eventMark(e, x, yy, P);
          this.hits.push({ x, y: yy, r: 8, frame: e.frame, text: (this.eps.length > 1 ? ["A", "B"][ei] + " · " : "") + e.text });
        }
      });
      y += ROW.rail;

      this.eps.forEach((ep, ei) => {
        y += ROW.group;
        if (this.eps.length > 1) {
          ctx.font = `600 10px ${MONO}`; ctx.fillStyle = P.ink2; ctx.textAlign = "left"; ctx.textBaseline = "middle";
          ctx.fillText(`${["A", "B"][ei]} · ${ep.label}`.toUpperCase(), GUTTER, y + ROW.head / 2);
          y += ROW.head;
        }
        for (let i = 0; i < ep.nb; i++) {
          const sel = selected && selected.team === "blue" && selected.id === i;
          this.blueLane(ep, i, y, P, sel);
          y += ROW.blue + ROW.gapB;
        }
        y += 4;
        for (let j = 0; j < ep.nr; j++) {
          const sel = selected && selected.team === "red" && selected.id === j;
          this.redLane(ep, j, y, P, sel);
          y += ROW.red + ROW.gapR;
        }
        y += 4;
        this.cbfTrack(ep, y, P);
        y += ROW.cbf;
        // ended early: the rest of this episode's row is a quiet void
        if (ep.lastFrame < this.frames) {
          const x0 = this.x(ep.lastFrame);
          ctx.fillStyle = P.voidFill;
          const top = y - ROW.cbf - 8 - ep.nr * (ROW.red + ROW.gapR) - ep.nb * (ROW.blue + ROW.gapB);
          ctx.fillRect(x0, top, this.W - RIGHT - x0, y - top);
        }
      });

      // playhead
      const px = Math.round(this.x(f)) + 0.5;
      ctx.strokeStyle = P.ink; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(px, ROW.axis - 2); ctx.lineTo(px, this.H - 4); ctx.stroke();
      ctx.fillStyle = P.ink; ctx.beginPath(); ctx.moveTo(px - 4, ROW.axis - 5); ctx.lineTo(px + 4, ROW.axis - 5); ctx.lineTo(px, ROW.axis); ctx.closePath(); ctx.fill();
    }

    caption(text, cy, P, color) {
      const ctx = this.ctx;
      ctx.font = `9.5px ${MONO}`; ctx.fillStyle = color || P.ink3; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(text, GUTTER - 10, cy);
    }

    runs(n, key) {
      const out = []; let s = 0;
      for (let t = 1; t <= n; t++) if (t === n || key(t) !== key(s)) { out.push([s, t]); s = t; }
      return out;
    }

    blueLane(ep, i, y, P, sel) {
      const ctx = this.ctx, lane = ep.blueLanes[i], h = ROW.blue, n = lane.length;
      this.caption(`D${i + 1}`, y + h / 2, P, sel ? P.ink : P.blue);
      if (sel) { ctx.fillStyle = P.selRow; ctx.fillRect(GUTTER - 2, y - 2, this.W - RIGHT - GUTTER + 4, h + 4); }
      const key = (t) => (lane[t].disabled ? "X" : lane[t].kind + (lane[t].explicit ? "" : "~"));
      for (const [a, b] of this.runs(n, key)) {
        const x0 = this.x(a), x1 = this.x(b), c = lane[a];
        this.ribbon(x0, y, x1 - x0, h, c.disabled ? "X" : c.kind, c.explicit, P);
      }
      // divergence marks above, own detection line below
      // divergence: a row of ink notches, distinct from the solid detection line
      ctx.fillStyle = P.ink;
      for (const [a, b] of this.runs(n, (t) => lane[t].diverge)) {
        if (!lane[a].diverge) continue;
        const x0 = this.x(a), x1 = Math.max(this.x(b), x0 + 1.5);
        for (let xx = x0; xx < x1; xx += 4) ctx.fillRect(xx, y - 4, Math.min(2, x1 - xx), 3);
      }
      ctx.fillStyle = P.red;
      for (const [a, b] of this.runs(n, (t) => lane[t].detect)) {
        if (!lane[a].detect) continue;
        ctx.fillRect(this.x(a), y + h + 0.5, Math.max(1.5, this.x(b) - this.x(a)), 2);
      }
    }

    ribbon(x, y, w, h, kind, explicit, P) {
      const ctx = this.ctx;
      ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
      if (kind === "X") {
        ctx.fillStyle = P.voidFill; ctx.fillRect(x, y, w, h);
        this.hatch(x, y, w, h, P.ink3, 5, 1); this.hatch(x, y, w, h, P.ink3, 5, -1);
      } else if (!explicit) {
        ctx.strokeStyle = P.blueA(0.55); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
        if (kind === "SCREEN") this.hatch(x, y, w, h, P.blueA(0.35), 6, 1);
      } else if (kind === "NEUTRALIZE") {
        ctx.fillStyle = P.blue; ctx.fillRect(x, y, w, h);
      } else if (kind === "SCREEN") {
        ctx.fillStyle = P.blueA(0.14); ctx.fillRect(x, y, w, h); this.hatch(x, y, w, h, P.blue, 5, 1);
      } else if (kind === "STATION") {
        ctx.fillStyle = P.blueA(0.24); ctx.fillRect(x, y, w, h);
      } else if (kind === "RECON") {
        ctx.fillStyle = P.blueA(0.1); ctx.fillRect(x, y, w, h);
        ctx.fillStyle = P.blue;
        for (let xx = x + 2; xx < x + w; xx += 5) for (let yy = y + 3; yy < y + h; yy += 5) ctx.fillRect(xx, yy, 1.4, 1.4);
      }
      ctx.restore();
      // a hairline gap between runs keeps adjacent tasks legible
      ctx.fillStyle = P.bg; ctx.fillRect(x, y, 1, h);
    }

    hatch(x, y, w, h, color, step, dir) {
      const ctx = this.ctx;
      ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.beginPath();
      for (let d = -h; d < w + h; d += step) {
        if (dir > 0) { ctx.moveTo(x + d, y + h); ctx.lineTo(x + d + h, y); }
        else { ctx.moveTo(x + d, y); ctx.lineTo(x + d + h, y + h); }
      }
      ctx.stroke();
    }

    redLane(ep, j, y, P, sel) {
      const ctx = this.ctx, lane = ep.redLanes[j], h = ROW.red, n = lane.length;
      const decoy = ep.contacts && ep.contacts.kinds[j] !== "drone";
      this.caption(PEV.scene.contactName(ep, j).replace("Civilian ", "Civ "), y + h / 2, P, sel ? P.ink : decoy ? P.ink3 : P.red);
      if (sel) { ctx.fillStyle = P.selRow; ctx.fillRect(GUTTER - 2, y - 2, this.W - RIGHT - GUTTER + 4, h + 4); }
      for (const [a, b] of this.runs(n, (t) => lane[t].outcome + lane[t].role)) {
        const c = lane[a], x0 = this.x(a), w = this.x(b) - x0;
        if (c.outcome !== "active") { ctx.fillStyle = P.hair; ctx.fillRect(x0, y + h / 2 - 0.5, w, 1); continue; }
        if (c.role === "evade") { ctx.strokeStyle = P.red; ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, y + 0.5, w - 1, h - 1); }
        else { ctx.fillStyle = decoy ? P.hair : P.redA(0.62); ctx.fillRect(x0, y, w, h); }
        ctx.fillStyle = P.bg; ctx.fillRect(x0, y, 1, h);
      }
      ctx.fillStyle = P.blue;
      for (const [a, b] of this.runs(n, (t) => lane[t].seen)) {
        if (!lane[a].seen) continue;
        ctx.fillRect(this.x(a), y + h + 0.5, Math.max(1.5, this.x(b) - this.x(a)), 1.5);
      }
    }

    cbfTrack(ep, y, P) {
      const ctx = this.ctx, h = ROW.cbf - 4, top = y + 2, max = Math.max(1, ep.cbfMax);
      this.caption("filter", top + h / 2, P);
      ctx.fillStyle = P.hair; ctx.fillRect(GUTTER, top + h, this.W - RIGHT - GUTTER, 1);
      ctx.beginPath(); ctx.moveTo(this.x(0), top + h);
      ep.cbf.forEach((m, t) => { ctx.lineTo(this.x(t), top + h - (m / max) * h); ctx.lineTo(this.x(t + 1), top + h - (m / max) * h); });
      ctx.lineTo(this.x(ep.cbf.length), top + h); ctx.closePath();
      ctx.fillStyle = P.cbfFill; ctx.fill();
      ctx.strokeStyle = P.ink2; ctx.lineWidth = 0.8; ctx.stroke();
      ctx.font = `8.5px ${MONO}`; ctx.fillStyle = P.ink3; ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText(`max ${ep.cbfMax.toFixed(2)}`, this.x(ep.cbf.length) + 3 > this.W - 40 ? GUTTER + 2 : this.x(ep.cbf.length) + 3, top - 1);
    }

    eventMark(e, x, y, P) {
      const ctx = this.ctx;
      ctx.lineWidth = 1.3;
      if (e.kind === "first_detection") {
        ctx.strokeStyle = P.blue; ctx.fillStyle = P.bg;
        ctx.beginPath(); ctx.arc(x, y, 3.8, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.fillStyle = P.blue; ctx.beginPath(); ctx.arc(x, y, 1.4, 0, Math.PI * 2); ctx.fill();
      } else if (e.kind === "neutralization" || e.kind === "decoy_destroyed") {
        ctx.strokeStyle = P.ink; ctx.fillStyle = P.bg; ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x - 2.5, y - 2.5); ctx.lineTo(x + 2.5, y + 2.5); ctx.moveTo(x + 2.5, y - 2.5); ctx.lineTo(x - 2.5, y + 2.5); ctx.stroke();
      } else if (e.kind === "breach") {
        ctx.fillStyle = P.red; ctx.beginPath(); ctx.moveTo(x, y - 6); ctx.lineTo(x + 5.5, y + 4); ctx.lineTo(x - 5.5, y + 4); ctx.closePath(); ctx.fill();
      } else if (e.kind === "escape" || e.kind === "decoy_left") {
        ctx.strokeStyle = P.red; ctx.beginPath(); ctx.moveTo(x - 4, y + 4); ctx.lineTo(x + 4, y - 4); ctx.moveTo(x, y - 4); ctx.lineTo(x + 4, y - 4); ctx.lineTo(x + 4, y); ctx.stroke();
      } else {
        ctx.strokeStyle = P.ink; ctx.beginPath(); ctx.moveTo(x - 4, y - 4); ctx.lineTo(x + 4, y + 4); ctx.moveTo(x + 4, y - 4); ctx.lineTo(x - 4, y + 4); ctx.moveTo(x, y - 5); ctx.lineTo(x, y + 5); ctx.stroke();
      }
    }

    /* ---- pointer --------------------------------------------------------- */

    local(e) { const r = this.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
    hitAt(x, y) { return this.hits.find((h) => Math.hypot(h.x - x, h.y - y) <= h.r); }
    down(e) {
      const [x, y] = this.local(e), hit = this.hitAt(x, y);
      if (hit) { this.onSeek(hit.frame); return; }
      if (x < GUTTER - 6) return;
      this.dragging = true; this.canvas.setPointerCapture(e.pointerId);
      this.onSeek(this.frameAt(x));
    }
    move(e) {
      const [x, y] = this.local(e);
      if (this.dragging) { this.onSeek(this.frameAt(x)); return; }
      const hit = this.hitAt(x, y);
      this.canvas.style.cursor = hit ? "pointer" : x >= GUTTER - 6 ? "col-resize" : "default";
      if (!this.eps.length) return;
      const frame = hit ? hit.frame : this.frameAt(x);
      const dt = this.eps[0].dims.dt;
      this.onHover(x < GUTTER - 6 ? null : { x: e.clientX, y: e.clientY,
        text: hit ? hit.text : `${(frame * dt).toFixed(1)} s · step ${Math.floor(frame)}` });
    }
    up(e) { this.dragging = false; try { this.canvas.releasePointerCapture(e.pointerId); } catch (_) { /* not captured */ } }
  }

  PEV.Timeline = Timeline;
})(window.PEV = window.PEV || {});
