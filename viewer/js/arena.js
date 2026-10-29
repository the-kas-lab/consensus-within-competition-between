/* The arena: one episode at one fractional frame, drawn by scene.js into an SVG.
 *
 * This is the SAME drawing a still, an SI figure panel or a video frame is
 * (scene.panel): paper grammar, sizes in points. The screen shows it at the
 * paper size of an exported frame (PAPER_MM, the panel of scene.frameDoc at
 * 120 mm), enlarged to fit: what you see at the fitted view is the exported
 * still, mark for mark. Zooming changes only the world-to-paper scale, never
 * the size of a mark.
 *
 * view = {cx, cy, scale}: world centre and CSS px per world unit.
 *
 * Probe mode: when `this.probe` is an array, every agent's drawn centre and
 * FOV wedge are logged FROM THE VALUES WRITTEN INTO THE SVG (converted to CSS
 * px), so verification/verify_replay_viewer.py checks what was drawn. */
(function (PEV) {
  "use strict";
  const PAPER_MM = 115;                 // scene.frameDoc's panel at 120 mm (120 - 2 x 2.5)

  class Arena {
    constructor(host) {
      this.host = host;
      this.view = null;
      this.W = 0; this.H = 0; this.dpr = 1;
      this.probe = null;
      this.tr = null;
      this.PT = 0.75;                   // points per CSS px, set by resize()
    }

    resize() {
      const r = this.host.getBoundingClientRect();
      this.dpr = window.devicePixelRatio || 1;
      this.W = Math.max(1, Math.round(r.width));
      this.H = Math.max(1, Math.round(r.height));
      this.PT = (PAPER_MM * 72 / 25.4) / Math.min(this.W, this.H);
    }

    // Fit the arena with a small margin.
    static fitView(ep, W, H) {
      const a = ep.geometry.arena, pad = 0;
      const s = Math.min(W, H) / (a.size + 2 * pad);
      return { cx: (a.min + a.max) / 2, cy: (a.min + a.max) / 2, scale: s };
    }

    window() {
      const v = this.view, w = this.W / v.scale, h = this.H / v.scale;
      return { x0: v.cx - w / 2, y0: v.cy - h / 2, w, h };
    }

    draw(ep, f, st) {
      if (!ep) { this.host.innerHTML = ""; return; }
      const box = { X: 0, Y: 0, W: this.W * this.PT, H: this.H * this.PT };
      const probe = this.probe ? [] : null;
      const r = PEV.scene.panel(ep, f, {
        box, win: this.window(), layers: st.layers, selected: st.selected, id: `a${ep.index}`,
        probe, photoRoot: st.photoRoot,
      });
      this.tr = r.transform;
      this.state = r.state;
      this.host.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="${this.W}" height="${this.H}" ` +
        `viewBox="0 0 ${box.W} ${box.H}" role="img" aria-label="Arena at ${(r.state.k * ep.dims.dt).toFixed(1)} s">${r.svg}</svg>`;
      if (probe) {
        for (const d of probe) {
          const out = Object.assign({}, d);
          if (d.x != null) { out.sx = d.x / this.PT; out.sy = d.y / this.PT; }
          if (d.cone) out.cone = { apex: [d.cone.apex[0] / this.PT, d.cone.apex[1] / this.PT], r: d.cone.r / this.PT, a0: d.cone.a0, a1: d.cone.a1 };
          this.probe.push(out);
        }
      }
    }

    // World point -> CSS px, through the transform the last drawing used.
    S(x, y) { const [px, py] = this.tr.p(x, y); return [px / this.PT, py / this.PT]; }

    pick(ep, f, sx, sy) {
      if (!this.tr) return null;
      const P = this.state.P;
      let best = null, bestD = 14;
      P.b.forEach((q, i) => { const [x, y] = this.S(q.x, q.y), d = Math.hypot(x - sx, y - sy); if (d < bestD) { bestD = d; best = { team: "blue", id: i }; } });
      P.r.forEach((q, j) => { const [x, y] = this.S(q.x, q.y), d = Math.hypot(x - sx, y - sy); if (d < bestD) { bestD = d; best = { team: "red", id: j }; } });
      return best;
    }
  }

  PEV.Arena = Arena;
  PEV.Arena.PAPER_MM = PAPER_MM;
})(window.PEV = window.PEV || {});
