/* render.html: one figure, still or video frame, drawn by scene.js and nothing
 * else, for scripts/replay_media.py to print (PDF) or capture (PNG frames).
 *
 *   render.html?bundle=x.replay.js&kind=figure&spec=<URI-encoded JSON>
 *   render.html?bundle=x.replay.js&kind=frame&spec=...   then PEV.render.frame(f)
 *
 * The page sizes itself to the drawing (CSS @page for print) and sets
 * document.body.dataset.ready once the bundle is in and the first drawing is up.
 */
(function (PEV) {
  "use strict";
  const params = new URLSearchParams(location.search);
  const out = document.getElementById("out");
  let ep = null, spec = null, kind = params.get("kind") || "frame";

  function size(wmm, hmm) {
    let s = document.getElementById("page-size");
    if (!s) { s = document.createElement("style"); s.id = "page-size"; document.head.appendChild(s); }
    s.textContent = `@page { size: ${wmm}mm ${hmm}mm; margin: 0; } html, body { width: ${wmm}mm; height: ${hmm}mm; overflow: hidden; }`;
  }

  function figure() {
    const r = PEV.scene.figure(ep, spec);
    out.innerHTML = r.svg;
    size(r.width_mm, r.height_mm);
    return { width_mm: r.width_mm, height_mm: r.height_mm, keyWidth_mm: r.keyWidth_mm };
  }

  function frame(f) {
    const r = PEV.scene.frameDoc(ep, f, spec);
    out.innerHTML = r.svg;
    size(r.width_mm, r.height_mm);
    return { width_mm: r.width_mm, height_mm: r.height_mm, lastStep: PEV.scene.lastStep(ep) };
  }

  window.PEV_REPLAY = (doc) => {
    try {
      PEV.model.validate(doc);
      ep = doc.episodes[+(params.get("episode") || 0)];
      spec = JSON.parse(params.get("spec") || "{}");
      if (spec.photoRoot == null) spec.photoRoot = "../";
      const info = kind === "figure" ? figure() : frame(+(params.get("f") || 0));
      document.getElementById("probe-out").textContent = JSON.stringify(info);
      document.body.dataset.ready = "1";
    } catch (e) {
      document.getElementById("probe-out").textContent = JSON.stringify({ error: String(e && e.stack || e) });
      document.body.dataset.ready = "error";
    }
  };

  PEV.render = { frame: (f) => frame(f), figure: () => figure(), episode: () => ep };

  const b = params.get("bundle");
  if (b) {
    const s = document.createElement("script");
    s.src = b;
    s.onerror = () => { document.getElementById("probe-out").textContent = JSON.stringify({ error: "could not load " + b }); document.body.dataset.ready = "error"; };
    document.head.appendChild(s);
  }
})(window.PEV = window.PEV || {});
