/* The replay app: loading, the clock, interaction, compare mode, export.
 *
 * One fractional frame drives everything. At 1x a frame lasts dt seconds, so
 * a 400-step episode plays in 40 s. Compare mode shares the clock and the view
 * transform; an episode that ended earlier holds its terminal frame.
 *
 * Loading works from file:// (no server): drag-and-drop, the file picker, or
 * ?bundle=path.replay.js, which is injected as a script calling PEV_REPLAY().
 * .json bundles load by fetch, which browsers refuse over file://. */
(function (PEV) {
  "use strict";
  const G = PEV.geom, $ = (s) => document.querySelector(s);
  const SPEEDS = [0.25, 0.5, 1, 2, 4, 8];
  const params = new URLSearchParams(location.search);

  const state = {
    eps: [], f: 0, playing: false, speed: 1, selected: null, view: null, autoFit: true,
    // What an SI figure draws (scene.FIGURE_LAYERS), plus id labels; the rest are for exploring.
    layers: { wedges: true, region: true, tracks: true, pursuit: true, commander: true, links: true, orders: true,
              falseBeliefs: true, time: false, scaleBar: true, labels: true, beliefs: false, tasks: false, sight: false,
              coverage: false, cbf: false, photos: false },
    photoRoot: "../",
    dirty: true, inspKey: "", recording: null,
  };
  let arenas = [], timeline = null, pal = null, lastTs = null;

  /* ---- palette, from the CSS tokens ------------------------------------ */

  function rgba(hex, a) {
    const h = hex.trim().replace("#", "");
    const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }
  function readPalette() {
    const cs = getComputedStyle(document.documentElement), v = (k) => cs.getPropertyValue(k).trim();
    const ink = v("--ink"), blue = v("--blue"), red = v("--red"), sheet = v("--sheet");
    const dark = document.documentElement.dataset.theme === "dark" ||
      (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
    return {
      dark, table: v("--table"), sheet, bg: v("--table"), ink, ink2: v("--ink-2"), ink3: v("--ink-3"), hair: v("--hair"),
      blue, red, shadow: v("--shadow"), selRow: v("--sel-row"), voidFill: v("--void"),
      grid: rgba(ink, dark ? 0.045 : 0.05), gridMajor: rgba(ink, dark ? 0.1 : 0.11), ring: rgba(ink, dark ? 0.16 : 0.17),
      zone: rgba(ink, dark ? 0.06 : 0.055), contour: rgba(ink, dark ? 0.14 : 0.13), guard: rgba(ink, 0.55),
      inkHatch: rgba(ink, dark ? 0.09 : 0.08), stale: rgba(ink, 0).replace(",0)", ",A)"),
      cbfFill: rgba(ink, 0.12),
      blueA: (a) => rgba(blue, a), redA: (a) => rgba(red, a), sheetA: (a) => rgba(sheet, a),
      serif: v("--font-serif"),
    };
  }

  /* ---- loading ------------------------------------------------------------ */

  function showError(msg) {
    $("#load-error").textContent = msg;
    if (state.eps.length) alert(msg);
  }

  function acceptDoc(doc, append) {
    let eps;
    try { eps = PEV.model.load(doc); } catch (e) { showError(e.message); return; }
    if (append && state.eps.length === 1) state.eps = [state.eps[0], eps[0]];
    else state.eps = eps.slice(0, 2);
    if (eps.length > 2) console.warn(`Bundle holds ${eps.length} episodes; showing the first two.`);
    setup();
  }
  // Script-tag bundles call this.
  window.PEV_REPLAY = (doc) => acceptDoc(doc, state.pendingAppend);

  function loadFiles(files) {
    const list = Array.from(files);
    let append = state.eps.length === 1 && list.length === 1;
    list.forEach((file, idx) => {
      const r = new FileReader();
      r.onload = () => {
        try { acceptDoc(PEV.model.parseText(r.result), append || idx > 0); }
        catch (e) { showError(`${file.name}: ${e.message}`); }
      };
      r.readAsText(file);
    });
  }

  function loadUrl(url, append) {
    if (/\.js(\?|$)/.test(url)) {
      state.pendingAppend = append;
      const s = document.createElement("script");
      s.src = url;
      s.onerror = () => showError(`Could not load ${url}.`);
      document.head.appendChild(s);
    } else {
      fetch(url).then((r) => r.text()).then((t) => acceptDoc(PEV.model.parseText(t), append))
        .catch(() => showError(`Could not fetch ${url}. Over file://, use the .replay.js form or drop the file.`));
    }
  }

  /* ---- setup ------------------------------------------------------------ */

  function setup() {
    const eps = state.eps, app = $("#app");
    app.dataset.mode = eps.length > 1 ? "compare" : "single";
    const holder = $("#arenas");
    holder.innerHTML = "";
    arenas = eps.map((ep, k) => {
      const fig = document.createElement("figure");
      fig.className = "arena-fig";
      const cap = document.createElement("figcaption");
      if (eps.length > 1) {
        const oc = ep.outcomeSentence();
        cap.innerHTML = `<b>${["A", "B"][k]}</b>${esc(ep.label)}<span class="oc">${esc(oc.lead)} · ${esc(oc.rest)}</span>`;
      }
      const cv = document.createElement("div");
      cv.className = "arena-host";
      fig.append(cap, cv); holder.append(fig);
      const a = new PEV.Arena(cv);
      bindArena(a, k);
      return a;
    });
    timeline.setEpisodes(eps);
    header();
    state.f = Math.min(state.f, maxFrame());
    state.autoFit = true;
    ["#btn-still", "#btn-video", "#btn-svg"].forEach((s) => ($(s).disabled = false));
    resize();
    markDirty();
  }

  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const maxFrame = () => Math.max(0, ...state.eps.map((e) => e.lastFrame));

  function header() {
    const eps = state.eps, ep = eps[0], h = ep.header;
    const fam = h.family.replace("_", "-");
    $("#title").innerHTML = eps.length > 1 && eps[1].header.seed === h.seed
      ? `Seed ${h.seed} <em>${fam}</em> · A vs B`
      : `Seed ${h.seed} <em>${fam}</em>`;
    const chips = eps.length > 1
      ? [`A · ${eps[0].label}`, `B · ${eps[1].label}`, `belief ${h.belief_mode}`, `stations ${h.station_heading}`]
      : ep.metaChips();
    $("#meta").innerHTML = chips.map((c) => `<span>${esc(c)}</span>`).join("");
    if (eps.length === 1) {
      const oc = ep.outcomeSentence();
      $("#outcome").innerHTML = `<div class="line"><b>${esc(oc.lead)}.</b> <span class="rest">${esc(oc.rest)}</span></div>`;
    } else $("#outcome").innerHTML = "";
    document.title = `Seed ${h.seed} · ${fam} · Episode Replay`;
  }

  /* ---- layout -------------------------------------------------------------- */

  function resize() {
    arenas.forEach((a) => a.resize());
    if (arenas.length && (state.autoFit || !state.view)) {
      const a = arenas[0];
      state.view = PEV.Arena.fitView(state.eps[0], a.W, a.H);
    }
    timeline.resize();
    markDirty();
  }

  /* ---- clock and render ---------------------------------------------------- */

  function markDirty() { state.dirty = true; }

  function frame(ts) {
    const dt = lastTs == null ? 0 : Math.min(0.1, (ts - lastTs) / 1000);
    lastTs = ts;
    if (state.playing && state.eps.length) {
      state.f += (dt * state.speed) / state.eps[0].dims.dt;
      if (state.f >= maxFrame()) { state.f = maxFrame(); setPlaying(false); }
      state.dirty = true;
    }
    if (state.dirty) { render(); state.dirty = false; }
    requestAnimationFrame(frame);
  }

  function renderState() {
    return { pal, layers: state.layers, selected: state.selected, photoRoot: state.photoRoot };
  }

  function render() {
    if (!state.eps.length) return;
    const st = renderState();
    arenas.forEach((a, k) => { a.view = state.view; a.draw(state.eps[k], state.f, st); });
    timeline.draw(state.f, pal, state.selected);
    const ep = state.eps[0];
    $("#clock-t").textContent = (Math.floor(state.f) * ep.dims.dt).toFixed(1);
    $("#clock-step").textContent = `step ${Math.floor(state.f)} / ${maxFrame()}`;
    const key = `${Math.floor(state.f)}|${JSON.stringify(state.selected)}|${state.eps.length}|${pal.dark}`;
    if (key !== state.inspKey) {
      state.inspKey = key;
      PEV.inspector.render($("#inspector"), state.eps, state.f, state.selected);
      eventLabel();
    }
  }

  function setPlaying(p) {
    if (p && state.f >= maxFrame()) state.f = 0;
    state.playing = p;
    $("#play-icon").innerHTML = p ? '<path d="M4,2H7V14H4ZM9,2H12V14H9Z" fill="currentColor"/>' : '<path d="M4,2L13,8L4,14Z" fill="currentColor"/>';
    $("#btn-play").setAttribute("aria-label", p ? "Pause" : "Play");
    markDirty();
  }
  function seek(f) { state.f = Math.max(0, Math.min(maxFrame(), f)); markDirty(); }
  function step(n) { setPlaying(false); seek(Math.round(state.f) + n); }
  function setSpeed(s) {
    state.speed = s;
    document.querySelectorAll("#speeds button").forEach((b) => b.classList.toggle("on", +b.dataset.speed === s));
  }

  /* ---- events ------------------------------------------------------------ */

  function allEvents() {
    const out = [];
    state.eps.forEach((ep, k) => ep.events.forEach((e) => out.push(Object.assign({ ep: k }, e))));
    return out.sort((a, b) => a.frame - b.frame);
  }
  function jumpEvent(dir) {
    const evs = allEvents(), f = state.f;
    const e = dir > 0 ? evs.find((x) => x.frame > f + 1e-6) : [...evs].reverse().find((x) => x.frame < f - 1e-6);
    if (e) { setPlaying(false); seek(e.frame); }
  }
  function eventLabel() {
    const evs = allEvents(), el = $("#ev-label");
    if (!evs.length) { el.textContent = "no events recorded"; return; }
    const next = evs.find((x) => x.frame > state.f + 1e-6);
    const prefix = (e) => (state.eps.length > 1 ? ["A", "B"][e.ep] + " · " : "");
    el.textContent = next ? `next: ${prefix(next)}${next.text} · ${(next.frame * state.eps[0].dims.dt).toFixed(1)} s`
      : `last: ${prefix(evs[evs.length - 1])}${evs[evs.length - 1].text}`;
  }

  /* ---- arena interaction ------------------------------------------------- */

  function bindArena(a, k) {
    const cv = a.host;
    let drag = null;
    cv.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = cv.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top;
      const before = G.toWorld(state.view, a.W, a.H, sx, sy);
      const z = Math.exp(-e.deltaY * 0.0015);
      state.view = Object.assign({}, state.view, { scale: Math.max(1, Math.min(200, state.view.scale * z)) });
      const after = G.toWorld(state.view, a.W, a.H, sx, sy);
      state.view.cx += before[0] - after[0]; state.view.cy += before[1] - after[1];
      state.autoFit = false; markDirty();
    }, { passive: false });
    cv.addEventListener("pointerdown", (e) => {
      const r = cv.getBoundingClientRect();
      drag = { x: e.clientX, y: e.clientY, sx: e.clientX - r.left, sy: e.clientY - r.top, moved: false, view: Object.assign({}, state.view) };
      cv.setPointerCapture(e.pointerId);
    });
    cv.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (!drag.moved && Math.hypot(dx, dy) < 3) return;
      drag.moved = true; cv.classList.add("panning");
      state.view = Object.assign({}, drag.view, { cx: drag.view.cx - dx / drag.view.scale, cy: drag.view.cy + dy / drag.view.scale });
      state.autoFit = false; markDirty();
    });
    cv.addEventListener("pointerup", () => {
      if (drag && !drag.moved) {
        const hit = a.pick(state.eps[k], state.f, drag.sx, drag.sy);
        state.selected = hit && state.selected && hit.team === state.selected.team && hit.id === state.selected.id ? null : hit;
        markDirty();
      }
      drag = null; cv.classList.remove("panning");
    });
    cv.addEventListener("dblclick", () => { state.autoFit = true; resize(); });
  }

  /* ---- export: still, SVG and video -------------------------------------- */

  // The exported drawing is scene.frameDoc -- exactly what scripts/replay_media.py
  // captures from render.html: a heading line, the panel, the key. In compare
  // mode the two episodes sit side by side in one SVG.
  function exportSvg(f) {
    const docs = state.eps.map((ep) => PEV.scene.frameDoc(ep, Math.min(f, ep.lastFrame), {
      width_mm: 120, title: ep.label, layers: Object.assign({}, state.layers, { labels: false, time: true }),
      photoRoot: state.photoRoot,
    }));
    if (docs.length === 1) return docs[0];
    const MM = PEV.scene.MM, w = docs.reduce((a, d) => a + d.width_mm, 0), h = Math.max(...docs.map((d) => d.height_mm));
    let x = 0;
    const inner = docs.map((d) => { const s = d.svg.replace("<svg ", `<svg x="${x * MM}" y="0" `); x += d.width_mm; return s; }).join("");
    return { svg: `<svg xmlns="http://www.w3.org/2000/svg" width="${w}mm" height="${h}mm" viewBox="0 0 ${w * MM} ${h * MM}">${inner}</svg>`, width_mm: w, height_mm: h };
  }

  // Rasterise an SVG document onto a canvas at `pxPerMm`.
  function rasterise(doc, canvas, pxPerMm) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      const url = URL.createObjectURL(new Blob([doc.svg], { type: "image/svg+xml" }));
      img.onload = () => {
        canvas.width = Math.round(doc.width_mm * pxPerMm); canvas.height = Math.round(doc.height_mm * pxPerMm);
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#FFFFFF"; ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(url); resolve(canvas);
      };
      img.onerror = (e) => { URL.revokeObjectURL(url); reject(e); };
      img.src = url;
    });
  }

  function download(blob, name) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }
  const stem = () => {
    const h = state.eps[0].header;
    return `replay_${h.seed}${state.eps.length > 1 ? "_compare" : ""}_t${(Math.floor(state.f) * state.eps[0].dims.dt).toFixed(1)}`;
  };

  async function still() {
    const c = await rasterise(exportSvg(Math.floor(state.f)), document.createElement("canvas"), 12);
    c.toBlob((b) => download(b, stem() + ".png"), "image/png");
  }
  function saveSvg() {
    download(new Blob([exportSvg(Math.floor(state.f)).svg], { type: "image/svg+xml" }), stem() + ".svg");
  }

  // In-browser recording: frames rendered one by one (not tied to the screen's
  // frame rate) at the current speed, 30 frames/s. For SI-quality H.264 use
  // scripts/replay_media.py, which captures the same drawing in headless Chrome.
  async function video() {
    if (state.recording) { state.recording.stop = true; return; }
    const types = ["video/mp4;codecs=avc1", "video/webm;codecs=vp9", "video/webm"];
    const mime = types.find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t));
    if (!mime) { alert("This browser cannot record canvas video."); return; }
    setPlaying(false);
    const c = document.createElement("canvas"), fps = 30, per = state.speed / fps / state.eps[0].dims.dt;
    await rasterise(exportSvg(state.f), c, 8);
    const stream = c.captureStream(0), track = stream.getVideoTracks()[0];
    const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 12e6 });
    const chunks = [], name = stem();
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    const done = new Promise((r) => (rec.onstop = r));
    state.recording = { stop: false };
    $("#rec").hidden = false;
    rec.start(250);
    const frames = [];
    for (let f = Math.floor(state.f); f < maxFrame(); f += per) frames.push(f);
    frames.push(maxFrame());
    for (let i = 0; i < fps; i++) frames.push(maxFrame());            // hold the last frame for 1 s
    for (const f of frames) {
      if (state.recording.stop) break;
      await rasterise(exportSvg(f), c, 8);
      track.requestFrame && track.requestFrame();
      seek(f);
      $("#rec-t").textContent = (Math.floor(f) * state.eps[0].dims.dt).toFixed(1);
      await new Promise((r) => setTimeout(r, 1000 / fps));
    }
    rec.stop(); await done;
    state.recording = null; $("#rec").hidden = true;
    download(new Blob(chunks, { type: mime }), `${name}.${mime.startsWith("video/mp4") ? "mp4" : "webm"}`);
  }

  /* ---- theme ------------------------------------------------------------ */

  function applyTheme(t) {
    if (t) document.documentElement.dataset.theme = t;
    pal = readPalette();
    state.inspKey = "";
    markDirty();
  }
  function toggleTheme() {
    const t = readPalette().dark ? "light" : "dark";
    try { localStorage.setItem("pev-theme", t); } catch (_) { /* storage unavailable */ }
    applyTheme(t);
  }

  function syncLayerChips() {
    document.querySelectorAll("[data-layer]").forEach((b) => b.classList.toggle("on", !!state.layers[b.dataset.layer]));
  }
  function toggleLayer(k) { state.layers[k] = !state.layers[k]; syncLayerChips(); markDirty(); }

  /* ---- probe mode (verification) ------------------------------------------- */

  // ?probe=1&steps=0,120,240: render each step exactly and publish what the
  // renderer drew, for verification/verify_replay_viewer.py.
  function runProbe() {
    const steps = (params.get("steps") || "0").split(",").map(Number);
    const out = { view: null, arenas: [] };
    arenas.forEach((a, k) => {
      const rec = { episode: k, W: a.W, H: a.H, dpr: a.dpr, samples: [] };
      for (const s of steps) {
        if (s > state.eps[k].lastFrame) continue;
        state.f = s; a.probe = [];
        a.view = state.view;
        a.draw(state.eps[k], s, renderState());
        rec.samples.push({ step: s, view: Object.assign({}, a.view), drawn: a.probe });
        a.probe = null;
      }
      out.arenas.push(rec);
    });
    const pre = $("#probe-out");
    pre.textContent = JSON.stringify(out);
    pre.dataset.ready = "1";
  }

  // ?selftest=export: render a still and record ~1 s of video in-page, then
  // publish their sizes -- lets headless Chrome confirm both export paths work.
  async function selfTestExport() {
    const out = { still: null, video: null, svg: null };
    const doc = exportSvg(Math.floor(state.f));
    out.svg = { bytes: doc.svg.length, width_mm: doc.width_mm, height_mm: doc.height_mm };
    const c = await rasterise(doc, document.createElement("canvas"), 8);
    const png = await new Promise((r) => c.toBlob(r, "image/png"));
    out.still = { width: c.width, height: c.height, bytes: png ? png.size : 0 };
    if (params.get("emit") === "1") out.still.dataUrl = c.toDataURL("image/png");
    const types = ["video/mp4;codecs=avc1", "video/webm;codecs=vp9", "video/webm"];
    const mime = types.find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t));
    if (mime) {
      const rec = new MediaRecorder(c.captureStream(30), { mimeType: mime });
      const chunks = [];
      rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      const done = new Promise((r) => (rec.onstop = r));
      rec.start(100);
      for (let k = 0; k < 30; k++) {
        state.f = Math.min(maxFrame(), state.f + 4);
        await rasterise(exportSvg(state.f), c, 8);
        await new Promise((r) => setTimeout(r, 33));
      }
      rec.stop(); await done;
      out.video = { mime, bytes: chunks.reduce((a, b) => a + b.size, 0) };
    }
    const pre = $("#probe-out");
    pre.textContent = JSON.stringify(out);
    pre.dataset.ready = "1";
  }

  /* ---- wiring ------------------------------------------------------------ */

  function init() {
    let saved = null;
    try { saved = localStorage.getItem("pev-theme"); } catch (_) { /* storage unavailable */ }
    applyTheme(params.get("theme") || saved);
    timeline = new PEV.Timeline($("#timeline"), (f) => { setPlaying(false); seek(f); }, (h) => {
      const tip = $("#tip");
      if (!h) { tip.hidden = true; return; }
      tip.hidden = false; tip.textContent = h.text; tip.style.left = h.x + "px"; tip.style.top = h.y + "px";
    });
    if (params.get("layers")) params.get("layers").split(",").forEach((kv) => {
      const [k, v] = kv.split(":"); if (k in state.layers) state.layers[k] = v !== "0";
    });
    syncLayerChips(); setSpeed(1);

    $("#btn-open").onclick = $("#btn-open-2").onclick = () => $("#file").click();
    $("#file").onchange = (e) => { loadFiles(e.target.files); e.target.value = ""; };
    $("#btn-play").onclick = () => setPlaying(!state.playing);
    $("#btn-back").onclick = () => step(-1);
    $("#btn-fwd").onclick = () => step(1);
    $("#btn-start").onclick = () => { setPlaying(false); seek(0); };
    $("#btn-end").onclick = () => { setPlaying(false); seek(maxFrame()); };
    $("#btn-prev-ev").onclick = () => jumpEvent(-1);
    $("#btn-next-ev").onclick = () => jumpEvent(1);
    $("#btn-theme").onclick = toggleTheme;
    $("#btn-still").onclick = still;
    $("#btn-video").onclick = video;
    $("#btn-svg").onclick = saveSvg;
    $("#btn-help").onclick = () => ($("#help").hidden = !$("#help").hidden);
    $("#help").onclick = () => ($("#help").hidden = true);
    document.querySelectorAll("#speeds button").forEach((b) => (b.onclick = () => setSpeed(+b.dataset.speed)));
    document.querySelectorAll("[data-layer]").forEach((b) => (b.onclick = () => toggleLayer(b.dataset.layer)));

    const app = $("#app");
    window.addEventListener("dragover", (e) => { e.preventDefault(); app.classList.add("dropping"); });
    window.addEventListener("dragleave", (e) => { if (e.relatedTarget === null) app.classList.remove("dropping"); });
    window.addEventListener("drop", (e) => { e.preventDefault(); app.classList.remove("dropping"); loadFiles(e.dataTransfer.files); });
    window.addEventListener("resize", resize);
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => applyTheme(null));

    window.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key;
      if (k === "?") { $("#help").hidden = !$("#help").hidden; return; }
      if (k === "o") { $("#file").click(); return; }
      if (k === "t") { toggleTheme(); return; }
      if (!state.eps.length) return;
      if (k === " ") { e.preventDefault(); setPlaying(!state.playing); }
      else if (k === "ArrowRight") { e.preventDefault(); step(e.shiftKey ? 10 : 1); }
      else if (k === "ArrowLeft") { e.preventDefault(); step(e.shiftKey ? -10 : -1); }
      else if (k === "Home") { setPlaying(false); seek(0); }
      else if (k === "End") { setPlaying(false); seek(maxFrame()); }
      else if (k === "]") setSpeed(SPEEDS[Math.min(SPEEDS.length - 1, SPEEDS.indexOf(state.speed) + 1)]);
      else if (k === "[") setSpeed(SPEEDS[Math.max(0, SPEEDS.indexOf(state.speed) - 1)]);
      else if (k === "e") jumpEvent(1);
      else if (k === "E") jumpEvent(-1);
      else if (/^[1-9]$/.test(k)) { const id = +k - 1; if (id < state.eps[0].nb) { state.selected = { team: "blue", id }; markDirty(); } }
      else if (k === "Escape") { if (state.recording) state.recording.stop = true; else { state.selected = null; $("#help").hidden = true; markDirty(); } }
      else if (k === "c") toggleLayer("wedges");
      else if (k === "b") toggleLayer("beliefs");
      else if (k === "r") toggleLayer("tracks");
      else if (k === "g") toggleLayer("coverage");
      else if (k === "l") toggleLayer("labels");
      else if (k === "n") { toggleLayer("links"); toggleLayer("orders"); }
      else if (k === "p") toggleLayer("photos");
      else if (k === "f") { state.autoFit = true; resize(); }
      else if (k === "s") still();
      else if (k === "v") video();
    });

    requestAnimationFrame(frame);

    const bundles = params.getAll("bundle");
    if (bundles.length) {
      // load sequentially so the second one appends into compare mode
      const next = (i) => {
        if (i >= bundles.length) return afterLoad();
        const before = state.eps.length;
        loadUrl(bundles[i], i > 0);
        const wait = () => (state.eps.length > before || $("#load-error").textContent ? next(i + 1) : setTimeout(wait, 30));
        wait();
      };
      next(0);
    }
  }

  // URL conveniences, applied once the bundle(s) are in.
  function afterLoad() {
    if (!state.eps.length) return;
    if (params.get("f")) seek(+params.get("f"));
    const sel = params.get("select");
    if (sel && /^[br]\d+$/.test(sel)) state.selected = { team: sel[0] === "b" ? "blue" : "red", id: +sel.slice(1) };
    if (params.get("play") === "1") setPlaying(true);
    if (params.get("probe") === "1") { resize(); runProbe(); }
    if (params.get("selftest") === "export") selfTestExport();
    if (params.get("selftest") === "perf") {
      // Per-part timing at 60 scattered frames. A 1-pixel read forces the
      // deferred rasterisation into each measurement; "flush" is that read alone.
      const T = { arena: [], timeline: [], inspector: [], flush: [] }, st = renderState();
      const flush = () => { arenas.forEach((a) => a.host.getBoundingClientRect() && a.host.firstChild && a.host.firstChild.getBBox()); timeline.ctx.getImageData(0, 0, 1, 1); };
      const time = (k, fn) => { const t0 = performance.now(); fn(); flush(); T[k].push(performance.now() - t0); };
      for (let k = 0; k < 60; k++) {
        state.f = (k * 7.37) % maxFrame();
        time("flush", () => {});
        time("arena", () => arenas.forEach((a, i) => { a.view = state.view; a.draw(state.eps[i], state.f, st); }));
        time("timeline", () => timeline.draw(state.f, pal, state.selected));
        time("inspector", () => PEV.inspector.render($("#inspector"), state.eps, state.f, { team: "blue", id: 0 }));
      }
      const med = (xs) => xs.sort((p, q) => p - q)[Math.floor(xs.length / 2)];
      const out = { W: arenas[0].W, H: arenas[0].H, dpr: arenas[0].dpr };
      for (const k of Object.keys(T)) out[k + "_median_ms"] = +med(T[k]).toFixed(2);
      const pre = $("#probe-out"); pre.textContent = JSON.stringify(out); pre.dataset.ready = "1";
    }
    render();
    document.body.dataset.ready = "1";
  }

  PEV.app = { state, seek, setPlaying, exportSvg };
  document.addEventListener("DOMContentLoaded", init);
})(window.PEV = window.PEV || {});
