/* Bundle -> Episode: version check, and everything the timeline and inspector
 * precompute once per episode (task lanes, detection lanes, CBF track, events).
 * Nothing here invents data: every lane is a per-step read of the recorder
 * (or, where named, the exporter's read-only instrumentation). */
(function (PEV) {
  "use strict";
  const G = PEV.geom;
  const FORMAT = "pev-replay", VERSIONS = [1, 2];

  // Task codes shared by the ribbons, the glyphs and the legend.
  const KIND = { NONE: 0, NEUTRALIZE: 1, SCREEN: 2, STATION: 3, RECON: 4 };
  const TAG = { NEUTRALIZE: "NEU", SCREEN: "SCR", STATION: "DEF", RECON: "REC", DEFAULT: "DEF", NONE: "—" };

  function parseText(text) {
    let t = text.trim();
    if (t.startsWith("PEV_REPLAY(")) t = t.slice("PEV_REPLAY(".length).replace(/\);?\s*$/, "");
    return JSON.parse(t);
  }

  function validate(doc) {
    if (!doc || doc.format !== FORMAT) throw new Error("Not a pev-replay bundle (missing format tag).");
    if (!VERSIONS.includes(doc.format_version))
      throw new Error(`Bundle format_version ${doc.format_version} is not supported by this viewer (knows ${VERSIONS.join(", ")}).`);
    if (!Array.isArray(doc.episodes) || !doc.episodes.length) throw new Error("Bundle holds no episodes.");
  }

  // Event wording in the paper's terms: defenders D1.., intruders I1.., birds and civilian drones by kind.
  const D = (i) => `D${i + 1}`;
  const EVENT_TEXT = {
    first_detection: (e, ep) => `First sighting · ${e.reds.map((r) => name(ep, r)).join(", ")}`,
    neutralization: (e, ep) => `${name(ep, e.red)} destroyed · ${D(e.blue)} disabled`,
    decoy_destroyed: (e, ep) => `${name(ep, e.red)} destroyed · ${D(e.blue)} disabled`,
    escape: (e, ep) => `${name(ep, e.red)} left the arena`,
    decoy_left: (e, ep) => `${name(ep, e.red)} left the arena`,
    breach: (e, ep) => `Breach · ${name(ep, e.red)} inside the defended region`,
    collision: (e) => `Defender collision · ${D(e.blues[0])}, ${D(e.blues[1])}`,
    solver_failure: () => "Solver failure · episode aborted",
  };
  function name(ep, j) {
    if (PEV.scene && ep.contacts) return PEV.scene.contactName(ep, j);
    return `I${j + 1}`;
  }

  class Episode {
    constructor(raw, index) {
      Object.assign(this, raw);
      this.index = index;
      const d = this.dims, n = d.n_steps;
      this.nb = d.num_blues; this.nr = d.num_reds;
      this.lastFrame = d.n_steps - 1;
      // The last recorded step is drawn with the stored outcome, so an event first
      // visible in the terminal frame is shown (and jumped to) at that step.
      const lastStep = n - 1;
      this.events = raw.derived.events.map((e) => Object.assign({}, e, {
        frame: Math.min(e.frame, lastStep), text: EVENT_TEXT[e.kind](e, raw),
      }));

      // Per blue, per step: assigned kind (the allocator's word), whether it is
      // explicit, the executed kind, divergence, own detection, disabled.
      this.blueLanes = [];
      for (let i = 0; i < this.nb; i++) {
        const lane = [];
        for (let t = 0; t < n; t++) {
          const st = this.steps[t];
          const as = G.assignedTask(this, t, i), ef = G.effectiveTask(this, t, i);
          lane.push({
            disabled: st.blue_disabled[i],
            kind: as.kind === "DEFAULT" ? ef.kind : as.kind,
            explicit: as.kind !== "DEFAULT",
            diverge: !!G.divergence(this, t, i),
            detect: (st.per_blue_detected_reds[String(i)] || []).length > 0,
          });
        }
        this.blueLanes.push(lane);
      }
      this.redLanes = [];
      for (let j = 0; j < this.nr; j++) {
        const lane = [];
        for (let t = 0; t < n; t++) {
          const st = this.steps[t];
          lane.push({ role: st.red_roles[String(j)] || null, outcome: st.red_outcomes[j],
                      seen: st.detected_red_union.includes(j) });
        }
        this.redLanes.push(lane);
      }
      // CBF: the largest per-agent intervention among active blues.
      this.cbf = [];
      this.cbfMax = 0;
      for (let t = 0; t < n; t++) {
        let m = 0;
        for (let i = 0; i < this.nb; i++) if (!this.steps[t].blue_disabled[i]) m = Math.max(m, G.intervention(this, t, i).mag);
        this.cbf.push(m);
        this.cbfMax = Math.max(this.cbfMax, m);
      }
      this.interventionFraction = [];
      for (let i = 0; i < this.nb; i++) {
        let hit = 0, act = 0;
        for (let t = 0; t < n; t++) {
          if (this.steps[t].blue_disabled[i]) continue;
          act++;
          if (G.intervention(this, t, i).mag > this.geometry.intervention_tol) hit++;
        }
        this.interventionFraction.push(act ? hit / act : 0);
      }
    }

    time(f) { return f * this.dims.dt; }

    // One sentence, stated once, in the paper's terms. Not a score: see the design note.
    outcomeSentence() {
      const h = this.header, T = (this.lastFrame * this.dims.dt).toFixed(1);
      const nInt = this.contacts ? this.contacts.num_intruders : this.nr;
      const lost = h.n_blues_disabled, parts = [
        `${h.num_neutralized} of ${nInt} intruders destroyed`,
        ...(h.num_escaped ? [`${h.num_escaped} left the arena`] : []),
        lost ? `${lost} defender${lost > 1 ? "s" : ""} disabled` : "no defender disabled",
      ];
      const lead = {
        breach: `Breach at ${T} s`,
        collision: `Defender collision at ${T} s`,
        solver_failure: `Aborted: solver failure at ${T} s`,
        all_reds_resolved: `Every intruder resolved by ${T} s`,
        horizon: `No breach by the ${T} s limit`,
      }[h.reason] || h.reason;
      return { lead, rest: parts.join(" · ") };
    }

    metaChips() {
      const h = this.header, s = this.source || {}, cfg = s.cfg || {}, chips = [];
      if (s.paper) {
        chips.push(s.paper.team, s.paper.condition);
        if (s.paper.setting && s.paper.setting !== "main study") chips.push(s.paper.setting);
      } else {
        const PLANNER = { RuleBasedBlueRolePlanner: "rule planner", DeduplicatingBlueRolePlanner: "dedup planner",
                          LLMRolePlanner: "LLM planner" };
        if (s.preset === "exp4") chips.push(`Exp 4 arm ${s.arm}`);
        else if (s.preset === "exp0") chips.push("Exp 0 config");
        chips.push(h.allocation === "cbaa"
          ? (h.transport === "none" ? "independent selection" : `consensus (${h.transport}), ${h.slots} slot${h.slots > 1 ? "s" : ""}`)
          : `${PLANNER[h.planner] || h.planner || "role planner"}${cfg.rule ? `, ${cfg.rule}` : ""}`);
      }
      const net = this.network, rc = net ? net.r_comm : cfg.r_comm;
      if (h.belief_mode === "networked") chips.push(rc == null || rc === "inf" || rc === Infinity ? "radio range unlimited" : `radio range ${rc} units`);
      if (net && net.loss_model) chips.push(`message loss: ${typeof net.loss_model === "object" ? (net.loss_model.model || "on") : net.loss_model}`);
      chips.push(`plans every ${h.planning_interval_s.toFixed(1)} s`);
      return chips;
    }
  }

  function load(doc) {
    validate(doc);
    return doc.episodes.map((raw, i) => new Episode(raw, i));
  }

  PEV.model = { load, parseText, validate, KIND, TAG, Episode };
})(window.PEV = window.PEV || {});
