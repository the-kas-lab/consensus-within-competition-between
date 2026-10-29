"""Supplementary episode figures, drawn by the replay viewer: one figure per example episode, with screenshots at
hand-picked steps (figures/episode_moments.json).

Every panel is the viewer's own drawing (viewer/js/scene.js), printed to PDF by headless Chrome
(scripts/replay_media.py), so a figure is exactly what the viewer shows at those steps. The episodes are exported
as bundles by analysis.episode_bundles; every re-simulation equals its stored record (asserted here from
export_log.json). Illustrative, not evidence. Each PDF gets the sidecar the figure tests read, through
style.register_pdf.

    python3 -m analysis.episode_bundles --data DATA        # once, writes OUT/bundles
    python3 -m figures.episodes_si --data DATA [--out OUT] [--figures DIR]
"""
from __future__ import annotations

import json
from pathlib import Path

from figures import paths, style
from scripts import replay_media as rm

EP = paths.AN / "bundles"
OUT = paths.FIG / "supp"
MOMENTS = Path(__file__).resolve().parent / "episode_moments.json"

# figure name, moments key, bundle, extra layers (limited range: radio links and orders never received)
FIGS = [
    ("fig_episode_consensus", "consensus", "E1_cbaa", {}),
    ("fig_episode_independent", "independent", "E1_independent", {}),
    ("fig_episode_limited_range", "limited_range", "c_commander", {"links": True, "orders": True}),
    ("fig_episode_airspace_lowcost", "airspace_lowcost", "d_commander", {}),
    ("fig_episode_airspace_nearfrontier", "airspace_nearfrontier", "e_commander_nearfrontier", {}),
    ("fig_episode_disguise_nowarning", "disguise_nowarning", "f_nowarning", {}),
    ("fig_episode_disguise_warning", "disguise_warning", "g_warning", {}),
]


def main():
    log = json.loads((EP / "export_log.json").read_text())
    assert log["outcome_mismatches"] == 0 and log["format_version"] == 2
    exported = {e["key"]: e for e in log["episodes"]}
    sel = json.loads(MOMENTS.read_text())["moments"]
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    with rm.Chrome() as chrome:
        for name, mkey, key, layers in FIGS:
            e = exported[key]
            assert e["record_equal"] and e["bundle_check"] == "passed", key
            steps = [k for k, _ in sel[mkey]]
            assert max(steps) < e["n_steps"], (name, steps, e["n_steps"])
            spec = {"steps": steps, "headings": [h for _, h in sel[mkey]], "width_mm": style.W2, "layers": layers}
            pdf = OUT / f"{name}.pdf"
            info = rm.figure(str(EP / f"{key}.replay.js"), spec, str(pdf), chrome=chrome)
            style.register_pdf(pdf, supplementary=True)
            meta[name] = {"episode_id": e["episode_id"], "bundle": f"bundles/{key}.replay.js", "steps": steps,
                          "times_s": [round(k * 0.1, 1) for k in steps], "outcome": e["outcome"],
                          "size_mm": [info["width_mm"], info["height_mm"]], "layers": layers}
            print(name, meta[name]["times_s"], e["outcome"]["touchdown"])
    (OUT / "fig_episodes_moments.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
