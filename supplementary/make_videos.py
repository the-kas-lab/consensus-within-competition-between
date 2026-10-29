"""Supplementary Videos 1-3, drawn by the replay viewer from the bundles of analysis.episode_bundles.

Every frame is the viewer's own drawing (viewer/js/scene.js), captured in headless Chrome and encoded by ffmpeg through
scripts/replay_media.py: the same picture as the viewer on screen and the Supplementary episode figures.

Episodes: one per main-study team on the ideal network with four intruders, seed 4003: the first seed, in order from 4001,
on which every team's breach outcome equals its majority outcome in that cell. Each re-simulation equals its stored record
(bundles/export_log.json).

Frames: four times real time at 20 frames per second (two steps per frame), last frame held for 2 s, 1080 px wide, H.264
(yuv420p, CRF 18).

    python3 -m analysis.episode_bundles --data DATA E1_commander E1_cbaa E1_independent
    python3 supplementary/make_videos.py --bundles DATA/analysis/bundles --out supplementary
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import replay_media as rm  # noqa: E402

VIDS = [("Supplementary_Video_1", "E1_commander", "Commander team"),
        ("Supplementary_Video_2", "E1_cbaa", "Consensus team"),
        ("Supplementary_Video_3", "E1_independent", "Independent team")]


def outcome_text(e):
    o = e["outcome"]
    if o["touchdown"]:
        return f"breach at {o['touchdown_step'] * 0.1:.1f} s"
    return f"no breach; {o['num_neutralized']} of 4 intruders destroyed"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundles", required=True, help="directory with the bundles and export_log.json")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    a = ap.parse_args()
    EP, HERE = Path(a.bundles), Path(a.out)
    log = json.loads((EP / "export_log.json").read_text())
    assert log["outcome_mismatches"] == 0 and log["format_version"] == 2
    ex = {e["key"]: e for e in log["episodes"]}
    out = {}
    for name, key, team in VIDS:
        e = ex[key]
        assert e["record_equal"] and e["bundle_check"] == "passed", key
        spec = {"width_mm": 120, "title": f"{team} · ideal network · four intruders · seed 4003",
                "subtitle": f"Illustrative episode ({outcome_text(e)}) · four times real time"}
        mp4 = HERE / f"{name}.mp4"
        n = rm.video(str(EP / f"{key}.replay.js"), str(mp4), spec=spec, speed=4.0, fps=20, hold_s=2.0, width_px=1080)
        out[name] = {"episode_id": e["episode_id"], "frames": n, "bytes": mp4.stat().st_size,
                     "outcome": outcome_text(e), "duration_s": round(n / 20, 1), "drawn_by": "viewer/js/scene.js"}
        print(name, out[name])
    (HERE / "videos.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
