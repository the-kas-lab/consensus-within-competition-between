"""Per-episode role counts of the main study, measured while its 900 episodes are re-simulated from the cache.

Per step, each active defender's executed role kind (RECON / NEUTRALIZE / DEFEND), split at the team's first
contact (the first step with a non-empty detected_red_union) into "pre:<kind>" and "post:<kind>" defender-steps.

    python3 -m analysis.behaviour_replays --data DATA [--out OUT] [--workers 8] [--limit N]

Writes OUT/behaviour_replays.json.
"""
import os
from collections import Counter

from analysis import common, replay


def extract(e, sim):
    steps = sim.recorder.steps
    first = next((st["step"] for st in steps if st["detected_red_union"]), None)
    active = Counter()
    for st in steps:
        phase = "pre" if first is None or st["step"] < first else "post"
        for a, role in st["blue_roles"].items():
            if st["blue_disabled"][int(a)]:
                continue
            kind = "RECON" if role == "recon" else "NEUTRALIZE" if role == "neutralize" else "DEFEND"
            active[f"{phase}:{kind}"] += 1
    return {"first_contact_step": first, "active": dict(active)}


def main(data, out_path, workers=4, limit=None):
    return replay.collect(data, common.MAIN, extract, workers, limit, out_path=out_path,
                          note="main study, re-simulated from the cache; per-episode active role counts")


if __name__ == "__main__":
    ap = common.parser(__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    res = main(a.data, os.path.join(common.out_dir(a), "behaviour_replays.json"), a.workers, a.limit)
    print(f"{res['n']} episodes, {res['n_reproduced']} reproduced")
