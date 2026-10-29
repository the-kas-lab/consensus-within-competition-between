"""Does the replay viewer draw what the recorder recorded?

Runs a short, detection-rich episode, exports it with ``scripts.export_replay``,
renders sampled steps in headless Chrome with the viewer's probe mode
(``viewer/index.html?probe=1``), and compares what the renderer DREW -- the
coordinates it wrote into the arena's SVG (scene.js, the drawing every still,
figure and video frame also is) for each agent's centre and FOV wedge --
against:

* the recorder's own arrays on the live ``Simulation`` (not the bundle), pushed
  through the viewer's world->screen transform;
* the FOV half-angle and range from the recorded *config*
  (``fov_half_angle_deg``, ``fov_range``), not from the bundle's geometry;
* the recorder's ``per_blue_detected_reds``: a red must lie inside a blue's
  drawn wedge exactly when that blue's sensor recorded it as detected.

The last check is the one that matters for reading a replay: a drawn cone that
disagreed with detection would make every "why didn't it see that?" answer wrong.

Free: rule-based, no network. Needs Google Chrome/Chromium; exits 2 (skipped)
without one. Run as a package:

    PYTHONPATH=$PWD python3 -m verification.verify_replay_viewer
"""

import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception import sensing  # noqa: E402
from planning.planner import RuleBasedBlueRolePlanner, RuleBasedRedRolePlanner  # noqa: E402
from scripts import export_replay as er  # noqa: E402
from scripts.episode_builder import _Quiet  # noqa: E402
from simulation.testbed import Simulation  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(REPO, "viewer", "index.html")
TOL_PX = 5e-3       # SVG coordinates are written to 0.001 pt (< 0.002 CSS px at the fitted view)
TOL_RAD = 1e-9

# Detects from step 0, ages beliefs, ends with both reds escaping (see
# tests/test_export_replay.py). Seed 7 is the golden seed; nothing is spent.
EPISODE = dict(
    num_blues=3, num_reds=2, seed=7, sim_steps=120, dt=0.1, horizon=60, arena_size=40.0,
    planning_interval_seconds=1.0, belief_mode="local",
    enable_blue_blue_cbf=True, enable_blue_red_cbf=True,
)


def find_chrome():
    env = os.environ.get("CHROME")
    candidates = [env] if env else []
    candidates += [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser"),
    ]
    return next((c for c in candidates if c and os.path.exists(c)), None)


def run_episode():
    sim = Simulation(
        **EPISODE,
        role_planner=RuleBasedBlueRolePlanner(recon_fraction=0.5, min_recon=1),
        red_role_planner=RuleBasedRedRolePlanner(),
    )
    capture = er.attach_capture(sim)
    with _Quiet():
        sim.run()
    return sim, capture


def probe(chrome, bundle_path, steps):
    url = "file://" + VIEWER + "?" + "&".join([
        "bundle=" + bundle_path, "probe=1", "steps=" + ",".join(map(str, steps)), "theme=light",
    ])
    out = subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--window-size=1400,900",
         "--force-device-scale-factor=1", "--allow-file-access-from-files",
         "--virtual-time-budget=8000", "--dump-dom", url],
        capture_output=True, text=True, timeout=120,
    ).stdout
    m = re.search(r'<pre id="probe-out"[^>]*>(.*?)</pre>', out, re.S)
    if not m or not m.group(1).strip():
        raise RuntimeError("the viewer produced no probe output (did the bundle load?)")
    return json.loads(html.unescape(m.group(1)))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def main():
    chrome = find_chrome()
    if chrome is None:
        print("SKIPPED: no Chrome/Chromium found (set CHROME=/path/to/chrome)")
        return 2

    sim, capture = run_episode()
    ep = er.episode_bundle(sim, capture, {"preset": None, "arm": None, "seed": sim.seed,
                                          "family": sim.scenario_family, "belief_mode": sim.belief_mode,
                                          "cfg": {}, "label": "verify"})
    er.check_bundle(ep, sim, capture)

    arrays = sim.recorder.arrays()
    config = sim.recorder.config
    half = math.radians(float(config["fov_half_angle_deg"]))
    rng = float(config["fov_range"])
    n = len(sim.recorder.steps)
    nb, nr, bnx, rnx = sim.num_blues, sim.num_reds, sim.blue_nx, sim.red_nx
    # Recorder rows. The viewer's last frame is the last recorded step, drawn
    # with the stored (terminal) outcome.
    blue_rows = list(arrays["blue_state"])
    red_rows = list(arrays["red_state"])
    steps = sorted({0, 1, n // 4, n // 2, (3 * n) // 4, n - 1})

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "verify.replay.js")
        er.write_bundle(er.bundle_document([ep]), path)
        result = probe(chrome, path, steps)

    arena = result["arenas"][0]
    W, H = arena["W"], arena["H"]
    failures, checks = [], 0
    detect_pairs = detect_pos = 0

    def fail(msg):
        failures.append(msg)

    for sample in arena["samples"]:
        t, v = sample["step"], sample["view"]
        to_screen = lambda x, y: (W / 2 + (x - v["cx"]) * v["scale"], H / 2 - (y - v["cy"]) * v["scale"])  # noqa: E731
        drawn = {}
        for d in sample["drawn"]:                    # a wedge entry, then the glyph entry
            drawn.setdefault((d["team"], d["id"]), {"cone": None}).update(d)
        status = sim.recorder.steps[t] if t < n - 1 else {
            "blue_disabled": ep["terminal"]["blue_disabled"], "red_outcomes": ep["terminal"]["red_outcomes"]}
        for team, count, rows, nx in (("blue", nb, blue_rows, bnx), ("red", nr, red_rows, rnx)):
            for i in range(count):
                d = drawn.get((team, i))
                if d is None or "sx" not in d:
                    fail(f"step {t}: {team} {i} not drawn"); continue
                x, y, theta = (float(rows[t][i * nx + k]) for k in (0, 1, 4))
                sx, sy = to_screen(x, y)
                checks += 1
                if abs(d["sx"] - sx) > TOL_PX or abs(d["sy"] - sy) > TOL_PX:
                    fail(f"step {t}: {team} {i} drawn at ({d['sx']:.6f},{d['sy']:.6f}), recorder ({sx:.6f},{sy:.6f})")
                if team != "blue":
                    continue
                c = d["cone"]
                if status["blue_disabled"][i]:
                    if c is not None:
                        fail(f"step {t}: disabled defender {i} still has a wedge")
                    continue
                if c is None:
                    fail(f"step {t}: defender {i} active but no FOV wedge drawn"); continue
                checks += 1
                # the wedge spans [theta - half, theta + half] (world angles) out to the range
                if (abs(c["apex"][0] - sx) > TOL_PX or abs(c["apex"][1] - sy) > TOL_PX
                        or abs(c["r"] - rng * v["scale"]) > TOL_PX
                        or abs(wrap(c["a0"] - (theta - half))) > TOL_RAD
                        or abs(wrap(c["a1"] - (theta + half))) > TOL_RAD):
                    fail(f"step {t}: defender {i} wedge {c} != apex ({sx},{sy}) r {rng * v['scale']} "
                         f"span [{theta - half}, {theta + half}]")

        # The drawn wedge is the sensor: inside it <=> recorded as detected.
        det = sim.recorder.steps[t]["per_blue_detected_reds"]
        live = sim.recorder.steps[t]
        for i in range(nb):
            c = drawn[("blue", i)]["cone"]
            if c is None:
                continue
            for j in range(nr):
                if live["red_outcomes"][j] != "active":
                    continue
                rx, ry = to_screen(float(red_rows[t][j * rnx]), float(red_rows[t][j * rnx + 1]))
                dx, dy = rx - c["apex"][0], ry - c["apex"][1]
                # screen y points down, so the world angle of the offset is atan2(-dy, dx)
                inside = math.hypot(dx, dy) <= c["r"] + 1e-9 and (
                    math.hypot(dx, dy) <= 1e-9
                    or abs(wrap(math.atan2(-dy, dx) - (c["a0"] + c["a1"]) / 2)) <= (c["a1"] - c["a0"]) / 2 + 1e-12)
                recorded = j in det.get(str(i), [])
                # same predicate on world numbers, from the sensing module itself
                truth = sensing.is_detected(blue_rows[t][i * bnx : i * bnx + 2], blue_rows[t][i * bnx + 4],
                                            red_rows[t][j * rnx : j * rnx + 2], half, rng)
                checks += 1
                detect_pairs += 1
                detect_pos += recorded
                if inside != recorded or truth != recorded:
                    fail(f"step {t}: defender {i} / intruder {j}: inside drawn wedge={inside}, "
                         f"recorded detection={recorded}, sensing.is_detected={truth}")

    n_samples = len(arena["samples"])
    print(f"viewer probe: {n_samples} sampled steps {steps}, arena {W}x{H} px, {checks} checks "
          f"({detect_pairs} blue-red detection pairs, {detect_pos} of them detected)")
    if failures:
        print(f"FAIL: {len(failures)} mismatch(es)")
        for f in failures[:20]:
            print("  " + f)
        return 1
    print("PASS: drawn centres and FOV wedges match the recorder to "
          f"{TOL_PX:g} px / {TOL_RAD:g} rad; drawn wedges agree with every recorded detection")
    return 0


if __name__ == "__main__":
    sys.exit(main())
