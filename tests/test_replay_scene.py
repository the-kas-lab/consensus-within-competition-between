"""The viewer's one drawing (viewer/js/scene.js) keeps the paper's figure grammar, and the driver
(scripts/replay_media.py) turns it into files without a second renderer.

* The scene runs headless in Node, exactly the code the viewer runs; every colour it writes is a token of
  viewer/js/tokens.js, every stroke is at least 1 pt and every glyph is 7 pt Arial.
* The driver prints a figure to a one-page vector PDF at the declared width with Arial embedded as TrueType,
  and a still to a PNG of the requested width.

Free: a rule-based episode, no network. Skips without node (scene) or Chrome (driver).
"""

import json
import os
import re
import shutil
import subprocess

import pytest

from conftest import build_simulation, run_quiet
from scripts import export_replay as er

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = os.path.join(ROOT, "viewer", "js")
EPISODE = dict(belief_mode="local", seed=7, sim_steps=120, arena_size=40.0, num_blues=3, num_reds=2)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    sim = build_simulation(**EPISODE)
    capture = er.attach_capture(sim)
    run_quiet(sim)
    ep = er.episode_bundle(sim, capture, {"preset": None, "arm": None, "seed": 7, "family": sim.scenario_family,
                                          "belief_mode": sim.belief_mode, "cfg": {}, "label": "scene test"})
    er.check_bundle(ep, sim, capture)
    path = tmp_path_factory.mktemp("scene") / "b.replay.js"
    er.write_bundle(er.bundle_document([ep]), str(path))
    return str(path), ep


def node(script):
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, cwd=JS, check=True)
    return out.stdout


def render_svgs(path, layers):
    return json.loads(node(f"""
const fs = require("fs"), sc = require("./scene.js");
const t = fs.readFileSync({json.dumps(path)}, "utf8");
const ep = JSON.parse(t.slice("PEV_REPLAY(".length).replace(/\\);\\s*$/, "")).episodes[0];
const L = {json.dumps(layers)};
const last = sc.lastStep(ep);
const fig = sc.figure(ep, {{steps: [0, Math.floor(last / 2), last], headings: ["Start", "Middle", "End"], layers: L}});
const frame = sc.frameDoc(ep, last / 3 + 0.5, {{title: "Scene test", layers: L}});
process.stdout.write(JSON.stringify({{figure: fig.svg, width_mm: fig.width_mm, frame: frame.svg}}));
"""))


@pytest.mark.parametrize("layers", [{}, {"labels": True, "beliefs": True, "tasks": True, "sight": True,
                                         "coverage": True, "cbf": True, "links": True, "orders": True}],
                         ids=["figure", "explore"])
def test_scene_writes_only_the_paper_grammar(bundle, layers):
    tokens = json.loads(node('process.stdout.write(JSON.stringify(require("./tokens.js")))'))
    allowed = {tokens[k].upper() for k in ("TEXT", "WHITE", "ZERO", "FILL", "GRID", "DEF", "DEF_FILL", "INT", "INT_FILL")}
    out = render_svgs(bundle[0], layers)
    assert out["width_mm"] == 180
    for svg in (out["figure"], out["frame"]):
        colours = set(re.findall(r'(?:fill|stroke)="(#[0-9A-Fa-f]{6})"', svg))
        assert colours and {c.upper() for c in colours} <= allowed, colours - allowed
        widths = [float(w) for w in re.findall(r'stroke-width="([\d.]+)"', svg)]
        assert widths and min(widths) >= tokens["LW_MIN"]
        texts = re.findall(r"<text [^>]*>", svg)
        assert texts
        for t in texts:
            assert f'font-size="{tokens["FS"]:g}"' in t and f'font-family="{tokens["FONT"]}"' in t, t
        assert "opacity" not in svg and "pattern" not in svg      # no alpha, no hatching


def test_driver_prints_a_figure_and_a_still(bundle, tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    from scripts import replay_media as rm
    try:
        rm.find_chrome()
    except SystemExit:
        pytest.skip("no Chrome/Chromium")
    path, ep = bundle
    last = ep["dims"]["n_steps"] - 1
    pdf = tmp_path / "fig.pdf"
    info = rm.figure(path, {"steps": [0, last // 2, last], "headings": ["Start", "Middle", "End"]}, str(pdf))
    doc = pymupdf.open(str(pdf))
    assert doc.page_count == 1
    assert abs(doc[0].rect.width / 72 * 25.4 - info["width_mm"]) < 0.15
    if shutil.which("pdffonts"):
        fonts = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True, check=True).stdout
        assert "Arial" in fonts and "TrueType" in fonts and "Type 3" not in fonts
    png = tmp_path / "still.png"
    rm.still(path, last / 2, str(png), width_px=800)
    with open(png, "rb") as f:
        head = f.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n" and int.from_bytes(head[16:20], "big") == 800
