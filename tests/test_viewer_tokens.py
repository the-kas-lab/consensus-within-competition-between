"""The replay viewer draws the Supplementary episode figures (viewer/js/scene.js, printed by headless Chrome), so
its paper tokens (viewer/js/tokens.js) must equal the figure registry (figures/style.py). This pins the source
values, so a registry change cannot leave the viewer behind silently.

    python3 -m pytest -q tests/test_viewer_tokens.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures import style  # noqa: E402

TOKENS = ROOT / "viewer/js/tokens.js"


def tokens():
    out = subprocess.run(["node", "-e", f"process.stdout.write(JSON.stringify(require({json.dumps(str(TOKENS))})))"],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def test_tokens_equal_registry():
    t = tokens()
    want = {
        "REGISTRY_VERSION": style.REGISTRY_VERSION,
        "TEXT": style.TEXT, "WHITE": style.WHITE, "ZERO": style.ZERO, "FILL": style.FILL, "GRID": style.GRID,
        "DEF": style.TEAM["blue"]["color"], "DEF_FILL": style.TEAM["blue"]["fill"],
        "INT": style.TEAM["red"]["color"], "INT_FILL": style.TEAM["red"]["fill"],
        "FONT": style.FONT, "FS": style.FS, "LW_MIN": style.LW_MIN, "LW_RULE": style.LW_RULE, "LW_DATA": style.LW_DATA,
        "WIDTHS_MM": list(style.WIDTHS), "H_MAX_MM": style.H_MAX,
    }
    got = {k: t[k] for k in want}
    assert {k: (v.upper() if isinstance(v, str) else v) for k, v in got.items()} == \
        {k: (v.upper() if isinstance(v, str) else v) for k, v in want.items()}


def test_strokes_never_below_floor():
    t = tokens()
    assert min(t["LW_MIN"], t["LW_RULE"], t["LW_DATA"]) >= style.LW_MIN
