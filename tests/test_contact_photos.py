"""Contact photos, contact photographs (assets/contact_photos/).

The nine files must be exactly the ones recorded in manifest.json (their
hashes enter the LLM cache keys and the episode records), every one must be
credited in CREDITS.md, and they are the simulator's default image pools.
"""

import hashlib
import json
import os
import re
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import testbed  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SET = os.path.join(ROOT, "assets", "contact_photos")
NAMES = [f"{kind}_{i}" for kind in ("intruder", "bird", "civilian") for i in (1, 2, 3)]


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _manifest():
    with open(os.path.join(SET, "manifest.json")) as f:
        return json.load(f)["files"]


def test_files_match_manifest():
    files = _manifest()
    assert sorted(files) == sorted(NAMES)
    for name in NAMES:
        assert _sha256(os.path.join(SET, f"{name}.jpeg")) == files[name]["sha256"], name


def test_format_and_no_metadata():
    for name in NAMES:
        with Image.open(os.path.join(SET, f"{name}.jpeg")) as im:
            assert im.format == "JPEG" and im.size == (640, 480) and im.mode == "RGB", name
            assert not im.getexif() and "icc_profile" not in im.info and "xmp" not in im.info, name


def test_every_file_credited_with_its_hash():
    with open(os.path.join(SET, "CREDITS.md")) as f:
        credits = f.read()
    files = _manifest()
    for name in NAMES:
        row = [line for line in credits.splitlines() if line.startswith(f"| `{name}.jpeg`")]
        assert len(row) == 1, name
        assert files[name]["sha256"] in row[0], name
        assert re.search(r"CC BY|CC0|Public domain|Unsplash License", row[0]), name


def test_sources_cover_every_original():
    with open(os.path.join(SET, "sources.json")) as f:
        sources = json.load(f)
    for name, rec in _manifest().items():
        src = sources[rec["original"]]
        assert re.fullmatch(r"[0-9a-f]{64}", src["sha256"]) and src["download_url"].startswith("https://"), name


def test_default_image_pools_are_this_set():
    pools = testbed._RED_DRONE_IMAGES + testbed._BIRD_IMAGES + testbed._CIVILIAN_IMAGES
    assert sorted(os.path.basename(p)[:-5] for p in pools) == sorted(NAMES)
