"""Builds the decoy-run contact photos (contact photographs) from their originals.

Every output is a 4:3 crop, resized to 640x480 with Lanczos, saved as baseline JPEG at quality 90 with no
metadata. The crops remove the ArmyInform watermark (intruder_1) and the race gate and banner (intruder_2).
All readable text on the objects (brand names, lens markings, the grenade stencil, a sticker glyph) is
removed before cropping: inside a polygon on the object's surface, the text pixels are selected by a
brightness/colour rule, dilated, and replaced by the harmonic (Laplace) interpolation of the surrounding
pixels -- a deterministic smooth fill, no blur and no learned inpainting. Nothing else is edited.

The originals are not in the repository. ``--fetch`` downloads them from the URLs in sources.json into
originals/ (git-ignored); every original is checked against its SHA-256 in sources.json before use.

    python3 -m assets.contact_photos.prepare_stimuli --fetch   # download originals, rebuild the nine files
    python3 -m assets.contact_photos.prepare_stimuli --check   # rebuild into a temp dir, compare with manifest.json

Output bytes depend on the Pillow / libjpeg / NumPy / SciPy builds; the versions used are in manifest.json.
Sources, authors and licences: CREDITS.md.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np
import PIL
import scipy
from PIL import Image, ImageDraw, features
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "originals")
SOURCES = os.path.join(HERE, "sources.json")
MANIFEST = os.path.join(HERE, "manifest.json")
SIZE = (640, 480)
QUALITY = 90

# output name -> (original key in sources.json, crop box (left, upper, right, lower) in original pixels)
CROPS = {
    "intruder_1": ("intruder_A1", (375, 40, 1375, 790)),
    "intruder_2": ("intruder_A2", (2220, 0, 4753, 1900)),
    "intruder_3": ("intruder_S29", (2270, 1048, 4970, 3073)),
    "bird_1": ("bird_kept1", (0, 560, 2952, 2774)),
    "bird_2": ("bird_BA", (769, 200, 3436, 2200)),
    "bird_3": ("bird_BB", (0, 0, 2900, 2175)),
    "civilian_1": ("civilian_kept2", (3, 0, 1978, 1481)),
    "civilian_2": ("civilian_CA", (200, 0, 5064, 3648)),
    "civilian_3": ("civilian_CB", (88, 0, 5213, 3844)),
}


# original key -> text regions: (label, polygon in original pixels, selection rule, threshold, dilation px).
# Rules: "all" = the whole polygon; "bright"/"dark" = luminance above/below the polygon's median by the
# threshold; "yellow" = (R+G)/2 - B above the threshold.
TEXT = {
    "civilian_kept2": [("PHANTOM emboss", [(912, 638), (1048, 638), (1048, 682), (912, 682)], "all", 0, 0)],
    "civilian_CA": [
        ("PHANTOM", [(2432, 1922), (2655, 1928), (2655, 2003), (2432, 1995)], "dark", 10, 4),
        ("dji on lens ring", [(2525, 2395), (2600, 2395), (2600, 2435), (2525, 2435)], "bright", 10, 3),
        ("lens markings", [(2478, 2560), (2520, 2572), (2560, 2576), (2600, 2571), (2625, 2565), (2630, 2598),
                           (2610, 2610), (2560, 2615), (2510, 2607), (2474, 2587)], "bright", 8, 3)],
    "civilian_CB": [
        ("HASSELBLAD", [(2568, 1462), (2785, 1408), (2797, 1453), (2570, 1488)], "bright", 8, 6),
        ("MAVIC 4 PRO", [(3585, 1840), (3935, 1730), (3960, 1790), (3605, 1905)], "bright", 8, 7)],
    "intruder_S29": [("grenade stencil", [(3485, 2175), (3735, 2175), (3735, 2300), (3485, 2300)], "yellow", 10, 5)],
    "intruder_A2": [("sticker", [(3410, 1164), (3465, 1164), (3465, 1208), (3410, 1208)], "all", 0, 0)],
}


def text_mask(arr, poly, rule, thr, dil):
    h, w = arr.shape[:2]
    img = Image.new("1", (w, h), 0)
    ImageDraw.Draw(img).polygon(poly, fill=1)
    inside = np.array(img, dtype=bool)
    lum = arr @ np.array([0.299, 0.587, 0.114])
    if rule == "all":
        sel = inside.copy()
    elif rule in ("bright", "dark"):
        med = np.median(lum[inside])
        sel = inside & (lum > med + thr if rule == "bright" else lum < med - thr)
    elif rule == "yellow":
        sel = inside & ((arr[..., 0] + arr[..., 1]) / 2 - arr[..., 2] > thr)
    else:
        raise ValueError(rule)
    if dil:
        sel = ndimage.binary_dilation(sel, iterations=dil)
    return sel & inside if rule != "all" else sel


def harmonic_fill(arr, mask):
    """Replaces the masked pixels by the solution of Laplace's equation with the unmasked neighbours as
    boundary values (4-neighbour stencil), per channel."""
    ys, xs = np.nonzero(mask)
    index = -np.ones(mask.shape, dtype=np.int64)
    index[ys, xs] = np.arange(len(ys))
    n = len(ys)
    rows, cols, vals = [], [], []
    b = np.zeros((n, arr.shape[2]))
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < mask.shape[0]) & (nx >= 0) & (nx < mask.shape[1])
        rows.append(np.arange(n)[ok]); cols.append(np.arange(n)[ok]); vals.append(np.ones(ok.sum()))
        inner = ok.copy(); inner[ok] = mask[ny[ok], nx[ok]]
        rows.append(np.arange(n)[inner]); cols.append(index[ny[inner], nx[inner]]); vals.append(-np.ones(inner.sum()))
        edge = ok & ~inner
        b[np.arange(n)[edge]] += arr[ny[edge], nx[edge]]
    a = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    out = arr.copy()
    for c in range(arr.shape[2]):
        out[ys, xs, c] = spsolve(a, b[:, c])
    return out


def remove_text(im, regions):
    arr = np.asarray(im, dtype=np.float64)
    report = []
    for label, poly, rule, thr, dil in regions:
        mask = text_mask(arr, poly, rule, thr, dil)
        arr = harmonic_fill(arr, mask)
        report.append({"label": label, "polygon": [list(p) for p in poly], "rule": rule, "threshold": thr,
                       "dilation": dil, "pixels": int(mask.sum())})
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8)), report


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fetch(sources):
    os.makedirs(ORIG, exist_ok=True)
    for key, s in sources.items():
        path = os.path.join(ORIG, s["file"])
        if os.path.isfile(path) and sha256(path) == s["sha256"]:
            continue
        req = urllib.request.Request(s["download_url"], headers={"User-Agent": "pursuit-evasion-dpc-stimulus-fetch/1.0"})
        for attempt in range(6):  # Wikimedia answers bursts with HTTP 429
            time.sleep(3)
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 5:
                    raise
                time.sleep(20 * (attempt + 1))
        with open(path, "wb") as f:
            f.write(data)
        if sha256(path) != s["sha256"]:
            raise SystemExit(f"{key}: downloaded file does not match its SHA-256 in sources.json")
        print("fetched", s["file"])


def build(sources, out_dir):
    records = {}
    for name, (key, box) in CROPS.items():
        src = os.path.join(ORIG, sources[key]["file"])
        if not os.path.isfile(src) or sha256(src) != sources[key]["sha256"]:
            raise SystemExit(f"{src}: missing or not matching sources.json (run with --fetch)")
        im = Image.open(src).convert("RGB")
        im, edits = remove_text(im, TEXT.get(key, []))
        w, h = box[2] - box[0], box[3] - box[1]
        if abs(w / h - 4 / 3) > 0.002 or box[2] > im.width or box[3] > im.height:
            raise SystemExit(f"{name}: crop {box} is not 4:3 inside {im.size}")
        out = im.crop(box).resize(SIZE, Image.LANCZOS)
        path = os.path.join(out_dir, f"{name}.jpeg")
        out.save(path, "JPEG", quality=QUALITY, optimize=False, progressive=False)
        records[name] = {"original": key, "original_size": list(im.size), "crop": list(box), "text_removed": edits,
                         "sha256": sha256(path), "source_page": sources[key]["page"]}
    return {"size": list(SIZE), "jpeg_quality": QUALITY, "resample": "LANCZOS", "pillow": PIL.__version__,
            "libjpeg": features.version("jpg"), "numpy": np.__version__, "scipy": scipy.__version__, "files": records}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fetch", action="store_true", help="download missing originals first")
    ap.add_argument("--check", action="store_true", help="rebuild into a temp dir and compare with manifest.json")
    args = ap.parse_args(argv)
    sources = json.load(open(SOURCES))
    if args.fetch:
        fetch(sources)
    if args.check:
        expected = json.load(open(MANIFEST))["files"]
        with tempfile.TemporaryDirectory() as tmp:
            got = build(sources, tmp)["files"]
        bad = [n for n in expected if got[n]["sha256"] != expected[n]["sha256"]]
        print("rebuild matches manifest.json" if not bad else f"MISMATCH: {bad}")
        return 1 if bad else 0
    meta = build(sources, HERE)
    json.dump(meta, open(MANIFEST, "w"), indent=1)
    for name, r in meta["files"].items():
        print(f"{name:11s} {r['sha256'][:16]}  crop {r['crop']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
