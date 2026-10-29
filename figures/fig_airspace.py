"""Fig. 7: airspace challenges.

Message: with birds and civilian drones, the near-frontier model set them aside and lost no defender, the
low-cost model did not; shown bird-drone photographs instead of attack-drone photographs, every team set
intruders aside more often; a one-sentence forewarning restored the near-frontier model only.

a, one frame of a consensus-team episode (low authority, low-cost model) with one bird and two civilian drones among
   four intruders, drawn by the replay viewer (viewer/js/scene.js, printed by headless Chrome through
   scripts/replay_media.py). The episode is exported as a bundle by analysis.episode_bundles (key fig7a_cbaa: a
   re-simulation from the cache equal to its stored record and checked against the bundle). The viewer's PDF page
   is set into this figure as vector content (PyMuPDF show_pdf_page); its letter "a" is the viewer's, at the size
   and place of the other panel letters.
b, the photographs the models received in that episode, and the four bird-drone photographs of the disguise
   (supplied with --bird-drone-dir; they are not part of this repository).
c, share of the models' answers (or contact-epochs, for the commanders) that set a contact aside, by contact
   kind and team, without the disguise.
d, intruders set aside with attack-drone photographs, with bird-drone photographs, and with bird-drone
   photographs plus the forewarning sentence.
e, defenders lost to birds and civilian drones, summed over the five episodes without the disguise.
Five seeds: every value is shown, without CIs. Values are read from the numbers.tex macros the text prints
(airspace.json and disguise.json).

    python3 -m analysis.episode_bundles --data DATA fig7a_cbaa
    python3 -m figures.fig_airspace --data DATA --bird-drone-dir DIR [--out OUT] [--figures DIR]
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

from experiments import studies
from figures import nums, paths, style

OUT = paths.FIG / "fig_airspace.pdf"
EP = paths.AN / "bundles"
PHOTOS = paths.FIG / "photos"
PHOTO_DPI = 480                                      # target; >= style.BITMAP_DPI_MIN (450)
FRAME = 250                                          # panel a: step shown (25.0 s), 0.7 s before the loss
W, H = style.W2, 133.0
A_W, A_PANEL = 95.0, 57.0                            # mm: panel a's page (map and key) and its square map
A_LEFT = 1.5                                         # mm: scene.figure's left margin, where it draws the letter


def episode_images():
    """The photograph of each contact of the seed-4001 airspace episode (contact index -> file)."""
    return {str(k): Path(v) for k, v in studies.airspace_images(0).items()}


def photo(src, tw, th, top=False):
    """Centre-crop a photograph to the tile's aspect and resample it to PHOTO_DPI at the tile's print size with
    Lanczos (classic interpolation, no generative upscaling; it adds no detail). The scaled copy is written to
    PHOTOS for inspection. top=True crops a portrait photograph from the top instead of the centre (keeps a
    bird's head)."""
    im = Image.open(src).convert("RGB")
    w, h = im.size
    cw, ch = (w, round(w * th / tw)) if w / h < tw / th else (round(h * tw / th), h)
    box = ((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch)
    if top and w / h < tw / th:
        box = (0, 0, cw, ch)
    px = (round(tw / 25.4 * PHOTO_DPI), round(th / 25.4 * PHOTO_DPI))
    out = im.crop(box).resize(px, Image.LANCZOS)
    PHOTOS.mkdir(parents=True, exist_ok=True)
    out.save(PHOTOS / f"{Path(src).parent.name}_{Path(src).stem}_{px[0]}x{px[1]}.png")
    return np.asarray(out)
# (key, macro tags, arm style, near-frontier, label), in the house order
TEAMS = (("CMD", ("CallCmdAside", "Cmd"), "commander", False, "Commander, low-cost"),
         ("SOLCMD", ("CallSolCmdAside", "SolCmd"), "commander", True, "Commander, near-frontier"),
         ("B", ("CallBLow", "B"), "cbaa", False, "Low authority, low-cost"),
         ("W", ("CallWZero", "W"), "bidweight", False, "High authority, low-cost"),
         ("WSOL", ("CallWSolZero", "WSol"), "bidweight", True, "High authority, near-frontier"))


def m(name):
    return float(nums.macro(name))


def mk(arm, hollow, size=4.4):
    st = style.ARM[arm]
    if hollow:
        return dict(marker=st["marker"], markerfacecolor=style.WHITE, markeredgecolor=st["color"],
                    markeredgewidth=style.LW_RULE, markersize=size)
    return dict(marker=st["marker"], color=st["color"], markeredgewidth=0, markersize=size)


def viewer_panel(pdf):
    """Panel a, drawn by the replay viewer at the frame (step 250, 25.0 s): defender 3 is pursuing civilian
    drone 6, with which it is lost 0.7 s later."""
    from scripts import replay_media as rm
    d = {"images": episode_images()}
    log = {e["key"]: e for e in json.loads((EP / "export_log.json").read_text())["episodes"]}["fig7a_cbaa"]
    assert log["record_equal"] and log["bundle_check"] == "passed", log
    assert "|s4001|repimg-B|" in log["episode_id"] and log["n_steps"] > FRAME, log
    bundle = str(EP / "fig7a_cbaa.replay.js")
    spec = {"steps": [FRAME], "width_mm": A_W, "panel_mm": A_PANEL, "key_right": True}
    with rm.Chrome() as chrome:
        chrome.open(rm.render_url(bundle, "figure", spec))
        st = json.loads(chrome.evaluate("JSON.stringify((()=>{const ep=PEV.render.episode(), S=PEV.scene.frameState("
                                        f"ep,{FRAME}); return {{p:S.pursuits, ev:ep.contacts.decoy_events}}}})())"))
        assert [3, 6] in st["p"] and st["ev"][0]["step"] - FRAME == 7 and st["ev"][0]["blue"] == 3, st
        info = rm.figure(bundle, spec, str(pdf), png_dpi=0, chrome=chrome)
    assert A_LEFT + A_PANEL + 3.5 + info["keyWidth_mm"] <= A_W, info
    return d, info


def tiles(fig, x0, ytop, d):
    imgs = d["images"]
    row1 = [("Intruder", imgs["0"]), ("Bird", imgs["4"]), ("Civilian drone", imgs["5"]), ("Civilian drone", imgs["6"])]
    if paths.BIRD_DRONE_DIR is None:
        raise SystemExit("Fig. 7b needs the four bird-drone photographs: pass --bird-drone-dir (see the README)")
    row2 = [("Bird drone", paths.BIRD_DRONE_DIR / f"bird_drone_{i}.jpg") for i in (1, 2, 3, 4)]
    tw, th, gx = 18.6, 13.95, 1.8
    style.text_mm(fig, x0, ytop, "Photographs in the airspace challenges", ha="left", va="top", fontweight="bold")
    style.text_mm(fig, x0, ytop - 25.0, "Photographs in the disguise", ha="left", va="top", fontweight="bold")
    for r, (row, top) in enumerate(((row1, ytop - 4.0), (row2, ytop - 29.0))):
        for k, (name, f) in enumerate(row):
            crop = photo(paths.REPO / f if r == 0 and not Path(f).is_absolute() else f, tw, th, top=(r == 1))
            assert crop.shape[1] / (tw / 25.4) >= style.BITMAP_DPI_MIN
            cx = x0 + k * (tw + gx)
            a = style.axes_mm(fig, cx, top - th, tw, th)
            a.imshow(crop, interpolation="none", extent=(0, 1, 0, 1), aspect="auto")
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(style.LW_RULE)
                sp.set_edgecolor(style.INK)
            style.text_mm(fig, cx, top - th - 1.0, name, ha="left", va="top")


def main():
    style.apply()
    fig = style.figure(W, H)
    top = H - 1.0
    # a (the viewer's page, set in after saving), b
    tmp = tempfile.TemporaryDirectory()
    apdf = Path(tmp.name) / "a.pdf"
    d, ainfo = viewer_panel(apdf)
    assert ainfo["height_mm"] <= H - 53.0 - 2.0, ainfo                # clear of the letters of c, d and e
    style.panel_label(fig, 97.0, top, "b")
    tiles(fig, 101.0, top, d)

    # c, d, e: one key for the five teams, as a grid: team and authority in rows, model in columns,
    # so the commander and the consensus team at each authority level read apart, and the empty cell shows the
    # level that was not run with the near-frontier model
    y0, hh = 9.0, 40.0
    kx = fig.add_axes([0, 0, 1, 1], facecolor="none")
    kx.set_xlim(0, W)
    kx.set_ylim(0, H)
    kx.axis("off")
    cols = {False: 96.0, True: 118.0}
    for hol, lab in ((False, "Low-cost"), (True, "Near-frontier")):
        kx.text(cols[hol], 66.5, lab, ha="center", va="center")
    rows = {"commander": ("Commander", 63.2), "cbaa": ("Consensus, low authority", 59.9),
            "bidweight": ("Consensus, high authority", 56.6)}
    for lab, y in rows.values():
        kx.text(cols[False] - 9.0, y, lab, ha="right", va="center")
    for _, _, arm, hol, _ in TEAMS:
        kx.plot([cols[hol]], [rows[arm][1]], ls="none", clip_on=False, **mk(arm, hol))
    offs = np.linspace(-0.3, 0.3, len(TEAMS))

    # c: set aside by contact kind
    cx = style.axes_mm(fig, 13.0, y0, 62.0, hh)
    kinds = (("Red", "Intruders"), ("Bird", "Birds"), ("Civ", "Civilian\ndrones"))
    for i, (kt, _) in enumerate(kinds):
        for o, (key, (pre, _), arm, hol, _) in zip(offs, TEAMS):
            cx.plot([i + o], [m(f"num{pre}{kt}")], ls="none", zorder=3, **mk(arm, hol))
    cx.set_xticks(range(3), [n for _, n in kinds])
    cx.tick_params(axis="x", length=0)
    cx.set_xlim(-0.55, 2.55)
    cx.set_ylim(-4, 104)
    cx.set_yticks([0, 50, 100])
    cx.set_ylabel("Contacts set aside (%)")
    style.panel_label(fig, 1.0, y0 + hh + 4.0, "c")

    # d: intruders set aside, three photograph conditions
    dx = style.axes_mm(fig, 96.0, y0, 44.0, hh)
    for key, (pre, tag), arm, hol, _ in TEAMS:
        ys = [m(f"num{pre}Red"), m(f"numCamoAsideRed{tag}"), m(f"numWarnAsideRed{tag}")]
        dx.plot([0, 1, 2], ys, color=style.ARM[arm]["color"], lw=style.LW_DATA, zorder=2,
                ls=(0, (2.0, 1.2)) if hol else "solid")
        dx.plot([0, 1, 2], ys, ls="none", zorder=3, **mk(arm, hol))
    dx.set_xticks(range(3), ["Attack-drone\nphoto", "Bird-drone\nphoto", "Bird-drone photo\nand forewarning"])
    dx.tick_params(axis="x", length=0)
    dx.set_xlim(-0.3, 2.3)
    dx.set_ylim(-4, 104)
    dx.set_yticks([0, 50, 100])
    dx.set_ylabel("Intruders set aside (%)")
    style.panel_label(fig, 84.0, y0 + hh + 4.0, "d")

    # e: defenders lost to birds and civilian drones
    ex = style.axes_mm(fig, 157.0, y0, 21.0, hh)
    for j, (key, (_, tag), arm, hol, _) in enumerate(TEAMS):
        n = m(f"numDecoyLostBird{tag}") + m(f"numDecoyLostCiv{tag}")
        assert n == m(f"numDecoyLost{tag}")
        st = style.ARM[arm]
        ex.bar(j, n, width=0.7, color=style.WHITE if hol else st["color"], edgecolor=st["color"],
               linewidth=style.LW_RULE, zorder=2)
        ex.plot([j], [-0.55], ls="none", clip_on=False, **mk(arm, hol, 3.6))
    ex.set_xlim(-0.6, 4.6)
    ex.set_ylim(0, 5.4)
    ex.set_yticks([0, 1, 2, 3, 4, 5])
    ex.set_xticks([])
    ex.set_ylabel("Defenders lost (of 20)")
    style.panel_label(fig, 146.0, y0 + hh + 4.0, "e")
    style.save(fig, OUT)
    place_viewer_page(apdf, 1.0 - A_LEFT, 0.0)                     # letter "a" at x = 1 mm, like c
    tmp.cleanup()


def place_viewer_page(apdf, x_mm, y_mm):
    """Set the viewer's page into the saved figure at (x, y) mm from the top-left, as vector content, then redraw
    the 300 dpi preview from the combined PDF. The sidecar style.save wrote stays: the canvas is unchanged."""
    mm = 72 / 25.4
    src = pymupdf.open(apdf)
    doc = pymupdf.open(OUT)
    r = src[0].rect
    doc[0].show_pdf_page(pymupdf.Rect(x_mm * mm, y_mm * mm, x_mm * mm + r.width, y_mm * mm + r.height), src, 0)
    doc.save(str(OUT) + ".tmp", garbage=3, deflate=True)
    Path(str(OUT) + ".tmp").replace(OUT)
    pymupdf.open(OUT)[0].get_pixmap(dpi=300).save(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
