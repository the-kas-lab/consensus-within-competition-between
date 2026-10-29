"""Plotting sheet: the figure registry of the paper.

Every figure script imports from here and saves through ``save()``; no script sets its own colour,
font, size or stroke.

* Text is 7 pt everywhere (5 pt minimum, 7 pt maximum; one size in every figure). Panel letters are
  bold and the same size.
* Widths are 88 / 120 / 180 mm and the height is at most 170 mm.
* All text is black. Results are black too.
* Colours fall into three classes (text, data, neutral); the palette keeps every mark at 3.0:1 or more against
  white and every pair that shares a figure at CIEDE2000 10 or more under normal, protan, deutan and tritan vision.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

REGISTRY_VERSION = 3
MM = 1 / 25.4                               # inches per millimetre
W1, W15, W2 = 88.0, 120.0, 180.0            # print widths (mm)
WIDTHS = (W1, W15, W2)
H_MAX = 170.0                               # mm, leaves room for the legend below

# Type (pt).
FONT = "Arial"
FONTS_ALLOWED = ("ArialMT", "Arial-BoldMT", "Helvetica", "Helvetica-Bold")
FS = 7.0                                    # every glyph
PANEL_FS = FS                               # panel letters "in the same type size"
FS_MIN, FS_MAX = 5.0, 7.0                   # smallest and largest text size, in pt
MINUS = "−"

# Strokes (pt). Floor 1.0.
LW_MIN = 1.0
LW_RULE = 1.0
LW_DATA = 1.25
LW_EMPH = 2.0

# Colour classes.
TEXT = "#000000"                            # all text, axes and results
WHITE = "#FFFFFF"
INK = TEXT
ZERO = "#8F8B85"                            # reference lines (zero, sector edges): warm grey
# Palette: Eggshell / Burnt Peach / Twilight Indigo / Muted Teal / Apricot Cream (coolors.co
# f4f1de-e07a5f-3d405b-81b29a-f2cc8f). Hues kept; lightness lowered until every mark has >= 3.4:1 on white and
# every pair that shares a figure has CIEDE2000 >= 10.7 under normal, protan, deutan and tritan vision.
TEAM = {
    "blue": {"color": "#3D405B", "fill": "#ECECF1", "label": "defenders"},      # Twilight Indigo
    "red": {"color": "#B34E36", "fill": "#FDE8E2", "label": "intruders"},       # Burnt Peach, deeper
}
ARM = {
    "cbaa": {"color": "#3C7E61", "fill": "#E4EFE9", "marker": "o", "label": "Consensus"},  # Muted Teal, darker
    "independent": {"color": "#3D405B", "fill": "#ECECF1", "marker": "s", "label": "Independent"},  # plain defender indigo
    "commander": {"color": "#B3830D", "fill": "#F7EBDA", "marker": "^", "label": "Commander"},  # Apricot Cream -> ochre
    "bidweight": {"color": "#1D503A", "fill": "#E6EEEA", "marker": "D", "label": "High authority"},  # deep Muted Teal
}
GRID = "#E8E4D0"                            # separators only, never a background grid
FILL = "#F4F1DE"                            # Eggshell: region, bands, "no event" tiles
PAPER = WHITE

TEXT_COLOURS = {TEXT, WHITE}
DATA_COLOURS = {TEXT, ZERO, *(v["color"] for v in TEAM.values()), *(v["color"] for v in ARM.values())}
NEUTRAL_COLOURS = {WHITE, GRID, FILL, *(v["fill"] for v in TEAM.values()), *(v["fill"] for v in ARM.values())}
DATA_CONTRAST_MIN = 3.0                     # WCAG 2.1 SC 1.4.11, against white
TEXT_CONTRAST_MIN = 4.5                     # WCAG 2.1 SC 1.4.3
NEUTRAL_CONTRAST_MAX = 1.35                 # a neutral must stay a background
BITMAP_DPI_MIN = 450                        # minimum resolution of any bitmap in a figure
LEGEND_WORDS_MAX = 150                      # cap on legend bodies, in words
CVD_DE_MIN = 10.0                           # CIEDE2000, worst of normal/protan/deutan/tritan

# Words that may appear in capitals (>= 3 letters), and numbers that are
# identifiers rather than quantities (exempt from the thousands-comma rule).
ACRONYMS = {"CBAA", "LLM", "LLMs", "MPC", "REF", "ATTACK", "EVADE", "RGB"}
IDENTIFIER_RE = r"^40[0-3]\d$"              # seeds 4001-4030


def apply():
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": [FONT],
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT,
        "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold",
        "font.size": FS,
        "axes.labelsize": FS,
        "axes.titlesize": FS,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS,
        "legend.title_fontsize": FS,
        "axes.edgecolor": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "text.color": TEXT,
        "axes.linewidth": LW_RULE,
        "xtick.major.width": LW_RULE,
        "ytick.major.width": LW_RULE,
        "xtick.minor.width": LW_RULE,
        "ytick.minor.width": LW_RULE,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "lines.linewidth": LW_DATA,
        "lines.markeredgewidth": LW_RULE,
        "patch.linewidth": LW_RULE,
        "hatch.linewidth": LW_RULE,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": LW_RULE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "legend.frameon": False,
        "axes.unicode_minus": True,
        "pdf.compression": 6,
    })


def figure(w_mm, h_mm):
    assert w_mm in WIDTHS, w_mm
    assert h_mm <= H_MAX, h_mm
    return plt.figure(figsize=(w_mm * MM, h_mm * MM))


def axes_mm(fig, x, y, w, h, **kw):
    """Axes placed in millimetres from the figure's lower-left corner."""
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def text_mm(fig, x, y, s, **kw):
    W, H = fig.get_size_inches() / MM
    return fig.text(x / W, y / H, s, **kw)


def panel_label(fig, x, y, letter):
    """Bold lower-case panel letter, top-left anchored at (x, y) mm."""
    assert len(letter) == 1 and letter.islower(), letter
    return text_mm(fig, x, y, letter, fontweight="bold", fontsize=PANEL_FS, ha="left", va="top")


def key(fig, handles, labels, x, y, ncol, **kw):
    """The figure's one key: frameless, anchored at (x, y) mm (lower-left)."""
    W, H = fig.get_size_inches() / MM
    opts = dict(frameon=False, handlelength=1.6, columnspacing=1.6, handletextpad=0.5,
                borderaxespad=0.0, borderpad=0.0)
    opts.update(kw)
    return fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(x / W, y / H),
                      ncol=ncol, **opts)


def _is_number(s):
    s = s.replace(MINUS, "-").replace("+", "").replace(",", "").rstrip("%")
    try:
        float(s)
        return True
    except ValueError:
        return False


def _axes_meta(fig):
    """What the tests need to know about every axes that draws a scale."""
    out = []
    for ax in fig.axes:
        if not ax.axison:
            continue
        rec = {}
        for name, axis, spine in (("x", ax.xaxis, "bottom"), ("y", ax.yaxis, "left")):
            labels = [t.get_text() for t in axis.get_ticklabels() if t.get_visible() and t.get_text()]
            ticks = axis.get_major_ticks()
            rec[name] = {
                "label": axis.get_label_text(),
                "n_tick_labels": len(labels),
                "numeric": bool(labels) and all(_is_number(s) for s in labels),
                "tick_length": max((t.tick1line.get_markersize() for t in ticks if t.tick1line.get_visible()),
                                   default=0.0),
                "spine_visible": bool(ax.spines[spine].get_visible()),
                "grid": any(g.get_visible() for g in axis.get_gridlines()),
            }
        out.append(rec)
    return out


def save(fig, path, supplementary=False):
    """Vector PDF at the declared canvas (no tight crop), a 300 dpi PNG preview
    and a JSON sidecar that the tests read. The only way figures are written."""
    W, H = fig.get_size_inches() / MM
    assert round(W, 3) in WIDTHS and H <= H_MAX + 1e-6, (W, H)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"registry_version": REGISTRY_VERSION, "width_mm": round(W, 3), "height_mm": round(H, 3),
            "supplementary": supplementary, "axes": _axes_meta(fig)}
    fig.savefig(path, bbox_inches=None, pad_inches=0)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches=None, pad_inches=0)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    plt.close(fig)


def register_pdf(path, supplementary=False):
    """The sidecar for a figure PDF drawn outside matplotlib: the replay viewer's scene.js, printed by headless
    Chrome (scripts/replay_media.py). Checks the page against the declared widths and the height cap, then writes
    the sidecar ``save`` writes."""
    import pymupdf
    doc = pymupdf.open(path)
    assert doc.page_count == 1, f"{path}: {doc.page_count} pages"
    r = doc[0].rect
    W, H = r.width / 72 * 25.4, r.height / 72 * 25.4
    width = min(WIDTHS, key=lambda w: abs(w - W))
    assert abs(width - W) < 0.15 and H <= H_MAX + 0.15, (W, H)
    meta = {"registry_version": REGISTRY_VERSION, "width_mm": width, "height_mm": round(H, 3),
            "supplementary": supplementary, "axes": [], "drawn_by": "viewer/js/scene.js"}
    Path(path).with_suffix(".json").write_text(json.dumps(meta, indent=1))


def rnd(x, d=2):
    """Half away from zero on the stored decimal value; the same rule as the
    numbers macros, so figure and text agree."""
    from decimal import ROUND_HALF_UP, Decimal
    return float(Decimal(str(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def signed(x, d=2):
    v = rnd(x, d)
    if v == 0:
        return f"{0:.{d}f}"
    return ("+" if v > 0 else MINUS) + f"{abs(v):.{d}f}"
