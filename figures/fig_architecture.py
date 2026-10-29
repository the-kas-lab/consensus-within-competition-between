"""Fig. 1: how the teams decide.

Panels:

a, one LLM call, with real content: the state-based observation (a plain-language excerpt of the recorded
   bounded-auction call in Supplementary Note 1, sim step 110, defender 0), the multimodal observation (the
   same text plus a photograph of each contact, labelled as in the prompts), the LLM, and the answer of that
   same call (balanced profile; contact 0 high priority; preferred task: pursue contact 0).
b, the four teams as columns: the network of each (who runs an LLM; beliefs, orders and bids on the links;
   defender links dashed because the network conditions degrade them, intruder links solid because the
   intruders share sightings over an ideal channel), then what the LLM sees, what it answers and who sets the
   task.
c, the tasks (defenders: pursue an intruder, hold a station, scout a cell; intruders: attack a gate, evade a
   defender) and their execution every 0.1 s: MPC, the barrier filter and motion (one word per stage).

Facts checked: Supplementary Note 1 (example requests and responses; the interfaces m8b.7, commander.10,
intruder.4); Methods (Communication and network conditions; Defender teams; Intruders). The photograph is
intruder_1 of assets/contact_photos (credited in the legend).

    python3 -m figures.fig_architecture --data DATA [--out OUT] [--figures DIR]
"""
from __future__ import annotations

import itertools

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge
from PIL import Image

from figures import paths, style

OUT = paths.FIG / "fig_architecture.pdf"
PHOTO = paths.REPO / "assets" / "contact_photos" / "intruder_1.jpeg"
W, H = style.W2, 147.0
BLUE, RED = style.TEAM["blue"], style.TEAM["red"]
LH = 2.75                      # mm per text line at 7 pt
DASH = (0, (2.2, 1.2))


# ---------------------------------------------------------------- primitives (all in mm)
def photo(tw, th, dpi=480):
    im = Image.open(PHOTO).convert("RGB")
    w, h = im.size
    cw, ch = (w, round(w * th / tw)) if w / h < tw / th else (round(h * tw / th), h)
    box = ((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch)
    px = (round(tw / 25.4 * dpi), round(th / 25.4 * dpi))
    return np.asarray(im.crop(box).resize(px, Image.LANCZOS))


def rbox(ax, x, y, w, h, fill=style.WHITE, edge=style.TEXT, lw=style.LW_RULE, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", facecolor=fill,
                                edgecolor=edge, linewidth=lw, zorder=z))


def arrow(ax, p, q, colour=style.TEXT, lw=style.LW_RULE, z=2, shrink=0.0, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=5.0, color=colour, linewidth=lw,
                                 shrinkA=shrink, shrinkB=shrink, zorder=z, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def lines(ax, x, y, rows, bold_first=False):
    for k, s in enumerate(rows):
        ax.text(x, y - k * LH, s, ha="left", va="top", zorder=5,
                fontweight="bold" if (bold_first and k == 0) else "normal")


def heading(ax, x, y, letter, text, fig):
    style.panel_label(fig, x, y, letter)
    ax.text(x + 4.0, y, text, ha="left", va="top", fontweight="bold")


def clock(ax, y, text):
    ax.text(W - 2.0, y, text, ha="right", va="top")


# ---------------------------------------------------------------- a: one LLM call
STATE = ("You: (92.9, 82.0), on station 0",
         "Contact 0: (41.4, 39.1), seen 0.1 s ago",
         "Pursue contact 0: bid 1.26",
         "Hold station 0: bid 0.34",
         "Scout cell 13: bid 0.04")
ANSWER = ("Profile: balanced",
          "Contact 0: high priority",
          "Preferred: pursue contact 0")


def stage(ax, x, y, w, h, label, sub="", fill=style.WHITE, edge=style.TEXT, bold=False):
    """A stage of the loop: a rounded box with a label and an optional second line."""
    rbox(ax, x, y, w, h, fill=fill, edge=edge)
    if sub:
        ax.text(x + w / 2, y + h / 2 + 1.4, label, ha="center", va="center", zorder=5,
                fontweight="bold" if bold else "normal")
        ax.text(x + w / 2, y + h / 2 - 1.6, sub, ha="center", va="center", zorder=5)
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", zorder=5,
                fontweight="bold" if bold else "normal")


def band(ax, x, y, w, h, label, bottom=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.8", facecolor=style.FILL,
                                edgecolor="none", zorder=0))
    if bottom:
        ax.text(x + 2.0, y + 1.6, label, ha="left", va="bottom", fontweight="bold", zorder=5)
    else:
        ax.text(x + 2.0, y + h - 1.6, label, ha="left", va="top", fontweight="bold", zorder=5)


def panel_a(ax, fig, top):
    """One defender's loop. Top band: planning, once per second. Bottom band: sensing and control, every
    0.1 s, drawn right to left so that the loop closes: task -> MPC -> barrier filter -> motion -> sensing ->
    beliefs -> back up into the observation."""
    heading(ax, 1.0, top, "a", "One defender's decision loop", fig)
    # planning band
    pb_top, pb_h = top - 5.5, 47.0
    band(ax, 3.0, pb_top - pb_h, W - 5.0, pb_h, "Planning, once per second")
    x0, cw = 6.0, 60.0
    ch1 = 4.0 + LH * (len(STATE) + 1)
    y0 = pb_top - 6.5
    ax.add_patch(Rectangle((x0, y0 - ch1), cw, ch1, facecolor=style.WHITE, edgecolor=style.TEXT,
                           linewidth=style.LW_RULE, zorder=3))
    lines(ax, x0 + 2.0, y0 - 1.6, ("State-based observation",) + STATE, bold_first=True)
    ch2 = 15.0
    y1 = y0 - ch1 - 2.0
    ax.add_patch(Rectangle((x0, y1 - ch2), cw, ch2, facecolor=style.WHITE, edgecolor=style.TEXT,
                           linewidth=style.LW_RULE, zorder=3))
    lines(ax, x0 + 2.0, y1 - 1.6, ("Multimodal observation", "Same text, plus"), bold_first=True)
    pw, ph = 14.4, 10.8
    px0 = x0 + cw - 2.0 - pw
    ax.imshow(photo(pw, ph), extent=(px0, px0 + pw, y1 - ch2 + 2.0, y1 - ch2 + 2.0 + ph), zorder=4,
              interpolation="none")
    ax.text(px0 - 1.5, y1 - ch2 + 2.0 + ph / 2, "Image of\ncontact 0", ha="right", va="center", zorder=5,
            linespacing=1.3)
    mid = (y0 + y1 - ch2) / 2
    lx, lw_ = 74.0, 16.0
    stage(ax, lx, mid - 5.0, lw_, 10.0, "LLM", fill=BLUE["fill"], edge=BLUE["color"], bold=True)
    arrow(ax, (x0 + cw, y0 - ch1 / 2), (lx, mid + 1.5))
    arrow(ax, (x0 + cw, y1 - ch2 / 2), (lx, mid - 1.5))
    ax0, aw, ah = 98.0, 42.0, 4.0 + LH * (len(ANSWER) + 1)
    ax.add_patch(Rectangle((ax0, mid - ah / 2), aw, ah, facecolor=style.WHITE, edgecolor=style.TEXT,
                           linewidth=style.LW_RULE, zorder=3))
    lines(ax, ax0 + 2.0, mid + ah / 2 - 1.6, ("Answer",) + ANSWER, bold_first=True)
    arrow(ax, (lx + lw_, mid), (ax0, mid))
    tx, tw = 148.0, 27.0
    stage(ax, tx, mid - 5.5, tw, 11.0, "Task allocation", sub="by team (b)")
    arrow(ax, (ax0 + aw, mid), (tx, mid))
    # control band, right to left
    cb_top = pb_top - pb_h - 3.0
    cb_h = 29.0
    band(ax, 3.0, cb_top - cb_h, W - 5.0, cb_h, "Sensing and control, every 0.1 s", bottom=True)
    cy = cb_top - cb_h / 2 + 2.0
    bh = 10.0
    # task box with the three defender tasks
    kx, kw = 136.0, 39.0
    rbox(ax, kx, cy - 9.0, kw, 18.0, z=1)
    ax.text(kx + kw / 2, cy + 7.6, "Task", ha="center", va="top", zorder=5)
    for k, (kind, lab) in enumerate((("pursue", "Pursue"), ("hold", "Hold"), ("scout", "Scout"))):
        gx = kx + 7.0 + k * 12.5
        glyph(ax, kind, gx, cy - 0.3, s=0.7)
        ax.text(gx, cy - 4.4, lab, ha="center", va="top", zorder=5)
    arrow(ax, (tx + tw / 2, mid - 5.5), (tx + tw / 2, cy + 9.0))
    stages = (("MPC", "", 18.0), ("Barrier filter", "", 22.0), ("Motion", "", 18.0),
              ("Sensing", "", 20.0), ("Beliefs", "", 20.0))   # one word per stage
    x_right = kx
    pos = {}
    for label, sub, w in stages:
        x = x_right - 6.0 - w
        stage(ax, x, cy - bh / 2, w, bh, label, sub=sub)
        arrow(ax, (x_right, cy), (x + w, cy))
        pos[label] = (x, w)
        x_right = x
    bx, bw = pos["Beliefs"]
    arrow(ax, (bx + bw / 2, cy + bh / 2), (bx + bw / 2, y1 - ch2))
    return cb_top - cb_h


# ---------------------------------------------------------------- b: the four teams
NODES = {"N": (0.0, 1.0), "E": (1.0, 0.0), "S": (0.0, -1.0), "W": (-1.0, 0.0)}
TEAMS = (
    ("commander", "Commander",
     ("Every defender and", "every reported intruder"),
     ("A task for", "every defender"),
     ("Orders, sent", "over the network")),
    ("cbaa", "Consensus",
     ("Own beliefs and", "candidate tasks with bids"),
     ("Preset profile, priorities", "and preferred tasks"),
     ("Auction agreement", "among defenders")),
    ("independent", "Independent",
     ("Own beliefs and", "candidate tasks with bids"),
     ("Preset profile, priorities", "and preferred tasks"),
     ("Own highest bid",)),
    ("intruders", "Intruders",
     ("Own state, gates and", "shared defender sightings"),
     ("Attack a gate", "or evade a defender"),
     ("Own LLM",)),
)
ROWS = ("LLM sees", "LLM answers", "Task set by")


def network(ax, cx, cy, arm):
    r, rn = 8.2, 3.5
    pos = {k: (cx + r * x, cy + r * y) for k, (x, y) in NODES.items()}
    intr = arm == "intruders"
    st = {"color": RED["color"], "fill": RED["fill"]} if intr else style.ARM[arm]
    ls = "-" if intr else DASH
    for a, b in itertools.combinations(pos, 2):
        ax.plot(*zip(pos[a], pos[b]), color=style.ZERO, lw=style.LW_RULE, zorder=1, linestyle=ls)
    if arm == "commander":
        for k in ("E", "S", "W"):
            arrow(ax, pos["N"], pos[k], colour=st["color"], lw=style.LW_EMPH, z=2, shrink=rn * 2.835 + 0.5, ls=DASH)
    elif arm == "cbaa":
        for a, b in itertools.combinations(pos, 2):
            ax.plot(*zip(pos[a], pos[b]), color=st["color"], lw=style.LW_EMPH, zorder=2, linestyle=DASH)
    for k, p in pos.items():
        has_llm = arm != "commander" or k == "N"
        ax.add_patch(Circle(p, rn, facecolor=st["fill"] if has_llm else style.WHITE, edgecolor=st["color"],
                            linewidth=style.LW_DATA, zorder=3))
        if has_llm:
            ax.text(*p, "LLM", ha="center", va="center", zorder=4)


def panel_b(ax, fig, top):
    heading(ax, 1.0, top, "b", "Four teams", fig)
    x_lab, x0, cwid = 6.0, 30.0, 37.5
    ty = top - 8.0
    y_net = ty - 13.0
    for k, (arm, title, sees, answers, setby) in enumerate(TEAMS):
        cx = x0 + cwid * k + cwid / 2
        mk, col = ("^", RED["color"]) if arm == "intruders" else (style.ARM[arm]["marker"], style.ARM[arm]["color"])
        ax.plot([cx - 9.5], [ty], marker=mk, color=col, markersize=4.2, markeredgewidth=0, ls="none")
        ax.text(cx - 7.6, ty, title, ha="left", va="center", fontweight="bold")
        network(ax, cx, y_net, arm)
        for r, cell in enumerate((sees, answers, setby)):
            lines(ax, cx - cwid / 2 + 2.0, y_net - 12.5 - r * 8.6, cell)
    for r, lab in enumerate(ROWS):
        ax.text(x_lab, y_net - 12.5 - r * 8.6, lab, ha="left", va="top", fontweight="bold")
    bottom = y_net - 12.5 - 2 * 8.6 - 6.0
    xs = x0 + 3 * cwid
    ax.plot([xs, xs], [ty + 2.0, bottom], color=style.GRID, lw=style.LW_RULE, zorder=0)
    key(ax, top)
    return bottom


def key(ax, y):
    from matplotlib.lines import Line2D
    items = [
        (Line2D([], [], marker="o", ls="none", markersize=7, markerfacecolor=style.WHITE,
                markeredgecolor=style.TEXT, markeredgewidth=style.LW_DATA), "No LLM"),
        (Line2D([], [], color=style.ZERO, lw=style.LW_RULE), "Shared beliefs"),
        (Line2D([], [], color=style.ARM["commander"]["color"], lw=style.LW_EMPH), "Orders"),
        (Line2D([], [], color=style.ARM["cbaa"]["color"], lw=style.LW_EMPH), "Bids"),
        (Line2D([], [], color=style.TEXT, lw=style.LW_RULE, linestyle=DASH), "Degraded by network conditions"),
    ]
    ax.legend([h for h, _ in items], [s for _, s in items], loc="upper left", bbox_to_anchor=(26.0, y - 0.2),
              bbox_transform=ax.transData, ncol=5, frameon=False, handlelength=1.6, columnspacing=1.2,
              borderaxespad=0, borderpad=0)


# ---------------------------------------------------------------- c: tasks and execution
def dot(ax, p, team, r=1.5):
    t = BLUE if team == "blue" else RED
    if team == "blue":
        ax.add_patch(Circle(p, r, facecolor=t["color"], edgecolor="none", zorder=4))
    else:
        x, y = p
        ax.add_patch(Polygon([(x - r, y - r * 0.85), (x + r, y - r * 0.85), (x, y + r)], closed=True,
                             facecolor=t["color"], edgecolor="none", zorder=4))


def glyph(ax, kind, cx, cy, s=1.0):
    """A sketch of one task, 16 x 9 mm at s = 1."""
    P = lambda dx, dy: (cx + s * dx, cy + s * dy)
    if kind == "pursue":
        dot(ax, P(-5.5, -1.0), "blue", r=1.5 * s)
        dot(ax, P(5.5, 1.0), "red", r=1.5 * s)
        arrow(ax, P(-3.8, -0.6), P(3.6, 0.7), colour=BLUE["color"], lw=style.LW_DATA)
    elif kind == "hold":
        ax.add_patch(Wedge(P(-3.5, -1.5), 8.5 * s, 20, 70, facecolor=BLUE["fill"], edgecolor=BLUE["color"],
                           linewidth=style.LW_RULE, zorder=2))
        dot(ax, P(-3.5, -1.5), "blue", r=1.5 * s)
    elif kind == "scout":
        ax.add_patch(Rectangle(P(1.0, -2.5), 6.0 * s, 6.0 * s, facecolor=style.WHITE, edgecolor=BLUE["color"],
                               linewidth=style.LW_RULE, linestyle=(0, (1.6, 1.0)), zorder=2))
        dot(ax, P(-6.0, -1.0), "blue", r=1.5 * s)
        arrow(ax, P(-4.3, -0.6), P(1.0, 0.4), colour=BLUE["color"], lw=style.LW_DATA)
    elif kind == "attack":
        ax.add_patch(Polygon([P(4.0, 3.0), P(8.5, 3.5), P(8.5, -1.5)], closed=True,
                             facecolor=style.FILL, edgecolor=style.TEXT, linewidth=style.LW_RULE, zorder=2))
        ax.plot([cx + 3.0], [cy + 0.5], marker="D", markersize=3.6, markerfacecolor=style.WHITE,
                markeredgecolor=style.TEXT, markeredgewidth=style.LW_RULE, zorder=3)
        dot(ax, P(-6.0, -2.0), "red", r=1.5 * s)
        arrow(ax, P(-4.4, -1.4), P(1.8, 0.1), colour=RED["color"], lw=style.LW_DATA)
    elif kind == "evade":
        dot(ax, P(3.5, 1.5), "blue", r=1.5 * s)
        dot(ax, P(-1.0, -0.5), "red", r=1.5 * s)
        arrow(ax, P(-1.8, -1.0), P(-7.5, 0.5), colour=RED["color"], lw=style.LW_DATA, rad=0.35)


def main():
    style.apply()
    fig = style.figure(W, H)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    ya = panel_a(ax, fig, H - 1.0)
    yb = panel_b(ax, fig, ya - 4.0)
    print(f"bottom of panel b at {yb:.1f} mm")
    style.save(fig, OUT)


if __name__ == "__main__":
    main()
