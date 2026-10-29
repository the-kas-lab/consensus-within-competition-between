"""Fig. 2: the arena to scale, from the simulator's own geometry.

Message: intruders start about 90 units from the defended region and every defender sensor faces the south-west
approach (a, b). a, the whole arena; b, the defended region enlarged.

Defenders are circles, the replay viewer's symbol, as in Fig. 7a, the SI's episode figures, the Supplementary
Videos and the README GIFs.

Source: an episode built by scripts.run_competing_teams.build (seed 4001, 4 intruders, multi_bearing, design
v2.3mb; setup_agents() only, no step is simulated). The build uses the rule backend only because no model is
consulted before the first step; nothing here is an outcome.

    python3 -m figures.fig_arena [--figures DIR]
"""
from __future__ import annotations

import math

import numpy as np

from figures import style
from figures import paths
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle, Wedge

OUT = paths.FIG / "fig_arena.pdf"

NAVY = style.TEAM["blue"]["color"]
NAVY_F = to_rgba(style.TEAM["blue"]["fill"], 0.6)   # alpha on the fill only, to show overlap
BRICK = style.TEAM["red"]["color"]
ZOOM = (64.5, 99.5, 64.0, 99.0)                     # x0, x1, y0, y1 of panel b
W, H = style.W2, 72.0
SIDE = 58.0                                          # mm, square map panels
FRAME = 250                                          # panel c: step shown (25.0 s), 0.7 s before the loss


def geometry():
    from scripts.run_competing_teams import build
    e = {"arm": "cbaa", "reds": 4, "family": "multi_bearing", "condition": "REF",
         "attacked": False, "seed": 4001, "rep": "fig", "design": "v2.3mb"}
    sim = build(e, backend="rule")[0]
    sim.setup_agents()
    reds = np.asarray(sim.red_state, dtype=float).reshape(sim.num_reds, -1)[:, :2]
    g = dict(
        center=np.asarray(sim.defense_center, float),
        shape=np.asarray(sim.defended_shape_points, float),
        poly=np.asarray(sim.defense_polygon, float),
        stations={int(k): np.asarray(v, float) for k, v in sim.solver.defend_stations.items()},
        headings={int(k): float(v) for k, v in sim.configured_station_headings.items()},
        gates=[np.asarray(gt.position, float) for gt in sim.gates],
        reds=reds, fov_half=math.degrees(sim.fov_half_angle_rad), fov_range=sim.fov_range,
        arena=float(sim.arena_size),
    )
    assert abs(g["fov_half"] - 30.0) < 1e-9 and g["fov_range"] == 35.0 and g["arena"] == 100.0
    assert [round(math.degrees(g["headings"][k])) for k in range(4)] == [255, 220, 185, 230]
    assert len(g["gates"]) == 4 and g["reds"].shape == (4, 2)
    return g


def wedges(ax, g):
    for k, p in g["stations"].items():
        h = math.degrees(g["headings"][k])
        ax.add_patch(Wedge(p, g["fov_range"], h - g["fov_half"], h + g["fov_half"],
                           facecolor=NAVY_F, edgecolor=NAVY, linewidth=style.LW_RULE, zorder=1))


def region(ax, shape):
    ax.add_patch(Polygon(shape, closed=True, facecolor=style.FILL, edgecolor=style.INK,
                         linewidth=style.LW_DATA, zorder=3))


def frame(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    ax.set_aspect("equal")


def main():
    """Fig. 2: a, the arena to scale; b, the defended region enlarged. One key, beside b."""
    style.apply()
    g = geometry()
    fig = style.figure(W, H)
    c = g["center"]
    y0 = 10.0

    # a: the whole arena.
    ax = style.axes_mm(fig, 13.0, y0, SIDE, SIDE)
    A = g["arena"]
    ax.set_xlim(0, A)
    ax.set_ylim(0, A)
    frame(ax)
    for b in (160.0, 300.0):
        r = math.radians(b)
        ax.plot([c[0], c[0] + 140 * math.cos(r)], [c[1], c[1] + 140 * math.sin(r)],
                color=style.ZERO, lw=style.LW_RULE, ls=(0, (1.0, 2.0)), zorder=0)
    wedges(ax, g)
    region(ax, g["shape"])
    for p in g["stations"].values():
        ax.plot(*p, marker="o", color=NAVY, markersize=3.6, ls="none", zorder=5)
    sc = A - c
    ax.add_patch(Rectangle(sc - 10, 20, 20, fill=False, edgecolor=BRICK, linewidth=style.LW_RULE,
                           linestyle=(0, (3, 2)), zorder=2))
    ax.plot(g["reds"][:, 0], g["reds"][:, 1], marker="^", color=BRICK, markersize=5.0, ls="none", zorder=5)
    x0, x1, yb0, yb1 = ZOOM
    ax.add_patch(Rectangle((x0, yb0), x1 - x0, yb1 - yb0, fill=False, edgecolor=style.INK,
                           linewidth=style.LW_RULE, zorder=6))
    ax.text(x0 - 1.5, yb1 - 1.0, "b", ha="right", va="top", fontweight="bold")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("x (units)", labelpad=2)
    ax.set_ylabel("y (units)", labelpad=2)

    # b: the defended region enlarged.
    bx = style.axes_mm(fig, 84.0, y0, SIDE, SIDE)
    bx.set_xlim(x0, x1)
    bx.set_ylim(yb0, yb1)
    frame(bx)
    wedges(bx, g)
    region(bx, g["shape"])
    for p in g["stations"].values():
        bx.plot(*p, marker="o", color=NAVY, markersize=6.0, ls="none", zorder=5)
    for q in g["gates"]:
        bx.plot(*q, marker="D", markerfacecolor=style.PAPER, markeredgecolor=style.INK,
                markeredgewidth=style.LW_RULE, markersize=5.0, ls="none", zorder=6)
    bx.set_xticks([65, 75, 85, 95])
    bx.set_yticks([65, 75, 85, 95])
    bx.set_xlabel("x (units)", labelpad=2)
    bx.set_ylabel("y (units)", labelpad=2)
    style.panel_label(fig, 1.0, H - 1.0, "a")
    style.panel_label(fig, 73.0, H - 1.0, "b")

    handles = [
        Line2D([], [], marker="o", color=NAVY, ls="none", markersize=4.5),
        Patch(facecolor=NAVY_F, edgecolor=NAVY, linewidth=style.LW_RULE),
        Patch(facecolor=style.FILL, edgecolor=style.INK, linewidth=style.LW_DATA),
        Line2D([], [], marker="D", markerfacecolor=style.PAPER, markeredgecolor=style.INK,
               markeredgewidth=style.LW_RULE, ls="none", markersize=4.5),
        Line2D([], [], marker="^", color=BRICK, ls="none", markersize=5.0),
        Patch(fill=False, edgecolor=BRICK, linewidth=style.LW_RULE, linestyle=(0, (3, 2))),
        Line2D([], [], color=style.ZERO, lw=style.LW_RULE, ls=(0, (1.0, 2.0))),
    ]
    labels = ["Defender station", "Sensor field of view", "Defended region", "Gate", "Intruder start",
              "Intruder start area", "Approach sector"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(146.0 / W, (y0 + SIDE) / H), frameon=False,
               handlelength=1.6, borderaxespad=0, labelspacing=0.9)
    style.save(fig, OUT)
    print("wrote", OUT, f"{W} x {H} mm")


if __name__ == "__main__":
    main()
