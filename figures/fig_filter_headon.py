"""Supplementary figure: the safety filter's condition cannot be met in some head-on states.

Message: two defenders closing head-on just outside the safety distance can be in the safe set (h > 0) with no
admissible acceleration that keeps h_{k+1} >= 0, so the discrete-time condition fails for every gamma; the filter's
condition with gamma = 0.05 fails in a wider band.

Source: barrier_tests.json (analysis.barrier_tests): A_headon_map.coop (two defenders, both braking, best
attainable h_{k+1} on a grid of gaps and closing speeds) and B_counterexample. D = 3 units, gamma = 0.05.

    python3 -m figures.fig_filter_headon --data DATA [--out OUT] [--figures DIR]
"""
from __future__ import annotations

import numpy as np

from figures import paths, style
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

OUT = paths.FIG / "supp" / "fig_filter_headon.pdf"
D, GAMMA = 3.0, 0.05


def main():
    style.apply()
    d = paths.analysis("barrier_tests.json")
    m = d["A_headon_map"]["coop"]
    g, v = np.array(m["gaps"]), np.array(m["speeds"])
    best = np.array(m["best_h_next"])                  # [speed][gap]
    G, V = np.meshgrid(g, v)
    h = G ** 2 - D ** 2
    assert (h > 0).all()
    cls = np.where(best < 0, 2, np.where(best < (1 - GAMMA) * h, 1, 0))
    assert int((cls == 2).sum()) == 612 and h.size == 68850  # numbers.tex \numFiltGridFail of \numFiltGridStates
    cx = d["B_counterexample"]
    assert (cx["gap"], cx["closing"]) == (3.1, 4.0) and cx["best_h_next_exact"] < 0 < cx["h"]

    fig = style.figure(88, 66)
    ax = style.axes_mm(fig, 13, 22, 70, 41)
    cols = [style.FILL, style.TEAM["red"]["fill"], style.TEAM["red"]["color"]]
    keep = g <= 4.0 + 1e-9  # draw only the window shown, so no fill extends beyond the axes
    ax.contourf(G[:, keep], V[:, keep], cls[:, keep], levels=[-0.5, 0.5, 1.5, 2.5], colors=cols)
    ax.plot([cx["gap"]], [cx["closing"]], marker="o", ms=4, color=style.TEXT, ls="none", clip_on=False)
    ax.set_xlim(3.0, 4.0)
    ax.set_ylim(0, 4.0)
    ax.set_xticks([3.0, 3.25, 3.5, 3.75, 4.0])
    ax.set_xlabel("Gap between the two defenders (units)")
    ax.set_ylabel("Closing speed (units/s)")
    handles = [Patch(facecolor=cols[0], edgecolor="none"), Patch(facecolor=cols[1], edgecolor="none"),
               Patch(facecolor=cols[2], edgecolor="none"),
               Line2D([], [], marker="o", ms=4, color=style.TEXT, ls="none")]
    labels = ["Barrier condition can be met", "Cannot be met with γ = 0.05",
              "Cannot be met for any γ", "Counterexample"]
    style.key(fig, handles, labels, 6, 2.5, 2, columnspacing=1.0)
    style.save(fig, OUT, supplementary=True)


if __name__ == "__main__":
    main()
