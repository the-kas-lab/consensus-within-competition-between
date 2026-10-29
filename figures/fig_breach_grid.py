"""Supplementary Fig. 1, breach grid: every episode's breach outcome, by team,
intruder count, network condition and seed (30 seeds), with per-row counts.

Message: breaches concentrated in four-intruder episodes in every team, and the
auction team had the fewest four-intruder breaches.

Full width, two stacked panels (a, two intruders; b, four intruders). Each team-condition row is one line of
30 square tiles, seeds 4001 to 4030 left to right.

Sources: the main study's records (outcome.touchdown per episode), cross-checked against matrix_analysis.json
cells.

    python3 -m figures.fig_breach_grid --data DATA [--out OUT] [--figures DIR]
"""
from __future__ import annotations

from figures import paths, style
from matplotlib.patches import Rectangle

OUT = paths.FIG / "supp" / "fig_breach_grid.pdf"

ARMS = ("commander", "cbaa", "independent")
CONDS = ("REF", "LI", "LB", "REF+attack", "MH")
CLAB = {"REF": "Ideal", "LI": "Independent\nloss", "LB": "Burst\nloss", "REF+attack": "Falsified\nreports",
        "MH": "Limited\nrange"}
SEEDS = tuple(range(4001, 4031))

W, H = style.W2, 162.0
XC, XA, XT = 0.5, 20.0, 47.0       # condition label, arm label, first tile (mm)
P, T = 3.8, 3.2                    # tile pitch, tile side (mm)
GG = 2.2                           # gap between condition groups (mm)
XN = XT + 30 * P + 1.0             # counts column
NR = 10.5                          # right edge of the LLM count, from XN (mm)


def load():
    out = {}
    for r in paths.records(*paths.MAIN_RUNS):
        cond = "REF+attack" if r["attacked"] else r["condition"]
        k = (r["arm"], r["reds"], cond, r["seed"])
        assert k not in out, k
        out[k] = bool(r["outcome"]["touchdown"])
    assert len(out) == 900
    return out


def panel(ax, top, n, llm, cells):
    """Draw one red-team size; return the y of its bottom edge."""
    y = top
    for c in CONDS:
        y0 = y
        for arm in ARMS:
            k = 0
            for i, seed in enumerate(SEEDS):
                hit = llm[(arm, n, c, seed)]
                k += hit
                ax.add_patch(Rectangle((XT + i * P, y - T), T, T, linewidth=0,
                                       facecolor=style.INK if hit else style.FILL))
            assert k == cells[f"{arm}|r{n}|{c}"]["breaches"], (arm, n, c)
            st = style.ARM[arm]
            yc = y - T / 2
            ax.plot([XA + 1.2], [yc], marker=st["marker"], color=st["color"], markersize=4.4,
                    ls="none", markeredgewidth=0)
            ax.text(XA + 3.2, yc, st["label"], va="center", ha="left")
            ax.text(XN + NR, yc, f"{k}", va="center", ha="right", fontweight="bold")
            y -= P
        ax.text(XC, (y0 + y + (P - T)) / 2, CLAB[c], va="center", ha="left", linespacing=1.3)
        y -= GG
    return y + GG


def seed_ticks(ax, y):
    for s in (4001, 4010, 4020, 4030):
        i = s - 4001
        x, ha = {4001: (XT, "left"), 4030: (XT + i * P + T, "right")}.get(s, (XT + i * P + T / 2, "center"))
        ax.text(x, y, f"{s}", ha=ha, va="bottom")


def main():
    style.apply()
    llm = load()
    cells = paths.analysis("matrix_analysis.json")["cells"]
    fig = style.figure(W, H)
    ax = style.axes_mm(fig, 0, 0, W, H)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    assert XN + NR <= W - 0.5, XN

    top = H - 15.0
    for n, letter, name in ((2, "a", "Two intruders"), (4, "b", "Four intruders")):
        style.panel_label(fig, XC, top + 4.6, letter)
        ax.text(XC + 3.5, top + 4.6, name, ha="left", va="top", fontweight="bold")
        seed_ticks(ax, top + 0.9)
        ax.text(XT - 2.0, top + 0.9, "Seed", ha="right", va="bottom")
        ax.text(XN + NR, top + 0.9, "Breaches", ha="right", va="bottom", fontweight="bold")
        bottom = panel(ax, top, n, llm, cells)
        top = bottom - 11.0
    assert bottom >= 0.5, bottom

    handles = [Rectangle((0, 0), 1, 1, facecolor=style.INK, linewidth=0),
               Rectangle((0, 0), 1, 1, facecolor=style.FILL, linewidth=0)]
    style.key(fig, handles, ["Breach", "No breach"],
              XT, H - 5.0, ncol=2, handleheight=1.2, handlelength=1.2)
    style.text_mm(fig, W - 0.5, H - 5.0, "One tile per episode", ha="right", va="bottom")
    style.save(fig, OUT, supplementary=True)
    print("wrote", OUT, f"{W} x {H} mm; bottom {bottom:.1f} mm")


if __name__ == "__main__":
    main()
