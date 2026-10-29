"""Fig. 4: duplicate commitments and missed scouting.

Message: the auction removed duplicate commitments (except under limited range); the independent team left
known intruders unpursued longest; the commander's defenders barely scouted and left intruders unseen longest.

Four small multiples in the house grammar: one dot per seed (beeswarm), black mean and 95% CI from the t
distribution over the 30 seeds, teams in the fixed order.
a, duplicate commitments: share of the once-per-second checks at which two or more active defenders were
   committed to the same task, by network condition (ideal; independent loss, burst loss and falsified reports
   pooled per seed; limited range). For the commander team a defender's commitment is its newest accepted order.
b, unpursued time: share of known-intruder time with no defender pursuing (four conditions without falsified
   reports).
c, unseen time: share of active intruder time unseen by every defender (same episodes).
d, scouting before the team's first contact: share of defender time spent scouting (all five conditions).

Sources and checks: allocation_data.py loaders (each asserts the numbers.tex macros the text cites):
conflicts() and commander_conflicts() for a (grouped below from the same records), uncovered() for b,
unseen() for c, scouting() for d.

    python3 -m figures.fig_allocation --data DATA [--out OUT] [--figures DIR]
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from figures import allocation_data as cur
from figures import grammar, nums, paths, style

OUT = paths.FIG / "fig_allocation.pdf"
SEEDS = list(range(4001, 4031))
TEAMS = ("commander", "cbaa", "independent")
GROUPS = (("Ideal", ("REF",)), ("Loss or falsified\nreports", ("LI", "LB", "REF+attack")),
          ("Limited range", ("MH",)))
W, H = style.W2, 66.0


def grouped_duplicates():
    """Per seed, share of checks with duplicate commitments, pooled over the conditions of each group
    (numerators and denominators summed, as the text pools them)."""
    agg = defaultdict(lambda: [0, 0])
    rows = paths.records(*paths.MAIN_RUNS)
    for r in rows:
        if r["arm"] == "commander":
            continue
        a = r["outcome"]["allocation"]
        k = (r["arm"], cur.cond_of(r), r["seed"])
        agg[k][0] += a["epochs_with_commitment_conflicts"]
        agg[k][1] += a["epochs"] - 1
    R = paths.analysis("commander_commitments.json")
    assert R["n"] == 300 and R["n_reproduced"] == 300
    for x in R["episodes"]:
        k = ("commander", cur.cond_of(x), x["seed"])
        agg[k][0] += x["boundaries_with_duplicates"]
        agg[k][1] += x["boundaries"]
    per_cond = cur.conflicts()                       # asserts the text's macros
    per_cond.update(cur.commander_conflicts())
    out = {}
    for t in TEAMS:
        for name, conds in GROUPS:
            out[(t, name)] = [100 * sum(agg[(t, c, s)][0] for c in conds) / sum(agg[(t, c, s)][1] for c in conds)
                              for s in SEEDS]
            if len(conds) == 1:
                assert np.allclose(out[(t, name)], [100 * v for v in per_cond[(t, conds[0])]])
    return out


def column(ax, x, values, arm):
    grammar.swarm_dots(ax, x - 0.1, values, arm, half_width=0.32)
    m, (lo, hi) = nums.t_ci(values)
    grammar.mean_ci(ax, x + 0.3, m, max(lo, 0.0), hi)


def frame(ax, ylim, yticks, ylabel):
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel)


def main():
    style.apply()
    dup = grouped_duplicates()
    unp = {t: [100 * v for v in vals] for t, vals in cur.uncovered().items()}
    uns = {t: [100 * v for v in vals] for t, vals in cur.unseen().items()}
    sco = {t: [100 * v for v in vals] for t, vals in cur.scouting().items()}
    fig = style.figure(W, H)
    y0, hh = 11.0, 41.0

    # a
    ax = style.axes_mm(fig, 13.0, y0, 66.0, hh)
    ax.set_xlim(-0.6, 3 * 3.2 - 1.0)
    frame(ax, (-3, 100), [0, 25, 50, 75, 100], "Duplicate commitments (% of checks)")
    for g, (name, _) in enumerate(GROUPS):
        for k, t in enumerate(TEAMS):
            column(ax, g * 3.2 + k, dup[(t, name)], t)
    ax.set_xticks([g * 3.2 + 1 for g in range(3)], [n for n, _ in GROUPS])
    ax.tick_params(axis="x", length=0)
    style.panel_label(fig, 1.0, H - 8.0, "a")

    # b, c, d
    specs = (("b", unp, (0, 50), [0, 25, 50], "Unpursued time (%)"),
             ("c", uns, (30, 70), [30, 50, 70], "Unseen time (%)"),
             ("d", sco, (-2, 60), [0, 20, 40, 60], "Scouting before first contact (%)"))
    for k, (letter, data, ylim, yt, lab) in enumerate(specs):
        x = 96.0 + k * 29.0
        ax = style.axes_mm(fig, x, y0, 19.0, hh)
        ax.set_xlim(-0.6, 2.7)
        frame(ax, ylim, yt, lab)
        for j, t in enumerate(TEAMS):
            assert min(data[t]) >= ylim[0] and max(data[t]) <= ylim[1], (letter, t, min(data[t]), max(data[t]))
            column(ax, j, data[t], t)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        style.panel_label(fig, x - 13.0, H - 8.0, letter)

    # one key for the figure
    hs, ls = grammar.team_handles(TEAMS)
    style.key(fig, hs, ls, 13.0, H - 4.6, ncol=3, handletextpad=0.3, columnspacing=1.6)
    kx = fig.add_axes([0, 0, 1, 1], facecolor="none")
    kx.set_xlim(0, W)
    kx.set_ylim(0, H)
    kx.axis("off")
    grammar.mean_ci_key(kx, 0.43, H - 2.9, length=0.025)
    style.save(fig, OUT)


if __name__ == "__main__":
    main()
