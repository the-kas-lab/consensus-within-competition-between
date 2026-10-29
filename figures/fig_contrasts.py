"""Fig. 3: consensus allocation lowered breaches.

Message: the auction team was breached least often, and its gaps to the other two teams came from
four-intruder episodes.

a, breach rate of each team with two and with four intruders (four conditions without falsified reports,
   120 episodes per bar); bars from zero, black 95% CI from the t distribution over the 30 per-seed rates.
b, the three team comparisons as a forest: per-seed paired differences (grey dots), mean and 95% CI over
   seeds (black), for both intruder counts together (the prespecified comparison) and for each count.
   Values printed as in the text.

Sources: the main study's records (outcome.touchdown), recomputed per seed and asserted against
matrix_analysis.json (primary_family per_seed_d, estimate and CI; contrasts "| r2" and "| r4" estimate and CI)
and against the numbers.tex macros the text prints.

    python3 -m figures.fig_contrasts --data DATA [--out OUT] [--figures DIR]
"""
from __future__ import annotations

import numpy as np

from figures import grammar, nums, paths, style

OUT = paths.FIG / "fig_contrasts.pdf"
M = style.MINUS
SEEDS = list(range(4001, 4031))
ARMS = ("commander", "cbaa", "independent")
TAG = {"commander": "Cmd", "cbaa": "Cbaa", "independent": "Ind"}
CONDS = ("REF", "LI", "LB", "MH")
# (first, second, analysis key, sign that turns the stored difference into first minus second, macro stem)
COMPARISONS = (("commander", "cbaa", "A1 commander - cbaa", +1, "AOne"),
               ("independent", "cbaa", "A3 cbaa - independent", -1, "AThree"),
               ("commander", "independent", "A2 commander - independent", +1, "ATwo"))
NAME = {"commander": "Commander", "cbaa": "Consensus", "independent": "Independent"}


# ---------------------------------------------------------------- data
def breaches():
    rows = paths.records(*paths.MAIN_RUNS)
    assert len(rows) == 900
    b = {(r["arm"], r["reds"], r["condition"], r["seed"]): int(bool(r["outcome"]["touchdown"]))
         for r in rows if not r["attacked"]}
    assert len(b) == 3 * 2 * 4 * 30
    return b


def seed_rate(b, arm, reds):
    return np.array([np.mean([b[(arm, n, c, s)] for n in reds for c in CONDS]) for s in SEEDS])


def rates(b):
    out = {}
    for arm in ARMS:
        both = seed_rate(b, arm, (2, 4))
        nums.assert_macro(f"numRate{TAG[arm]}Pct", nums.pct(both.mean()))
        nums.assert_macro(f"numRate{TAG[arm]}FourPct", nums.pct(seed_rate(b, arm, (4,)).mean()))
        for n in (2, 4):
            v = seed_rate(b, arm, (n,))
            m, (lo, hi) = nums.t_ci(v)
            out[(arm, n)] = (100 * m, 100 * max(lo, 0.0), 100 * hi)
    return out


def comparisons(b):
    a = paths.analysis("matrix_analysis.json")
    out = []
    for first, second, key, sign, stem in COMPARISONS:
        rows = []
        for reds, label in (((2, 4), "Two and four intruders"), ((2,), "Two intruders"), ((4,), "Four intruders")):
            d = seed_rate(b, first, reds) - seed_rate(b, second, reds)
            if reds == (2, 4):
                r = a["primary_family"][key]
                stored, ci = r["estimate"], r["ci95_t_seeds"]
                assert np.allclose(sign * np.array(r["per_seed_d"]), d, atol=1e-9)
            else:
                r = a["contrasts"][f"{key} | r{reds[0]}"]["breach_difference"]
                stored, ci = r["estimate"], r["ci95_t_seeds"]
            m, (lo, hi) = nums.t_ci(d)
            s_ci = sorted(sign * np.array(ci))
            assert abs(m - sign * stored) < 6e-5 and abs(lo - s_ci[0]) < 6e-5 and abs(hi - s_ci[1]) < 6e-5
            text = f"{_signed(m)} ({_ci(lo, hi)})"
            _check_text(stem, reds, sign, m, lo, hi)
            rows.append(dict(label=label, d=100 * d, m=100 * m, lo=100 * lo, hi=100 * hi, text=text))
        out.append(((f"{NAME[first]} vs {NAME[second]}", NAME[first], NAME[second]), rows))
    return out


def _signed(v):
    s = nums.pct(abs(v))
    return s if s == "0" else (("+" if v > 0 else M) + s)


def _ci(lo, hi):
    if lo > 0 or hi < 0:                       # text style: no signs when the CI excludes zero
        return f"{_signed(lo).lstrip('+')} to {_signed(hi).lstrip('+')}"
    return f"{_signed(lo)} to {_signed(hi)}"


def _check_text(stem, reds, sign, m, lo, hi):
    """Where the text prints this comparison, the figure prints the same numbers."""
    part = {(2, 4): "", (2,): "Two", (4,): "Four"}[reds]
    est = nums.signed_pp(sign * m if sign < 0 else m)
    if sign < 0:   # the text states A3 as independent minus auction through the unsigned macros
        if (lo > 0 or hi < 0):
            nums.assert_macro(f"num{stem}{part}PPAbs", nums.pct(abs(m)))
            nums.assert_macro(f"num{stem}{part}PPAbsCI", f"{nums.pct(min(abs(lo), abs(hi)))} to {nums.pct(max(abs(lo), abs(hi)))}")
        return
    pre = "num" if f"num{stem}{part}PP" in _names() else "si"     # by-count rows the text omits are in SI Table 2
    nums.assert_macro(f"{pre}{stem}{part}PP", est)
    nums.assert_macro(f"{pre}{stem}{part}PPCI", nums.pp_ci(lo, hi))


def _names():
    nums.macro("numRateCmdPct")
    return nums._CACHE


# ---------------------------------------------------------------- drawing
W, H = style.W2, 70.0


def panel_a(fig, r):
    ax = style.axes_mm(fig, 12.0, 9.0, 38.0, 53.0)
    bw = 0.26
    for gi, n in enumerate((2, 4)):
        for ti, arm in enumerate(ARMS):
            x = gi * 1.15 + (ti - 1) * (bw + 0.05)
            m, lo, hi = r[(arm, n)]
            ax.bar(x, m, width=bw, color=style.ARM[arm]["color"], lw=0, zorder=2)
            ax.plot([x, x], [lo, hi], color=style.TEXT, lw=grammar.CI_LW, solid_capstyle="butt", zorder=4)
    ax.set_xticks([0, 1.15], ["Two intruders", "Four intruders"])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, 1.65)
    ax.set_ylim(0, 70)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_ylabel("Breach rate (%)")
    hs, ls = grammar.team_handles(ARMS)
    for h in hs:
        h.set_marker("s")
    hs, ls = hs + [grammar.ci_handle()], ls + ["95% CI"]
    ax.legend(hs, ls, loc="upper left", bbox_to_anchor=(0.0, 1.06), frameon=False, handlelength=0.8,
              handletextpad=0.4, borderaxespad=0, labelspacing=0.3)
    style.panel_label(fig, 1.5, H - 2.0, "a")


def panel_b(fig, blocks):
    x0, w = 96.0, 62.0
    rows = sum(1 + len(r) for _, r in blocks)
    ax = style.axes_mm(fig, x0, 9.0, w, 57.0)
    y = 0.0
    yt, yl = [], []
    for (title, first, second), rws in blocks:
        tr = ax.get_yaxis_transform()
        ax.text(-0.02, y, title, transform=tr, ha="right", va="center", fontweight="bold")
        # the direction, named: left of zero the first team had fewer breaches, right of zero the second
        ax.text(-3, y, f"← {first} better", ha="right", va="center")
        ax.text(3, y, f"{second} better →", ha="left", va="center")
        y += 1
        for rw in rws:
            grammar.diff_row(ax, y, rw["d"], rw["m"], rw["lo"], rw["hi"])
            yt.append(y)
            yl.append(rw["label"])
            ax.text(1.03, y, rw["text"], transform=ax.get_yaxis_transform(), ha="left", va="center")
            y += 1
    ax.set_yticks(yt, yl)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(y - 0.4, -1.0)
    grammar.mean_ci_key(ax, -0.52, -0.9)
    grammar.bubble_key(ax, 0.10, -0.9, label="Seeds")
    ax.set_xlim(-100, 100)
    ax.set_xticks([-100, -50, 0, 50, 100], [f"{M}100", f"{M}50", "0", "+50", "+100"])
    ax.plot([0, 0], [-0.45, y - 0.4], color=style.ZERO, lw=style.LW_RULE, zorder=1)   # zero line, not through the key row
    ax.set_xlabel("Difference in breach rate (percentage points)")
    ax.text(1.03, -0.9, "Mean (95% CI)", transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontweight="bold")
    style.panel_label(fig, 54.0, H - 2.0, "b")


def main():
    style.apply()
    b = breaches()
    r = rates(b)
    blocks = comparisons(b)
    fig = style.figure(W, H)
    panel_a(fig, r)
    panel_b(fig, blocks)
    style.save(fig, OUT)


if __name__ == "__main__":
    main()
