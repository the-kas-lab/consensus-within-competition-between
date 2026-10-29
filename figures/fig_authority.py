"""Fig. 6: more LLM authority, and model capability.

Message: wider authority raised breaches with the low-cost model but not with the
near-frontier model, which used it to leave known intruders unpursued without losing
detection; in the commander's seat the near-frontier model was breached less often
under falsified reports.

a, bid-weight minus bounded auction team (low-cost LLM), per-seed paired difference in
   breach rate: four conditions without falsified reports, two and four intruders
   (8 pairs per seed), then four intruders only (4 pairs per seed).
b, ideal network, four intruders, one episode per seed: breach outcome per seed for the
   bounded team, the low-cost bid-weight team and the near-frontier bid-weight team, and
   the paired differences with Newcombe (method 10) intervals; dots are not drawn
   because each seed's difference is -1, 0 or +1.
c, same episodes as b: share of active intruder time unseen by every defender, and share
   of known-intruder time with no pursuer.
d, near-frontier minus low-cost commander, per-seed paired difference in breach rate,
   ideal network and falsified reports, two and four intruders (2 pairs per seed).

Sources: a, high_authority.json (rows N2 W - B, | r4);
b, records of the main study (bounded = auction team), the high-authority low-cost study and the high-authority
near-frontier study (the seed -> run map of high_authority_near_frontier.json seeds_by_run), checked against
high_authority_near_frontier.json; c, coverage_exposure_main_study.json (bounded),
coverage_exposure_high_authority_low_cost.json and coverage_exposure_high_authority_near_frontier.json (valid
records only, as high_authority_near_frontier_secondary.json), checked against that file; d, summary.json
near_frontier_commander['3_sol_minus_gemini|REF' / '|REF+attack'] (= key_rows X12, X13).
Every printed value and every cited macro is asserted against numbers.tex.

    python3 -m figures.fig_authority --data DATA [--out OUT] [--figures DIR]
"""
from __future__ import annotations

import numpy as np

from analysis import common
from figures import nums, paths, style
from figures.allocation_data import summary, swarm
from matplotlib.patches import Rectangle

OUT = paths.FIG / "fig_authority.pdf"
M = style.MINUS
SEEDS = list(range(4001, 4031))
W, H = style.W2, 132.0
DOT_STEP, DOT_D = 0.5, 1.5

TEAM = {  # key -> (arm style, open marker = near-frontier, label)
    "B": ("cbaa", False, "Low authority, low-cost"),
    "W": ("bidweight", False, "High authority, low-cost"),
    "WSOL": ("bidweight", True, "High authority, near-frontier"),
    "CMD": ("commander", False, "Commander, low-cost"),
    "SOLCMD": ("commander", True, "Commander, near-frontier"),
}


def marker(key, size=4.2):
    arm, hollow, _ = TEAM[key]
    st = style.ARM[arm]
    if hollow:
        return dict(marker=st["marker"], markerfacecolor=style.PAPER, markeredgecolor=st["color"],
                    markeredgewidth=style.LW_RULE, markersize=size, ls="none")
    return dict(marker=st["marker"], color=st["color"], markeredgewidth=0, markersize=size, ls="none")


# ---------------------------------------------------------------- data
def panel_a():
    S = paths.analysis("high_authority.json")
    out = []
    for key, name, tag in (("N2 W - B", "Two and four\nintruders", "numWminusBPP"),
                           ("N2 W - B | r4", "Four intruders", "numWminusBFourPP")):
        r = S["rows"][key]
        b = r["breach_difference"]
        seeds = [x["d_breach"] for x in r["per_seed"]]
        nums.assert_seed_summary(seeds, b["estimate"], b["ci95_t_seeds"])
        nums.assert_macro(tag, nums.signed_pp(b["estimate"]))
        nums.assert_macro(tag + "CI", nums.pp_ci(*b["ci95_t_seeds"]))
        if key == "N2 W - B":
            assert abs(b["estimate"] - S["family_N_holm"][key]["estimate"]) < 1e-9
            nums.assert_macro("numRateWSwPct", nums.pct(r["first_breach_rate"]))
            nums.assert_macro("numRateBSwPct", nums.pct(r["second_breach_rate"]))
        out.append(dict(label=name, seeds=[100 * v for v in seeds], est=100 * b["estimate"],
                        ci=[100 * v for v in b["ci95_t_seeds"]]))
    return out


def ref4(recs, keep=lambda r: True):
    out = {}
    for r in recs:
        if r["reds"] == 4 and r["condition"] == "REF" and not r["attacked"] and keep(r):
            out.setdefault(r["seed"], []).append(r)
    return out


def panel_b():
    W_ = paths.analysis("high_authority_near_frontier.json")
    M_ = paths.records(*paths.MAIN_RUNS)
    B = ref4(M_, lambda r: r["arm"] == "cbaa")
    Wl = ref4(paths.records("high_authority_low_cost/seeds_4001_4030"))
    sol = {}
    for run, seeds in W_["seeds_by_run"].items():        # the valid record of each seed, by run
        recs = ref4(paths.records(f"high_authority_near_frontier/{run}"))
        for s in seeds:
            assert len(recs[s]) == 1 and s not in sol, (run, s)
            sol[s] = recs[s]
    assert sorted(sol) == SEEDS
    out = {}
    for k, d in (("B", B), ("W", Wl), ("WSOL", sol)):
        out[k] = []
        for s in SEEDS:
            assert len(d[s]) == 1, (k, s, len(d[s]))
            out[k].append(int(bool(d[s][0]["outcome"]["touchdown"])))
    lv = W_["levels"]
    for k, lk, mac in (("WSOL", "W-Sol", "numBrWSol"), ("W", "W", "numBrW"), ("B", "B", "numBrB")):
        assert f"{sum(out[k])}/30" == lv[lk]["breaches"], (k, sum(out[k]))
        nums.assert_macro(mac, f"{sum(out[k])} of 30")
    rows = []
    for rid, a, b, tag, label in (("S1 W-Sol - W", "WSOL", "W", "numSOnePP", f"Near-frontier {M}\nlow-cost, high authority"),
                                  ("secondary W-Sol - B", "WSOL", "B", "numWSolBPP", f"Near-frontier, high authority\n{M} low authority")):
        r = W_["rows"][rid]
        f = sum(x and not y for x, y in zip(out[a], out[b]))
        g = sum(y and not x for x, y in zip(out[a], out[b]))
        assert (f, g) == (r["first_only"], r["second_only"]), (rid, f, g)
        nums.assert_macro(tag, nums.signed_pp(r["estimate"]))
        nums.assert_macro(tag + "CI", nums.pp_ci(*r["newcombe10"]))
        rows.append(dict(label=label, est=100 * r["estimate"], ci=[100 * v for v in r["newcombe10"]], fg=(f, g)))
    nums.assert_macro("numSOneMcN", nums.pval(W_["rows"]["S1 W-Sol - W"]["exact_mcnemar_p"]))
    return out, rows


def panel_c():
    valid = {r["episode_id"] for r in common.valid_records(str(paths.DATA), common.HIGH_AUTHORITY_NEAR_FRONTIER)}
    d = paths.analysis("coverage_exposure_high_authority_near_frontier.json")
    assert d["outcome_mismatches"] == []
    solw = {}
    for eid, v in d["episodes"].items():
        if eid in valid:
            assert v["seed"] not in solw
            solw[v["seed"]] = v["coverage"]

    def cov(name, arm=None):
        d = paths.analysis(name)
        return {e["seed"]: e["coverage"] for e in d["episodes"].values()
                if e["reds"] == 4 and e["condition"] == "REF" and not e["attacked"] and (arm is None or e["arm"] == arm)}
    arms = {"B": cov("coverage_exposure_main_study.json", "cbaa"),
            "W": cov("coverage_exposure_high_authority_low_cost.json"), "WSOL": solw}
    SS = paths.analysis("high_authority_near_frontier_secondary.json")
    unseen, uncov = {}, {}
    for k, lk, tag in (("B", "B", "B"), ("W", "W", "W"), ("WSOL", "W-Sol", "WSol")):
        v = arms[k]
        assert sorted(v) == SEEDS, k
        pu = sum(v[s]["unseen"] for s in SEEDS) / sum(v[s]["active"] for s in SEEDS)
        pk = sum(v[s]["known_team_no_exec"] for s in SEEDS) / sum(v[s]["known_team"] for s in SEEDS)
        assert abs(pu - SS["unseen_share_pooled"][lk]) < 6e-5 and abs(pk - SS["uncovered_known_pooled"][lk]) < 6e-5
        nums.assert_macro(f"numUnseenRef{tag}", nums.pct(SS["unseen_share_pooled"][lk]))
        nums.assert_macro(f"numUncov{tag}", nums.pct(SS["uncovered_known_pooled"][lk]))
        unseen[k] = [v[s]["unseen"] / v[s]["active"] for s in SEEDS]
        uncov[k] = [v[s]["known_team_no_exec"] / v[s]["known_team"] for s in SEEDS if v[s]["known_team"] > 0]
    return unseen, uncov


def panel_d():
    V = paths.analysis("summary.json")["near_frontier_commander"]
    P = paths.analysis("key_rows.json")["rows"]
    out = []
    for key, x, tag, name in (("3_sol_minus_gemini|REF", "X12", "numSolCmdRefPP", "Ideal network"),
                              ("3_sol_minus_gemini|REF+attack", "X13", "numSolCmdAtkPP", "Falsified reports")):
        r = V[key]
        nums.assert_seed_summary(r["per_seed"], r["estimate"], r["ci95_t_seeds"])
        assert abs(P[x]["pooled"]["estimate"] - r["estimate"]) < 1e-9
        nums.assert_macro(tag, nums.signed_pp(r["estimate"]))
        nums.assert_macro(tag + "CI", nums.pp_ci(*P[x]["pooled"]["ci95_t"]))
        out.append(dict(label=name, seeds=[100 * v for v in r["per_seed"]], est=100 * r["estimate"],
                        ci=[100 * v for v in r["ci95_t_seeds"]]))
    cells = V["cells"]
    nums.assert_macro("numCTwoBrRef", str(cells["solcmd|r2|REF"] + cells["solcmd|r4|REF"]))
    nums.assert_macro("numCTwoBrAtk", str(cells["solcmd|r2|REF+attack"] + cells["solcmd|r4|REF+attack"]))
    return out


# ---------------------------------------------------------------- drawing: a, one forest of every authority
# comparison; b, seed strips on the ideal network with four intruders; c, unseen and unpursued time for the teams
# of b.
from figures import grammar  # noqa: E402

W, H = style.W2, 170.0
STRIPS = tuple((k, TEAM[k][2]) for k in ("B", "W", "WSOL"))


def row(label, seeds_pp, est_pp, ci_pp):
    return dict(label=label, d=np.asarray(seeds_pp, float), m=est_pp, lo=ci_pp[0], hi=ci_pp[1],
                text=f"{grammar.fmt_est(est_pp / 100)} ({grammar.fmt_ci(ci_pp[0] / 100, ci_pp[1] / 100)})")


def blocks(a, outcomes, brows, d):
    s1, wsb = brows
    diff = lambda x, y: [100 * (p - q) for p, q in zip(outcomes[x], outcomes[y])]  # noqa: E731
    return [
        ("High vs low authority, low-cost",
         [row("Two and four intruders", *[a[0][k] for k in ("seeds", "est", "ci")]),
          row("Four intruders", *[a[1][k] for k in ("seeds", "est", "ci")])],
         "High authority better", "Low authority better"),
        ("Near-frontier vs low-cost, ideal network",
         [row("Both at high authority", diff("WSOL", "W"), s1["est"], s1["ci"]),
          row("High vs low authority", diff("WSOL", "B"), wsb["est"], wsb["ci"])],
         "Near-frontier better", "Low-cost better"),
        ("Near-frontier vs low-cost commander",
         [row("Ideal network", *[d[0][k] for k in ("seeds", "est", "ci")]),
          row("Falsified reports", *[d[1][k] for k in ("seeds", "est", "ci")])],
         "Near-frontier better", "Low-cost better"),
    ]


def weights_panel(fig):
    """a: what each authority level lets the consensus team's LLM set, and what the LLMs set. One axis per weight.
    Black ticks: the values low
    authority can reach (Supplementary Table 12: the four profiles; the per-intruder priority is not a weight and is not drawn).
    Band: the range high authority allows (Supplementary Table 7). Markers: median of the LLMs' answers at high
    authority with the 25th-75th percentiles, low-cost (filled) and near-frontier (hollow), on the ideal network with
    four intruders (matched episodes), before and after the defender knew of an intruder. Before contact no pursuit
    task exists, so the pursuit weights act on nothing and are not drawn there.
    Source: bid_weights.json (W_ref4, WSOL), the source of Supplementary Table 7."""
    bw = paths.analysis("bid_weights.json")
    assert bw["episodes"]["W_ref4"] == 30 and bw["episodes"]["WSOL"] == 30
    rows = (("w_threat", "Threat value", (1.2,), True),
            ("w_engage", "Engagement value", (1.0,), True),
            ("w_coverage", "Station coverage", (0.5, 0.8), False),
            ("w_scout", "Scouting information", (0.45, 0.75), False),
            ("w_attrition", "Attrition cost", (0.2, 0.5, 0.9), False))
    st = style.ARM["bidweight"]
    top, rh = 157.5, 4.7                          # first row centre (mm), row pitch
    for k, (phase, head, x0) in enumerate((("before_contact", "Before contact", 42.0),
                                           ("with_contact", "After contact", 116.0))):
        ax = style.axes_mm(fig, x0, top - (len(rows) - 0.5) * rh, 60.0, len(rows) * rh)
        ax.set_xlim(0, 3.0)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks(range(len(rows)), [lab for _, lab, *_ in rows] if k == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Weight (dimensionless)", labelpad=1)
        ax.set_title(head, fontweight="bold", loc="left", pad=2)
        for i, (w, _, opts, pursuit) in enumerate(rows):
            lo, hi = bw["ranges"][w]
            ax.add_patch(Rectangle((lo, i - 0.32), hi - lo, 0.64, facecolor=st["fill"], edgecolor="none", zorder=0))
            for v in opts:
                ax.plot([v, v], [i - 0.42, i + 0.42], color=style.TEXT, lw=style.LW_RULE, solid_capstyle="butt", zorder=2)
            if pursuit and phase == "before_contact":
                continue
            for dy, (g, hollow) in ((-0.17, ("W_ref4", False)), (0.17, ("WSOL", True))):
                q = bw[g][phase]["weights_p10_25_50_75_90"][w]
                ax.plot([q[1], q[3]], [i + dy, i + dy], color=st["color"], lw=style.LW_DATA, zorder=3)
                kw = dict(marker=st["marker"], markersize=3.6, ls="none", zorder=4)
                kw.update(dict(markerfacecolor=style.WHITE, markeredgecolor=st["color"], markeredgewidth=style.LW_RULE)
                          if hollow else dict(color=st["color"], markeredgewidth=0))
                ax.plot([q[2]], [i + dy], **kw)
    # key: one row above the two axes
    kx = fig.add_axes([0, 0, 1, 1], facecolor="none")
    kx.set_xlim(0, W)
    kx.set_ylim(0, H)
    kx.axis("off")
    y = H - 3.0
    kx.plot([42.0, 42.0], [y - 1.4, y + 1.4], color=style.TEXT, lw=style.LW_RULE, solid_capstyle="butt")
    kx.text(43.6, y, "Low-authority option", ha="left", va="center")
    kx.add_patch(Rectangle((76.0, y - 1.4), 5.0, 2.8, facecolor=st["fill"], edgecolor="none"))
    kx.text(82.6, y, "High-authority range", ha="left", va="center")
    for x, hollow, lab in ((116.0, False, "Low-cost answer"), (146.0, True, "Near-frontier answer")):
        kx.plot([x - 2.0, x + 2.0], [y, y], color=st["color"], lw=style.LW_DATA)
        kw = dict(marker=st["marker"], markersize=3.6, ls="none")
        kw.update(dict(markerfacecolor=style.WHITE, markeredgecolor=st["color"], markeredgewidth=style.LW_RULE)
                  if hollow else dict(color=st["color"], markeredgewidth=0))
        kx.plot([x], [y], **kw)
        kx.text(x + 3.2, y, lab, ha="left", va="center")
    style.panel_label(fig, 1.0, H - 1.0, "a")


def strip_marker(ax, x, y, key):
    arm, hollow, _ = TEAM[key]
    st = style.ARM[arm]
    kw = dict(marker=st["marker"], ls="none", markersize=4.2, clip_on=False)
    kw.update(dict(markerfacecolor=style.WHITE, markeredgecolor=st["color"], markeredgewidth=style.LW_RULE)
              if hollow else dict(color=st["color"], markeredgewidth=0))
    ax.plot([x], [y], **kw)


def main():
    style.apply()
    a = panel_a()
    outcomes, brows = panel_b()
    unseen, uncov = panel_c()
    d = panel_d()
    fig = style.figure(W, H)

    # a: forest
    ax = style.axes_mm(fig, 62.0, 61.0, 70.0, 58.0)
    y = grammar.forest(ax, blocks(a, outcomes, brows, d), (-112, 112), [-100, -50, 0, 50, 100], "", "")
    ax.set_xlabel("Difference in breach rate (percentage points)")
    grammar.mean_ci_key(ax, -0.75, -2.2)
    grammar.bubble_key(ax, -0.2, -2.2, label="Seeds")
    ax.set_ylim(y - 0.4, -2.6)
    style.panel_label(fig, 1.0, 125.0, "b")
    weights_panel(fig)

    # c: seed strips, ideal network, four intruders
    sx = fig.add_axes([0, 0, 1, 1], facecolor="none")
    sx.set_xlim(0, W)
    sx.set_ylim(0, H)
    sx.axis("off")
    style.panel_label(fig, 1.0, 48.0, "c")
    tile, gap = 1.8, 0.3
    x0 = 36.0
    for r, (key, label) in enumerate(STRIPS):
        yy = 36.0 - r * 7.5
        sx.text(4.0, yy, label, ha="left", va="center")      # row label; the markers key panel c
        for k, v in enumerate(outcomes[key]):
            sx.add_patch(Rectangle((x0 + k * (tile + gap), yy - tile / 2), tile, tile, facecolor=style.TEXT if v else style.FILL,
                                   edgecolor="none"))
        sx.text(x0 + 30 * (tile + gap) + 1.0, yy, f"{sum(outcomes[key])} of 30", ha="left", va="center")
    sx.add_patch(Rectangle((x0, 42.6), tile, tile, facecolor=style.TEXT, edgecolor="none"))
    sx.text(x0 + tile + 1.0, 42.6 + tile / 2, "Breach", ha="left", va="center")
    sx.add_patch(Rectangle((x0 + 16.0, 42.6), tile, tile, facecolor=style.FILL, edgecolor="none"))
    sx.text(x0 + 16.0 + tile + 1.0, 42.6 + tile / 2, "No breach", ha="left", va="center")
    sx.text(x0 + 30 * (tile + gap), 42.6 + tile / 2, "One tile per seed", ha="right", va="center")

    # d: unseen and unpursued time, same episodes as c
    for k, (letter_x, data, lab, lim, ticks) in enumerate(((126.0, unseen, "Unseen time (%)", (25, 75), [25, 50, 75]),
                                                            (158.0, uncov, "Unpursued time (%)", (0, 60), [0, 30, 60]))):
        cx = style.axes_mm(fig, letter_x, 8.0, 19.0, 25.0)
        cx.set_xlim(-0.6, 2.6)
        cx.set_ylim(*lim)
        cx.set_yticks(ticks)
        cx.set_ylabel(lab)
        for j, (key, _) in enumerate(STRIPS):
            v = [100 * x for x in data[key]]
            assert lim[0] <= min(v) and max(v) <= lim[1], (lab, key, min(v), max(v))
            arm, hollow, _ = TEAM[key]
            grammar.swarm_dots(cx, j - 0.05, v, arm, open_marker=hollow, half_width=0.3)
            m, (lo, hi) = nums.t_ci(v)
            grammar.mean_ci(cx, j + 0.38, m, lo, hi)
        cx.set_xticks([])
        cx.spines["bottom"].set_visible(False)
    style.panel_label(fig, 113.0, 48.0, "d")
    for r, (key, label) in enumerate(STRIPS):                 # d's key: the markers it uses
        strip_marker(sx, 119.0, 45.0 - r * 3.4, key)
        sx.text(121.5, 45.0 - r * 3.4, label, ha="left", va="center")
    style.save(fig, OUT)


if __name__ == "__main__":
    main()
