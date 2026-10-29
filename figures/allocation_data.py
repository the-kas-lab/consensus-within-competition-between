"""Data loaders and strip helpers for Fig. 4 (and Fig. 6's swarm/summary). Every loader asserts the numbers.tex
macros the text cites.

Fig. 4: how the teams allocated.

Message: independent defenders duplicated commitments and left known intruders without a pursuer, which the
auction prevented, while the commander's defenders rarely scouted and left intruders unseen for longer.

One dot per seed (30 seeds; each seed pools its episodes as stated per panel); black diamonds and bars are the
mean over seeds and its 95 % t interval.
a, share of sampled epoch boundaries (each epoch boundary after an episode's first) at which two or more live
   defenders were committed to the same task, by network condition. For the commander team a defender's
   commitment is its newest accepted order.
b, share of known-intruder time with no defender pursuing, four conditions without falsified reports.
c, scouting before the team's first contact (share of defender-steps), all five conditions; intruders unseen by
   every defender (share of active intruder time), four conditions without falsified reports.

Sources: a, the main study's records outcome.allocation (epochs_with_commitment_conflicts / (epochs - 1)) for the
consensus and independent teams; the commander team's records hold no allocation log, so its orders in force were
sampled at the same boundaries by analysis.commander_replays (re-simulation from the cache, 300 of 300
reproduced); b, summary.json coverage.llm['<team>|both|base'].per_seed_weighted; c, scouting from
behaviour_replays.json (900 of 900 reproduced, totals equal behaviour_pool.json), unseen from
coverage_exposure_main_study.json.
Checks: the pooled values behind every numbers.tex macro the text cites for this figure are recomputed here and
asserted; the plotted seed means are within 1 point of them.
"""
from collections import defaultdict

import numpy as np

from figures import nums, paths, style

OUT = paths.FIG / "fig_allocation.pdf"
SEEDS = list(range(4001, 4031))
CONDS = [("REF", "Ideal"), ("LI", "Independent\nloss"), ("LB", "Burst\nloss"),
         ("REF+attack", "Falsified\nreports"), ("MH", "Limited\nrange")]
TEAMS = ("commander", "cbaa", "independent")
SHORT = {"commander": "Commander", "cbaa": "Consensus", "independent": "Independent"}
DOT = 2.0          # dot diameter (pt)


# ---------------------------------------------------------------- data
def cond_of(r):
    return "REF+attack" if r["attacked"] else r["condition"]


def conflicts():
    rows = paths.records(*paths.MAIN_RUNS)
    assert len(rows) == 900
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        a = r["outcome"]["allocation"]
        if r["arm"] == "commander":
            assert a == {}, "the commander team logs no commitments"
            continue
        k = (r["arm"], cond_of(r), r["seed"])
        agg[k][0] += a["epochs_with_commitment_conflicts"]
        agg[k][1] += a["epochs"] - 1
    share = {(arm, c): [agg[(arm, c, s)][0] / agg[(arm, c, s)][1] for s in SEEDS]
             for arm in ("cbaa", "independent") for c, _ in CONDS}
    tot = lambda arm, cs: [sum(agg[(arm, c, s)][i] for c in cs for s in SEEDS) for i in (0, 1)]  # noqa: E731
    # the macros the text cites for Fig. 4a
    n, d = tot("cbaa", ["REF"])
    nums.assert_macro("numConfNCbaaRef", f"{n} of {d:,}")
    n, d = tot("independent", ["REF"])
    nums.assert_macro("numConfNIndRef", f"{n:,} of {d:,}")
    nums.assert_macro("numConfIndRef", nums.pct(n / d) + "\\%")
    n, d = tot("cbaa", ["MH"])
    nums.assert_macro("numConfNCbaaMh", f"{n} of {d:,}")
    nums.assert_macro("numConfCbaaMh", nums.pct(n / d) + "\\%")
    n, d = tot("independent", ["MH"])
    nums.assert_macro("numConfIndMh", nums.pct(n / d) + "\\%")
    other = ["REF", "LI", "LB", "REF+attack"]
    n, d = tot("cbaa", other)
    nums.assert_macro("numConfCbaaOtherN", f"{n} of {d:,}")
    seed_other = {arm: [sum(agg[(arm, c, s)][0] for c in other) / sum(agg[(arm, c, s)][1] for c in other)
                        for s in SEEDS] for arm in ("cbaa", "independent")}
    nums.assert_macro("numConfCbaaSeedMax", nums.pct(max(seed_other["cbaa"]), 1) + "\\%")
    nums.assert_macro("numConfIndSeedLo", nums.pct(min(seed_other["independent"])) + "\\%")
    nums.assert_macro("numConfIndSeedHi", nums.pct(max(seed_other["independent"])) + "\\%")
    lower = sum(a < b for a, b in zip(share[("cbaa", "MH")], share[("independent", "MH")]))
    nums.assert_macro("numConfMhSeedsLower", str(lower))
    for k, v in share.items():
        agg_pooled = sum(agg[(k[0], k[1], s)][0] for s in SEEDS) / sum(agg[(k[0], k[1], s)][1] for s in SEEDS)
        assert abs(np.mean(v) - agg_pooled) <= 0.01, (k, np.mean(v), agg_pooled)
    return share


def commander_conflicts():
    """The commander team's orders in force, sampled at the same boundaries by a re-simulation from the cache
    (analysis.commander_replays)."""
    R = paths.analysis("commander_commitments.json")
    assert R["n"] == 300 and R["n_reproduced"] == 300, (R["n"], R["n_reproduced"])
    agg = defaultdict(lambda: [0, 0])
    for x in R["episodes"]:
        k = (cond_of(x), x["seed"])
        agg[k][0] += x["boundaries_with_duplicates"]
        agg[k][1] += x["boundaries"]
    tot = {c: [sum(agg[(c, s)][i] for s in SEEDS) for i in (0, 1)] for c, _ in CONDS}
    for c, name in (("REF", "numConfNCmdRef"), ("MH", "numConfNCmdMh")):
        val = f"{tot[c][0]} of {tot[c][1]:,}"
        try:
            nums.assert_macro(name, val)
        except KeyError:
            print(f"  (macro \\{name} is not defined; value {val})")
    return {("commander", c): [agg[(c, s)][0] / agg[(c, s)][1] for s in SEEDS] for c, _ in CONDS}


def uncovered():
    V = paths.analysis("summary.json")
    assert V["seeds"] == [4001, 4030, 30]
    out = {}
    for t, tag in (("commander", "Cmd"), ("cbaa", "Cbaa"), ("independent", "Ind")):
        c = V["coverage"]["llm"][f"{t}|both|base"]
        assert c["episodes"] == 240 and c["undefined_episodes"] == 0
        nums.assert_macro(f"numCovUnc{tag}", nums.pct(c["share_uncovered_exec"]) + "\\%")
        out[t] = c["per_seed_weighted"]
        assert len(out[t]) == 30 and abs(np.mean(out[t]) - c["share_uncovered_exec"]) <= 0.01
    k5 = paths.analysis("key_rows.json")["rows"]["K5"]["pooled"]
    d = np.array(out["cbaa"]) - np.array(out["independent"])
    assert abs(d.mean() - k5["estimate"]) < 1e-4
    assert int((d < 0).sum()) == k5["seeds_pos_neg_zero"][1] == 30
    nums.assert_macro("numKFiveSeedsNeg", "30")
    return out


def scouting():
    B = paths.analysis("behaviour_replays.json")
    assert B["n"] == 900 and B["n_reproduced"] == 900
    pool = paths.analysis("behaviour_pool.json")["cells"]
    for arm in TEAMS:
        for n in (2, 4):
            tot = defaultdict(int)
            for x in B["episodes"]:
                if x["arm"] == arm and x["reds"] == n:
                    for k, v in x["active"].items():
                        tot[k] += v
            assert dict(tot) == pool[f"{arm}|r{n}|ALL"]["exact_counts"]["active_blue_steps"], (arm, n)
    agg = defaultdict(lambda: [0, 0])
    cell = defaultdict(lambda: [0, 0])
    for x in B["episodes"]:
        a = x["active"]
        rc, tot = a.get("pre:RECON", 0), sum(v for k, v in a.items() if k.startswith("pre:"))
        for key, d in (((x["arm"], x["seed"]), agg), ((x["arm"], x["reds"]), cell)):
            d[key][0] += rc
            d[key][1] += tot
    out = {t: [agg[(t, s)][0] / agg[(t, s)][1] for s in SEEDS] for t in TEAMS}
    nums.assert_macro("numCmdPreRecon", str(sum(agg[("commander", s)][0] for s in SEEDS)))
    nums.assert_macro("numCmdPreSteps", f"{sum(agg[('commander', s)][1] for s in SEEDS):,}")
    dec = [cell[(t, n)][0] / cell[(t, n)][1] for t in ("cbaa", "independent") for n in (2, 4)]
    nums.assert_macro("numPreReconDecLo", nums.pct(min(dec)))
    nums.assert_macro("numPreReconDecHi", nums.pct(max(dec)))
    return out


def unseen():
    d = paths.analysis("coverage_exposure_main_study.json")
    assert not d["outcome_mismatches"]
    cov = d["episodes"]
    assert len(cov) == 900
    out = {}
    for t, tag in (("commander", "Cmd"), ("cbaa", "Cbaa"), ("independent", None)):
        U = defaultdict(lambda: [0, 0])
        for e in cov.values():
            if e["arm"] == t and not e["attacked"] and e["condition"] in ("REF", "MH", "LI", "LB"):
                U[e["seed"]][0] += e["coverage"]["unseen"]
                U[e["seed"]][1] += e["coverage"]["active"]
        pooled = sum(v[0] for v in U.values()) / sum(v[1] for v in U.values())
        if tag:
            nums.assert_macro(f"numCovUnseen{tag}", nums.pct(pooled) + "\\%")
        out[t] = [U[s][0] / U[s][1] for s in SEEDS]
        assert abs(np.mean(out[t]) - pooled) <= 0.01
    return out


# ---------------------------------------------------------------- drawing
def swarm(values, centre, ax, colour, marker, bin_w, dx, maxw=0.42, compact=False, label=True):
    """Dots binned in y; the dots of one bin are spread symmetrically about the centre,
    at most +/- maxw (a large tie then overlaps into a band and is labelled with its count)."""
    v = np.asarray(values, float)
    bins = np.round(v / bin_w).astype(int)
    xs = np.empty_like(v)
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        n = len(idx)
        step = dx if n < 2 else min(dx, 2 * maxw / (n - 1))
        xs[idx] = centre + (np.arange(n) - (n - 1) / 2) * step
        if label and n * dx > 2 * maxw + 1e-9:
            tied = v[idx]
            exact = np.allclose(tied, tied[0])
            word = "" if compact else " seeds"
            label = f"{n}{word} at {tied[0]:.0f}" if exact else f"{n}{word},\n{tied.min():.1f} to {tied.max():.1f}"
            ax.text(centre, tied.max() + 2.2 * bin_w, label, ha="center", va="bottom", linespacing=1.25)
    ax.scatter(xs, v, s=DOT ** 2, color=colour, marker=marker, linewidth=0, zorder=3)


def summary(values, centre, ax):
    m, (lo, hi) = nums.t_ci(values)
    ax.plot([centre, centre], [max(lo, 0.0), hi], color=style.INK, lw=style.LW_EMPH, solid_capstyle="butt", zorder=4)
    ax.plot([centre], [m], marker="D", markersize=4.2, color=style.INK, markeredgewidth=0, ls="none", zorder=5)


def frame(ax, ylim, yticks, ylabel):
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)


def strip(ax, data, teams, pos, scale, bin_w, dx, maxw=0.42, moff=0.42, compact=False):
    for t, x in zip(teams, pos):
        st = style.ARM[t]
        vals = [scale * v for v in data[t]]
        swarm(vals, x, ax, st["color"], st["marker"], bin_w, dx, maxw, compact)
        summary(vals, x + moff, ax)
