"""Supplementary Data 1: every paired comparison of Supplementary Tables 2, 4 and 8 with its per-seed values (CSV).

Per-seed values are recomputed here from the episode records (one value per seed: the mean over that seed's matched
pairs of the first breach indicator minus the second), and their mean is asserted against the stored estimate of
the same comparison; confidence intervals and P values are copied from the analysis outputs.

    python3 -m analysis.make_si_data --data DATA [--out OUT]       ->  OUT/Supplementary_Data_1.csv
"""
import csv
import json
import os

import numpy as np

from analysis import common, numfmt
from analysis.numfmt import leaf, load

_args = common.parser(__doc__.split("\n")[0]).parse_args()
DATA = _args.data
numfmt.set_dir(common.out_dir(_args))
OUT = os.path.join(common.out_dir(_args), "Supplementary_Data_1.csv")
SEEDS = list(range(4001, 4031))


def records(*runs):
    return common.records(DATA, [tuple(r.split("/")) for r in runs])


# ---- episode outcomes by (team, intruders, condition, seed)
C = {}


def put(team, r):
    cond = "ATK" if r["attacked"] else r["condition"]
    o = r["outcome"]
    C[(team, r["reds"], cond, r["seed"])] = (float(o["touchdown"]), o["touchdown_step"] * 0.1 if o["touchdown"] else 40.0)


for r in records("main_study/seeds_4001_4010", "main_study/seeds_4011_4030"):
    put(r["arm"], r)
for r in records("high_authority_low_cost/seeds_4001_4030"):
    put("bidweight", r)
_valid = {x["episode_id"] for x in load("endings_WSOL.json")["episodes"]}
for r in records("high_authority_near_frontier/first_run", "high_authority_near_frontier/second_run"):
    if r["episode_id"] in _valid:
        put("bidweight_nearfrontier", r)
for r in records("near_frontier_commander/seeds_4001_4010", "near_frontier_commander/seeds_4011_4030"):
    put("commander_nearfrontier", r)

FOUR = ["REF", "LI", "LB", "MH"]
NAME = {"REF": "ideal network", "LI": "independent loss", "LB": "burst loss", "ATK": "falsified reports", "MH": "limited range"}


def diff_teams(a, b, reds, conds, k=0):
    return [np.mean([C[(a, r, c, s)][k] - C[(b, r, c, s)][k] for r in reds for c in conds]) for s in SEEDS]


def effect(t, cond, base, reds, k=0):
    return [np.mean([C[(t, r, cond, s)][k] - C[(t, r, base, s)][k] for r in reds]) for s in SEEDS]


rows = []
MA = "matrix_analysis.json"


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


PX = "key_rows.json"
KH = dict(zip(["K1", "K2", "K4", "K5"], holm([leaf(PX, ["rows", k, "pooled", "p_t"]) for k in ("K1", "K2", "K4", "K5")])))
_sw = "high_authority.json"
NH = dict(zip(["N2", "N5"], holm([leaf(_sw, ["family_N_holm", k, "p_t_raw"]) for k in ("N2 W - B", "N5 attack effect W - B")])))
KFAM = "Holm, family of 4 added during the study (recomputed over its LLM rows)"
NFAM = "Holm, family of 2 added during the study (recomputed over its LLM rows)"


def stored_t(d):
    return d["estimate"], d["ci95_t_seeds"], "t", d.get("p_t_raw"), d.get("p_sign_flip", d.get("p_sign_flip_exact_conditional_on_symmetry"))


def add(table, comp, measure, first, second, conds, reds, vals, est, ci, method, pt, psf, padj="", adj="", tol=1e-3):
    m = float(np.mean(vals))
    assert abs(m - est) < tol, (comp, m, est)
    pps = len(reds) * len(conds) if isinstance(conds, list) else len(reds)
    rows.append({"supplementary_table": table, "comparison": comp, "measure": measure, "first": first, "second": second,
                 "conditions": "; ".join(NAME[c] for c in conds) if isinstance(conds, list) else conds,
                 "intruders": " and ".join(str(r) for r in reds), "pairs_per_seed": pps,
                 "estimate": round(est, 4), "ci95_low": "" if ci is None else round(ci[0], 4),
                 "ci95_high": "" if ci is None else round(ci[1], 4), "ci_method": method,
                 "p_t": "" if pt is None else pt, "p_sign_flip_or_mcnemar": "" if psf is None else psf,
                 "p_adjusted": padj, "adjustment": adj, **{f"seed_{s}": round(float(v), 4) for s, v in zip(SEEDS, vals)}})


# ---- Supplementary Table 2: team contrasts (breach rate)
T2 = [("commander", "cbaa", "A1 commander - cbaa"), ("commander", "independent", "A2 commander - independent"),
      ("cbaa", "independent", "A3 cbaa - independent")]
TN = {"commander": "commander", "cbaa": "consensus", "independent": "independent",
      "bidweight": "consensus at high authority, low-cost", "bidweight_nearfrontier": "consensus at high authority, near-frontier",
      "commander_nearfrontier": "commander, near-frontier"}
for a, b, k in T2:
    d = leaf(MA, ["primary_family", k])
    add(2, f"{TN[a]} minus {TN[b]}", "breach", TN[a], TN[b], FOUR, [2, 4], diff_teams(a, b, [2, 4], FOUR), d["estimate"],
        d["ci95_t_seeds"], "t", d["p_t_raw"], d["p_sign_flip_exact_conditional_on_symmetry"], d["p_t_holm"],
        "Holm, primary family (4)")
    for r in (2, 4):
        d = leaf(MA, ["contrasts", f"{k} | r{r}", "breach_difference"])
        add(2, f"{TN[a]} minus {TN[b]}", "breach", TN[a], TN[b], FOUR, [r], diff_teams(a, b, [r], FOUR), *stored_t(d))
    for r in (2, 4):
        for c in FOUR:
            d = leaf(MA, ["contrasts", f"{k} | r{r} | {c}", "breach_difference"])
            ci = d["ci95_t_seeds"] if isinstance(d["ci95_t_seeds"], list) else None
            add(2, f"{TN[a]} minus {TN[b]}", "breach", TN[a], TN[b], [c], [r], diff_teams(a, b, [r], [c]),
                d["estimate"], ci, "t" if ci else "not estimable", d.get("p_t_raw"), d["p_sign_flip"])

# ---- Supplementary Table 4: condition effects
EFF = [("LI", "D2 LI - REF"), ("LB", "D3 LB - REF"), ("ATK", "F1 REF+attack - REF"), ("MH", "D1 MH - REF")]
for c, k in EFF:
    if c == "ATK":
        d = leaf(MA, ["primary_family", k])
        vals = [np.mean([C[(t, r, c, s)][0] - C[(t, r, "REF", s)][0] for t in ("commander", "cbaa", "independent")
                         for r in (2, 4)]) for s in SEEDS]
        add(4, f"{NAME[c]} minus ideal network, all teams", "breach", NAME[c], "ideal network", "all three teams", [2, 4],
            vals, d["estimate"], d["ci95_t_seeds"], "t", d["p_t_raw"], d["p_sign_flip_exact_conditional_on_symmetry"],
            d["p_t_holm"], "Holm, primary family (4)")
    else:
        d = leaf(MA, ["contrasts", f"{k} | pooled summary", "breach_difference"])
        vals = [np.mean([C[(t, r, c, s)][0] - C[(t, r, "REF", s)][0] for t in ("commander", "cbaa", "independent")
                         for r in (2, 4)]) for s in SEEDS]
        add(4, f"{NAME[c]} minus ideal network, all teams", "breach", NAME[c], "ideal network", "all three teams", [2, 4],
            vals, *stored_t(d))
    for t in ("commander", "cbaa", "independent"):
        for reds, sk in (([2], "r2"), ([4], "r4"), ([2, 4], "pooled over sizes")):
            d = leaf(MA, ["contrasts", f"{k} | {t} | {sk}", "breach_difference"])
            hk = {"padj": round(KH["K1"], 5), "adj": KFAM} if (c, t, len(reds)) == ("ATK", "commander", 2) else {}
            add(4, f"{NAME[c]} minus ideal network, {TN[t]} team", "breach", NAME[c], "ideal network", TN[t], reds,
                effect(t, c, "REF", reds), *stored_t(d), **hk)
    if c in ("ATK", "MH"):
        eff = "(REF+attack - REF)" if c == "ATK" else "(MH - REF)"
        for a, b, _ in T2:
            for reds, sk in (([2], "r2"), ([4], "r4"), ([2, 4], "pooled over sizes")):
                d = leaf(MA, ["interactions_exploratory", f"{eff} effect: {a} - {b} | {sk}", "breach"])
                vals = [x - y for x, y in zip(effect(a, c, "REF", reds), effect(b, c, "REF", reds))]
                hk = {"padj": round(KH["K2"], 5), "adj": KFAM} if (c, a, b, len(reds)) == ("ATK", "commander", "cbaa", 2) else {}
                add(4, f"change under {NAME[c]}, {TN[a]} team minus {TN[b]} team", "breach", TN[a], TN[b], f"{NAME[c]} minus ideal network",
                    reds, vals, *stored_t(d), **hk)
# Labels that name a network condition say "team" after each team name, so "independent loss" and the
# independent team cannot be read as one another.
LBLI = "lb_minus_li.json"
for t in ("commander", "cbaa", "independent"):
    d = leaf(LBLI, ["arms", t, "LB - LI"])
    add(4, f"burst loss minus independent loss, {TN[t]} team", "breach", "burst loss", "independent loss", TN[t], [2, 4],
        effect(t, "LB", "LI", [2, 4]), d["estimate"], d["ci95_t_seeds"], "t", None, None)
for a, b in (("commander", "cbaa"), ("commander", "independent"), ("independent", "cbaa")):
    d = leaf(LBLI, ["interactions", f"(LB - LI): {a} - {b}"])
    vals = [x - y for x, y in zip(effect(a, "LB", "LI", [2, 4]), effect(b, "LB", "LI", [2, 4]))]
    add(4, f"change from independent loss to burst loss, {TN[a]} team minus {TN[b]} team", "breach", TN[a], TN[b],
        "burst loss minus independent loss", [2, 4], vals, d["estimate"], d["ci95_t_seeds"], "t", None, None)

# ---- Supplementary Table 8: authority and model capability
SW = "high_authority.json"
d = leaf(SW, ["family_N_holm", "N2 W - B"])
add(8, "high minus low authority, consensus, low-cost", "breach", TN["bidweight"], TN["cbaa"], FOUR, [2, 4],
    diff_teams("bidweight", "cbaa", [2, 4], FOUR), d["estimate"], d["ci95_t_seeds"], "t", d["p_t_raw"], d["p_sign_flip"],
    round(NH["N2"], 5), NFAM)
for r in (2, 4):
    d = leaf(SW, ["rows", f"N2 W - B | r{r}", "breach_difference"])
    add(8, "high minus low authority, consensus, low-cost", "breach", TN["bidweight"], TN["cbaa"], FOUR, [r],
        diff_teams("bidweight", "cbaa", [r], FOUR), *stored_t(d))
d = leaf(SW, ["family_N_holm", "N5 attack effect W - B"])
vals = [x - y for x, y in zip(effect("bidweight", "ATK", "REF", [2, 4]), effect("cbaa", "ATK", "REF", [2, 4]))]
add(8, "change under falsified reports, high minus low authority, consensus", "breach", TN["bidweight"], TN["cbaa"],
    "falsified reports minus ideal network", [2, 4], vals, d["estimate"], d["ci95_t_seeds"], "t", d["p_t_raw"],
    d["p_sign_flip"], round(NH["N5"], 5), NFAM)
for r in (2, 4):
    d = leaf(SW, ["rows", f"N5 attack effect W - B | r{r}", "breach_difference"])
    vals = [x - y for x, y in zip(effect("bidweight", "ATK", "REF", [r]), effect("cbaa", "ATK", "REF", [r]))]
    add(8, "change under falsified reports, high minus low authority, consensus", "breach", TN["bidweight"], TN["cbaa"],
        "falsified reports minus ideal network", [r], vals, *stored_t(d))
for other, k in (("commander", "S W - commander"), ("independent", "S W - independent")):
    d = leaf(SW, ["rows", k, "breach_difference"])
    add(8, f"consensus at high authority minus {TN[other]}, low-cost", "breach", TN["bidweight"], TN[other], FOUR, [2, 4],
        diff_teams("bidweight", other, [2, 4], FOUR), *stored_t(d))
SOLW = "high_authority_near_frontier.json"
for other, k in (("bidweight", "S1 W-Sol - W"), ("cbaa", "secondary W-Sol - B"), ("commander_nearfrontier", "secondary W-Sol - Sol commander")):
    d = leaf(SOLW, ["rows", k])
    add(8, f"consensus at high authority, near-frontier, minus {TN[other]}", "breach", TN["bidweight_nearfrontier"], TN[other], ["REF"], [4],
        diff_teams("bidweight_nearfrontier", other, [4], ["REF"]), d["estimate"], d["newcombe10"], "Newcombe method 10",
        None, d["exact_mcnemar_p"])
C2 = "summary.json"
for c, k in (("REF", "REF"), ("ATK", "REF+attack")):
    d = leaf(C2, ["near_frontier_commander", f"3_sol_minus_gemini|{k}"])
    add(8, "near-frontier minus low-cost commander", "breach", TN["commander_nearfrontier"], TN["commander"], [c], [2, 4],
        diff_teams("commander_nearfrontier", "commander", [2, 4], [c]), *stored_t(d))
    for r in (2, 4):
        d = leaf(C2, ["near_frontier_commander", f"3_sol_minus_gemini|{k}|r{r}"])
        add(8, "near-frontier minus low-cost commander", "breach", TN["commander_nearfrontier"], TN["commander"], [c], [r],
            diff_teams("commander_nearfrontier", "commander", [r], [c]), d["diff"], d["newcombe10"], "Newcombe method 10",
            None, d["mcnemar_exact_p"])
for reds, k in (([2, 4], "2_attack_effect_sol_minus_gemini"), ([2], "2_attack_effect_sol_minus_gemini_r2"),
                ([4], "2_attack_effect_sol_minus_gemini_r4")):
    d = leaf(C2, ["near_frontier_commander", k])
    vals = [x - y for x, y in zip(effect("commander_nearfrontier", "ATK", "REF", reds), effect("commander", "ATK", "REF", reds))]
    add(8, "change under falsified reports, near-frontier minus low-cost commander", "breach", TN["commander_nearfrontier"],
        TN["commander"], "falsified reports minus ideal network", reds, vals, *stored_t(d),
        **({"padj": round(KH["K4"], 5), "adj": KFAM} if len(reds) == 2 else {}))

with open(OUT, "w", newline="") as fh:
    # the SI reports no P values; they are still computed above so the stored-estimate checks run
    rows = [{k: v for k, v in r.items() if k not in ('p_t', 'p_sign_flip_or_mcnemar', 'p_adjusted', 'adjustment')} for r in rows]
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} comparisons -> {OUT}; every recomputed per-seed mean equals its stored estimate")
