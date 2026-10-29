"""The consensus team at high authority (the model writes its own bid weights) against the consensus team at
low authority, on the same seeds and conditions.

Arms: B = the main study's consensus team (low authority), W = the consensus team at high authority (low-cost
model), with the commander and independent teams of the main study for reference. Key family: N2 (W - B over
the ideal network, limited range, independent loss and burst loss, both intruder counts, 8 pairs per seed) and
N5 (the difference of the attack effects, [W(falsified) - W(ideal)] - [B(falsified) - B(ideal)], mean of the two
intruder counts per seed), Holm over the two on the seed-level t p-value, sign-flip Holm as sensitivity.
Secondary, unadjusted: N2 and N5 by intruder count, W - commander, W - independent, each team's attack effect
(by size with McNemar / Newcombe 10), breach-free duration in every contrast, per-cell rates.

Whole-seed rule: if any arm lacks a valid record for a cell, only the seeds complete in every arm are used.

Also: the share of active-intruder time before any defender had seen the intruder ("unseen"), B and W, and
the seed-level W - B difference.

    python3 -m analysis.high_authority --data DATA [--out OUT]      ->  OUT/high_authority.json, OUT/high_authority_unseen.json

The unseen shares read OUT/coverage_exposure_main_study.json and OUT/coverage_exposure_high_authority_low_cost.json
(analysis.coverage_replay).
"""
import json
import os

import numpy as np
from scipy.stats import binomtest

from analysis import common, matrix_stats
from analysis.common import BASE, CONDS, REDS, cond
from scripts.run_competing_teams import record_problem

ARM_OF = {"cbaa": "B", "cbaa_sw2": "W", "commander": "commander", "independent": "independent"}
ALL_SEEDS = common.SEEDS


def load(data):
    by, problems = {}, []
    for r in common.records(data, common.MAIN):
        by[(ARM_OF[r["arm"]], r["reds"], cond(r), r["seed"])] = r
    run = common.HIGH_AUTHORITY_LOW_COST[0]
    ident = common.identity(data, run)
    seen = {}
    for r in common.records(data, [run]):
        seen.setdefault(r["episode_id"], []).append(r)
    for eid, rs in seen.items():
        e = {k: rs[0].get(k) for k in ("arm", "reds", "family", "condition", "attacked", "seed",
                                        "rep", "design")}
        bad = record_problem(e, rs, ident)
        if bad:
            problems.append(["W", eid, bad])
            continue
        r = rs[0]
        by[(ARM_OF[r["arm"]], r["reds"], cond(r), r["seed"])] = r
    return by, problems


def analyse(by, arms_needed, seeds):
    def breach(k):
        return int(bool(by[k]["outcome"]["touchdown"]))

    def dur(k):
        o = by[k]["outcome"]
        return o["touchdown_step"] * 0.1 if o["touchdown"] else by[k]["horizon_steps"] * 0.1

    def contrast(pairs_by_seed, label):
        ds, dd, per_seed = [], [], []
        tot = dict(e=0, f=0, g=0, h=0)
        for s in seeds:
            x = [(breach(a), breach(b)) for a, b in pairs_by_seed[s]]
            e = sum(p and q for p, q in x)
            f = sum(p and not q for p, q in x)
            g = sum(q and not p for p, q in x)
            h = len(x) - e - f - g
            for kk, v in zip("efgh", (e, f, g, h)):
                tot[kk] += v
            ds.append((f - g) / len(x))
            dd.append(float(np.mean([dur(a) - dur(b) for a, b in pairs_by_seed[s]])))
            per_seed.append({"seed": s, "pairs": len(x), "first_only": f, "second_only": g,
                             "d_breach": round(ds[-1], 4)})
        n = sum(tot.values())
        out = {"label": label, "n_pairs": n, "pairs_per_seed": n // len(seeds),
               "first_breach_rate": round((tot["e"] + tot["f"]) / n, 4),
               "second_breach_rate": round((tot["e"] + tot["g"]) / n, 4),
               "first_only": tot["f"], "second_only": tot["g"],
               "breach_difference": matrix_stats.cluster_interval(ds),
               "breach_free_duration_s": matrix_stats.cluster_interval(dd), "per_seed": per_seed}
        if n // len(seeds) == 1:
            out["breach_newcombe10_independent_pairs"] = [round(z, 4) for z in matrix_stats.newcombe10(
                tot["e"], tot["f"], tot["g"], tot["h"])]
            nd = tot["f"] + tot["g"]
            out["breach_exact_mcnemar_p"] = 1.0 if nd == 0 else round(float(binomtest(tot["f"], nd, 0.5).pvalue), 5)
        return out

    def between(x, y, cells):
        return {s: [((x, n, c, s), (y, n, c, s)) for n, c in cells] for s in seeds}

    def attack(arm, sizes):
        return {s: [((arm, n, "REF+attack", s), (arm, n, "REF", s)) for n in sizes] for s in seeds}

    def attack_diff(x, y, sizes):
        v = [float(np.mean([(breach((x, n, "REF+attack", s)) - breach((x, n, "REF", s)))
                            - (breach((y, n, "REF+attack", s)) - breach((y, n, "REF", s))) for n in sizes]))
             for s in seeds]
        w = [float(np.mean([(dur((x, n, "REF+attack", s)) - dur((x, n, "REF", s)))
                            - (dur((y, n, "REF+attack", s)) - dur((y, n, "REF", s))) for n in sizes]))
             for s in seeds]
        return {"label": "attack effect difference", "breach_difference": matrix_stats.cluster_interval(v),
                "breach_free_duration_s": matrix_stats.cluster_interval(w), "per_seed_d": [round(z, 4) for z in v]}

    pooled = [(n, c) for n in REDS for c in BASE]
    rows = {}
    for rid, x, y in (("N1 B - R", "B", "R"), ("N2 W - B", "W", "B"), ("N3 W - R", "W", "R"),
                      ("N4 W - H", "W", "H")):
        if x in arms_needed and y in arms_needed:
            rows[rid] = contrast(between(x, y, pooled), "REF/MH/LI/LB x both sizes")
            for n in REDS:
                rows[f"{rid} | r{n}"] = contrast(between(x, y, [(n, c) for c in BASE]), f"r{n} over REF/MH/LI/LB")
    if {"W", "B"} <= set(arms_needed):
        rows["N5 attack effect W - B"] = attack_diff("W", "B", REDS)
        for n in REDS:
            rows[f"N5 attack effect W - B | r{n}"] = attack_diff("W", "B", (n,))
    for rid, x, y in (("S H - R", "H", "R"), ("S H - B", "H", "B"), ("S W - commander", "W", "commander"),
                      ("S W - independent", "W", "independent")):
        if x in arms_needed and y in arms_needed:
            rows[rid] = contrast(between(x, y, pooled), "REF/MH/LI/LB x both sizes")
    for arm in [a for a in ("R", "W", "H", "B") if a in arms_needed]:
        rows[f"S attack effect {arm}"] = contrast(attack(arm, REDS), "pooled over sizes")
        for n in REDS:
            rows[f"S attack effect {arm} | r{n}"] = contrast(attack(arm, (n,)), f"r{n}")
    fam = [k for k in ("N1 B - R", "N2 W - B", "N3 W - R", "N4 W - H", "N5 attack effect W - B") if k in rows]
    pt = [rows[k]["breach_difference"]["p_t_raw"] for k in fam]
    ps = [rows[k]["breach_difference"]["p_sign_flip"] for k in fam]
    family = {}
    for k, a, b in zip(fam, matrix_stats.holm(pt), matrix_stats.holm(ps)):
        bd = rows[k]["breach_difference"]
        family[k] = {"estimate": bd["estimate"], "ci95_t_seeds": bd["ci95_t_seeds"], "p_t_raw": bd["p_t_raw"],
                     "p_t_holm": None if a is None else round(a, 5),
                     "ci95_seed_bootstrap": bd["ci95_seed_bootstrap"], "p_sign_flip": bd["p_sign_flip"],
                     "p_sign_flip_holm": None if b is None else round(b, 5),
                     "seeds_pos_neg_zero": [bd["seeds_positive"], bd["seeds_negative"], bd["seeds_zero"]]}
    cells = {}
    for arm in arms_needed:
        for n in REDS:
            for c in CONDS:
                ks = [(arm, n, c, s) for s in seeds]
                rs = [by[k] for k in ks]
                kk = sum(breach(k) for k in ks)
                cells[f"{arm}|r{n}|{c}"] = {
                    "breaches": kk, "n": len(seeds), "rate": round(kk / len(seeds), 4),
                    "wilson95": [round(z, 4) for z in matrix_stats.wilson(kk, len(seeds))],
                    "breach_free_mean_s": round(float(np.mean([dur(k) for k in ks])), 2),
                    "neutralized_mean": round(float(np.mean([r["outcome"]["num_neutralized"] for r in rs])), 2),
                    "escaped_mean": round(float(np.mean([r["outcome"]["num_escaped"] for r in rs])), 2),
                    "blues_disabled_mean": round(float(np.mean([r["outcome"]["n_blues_disabled"] for r in rs])), 2)}
    return {"family_N_holm": family, "rows": rows, "cells": cells}


def unseen(out_dir):
    """Share of active-intruder time before any defender had seen the intruder (sum of unseen over sum of
    active, ideal network, limited range, independent loss and burst loss, both intruder counts)."""
    arms = {"B": [], "W": []}
    d = json.load(open(os.path.join(out_dir, "coverage_exposure_main_study.json")))
    arms["B"] += [e for e in d["episodes"].values() if e["arm"] == "cbaa"]
    d = json.load(open(os.path.join(out_dir, "coverage_exposure_high_authority_low_cost.json")))
    arms["W"] += list(d["episodes"].values())
    out = {"definition": "sum(unseen)/sum(active) over REF/MH/LI/LB x both sizes"}
    per = {}
    for a, eps in arms.items():
        sel = [e for e in eps if not e["attacked"] and e["condition"] in BASE and 4001 <= e["seed"] <= 4030]
        out[f"share_unseen_{a}"] = round(sum(e["coverage"]["unseen"] for e in sel)
                                         / sum(e["coverage"]["active"] for e in sel), 4)
        out[f"episodes_{a}"] = len(sel)
        per[a] = []
        for s in ALL_SEEDS:
            ss = [e for e in sel if e["seed"] == s]
            per[a].append(sum(e["coverage"]["unseen"] for e in ss) / sum(e["coverage"]["active"] for e in ss))
    out["unseen W - B"] = matrix_stats.cluster_interval([p - q for p, q in zip(per["W"], per["B"])])
    return out


def main(data, out_dir):
    by, problems = load(data)
    arms = ["B", "W", "commander", "independent"]
    complete = [s for s in ALL_SEEDS if all((a, n, c, s) in by for a in arms for n in REDS for c in CONDS)]
    res = analyse(by, arms, tuple(complete))
    res.update({"seeds_used": complete, "n_seeds": len(complete), "whole_seed_rule_applied": len(complete) < 30,
                "invalid_records": problems})
    common.dump(res, os.path.join(out_dir, "high_authority.json"))
    path = os.path.join(out_dir, "coverage_exposure_high_authority_low_cost.json")
    if os.path.exists(path):
        common.dump(unseen(out_dir), os.path.join(out_dir, "high_authority_unseen.json"))
    return res


if __name__ == "__main__":
    a = common.parser(__doc__.split("\n")[0]).parse_args()
    main(a.data, common.out_dir(a))
