"""Seed-clustered statistics of the main study.

Every contrast is reduced to one number per seed (the mean over that seed's matched pairs of first minus
second); the 30 seeds are the sample. Per contrast the output reports the estimate, the seed-level t interval
and t p-value, the seed-bootstrap interval, the sign-flip p-value (exact by convolution over the integer
values, Monte Carlo otherwise), Holm adjustment over the primary family, and for cells with one pair per seed
Newcombe's method-10 interval and the exact McNemar p. Interactions are exploratory.

    python3 -m analysis.matrix_stats --data DATA [--out OUT]       ->  OUT/matrix_analysis.json
"""
import itertools
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest
from scipy.stats import t as student_t

from analysis import common
from analysis.common import ARMS, BASE, CONDS, REDS, cond

SEEDS = common.SEEDS
BOOT, BOOT_SEED, Z = 10_000, 20261004, 1.959963984540054


def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = p + Z * Z / (2 * n)
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (float((c - h) / d), float((c + h) / d))


def newcombe10(e, f, g, h):
    """Newcombe (1998) method 10 for the paired difference (f - g) / n; e both,
    f first only, g second only, h neither."""
    n = e + f + g + h
    p1, p2 = (e + f) / n, (e + g) / n
    l1, u1 = wilson(e + f, n)
    l2, u2 = wilson(e + g, n)
    den = (e + f) * (g + h) * (e + g) * (f + h)
    num = e * h - f * g
    if num > 0:
        num = max(num - n / 2, 0.0)
    phi = 0.0 if den == 0 else num / np.sqrt(den)
    dl = np.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = np.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return (float(p1 - p2 - dl), float(p1 - p2 + du))


_PATTERNS = {}


def sign_flip_exact(values):
    """Two-sided exact sign-flip p over all 2^k sign patterns (identity
    included): p = #{|mean(e*d)| >= |mean(d)|} / 2^k. Valid in finite samples
    only if each seed difference is symmetric about 0 under H0. p = 1 when
    every d_s = 0."""
    v = np.asarray(values, dtype=float)
    if np.allclose(v, 0):
        return 1.0
    k = v.size
    if k not in _PATTERNS:
        _PATTERNS[k] = np.array(list(itertools.product((1.0, -1.0), repeat=k)))
    t0 = abs(v.mean())
    tb = np.abs(_PATTERNS[k] @ v) / k
    return float(np.count_nonzero(tb >= t0 - 1e-12) / len(tb))


MC_B, MC_SEED = 10 ** 6, 20261006


def sign_flip_exact_conv(values, scale=24, tol=1e-6):
    """The same exact two-sided sign-flip p as sign_flip_exact
    (all 2^k patterns, identity included, p = #{|sum(e*d)| >= |sum(d)|} / 2^k),
    computed by convolution over the integers n_s = scale * d_s instead of a
    dense 2^k x k pattern array, so k = 30 is feasible. Breach rows: scale 24
    (pairs per seed 8, 6, 4, 2 or 1); breach-free durations (0.1 s steps):
    scale 240. Raises ValueError if some scale * d_s is not an integer (within
    ``tol``): such rows are continuous and go to sign_flip_mc."""
    v = np.asarray(values, dtype=float)
    x = v * scale
    n = np.rint(x)
    if np.max(np.abs(x - n)) > tol:
        raise ValueError(f"values are not integers at scale {scale}")
    n = n.astype(np.int64)
    k = n.size
    if not n.any():
        return 1.0
    if k > 62:
        raise ValueError("k > 62 overflows the int64 counts")
    a = np.abs(n)
    total = int(a.sum())
    dist = np.zeros(2 * total + 1, dtype=np.int64)  # dist[j]: patterns with sum j - total
    dist[total] = 1
    for m in a:
        if m == 0:
            dist = 2 * dist
            continue
        new = np.zeros_like(dist)
        new[m:] += dist[:-m]
        new[:-m] += dist[m:]
        dist = new
    t0 = abs(int(n.sum()))
    sums = np.abs(np.arange(-total, total + 1))
    return float(int(dist[sums >= t0].sum()) / 2 ** k)


def sign_flip_mc(values, B=MC_B, seed=MC_SEED, chunk=50_000):
    """Continuous rows: Monte Carlo sign-flip with B random
    sign vectors (generator seed 20261006), p = (1 + #{|T*| >= |T|}) / (1 + B).
    Returns (p, Monte Carlo SE)."""
    v = np.asarray(values, dtype=float)
    if np.allclose(v, 0):
        return 1.0, 0.0
    t0 = abs(v.sum())
    eps = 1e-12 * max(1.0, float(np.abs(v).sum()))
    rng = np.random.default_rng(seed)
    hits, done = 0, 0
    while done < B:
        m = min(chunk, B - done)
        e = rng.integers(0, 2, size=(m, v.size), dtype=np.int8) * 2 - 1
        hits += int(np.count_nonzero(np.abs(e @ v) >= t0 - eps))
        done += m
    p = (1 + hits) / (1 + B)
    return float(p), float(np.sqrt(p * (1 - p) / B))


def sign_flip_p(values, scales=(24, 240)):
    """Dispatcher: exact convolution at the first scale that makes every seed
    value an integer, otherwise Monte Carlo. Returns {"p", "method", ...}."""
    for s in scales:
        try:
            return {"p": sign_flip_exact_conv(values, scale=s), "method": f"exact_conv_scale{s}"}
        except ValueError:
            pass
    p, se = sign_flip_mc(values)
    return {"p": p, "method": f"monte_carlo_B{MC_B}_seed{MC_SEED}", "mc_se": se}


def t_test_p(values):
    """Two-sided one-sample t p for mean 0 on the seed values; None when the
    seed values do not vary (not estimable)."""
    v = np.asarray(values, dtype=float)
    if np.allclose(v, v[0]):
        return None
    t = v.mean() / (v.std(ddof=1) / np.sqrt(v.size))
    return float(2 * student_t.sf(abs(t), v.size - 1))


def holm(ps):
    """Holm-adjusted p-values (None stays None, counted in m)."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: (ps[i] is None, ps[i] if ps[i] is not None else 2))
    adj, run = [None] * m, 0.0
    for r, i in enumerate(order):
        if ps[i] is None:
            continue
        run = max(run, min(1.0, (m - r) * ps[i]))
        adj[i] = run
    return adj


def cluster_interval(values):
    """t (headline, approximate) and seed-bootstrap (sensitivity) 95 % intervals
    of the mean, the raw t p, and the exact sign-flip p (sensitivity,
    conditional on symmetry)."""
    v = np.asarray(values, dtype=float)
    k = len(v)
    sf = sign_flip_p(v)
    out = {"n_seeds": k, "estimate": round(float(v.mean()), 4),
           "p_t_raw": t_test_p(v), "p_sign_flip": sf["p"], "sign_flip_method": sf["method"],
           "seeds_positive": int((v > 1e-12).sum()), "seeds_negative": int((v < -1e-12).sum()),
           "seeds_zero": int((np.abs(v) <= 1e-12).sum())}
    if out["p_t_raw"] is not None:
        out["p_t_raw"] = round(out["p_t_raw"], 5)
    out["p_sign_flip"] = round(out["p_sign_flip"], 5)
    if "mc_se" in sf:
        out["p_sign_flip_mc_se"] = round(sf["mc_se"], 6)
    else:
        out["p_sign_flip_exact"] = out["p_sign_flip"]
    if out["p_t_raw"] is not None:
        out["t_vs_sign_flip_disagree_at_0.05"] = (out["p_t_raw"] <= 0.05) != (out["p_sign_flip"] <= 0.05)
    if np.allclose(v, v[0]):
        out["ci95_t_seeds"] = out["ci95_seed_bootstrap"] = "not estimable (no between-seed variation)"
        return out
    half = student_t.ppf(0.975, k - 1) * v.std(ddof=1) / np.sqrt(k)
    rng = np.random.default_rng(BOOT_SEED)
    boots = v[rng.integers(0, k, size=(BOOT, k))].mean(axis=1)
    out["ci95_t_seeds"] = [round(float(v.mean() - half), 4), round(float(v.mean() + half), 4)]
    out["ci95_seed_bootstrap"] = [round(float(np.percentile(boots, 2.5)), 4),
                                  round(float(np.percentile(boots, 97.5)), 4)]
    return out


def main(data, out_path=None, seeds=None):
    global SEEDS
    if seeds is not None:
        SEEDS = tuple(seeds)
    out_path = out_path or os.path.join(data, "analysis", "matrix_analysis.json")
    recs = common.records(data, common.MAIN)
    by = {}
    for r in recs:
        k = (r["arm"], r["reds"], cond(r), r["seed"])
        assert k not in by, k
        by[k] = r
    assert len(by) == len(ARMS) * len(REDS) * len(CONDS) * 30, len(by)
    assert sorted({k[3] for k in by}) == list(common.SEEDS)
    recs = [r for r in recs if r["seed"] in SEEDS]
    by = {k: v for k, v in by.items() if k[3] in SEEDS}
    assert len(by) == len(ARMS) * len(REDS) * len(CONDS) * len(SEEDS)

    def breach(k):
        return int(bool(by[k]["outcome"]["touchdown"]))

    def dur(k):
        o = by[k]["outcome"]
        return o["touchdown_step"] * 0.1 if o["touchdown"] else by[k]["horizon_steps"] * 0.1

    cells = {}
    for a in ARMS:
        for n in REDS:
            for c in CONDS:
                ks = [(a, n, c, s) for s in SEEDS]
                kk = sum(breach(x) for x in ks)
                rs = [by[x] for x in ks]
                cells[f"{a}|r{n}|{c}"] = {
                    "breaches": kk, "n": len(SEEDS), "rate": kk / len(SEEDS),
                    "wilson95": [round(x, 4) for x in wilson(kk, len(SEEDS))],
                    "breach_free_mean_s": round(float(np.mean([dur(x) for x in ks])), 2),
                    "neutralized_mean": round(float(np.mean([r["outcome"]["num_neutralized"] for r in rs])), 2),
                    "escaped_mean": round(float(np.mean([r["outcome"]["num_escaped"] for r in rs])), 2),
                    "blues_disabled_mean": round(float(np.mean([r["outcome"]["n_blues_disabled"] for r in rs])), 2),
                    "blue_score_mean_bookkeeping": round(float(np.mean([r["outcome"]["blue_score"] for r in rs])), 2)}

    def contrast(pairs_by_seed, label):
        """pairs_by_seed: {seed: [(first_key, second_key), ...]} (equal sizes)."""
        per_seed, ds, dd = [], [], []
        tot = {"e": 0, "f": 0, "g": 0, "h": 0}
        for s in SEEDS:
            pr = pairs_by_seed[s]
            x = [(breach(a), breach(b)) for a, b in pr]
            e = sum(1 for p, q in x if p and q)
            f = sum(1 for p, q in x if p and not q)
            g = sum(1 for p, q in x if q and not p)
            h = len(x) - e - f - g
            for kk, v in zip("efgh", (e, f, g, h)):
                tot[kk] += v
            d = (f - g) / len(x)
            ds.append(d)
            dd.append(float(np.mean([dur(a) - dur(b) for a, b in pr])))
            per_seed.append({"seed": s, "pairs": len(x), "first_only": f, "second_only": g,
                             "d_breach": round(d, 4), "d_breach_free_s": round(dd[-1], 3)})
        n = sum(tot.values())
        pairs_per_seed = n // len(SEEDS)
        out = {"label": label, "n_pairs": n, "pairs_per_seed": pairs_per_seed,
               "first_breach_rate": round((tot["e"] + tot["f"]) / n, 4),
               "second_breach_rate": round((tot["e"] + tot["g"]) / n, 4),
               "both": tot["e"], "first_only": tot["f"], "second_only": tot["g"], "neither": tot["h"],
               "breach_difference": cluster_interval(ds),
               "breach_free_duration_s": cluster_interval(dd),
               "per_seed": per_seed}
        if pairs_per_seed == 1:
            out["breach_newcombe10_independent_pairs"] = [round(x, 4) for x in newcombe10(
                tot["e"], tot["f"], tot["g"], tot["h"])]
            nd = tot["f"] + tot["g"]
            out["breach_exact_mcnemar_p"] = 1.0 if nd == 0 else round(float(
                binomtest(tot["f"], nd, 0.5).pvalue), 5)
        return out

    def pairs(first, second, keys):
        d = {s: [] for s in SEEDS}
        for k in keys:
            for s in SEEDS:
                d[s].append((first(k) + (s,), second(k) + (s,)))
        return d

    contrasts = {}
    for cid, x, y in (("A1 commander - cbaa", "commander", "cbaa"),
                      ("A2 commander - independent", "commander", "independent"),
                      ("A3 cbaa - independent", "cbaa", "independent")):
        for n in REDS:
            for c in BASE:
                contrasts[f"{cid} | r{n} | {c}"] = contrast(
                    pairs(lambda k, x=x: (x,) + k, lambda k, y=y: (y,) + k, [(n, c)]), "cell")
            contrasts[f"{cid} | r{n}"] = contrast(
                pairs(lambda k, x=x: (x,) + k, lambda k, y=y: (y,) + k, [(n, c) for c in BASE]),
                "red size (over REF/MH/LI/LB)")
        contrasts[f"{cid} | pooled summary"] = contrast(
            pairs(lambda k, x=x: (x,) + k, lambda k, y=y: (y,) + k,
                  [(n, c) for n in REDS for c in BASE]),
            "POOLED SUMMARY over red sizes and conditions")
    for cid, c in (("D1 MH - REF", "MH"), ("D2 LI - REF", "LI"), ("D3 LB - REF", "LB"),
                   ("F1 REF+attack - REF", "REF+attack")):
        first = lambda k, c=c: (k[0], k[1], c)  # noqa: E731
        second = lambda k: (k[0], k[1], "REF")  # noqa: E731
        for a in ARMS:
            for n in REDS:
                contrasts[f"{cid} | {a} | r{n}"] = contrast(pairs(first, second, [(a, n)]), "cell")
            contrasts[f"{cid} | {a} | pooled over sizes"] = contrast(
                pairs(first, second, [(a, n) for n in REDS]), "POOLED SUMMARY over red sizes")
        contrasts[f"{cid} | pooled summary"] = contrast(
            pairs(first, second, [(a, n) for a in ARMS for n in REDS]),
            "POOLED SUMMARY over arms and red sizes")

    # Exploratory interactions: within-seed difference of two arms' condition effects.
    def effect(arm, c, sizes, s, f):
        return float(np.mean([f((arm, n, c, s)) - f((arm, n, "REF", s)) for n in sizes]))

    interactions = {}
    for c in ("REF+attack", "MH"):
        for x, y in (("commander", "cbaa"), ("commander", "independent"), ("cbaa", "independent")):
            for sizes, tag in (((2,), "r2"), ((4,), "r4"), (REDS, "pooled over sizes")):
                v = [effect(x, c, sizes, s, breach) - effect(y, c, sizes, s, breach) for s in SEEDS]
                w = [effect(x, c, sizes, s, dur) - effect(y, c, sizes, s, dur) for s in SEEDS]
                interactions[f"({c} - REF) effect: {x} - {y} | {tag}"] = {
                    "EXPLORATORY": True, "breach": cluster_interval(v),
                    "breach_free_duration_s": cluster_interval(w),
                    "per_seed_breach": [round(z, 3) for z in v]}

    # PRIMARY FAMILY (original weights): Holm over the four, on the headline
    # t p and, as sensitivity, on the exact sign-flip p.
    fam = ["A1 commander - cbaa | pooled summary", "A2 commander - independent | pooled summary",
           "A3 cbaa - independent | pooled summary", "F1 REF+attack - REF | pooled summary"]
    pt = [contrasts[k]["breach_difference"]["p_t_raw"] for k in fam]
    ps = [contrasts[k]["breach_difference"]["p_sign_flip"] for k in fam]
    primary = {}
    for k, a, b in zip(fam, holm(pt), holm(ps)):
        c = contrasts[k]
        bd = c["breach_difference"]
        primary[k.split(" |")[0]] = {
            "contrast_key": k, "pairs_per_seed": c["pairs_per_seed"], "n_pairs": c["n_pairs"],
            "first_breach_rate": c["first_breach_rate"], "second_breach_rate": c["second_breach_rate"],
            "first_only": c["first_only"], "second_only": c["second_only"],
            "estimate": bd["estimate"], "ci95_t_seeds": bd["ci95_t_seeds"],
            "p_t_raw": bd["p_t_raw"], "p_t_holm": None if a is None else round(a, 5),
            "ci95_seed_bootstrap": bd["ci95_seed_bootstrap"],
            "p_sign_flip_exact_conditional_on_symmetry": bd["p_sign_flip"],
            "sign_flip_method": bd["sign_flip_method"],
            "p_sign_flip_holm": None if b is None else round(b, 5),
            "seeds_pos_neg_zero": [bd["seeds_positive"], bd["seeds_negative"], bd["seeds_zero"]],
            "per_seed_d": [x["d_breach"] for x in c["per_seed"]]}
        for m in ("first_breach_rate", "second_breach_rate"):
            primary[k.split(" |")[0]][m] = c[m]


    validity = defaultdict(lambda: defaultdict(int))
    for r in recs:
        for k, v in (r.get("defender") or {}).get("events_by_kind", {}).items():
            validity["defender_events"][k] += v
        i = r.get("intruder") or {}
        validity["intruder"]["decisions"] += i.get("decisions", 0)
        for k, v in (i.get("invalid_by_reason") or {}).items():
            validity["intruder_invalid"][k] += v
        c = r.get("commander") or {}
        for k in ("epochs", "unusable_epochs", "epochs_with_agent_fallback"):
            validity["commander"][k] += c.get(k, 0)
        validity["episodes"]["solver_failed"] += bool(r.get("solver_failed"))
    out = {"note": ("Seed-clustered analysis of the main study (30 seeds, 4001-4030); sign-flip by "
                    "sign_flip_p (exact convolution / Monte Carlo). Primary family A1/A2/A3/F1: headline "
                    "seed-level t (approximate, mean contrast) with raw and Holm p; seed bootstrap and "
                    "exact sign-flip (conditional on symmetry) as sensitivity. All other rows unadjusted; "
                    "interactions exploratory."),
           "n_episodes": len(recs), "primary_family": primary, "cells": cells, "contrasts": contrasts,
           "interactions_exploratory": interactions,
           "validity": {k: dict(v) for k, v in validity.items()},
           "cost_usd_written": round(sum(r.get("cost_usd_written") or 0 for r in recs), 4)}
    common.dump(out, out_path)
    return out


if __name__ == "__main__":
    ap = common.parser(__doc__.split("\n")[0])
    a = ap.parse_args()
    main(a.data, os.path.join(common.out_dir(a), "matrix_analysis.json"))
