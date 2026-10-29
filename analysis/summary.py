"""Commander-team contrasts with the near-frontier model, and the coverage and attack-exposure summaries of the
main study.

  * near-frontier commander against the low-cost commander on the ideal network and with falsified reports (the
    attack effect, and the difference of attack effects), one pair per seed by intruder count;
    commander-call cost and latency from the cached answers;
  * coverage: the share of known intruders that no defender pursued, and the share of active-intruder time
    before any defender had seen the intruder, per team and condition set, with seed-level contrasts between
    teams and between the ideal network and falsified reports;
  * attack exposure: in how many episodes with falsified reports altered records were stored and accepted
    (the main study and the near-frontier commander study).

    python3 -m analysis.summary --data DATA [--out OUT]       ->  OUT/summary.json

Reads OUT/coverage_exposure_main_study.json and OUT/coverage_exposure_near_frontier_commander.json
(analysis.coverage_replay).
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

from analysis import common
from analysis.common import ARMS, cond
from analysis.matrix_stats import cluster_interval, newcombe10

ALL_SEEDS = common.SEEDS
SEEDS = ALL_SEEDS
DATA = None
OUT = None
SRC = {"main": common.MAIN, "near_frontier_commander": common.NEAR_FRONTIER_COMMANDER}


def pooled(which, arm_key=False, arm=None):
    """One record per key over both seed ranges (asserted), restricted to SEEDS."""
    by = {}
    for r in common.records(DATA, SRC[which]):
        if arm is not None and r["arm"] != arm:
            continue
        k = ((r["arm"],) if arm_key else ()) + (r["reds"], cond(r), r["seed"])
        assert k not in by, k
        by[k] = r
    assert {k[-1] for k in by} == set(ALL_SEEDS), which
    return {k: v for k, v in by.items() if k[-1] in SEEDS}


def br(r):
    return int(bool(r["outcome"]["touchdown"]))


def mcnemar_exact(f, g):
    return 1.0 if f + g == 0 else float(binomtest(f, f + g, 0.5).pvalue)


def one_pair(first, second):
    """first/second: per-seed binary lists (one pair per seed): McNemar + Newcombe 10."""
    a = list(zip(first, second))
    e = sum(1 for x, y in a if x and y)
    f = sum(1 for x, y in a if x and not y)
    g = sum(1 for x, y in a if y and not x)
    h = len(a) - e - f - g
    return {"diff": (f - g) / len(a), "first_only": f, "second_only": g, "both": e, "neither": h,
            "newcombe10": newcombe10(e, f, g, h), "mcnemar_exact_p": mcnemar_exact(f, g)}


def near_frontier_commander():
    sc = pooled("near_frontier_commander")
    lc = pooled("main", arm="commander")
    out = {"tier": "std", "n_records": len(sc), "cells": {}}
    for reds in (2, 4):
        for c in ("REF", "REF+attack"):
            ks = [(reds, c, s) for s in SEEDS if (reds, c, s) in sc]
            out["cells"][f"solcmd|r{reds}|{c}"] = sum(br(sc[k]) for k in ks)
    assert len(sc) == 4 * len(SEEDS)

    def eff(rec, reds, s):
        return br(rec[(reds, "REF+attack", s)]) - br(rec[(reds, "REF", s)])

    per = [np.mean([eff(sc, r, s) for r in (2, 4)]) for s in SEEDS]
    out["1_sol_attack_effect"] = dict(cluster_interval(per), per_seed=per)
    for reds in (2, 4):  # v4 by-size addition
        out[f"1_sol_attack_effect_r{reds}"] = one_pair([br(sc[(reds, "REF+attack", s)]) for s in SEEDS],
                                                       [br(sc[(reds, "REF", s)]) for s in SEEDS])
    for name, other in (("gemini", lc),):
        per = [np.mean([eff(sc, r, s) - eff(other, r, s) for r in (2, 4)]) for s in SEEDS]
        out[f"2_attack_effect_sol_minus_{name}"] = dict(cluster_interval(per), per_seed=per)
        for reds in (2, 4):  # v4 by-size addition
            per = [float(eff(sc, reds, s) - eff(other, reds, s)) for s in SEEDS]
            out[f"2_attack_effect_sol_minus_{name}_r{reds}"] = dict(cluster_interval(per), per_seed=per)
        for c in ("REF", "REF+attack"):
            per = [np.mean([br(sc[(r, c, s)]) - br(other[(r, c, s)]) for r in (2, 4)]) for s in SEEDS]
            out[f"3_sol_minus_{name}|{c}"] = dict(cluster_interval(per), per_seed=per)
            for reds in (2, 4):  # v4 by-size addition
                out[f"3_sol_minus_{name}|{c}|r{reds}"] = one_pair([br(sc[(reds, c, s)]) for s in SEEDS],
                                                                  [br(other[(reds, c, s)]) for s in SEEDS])
    out["commander_calls"] = sum((r.get("commander") or {}).get("calls", 0) for r in sc.values())
    seedset = {f"s{s}" for s in SEEDS}

    def tally(patterns, keep):
        cost, lat, n = 0.0, [], 0
        for pattern in patterns:
            for f in sorted(glob.glob(pattern, recursive=True)):
                if not keep(f) or not any(f"_{x}_" in f for x in seedset):
                    continue
                rec = json.load(open(f))
                prov = rec.get("provenance") or {}
                if not str(prov.get("prompt_builder_version", "")).startswith("commander"):
                    continue
                n += 1
                cost += float((rec.get("usage") or {}).get("cost_usd") or 0)
                lat.append(float(prov.get("latency_s") or 0))
        return {"calls": n, "cost_usd": cost, "latency_median_s": float(np.median(lat)) if lat else None}
    out["cmd_cost_sol"] = tally([os.path.join(common.run_dir(DATA, run), "cache/**/*.json")
                                 for run in common.NEAR_FRONTIER_COMMANDER], lambda f: "__near_frontier" in f)
    out["cmd_cost_flashlite"] = tally(
        [os.path.join(common.run_dir(DATA, run), "cache/**/*.json") for run in common.MAIN],
        lambda f: re.search(r"/commander_r\d_multi_bearing_(REF|REF_attack)_s\d+_", f) is not None)
    out["unusable_epochs"] = sum((r.get("commander") or {}).get("unusable_epochs", 0) for r in sc.values())
    return out


def cov_episodes(study="main_study"):
    d = json.load(open(os.path.join(OUT, f"coverage_exposure_{study}.json")))
    assert d["outcome_mismatches"] == [], d["outcome_mismatches"]
    eps = list(d["episodes"].values())
    assert {e["seed"] for e in eps} == set(ALL_SEEDS)
    return d["n"], [e for e in eps if e["seed"] in SEEDS]


def coverage():
    out = {}
    for pop in ("llm",):
        n, eps = cov_episodes()
        out[pop] = {"n": len(eps), "outcome_mismatches": []}
        for a in ARMS:
            for reds in (2, 4, "both"):
                for cset_name, cset in (("base", ("REF", "MH", "LI", "LB")), ("REF", ("REF",)),
                                        ("MH", ("MH",)), ("LI", ("LI",)), ("LB", ("LB",)),
                                        ("attack", ("REF+attack",))):
                    sel = [e for e in eps if e["arm"] == a and (reds == "both" or e["reds"] == reds)
                           and cond(e) in cset]
                    if not sel:
                        continue
                    tot = defaultdict(int)
                    for e in sel:
                        for k, v in e["coverage"].items():
                            tot[k] += v
                    per_seed_w, per_seed_ep = [], []
                    for s in SEEDS:
                        ss = [e for e in sel if e["seed"] == s]
                        kt = sum(e["coverage"].get("known_team", 0) for e in ss)
                        un = sum(e["coverage"].get("known_team_no_exec", 0) for e in ss)
                        per_seed_w.append(un / kt if kt else None)
                        sh = [e["coverage"]["known_team_no_exec"] / e["coverage"]["known_team"]
                              for e in ss if e["coverage"].get("known_team")]
                        per_seed_ep.append(float(np.mean(sh)) if sh else None)
                    out[pop][f"{a}|{reds}|{cset_name}"] = {
                        "episodes": len(sel), "undefined_episodes": sum(1 for e in sel if not e["coverage"].get("known_team")),
                        "totals": dict(tot),
                        "share_uncovered_exec": tot["known_team_no_exec"] / tot["known_team"] if tot["known_team"] else None,
                        "share_uncovered_assigned": tot["known_team_no_assigned"] / tot["known_team"] if tot["known_team"] else None,
                        "share_assigned_not_exec": tot["known_team_assigned_no_exec"] / tot["known_team"] if tot["known_team"] else None,
                        "share_unseen_of_active": tot["unseen"] / tot["active"] if tot["active"] else None,
                        "per_seed_weighted": per_seed_w, "per_seed_episode_mean": per_seed_ep}
    return out


def _diff(x, y):
    return [p - q for p, q in zip(x, y) if p is not None and q is not None]


def coverage_contrasts(cov):
    out = {}
    c = cov["llm"]
    pop = "llm"
    for lab, a, b in (("A1 commander - cbaa", "commander", "cbaa"),
                      ("A2 commander - independent", "commander", "independent"),
                      ("A3 cbaa - independent", "cbaa", "independent")):
        for reds in (2, 4, "both"):
            for w in ("per_seed_weighted", "per_seed_episode_mean"):
                d = _diff(c[f"{a}|{reds}|base"][w], c[f"{b}|{reds}|base"][w])
                out[f"{pop}|{lab}|r{reds}|{w}"] = dict(cluster_interval(d), per_seed=d) if len(d) >= 3 else None
    for a in ARMS:
        for reds in ("both", 2, 4):
            d = _diff(c[f"{a}|{reds}|attack"]["per_seed_weighted"], c[f"{a}|{reds}|REF"]["per_seed_weighted"])
            tag = "" if reds == "both" else f"|r{reds}"
            out[f"{pop}|attack-REF|{a}|per_seed_weighted{tag}"] = dict(cluster_interval(d), per_seed=d) if len(d) >= 3 else None
    return out


def exposure():
    """Attack exposure of the episodes with falsified reports (the attack log of the re-simulation)."""
    out = {}
    for pop, study in (("llm", "main_study"), ("near_frontier", "near_frontier_commander")):
        _, eps = cov_episodes(study)
        eps = [e for e in eps if e["exposure"] is not None]
        for a in ARMS:
            sel = [e for e in eps if e["arm"] == a]
            if not sel:
                continue
            x = [e["exposure"] for e in sel]
            out[f"{pop}|{a}"] = {
                "attack_episodes": len(sel),
                "with_altered_reports": sum(1 for v in x if v["reports_altered"] > 0),
                "with_acceptance": sum(1 for v in x if v["accepted"] > 0),
                "reports_altered_median": float(np.median([v["reports_altered"] for v in x])),
                "accepted_median": float(np.median([v["accepted"] for v in x])),
                "target_selected_step_median": float(np.median([v["target_selected_step"] for v in x
                                                               if v["target_selected_step"] is not None])),
                "b0_accepted_episodes": sum(1 for v in x if v["accepted_by_b0"] > 0),
                "b0_stored_false_steps_median": float(np.median([v["stored_false_steps"].get("0", 0) for v in x])),
                "any_blue_stored_false_steps_median": float(np.median([sum(v["stored_false_steps"].values()) for v in x])),
                "episodes_with_false_commitment": sum(1 for v in x if sum(v["commitment_steps"].values()) > 0),
                "breaches": sum(1 for e in sel if e["outcome"]["touchdown"]),
                "breach_by_target_red": sum(1 for e in sel if e["outcome"]["touchdown"]
                                            and e["outcome"]["touchdown_red_index"] == e["exposure"]["target_red"]),
            }
    return out


def main(data, out_dir, seeds=ALL_SEEDS):
    global DATA, OUT, SEEDS
    DATA, OUT, SEEDS = data, out_dir, tuple(seeds)
    o = {"seeds": [SEEDS[0], SEEDS[-1], len(SEEDS)], "near_frontier_commander": near_frontier_commander()}
    cov = coverage()
    o["coverage"] = cov
    o["coverage_contrasts"] = coverage_contrasts(cov)
    o["exposure"] = exposure()
    return o


if __name__ == "__main__":
    a = common.parser(__doc__.split("\n")[0]).parse_args()
    o = common.out_dir(a)
    common.dump(main(a.data, o), os.path.join(o, "summary.json"))
