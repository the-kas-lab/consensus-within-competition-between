"""The key rows of the paper, assembled from the matrix statistics and the summary.

  K1  attack effect (falsified reports minus ideal network), commander team
  K2  attack effect, commander team minus consensus team (difference of the two effects)
  K4  attack effect, near-frontier commander minus low-cost commander
  K5  share of known intruders that no defender pursued, consensus team minus independent team
  X12 near-frontier minus low-cost commander on the ideal network
  X13 near-frontier minus low-cost commander with falsified reports

Each row holds the pooled estimate with its seed-level t interval and p-values, and the same by intruder count
(one pair per seed: difference, Newcombe interval, exact McNemar p).

    python3 -m analysis.key_rows --out OUT       ->  OUT/key_rows.json

Reads OUT/matrix_analysis.json and OUT/summary.json.
"""
import json
import os

from analysis import common


def ci_row(d, src):
    """A cluster_interval dict -> a compact row."""
    if d is None:
        return {"src": src, "missing": True}
    return {"src": src, "estimate": d["estimate"], "ci95_t": d.get("ci95_t_seeds"), "p_t": d.get("p_t_raw"),
            "p_sign_flip": d.get("p_sign_flip"), "sign_flip_method": d.get("sign_flip_method"),
            "sign_flip_mc_se": d.get("p_sign_flip_mc_se"), "boot95": d.get("ci95_seed_bootstrap"),
            "seeds_pos_neg_zero": [d["seeds_positive"], d["seeds_negative"], d["seeds_zero"]],
            "n_seeds": d["n_seeds"], "t_vs_sf_disagree_0.05": d.get("t_vs_sign_flip_disagree_at_0.05")}


def pair_row(d, src):
    """A one-pair-per-seed dict (summary one_pair or matrix cell) -> row."""
    if "newcombe10" in d:
        return {"src": src, "one_pair_per_seed": True, "diff": d["diff"], "newcombe10": d["newcombe10"],
                "mcnemar_exact_p": d["mcnemar_exact_p"], "first_only": d["first_only"],
                "second_only": d["second_only"]}
    bd = d["breach_difference"]
    return {"src": src, "one_pair_per_seed": True, "diff": bd["estimate"],
            "newcombe10": d["breach_newcombe10_independent_pairs"], "mcnemar_exact_p": d["breach_exact_mcnemar_p"],
            "first_only": d["first_only"], "second_only": d["second_only"], "t_row": ci_row(bd, src)}


def main(out_dir):
    M = json.load(open(os.path.join(out_dir, "matrix_analysis.json")))
    S = json.load(open(os.path.join(out_dir, "summary.json")))
    X = {}

    def add(i, label, pooled, sizes=None):
        X[i] = {"label": label, "pooled": pooled, "by_size": sizes or {}}

    def mrow(key):
        return ci_row(M["contrasts"][key]["breach_difference"], f"matrix.contrasts['{key}'].breach_difference")

    def irow(key):
        return ci_row(M["interactions_exploratory"][key]["breach"], f"matrix.interactions_exploratory['{key}'].breach")

    def srow(sec, key):
        return ci_row(S[sec][key], f"summary.{sec}['{key}']")

    def sprow(sec, key):
        return pair_row(S[sec][key], f"summary.{sec}['{key}']")

    c2 = "near_frontier_commander"
    add("K1", "attack effect, commander team",
        mrow("F1 REF+attack - REF | commander | pooled over sizes"),
        {f"r{n}": pair_row(M["contrasts"][f"F1 REF+attack - REF | commander | r{n}"],
                           f"matrix.contrasts['F1 REF+attack - REF | commander | r{n}']") for n in (2, 4)})
    add("K2", "attack effect, commander - consensus team",
        irow("(REF+attack - REF) effect: commander - cbaa | pooled over sizes"),
        {f"r{n}": irow(f"(REF+attack - REF) effect: commander - cbaa | r{n}") for n in (2, 4)})
    add("K4", "attack effect, near-frontier commander - low-cost commander",
        srow(c2, "2_attack_effect_sol_minus_gemini"),
        {f"r{n}": srow(c2, f"2_attack_effect_sol_minus_gemini_r{n}") for n in (2, 4)})
    add("K5", "uncovered known-intruder share, consensus - independent team, REF/MH/LI/LB, exposure-weighted",
        srow("coverage_contrasts", "llm|A3 cbaa - independent|rboth|per_seed_weighted"),
        {f"r{n}": srow("coverage_contrasts", f"llm|A3 cbaa - independent|r{n}|per_seed_weighted") for n in (2, 4)})
    for i, c in (("X12", "REF"), ("X13", "REF+attack")):
        add(i, f"near-frontier - low-cost commander, under {c}", srow(c2, f"3_sol_minus_gemini|{c}"),
            {f"r{n}": sprow(c2, f"3_sol_minus_gemini|{c}|r{n}") for n in (2, 4)})
    return {"rows": X}


if __name__ == "__main__":
    ap = common.parser(__doc__.split("\n")[0])
    a = ap.parse_args()
    o = common.out_dir(a)
    common.dump(main(o), os.path.join(o, "key_rows.json"))
