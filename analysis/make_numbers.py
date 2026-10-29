"""Every number the paper's text and figures quote, written as LaTeX macros.

Each macro is read from a named leaf of the analysis outputs and carries that source in a comment.
Percentages are rounded half-up to whole percent; percentage points (pp) likewise.

    python3 -m analysis.make_numbers --out OUT       ->  OUT/numbers.tex

Reads the outputs of the other analysis scripts in OUT (see analysis/README.md for the order). The macro names
are the identifiers of the paper's LaTeX source.
"""
import json
import os
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

from analysis import common, numfmt
from analysis.common import ARMS
from analysis.numfmt import holm, leaf, load, pct, pval, signed_pp, signed_pp1

_args = common.parser(__doc__.split("\n")[0]).parse_args()
numfmt.set_dir(common.out_dir(_args))
OUT = os.path.join(common.out_dir(_args), "numbers.tex")


MACROS = []


def add(name, value, rel, path, note=""):
    MACROS.append((name, value, f"{rel} :: {'.'.join(str(p) for p in path)}{(' ' + note) if note else ''}"))


M = "matrix_analysis.json"
PF = ["primary_family"]
A1, A2, A3 = "A1 commander - cbaa", "A2 commander - independent", "A3 cbaa - independent"

# --- primary family: pooled over REF, MH, LI, LB and both sizes (8 pairs per seed, 30 seeds)
for name, k, side in (("numRateCmdPct", A1, "first_breach_rate"), ("numRateCbaaPct", A1, "second_breach_rate"),
                      ("numRateIndPct", A2, "second_breach_rate")):
    add(name, pct(leaf(M, PF + [k, side])), M, PF + [k, side], "(percent)")
add("numEpisodesPerArmPrimary", str(leaf(M, PF + [A1, "n_pairs"])), M, PF + [A1, "n_pairs"],
    "(episodes per arm in the primary family)")
for tag, k in (("AOne", A1), ("ATwo", A2), ("AThree", A3)):
    est = leaf(M, PF + [k, "estimate"])
    lo, hi = leaf(M, PF + [k, "ci95_t_seeds"])
    add(f"num{tag}PP", signed_pp(est), M, PF + [k, "estimate"], "(pp)")
    add(f"num{tag}PPCI", f"{signed_pp(lo)} to {signed_pp(hi)}", M, PF + [k, "ci95_t_seeds"], "(pp)")
# --- four reds
for name, k, side in (("numRateCmdFourPct", A1, "first_breach_rate"), ("numRateCbaaFourPct", A1, "second_breach_rate"),
                      ("numRateIndFourPct", A3, "second_breach_rate")):
    path = ["contrasts", k + " | r4", side]
    add(name, pct(leaf(M, path)), M, path, "(percent)")

# --- commander, MH - REF, pooled over sizes (secondary, unadjusted; t vs sign-flip disagree at 0.05)
p = ["contrasts", "D1 MH - REF | commander | pooled over sizes", "breach_difference", "estimate"]
add("numCmdMhPP", signed_pp(leaf(M, p)), M, p, "(pp)")
p = ["contrasts", "D1 MH - REF | commander | pooled over sizes", "breach_difference", "ci95_t_seeds"]
lo, hi = leaf(M, p)
add("numCmdMhPPCI", f"{signed_pp(lo)} to {signed_pp(hi)}", M, p, "(pp, 95% t interval over 30 seeds)")

# --- strategy weights (W) vs bounded (B), Flash-Lite, pooled REF/MH/LI/LB x sizes
S = "high_authority.json"
p = ["family_N_holm", "N2 W - B", "estimate"]
add("numWminusBPP", signed_pp(leaf(S, p)), S, p, "(pp)")
add("numWminusBPPAbs", pct(abs(leaf(S, p))), S, p, "(pp, unsigned, for running text)")
_ci = leaf(S, ["family_N_holm", "N2 W - B", "ci95_t_seeds"])
assert min(_ci) > 0
add("numWminusBPPAbsCI", f"{pct(_ci[0])} to {pct(_ci[1])}", S, ["family_N_holm", "N2 W - B", "ci95_t_seeds"], "(pp, unsigned, 95% interval)")
p = ["family_N_holm", "N2 W - B", "ci95_t_seeds"]
lo, hi = leaf(S, p)
add("numWminusBPPCI", f"{signed_pp(lo)} to {signed_pp(hi)}", S, p, "(pp)")

# --- W-Sol (REF, four reds, 30 seeds)
W = "high_authority_near_frontier.json"
for name, lvl in (("numBrWSol", "W-Sol"), ("numBrW", "W"), ("numBrB", "B")):
    p = ["levels", lvl, "breaches"]
    add(name, leaf(W, p).replace("/", " of "), W, p, "(breaches of 30, REF, four reds)")

# --- airspace challenge (5 seeds, ideal network, four intruders): defenders lost to birds and civilian drones, per team
D = "airspace.json"
for name, arm in (("numDecoyLostB", "B"), ("numDecoyLostCmd", "CMD"), ("numDecoyLostW", "W"),
                  ("numDecoyLostWSol", "WSOL"), ("numDecoyLostSolCmd", "SOLCMD")):
    per = leaf(D, ["arms", arm, "per_seed"])
    add(name, str(sum(v["blues_lost_to_contacts"] for v in per.values())), D,
        ["arms", arm, "per_seed", "*", "blues_lost_to_contacts"], f"(sum over {len(per)} seeds)")

# --- commander call latency (Flash-Lite, matched REF/attack cells)
V = "summary.json"
p = ["near_frontier_commander", "cmd_cost_flashlite", "latency_median_s"]
add("numCmdLatencyS", f"{leaf(V, p):.1f}", V, p, "(s, median per call)")

# =====================================================================================
# Results
# =====================================================================================
RF = "run_facts.json"
COPIED = {}


def _rnd(x, d):
    return float(Decimal(str(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def _pctw(x, d=1):
    return f"{_rnd(round(100 * float(x), 6), d):.{d}f}\\%"


def _put(name, value, src):
    COPIED[name] = (str(value), src)


def run_facts():
    """The macros about the runs themselves (validity, calls, conflicts, coverage, exposure, behaviour)."""
    rf, cells, vd = load(RF), load(M)["cells"], load(M)["validity"]
    _put("numValidLLM", rf["main_valid"], f"{RF} :: main_valid")
    _put("numCalls", f"{rf['main_calls']:,}", f"{RF} :: main_calls")
    _put("numTransportFail", rf["main_transport_failures"], f"{RF} :: main_transport_failures")
    _put("numRetries", rf["main_retries"], f"{RF} :: main_retries")
    _put("numCmdFallbackEpochs", vd["commander"]["epochs_with_agent_fallback"], f"{M} :: validity.commander.epochs_with_agent_fallback")
    _put("numCmdEpochs", f"{vd['commander']['epochs']:,}", f"{M} :: validity.commander.epochs")
    for r, n in ((2, "numBrTwoTotal"), (4, "numBrFourTotal")):
        _put(n, sum(v["breaches"] for k, v in cells.items() if k.split("|")[1] == f"r{r}"), f"{M} :: sum cells r{r} breaches")
    _put("numEpsPerSize", rf["episodes_per_size"]["2"], f"{RF} :: episodes_per_size")
    cc, cs = rf["conflicts_by_cell"], rf["conflicts_by_seed"]
    for arm, at, c, ct in (("cbaa", "Cbaa", "REF", "Ref"), ("independent", "Ind", "REF", "Ref"), ("cbaa", "Cbaa", "MH", "Mh"),
                           ("independent", "Ind", "MH", "Mh")):
        conf, smp = cc[f"{arm}|{c}"]
        if (arm, c) != ("independent", "MH"):
            _put(f"numConfN{at}{ct}", f"{conf:,} of {smp:,}", f"{RF} :: conflicts_by_cell[{arm}|{c}]")
        if (arm, c) != ("cbaa", "REF"):
            _put(f"numConf{at}{ct}", _pctw(conf / smp, 0), f"{RF} :: conflicts_by_cell[{arm}|{c}] share")
    seeds = [str(s) for s in common.SEEDS]
    cb = [cs[f"cbaa|other|{s}"] for s in seeds]
    ind = [cs[f"independent|other|{s}"][0] / cs[f"independent|other|{s}"][1] for s in seeds]
    _put("numConfCbaaOtherN", f"{sum(c[0] for c in cb):,} of {sum(c[1] for c in cb):,}", f"{RF} :: conflicts_by_seed cbaa other")
    _put("numConfCbaaSeedMax", _pctw(max(c[0] / c[1] for c in cb)), f"{RF} :: max per-seed share, consensus team, other conditions")
    _put("numConfIndSeedLo", _pctw(min(ind), 0), f"{RF} :: min per-seed share, independent team, other conditions")
    _put("numConfIndSeedHi", _pctw(max(ind), 0), f"{RF} :: max per-seed share, independent team, other conditions")
    _put("numConfMhSeedsLower", sum(cs[f"cbaa|MH|{s}"][0] / cs[f"cbaa|MH|{s}"][1] < cs[f"independent|MH|{s}"][0] / cs[f"independent|MH|{s}"][1]
                                    for s in seeds), f"{RF} :: seeds with the consensus MH share below the independent MH share")
    cov = load(V)["coverage"]["llm"]
    for a, n in (("cbaa", "Cbaa"), ("independent", "Ind"), ("commander", "Cmd")):
        _put(f"numCovUnc{n}", _pctw(cov[f"{a}|both|base"]["share_uncovered_exec"], 0), f"{V} :: coverage.llm['{a}|both|base'].share_uncovered_exec")
    for a, n in (("commander", "Cmd"), ("cbaa", "Cbaa")):
        _put(f"numCovUnseen{n}", _pctw(cov[f"{a}|both|base"]["share_unseen_of_active"], 0), f"{V} :: coverage.llm['{a}|both|base'].share_unseen_of_active")
    bp = load("behaviour_pool.json")["cells"]
    ans = {}
    for n in (2, 4):
        for k, v in bp[f"commander|r{n}|ALL"]["exact_counts"]["commander_answered_kinds"].items():
            ans[k] = ans.get(k, 0) + v
    BP = "behaviour_pool.json"
    _put("numCmdReconAns", ans["RECON"], f"{BP} :: commander answered kinds RECON, both sizes")
    _put("numCmdAnsTotal", f"{ans['DEFEND'] + ans['NEUTRALIZE'] + ans['RECON']:,}", f"{BP} :: sum answered kinds, both sizes")
    act = lambda arm, n: bp[f"{arm}|r{n}|ALL"]["exact_counts"]["active_blue_steps"]  # noqa: E731
    _put("numCmdPreRecon", f"{sum(act('commander', n).get('pre:RECON', 0) for n in (2, 4)):,}", f"{BP} :: commander active_blue_steps pre:RECON")
    _put("numCmdPreSteps", f"{sum(act('commander', n).get(k, 0) for n in (2, 4) for k in ('pre:DEFEND', 'pre:RECON', 'pre:NEUTRALIZE')):,}",
         f"{BP} :: commander active_blue_steps pre:*")
    shares = [act(a, n).get("pre:RECON", 0) / sum(v for k, v in act(a, n).items() if k.startswith("pre:"))
              for a in ("cbaa", "independent") for n in (2, 4)]
    _put("numPreReconDecLo", f"{100 * min(shares):.0f}", f"{BP} :: min pre:RECON share, decentralized arms")
    _put("numPreReconDecHi", f"{100 * max(shares):.0f}", f"{BP} :: max pre:RECON share, decentralized arms")
    att = [bp[f"{a}|r{n}|ALL"]["red_attack_share"] * 100 for a in ARMS for n in (2, 4)]
    _put("numRedAttackPctLo", f"{min(att):.0f}\\%", f"{BP} :: min red_attack_share * 100")
    _put("numRedAttackPctHi", f"{max(att):.0f}\\%", f"{BP} :: max red_attack_share * 100")
    ex = load(V)["exposure"]
    for a, n in (("cbaa", "Cbaa"), ("independent", "Ind"), ("commander", "Cmd")):
        _put(f"numExpAlt{n}", ex[f"llm|{a}"]["with_altered_reports"], f"{V} :: exposure['llm|{a}'].with_altered_reports")
        _put(f"numExpBzero{n}", ex[f"llm|{a}"]["b0_accepted_episodes"], f"{V} :: exposure['llm|{a}'].b0_accepted_episodes")
    _put("numAtkEps", 2 * len(common.SEEDS), f"{V} :: attack episodes per team (2 intruder counts x 30 seeds)")
    u = load("high_authority_unseen.json")
    for a in ("W", "B"):
        _put(f"swUnseen{a}", f"{100 * u[f'share_unseen_{a}']:.0f}\\%", f"high_authority_unseen.json :: share_unseen_{a}")
    _put("swEpsPerArm", 10 * load(S)["n_seeds"], f"{S} :: n_seeds x 2 sizes x 5 conditions")
    c2 = load(V)["near_frontier_commander"]["cells"]
    _put("numCTwoBrRef", sum(c2[f"solcmd|r{n}|REF"] for n in (2, 4)), f"{V} :: near_frontier_commander.cells r2+r4 REF")
    _put("numCTwoBrAtk", sum(c2[f"solcmd|r{n}|REF+attack"] for n in (2, 4)), f"{V} :: near_frontier_commander.cells r2+r4 REF+attack")
    _put("numCTwoEps", load(RF)["near_frontier_commander_valid"], f"{RF} :: near_frontier_commander_valid")
    sb = load(W)["seeds_by_run"]
    _put("solwInvalid", load(RF)["near_frontier_first_run_invalid"], f"{RF} :: near_frontier_first_run_invalid")
    _put("solwSeedsRerun", len(sb["second_run"]), f"{W} :: seeds_by_run.second_run")
    _put("solwSeedsFirst", len(sb["first_run"]), f"{W} :: seeds_by_run.first_run (valid records used from the first run)")


def copy(name, new=None):
    v, src = COPIED[name]
    MACROS.append((new or name, v, src))


run_facts()


def pp_ci(name, rel, path_est, path_ci, note=""):
    est, (lo, hi) = leaf(rel, path_est), leaf(rel, path_ci)
    add(name, signed_pp(est), rel, path_est, "(pp) " + note)
    add(name + "CI", f"{signed_pp(lo)} to {signed_pp(hi)}", rel, path_ci, "(pp, 95% interval) " + note)


def pp_ci_up(name, rel, path_est, path_ci, note=""):
    """Unsigned copies of a positive difference and its all-positive interval, for running text ("38 points higher")."""
    est, (lo, hi) = leaf(rel, path_est), leaf(rel, path_ci)
    assert est > 0 and lo > 0, name
    add(name + "Abs", pct(est), rel, path_est, "(pp, unsigned) " + note)
    add(name + "AbsCI", f"{pct(lo)} to {pct(hi)}", rel, path_ci, "(pp, unsigned, 95% interval) " + note)


def pp1_ci(name, rel, path_est, path_ci, note=""):
    est, (lo, hi) = leaf(rel, path_est), leaf(rel, path_ci)
    add(name, signed_pp1(est), rel, path_est, "(pp, 1 dp) " + note)
    add(name + "CI", f"{signed_pp1(lo)} to {signed_pp1(hi)}", rel, path_ci, "(pp, 1 dp, 95% interval) " + note)


# --- validity and design of the runs
for n in ("numValidLLM", "numCalls", "numTransportFail", "numRetries", "numCmdFallbackEpochs", "numCmdEpochs",
          "numBrTwoTotal", "numBrFourTotal", "numEpsPerSize"):
    copy(n)

# --- time-limit endings (cache re-simulations, main_replays.json)
R = "main_replays.json"
_E = load(R)["episodes"]
assert load(R)["n_reproduced"] == 900
def _hz(arm, reds):
    s = [x for x in _E if (arm == "all" or x["arm"] == arm) and x["reds"] == reds]
    nb = [x for x in s if not x["outcome"]["touchdown"]]
    hz = [x for x in nb if x["live"]]
    return nb, hz
for arm, tag in (("all", "All"), ("commander", "Cmd"), ("cbaa", "Cbaa"), ("independent", "Ind")):
    for reds, rt in ((2, "Two"), (4, "Four")):
        nb, hz = _hz(arm, reds)
        add(f"numHz{tag}{rt}", f"{len(hz)} of {len(nb)}", R, ["episodes", f"{arm}|r{reds}", "no breach & ran to horizon with a red alive"], "(of no-breach episodes)")
_live4 = [l for x in _hz("all", 4)[1] for l in x["live"]]
add("numHzLiveFour", str(len(_live4)), R, ["episodes", "r4", "live reds at horizon"])
add("numHzClosingFour", str(sum(l["class"] == "approach" for l in _live4)), R, ["episodes", "r4", "live reds class approach"])
add("numHzStandoffFour", str(sum(l["class"] == "stand-off" for l in _live4)), R, ["episodes", "r4", "live reds class stand-off"])
_d = sorted(l["dist_to_region_end"] for l in _live4 if l["class"] == "approach")
add("numHzClosingDist", f"{float(np.median(_d)):.1f}", R, ["episodes", "r4", "median dist_to_region_end, approach"], "(units)")

# --- primary family
for tag, k in (("AOne", A1), ("ATwo", A2), ("AThree", A3)):
    add(f"num{tag}Holm", pval(leaf(M, PF + [k, "p_t_holm"])), M, PF + [k, "p_t_holm"])
    add(f"num{tag}SFHolm", pval(leaf(M, PF + [k, "p_sign_flip_holm"])), M, PF + [k, "p_sign_flip_holm"])
for tag, k in (("AOne", A1), ("AThree", A3)):
    for rt, r in (("Two", "r2"), ("Four", "r4")):
        pp_ci(f"num{tag}{rt}PP", M, ["contrasts", f"{k} | {r}", "breach_difference", "estimate"],
              ["contrasts", f"{k} | {r}", "breach_difference", "ci95_t_seeds"], "(secondary, unadjusted)")
_c = load(M)["cells"]
for arm, tag in (("commander", "Cmd"), ("cbaa", "Cbaa"), ("independent", "Ind")):
    for reds, rt in ((2, "Two"), (4, "Four")):
        n = sum(_c[f"{arm}|r{reds}|{c}"]["breaches"] for c in ("REF", "MH", "LI", "LB"))
        add(f"numBr{tag}{rt}Primary", f"{n} of 120", M, ["cells", f"{arm}|r{reds}|REF+MH+LI+LB", "breaches"])

# --- coordination
for n in ("numConfNCbaaRef", "numConfNIndRef", "numConfIndRef", "numConfCbaaMh", "numConfNCbaaMh", "numConfIndMh",
          "numConfMhSeedsLower", "numConfCbaaOtherN", "numConfCbaaSeedMax", "numConfIndSeedLo", "numConfIndSeedHi",
          "numCovUncCbaa", "numCovUncInd", "numCovUncCmd", "numCovUnseenCmd", "numCovUnseenCbaa",
          "numCmdReconAns", "numCmdAnsTotal", "numCmdPreRecon", "numCmdPreSteps", "numPreReconDecLo", "numPreReconDecHi",
          "numRedAttackPctLo", "numRedAttackPctHi"):
    copy(n)
P = "key_rows.json"
pp1_ci("numKFivePP", P, ["rows", "K5", "pooled", "estimate"], ["rows", "K5", "pooled", "ci95_t"], "(uncovered known-red share)")
_e, (_lo, _hi) = leaf(P, ["rows", "K5", "pooled", "estimate"]), leaf(P, ["rows", "K5", "pooled", "ci95_t"])
assert _e < 0 and _hi < 0
add("numKFivePPAbs", signed_pp1(-_e).lstrip("+"), P, ["rows", "K5", "pooled", "estimate"], "(pp, unsigned, auction below independent)")
add("numKFivePPAbsCI", f"{signed_pp1(-_hi).lstrip('+')} to {signed_pp1(-_lo).lstrip('+')}", P, ["rows", "K5", "pooled", "ci95_t"], "(pp, unsigned)")
# Holm over the four key rows K1, K2, K4, K5, on the raw seed-level t p values of key_rows.json.
_KF = ["K1", "K2", "K4", "K5"]
_kadj = dict(zip(_KF, holm([leaf(P, ["rows", k, "pooled", "p_t"]) for k in _KF])))
_kadj_sf = dict(zip(_KF, holm([leaf(P, ["rows", k, "pooled", "p_sign_flip"]) for k in _KF])))
add("numKFiveHolm", pval(_kadj["K5"]), P, ["rows", "{K1,K2,K4,K5}", "pooled", "p_t"], "(Holm over 4 LLM-only rows, recomputed)")
add("numKFiveSFHolm", pval(_kadj_sf["K5"]), P, ["rows", "{K1,K2,K4,K5}", "pooled", "p_sign_flip"], "(Holm over 4, recomputed)")
add("numKOneHolm", pval(_kadj["K1"]), P, ["rows", "{K1,K2,K4,K5}", "pooled", "p_t"], "(Holm over 4, recomputed)")
add("numKFiveSeedsNeg", str(leaf(P, ["rows", "K5", "pooled", "seeds_pos_neg_zero"])[1]), P, ["rows", "K5", "pooled", "seeds_pos_neg_zero", 1])
VS = "summary.json"
_k = "llm|A1 commander - cbaa|rboth|per_seed_weighted"
pp1_ci("numCovAOnePP", VS, ["coverage_contrasts", _k, "estimate"], ["coverage_contrasts", _k, "ci95_t_seeds"])
assert leaf(VS, ["coverage_contrasts", _k, "estimate"]) > 0
add("numCovAOnePPAbs", signed_pp1(leaf(VS, ["coverage_contrasts", _k, "estimate"])).lstrip("+"), VS, ["coverage_contrasts", _k, "estimate"], "(pp, unsigned, commander above auction)")
S2 = "main_replays_summary.json"
for arm, tag in (("commander", "Cmd"), ("cbaa", "Cbaa"), ("independent", "Ind")):
    f = leaf(S2, ["filter", f"{arm}|all|rboth"])
    add(f"numFiltPurs{tag}", pct(f["pursuer_barrier_share"]), S2, ["filter", f"{arm}|all|rboth", "pursuer_barrier_share"], "(percent)")
    add(f"numFiltMates{tag}", pct(f["pursuer_barrier_teammate_wreck_share"]), S2, ["filter", f"{arm}|all|rboth", "pursuer_barrier_teammate_wreck_share"], "(percent)")
    add(f"numFiltRed{tag}", pct(f["pursuer_barrier_red_share"]), S2, ["filter", f"{arm}|all|rboth", "pursuer_barrier_red_share"], "(percent)")

# --- network conditions
for arm, tag in (("commander", "Cmd"), ("cbaa", "Cbaa"), ("independent", "Ind")):
    for c, ct in (("D2 LI - REF", "Li"), ("D3 LB - REF", "Lb"), ("D1 MH - REF", "Mh"), ("F1 REF+attack - REF", "Atk")):
        k = f"{c} | {arm} | pooled over sizes"
        if (tag, ct) == ("Cmd", "Mh"):
            continue  # numCmdMhPP already defined above
        pp_ci(f"num{tag}{ct}PP", M, ["contrasts", k, "breach_difference", "estimate"], ["contrasts", k, "breach_difference", "ci95_t_seeds"], "(secondary, unadjusted)")
add("numCmdMhSF", f"{leaf(M, ['contrasts', 'D1 MH - REF | commander | pooled over sizes', 'breach_difference', 'p_sign_flip']):.3f}", M,
    ["contrasts", "D1 MH - REF | commander | pooled over sizes", "breach_difference", "p_sign_flip"])
_ik = "(MH - REF) effect: commander - cbaa | pooled over sizes"
pp_ci("numIMhCmdCbaaPP", M, ["interactions_exploratory", _ik, "breach", "estimate"], ["interactions_exploratory", _ik, "breach", "ci95_t_seeds"], "(exploratory)")
L = "lb_minus_li.json"
for arm, tag in (("commander", "Cmd"), ("cbaa", "Cbaa"), ("independent", "Ind")):
    pp_ci(f"num{tag}LbLiPP", L, ["arms", arm, "LB - LI", "estimate"], ["arms", arm, "LB - LI", "ci95_t_seeds"], "(post hoc)")
F1 = "F1 REF+attack - REF"
pp_ci("numFOnePP", M, PF + [F1, "estimate"], PF + [F1, "ci95_t_seeds"])
add("numFOneHolm", pval(leaf(M, PF + [F1, "p_t_holm"])), M, PF + [F1, "p_t_holm"])
for n in ("numExpAltCbaa", "numExpAltInd", "numExpAltCmd", "numAtkEps", "numExpBzeroCbaa", "numExpBzeroInd", "numExpBzeroCmd"):
    copy(n)
for arm, tag in (("commander", "Cmd"), ("cbaa", "Cbaa"), ("independent", "Ind")):
    for reds, rt in ((2, "Two"), (4, "Four"), ("both", "")):
        x = leaf(S2, ["comm", f"{arm}|MH|r{reds}"])
        add(f"numConn{tag}{rt}", pct(x["connected_fraction_mean"]), S2, ["comm", f"{arm}|MH|r{reds}", "connected_fraction_mean"], "(percent of rounds)")
    x = leaf(S2, ["comm", f"{arm}|MH|rboth"])
    add(f"numPart{tag}", f"{x['mean_partition_s']:.1f}", S2, ["comm", f"{arm}|MH|rboth", "mean_partition_s"], "(s)")

# --- authority and capability
_sc = load(S)["cells"]
for lvl, tag in (("W", "W"), ("B", "B")):
    n = sum(_sc[f"{lvl}|r{r}|{c}"]["breaches"] for r in (2, 4) for c in ("REF", "MH", "LI", "LB"))
    tot = sum(_sc[f"{lvl}|r{r}|{c}"].get("n", 30) for r in (2, 4) for c in ("REF", "MH", "LI", "LB"))
    add(f"numRate{tag}SwPct", pct(n / tot), S, ["cells", f"{lvl}|r2+r4|REF+MH+LI+LB", "breaches / n"], f"(percent; {n} of {tot})")
pp_ci("numWminusBFourPP", S, ["rows", "N2 W - B | r4", "breach_difference", "estimate"], ["rows", "N2 W - B | r4", "breach_difference", "ci95_t_seeds"], "(secondary)")
pp_ci_up("numWminusBFourPP", S, ["rows", "N2 W - B | r4", "breach_difference", "estimate"], ["rows", "N2 W - B | r4", "breach_difference", "ci95_t_seeds"], "(secondary)")
for lvl, tag in (("W", "W"), ("B", "B")):
    n = sum(_sc[f"{lvl}|r4|{c}"]["breaches"] for c in ("REF", "MH", "LI", "LB"))
    tot = sum(_sc[f"{lvl}|r4|{c}"].get("n", 30) for c in ("REF", "MH", "LI", "LB"))
    add(f"numRate{tag}SwFourPct", pct(n / tot), S, ["cells", f"{lvl}|r4|REF+MH+LI+LB", "breaches / n"], f"(percent; {n} of {tot})")
# Holm over the two rows of family N: N2 (W - B) and N5 (attack effect W - B).
# N1, N3, N4 (rule bids R / adaptive rule H) removed. S family reduces to S1 alone: reported unadjusted (exact McNemar).
_NF = ["N2 W - B", "N5 attack effect W - B"]
_nadj = dict(zip(_NF, holm([leaf(S, ["family_N_holm", k, "p_t_raw"]) for k in _NF])))
add("numWminusBHolm", pval(_nadj["N2 W - B"]), S, ["family_N_holm", "{N2,N5}", "p_t_raw"], "(Holm over 2 LLM-only rows, recomputed)")
for n in ("swUnseenW", "swUnseenB", "swEpsPerArm"):
    copy(n)
pp_ci("numUnseenWBPP", "high_authority_unseen.json", ["unseen W - B", "estimate"], ["unseen W - B", "ci95_t_seeds"])
pp_ci_up("numUnseenWBPP", "high_authority_unseen.json", ["unseen W - B", "estimate"], ["unseen W - B", "ci95_t_seeds"])
_s1 = ["rows", "S1 W-Sol - W"]
add("numSOnePP", signed_pp(leaf(W, _s1 + ["estimate"])), W, _s1 + ["estimate"], "(pp)")
lo, hi = leaf(W, _s1 + ["newcombe10"])
add("numSOnePPCI", f"{signed_pp(lo)} to {signed_pp(hi)}", W, _s1 + ["newcombe10"], "(pp, Newcombe method 10)")
add("numSOneMcN", pval(leaf(W, _s1 + ["exact_mcnemar_p"])), W, _s1 + ["exact_mcnemar_p"])
_sb = ["rows", "secondary W-Sol - B"]
add("numWSolBPP", signed_pp(leaf(W, _sb + ["estimate"])), W, _sb + ["estimate"], "(pp)")
lo, hi = leaf(W, _sb + ["newcombe10"])
add("numWSolBPPCI", f"{signed_pp(lo)} to {signed_pp(hi)}", W, _sb + ["newcombe10"], "(pp, Newcombe method 10)")
SS = "high_authority_near_frontier_secondary.json"
for lvl, tag in (("W-Sol", "WSol"), ("W", "W"), ("B", "B")):
    add(f"numUncov{tag}", pct(leaf(SS, ["uncovered_known_pooled", lvl])), SS, ["uncovered_known_pooled", lvl], "(percent, REF four reds)")
    add(f"numUnseenRef{tag}", pct(leaf(SS, ["unseen_share_pooled", lvl])), SS, ["unseen_share_pooled", lvl], "(percent, REF four reds)")
add("numBrSolCmdFourRef", f"{leaf(W, ['levels', 'Sol commander', 'breaches']).split('/')[0]} of 30", W, ["levels", "Sol commander", "breaches"])
add("numBrCmdFourRef", f"{_c['commander|r4|REF']['breaches']} of 30", M, ["cells", "commander|r4|REF", "breaches"])
for x, tag in (("X12", "Ref"), ("X13", "Atk")):
    pp_ci(f"numSolCmd{tag}PP", P, ["rows", x, "pooled", "estimate"], ["rows", x, "pooled", "ci95_t"], "(unadjusted)")
for n in ("numCTwoBrRef", "numCTwoBrAtk"):
    copy(n)
add("numBrCmdRefBoth", f"{_c['commander|r2|REF']['breaches'] + _c['commander|r4|REF']['breaches']}", M, ["cells", "commander|r2+r4|REF", "breaches"], "(of 60)")
add("numBrCmdAtkBoth", f"{_c['commander|r2|REF+attack']['breaches'] + _c['commander|r4|REF+attack']['breaches']}", M, ["cells", "commander|r2+r4|REF+attack", "breaches"], "(of 60)")

# --- airspace challenge (5 seeds, ideal network, four intruders)
for arm, tag in (("B", "B"), ("CMD", "Cmd"), ("W", "W"), ("WSOL", "WSol"), ("SOLCMD", "SolCmd")):
    a = leaf(D, ["arms", arm])
    add(f"numDecoyBr{tag}", str(a["breaches"]), D, ["arms", arm, "breaches"], "(of 5)")
    add(f"numDecoyBrRef{tag}", str(a["no_contact_reference"]["breaches"]), D, ["arms", arm, "no_contact_reference", "breaches"], "(of 5, no birds or civilian drones)")
    add(f"numDecoyLostCiv{tag}", str(a["blues_lost_to_contacts_by_kind"].get("civilian", 0)), D, ["arms", arm, "blues_lost_to_contacts_by_kind", "civilian"])
    add(f"numDecoyLostBird{tag}", str(a["blues_lost_to_contacts_by_kind"].get("bird", 0)), D, ["arms", arm, "blues_lost_to_contacts_by_kind", "bird"])
    per = [v["blues_lost_to_contacts"] for v in a["per_seed"].values()]
    add(f"numDecoyLostRange{tag}", f"{min(per)}--{max(per)}", D, ["arms", arm, "per_seed", "*", "blues_lost_to_contacts"], "(range over seeds)")
# low-cost teams (B, CMD, W) together: range of defenders lost to birds and civilian drones, the total, how many to
# civilian drones, and the per-episode maximum the text states ("never more than one per episode")
_low = [leaf(D, ["arms", x]) for x in ("B", "CMD", "W")]
_lost = [x["blues_lost_to_contacts"] for x in _low]
add("numDecoyLostLowLo", str(min(_lost)), D, ["arms", "B|CMD|W", "blues_lost_to_contacts"], "(min over the low-cost teams)")
add("numDecoyLostLowHi", str(max(_lost)), D, ["arms", "B|CMD|W", "blues_lost_to_contacts"], "(max over the low-cost teams)")
add("numDecoyLostLowAll", str(sum(_lost)), D, ["arms", "B+CMD+W", "blues_lost_to_contacts"], "(sum)")
add("numDecoyLostLowCiv", str(sum(x["blues_lost_to_contacts_by_kind"].get("civilian", 0) for x in _low)), D,
    ["arms", "B+CMD+W", "blues_lost_to_contacts_by_kind", "civilian"], "(sum)")
assert max(v["blues_lost_to_contacts"] for x in _low for v in x["per_seed"].values()) == 1, "text: at most one per episode"
add("numDecoyEpisodesLLM", str(sum(leaf(D, ["arms", a, "episodes"]) for a in ("B", "CMD", "W", "WSOL", "SOLCMD"))), D, ["arms", "B+CMD+W+WSOL+SOLCMD", "episodes"])
for _n in ("numCTwoEps", "solwInvalid", "solwSeedsRerun", "solwSeedsFirst"):
    copy(_n)
CA = ["calls_answered"]
_sc2 = leaf(D, CA + ["SOLCMD"])
add("numSolCmdDecoyEpochs", str(sum(_sc2["bird"].values()) + sum(_sc2["civilian"].values())), D, CA + ["SOLCMD", "bird+civilian", "*"], "(bird and civilian-drone contact-epochs seen)")
add("numSolCmdDecoyAsg", str(_sc2["bird"].get("True", 0) + _sc2["civilian"].get("True", 0)), D, CA + ["SOLCMD", "bird+civilian", "True"], "(assigned)")
def _share(arm, kind, keys):
    d = leaf(D, CA + [arm, kind]); return sum(d.get(k, 0) for k in keys) / sum(d.values())
for kind, kt in (("drone", "Red"), ("bird", "Bird"), ("civilian", "Civ")):
    add(f"numCallBLow{kt}", pct(_share("B", kind, ["LOW"])), D, CA + ["B", kind, "LOW"], "(percent of answers)")
    add(f"numCallWZero{kt}", pct(_share("W", kind, ["0"])), D, CA + ["W", kind, "0"], "(percent of answers)")
    add(f"numCallWSolZero{kt}", pct(_share("WSOL", kind, ["0"])), D, CA + ["WSOL", kind, "0"], "(percent of answers)")
    add(f"numCallCmdAsg{kt}", pct(_share("CMD", kind, ["True"])), D, CA + ["CMD", kind, "True"], "(percent of contact-epochs assigned)")
    add(f"numCallSolCmdAsg{kt}", pct(_share("SOLCMD", kind, ["True"])), D, CA + ["SOLCMD", kind, "True"], "(percent of contact-epochs assigned)")
    add(f"numCallCmdAside{kt}", pct(_share("CMD", kind, ["False"])), D, CA + ["CMD", kind, "False"], "(percent of contact-epochs NOT assigned)")
    add(f"numCallSolCmdAside{kt}", pct(_share("SOLCMD", kind, ["False"])), D, CA + ["SOLCMD", kind, "False"], "(percent of contact-epochs NOT assigned)")

# --- disguised intruders (5 seeds, ideal network, four intruders): each real intruder shown as a
# bird-drone photograph, without (U, "Camo") and with (W, "Warn") the one-sentence forewarning.
# The airspace rows (attack-drone photographs) come from the same analysis's airspace_reference, which
# reproduces every airspace number above with the same code.
CM = "disguise.json"
_camo_tags = (("B", "B"), ("CMD", "Cmd"), ("W", "W"), ("WSOL", "WSol"), ("SOLCMD", "SolCmd"))
assert leaf(CM, ["n_records"]) == 50
add("numCamoEpisodes", str(leaf(CM, ["n_records"])), CM, ["n_records"], "(5 teams x 2 conditions x 5 seeds, all valid)")
for arm, tag in _camo_tags:
    _ref = leaf(CM, ["airspace_reference", "teams", f"{arm}|airspace"])
    assert _ref["breach"] == leaf(D, ["arms", arm, "breaches"]), arm   # the same airspace numbers as above
    add(f"numDecoyKill{tag}", str(_ref["reds_destroyed"]), CM, ["airspace_reference", "teams", f"{arm}|airspace", "reds_destroyed"], "(of 20)")
    for cond, ct in (("U", "Camo"), ("W", "Warn")):
        t = leaf(CM, ["teams", f"{arm}|{cond}"])
        base = ["teams", f"{arm}|{cond}"]
        add(f"num{ct}Br{tag}", str(t["breach"]), CM, base + ["breach"], "(of 5)")
        add(f"num{ct}Kill{tag}", str(t["reds_destroyed"]), CM, base + ["reds_destroyed"], "(of 20)")
        add(f"num{ct}Lost{tag}", str(t["blues_lost_to_contacts"]), CM, base + ["blues_lost_to_contacts"], "(defenders lost to birds and civilian drones)")
        for kind, kt in (("drone", "Red"), ("bird", "Bird"), ("civilian", "Civ")):
            c = leaf(CM, ["calls", f"{arm}|{cond}", kind])
            add(f"num{ct}Aside{kt}{tag}", pct(c["set_aside"] / c["n"]), CM, ["calls", f"{arm}|{cond}", kind, "set_aside/n"],
                f"(percent of answers set aside; {c['set_aside']} of {c['n']})")
# speed cue (U): teams that set fast intruders aside less often than slower ones, and their fast-intruder range
_fast = []
for arm, tag in _camo_tags:
    c = leaf(CM, ["calls", f"{arm}|U"])
    f, s = c["drone_fast"], c["drone_not_fast"]
    if f["set_aside"] / f["n"] < s["set_aside"] / s["n"]:
        _fast.append(f["set_aside"] / f["n"])
add("numCamoFastTeams", str(len(_fast)), CM, ["calls", "*|U", "drone_fast vs drone_not_fast"], "(teams with fast < not fast)")
add("numCamoFastLo", pct(min(_fast)), CM, ["calls", "*|U", "drone_fast", "set_aside/n"], "(percent, min over those teams)")
add("numCamoFastHi", pct(max(_fast)), CM, ["calls", "*|U", "drone_fast", "set_aside/n"], "(percent, max over those teams)")
add("numCamoFastThr", str(leaf(CM, ["fast_threshold"])), CM, ["fast_threshold"], "(units/s)")
# the photo probe (disguise_probe/calls.jsonl in the data deposit): the 192 single-photograph calls that asked
# each model to categorise a photograph, with and without the forewarning
_PSRC = "disguise_probe/calls.jsonl"
_pr = [json.loads(x) for x in open(os.path.join(_args.data, _PSRC))]
assert all(r.get("category") is not None for r in _pr), "probe call without a category"
assert len(_pr) == 192, len(_pr)
# text (Methods): warned, both models called almost every bird-drone photograph a red intruder, and
# only the near-frontier model called a real-bird photograph one, for exactly one of the three
_rb = {(r["model_key"], r["photo"]) for r in _pr if r["kind"] == "bird" and r["framing"] == "game_w"
       and r["category"] == "red intruder"}
assert {m for m, _ in _rb} == {"sol"} and len(_rb) == 1, _rb
assert not any(r["category"] == "red intruder" for r in _pr if r["kind"] == "bird_drone" and r["framing"] == "game_u")
add("numProbeCalls", str(len(_pr)), _PSRC, ["lines"], "(calls)")
_bd = lambda m, fr: [r for r in _pr if r["model_key"] == m and r["kind"] == "bird_drone" and r["framing"] == fr]
assert len({r["photo"] for r in _pr if r["kind"] == "bird_drone"}) == 7
assert all(r["category"] == "bird" for r in _bd("sol", "game_u"))
assert {r["category"] for r in _bd("flashlite", "game_u")} == {"bird", "civilian drone"}
add("numProbeBdPhotos", "seven", _PSRC, ["distinct bird_drone photos"], "(taxidermy bird-drone photographs in the probe)")
for _m, _tag in (("flashlite", "Low"), ("sol", "Sol")):
    _w = _bd(_m, "game_w")
    add(f"numProbeBdAns{_tag}", str(len(_w)), _PSRC, [_m, "bird_drone", "game_w", "answers"])
    add(f"numProbeBdRedWarn{_tag}", str(sum(r["category"] == "red intruder" for r in _w)), _PSRC, [_m, "bird_drone", "game_w", "red intruder"])

# --- Methods, safety filter: counterexample, head-on grid, braking-barrier sampled test (analysis.barrier_tests)
BT = "barrier_tests.json"
add("numFiltCexReq", str(Decimal(str(leaf(BT, ["B_counterexample", "required"]))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
    BT, ["B_counterexample", "required"], "((1-gamma) h at gap 3.1, closing 4; 2 dp)")
_cb = Decimal(str(leaf(BT, ["B_counterexample", "best_h_next_exact"]))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
add("numFiltCexBest", ("\\textminus" if _cb < 0 else "") + str(abs(_cb)), BT, ["B_counterexample", "best_h_next_exact"],
    "(best h_{k+1} over admissible inputs; 2 dp)")
_hm = leaf(BT, ["A_headon_map", "coop"])
_fail = sum(1 for row in _hm["best_h_next"] for g, x in zip(_hm["gaps"], row) if g ** 2 - 9 > 0 and x < 0)
add("numFiltGridFail", f"{_fail:,}", BT, ["A_headon_map", "coop", "best_h_next"], "(states with h > 0 and best h_{k+1} < 0; D = 3)")
add("numFiltGridStates", f"{len(_hm['gaps']) * len(_hm['speeds']):,}", BT, ["A_headon_map", "coop", "gaps x speeds"], "(two-defender head-on grid)")

# --- Results overview, limited range: station spacing as the scale for the 15-unit range
SG = "station_geometry.json"
for _n, _k in (("numStationGapLo", "station_pair_distance_min"), ("numStationGapHi", "station_pair_distance_max")):
    add(_n, str(Decimal(str(leaf(SG, [_k]))).quantize(Decimal("1"), rounding=ROUND_HALF_UP)), SG, [_k],
        "(units, distance between two defender stations; simulator geometry code)")

# --- Results, limited range: the commander's task-changing orders that arrived
CMO = "commander_missed_orders.json"
_cmo = load(CMO)
assert _cmo["n"] == 300 and _cmo["n_reproduced"] == 300


def _task_orders_arrived(cond, attacked):
    agg = {}
    for x in _cmo["episodes"]:
        if x["condition"] == cond and x["attacked"] == attacked:
            a = agg.setdefault(x["seed"], [0, 0])
            a[0] += x["applied_changed"]
            a[1] += x["applied_changed"] + x["missed_changed"]
    assert len(agg) == 30 and all(b > 0 for _, b in agg.values())
    return sum(a / b for a, b in agg.values()) / 30


assert _task_orders_arrived("REF", False) == 1.0
add("numTaskOrdersArrivedMh", pct(_task_orders_arrived("MH", False)), CMO, ["episodes", "MH", "applied_changed / (applied_changed + missed_changed)"],
    "(percent, mean over 30 seeds; ideal network 100)")

# --- unsigned copies for running text ("17 percentage points higher (95% CI: 9 to 24)"); the direction is in words.
# name+"Abs" strips the sign of the estimate; name+"AbsCI" exists only when the interval excludes zero.
def _unsigned(v):
    return v.replace("\\textminus", "").lstrip("+")
_vals = {n: (v, s) for n, v, s in MACROS}
for _n in ("numAOnePP", "numAThreePP", "numATwoPP", "numAOneFourPP", "numAThreeFourPP", "numAOneTwoPP", "numAThreeTwoPP",
           "numFOnePP", "numCmdMhPP", "numIMhCmdCbaaPP", "numCbaaMhPP", "numSOnePP", "numSolCmdRefPP", "numSolCmdAtkPP"):
    _v, _s = _vals[_n]
    MACROS.append((_n + "Abs", _unsigned(_v), _s + " (unsigned)"))
    _ci, _cs = _vals[_n + "CI"]
    _lo, _hi = _ci.split(" to ")
    if (_lo.startswith("+") and _hi.startswith("+")) or ("textminus" in _lo and "textminus" in _hi):
        _a, _b = sorted((_unsigned(_lo), _unsigned(_hi)), key=float)
        MACROS.append((_n + "AbsCI", f"{_a} to {_b}", _cs + " (unsigned)"))

with open(OUT, "w") as fh:
    fh.write("% AUTO-GENERATED by analysis/make_numbers.py. Do not edit by hand.\n")
    for n, v, src in MACROS:
        fh.write(f"\\newcommand{{\\{n}}}{{{v}}} % {src}\n")
print(f"{len(MACROS)} macros -> {OUT}")
for n, v, _ in MACROS:
    print(f"  \\{n} = {v}")
