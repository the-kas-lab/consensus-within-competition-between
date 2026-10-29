"""Connectivity data for Fig. 5a: realised connectivity of the defenders' network under limited range.

Per team and intruder count, the share of communication rounds in which the live defenders formed one connected
group, one value per seed (30 seeds), from the re-simulation of the main study (main_replays.json, 900 of 900
reproduced) and checked against its summary and against the numConn* macros.
"""
import numpy as np

from figures import nums, paths

TEAMS = (("commander", "Cmd", "Commander"), ("cbaa", "Cbaa", "Consensus"), ("independent", "Ind", "Independent"))


def connectivity():
    R = paths.analysis("main_replays.json")
    assert R["n"] == 900 and R["n_reproduced"] == 900
    S = paths.analysis("main_replays_summary.json")["comm"]
    out = {}
    for arm, tag, _ in TEAMS:
        both = []
        for n, word in ((2, "Two"), (4, "Four")):
            eps = sorted((x for x in R["episodes"] if x["arm"] == arm and x["condition"] == "MH"
                          and not x["attacked"] and x["reds"] == n), key=lambda x: x["seed"])
            assert [x["seed"] for x in eps] == list(range(4001, 4031))
            v = [x["comm"]["connected_fraction"] for x in eps]
            assert abs(np.mean(v) - S[f"{arm}|MH|r{n}"]["connected_fraction_mean"]) < 6e-4
            nums.assert_macro(f"numConn{tag}{word}", nums.pct(S[f"{arm}|MH|r{n}"]["connected_fraction_mean"]))
            out[(arm, n)] = v
            both += v
            # cited in the legend: two-intruder episodes never connected in any round
            zeros = {("cbaa", 2): 23, ("independent", 2): 22}
            if (arm, n) in zeros:
                assert sum(x == 0 for x in v) == zeros[(arm, n)]
        nums.assert_macro(f"numConn{tag}", nums.pct(S[f"{arm}|MH|rboth"]["connected_fraction_mean"]))
        assert abs(np.mean(both) - S[f"{arm}|MH|rboth"]["connected_fraction_mean"]) < 6e-4
    return out
