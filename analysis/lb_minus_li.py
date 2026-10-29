"""LB - LI breach differences per arm and their between-arm interactions (main study, 30 seeds).

Seed-level paired differences pooled over both intruder counts (2 pairs per seed), 95 % t interval (29 df).
Secondary, unadjusted.

    python3 -m analysis.lb_minus_li --data DATA [--out OUT]       ->  OUT/lb_minus_li.json
"""
import os
from collections import defaultdict

import numpy as np

from analysis import common

T = 2.045229642132703  # t(0.975, 29)


def summ(d):
    d = np.asarray(d, float)
    m = d.mean()
    h = T * d.std(ddof=1) / np.sqrt(len(d))
    return {"estimate": round(float(m), 4), "ci95_t_seeds": [round(float(m - h), 4), round(float(m + h), 4)],
            "seeds_pos_neg_zero": [int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())]}


def main(data):
    B = defaultdict(dict)
    for r in common.records(data, common.MAIN):
        B[(r["arm"], r["reds"], common.cond(r))][r["seed"]] = int(r["outcome"]["touchdown"])
    seeds = sorted(B[("cbaa", 2, "REF")])
    assert len(seeds) == 30

    def per(a, c1, c0):
        return [np.mean([B[(a, s, c1)][sd] - B[(a, s, c0)][sd] for s in (2, 4)]) for sd in seeds]

    out = {"note": __doc__.strip().split("\n")[0], "arms": {}, "interactions": {}}
    for a in ("commander", "cbaa", "independent"):
        out["arms"][a] = {"LB - LI": summ(per(a, "LB", "LI")),
                          "breaches": {f"r{s}|{c}": sum(B[(a, s, c)].values()) for s in (2, 4) for c in ("REF", "LI", "LB")},
                          "same_outcome_pairs": int(sum(B[(a, s, "LI")][sd] == B[(a, s, "LB")][sd]
                                                        for s in (2, 4) for sd in seeds))}
    for a, b in (("commander", "cbaa"), ("independent", "cbaa"), ("commander", "independent")):
        out["interactions"][f"(LB - LI): {a} - {b}"] = summ(np.array(per(a, "LB", "LI")) - np.array(per(b, "LB", "LI")))
    return out


if __name__ == "__main__":
    a = common.parser(__doc__.split("\n")[0]).parse_args()
    common.dump(main(a.data), os.path.join(common.out_dir(a), "lb_minus_li.json"))
