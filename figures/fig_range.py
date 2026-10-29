"""Fig. 5: limited range harmed the commander.

Message: message loss and falsified reports changed no team's breach rate detectably; limited range split the
decentralized teams' networks most, yet only the commander's breach rate rose, and its orders often did not
arrive.

a, change in breach rate from the ideal network under each other condition, by team (pooled over two and
   four intruders, 2 episode pairs per seed): per-seed changes as count-sized grey circles, mean and 95% CI over
   seeds in black, values as the text prints them.
b, connected rounds under limited range: per seed (its two episodes averaged), the share of communication
   rounds in which the active defenders formed one connected group.
c, the commander's task-changing orders that arrived, by network condition (per seed; mean over seeds is
   \\numTaskOrdersArrivedMh under limited range).

Sources: matrix_analysis.json contrasts "D1 MH|D2 LI|D3 LB|F1 REF+attack - REF | <team> | pooled over sizes"
(per_seed d_breach; estimate and CI asserted against the num<Team><Cond>PP(CI) macros); range_data.connectivity()
(asserts numConn*); fig_range.task_orders() from commander_missed_orders.json (re-simulation, 300 of 300
reproduced).

    python3 -m figures.fig_range --data DATA [--out OUT] [--figures DIR]
"""
from __future__ import annotations

import numpy as np

from figures import grammar, nums, paths, range_data, style

OUT = paths.FIG / "fig_range.pdf"
SEEDS = list(range(4001, 4031))
TEAMS = (("commander", "Cmd", "Commander"), ("cbaa", "Cbaa", "Consensus"), ("independent", "Ind", "Independent"))
CONDITIONS = (("D2 LI - REF", "Li", "Independent loss"), ("D3 LB - REF", "Lb", "Burst loss"),
              ("F1 REF+attack - REF", "Atk", "Falsified reports"), ("D1 MH - REF", "Mh", "Limited range"))
ORDER_CONDS = (("REF", False, "Ideal"), ("LI", False, "Independent loss"), ("LB", False, "Burst loss"),
               ("REF", True, "Falsified reports"), ("MH", False, "Limited range"))
W, H = style.W2, 92.0


def changes():
    a = paths.analysis("matrix_analysis.json")
    blocks = []
    for key, ctag, name in CONDITIONS:
        rows = []
        for arm, tag, label in TEAMS:
            c = a["contrasts"][f"{key} | {arm} | pooled over sizes"]
            assert c["pairs_per_seed"] == 2
            b = c["breach_difference"]
            d = np.array([s["d_breach"] for s in c["per_seed"]])
            nums.assert_seed_summary(d, b["estimate"], b["ci95_t_seeds"])
            m, (lo, hi) = nums.t_ci(d)
            nums.assert_macro(f"num{tag}{ctag}PP", nums.signed_pp(b["estimate"]))
            nums.assert_macro(f"num{tag}{ctag}PPCI", nums.pp_ci(*b["ci95_t_seeds"]))
            rows.append(dict(label=label, d=100 * d, m=100 * m, lo=100 * lo, hi=100 * hi,
                             text=f"{grammar.fmt_est(m)} ({grammar.fmt_ci(lo, hi)})"))
        blocks.append((name, rows))
    return blocks


def task_orders():
    """Per seed and pooled, the share of the commander's task-changing orders that arrived."""
    R = paths.analysis("commander_missed_orders.json")
    assert R["n"] == 300 and R["n_reproduced"] == 300, (R["n"], R["n_reproduced"])
    out = {}
    for cond, att, name in ORDER_CONDS:
        agg = {s: [0, 0] for s in SEEDS}
        for x in R["episodes"]:
            if x["condition"] == cond and x["attacked"] == att:
                agg[x["seed"]][0] += x["applied_changed"]
                agg[x["seed"]][1] += x["applied_changed"] + x["missed_changed"]
        assert all(b > 0 for _, b in agg.values()), name
        out[name] = [100 * a / b for a, b in agg.values()]
    nums.assert_macro("numTaskOrdersArrivedMh", nums.pct(np.mean(out["Limited range"]) / 100))
    assert min(out["Ideal"]) == 100
    return out


def main():
    style.apply()
    blocks = changes()
    conn = range_data.connectivity()
    orders = task_orders()
    fig = style.figure(W, H)

    # a: forest of condition changes
    ax = style.axes_mm(fig, 30.0, 10.0, 58.0, 74.0)
    y = grammar.forest(ax, blocks, (-112, 112), [-100, -50, 0, 50, 100], "Fewer breaches", "More breaches",
                       ends_every_block=False)
    ax.set_xlabel("Change in breach rate from the ideal network (percentage points)")
    grammar.mean_ci_key(ax, -0.45, -2.2)
    grammar.bubble_key(ax, 0.22, -2.2, label="Seeds")
    ax.set_ylim(y - 0.4, -2.6)
    style.panel_label(fig, 1.0, H - 1.0, "a")

    # b: connected rounds under limited range, by team
    bx = style.axes_mm(fig, 132.0, 52.0, 46.0, 33.0)
    bx.set_xlim(-0.6, 2.7)
    bx.set_ylim(-3, 100)
    bx.set_yticks([0, 50, 100])
    bx.set_ylabel("Connected rounds (%)")
    for k, (arm, _, _) in enumerate(TEAMS):
        v = [100 * (a + b) / 2 for a, b in zip(conn[(arm, 2)], conn[(arm, 4)])]
        grammar.swarm_dots(bx, k - 0.1, v, arm, half_width=0.32)
        m, (lo, hi) = nums.t_ci(v)
        grammar.mean_ci(bx, k + 0.3, m, max(lo, 0), min(hi, 100))
    bx.set_xticks([0.1, 1.1, 2.1], [n for _, _, n in TEAMS])
    bx.tick_params(axis="x", length=0)
    bx.spines["bottom"].set_visible(False)
    style.panel_label(fig, 122.0, H - 1.0, "b")

    # c: commander's task-changing orders that arrived, by condition
    cx = style.axes_mm(fig, 146.0, 10.0, 31.0, 30.0)
    cx.set_xlim(0, 103)
    cx.set_xticks([0, 50, 100])
    cx.set_xlabel("Commander's orders arrived (%)")
    names = [n for _, _, n in ORDER_CONDS]
    cx.set_ylim(len(names) - 0.4, -0.6)
    for k, n in enumerate(names):
        grammar.swarm_dots(cx, k - 0.12, orders[n], "commander", vertical=False, half_width=0.28)
        m, (lo, hi) = nums.t_ci(orders[n])
        grammar.mean_ci(cx, k + 0.28, m, max(lo, 0), min(hi, 100), vertical=False)
    cx.set_yticks(range(len(names)), names)
    cx.tick_params(axis="y", length=0)
    cx.spines["left"].set_visible(False)
    style.panel_label(fig, 122.0, 43.0, "c")
    style.save(fig, OUT)


if __name__ == "__main__":
    main()
