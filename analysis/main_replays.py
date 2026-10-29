"""Measures taken while the main study's 900 episodes are re-simulated from their cached answers.

Each episode is rebuilt through ``analysis.replay`` and must reproduce its stored record before any measure is
used. Measures per episode:

  comm   ``Simulation.comm_metrics()``: the realised connectivity of the defenders' network (the study
         records do not store it). Read for every condition; the limited-range condition is the one of interest.
  live   for an episode that ended at the horizon with no breach and an intruder still active: each live
         intruder's last 10 s, classified as stand-off (the intruder evades on at least half the steps and a
         defender is within 6 units on at least half), approach (otherwise, and its distance to the region
         fell by more than 1 unit) or other.
  filt   the defenders' local safety-filter calls: d_box is the change by the box clip of the MPC acceleration,
         d_bar the change by the barrier rows; "changed" means more than 0.01. Counted by role group (pursuer
         with an installed target, pursue role without an installed target, other roles) and, for barrier
         changes, by the binding row kinds (intruder, teammate, wreck); intruder rows also by whether any
         active defender believed in that intruder.

    python3 -m analysis.main_replays --data DATA [--out OUT] [--workers 8] [--limit N]

Writes OUT/main_replays.json.
"""
import os
from collections import Counter

import numpy as np

from analysis import common, replay

WINDOW = 100  # steps = 10 s
COMM_KEYS = ("rounds", "connected_fraction", "mean_lcc_fraction", "mean_n_components", "n_partitions",
             "mean_partition_steps", "max_partition_steps", "partitioned_fraction")


class _State:
    """Read-only instrumentation of one simulation."""

    def __init__(self, tb, orig_fa, filt, rows):
        self.tb, self.orig_fa, self.filt, self.rows = tb, orig_fa, filt, rows

    def close(self):
        self.tb.filter_agent = self.orig_fa


def setup(e, sim):
    import simulation.testbed as tb
    from control.local_cbf import pair_row

    # filter instrumentation (read-only wrappers)
    ctx = {}
    filt = Counter()
    orig_bn, orig_rn = sim._local_blue_neighbors, sim._local_red_neighbors

    def bn(i, snap):
        ctx["step"], ctx["snap"], ctx["who"] = len(sim.history_blues) - 1, snap, ("blue", i)
        return orig_bn(i, snap)

    def rn(j, snap):
        ctx["who"] = ("red", j)
        return orig_rn(j, snap)
    sim._local_blue_neighbors, sim._local_red_neighbors = bn, rn
    orig_fa = tb.filter_agent

    def fa(p, v, u_nom, neighbors, cfg):
        r = orig_fa(p, v, u_nom, neighbors, cfg)
        team, idx = ctx.get("who", ("?", -1))
        if team != "blue":
            return r
        p_, v_, un = np.asarray(p, float), np.asarray(v, float), np.asarray(u_nom, float)
        lo = np.maximum(-cfg.a_max, (-cfg.v_max - v_) / cfg.dt)
        hi = np.minimum(cfg.a_max, (cfg.v_max - v_) / cfg.dt)
        u_box = np.clip(un, lo, hi)
        d_box, d_bar = float(np.linalg.norm(u_box - un)), float(np.linalg.norm(r.u - u_box))
        role = sim.role_assignment.get(idx)
        role = "defend(default)" if role is None else role.value
        if role == "neutralize":
            grp = "pursuer_installed" if idx in ctx["snap"]["pursued"] else "neutralize_no_target"
        else:
            grp = "other_role"
        filt[f"{grp}|calls"] += 1
        filt[f"path|{r.path}"] += 1
        if d_bar > 0.01:
            kinds, belief = set(), set()
            for nb in neighbors:
                g, b = pair_row(p_, v_, nb.position, nb.velocity, nb.d_safe, cfg.gamma, cfg.dt)
                if nb.share * b - g @ r.u >= -1e-6 and nb.share * b - g @ u_box > 1e-9:
                    k = nb.key[0]
                    if k == "blue":
                        k = "teammate" if nb.share < 1 else "wreck"
                    kinds.add(k)
                    if k == "red":
                        rid = nb.key[1]
                        belief.add("believed" if any(rid in sim.belief_stores[b2].targets
                                                      for b2 in range(sim.num_blues)
                                                      if not sim.blue_disabled[b2]) else "unbelieved")
            filt[f"{grp}|barrier"] += 1
            filt[f"{grp}|barrier|{'red' if 'red' in kinds else ('teammate_wreck' if kinds else 'unclassified')}"] += 1
            for x in belief:
                filt[f"{grp}|barrier_red|{x}"] += 1
        elif d_box > 0.01:
            filt[f"{grp}|box_only"] += 1
        else:
            filt[f"{grp}|unchanged"] += 1
        return r
    tb.filter_agent = fa

    # end-of-horizon snapshots
    rows = []
    orig = sim.solver.solve_decentralized
    horizon = sim.sim_steps

    def w(*a, **k):
        s = len(sim.history_blues) - 1
        if s >= horizon - WINDOW:
            rows.append({"s": s, "red": sim.red_state.copy(), "blue": sim.x_current.copy(),
                         "roles": {j: r.value for j, r in sim.red_role_assignment.items()},
                         "dis": list(sim.blue_disabled),
                         "pursued": {j: any(t == j and not sim.blue_disabled[i]
                                            and sim.role_assignment.get(i) is not None
                                            and sim.role_assignment.get(i).value == "neutralize"
                                            for i, t in sim.blue_target_red_idx.items())
                                     for j in range(sim.num_reds)}})
        return orig(*a, **k)
    sim.solver.solve_decentralized = w
    return _State(tb, orig_fa, filt, rows)


def extract(e, sim, st):
    from coordination.tasks import closest_point_on_polygon

    rows, filt = st.rows, st.filt
    horizon = sim.sim_steps
    o = sim.episode_outcome()
    m = sim.comm_metrics()
    live = []
    if not o["touchdown"] and "active" in o["red_outcomes"] and len(sim.history_blues) - 1 >= horizon:
        poly, rx, bx = sim.defended_shape_points, sim.red_nx, sim.blue_nx
        for j in range(sim.num_reds):
            if sim.red_outcomes[j] != "active":
                continue
            ev = near = pur = 0
            dpoly = []
            for r in rows:
                pj = r["red"][j * rx: j * rx + 2]
                ev += r["roles"].get(j) == "evade"
                act = [i for i in range(sim.num_blues) if not r["dis"][i]]
                dmin = min((float(np.linalg.norm(pj - r["blue"][i * bx: i * bx + 2])) for i in act), default=np.inf)
                near += dmin < 6.0
                pur += bool(r["pursued"].get(j))
                dpoly.append(float(np.linalg.norm(pj - closest_point_on_polygon(pj, poly))))
            n = max(1, len(rows))
            so = ev / n >= 0.5 and near / n >= 0.5
            ap = (not so) and bool(dpoly) and dpoly[-1] < dpoly[0] - 1.0
            live.append({"red": j, "class": "stand-off" if so else ("approach" if ap else "other"),
                         "evade_share": ev / n, "blue_within_6_share": near / n, "pursued_share": pur / n,
                         "dist_to_region_end": dpoly[-1] if dpoly else None})
    return {"comm": {k: m.get(k) for k in COMM_KEYS}, "live": live, "filt": dict(filt)}


def main(data, out_path, workers=4, limit=None):
    return replay.collect(data, common.MAIN, extract, workers, limit, setup=setup, out_path=out_path,
                          note="main study, re-simulated from the cache; see the module docstring")


if __name__ == "__main__":
    ap = common.parser(__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    res = main(a.data, os.path.join(common.out_dir(a), "main_replays.json"), a.workers, a.limit)
    print(f"{res['n']} episodes, {res['n_reproduced']} reproduced")
