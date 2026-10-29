"""Time-limit endings of the other studies' episodes, measured while they are re-simulated from the cache.

For the table of every cell of the Supplementary Information. Groups:

  W            consensus team at high authority, low-cost model (300 episodes)
  WSOL         the same team with the near-frontier model, valid records only (30 episodes)
  SOLCMD_STD   commander team with the near-frontier model, seeds 4001-4010 (40 episodes)
  SOLCMD_EXT   the same, seeds 4011-4030 (80 episodes)

Each episode must reproduce its stored record. Measures: the time-limit classification of every live intruder
(last 10 s; stand-off = the intruder evades on at least half the steps and a defender is within 6 units on at
least half; approach = otherwise, and the distance to the region fell by more than 1 unit; other = neither)
and ``Simulation.comm_metrics()``.

    python3 -m analysis.endings_replays W|WSOL|SOLCMD_STD|SOLCMD_EXT --data DATA [--out OUT] [--workers 8]

Writes OUT/endings_<group>.json.
"""
import os

import numpy as np

from analysis import common, replay

WINDOW = 100  # steps = 10 s
COMM_KEYS = ("rounds", "connected_fraction", "mean_lcc_fraction", "mean_n_components", "n_partitions",
             "mean_partition_steps", "max_partition_steps", "partitioned_fraction")
GROUPS = {"W": common.HIGH_AUTHORITY_LOW_COST, "WSOL": common.HIGH_AUTHORITY_NEAR_FRONTIER,
          "SOLCMD_STD": common.NEAR_FRONTIER_COMMANDER[:1], "SOLCMD_EXT": common.NEAR_FRONTIER_COMMANDER[1:]}


def setup(e, sim):
    rows = []
    orig = sim.solver.solve_decentralized
    horizon = sim.sim_steps

    def w(*a, **k):
        s = len(sim.history_blues) - 1
        if s >= horizon - WINDOW:
            rows.append({"s": s, "red": sim.red_state.copy(), "blue": sim.x_current.copy(),
                         "roles": {j: r.value for j, r in sim.red_role_assignment.items()},
                         "dis": list(sim.blue_disabled)})
        return orig(*a, **k)
    sim.solver.solve_decentralized = w
    return rows


def extract(e, sim, rows):
    from coordination.tasks import closest_point_on_polygon

    horizon = sim.sim_steps
    o = sim.episode_outcome()
    m = sim.comm_metrics()
    live = []
    if not o["touchdown"] and "active" in o["red_outcomes"] and len(sim.history_blues) - 1 >= horizon:
        poly, rx, bx = sim.defended_shape_points, sim.red_nx, sim.blue_nx
        for j in range(sim.num_reds):
            if sim.red_outcomes[j] != "active":
                continue
            ev = near = 0
            dpoly = []
            for r in rows:
                pj = r["red"][j * rx: j * rx + 2]
                ev += r["roles"].get(j) == "evade"
                act = [i for i in range(sim.num_blues) if not r["dis"][i]]
                dmin = min((float(np.linalg.norm(pj - r["blue"][i * bx: i * bx + 2])) for i in act), default=np.inf)
                near += dmin < 6.0
                dpoly.append(float(np.linalg.norm(pj - closest_point_on_polygon(pj, poly))))
            n = max(1, len(rows))
            so = ev / n >= 0.5 and near / n >= 0.5
            ap = (not so) and bool(dpoly) and dpoly[-1] < dpoly[0] - 1.0
            live.append({"red": j, "class": "stand-off" if so else ("approach" if ap else "other"),
                         "dist_to_region_end": dpoly[-1] if dpoly else None})
    return {"comm": {k: m.get(k) for k in COMM_KEYS}, "live": live}


def main(group, data, out_path, workers=4, limit=None):
    only = None
    if group == "WSOL":
        only = {r["episode_id"] for r in common.valid_records(data, GROUPS[group])}
    return replay.collect(data, GROUPS[group], extract, workers, limit, setup=setup, out_path=out_path,
                          only=only, note=f"time-limit endings, group {group}, re-simulated from the cache")


if __name__ == "__main__":
    ap = common.parser(__doc__.split("\n")[0])
    ap.add_argument("group", choices=sorted(GROUPS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    res = main(a.group, a.data, os.path.join(common.out_dir(a), f"endings_{a.group}.json"), a.workers, a.limit)
    print(f"{res['n']} episodes, {res['n_reproduced']} reproduced")
