"""Station geometry for the limited-range sentence: how far apart the four defender stations are, and the radius
of the defended region, computed with the simulator's own geometry code (Simulation._build_perimeter_geometry
and DecentralizedPursuitEvasionSolver._build_defend_stations) on stub objects, so no Simulation is built and no
episode runs.

    python3 -m analysis.station_geometry [--out OUT]       ->  OUT/station_geometry.json
"""
import argparse
import itertools
import os
from types import SimpleNamespace

import numpy as np

from analysis import common
from simulation.decentralized_solver import DecentralizedPursuitEvasionSolver as Solver
from simulation.testbed import Simulation


def main():
    ARENA, BLUES, RANGE = 100.0, 4, 15.0  # main-study arena size, defenders per team, limited-range link length
    sim = SimpleNamespace(arena_size=ARENA, num_blues=BLUES, _sample_polygon_perimeter=Simulation._sample_polygon_perimeter)
    Simulation._build_perimeter_geometry(sim)
    stations = Solver._build_defend_stations(SimpleNamespace(defense_polygon=sim.defense_polygon, N_p=BLUES))
    pts = {k: np.asarray(v, dtype=float) for k, v in stations.items()}
    pairs = {f"{i}-{j}": float(np.linalg.norm(pts[i] - pts[j])) for i, j in itertools.combinations(sorted(pts), 2)}
    region_r = np.linalg.norm(np.asarray(sim.defended_shape_points) - sim.defense_center, axis=1)  # breach test shape
    out = {
        "arena_size": ARENA, "num_blues": BLUES, "range_units": RANGE,
        "stations": {str(k): pts[k].round(4).tolist() for k in sorted(pts)},
        "station_pair_distance": pairs,
        "station_pair_distance_min": min(pairs.values()), "station_pair_distance_max": max(pairs.values()),
        "station_pairs_within_range": sum(d < RANGE for d in pairs.values()),
        "region_radius_min": float(region_r.min()), "region_radius_max": float(region_r.max()),
    }

    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    common.dump(main(), os.path.join(ap.parse_args().out, "station_geometry.json"))
