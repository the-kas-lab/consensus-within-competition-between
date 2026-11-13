"""Run the whole analysis, in order, from a data deposit: statistics, the paper's and the Supplementary
Information's numbers, and the figures.

    python3 -m analysis.run_all --data DATA [--out OUT] [--workers 8] [--replays] [--figures] [--bird-drone-dir DIR]

Without ``--replays`` the results of the re-simulation stage are taken from the deposit (OUT must already hold
them: coverage_exposure_*.json, main_replays.json, behaviour_replays.json, commander_commitments.json,
commander_missed_orders.json, endings_*.json). With ``--replays`` they are recomputed first by re-simulating
every episode from its cached answers (analysis.replay_all; about 1,400 episodes, roughly two hours on eight
cores; no model is called).

Without the cached answers (only records_and_analysis.zip unpacked), the modules that count them (NEEDS_CACHE) are
skipped and their results are taken from the deposit, so every number and figure is still recomputed from those results.
"""
import os
import subprocess
import sys

from analysis import common

STATISTICS = ["matrix_stats", "high_authority", "high_authority_near_frontier", "airspace", "summary", "key_rows",
              "run_facts", "behaviour_pool", "replay_summary", "lb_minus_li", "station_geometry", "barrier_tests",
              "bid_weights"]
# modules that count the cached model answers, with the files they write
NEEDS_CACHE = {"airspace": ("airspace.json", "disguise.json"), "summary": ("summary.json",), "run_facts": ("run_facts.json",),
               "behaviour_pool": ("behaviour_pool.json",), "bid_weights": ("bid_weights.json",)}
NUMBERS = ["make_numbers", "make_si_numbers", "make_si_data"]
FIGURES = ["fig_architecture", "fig_arena", "fig_contrasts", "fig_allocation", "fig_range", "fig_authority",
           "fig_breach_grid", "fig_filter_headon"]


def run(module, *args):
    cmd = [sys.executable, "-m", module, *args]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = common.parser(__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--replays", action="store_true", help="recompute the re-simulation stage first")
    ap.add_argument("--figures", action="store_true", help="also draw the figures")
    ap.add_argument("--bird-drone-dir", default=None, help="the four bird-drone photographs (Fig. 7)")
    a = ap.parse_args()
    out = common.out_dir(a)
    base = ["--data", a.data, "--out", out]
    if a.replays:
        run("analysis.replay_all", *base, "--workers", str(a.workers))
    cached = common.has_cache(a.data)
    if not cached:
        print("The cached model answers are not unpacked: " + ", ".join(NEEDS_CACHE) + " are skipped and their results "
              "are taken from the deposit's analysis/ folder.", flush=True)
    for m in STATISTICS:
        if m in NEEDS_CACHE and not cached:
            for f in NEEDS_CACHE[m]:
                assert os.path.exists(os.path.join(out, f)), f"{f} is neither in {out} nor computable without the cache"
            continue
        if m in ("station_geometry", "barrier_tests"):
            run(f"analysis.{m}", "--out", out)
        else:
            run(f"analysis.{m}", *base)
    for m in NUMBERS:
        run(f"analysis.{m}", *base)
    if a.figures:
        for m in FIGURES:
            run(f"figures.{m}", *base)
        if a.bird_drone_dir:
            run("analysis.episode_bundles", *base, "fig7a_cbaa")
            run("figures.fig_airspace", *base, "--bird-drone-dir", a.bird_drone_dir)


if __name__ == "__main__":
    main()
