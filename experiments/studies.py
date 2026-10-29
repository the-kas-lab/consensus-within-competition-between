"""The reported studies: which episodes, which models, and where their records live.

Each study is a list of *runs*. A run is a directory under the data directory:

    <data>/<study>/<run>/records.jsonl                 one record per episode
    <data>/<study>/<run>/records.jsonl.identity.json   the execution identity
    <data>/<study>/<run>/cache/<identity>/<episode>/   every model answer, by request hash

and an ``entries()`` function that lists the run's episodes (arm, intruders, network
condition, seed, ...). ``experiments.run`` replays or runs them.

Two model profiles are used (``experiments/profiles.json``): ``low-cost`` for the
defenders' and intruders' calls in every study, and ``near-frontier`` where a study
puts that model in the commander's or the defenders' seat (``entry["near_frontier"]``).
"""

import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from scripts.inference_profile import DESIGN, InferenceProfile, load_profiles
from scripts.run_competing_teams import episode_id

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROFILES = os.path.join(HERE, "profiles.json")
STIMULI = os.path.join(ROOT, "assets", "contact_photos")

SEEDS_FIRST = tuple(range(4001, 4011))
SEEDS_SECOND = tuple(range(4011, 4031))
SEEDS_ALL = SEEDS_FIRST + SEEDS_SECOND
SEEDS_AIRSPACE = tuple(range(4001, 4006))
# Network conditions of the main study: ideal network, limited range, independent
# loss, burst loss (unattacked) and falsified reports on the ideal network.
CONDITIONS = (("REF", False), ("MH", False), ("LI", False), ("LB", False), ("REF", True))
# Seeds whose first near-frontier high-authority runs failed under a provider's rate
# limit; they were run again as a second run before any outcome was read.
SEEDS_NEAR_FRONTIER_RERUN = (4019, 4020, 4021, 4025, 4026, 4028, 4029, 4030)

# Red-array order of an airspace episode: four intruders, one bird, two civilian drones.
KINDS = ["drone"] * 4 + ["bird"] + ["civilian"] * 2
AIRSPACE_ARMS = {"B": "cbaa", "CMD": "commander", "W": "cbaa_sw2", "SOLCMD": "commander",
                 "WSOL": "cbaa_sw2"}
NEAR_FRONTIER_SEAT = {"SOLCMD": "commander", "WSOL": "defender"}
AIRSPACE_TAG = "b1c2.img2"
DISGUISE_TAG = "b1c2.img2.camo1"
# The prompt version of each disguise condition: without and with the forewarning.
DISGUISE_CONDITIONS = {"U": "img.1", "W": "img.1w"}
BIRD_DRONE_FILES = ("bird_drone_1.jpg", "bird_drone_2.jpg", "bird_drone_3.jpg", "bird_drone_4.jpg")


def profile(name: str) -> InferenceProfile:
    return load_profiles(PROFILES)[name]


def _entry(arm, reds, condition, attacked, seed, rep, **extra) -> Dict:
    e = {"arm": arm, "reds": reds, "family": "multi_bearing", "condition": condition,
         "attacked": attacked, "seed": seed, "rep": rep, "design": DESIGN, **extra}
    e["episode_id"] = episode_id(e)
    return e


# -- main study and the high-authority arm -------------------------------------------------

def main_entries(seeds) -> List[Dict]:
    return [_entry(arm, reds, cond, attacked, seed, "matrix")
            for seed in seeds for cond, attacked in CONDITIONS
            for arm in ("cbaa", "independent", "commander") for reds in (2, 4)]


def high_authority_entries(seeds) -> List[Dict]:
    """The consensus team at high authority, low-cost model, every network condition."""
    return [_entry("cbaa_sw2", reds, cond, attacked, seed, "sweep")
            for seed in seeds for cond, attacked in CONDITIONS for reds in (2, 4)]


def near_frontier_high_authority_entries(seeds, rep) -> List[Dict]:
    """The consensus team at high authority with the near-frontier model writing the
    defenders' weights: ideal network, four intruders."""
    return [_entry("cbaa_sw2", 4, "REF", False, seed, rep, near_frontier="defender")
            for seed in seeds]


def near_frontier_commander_entries(seeds) -> List[Dict]:
    """The commander team with the near-frontier model as commander, low-cost intruders."""
    return [_entry("commander", reds, "REF", attacked, seed, "c2sol", near_frontier="commander")
            for seed in seeds for attacked in (False, True) for reds in (2, 4)]


# -- airspace challenge and disguised intruders ------------------------------------------

def photo_pools() -> Dict[str, List[str]]:
    return {"drone": [f"{STIMULI}/intruder_{i}.jpeg" for i in (1, 2, 3)],
            "bird": [f"{STIMULI}/bird_{i}.jpeg" for i in (1, 2, 3)],
            "civilian": [f"{STIMULI}/civilian_{i}.jpeg" for i in (1, 2, 3)]}


def airspace_images(seed_index: int, bird_drones: Optional[List[str]] = None) -> Dict[int, str]:
    """The photograph of each contact (red-array index -> file): balanced and
    deterministic, the same in every arm. The k-th contact of a kind on the s-th seed
    shows photo (s * n_kind + k) mod 3. With ``bird_drones`` (four photographs of
    bird-shaped drones) the four intruders show photo (s + k) mod 4 instead."""
    pools, n = photo_pools(), Counter(KINDS)
    seen, out = Counter(), {}
    for j, kind in enumerate(KINDS):
        if kind == "drone" and bird_drones is not None:
            out[j] = bird_drones[(seed_index + seen[kind]) % len(bird_drones)]
        else:
            pool = pools[kind]
            out[j] = pool[(seed_index * n[kind] + seen[kind]) % len(pool)]
        seen[kind] += 1
    return out


def _contacts(images) -> Dict:
    return {"birds": 1, "civilians": 2, "motion": "sector_waypoints.1", "images": images}


def airspace_entries(seeds=SEEDS_AIRSPACE) -> List[Dict]:
    out = []
    for s_idx, seed in enumerate(seeds):
        images = airspace_images(s_idx)
        for label, arm in AIRSPACE_ARMS.items():
            extra = {"near_frontier": NEAR_FRONTIER_SEAT[label]} if label in NEAR_FRONTIER_SEAT else {}
            out.append(_entry(arm, 4, "REF", False, seed, f"img-{label}", decoys=_contacts(images),
                              decoy_tag=AIRSPACE_TAG, **extra))
    return out


def disguise_entries(bird_drone_dir: str, seeds=SEEDS_AIRSPACE) -> List[Dict]:
    """Airspace episodes whose four intruders are shown as photographs of bird-shaped
    drones, without (U) and with (W) the forewarning that intruders may be disguised.
    ``bird_drone_dir`` holds the four photographs (``BIRD_DRONE_FILES``); they are not
    part of this repository."""
    photos = [os.path.join(bird_drone_dir, f) for f in BIRD_DRONE_FILES]
    out = []
    for s_idx, seed in enumerate(seeds):
        images = airspace_images(s_idx, photos)
        for cond, mode in DISGUISE_CONDITIONS.items():
            for label, arm in AIRSPACE_ARMS.items():
                extra = {"near_frontier": NEAR_FRONTIER_SEAT[label]} if label in NEAR_FRONTIER_SEAT else {}
                if mode != "img.1":
                    extra["image_mode"] = mode
                out.append(_entry(arm, 4, "REF", False, seed, f"camo-{cond}-{label}",
                                  decoys=_contacts(images), decoy_tag=DISGUISE_TAG, **extra))
    return out


# -- registry ------------------------------------------------------------------------

@dataclass(frozen=True)
class Run:
    study: str
    name: str
    entries: Callable[..., List[Dict]]
    needs_bird_drones: bool = False

    @property
    def subdir(self) -> str:
        return os.path.join(self.study, self.name)


RUNS: Tuple[Run, ...] = (
    Run("main_study", "seeds_4001_4010", lambda: main_entries(SEEDS_FIRST)),
    Run("main_study", "seeds_4011_4030", lambda: main_entries(SEEDS_SECOND)),
    Run("high_authority_low_cost", "seeds_4001_4030", lambda: high_authority_entries(SEEDS_ALL)),
    Run("high_authority_near_frontier", "first_run",
        lambda: near_frontier_high_authority_entries(SEEDS_ALL, "solw")),
    Run("high_authority_near_frontier", "second_run",
        lambda: near_frontier_high_authority_entries(SEEDS_NEAR_FRONTIER_RERUN, "solw-r2")),
    Run("near_frontier_commander", "seeds_4001_4010",
        lambda: near_frontier_commander_entries(SEEDS_FIRST)),
    Run("near_frontier_commander", "seeds_4011_4030",
        lambda: near_frontier_commander_entries(SEEDS_SECOND)),
    Run("airspace", "all_arms", airspace_entries),
    Run("disguise", "all_arms", disguise_entries, needs_bird_drones=True),
)
STUDIES = tuple(dict.fromkeys(r.study for r in RUNS))


def runs_of(study: str) -> List[Run]:
    rs = [r for r in RUNS if r.study == study]
    if not rs:
        raise KeyError(f"unknown study {study!r}; expected one of {STUDIES}")
    return rs


def stored_identity(run_dir: str) -> Optional[Dict]:
    p = os.path.join(run_dir, "records.jsonl.identity.json")
    return json.load(open(p)) if os.path.exists(p) else None


def stored_records(run_dir: str) -> List[Dict]:
    p = os.path.join(run_dir, "records.jsonl")
    return [json.loads(line) for line in open(p)] if os.path.exists(p) else []
