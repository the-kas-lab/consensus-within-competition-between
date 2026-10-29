"""Where the figure scripts read from and write to, from the command line.

    --data     the data deposit (episode records)
    --out      the analysis results (default <data>/analysis): the JSON files of the analysis scripts and the
               macro files numbers.tex and si_numbers.tex
    --figures  where the figures are written (default <out>/figures)
    --bird-drone-dir  the four bird-drone photographs bird_drone_1.jpg ... bird_drone_4.jpg (Fig. 7 only; not
               part of this repository)
"""
import argparse
import json
import os
from pathlib import Path
from typing import List

from analysis import common

_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("--data", required=True)
_ap.add_argument("--out", default=None)
_ap.add_argument("--figures", default=None)
_ap.add_argument("--bird-drone-dir", default=None)
_ARGS, _REST = _ap.parse_known_args()

DATA = Path(_ARGS.data)
AN = Path(_ARGS.out or DATA / "analysis")
FIG = Path(_ARGS.figures or AN / "figures")
BIRD_DRONE_DIR = Path(_ARGS.bird_drone_dir) if _ARGS.bird_drone_dir else None
REPO = Path(__file__).resolve().parent.parent


def records(*runs: str) -> List[dict]:
    """Stored records of runs named "<study>/<run>"."""
    return common.records(str(DATA), [tuple(r.split("/")) for r in runs])


def analysis(name: str) -> dict:
    """A JSON file of the analysis results."""
    return json.loads((AN / name).read_text())


MAIN_RUNS = ("main_study/seeds_4001_4010", "main_study/seeds_4011_4030")
