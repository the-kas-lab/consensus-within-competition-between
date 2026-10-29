"""Recorded end-to-end baselines: the regression net for everything after M1.

The belief layer (M2) rewrites how a controller's target is resolved -- see
 L1 and L2. That change is
invasive enough that "did anything else move?" is not answerable by inspection.
These baselines answer it.

The contract is deliberately blunt: ``belief_mode="ground_truth"`` and
``R_comm=inf`` must reproduce these fingerprints byte-for-byte. If they do not,
either the refactor changed behavior it should not have, or the change was
intended -- in which case re-record with ``--update-golden`` and the golden
diff becomes the written record of what moved and why.

Recorded at 678d8ab + M1 (OSQP verbosity fix, RNG streams in legacy
mode). Both M1 changes were verified byte-identical to pre-M1 behavior, so these
fingerprints also describe the codebase as it stood before any of this work.
"""

import json
import os

import pytest

from conftest import GOLDEN_DIR, build_simulation, fingerprint, run_quiet

GOLDEN_PATH = os.path.join(GOLDEN_DIR, "episodes.json")

# Chosen to span the branches that actually differ in the control path: RECON
# assignment on/off, single vs. multiple reds, and a team large enough that the
# blue-blue CBF has real pairwise constraints to enforce.
CASES = {
    "two_blue_one_red_recon": dict(num_blues=2, num_reds=1, seed=7, sim_steps=60),
    "four_blue_one_red_recon": dict(num_blues=4, num_reds=1, seed=130, sim_steps=60),
    "four_blue_two_red_recon": dict(num_blues=4, num_reds=2, seed=130, sim_steps=60),
    "four_blue_two_red_no_recon": dict(
        num_blues=4, num_reds=2, seed=42, sim_steps=60, recon_fraction=0.0, min_recon=0
    ),
}


def _build(case):
    params = dict(case)
    recon_fraction = params.pop("recon_fraction", 0.5)
    min_recon = params.pop("min_recon", 1)
    from planning.planner import RuleBasedBlueRolePlanner

    params["role_planner"] = RuleBasedBlueRolePlanner(
        recon_fraction=recon_fraction, min_recon=min_recon
    )
    return build_simulation(**params)


def _load():
    if not os.path.exists(GOLDEN_PATH):
        return {}
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def test_golden_episodes(update_golden):
    recorded = _load()
    actual = {name: fingerprint(run_quiet(_build(case))) for name, case in CASES.items()}

    if update_golden or not recorded:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(GOLDEN_PATH, "w") as f:
            json.dump(actual, f, indent=2, sort_keys=True)
            f.write("\n")
        if not recorded:
            pytest.skip(f"recorded initial baseline at {GOLDEN_PATH}; re-run to assert")
        pytest.skip("golden baselines re-recorded on request")

    mismatches = []
    for name in CASES:
        if name not in recorded:
            mismatches.append(f"{name}: missing from golden file (re-record)")
        elif recorded[name] != actual[name]:
            mismatches.append(
                f"{name}:\n    golden={recorded[name]}\n    actual={actual[name]}"
            )
    assert not mismatches, (
        "Episode behavior changed against the recorded baseline:\n"
        + "\n".join(mismatches)
        + "\n\nIf this change is intended, re-record with:\n"
          "  pytest tests/test_golden_episode.py --update-golden\n"
          "and commit the golden diff as the record of what moved."
    )


def test_golden_file_covers_every_case():
    """A case silently dropped from the golden file would make the regression
    net quietly smaller. Keep the two in sync."""
    recorded = _load()
    if not recorded:
        pytest.skip("no baseline recorded yet")
    assert set(recorded) == set(CASES)
