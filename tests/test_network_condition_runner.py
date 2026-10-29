"""scripts/network_conditions.py: the calibration rule, the calibration's blindness to
outcomes, and the resolved-target holding count.

 and Sec 9.4. The rule
is applied mechanically, so it is pinned on synthetic tables where every
branch is known (closest to target, the diameter eligibility, ties to the
larger range, the declared fallback). The holding count is checked on live
episodes against an independent derivation from the smoke's execution causes.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import network_conditions  # noqa: E402
from scripts.network_conditions import run_episode_record, select  # noqa: E402


def _rec(r, cf, diam, family="symmetric", arm="C"):
    return {"r_comm": r, "family": family, "arm": arm, "unsettled_at_epoch_end": 0,
            "allocation": {"commitment_conflicted_task_epochs": 0, "conflicted_task_epochs": 0},
            "comm": {"connected_fraction": cf, "mean_partition_steps": 0.0,
                     "mean_diameter_when_connected": diam}}


def _table(rows):
    """rows: {range: (connected_fraction, diameter)}, one record per cell."""
    return [_rec(r, cf, d, fam, arm) for r, (cf, d) in rows.items()
            for fam in ("symmetric", "multi_bearing") for arm in ("I", "C")]


def test_the_rule_picks_closest_to_090_and_050():
    sel = select(_table({8.0: (0.30, 2.5), 10.0: (0.48, 2.4), 12.0: (0.70, 2.1),
                         15.0: (0.88, 1.9), 20.0: (0.97, 1.4)}))
    assert sel["R_MH"] == 15.0 and sel["R_IP"] == 10.0
    assert not sel["fallback_needed"]


def test_r_mh_requires_multi_hop_paths():
    # 20 is closest to 0.90 but its paths are almost all one hop.
    sel = select(_table({8.0: (0.30, 2.5), 10.0: (0.50, 2.4), 12.0: (0.70, 2.1),
                         15.0: (0.80, 1.6), 20.0: (0.91, 1.2)}))
    assert sel["R_MH"] == 15.0
    assert 20.0 not in sel["R_MH_eligible"]


def test_ties_go_to_the_larger_range():
    sel = select(_table({8.0: (0.45, 2.5), 10.0: (0.55, 2.4), 12.0: (0.85, 2.1),
                         15.0: (0.95, 1.9), 20.0: (0.99, 1.4)}))
    assert sel["R_IP"] == 10.0 and sel["R_MH"] == 15.0


def test_the_declared_fallback_is_flagged_when_nothing_is_below_065():
    rows = {8.0: (0.66, 2.5), 10.0: (0.75, 2.4), 12.0: (0.85, 2.1),
            15.0: (0.92, 1.9), 20.0: (0.99, 1.4)}
    assert select(_table(rows))["fallback_needed"]
    rows.update({6.0: (0.50, 2.8), 4.0: (0.20, 2.9)})
    sel = select(_table(rows))
    assert not sel["fallback_needed"] and sel["R_IP"] == 6.0


def test_the_statistic_weights_each_family_and_arm_cell_equally():
    recs = [_rec(10.0, 0.2, 2.0, "symmetric", "I"), _rec(10.0, 0.4, 2.0, "symmetric", "C"),
            _rec(10.0, 0.6, 2.0, "multi_bearing", "I"), _rec(10.0, 0.8, 2.0, "multi_bearing", "C"),
            _rec(10.0, 0.8, 2.0, "multi_bearing", "C")]
    assert select(recs)["table"]["10.0"]["connected_fraction"] == pytest.approx(0.5)


def test_run_refuses_until_the_ranges_are_frozen(monkeypatch):
    monkeypatch.setattr(network_conditions, "R_MH", None)
    with pytest.raises(SystemExit):
        network_conditions.episode_cfg("symmetric", "MH", "C")
    # REF, LI and LB need no calibrated range.
    assert network_conditions.episode_cfg("symmetric", "LB", "C")[1]["comm_loss_model"] == network_conditions.GE_03


def test_the_arms_differ_in_transport_only_and_nx_is_local_with_no_auction_traffic():
    for c in ("REF", "LI", "LB"):
        (bi, ci), (bc, cc) = (network_conditions.episode_cfg("multi_bearing", c, a) for a in ("I", "C"))
        assert bi == bc == "networked"
        diff = {k for k in set(ci) | set(cc) if ci.get(k) != cc.get(k)}
        assert diff == {"cbaa_transport"}
        assert ci["cbaa_fallback"] == "station" and ci["comm_interval_seconds"] == 0.1
    belief, nx = network_conditions.episode_cfg("symmetric", "NX", "N")
    assert belief == "local" and nx["cbaa_transport"] == "none"


def test_calibration_records_carry_no_mission_outcome():
    rec = run_episode_record(3705, "symmetric", "CAL", "C", r_comm=12.0, with_outcomes=False)
    for field in ("breach", "attrition_rate", "blue_score", "num_neutralized", "num_escaped"):
        assert field not in rec
    assert rec["comm"]["connected_fraction"] is not None


def test_resolved_target_holding_agrees_with_the_smokes_execution_causes():
    """Live episodes where committed contacts do resolve (22 Part 4 found
    SCREEN non-execution, all target_resolved, in this configuration). The
    smoke's ``cause()`` path classifies every NON-executed committed step on a
    resolved contact as target_resolved. The direct count, less the steps that
    were still executed, must equal it. And the count must be positive
    somewhere, or the check is vacuous."""
    held = []
    for seed in (3705, 3706, 3707):
        rec = run_episode_record(seed, "multi_bearing", "REF", "C", with_outcomes=False)
        h = rec["resolved_target_holding"]
        causes = sum(v for k, v in rec["not_executed_causes"].items()
                     if k.endswith(":target_resolved"))
        assert h["agent_steps"] + h["at_resolution_step"] - h["executed"] == causes
        assert sum(h["by_kind"].values()) == h["agent_steps"]
        held.append(h["agent_steps"])
    assert sum(held) > 0


def test_the_frozen_ranges_are_the_calibrated_ones():
    # 23 Amendments 4-5, the range-calibration rule. A change here is a design change, not a fix.
    assert (network_conditions.R_MH, network_conditions.R_IP) == (15.0, 14.0)
    assert network_conditions.conditions()["IPL"] == {"r_comm": 14.0, "comm_loss_model": network_conditions.IID_03}
