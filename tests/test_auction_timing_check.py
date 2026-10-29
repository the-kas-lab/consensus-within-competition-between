"""The auction-timing check's instrumentation (scripts/auction_smoke.py,
``--grid timing``).

Three things must hold before its numbers mean anything:

1. the smoke's existing Exp 4 grid builds exactly the cfg it always did, so
   extending the cell handling cannot have changed 18's or 21's runs;
2. the 17 C3 relay classifier moves an entry one hop per round and only through
   agents that believe in the task -- checked on hand-built graphs, where the
   right answer is known;
3. on a live Simulation, the wrappers agree with the shipped code's own
   counters (unsettled agent-epochs, no-commitment agent-steps).
"""

import pytest

from scripts.episode_builder import _Quiet, build_episode
from scripts.auction_smoke import (
    BASE_CFG,
    BELIEF_MODE,
    CELLS,
    TIMING_CELLS,
    attach_ledger,
    attach_timing_log,
    cell_config,
    classify_relay,
    measure_timing,
    reachable,
)

NEUTRALIZE_5 = "NEUTRALIZE:5:0"


@pytest.mark.parametrize("cell", sorted(CELLS))
@pytest.mark.parametrize("family", ["symmetric", "multi_bearing"])
@pytest.mark.parametrize("heading", ["inherited", "outward"])
def test_exp4_cells_build_the_cfg_they_always_did(cell, family, heading):
    # The expression run_one used before the timing grid existed.
    before = (BELIEF_MODE, dict(BASE_CFG, family=family, station_heading=heading, **CELLS[cell]))
    assert cell_config(CELLS[cell], family, heading) == before


class _Built(Exception):
    pass


@pytest.mark.parametrize("grid,cell", [(None, c) for c in sorted(CELLS)]
                         + [(TIMING_CELLS, "r20_p0.1_net"), (TIMING_CELLS, "rinf_p0.5_none")])
def test_run_one_passes_the_cell_config_to_build_episode(monkeypatch, grid, cell):
    # The call site: what run_one actually hands the one episode definition.
    import scripts.auction_smoke as smoke

    seen = {}

    def stub(seed, belief_mode, cfg):
        seen.update(seed=seed, belief_mode=belief_mode, cfg=cfg)
        raise _Built

    monkeypatch.setattr(smoke, "build_episode", stub)
    with pytest.raises(_Built):
        smoke.run_one(3705, "symmetric", cell, "outward", cells=grid)
    expected = cell_config((grid or CELLS)[cell], "symmetric", "outward")
    assert (seen["belief_mode"], seen["cfg"]) == expected
    if grid is None:
        assert expected == (BELIEF_MODE, dict(BASE_CFG, family="symmetric",
                                              station_heading="outward", **CELLS[cell]))


def test_timing_cells_carry_their_own_belief_mode_and_comms():
    belief, cfg = cell_config(TIMING_CELLS["r15_p0.1_net"], "symmetric", "outward")
    assert belief == "networked"
    assert "belief_mode" not in cfg  # popped, not passed on to build_episode
    assert (cfg["r_comm"], cfg["comm_interval_seconds"], cfg["cbaa_transport"]) == (15.0, 0.1, "network")
    # The spec dict itself is not mutated by the pop.
    assert TIMING_CELLS["r15_p0.1_net"]["belief_mode"] == "networked"


# -- reachability ------------------------------------------------------------

LINE = [(0, 1), (1, 2)]


def test_one_hop_per_round():
    # Two hops need two rounds even when both edges exist in the first. An
    # in-place update of the known set would let 0 -> 1 -> 2 cross in one.
    assert not reachable(0, 2, [LINE], relays={0, 1, 2})
    assert reachable(0, 2, [LINE, LINE], relays={0, 1, 2})


def test_a_non_believer_does_not_relay():
    # 17 C3: 0 and 2 believe, 1 does not. Connected as a whole, never through
    # believers.
    rounds = [LINE] * 5
    assert not reachable(0, 2, rounds, relays={0, 2})
    assert reachable(0, 2, rounds, relays={0, 1, 2})


def test_order_of_rounds_matters():
    # A time-respecting path: 1-2 must exist AFTER 0-1, not before.
    assert not reachable(0, 2, [[(1, 2)], [(0, 1)]], relays={0, 1, 2})
    assert reachable(0, 2, [[(0, 1)], [(1, 2)]], relays={0, 1, 2})


def _epoch(believers, bids):
    return {
        "tasks": {i: ({NEUTRALIZE_5} if i in believers else set()) | {"DEFEND:0"}
                  for i in range(3)},
        "bids": {i: {NEUTRALIZE_5: bids.get(i, 0.0), "DEFEND:0": 0.0} for i in range(3)},
    }


def _rounds(*edge_sets):
    """(step, edges) per round, steps 1, 2, ..."""
    return [(n + 1, edges) for n, edges in enumerate(edge_sets)]


def test_classify_relay_failure():
    counts = classify_relay(_epoch({0, 2}, {0: 5.0, 2: 3.0}), _rounds(*[LINE] * 4), [0, 1, 2],
                            {NEUTRALIZE_5: [0, 2]})
    assert counts["relay_failure"] == 1
    assert counts["no_path_in_time"] == 0
    assert counts["structural_relay_opportunity"] == 1


def test_classify_no_path_in_time():
    counts = classify_relay(_epoch({0, 2}, {0: 5.0, 2: 3.0}), _rounds([(0, 1)]), [0, 1, 2],
                            {NEUTRALIZE_5: [0, 2]})
    assert counts["relay_failure"] == 0
    assert counts["no_path_in_time"] == 1
    assert counts["believer_pair_no_path"] == 1


def test_classify_uses_the_winner_to_loser_direction():
    # 2 outbids 0; 2's entry can reach 0 through believer 1 (edge 2-1 then 1-0),
    # but 0's cannot reach 2 in time. The loser (0) heard the winner, so this
    # is flagged as the impossible case, not as a relay failure.
    rounds = _rounds([(1, 2)], [(0, 1)])
    counts = classify_relay(_epoch({0, 1, 2}, {0: 3.0, 2: 5.0}), rounds, [0, 1, 2],
                            {NEUTRALIZE_5: [0, 2]})
    assert counts["path_via_believers_still_conflicted"] == 1
    assert counts["relay_failure"] == 0


def test_a_claim_travels_only_from_when_it_was_made():
    # Complete graph every round, but the winner (0) took the task only in the
    # last round (step 3): it never sent that claim, so no path in time -- not
    # the impossible case. Found by the dry run on spent seed 2001.
    k3 = [(0, 1), (0, 2), (1, 2)]
    epoch = _epoch({0, 1, 2}, {0: 5.0, 2: 3.0})
    late = classify_relay(epoch, _rounds(k3, k3, k3), [0, 1, 2], {NEUTRALIZE_5: [0, 2]},
                          claim_since={0: 4, 2: 1})
    assert late["no_path_in_time"] == 1
    assert late["path_via_believers_still_conflicted"] == 0
    early = classify_relay(epoch, _rounds(k3, k3, k3), [0, 1, 2], {NEUTRALIZE_5: [0, 2]},
                           claim_since={0: 1, 2: 1})
    assert early["path_via_believers_still_conflicted"] == 1


def test_station_and_recon_conflicts_are_never_relay_failures():
    # Every agent generates every DEFEND/RECON task, so there are no
    # non-believers to fail to relay them.
    counts = classify_relay(_epoch({0, 2}, {}), _rounds(*[LINE] * 4), [0, 1, 2], {"DEFEND:0": [0, 2]})
    assert counts["relay_failure"] == 0 and counts["no_path_in_time"] == 0


# -- live Simulation ----------------------------------------------------------


@pytest.mark.parametrize("cell,family,steps", [
    ("r15_p0.1_net", "multi_bearing", 120),
    # Full length for the same reason: "none" conflicts on contact tasks exist
    # here, and the relay classifier must NOT read the belief graph as an
    # auction graph (the dry run's second defect).
    ("rinf_p0.1_none", "multi_bearing", 400),
    # Full length: on this seed contacts arrive late, and only then do
    # boundary conflicts on contact tasks exist for the relay classifier. The
    # dry run found its claim-time defect here.
    ("rinf_p0.5_net", "symmetric", 400),
])
def test_wrappers_agree_with_the_shipped_counters(cell, family, steps):
    belief, cfg = cell_config(TIMING_CELLS[cell], family, "outward")
    cfg["steps"] = steps
    sim = build_episode(2001, belief, cfg)
    ledger = attach_ledger(sim)
    log = attach_timing_log(sim)
    with _Quiet():
        sim.run()
    t = measure_timing(sim, log, ledger)

    assert t["agent_epochs_sampled"] > 0
    assert t["unsettled_at_epoch_end"] == t["runtime_unsettled_at_epoch_end"]
    assert t["agent_steps_no_commitment"] == t["ledger_agent_steps_no_commitment"]
    # Rounds are recorded only where a Network exists, one per comm step.
    assert len(log["rounds"]) == sim.comm_stats["rounds"]
    assert t["relay"].get("path_via_believers_still_conflicted", 0) == 0
    # The communication-round schedule: the runtime's own commitment-conflict count is the wrapper's.
    alloc = sim.episode_outcome()["allocation"]
    assert alloc["commitment_conflicted_task_epochs"] == sum(
        t["commitment_conflicts_by_kind"].values())
    if TIMING_CELLS[cell]["cbaa_transport"] == "none":
        assert not t["relay_applicable"] and t["relay"] == {}
        assert t["x_conflicts_by_kind"].get("NEUTRALIZE", 0) + t["x_conflicts_by_kind"].get("SCREEN", 0) > 0
    if cell == "rinf_p0.5_net":
        # Non-vacuous: the classifier actually ran on contact-task conflicts.
        assert t["relay"].get("no_path_in_time", 0) > 0
