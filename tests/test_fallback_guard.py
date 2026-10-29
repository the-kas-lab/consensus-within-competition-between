"""The communication-round schedule.

1. **The station fallback.** An agent with no commitment yet executes an
   explicit DEFEND(station i) with the station-heading rule posture -- when asked. It is a
   declared fallback, not a commitment: the auction never sees it. The default
   stays the legacy implicit DEFEND, so every existing run is unchanged.
2. **The schedule guard.** A schedule whose MINIMUM rounds per epoch fall
   short of local stability is rejected unless overridden for diagnostics.
   Under a non-dividing period epochs hold
   different round counts, so an insufficient minimum does not mean that no
   epoch could ever settle. It certifies nothing, and it must not use a
   heuristic round budget.
3. **Commitment conflicts.** Duplicates in force are logged next to duplicate
   claims, sampled at the same boundary, because the two differ.
"""

import pytest

from coordination.cbaa import CBAAAgent, commitment_conflicts, conflicts
from coordination.runtime import CBAARuntime, output_for
from planning.roles import BlueRole
from scripts.episode_builder import _Quiet, build_episode
from scripts.auction_smoke import (
    TIMING_CELLS,
    attach_bearing_log,
    attach_ledger,
    cell_config,
    measure_fallback,
)
from simulation.testbed import Simulation, rounds_per_epoch_cycle


# -- 1. the fallback, at the method ---------------------------------------------


def test_legacy_default_is_the_implicit_defend():
    out = output_for({0: None, 1: "DEFEND:1"})
    assert out.roles[0] == BlueRole.DEFEND
    assert 0 not in out.defend_targets  # no override: the solver's implicit DEFEND
    assert out.defend_targets[1] == ("station", 1)


def test_station_fallback_is_the_agents_own_station():
    out = output_for({0: None, 2: None, 1: "SCREEN:7"}, fallback="station")
    assert out.roles[0] == out.roles[2] == BlueRole.DEFEND
    assert out.defend_targets[0] == ("station", 0)
    assert out.defend_targets[2] == ("station", 2)
    assert out.defend_targets[1] == ("contact", 7)  # a commitment is untouched


def test_unknown_fallback_is_refused():
    with pytest.raises(ValueError):
        output_for({0: None}, fallback="nearest")
    with pytest.raises(ValueError):
        CBAARuntime(fallback="nearest")


def test_the_fallback_is_not_a_commitment():
    # Rendering the fallback must not write it into auction state, or it would
    # enter the next epoch's switch cost as if it were a task held.
    rt = CBAARuntime(fallback="station")
    rt.agents = {i: CBAAAgent(i, stable_rounds=3) for i in range(3)}
    out = rt.output([0, 1, 2])
    assert all(out.defend_targets[i] == ("station", i) for i in range(3))
    assert rt.commitments([0, 1, 2]) == {0: None, 1: None, 2: None}


# -- 1. the fallback, on a live Simulation (runtime and testbed call sites) -----


@pytest.mark.parametrize("cell", ["rinf_p0.1_net", "rinf_p0.1_none"])
def test_uncommitted_agents_hold_their_own_station_facing_outward(cell):
    # Both transports: the fallback applies identically to independent
    # selection (Exp 4's arm 1) and to the auction.
    belief, cfg = cell_config(TIMING_CELLS[cell], "multi_bearing", "outward")
    cfg.update(steps=40, cbaa_fallback="station")
    sim = build_episode(2001, belief, cfg)
    ledger = attach_ledger(sim)
    bearing_log = attach_bearing_log(sim)
    with _Quiet():
        sim.run()
    f = measure_fallback(sim, ledger, bearing_log)

    n = f.get("uncommitted_agent_steps", 0)
    assert n > 0  # non-vacuous: the first epoch has unsettled steps
    assert f.get("recorded_own_station", 0) == n
    assert f.get("source:station", 0) == n  # never nearest_believed / screen
    assert f.get("heading:station_outward", 0) + f.get("heading:detection", 0) == n
    assert sim.episode_outcome()["allocation"]["fallback"] == "station"


def test_legacy_fallback_leaves_uncommitted_agents_without_an_override():
    belief, cfg = cell_config(TIMING_CELLS["rinf_p0.1_net"], "multi_bearing", "outward")
    cfg["steps"] = 40
    sim = build_episode(2001, belief, cfg)
    ledger = attach_ledger(sim)
    bearing_log = attach_bearing_log(sim)
    with _Quiet():
        sim.run()
    f = measure_fallback(sim, ledger, bearing_log)
    assert f.get("uncommitted_agent_steps", 0) > 0
    assert f.get("recorded_own_station", 0) == 0
    assert f.get("heading:station_outward", 0) == 0  # the station-heading rule never reaches it


# -- 2. the guard ---------------------------------------------------------------


def _sim(period, **kw):
    kw.setdefault("belief_mode", "networked")
    return Simulation(
        num_blues=4, num_reds=2, seed=2001, sim_steps=5, dt=0.1,
        planning_interval_seconds=1.0, allocation="cbaa",
        comm_interval_seconds=period, **kw,
    )


@pytest.mark.parametrize("period", [0.4, 0.5, 1.0, 5.0])
def test_impossible_schedules_are_rejected(period):
    with pytest.raises(ValueError, match="insufficient minimum"):
        _sim(period)


@pytest.mark.parametrize("period,rounds", [(0.1, 10), (0.2, 5), (0.3, 3)])
def test_possible_schedules_are_accepted_and_certified_nothing(period, rounds):
    # 0.3 s holds exactly stable_rounds = 3 rounds: possible, so accepted,
    # although with auction traffic it will rarely settle. The guard rejects
    # the impossible only -- a heuristic budget (e.g. 7 rounds) would reject it.
    rt = _sim(period).cbaa_runtime
    assert rt.min_rounds_per_epoch == rounds
    assert rt.stability_impossible is False
    assert not hasattr(rt, "auction_can_settle")


def _observed_rounds_per_epoch(period, steps=62):
    """Rounds each agent actually ran in each completed epoch of a live
    episode, read off the agents just before every reset."""
    sim = Simulation(
        num_blues=3, num_reds=2, seed=2001, sim_steps=steps, dt=0.1,
        planning_interval_seconds=1.0, allocation="cbaa", belief_mode="shared",
        comm_interval_seconds=period, cbaa_allow_unsettleable=True,
        enable_blue_blue_cbf=True, enable_blue_red_cbf=True,
    )
    runtime, seen = sim.cbaa_runtime, []
    original = runtime.start_epoch

    def start_epoch(step, views, *a, **kw):
        if runtime.epoch >= 0:
            seen.append({runtime.agents[i]._rounds_this_epoch for i in views})
        return original(step, views, *a, **kw)

    runtime.start_epoch = start_epoch
    with _Quiet():
        sim.run()
    assert all(len(s) == 1 for s in seen), "every live agent runs every round"
    return [s.pop() for s in seen]


@pytest.mark.parametrize("period,cycle", [(0.1, [10]), (0.3, [3, 3, 4]), (0.4, [2, 3]),
                                          (0.5, [2]), (5.0, [0, 0, 0, 0, 1])])
def test_the_cycle_is_the_schedule(period, cycle):
    assert rounds_per_epoch_cycle(10, round(period / 0.1)) == cycle


@pytest.mark.parametrize("period", [0.3, 0.4])
def test_non_dividing_periods_give_epochs_different_round_counts_on_a_live_episode(period):
    # Measured rather than asserted: at 0.3 s an epoch
    # does NOT hold exactly 3 rounds every time.
    observed = _observed_rounds_per_epoch(period)
    cycle = rounds_per_epoch_cycle(10, round(period / 0.1))
    assert len(observed) >= 2 * len(cycle)
    assert observed == [cycle[k % len(cycle)] for k in range(len(observed))]
    assert min(observed) == 10 // round(period / 0.1) < max(observed)


def test_the_message_says_when_only_some_epochs_fall_short():
    with pytest.raises(ValueError) as short_min:
        _sim(0.4)
    assert "1 of every 2 epochs hold too few" in str(short_min.value)
    assert "never commit" not in str(short_min.value)
    with pytest.raises(ValueError) as none_can:
        _sim(0.5)
    assert "No epoch holds enough rounds" in str(none_can.value)


def test_the_guard_follows_the_configured_stable_rounds():
    assert _sim(0.5, cbaa_stable_rounds=2).cbaa_runtime.stability_impossible is False
    with pytest.raises(ValueError):
        _sim(0.3, cbaa_stable_rounds=4)


def test_the_guard_covers_every_transport():
    # "ideal" and "none" run their rounds on the same communication clock.
    with pytest.raises(ValueError):
        _sim(0.5, belief_mode="shared")
    with pytest.raises(ValueError):
        _sim(0.5, cbaa_transport="none")


def test_a_diagnostic_override_runs_and_says_so():
    rt = _sim(0.5, cbaa_allow_unsettleable=True).cbaa_runtime
    assert rt.stability_impossible is True
    assert rt.min_rounds_per_epoch == 2


def test_no_guard_without_an_auction():
    sim = Simulation(num_blues=4, num_reds=2, seed=2001, sim_steps=5, dt=0.1,
                     planning_interval_seconds=1.0, belief_mode="networked",
                     comm_interval_seconds=5.0)
    assert sim.cbaa_runtime is None


# -- 3. commitment conflicts ------------------------------------------------------


def _agent(i, x, commitment):
    a = CBAAAgent(i, stable_rounds=3)
    a.x, a.commitment = x, commitment
    return a


def test_commitment_conflicts_read_what_is_in_force_not_the_claims():
    agents = [
        _agent(0, "SCREEN:1", "DEFEND:0"),    # claim differs from what it executes
        _agent(1, "SCREEN:1", "DEFEND:0"),
        _agent(2, "RECON:3", None),           # nothing in force
        _agent(3, "RECON:3", None),
    ]
    assert commitment_conflicts(agents) == {"DEFEND:0": [0, 1]}
    assert conflicts(agents) == {"SCREEN:1": [0, 1], "RECON:3": [2, 3]}
