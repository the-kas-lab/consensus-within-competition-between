"""resolution_info="public": every resolution is
announced to the opposing team at once; the resolved agent leaves those belief
stores for good. "legacy" (the default) is unchanged and unrecorded."""

import numpy as np
import pytest

from conftest import build_simulation, capture_fd_output
from perception.belief.records import Provenance, TargetBelief
from perception.belief.store import LocalBeliefStore


def _belief(cid, step, origin=1):
    return TargetBelief(cid, np.array([1.0, 2.0, 0.0, 0.0, 0.0]),
                        Provenance(origin_agent=origin, observed_at=step, received_at=step,
                                   relay_agent=None, source_sequence=1, planning_epoch=None))


def test_a_resolved_contact_is_dropped_and_never_re_added():
    store = LocalBeliefStore(0)
    store.observe_target(3, np.zeros(5), step=1)
    store.merge_targets({4: _belief(4, 2)})
    store.mark_resolved(3)
    assert 3 not in store.targets and 4 in store.targets
    assert store.observe_target(3, np.zeros(5), step=5) is None
    store.merge_targets({3: _belief(3, 9), 5: _belief(5, 9)})  # a relayed record
    assert 3 not in store.targets and 5 in store.targets


def test_legacy_is_the_default_unrecorded_and_values_are_checked():
    sim = build_simulation()
    assert sim.resolution_info == "legacy" and "resolution_info" not in sim._init_args
    assert build_simulation(resolution_info="public")._init_args["resolution_info"] == "public"
    with pytest.raises(ValueError, match="resolution_info"):
        build_simulation(resolution_info="private")


def _episode_with_kills(resolution_info):
    """Rule CBAA, 4 gate intruders, safety.2, spent seed 3904: two
    neutralizations."""
    from scripts import experiment_policy as ep
    from scripts import run_competing_teams as rct

    saved = dict(ep.NEW_EXPERIMENT_COMMON)
    ep.NEW_EXPERIMENT_COMMON["resolution_info"] = resolution_info
    try:
        e = {"arm": "cbaa", "reds": 4, "family": "multi_bearing", "condition": "REF",
             "attacked": False, "seed": 3904}
        e["episode_id"] = rct.episode_id(e)
        sim, _ = rct.build(e, "rule")
    finally:
        ep.NEW_EXPERIMENT_COMMON.clear()
        ep.NEW_EXPERIMENT_COMMON.update(saved)
    with capture_fd_output():
        sim.run()
    return sim


def test_public_resolution_empties_every_opposing_store_and_is_logged():
    pub = _episode_with_kills("public")
    dead_reds = [j for j, o in enumerate(pub.red_outcomes) if o != "active"]
    dead_blues = [i for i, d in enumerate(pub.blue_disabled) if d]
    assert dead_reds and dead_blues  # the fixture must exercise the feature
    for store in pub.belief_stores.values():
        assert not set(dead_reds) & set(store.targets)
    for store in pub.red_belief_stores.values():
        assert not set(dead_blues) & set(store.targets)
    logged = {(r["team"], r["agent"]) for r in pub.resolution_log}
    assert logged == {("red", j) for j in dead_reds} | {("blue", i) for i in dead_blues}
    # Under legacy the same kind of episode leaves a resolved red believed in.
    leg = _episode_with_kills("legacy")
    assert leg.resolution_log == []
    gone = [j for j, o in enumerate(leg.red_outcomes) if o != "active"]
    assert any(set(gone) & set(s.targets) for s in leg.belief_stores.values())
