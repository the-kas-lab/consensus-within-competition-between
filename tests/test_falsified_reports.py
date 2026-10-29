"""falsified_reports: one compromised blue adds a fixed displacement to the
position in its own outgoing first-hand reports about one red. Rule-based,
free; spent seed 3705."""

import numpy as np
import pytest

from conftest import build_simulation, capture_fd_output, fingerprint
from scripts import network_conditions
from scripts.experiment_policy import (
    NEW_EXPERIMENT_INTRUDER_INFO, NEW_EXPERIMENT_PURSUIT, NEW_EXPERIMENT_SAFETY,
    NEW_EXPERIMENT_TASKS,
)
from scripts.episode_builder import build_episode

SEED, FAMILY = 3705, "multi_bearing"


def _cfg(arm="C", **extra):
    belief, cfg = network_conditions.episode_cfg(FAMILY, "REF", arm)
    cfg = dict(cfg, **NEW_EXPERIMENT_PURSUIT, **NEW_EXPERIMENT_SAFETY,
               **NEW_EXPERIMENT_INTRUDER_INFO, **NEW_EXPERIMENT_TASKS, **extra)
    return belief, cfg


def _spec(**kw):
    spec = dict(onset_step=50, duration_steps=300, displacement=[8.0, 0.0], report_cap=400)
    spec.update(kw)
    return spec


def _run(arm="C", **extra):
    belief, cfg = _cfg(arm, **extra)
    sim = build_episode(SEED, belief, cfg)
    with capture_fd_output():
        sim.run()
    return sim


def test_default_off_and_not_recorded():
    sim = build_simulation()
    assert sim.falsified_reports is None and "falsified_reports" not in sim._init_args


@pytest.mark.parametrize("spec, match", [
    (dict(onset_step=0, duration_steps=10, displacement=[1, 0]), "needs exactly"),
    (dict(onset_step=0, duration_steps=0, displacement=[1, 0], report_cap=1), "out of range"),
])
def test_bad_specs_are_refused(spec, match):
    belief, cfg = _cfg(falsified_reports=spec)
    with pytest.raises(ValueError, match=match):
        build_episode(SEED, belief, cfg)


def test_needs_the_networked_belief_layer():
    with pytest.raises(ValueError, match="networked"):
        build_simulation(belief_mode="shared", falsified_reports=_spec())


@pytest.mark.parametrize("commander", [0, 1, 2, 3])
def test_sender_is_never_the_commander_and_is_shared_by_matched_arms(commander):
    senders = []
    for arm in ("C", "I"):
        belief, cfg = _cfg(arm, falsified_reports=_spec(), commander_id=commander)
        sim = build_episode(SEED, belief, cfg)
        sim._reset_falsified_reports()
        senders.append(sim.attack_log["sender"])
    assert senders[0] == senders[1] != commander


def test_zero_displacement_changes_nothing_but_is_still_traced():
    base = _run()
    inert = _run(falsified_reports=_spec(displacement=[0.0, 0.0]))
    assert inert.attack_log["reports_altered"] > 0
    assert fingerprint(inert) == fingerprint(base)


@pytest.fixture(scope="module")
def attacked():
    return _run(falsified_reports=_spec())


def test_alterations_shift_position_only_within_window_and_budget(attacked):
    log, spec = attacked.attack_log, attacked.falsified_reports
    assert log["target_red"] is not None and log["alterations"]
    hist = np.asarray(attacked.history_red)
    nx = attacked.red_nx
    for k, a in enumerate(log["alterations"], start=1):
        assert a["red"] == log["target_red"] and a["report_index"] == k
        assert spec["onset_step"] <= a["step"] < spec["onset_step"] + spec["duration_steps"]
        assert a["observed_at"] >= spec["onset_step"]
        np.testing.assert_allclose(np.subtract(a["reported_position"], a["true_position"]),
                                   spec["displacement"])
        # The sender's own sighting was correct: the red really was there then.
        red = a["red"]
        np.testing.assert_allclose(a["true_position"],
                                   hist[a["observed_at"]][red * nx : red * nx + 2])
    assert log["reports_altered"] == len(log["alterations"]) <= spec["report_cap"]


def test_altered_records_keep_identity_velocity_and_time(attacked):
    for fake, true in attacked._falsified_by_id.values():
        assert fake.contact_id == true.contact_id
        assert fake.provenance is true.provenance
        np.testing.assert_array_equal(fake.state[2:], true.state[2:])


def test_the_sender_never_holds_its_own_falsification(attacked):
    sender = attacked.attack_log["sender"]
    for b in attacked.belief_stores[sender].targets.values():
        assert id(b) not in attacked._falsified_by_id
    for d in attacked.attack_log["deliveries"]:
        assert d["receiver"] != sender or d not in attacked.attack_log["accepted"]


def test_deliveries_distinguish_relays_and_acceptance_is_a_subset(attacked):
    log = attacked.attack_log
    assert log["deliveries"]
    for d in log["deliveries"]:
        assert d["relayed"] == (d["sender"] != log["sender"])
    assert all(a in log["deliveries"] for a in log["accepted"])
    assert any(d["relayed"] for d in log["deliveries"])


def test_report_cap_stops_alteration():
    sim = _run(falsified_reports=_spec(report_cap=5))
    assert sim.attack_log["reports_altered"] == 5
    assert sim.attack_log["stop_reason"] == "report_cap"


# ---------------------------------------------------------------------------
# Equal-timestamp conflicts are order-dependent (perception/belief/merge.py)
# ---------------------------------------------------------------------------


def test_truthful_first_and_false_first_delivery_keep_different_positions():
    from perception.belief.merge import merge_target_dicts
    from perception.belief.records import Provenance, TargetBelief

    prov = Provenance(origin_agent=2, observed_at=40, received_at=40)
    true = TargetBelief(1, np.array([30.0, 20.0, 1.0, 0.5]), prov)
    false = TargetBelief(1, np.array([30.0 - 7.07, 20.0 - 7.07, 1.0, 0.5]), prov)
    truthful_first = merge_target_dicts(merge_target_dicts({}, {1: true}), {1: false})
    false_first = merge_target_dicts(merge_target_dicts({}, {1: false}), {1: true})
    assert truthful_first[1] is true and false_first[1] is false
    # A NEWER sighting still wins from either state.
    newer = TargetBelief(1, np.array([31.0, 20.5, 1.0, 0.5]),
                         Provenance(origin_agent=0, observed_at=41, received_at=41))
    for state in (truthful_first, false_first):
        assert merge_target_dicts(state, {1: newer})[1] is newer


def test_delivery_records_reproduce_order_and_what_was_held(attacked):
    log = attacked.attack_log
    seqs = [d["delivery_seq"] for d in log["deliveries"]]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    kinds = {"none", "same_altered_record", "other_altered_record",
             "older_record", "same_time_record", "newer_record"}
    for d in log["deliveries"]:
        assert d["held"] in kinds and d["message_id"]
        # Acceptance happens exactly when the held record could lose the merge.
        if d in log["accepted"]:
            assert d["held"] in ("none", "older_record", "other_altered_record")
        elif d["held"] in ("none", "older_record"):
            raise AssertionError(f"should have been accepted: {d}")


def test_stored_fresh_and_committed_are_nested_and_counted_separately(attacked):
    log = attacked.attack_log
    assert log["belief_validity_age_steps"] is None  # the runners' configuration
    for i, n in log["commitment_steps"].items():
        assert n <= log["fresh_false_steps"].get(i, 0)
    for i, n in log["fresh_false_steps"].items():
        assert n <= log["stored_false_steps"][i]
    # With no age limit nothing expires: every stored record is fresh.
    assert log["fresh_false_steps"] == log["stored_false_steps"]


def test_distinct_observations_are_counted_apart_from_broadcasts(attacked):
    log = attacked.attack_log
    distinct = {a["observed_at"] for a in log["alterations"]}
    assert log["distinct_observations_falsified"] == len(distinct) <= log["reports_altered"]


def test_primary_parameters_are_the_declared_persistent_setting():
    from scripts.experiment_policy import NEW_EXPERIMENT_FALSIFIED_REPORTS as P

    assert (P["onset_step"], P["duration_steps"], P["report_cap"]) == (0, 400, 400)
    assert np.hypot(*P["displacement"]) == pytest.approx(10.0)
    np.testing.assert_allclose(P["displacement"], [-10 / np.sqrt(2)] * 2)
