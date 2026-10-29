"""The replay exporter shows the episode that ran, and nothing else.

Three claims, each pinned here because a viewer that draws a plausible but
wrong episode is worse than no viewer:

1. **A bundle reproduces the recorded episode.** The conftest ``fingerprint``,
   rebuilt from the bundle *alone*, equals the golden fingerprints -- the same
   baselines ``test_golden_episode`` holds the simulator to.
2. **The exporter's instrumentation is inert.** The ledger, bearing-log and
   believed-targets wrappers are attached from outside the testbed; a wrapped
   episode must be byte-identical to an unwrapped one.
3. **What the bundle derives, it derives exactly.** Believed positions (the
   recorder stores only ages), coverage staleness and neutralization pairs are
   reconstructed, then checked against the live episode -- and the check is
   shown to fail on a tampered bundle, because a check that cannot fail proves
   nothing (the LLM planner's order-invariance lesson).
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import build_simulation, fingerprint, run_quiet
from planning.roles import BlueRole, RedRole
from scripts import export_replay as er
from scripts.auction_smoke import attach_bearing_log, attach_ledger
from test_golden_episode import CASES, GOLDEN_PATH, _build

# Detects from step 0, holds beliefs up to 72 steps old, falls back to nearest
# believed contacts, and ends with both reds escaping -- so the belief, event
# and terminal-frame paths are all exercised, not vacuously passed. Seed 7 is
# the golden seed; nothing here spends a seed.
AGING = dict(belief_mode="local", seed=7, sim_steps=120, arena_size=40.0, num_blues=3, num_reds=2)


def fingerprint_from_bundle(ep):
    """``conftest.fingerprint`` over an object rebuilt from the bundle only."""
    outcome = ep["recorder"]["outcome"]
    steps = ep["steps"]
    return fingerprint(SimpleNamespace(
        history_blues=np.asarray(ep["frames"]["blue_state"], dtype=float),
        history_red=np.asarray(ep["frames"]["red_state"], dtype=float),
        # run() seeds both histories with the (empty) assignment in force before
        # step 0, then appends the assignment each recorded step was solved with.
        role_assignment_history=[{}] + [
            {int(k): BlueRole(v) for k, v in st["blue_roles"].items()} for st in steps
        ],
        red_role_assignment_history=[{}] + [
            {int(k): RedRole(v) for k, v in st["red_roles"].items()} for st in steps
        ],
        touchdown=outcome["touchdown"],
        num_neutralized=outcome["num_neutralized"],
        num_escaped=outcome["num_escaped"],
        blue_score=outcome["blue_score"],
        solver_failed=outcome["solver_failed"],
        blue_disabled=outcome["blue_disabled"],
    ))


def wrapped_run(sim):
    capture = {
        "ledger": attach_ledger(sim),
        "bearing_log": attach_bearing_log(sim),
        "believed": er.attach_believed_log(sim),
    }
    run_quiet(sim)
    return sim, capture


def bundle_of(sim, capture, label="test"):
    source = {"preset": None, "arm": None, "seed": int(sim.seed), "family": sim.scenario_family,
              "belief_mode": sim.belief_mode, "cfg": {}, "label": label}
    return er.episode_bundle(sim, capture, source)


@pytest.fixture(scope="module")
def aging():
    plain = run_quiet(build_simulation(**AGING))
    sim, capture = wrapped_run(build_simulation(**AGING))
    return plain, sim, capture, bundle_of(sim, capture)


def test_bundle_reproduces_golden_fingerprints():
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    for name, case in CASES.items():
        sim, capture = wrapped_run(_build(case))
        ep = bundle_of(sim, capture, name)
        assert er.check_bundle(ep, sim, capture)
        assert fingerprint_from_bundle(ep) == golden[name], name


def test_instrumentation_is_inert(aging):
    plain, sim, _, ep = aging
    assert fingerprint(sim) == fingerprint(plain)
    # Every field by hash except roles. fingerprint() hashes repr() of each role
    # dict, which carries the planner's *insertion order*; the recorder
    # normalises keys to sorted order, so a planner that inserts neutralizers
    # first (this one) hashes differently with identical content. Order is not a
    # property of the episode, so roles are compared by content.
    from_bundle, live = fingerprint_from_bundle(ep), fingerprint(plain)
    assert {k: v for k, v in from_bundle.items() if k != "roles"} == \
        {k: v for k, v in live.items() if k != "roles"}
    rebuilt = [{}] + [{int(k): v for k, v in st["blue_roles"].items()} for st in ep["steps"]]
    assert rebuilt == [{int(k): r.value for k, r in d.items()} for d in plain.role_assignment_history]


def test_terminal_frame_is_the_state_that_ended_the_episode(aging):
    _, sim, _, ep = aging
    n = ep["dims"]["n_steps"]
    assert ep["dims"]["n_frames"] == n + 1 == len(sim.history_blues)
    assert ep["frames"]["red_state"][-1] == np.asarray(sim.history_red[-1]).tolist()
    assert ep["terminal"]["reason"] == "all_reds_resolved"
    escapes = [e for e in ep["derived"]["events"] if e["kind"] == "escape"]
    assert sorted(e["red"] for e in escapes) == [0, 1]
    size = ep["geometry"]["arena"]["size"]
    for e in escapes:
        x, y = e["pos"]
        assert not (0.0 <= x <= size and 0.0 <= y <= size), e
    # The last escape resolved the episode, so it is only visible in the
    # terminal frame the recorder never saw.
    assert max(e["frame"] for e in escapes) == n


def test_belief_derivation_is_exact_and_non_vacuous(aging):
    _, sim, capture, ep = aging
    ages = [a for st in ep["steps"] for per in st["beliefs"].values() for a in per.values()]
    assert sum(a > 0 for a in ages) > 50 and max(ages) > 30
    assert any(e["source"] == "nearest_believed"
               for st in ep["steps"] for e in st["effective_targets"].values())
    assert er.check_bundle(ep, sim, capture)


def test_belief_check_fails_on_a_tampered_bundle(aging):
    _, sim, capture, ep = aging
    # Move one red at an observation instant some agent is still acting on.
    t, contact, age = next(
        (t, int(c), a) for t, st in enumerate(ep["steps"])
        for per in st["beliefs"].values() for c, a in per.items() if a > 5
    )
    tampered = json.loads(json.dumps(ep))
    rnx = ep["dims"]["red_nx"]
    tampered["frames"]["red_state"][t - age][contact * rnx] += 1e-9
    # The history check fires first; bypass it to prove the belief check bites.
    sim_view = SimpleNamespace(
        history_blues=np.asarray(tampered["frames"]["blue_state"]),
        history_red=np.asarray(tampered["frames"]["red_state"]),
        coverage_tracker=sim.coverage_tracker,
    )
    with pytest.raises(AssertionError, match="derived"):
        er.check_bundle(tampered, sim_view, capture)


def test_coverage_derivation_matches_live_tracker(aging):
    _, sim, _, ep = aging
    assert ep["derived"]["coverage_last_seen"][-1] == sim.coverage_tracker.last_seen_step.tolist()
    assert len(ep["derived"]["coverage_last_seen"]) == ep["dims"]["n_frames"]


def test_unspent_seeds_are_refused_before_anything_runs(tmp_path):
    for seed in (3725, 3801, 3900):
        with pytest.raises(SystemExit):
            er.main(["--seed", str(seed), "--out", str(tmp_path / "x.json")])
    assert not list(tmp_path.iterdir())


def test_js_wrapper_round_trips_and_encodes_infinity(tmp_path):
    doc = er.bundle_document([{"r_comm": float("inf"), "x": [1.5, float("nan")]}])
    path = er.write_bundle(doc, tmp_path / "b.replay.js")
    text = open(path).read()
    assert text.startswith("PEV_REPLAY(") and text.rstrip().endswith(");")
    loaded = json.loads(text[len("PEV_REPLAY("):].rstrip().rstrip(";")[:-1])
    assert loaded["episodes"][0] == {"r_comm": "inf", "x": [1.5, "nan"]}
    assert loaded["format"] == er.FORMAT and loaded["format_version"] == er.FORMAT_VERSION


# ---------------------------------------------------------------------------
# Format 2: non-hostile contacts, the network, the commander's orders, false beliefs
# ---------------------------------------------------------------------------

def _rct_run(entry, steps):
    """A competing-teams episode with the free rule backend, cut to `steps`."""
    from scripts import run_competing_teams as rct
    sim, _ = rct.build(entry, "rule")
    sim.sim_steps = steps
    capture = er.attach_capture(sim)
    run_quiet(sim)
    return sim, capture


@pytest.fixture(scope="module")
def decoy_episode():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images = {4: os.path.join(root, "assets/contact_photos/bird_1.jpeg"),
              5: os.path.join(root, "assets/contact_photos/civilian_1.jpeg"),
              6: os.path.join(root, "assets/contact_photos/civilian_2.jpeg")}
    entry = {"arm": "commander", "reds": 4, "family": "multi_bearing", "condition": "REF", "attacked": False,
             "seed": 4001, "design": "v2.3mb", "decoys": {"birds": 1, "civilians": 2, "images": images},
             "decoy_tag": "b1c2"}
    sim, capture = _rct_run(entry, 250)
    return sim, capture, bundle_of(sim, capture, "decoys")


def test_decoy_rows_are_exported_with_their_kinds_and_photographs(decoy_episode):
    sim, capture, ep = decoy_episode
    assert er.check_bundle(ep, sim, capture)
    c = ep["contacts"]
    assert c["kinds"] == ["drone"] * 4 + ["bird", "civilian", "civilian"] and c["num_intruders"] == 4
    assert c["images"] == {"4": "assets/contact_photos/bird_1.jpeg", "5": "assets/contact_photos/civilian_1.jpeg",
                           "6": "assets/contact_photos/civilian_2.jpeg"}
    # every red-state row has an outcome, non-hostile contacts included (the outcome dict keeps intruders only)
    assert len(ep["terminal"]["red_outcomes"]) == 7 == ep["dims"]["num_reds"]
    assert ep["terminal"]["red_outcomes"][:4] == ep["recorder"]["outcome"]["red_outcomes"]
    # a non-hostile destroyed by contact takes its defender, paired exactly like a kill
    for e in sim.decoy_events:
        if e["event"] == "destroyed":
            n = [x for x in ep["derived"]["neutralizations"] if x["red"] == e["decoy"]]
            assert len(n) == 1 and n[0]["blue"] == e["blue"] and n[0]["outcome"] == "decoy_destroyed"


@pytest.fixture(scope="module")
def limited_range_commander():
    entry = {"arm": "commander", "reds": 4, "family": "multi_bearing", "condition": "MH", "attacked": False,
             "seed": 4001, "design": "v2.3mb"}
    sim, capture = _rct_run(entry, 200)
    return sim, capture, bundle_of(sim, capture, "limited range")


def test_link_rounds_are_the_disk_rule_on_the_recorded_positions(limited_range_commander):
    from comms.topologies import disk
    sim, capture, ep = limited_range_commander
    net = ep["network"]
    assert net["r_comm"] == 15.0 and net["links"]
    nx, n_links = ep["dims"]["blue_nx"], 0
    for t, edges in net["links"].items():
        t = int(t)
        row, down = ep["frames"]["blue_state"][t], ep["steps"][t]["blue_disabled"]
        pos = {i: row[i * nx: i * nx + 2] for i in range(ep["dims"]["num_blues"])}
        want = sorted(sorted(e) for e in disk(pos, net["r_comm"]) if not any(down[a] for a in e))
        assert edges == want, t
        n_links += len(edges)
    assert n_links > 0
    # some round must miss a pair, or the test says nothing about range
    nb = ep["dims"]["num_blues"]
    assert any(len(e) < nb * (nb - 1) // 2 for e in net["links"].values())


def test_every_order_has_a_fate_and_the_planner_is_recorded(limited_range_commander):
    sim, _, ep = limited_range_commander
    cmd = ep["command"]
    assert len(cmd["orders"]) == len(sim.command_log)
    assert {o["fate"] for o in cmd["orders"]} <= {"self", "applied", "addressee_lost", "missed_changed", "missed_same"}
    assert all((o["via"] == "self") == (o["fate"] == "self") for o in cmd["orders"])
    assert set(cmd["source_by_step"].values()) <= set(range(ep["dims"]["num_blues"]))
    assert cmd["source_by_step"][str(min(int(t) for t in cmd["source_by_step"]))] == sim.commander_id


def test_order_fates_follow_the_analysis_definition():
    # agent 1 holds DEFEND:1 by default; an applied pursuit order becomes the held task
    o = lambda epoch, agent, task, gen, applied, via="network": dict(  # noqa: E731
        epoch=epoch, agent=agent, task_id=task, generated_at=gen, applied_at=applied, via=via)
    log = [o(0, 0, "DEFEND:0", 0, 0, "self"),
           o(0, 1, "NEUTRALIZE:2:0", 0, None),        # never arrived, would change DEFEND:1 -> missed_changed
           o(1, 1, "DEFEND:1", 10, None),             # never arrived, repeats the held DEFEND:1 -> missed_same
           o(2, 1, "NEUTRALIZE:3:0", 20, 21),         # arrived
           o(3, 1, "NEUTRALIZE:3:0", 30, None),       # repeats the held pursuit -> missed_same
           o(3, 2, "NEUTRALIZE:3:0", 30, None)]       # addressee lost at 35, within 10 steps
    assert er.classify_orders(log, {2: 35}) == [
        "self", "missed_changed", "missed_same", "applied", "missed_same", "addressee_lost"]


def test_false_beliefs_are_shipped_only_as_held_and_tied_to_accepted_records():
    from scripts import network_conditions
    from scripts.experiment_policy import (
        NEW_EXPERIMENT_INTRUDER_INFO, NEW_EXPERIMENT_PURSUIT, NEW_EXPERIMENT_SAFETY, NEW_EXPERIMENT_TASKS,
    )
    from scripts.episode_builder import build_episode
    belief, cfg = network_conditions.episode_cfg("multi_bearing", "REF", "C")
    cfg = dict(cfg, **NEW_EXPERIMENT_PURSUIT, **NEW_EXPERIMENT_SAFETY, **NEW_EXPERIMENT_INTRUDER_INFO,
               **NEW_EXPERIMENT_TASKS, steps=220,
               falsified_reports=dict(onset_step=50, duration_steps=300, displacement=[8.0, 0.0], report_cap=400))
    sim = build_episode(3705, belief, cfg)   # spent seed, as in test_falsified_reports.py
    capture = er.attach_capture(sim)
    run_quiet(sim)
    ep = bundle_of(sim, capture, "falsified")
    held = ep.get("beliefs_held", {})
    assert held, "the attack altered no belief the controller acted on"
    target = sim.attack_log["target_red"]
    assert {int(c) for per in held.values() for b in per.values() for c in b} == {target}
    assert er.check_bundle(ep, sim, capture)


def test_held_beliefs_are_refused_without_an_accepted_falsified_record(aging):
    _, sim, capture, ep = aging
    t, agent, c = next((t, a, c) for t, st in enumerate(ep["steps"]) for a, per in st["beliefs"].items()
                       for c, age in per.items() if age > 5)
    tampered = json.loads(json.dumps(ep))
    rnx = ep["dims"]["red_nx"]
    tampered["frames"]["red_state"][t - ep["steps"][t]["beliefs"][agent][c]][int(c) * rnx] += 1.0
    with pytest.raises(AssertionError, match="without an accepted falsified record"):
        er.held_belief_overrides(tampered, sim, capture)
