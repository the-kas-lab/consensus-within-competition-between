"""The episode record is the substrate every metric reads back.

The properties that matter are (a) recording changes nothing about the episode,
(b) the record is internally consistent enough to replay, and (c) the
safety-filter numbers asks for are actually derivable from it.
"""

import json

import numpy as np
import pytest

from conftest import build_simulation, fingerprint, run_quiet


def test_recording_does_not_change_the_episode():
    """The one property that makes the recorder safe to leave on by default."""
    with_rec = fingerprint(run_quiet(build_simulation(seed=17, record_episode=True)))
    without = fingerprint(run_quiet(build_simulation(seed=17, record_episode=False)))
    assert with_rec == without


def test_disabled_recorder_is_absent():
    sim = run_quiet(build_simulation(seed=17, record_episode=False))
    assert sim.recorder is None


def test_manifest_shape_and_config_capture():
    sim = run_quiet(build_simulation(seed=17, num_blues=3, num_reds=2))
    m = sim.recorder.manifest()

    assert m["schema_version"] == 1
    assert m["seed"] == 17
    assert m["rng_mode"] == "legacy_global"
    assert m["dimensions"] == {
        "num_blues": 3, "num_reds": 2,
        "blue_nx": sim.blue_nx, "red_nx": sim.red_nx, "blue_nu": sim.blue.nu_single,
    }
    # Config capture must reflect the ACTUAL constructor arguments, including
    # the CBF parameters whose plumbing was broken until M1.
    assert m["config"]["num_blues"] == 3
    assert m["config"]["D_safe_blue"] == pytest.approx(3.0)
    assert m["config"]["enable_blue_blue_cbf"] is True
    # A planner is an object; it is recorded by type plus scalar config.
    assert m["config"]["role_planner"]["__type__"] == "RuleBasedBlueRolePlanner"
    assert m["config"]["role_planner"]["recon_fraction"] == pytest.approx(0.5)
    assert json.dumps(m)  # must be serializable end to end


def test_config_hash_is_sensitive_and_stable():
    a = run_quiet(build_simulation(seed=17)).recorder.manifest()["config_hash"]
    b = run_quiet(build_simulation(seed=17)).recorder.manifest()["config_hash"]
    c = run_quiet(build_simulation(seed=17, D_safe_blue=9.0)).recorder.manifest()["config_hash"]
    assert a == b
    assert a != c, "config hash must notice a CBF parameter change"


def test_step_records_align_with_arrays():
    sim = run_quiet(build_simulation(seed=17, num_blues=3, num_reds=2))
    rec = sim.recorder
    arrays = rec.arrays()
    n = len(rec.steps)

    assert n > 0
    for name in ("blue_state", "red_state", "u_requested", "u_executed"):
        assert arrays[name].shape[0] == n, f"{name} length disagrees with step records"
    assert arrays["blue_state"].shape[1] == 3 * sim.blue_nx
    assert arrays["u_requested"].shape[1] == 3 * sim.blue.nu_single


def test_engagement_authorized_tracks_neutralize_role():
    """The engagement-authority rule: the safety supervisor receives exactly one bit per agent,
    derived from the committed allocation. Auditable from the record."""
    sim = run_quiet(build_simulation(seed=130, num_blues=4, num_reds=2, sim_steps=120))
    for step in sim.recorder.steps:
        roles = step["blue_roles"]
        for i, authorized in enumerate(step["engagement_authorized"]):
            assert authorized == (roles.get(str(i)) == "neutralize"), (
                f"step {step['step']}, agent {i}: engagement bit disagrees with role"
            )


def test_safety_metrics_are_threshold_swept():
    """A single intervention fraction hides a strongly bimodal distribution --
    see simulation/recorder.INTERVENTION_THRESHOLDS."""
    sim = run_quiet(
        build_simulation(seed=130, num_blues=4, num_reds=2, sim_steps=120,
                         enable_blue_blue_cbf=True, enable_blue_red_cbf=True)
    )
    s = sim.recorder.manifest()["safety"]
    sweep = s["intervention_fraction_by_threshold"]

    assert set(sweep) == {"1e-06", "0.001", "0.01", "0.1", "0.5", "1.0"}
    # Monotone non-increasing in the threshold, by construction.
    vals = [sweep[k] for k in ("1e-06", "0.001", "0.01", "0.1", "0.5", "1.0")]
    assert all(a >= b for a, b in zip(vals, vals[1:])), vals
    assert s["max_intervention_magnitude"] >= s["mean_intervention_magnitude"]
    assert s["steps_with_cbf_active"] > 0


def test_cbf_off_means_no_intervention():
    """With the filter disabled, executed controls must be the nominal ones
    exactly -- this is the control that makes a nonzero reading meaningful."""
    sim = run_quiet(
        build_simulation(seed=17, sim_steps=30,
                         enable_blue_blue_cbf=False, enable_blue_red_cbf=False,
                         enable_convex_hull_containment=False)
    )
    a = sim.recorder.arrays()
    assert np.array_equal(a["u_requested"], a["u_executed"])
    assert not any(s["cbf_active"] for s in sim.recorder.steps)
    assert sim.recorder.manifest()["safety"]["max_intervention_magnitude"] == 0.0


def test_save_round_trip(tmp_path):
    sim = run_quiet(build_simulation(seed=17, sim_steps=30))
    stem = tmp_path / "episodes" / "ep0"
    json_path, npz_path = sim.recorder.save(stem)

    with open(json_path) as f:
        loaded = json.load(f)
    assert loaded["config_hash"] == sim.recorder.config_hash
    assert len(loaded["steps"]) == len(sim.recorder.steps)

    data = np.load(npz_path)
    assert np.allclose(data["blue_state"], sim.recorder.arrays()["blue_state"])
