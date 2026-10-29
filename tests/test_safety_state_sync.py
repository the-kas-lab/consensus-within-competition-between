"""safety_state_sync: which blue state the red safety filter sees.

"sequential" (default) is the historical order -- blues step first, and the
red filter sees their post-step state. "frozen" hands it the same pre-step
snapshot the blue filter used. Checked by spying on the filter's input against
the recorded pre-step blue history, on a short episode with the red CBF on."""

import numpy as np
import pytest

from conftest import build_simulation, run_quiet


def _spy_red_filter_inputs(sim):
    seen = []
    original = sim.solver.one_step_red_cbf_filter

    def spy(x_current, red_state, u_des):
        seen.append(np.array(x_current, dtype=float).copy())
        return original(x_current=x_current, red_state=red_state, u_des=u_des)

    sim.solver.one_step_red_cbf_filter = spy
    return seen


def _blue_inputs_vs_history(mode):
    sim = build_simulation(enable_red_cbf=True, sim_steps=12, safety_state_sync=mode)
    seen = _spy_red_filter_inputs(sim)
    run_quiet(sim)
    # history_blues[k] is the state at the start of step k; [k + 1] is after it.
    # One red, so exactly one filter call per step while it stays active.
    assert seen, "the red filter never ran"
    return seen, np.asarray(sim.history_blues)


def test_frozen_hands_the_red_filter_the_pre_step_blue_state():
    seen, hist = _blue_inputs_vs_history("frozen")
    for k, x in enumerate(seen):
        np.testing.assert_array_equal(x, hist[k])


def test_sequential_is_the_historical_post_step_state():
    seen, hist = _blue_inputs_vs_history("sequential")
    for k, x in enumerate(seen):
        np.testing.assert_array_equal(x, hist[k + 1])
    assert any(not np.array_equal(hist[k], hist[k + 1]) for k in range(len(seen)))


def test_default_is_sequential_and_not_recorded():
    sim = build_simulation()
    assert sim.safety_state_sync == "sequential"
    assert "safety_state_sync" not in sim._init_args
    assert build_simulation(safety_state_sync="frozen")._init_args[
        "safety_state_sync"] == "frozen"


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="safety_state_sync"):
        build_simulation(safety_state_sync="simultaneous")
