"""The safety configuration is a controlled variable. Pin it.

The safety-parameter choice
0-7 run under the CBF settings that have historically been in force. That is a
research decision with measured consequences -- the alternative configuration
changes intervention magnitude 3-4x and flips mission outcomes on roughly one
seed in three.

 requires the MPC and safety stack be held fixed across conditions, so
a silent drift in any of these defaults would invalidate every comparison made
before it was noticed. These tests make that drift loud.

The second test is the one that matters most: until M1 these parameters were
silently discarded before reaching the filter, so "the default is correct" and
"the default is in force" were two different things.
"""

import numpy as np
import pytest

from conftest import build_simulation, run_quiet

# The values in force for this phase. Changing any of these is a the safety-parameter choice re-decision,
# not a tuning tweak -- update the decision record first.
HISTORICALLY_ACTIVE = {
    "D_safe_blue": 3.0,
    "D_safe_blue_red": 3.0,
    "D_safe_red": 3.0,
    "blue_blue_cbf_slack_weight": 1e5,
    "blue_red_cbf_slack_weight": 1e5,
    "red_cbf_slack_weight": 1e5,
    "convex_hull_slack_weight": 1e5,
}


def test_safety_defaults_match_historically_active_values():
    """Simulation's declared defaults are the safety-parameter choice-settled configuration."""
    import inspect

    from simulation.testbed import Simulation

    declared = inspect.signature(Simulation.__init__).parameters
    for name, expected in HISTORICALLY_ACTIVE.items():
        assert name in declared, f"{name} is no longer a Simulation parameter"
        assert declared[name].default == pytest.approx(expected), (
            f"{name} default changed from the safety-parameter choice-settled value {expected!r} to "
            f"{declared[name].default!r}. If deliberate, update the safety-parameter choice in "
            " first -- this parameter "
            "moves mission outcomes."
        )


def test_safety_defaults_actually_reach_the_filter():
    """Declared defaults and *effective* defaults are not the same claim.

    Until M1 these were filtered out by _build_solver before ever reaching
    CBFFilterMixin, so the CBF ran on its own internal defaults regardless of
    configuration. Passing the settled values explicitly must be a no-op --
    which is only true if the defaults genuinely arrive.
    """
    implicit = run_quiet(build_simulation(seed=130, num_blues=4, num_reds=2, sim_steps=60))
    explicit = run_quiet(
        build_simulation(seed=130, num_blues=4, num_reds=2, sim_steps=60, **HISTORICALLY_ACTIVE)
    )
    assert np.array_equal(implicit.history_blues, explicit.history_blues)


def test_safety_parameters_are_live():
    """The converse control: a changed safety parameter must change behavior.

    Without this, test_safety_defaults_actually_reach_the_filter would pass
    vacuously in exactly the broken state M1 fixed -- if nothing reaches the
    filter, every configuration produces identical trajectories.
    """
    base = run_quiet(build_simulation(seed=130, num_blues=4, num_reds=2, sim_steps=60))
    widened = run_quiet(
        build_simulation(seed=130, num_blues=4, num_reds=2, sim_steps=60, D_safe_blue=25.0)
    )
    assert not np.array_equal(base.history_blues, widened.history_blues), (
        "D_safe_blue no longer affects the trajectory -- the parameter plumbing "
        "in Simulation._build_solver has regressed."
    )


def test_misspelled_simulation_kwarg_raises():
    """A typo at the Simulation boundary is caught by Python itself."""
    from simulation.testbed import Simulation

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        Simulation(num_blues=2, num_reds=1, seed=1, sim_steps=5, D_safe_bleu=3.0)


def test_solver_kwarg_discovery_sees_through_forwarding():
    """The failure M1 actually fixed, which Python does NOT catch.

    _build_solver used to filter its candidates by the solver's own signature.
    The solver declares **kwargs and forwards to CBFFilterMixin, so every CBF
    parameter looked unsupported and was dropped in silence -- a config value
    that simply never arrived. Discovery must walk the MRO.
    """
    from simulation.decentralized_solver import DecentralizedPursuitEvasionSolver
    from simulation.testbed import Simulation

    accepted = Simulation._accepted_solver_kwargs(DecentralizedPursuitEvasionSolver)
    for name in HISTORICALLY_ACTIVE:
        assert name in accepted, (
            f"{name} is not discoverable as a solver kwarg, so _build_solver "
            "would drop it silently -- this is the M1 regression."
        )


def test_unconsumable_candidate_kwarg_raises():
    """A candidate nobody in the solver chain accepts must raise, not vanish."""
    from simulation.decentralized_solver import DecentralizedPursuitEvasionSolver
    from simulation.testbed import Simulation

    sim = build_simulation(seed=1, sim_steps=5)
    # Save the raw descriptor, not the resolved function: _accepted_solver_kwargs
    # is a staticmethod, and restoring it by plain assignment would rebind it as
    # an instance method and break every later test in the session.
    original_descriptor = Simulation.__dict__["_accepted_solver_kwargs"]
    original_fn = original_descriptor.__func__

    # Pretend the solver stopped accepting a parameter we still send.
    try:
        Simulation._accepted_solver_kwargs = staticmethod(
            lambda cls: original_fn(cls) - {"D_safe_blue"}
        )
        with pytest.raises(TypeError, match="accepts none of"):
            sim._build_solver()
    finally:
        Simulation._accepted_solver_kwargs = original_descriptor
    # Restoration must be complete, or the failure surfaces in an unrelated test.
    assert "D_safe_blue" in Simulation._accepted_solver_kwargs(
        DecentralizedPursuitEvasionSolver
    )
