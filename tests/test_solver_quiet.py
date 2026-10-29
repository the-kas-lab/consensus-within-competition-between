"""Regression for the OSQP verbosity fix in control/base.py.

CasADi namespaces plugin options: a bare {"verbose": False} sets *CasADi's*
verbosity, not OSQP's, so the solver's iteration banner was printed on every
single MPC solve -- thousands of lines per episode, which buries planner output
and makes sweep logs unusable. The OSQP flag has to be nested under the plugin
name.

Verbosity only; the fix was verified to leave trajectories byte-identical.
"""

from conftest import build_simulation, capture_fd_output

OSQP_BANNER_MARKERS = ("iter   objective", "optimal rho estimate", "primal-dual integral")


def test_episode_does_not_print_osqp_banner():
    sim = build_simulation(sim_steps=20)
    with capture_fd_output() as read_back:
        sim.run()
        output = read_back()

    for marker in OSQP_BANNER_MARKERS:
        assert marker not in output, (
            f"OSQP solver banner leaked to stdout (marker {marker!r}). "
            "Check _OSQP_OPTS in control/base.py."
        )


def test_planner_output_still_visible():
    """The fix must silence the solver without silencing the planner -- the
    per-cycle role log is the main human-readable trace of an episode."""
    sim = build_simulation(sim_steps=20)
    with capture_fd_output() as read_back:
        sim.run()
        output = read_back()

    assert "[planner]" in output
