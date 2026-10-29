"""Shared fixtures for the pursuit-evasion test suite.

Design notes, because they constrain every test written against this:

* **Episodes are the unit under test.** There is no way to exercise the
  planner/controller/CBF interaction without running one, so the fixtures here
  make a *fast* episode cheap (2 blues, 1 red, tens of steps) rather than trying
  to mock the stack apart.
* **CBF is on by default here.** ``Simulation``'s own CBF defaults are all
  ``False`` and ``main.py`` runs unfiltered, but every experiment script enables
  them -- so tests assert against the configuration research actually uses.
* **stdout is captured at file-descriptor level, not via ``capsys``.** CasADi and
  OSQP print from C, which bypasses Python-level redirection entirely.
"""

import contextlib
import hashlib
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planning.planner import RuleBasedBlueRolePlanner, RuleBasedRedRolePlanner  # noqa: E402
from simulation.testbed import Simulation  # noqa: E402

# Bid-version presets for the bid tests (tests/test_bid_*.py). The reported runs use
# only the "v2.3mb" design; these name the earlier bid versions the library still has.
from scripts import experiment_policy as _ep  # noqa: E402

for _name in ("v2", "v2.2", "v2.3"):
    _ep.DESIGNS.setdefault(_name, {"cbaa_bid_version": _name, "station_heading": "configured"})

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")


@contextlib.contextmanager
def capture_fd_output():
    """Capture stdout+stderr at the OS file-descriptor level.

    Necessary because CasADi/OSQP write from C: ``capsys`` and
    ``contextlib.redirect_stdout`` both swap Python-level objects and see none
    of it. Yields a callable returning everything written so far.
    """
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        saved_out, saved_err = os.dup(1), os.dup(2)
        py_out, py_err = sys.stdout, sys.stderr
        try:
            py_out.flush()
            py_err.flush()
            os.dup2(tmp.fileno(), 1)
            os.dup2(tmp.fileno(), 2)
            # Redirecting the file descriptors alone is not enough under pytest:
            # it has already swapped sys.stdout for an object that never touches
            # fd 1, so Python-level prints would bypass the capture entirely.
            # Point sys.stdout/stderr back at the (now redirected) descriptors.
            sys.stdout = os.fdopen(os.dup(1), "w", buffering=1)
            sys.stderr = os.fdopen(os.dup(2), "w", buffering=1)

            def read_back():
                sys.stdout.flush()
                sys.stderr.flush()
                pos = tmp.tell()
                tmp.seek(0)
                data = tmp.read().decode("utf-8", errors="replace")
                tmp.seek(pos)
                return data

            yield read_back
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
                sys.stdout.close()
                sys.stderr.close()
            except (ValueError, OSError):
                pass
            sys.stdout, sys.stderr = py_out, py_err
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)


def build_simulation(**overrides):
    """A small, fully-specified episode. Every parameter that affects the
    trajectory is pinned explicitly -- a test must never depend on a
    ``Simulation`` default, because those defaults are exactly what the research
    work will be changing."""
    params = dict(
        num_blues=2,
        num_reds=1,
        seed=7,
        sim_steps=40,
        dt=0.1,
        horizon=60,
        arena_size=100.0,
        planning_interval_seconds=1.0,
        enable_blue_blue_cbf=True,
        enable_blue_red_cbf=True,
        enable_red_cbf=False,
        enable_convex_hull_containment=False,
    )
    params.update(overrides)
    params.setdefault(
        "role_planner", RuleBasedBlueRolePlanner(recon_fraction=0.5, min_recon=1)
    )
    params.setdefault("red_role_planner", RuleBasedRedRolePlanner())
    return Simulation(**params)


def run_quiet(sim):
    """Run an episode, discarding solver/planner chatter."""
    with capture_fd_output():
        sim.run()
    return sim


def fingerprint(sim):
    """A compact, comparable summary of everything an episode *did*.

    Hashes rather than raw arrays so golden files stay small and a diff is a
    single line; the per-field split means a regression says *which* layer moved
    (blue motion vs. red motion vs. role assignment vs. game outcome) instead of
    just "something changed".
    """

    def h(arr):
        return hashlib.sha256(
            np.ascontiguousarray(np.asarray(arr, dtype=float)).tobytes()
        ).hexdigest()[:16]

    roles = [
        {int(k): v.value for k, v in step.items()} for step in sim.role_assignment_history
    ]
    red_roles = [
        {int(k): v.value for k, v in step.items()}
        for step in sim.red_role_assignment_history
    ]
    return {
        "n_steps": len(sim.history_blues),
        "blue_traj": h(sim.history_blues),
        "red_traj": h(sim.history_red),
        "roles": hashlib.sha256(repr(roles).encode()).hexdigest()[:16],
        "red_roles": hashlib.sha256(repr(red_roles).encode()).hexdigest()[:16],
        "outcome": {
            "touchdown": bool(sim.touchdown),
            "num_neutralized": int(sim.num_neutralized),
            "num_escaped": int(sim.num_escaped),
            "blue_score": round(float(sim.blue_score), 9),
            "solver_failed": bool(sim.solver_failed),
            "blue_disabled": [bool(x) for x in sim.blue_disabled],
        },
    }


@pytest.fixture
def make_sim():
    """Factory fixture: ``make_sim(num_reds=2)`` -> a configured Simulation."""
    return build_simulation


@pytest.fixture
def run_episode():
    """Factory fixture: ``run_episode(seed=3)`` -> a completed Simulation."""

    def _run(**overrides):
        return run_quiet(build_simulation(**overrides))

    return _run


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Re-record tests/golden/*.json instead of asserting against it. "
             "Use ONLY when a behavior change is intended and understood; the "
             "diff of the golden file is the record of what moved.",
    )


@pytest.fixture(scope="session")
def update_golden(request):
    return request.config.getoption("--update-golden")
