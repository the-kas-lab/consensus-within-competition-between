"""The station-heading rule: the explicit DEFEND(station) posture faces outward ().

The auction smoke found that a blue holding a station with nothing detected never
turns -- no heading term is active -- so what it can see depended on the heading
its last motion left it with. the station-heading rule(a) gives the posture a surveillance heading:
the direction from the defended centre through the assigned station, reached
through the existing bearing-tracking term and yaw-rate bound, with
detection-driven tracking taking precedence. It is a behaviour change motivated
by the pilot; the "inherited" default keeps every existing mode and golden.

Every test drives a live Simulation, in BOTH the CBAA and the independent
(transport "none") arms: the posture must apply identically across Exp 4's arms.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import build_simulation, fingerprint, run_quiet  # noqa: E402

FIXTURE = dict(num_blues=3, num_reds=2, sim_steps=300, seed=7, belief_mode="shared",
               allocation="cbaa")
TRANSIENT_STEPS = 10  # 1 s: omega_max is 2*pi rad/s, so any turn fits inside it


def _run(**overrides):
    params = dict(FIXTURE)
    params.update(overrides)
    sim = build_simulation(**params)
    log = []
    original = sim.solver.solve_decentralized

    def spy(*args, **kwargs):
        out = original(*args, **kwargs)
        log.append((dict(kwargs.get("bearing_target_override") or {}),
                    dict(sim.solver.bearing_sources)))
        return out

    sim.solver.solve_decentralized = spy
    return run_quiet(sim), log


def _wrap_deg(rad):
    return abs((math.degrees(rad) + 180.0) % 360.0 - 180.0)


@pytest.fixture(scope="module", params=["ideal", "none"])
def outward(request):
    return _run(cbaa_transport=request.param, station_heading="outward")


def test_the_default_is_inherited_and_never_applies_the_posture():
    sim, log = _run()
    assert sim.station_heading == sim.solver.station_heading == "inherited"
    assert all(v != "station_outward" for _, b in log for v in b.values())


def test_an_unknown_posture_is_refused():
    with pytest.raises(ValueError, match="station_heading"):
        build_simulation(station_heading="sideways")


def test_only_a_committed_station_without_a_detection_gets_the_posture(outward):
    """The classification, every agent every step: detection first; else the
    outward posture iff the agent is realising DEFEND(station); else nothing.
    SCREEN, NEUTRALIZE, RECON and the uncommitted DEFEND never get it."""
    sim, log = outward
    both = 0
    for (override, sources), st in zip(log, sim.recorder.steps):
        for agent, source in sources.items():
            station = st["defend_targets"].get(str(agent), [None])[0] == "station"
            if agent in override:
                assert source == "detection"
                both += station
            elif station and st["blue_roles"][str(agent)] == "defend":
                assert source == "station_outward", (st["step"], agent)
            else:
                assert source is None, (st["step"], agent, source)
    assert both > 0, "fixture never had a station agent detect: precedence untested"


def test_a_station_holder_acquires_and_keeps_the_outward_heading(outward):
    """After a turning transient, the heading is the outward direction from the
    defended centre through the assigned station. Not required during the turn,
    in transit to another task, or while tracking."""
    sim, log = outward
    center = np.asarray(sim.defense_center, dtype=float).ravel()[:2]
    run, errors = {}, []
    for s, (_, sources) in enumerate(log[:-1]):
        st = sim.recorder.steps[s]
        for agent, source in sources.items():
            run[agent] = run.get(agent, 0) + 1 if source == "station_outward" else 0
            if run[agent] < TRANSIENT_STEPS:
                continue
            station = sim.solver.defend_stations[int(st["defend_targets"][str(agent)][1])]
            outward = math.atan2(*(np.asarray(station) - center)[::-1])
            heading = np.asarray(sim.history_blues[s + 1]).ravel()[agent * sim.blue_nx + 4]
            errors.append(_wrap_deg(heading - outward))
    assert len(errors) > 100, "fixture must hold stations long enough to measure"
    assert max(errors) < 5.0, max(errors)


def test_run_episode_passes_the_posture_through():
    """Call site: Exp 4 builds its episodes through run_episode's cfg."""
    from scripts.episode_builder import build_episode

    cfg = dict(blues=3, reds=2, steps=20, planning_interval_seconds=1.0, family="symmetric",
               rule="nearest", neutralize_fraction=0.5, recon_fraction=0.5,
               allocation="cbaa", station_heading="outward")
    assert build_episode(7, "shared", cfg).solver.station_heading == "outward"
    del cfg["station_heading"]
    assert build_episode(7, "shared", cfg).solver.station_heading == "inherited"


def test_the_posture_changes_behaviour_it_is_not_neutral():
    """Declared, not hidden: the default keeps the goldens, it does not make the
    posture free."""
    before, _ = _run()
    after, _ = _run(station_heading="outward")
    assert fingerprint(before) != fingerprint(after)
