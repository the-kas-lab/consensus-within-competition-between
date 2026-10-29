"""The auction step 4: the allocator can choose what a DEFEND agent screens.

 Drives the solver directly with
fabricated beliefs, the decisive shape test_information_isolation.py uses: these
tests establish what CAN reach the controller, rather than observing what
happens to.

The first test is the one that matters most. This is the only place the auction touches
existing motion, so the default must be unchanged.

**Controls are compared with ``atol=1e-9``, not bitwise**, following
test_information_isolation.py. Measured on the UNMODIFIED solver: two
consecutive in-process solves with identical inputs differ by ~5.8e-14, because
solver state survives ``reset_warm_starts()``. Episode-level byte-identity is a
separate, stronger claim, and the golden tests are what hold it -- a single
solve here cannot.
"""

import numpy as np
import pytest

from conftest import build_simulation
from planning.roles import BlueRole

D = BlueRole.DEFEND


def _prepared(num_blues=2, num_reds=2):
    sim = build_simulation(num_blues=num_blues, num_reds=num_reds)
    sim.setup_agents()
    sim.solver.set_fixed_adjacency_from_state(sim.x_current, sim.red_state)
    return sim


def _red(sim, *positions):
    rows = [np.array([x, y, 0.0, 0.0, 0.0])[: sim.red_nx] for x, y in positions]
    return np.concatenate(rows)


def _solve(sim, believed, red_state=None, roles=None, **kw):
    """One solve from a cold start, so consecutive solves are comparable --
    warm starts would otherwise make the second solve differ from the first for
    reasons unrelated to its inputs."""
    sim.solver.reset_warm_starts()
    roles = roles or {i: D for i in range(sim.num_blues)}
    u, _, _, _ = sim.solver.solve_decentralized(
        x_current=sim.x_current,
        red_states=_red(sim, (80, 20), (20, 80)) if red_state is None else red_state,
        role_assignment=roles,
        believed_targets=believed,
        **kw,
    )
    return np.asarray(u).copy(), dict(sim.solver.effective_targets)


def _belief(sim, **contacts):
    """{contact_id: (x, y)} -> the solver's believed_targets shape, agent 0."""
    per = {int(k[1:]): np.array([x, y, 0.0, 0.0, 0.0])[: sim.red_nx] for k, (x, y) in contacts.items()}
    return {0: per, 1: {}}


# --- default-off: byte-identical --------------------------------------------

def test_the_default_leaves_every_defend_agent_exactly_as_before():
    sim = _prepared()
    believed = _belief(sim, c0=(30, 30), c1=(70, 70))
    omitted = _solve(sim, believed)
    none = _solve(sim, believed, defend_target_override=None)
    empty = _solve(sim, believed, defend_target_override={})
    for other in (none, empty):
        assert np.allclose(omitted[0], other[0], atol=1e-9), "controls moved"
        assert omitted[1] == other[1], "effective targets moved"


# --- SCREEN: a chosen contact, not the nearest one ---------------------------

def test_screen_steers_at_the_chosen_contact_not_the_nearest():
    """Contacts on different sides of the defended region, so their
    interposition points differ -- asserted as a precondition, because the
    first version of this test put both contacts off the same side, where they
    project to the SAME perimeter vertex and the controller correctly produces
    the same control. A SCREEN test whose contacts share a screen point cannot
    tell a working override from a broken one."""
    sim = _prepared()
    controller = sim.solver._blue_role_controllers[D]
    me = sim.x_current[0:2]
    near, far = (90.0, 50.0), (40.0, 82.0)
    assert np.linalg.norm(me - near) < np.linalg.norm(me - far), "precondition: c0 is nearest"
    assert not np.allclose(
        controller._closest_point_on_polygon(np.array(near)),
        controller._closest_point_on_polygon(np.array(far)),
    ), "precondition: the two contacts must have different screen points"
    believed = _belief(sim, c0=near, c1=far)

    default_u, default_eff = _solve(sim, believed)
    assert default_eff[0]["contact_id"] == 0, "precondition: default screens the nearest"

    chosen_u, chosen_eff = _solve(sim, believed, defend_target_override={0: ("contact", 1)})
    assert chosen_eff[0] == {"source": "screen", "contact_id": 1, "controller": "defend"}
    assert not np.allclose(default_u[0:3], chosen_u[0:3], atol=1e-6), "the choice must move the agent"


def test_screen_reads_the_agents_belief_not_the_truth():
    """L1-shaped. Same belief about contact 1, two wildly different true red
    positions: if ground truth could reach the screening controller, the two
    solves would differ."""
    sim = _prepared()
    believed = _belief(sim, c0=(30, 30), c1=(60, 20))
    override = {0: ("contact", 1)}
    a, _ = _solve(sim, believed, red_state=_red(sim, (80, 20), (20, 80)), defend_target_override=override)
    b, _ = _solve(sim, believed, red_state=_red(sim, (10, 90), (95, 5)), defend_target_override=override)
    assert np.allclose(a[0:3], b[0:3], atol=1e-9)


def test_screening_a_contact_no_longer_believed_falls_back_rather_than_inventing_one():
    sim = _prepared()
    believed = _belief(sim, c0=(30, 30))
    _, eff = _solve(sim, believed, defend_target_override={0: ("contact", 9)})
    assert eff[0]["source"] == "nearest_believed" and eff[0]["contact_id"] == 0


# --- DEFEND(station): holds even while contacts are believed -----------------

def test_a_station_is_held_even_while_a_contact_is_believed():
    """What the default cannot do: any believed contact overrides the station
    there. As an allocated task it must not."""
    sim = _prepared()
    believed = _belief(sim, c0=(30, 30))
    _, default_eff = _solve(sim, believed)
    assert default_eff[0]["source"] == "nearest_believed", "precondition"

    _, eff = _solve(sim, believed, defend_target_override={0: ("station", 1)})
    assert eff[0]["source"] == "station" and eff[0]["contact_id"] is None


def test_the_override_is_ignored_for_agents_not_in_defend():
    """Checks the effective-target log as well as the controls. A mutation that
    routed a RECON agent into the SCREEN branch survived a controls-only version
    of this test: RECON ignores the target state, so the motion was unchanged --
    but the agent was logged as screening a contact it was not pursuing. That
    log is needed
    to say what the team was chasing at all, so corrupting it silently is a
    real failure even when nothing moves."""
    sim = _prepared()
    believed = _belief(sim, c0=(30, 30))
    roles = {0: BlueRole.RECON, 1: D}
    a = _solve(sim, believed, roles=roles)
    b = _solve(sim, believed, roles=roles, defend_target_override={0: ("contact", 0)})
    assert np.allclose(a[0], b[0], atol=1e-9)
    assert a[1] == b[1], "the effective-target log must not relabel a non-DEFEND agent"
    assert b[1][0]["source"] == "recon_waypoint"
