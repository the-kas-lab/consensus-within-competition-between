"""Exp 4's arm 1 is CBAA with the auction traffic removed, and nothing else.

The independent arm's "independent
proposals" must mean each agent selects from the SAME task generator and bid
function CBAA uses -- SCREEN, station holding, retention and controller mapping
all identical -- or the comparison confounds conflict resolution with a changed
policy. Arm 1 is ``allocation="cbaa", cbaa_transport="none"``, and these tests
establish that it differs from arm 2 in auction traffic only.

The decisive test is an identity, not a similarity: take an arm-2 episode,
silence its auction traffic (no state delivered in any round), and it must be
byte-identical to arm 1 -- recorder, trajectories, roles, allocation metrics
other than the message counts.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import build_simulation, fingerprint, run_quiet  # noqa: E402
from coordination.runtime import CBAARuntime  # noqa: E402

FIXTURE = dict(num_blues=3, num_reds=2, sim_steps=300, seed=7, belief_mode="shared",
               allocation="cbaa")


class _Silenced(CBAARuntime):
    """Arm 2 with its auction traffic removed: every round delivers nothing.
    Everything else -- tasks, bids, stability rule, commitment mapping -- is the
    arm-2 object's own."""

    def local_round(self, active):
        self._outbox = {}
        return super().local_round(active)


def _build(**overrides):
    params = dict(FIXTURE)
    params.update(overrides)
    return build_simulation(**params)


def _recorded(sim):
    return json.dumps(sim.recorder.steps, sort_keys=True, default=str)


def _allocation_sans_traffic(sim):
    m = dict(sim.episode_outcome()["allocation"])
    for k in ("transport", "auction_messages_sent", "stale_messages_dropped"):
        m.pop(k)
    return m


def test_arm1_and_arm2_are_configured_identically_except_transport():
    one = _build(cbaa_transport="none").cbaa_runtime
    two = _build(cbaa_transport="ideal").cbaa_runtime
    assert type(one) is type(two) is CBAARuntime
    assert one.transport == "none" and two.transport == "ideal"
    for attr in ("slots", "stable_rounds_config", "profile", "modifier_source"):
        assert getattr(one, attr) == getattr(two, attr), attr


def test_cbaa_with_its_auction_traffic_silenced_is_arm1_byte_for_byte():
    arm1 = run_quiet(_build(cbaa_transport="none"))

    silenced = _build(cbaa_transport="ideal")
    silenced.cbaa_runtime.__class__ = _Silenced
    silenced = run_quiet(silenced)

    arm2 = run_quiet(_build(cbaa_transport="ideal"))

    assert fingerprint(silenced) == fingerprint(arm1)
    assert _recorded(silenced) == _recorded(arm1)
    assert _allocation_sans_traffic(silenced) == _allocation_sans_traffic(arm1)
    # Not vacuous: the traffic does change the episode.
    assert fingerprint(arm2) != fingerprint(arm1)
    kinds = {v for st in arm1.recorder.steps for v in st["blue_roles"].values()}
    assert len(kinds) >= 2, "arm 1 must exercise more than one role on this fixture"


def test_slots_do_not_exist_for_arm1():
    """With no traffic, a second NEUTRALIZE slot is an equal-bid copy the
    canonical order never selects over slot 0 -- so arm 1 has no slots
    dimension, and one arm-1 run serves both CBAA arms."""
    s1 = run_quiet(_build(cbaa_transport="none", cbaa_slots=1))
    s2 = run_quiet(_build(cbaa_transport="none", cbaa_slots=2))
    assert fingerprint(s1) == fingerprint(s2)
    assert _recorded(s1) == _recorded(s2)
