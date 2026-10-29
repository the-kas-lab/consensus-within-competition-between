"""Exp 5's communication plumbing and its arm-identity pin.

 its
amendments. Three things must hold before any episode:

1. **A loss model is a plain cfg spec, built fresh for every episode.** The
   Gilbert-Elliott model memoizes per-link state, so one instance shared across
   episodes is excluded by construction and by test. Absent, nothing changes.
2. **Loss is a per-link, per-round erasure** (Amendment 1). Every envelope on
   one link in one round -- belief and auction, both directions -- shares one
   draw. Checked on a live Simulation, not on the model alone.
3. **Arm I is arm C with its auction traffic removed, and nothing else**, under
   ``networked`` belief with range and loss: the networked counterpart of
   tests/test_arm_variants.py. Arm C's auction envelopes are SENT (so they cross
   the same links and would consume draws, were draws consumed) and stripped
   at delivery. The result must be byte-identical to arm I.
"""

import json
import os
import sys
from collections import defaultdict

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import build_simulation, fingerprint, run_quiet  # noqa: E402
from comms.link_models import (  # noqa: E402
    GilbertElliottLossModel,
    IidLossModel,
    loss_model_from_spec,
)
from comms.messages import AUCTION_MESSAGE, TARGET_UPDATE  # noqa: E402
from scripts.episode_builder import build_episode  # noqa: E402

IID = {"model": "iid", "p": 0.3}
GE = {"model": "gilbert_elliott", "p_gb": 0.15, "p_bg": 0.30, "loss_good": 0.0, "loss_bad": 0.9}
# The Exp 4 arm-identity fixture (3 blues / 2 reds / 300 steps / seed 7, the
# configuration measured to believe contacts), under networked belief at 0.1 s.
FIXTURE = dict(num_blues=3, num_reds=2, sim_steps=300, seed=7, belief_mode="networked",
               comm_interval_seconds=0.1, allocation="cbaa", cbaa_fallback="station",
               station_heading="outward")
PILOT_CFG = dict(blues=4, reds=2, steps=5, planning_interval_seconds=1.0, rule="nearest",
                 neutralize_fraction=0.5, recon_fraction=0.5, family="symmetric",
                 allocation="cbaa", cbaa_transport="network", comm_interval_seconds=0.1)


# -- 1. the spec ------------------------------------------------------------------


def test_the_spec_builds_the_declared_models():
    iid = loss_model_from_spec(IID)
    ge = loss_model_from_spec(GE)
    assert isinstance(iid, IidLossModel) and iid.loss_probability == 0.3
    assert isinstance(ge, GilbertElliottLossModel)
    assert ge.describe() == {"model": "gilbert_elliott", **{k: GE[k] for k in GE if k != "model"}}
    assert loss_model_from_spec(None) is None
    with pytest.raises(ValueError):
        loss_model_from_spec({"model": "fade"})
    with pytest.raises(ValueError):
        loss_model_from_spec({"model": "iid", "p": 0.3, "burst": 2})


def test_every_episode_gets_its_own_loss_model_and_the_cfg_is_untouched():
    cfg = dict(PILOT_CFG, comm_loss_model=dict(GE))
    before = json.dumps(cfg, sort_keys=True)
    a = build_episode(3705, "networked", cfg)
    b = build_episode(3705, "networked", cfg)
    assert json.dumps(cfg, sort_keys=True) == before
    # The Network is built when the episode runs, from this instance.
    assert isinstance(a.comm_loss_model, GilbertElliottLossModel)
    assert a.comm_loss_model is not b.comm_loss_model
    assert a.comm_loss_model._cache == {} and b.comm_loss_model._cache == {}
    a.comm_loss_model.delivers(3705, 0, 1, 40)
    assert b.comm_loss_model._cache == {}, "one episode's memo must not reach another"


def test_the_network_uses_the_episodes_own_instance():
    sim = build_episode(3705, "networked", dict(PILOT_CFG, comm_loss_model=dict(IID)))
    run_quiet(sim)
    assert sim.network.loss_model is sim.comm_loss_model


def test_no_spec_is_no_loss_exactly_as_before():
    sim = build_episode(3705, "networked", dict(PILOT_CFG))
    assert sim.comm_loss_model is None
    run_quiet(sim)
    assert sim.network.loss_model.describe() == {"model": "none"}


# -- 2. one draw per link per round, on a live Simulation ----------------------------


def _on_network(sim, install):
    """Run ``install(network)`` once, as soon as the episode builds its Network
    (inside ``run``), before that round delivers or sends anything:
    ``_record_connectivity`` runs right after ``begin_round``."""
    original = sim._record_connectivity
    done = []

    def record():
        if not done:
            install(sim.network)
            done.append(True)
        return original()

    sim._record_connectivity = record
    return sim


def _outcomes_per_link_round(spec, r_comm=float("inf")):
    sim = build_simulation(cbaa_transport="network", r_comm=r_comm,
                           comm_loss_model=loss_model_from_spec(spec), **FIXTURE)
    seen = defaultdict(set)   # (tick, link) -> {delivered?}
    kinds = defaultdict(set)  # (tick, link) -> {message types}

    def install(net):
        original = net.send

        def send(envelope):
            receivers = net.neighbours(envelope.sender)
            before = {r: len(net._outbox[r]) for r in receivers}
            accepted = original(envelope)
            for r in receivers:
                link = tuple(sorted((int(envelope.sender), int(r))))
                seen[(net.comm_tick, link)].add(len(net._outbox[r]) > before[r])
                kinds[(net.comm_tick, link)].add((envelope.message_type, int(envelope.sender)))
            return accepted

        net.send = send

    run_quiet(_on_network(sim, install))
    return seen, kinds, sim.network.stats


@pytest.mark.parametrize("spec", [IID, GE])
def test_every_envelope_on_a_link_in_a_round_shares_one_draw(spec):
    seen, kinds, stats = _outcomes_per_link_round(spec)
    both = [k for k in seen
            if {m for m, _ in kinds[k]} >= {TARGET_UPDATE, AUCTION_MESSAGE}
            and len({s for _, s in kinds[k]}) == 2]
    # Not vacuous: many link-rounds carried belief AND auction traffic in BOTH
    # directions, and the channel did erase some of them.
    assert len(both) > 500
    assert stats["dropped_loss"] > 0 and stats["delivered"] > 0
    assert all(len(seen[k]) == 1 for k in seen), "one link-round, one outcome"


# -- 3. the networked arm-identity pin --------------------------------------------------


def _strip_auction_on_delivery(sim):
    """Arm C, with every auction envelope removed at DELIVERY: sent, carried,
    subject to the same links and draws, never received."""
    def install(net):
        original = net.deliver

        def deliver():
            out = {}
            for agent, envelopes in original().items():
                kept = [e for e in envelopes if e.message_type != AUCTION_MESSAGE]
                if kept:
                    out[agent] = kept
            return out

        net.deliver = deliver

    return _on_network(sim, install)


def _build(transport, spec, r_comm):
    return build_simulation(cbaa_transport=transport, r_comm=r_comm,
                            comm_loss_model=loss_model_from_spec(spec), **FIXTURE)


def _recorded(sim):
    return json.dumps(sim.recorder.steps, sort_keys=True, default=str)


def _allocation_sans_traffic(sim):
    m = dict(sim.episode_outcome()["allocation"])
    for k in ("transport", "auction_messages_sent", "stale_messages_dropped"):
        m.pop(k)
    return m


def _belief_traffic(sim):
    c = sim.comm_metrics()
    return {k: c[k] for k in ("delivered", "merged_targets", "connected_fraction",
                              "mean_lcc_fraction", "n_partitions", "mean_partition_steps")}


@pytest.mark.parametrize("spec,r_comm", [(IID, 20.0), (GE, float("inf"))])
def test_arm_c_with_its_auction_traffic_stripped_is_arm_i_byte_for_byte(spec, r_comm):
    arm_i = run_quiet(_build("none", spec, r_comm))
    stripped = run_quiet(_strip_auction_on_delivery(_build("network", spec, r_comm)))
    arm_c = run_quiet(_build("network", spec, r_comm))

    assert fingerprint(stripped) == fingerprint(arm_i)
    assert _recorded(stripped) == _recorded(arm_i)
    assert _allocation_sans_traffic(stripped) == _allocation_sans_traffic(arm_i)
    assert _belief_traffic(stripped) == _belief_traffic(arm_i)
    # The auction traffic was really there, on the same channel.
    assert stripped.network.stats["sent"] > arm_i.network.stats["sent"]
    assert stripped.network.stats["dropped_loss"] > arm_i.network.stats["dropped_loss"]
    # Not vacuous: delivered, the traffic changes the episode.
    assert fingerprint(arm_c) != fingerprint(arm_i)


def test_the_arms_are_configured_identically_except_transport():
    one = _build("none", IID, 20.0)
    two = _build("network", IID, 20.0)
    for attr in ("slots", "stable_rounds_config", "profile", "modifier_source", "fallback"):
        assert getattr(one.cbaa_runtime, attr) == getattr(two.cbaa_runtime, attr), attr
    assert one.comm_loss_model.describe() == two.comm_loss_model.describe()
    assert one.r_comm == two.r_comm
    assert one.comm_interval_steps == two.comm_interval_steps == 1
