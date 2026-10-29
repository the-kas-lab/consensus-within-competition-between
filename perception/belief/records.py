"""Belief records: what an agent knows about one thing, and how it came to know it.

Design commitments, all from and the decisions in
:

* **Contacts are keyed by an opaque ``contact_id``, never a raw red index.**
  For Paper 1 the mapping is the identity, so this costs nothing. It means
  Experiment II's shuffled contact roster can be reintroduced later without
  touching the belief, task or allocation layers -- see 01-plan-vs-repo.md Sec C5.
* **Relaying must never rejuvenate an observation.** ``observed_at`` is the time
  the *sensor* saw it. A belief relayed twice is still as old as its original
  sighting, which is the entire point of tracking age.
* **Nothing is ever deleted on a timeout.** Consumers apply their own validity
  threshold; the store keeps stale records for analysis and later reconciliation.
  A hard global timeout would destroy exactly the evidence the semantic-attack
  work needs (claim persistence, belief error over time).
"""

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Provenance:
    """Where a belief came from, and when.

    ``observed_at`` is sensor time and is immutable through any number of
    relays. ``received_at`` is when *this* agent learned it -- equal to
    ``observed_at`` for a first-hand observation. The gap between them is
    communication latency, and it is what's age-of-information
    metrics measure.
    """

    origin_agent: int
    observed_at: int
    received_at: int
    relay_agent: Optional[int] = None
    source_sequence: int = 0
    planning_epoch: Optional[int] = None

    @property
    def is_first_hand(self):
        return self.relay_agent is None

    def relayed_by(self, relay_agent, received_at):
        """A copy of this provenance as seen one hop further on.

        ``observed_at`` and ``origin_agent`` deliberately survive unchanged.
        """
        return replace(self, relay_agent=relay_agent, received_at=int(received_at))


@dataclass
class TargetBelief:
    """One agent's last-known state for one contact.

    ``state`` is the raw state vector as observed (4-wide, or 5-wide when the
    contact carries a heading), stored whole so a consumer can use velocity for
    prediction without a second lookup.
    """

    contact_id: int
    state: np.ndarray
    provenance: Provenance

    def __post_init__(self):
        self.state = np.asarray(self.state, dtype=float).ravel().copy()

    @property
    def position(self):
        return self.state[0:2]

    @property
    def velocity(self):
        return self.state[2:4]

    def age(self, now):
        """Steps since the observation was *made*, not since it arrived."""
        return int(now) - int(self.provenance.observed_at)

    def predicted_position(self, now, dt):
        """Constant-velocity extrapolation from the observation forward.

        p_hat(t) = p(t_o) + v(t_o) * (t - t_o)

        With ``age == 0`` this returns the observed position exactly, which is
        what makes ``belief_mode="ground_truth"`` reproduce pre-belief behavior
        byte-for-byte.
        """
        return self.position + self.velocity * (self.age(now) * float(dt))

    def predicted_state(self, now, dt):
        """The full state vector with position advanced; velocity and heading
        are carried forward unchanged, since nothing is observed about how they
        evolved after ``observed_at``."""
        out = self.state.copy()
        out[0:2] = self.predicted_position(now, dt)
        return out

    def uncertainty_radius(self, now, dt, v_max):
        """Conservative reachable-set radius: rho(t) = v_max * (t - t_o).

 asks for the indicator, not exact set propagation. A consumer
        decides whether a belief this uncertain is usable.
        """
        return float(v_max) * self.age(now) * float(dt)

    def is_fresh(self, now, max_age):
        """``max_age is None`` means "no staleness limit" -- the store's own
        semantics, not a consumer's."""
        return True if max_age is None else self.age(now) <= int(max_age)


@dataclass
class CoverageBelief:
    """One agent's belief about when each arena cell was last swept.

    Replaces the single team-global CoverageTracker for decision purposes. Own
    sweeps update it directly; a teammate's sweep reaches it only through
    communication, which is what makes coverage a *distributed* quantity rather
    than a free team oracle.

    ``-1`` means never swept, matching CoverageTracker's existing convention so
    the two are directly comparable.
    """

    n_regions: int
    last_seen_step: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.last_seen_step is None:
            self.last_seen_step = np.full(int(self.n_regions), -1, dtype=int)
        else:
            self.last_seen_step = np.asarray(self.last_seen_step, dtype=int).copy()

    def observe(self, region_id, step):
        region_id = int(region_id)
        if step > self.last_seen_step[region_id]:
            self.last_seen_step[region_id] = int(step)

    def merge(self, other):
        """Per: coverage merges by maximum ``swept_at``, per cell.

        Monotone and idempotent, so duplicate or reordered delivery cannot
        regress a cell.
        """
        np.maximum(self.last_seen_step, other.last_seen_step, out=self.last_seen_step)

    def steps_since_seen(self, now):
        return np.where(self.last_seen_step < 0, -1, int(now) - self.last_seen_step)

    def summarize(self, now, centers):
        """Same shape CoverageTracker.summarize emits, so planners and the
        prompt builder consume either without branching."""
        since = self.steps_since_seen(now)
        return [
            {
                "region_id": idx,
                "center": centers[idx],
                "steps_since_seen": int(since[idx]),
            }
            for idx in range(int(self.n_regions))
        ]

    def copy(self):
        return CoverageBelief(self.n_regions, self.last_seen_step)


@dataclass(frozen=True)
class AgentStatus:
    """One agent's belief about whether *another* agent is still alive.

    Separate from ``TargetBelief`` because it merges on a different rule and for
    a different reason. A target belief is a claim about the world that is
    superseded by a fresher sighting; a status is a claim about an agent, and in
    this model **an agent never revives** -- ``testbed._resolve_red_neutralized``
    disables a blue permanently. So "disabled" is *absorbing*, not merely newer.

    ``as_of`` is the step the claim refers to: the disable step for a tombstone,
    or the observation step for a liveness report.

    Why this matters for correctness: without absorbing semantics, a SELF_STATE
    broadcast that was in flight when an agent died would arrive afterwards and
    silently resurrect it in a teammate's model. The teammate would then rank a
    dead agent in the team-relative role split and assign it work, which the
    planner has no way to detect. See ``perception.belief.merge.merge_status``.
    """

    agent_id: int
    disabled: bool
    as_of: int

    @property
    def is_tombstone(self):
        return self.disabled
