"""LocalWorldView: the immutable snapshot a decision consumer is handed.

Everything downstream of belief -- the planner prompt, task generation, bid
computation, the allocator -- reads a view, never the store. Two reasons, both
load-bearing:

* **Frozen planning snapshots**. Every agent's decision for
  planning epoch k must be computed from beliefs frozen at the same logical
  instant. If consumers read a live store, agent 1 would see agent 0's
  same-epoch update simply because the loop ran in that order.
* **It is the enforcement point for information isolation.** asks
  that an unseen red appear in none of the observation, the prompt, task
  generation, bidding or the assignment protocol. One object to audit.

``teammates`` arrived at M3 and is deliberately NOT belief-gated -- see
``LocalWorldView.teammates`` below and the decentralization rule. Status tombstones and intent
still join when the communication layer gives them a way to arrive (M4) -- see

"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from perception.belief.records import CoverageBelief, TargetBelief


@dataclass(frozen=True)
class LocalWorldView:
    """What one agent knows, frozen at ``now``.

    ``teammates`` is ``{agent_id: state}`` for every *other active* blue, and it
    is **ground truth in every belief mode** -- the one field here that is not
    belief-gated. That is the decentralization rule commitment 1, and it is a declaration of
    something already true rather than a new channel: ``solve_decentralized``
    takes ``x_current`` whole and ``PlannerObservation``'s docstring already
    states "A team's own agents are always fully visible". Putting it here makes
    the assumption auditable at the one place asks us to audit.

    What follows from that, and is load-bearing for Experiment 0:

    * The lower arm withholds **target observations**, not all communication.
      It is never described as "zero communication".
    * Teammate state is *held fixed* across both arms, so it cannot explain the
      difference between them. M4 is where it stops being held fixed.
    * **Liveness is part of the channel.** Only active blues appear, because the
      centralized planner ranks only active blues (``obs.disabled_blues``) and
      the per-agent decomposition has to rank the same set or the equivalence
      gate cannot hold. M4 turns this into a status belief with tombstones.

    Self is excluded -- ``self_state`` already carries it. Use
    ``active_blue_states()`` for the full set an agent assigns roles over.
    """

    now: int
    dt: float
    agent_id: int
    self_state: np.ndarray
    targets: Dict[int, TargetBelief] = field(default_factory=dict)
    coverage: Optional[CoverageBelief] = None
    coverage_centers: Optional[np.ndarray] = None
    teammates: Dict[int, np.ndarray] = field(default_factory=dict)

    def active_blue_states(self):
        """``{agent_id: state}`` for every blue this agent assigns a role over,
        itself included, ordered by ``agent_id`` so a ranking built from it is
        deterministic.
        """
        merged = {int(self.agent_id): self.self_state}
        merged.update({int(i): s for i, s in self.teammates.items()})
        return {i: merged[i] for i in sorted(merged)}

    def fresh_targets(self, max_age=None):
        """Targets this agent considers usable for a decision.

        ``max_age=None`` returns everything the agent knows, stale included --
        the store never deletes, so a consumer that wants a staleness limit has
        to ask for one.
        """
        return {
            cid: b for cid, b in self.targets.items() if b.is_fresh(self.now, max_age)
        }

    def nearest_target(self, position, max_age=None):
        """The believed contact nearest ``position``, by *predicted* position.

        Returns ``(contact_id, belief)`` or ``(None, None)`` when the agent
        believes in no usable contact -- which is a normal state under fog of
        war, not an error, and is what the caller's fallback exists for.
        """
        candidates = self.fresh_targets(max_age)
        if not candidates:
            return None, None
        position = np.asarray(position, dtype=float).reshape(2)
        best_id = min(
            candidates,
            key=lambda cid: (
                float(np.linalg.norm(position - candidates[cid].predicted_position(self.now, self.dt))),
                cid,  # stable tiebreak, so the view is deterministic
            ),
        )
        return best_id, candidates[best_id]

    def coverage_regions(self):
        """CoverageTracker.summarize-compatible output, so existing planner and
        prompt code consumes a belief without branching."""
        if self.coverage is None or self.coverage_centers is None:
            return None
        return self.coverage.summarize(self.now, self.coverage_centers)

    def known_contact_ids(self):
        return sorted(self.targets)
