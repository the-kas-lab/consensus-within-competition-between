"""LocalBeliefStore: one blue agent's mutable belief state.

One store per agent. It accumulates first-hand observations, accepts merges
from other sources, and emits immutable ``LocalWorldView`` snapshots for
consumers.

What this store deliberately does NOT do:

* **It never deletes on a timeout.** Consumers apply their own validity age.
* **It never contracts on negative information.** declares for
  Version 1 that not seeing a contact does not shrink its reachable set. A red
  that walks out of an agent's cone stays believed-in, at its last position,
  ageing. This is an idealization and is declared as one.
* **It holds no probability.** Last-known state plus age, exactly as
  04-extension-points.md scoped it. No filter, no occupancy grid.

The one deletion it does support is an explicit RESOLUTION notice
(``mark_resolved``), used only under the simulator's opt-in
``resolution_info="public"``: the contact is dropped and no later
observation or merge (a relay of an older record, say) can re-add it.
"""

import numpy as np

from perception.belief.merge import merge_target, merge_target_dicts
from perception.belief.records import CoverageBelief, Provenance, TargetBelief
from perception.belief.view import LocalWorldView


class LocalBeliefStore:
    """Beliefs held by one agent.

    ``n_regions`` sizes the coverage belief; pass 0 to disable coverage
    tracking for an agent that has no use for it.
    """

    def __init__(self, agent_id, n_regions=0):
        self.agent_id = int(agent_id)
        self.targets = {}
        self.coverage = CoverageBelief(int(n_regions)) if n_regions else None
        self._sequence = 0
        self.resolved = set()

    # -- first-hand observation -----------------------------------------

    def observe_target(self, contact_id, state, step, planning_epoch=None):
        """Record something this agent's own sensor saw this step."""
        if int(contact_id) in self.resolved:
            return None
        self._sequence += 1
        belief = TargetBelief(
            contact_id=int(contact_id),
            state=state,
            provenance=Provenance(
                origin_agent=self.agent_id,
                observed_at=int(step),
                received_at=int(step),
                relay_agent=None,
                source_sequence=self._sequence,
                planning_epoch=planning_epoch,
            ),
        )
        self.targets[int(contact_id)] = merge_target(
            self.targets.get(int(contact_id)), belief
        )
        return self.targets[int(contact_id)]

    def observe_coverage(self, region_id, step):
        if self.coverage is not None:
            self.coverage.observe(region_id, step)

    # -- second-hand information ----------------------------------------

    def merge_targets(self, incoming):
        """Merge a mapping of ``{contact_id: TargetBelief}``, per contact.

        At M2 this is how the shared-belief arm is realized: every agent merges
        every other agent's first-hand observations, with no latency or loss.
        At M4 the same entry point takes what actually arrived over the network,
        so the belief layer does not change shape when communication does.
        """
        if self.resolved:
            incoming = {c: b for c, b in incoming.items() if int(c) not in self.resolved}
        self.targets = merge_target_dicts(self.targets, incoming)

    def mark_resolved(self, contact_id):
        """A public resolution notice: forget the contact, permanently."""
        self.resolved.add(int(contact_id))
        self.targets.pop(int(contact_id), None)

    def merge_coverage(self, other_coverage):
        if self.coverage is not None and other_coverage is not None:
            self.coverage.merge(other_coverage)

    # -- read -----------------------------------------------------------

    def snapshot(self, now, dt, self_state, coverage_centers=None, teammates=None):
        """An immutable view frozen at ``now``.

        Target beliefs are copied, so a consumer holding a snapshot cannot be
        perturbed by a later observation -- the frozen-snapshot property plan
        Sec 7 requires of per-agent planning.

        ``teammates`` is ``{agent_id: state}`` for the other *active* blues.
        The store does not hold it -- it is not a belief, it is the declared
        common-knowledge channel, so the caller
        supplies it and it is copied here for the same freezing reason.
        """
        return LocalWorldView(
            now=int(now),
            dt=float(dt),
            agent_id=self.agent_id,
            self_state=np.asarray(self_state, dtype=float).ravel().copy(),
            targets={
                cid: TargetBelief(b.contact_id, b.state, b.provenance)
                for cid, b in self.targets.items()
            },
            coverage=None if self.coverage is None else self.coverage.copy(),
            coverage_centers=coverage_centers,
            teammates={
                int(i): np.asarray(s, dtype=float).ravel().copy()
                for i, s in (teammates or {}).items()
                if int(i) != self.agent_id
            },
        )

    def __repr__(self):
        return (
            f"LocalBeliefStore(agent={self.agent_id}, "
            f"targets={sorted(self.targets)}, seq={self._sequence})"
        )
