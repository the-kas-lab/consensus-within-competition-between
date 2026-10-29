"""Synchronous communication rounds with double-buffered inboxes.

 Two non-negotiables, both of which exist to stop one message
traversing two hops in a single tick:

**1. Double buffering is mandatory.** Every sender reads a start-of-round
snapshot. Deliveries made during round *t* land in a *pending* buffer and only
become visible at round *t+1*. Without this, the iteration order over agents
would silently determine how far information travels: agent 0 sends to 1, agent
1 is processed next, already has the message, and forwards it onward in the same
tick. Information would then cross the graph at a speed set by a ``for`` loop
rather than by the protocol, and every latency result in the paper would be an
artifact of dict ordering.

**2. Link randomness is counter-based**, keyed ``(seed, sender, receiver,
comm_tick)``. See ``comms.link_models``. The network never draws from a
sequential stream, so adding a message type or reordering agents cannot reshuffle
unrelated packet outcomes, and paired conditions stay paired.

## What this layer does and does not do

It moves envelopes between inboxes. It does **not** merge beliefs -- merging is
``perception.belief.merge``'s job and happens at step 4 of's within-step
order, after this layer has delivered at step 3. Keeping them apart is what lets
the merge rules stay pure lattice joins that are testable without a network, and
lets the network be testable without a belief.

Relaying is opt-in (``store_and_forward=True``). With it off, this is a
one-hop broadcast medium, which is all the communication conditions's first sweeps need.
"""

from collections import defaultdict

from comms import topologies
from comms.link_models import DiskLinkModel, NoLoss
from comms.messages import BROADCAST


class Network:
    """A synchronous, round-based message fabric over a dynamic graph.

    Usage per communication tick::

        net.begin_round(positions, disabled)   # freeze the graph
        net.send(envelope)                     # senders enqueue
        delivered = net.deliver()              # swap buffers, return inboxes

    ``deliver`` returns ``{agent_id: [envelope, ...]}`` for *this* round and
    leaves anything generated during the round waiting for the next one.
    """

    def __init__(
        self,
        seed,
        link_model=None,
        loss_model=None,
        store_and_forward=False,
        fixed_edges=None,
    ):
        self.seed = int(seed)
        self.link_model = link_model or DiskLinkModel()
        self.loss_model = loss_model or NoLoss()
        self.store_and_forward = bool(store_and_forward)
        # When set, the graph is this fixed edge set rather than a disk over
        # live positions -- the controlled regimes of
        self.fixed_edges = fixed_edges

        self.comm_tick = -1
        self.edges = set()
        self.disabled = set()
        self._agents = ()

        # The two buffers. `_outbox` is what senders write during the current
        # round; `_inbox` is what receivers read. They are never the same object.
        self._outbox = defaultdict(list)
        self._inbox = defaultdict(list)

        # Diagnostics.
        self.stats = {
            "sent": 0,
            "delivered": 0,
            "dropped_no_link": 0,
            "dropped_loss": 0,
            "dropped_ttl": 0,
            "dropped_disabled": 0,
            "relayed": 0,
        }
        self._delivered_ids = defaultdict(set)

    # -- round lifecycle ---------------------------------------------------

    def begin_round(self, positions=None, disabled=(), comm_tick=None):
        """Freeze the graph for this round.

        The graph is computed **once** here and used for every send in the
        round. Recomputing per-send would let an agent that moved mid-round
        reach a different neighbour set than the one its peers saw, which breaks
        the "synchronous round" abstraction.
        """
        self.comm_tick = self.comm_tick + 1 if comm_tick is None else int(comm_tick)
        self.disabled = set(int(a) for a in disabled)
        if self.fixed_edges is not None:
            self.edges = set(self.fixed_edges)
            if positions:
                self._agents = tuple(sorted(positions))
        elif positions:
            self._agents = tuple(sorted(positions))
            self.edges = self._disk_edges(positions)
        else:
            self.edges = set()
        return self.edges

    def _disk_edges(self, positions):
        """Edges from live positions, honouring the link model.

        A stochastic link model gets its draw keyed on the canonical pair and
        this round's tick, so the same link makes the same decision no matter
        which endpoint is considered first.
        """
        import numpy as np

        ids = sorted(positions)
        edges = set()
        for index, i in enumerate(ids):
            for j in ids[index + 1:]:
                if i in self.disabled or j in self.disabled:
                    continue
                d = float(
                    np.linalg.norm(
                        np.asarray(positions[i], float) - np.asarray(positions[j], float)
                    )
                )
                if self.link_model.connected(
                    d, seed=self.seed, sender=i, receiver=j, comm_tick=self.comm_tick
                ):
                    edges.add(frozenset((int(i), int(j))))
        return edges

    def neighbours(self, agent):
        if agent in self.disabled:
            return set()
        return {n for n in topologies.neighbours(self.edges, agent) if n not in self.disabled}

    # -- sending -----------------------------------------------------------

    def send(self, envelope):
        """Enqueue one envelope for delivery at the *next* buffer swap.

        A disabled sender is dropped here rather than filtered by the caller, so
        "a disabled relay stops forwarding" holds however the caller is written.
        """
        self.stats["sent"] += 1
        if envelope.sender in self.disabled:
            self.stats["dropped_disabled"] += 1
            return 0

        recipients = (
            self.neighbours(envelope.sender)
            if envelope.is_broadcast
            else ({int(envelope.recipient)} & self.neighbours(envelope.sender))
        )
        if not envelope.is_broadcast and not recipients:
            self.stats["dropped_no_link"] += 1
            return 0

        accepted = 0
        for receiver in sorted(recipients):
            if not self.loss_model.delivers(
                self.seed, envelope.sender, receiver, self.comm_tick
            ):
                self.stats["dropped_loss"] += 1
                continue
            # Written to the OUTBOX: not visible to anyone until deliver().
            self._outbox[receiver].append(envelope)
            accepted += 1
        return accepted

    def broadcast(self, envelope):
        return self.send(envelope)

    # -- delivery ----------------------------------------------------------

    def deliver(self):
        """Swap buffers and hand each agent what arrived.

        This is the double-buffer swap and it is the whole reason this class
        exists. Everything enqueued since the last swap becomes visible now;
        anything enqueued *after* this call waits for the next round.
        """
        self._inbox = self._outbox
        self._outbox = defaultdict(list)

        delivered = {}
        for agent, envelopes in self._inbox.items():
            if agent in self.disabled:
                self.stats["dropped_disabled"] += len(envelopes)
                continue
            kept = []
            for envelope in envelopes:
                # Idempotence is a property of the merge rules, not of this
                # layer -- duplicates are counted and passed through rather
                # than silently swallowed, so a test can observe them.
                self._delivered_ids[agent].add(envelope.message_id)
                kept.append(envelope)
            if kept:
                delivered[agent] = kept
                self.stats["delivered"] += len(kept)
        return delivered

    def forward(self, agent, envelopes, sent_at):
        """Store-and-forward gossip: re-broadcast what ``agent`` just received.

        Called by the caller *after* ``deliver``, so a forwarded message lands in
        the outbox and cannot reach anyone until the next swap. That is what
        makes one hop per round true rather than aspirational.

        A message is not sent back to the peer it came from, and TTL-exhausted
        messages are dropped and counted.
        """
        if not self.store_and_forward or agent in self.disabled:
            return 0
        count = 0
        for envelope in envelopes:
            if envelope.ttl <= 0:
                self.stats["dropped_ttl"] += 1
                continue
            hop = envelope.forwarded_by(agent, BROADCAST, sent_at)
            for receiver in sorted(self.neighbours(agent)):
                if receiver == envelope.sender:
                    continue
                if not self.loss_model.delivers(self.seed, agent, receiver, self.comm_tick):
                    self.stats["dropped_loss"] += 1
                    continue
                self._outbox[receiver].append(hop)
                count += 1
        self.stats["relayed"] += count
        return count

    # -- diagnostics -------------------------------------------------------

    def connectivity_metrics(self, agents=None):
        """The mediators asks to be reported alongside R_comm.

        Reporting these rather than only the configured range is the difference
        between "performance degrades past R_comm = 15" and a claim about *why*.
        """
        agents = sorted(self._agents if agents is None else agents)
        live = [a for a in agents if a not in self.disabled]
        if not live:
            return {
                "n_agents": 0,
                "connected": False,
                "n_components": 0,
                "largest_component_fraction": 0.0,
                "diameter": None,
                "n_edges": 0,
            }
        comps = topologies.components(self.edges, live)
        largest = max((len(c) for c in comps), default=0)
        return {
            "n_agents": len(live),
            "connected": len(comps) == 1,
            "n_components": len(comps),
            "largest_component_fraction": largest / len(live),
            "diameter": topologies.diameter(self.edges, live),
            "n_edges": len(self.edges),
        }

    def describe(self):
        return {
            "seed": self.seed,
            "link_model": self.link_model.describe(),
            "loss_model": self.loss_model.describe(),
            "store_and_forward": self.store_and_forward,
            "fixed_topology": self.fixed_edges is not None,
        }
