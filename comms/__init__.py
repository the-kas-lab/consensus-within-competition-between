"""Communication substrate (M4 / 10 / 11).

**Not yet wired into ``Simulation``.** That -- not any property of the defaults
-- is why existing results are unaffected today.

A default ``Network`` is a lossless complete graph. That is a *degenerate
configuration*, **not** "communication off": once this layer is connected to the
episode loop it still imposes round structure, delivery timing and merge
semantics on every belief that crosses it. So integration needs regression
checks even under perfect links, and "the graph was complete" is not a reason to
skip them.

Two invariants carry the layer, and each has a test:

* **Double buffering** -- deliveries are visible only after the next swap, so no
  message traverses two hops in one tick and information speed is set by the
  protocol rather than by agent iteration order (``network.Network.deliver``).
* **Counter-based link randomness** keyed ``(seed, sender, receiver,
  comm_tick)`` -- so unrelated code changes cannot reshuffle packet outcomes and
  paired conditions stay paired (``link_models``).
"""

from comms.link_models import (
    DiskLinkModel,
    GilbertElliottLossModel,
    IidLossModel,
    LogisticLinkModel,
    NoLoss,
)
from comms.messages import (
    ALLOCATION_MESSAGE_TYPES,
    ASSIGNMENT_COMMAND,
    AUCTION_MESSAGE,
    BELIEF_MESSAGE_TYPES,
    BROADCAST,
    COMMAND_MESSAGE_TYPES,
    COVERAGE_UPDATE,
    EPOCH_REQUIRED_MESSAGE_TYPES,
    INTENT_UPDATE,
    REALLOCATION_PROPOSAL,
    SELF_STATE,
    STATUS_TOMBSTONE,
    TARGET_UPDATE,
    AssignmentCommand,
    AuctionMessage,
    BeliefUpdate,
    IntentUpdate,
    MessageEnvelope,
    ReallocationProposal,
    StatusTombstone,
    make_envelope,
)
from comms.network import Network
from comms.receive import (
    COMMAND_ACCEPTED,
    COMMAND_NOT_A_COMMAND,
    COMMAND_SUPERSEDED,
    COMMAND_TOO_OLD,
    accept,
    accept_command,
    classify_command,
    is_stale_allocation,
    partition_deliveries,
)

__all__ = [
    "Network",
    "accept",
    "accept_command",
    "classify_command",
    "partition_deliveries",
    "ALLOCATION_MESSAGE_TYPES",
    "BELIEF_MESSAGE_TYPES",
    "COMMAND_MESSAGE_TYPES",
    "EPOCH_REQUIRED_MESSAGE_TYPES",
    "COMMAND_ACCEPTED",
    "COMMAND_SUPERSEDED",
    "COMMAND_TOO_OLD",
    "COMMAND_NOT_A_COMMAND",
    "MessageEnvelope",
    "AssignmentCommand",
    "BeliefUpdate",
    "IntentUpdate",
    "AuctionMessage",
    "ReallocationProposal",
    "StatusTombstone",
    "make_envelope",
    "BROADCAST",
    "TARGET_UPDATE",
    "COVERAGE_UPDATE",
    "SELF_STATE",
    "STATUS_TOMBSTONE",
    "INTENT_UPDATE",
    "AUCTION_MESSAGE",
    "REALLOCATION_PROPOSAL",
    "ASSIGNMENT_COMMAND",
    "DiskLinkModel",
    "LogisticLinkModel",
    "NoLoss",
    "IidLossModel",
    "GilbertElliottLossModel",
]
