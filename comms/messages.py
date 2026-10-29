"""What travels over the network: typed state, never natural-language reasoning.

 The envelope is transport metadata; the payload is one of the
message classes below. An LLM planner's ``explanation`` field is logged locally
and **never** transmitted -- if reasoning text could cross the wire, the
semantic-attack study at M10 would be attacking prose rather than protocol
state, which is a different paper.

Three properties are load-bearing and each is asserted by a test:

* **Relaying never rejuvenates an observation.** ``observed_at`` is sensor time
  and survives any number of hops; only ``received_at`` and ``relay_agent``
  change. This is why ``perception.belief.Provenance`` owns that logic and this
  module defers to it rather than restamping anything.
* **A late message from epoch k must never overwrite epoch k+1.** Every
  envelope carries ``planning_epoch`` and the receiver drops stale-epoch
  allocation traffic.
* **Delivery is at-most-once but duplicates are harmless.** The merge rules in
  ``perception.belief.merge`` are lattice joins, so a duplicate or reordered
  delivery cannot regress a belief. ``message_id`` exists for logging and
  idempotence checks, not to make correctness depend on deduplication.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# Message types. Belief traffic is split by data type because the
# merge rules differ per type -- is explicit that one generic
# "freshest wins" rule is wrong.
TARGET_UPDATE = "target_update"
COVERAGE_UPDATE = "coverage_update"
SELF_STATE = "self_state"
STATUS_TOMBSTONE = "status_tombstone"
INTENT_UPDATE = "intent_update"
AUCTION_MESSAGE = "auction_message"
REALLOCATION_PROPOSAL = "reallocation_proposal"
ASSIGNMENT_COMMAND = "assignment_command"

MESSAGE_TYPES = (
    TARGET_UPDATE,
    COVERAGE_UPDATE,
    SELF_STATE,
    STATUS_TOMBSTONE,
    INTENT_UPDATE,
    AUCTION_MESSAGE,
    REALLOCATION_PROPOSAL,
    ASSIGNMENT_COMMAND,
)

# Allocation traffic is epoch-scoped: a late message from epoch k must never
# overwrite epoch k+1. Belief traffic is NOT in this
# set and must never be epoch-filtered -- an observation sent during an earlier
# epoch may still be the freshest sighting the receiver has, and its staleness
# is decided by ``observed_at`` in perception.belief.merge. See comms.receive.
ALLOCATION_MESSAGE_TYPES = (INTENT_UPDATE, AUCTION_MESSAGE, REALLOCATION_PROPOSAL)
BELIEF_MESSAGE_TYPES = (TARGET_UPDATE, COVERAGE_UPDATE, SELF_STATE, STATUS_TOMBSTONE)

# A commander's assignment to one subordinate. Deliberately
# NOT in ALLOCATION_MESSAGE_TYPES, which is the set the wall-clock epoch filter
# applies to. The two are ordered against different things and conflating them
# breaks condition B outright:
#
#   * an auction bid from a superseded epoch is **meaningless** -- the round it
#     belonged to is over, so it is ordered against the receiver's CURRENT
#     epoch;
#   * a command from a superseded epoch is **still a command**. If it were
#     ordered against the current epoch then every delayed command would be
#     rejected on arrival, because the clock has moved on in transit, and
#     condition B would measure "commands never arrive" rather than "commands
#     arrive late".
#
# Commands are therefore ordered against the LATEST COMMAND THE RECEIVER HAS
# ALREADY ACCEPTED -- see comms.receive.accept_command, and
# settled this before
# any delivery code was written.
COMMAND_MESSAGE_TYPES = (ASSIGNMENT_COMMAND,)

# Types that must declare a planning_epoch at construction. Commands join
# allocation traffic here for the same reason auctions are in it -- an untagged
# command cannot be ordered against anything -- while staying out of the
# epoch-FILTER set above. Keeping the two sets apart is the whole point.
EPOCH_REQUIRED_MESSAGE_TYPES = ALLOCATION_MESSAGE_TYPES + COMMAND_MESSAGE_TYPES

# A broadcast has no single addressee. Kept explicit rather than using None so a
# missing recipient is a bug rather than an accidental broadcast.
BROADCAST = "__broadcast__"


@dataclass(frozen=True)
class MessageEnvelope:
    """Transport metadata wrapping one payload.

    ``ttl`` bounds store-and-forward gossip. It decrements per hop and a message
    is dropped at zero, which keeps relay traffic finite on a ring or a dense
    disk graph without needing global knowledge of the topology.

    ``hop_count`` is diagnostic: wants delivered-hop distributions,
    and it is the cheapest way to verify "one hop per round" in a test.
    """

    message_id: str
    message_type: str
    sender: int
    recipient: Any            # agent id, or BROADCAST
    sent_at: int              # simulation step the message was handed to the network
    planning_epoch: Optional[int]
    payload: Any
    hop_count: int = 0
    ttl: int = 4
    origin_sender: Optional[int] = None

    def __post_init__(self):
        if self.message_type not in MESSAGE_TYPES:
            raise ValueError(
                f"Unknown message_type {self.message_type!r}. Known: "
                f"{', '.join(MESSAGE_TYPES)}."
            )
        if self.ttl < 0:
            raise ValueError(f"ttl must be >= 0, got {self.ttl}")
        if self.message_type in EPOCH_REQUIRED_MESSAGE_TYPES and self.planning_epoch is None:
            # Refused at construction rather than filtered at the receiver: an
            # untagged auction message cannot be checked for staleness, so the rule that an auction may span planning epochs's
            # drop rule would silently not apply to it. Better to fail here than
            # to let epoch k's bid quietly contaminate epoch k+1. The same
            # argument makes an untagged command unorderable against the
            # receiver's accepted-command history.
            raise ValueError(
                f"{self.message_type} requires a planning_epoch -- allocation "
                "traffic and commander assignments are epoch-scoped "
                "(the rule that an auction may span planning epochs, the command-delivery conditions contract Sec 3a)."
            )
        if self.origin_sender is None:
            object.__setattr__(self, "origin_sender", self.sender)

    @property
    def is_broadcast(self):
        return self.recipient == BROADCAST

    def forwarded_by(self, relay_agent, recipient, sent_at):
        """This message one hop further on.

        ``origin_sender`` and the payload survive; ``sender`` becomes the relay.
        Nothing here touches the payload's own timestamps -- rejuvenating an
        observation on relay is the single most damaging thing this layer could
        do, and it is prevented by never writing to the payload at all.
        """
        if self.ttl <= 0:
            raise ValueError("cannot forward a message with ttl exhausted")
        return MessageEnvelope(
            message_id=f"{self.message_id}/r{self.hop_count + 1}:{relay_agent}",
            message_type=self.message_type,
            sender=int(relay_agent),
            recipient=recipient,
            sent_at=int(sent_at),
            planning_epoch=self.planning_epoch,
            payload=self.payload,
            hop_count=self.hop_count + 1,
            ttl=self.ttl - 1,
            origin_sender=self.origin_sender,
        )


@dataclass(frozen=True)
class BeliefUpdate:
    """A compact belief snapshot.

    Version 1 serializes whole records because the state is tiny -- a handful of
    contacts per agent. The class is named ``BeliefUpdate`` rather than
    ``BeliefSnapshot`` deliberately, so it can later carry deltas or
    event-triggered updates without changing transport semantics or any test
    that asserts on message types.

    ``targets`` maps contact_id -> TargetBelief. ``coverage`` is a
    ``CoverageBelief`` or None. Both are merged by the lattice joins in
    ``perception.belief.merge``, so this payload is safe to deliver twice.
    """

    targets: Dict[int, Any] = field(default_factory=dict)
    coverage: Optional[Any] = None


@dataclass(frozen=True)
class IntentUpdate:
    """What an agent means to do, so teammates can avoid duplicating it.

    Distinct from a belief: a belief is a claim about the world and merges by
    observation time; an intent is a claim about the sender itself and merges by
    ``owner_sequence``, because only the owner may revise its own intent. That
    distinction is why this is a separate message class rather than a field on
    BeliefUpdate.
    """

    planning_epoch: int
    owner_agent: int
    proposed_task: Optional[Any] = None
    committed_task: Optional[Any] = None
    commitment_until: Optional[int] = None
    owner_sequence: int = 0


@dataclass(frozen=True)
class AuctionMessage:
    """CBAA protocol state. ``task_generation`` lets a receiver reject bids for a task set that
    has since been rebuilt, which is a different staleness axis from
    ``planning_epoch`` and needs its own field."""

    planning_epoch: int
    task_generation: int
    winning_agents: Tuple = ()
    winning_bids: Tuple = ()
    protocol_state: Optional[Any] = None


@dataclass(frozen=True)
class ReallocationProposal:
    """A bounded-reallocation proposal."""

    planning_epoch: int
    agent_id: int
    current_task: Optional[Any] = None
    proposed_task: Optional[Any] = None
    delta_utility: float = 0.0


@dataclass(frozen=True)
class AssignmentCommand:
    """One agent's role and target, as ordered by the commander.

    Addressed to a single ``agent_id`` rather than broadcast as a team plan,
    because the whole point of condition B is that different subordinates may be
    holding assignments from different epochs: a team-wide payload would arrive
    or not arrive as a unit and could not represent that.

    ``issued_at`` is the simulation step of the planning epoch that produced
    this command; ``planning_epoch`` is the epoch index it is ordered by. Both
    are carried so the generation -> receipt -> application chain is
    measurable rather than declared -- the command-delivery conditions contract Sec 3 limitation (B - A
    mixes inbound information latency with outbound assignment latency) can only
    be *observed* if the outbound half is timestamped.

    The fields mirror ``planning.planner.PlannerOutput``, one agent's slice of
    it, so applying a command is assembling the same object the local path
    builds.
    """

    planning_epoch: int
    commander_id: int
    agent_id: int
    issued_at: int
    role: Optional[Any] = None
    target_red_id: Optional[int] = None
    target_contact_id: Optional[int] = None
    recon_target_region: Optional[int] = None
    # The task model's identity, so a commanded DEFEND station and a
    # NEUTRALIZE slot survive delivery (None for planners without one):
    # ("station", k) | ("contact", c), and e.g. "NEUTRALIZE:3:1".
    defend_target: Optional[tuple] = None
    task_id: Optional[str] = None


@dataclass(frozen=True)
class StatusTombstone:
    """'Agent i is disabled as of step t.'

    A tombstone, not a deletion: requires it to dominate older live
    entries so a stale SELF_STATE cannot resurrect a dead agent. Merging is by
    ``(disabled, disabled_at)`` with disabled winning ties, which keeps the
    operation a lattice join like the others.
    """

    agent_id: int
    disabled_at: int


def make_envelope(
    message_type,
    sender,
    payload,
    sent_at,
    recipient=BROADCAST,
    planning_epoch=None,
    ttl=4,
    sequence=0,
):
    """Build an envelope with a deterministic id.

    The id is derived from ``(sender, type, sent_at, sequence)`` rather than
    ``uuid4`` on purpose: episodes must be a pure function of (seed, config), and
    a random id would make logs differ between two runs that are otherwise
    byte-identical. ``sequence`` disambiguates several messages of one type from
    one sender in one step.
    """
    return MessageEnvelope(
        message_id=f"{sender}:{message_type}:{sent_at}:{sequence}",
        message_type=message_type,
        sender=int(sender),
        recipient=recipient,
        sent_at=int(sent_at),
        planning_epoch=planning_epoch,
        payload=payload,
        ttl=int(ttl),
    )
