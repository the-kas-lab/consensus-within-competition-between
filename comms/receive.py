"""Receiver-side acceptance: which delivered messages the receiver may act on.

This is deliberately **not** in ``Network.deliver``. The transport's job is to
move envelopes across links; deciding whether a message is still *relevant* is a
property of the receiver's current state (its planning epoch), not of the wire.
Putting the rule here keeps the network testable without a planner and keeps the
rule testable without a network.

## Epoch rejection applies to allocation traffic ONLY

The rule that an auction may span planning epochs requires that a late allocation message from epoch *k* must never
overwrite state for epoch *k+1*: an old assignment must not replace a newer
assignment. That rule is correct for allocation and **wrong for beliefs**.

A target observation *sent* during an earlier planning epoch may still be the
**newest observation the receiver has**. Epoch is a property of the decision
cycle that emitted the message, not of the sighting inside it -- a contact seen
at step 300 and relayed twice is still a step-300 sighting however many planning
epochs have elapsed in transit. Dropping it because its envelope carries an old
epoch would discard information the receiver has nothing better than, and would
do so *silently*, biasing exactly the age-of-information mediator the study
measures.

Staleness for beliefs is therefore decided where it belongs: in
``perception.belief.merge``, by comparing ``observed_at``. A stale belief loses
to a fresher one on its merits; it is never rejected by the transport.

    | traffic                                          | epoch-filtered? | staleness decided by |
    |--------------------------------------------------|-----------------|----------------------|
    | AUCTION_MESSAGE, REALLOCATION_PROPOSAL, INTENT   | **yes**         | ``planning_epoch``   |
    | TARGET_UPDATE, COVERAGE_UPDATE, SELF_STATE, TOMB | no              | ``observed_at`` / merge rule |
    | ASSIGNMENT_COMMAND                               | no -- see below | the receiver's own accepted-command history |

``INTENT_UPDATE`` is grouped with allocation because an intent is a commitment
attached to a decision cycle: acting on epoch *k*'s stated intent during epoch
*k+1* is the same error as acting on its bid.

## Commander assignments are ordered against the receiver, not the clock

``ASSIGNMENT_COMMAND`` is **not** allocation traffic for the
purposes of the rule above, and ``accept`` passes it through untouched.

The reason is that the two are ordered against different references. A bid from
a superseded auction round is meaningless, so it is measured against the
receiver's *current* epoch. **A command from a superseded epoch is still a
command** -- it may be the newest order that agent has ever been given. Applying
the epoch filter to it would reject every command that took longer than one
planning interval to arrive, *because* it arrived late, and condition B would
then measure "commands never arrive" instead of "commands arrive late". The
failure would be silent and would look exactly like a disconnected team.

So commands get their own rule, in ``accept_command``: newer than the latest
command this receiver has already accepted. Any *age* limit is a separate,
separately-declared parameter that defaults to off, so "late" stays
distinguishable from "dropped". Settled in
 before any delivery code
existed, precisely because getting it wrong produces a plausible-looking null.
"""

from comms.messages import ALLOCATION_MESSAGE_TYPES, COMMAND_MESSAGE_TYPES


def is_stale_allocation(envelope, current_planning_epoch):
    """True iff this is allocation traffic from a superseded planning epoch.

    ``current_planning_epoch is None`` means the receiver is not running epoched
    allocation, so nothing can be stale.
    """
    if envelope.message_type not in ALLOCATION_MESSAGE_TYPES:
        return False
    if current_planning_epoch is None:
        return False
    # Envelopes carrying allocation traffic are required to declare an epoch --
    # MessageEnvelope refuses to construct one otherwise, so this cannot be None.
    return envelope.planning_epoch < int(current_planning_epoch)


def accept(envelope, current_planning_epoch=None):
    """Should the receiver act on this message?

    Belief traffic always passes; the merge rules decide what it is worth.
    Allocation traffic from a superseded epoch is rejected.
    """
    return not is_stale_allocation(envelope, current_planning_epoch)


def partition_deliveries(envelopes, current_planning_epoch=None):
    """Split one inbox into ``(accepted, rejected)``.

    Rejections are **returned, not discarded**, so a caller can count them.
 wants stale-message rates reported, and the rule that an auction may span planning epochs's drop rule is only
    trustworthy if it is observable -- a filter that silently ate messages would
    be indistinguishable from a link that never delivered them.
    """
    accepted, rejected = [], []
    for envelope in envelopes:
        (accepted if accept(envelope, current_planning_epoch) else rejected).append(envelope)
    return accepted, rejected


# Why a command was not accepted. Returned rather than folded into a bool so a
# caller can count the two apart: "superseded" is the protocol working, and
# "too_old" is a configured age limit binding. Reporting them as one number
# would hide which.
COMMAND_ACCEPTED = "accepted"
COMMAND_SUPERSEDED = "superseded"
COMMAND_TOO_OLD = "too_old"
COMMAND_NOT_A_COMMAND = "not_a_command"


def classify_command(envelope, latest_accepted_epoch=None, now=None, max_age_steps=None):
    """Why this command is or is not acted on. One of the COMMAND_* constants.

    ``latest_accepted_epoch`` is the ``planning_epoch`` of the newest command
    **this receiver has already accepted**, or None if it has accepted none.
    That -- not the global planning clock -- is the ordering reference; see the
    module docstring.

    ``max_age_steps`` is the separately-declared age limit and defaults to
    **None, meaning no limit**. It is deliberately not the ordering rule: with
    it off, a command that took twenty rounds to arrive is still applied, so an
    experiment can tell a late command from a lost one. When it is set, ``now``
    must be given, and age is measured from the epoch that ISSUED the command
    (``payload.issued_at``), not from the hop that last carried it -- otherwise
    a relay would rejuvenate an order, which is the same error the belief layer
    refuses at ``Provenance``.
    """
    if envelope.message_type not in COMMAND_MESSAGE_TYPES:
        return COMMAND_NOT_A_COMMAND
    if latest_accepted_epoch is not None and envelope.planning_epoch <= int(
        latest_accepted_epoch
    ):
        # Strictly newer, so a duplicate of the command in force is superseded
        # too. Re-applying it would be harmless today; rejecting it keeps the
        # accepted-command ledger a record of changes rather than of arrivals.
        return COMMAND_SUPERSEDED
    if max_age_steps is not None:
        if now is None:
            raise ValueError("max_age_steps requires now")
        issued_at = getattr(envelope.payload, "issued_at", envelope.sent_at)
        if int(now) - int(issued_at) > int(max_age_steps):
            return COMMAND_TOO_OLD
    return COMMAND_ACCEPTED


def accept_command(envelope, latest_accepted_epoch=None, now=None, max_age_steps=None):
    """``True`` iff the receiver should act on this command. See classify_command."""
    return (
        classify_command(envelope, latest_accepted_epoch, now, max_age_steps)
        == COMMAND_ACCEPTED
    )
