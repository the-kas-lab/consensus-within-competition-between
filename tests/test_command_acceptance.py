"""Commander assignments are ordered against the RECEIVER, not the wall clock.

The command-delivery conditions contract Sec 3a settled this before any delivery code was written,
because getting it wrong produces a plausible-looking null rather than a crash.

`comms/receive.py`'s epoch filter rejects allocation traffic whose
`planning_epoch` is older than the epoch the CALLER supplies. If condition B
supplied the wall-clock planning epoch for commands, then every delayed command
would be rejected -- *because* it was delayed, the clock having moved on while
it was in transit -- even when it is the newest order that agent has ever
received. B would then measure "commands never arrive" and report it as
"communication does not help", which is a wrong answer that looks like a
finding.

So a command is accepted iff its epoch is newer than the latest command this
receiver has already accepted. Any age limit is a separate, separately-declared
parameter that defaults to off, so "late" stays distinguishable from "dropped".

These tests cover the rule itself. The delivery path that uses it is tested
against a live Simulation in tests/test_commander_conditions.py.
"""

import pytest

from comms import receive
from comms.messages import (
    ALLOCATION_MESSAGE_TYPES,
    ASSIGNMENT_COMMAND,
    AUCTION_MESSAGE,
    COMMAND_MESSAGE_TYPES,
    TARGET_UPDATE,
    AssignmentCommand,
    AuctionMessage,
    BeliefUpdate,
    make_envelope,
)


def _command(epoch, agent=1, issued_at=None, sent_at=None, commander=0):
    issued_at = epoch * 10 if issued_at is None else issued_at
    return make_envelope(
        ASSIGNMENT_COMMAND,
        commander,
        AssignmentCommand(
            planning_epoch=epoch,
            commander_id=commander,
            agent_id=agent,
            issued_at=issued_at,
            role="neutralize",
            target_red_id=0,
        ),
        sent_at=issued_at if sent_at is None else sent_at,
        recipient=agent,
        planning_epoch=epoch,
    )


# ---------------------------------------------------------------------------
# The defect this rule exists to prevent
# ---------------------------------------------------------------------------


def test_a_delayed_command_is_accepted_however_far_the_clock_has_moved():
    """The whole point. A command issued at epoch 3 and delivered at epoch 40
    is still the newest order this agent has, and is applied."""
    late = _command(epoch=3)
    assert receive.accept_command(late, latest_accepted_epoch=None) is True
    assert receive.accept_command(late, latest_accepted_epoch=2) is True


def test_the_wall_clock_epoch_filter_would_have_rejected_that_command():
    """The counterfactual, asserted rather than described, so the reason this
    path exists cannot quietly stop being true.

    `accept` is the allocation rule. Were commands routed through it with the
    current planning epoch, the command above would be dropped. It is not, and
    the two assertions below are what say the separation is real.
    """
    late = _command(epoch=3)
    # If ASSIGNMENT_COMMAND were in ALLOCATION_MESSAGE_TYPES this would be
    # False, which is the failure mode the contract Sec 3a names.
    assert receive.accept(late, current_planning_epoch=40) is True
    assert ASSIGNMENT_COMMAND not in ALLOCATION_MESSAGE_TYPES

    equivalent_auction = make_envelope(
        AUCTION_MESSAGE, 0, AuctionMessage(3, 0), sent_at=30, planning_epoch=3
    )
    assert receive.accept(equivalent_auction, current_planning_epoch=40) is False


@pytest.mark.parametrize("message_type", ALLOCATION_MESSAGE_TYPES)
def test_the_allocation_epoch_rule_is_unchanged(message_type):
    """The rule that an auction may span planning epochs's rule is correct for auction and reallocation traffic -- a bid from a
    superseded round is meaningless -- and commands must not have altered it."""
    assert message_type not in COMMAND_MESSAGE_TYPES


# ---------------------------------------------------------------------------
# Ordering against the receiver's own accepted history
# ---------------------------------------------------------------------------


def test_a_reordered_older_command_is_rejected_and_the_newer_one_survives():
    """Reordering is the case the ordering rule is *for*. An agent holding
    epoch 7 must not be dragged back to epoch 5 by a straggler."""
    assert receive.classify_command(_command(epoch=5), latest_accepted_epoch=7) == (
        receive.COMMAND_SUPERSEDED
    )
    assert receive.classify_command(_command(epoch=8), latest_accepted_epoch=7) == (
        receive.COMMAND_ACCEPTED
    )


def test_a_duplicate_of_the_command_in_force_is_superseded():
    """Strictly newer, not newer-or-equal. Re-applying a duplicate would be
    harmless today; rejecting it keeps the accepted-command ledger a record of
    changes rather than of arrivals, which is what the delivery checks count."""
    assert receive.classify_command(_command(epoch=7), latest_accepted_epoch=7) == (
        receive.COMMAND_SUPERSEDED
    )


def test_an_agent_that_has_accepted_nothing_accepts_its_first_command():
    assert receive.classify_command(_command(epoch=0), latest_accepted_epoch=None) == (
        receive.COMMAND_ACCEPTED
    )


def test_ordering_is_per_receiver_not_global():
    """Two agents, one partitioned. The connected one is at epoch 9; the
    isolated one has accepted nothing, so epoch 4 is news to it and not to the
    other. A global clock cannot express that, which is why the reference is
    the receiver's own history."""
    stale_for_one = _command(epoch=4, agent=1)
    assert receive.accept_command(stale_for_one, latest_accepted_epoch=9) is False
    assert receive.accept_command(stale_for_one, latest_accepted_epoch=None) is True


# ---------------------------------------------------------------------------
# The age limit: separate, separately declared, off by default
# ---------------------------------------------------------------------------


def test_no_age_limit_by_default_so_late_stays_distinguishable_from_dropped():
    ancient = _command(epoch=1, issued_at=10)
    assert receive.accept_command(ancient, latest_accepted_epoch=0, now=100_000) is True


def test_the_age_limit_binds_only_when_declared():
    ancient = _command(epoch=1, issued_at=10)
    assert receive.classify_command(
        ancient, latest_accepted_epoch=0, now=40, max_age_steps=20
    ) == receive.COMMAND_TOO_OLD
    assert receive.classify_command(
        ancient, latest_accepted_epoch=0, now=30, max_age_steps=20
    ) == receive.COMMAND_ACCEPTED


def test_age_is_measured_from_issue_not_from_the_last_hop():
    """A relay must not rejuvenate an order -- the same error the belief layer
    refuses at `Provenance`. The envelope is re-stamped on forward; the
    payload's `issued_at` is not.
    """
    relayed = _command(epoch=1, issued_at=10, sent_at=95)
    assert receive.classify_command(
        relayed, latest_accepted_epoch=0, now=100, max_age_steps=20
    ) == receive.COMMAND_TOO_OLD


def test_an_age_limit_without_a_clock_is_refused_rather_than_ignored():
    with pytest.raises(ValueError):
        receive.classify_command(_command(epoch=1), max_age_steps=5)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_a_command_must_declare_an_epoch():
    """Refused at construction, for the same reason an untagged auction message
    is: an unordered command cannot be checked against anything, so the
    ordering rule would silently not apply to it."""
    with pytest.raises(ValueError, match="planning_epoch"):
        make_envelope(
            ASSIGNMENT_COMMAND,
            0,
            AssignmentCommand(planning_epoch=1, commander_id=0, agent_id=1, issued_at=0),
            sent_at=0,
            recipient=1,
        )


def test_classify_refuses_to_judge_traffic_that_is_not_a_command():
    """A belief update accidentally routed to the command path must be visible
    as a routing error, not silently accepted as an order."""
    belief = make_envelope(TARGET_UPDATE, 0, BeliefUpdate(), sent_at=1)
    assert receive.classify_command(belief) == receive.COMMAND_NOT_A_COMMAND
    assert receive.accept_command(belief) is False
