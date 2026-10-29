"""Merge rules, one per data type.

 is explicit that a single generic rule ("freshest message wins") is
wrong, because different fields have different correctness conditions. The two
rules implemented here are the two M2 needs; the rest arrive with the
communication layer at M4.

| Data type      | Rule                            | Property |
|----------------|---------------------------------|----------|
| Target track   | per-contact max ``observed_at`` | idempotent; order-independent across DISTINCT ``observed_at`` only |
| Coverage cell  | per-cell max ``swept_at``       | monotone, order-independent |

Duplicate delivery is harmless and reordering cannot regress a track to an
OLDER observation -- the properties's communication tests assert.

**Equal timestamps are order-dependent.** Two different records about one
contact carrying the same ``observed_at`` tie, and a tie keeps whichever
arrived first. With honest reports this is invisible: every sighting made at
one step is the same true state. With conflicting content (a falsified
report, ``Simulation(falsified_reports=...)``) the surviving position depends
on arrival order -- see tests/test_falsified_reports.py, truthful-first vs
false-first. The rule is kept as is; the order is logged.

Note the comparison is on ``observed_at`` (sensor time), never ``received_at``.
A freshly-relayed old sighting must lose to an older-arriving new one.
"""


def merge_target(existing, incoming):
    """Returns whichever belief rests on the more recent *observation*.

    Ties keep the existing record. That makes the merge idempotent under
    duplicate delivery. It does NOT make it order-independent when two
    different records share an ``observed_at``: the first to arrive is kept.
    """
    if existing is None:
        return incoming
    if incoming is None:
        return existing
    if incoming.provenance.observed_at > existing.provenance.observed_at:
        return incoming
    return existing


def merge_target_dicts(existing, incoming):
    """Merge per contact, not per message.

 calls this out specifically: a message carrying a stale track for
    contact A and a fresh one for contact B must not be accepted or rejected
    wholesale.
    """
    merged = dict(existing)
    for contact_id, belief in incoming.items():
        merged[contact_id] = merge_target(merged.get(contact_id), belief)
    return merged


def merge_status(existing, incoming):
    """Merge two claims about whether an agent is alive. **Disabled absorbs.**

    Extends the table above with the third data type the communication layer
    introduces:

    | Data type    | Rule                                 | Property |
    |--------------|--------------------------------------|----------|
    | Agent status | disabled absorbs; earliest disable   | idempotent, order-independent |

    Three cases:

    * **Either side disabled** -> disabled. Agents do not revive in this model,
      so a later liveness report is not newer information, it is *stale*
      information that happens to have arrived late. This is the whole point of
      a tombstone: it must dominate an older live entry **and** any live entry
      that arrives afterwards.
    * **Both disabled** -> the *earliest* ``as_of``. An agent died once, at one
      step; the earliest report is the one closest to that event, and ``min`` is
      commutative, associative and idempotent, so duplicates and reordering
      cannot move it.
    * **Both live** -> the later ``as_of``, ties keeping ``existing``, matching
      ``merge_target``'s tie rule so duplicate delivery is a no-op.

    This is a join on the lattice ``(live, t) < (disabled, t')``, ordered by
    ascending ``as_of`` among live records and *descending* ``as_of`` among
    disabled ones. Being a join is what makes duplicate and out-of-order
    delivery harmless without the transport having to deduplicate -- the same
    property ``merge_target`` and ``CoverageBelief.merge`` rely on.
    """
    if existing is None:
        return incoming
    if incoming is None:
        return existing
    if existing.disabled and incoming.disabled:
        return existing if existing.as_of <= incoming.as_of else incoming
    if existing.disabled:
        return existing
    if incoming.disabled:
        return incoming
    return incoming if incoming.as_of > existing.as_of else existing


def merge_status_dicts(existing, incoming):
    """Merge per agent, not per message -- same reasoning as
    ``merge_target_dicts``: one message may carry a stale record for agent A and
    a fresh one for agent B."""
    merged = dict(existing)
    for agent_id, status in incoming.items():
        merged[agent_id] = merge_status(merged.get(agent_id), status)
    return merged
