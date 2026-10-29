"""Per-agent belief state: what one agent knows, as distinct from what is true.

The package exists to make a single distinction enforceable in code: an agent
acts on its *beliefs*, while the simulator's physics, safety filter and outcome
resolution act on *ground truth*. Before this layer, the two were the same
object almost everywhere --

Public surface:

    Provenance        where a belief came from and when it was observed
    AgentStatus       whether a teammate is alive; 'disabled' is absorbing
    TargetBelief      last-known state of one contact, plus age and prediction
    CoverageBelief    per-agent "when did I last sweep each cell"
    LocalBeliefStore  one agent's mutable belief state
    LocalWorldView    an immutable snapshot handed to consumers
"""

from perception.belief.records import (
    AgentStatus,
    CoverageBelief,
    Provenance,
    TargetBelief,
)
from perception.belief.store import LocalBeliefStore
from perception.belief.view import LocalWorldView

__all__ = [
    "AgentStatus",
    "CoverageBelief",
    "LocalBeliefStore",
    "LocalWorldView",
    "Provenance",
    "TargetBelief",
]
