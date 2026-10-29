"""Errors that stop a run instead of becoming a planner fallback.

A planner turns an ordinary call failure into a recorded fallback so that a
400-step episode survives one bad response. Some failures must never be
absorbed that way, because the episode that continued would not be the
episode that was asked for:

* ``ReplayMiss`` (planning.local_llm) -- a replay asked a question its cache
  cannot answer; continuing would run a different episode;
* ``BudgetExhausted`` (planning.spend) -- the next paid attempt would exceed
  the spend cap; continuing on fallbacks would produce a rule run wearing an
  LLM's name.

Every site that catches planner exceptions re-raises ``StopRun`` first.
"""


class StopRun(RuntimeError):
    """Base class: propagate, never fall back."""
