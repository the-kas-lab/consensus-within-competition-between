"""Task-model commander: the centralized baseline over CBAA's task model.

The commander assigns each active agent one task from a catalogue built with
the SAME task-generation rules, task types, identifiers and configured
capacities CBAA uses (``coordination.tasks.generate_tasks``), applied to the
commander's OWN belief snapshot -- the store the network delivered to it, never
a union of its subordinates' stores. CBAA agents may hold different candidate
lists because their beliefs differ; so may the commander. That matches action
expressivity while keeping the intended information difference.

Validation (``validate_assignment``), per active agent:

* a missing assignment, an unknown task id, or conflicting entries for one
  agent -> that agent's explicit ``DEFEND(own station)`` fallback;
* every task id has capacity one (NEUTRALIZE slots are distinct ids, so two
  slots of one contact are allowed when configured): on duplicate claims the
  lowest agent id keeps the task and the others fall back;
* an unusable response (``None``: model-call failure, unparseable output)
  -> fallback for every active agent.

Valid assignments are never altered to make room for a fallback: a fallback
station that coincides with another agent's valid assignment is RECORDED as a
conflict, not repaired. Nor does a valid plan imply conflict-free executed
commitments -- network delivery can mix epochs.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from coordination.runtime import output_for
from planning.stops import StopRun
from coordination.tasks import DEFEND, NEUTRALIZE, RECON, THREE_TASK, TaskContext, generate_tasks

MISSING = "missing"
UNKNOWN_TASK = "unknown_task"
CONFLICTING_ENTRIES = "conflicting_entries"
CAPACITY_CONFLICT = "capacity_conflict"
UNUSABLE_RESPONSE = "unusable_response"
NOT_ACTIVE = "not_an_active_agent"


def own_station_task(agent: int) -> str:
    return f"{DEFEND}:{int(agent)}"


def validate_assignment(raw: Optional[Sequence[Tuple[int, str]]], catalogue_ids,
                        active_agents) -> Tuple[Dict[int, str], dict]:
    """``raw``: the decider's entries as ``(agent, task_id)`` pairs, in the order
    given, or ``None`` when the response is unusable. Returns the per-agent
    assignment to put into force and a report keeping the original response,
    failures, retained assignments, fallbacks and fallback conflicts apart."""
    catalogue_ids = set(catalogue_ids)
    active = sorted(int(a) for a in active_agents)
    report = {
        "response": None if raw is None else [(int(a), str(t)) for a, t in raw],
        "failures": [],             # (agent, reason, detail)
        "retained": {},             # agent -> task id, as decided
        "fallback": {},             # agent -> DEFEND(own station)
        "fallback_conflicts": [],   # (agent, fallback task, holder of that task)
    }
    candidates: Dict[int, str] = {}
    if raw is None:
        for agent in active:
            report["failures"].append((agent, UNUSABLE_RESPONSE, None))
    else:
        entries = defaultdict(list)
        for agent, task in raw:
            agent = int(agent)
            if agent not in active:
                report["failures"].append((agent, NOT_ACTIVE, str(task)))
                continue
            entries[agent].append(str(task))
        for agent in active:
            given = entries.get(agent, [])
            if not given:
                report["failures"].append((agent, MISSING, None))
            elif len(set(given)) > 1:
                report["failures"].append((agent, CONFLICTING_ENTRIES, sorted(set(given))))
            elif given[0] not in catalogue_ids:
                report["failures"].append((agent, UNKNOWN_TASK, given[0]))
            else:
                candidates[agent] = given[0]

    holders: Dict[str, int] = {}
    for agent in sorted(candidates):
        task = candidates[agent]
        if task in holders:
            report["failures"].append((agent, CAPACITY_CONFLICT, task))
            continue
        holders[task] = agent
        report["retained"][agent] = task

    for agent in active:
        if agent in report["retained"]:
            continue
        task = own_station_task(agent)
        report["fallback"][agent] = task
        if task in holders:
            report["fallback_conflicts"].append((agent, task, holders[task]))

    assignment = dict(report["retained"])
    assignment.update(report["fallback"])
    return assignment, report


class NearestCommanderDecider:
    """Deterministic rule decider for the free rule backend (plumbing, not a
    result of the paper): each believed contact's slot-0 NEUTRALIZE task goes to
    the nearest still-unassigned agent; everyone else holds their own station.
    Ties break by agent id."""

    def decide(self, view, catalogue, positions):
        free = sorted(positions)
        out = []
        for task in catalogue:
            if task.kind != NEUTRALIZE or task.slot != 0 or not free:
                continue
            loc = np.asarray(task.location, dtype=float)
            best = min(free, key=lambda a: (float(np.linalg.norm(positions[a] - loc)), a))
            out.append((best, task.task_id))
            free.remove(best)
        out.extend((a, own_station_task(a)) for a in free)
        return out


class CommanderTaskPlanner:
    """``plan_team(view, max_age)`` planner for ``centralized_from_belief``.

    ``task_model`` declares the task set this planner realizes; Simulation
    reads it to declare that set itself (and so disable implicit screening),
    rather than relying on the planner to behave like it. ``bind`` receives the
    episode's TaskContext, built from the live solver exactly as CBAA's is.
    """

    task_model = THREE_TASK

    def __init__(self, decider):
        self.decider = decider
        self.task_ctx: Optional[TaskContext] = None
        self.reports: List[dict] = []

    def bind(self, task_ctx: TaskContext):
        if task_ctx.task_set != self.task_model:
            raise ValueError(
                f"commander realizes {self.task_model!r} but was bound to "
                f"{task_ctx.task_set!r}"
            )
        self.task_ctx = task_ctx
        self.reports = []
        self.last_issued = None

    def plan_team(self, view, max_age=None):
        if self.task_ctx is None:
            raise RuntimeError("CommanderTaskPlanner.plan_team before bind()")
        catalogue = generate_tasks(view, self.task_ctx, max_age=max_age)
        positions = {int(view.agent_id): np.asarray(view.self_state, dtype=float).ravel()[0:2]}
        for j, state in (view.teammates or {}).items():
            positions[int(j)] = np.asarray(state, dtype=float).ravel()[0:2]
        observe = getattr(self.decider, "observe_history", None)
        if observe is not None:
            # The commander's own assignment history (what it last issued).
            observe(getattr(self, "last_issued", None))
        try:
            raw = self.decider.decide(view, catalogue, positions)
        except StopRun:
            raise  # a replay miss or the spend cap stops the run, never a fallback
        except Exception as exc:  # a failed model call is an unusable response
            raw, error = None, repr(exc)
        else:
            error = None
        assignment, report = validate_assignment(
            raw, [t.task_id for t in catalogue], positions.keys()
        )
        report.update(now=int(view.now), commander=int(view.agent_id), error=error,
                      catalogue=[t.task_id for t in catalogue])
        self.reports.append(report)
        self.last_issued = (int(view.now), dict(assignment))
        out = output_for(assignment, fallback="station", task_set=self.task_ctx.task_set)
        return out
