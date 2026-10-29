"""Tasks, generated deterministically from one agent's belief.

: *"Generate the candidate task set deterministically from the local
belief ... The task generator must use the same belief object as the LLM prompt
builder."* It does -- the frozen ``LocalWorldView`` -- plus a ``TaskContext`` of
static geometry, frozen at construction, exactly as the prompt builder takes a
``PromptContext``.

Four kinds:

    NEUTRALIZE(contact, slot)  close to contact: mutual destruction, +points
    SCREEN(contact)            interpose on the perimeter, contact -> centre
    DEFEND(station_k)          hold fixed perimeter station k
    RECON(cell)                patrol toward a coverage cell and sweep

**SCREEN exists because of a flaw, not for completeness**. A
DEFEND agent that believes in a contact already screens one -- but *its own
nearest believed* contact, chosen implicitly per agent per step by no allocator
(``decentralized_solver.py:362-371``). So two agents can screen one contact while
another goes unscreened, invisibly. In the episodes deconfliction lost, the
allocation diagnostic's traces support screening and induced evasion as a
*candidate* mechanism -- not an established cause
(, corrections). That
is reason enough to represent screening explicitly: SCREEN makes the screened
contact a choice the allocator can see.

**SCREEN can be switched off**: ``TaskContext.task_set`` selects one of
``TASK_SETS``. ``"four_task"``, the default, is the historical model every
earlier run used. ``"three_task"`` generates no SCREEN. The solver and runtime
close screening's implicit paths in that mode; see


**Task ids are stable strings**, so two agents that believe in the same contact
generate the same id for it -- the precondition for consensus. An agent that
does not believe in a contact generates no task for it, and
cannot acquire one from auction traffic.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

NEUTRALIZE = "NEUTRALIZE"
SCREEN = "SCREEN"
DEFEND = "DEFEND"
RECON = "RECON"

# Canonical generation order. Not a priority -- bids decide priority -- but a
# fixed order makes the task list, and therefore every tiebreak downstream of
# it, deterministic.
_KIND_ORDER = {NEUTRALIZE: 0, SCREEN: 1, DEFEND: 2, RECON: 3}

# Which kinds an auction may generate. The default is the historical
# four-task model; absent anywhere, a task set means this one.
FOUR_TASK = "four_task"
THREE_TASK = "three_task"
DEFAULT_TASK_SET = FOUR_TASK
TASK_SETS = {
    FOUR_TASK: frozenset({NEUTRALIZE, SCREEN, DEFEND, RECON}),
    THREE_TASK: frozenset({NEUTRALIZE, DEFEND, RECON}),
}


def check_task_set(task_set) -> str:
    if task_set not in TASK_SETS:
        raise ValueError(f"unknown task set {task_set!r}; expected one of {sorted(TASK_SETS)}")
    return task_set


def screen_available(task_set) -> bool:
    return SCREEN in TASK_SETS[check_task_set(task_set)]


@dataclass(frozen=True)
class Task:
    """One thing one agent could do this epoch.

    ``location`` is where the agent would have to go -- the point its ETA is
    measured to. It is derived from the *belief*, never from ground truth: a
    NEUTRALIZE location is the contact's **predicted** position from the
    agent's own (possibly old) sighting.
    """

    task_id: str
    kind: str
    location: Tuple[float, float]
    contact_id: Optional[int] = None
    slot: int = 0
    station_id: Optional[int] = None
    cell_id: Optional[int] = None

    @property
    def sort_key(self):
        ident = (
            self.contact_id if self.contact_id is not None
            else self.station_id if self.station_id is not None
            else self.cell_id if self.cell_id is not None
            else -1
        )
        return (_KIND_ORDER[self.kind], ident, self.slot)


@dataclass(frozen=True)
class TaskContext:
    """Static scenario geometry: the shape of the board, never the adversary.

    Frozen and built once, like ``PromptContext``. ``defense_polygon`` and
    ``stations`` are tuples so the frozenness is real.

    ``slots`` is how many agents may win pursuit of one contact. 1 is strict
    CBAA. 2 is the duplicate-permitting arm the allocation diagnostic requires
    Exp 4 to carry -- expressed as two *distinct tasks*, so CBAA's
    one-agent-per-task invariant is untouched.
    """

    defense_polygon: Tuple[Tuple[float, float], ...]
    stations: Tuple[Tuple[int, Tuple[float, float]], ...]
    slots: int = 1
    task_set: str = DEFAULT_TASK_SET

    def __post_init__(self):
        check_task_set(self.task_set)

    @staticmethod
    def from_simulation(sim, slots: int = 1, task_set: str = DEFAULT_TASK_SET):
        """Taken off a live Simulation's solver, so SCREEN and DEFEND locations
        are the points the controllers will actually steer to -- a hand-built
        polygon or station list could drift from them, and every ETA would then
        describe a journey no agent makes."""
        solver = sim.solver
        if solver is None:
            # Simulation builds its solver in __init__ (testbed.py:480), so this
            # guards a hand-assembled or partially torn-down object, not a
            # normal construction order.
            raise ValueError("Simulation has no solver; cannot read its geometry")
        polygon = tuple(
            (float(x), float(y)) for x, y in np.asarray(solver.defense_polygon, dtype=float)
        )
        stations = tuple(
            (int(k), (float(v[0]), float(v[1])))
            for k, v in sorted(solver.defend_stations.items())
        )
        return TaskContext(defense_polygon=polygon, stations=stations, slots=int(slots),
                           task_set=task_set)

    def polygon_array(self) -> np.ndarray:
        return np.asarray(self.defense_polygon, dtype=float).reshape(-1, 2)


def closest_point_on_polygon(point, polygon) -> np.ndarray:
    """The perimeter point nearest ``point`` -- SCREEN's interposition point.

    Deliberately the same computation as
    ``DefendController._closest_point_on_polygon``, and pinned to it by a test
    that takes the controller off a live Simulation: if the two diverged, a
    SCREEN bid's ETA would measure a journey to a point the controller never
    steers to.
    """
    point = np.asarray(point, dtype=float).reshape(2)
    polygon = np.asarray(polygon, dtype=float).reshape(-1, 2)
    closed = np.vstack([polygon, polygon[0]])
    best, best_d = polygon[0].copy(), np.inf
    for k in range(len(polygon)):
        start, end = closed[k], closed[k + 1]
        seg = end - start
        denom = float(np.dot(seg, seg))
        if denom <= 1e-12:
            cand = start.copy()
        else:
            alpha = np.clip(np.dot(point - start, seg) / denom, 0.0, 1.0)
            cand = start + alpha * seg
        d = float(np.sum((cand - point) ** 2))
        if d < best_d:
            best, best_d = cand, d
    return best


def _xy(v) -> Tuple[float, float]:
    v = np.asarray(v, dtype=float).ravel()
    return (float(v[0]), float(v[1]))


def generate_tasks(view, ctx: TaskContext, max_age=None) -> List[Task]:
    """Every task this agent could bid on this epoch, in canonical order.

    Contacts come from ``view.fresh_targets(max_age)`` -- the same call, with
    the same staleness semantics, as the prompt builder and the rule planner,
    so all three reason over the same contact set at identical settings.
    """
    tasks: List[Task] = []
    polygon = ctx.polygon_array()
    have_polygon = len(polygon) >= 3
    screen = screen_available(ctx.task_set)

    believed = view.fresh_targets(max_age)
    for contact_id in sorted(believed):
        predicted = believed[contact_id].predicted_position(view.now, view.dt)
        for slot in range(max(1, int(ctx.slots))):
            tasks.append(Task(
                task_id=f"{NEUTRALIZE}:{int(contact_id)}:{slot}",
                kind=NEUTRALIZE,
                location=_xy(predicted),
                contact_id=int(contact_id),
                slot=slot,
            ))
        if have_polygon and screen:
            tasks.append(Task(
                task_id=f"{SCREEN}:{int(contact_id)}",
                kind=SCREEN,
                location=_xy(closest_point_on_polygon(predicted, polygon)),
                contact_id=int(contact_id),
            ))

    for station_id, position in ctx.stations:
        tasks.append(Task(
            task_id=f"{DEFEND}:{int(station_id)}",
            kind=DEFEND,
            location=_xy(position),
            station_id=int(station_id),
        ))

    for region in view.coverage_regions() or []:
        tasks.append(Task(
            task_id=f"{RECON}:{int(region['region_id'])}",
            kind=RECON,
            location=_xy(region["center"]),
            cell_id=int(region["region_id"]),
        ))

    return sorted(tasks, key=lambda t: t.sort_key)


def tasks_by_id(tasks) -> Dict[str, Task]:
    return {t.task_id: t for t in tasks}
