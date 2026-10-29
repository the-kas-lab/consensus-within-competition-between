"""CBAA inside a running episode: epochs, transport, and what commitments mean.

The testbed owns one ``CBAARuntime`` when ``Simulation(allocation="cbaa")``, and
none otherwise -- every existing mode is untouched. The runtime keeps the
auction logic out of ``testbed.py``; the testbed only (1) starts an epoch at
each planning step, (2) routes auction traffic in each communication round, and
(3) puts changed commitments into force.

**Transport.** Three, one per belief mode by default, because belief sharing
and auction traffic ride the same radio:

    networked -> "network"  AUCTION_MESSAGE envelopes over the real Network,
                            dropped by the existing is_stale_allocation rule
    shared    -> "ideal"    complete, lossless, one round per comm tick -- the
                            same double-buffered semantics as the Network, with
                            the Network's graph replaced by a complete one
    local     -> "none"     no auction traffic: independent greedy, the
                            no-coordination bound

"ideal" deliberately mirrors the network's timing (deliver what was sent LAST
round), so an ideal auction differs from a networked R=inf auction only in the
graph and losses, never in when a round happens.

**Epoch lag, declared.** The communication round runs before planning within a
step (testbed order). So the round at a planning step still carries the
previous epoch's auction states, which the next round's receivers drop as
stale. Each epoch therefore spends one round on the handover. Honest, and the
same under every transport.

**Dead agents' claims.** A blue disabled mid-epoch stops broadcasting, but
others may still hold its last winning bid for a task until the next epoch
resets the auction. The task is orphaned for at most one epoch. Not repaired --
repairing it would need a liveness broadcast the protocol does not have.
"""

from typing import Dict, Iterable, List, Optional, Tuple

from coordination.bidding import BidContext, BidModifiers, bids
from coordination.cbaa import AuctionState, CBAAAgent, commitment_conflicts, conflicts
from coordination.tasks import (
    DEFAULT_TASK_SET,
    DEFEND,
    NEUTRALIZE,
    RECON,
    SCREEN,
    TASK_SETS,
    TaskContext,
    check_task_set,
    generate_tasks,
    screen_available,
)
from planning.planner import PlannerOutput
from planning.roles import BlueRole

# The LLM bridges that show each task's bid terms, and the bid version each one states.
BID_VERSION_STATED = {"m8b.7": "v2.3", "m8b.9": "v2.3"}

TRANSPORT_FOR_BELIEF_MODE = {"networked": "network", "shared": "ideal", "local": "none"}

# What an agent with no commitment yet executes:
#   "legacy"  -- DEFEND with no override: the solver's implicit DEFEND, which
#                holds the agent's own station or screens its own nearest
#                believed contact, and which the station-heading rule outward posture never
#                reaches. The default; every existing run is unchanged.
#   "station" -- an explicit DEFEND(station i), the agent's own station, through
#                the override path, so the station-heading rule applies and no contact is screened by
#                an allocator nobody runs. A DECLARED FALLBACK, not a neutral
#                one: it changes the first steps of every episode, so a
#                comparison using it needs references run with it too.
FALLBACKS = ("legacy", "station")


def check_fallback(fallback: str, task_set: str = DEFAULT_TASK_SET) -> str:
    """The fallback, validated against the task set. "legacy" is the
    solver's implicit DEFEND, which screens the agent's nearest believed
    contact, so it would reintroduce screening where SCREEN is disabled.
    Refused rather than quietly upgraded: the run's recorded configuration must
    say what the agents did."""
    if fallback not in FALLBACKS:
        raise ValueError(f"unknown CBAA fallback {fallback!r}; expected one of {FALLBACKS}")
    if fallback == "legacy" and not screen_available(task_set):
        raise ValueError(
            f"cbaa_fallback='legacy' is the implicit screening DEFEND, which task set "
            f"{task_set!r} excludes; use cbaa_fallback='station'"
        )
    return fallback


def output_for(commitments: Dict[int, Optional[str]], fallback: str = "legacy",
               task_set: str = DEFAULT_TASK_SET) -> PlannerOutput:
    """Commitments -> the PlannerOutput the testbed already knows how to apply.

    Parsed from the task id, which is self-describing, rather than looked up in
    the current epoch's task table: under the rule that an auction may span planning epochs an agent keeps executing the
    PREVIOUS epoch's commitment until the new auction settles, and that task id
    may not exist in the new table.

    An active agent with no commitment yet (the first epoch, before it settles)
    is recorded as DEFEND, so the role histories are complete rather than
    silently missing agents. With ``fallback="legacy"`` it gets no override
    (the implicit DEFEND); with ``"station"`` it gets its own station,
    explicitly. Either way it is NOT a commitment: the auction's state, and so
    every bid's switch cost, still sees ``None``.

    Under a task set without SCREEN, a SCREEN commitment is **refused**,
    not remapped. It cannot arise from shipped code -- an agent wins only tasks
    it generated, and it generates none -- so one here is a broken
    invariant, and mapping it to a station would hide that behind plausible
    behaviour.
    """
    check_fallback(fallback, task_set)
    screen = screen_available(task_set)
    out = PlannerOutput()
    for agent in sorted(commitments):
        task = commitments[agent]
        if task is None:
            out.roles[agent] = BlueRole.DEFEND
            if fallback == "station":
                out.defend_targets[agent] = ("station", int(agent))
            continue
        out.task_ids[agent] = task
        parts = task.split(":")
        kind, ident = parts[0], int(parts[1])
        if kind == NEUTRALIZE:
            out.roles[agent] = BlueRole.NEUTRALIZE
            out.target_red_ids[agent] = ident
        elif kind == SCREEN:
            if not screen:
                raise ValueError(
                    f"agent {agent} holds commitment {task!r}, but task set "
                    f"{task_set!r} has no SCREEN"
                )
            out.roles[agent] = BlueRole.DEFEND
            out.defend_targets[agent] = ("contact", ident)
        elif kind == DEFEND:
            out.roles[agent] = BlueRole.DEFEND
            out.defend_targets[agent] = ("station", ident)
        elif kind == RECON:
            out.roles[agent] = BlueRole.RECON
            out.recon_target_regions[agent] = ident
        else:
            raise ValueError(f"unknown task kind in {task!r}")
    return out


class CBAARuntime:
    def __init__(self, slots: int = 1, stable_rounds: Optional[int] = None,
                 profile: str = "balanced", transport: str = "ideal",
                 modifier_source=None, fallback: str = "legacy",
                 task_set: str = DEFAULT_TASK_SET):
        if transport not in ("network", "ideal", "none"):
            raise ValueError(f"unknown CBAA transport {transport!r}")
        # The three-task rule. Held here, not per arm: the independent arm is this runtime with
        # transport "none", so every arm generates from the same task set.
        self.task_set = check_task_set(task_set)
        self.fallback = check_fallback(fallback, self.task_set)
        # Set by the testbed's schedule guard: the fewest communication
        # rounds any planning epoch can hold, and whether that makes local
        # stability impossible. None until a Simulation fills them in.
        # Deliberately NOT a "can settle" flag: enough round opportunities do
        # not guarantee agreement under loss, partition or divergent tasks.
        self.min_rounds_per_epoch = None
        self.stability_impossible = None
        BidModifiers(scoring_profile=profile)  # validate the name now, not mid-episode
        self.slots = int(slots)
        self.stable_rounds_config = stable_rounds
        self.profile = str(profile)
        self.transport = transport
        # The LLM bridge (coordination/llm_bridge.py), or None -- the default,
        # under which every bid uses the run's profile and nothing else, exactly
        # as before. Consulted once per epoch, AFTER every agent's view and task
        # list is frozen, BEFORE any bid is computed.
        self.modifier_source = modifier_source

    def reset(self, num_blues: int, task_ctx: TaskContext, bid_ctx: BidContext):
        if task_ctx.task_set != self.task_set:
            raise ValueError(
                f"task context generates task set {task_ctx.task_set!r}, but this "
                f"runtime was configured with {self.task_set!r}"
            )
        stable = (
            max(1, num_blues - 1)
            if self.stable_rounds_config is None
            else int(self.stable_rounds_config)
        )
        self.agents: Dict[int, CBAAAgent] = {
            i: CBAAAgent(i, stable_rounds=stable) for i in range(num_blues)
        }
        self.task_ctx = task_ctx
        self.bid_ctx = bid_ctx
        self.epoch = -1
        self._outbox: Dict[int, AuctionState] = {}
        self.conflict_log: List[Dict] = []
        # Bid v2.3 only: every agent-epoch in which a
        # NEUTRALIZE task carried continuity relief, as (step, agent,
        # commitment, {task_id: relief}). Diagnostic; the bids are unchanged.
        self.relief_log: List[Tuple] = []
        self.stats = {
            "epochs": 0,
            "rounds": 0,
            "auction_messages_sent": 0,
            "recommitments": 0,
            "rounds_to_stable": [],
            "unsettled_at_epoch_end": 0,
            # Dropped by the Network's is_stale_allocation BEFORE any agent saw
            # them. Counted here so "stale" means the same thing under every
            # transport: first measured, a networked run reported 0 and an ideal
            # run 84 for the same one-round epoch handover, because only the
            # agent-level drops were being counted.
            "stale_dropped_by_transport": 0,
        }

    # -- epochs -------------------------------------------------------------

    def start_epoch(self, step: int, views: Dict, max_age=None,
                    modifiers: Optional[Dict[int, BidModifiers]] = None):
        """Inject epoch k's bids, computed by each agent from its own frozen
        view. The conflict sample for epoch k-1 is taken FIRST -- the rule that an auction may span planning epochs's
        follow-up: allocation state is sampled at the epoch boundary, before new
        bids are injected."""
        active = sorted(views)
        if self.epoch >= 0:
            live = [self.agents[i] for i in active]
            self.conflict_log.append({
                "step": int(step),
                "epoch": int(self.epoch),
                "conflicts": conflicts(live),
                # What is in force, sampled at the same instant.
                "commitment_conflicts": commitment_conflicts(live),
            })
            for a in live:
                self.stats["recommitments"] += a.recommitments
                if a.rounds_to_stable is None:
                    self.stats["unsettled_at_epoch_end"] += 1
                else:
                    self.stats["rounds_to_stable"].append(a.rounds_to_stable)

        self.epoch += 1
        self.stats["epochs"] += 1
        default_mods = BidModifiers(scoring_profile=self.profile)
        # Every agent's task list is generated before any modifier call, so a
        # bridge sees a frozen (view, tasks) pair per agent and nothing else.
        tasks_by_agent = {
            i: generate_tasks(views[i], self.task_ctx, max_age=max_age) for i in active
        }
        allowed = TASK_SETS[self.task_set]
        for i in active:
            stray = sorted({t.kind for t in tasks_by_agent[i]} - allowed)
            if stray:
                raise ValueError(
                    f"agent {i} generated {stray} outside task set {self.task_set!r}"
                )
        if modifiers is None and self.modifier_source is not None:
            extra = {}
            bridge = getattr(self.modifier_source, "bridge_version", None)
            if self.bid_ctx.v2 is not None and bridge in BID_VERSION_STATED:
                # The bridges show each task's actual terms under the default
                # modifiers: the same function and frozen inputs as the bids.
                from coordination.bidding import bid_terms_v2

                stated = BID_VERSION_STATED[bridge]
                if self.bid_ctx.v2.version != stated:
                    raise ValueError(f"{bridge} states bid version {stated}; this run bids with "
                                     f"{self.bid_ctx.v2.version!r}")
                v2 = self.bid_ctx.v2
                extra = {"bid_terms": {i: bid_terms_v2(views[i], tasks_by_agent[i], self.bid_ctx,
                                                       self.agents[i].commitment, default_mods)
                                       for i in active},
                         "current_tasks": {i: self.agents[i].commitment for i in active},
                         "remaining_s": max(0.0, (v2.horizon_steps - int(step)) * v2.dt)}
                expected = "independent" if self.transport == "none" else "cbaa"
                if getattr(self.modifier_source, "selection", None) != expected:
                    raise ValueError(f"{bridge} tells the agents they use {expected!r} selection "
                                     f"under transport {self.transport!r}; the bridge says "
                                     f"{getattr(self.modifier_source, 'selection', None)!r}")
            modifiers = self.modifier_source.modifiers_for_epoch(
                step,
                {i: (views[i], tasks_by_agent[i]) for i in active},
                default_profile=self.profile,
                max_age=max_age,
                **extra,
            )
        for i in active:
            view = views[i]
            tasks = tasks_by_agent[i]
            agent = self.agents[i]
            # None = the bridge failed totally for this agent: default
            # modifiers, i.e. exactly rule-based CBAA. Recorded by the bridge.
            mods = (modifiers or {}).get(i) or default_mods
            if self.bid_ctx.v2 is not None and self.bid_ctx.v2.version == "v2.3":
                from coordination.bidding import bid_terms_v2
                from coordination.strategy_weights import StrategyWeights, strategy_terms

                if isinstance(mods, StrategyWeights):
                    # The default terms plus exact per-weight deltas.
                    terms = strategy_terms(view, tasks, self.bid_ctx, agent.commitment, mods)
                else:
                    terms = bid_terms_v2(view, tasks, self.bid_ctx, agent.commitment, mods)
                b = {k: x["bid"] for k, x in terms.items()}  # = bids(): same function
                relief = {k: round(x["continuity_relief"], 4) for k, x in terms.items()
                          if x.get("continuity_relief", 0.0) > 0.0}
                if relief:
                    self.relief_log.append((int(step), int(i), agent.commitment, relief))
            else:
                b = bids(view, tasks, self.bid_ctx,
                         current_task_id=agent.commitment, modifiers=mods)
            agent.start_epoch(self.epoch, b, [t.task_id for t in tasks])

    # -- rounds ---------------------------------------------------------------

    def commitments(self, active: Iterable[int]) -> Dict[int, Optional[str]]:
        return {i: self.agents[i].commitment for i in sorted(active)}

    def output(self, active: Iterable[int]) -> PlannerOutput:
        """The commitments in force, with this run's fallback for agents that
        have none yet -- the one place the testbed turns auction state into
        roles, so the fallback cannot apply at one call site and not another."""
        return output_for(self.commitments(active), fallback=self.fallback,
                          task_set=self.task_set)

    def outgoing(self, agent: int) -> Optional[AuctionState]:
        if self.epoch < 0:
            return None
        self.stats["auction_messages_sent"] += 1
        return self.agents[agent].outgoing()

    def receive_round(self, delivered: Dict[int, List[AuctionState]], active: Iterable[int]):
        """One round's consensus for every active agent -- including agents that
        received nothing, because a quiet round is what local stability counts.
        Returns the agents whose commitment changed."""
        if self.epoch < 0:
            return set()
        active = sorted(active)
        before = self.commitments(active)
        for i in active:
            self.agents[i].receive(delivered.get(i, []))
        self.stats["rounds"] += 1
        after = self.commitments(active)
        return {i for i in active if before[i] != after[i]}

    def local_round(self, active: Iterable[int]):
        """One round under the "ideal" or "none" transport.

        Double-buffered exactly like the Network: deliver what was sent LAST
        round, then capture this round's outbox. "ideal" delivers every other
        active agent's state (a complete graph); "none" delivers nothing."""
        if self.epoch < 0:
            return set()
        active = sorted(active)
        if self.transport == "ideal":
            delivered = {
                i: [self._outbox[j] for j in active if j != i and j in self._outbox]
                for i in active
            }
        else:
            delivered = {}
        changed = self.receive_round(delivered, active)
        if self.transport == "ideal":
            # Only a transport that delivers builds an outbox. First written,
            # "none" built one anyway and reported hundreds of auction messages
            # sent in an episode where none could be received.
            self._outbox = {i: self.outgoing(i) for i in active}
        return changed

    def note_stale_transport_drop(self):
        self.stats["stale_dropped_by_transport"] += 1

    # -- read-outs --------------------------------------------------------------

    def metrics(self) -> Dict:
        rts = self.stats["rounds_to_stable"]
        out = {
            "transport": self.transport,
            "profile": self.profile,
            "slots": self.slots,
            "epochs": self.stats["epochs"],
            "rounds": self.stats["rounds"],
            "auction_messages_sent": self.stats["auction_messages_sent"],
            "mean_rounds_to_stable": (sum(rts) / len(rts)) if rts else None,
            "max_rounds_to_stable": max(rts) if rts else None,
            "unsettled_at_epoch_end": self.stats["unsettled_at_epoch_end"],
            "recommitments": self.stats["recommitments"],
            "epochs_with_conflicts": sum(1 for e in self.conflict_log if e["conflicts"]),
            "conflicted_task_epochs": sum(len(e["conflicts"]) for e in self.conflict_log),
            # The communication-round schedule: duplicates IN FORCE, which claims alone do not show.
            "epochs_with_commitment_conflicts": sum(
                1 for e in self.conflict_log if e["commitment_conflicts"]),
            "commitment_conflicted_task_epochs": sum(
                len(e["commitment_conflicts"]) for e in self.conflict_log),
            "fallback": self.fallback,
            "min_rounds_per_epoch": self.min_rounds_per_epoch,
            "stability_impossible": self.stability_impossible,
            # Agent-level and transport-level drops are mutually exclusive (a
            # message the transport drops never reaches an agent), so the sum is
            # the one number comparable across transports.
            "stale_messages_dropped": (
                sum(a.stale_messages_ignored for a in self.agents.values())
                + self.stats["stale_dropped_by_transport"]
            ),
            "foreign_entries_ignored": sum(a.foreign_entries_ignored for a in self.agents.values()),
        }
        # The three-task rule: present only when not the historical default. Exp 5 records
        # embed these metrics verbatim and re-simulations are verified against
        # them by canonical equality, so an unconditional key would make every
        # historical record "mismatch". Absent means four-task.
        if self.task_set != DEFAULT_TASK_SET:
            out["task_set"] = self.task_set
        return out
