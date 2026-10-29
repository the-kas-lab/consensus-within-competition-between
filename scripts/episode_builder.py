"""Experiment 0 pilot: does target-observation sharing change the outcome?

Two arms, paired on seed, identical in everything else:

    shared   observe locally, then merge every teammate's first-hand
             observations with no latency or loss
    local    observe locally, share no target observations

Teammate state is available in BOTH arms, and liveness with it. The lower arm is
therefore not communication-free and is never described as "zero communication"
(the decentralization rule commitment 6). What is bracketed is target observations.

Both arms run ``per_agent_planning=True``. That is the whole point of M3: before
it, role assignment stayed centralized and fed by the team-wide detection union
regardless of belief, so the arms differed only in what the controllers acted on.

Held fixed, deliberately, as controlled variables:

* the safety configuration. requires the MPC and safety
  stack be held fixed across conditions.
* the planner, its parameters, the scenario family, and the red team.

  The red team is **not omniscient**, despite what an earlier version of this
  docstring said. Audited for the intruder-team rule (): reds detect
  blues through the *same* FOV half-angle and range as blues
  (``testbed.py:958-962``), so their sensing is genuinely limited. What they do
  have is perfect zero-latency sharing across the red team -- one detection
  union for all of them -- exact ground-truth positions for anything detected
  (``planner.py:536`` reads ``x_current`` directly), and, in EVADE, the detected
  blues' full planned MPC trajectories over the horizon
  (``red_controllers.py:65-77``). That last one is an asymmetry with no blue
  counterpart: blues extrapolate red *positions* from a possibly-stale belief,
  reds react to blue *plans*.

  It is held **constant across every condition**, which is the point -- it is a
  control variable, not an adversary-strength claim. If red information varied
  with the condition, no measured difference could be attributed to the blue
  manipulation.

Free: ``belief_mode`` only.

Run:
    PYTHONPATH=$PWD python3 -m scripts.episode_builder --seeds 30

Costs nothing -- rule-based planners, no network calls.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planning.planner import (  # noqa: E402
    DeduplicatingBlueRolePlanner,
    RuleBasedBlueRolePlanner,
    RuleBasedRedRolePlanner,
)
from comms.link_models import loss_model_from_spec  # noqa: E402
from planning.roles import BlueRole  # noqa: E402
from simulation.testbed import SCENARIO_FAMILIES, Simulation  # noqa: E402

ARMS = ("shared", "local")

# The allocation rule is a separate experimental condition, not a fix. `nearest`
# is the baseline every equivalence test is attached to and stays the default;
# `dedup` is the step-3 probe, run in BOTH arms so the residual duplication that
# only communication could remove stays visible.
RULES = {
    "nearest": RuleBasedBlueRolePlanner,
    "dedup": DeduplicatingBlueRolePlanner,
}


# ---------------------------------------------------------------------------
# Running one episode
# ---------------------------------------------------------------------------


class _Quiet:
    """Silence CasADi/OSQP *and* the planner's per-epoch prints.

    Redirection has to happen at the file-descriptor level because CasADi and
    OSQP write from C, which Python-level redirection never sees. The flushes
    are not optional: Python buffers sys.stdout when it is not a tty, so
    anything printed inside the block would otherwise sit in the buffer and be
    written *after* fd 1 is restored -- appearing on the terminal exactly as if
    nothing had been suppressed.
    """

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self._saved = os.dup(1), os.dup(2)
        self._null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._null, 1)
        os.dup2(self._null, 2)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._saved[0], 1)
        os.dup2(self._saved[1], 2)
        for fd in (*self._saved, self._null):
            os.close(fd)
        return False


def run_episode(seed, belief_mode, cfg):
    sim = build_episode(seed, belief_mode, cfg)
    with _Quiet():
        sim.run()
    return sim


def build_episode(seed, belief_mode, cfg):
    """The episode ``run_episode`` runs, constructed but not yet run.

    Split out so a caller can attach instrumentation from outside the testbed
    before the first step (the auction smoke's assignment ledger does), while every
    runner keeps going through this one definition.
    """
    return Simulation(
        num_blues=cfg["blues"],
        num_reds=cfg["reds"],
        seed=seed,
        sim_steps=cfg["steps"],
        dt=0.1,
        planning_interval_seconds=cfg["planning_interval_seconds"],
        belief_mode=belief_mode,
        # Defaults reproduce Exp 0 and the communication conditions exactly: those runs are decision-
        # decentralized and never touched the command-delivery conditions options, so a cfg that
        # omits these keys builds the same Simulation it always did. the command-delivery conditions's
        # centralized conditions set them; nothing else does.
        per_agent_planning=cfg.get("per_agent_planning", True),
        centralized_from_belief=cfg.get("centralized_from_belief", False),
        commander_id=cfg.get("commander_id", 0),
        commander_failure_stops_planning=cfg.get(
            "commander_failure_stops_planning", False
        ),
        deliver_assignments_over_network=cfg.get(
            "deliver_assignments_over_network", False
        ),
        command_max_age_steps=cfg.get("command_max_age_steps", None),
        command_max_hops=cfg.get("command_max_hops", None),
        # The auction allocation. Same pattern as the command-delivery conditions keys above: every default is
        # Simulation's own, so a cfg that omits them builds the Simulation it
        # always did and every existing runner stays byte-identical. With
        # allocation="cbaa" the role planner below is built but superseded --
        # roles come from each agent's settled auction commitment.
        allocation=cfg.get("allocation", None),
        cbaa_slots=cfg.get("cbaa_slots", 1),
        cbaa_stable_rounds=cfg.get("cbaa_stable_rounds", None),
        cbaa_profile=cfg.get("cbaa_profile", "balanced"),
        cbaa_transport=cfg.get("cbaa_transport", None),
        # The communication-round schedule. Both default to Simulation's own: the implicit DEFEND before a
        # first commitment, and the schedule guard active.
        cbaa_fallback=cfg.get("cbaa_fallback", "legacy"),
        cbaa_allow_unsettleable=cfg.get("cbaa_allow_unsettleable", False),
        # The three-task rule: "three_task" disables SCREEN (needs cbaa_fallback="station").
        # Absent, it is Simulation's own default, the historical four-task
        # model, so every existing runner is unchanged.
        cbaa_task_set=cfg.get("cbaa_task_set", "four_task"),
        cbaa_bid_version=cfg.get("cbaa_bid_version", "v1"),
        # The station-heading rule: the explicit DEFEND(station) posture's heading. Default is
        # Simulation's own, so every existing runner is unchanged.
        station_heading=cfg.get("station_heading", "inherited"),
        # The pursuer aim-point rule: the NEUTRALIZE aim point between assignment applications.
        # Default is Simulation's own (historical "frozen"), so every existing
        # runner is unchanged.
        neutralize_aim=cfg.get("neutralize_aim", "frozen"),
        # The pursuer sensor-heading rule: where a NEUTRALIZE agent points its sensor. Default is
        # Simulation's own (historical "inherited").
        neutralize_heading=cfg.get("neutralize_heading", "inherited"),
        # Communication parameters. Both default to Simulation's own defaults,
        # so an Exp 0 cfg that omits them is byte-identical to before -- and a
        # `networked` cell from the communication conditions sweep reuses this one episode definition
        # rather than a second copy of it, which is what keeps the communication conditions's numbers
        # comparable with Exp 0's.
        r_comm=cfg.get("r_comm", float("inf")),
        comm_interval_seconds=cfg.get("comm_interval_seconds", None),
        # Packet loss (Exp 5).
        # A plain spec, built into a NEW model here for every episode. Absent,
        # it is None, Simulation's own default (no loss), so every existing
        # runner is byte-identical.
        comm_loss_model=loss_model_from_spec(cfg.get("comm_loss_model")),
        # The LLM bid-modifier bridge (coordination/llm_bridge.py), or None,
        # Simulation's own default: rule-based CBAA.
        cbaa_modifier_source=cfg.get("cbaa_modifier_source", None),
        # Held fixed within a run, varied across runs. `symmetric` is retained
        # alongside `multi_bearing` rather than replaced by it, so that "it only
        # helps under condition X" stays a statement the paper can make.
        scenario_family=cfg["family"],
        # Safety: enabled and at the safety-parameter choice defaults. Simulation's own CBF defaults
        # are all False and main.py runs unfiltered, so this must be explicit.
        enable_blue_blue_cbf=True,
        enable_blue_red_cbf=True,
        enable_red_cbf=cfg.get("enable_red_cbf", False),
        enable_convex_hull_containment=False,
        # The competing-teams configuration. Every default is
        # Simulation's own, so a cfg that omits them builds the same episode.
        safety_state_sync=cfg.get("safety_state_sync", "sequential"),
        safety_filter=cfg.get("safety_filter", "joint"),
        local_cbf_gamma=cfg.get("local_cbf_gamma", None),
        local_cbf_red_red=cfg.get("local_cbf_red_red", True),
        local_cbf_red_blue=cfg.get("local_cbf_red_blue", True),
        intruder_info=cfg.get("intruder_info", "legacy"),
        falsified_reports=cfg.get("falsified_reports", None),
        intruder_team=cfg.get("intruder_team", None),
        unavailable_target=cfg.get("unavailable_target", "nearest_believed"),
        resolution_info=cfg.get("resolution_info", "legacy"),
        red_sensor_control=cfg.get("red_sensor_control", "fixed"),
        # The task-model commander (coordination/commander.py), when given;
        # otherwise the rule role planner every existing runner uses.
        role_planner=cfg.get("commander_planner") or RULES[cfg.get("rule", "nearest")](
            neutralize_fraction=cfg["neutralize_fraction"],
            min_neutralizers=1,
            recon_fraction=cfg["recon_fraction"],
            min_recon=1,
        ),
        red_role_planner=cfg.get("red_planner") or RuleBasedRedRolePlanner(),
        # image/non-hostile experiment; absent = None, Simulation's own default.
        decoys=cfg.get("decoys", None),
    )


# ---------------------------------------------------------------------------
# Metrics -- all read from the episode recorder, one source of truth
# ---------------------------------------------------------------------------


def _planning_steps(sim):
    """Only planning instants carry a *fresh* decision. target_red_ids persists
    between epochs, so counting coordination failures every step would weight
    one decision by the planning interval."""
    interval = sim.planning_interval_steps
    return [s for s in sim.recorder.steps if s["step"] % interval == 0]


def episode_metrics(sim):
    steps = sim.recorder.steps
    n = max(1, len(steps))
    m = dict(sim.episode_outcome())

    # -- detection availability ---------------------------------------------
    # The information bracket M0 measured at 1.29x: how much more the team sees
    # than any one agent. If this is ~1.0 the arms cannot differ for want of
    # anything to share.
    union = sum(1 for s in steps if s["detected_red_union"]) / n
    per_blue = [
        sum(1 for s in steps if s["per_blue_detected_reds"].get(str(i))) / n
        for i in range(sim.num_blues)
    ]
    individual = float(np.mean(per_blue)) if per_blue else 0.0
    m["union_detect_fraction"] = union
    m["individual_detect_fraction"] = individual
    m["information_bracket"] = (union / individual) if individual > 0 else float("nan")

    # -- belief age (age of information at the decision instant) -------------
    plan_steps = _planning_steps(sim)
    ages = [
        age
        for s in plan_steps
        for per in s["beliefs"].values()
        for age in per.values()
    ]
    m["mean_belief_age"] = float(np.mean(ages)) if ages else float("nan")
    m["p90_belief_age"] = float(np.percentile(ages, 90)) if ages else float("nan")
    m["mean_contacts_believed"] = float(
        np.mean([len(per) for s in plan_steps for per in s["beliefs"].values()])
    ) if plan_steps else float("nan")

    # -- belief disagreement -------------------------------------------------
    # Mean pairwise Jaccard distance between agents' believed contact sets, at
    # planning instants. Zero in the shared arm by construction; it is the
    # mediator the local arm is supposed to move.
    disagreements = []
    for s in plan_steps:
        sets = [set(per) for per in s["beliefs"].values()]
        pairs = [
            (a, b) for i, a in enumerate(sets) for b in sets[i + 1 :]
        ]
        for a, b in pairs:
            union_ab = a | b
            disagreements.append(
                0.0 if not union_ab else 1.0 - len(a & b) / len(union_ab)
            )
    m["belief_disagreement"] = (
        float(np.mean(disagreements)) if disagreements else 0.0
    )

    m.update(_live_belief_metrics(plan_steps))

    # -- coordination failures ----------------------------------------------
    # Made observable only by per-agent planning: a team planner cannot produce
    # either of these.
    dup_epochs = 0
    unassigned_epochs = 0
    for s in plan_steps:
        targets = {
            int(a): v
            for a, v in s["target_red_ids"].items()
            if v is not None and s["blue_roles"].get(a) == BlueRole.NEUTRALIZE.value
        }
        assigned = list(targets.values())
        if len(assigned) != len(set(assigned)):
            dup_epochs += 1
        believed = {int(c) for per in s["beliefs"].values() for c in per}
        active_believed = {
            c for c in believed if s["red_outcomes"][c] == "active"
        }
        if active_believed - set(assigned):
            unassigned_epochs += 1
    n_epochs = max(1, len(plan_steps))
    m["duplicate_assignment_rate"] = dup_epochs / n_epochs
    m["unassigned_contact_rate"] = unassigned_epochs / n_epochs
    # Logged per episode, not inferred from sim_steps / planning_interval_steps:
    # episodes terminate early on breach or on defenders winning, so the nominal
    # figure overstates how many decisions were actually taken.
    m["n_planning_epochs"] = len(plan_steps)
    m["n_recorded_steps"] = len(steps)

    # -- safety interventions ------------------------------------------------
    sm = sim.recorder.safety_metrics()
    pct = sm.get("magnitude_percentiles", {})
    m["cbf_du_p50"] = pct.get("50", float("nan"))
    m["cbf_du_p90"] = pct.get("90", float("nan"))
    m["cbf_intervention_fraction"] = sm.get(
        "intervention_fraction_by_threshold", {}
    ).get("0.01", float("nan"))
    m["cbf_max_du"] = sm.get("max_intervention_magnitude", float("nan"))

    # n_blues_disabled and attrition_rate now come from episode_outcome(), which
    # also documents why they are an identity with num_neutralized rather than
    # an independent corroborating measurement.
    return m


def _live_belief_metrics(plan_steps):
    """Belief mediators restricted to blues that are still alive, plus the
    decomposition of staleness into the part a protocol can do something about.

    **Why the restriction.** ``EpisodeRecorder`` snapshots every entry in
    ``belief_stores``, including blues that have been disabled, while every
    belief-consuming path in ``Simulation`` -- the merge in ``_update_beliefs``,
    ``_plan_per_agent``, ``_believed_targets`` -- filters to ``active_blues``. A
    disabled blue's store therefore freezes at the moment of its death and ages
    for the rest of the episode while nothing ever reads it. The unqualified
    ``mean_belief_age`` / ``belief_disagreement`` above average over those
    frozen stores, so they describe a store no decision consumes.

    Measured, seed 3001, symmetric, 400 steps, one blue lost: mean belief age
    31.70 steps over all stores against **22.40** over live ones -- a 9.3-step
    (0.93 s) inflation from a single dead agent out of four. The bias is not
    symmetric between conditions either: the arm that loses more defenders
    accumulates more frozen stores, and Exp 0 measured the local arm losing
    significantly more (attrition -0.046 [-0.069, -0.024]).

    The original keys are kept unchanged so earlier runs still reproduce; these
    are added beside them and are what the communication conditions reports.

    **The decomposition.** Raw age conflates two different things:

    * *sensing* staleness -- nobody has seen the contact recently. Present in
      every belief mode, including ``shared``, and it dominates: ~22 steps at
      400-step episodes.
    * *protocol* staleness -- somebody has a fresher sighting and it has not
      arrived. This is the only part communication controls, and it is the
      quantity the communication conditions is actually sweeping.

    ``belief_age_excess`` isolates the second: for each (live agent, contact)
    pair it believes in, how much older its record is than the freshest record
    of that contact anywhere in the live team. It is **0.00 by construction
    under ``shared``** -- verified on every seed tried, and pinned by
    ``tests/test_belief_mediators.py`` -- so it does not merely correlate with
    the manipulation, it measures it.

    ``contact_coverage_deficit`` is the complement: the fraction of (live agent,
    contact-known-to-someone) pairs where that agent holds **no** record at all.
    Excess measures how stale what you have is; deficit measures what you do not
    have. Together they are the information gap, and reporting only the first
    would score an agent that knows nothing as perfectly fresh.
    """
    ages, excesses, disagreements = [], [], []
    covered, total_pairs = 0, 0
    contacts_believed = []
    for step in plan_steps:
        dead = {i for i, is_dead in enumerate(step["blue_disabled"]) if is_dead}
        live = {
            agent: per
            for agent, per in step["beliefs"].items()
            if int(agent) not in dead
        }
        if not live:
            continue
        # Freshest record of each contact anywhere in the live team.
        best = {}
        for per in live.values():
            for contact, age in per.items():
                best[contact] = min(best.get(contact, age), age)
        for per in live.values():
            contacts_believed.append(len(per))
            for contact in best:
                total_pairs += 1
                if contact in per:
                    covered += 1
                    ages.append(per[contact])
                    excesses.append(per[contact] - best[contact])
        sets = [set(per) for per in live.values()]
        for index, a in enumerate(sets):
            for b in sets[index + 1:]:
                union = a | b
                disagreements.append(
                    0.0 if not union else 1.0 - len(a & b) / len(union)
                )

    mean = lambda v: float(np.mean(v)) if v else float("nan")  # noqa: E731
    return {
        "live_mean_belief_age": mean(ages),
        "live_p90_belief_age": (
            float(np.percentile(ages, 90)) if ages else float("nan")
        ),
        "live_belief_disagreement": mean(disagreements),
        "live_mean_contacts_believed": mean(contacts_believed),
        "belief_age_excess": mean(excesses),
        "contact_coverage_deficit": (
            1.0 - covered / total_pairs if total_pairs else float("nan")
        ),
    }


def assignment_difference(sim_a, sim_b):
    """How much the two arms' *decisions* differ, at paired planning instants.

    Roles only. Hamming over agents, averaged across epochs, plus the fraction
    of epochs in which the arms assign anything differently at all.
    """
    a = {s["step"]: s["blue_roles"] for s in _planning_steps(sim_a)}
    b = {s["step"]: s["blue_roles"] for s in _planning_steps(sim_b)}
    common = sorted(set(a) & set(b))
    if not common:
        return {"role_hamming": float("nan"), "epochs_differing": float("nan")}
    hammings, differing = [], 0
    for step in common:
        agents = set(a[step]) | set(b[step])
        h = sum(1 for k in agents if a[step].get(k) != b[step].get(k))
        hammings.append(h)
        differing += h > 0
    return {
        "role_hamming": float(np.mean(hammings)),
        "epochs_differing": differing / len(common),
        "n_paired_epochs": len(common),
    }


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------


def paired_binary(shared_vals, local_vals):
    """Exact McNemar / sign test on a paired binary outcome."""
    from scipy.stats import binomtest

    s_only = sum(1 for s, l in zip(shared_vals, local_vals) if s and not l)
    l_only = sum(1 for s, l in zip(shared_vals, local_vals) if l and not s)
    n_disc = s_only + l_only
    p = binomtest(s_only, n_disc, 0.5).pvalue if n_disc else 1.0
    return {
        "shared_rate": float(np.mean(shared_vals)),
        "local_rate": float(np.mean(local_vals)),
        "discordant_favouring_shared": l_only,   # local had it, shared did not
        "discordant_favouring_local": s_only,
        "n_discordant": n_disc,
        "exact_p": float(p),
    }


def paired_continuous(shared_vals, local_vals):
    """Paired difference with a bootstrap CI, plus Wilcoxon signed-rank.

    Bootstrap rather than a t-interval because several of these metrics are
    bounded rates with skewed distributions at this sample size.
    """
    from scipy.stats import wilcoxon

    s = np.asarray(shared_vals, dtype=float)
    l = np.asarray(local_vals, dtype=float)
    ok = ~(np.isnan(s) | np.isnan(l))
    s, l = s[ok], l[ok]
    if s.size == 0:
        return {}
    d = s - l
    rng = np.random.default_rng(0)
    boot = np.array(
        [rng.choice(d, size=d.size, replace=True).mean() for _ in range(5000)]
    )
    try:
        p = float(wilcoxon(s, l).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    return {
        "shared_mean": float(s.mean()),
        "local_mean": float(l.mean()),
        "mean_diff": float(d.mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "wilcoxon_p": p,
        "n": int(s.size),
    }


# ---------------------------------------------------------------------------


BINARY_METRICS = ["touchdown"]
CONTINUOUS_METRICS = [
    "num_neutralized",
    "num_escaped",
    "blue_score",
    "n_blues_disabled",
    "attrition_rate",
    "union_detect_fraction",
    "individual_detect_fraction",
    "information_bracket",
    "mean_belief_age",
    "p90_belief_age",
    "mean_contacts_believed",
    "belief_disagreement",
    # Restricted to live blues, and the protocol-attributable decomposition.
    # See _live_belief_metrics: the four keys above average over the frozen
    # stores of disabled blues, which no decision path ever reads.
    "live_mean_belief_age",
    "live_p90_belief_age",
    "live_mean_contacts_believed",
    "live_belief_disagreement",
    "belief_age_excess",
    "contact_coverage_deficit",
    "duplicate_assignment_rate",
    "unassigned_contact_rate",
    "cbf_du_p50",
    "cbf_du_p90",
    "cbf_intervention_fraction",
    "n_planning_epochs",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--first-seed", type=int, default=1)
    ap.add_argument("--blues", type=int, default=4)
    ap.add_argument("--reds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--planning-interval-seconds", type=float, default=5.0)
    ap.add_argument("--neutralize-fraction", type=float, default=0.5)
    ap.add_argument("--recon-fraction", type=float, default=0.5)
    ap.add_argument("--rule", default="nearest", choices=sorted(RULES),
                    help="allocation rule: 'nearest' is the baseline every "
                         "equivalence test is attached to; 'dedup' is the "
                         "step-3 condition, run in both arms")
    ap.add_argument("--family", default="symmetric", choices=SCENARIO_FAMILIES,
                    help="scenario geometry family; "
                         "selects the spawn geometry and is recorded in the output")
    ap.add_argument("--out", default=None, help="write the full result JSON here")
    args = ap.parse_args()

    cfg = {
        "blues": args.blues,
        "reds": args.reds,
        "steps": args.steps,
        "planning_interval_seconds": args.planning_interval_seconds,
        "neutralize_fraction": args.neutralize_fraction,
        "recon_fraction": args.recon_fraction,
        "family": args.family,
        "rule": args.rule,
    }
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))

    per_arm = {arm: [] for arm in ARMS}
    pairwise = []
    for k, seed in enumerate(seeds, 1):
        sims = {}
        for arm in ARMS:
            sim = run_episode(seed, arm, cfg)
            sims[arm] = sim
            per_arm[arm].append(episode_metrics(sim))
        pairwise.append(assignment_difference(sims["shared"], sims["local"]))
        print(f"  seed {seed} ({k}/{len(seeds)}) done", flush=True)

    results = {"config": cfg, "seeds": seeds, "per_arm": per_arm,
               "pairwise": pairwise, "tests": {}}

    for name in BINARY_METRICS:
        results["tests"][name] = paired_binary(
            [m[name] for m in per_arm["shared"]],
            [m[name] for m in per_arm["local"]],
        )
    for name in CONTINUOUS_METRICS:
        results["tests"][name] = paired_continuous(
            [m[name] for m in per_arm["shared"]],
            [m[name] for m in per_arm["local"]],
        )

    _report(results, pairwise)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2, default=float)
        print(f"\nfull results -> {args.out}")
    return results


def _report(results, pairwise):
    cfg = results["config"]
    print(
        f"\nExperiment 0 pilot -- family={cfg['family']} "
        f"{cfg['blues']} blues / {cfg['reds']} reds / {cfg['steps']} steps, "
        f"{len(results['seeds'])} paired seeds\n"
    )

    t = results["tests"]["touchdown"]
    print("MISSION")
    print(
        f"  breach rate         shared {t['shared_rate']:.2f}   "
        f"local {t['local_rate']:.2f}   "
        f"discordant {t['n_discordant']} "
        f"({t['discordant_favouring_shared']} favour shared, "
        f"{t['discordant_favouring_local']} favour local)  exact p={t['exact_p']:.3f}"
    )

    def line(label, key, fmt="{:.3f}"):
        r = results["tests"].get(key)
        if not r:
            return
        lo, hi = r["ci95"]
        star = "*" if (lo > 0) or (hi < 0) else " "
        print(
            f"  {label:<22s}shared {fmt.format(r['shared_mean'])}   "
            f"local {fmt.format(r['local_mean'])}   "
            f"diff {fmt.format(r['mean_diff'])} "
            f"[{fmt.format(lo)}, {fmt.format(hi)}]{star} p={r['wilcoxon_p']:.3f}"
        )

    # The escape-scoring rule: denial is the mission. blue_score is reported beside breach rate, not
    # in place of it, because the two genuinely move in opposite directions.
    for label, key in [("escaped", "num_escaped"),
                       ("blue_score", "blue_score")]:
        line(label, key)

    # Reported separately from MISSION, and deliberately so. blue_score credits
    # +1 per kill and charges nothing for the defender spent, so it favours
    # whichever arm trades more; this section is the other side of that ledger.
    print("\nATTRITION (the cost blue_score does not charge for)")
    for label, key in [("blues lost", "n_blues_disabled"),
                       ("attrition rate", "attrition_rate"),
                       ("reds neutralized", "num_neutralized")]:
        line(label, key)
    print(
        "  note: blues lost == reds neutralized is an IDENTITY "
        "(_resolve_red_neutralized\n"
        "        is the only site setting blue_disabled, one per kill), so these "
        "are one\n        event counted twice and cannot corroborate each other. "
        "Exchange ratio is\n        identically 1.0 and is not reported."
    )

    print("\nDETECTION AVAILABILITY")
    for label, key in [("union detect frac", "union_detect_fraction"),
                       ("individual detect", "individual_detect_fraction"),
                       ("information bracket", "information_bracket")]:
        line(label, key)

    print("\nINFORMATION (the mediator)")
    for label, key in [("mean belief age", "mean_belief_age"),
                       ("p90 belief age", "p90_belief_age"),
                       ("contacts believed", "mean_contacts_believed"),
                       ("belief disagreement", "belief_disagreement")]:
        line(label, key, "{:.2f}")

    print("\nCOORDINATION")
    for label, key in [("duplicate assign", "duplicate_assignment_rate"),
                       ("unassigned contact", "unassigned_contact_rate")]:
        line(label, key)
    line("decisions/episode", "n_planning_epochs", "{:.1f}")
    valid = [p for p in pairwise if not np.isnan(p.get("role_hamming", np.nan))]
    if valid:
        print(
            f"  {'role hamming':<22s}mean {np.mean([p['role_hamming'] for p in valid]):.2f} "
            f"agents/epoch;  epochs differing "
            f"{np.mean([p['epochs_differing'] for p in valid]):.2f}"
        )

    print("\nSAFETY (held fixed -- differences here are consequences, not causes)")
    for label, key in [("CBF |du| p50", "cbf_du_p50"),
                       ("CBF |du| p90", "cbf_du_p90"),
                       ("intervention frac", "cbf_intervention_fraction")]:
        line(label, key)

    print("\n  * = 95% bootstrap CI on the paired difference excludes zero")


if __name__ == "__main__":
    main()
