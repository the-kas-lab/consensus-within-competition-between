"""Episode builder and batch runner for the competing-teams experiments.

Arms: the consensus team (``cbaa``), the independent team (``independent``), the
commander team (``commander``), and the consensus team at high authority
(``cbaa_sw2``). Intruders {2, 4} x families {symmetric, multi_bearing} x
communication conditions (REF = ideal network, MH = limited range, LI =
independent loss, LB = burst loss, plus REF+attack = falsified reports) x seeds.
Each attacked episode is PAIRED with the unattacked REF episode of the same arm,
intruder count, family and seed (``pairs``).

Every arm runs the same configuration (experiment_policy: pursuit, safety,
intruder information, three-task model, unavailable-target execution) with gate
intruders; the commander arm takes the task-model commander over the same
network. Backends: ``rule`` (free: default bids, NearestCommanderDecider,
GateRuleIntruderPlanner), ``profile`` (LLMs from an inference profile; paid) and
``replay_profile`` (replay-only from the cache: a miss stops the episode).

    python3 -m scripts.run_competing_teams manifest --seeds 4001 --out m.jsonl \\
        --conditions REF --families multi_bearing
    python3 -m scripts.run_competing_teams run --manifest m.jsonl --out r.jsonl --backend rule
    python3 -m scripts.run_competing_teams pairs --manifest m.jsonl --results r.jsonl
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARMS = ("cbaa", "independent", "commander")
# Opt-in arm, never in the default manifest: the consensus team at high
# authority, whose defenders write the coefficients of their own bid formula.
STRATEGY_ARMS = ("cbaa_sw2",)
INTRUDERS = (2, 4)
FAMILIES = ("symmetric", "multi_bearing")
CONDITIONS = ("REF", "MH", "LI", "LB")
ATTACK_CONDITION = "REF"


# ---------------------------------------------------------------------------
# manifest and pairs
# ---------------------------------------------------------------------------


def episode_id(e):
    """Unique per prespecified execution. ``rep`` (optional) separates fresh
    deployment repetitions of the same cell and seed; absent for the outcome
    matrix, so its ids are unchanged."""
    return (f"{e['arm']}|r{e['reds']}|{e['family']}|{e['condition']}"
            f"{'+attack' if e['attacked'] else ''}|s{e['seed']}"
            f"{'|rep' + str(e['rep']) if e.get('rep') is not None else ''}"
            f"{'|design-' + str(e['design']) if e.get('design') is not None else ''}"
            f"{'|decoys-' + str(e['decoy_tag']) if e.get('decoy_tag') else ''}"
            f"{'|imgmode-' + str(e['image_mode']) if e.get('image_mode') else ''}")


def build_manifest(seeds, arms=ARMS, intruders=INTRUDERS, families=FAMILIES,
                   conditions=CONDITIONS, attack=True):
    out = []
    for arm in arms:
        for reds in intruders:
            for family in families:
                for seed in seeds:
                    for condition in conditions:
                        out.append({"arm": arm, "reds": reds, "family": family,
                                    "condition": condition, "attacked": False, "seed": seed})
                    if attack:
                        out.append({"arm": arm, "reds": reds, "family": family,
                                    "condition": ATTACK_CONDITION, "attacked": True, "seed": seed})
    for e in out:
        e["episode_id"] = episode_id(e)
    return out


def prespecified_pairs(manifest):
    """Every attacked episode with its unattacked reference (same arm,
    intruders, family, seed, ATTACK_CONDITION). The reference must be in the
    manifest: a pair whose reference was not prespecified is an error."""
    ids = {e["episode_id"] for e in manifest}
    pairs = []
    for e in manifest:
        if not e["attacked"]:
            continue
        ref = episode_id(dict(e, attacked=False))
        if ref not in ids:
            raise ValueError(f"attacked episode {e['episode_id']} has no prespecified reference")
        pairs.append({"arm": e["arm"], "reds": e["reds"], "family": e["family"],
                      "seed": e["seed"], "attacked": e["episode_id"], "reference": ref})
    return pairs


def terminal_problem(rec):
    """None when a record ends in a valid terminal state -- breach, every red
    resolved, or the horizon reached, with no solver failure -- else why not.
    Records written before ``horizon_steps`` existed assume the 400-step
    horizon every competing-teams run used."""
    o = rec.get("outcome")
    if not o:
        return "no_outcome"
    if rec.get("solver_failed") or o.get("solver_failed"):
        return "solver_failed"
    if o.get("touchdown"):
        return None
    if o.get("red_outcomes") and all(x != "active" for x in o["red_outcomes"]):
        return None
    if rec.get("steps_run", -1) >= rec.get("horizon_steps", 400):
        return None
    return "not_terminal"


# The fields every record copies from its manifest entry (episode_record).
RECORD_MANIFEST_KEYS = ("arm", "reds", "family", "condition", "attacked", "seed", "rep", "design",
                        "decoy_tag", "image_mode")


def record_problem(entry, records, identity_hash=None):
    """None when the manifest entry has exactly one record, that record
    matches the entry's fields and (when given) the expected execution
    identity, and it ends in a valid terminal state; else why not
    ("missing", "duplicate", "manifest_mismatch", "identity_mismatch" or a
    ``terminal_problem``)."""
    if not records:
        return "missing"
    if len(records) > 1:
        return "duplicate"
    rec = records[0]
    if any(rec.get(k) != entry.get(k) for k in RECORD_MANIFEST_KEYS):
        return "manifest_mismatch"
    if identity_hash is not None and rec.get("identity_hash") != identity_hash:
        return "identity_mismatch"
    return terminal_problem(rec)


def records_by_id(results):
    by_id = defaultdict(list)
    for r in results:
        by_id[r["episode_id"]].append(r)
    return by_id


def manifest_status(manifest, results, identity_hash=None):
    """Every prespecified execution's ``record_problem`` (None = validly
    completed), plus records that belong to no manifest entry."""
    by_id = records_by_id(results)
    ids = {e["episode_id"] for e in manifest}
    return {"problems": {e["episode_id"]: record_problem(e, by_id.get(e["episode_id"]), identity_hash)
                         for e in manifest},
            "extra_records": sorted(i for i in by_id if i not in ids)}


def pair_table(manifest, results, identity_hash=None):
    """One row per prespecified pair, none dropped. "complete" only when each
    half has exactly one record that matches its manifest entry, both halves
    share one execution identity (the expected one when given), and both end
    in a valid terminal state (``record_problem``); exposure fields come from
    the attacked episode's attack log."""
    by_id = records_by_id(results)
    entries = {e["episode_id"]: e for e in manifest}
    rows = []
    for p in prespecified_pairs(manifest):
        ra, rr = by_id.get(p["attacked"], []), by_id.get(p["reference"], [])
        bad_a = record_problem(entries[p["attacked"]], ra, identity_hash)
        bad_r = record_problem(entries[p["reference"]], rr, identity_hash)
        if not (bad_a or bad_r) and ra[0].get("identity_hash") != rr[0].get("identity_hash"):
            bad_a = bad_r = "identity_mismatch"
        a = ra[0] if len(ra) == 1 else None
        r = rr[0] if len(rr) == 1 else None
        status = ("missing_both" if not (ra or rr) else "missing_attacked" if not ra else
                  "missing_reference" if not rr else
                  "complete" if not (bad_a or bad_r) else
                  "invalid_both" if bad_a and bad_r else
                  "invalid_attacked" if bad_a else "invalid_reference")
        exp = (a or {}).get("attack") or {}
        rows.append(dict(p, status=status, attacked_problem=bad_a, reference_problem=bad_r,
                         eligible_report=exp.get("target_red") is not None,
                         accepted_falsification=bool(exp.get("n_accepted")),
                         any_commitment=bool(exp.get("any_commitment")),
                         attacked_outcome=(a or {}).get("outcome"),
                         reference_outcome=(r or {}).get("outcome")))
    return rows


OUTCOME_FIELDS = {
    "breach": lambda o: float(bool(o["touchdown"])),
    "attrition_rate": lambda o: float(o["attrition_rate"]),
    "blue_score": lambda o: float(o["blue_score"]),
}


def paired_differences(rows):
    """attacked - reference per complete pair, by (arm, reds), over every
    complete pair (exposed or not), with the exposure counts beside them. The
    statistical design (intervals, multiplicity) is frozen separately."""
    cells = defaultdict(list)
    for row in rows:
        cells[(row["arm"], row["reds"])].append(row)
    out = {}
    for (arm, reds), rs in sorted(cells.items()):
        done = [r for r in rs if r["status"] == "complete"]
        cell = {"prespecified": len(rs), "complete": len(done),
                "no_eligible_report": sum(not r["eligible_report"] for r in done),
                "no_accepted_falsification": sum(not r["accepted_falsification"] for r in done),
                "with_commitment": sum(r["any_commitment"] for r in done)}
        for name, f in OUTCOME_FIELDS.items():
            d = [f(r["attacked_outcome"]) - f(r["reference_outcome"]) for r in done]
            cell[f"mean_diff_{name}"] = sum(d) / len(d) if d else None
        out[f"{arm}|r{reds}"] = cell
    return out


# ---------------------------------------------------------------------------
# one episode
# ---------------------------------------------------------------------------


def _profile_kwargs(profile, interface, args):
    """Planner kwargs for one interface from an InferenceProfile: the
    episode's own cache (namespaced by execution identity), one shared client
    and the shared ledger per process."""
    from planning.response_cache import ResponseCache

    replay = args.get("replay", False)
    cache = args.get("_cache")
    if cache is None:
        cache = args["_cache"] = ResponseCache(path=args["cache_dir"])
    client = None
    ledger = None
    if not replay:
        client = args.get("_client")
        if client is None:
            client = args["_client"] = profile.build_client()
        if profile.backend == "openrouter":
            ledger = args.get("_ledger")
            if ledger is None:
                from planning.spend import SpendLedger

                ledger = args["_ledger"] = SpendLedger(args["cap_usd"], profile.ledger_price(),
                                                       path=args["ledger"])
    return profile.planner_kwargs(interface, cache, client=client, ledger=ledger,
                                  replay_only=replay)


def execution_identity(backend, profile=None, study_version=None, designs=(), arms=(),
                       decoys=False, image_modes=()):
    """Everything that defines how an execution runs, apart from the episode
    cell itself: code, backend, inference profile, interface versions, the
    policy (safety version included) and the study version. A resume whose
    identity differs is refused (cmd_run)."""
    import hashlib

    from scripts import experiment_policy as ep
    from scripts.inference_profile import INTERFACE_VERSIONS, code_version

    ident = {
        "study_version": study_version,
        "code": code_version(),
        "backend": backend,
        "profile": None if profile is None else profile.as_dict(),
        "profile_hash": None if profile is None else profile.profile_hash(),
        "interfaces": INTERFACE_VERSIONS if profile is not None else "rule",
        "safety_version": ep.NEW_EXPERIMENT_SAFETY_VERSION,
        "policy": {"common": ep.NEW_EXPERIMENT_COMMON, "tasks": ep.NEW_EXPERIMENT_TASKS,
                   "falsified_reports": ep.NEW_EXPERIMENT_FALSIFIED_REPORTS},
    }
    designs = sorted({d for d in designs if d is not None})
    if designs:
        ident["design_interfaces"] = ({d: INTERFACE_VERSIONS for d in designs}
                                      if profile is not None else "rule")
        ident["design_policy"] = {d: ep.DESIGNS[d] for d in designs}
    if any(a in STRATEGY_ARMS for a in arms):
        from coordination.strategy_bridge import PROMPT_VERSION, SCHEMA
        from coordination.strategy_weights import spec_identity

        ident["strategy_arms"] = {"arms": sorted(set(arms) & set(STRATEGY_ARMS)),
                                  "spec": spec_identity(),
                                  "bridge": {"prompt_version": PROMPT_VERSION, "schema": SCHEMA}}
    if decoys:
        # Recorded only for runs with birds and civilian drones: the image
        # mode, the contact brief and the motion constants.
        from planning.local_llm import DECOY_BRIEF_V1, IMAGE_MODE_VERSION, IMAGE_RESERVE_BYTES
        from simulation.testbed import Simulation

        ident["decoys"] = {
            "image_mode": IMAGE_MODE_VERSION, "brief": DECOY_BRIEF_V1,
            "image_reserve_bytes": IMAGE_RESERVE_BYTES,
            "motion": {k: getattr(Simulation, k) for k in sorted(dir(Simulation))
                       if k.startswith("DECOY_")},
        }
        extra = sorted(set(image_modes) - {IMAGE_MODE_VERSION})
        if extra:
            # Recorded only when a manifest uses a non-default image mode:
            # each such mode's brief.
            from planning.local_llm import DECOY_BRIEFS

            ident["decoys"]["image_modes"] = {m: DECOY_BRIEFS[m] for m in extra}
    ident = json.loads(json.dumps(ident, sort_keys=True, default=str))
    ident["identity_hash"] = hashlib.sha256(
        json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
    return ident


def _condition_cfg(family, condition, exp5_arm):
    from scripts import network_conditions

    return network_conditions.episode_cfg(family, condition, exp5_arm)


def _build_strategy_arm(e, backend, llm_args, profile, common, planners):
    """The consensus team at high authority: the CBAA arm's episode
    configuration and the run's intruders, with each defender's bid weights
    written by its LLM."""
    from scripts.experiment_policy import NEW_EXPERIMENT_TASKS
    from scripts.episode_builder import build_episode

    belief, cfg = _condition_cfg(e["family"], e["condition"], "C")
    cfg = dict(cfg, **common, **NEW_EXPERIMENT_TASKS)
    src = None
    if profile is not None:
        from coordination.strategy_bridge import StrategyWeightSource
        from planning.prompt_builder import PromptContext

        prof, args = _serving(e, "defender", profile, llm_args)
        kw = _profile_kwargs(prof, "defender", args)
        base = kw.pop("endpoint")
        src = StrategyWeightSource(PromptContext(arena_size=0.0), base_url=base,
                                   max_workers=llm_args.get("agent_workers", 4),
                                   build_client_on_demand=False, **kw)
    elif backend != "rule":
        raise ValueError(f"{e['arm']} needs a profile backend: its defenders are LLMs")
    if src is not None:
        cfg["cbaa_modifier_source"] = src
        planners["defenders"] = src
    sim = build_episode(e["seed"], belief, cfg)
    if sim.cbaa_bid_version != "v2.3":
        raise ValueError(f"{e['arm']} needs bid version v2.3, got {sim.cbaa_bid_version!r}")
    if src is not None:
        from planning.prompt_builder import PromptContext

        src.context = PromptContext.from_simulation(sim)
    return sim


def _serving(e, interface, profile, llm_args):
    """The (profile, args) that serve one interface of this episode: the run's
    own, or the near-frontier model's when the entry names this interface
    (``e["near_frontier"]`` is ``"defender"`` or ``"commander"``). The
    near-frontier model has its own cache next to the episode's."""
    nf = (llm_args or {}).get("near_frontier")
    if nf is not None and e.get("near_frontier") == interface:
        args = dict(nf)
        args.setdefault("cache_dir", llm_args["cache_dir"] + "__near_frontier")
        args["replay"] = llm_args.get("replay", False)
        return nf["profile_obj"], args
    return profile, llm_args


def build(e, backend="rule", llm_args=None):
    """The episode's Simulation (built, not run) and its planner objects."""
    from coordination.commander import CommanderTaskPlanner, NearestCommanderDecider
    from planning.intruder import GateRuleIntruderPlanner
    from scripts.experiment_policy import (DESIGNS, NEW_EXPERIMENT_COMMON,
                                           NEW_EXPERIMENT_FALSIFIED_REPORTS,
                                           NEW_EXPERIMENT_TASKS)
    from scripts.episode_builder import build_episode

    planners = {}
    profile = (llm_args or {}).get("profile_obj")
    if backend != "rule" and profile is None:
        raise ValueError("an LLM backend needs an inference profile (llm_args['profile_obj'])")
    if backend == "rule":
        red = GateRuleIntruderPlanner()
    else:
        from planning.intruder_llm import LLMIntruderTeam

        kw = _profile_kwargs(profile, "intruder", llm_args)
        red = LLMIntruderTeam(max_workers=llm_args.get("agent_workers", 4), **kw)
        planners["intruders"] = red
    common = dict(NEW_EXPERIMENT_COMMON, intruder_team="gates", red_planner=red,
                  reds=e["reds"])
    if e.get("decoys") is not None:
        common["decoys"] = e["decoys"]
    if e.get("design") is not None:
        if e["design"] not in DESIGNS:
            raise ValueError(f"unknown design {e['design']!r}")
        common.update(DESIGNS[e["design"]])
    if e["attacked"]:
        common["falsified_reports"] = dict(NEW_EXPERIMENT_FALSIFIED_REPORTS)
    if e["arm"] in ("cbaa", "independent"):
        belief, cfg = _condition_cfg(e["family"], e["condition"],
                                     "C" if e["arm"] == "cbaa" else "I")
        cfg = dict(cfg, **common, **NEW_EXPERIMENT_TASKS)
        if backend != "rule":
            from coordination.llm_bridge import LLMModifierSource
            from planning.prompt_builder import PromptContext

            prof, args = _serving(e, "defender", profile, llm_args)
            kw = _profile_kwargs(prof, "defender", args)
            base = kw.pop("endpoint")
            src = LLMModifierSource(PromptContext(arena_size=0.0), base_url=base,
                                    task_set="three_task",
                                    selection=("independent" if e["arm"] == "independent"
                                               else "cbaa"),
                                    max_workers=llm_args.get("agent_workers", 4),
                                    build_client_on_demand=False, **kw)
            cfg["cbaa_modifier_source"] = src
            planners["defenders"] = src
        sim = build_episode(e["seed"], belief, cfg)
        if "defenders" in planners:
            planners["defenders"].context = PromptContext.from_simulation(sim)
    elif e["arm"] in STRATEGY_ARMS:
        sim = _build_strategy_arm(e, backend, llm_args, profile, common, planners)
    elif e["arm"] != "commander":
        raise ValueError(f"unknown arm {e['arm']!r}")
    else:
        belief, cfg = _condition_cfg(e["family"], e["condition"], "C")
        cfg = {k: v for k, v in cfg.items() if not k.startswith("cbaa_") and k != "allocation"}
        if backend == "rule":
            decider = NearestCommanderDecider()
        else:
            from coordination.commander_llm import LLMCommanderDecider

            prof, args = _serving(e, "commander", profile, llm_args)
            decider = LLMCommanderDecider(**_profile_kwargs(prof, "commander", args))
            planners["commander"] = decider
        planner = CommanderTaskPlanner(decider)
        planners["commander_planner"] = planner
        cfg.update(common, per_agent_planning=False, centralized_from_belief=True,
                   deliver_assignments_over_network=True, cbaa_task_set="three_task",
                   commander_planner=planner)
        sim = build_episode(e["seed"], belief, cfg)
        if backend != "rule":
            from coordination.commander_llm import commander_geometry
            from coordination.tasks import TaskContext

            decider.geometry = commander_geometry(
                sim, TaskContext.from_simulation(sim, slots=sim.commander_slots,
                                                 task_set="three_task"))
    if e.get("decoys") is not None and sim.contact_image_paths:
        # Every LLM defender interface sees one image per believed contact, in
        # the entry's image mode (absent = img.1).
        lookup = sim.contact_image_paths.get
        mode = {"mode": e["image_mode"]} if e.get("image_mode") else {}
        for key in ("defenders", "commander"):
            if key in planners and hasattr(planners[key], "enable_images"):
                planners[key].enable_images(lookup, **mode)
    if "intruders" in planners:
        # The intruder prompt states the episode length: the built episode's own.
        planners["intruders"].set_episode_horizon(sim.sim_steps, sim.dt)
    if backend != "rule":
        from scripts.inference_profile import check_heading_prompt_geometry

        check_heading_prompt_geometry(sim)
    return sim, planners


def _usage(cache_totals):
    return {k: cache_totals.get(k, 0) for k in ("calls", "prompt_tokens", "completion_tokens",
                                                "transport_failures")}


def _cost_from_cache(planner_objs):
    """Billed cost this process wrote, summed from the cache records' usage."""
    total, seen = 0.0, set()
    for p in planner_objs:
        cache = getattr(p, "cache", None)
        if cache is None or id(cache) in seen:
            continue
        seen.add(id(cache))
        with cache._lock:
            for rec in cache._memory.values():
                total += float((rec.get("usage") or {}).get("cost_usd") or 0.0)
    return total


def episode_record(e, backend="rule", llm_args=None):
    from scripts.episode_builder import _Quiet

    if llm_args and (llm_args.get("cache_per_episode") or llm_args.get("profile_obj") is not None):
        # One cache directory per execution: no answer cached by another
        # episode (e.g. another arm in an identical early state, or another
        # repetition) is reused, so every call is live and the runtime is the
        # episode's own. A profile run always does this, under its identity
        # namespace; a re-run of the SAME execution (recovery) reuses only
        # its own records, and says so (cache_hits).
        slug = e["episode_id"].replace("|", "_").replace("+", "_")
        root = llm_args["cache_dir"]
        if llm_args.get("identity_hash"):
            root = os.path.join(root, llm_args["identity_hash"])
        llm_args = {k: v for k, v in llm_args.items() if not k.startswith("_")}
        llm_args["cache_dir"] = os.path.join(root, slug)

    from simulation.recorder import config_hash, describe_config

    sim, planners = build(e, backend, llm_args)
    t_start = time.time()
    t0 = time.monotonic()
    with _Quiet():
        sim.run()
    wall = time.monotonic() - t0
    rec = {"episode_id": e["episode_id"], **{k: e[k] for k in
                                            ("arm", "reds", "family", "condition", "attacked", "seed")},
           "rep": e.get("rep"), "backend": backend,
           "identity_hash": (llm_args or {}).get("identity_hash"),
           "scenario_config_hash": config_hash(describe_config(sim._init_args)),
           "started_unix": round(t_start, 3), "finished_unix": round(time.time(), 3),
           "wall_s": round(wall, 2), "steps_run": len(sim.history_blues) - 1,
           "horizon_steps": int(sim.sim_steps),
           "design": e.get("design"),
           **({"decoy_tag": e["decoy_tag"]} if e.get("decoy_tag") else {}),
           **({"image_mode": e["image_mode"]} if e.get("image_mode") else {}),
           "outcome": sim.episode_outcome(),
           "solver_failed": bool(sim.solver_failed), "solver_failure_step": sim.solver_failure_step,
           "resolution_events": list(getattr(sim, "resolution_log", []))}
    # Defender LLM (CBAA / independent): every drop and total failure, by kind.
    src = planners.get("defenders")
    defender_events = Counter(ev["kind"] for ev in sim.planner_fallback_events())
    rec["defender"] = {"events_by_kind": dict(defender_events)}
    if src is not None:
        d = src.diagnostics()
        rec["defender"].update(calls=d["calls"], cache_hits=d["cache_hits"],
                               epochs_x_agents=d["epochs_x_agents"],
                               defaulted_after_total_failure=d["defaulted_after_total_failure"],
                               transport_failures=d["transport_failures"],
                               usage=_usage(d["usage"]))
    # Intruders: decisions, invalid outputs by reason, fallbacks by kind.
    decisions = [x for x in sim.intruder_log if "decision" in x]
    rec["intruder"] = {
        "decisions": len(decisions),
        "invalid_by_reason": dict(Counter((x.get("planner_reason") or x["invalid_reason"]).split(":")[0]
                                          for x in decisions if x["invalid_reason"])),
        "fallback_by_kind": dict(Counter(x["fallback"] for x in decisions if x["fallback"])),
        "actions": dict(Counter(x["decision"].role for x in decisions)),
        "phase_changes": sum(1 for x in sim.intruder_log if x.get("event") == "phase"),
    }
    if "intruders" in planners:
        rec["intruder"].update(planners["intruders"].diagnostics())
    # Commander: per-epoch validation reports.
    cp = planners.get("commander_planner")
    if cp is not None:
        reps = cp.reports
        rec["commander"] = {
            "epochs": len(reps),
            "unusable_epochs": sum(1 for r in reps if r["response"] is None),
            "epochs_with_agent_fallback": sum(1 for r in reps if r["fallback"]),
            "agent_failures_by_reason": dict(Counter(f[1] for r in reps for f in r["failures"])),
            "fallback_conflicts": sum(len(r["fallback_conflicts"]) for r in reps),
            "pending_events": dict(Counter(x["event"] for x in sim.commander_pending_log)),
        }
        dec = planners.get("commander")
        if dec is not None:
            rec["commander"].update(calls=dec.calls, cache_hits=dec.cache_hits,
                                    transport_failures=dec.transport_failures)
    if e["arm"] in STRATEGY_ARMS and src is not None:
        rec["strategy_log"] = list(src.modifier_log)
    rec["unavailable_target_events"] = dict(Counter(x["event"] for x in sim.unavailable_target_log))
    rec["red_red_separation"] = getattr(sim, "red_red_separation", None)
    # bid v2.3: agent-epochs whose bids carried continuity relief
    rt = getattr(sim, "cbaa_runtime", None)
    if rt is not None and getattr(sim, "cbaa_bid_version", None) == "v2.3":
        rec["continuity_relief_log"] = [list(x) for x in rt.relief_log]
    if sim.attack_log is not None:
        log = sim.attack_log
        rec["attack"] = {"sender": log["sender"], "target_red": log["target_red"],
                         "target_selected_step": log["target_selected_step"],
                         "reports_altered": log["reports_altered"], "stop_reason": log["stop_reason"],
                         "n_deliveries": len(log["deliveries"]), "n_accepted": len(log["accepted"]),
                         "distinct_observations_falsified": log["distinct_observations_falsified"],
                         "any_commitment": any(v > 0 for v in log["commitment_steps"].values())}
    if backend != "rule":
        rec["cache_hits_total"] = sum(getattr(p, "cache_hits", 0) for p in
                                      ([planners.get("commander")] if planners.get("commander") else [])
                                      + (list(planners["intruders"].planners.values())
                                         if "intruders" in planners else [])
                                      + (list(src.planners.values()) if src is not None else []))
        objs = [p for k, p in planners.items() if k != "commander_planner"]
        objs += list(planners["intruders"].planners.values()) if "intruders" in planners else []
        objs += list(src.planners.values()) if src is not None else []
        rec["cost_usd_written"] = round(_cost_from_cache(objs), 6)
        rec["llm_calls"] = (rec["defender"].get("calls", 0) + rec["intruder"].get("calls", 0)
                            + rec.get("commander", {}).get("calls", 0))
    return json.loads(json.dumps(rec, default=str))


def _job(payload):
    e, backend, llm_args = payload
    return episode_record(e, backend, llm_args)


def cmd_run(args):
    manifest = [json.loads(line) for line in open(args.manifest)]
    profile = None
    if args.backend in ("profile", "replay_profile"):
        from scripts.inference_profile import load_profiles

        profile = load_profiles(args.profile_file)[args.profile_id]
    ident = execution_identity(args.backend.replace("replay_", ""), profile, args.study_version,
                               designs=[e.get("design") for e in manifest],
                               arms=[e["arm"] for e in manifest])
    if args.backend == "profile" and ident["code"]["dirty"] and not args.allow_dirty:
        raise SystemExit("tracked files differ from HEAD: a live profile run records the commit "
                         "as its code identity, so commit first (or --allow-dirty for a smoke)")
    id_path = args.out + ".identity.json"
    if os.path.exists(id_path):
        held = json.load(open(id_path))
        if held["identity_hash"] != ident["identity_hash"]:
            raise SystemExit(f"{args.out} was produced under identity {held['identity_hash']}, "
                             f"not {ident['identity_hash']}: refusing to resume a different "
                             f"configuration (diff the two identities; start a new output)")
    elif os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        raise SystemExit(f"{args.out} exists without an identity file: refusing to resume")
    else:
        with open(id_path, "w") as f:
            json.dump(ident, f, indent=1, sort_keys=True)
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            if r.get("identity_hash", ident["identity_hash"]) != ident["identity_hash"]:
                raise SystemExit(f"record {r['episode_id']} in {args.out} has identity "
                                 f"{r.get('identity_hash')}: refusing to resume")
            done.add(r["episode_id"])
    todo = [e for e in manifest if e["episode_id"] not in done]
    llm_args = None
    if args.backend != "rule":
        llm_args = {"cache_dir": args.cache_dir, "agent_workers": args.agent_workers,
                    "timeout": args.timeout, "max_tokens": args.max_tokens,
                    "cap_usd": args.cap_usd, "ledger": args.ledger,
                    "cache_per_episode": args.cache_per_episode,
                    "identity_hash": ident["identity_hash"]}
        if profile is not None:
            llm_args.update(profile_obj=profile, replay=args.backend == "replay_profile")
    print(f"{len(manifest)} prespecified, {len(done)} already recorded, {len(todo)} to run "
          f"({args.backend}, {args.workers} episode workers)", flush=True)
    t0 = time.monotonic()
    with open(args.out, "a") as sink:
        if args.workers == 1:
            for e in todo:
                rec = episode_record(e, args.backend, llm_args)
                rec["identity_hash"] = ident["identity_hash"]
                sink.write(json.dumps(rec) + "\n")
                sink.flush()
                print(f"  {rec['episode_id']}: {rec['wall_s']} s", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futs = {pool.submit(_job, (e, args.backend, llm_args)): e for e in todo}
                for f in as_completed(futs):
                    rec = f.result()  # StopRun (replay miss, spend cap) stops the batch
                    rec["identity_hash"] = ident["identity_hash"]
                    sink.write(json.dumps(rec) + "\n")
                    sink.flush()
                    print(f"  {rec['episode_id']}: {rec['wall_s']} s", flush=True)
    print(f"batch wall {time.monotonic() - t0:.1f} s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], required=True)
    m.add_argument("--arms", default=",".join(ARMS))
    m.add_argument("--intruders", default="2,4")
    m.add_argument("--families", default=",".join(FAMILIES))
    m.add_argument("--conditions", default=",".join(CONDITIONS))
    m.add_argument("--no-attack", action="store_true")
    m.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--manifest", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--backend", choices=("rule", "profile", "replay_profile"), default="rule")
    r.add_argument("--profile-file", default=None)
    r.add_argument("--profile-id", default=None)
    r.add_argument("--study-version", default=None)
    r.add_argument("--allow-dirty", action="store_true")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--agent-workers", type=int, default=4)
    r.add_argument("--cache-dir", default=None)
    r.add_argument("--cache-per-episode", action="store_true")
    r.add_argument("--timeout", type=float, default=180.0)
    r.add_argument("--max-tokens", type=int, default=2000)
    r.add_argument("--cap-usd", type=float, default=5.0, help="spend cap of a paid profile run")
    r.add_argument("--ledger", default=None, help="spend ledger file of a paid profile run")
    p = sub.add_parser("pairs")
    p.add_argument("--manifest", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "manifest":
        man = build_manifest(args.seeds, args.arms.split(","),
                             [int(x) for x in args.intruders.split(",")],
                             args.families.split(","), args.conditions.split(","),
                             attack=not args.no_attack)
        with open(args.out, "w") as f:
            for e in man:
                f.write(json.dumps(e) + "\n")
        print(f"wrote {len(man)} episodes ({len(prespecified_pairs(man))} attacked/unattacked "
              f"pairs) to {args.out}")
    elif args.cmd == "run":
        if args.backend != "rule" and not args.cache_dir:
            raise SystemExit("an LLM backend needs --cache-dir")
        if args.backend in ("profile", "replay_profile") and not (args.profile_file and args.profile_id):
            raise SystemExit("a profile run needs --profile-file and --profile-id")
        if args.backend == "profile" and not args.ledger:
            from scripts.inference_profile import load_profiles

            if load_profiles(args.profile_file)[args.profile_id].backend == "openrouter":
                raise SystemExit("a paid profile run needs --ledger (and --cap-usd)")
        cmd_run(args)
    elif args.cmd == "pairs":
        man = [json.loads(line) for line in open(args.manifest)]
        res = [json.loads(line) for line in open(args.results)]
        id_path = args.results + ".identity.json"
        ident = json.load(open(id_path))["identity_hash"] if os.path.exists(id_path) else None
        rows = pair_table(man, res, identity_hash=ident)
        out = {"pairs": rows, "summary": paired_differences(rows),
               "status": dict(Counter(r["status"] for r in rows))}
        if args.out:
            json.dump(out, open(args.out, "w"), indent=1)
        print(json.dumps({"status": out["status"], "summary": out["summary"]}, indent=1))


if __name__ == "__main__":
    main()
