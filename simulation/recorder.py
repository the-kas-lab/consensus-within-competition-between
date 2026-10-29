"""Structured per-episode recording: the substrate every later metric reads.

Why this exists
---------------
 asks for five families of metric (mission, information, network,
coordination, safety-filter) and Sec 20 asks for paired statistics over them.
None of that can be computed from what an episode currently leaves behind:
``Simulation`` keeps ``history_blues`` / ``history_red`` / ``history_cbf_slack``
and a handful of scalars, which is enough to draw a GIF and nothing else.

Two specific gaps this closes now, because both are load-bearing later:

* **Requested vs. executed controls.** requires
  ``delta_u_i = ||u_requested - u_executed||`` per agent per step, and reports
  intervention frequency and magnitude by condition. The nominal MPC control was
  previously discarded the instant the CBF filter replaced it, so the
  intervention was unobservable after the fact.
* **``engagement_authorized``.** the engagement-authority rule
  receives exactly one bit per agent, derived from the committed allocation, and
  that the bit is logged every step. This is where that logging lives. It is also
  the evidence for the Sec 23 test that the CBF's inputs are *states plus that
  bit and nothing else*.

Storage
-------
Two files per episode, sharing a stem:

* ``<stem>.json`` -- manifest: config, config hash, seed, RNG mode, outcome,
  derived safety metrics, and the per-step structured (non-array) fields.
* ``<stem>.npz`` -- the float arrays (states, controls, slacks), which are the
  bulk and do not belong in JSON.

Recording is pure observation: it never feeds anything back into the simulation,
so an episode's trajectory is identical whether or not it is being recorded.
``tests/test_golden_episode.py`` is what holds that to account.
"""

import datetime
import hashlib
import json
import os
import subprocess

import numpy as np

SCHEMA_VERSION = 1

# The nominal MPC and the CBF-QP are solved separately, so ||u_req - u_exe||
# is never exactly zero even when the filter is inactive: OSQP's termination
# tolerances put a noise floor somewhere around 1e-3. A single "intervention
# fraction" therefore says as much about the chosen threshold as about the
# filter -- measured on this testbed, the fraction runs 0.90 at 1e-6 and 0.52
# at 1e-2 for the *same* episodes.
#
# So the sweep is the metric, not any one entry in it. INTERVENTION_TOL is kept
# only as the headline number's threshold and is set above the solver noise
# floor deliberately.
INTERVENTION_TOL = 1e-2
INTERVENTION_THRESHOLDS = (1e-6, 1e-3, 1e-2, 0.1, 0.5, 1.0)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def describe_config(init_args):
    """Render ``Simulation.__init__``'s bound arguments as something hashable.

    Planner instances are the awkward part: they are objects, they carry
    configuration that changes the trajectory (``recon_fraction`` and similar),
    and they are not JSON. They are recorded as their class name plus their
    scalar attributes, which is exactly the part that affects behavior.
    """

    def describe(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, np.ndarray):
            return {"__ndarray__": value.tolist()}
        if isinstance(value, (list, tuple)):
            return [describe(v) for v in value]
        if isinstance(value, dict):
            return {str(k): describe(v) for k, v in value.items()}
        # An object -- a planner, most likely. Keep its type and its scalar
        # configuration; drop anything stateful or unserializable.
        scalars = {
            k: v
            for k, v in sorted(vars(value).items())
            if not k.startswith("_") and isinstance(v, (bool, int, float, str, type(None)))
        }
        return {"__type__": type(value).__name__, **scalars}

    return {k: describe(v) for k, v in sorted(init_args.items()) if k != "self"}


def config_hash(config):
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class EpisodeRecorder:
    """Accumulates one episode's structured record.

    Deliberately dumb: it stores what it is handed and derives nothing until
    ``manifest()``. Any interpretation of the data (age of information, belief
    disagreement, allocation churn) belongs in the evaluation layer reading these
    files back, not here -- otherwise the recorder grows a dependency on every
    research question in turn.
    """

    def __init__(self, config, seed, rng_mode, num_blues, num_reds, blue_nx, red_nx, blue_nu):
        self.config = config
        self.config_hash = config_hash(config)
        self.seed = int(seed)
        self.rng_mode = rng_mode
        self.num_blues = int(num_blues)
        self.num_reds = int(num_reds)
        self.blue_nx = int(blue_nx)
        self.red_nx = int(red_nx)
        self.blue_nu = int(blue_nu)

        self.steps = []          # structured, JSON-safe per-step fields
        self._blue_state = []    # arrays, npz-bound
        self._red_state = []
        self._u_requested = []
        self._u_executed = []
        self._cbf_slack = []

        self.outcome = {}
        self.recorded_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.git_commit = _git_commit()

    def record_step(
        self,
        step,
        blue_state,
        red_state,
        u_requested,
        u_executed,
        cbf_slack,
        cbf_active,
        engagement_authorized,
        blue_disabled,
        blue_roles,
        red_roles,
        detected_red_union,
        per_blue_detected_reds,
        target_red_ids,
        recon_target_regions,
        red_outcomes,
        beliefs=None,
        effective_targets=None,
        defend_targets=None,
    ):
        """Called once per control step, immediately before integration.

        States are the values the controls were computed *from*, so a step's
        (state, u_requested, u_executed) triple is internally consistent and can
        be replayed.
        """
        self._blue_state.append(np.asarray(blue_state, dtype=float).copy())
        self._red_state.append(np.asarray(red_state, dtype=float).copy())
        self._u_requested.append(np.asarray(u_requested, dtype=float).copy())
        self._u_executed.append(np.asarray(u_executed, dtype=float).copy())
        self._cbf_slack.append(
            np.zeros(0) if cbf_slack is None else np.asarray(cbf_slack, dtype=float).ravel().copy()
        )
        self.steps.append(
            {
                "step": int(step),
                "cbf_active": bool(cbf_active),
                "engagement_authorized": [bool(x) for x in engagement_authorized],
                "blue_disabled": [bool(x) for x in blue_disabled],
                "blue_roles": {str(k): v for k, v in sorted(blue_roles.items())},
                "red_roles": {str(k): v for k, v in sorted(red_roles.items())},
                "detected_red_union": sorted(int(x) for x in detected_red_union),
                "per_blue_detected_reds": {
                    str(k): sorted(int(x) for x in v)
                    for k, v in sorted(per_blue_detected_reds.items())
                },
                "target_red_ids": {
                    str(k): (None if v is None else int(v))
                    for k, v in sorted(target_red_ids.items())
                },
                "recon_target_regions": {
                    str(k): int(v) for k, v in sorted(recon_target_regions.items())
                },
                # The auction / the screening-task rule: what each DEFEND agent was ALLOCATED to screen or
                # hold -- ("contact", c) or ("station", k). Empty for every
                # planner that predates the auction. Without it, "what did the team
                # screen" would have to be reconstructed from motion, which is not
                # the same thing as the assignment.
                "defend_targets": {
                    str(k): [str(v[0]), int(v[1])]
                    for k, v in sorted((defend_targets or {}).items())
                },
                "red_outcomes": list(red_outcomes),
                # {agent_id: {contact_id: age_in_steps}} -- what each agent
                # believed at this step, and how old each belief was. This is
                # the raw material for's information metrics:
                # age of information at the decision instant, and pairwise
                # belief disagreement between agents.
                # {agent_id: {"source", "contact_id", "controller"}} -- what
                # each blue was ACTUALLY steered at this step, which is not the
                # same thing as target_red_ids above. A NEUTRALIZE agent whose
                # assigned target has gone inactive keeps that assignment on
                # record while the controller silently falls back to its
                # nearest believed contact
                # So
                # "the team held its commitments" and "the team kept chasing
                # the same contacts" are separate claims and need separate
                # evidence.
                "effective_targets": (
                    {}
                    if effective_targets is None
                    else {
                        str(agent): {
                            "source": str(entry["source"]),
                            "contact_id": (
                                None
                                if entry["contact_id"] is None
                                else int(entry["contact_id"])
                            ),
                            "controller": str(entry["controller"]),
                        }
                        for agent, entry in sorted(effective_targets.items())
                    }
                ),
                "beliefs": (
                    {}
                    if beliefs is None
                    else {
                        str(agent): {str(cid): int(age) for cid, age in sorted(per.items())}
                        for agent, per in sorted(beliefs.items())
                    }
                ),
            }
        )

    def finalize(self, outcome):
        self.outcome = outcome

    # -- derived ---------------------------------------------------------

    def safety_metrics(self):
        """'s required safety-filter reporting.

        Only translational control is compared. Yaw rate is passed through the
        filter unmodified by construction (see the CBF block in
        Simulation.run), so including it would dilute the intervention
        magnitude with a channel the filter never touches.
        """
        if not self._u_requested:
            return {}

        req = np.asarray(self._u_requested)
        exe = np.asarray(self._u_executed)
        n_steps, n_agents = req.shape[0], self.num_blues

        per_step_agent = np.zeros((n_steps, n_agents))
        for i in range(n_agents):
            lo = i * self.blue_nu
            d = req[:, lo : lo + 2] - exe[:, lo : lo + 2]
            per_step_agent[:, i] = np.linalg.norm(d, axis=1)

        intervened = per_step_agent > INTERVENTION_TOL
        active = np.array([s["cbf_active"] for s in self.steps], dtype=bool)
        flat = per_step_agent.ravel()

        return {
            "intervention_tol": INTERVENTION_TOL,
            "steps_with_cbf_active": int(active.sum()),
            # The honest characterization: a sweep, because the fraction is
            # threshold-sensitive and the distribution is strongly bimodal
            # (a near-zero mode at the solver noise floor, and a mode at full
            # control authority). Keyed by threshold, over agent-steps.
            "intervention_fraction_by_threshold": {
                str(t): float((flat > t).mean()) for t in INTERVENTION_THRESHOLDS
            },
            "magnitude_percentiles": {
                str(q): float(np.percentile(flat, q)) for q in (50, 75, 90, 95, 99)
            },
            "intervention_fraction_all_steps": float(intervened.any(axis=1).mean()),
            "intervention_fraction_of_active_steps": (
                float(intervened.any(axis=1)[active].mean()) if active.any() else 0.0
            ),
            "mean_intervention_magnitude": float(per_step_agent[intervened].mean())
            if intervened.any() else 0.0,
            "max_intervention_magnitude": float(per_step_agent.max()),
            "per_agent_intervention_fraction": [
                float(intervened[:, i].mean()) for i in range(n_agents)
            ],
            "per_agent_mean_intervention": [
                float(per_step_agent[intervened[:, i], i].mean()) if intervened[:, i].any() else 0.0
                for i in range(n_agents)
            ],
        }

    def manifest(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "recorded_at": self.recorded_at,
            "git_commit": self.git_commit,
            "seed": self.seed,
            "rng_mode": self.rng_mode,
            "config": self.config,
            "config_hash": self.config_hash,
            "dimensions": {
                "num_blues": self.num_blues,
                "num_reds": self.num_reds,
                "blue_nx": self.blue_nx,
                "red_nx": self.red_nx,
                "blue_nu": self.blue_nu,
            },
            "n_recorded_steps": len(self.steps),
            "outcome": self.outcome,
            "safety": self.safety_metrics(),
            "steps": self.steps,
        }

    def arrays(self):
        """Ragged only in cbf_slack, whose width depends on which CBF
        constraints are enabled and how many pairs exist that step; stored
        padded with NaN plus an explicit width column so a reader can tell a
        genuine zero slack from an absent constraint."""
        slack_width = max((len(s) for s in self._cbf_slack), default=0)
        slack = np.full((len(self._cbf_slack), slack_width), np.nan)
        for t, s in enumerate(self._cbf_slack):
            slack[t, : len(s)] = s
        return {
            "blue_state": np.asarray(self._blue_state),
            "red_state": np.asarray(self._red_state),
            "u_requested": np.asarray(self._u_requested),
            "u_executed": np.asarray(self._u_executed),
            "cbf_slack": slack,
            "cbf_slack_width": np.array([len(s) for s in self._cbf_slack]),
        }

    def save(self, stem):
        """Writes ``<stem>.json`` and ``<stem>.npz``. Returns both paths."""
        stem = os.fspath(stem)
        parent = os.path.dirname(stem)
        if parent:
            os.makedirs(parent, exist_ok=True)
        json_path, npz_path = stem + ".json", stem + ".npz"
        with open(json_path, "w") as f:
            json.dump(self.manifest(), f, indent=2, sort_keys=True)
            f.write("\n")
        np.savez_compressed(npz_path, **self.arrays())
        return json_path, npz_path
