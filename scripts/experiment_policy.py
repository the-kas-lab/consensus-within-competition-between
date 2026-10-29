"""Behaviour selections for NEW experiments (not for reproducing old ones).

A NEUTRALIZE agent
steers at, and points its sensor toward, the latest belief it holds about the
contact it is pursuing.

* ``neutralize_aim="live"``. The committed contact is fixed at
  assignment; the aim point is the agent's current belief about it, predicted
  from its observation time to now. Updating the aim point changes neither
  the commitment nor any allocation.
* ``neutralize_heading="pursued_contact"``. The sensor points at that
  same belief, ahead of unrelated detections and of the detection grace hold,
  turning the short way.

**Timing assumption, stated explicitly:** every agent shares one simulation
clock (an integer step). A belief's age is ``now - observed_at`` with the
sensor timestamp preserved through relays, so communication delay is already
inside the belief layer's prediction and is not compensated again. No agent
reads another's delay, a network parameter, or true target state.

``Simulation``'s own defaults (``"frozen"``, ``"inherited"``) and every
existing experiment configuration (Exp 0-5 runners) are deliberately left
unchanged, so historical results stay reproducible and are not reinterpreted.
A new experiment opts in explicitly, e.g.::

    from scripts.experiment_policy import NEW_EXPERIMENT_PURSUIT
    cfg = dict(SOME_BASE_CFG, family=family, **NEW_EXPERIMENT_PURSUIT)
    sim = build_episode(seed, belief_mode, cfg)

``neutralize_aim="aligned"`` remains available as a diagnostic option only.
"""

NEW_EXPERIMENT_PURSUIT = {
    "neutralize_aim": "live",
    "neutralize_heading": "pursued_contact",
}


# Safety layer for the competing-teams comparisons (every arm uses the SAME
# configuration). Selecting safety_filter="local" alone does NOT filter reds:
# the red filter still runs only under enable_red_cbf, so it is set here
# explicitly, as is red-red avoidance. Red-red contact is not a simulator
# event; avoidance violations and contacts are reported, never a termination.
# local_cbf_gamma = 0.05 (see verification.verify_local_cbf) is tied to dt = 0.1 and v_max = a_max = 2.0 for both teams.
#
# Red constraints: red-red
# separation (cooperative halves) and the red's own control bounds ONLY.
# local_cbf_red_blue=False removes the red-side blue-avoidance rows, which in
# "safety.1" kept every pursuer at D = 3 and suppressed contact in the tested
# configuration. Avoiding blues is the intruder's ATTACK/EVADE
# decision. Blue constraints are unchanged: cooperative mobile teammates,
# disabled blues as immobile obstacles, active reds at constant velocity, and
# the exemption only for the installed pursuit pair.
NEW_EXPERIMENT_SAFETY_VERSION = "safety.2"
NEW_EXPERIMENT_SAFETY = {
    "safety_state_sync": "frozen",
    "safety_filter": "local",
    "local_cbf_gamma": 0.05,
    "enable_red_cbf": True,
    "local_cbf_red_red": True,
    "local_cbf_red_blue": False,
}

# Intruder information for the competing-teams comparisons: each intruder's
# own sightings over an ideal intruder channel; EVADE from those beliefs.
NEW_EXPERIMENT_INTRUDER_INFO = {
    "intruder_info": "shared_beliefs",
}

# Task model for the competing-teams comparisons:
# SCREEN disabled, explicit and implicit -- DEFEND(station), RECON(cell),
# NEUTRALIZE(contact, slot) only. three_task requires the explicit station
# fallback. Applies to the CBAA and independent arms here; the
# centralized commander arm adopts the same task model through its adapter
# (Simulation currently refuses three_task without allocation="cbaa", because
# the legacy role planners' DEFEND screens implicitly). Historical four-task
# behaviour stays the Simulation default for reproduction; its SCREEN
# agents get no pursued-contact heading (a historical-mode limitation).
NEW_EXPERIMENT_TASKS = {
    "cbaa_task_set": "three_task",
    "cbaa_fallback": "station",
}

# Falsified-report stress test, primary setting (identical at 2 and
# 4 intruders). "Persistent compromise with opportunistic report
# falsification": compromise from step 0, but exposure starts only when the
# sender holds an eligible first-hand report. Window = the 400-step (40 s at
# dt = 0.1) episode; the cap of 400 altered originating broadcasts is an
# implementation bound and does not bind here (one broadcast per 0.1 s round).
# Displacement: a fixed 10-unit offset toward the south-west, the direction
# the defended region is mainly approached from; it is NOT farther from the
# defended polygon for every red position.
NEW_EXPERIMENT_FALSIFIED_REPORTS = {
    "onset_step": 0,
    "duration_steps": 400,
    "displacement": [-10.0 / 2 ** 0.5, -10.0 / 2 ** 0.5],
    "report_cap": 400,
}

# Executing an unavailable NEUTRALIZE target, every defender arm. The task-model commander
# path executes a commanded contact that is resolved, or absent from the
# recipient's own belief, as DEFEND(own station). Without this setting the
# CBAA and independent arms would instead pursue the nearest believed contact
# (one nobody assigned) -- an execution difference between arms that is not
# part of the comparison. With it, all three arms hold station, logged in
# Simulation.unavailable_target_log. Commitments are untouched.
NEW_EXPERIMENT_EXECUTION = {
    "unavailable_target": "station",
}

# How agents learn that an opponent is gone (USER DECISION: public
# notification). Every resolution -- a red neutralized or escaped, a blue
# disabled -- is announced at once to the opposing team: the resolved agent
# leaves every opposing belief store (the commander's included) and no relayed
# record re-adds it, so task lists, prompts and the commander's catalogue agree
# with the execution gates that already read the simulator's red_outcomes.
# An explicit, declared information assumption, not belief-local inference.
NEW_EXPERIMENT_INFORMATION = {
    "resolution_info": "public",
}

# Red sensor (heading) control: the blue bearing-tracking rule
# mirrored for reds -- track a blue the red personally detects (its EVADE target
# first), hold the last sighting for bearing_hold_steps, else face its own
# velocity direction. Bounded by the
# red yaw-rate limit; heading never affects translation. Versioned separately
# from heading OBSERVABILITY, which is the prompt versions m8b.3 / commander.3 /
# intruder.3 (scripts/inference_profile.NEW_EXPERIMENT_INTERFACES).
NEW_EXPERIMENT_SENSING = {
    "red_sensor_control": "track_or_velocity",
}

# Every new-experiment selection above, in one place, so a runner cannot miss
# one (measure_inference's capture once omitted NEW_EXPERIMENT_EXECUTION).
# Arm-specific settings (task model for CBAA/independent, the commander's
# adapter, falsified reports) are added by the runner.
NEW_EXPERIMENT_COMMON = dict(
    NEW_EXPERIMENT_PURSUIT, **NEW_EXPERIMENT_SAFETY, **NEW_EXPERIMENT_INTRUDER_INFO,
    **NEW_EXPERIMENT_EXECUTION, **NEW_EXPERIMENT_INFORMATION, **NEW_EXPERIMENT_SENSING,
)


# The bid version and station posture every reported episode uses ("v2.3mb" is
# the design label stored in every episode id): bids with the gate-aware breach
# forecast, the optimistic interception forecast, the threat value every
# believed red carries and conditional pursuit continuity, and a static station
# posture from the 160-300 degree approach prior.
DESIGN_POLICY = {
    "cbaa_bid_version": "v2.3",
    "station_heading": "configured",
}
DESIGNS = {"v2.3mb": DESIGN_POLICY}
