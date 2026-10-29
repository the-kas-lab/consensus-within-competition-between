import glob
import inspect
import os
import random
from collections import defaultdict

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from agents.dynamics import Agent
from control.local_cbf import COOPERATIVE, FULL, LOCAL_CBF_VERSION, LocalCBFConfig, Neighbor, filter_agent
from simulation.rng import RngStreams
from coordination.tasks import DEFAULT_TASK_SET, TaskContext, check_task_set, screen_available
from simulation.recorder import EpisodeRecorder, describe_config
from simulation.decentralized_solver import DecentralizedPursuitEvasionSolver
from planning.planner import (
    LLMRedRolePlanner,
    LLMRolePlanner,
    PlannerObservation,
    PlannerOutput,
    plan_full_from,
    RuleBasedRedRolePlanner,
    RuleBasedBlueRolePlanner,
)
from planning.roles import RedRole, BlueRole
import perception.sensing as sensing
from perception.coverage import CoverageTracker
from comms import BROADCAST, BeliefUpdate, DiskLinkModel, Network, make_envelope
from comms.messages import AUCTION_MESSAGE, ASSIGNMENT_COMMAND, TARGET_UPDATE, AssignmentCommand
from comms.receive import (
    is_stale_allocation,
    COMMAND_ACCEPTED,
    COMMAND_SUPERSEDED,
    classify_command,
)
from perception.belief import LocalBeliefStore
from perception.belief.records import TargetBelief

_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
)
_IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "JPG", "JPEG", "PNG")


def _glob_images(subdir, prefix):
    """All prefix_*.<ext> files under assets/<subdir>/, any of
    _IMAGE_EXTENSIONS; planner._encode_image_data_uri handles both png and
    jpg/jpeg for MIME typing."""
    paths = []
    for ext in _IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(_ASSETS_DIR, subdir, f"{prefix}_*.{ext}")))
    return sorted(paths)


def rounds_per_epoch_cycle(planning_steps, comm_steps):
    """Communication rounds held by each planning epoch, over one repeating
    cycle of the schedule.

    Epoch k starts at step k*P and owns the rounds at steps s in (k*P, (k+1)*P]
    with s % C == 0: the round at a planning step runs before planning in the
    step, so it still belongs to the old epoch. The pattern repeats every
    C / gcd(P, C) epochs. ``[3, 3, 4]`` for P = 10, C = 3.
    """
    from math import gcd

    p, c = int(planning_steps), int(comm_steps)
    return [(k + 1) * p // c - k * p // c for k in range(c // gcd(p, c))]


_RED_DRONE_IMAGES = _glob_images("contact_photos", "intruder")
_BIRD_IMAGES = _glob_images("contact_photos", "bird")
_CIVILIAN_IMAGES = _glob_images("contact_photos", "civilian")


# Scenario families. ``symmetric`` is the geometry
# every result to date was produced under and must stay byte-identical;
# ``multi_bearing`` is added *alongside* it, never in place of it, so that "it
# only helps under condition X" stays a statement we can make.
SCENARIO_FAMILIES = ("symmetric", "multi_bearing")
# The pursuer aim-point rule: the NEUTRALIZE aim point
# between assignment applications. "frozen" is the historical behaviour.
NEUTRALIZE_AIM_MODES = ("frozen", "aligned", "live")
# The pursuer sensor-heading rule: where a NEUTRALIZE agent points its sensor. "inherited" is the historical
# behaviour (own detection only).
NEUTRALIZE_HEADING_MODES = ("inherited", "pursued_contact")
# Which blue state the red safety filter sees. "sequential" -- the historical
# behaviour -- hands it the blues AFTER they have stepped; "frozen" hands every
# safety filter the same pre-step snapshot at time t.
SAFETY_STATE_SYNC_MODES = ("sequential", "frozen")
# "joint" -- the historical one-QP-over-all-blues filter plus the red one-step
# filter; "local" -- one QP per agent (control/local_cbf.py).
SAFETY_FILTER_MODES = ("joint", "local")
# What intruders know about blues. "legacy" -- the historical path: EVADE
# reads the blues' actual MPC plans, gated by this step's team detection.
# "shared_beliefs" -- each red's own first-hand sightings, shared over an
# ideal (instant, lossless, timestamp-preserving) intruder channel; EVADE
# reads constant-velocity predictions from those beliefs.
INTRUDER_INFO_MODES = ("legacy", "shared_beliefs")

# Why the bearing offset is bounded, and bounded by *this*: the defended region
# sits at (0.82, 0.82) * arena_size -- 17.5 units from the NE arena walls
# against a red standoff radius of ~90.5. Measured, only the bearing arc
# 206deg-244deg (11% of the circle) can hold a red at that radius, and no
# smaller radius rescues it: R=50 opens only 36% of bearings and R=20 (68%)
# is *inside* the blue spawn ring, which would put reds on top of defenders.
# Omnidirectional approach is therefore unavailable without relocating the
# defended region -- which would change standoff distance as well as bearing
# and stop being a one-variable manipulation. +-10deg spans the feasible arc
# (37.7deg total, against 12deg for ``symmetric``) and separates two reds at
# the extremes by 58.5 units, comfortably beyond the 35-unit sensor range,
# which is the mechanism the scenario-geometry rule actually asks for: blue A sees red 1 while blue B
# sees red 2.
DEFAULT_MULTI_BEARING_MAX_OFFSET_DEG = 10.0


class Simulation:
    def __init__(
        self,
        num_blues=4,
        num_reds=1,
        dt=0.1,
        horizon=60,
        arena_size=100.0,
        D_safe_blue=3.0,
        D_safe_blue_red=3.0,
        D_safe_red=3.0,
        solver_mode="decentralized",
        sim_steps=600,
        seed=130,
        blue_v_max=2.0,
        blue_a_max=2.0,
        red_v_max=2.0,
        red_a_max=2.0,
        red_random_policy=False,
        enable_blue_blue_cbf=False,
        enable_blue_red_cbf=False,
        enable_red_cbf=False,
        enable_convex_hull_containment=False,
        blue_blue_cbf_slack_weight=1e5,
        blue_red_cbf_slack_weight=1e5,
        red_cbf_slack_weight=1e5,
        convex_hull_slack_weight=1e5,
        contact_tolerance=4e-1,
        planning_interval_seconds=10.0,
        bearing_hold_seconds=1.0,
        role_planner=None,
        neutralize_fraction=0.5,
        red_role_planner=None,
        neutralize_points=1.0,
        escape_points=1.0,
        num_bird_slots=0,
        bird_spawn_prob=0.2,
        bird_v_max=1.5,
        bird_a_max=1.0,
        num_contact_slots=0,
        p_bird=0.0,
        p_civilian=0.0,
        civilian_v_max=1.5,
        civilian_a_max=1.0,
        fov_half_angle_deg=sensing.DEFAULT_FOV_HALF_ANGLE_DEG,
        fov_range=sensing.DEFAULT_FOV_RANGE,
        fov_bearing_speed_threshold=sensing.DEFAULT_BEARING_SPEED_THRESHOLD,
        omega_max=2.0 * np.pi,
        coverage_grid_dim=4,
        legacy_global_rng=True,
        record_episode=True,
        belief_mode="ground_truth",
        belief_validity_age_steps=None,
        # command-delivery condition A and B. Default None/False keeps every pre-the command-delivery conditions mode
        # byte-identical -- the centralized planner still reads the
        # detection-gated ground-truth observation it always did.
        centralized_from_belief=False,
        commander_id=0,
        commander_failure_stops_planning=False,
        deliver_assignments_over_network=False,
        command_max_age_steps=None,
        command_max_hops=None,
        per_agent_planning=False,
        allocation=None,
        cbaa_slots=1,
        cbaa_stable_rounds=None,
        cbaa_profile="balanced",
        cbaa_transport=None,
        cbaa_modifier_source=None,
        cbaa_fallback="legacy",
        cbaa_allow_unsettleable=False,
        cbaa_task_set="four_task",
        cbaa_bid_version="v1",
        station_heading="inherited",
        neutralize_aim="frozen",
        neutralize_heading="inherited",
        safety_state_sync="sequential",
        safety_filter="joint",
        local_cbf_gamma=None,
        local_cbf_red_red=True,
        local_cbf_red_blue=True,
        intruder_info="legacy",
        falsified_reports=None,
        intruder_team=None,
        unavailable_target="nearest_believed",
        resolution_info="legacy",
        red_sensor_control="fixed",
        scenario_family="symmetric",
        multi_bearing_max_offset_deg=DEFAULT_MULTI_BEARING_MAX_OFFSET_DEG,
        comm_interval_seconds=None,
        r_comm=float("inf"),
        comm_loss_model=None,
        comm_message_ttl=4,
        decoys=None,
    ):
        # Captured before any mutation so the manifest records exactly the
        # arguments this episode was constructed with, and stays correct when
        # the signature grows. See simulation/recorder.describe_config.
        self._init_args = {k: v for k, v in locals().items() if k != "self"}
        # The three-task rule: recorded only when it is not the historical default, so that
        # every existing configuration -- and its config_hash -- is exactly what
        # it was, and historical records still verify. Absent means four-task.
        if cbaa_task_set == "four_task":
            del self._init_args["cbaa_task_set"]
        # The pursuer aim-point rule, same rule: absent means the historical frozen aim point.
        if neutralize_aim == "frozen":
            del self._init_args["neutralize_aim"]
        # The pursuer sensor-heading rule, same rule: absent means the historical NEUTRALIZE heading.
        if neutralize_heading == "inherited":
            del self._init_args["neutralize_heading"]
        # Same rule: absent means the historical sequential safety state.
        if safety_state_sync == "sequential":
            del self._init_args["safety_state_sync"]
        if intruder_info == "legacy":
            del self._init_args["intruder_info"]
        if falsified_reports is None:
            del self._init_args["falsified_reports"]
        # Image/non-hostile experiment: absent when off, so every existing
        # configuration and its config_hash are unchanged.
        if decoys is None:
            del self._init_args["decoys"]
        if intruder_team is None:
            del self._init_args["intruder_team"]
        if unavailable_target == "nearest_believed":
            del self._init_args["unavailable_target"]
        if resolution_info == "legacy":
            del self._init_args["resolution_info"]
        if red_sensor_control == "fixed":
            del self._init_args["red_sensor_control"]
        # Where each red's sensor (its heading) points. "fixed" (historical):
        # the red controllers carry no heading objective, so a red keeps its
        # spawn heading (measured: 0.0 deg change over whole episodes).
        # "track_or_velocity": face a blue this red PERSONALLY detects -- the blue it is
        # evading if that one is detected, else the nearest detected -- keep
        # facing the last sighting for bearing_hold_steps after losing it,
        # otherwise face its own direction of motion, holding the current
        # heading below RED_SENSOR_MIN_SPEED. Turning is bounded by the red
        # yaw-rate limit. Heading never enters the translational controls.
        if red_sensor_control not in ("fixed", "track_or_velocity"):
            raise ValueError(f"red_sensor_control must be 'fixed' or 'track_or_velocity', "
                             f"got {red_sensor_control!r}")
        self.red_sensor_control = red_sensor_control
        self.red_sensor_log = []
        # How agents learn that an opponent is gone.
        # "legacy" (historical): belief stores never drop a resolved red or a
        # disabled blue, so task lists, prompts and the commander's catalogue
        # can keep listing them as ageing contacts, while blue EXECUTION gates
        # on the simulator's red_outcomes. "public": an explicit information
        # assumption -- every resolution (a red neutralized or escaped, a blue
        # disabled) is announced at once to every agent of the opposing team
        # (blues for reds, reds for blues): the contact is dropped from those
        # belief stores and no later observation or relayed record re-adds it.
        # The execution gates on red_outcomes then read information every
        # agent holds. Logged in resolution_log.
        if resolution_info not in ("legacy", "public"):
            raise ValueError(f"resolution_info must be 'legacy' or 'public', got {resolution_info!r}")
        self.resolution_info = resolution_info
        self.resolution_log = []
        # What a NEUTRALIZE commitment executes when its contact is resolved
        # or absent from the agent's own belief at assignment application.
        # "nearest_believed" (historical): the role stays NEUTRALIZE and the
        # solver pursues the nearest believed contact -- one nobody assigned.
        # "station": the agent holds its own station until the next
        # application, the rule the task-model commander path already applies
        # (_executable_commander_output), so every defender arm executes an
        # unavailable target the same way. Logged in unavailable_target_log.
        if unavailable_target not in ("nearest_believed", "station"):
            raise ValueError(f"unavailable_target must be 'nearest_believed' or 'station', "
                             f"got {unavailable_target!r}")
        self.unavailable_target = unavailable_target
        self.unavailable_target_log = []
        self._unavailable_logged = {}
        # Same rule: absent means the lcbf.1 red filter (reds avoid active
        # blues). Recorded only when switched off.
        if local_cbf_red_blue:
            del self._init_args["local_cbf_red_blue"]
        if safety_filter == "joint":
            del self._init_args["safety_filter"]
            del self._init_args["local_cbf_gamma"]
            del self._init_args["local_cbf_red_red"]
            if not local_cbf_red_blue:
                raise ValueError('local_cbf_red_blue=False requires safety_filter="local"')

        if solver_mode != "decentralized":
            raise ValueError(
                f"Unsupported solver_mode '{solver_mode}'. Only 'decentralized' is "
                "supported."
            )
        if scenario_family not in SCENARIO_FAMILIES:
            raise ValueError(
                f"Unknown scenario_family '{scenario_family}'. Known families: "
                f"{', '.join(SCENARIO_FAMILIES)}."
            )
        if multi_bearing_max_offset_deg < 0.0:
            raise ValueError(
                "multi_bearing_max_offset_deg must be >= 0.0, got "
                f"{multi_bearing_max_offset_deg}"
            )
        if int(num_contact_slots) > 0:
            if int(num_reds) != 1:
                raise ValueError(
                    "num_reds and num_contact_slots are mutually exclusive -- "
                    "num_contact_slots > 0 derives the real-red count itself from "
                    "the per-slot draw, so passing a non-default num_reds alongside "
                    "it is ambiguous. Leave num_reds at its default."
                )
            if not (0.0 <= p_bird):
                raise ValueError(f"p_bird must be >= 0.0, got {p_bird}")
            if not (0.0 <= p_civilian):
                raise ValueError(f"p_civilian must be >= 0.0, got {p_civilian}")
            if p_bird + p_civilian > 1.0:
                raise ValueError(
                    f"p_bird + p_civilian must be <= 1.0, got {p_bird + p_civilian}"
                )

        self.seed = seed
        # Per-purpose RNG streams (see simulation/rng.py). legacy_global_rng=True
        # keeps spawn/patrol/inert on the process-global np.random so episodes
        # stay byte-identical to pre-refactor runs; newly added purposes (comm,
        # attack, scenario) are independent Generators either way, so a new
        # subsystem can never perturb an existing draw.
        self.rng = RngStreams(self.seed, legacy_global=legacy_global_rng)
        if self.rng.legacy_global:
            np.random.seed(self.seed)

        # Pure observation -- the recorder never feeds anything back into the
        # episode, so a recorded run and an unrecorded one are identical.
        # tests/test_golden_episode.py holds that to account.
        self.record_episode = bool(record_episode)
        self.recorder = None

        # How each blue's belief about red contacts is populated each step.
        #
        #   "ground_truth" -- every agent believes every red's true state, age 0.
        #                     Reproduces pre-M2 behavior byte-for-byte and is the
        #                     regression baseline, NOT an experimental condition.
        #   "shared"       -- agents observe locally, then merge every other
        #                     agent's first-hand observations with no latency or
        #                     loss. The perfect-communication arm.
        #   "local"        -- agents observe locally and share nothing. The
        #                     zero-communication arm.
        #   "networked"    -- agents observe locally, then exchange belief
        #                     snapshots over a real comms.Network (M4): a
        #                     round-based graph with one hop per round,
        #                     configurable range and loss.
        #
        # "shared" and "local" are Experiment 0's bracket.
        #
        # "networked" is NOT a drop-in for "shared" and must never be described
        # as one. Even on a complete, lossless graph it differs, because
        # exchange costs at least one communication round: a sighting reaches a
        # teammate one comm tick after it is made, where "shared" delivers it in
        # the same step. That gap is the thing the communication conditions varies, so it is a feature --
        # but it means "the graph was complete" is not grounds for expecting
        # "shared" numbers.
        if belief_mode not in ("ground_truth", "shared", "local", "networked"):
            raise ValueError(
                f"Unsupported belief_mode {belief_mode!r}. "
                "Expected 'ground_truth', 'shared', 'local' or 'networked'."
            )
        self.belief_mode = belief_mode
        # None = never stale. Consumers apply their own limit; the store itself
        # never deletes.
        self.belief_validity_age_steps = belief_validity_age_steps
        # See `centralized_from_belief`
        # is the fix for the defect that made the command-delivery conditions 2x2 non-factorial: the old
        # centralized branch planned from ground-truth red states gated by this
        # step's FOV union, never touching belief_stores, so "centralized" also
        # silently meant "memoryless, no staleness".
        self.centralized_from_belief = bool(centralized_from_belief)
        self.commander_id = int(commander_id)
        self.commander_failure_stops_planning = bool(commander_failure_stops_planning)
        # Condition B's OUTBOUND half: the commander's
        # assignment reaches each subordinate over the same network its beliefs
        # travel on, one hop per communication round, instead of appearing in
        # everyone's hands the instant it is computed. Off by default, which is
        # condition A -- immediate delivery, no network.
        self.deliver_assignments_over_network = bool(deliver_assignments_over_network)
        # The age limit of contract Sec 3a, kept SEPARATE from the ordering rule
        # and defaulting to off. With it off a command that took twenty rounds
        # to arrive is still applied, which is what keeps "arrived late"
        # distinguishable from "never arrived" -- the distinction B exists to
        # measure.
        self.command_max_age_steps = (
            None if command_max_age_steps is None else int(command_max_age_steps)
        )
        # An optional declared cap on how far a command may be relayed. None --
        # the default -- means the flood's own bound applies: each agent
        # forwards a given command at most once, so it visits at most
        # num_blues agents whatever this is. Kept separate from
        # comm_message_ttl, which is inert and stays inert. See
        # _command_hop_budget.
        self.command_max_hops = (
            None if command_max_hops is None else int(command_max_hops)
        )
        if self.deliver_assignments_over_network:
            if not self.centralized_from_belief:
                raise ValueError(
                    "deliver_assignments_over_network=True requires "
                    "centralized_from_belief=True: there is no commander to "
                    "deliver from otherwise. Conditions C and D compute their "
                    "own assignments and have nothing to send."
                )
            if belief_mode != "networked":
                raise ValueError(
                    "deliver_assignments_over_network=True requires "
                    f"belief_mode='networked', got {belief_mode!r}. Commands "
                    "travel over the same network the beliefs do, and the other "
                    "belief modes do not build one."
                )
        # Initialised here as well as in run(), because episode_outcome() reads
        # it and callers are entitled to ask for the outcome of a Simulation
        # that has not been run -- two scenario-family tests do exactly that.
        self.commander_lost_step = None
        self.planning_source = self.commander_id
        self._reset_command_state()
        self.belief_stores = {}

        # Whether role assignment is decentralized too, or only the controllers
        # act on local belief. Default off: with it off, Simulation reproduces
        # the pre-M3 golden baselines byte-for-byte. See _plan_per_agent and
        # The decentralization rule.
        self.per_agent_planning = bool(per_agent_planning)
        # The auction. None -- the default -- builds no auction at all, and every existing
        # mode is untouched. "cbaa" supersedes the blue role planner: roles come
        # from each agent's settled auction commitment. See
        #
        self.allocation = allocation
        self.defend_target_override = {}
        self.cbaa_runtime = None
        if allocation is not None:
            if allocation != "cbaa":
                raise ValueError(f"unknown allocation {allocation!r}; expected None or 'cbaa'")
            if belief_mode not in ("shared", "networked", "local"):
                raise ValueError(
                    "allocation='cbaa' generates tasks from each agent's belief, so it "
                    f"needs belief_mode shared/networked/local, not {belief_mode!r}"
                )
            if centralized_from_belief or deliver_assignments_over_network:
                raise ValueError(
                    "allocation='cbaa' is decentralized by construction; it cannot be "
                    "combined with the command-delivery conditions commander conditions"
                )
            from coordination.runtime import TRANSPORT_FOR_BELIEF_MODE, CBAARuntime

            # The three-task rule: one task set for the auction, the bridge and the solver.
            # Every arm is this runtime (the independent one has transport
            # "none"), so they cannot differ. A bridge built for another task
            # set would advertise tasks the auction does not have, or hide
            # ones it does; refused here rather than at its first contact.
            bridge_tasks = getattr(cbaa_modifier_source, "task_set", DEFAULT_TASK_SET)
            if cbaa_modifier_source is not None and bridge_tasks != cbaa_task_set:
                raise ValueError(
                    f"cbaa_modifier_source was built for task set {bridge_tasks!r}, "
                    f"but cbaa_task_set is {cbaa_task_set!r}"
                )
            self.cbaa_runtime = CBAARuntime(
                slots=cbaa_slots,
                stable_rounds=cbaa_stable_rounds,
                profile=cbaa_profile,
                transport=cbaa_transport or TRANSPORT_FOR_BELIEF_MODE[belief_mode],
                modifier_source=cbaa_modifier_source,
                fallback=cbaa_fallback,
                task_set=cbaa_task_set,
            )
        elif cbaa_modifier_source is not None:
            raise ValueError("cbaa_modifier_source needs allocation='cbaa'")
        elif check_task_set(cbaa_task_set) != DEFAULT_TASK_SET:
            # A commander that chooses from the task model (coordination.
            # commander) realizes a task set itself and DECLARES it; the
            # simulator then declares the same set (implicit screening off),
            # rather than trusting an adapter to behave like it.
            model = getattr(role_planner, "task_model", None)
            if not (centralized_from_belief and not per_agent_planning
                    and model == check_task_set(cbaa_task_set)):
                raise ValueError(
                    f"cbaa_task_set={cbaa_task_set!r} needs allocation='cbaa' or a "
                    "task-model commander (centralized_from_belief, per_agent_planning="
                    "False, role_planner.task_model == the task set): the legacy role "
                    "planners have no task model, and their DEFEND screens implicitly"
                )
        self.cbaa_task_set = check_task_set(cbaa_task_set)
        # The task-model commander path (coordination/commander.py).
        self.commander_task_model = (
            getattr(role_planner, "task_model", None) is not None and bool(centralized_from_belief)
        )
        if self.commander_task_model and getattr(role_planner, "task_model") != self.cbaa_task_set:
            raise ValueError(
                f"the commander realizes {role_planner.task_model!r}; declare "
                f"cbaa_task_set={role_planner.task_model!r} so the simulator agrees"
            )
        self.commander_slots = int(cbaa_slots)
        self.commander_pending = {}
        self.commander_pending_log = []
        self._commander_in_force = None
        self._commander_resolved = {}

        self.num_blues = num_blues
        self.num_reds = int(num_reds)
        self.num_contact_slots = int(num_contact_slots)
        self.p_bird = float(p_bird)
        self.p_civilian = float(p_civilian)
        self.contact_substitution_enabled = self.num_contact_slots > 0
        self.slot_kinds = []
        self.num_civilians = 0
        self.num_substitution_birds = 0
        if self.contact_substitution_enabled:
            # Local RNG (never np.random), offset from _build_contact_roster's
            # own offset (+4242) so the two draws never collide -- this keeps
            # the whole substitution mechanic a structural no-op with respect
            # to every other seeded draw in the episode (spawn positions,
            # velocities, contact-roster shuffling) whether or not it's used.
            slot_rng = random.Random(self.seed + 9137)

            def _draw_slot_kinds():
                kinds = []
                for _ in range(self.num_contact_slots):
                    r = slot_rng.random()
                    if r < self.p_bird:
                        kinds.append("bird")
                    elif r < self.p_bird + self.p_civilian:
                        kinds.append("civilian")
                    else:
                        kinds.append("drone")
                return kinds

            slot_kinds = _draw_slot_kinds()
            # Guarantee at least one real drone per episode -- an all-non-hostile contact
            # episode would otherwise resolve as an instant, degenerate
            # "defenders won" before the simulation even starts (num_reds==0
            # makes num_neutralized+num_escaped>=num_reds trivially true at
            # step 0). This conditions the realized distribution on >=1
            # success rather than the raw Binomial(num_contact_slots,
            # 1-p_bird-p_civilian) -- an explicit, describable adjustment,
            # not a silent bias.
            while "drone" not in slot_kinds:
                slot_kinds = _draw_slot_kinds()
            self.slot_kinds = slot_kinds
            self.num_reds = slot_kinds.count("drone")
            self.num_substitution_birds = slot_kinds.count("bird")
            self.num_civilians = slot_kinds.count("civilian")
        # non-hostile contacts: birds and civilian drones that live in the red state
        # arrays at indices num_real_reds..num_reds-1, so that sensing, every
        # blue's beliefs, the network, task generation, the commander's
        # catalogue, the safety filter and contact resolution treat them as
        # contacts by index, unchanged. They are not intruders: no intruder
        # planner, no red beliefs, no breach, no score; their motion is
        # scripted. Off (None) everywhere else: num_real_reds == num_reds.
        self.decoy_spec = self._check_decoys(decoys, num_contact_slots)
        self.num_real_reds = self.num_reds
        self.red_kinds = ["drone"] * self.num_reds
        if self.decoy_spec is not None:
            self.red_kinds += (["bird"] * self.decoy_spec["birds"]
                               + ["civilian"] * self.decoy_spec["civilians"])
            self.num_reds = len(self.red_kinds)
        self.decoy_indices = frozenset(j for j, k in enumerate(self.red_kinds) if k != "drone")
        self.contact_image_paths = dict((self.decoy_spec or {}).get("images") or {})
        self.dt = dt
        self.horizon = horizon
        self.arena_size = float(arena_size)
        self.arena_min = 0.0
        self.arena_max = self.arena_size
        self.D_safe_blue = D_safe_blue
        self.D_safe_blue_red = D_safe_blue_red
        self.D_safe_red = D_safe_red
        self.solver_mode = solver_mode
        self.sim_steps = sim_steps
        self.red_random_policy = red_random_policy
        self.scenario_family = str(scenario_family)
        self.multi_bearing_max_offset_deg = float(multi_bearing_max_offset_deg)
        # Communication clock, separate from control and planning.
        # None means "every control step", the fastest the substrate can run.
        self.comm_interval_seconds = comm_interval_seconds
        self.r_comm = float(r_comm)
        self.comm_loss_model = comm_loss_model
        self.comm_message_ttl = int(comm_message_ttl)
        self.network = None
        self.enable_blue_blue_cbf = enable_blue_blue_cbf
        self.enable_blue_red_cbf = enable_blue_red_cbf
        self.enable_red_cbf = enable_red_cbf
        self.enable_convex_hull_containment = enable_convex_hull_containment
        self.blue_blue_cbf_slack_weight = float(blue_blue_cbf_slack_weight)
        self.blue_red_cbf_slack_weight = float(blue_red_cbf_slack_weight)
        self.red_cbf_slack_weight = float(red_cbf_slack_weight)
        self.convex_hull_slack_weight = convex_hull_slack_weight
        self.contact_tolerance = float(contact_tolerance)
        self.fov_half_angle_rad = np.deg2rad(float(fov_half_angle_deg))
        self.fov_range = float(fov_range)
        self.fov_bearing_speed_threshold = float(fov_bearing_speed_threshold)
        self.omega_max = float(omega_max)
        self.coverage_tracker = CoverageTracker(self.arena_size, grid_dim=coverage_grid_dim)
        self.defended_shape_points = None
        self.defense_polygon = None
        self.defense_center = None
        self.defender_spawn_points = None
        self.planning_interval_seconds = float(planning_interval_seconds)
        self.planning_interval_steps = max(
            1, round(self.planning_interval_seconds / self.dt)
        )
        # Third clock. Defaults to every control step, which is the
        # fastest the substrate can run and the least latency it can impose;
        # The communication conditions sweeps it upward. Only consulted when belief_mode == 'networked'.
        self.comm_interval_steps = (
            1
            if self.comm_interval_seconds is None
            else max(1, int(round(float(self.comm_interval_seconds) / self.dt)))
        )
        if self.cbaa_runtime is not None:
            self._guard_auction_schedule(
                num_blues, cbaa_stable_rounds, bool(cbaa_allow_unsettleable)
            )
        self.bearing_hold_seconds = float(bearing_hold_seconds)
        self.bearing_hold_steps = max(0, round(self.bearing_hold_seconds / self.dt))
        self.role_planner = (
            role_planner
            if role_planner is not None
            else RuleBasedBlueRolePlanner(
                neutralize_fraction=neutralize_fraction,
            )
        )
        if self.per_agent_planning and not hasattr(self.role_planner, "plan_one"):
            # Fail at construction, not 400 steps into a sweep. The LLM planners
            # have no plan_one until the LLM planner, and silently falling back to the team
            # planner would produce an episode that looks decentralized in the
            # config and is not -- exactly the kind of result that is worse than
            # a crash.
            raise TypeError(
                f"per_agent_planning=True requires a planner with plan_one(view); "
                f"{type(self.role_planner).__name__} has none. Per-agent planning "
                "arrives for LLM planners at the LLM planner."
            )
        self.role_assignment = {}
        self.planner_output = PlannerOutput()
        self.blue_target_red_idx = {}
        # Initialised here as well as in run(), for the same reason
        # commander_lost_step is: _apply_assignment is now reachable from the
        # communication round (a command can land between planning epochs), so
        # these are no longer touched for the first time by the run loop and a
        # caller driving the comms path directly must not meet an
        # AttributeError.
        self.target_override = {}
        self.target_override_ids = {}
        self.target_override_applied = {}
        self.recon_target_override = {}
        self.blue_last_detected_red_pos = {}
        self.blue_steps_since_detected = {}
        self.red_role_planner = (
            red_role_planner
            if red_role_planner is not None
            else RuleBasedRedRolePlanner()
        )
        self.red_role_assignment = {}
        self.role_assignment_history = []
        self.red_role_assignment_history = []
        self.neutralize_points = float(neutralize_points)
        self.escape_points = float(escape_points)
        self.blue_disabled = [False] * self.num_blues
        self.blue_disabled_history = []

        self.num_bird_slots = int(num_bird_slots)
        self.bird_spawn_prob = float(bird_spawn_prob)
        self.bird_v_max = float(bird_v_max)
        self.bird_a_max = float(bird_a_max)
        self.num_birds = 0
        self.bird_state = np.zeros(0)
        self.history_birds = []
        self.civilian_v_max = float(civilian_v_max)
        self.civilian_a_max = float(civilian_a_max)
        self.civilian_state = np.zeros(0)
        self.history_civilians = []
        self.contact_roster = []  # list of ("red"|"bird"|"civilian", index), position = contact_id
        self.contact_images = {}  # contact_id -> image path
        self.num_bird_engagements = 0
        self.num_civilian_engagements = 0

        self.blue_v_max = float(blue_v_max)
        self.blue_a_max = float(blue_a_max)
        self.red_v_max = float(red_v_max)
        self.red_a_max = float(red_a_max)

        heading_kwargs = {"include_heading": True, "omega_max": self.omega_max}

        self.blue = Agent(
            agent_type=Agent.BLUE,
            dt=self.dt,
            count=self.num_blues,
            v_max=self.blue_v_max,
            a_max=self.blue_a_max,
            **heading_kwargs,
        )
        self.red = Agent(
            agent_type=Agent.RED,
            dt=self.dt,
            count=self.num_reds,
            v_max=self.red_v_max,
            a_max=self.red_a_max,
            use_random_policy=self.red_random_policy,
            **heading_kwargs,
        )
        self.blue_nx = self.blue.nx_single
        self.red_nx = self.red.nx_single
        self._build_perimeter_geometry()
        self.solver = self._build_solver()
        if hasattr(self.solver, "scripted_red_ids"):
            self.solver.scripted_red_ids = self.decoy_indices
        elif self.decoy_indices:
            raise ValueError("decoys need the decentralized solver (scripted red ids)")
        # The station-heading rule. Validated here, applied by the solver to the explicit
        # DEFEND(station) posture only. "inherited" is byte-identical to before.
        # "configured": a static per-station heading computed
        # once from the approach prior and each station's actual sensor origin
        # (coordination.coverage_value.configured_station_headings).
        if station_heading not in ("inherited", "outward", "configured"):
            raise ValueError(f"station_heading must be 'inherited', 'outward' or 'configured', "
                             f"not {station_heading!r}")
        self.station_heading = station_heading
        self.solver.station_heading = station_heading
        self.configured_station_headings = None
        if station_heading == "configured":
            from coordination import coverage_value as cv

            stations = sorted((int(k), np.asarray(v, dtype=float)) for k, v in
                              self.solver.defend_stations.items())
            f = cv.approach_field(self.defense_center, self.solver.defense_polygon, self.arena_min,
                                  self.arena_max, self.fov_half_angle_rad, self.fov_range)
            self.configured_station_headings = cv.configured_station_headings(f, stations)
            self.solver.configured_station_headings = dict(self.configured_station_headings)
        # Bid version: "v1" -- the default -- is every
        # historical bid; "v2" adds coverage, information and threat-timing
        # terms, three-task only; "v2.2" adds the gate-aware
        # forecast, reachable-set interception, the threat value and pursuit
        # continuity; "v2.3" makes that continuity conditional.
        if cbaa_bid_version not in ("v1", "v2", "v2.2", "v2.3"):
            raise ValueError("cbaa_bid_version must be 'v1', 'v2', 'v2.2' or 'v2.3', "
                             f"not {cbaa_bid_version!r}")
        if cbaa_bid_version != "v1" and station_heading not in ("outward", "configured"):
            raise ValueError(f"cbaa_bid_version={cbaa_bid_version!r} values stations by their "
                             "sensor coverage, so it needs a station posture: station_heading "
                             "'outward' or 'configured'")
        if cbaa_bid_version != "v1" and screen_available(self.cbaa_task_set):
            raise ValueError(f"cbaa_bid_version={cbaa_bid_version!r} is defined for the "
                             "three-task model only")
        self.cbaa_bid_version = cbaa_bid_version
        # The pursuer aim-point rule. Which aim point a NEUTRALIZE controller is handed between
        # assignment applications. "frozen" -- the default -- is the historical
        # behaviour: the believed state at the application step, used as if it
        # were current. See _neutralize_aim_points.
        if neutralize_aim not in NEUTRALIZE_AIM_MODES:
            raise ValueError(
                f"neutralize_aim must be one of {NEUTRALIZE_AIM_MODES}, not {neutralize_aim!r}"
            )
        self.neutralize_aim = neutralize_aim
        # {blue_id: "snapshot" | "aligned" | "belief"} for the last solve: which
        # aim point each overridden agent was handed. Diagnostic only; left
        # empty under "frozen", whose aim point is the snapshot itself.
        self.neutralize_aim_sources = {}
        # The pursuer sensor-heading rule. Validated here, applied by the solver to agents flying the
        # NEUTRALIZE controller only. "inherited" is byte-identical to before.
        if neutralize_heading not in NEUTRALIZE_HEADING_MODES:
            raise ValueError(
                f"neutralize_heading must be one of {NEUTRALIZE_HEADING_MODES}, "
                f"not {neutralize_heading!r}"
            )
        self.neutralize_heading = neutralize_heading
        self.solver.neutralize_heading = neutralize_heading
        if safety_state_sync not in SAFETY_STATE_SYNC_MODES:
            raise ValueError(
                f"safety_state_sync must be one of {SAFETY_STATE_SYNC_MODES}, "
                f"not {safety_state_sync!r}"
            )
        self.safety_state_sync = safety_state_sync
        if safety_filter not in SAFETY_FILTER_MODES:
            raise ValueError(
                f"safety_filter must be one of {SAFETY_FILTER_MODES}, not {safety_filter!r}"
            )
        self.safety_filter = safety_filter
        if intruder_info not in INTRUDER_INFO_MODES:
            raise ValueError(
                f"intruder_info must be one of {INTRUDER_INFO_MODES}, not {intruder_info!r}"
            )
        self.intruder_info = intruder_info
        self.falsified_reports = self._check_falsified_reports(falsified_reports, belief_mode)
        # Episode-scoped; rebuilt by run(). Initialized here too, because some
        # callers drive communication rounds without run().
        self.attack_log = None
        self._falsified_by_id = {}
        self._falsified_copy_of = {}
        # After intruder_info, the red planner and the solver all exist.
        self._init_intruder_team(intruder_team)
        self.local_cbf_red_red = bool(local_cbf_red_red)
        # True (lcbf.1, historical local default): each red treats every
        # active blue -- including one pursuing it -- as a constant-velocity
        # obstacle at D, which in the tested configuration kept every pursuer
        # at D and suppressed contact. False: the red filter enforces red-red separation and its own
        # control bounds only; avoiding blues is left to the intruder's
        # ATTACK/EVADE decision. Blue constraints are unchanged either way.
        self.local_cbf_red_blue = bool(local_cbf_red_blue)
        self.local_cbf_log = []
        if safety_filter == "local":
            # Every local filter must read one snapshot taken before any agent
            # advances, and gamma is a declared, versioned choice tied to dt --
            # never inherited from the joint filter's defaults.
            if safety_state_sync != "frozen":
                raise ValueError('safety_filter="local" requires safety_state_sync="frozen"')
            if local_cbf_gamma is None:
                raise ValueError('safety_filter="local" requires an explicit local_cbf_gamma')
            if enable_convex_hull_containment:
                raise ValueError('safety_filter="local" does not support convex-hull containment')
            self.local_cbf_blue_cfg = LocalCBFConfig(
                gamma=float(local_cbf_gamma), dt=self.dt,
                a_max=self.blue_a_max, v_max=self.blue_v_max,
            )
            self.local_cbf_red_cfg = LocalCBFConfig(
                gamma=float(local_cbf_gamma), dt=self.dt,
                a_max=self.red_a_max, v_max=self.red_v_max,
            )
        # The three-task rule. True -- every historical configuration -- leaves the solver's
        # DEFEND exactly as it was. Without SCREEN, every DEFEND holds an
        # explicit station and nothing can screen implicitly.
        self.solver.defend_screening = screen_available(self.cbaa_task_set)
        self.solver_supports_cbf = bool(
            getattr(self.solver, "supports_one_step_cbf", False)
        )

        self.x_current = None
        self.red_state = None
        self.history_blues = []
        self.history_red = []
        self.history_cbf_slack = []
        self.neutralized = False
        self.neutralize_step = None
        self.neutralize_agent = None
        self.touchdown = False
        self.touchdown_step = None
        self.touchdown_red_index = None
        self.escaped = False
        self.escape_step = None
        self.collided = False
        self.collision_step = None
        self.collision_agents = None
        self.collision_kind = None
        self.defenders_won = False
        self.defenders_win_step = None
        self.defenders_win_agents = None
        self.red_outcomes = ["active"] * self.num_reds
        self.decoy_events = []  #: one entry per non-hostile destroyed or gone
        self.num_neutralized = 0
        self.num_escaped = 0
        self.blue_score = 0.0
        self.solver_failed = False
        self.solver_failure_step = None

    def _build_solver(self):
        solver_cls = self._select_solver_class()
        candidate_kwargs = {
            "blue_agent": self.blue,
            "red_agent": self.red,
            "horizon": self.horizon,
            "arena_size": self.arena_size,
            "D_safe_blue": self.D_safe_blue,
            "D_safe_blue_red": self.D_safe_blue_red,
            "D_safe_red": self.D_safe_red,
            "enable_blue_blue_cbf": self.enable_blue_blue_cbf,
            "enable_blue_red_cbf": self.enable_blue_red_cbf,
            "enable_convex_hull_containment": self.enable_convex_hull_containment,
            "blue_blue_cbf_slack_weight": self.blue_blue_cbf_slack_weight,
            "blue_red_cbf_slack_weight": self.blue_red_cbf_slack_weight,
            "red_cbf_slack_weight": self.red_cbf_slack_weight,
            "convex_hull_slack_weight": self.convex_hull_slack_weight,
            "defense_center": self.defense_center,
            "defense_polygon": self.defense_polygon,
            "defended_shape_points": self.defended_shape_points,
            "patrol_rng": self.rng.patrol,
        }
        unsupported = sorted(set(candidate_kwargs) - self._accepted_solver_kwargs(solver_cls))
        if unsupported:
            raise TypeError(
                f"{solver_cls.__name__} accepts none of {unsupported}. These would "
                "previously have been dropped in silence -- a config value that "
                "appears to have no effect because it never arrives. Either the "
                "solver should accept them or they should not be sent."
            )
        return solver_cls(**candidate_kwargs)

    @staticmethod
    def _accepted_solver_kwargs(solver_cls):
        """Every keyword the solver can actually consume, including via the
        mixins it forwards **kwargs to.

        Signature inspection of the solver alone is not enough: it declares
        **kwargs and hands them to CBFFilterMixin, so the CBF's own parameters
        (D_safe_*, the slack weights) never appear in the solver's signature.
        Filtering on that signature is what silently discarded every CBF safety
        distance and slack weight Simulation was configured with -- see

        """
        accepted = set()
        for klass in inspect.getmro(solver_cls):
            init = klass.__dict__.get("__init__")
            if init is None:
                continue
            for name, param in inspect.signature(init).parameters.items():
                if param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
                    continue
                if name == "self":
                    continue
                accepted.add(name)
        return accepted

    def _select_solver_class(self):
        return DecentralizedPursuitEvasionSolver

    def _build_perimeter_geometry(self):
        center = np.array([0.82 * self.arena_size, 0.82 * self.arena_size], dtype=float)
        angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        base_radius = 0.08 * self.arena_size
        radial_scale = 1.0 + 0.18 * np.cos(3.0 * angles) + 0.08 * np.sin(5.0 * angles)
        radii = base_radius * radial_scale
        shape = center + np.column_stack(
            [radii * np.cos(angles), radii * np.sin(angles)]
        )
        polygon_stride = max(1, len(shape) // 12)
        polygon = center + 1.15 * (shape[::polygon_stride] - center)
        self.defended_shape_points = shape
        self.defense_polygon = polygon
        self.defense_center = center
        self.defender_spawn_points = self._sample_polygon_perimeter(
            polygon,
            self.num_blues,
        )

    @staticmethod
    def _sample_polygon_perimeter(polygon, n_points):
        vertices = np.asarray(polygon, dtype=float)
        if n_points <= 0:
            return np.zeros((0, 2), dtype=float)
        closed_vertices = np.vstack([vertices, vertices[0]])
        edge_vectors = np.diff(closed_vertices, axis=0)
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        perimeter = float(np.sum(edge_lengths))
        distances = np.linspace(0.0, perimeter, n_points, endpoint=False)
        anchors = []
        cumulative = np.concatenate([[0.0], np.cumsum(edge_lengths)])
        for distance_along in distances:
            edge_idx = np.searchsorted(cumulative[1:], distance_along, side="right")
            edge_start = vertices[edge_idx]
            edge_vector = edge_vectors[edge_idx]
            edge_length = max(edge_lengths[edge_idx], 1e-9)
            offset = distance_along - cumulative[edge_idx]
            anchors.append(edge_start + (offset / edge_length) * edge_vector)
        return np.asarray(anchors, dtype=float)

    @staticmethod
    def _point_in_polygon(point, polygon, boundary_tol=1e-9):
        """Is ``point`` in the CLOSED polygon (interior or boundary)?

        The breach-test rule.
        The original guarded the ray-crossing division with
        ``max(y_prev - y_curr, 1e-9)``, which also replaced every NEGATIVE
        denominator (every edge whose y increases) with 1e-9. Such an edge then
        counted as a crossing always or never, whatever the point's x. A band
        outside the defended region read as inside (false breaches that ended
        episodes), and much of the region read as outside (missed breaches).

        * **Boundary:** a point within ``boundary_tol`` of any edge is inside,
          so a red that reaches the perimeter has breached. It is decided
          first, explicitly, because a ray cast is arbitrary on the boundary.
        * **Interior:** even-odd ray casting towards +x. An edge is considered
          only if it straddles the point's y (``(y_curr > y) != (y_prev > y)``),
          so horizontal edges are excluded and the signed denominator
          ``y_prev - y_curr`` is never zero where it is used. It is not
          clamped.

        Independent of vertex order, and correct for concave polygons.
        """
        x, y = float(point[0]), float(point[1])
        vertices = np.asarray(polygon, dtype=float)
        x_prev, y_prev = vertices[-1]
        for x_curr, y_curr in vertices:
            dx, dy = x_curr - x_prev, y_curr - y_prev
            length2 = dx * dx + dy * dy
            t = 0.0 if length2 == 0.0 else min(1.0, max(0.0, ((x - x_prev) * dx + (y - y_prev) * dy) / length2))
            if np.hypot(x - (x_prev + t * dx), y - (y_prev + t * dy)) <= boundary_tol:
                return True
            x_prev, y_prev = x_curr, y_curr
        inside = False
        x_prev, y_prev = vertices[-1]
        for x_curr, y_curr in vertices:
            if (y_curr > y) != (y_prev > y):
                x_cross = x_curr + (x_prev - x_curr) * (y - y_curr) / (y_prev - y_curr)
                if x < x_cross:
                    inside = not inside
            x_prev, y_prev = x_curr, y_curr
        return inside

    # Rejection budget per red. The measured rejection rate at the default
    # +-10deg is 12.7% of draws (large-radius reds near the box's far corner
    # are the ones that can rotate into a wall), so exhausting 200 draws is
    # not something a sane configuration does -- it means the requested offset
    # is too wide for the geometry, and saying so loudly beats silently
    # narrowing the distribution.
    _MAX_BEARING_DRAWS = 200

    def _randomize_red_bearings(self, red_positions, min_red_spawn_dist):
        """``multi_bearing``: rotate each red's spawn about the defended region.

        A *one-variable* manipulation, and deliberately built to be one. Each
        red keeps the radius it was already assigned -- the draw from the
        ``spawn`` stream is untouched, so the standoff distribution is
        identical to ``symmetric`` by construction -- and only its bearing
        moves. The offset is drawn per red, independently, because that is what
        creates the belief *disagreement* the scenario-geometry rule is after: two reds on different
        bearings can be separated by more than one blue's 35-unit sensor range,
        so blue A sees red 1 while blue B sees red 2. A single per-episode
        rotation would move both reds together and produce shared ignorance
        instead, which is not the same experiment.

        Offsets come from the ``scenario`` stream (see simulation/rng.py), which
        is always an independent Generator and which ``symmetric`` never draws
        from. That is what keeps ``symmetric`` byte-identical rather than merely
        unchanged-in-expectation.

        Rejection is on the *offset only*, never the radius, so the realized
        distribution is exactly ``delta | radius ~ Uniform`` over the offsets
        that keep the red inside the arena and at least ``min_red_spawn_dist``
        from every red already placed.
        """
        if self.multi_bearing_max_offset_deg <= 0.0 or not red_positions:
            return list(red_positions)

        center = np.asarray(self.defense_center, dtype=float)
        max_offset = np.radians(self.multi_bearing_max_offset_deg)
        margin = 0.5
        lo, hi = self.arena_min + margin, self.arena_max - margin

        rotated = []
        for index, red_pos in enumerate(red_positions):
            radial = np.asarray(red_pos, dtype=float) - center
            radius = float(np.linalg.norm(radial))
            bearing = float(np.arctan2(radial[1], radial[0]))
            for _ in range(self._MAX_BEARING_DRAWS):
                delta = float(self.rng.scenario.uniform(-max_offset, max_offset))
                angle = bearing + delta
                candidate = center + radius * np.array(
                    [np.cos(angle), np.sin(angle)], dtype=float
                )
                if np.any(candidate < lo) or np.any(candidate > hi):
                    continue
                if any(
                    np.linalg.norm(candidate - prev) < min_red_spawn_dist
                    for prev in rotated
                ):
                    continue
                rotated.append(candidate)
                break
            else:
                raise RuntimeError(
                    f"multi_bearing: could not place red {index} within "
                    f"{self._MAX_BEARING_DRAWS} draws at radius {radius:.1f} "
                    f"(multi_bearing_max_offset_deg="
                    f"{self.multi_bearing_max_offset_deg}, "
                    f"min_red_spawn_dist={min_red_spawn_dist}). Either the "
                    f"defended region at {tuple(center)} leaves too narrow a "
                    f"feasible arc at this radius -- lower "
                    f"multi_bearing_max_offset_deg -- or too many reds are "
                    f"competing for it. Failing rather than retrying without "
                    f"the constraint, because a silently narrowed spawn "
                    f"distribution reads as a null result."
                )
        return rotated

    def setup_agents(self):
        arena_margin = 0.5

        min_red_spawn_dist = 4.0
        # Reflecting defense_center through the arena's center puts red's
        # spawn box diametrically opposite the defended region, rather
        # than near the arena's middle -- same half-width (0.10*arena_size)
        # as the old fixed [0.45, 0.65] window, just recentered.
        red_spawn_center = self.arena_size - self.defense_center
        red_spawn_half_width = 0.10 * self.arena_size
        red_positions = []
        for _ in range(self.num_real_reds):
            red_pos = None
            while red_pos is None or any(
                np.linalg.norm(red_pos - prev) < min_red_spawn_dist
                for prev in red_positions
            ):
                red_pos = np.array(
                    [
                        self.rng.spawn.uniform(
                            red_spawn_center[0] - red_spawn_half_width,
                            red_spawn_center[0] + red_spawn_half_width,
                        ),
                        self.rng.spawn.uniform(
                            red_spawn_center[1] - red_spawn_half_width,
                            red_spawn_center[1] + red_spawn_half_width,
                        ),
                    ],
                    dtype=float,
                )
            red_positions.append(red_pos)

        # Applied here, between position and state, so that the spawn heading
        # derived below ("always faces the target directly") follows the
        # rotated position for free. Moving it after the state loop would leave
        # every red facing the *nominal* 225deg approach axis regardless of
        # where it actually spawned, which is a second manipulation, not this
        # one. Blue's spawn heading is deliberately NOT rotated: blues keep
        # facing the nominal threat axis in both families, and at +-10deg the
        # red stays inside blue's 30deg FOV half-angle, so this withholds no
        # detection that `symmetric` would have granted.
        if self.scenario_family == "multi_bearing":
            red_positions = self._randomize_red_bearings(
                red_positions, min_red_spawn_dist
            )

        red_state_list = []
        for red_pos in red_positions:
            angle_e = self.rng.spawn.random() * 2 * np.pi
            vx_e = self.red.v_max * np.cos(angle_e)
            vy_e = self.red.v_max * np.sin(angle_e)
            row = [red_pos[0], red_pos[1], vx_e, vy_e]
            if self.red.include_heading:
                # Spawn heading always faces the target directly,
                # independent of the (separately randomized) initial
                # velocity direction above.
                to_target = self.defense_center - red_pos
                theta_0 = (
                    float(np.arctan2(to_target[1], to_target[0]))
                    if np.linalg.norm(to_target) > 1e-9
                    else 0.0
                )
                row.append(theta_0)
            red_state_list.append(row)
        self.red_state = np.array(red_state_list, dtype=float).reshape(-1)

        x0_list = []
        for spawn_point in self.defender_spawn_points:
            outward_dir = np.asarray(spawn_point, dtype=float) - self.defense_center
            outward_norm = float(np.linalg.norm(outward_dir))
            if outward_norm > 1e-9:
                outward_dir = outward_dir / outward_norm
            else:
                outward_dir = np.array([1.0, 0.0], dtype=float)
            spawn_base = np.asarray(spawn_point, dtype=float) + 8.0 * outward_dir
            px = np.clip(
                spawn_base[0] + self.rng.spawn.uniform(-0.75, 0.75),
                arena_margin,
                self.arena_max - arena_margin,
            )
            py = np.clip(
                spawn_base[1] + self.rng.spawn.uniform(-0.75, 0.75),
                arena_margin,
                self.arena_max - arena_margin,
            )
            row = [px, py, 0.0, 0.0]
            if self.blue.include_heading:
                # Face the reflection of defense_center through the
                # arena's center -- the same point red's spawn box is
                # centered on -- so blue starts out looking toward where
                # the threat actually spawns, not just radially outward
                # from its own guard post.
                row.append(
                    sensing.compute_bearing(
                        [px, py], [0.0, 0.0], self.arena_size - self.defense_center, True,
                        self.fov_bearing_speed_threshold,
                    )
                )
            x0_list.append(row)
        self.x_current = np.asarray(x0_list, dtype=float).reshape(-1)
        if self.decoy_spec is not None:
            # After every real-red and blue draw, from the non-hostile contacts' own stream:
            # the real reds and blues spawn exactly as without non-hostile contacts.
            decoy_rows = self._spawn_decoys()
            self.red_state = np.concatenate(
                [self.red_state, np.asarray(decoy_rows, dtype=float).reshape(-1)])
        self._spawn_birds()
        self._spawn_civilians()
        self._build_contact_roster()

    def _random_inert_spawn_state(self, v_max):
        """One [px, py, vx, vy] draw for a harmless inert entity (bird or
        civilian drone): uniform position in the inner 80% of the arena,
        random heading, speed uniform in [0.3, 1.0] * v_max. Uses np.random
        (the global RNG), matching the pre-existing bird-spawn draw exactly."""
        pos = self.rng.spawn.uniform(0.1 * self.arena_size, 0.9 * self.arena_size, size=2)
        angle = self.rng.spawn.random() * 2 * np.pi
        speed = self.rng.spawn.uniform(0.3, 1.0) * v_max
        vel = speed * np.array([np.cos(angle), np.sin(angle)])
        return [pos[0], pos[1], vel[0], vel[1]]

    def _spawn_birds(self):
        """Spawns 0..num_bird_slots harmless non-hostile agents (independent Bernoulli
        draw per slot -- the original additive mechanic, unchanged) plus
        self.num_substitution_birds more (from the three-way per-slot draw in
        __init__, zero unless num_contact_slots was set). Both are motion-model
        identical, so they're merged into one bird_state array/kind label.
        Birds are tactically inert -- pure random-walk motion, never
        neutralized/escaped/counted toward any win condition."""
        bird_states = []
        for _ in range(self.num_bird_slots):
            if self.rng.spawn.random() >= self.bird_spawn_prob:
                continue
            bird_states.append(self._random_inert_spawn_state(self.bird_v_max))
        for _ in range(self.num_substitution_birds):
            bird_states.append(self._random_inert_spawn_state(self.bird_v_max))
        self.num_birds = len(bird_states)
        self.bird_state = (
            np.array(bird_states, dtype=float).reshape(-1)
            if bird_states
            else np.zeros(0)
        )

    def _spawn_civilians(self):
        """Spawns self.num_civilians harmless civilian-drone non-hostile contacts (from
        the three-way per-slot draw in __init__, zero unless num_contact_slots
        was set) -- same inert motion model as birds, but a deliberately
        harder non-hostile kind visually (a civilian drone is still a drone, unlike
        a bird's plainly different silhouette): the two are meant to be
        distinguishable in the same image pool by something other than
        shape."""
        civilian_states = [
            self._random_inert_spawn_state(self.civilian_v_max)
            for _ in range(self.num_civilians)
        ]
        self.civilian_state = (
            np.array(civilian_states, dtype=float).reshape(-1)
            if civilian_states
            else np.zeros(0)
        )

    @staticmethod
    def _step_random_walk(state, count, v_max, a_max, dt, arena_min, arena_max, rng):
        """Bounded random-walk update shared by birds and civilians: small
        random acceleration, velocity/position integration, and reflection
        off the arena boundary.

        rng is the caller's "inert" stream (see simulation/rng.py) rather than
        the global np.random -- these draws happen every step, interleaved with
        ReconController's patrol draws, so sharing a stream with them is exactly
        the coupling the stream split exists to remove."""
        state = state.copy()
        for b in range(count):
            idx = b * 4
            px, py, vx, vy = state[idx : idx + 4]
            ax, ay = np.clip(
                rng.normal(0.0, 0.5 * a_max, size=2), -a_max, a_max
            )
            vx = float(np.clip(vx + ax * dt, -v_max, v_max))
            vy = float(np.clip(vy + ay * dt, -v_max, v_max))
            px, py = px + vx * dt, py + vy * dt
            if px < arena_min or px > arena_max:
                vx = -vx
                px = np.clip(px, arena_min, arena_max)
            if py < arena_min or py > arena_max:
                vy = -vy
                py = np.clip(py, arena_min, arena_max)
            state[idx : idx + 4] = [px, py, vx, vy]
        return state

    def _step_birds(self):
        self.bird_state = self._step_random_walk(
            self.bird_state, self.num_birds, self.bird_v_max, self.bird_a_max,
            self.dt, self.arena_min, self.arena_max, self.rng.inert,
        )

    def _step_civilians(self):
        self.civilian_state = self._step_random_walk(
            self.civilian_state, self.num_civilians, self.civilian_v_max,
            self.civilian_a_max, self.dt, self.arena_min, self.arena_max,
            self.rng.inert,
        )

    def _build_contact_roster(self):
        """Builds the fixed, per-episode (kind, index) -> contact_id mapping
        used by vision planners: reds, birds, and civilian drones shuffled
        together so a contact's id carries no positional hint about its true
        identity, plus a stable image assignment per contact.

        Uses a local random.Random instance (seeded from self.seed, offset to
        avoid correlating with anything else), never numpy's global RNG, so
        this is a strict no-op with respect to every other seeded draw in the
        episode -- Experiment I's spawn/motion sequence is unaffected whether
        or not this method runs.
        """
        local_rng = random.Random(self.seed + 4242)

        roster = (
            [("red", j) for j in range(self.num_reds)]
            + [("bird", b) for b in range(self.num_birds)]
            + [("civilian", c) for c in range(self.num_civilians)]
        )
        local_rng.shuffle(roster)
        self.contact_roster = roster

        image_pools = {"bird": _BIRD_IMAGES, "civilian": _CIVILIAN_IMAGES}
        self.contact_images = {}
        for contact_id, (kind, _idx) in enumerate(roster):
            pool = image_pools.get(kind, _RED_DRONE_IMAGES)
            if pool:
                self.contact_images[contact_id] = local_rng.choice(pool)

    def _active_contact_states(self):
        """contact_id -> current 4-state (position+velocity only -- Experiment
        II's vision planner never needs/expects bearing), excluding reds that
        have already been resolved (neutralized/escaped) -- birds/civilians are
        always shown, since they never resolve. NOT detection-filtered --
        callers that build a PlannerObservation must additionally filter this
        down to detected_contact_indices (see _compute_detected_indices)."""
        states = {}
        for contact_id, (kind, idx) in enumerate(self.contact_roster):
            if kind == "red":
                if self.red_outcomes[idx] != "active":
                    continue
                states[contact_id] = self.red_state[idx * self.red_nx : idx * self.red_nx + 4]
            elif kind == "civilian":
                states[contact_id] = self.civilian_state[idx * 4 : idx * 4 + 4]
            else:
                states[contact_id] = self.bird_state[idx * 4 : idx * 4 + 4]
        return states

    def _active_contact_images(self, active_contact_ids):
        return {
            contact_id: path
            for contact_id, path in self.contact_images.items()
            if contact_id in active_contact_ids
        }

    def _resolve_target_override(self, target_contact_ids):
        """Maps a role planner's {blue_id: contact_id} (PlannerOutput
        .target_contact_ids) to {blue_id: 4-state} for solve_decentralized, and
        counts how many resolve to a bird or civilian drone (a wasted
        NEUTRALIZE commitment -- neither is ever in red_outcomes, so
        engaging one can never earn neutralize credit).

        Returns ``(target_override, target_override_ids)``. The second is the
        subset of ``target_contact_ids`` that actually resolved, carried so the
        solver's effective-target log can name what it steered at -- the solver
        is handed positions and cannot recover the id from them.
        """
        target_override = {}
        target_override_ids = {}
        bird_engagements_this_cycle = 0
        civilian_engagements_this_cycle = 0
        for blue_id, contact_id in target_contact_ids.items():
            if contact_id is None or not (0 <= contact_id < len(self.contact_roster)):
                continue
            kind, idx = self.contact_roster[contact_id]
            target_override_ids[blue_id] = int(contact_id)
            if kind == "red":
                target_override[blue_id] = self.red_state[idx * self.red_nx : idx * self.red_nx + 4]
            elif kind == "civilian":
                target_override[blue_id] = self.civilian_state[idx * 4 : idx * 4 + 4]
                civilian_engagements_this_cycle += 1
            else:
                target_override[blue_id] = self.bird_state[idx * 4 : idx * 4 + 4]
                bird_engagements_this_cycle += 1
        self.num_bird_engagements += bird_engagements_this_cycle
        self.num_civilian_engagements += civilian_engagements_this_cycle
        return target_override, target_override_ids

    def _compute_bearings(self, states, count, nx):
        """Direct read of each agent's persisted bearing (state[...+4]) --
        theta is now genuine state, evolved by omega via the agent's own
        dynamics (Agent.include_heading), not recomputed from velocity each
        cycle. Every controller currently regularizes omega toward 0, so
        theta stays at its spawn-time value until a real recon controller
        starts spending omega -- see sensing.compute_bearing for that
        spawn-time convention."""
        return {idx: float(states[idx * nx + 4]) for idx in range(count)}

    def _compute_detected_indices(self, sim_step):
        """Returns (detected_red_indices, detected_blue_indices,
        detected_contact_indices, per_blue_detected_reds): which opposing
        agents (and, for the vision-augmented Experiment II contact roster,
        which contact_ids) each team's combined field of view can see this
        cycle. Disabled blues and non-active reds are excluded from being
        both observers (their sensors are presumably down/gone) and
        detection targets (already-resolved reds aren't real threats).
        Own-team visibility is unaffected by this -- it's handled entirely in
        planner._format_observation, not here.

        detected_contact_indices reuses the same blue_observers/blue_bearings
        against _active_contact_states()'s positions (reds+birds+civilians,
        keyed by contact_id) instead of red_targets -- the same FOV primitive,
        just fed a different target set. This is always computed (cheap, a
        handful of contacts at most); it's simply unused/empty whenever
        contact_roster is red-only (Experiment I, or Experiment II with no
        detection filtering applied by the caller).

        per_blue_detected_reds ({blue_idx: set(red_idx)}) is the per-observer
        analogue of detected_red_indices' team-wide union -- which reds each
        specific blue's own sensor cone currently sees, used to drive
        bearing-tracking (a blue only turns to face a red it personally
        detected, not one only some other blue on the team can see).

        Also updates self.coverage_tracker from the same blue observer/
        bearing data, since "is this cell within some blue's FOV" is the
        same primitive as "is this opposing agent within some blue's FOV" --
        just tested against grid cell centers instead of red positions."""
        active_blues = [i for i in range(self.num_blues) if not self.blue_disabled[i]]
        active_reds = [j for j in range(self.num_reds) if self.red_outcomes[j] == "active"]

        blue_bearings = self._compute_bearings(self.x_current, self.num_blues, self.blue_nx)
        red_bearings = self._compute_bearings(self.red_state, self.num_reds, self.red_nx)

        blue_observers = [(i, self.x_current[i * self.blue_nx : i * self.blue_nx + 2]) for i in active_blues]
        red_targets = [(j, self.red_state[j * self.red_nx : j * self.red_nx + 2]) for j in active_reds]
        detected_red_indices = sensing.team_detected_indices(
            blue_observers, blue_bearings, red_targets, self.fov_half_angle_rad, self.fov_range,
        )
        per_blue_detected_reds = sensing.per_observer_detected_indices(
            blue_observers, blue_bearings, red_targets, self.fov_half_angle_rad, self.fov_range,
        )

        red_observers = [(j, self.red_state[j * self.red_nx : j * self.red_nx + 2]) for j in active_reds
                         if j < self.num_real_reds]  # non-hostile contacts sense nothing
        blue_targets = [(i, self.x_current[i * self.blue_nx : i * self.blue_nx + 2]) for i in active_blues]
        detected_blue_indices = sensing.team_detected_indices(
            red_observers, red_bearings, blue_targets, self.fov_half_angle_rad, self.fov_range,
        )

        self.coverage_tracker.update(
            sim_step, blue_observers, blue_bearings, self.fov_half_angle_rad, self.fov_range,
        )

        contact_targets = [
            (contact_id, state[0:2]) for contact_id, state in self._active_contact_states().items()
        ]
        detected_contact_indices = sensing.team_detected_indices(
            blue_observers, blue_bearings, contact_targets, self.fov_half_angle_rad, self.fov_range,
        )

        return detected_red_indices, detected_blue_indices, detected_contact_indices, per_blue_detected_reds

    def _update_beliefs(self, step, per_blue_detected_reds, blue_observers, blue_bearings):
        """Advance every blue's belief one step, per self.belief_mode.

        Beliefs are keyed by contact_id. For Paper 1 contact_id is the red index
        (identity map); keeping the namespace separate now means Experiment II's
        shuffled roster needs no belief-layer change later -- see

        """
        active_blues = [i for i in range(self.num_blues) if not self.blue_disabled[i]]

        if self.belief_mode == "ground_truth":
            # Every agent believes every red, at its true state, with age 0.
            # Deliberately includes already-resolved reds, because the pre-M2
            # nearest-red heuristic scanned all N_red without filtering on
            # outcome -- reproducing that exactly is what makes this mode a
            # valid regression baseline.
            for i in active_blues:
                for j in range(self.num_reds):
                    self.belief_stores[i].observe_target(
                        j, self.red_state[j * self.red_nx : j * self.red_nx + self.red_nx], step
                    )
                for cell_idx in range(len(self.coverage_tracker.centers)):
                    if self.coverage_tracker.last_seen_step[cell_idx] == step:
                        self.belief_stores[i].observe_coverage(cell_idx, step)
            return

        # First-hand observation: only what THIS agent's own sensor saw.
        for i in active_blues:
            for j in per_blue_detected_reds.get(i, ()):  # own cone only
                self.belief_stores[i].observe_target(
                    j, self.red_state[j * self.red_nx : j * self.red_nx + self.red_nx], step
                )
            own_bearing = blue_bearings[i]
            own_pos = self.x_current[i * self.blue_nx : i * self.blue_nx + 2]
            for cell_idx, center in enumerate(self.coverage_tracker.centers):
                if sensing.is_detected(
                    own_pos, own_bearing, center, self.fov_half_angle_rad, self.fov_range
                ):
                    self.belief_stores[i].observe_coverage(cell_idx, step)

        if self.belief_mode == "shared":
            # Perfect, instantaneous, unlimited-range sharing -- the upper
            # bracket of Experiment 0. Merged per contact by max observed_at
            # (perception/belief/merge.py), so this is the same entry point the
            # real communication layer will use at M4.
            union_targets = {}
            for i in active_blues:
                for cid, belief in self.belief_stores[i].targets.items():
                    current = union_targets.get(cid)
                    if (
                        current is None
                        or belief.provenance.observed_at > current.provenance.observed_at
                    ):
                        union_targets[cid] = belief
            union_coverage = None
            for i in active_blues:
                store = self.belief_stores[i]
                if store.coverage is None:
                    continue
                if union_coverage is None:
                    union_coverage = store.coverage.copy()
                else:
                    union_coverage.merge(store.coverage)
            for i in active_blues:
                self.belief_stores[i].merge_targets(union_targets)
                self.belief_stores[i].merge_coverage(union_coverage)

        elif self.belief_mode == "networked":
            self._exchange_beliefs_over_network(step, active_blues)

    def _update_red_beliefs(self, step):
        """intruder_info="shared_beliefs": each active red records the blues
        its OWN sensor cone sees (true state at that step, as for blues), then
        the ideal intruder channel merges every red's targets by latest
        observed_at -- the same rule as belief_mode="shared" -- into every
        active red's store. Timestamps are preserved, so beliefs age and are
        extrapolated from when they were seen. Disabled blues are never
        detection targets, matching _compute_detected_indices."""
        active_reds = [j for j in range(self.num_real_reds) if self.red_outcomes[j] == "active"]
        active_blues = [i for i in range(self.num_blues) if not self.blue_disabled[i]]
        red_bearings = self._compute_bearings(self.red_state, self.num_reds, self.red_nx)
        per_red = sensing.per_observer_detected_indices(
            [(j, self.red_state[j * self.red_nx : j * self.red_nx + 2]) for j in active_reds],
            red_bearings,
            [(i, self.x_current[i * self.blue_nx : i * self.blue_nx + 2]) for i in active_blues],
            self.fov_half_angle_rad,
            self.fov_range,
        )
        for j in active_reds:
            for i in sorted(per_red.get(j, ())):
                self.red_belief_stores[j].observe_target(
                    i, self.x_current[i * self.blue_nx : i * self.blue_nx + self.blue_nx], step
                )
        union = {}
        for j in active_reds:
            for cid, belief in self.red_belief_stores[j].targets.items():
                current = union.get(cid)
                if current is None or belief.provenance.observed_at > current.provenance.observed_at:
                    union[cid] = belief
        for j in active_reds:
            self.red_belief_stores[j].merge_targets(union)

    def _red_believed_blues(self, step):
        """{red_id: {blue_id: state predicted to now}} for fresh beliefs, per
        active red. Identical across reds under the ideal channel; kept per
        agent so a consumer never reads a team object."""
        return {
            j: {
                cid: belief.predicted_state(step, self.dt)
                for cid, belief in store.targets.items()
                if belief.is_fresh(step, self.belief_validity_age_steps)
            }
            for j, store in self.red_belief_stores.items()
            if self.red_outcomes[j] == "active"
        }

    # -- falsified teammate reports (stress test) -------------------------
    #
    # One compromised blue, chosen before the episode from the "attack" RNG
    # stream (never the commander, so matched arms share it), adds a fixed
    # displacement to the POSITION in its own outgoing first-hand reports about
    # one selected red. Identity, velocity and the original observation time
    # are kept; its own store is never altered; auction messages, commands,
    # other senders and the safety layer are untouched. Recipients merge by
    # the ordinary rule (later observed_at wins, ties keep what is held), so
    # nothing forces acceptance. The altered record is a distinct object that
    # travels by reference, which is how relays, deliveries and acceptances
    # are traced without changing the belief record format.

    _FALSIFIED_KEYS = {"onset_step", "duration_steps", "displacement", "report_cap"}

    def _check_falsified_reports(self, spec, belief_mode):
        if spec is None:
            return None
        spec = dict(spec)
        if set(spec) != self._FALSIFIED_KEYS:
            raise ValueError(
                f"falsified_reports needs exactly {sorted(self._FALSIFIED_KEYS)}, got {sorted(spec)}"
            )
        if belief_mode != "networked":
            raise ValueError("falsified_reports alters network reports: needs belief_mode='networked'")
        disp = np.asarray(spec["displacement"], dtype=float).reshape(2)
        out = {
            "onset_step": int(spec["onset_step"]),
            "duration_steps": int(spec["duration_steps"]),
            "displacement": (float(disp[0]), float(disp[1])),
            "report_cap": int(spec["report_cap"]),
        }
        if out["onset_step"] < 0 or out["duration_steps"] <= 0 or out["report_cap"] <= 0:
            raise ValueError(f"falsified_reports values out of range: {out}")
        return out

    def _reset_falsified_reports(self):
        self.attack_log = None
        self._falsified_by_id = {}
        self._falsified_copy_of = {}
        if self.falsified_reports is None:
            return
        eligible = [i for i in range(self.num_blues) if i != self.commander_id]
        sender = int(self.rng.attack.choice(eligible))
        self.attack_log = {
            "sender": sender,
            "eligible_senders": eligible,
            "spec": dict(self.falsified_reports),
            "target_red": None,
            "target_selected_step": None,
            "reports_altered": 0,
            "stop_reason": None,
            "alterations": [],   # one per altered outgoing report (counts against the cap)
            "deliveries": [],    # every delivery of an altered record, own send or relay
            "accepted": [],      # deliveries after which the recipient holds the altered record
            "distinct_observations_falsified": 0,  # underlying sightings, not broadcasts
            # Per blue, per step, three different things -- never conflated:
            "stored_false_steps": {},      # holds the altered record
            # And fresh under belief_validity_age_steps. "Fresh" only: not
            # necessarily eligible for every task, nor actually used by control.
            # With belief_validity_age_steps=None nothing expires by age -- a
            # substantive assumption of the stress test, recorded below.
            "fresh_false_steps": {},
            "commitment_steps": {},        # ...and NEUTRALIZE-committed to the target on it
            "first_commitment_step": {},
            "belief_validity_age_steps": self.belief_validity_age_steps,
            "delivery_seq": 0,
        }

    def _outgoing_targets(self, agent, store, step):
        """The targets ``agent`` broadcasts this round: its store, unless it is
        the compromised sender inside the attack window."""
        targets = dict(store.targets)
        log = self.attack_log
        if log is None or agent != log["sender"] or log["stop_reason"] is not None:
            return targets
        spec = self.falsified_reports
        if step < spec["onset_step"]:
            return targets
        if step >= spec["onset_step"] + spec["duration_steps"]:
            log["stop_reason"] = "window_expired"
            return targets
        if log["reports_altered"] >= spec["report_cap"]:
            log["stop_reason"] = "report_cap"
            return targets
        if log["target_red"] is None:
            eligible = sorted(
                cid for cid, b in targets.items()
                if b.provenance.origin_agent == agent
                and b.provenance.observed_at >= spec["onset_step"]
                and 0 <= cid < self.num_reds
            )
            if not eligible:
                return targets
            log["target_red"], log["target_selected_step"] = eligible[0], int(step)
        red = log["target_red"]
        true = targets.get(red)
        if true is None or true.provenance.origin_agent != agent:
            return targets  # only the sender's own first-hand reports
        fake = self._falsified_copy_of.get(id(true))
        if fake is None:
            state = true.state.copy()
            state[0:2] = state[0:2] + np.asarray(spec["displacement"])
            fake = TargetBelief(contact_id=true.contact_id, state=state, provenance=true.provenance)
            # Keep both alive so ids are never reused.
            self._falsified_copy_of[id(true)] = fake
            self._falsified_by_id[id(fake)] = (fake, true)
            log["distinct_observations_falsified"] += 1
        targets[red] = fake
        log["reports_altered"] += 1
        log["alterations"].append({
            "step": int(step), "red": int(red),
            "observed_at": int(true.provenance.observed_at),
            "true_position": true.state[0:2].tolist(),
            "reported_position": fake.state[0:2].tolist(),
            "report_index": log["reports_altered"],
        })
        return targets

    def _falsified_entries(self, targets):
        if not self._falsified_by_id:
            return {}
        return {cid: b for cid, b in targets.items() if id(b) in self._falsified_by_id}

    def _held_kind(self, held, incoming):
        if held is None:
            return "none"
        if held is incoming:
            return "same_altered_record"
        if id(held) in self._falsified_by_id:
            return "other_altered_record"
        t_held, t_in = held.provenance.observed_at, incoming.provenance.observed_at
        if t_held < t_in:
            return "older_record"
        return "same_time_record" if t_held == t_in else "newer_record"

    def _log_falsified_delivery(self, step, receiver, sender, entries, held_before,
                                message_id=None):
        """Every delivery of an altered record; ``accepted`` only when this
        merge newly installed it (a duplicate of a record already held is a
        delivery, not a new acceptance). ``delivery_seq`` is the processing
        order and ``held`` what the receiver held just before, which together
        reproduce equal-timestamp outcomes (a tie keeps the held record)."""
        log = self.attack_log
        for cid, belief in entries.items():
            log["delivery_seq"] += 1
            held = held_before.get(cid)
            rec = {
                "step": int(step), "receiver": int(receiver), "sender": int(sender),
                "relayed": int(sender) != log["sender"], "red": int(cid),
                "observed_at": int(belief.provenance.observed_at),
                "origin_agent": int(belief.provenance.origin_agent),
                "delivery_seq": log["delivery_seq"],
                "message_id": message_id,
                "held": self._held_kind(held, belief),
                "held_observed_at": None if held is None else int(held.provenance.observed_at),
                "held_origin_agent": None if held is None else int(held.provenance.origin_agent),
            }
            log["deliveries"].append(rec)
            if (self.belief_stores[receiver].targets.get(cid) is belief
                    and held_before.get(cid) is not belief):
                log["accepted"].append(rec)

    def _log_falsified_commitments(self, step):
        """Per step, per active blue: holds the altered record (stored); that
        record is fresh under the configured belief-age limit (fresh -- not
        necessarily eligible for every task or used by control);
        and NEUTRALIZE-committed to the target while holding it. A diagnostic
        only: a false report can also move priorities, stations, recon or a
        commander's assignment, so no commitment here is not "no effect"."""
        log = self.attack_log
        if log is None:
            return
        if log["stop_reason"] is None and self.blue_disabled[log["sender"]]:
            log["stop_reason"] = "sender_inactive"
        if log["target_red"] is None:
            return
        red = log["target_red"]
        for i in range(self.num_blues):
            if self.blue_disabled[i]:
                continue
            held = self.belief_stores[i].targets.get(red)
            if held is None or id(held) not in self._falsified_by_id:
                continue
            log["stored_false_steps"][i] = log["stored_false_steps"].get(i, 0) + 1
            if held.is_fresh(step, self.belief_validity_age_steps):
                log["fresh_false_steps"][i] = log["fresh_false_steps"].get(i, 0) + 1
            if self.role_assignment.get(i) != BlueRole.NEUTRALIZE:
                continue
            if self.blue_target_red_idx.get(i) != red:
                continue
            log["commitment_steps"][i] = log["commitment_steps"].get(i, 0) + 1
            log["first_commitment_step"].setdefault(i, int(step))

    def _exchange_beliefs_over_network(self, step, active_blues):
        """One communication round: deliver, merge, then send.

        That order is's within-step order (3. enqueue deliveries due
        now, 4. merge them) and it is not interchangeable with send-then-deliver.
        Because ``Network`` is double-buffered, what is delivered here was sent
        on the *previous* round, so a sighting reaches a one-hop neighbour one
        comm tick after it is made. Sending *after* merging is what makes
        multi-hop gossip work without an explicit relay step: an agent's next
        broadcast includes what it just learned, so information advances exactly
        one hop per round, which is the property M4's line-graph test pins.

        Relaying by re-broadcasting a merged store cannot rejuvenate anything --
        ``merge_target`` compares ``observed_at`` and ``Provenance`` carries it
        unchanged, so a twice-relayed sighting is still as old as the moment the
        sensor made it.

        Coverage is exchanged in the same payload; it merges by per-cell max
        ``swept_at``, which is monotone, so duplicate and reordered delivery are
        harmless on that field too.
        """
        if self.network is None or step % self.comm_interval_steps != 0:
            return

        positions = {
            i: self.x_current[i * self.blue_nx : i * self.blue_nx + 2]
            for i in range(self.num_blues)
        }
        disabled = [i for i in range(self.num_blues) if self.blue_disabled[i]]
        self.network.begin_round(
            positions=positions, disabled=disabled, comm_tick=step // self.comm_interval_steps
        )
        self.comm_stats["rounds"] += 1
        self._record_connectivity()

        # 3 + 4: deliver what was sent last round, and merge it.
        commands_applied = set()
        auction_inbox = {}
        for agent, envelopes in self.network.deliver().items():
            if self.blue_disabled[agent]:
                continue
            store = self.belief_stores[agent]
            for envelope in envelopes:
                if envelope.message_type == AUCTION_MESSAGE:
                    # Allocation traffic: the existing epoch filter applies
                    # (comms.receive), and the payload goes to the auction --
                    # never to the belief store. the rule that auction traffic never creates a belief: auction traffic cannot
                    # create a belief, and routing it here is half of that.
                    if self.cbaa_runtime is None:
                        continue
                    if is_stale_allocation(envelope, self.cbaa_runtime.epoch):
                        self.cbaa_runtime.note_stale_transport_drop()
                        continue
                    auction_inbox.setdefault(agent, []).append(envelope.payload)
                    continue
                if envelope.message_type == ASSIGNMENT_COMMAND:
                    # Commands are broadcast and addressed in the payload, so
                    # most arrivals are for somebody else. An intermediary
                    # CARRIES an order it is not addressed to; it never reads
                    # it into its own commitment, and the acceptance rule is
                    # evaluated against the addressee's history, not the
                    # relay's.
                    if envelope.payload.agent_id != agent:
                        self._queue_command_relay(agent, envelope, step)
                        continue
                    # Ordered against this receiver's own accepted history,
                    # never the global epoch -- contract Sec 3a, and the reason
                    # commands are not routed through comms.receive's
                    # allocation filter.
                    if self._receive_command(agent, envelope, step):
                        commands_applied.add(agent)
                    continue
                # No epoch filtering here, deliberately: this is belief traffic,
                # and an observation sent during an earlier planning epoch may
                # still be the freshest sighting this agent has. Staleness is
                # decided by observed_at inside merge_targets. See comms.receive.
                payload = envelope.payload
                falsified_in = self._falsified_entries(payload.targets)
                held_before = {cid: store.targets.get(cid) for cid in falsified_in}
                store.merge_targets(payload.targets)
                if falsified_in:
                    self._log_falsified_delivery(
                        step, agent, envelope.sender, falsified_in, held_before,
                        message_id=envelope.message_id,
                    )
                if payload.coverage is not None:
                    store.merge_coverage(payload.coverage)
                self.comm_stats["merged_targets"] += len(payload.targets)
                self.comm_stats["delivered"] += 1

        if commands_applied:
            # Act on the order now, not at the next planning epoch -- which,
            # once the commander is gone, may never come. The assignment in
            # force is rebuilt from what every agent has accepted.
            self._apply_assignment(
                self._commitment_output(), step, self._believed_targets(step),
                # Only the agents that just accepted an order. See the method's
                # docstring: re-resolving the whole team here would give B a
                # fresher aim point than A, which is a motion-layer difference
                # in an information experiment.
                resolve_only=commands_applied,
            )

        if self.cbaa_runtime is not None and self.cbaa_runtime.transport == "network":
            changed = self.cbaa_runtime.receive_round(auction_inbox, active_blues)
            if changed:
                self._apply_cbaa(step, changed)

        self._flush_outbound_commands(step)
        # One hop further, in the send phase of the round they arrived on --
        # the same rule the belief re-broadcast below follows.
        self._relay_commands(step)

        # Then send: the snapshot now includes anything merged above, which is
        # what advances information one hop per round.
        for agent in active_blues:
            store = self.belief_stores[agent]
            if not store.targets and store.coverage is None:
                continue
            self.network.send(
                make_envelope(
                    TARGET_UPDATE,
                    agent,
                    BeliefUpdate(
                        targets=self._outgoing_targets(agent, store, step),
                        coverage=store.coverage.copy() if store.coverage else None,
                    ),
                    sent_at=step,
                    ttl=self.comm_message_ttl,
                )
            )

        # The auction re-broadcasts its merged state every round, like belief
        # gossip, so it needs no relay step: consensus itself carries a winning
        # bid one hop further per round.
        if self.cbaa_runtime is not None and self.cbaa_runtime.transport == "network":
            for agent in active_blues:
                state = self.cbaa_runtime.outgoing(agent)
                if state is None:
                    break
                self.network.send(
                    make_envelope(
                        AUCTION_MESSAGE,
                        agent,
                        state,
                        sent_at=step,
                        planning_epoch=state.epoch,
                        ttl=self.comm_message_ttl,
                    )
                )

    @staticmethod
    def _fresh_comm_stats():
        """The zeroed per-episode communication counters.

        A factory rather than a dict literal at the reset site, because the
        shape is needed in more than one place -- ``reset_episode_state`` builds
        an episode's counters, and tests that drive
        ``_exchange_beliefs_over_network`` directly (to stage a disabled relay,
        say) have to build the same ones. Two literals drift, and the second one
        silently loses whichever field was added last.

        Traffic counters first, then the connectivity mediators.
        Only ever written in the networked branch, so the other three belief
        modes are untouched by any of it.
        """
        return {
            "rounds": 0,
            "delivered": 0,
            "merged_targets": 0,
            "connected_rounds": 0,
            "lcc_fraction_sum": 0.0,
            "n_components_sum": 0,
            "diameter_sum": 0,
            "diameter_rounds": 0,
            # Closed partition runs, in rounds. A run still open when the
            # episode ends is held in _open_partition_run and flushed by
            # comm_metrics(), which is why that method does not mutate state.
            "partition_runs": [],
            "_open_partition_run": 0,
        }

    def _record_connectivity(self):
        """Sample the mediators for the round just frozen.

        Called immediately after ``begin_round`` and before any delivery, so it
        describes the graph the round's sends actually used -- not a graph the
        agents had already moved off.

        **Why this is sampled at all.** forbids reporting results
        against the configured ``R_comm`` alone. M0 went further and measured
        that *partition duration*, not graph diameter, is the mediator that
        carries the effect: the diameter is 1.0-1.9 hops at every range worth
        running, so it cannot discriminate between conditions, while mean
        partition length moves 15 -> 94 steps across the same grid.

        A "partition" here is a maximal run of consecutive communication rounds
        in which the live blues do not form one component. Disabled blues are
        not nodes (``connectivity_metrics`` filters them), matching M0.
        """
        metrics = self.network.connectivity_metrics()
        stats = self.comm_stats
        stats["lcc_fraction_sum"] += metrics["largest_component_fraction"]
        stats["n_components_sum"] += metrics["n_components"]
        if metrics["connected"]:
            stats["connected_rounds"] += 1
            # Diameter is None when disconnected, which is the honest answer
            # rather than 0 -- so it is averaged over connected rounds only.
            if metrics["diameter"] is not None:
                stats["diameter_sum"] += metrics["diameter"]
                stats["diameter_rounds"] += 1
            if stats["_open_partition_run"]:
                stats["partition_runs"].append(stats["_open_partition_run"])
                stats["_open_partition_run"] = 0
        else:
            stats["_open_partition_run"] += 1

    def comm_metrics(self):
        """Per-episode communication mediators, or ``None`` outside networked mode.

        ``None`` rather than a dict of zeros, deliberately: the other three
        belief modes construct no ``Network`` at all (see ``self.network``), so
        there is no graph to report on and a zeroed dict would read as "measured
        and found disconnected". A caller that gets ``None`` knows the question
        did not apply.

        Durations are reported in **steps** as well as rounds. Rounds are the
        protocol's clock and steps are the mission's, and the two differ by
        ``comm_interval_steps`` -- which is precisely the axis the communication conditions sweeps, so
        reporting only rounds would make the period sweep look flat by
        construction.
        """
        if self.network is None:
            return None
        stats = self.comm_stats
        rounds = max(1, stats["rounds"])
        # Flush a run still open at termination without mutating the counters:
        # the episode ended mid-partition, and dropping that run would bias
        # mean duration downward exactly in the conditions that partition most.
        runs = list(stats["partition_runs"])
        if stats["_open_partition_run"]:
            runs.append(stats["_open_partition_run"])
        per_round_steps = self.comm_interval_steps
        return {
            "rounds": stats["rounds"],
            "delivered": stats["delivered"],
            "merged_targets": stats["merged_targets"],
            "connected_fraction": stats["connected_rounds"] / rounds,
            "mean_lcc_fraction": stats["lcc_fraction_sum"] / rounds,
            "mean_n_components": stats["n_components_sum"] / rounds,
            "mean_diameter_when_connected": (
                stats["diameter_sum"] / stats["diameter_rounds"]
                if stats["diameter_rounds"]
                else float("nan")
            ),
            "n_partitions": len(runs),
            "mean_partition_steps": (
                float(np.mean(runs)) * per_round_steps if runs else 0.0
            ),
            "max_partition_steps": (max(runs) * per_round_steps if runs else 0),
            "partitioned_fraction": 1.0 - stats["connected_rounds"] / rounds,
            "comm_interval_steps": per_round_steps,
        }

    def commander_is_active(self):
        """Is the designated commander still alive?

        The commander is a **fixed** blue, chosen at construction and never
        re-elected -- decision recorded in
 No handover machinery
        exists at the command-delivery conditions and none is implied: an election protocol is itself a
        communication mechanism and would confound the very contrast B - A is
        meant to isolate.
        """
        return not self.blue_disabled[self.commander_id]

    def _plan_centralized_from_belief(self, step):
        """Conditions A and B: one planner call, on ONE agent's belief store.

        A and B share this code path entirely. The only difference between them
        is ``belief_mode``, which is the information axis:

        * ``"shared"`` -- every active store holds the same merged contents, so
          the commander's store *is* the pooled team belief. That is condition A.
        * ``"networked"`` -- the commander's store holds only what the network
          delivered to it. That is condition B.

        Routing both through one function is what makes the 2x2 a factorial
        rather than four separately-built configurations, and it is why the
        test that compares conditions A and C checks plumbing instead of arithmetic.

        Returns ``None`` when the commander is disabled and the failure rule is
        active; the caller then holds the previous commitment.
        """
        if self.commander_failure_stops_planning and not self.commander_is_active():
            return None

        # A disabled commander's store is frozen and it is not an assignable
        # agent, so planning from it would rank a team that does not include
        # whoever is actually alive. Without the failure rule, fall back to the
        # lowest-indexed active blue so the condition still runs.
        source = self.commander_id
        if self.blue_disabled[source]:
            active = [i for i in range(self.num_blues) if not self.blue_disabled[i]]
            if not active:
                return None
            source = active[0]
        # Recorded so condition B issues commands from whoever actually planned.
        # With the failure rule off and the commander dead, the substitute above
        # is the real source; sending its orders under the dead commander's id
        # would have Network.send drop every one of them as a disabled sender,
        # freezing the team silently and for the wrong reason.
        self.planning_source = source

        states = {
            i: self.x_current[i * self.blue_nx : (i + 1) * self.blue_nx]
            for i in range(self.num_blues)
            if not self.blue_disabled[i]
        }
        view = self.belief_stores[source].snapshot(
            now=step,
            dt=self.dt,
            self_state=states[source],
            coverage_centers=self.coverage_tracker.centers,
            # Ground truth in every belief mode -- the decentralization rule commitment 1, unchanged by
            # The command-delivery conditions. The centralized conditions delay target information, not
            # teammate state.
            teammates={j: s for j, s in states.items() if j != source},
        )
        return self.role_planner.plan_team(
            view, max_age=self.belief_validity_age_steps
        )

    # -- condition B: assignment delivery over the network ------------------
    #
    # The command-delivery conditions contract Sec 3. B - A mixes two latencies -- information reaching the
    # commander, and assignments reaching the agents -- and this design CANNOT
    # separate them. That is declared, not discovered, and the ledger below is
    # what makes the outbound half observable rather than only declared: every
    # command records the step it was generated, handed to the network,
    # received and applied.

    def _reset_command_state(self):
        """The per-episode command ledger. Empty in every mode but B."""
        # {agent: the AssignmentCommand in force}. An agent with no entry has
        # never been given an order and holds the initial assignment, which at
        # episode start is empty -- it never idles waiting.
        self.agent_commitments = {}
        # {agent: planning_epoch of its newest ACCEPTED command}. This, not the
        # wall clock, is what a command is ordered against -- see
        # comms.receive.classify_command and contract Sec 3a.
        self.last_accepted_command_epoch = {}
        # Generated but not yet handed to the network: commands wait for the
        # next communication round, so they are transmitted on the graph that
        # is actually in force when they go out rather than one the team has
        # since moved off.
        self._outbound_commands = []
        # (relay_agent, envelope) pairs received this round and addressed to
        # somebody else, to be carried one hop further in this round's send
        # phase. See _relay_commands.
        self._pending_relays = []
        # {agent: {(commander, epoch, addressee)}} already carried by that
        # agent. One relay per agent per command is what terminates the flood.
        self._relayed_commands = defaultdict(set)
        # One row per command, carrying the whole generation -> handoff ->
        # receipt -> application chain. A list rather than a counter because
        # the delivery checks assert on individual commands and a mean latency
        # cannot show that a particular late command was applied.
        self.command_log = []
        self.command_stats = {
            "generated": 0,
            "self_applied": 0,
            "handed_to_network": 0,
            "not_sent_sender_disabled": 0,
            "no_link": 0,
            "delivered": 0,
            "accepted": 0,
            "rejected_superseded": 0,
            "rejected_too_old": 0,
            "relayed": 0,
            "relay_dropped": 0,
            "relay_suppressed_duplicate": 0,
        }

    def _issue_commands(self, planned, step, believed_targets, generated=True):
        """Turn the commander's team plan into one command per subordinate.

        Returns the assignment **actually in force**, which is the accumulation
        of commands each agent has accepted -- NOT the plan just computed. That
        is the whole of condition B: the commander's decision and the team's
        behaviour are separated by the network.

        ``generated=False`` means no planning epoch happened (the commander is
        gone and the failure rule is active). Nothing new is issued. Commands
        already handed to the network still travel and are still applied, per
        contract Sec 3b -- the network does not know the sender died -- so
        survivors may keep updating for a few rounds after the commander is
        already lost. That is more precise than freezing the team at the
        commander's final epoch.

        The commander applies its own command immediately. It is the source; no
        hop is needed for it to talk to itself, and routing it through the
        network would model a radio link an agent has with itself.
        """
        if generated:
            epoch = step // self.planning_interval_steps
            source = self.planning_source
            for agent in range(self.num_blues):
                if self.blue_disabled[agent]:
                    continue
                command = AssignmentCommand(
                    planning_epoch=epoch,
                    commander_id=source,
                    agent_id=agent,
                    issued_at=step,
                    role=planned.roles.get(agent),
                    target_red_id=planned.target_red_ids.get(agent),
                    target_contact_id=planned.target_contact_ids.get(agent),
                    recon_target_region=planned.recon_target_regions.get(agent),
                    defend_target=planned.defend_targets.get(agent),
                    task_id=planned.task_ids.get(agent),
                )
                self.command_stats["generated"] += 1
                self.command_log.append({
                    "epoch": epoch,
                    "agent": agent,
                    # "self" never crosses a link and so carries no latency;
                    # it is kept apart from "network" rather than averaged in,
                    # which would dilute the measured delivery latency by the
                    # commander's own always-zero row.
                    "via": "self" if agent == source else "network",
                    "task_id": planned.task_ids.get(agent),
                    "defend_target": planned.defend_targets.get(agent),
                    "generated_at": step,
                    "sent_at": None,
                    "received_at": None,
                    "applied_at": None,
                    # 0 for the commander's own order, which crosses no link.
                    "hops": 0 if agent == source else None,
                    "outcome": None,
                })
                entry = self.command_log[-1]
                if agent == source:
                    entry["sent_at"] = step
                    entry["received_at"] = step
                    self.command_stats["self_applied"] += 1
                    self._accept_command(command, step, entry)
                else:
                    self._outbound_commands.append((command, entry))
        return self._commitment_output()

    def _flush_outbound_commands(self, step):
        """Hand queued commands to the network, inside a frozen round.

        Called twice per step, and both calls matter. Planning runs *after* the
        step's communication round, so the call right after generation is what
        puts a command into the round already open at that step -- the same
        outbox this step's belief broadcasts went into, delivered one round
        later by the double buffer, exactly like every other message. Without
        it a command would wait for the *next* round and cost two comm
        intervals instead of one.

        A command generated at a planning step with no round open is held until
        the next round and transmitted there, so it always travels on the graph
        in force when it actually goes out rather than one the team has since
        moved off.

        A command whose commander died between generation and this call is
        dropped by ``Network.send``, which refuses a disabled sender. Sender
        death cancels a transmission **not yet under way**; it does not cancel
        one already under way, and it is not a guarantee of delivery either --
        recipient death, an absent link and the acceptance rule all still apply.
        It is counted rather than silent, because a drop and a rejection are
        different findings.

        **Commands are broadcast, addressed in the payload**, not sent
        point-to-point. ``Network.send`` intersects an addressed envelope with
        the sender's immediate neighbours, so an addressed command could only
        ever reach a one-hop subordinate -- while beliefs reach the whole
        connected component by re-broadcast. Condition B would then have
        measured multi-hop inbound information against direct-only outbound
        control, which is a protocol restriction this workstream never declared
        and which bites hardest in exactly the finite-range regimes the communication conditions selected.
        See ``_relay_commands``.
        """
        if not self._outbound_commands:
            return
        if self.network is None or step % self.comm_interval_steps != 0:
            # No round is open this step. Hold, rather than send against a
            # graph frozen at some earlier step.
            return
        pending, self._outbound_commands = self._outbound_commands, []
        for sequence, (command, entry) in enumerate(pending):
            if self.blue_disabled[command.commander_id]:
                entry["outcome"] = "not_sent_commander_disabled"
                self.command_stats["not_sent_sender_disabled"] += 1
                continue
            reached = self.network.send(
                make_envelope(
                    ASSIGNMENT_COMMAND,
                    command.commander_id,
                    command,
                    sent_at=step,
                    # BROADCAST, with the addressee in the payload. See the
                    # docstring: an addressed envelope cannot leave the
                    # commander's own neighbourhood.
                    recipient=BROADCAST,
                    planning_epoch=command.planning_epoch,
                    # A backstop only. Reach is decided by _command_hop_limit
                    # against hop_count, explicitly, so that the bound is the
                    # declared parameter and not an envelope field whose
                    # belief-layer namesake is inert.
                    ttl=int(self.num_blues) + 1,
                    sequence=sequence,
                )
            )
            if reached:
                entry["sent_at"] = step
                self.command_stats["handed_to_network"] += 1
            else:
                # The commander had no neighbour at all this round. Nothing
                # left it, and the addressee holds its last accepted command.
                entry["outcome"] = "no_link"
                self.command_stats["no_link"] += 1

    def _command_hop_limit(self):
        """The most links a command may traverse before it stops being carried.

        Counted in **links traversed at receipt**, not forwards remaining: a
        direct neighbour receives at 1 hop, an agent two links away at 2. That
        is the quantity a reader means by "how far can an order reach", and
        expressing the cap any other way makes ``command_max_hops=1`` silently
        allow a two-link delivery, which is what the first version did.

        Deliberately **not** ``comm_message_ttl``. That parameter is inert
        ([08 Flaw C]()) -- it is
        read only by ``Network.forward``, which production code never calls --
        and it is pinned inert by a test so that no reader reports a
        TTL-bounded relay the code does not have. Resurrecting it here for
        commands only would make that pin true in the letter and false in the
        spirit, and would silently couple command reach to a belief-layer
        setting that does nothing to beliefs.

        With ``command_max_hops`` unset the limit is the team size, which
        **cannot bind**: relaying is already bounded by each agent forwarding a
        given command at most once, so a command crosses at most
        ``num_blues - 1`` links however large this is. The bound is a property
        of the flood, not a tuning parameter, and it is stated that way rather
        than left to a number someone has to choose.
        """
        if self.command_max_hops is not None:
            return int(self.command_max_hops)
        return int(self.num_blues)

    def _relay_commands(self, step):
        """Advance commands addressed to someone else one hop further.

        The same shape as the belief layer's multi-hop: an agent re-broadcasts
        what it received, in the send phase of the round it received it, so
        information advances exactly one hop per communication round and the
        double buffer makes that true rather than aspirational.

        Three properties, each of which a test pins:

        * **An intermediary never executes an order addressed elsewhere.** It
          carries it. Only ``command.agent_id`` applies a command, and the
          acceptance rule is evaluated against *that* agent's history.
        * **Relaying preserves origin, issue time and epoch.**
          ``MessageEnvelope.forwarded_by`` rewrites only ``sender``,
          ``hop_count`` and ``ttl``; it never touches the payload, so
          ``commander_id``, ``issued_at`` and ``planning_epoch`` survive any
          number of hops. A relay cannot rejuvenate an order any more than it
          can rejuvenate an observation.
        * **A relay that is still alive keeps carrying an order after the
          commander is gone.** The sender of a relayed envelope is the relay,
          not the dead commander, so ``Network.send``'s disabled-sender rule
          does not apply to it -- which is contract Sec 3b holding at range
          rather than only for direct neighbours.

        Each agent forwards a given command **once**, which is what terminates
        the flood: at most one relay per agent per command.
        """
        if not self._pending_relays:
            return
        pending, self._pending_relays = self._pending_relays, []
        for relay_agent, envelope in pending:
            if self.blue_disabled[relay_agent] or envelope.ttl <= 0:
                self.command_stats["relay_dropped"] += 1
                continue
            forwarded = envelope.forwarded_by(relay_agent, BROADCAST, step)
            if self.network.send(forwarded):
                self.command_stats["relayed"] += 1
            else:
                self.command_stats["relay_dropped"] += 1

    def _queue_command_relay(self, agent, envelope, step):
        """This agent holds an order for someone else. Carry it, do not read it."""
        command = envelope.payload
        key = (command.commander_id, command.planning_epoch, command.agent_id)
        if key in self._relayed_commands[agent]:
            # Already carried by this agent. Forwarding it again would make the
            # flood non-terminating without moving it any further.
            self.command_stats["relay_suppressed_duplicate"] += 1
            return
        # This envelope arrived having crossed hop_count + 1 links; the copy
        # this agent would send arrives one link further out.
        if envelope.hop_count + 2 > self._command_hop_limit() or envelope.ttl <= 0:
            self.command_stats["relay_dropped"] += 1
            return
        self._relayed_commands[agent].add(key)
        self._pending_relays.append((agent, envelope))

    def _receive_command(self, agent, envelope, step):
        """Apply the command-delivery conditions Sec 3a acceptance rule to one delivered command."""
        self.command_stats["delivered"] += 1
        command = envelope.payload
        if command.agent_id != agent:
            # Commands are addressed, never broadcast, so this cannot happen
            # today. Refused rather than tolerated because the alternative is
            # one agent's ledger quietly recording another's orders, which
            # would corrupt the ordering rule for both.
            raise ValueError(
                f"command for blue {command.agent_id} was delivered to blue "
                f"{agent}; commander assignments are addressed, not broadcast"
            )
        entry = self._command_log_entry(command)
        if entry is not None and entry["received_at"] is None:
            # FIRST receipt, not the last. Under a broadcast flood the same
            # command arrives again by longer paths -- a direct neighbour hears
            # it from the commander on one round and from its relaying
            # teammates on the next -- and those later copies are superseded
            # duplicates, not the delivery. Overwriting here would have made
            # the measured latency the length of the slowest path that happened
            # to exist rather than the one the order actually travelled.
            entry["received_at"] = step
            # Links traversed, so a direct neighbour reads 1 and an agent two
            # links out reads 2. Recorded because generation-to-receipt is
            # waiting-for-a-round PLUS one round per hop, and the headline
            # "one communication round" is only the direct case.
            entry["hops"] = envelope.hop_count + 1
        verdict = classify_command(
            envelope,
            latest_accepted_epoch=self.last_accepted_command_epoch.get(agent),
            now=step,
            max_age_steps=self.command_max_age_steps,
        )
        if verdict == COMMAND_ACCEPTED:
            self._accept_command(command, step, entry)
            return True
        if entry is not None and entry["outcome"] != COMMAND_ACCEPTED:
            # A command that was accepted stays accepted. Under a broadcast
            # flood its own later copies arrive and are correctly judged
            # superseded, and letting them rewrite the outcome would record
            # every delivered order as rejected.
            entry["outcome"] = verdict
        self.command_stats[
            "rejected_superseded" if verdict == COMMAND_SUPERSEDED else "rejected_too_old"
        ] += 1
        return False

    def _accept_command(self, command, step, entry=None):
        agent = command.agent_id
        self.agent_commitments[agent] = command
        self.last_accepted_command_epoch[agent] = command.planning_epoch
        self.command_stats["accepted"] += 1
        if entry is not None:
            entry["applied_at"] = step
            entry["outcome"] = COMMAND_ACCEPTED

    def _command_log_entry(self, command):
        for entry in reversed(self.command_log):
            if entry["agent"] == command.agent_id and entry["epoch"] == command.planning_epoch:
                return entry
        return None

    def _commitment_output(self):
        """The assignment in force: what each agent has been told, not what the
        commander last decided.

        Agents are inserted in ascending order, matching ``plan_team`` and
        ``_plan_per_agent`` -- ``tests/conftest.py``'s fingerprint hashes
        ``repr(dict)``, which is insertion-ordered, so an arbitrary order here
        would surface as a spurious episode mismatch between conditions holding
        identical assignments.
        """
        out = PlannerOutput()
        for agent in sorted(self.agent_commitments):
            command = self.agent_commitments[agent]
            if command.role is not None:
                out.roles[agent] = command.role
            if command.target_red_id is not None:
                out.target_red_ids[agent] = command.target_red_id
            if command.target_contact_id is not None:
                out.target_contact_ids[agent] = command.target_contact_id
            if command.recon_target_region is not None:
                out.recon_target_regions[agent] = command.recon_target_region
            if command.defend_target is not None:
                out.defend_targets[agent] = command.defend_target
            if command.task_id is not None:
                out.task_ids[agent] = command.task_id
        return out

    def command_metrics(self):
        """Per-episode command traffic, for the command-delivery conditions results table."""
        applied = [
            e for e in self.command_log
            if e["applied_at"] is not None and e["via"] == "network"
        ]
        latencies = [e["applied_at"] - e["generated_at"] for e in applied]
        return {
            **self.command_stats,
            "n_delivered_commands_applied": len(applied),
            "mean_command_latency_steps": (
                float(np.mean(latencies)) if latencies else float("nan")
            ),
            "max_command_latency_steps": max(latencies) if latencies else 0,
        }

    # -- task-model commander: pending commanded contacts ------------------

    def _reset_commander_task_model(self):
        self.role_planner.bind(TaskContext.from_simulation(
            self, slots=self.commander_slots, task_set=self.cbaa_task_set
        ))
        self.commander_pending = {}
        self.commander_pending_log = []
        self._commander_in_force = None
        self._commander_resolved = {}

    def _executable_commander_output(self, planned, step, believed_targets):
        """The commanded assignment as it can be EXECUTED by each recipient.

        A NEUTRALIZE task whose contact the recipient does not currently
        believe in (no fresh entry in its OWN store) is kept as the commanded
        task, marked pending, and executed as an explicit DEFEND(own station)
        until usable target information arrives. Never the nearest red, never
        ground truth. The commitment ledger is not touched.

        A commanded contact that has been RESOLVED (the same red_outcomes gate
        _apply_assignment already applies to every NEUTRALIZE target) is also
        executed as DEFEND(own station) until the next command -- otherwise the
        solver's nearest-believed fallback would pursue some other contact the
        commander never assigned. Logged as "commanded_contact_resolved"."""
        out = PlannerOutput(
            roles=dict(planned.roles),
            target_red_ids=dict(planned.target_red_ids),
            target_contact_ids=dict(planned.target_contact_ids),
            recon_target_regions=dict(planned.recon_target_regions),
            defend_targets=dict(planned.defend_targets),
            task_ids=dict(planned.task_ids),
        )
        pending_now = {}
        for agent, role in planned.roles.items():
            if self.blue_disabled[agent] or role != BlueRole.NEUTRALIZE:
                continue
            red = planned.target_red_ids.get(agent)
            resolved = red is not None and 0 <= red < self.num_reds and self.red_outcomes[red] != "active"
            if not resolved and red is not None and red in believed_targets.get(agent, {}):
                continue
            out.roles[agent] = BlueRole.DEFEND
            out.defend_targets[agent] = ("station", int(agent))
            out.target_red_ids.pop(agent, None)
            if resolved:
                key = (planned.task_ids.get(agent), red)
                if self._commander_resolved.get(agent) != key:
                    self._commander_resolved[agent] = key
                    self.commander_pending_log.append(
                        {"step": int(step), "agent": int(agent), "task_id": key[0],
                         "contact": red, "event": "commanded_contact_resolved"})
                continue
            pending_now[agent] = (planned.task_ids.get(agent), red)
        for agent, (task, red) in pending_now.items():
            if self.commander_pending.get(agent) != (task, red):
                self.commander_pending_log.append(
                    {"step": int(step), "agent": int(agent), "task_id": task,
                     "contact": red, "event": "pending_no_local_belief"})
        for agent, (task, red) in self.commander_pending.items():
            if agent not in pending_now:
                self.commander_pending_log.append(
                    {"step": int(step), "agent": int(agent), "task_id": task,
                     "contact": red, "event": "cleared"})
        self.commander_pending = pending_now
        return out

    def _activate_ready_pending(self, step, believed_targets):
        """Between epochs: a pending agent whose own belief now holds the
        commanded contact starts executing it, re-resolving only that agent."""
        ready = {a for a, (_, red) in self.commander_pending.items()
                 if red is not None and red in believed_targets.get(a, {})}
        if ready and self._commander_in_force is not None:
            self._apply_assignment(self._commander_in_force, step, believed_targets,
                                   resolve_only=ready)

    def _apply_assignment(self, planned, step, believed_targets, resolve_only=None):
        """Put a PlannerOutput into force: roles, targets and waypoints.

        Extracted from ``run`` unchanged. Every mode without command delivery calls it exactly
        once per planning epoch, so nothing about those modes moves. Condition B needs it callable a second time --
        a command can land between epochs, and the agent that accepted it must
        act on it then rather than at the next epoch it may never see.

        ``resolve_only`` restricts which agents' TARGET RESOLUTION is redone,
        and exists to keep B - A honest. The resolved override is a believed
        position extrapolated to the step it was resolved on
        (``believed_targets`` is ``predicted_state(step, dt)``), and between
        epochs it stays frozen at that value. So re-resolving the whole team
        every time any command lands would hand condition B a fresher aim point
        than condition A ever gets -- a **motion-layer** advantage created by
        the delivery path, in the one comparison that is supposed to isolate
        information.

        With it set, only those agents' entries are recomputed; everyone else
        keeps the override frozen from their own last application, exactly as
        under A. Roles and the recorded output are always taken whole, because
        they come from the commitment ledger and are already per-agent.

        ``resolve_only=None`` is the unrestricted path every pre-the command-delivery conditions mode takes.
        """
        if self.commander_task_model:
            self._commander_in_force = planned
            planned = self._executable_commander_output(planned, step, believed_targets)
        if self.unavailable_target == "station":
            # On the FULL output, like the commander conversion above: the
            # restricted copy below carries every role but only the
            # re-resolved agents' targets, so judged there every other
            # NEUTRALIZE agent would look targetless.
            planned = self._station_for_unavailable(planned, step, believed_targets)
        if resolve_only is not None:
            resolve_only = {int(a) for a in resolve_only}
            kept = PlannerOutput(
                roles=planned.roles,
                # Taken whole, like roles: a defend target names a contact or a
                # station, and the solver reads the agent's LIVE belief about it
                # each step -- there is no frozen aim point to keep honest.
                defend_targets=planned.defend_targets,
                task_ids=planned.task_ids,
                target_red_ids={
                    a: v for a, v in planned.target_red_ids.items() if a in resolve_only
                },
                target_contact_ids={
                    a: v for a, v in planned.target_contact_ids.items()
                    if a in resolve_only
                },
                recon_target_regions={
                    a: v for a, v in planned.recon_target_regions.items()
                    if a in resolve_only
                },
            )
            previous = (
                dict(self.target_override), dict(self.target_override_ids),
                dict(self.blue_target_red_idx), dict(self.recon_target_override),
                dict(self.target_override_applied),
            )
            self._put_assignment_in_force(kept, step, believed_targets)
            # Restore the untouched agents, then record the FULL output -- the
            # restricted copy above exists only to bound the re-resolution.
            for live, old in zip(
                (self.target_override, self.target_override_ids,
                 self.blue_target_red_idx, self.recon_target_override,
                 self.target_override_applied),
                previous,
            ):
                for agent, value in old.items():
                    if agent not in resolve_only:
                        live[agent] = value
            self.planner_output = planned
            return
        self._put_assignment_in_force(planned, step, believed_targets)

    def _station_for_unavailable(self, planned, step, believed_targets):
        """unavailable_target="station": a NEUTRALIZE commitment whose red is
        resolved, or which the agent does not believe in, executes DEFEND(own
        station) at this application. The commitment itself (auction state,
        commander ledger) is untouched; only what is executed changes. Same
        gates as the commander path: red_outcomes for resolution (the gate
        that already withholds the aim point), the agent's OWN fresh belief
        for availability. Contact-roster targets (Experiment II) pass through."""
        out = PlannerOutput(
            roles=dict(planned.roles),
            target_red_ids=dict(planned.target_red_ids),
            target_contact_ids=dict(planned.target_contact_ids),
            recon_target_regions=dict(planned.recon_target_regions),
            defend_targets=dict(planned.defend_targets),
            task_ids=dict(planned.task_ids),
        )
        for agent, role in planned.roles.items():
            if self.blue_disabled[agent] or role != BlueRole.NEUTRALIZE:
                continue
            if agent in planned.target_contact_ids:
                continue
            red = planned.target_red_ids.get(agent)
            if red is not None and 0 <= red < self.num_reds and self.red_outcomes[red] != "active":
                reason = "committed_contact_resolved"
            elif red is None or red not in believed_targets.get(agent, {}):
                reason = "committed_contact_not_believed"
            else:
                self._unavailable_logged.pop(agent, None)
                continue
            out.roles[agent] = BlueRole.DEFEND
            out.defend_targets[agent] = ("station", int(agent))
            out.target_red_ids.pop(agent, None)
            key = (planned.task_ids.get(agent), red, reason)
            if self._unavailable_logged.get(agent) != key:
                self._unavailable_logged[agent] = key
                self.unavailable_target_log.append(
                    {"step": int(step), "agent": int(agent), "task_id": key[0],
                     "contact": red, "event": reason})
        return out

    def _put_assignment_in_force(self, planned, step, believed_targets):
        """``_apply_assignment``'s body, unrestricted.

        Separated so the restricted path can reuse it **without recursing
        through the public method**. The recursion was not merely inelegant: it
        re-entered whatever `_apply_assignment` names on the instance, so any
        wrapper around it -- a test's, a future subclass's -- saw a phantom
        second call carrying ``resolve_only=None`` and could not tell it from a
        planning epoch.
        """
        self.role_assignment = planned.roles
        # The screening-task rule. Empty for every planner that predates the auction. A screened contact that
        # has already been resolved is dropped, exactly as target_override
        # gates NEUTRALIZE on red_outcomes below.
        self.defend_target_override = {
            agent: (kind, ident)
            for agent, (kind, ident) in planned.defend_targets.items()
            if kind == "station"
            or (0 <= ident < self.num_reds and self.red_outcomes[ident] == "active")
        }
        # Two independent target-assignment conventions can coexist:
        # target_contact_ids (Experiment II, contact_id space -- reds+
        # birds merged/shuffled) and target_red_ids (Experiment I, raw
        # red index space -- Exp I never obfuscates red identity, only
        # gates visibility via FOV). A given role_planner only ever
        # populates one of the two, so there's no collision.
        target_contact_ids = planned.target_contact_ids
        target_red_ids = planned.target_red_ids
        self.target_override, self.target_override_ids = (
            self._resolve_target_override(target_contact_ids)
        )
        belief_keys = {}
        for blue_id, red_id in target_red_ids.items():
            if (
                red_id is not None
                and 0 <= red_id < self.num_reds
                and self.red_outcomes[red_id] == "active"
            ):
                # The interceptor flies to where IT BELIEVES the target
                # is, not where the target actually is. Before M2 this
                # read self.red_state -- ground truth -- so a blue could
                # commit to a stale or falsified belief and still track
                # the real red perfectly. Closing
                # this is what makes a false-location attack able to
                # move a trajectory at all.
                believed = believed_targets.get(blue_id, {}).get(red_id)
                if believed is not None:
                    self.target_override[blue_id] = np.asarray(
                        believed, dtype=float
                    ).ravel()[0:4]
                    self.target_override_ids[blue_id] = int(red_id)
                    belief_keys[blue_id] = int(red_id)
        # The pursuer aim-point rule. When, and from which belief, each aim point was resolved:
        # {blue_id: (application step, belief key or None)}. The key is None
        # for a contact-roster entry (Experiment II), which is not a belief.
        # Read only by _neutralize_aim_points; the historical "frozen" mode
        # never looks at it.
        self.target_override_applied = {
            blue_id: (step, belief_keys.get(blue_id)) for blue_id in self.target_override
        }
        # {blue_id: red_idx} for bearing-tracking's tiebreak (see
        # _resolve_bearing_targets) -- unifies Experiment I's raw red
        # index space and Experiment II's contact_id space (resolved
        # through contact_roster, keeping only contacts that are
        # actually reds -- a targeted bird/civilian carries no
        # bearing-tracking priority). Reassigned in full each
        # planning cycle, same as role_assignment/target_override.
        self.blue_target_red_idx = {}
        for blue_id, red_id in target_red_ids.items():
            if red_id is not None and 0 <= red_id < self.num_reds and self.red_outcomes[red_id] == "active":
                self.blue_target_red_idx[blue_id] = red_id
        for blue_id, contact_id in target_contact_ids.items():
            if contact_id is None or not (0 <= contact_id < len(self.contact_roster)):
                continue
            kind, idx = self.contact_roster[contact_id]
            if kind == "red" and self.red_outcomes[idx] == "active":
                self.blue_target_red_idx[blue_id] = idx
        # RECON targeting: separate from target_override above
        # (which feeds NEUTRALIZE's intercept controller with a
        # 4-vector red/contact state) -- ReconController only ever
        # wants a 2D waypoint, resolved from a region id via
        # self.coverage_tracker rather than any agent's state.
        target_region_ids = planned.recon_target_regions
        # Held on self because the recorder reads it every step, not
        # only on planning steps -- same persistence the last_* reads
        # had, now from an object with a defined epoch.
        self.planner_output = planned
        self.recon_target_override = {
            blue_id: self.coverage_tracker.region_center(region_id)
            for blue_id, region_id in target_region_ids.items()
        }

    def _plan_per_agent(self, step):
        """One planner call per active blue, each on its own LocalWorldView.

        This is what makes Experiment 0's lower arm a decision-level condition
        rather than a controller-level one. With the team planner, role
        assignment stays centralized and fed by the team-wide detection union no
        matter what the agents believe, so "no target-observation sharing" has
        nothing to bite on.

        **Every view is frozen before any planning call runs.**
        requires that all of epoch k's decisions read beliefs captured at the
        same logical instant; if views were built lazily inside the loop, blue 1
        would see blue 0's same-epoch effects purely because the loop ran in that
        order. Snapshots copy, so this holds even though plan_one does not mutate
        the store -- the guarantee should not rest on the planner's good
        behavior.

        Teammate states are ground truth in every belief mode, by declaration
. Only *active* blues appear, matching the team
        planner's obs.disabled_blues filter -- the two paths must agree on who
        is assignable or the equivalence gate cannot hold.
        """
        active, views = self._frozen_views(step)

        merged = PlannerOutput()
        for i in active:
            out = self.role_planner.plan_one(
                views[i], max_age=self.belief_validity_age_steps
            )
            merged.roles.update(out.roles)
            merged.target_red_ids.update(out.target_red_ids)
            merged.target_contact_ids.update(out.target_contact_ids)
            merged.recon_target_regions.update(out.recon_target_regions)
        return merged

    def _frozen_views(self, step):
        """Every active blue's LocalWorldView, all frozen before anyone decides.

        Extracted from ``_plan_per_agent`` unchanged, so the per-agent planners
        and the auction freeze views by one piece of code rather than two that
        could drift.: epoch k's decisions read beliefs captured at
        one logical instant.
        """
        active = [i for i in range(self.num_blues) if not self.blue_disabled[i]]
        states = {
            i: self.x_current[i * self.blue_nx : (i + 1) * self.blue_nx] for i in active
        }
        views = {
            i: self.belief_stores[i].snapshot(
                now=step,
                dt=self.dt,
                self_state=states[i],
                coverage_centers=self.coverage_tracker.centers,
                teammates={j: s for j, s in states.items() if j != i},
            )
            for i in active
        }
        return active, views

    def _guard_auction_schedule(self, num_blues, stable_rounds, allow):
        """The communication-round schedule: reject a schedule whose MINIMUM rounds per planning
        epoch cannot meet the local-stability requirement, unless deliberately
        overridden for diagnostics.

        The auction resets at every planning epoch and an agent
        commits only after ``stable_rounds`` consecutive unchanged rounds, all
        inside one epoch. The round at a planning step belongs to the old epoch
        (testbed order), so epoch k of P steps, with a round every C steps,
        holds floor((k+1)P / C) - floor(kP / C) rounds. When C divides P that
        is P / C in every epoch. When it does not, epochs differ: they hold
        floor(P / C) or one more. (0.3 s against 1 s epochs is 3, 3, 4 rounds,
        not 3 in every epoch. 0.4 s is 2, 3.)

        The guard compares the MINIMUM, floor(P / C), with ``stable_rounds``.
        Below it, no agent can commit in any epoch that holds only the minimum,
        and such epochs recur throughout the episode. That identifies an
        insufficient minimum. It does NOT show that no epoch could ever settle:
        under a non-dividing period some epochs hold one round more. Only when
        even the maximum falls short (0.5 s: 2 rounds in every epoch) is
        commitment impossible everywhere. The auction-timing check
 found that case
        running whole episodes with nothing committed and nothing warning.

        ``stability_impossible`` keeps its recorded name, so saved records stay
        comparable. It means "impossible in the epochs holding the minimum",
        not "impossible in every epoch".

        It certifies nothing either way: enough round opportunities do not
        guarantee agreement under loss, partition or divergent task knowledge,
        and with auction traffic the usual minimum is higher still. Unsettled
        epochs remain a measured outcome.
        """
        stable = max(1, num_blues - 1) if stable_rounds is None else int(stable_rounds)
        min_rounds = self.planning_interval_steps // self.comm_interval_steps
        impossible = min_rounds < stable
        self.cbaa_runtime.min_rounds_per_epoch = int(min_rounds)
        self.cbaa_runtime.stability_impossible = bool(impossible)
        if impossible and not allow:
            cycle = rounds_per_epoch_cycle(self.planning_interval_steps, self.comm_interval_steps)
            short = sum(1 for n in cycle if n < stable)
            if max(cycle) < stable:
                reach = "No epoch holds enough rounds, so no agent would ever commit."
            else:
                reach = (
                    f"{short} of every {len(cycle)} epochs hold too few, so no agent can "
                    f"commit in those; the others hold up to {max(cycle)} rounds."
                )
            raise ValueError(
                f"allocation='cbaa' has an insufficient minimum under this schedule: a "
                f"planning epoch of {self.planning_interval_steps} steps with a communication "
                f"round every {self.comm_interval_steps} steps holds {cycle} rounds in "
                f"repeating epochs, as few as {min_rounds}, and local stability needs "
                f"{stable} consecutive unchanged rounds within one epoch. {reach} Pass "
                f"cbaa_allow_unsettleable=True only to diagnose that failure."
            )

    def station_heading_angles(self, stations):
        """{station id: heading} the run's station posture gives each station:
        the configured heading under "configured", else the outward heading
        (from the defended centre through the station)."""
        out = {}
        for k, p in stations:
            k = int(k)
            if self.configured_station_headings is not None:
                out[k] = float(self.configured_station_headings[k])
            else:
                v = np.asarray(p, dtype=float) - np.asarray(self.defense_center, dtype=float)
                out[k] = float(np.arctan2(v[1], v[0]))
        return out

    def _reset_allocation(self):
        """A fresh auction for a new episode, over this run's geometry and task
        set. Extracted from ``run`` unchanged except for the task set, so the
        context the auction actually generates from can be checked without
        running an episode."""
        from coordination.bidding import BidContext

        task_ctx = TaskContext.from_simulation(
            self, slots=self.cbaa_runtime.slots, task_set=self.cbaa_runtime.task_set
        )
        self.cbaa_runtime.reset(
            self.num_blues, task_ctx, BidContext.from_simulation(self, task_ctx)
        )

    def _plan_cbaa(self, step):
        """The auction: start epoch k's auction, and report what is in force now.

        Every agent bids from its own frozen view. What is returned is the set
        of COMMITMENTS, which under the rule that an auction may span planning epochs are still the previous epoch's until each
        agent's new auction settles locally -- so the new bids take effect in a
        later communication round, through ``_apply_cbaa``, not here.
        """
        active, views = self._frozen_views(step)
        self.cbaa_runtime.start_epoch(step, views, max_age=self.belief_validity_age_steps)
        return self.cbaa_runtime.output(active)

    def _apply_cbaa(self, step, changed):
        """Put changed auction commitments into force between planning epochs.

        ``resolve_only`` restricts target re-resolution to the agents whose
        commitment changed -- the rule command-delivery condition B established, so an agent
        whose commitment did not move keeps its aim point frozen from its own
        last application instead of getting a fresher one for free.
        """
        active = [i for i in range(self.num_blues) if not self.blue_disabled[i]]
        self._apply_assignment(
            self.cbaa_runtime.output(active),
            step,
            self._believed_targets(step),
            resolve_only=changed,
        )

    def _believed_targets(self, step):
        """{blue_id: {contact_id: predicted state}} for the solver.

        Positions are extrapolated from each belief's own observation time to
        now, so an agent acting on a 30-step-old sighting aims where the contact
        would be if it had kept going -- not where it actually is. With age 0
        this returns the observed state exactly.
        """
        out = {}
        for i in range(self.num_blues):
            store = self.belief_stores.get(i)
            if store is None:
                out[i] = {}
                continue
            out[i] = {
                cid: belief.predicted_state(step, self.dt)
                for cid, belief in store.targets.items()
                if belief.is_fresh(step, self.belief_validity_age_steps)
            }
        return out

    def _neutralize_aim_points(self, step, believed_targets):
        """The pursuer aim-point rule: the {blue_id: 4-state} NEUTRALIZE aim points for this solve.

        ``target_override`` holds each agent's aim point as resolved at its last
        assignment application: a believed state extrapolated to THAT step, then
        frozen (see _apply_assignment). NeutralizeController predicts the target
        from it at k = 0 = now, so under "frozen" -- the historical default,
        returned as the very same dict -- a state up to one planning epoch old
        is treated as current.

        * "aligned": the same frozen state, advanced at its own velocity from its
          application step to ``step``. No information newer than the
          application enters, so the command-delivery conditions rule that restricts re-resolution
          (resolve_only) keeps exactly its meaning.
        * "live": the CONTACT stays the one resolved at application; the aim
          point is the agent's current belief about it (``believed_targets``,
          already predicted to now), or the aligned snapshot if the agent holds
          no belief entry for it.

        Entries that did not come from a belief (Experiment II's contact roster,
        whose ids are not belief keys) are passed through unchanged in every
        mode.
        """
        self.neutralize_aim_sources = {}
        if self.neutralize_aim == "frozen":
            return self.target_override
        out = {}
        for blue_id, snapshot in self.target_override.items():
            applied_step, key = self.target_override_applied.get(blue_id, (None, None))
            if key is None or applied_step is None:
                out[blue_id] = snapshot
                self.neutralize_aim_sources[blue_id] = "snapshot"
                continue
            if self.neutralize_aim == "live":
                believed = believed_targets.get(blue_id, {}).get(key)
                if believed is not None:
                    out[blue_id] = np.asarray(believed, dtype=float).ravel()[0:4].copy()
                    self.neutralize_aim_sources[blue_id] = "belief"
                    continue
            aligned = np.asarray(snapshot, dtype=float).ravel()[0:4].copy()
            aligned[0:2] = aligned[0:2] + (step - applied_step) * self.dt * aligned[2:4]
            out[blue_id] = aligned
            self.neutralize_aim_sources[blue_id] = "aligned"
        return out

    def _resolve_bearing_targets(self, per_blue_detected_reds):
        """{blue_idx: 2-vector red position} for every blue that either
        personally detected at least one red this step (see
        _compute_detected_indices), or is still within its post-detection
        grace period (see below). Tiebreak when a blue detects more than
        one: its currently assigned NEUTRALIZE target (self.blue_target_red_idx,
        refreshed each planning cycle in run()) if that red is among what it
        personally detected, else the nearest of what it personally detected.

        Grace period: once a blue's own sensor stops detecting anything, the
        last detected red's position is held as its bearing target for up to
        self.bearing_hold_steps more steps before falling back to the role's
        default heading behavior. Without this, a single-step detection blip
        (e.g. the blue's own independent translation -- RECON's patrol
        motion keeps flying toward its waypoint regardless of what's
        detected -- dragging it out of the FOV cone/range for just one step)
        immediately and fully releases the tracking cost, snapping heading
        back to the role's stale default target with no chance to
        re-acquire. A blue past its grace period (or that's never detected
        anything) has no entry -- its controller falls back to its own
        role's default heading behavior (see BaseBlueRoleController)."""
        bearing_targets = {}
        for blue_idx in range(self.num_blues):
            if self.blue_disabled[blue_idx]:
                continue
            detected = per_blue_detected_reds.get(blue_idx, set())
            if detected:
                assigned_red_idx = self.blue_target_red_idx.get(blue_idx)
                if assigned_red_idx in detected:
                    red_idx = assigned_red_idx
                else:
                    own_pos = self.x_current[blue_idx * self.blue_nx : blue_idx * self.blue_nx + 2]
                    red_idx = min(
                        detected,
                        key=lambda j: np.linalg.norm(
                            own_pos - self.red_state[j * self.red_nx : j * self.red_nx + 2]
                        ),
                    )
                pos = self.red_state[red_idx * self.red_nx : red_idx * self.red_nx + 2].copy()
                bearing_targets[blue_idx] = pos
                self.blue_last_detected_red_pos[blue_idx] = pos
                self.blue_steps_since_detected[blue_idx] = 0
            elif (
                blue_idx in self.blue_last_detected_red_pos
                and self.blue_steps_since_detected.get(blue_idx, 0) < self.bearing_hold_steps
            ):
                bearing_targets[blue_idx] = self.blue_last_detected_red_pos[blue_idx]
                self.blue_steps_since_detected[blue_idx] += 1
            else:
                self.blue_last_detected_red_pos.pop(blue_idx, None)
                self.blue_steps_since_detected.pop(blue_idx, None)
        return bearing_targets

    RED_SENSOR_MIN_SPEED = 0.05  # units/s; below it the heading is held (2.5% of v_max)

    def _red_sensor_omegas(self, step, red_state, blue_state):
        """red_sensor_control="track_or_velocity": each active red's yaw rate
        this step, from the pre-step states. Detection is the red's OWN cone
        (its current heading), the same sensing primitive the blues use."""
        active_reds = [j for j in range(self.num_real_reds) if self.red_outcomes[j] == "active"]
        active_blues = [i for i in range(self.num_blues) if not self.blue_disabled[i]]
        bearings = self._compute_bearings(red_state, self.num_reds, self.red_nx)
        detected = sensing.per_observer_detected_indices(
            [(j, red_state[j * self.red_nx: j * self.red_nx + 2]) for j in active_reds],
            bearings,
            [(i, blue_state[i * self.blue_nx: i * self.blue_nx + 2]) for i in active_blues],
            self.fov_half_angle_rad, self.fov_range,
        )
        omega_max = float(getattr(self.red, "omega_max", self.omega_max))
        out = {}
        for j in active_reds:
            x = red_state[j * self.red_nx: j * self.red_nx + self.red_nx]
            p, v, theta = x[0:2], x[2:4], float(x[4])
            seen = detected.get(j, set())
            target_blue, mode = None, None
            if seen:
                d = getattr(self, "red_decisions", {}).get(j)
                evading = getattr(d, "evade_blue_id", None) if getattr(d, "role", None) == "evade" else None
                target_blue = evading if evading in seen else min(
                    seen, key=lambda i: float(np.linalg.norm(
                        blue_state[i * self.blue_nx: i * self.blue_nx + 2] - p)))
                tgt = blue_state[target_blue * self.blue_nx: target_blue * self.blue_nx + 2].copy()
                self._red_last_seen_blue_pos[j] = tgt
                self._red_steps_since_seen[j] = 0
                mode = "track"
            elif (j in self._red_last_seen_blue_pos
                  and self._red_steps_since_seen.get(j, 0) < self.bearing_hold_steps):
                tgt = self._red_last_seen_blue_pos[j]
                self._red_steps_since_seen[j] += 1
                mode = "hold_sighting"
            else:
                self._red_last_seen_blue_pos.pop(j, None)
                self._red_steps_since_seen.pop(j, None)
                tgt = None
                mode = "velocity" if float(np.linalg.norm(v)) >= self.RED_SENSOR_MIN_SPEED else "stationary"
            if tgt is not None:
                desired = float(np.arctan2(tgt[1] - p[1], tgt[0] - p[0]))
            elif mode == "velocity":
                desired = float(np.arctan2(v[1], v[0]))
            else:
                desired = theta
            err = (desired - theta + np.pi) % (2 * np.pi) - np.pi
            omega = float(np.clip(err / self.dt, -omega_max, omega_max))
            out[j] = omega
            self.red_sensor_log.append({"step": int(step), "red": int(j), "mode": mode,
                                        "blue": None if target_blue is None else int(target_blue),
                                        "omega": omega, "heading_error": float(err)})
        return out

    def _local_safety_snapshot(self, blue_state, red_state):
        """Everything every local filter reads this step, frozen before any
        agent advances: states, activity flags, and the engagement
        authorizations. ``pursued`` maps a NEUTRALIZE blue to the red index
        of the contact whose pursuit target is actually installed; it is the
        only pair exempted from avoidance and is never used to correct a
        belief."""
        return {
            "blue": np.asarray(blue_state, dtype=float).copy(),
            "red": np.asarray(red_state, dtype=float).copy(),
            "blue_disabled": tuple(bool(d) for d in self.blue_disabled),
            "red_active": tuple(o == "active" for o in self.red_outcomes),
            "pursued": {
                i: self.blue_target_red_idx[i]
                for i, role in self.role_assignment.items()
                if role == BlueRole.NEUTRALIZE
                and i in self.target_override
                and i in self.blue_target_red_idx
            },
        }

    def _snap_pv(self, snap, team, idx):
        nx = self.blue_nx if team == "blue" else self.red_nx
        x = snap[team][idx * nx : idx * nx + 4]
        return x[0:2], x[2:4]

    def _local_blue_neighbors(self, i, snap):
        """Active teammates: cooperative half. Disabled blues: immobile
        obstacles (zero velocity, full responsibility) -- they stay collision
        obstacles in _blue_blue_collision_status. Active reds other than the
        pursued one: constant-velocity prediction, full responsibility (a
        heuristic, no cross-team guarantee). Resolved reds are excluded, as
        in _resolve_active_reds."""
        out = []
        if self.enable_blue_blue_cbf:
            for j in range(self.num_blues):
                if j == i:
                    continue
                p, v = self._snap_pv(snap, "blue", j)
                if snap["blue_disabled"][j]:
                    out.append(Neighbor(("blue", j), p, np.zeros(2), self.D_safe_blue, FULL))
                else:
                    out.append(Neighbor(("blue", j), p, v, self.D_safe_blue, COOPERATIVE))
        if self.enable_blue_red_cbf:
            pursued = snap["pursued"].get(i)
            for k in range(self.num_reds):
                if not snap["red_active"][k] or k == pursued:
                    continue
                p, v = self._snap_pv(snap, "red", k)
                out.append(Neighbor(("red", k), p, v, self.D_safe_blue_red, FULL))
        return out

    def _local_red_neighbors(self, j, snap):
        """Active blues (including any pursuing this red -- the target gets no
        reciprocal exemption), only under local_cbf_red_blue: constant-velocity
        prediction, full responsibility. Disabled blues are excluded: a red
        touching one has no consequence in _resolve_active_reds. Active red
        teammates: cooperative half (optional, local_cbf_red_red). With
        neither, the filter only applies the red's control bounds."""
        out = []
        for i in range(self.num_blues):
            if not self.local_cbf_red_blue or snap["blue_disabled"][i]:
                continue
            p, v = self._snap_pv(snap, "blue", i)
            out.append(Neighbor(("blue", i), p, v, self.D_safe_red, FULL))
        if self.local_cbf_red_red:
            for k in range(self.num_reds):
                if k == j or not snap["red_active"][k]:
                    continue
                p, v = self._snap_pv(snap, "red", k)
                out.append(Neighbor(("red", k), p, v, self.D_safe_red, COOPERATIVE))
        return out

    @staticmethod
    def _local_cbf_summary(result):
        return {
            "path": result.path,
            "reason": result.reason,
            "hard_status": result.hard_status,
            "relaxed_status": result.relaxed_status,
            "max_slack": max(result.slacks.values(), default=0.0),
            "slacks": dict(result.slacks),
            "barrier_residual": result.barrier_residual,
            "velocity_box_empty": result.velocity_box_empty,
            "num_rows": result.num_rows,
        }

    def _record_red_red_separation(self, step):
        active = [j for j in range(self.num_real_reds) if self.red_outcomes[j] == "active"]
        rec = self.red_red_separation
        for a in range(len(active)):
            for b in range(a + 1, len(active)):
                j, k = active[a], active[b]
                d = float(np.linalg.norm(
                    self.red_state[j * self.red_nx : j * self.red_nx + 2]
                    - self.red_state[k * self.red_nx : k * self.red_nx + 2]
                ))
                rec["min_distance"] = min(rec["min_distance"], d)
                if d < self.D_safe_red:
                    rec["pair_steps_below_d_safe"] += 1
                if d <= self.contact_tolerance:
                    rec["pair_steps_at_contact"] += 1
                    rec["contact_events"].append((step, j, k, d))

    # -- intruder team: gates and per-intruder decisions -------------------

    def _init_intruder_team(self, intruder_team):
        """intruder_team="gates": four public approach gates, a one-way
        approach -> interior phase per intruder, and per-intruder decisions
        (planning/intruder.py) from frozen observations. Needs the shared
        intruder beliefs and a red planner with ``decide_all``."""
        self.intruder_team = intruder_team
        self.gates = ()
        self.intruder_log = []
        if intruder_team is None:
            return
        if intruder_team != "gates":
            raise ValueError(f"intruder_team must be None or 'gates', not {intruder_team!r}")
        if self.intruder_info != "shared_beliefs":
            raise ValueError('intruder_team="gates" needs intruder_info="shared_beliefs"')
        if not hasattr(self.red_role_planner, "decide_all"):
            raise ValueError(
                f'intruder_team="gates" needs an intruder planner with decide_all; '
                f"{type(self.red_role_planner).__name__} has none"
            )
        from control.gates import build_gates
        from control.red_controllers import GateAttackController

        self.gates = build_gates(self.defense_center, self.defended_shape_points,
                                 self.arena_min, self.arena_max)
        heading = ({"include_heading": True, "omega_max": self.red.omega_max}
                   if self.red.include_heading else {})
        self.gate_attack_controller = GateAttackController(
            horizon=self.solver.N, dt=self.dt, a_max=self.red_a_max, v_max=self.red_v_max,
            defense_center=self.defense_center, arena_size=self.arena_size, **heading,
        )
        self.solver._red_role_controllers[RedRole.ATTACK] = self.gate_attack_controller

    def _reset_intruder_team(self):
        from control.gates import APPROACH

        self.red_phase = {j: APPROACH for j in range(self.num_real_reds)}
        self.red_decisions = {}          # in force: {red: IntruderDecision}
        self.red_published_intents = {}  # previous completed epoch: {red: (decision, step)}
        self.intruder_log = []
        self._selected_gate = {}         # the gate each intruder last chose
        self.gate_attack_controller.goals = {}

    def _intruder_observations(self, step):
        """Every active intruder's frozen observation, built before any
        decision is requested."""
        from planning.intruder import IntruderObservation

        believed = self._red_believed_blues_with_time(step)
        active = [j for j in range(self.num_real_reds) if self.red_outcomes[j] == "active"]
        states = {j: tuple(float(v) for v in self.red_state[j * self.red_nx : (j + 1) * self.red_nx])
                  for j in active}
        out = []
        for j in active:
            out.append(IntruderObservation(
                agent_id=j, now=int(step), dt=self.dt, own_state=states[j],
                phase=self.red_phase[j], commitment=self.red_decisions.get(j),
                teammates={k: s for k, s in states.items() if k != j},
                teammate_intents={k: v for k, v in self.red_published_intents.items()
                                  if k != j and k in states},
                believed_blues=believed.get(j, {}),
                gates=tuple((g.gate_id, g.position) for g in self.gates),
                defense_center=(float(self.defense_center[0]), float(self.defense_center[1])),
            ))
        return out

    def _red_believed_blues_with_time(self, step):
        return {
            j: {cid: (tuple(float(v) for v in b.predicted_state(step, self.dt)),
                      int(b.provenance.observed_at))
                for cid, b in store.targets.items()
                if b.is_fresh(step, self.belief_validity_age_steps)}
            for j, store in self.red_belief_stores.items()
            if self.red_outcomes[j] == "active"
        }

    def _plan_intruders(self, step):
        from planning.intruder import ATTACK, fallback_decision, validate_decision
        from planning.stops import StopRun

        observations = self._intruder_observations(step)
        try:
            raw = self.red_role_planner.decide_all(observations)
        except StopRun:
            raise  # a replay miss or the spend cap stops the run, never a fallback
        except Exception as exc:  # a failed planner call: every intruder falls back
            raw = {o.agent_id: exc for o in observations}
        for obs in observations:
            j = obs.agent_id
            decision = raw.get(j)
            reason = validate_decision(decision, obs)
            fallback = None
            if reason is not None:
                decision, fallback = fallback_decision(obs)
            self.red_decisions[j] = decision
            self.red_role_assignment[j] = RedRole.ATTACK if decision.role == ATTACK else RedRole.EVADE
            given = raw.get(j)
            self.intruder_log.append({
                "step": int(step), "red": j, "phase": obs.phase,
                "raw": repr(given), "invalid_reason": reason, "fallback": fallback,
                "decision": decision,
                # Why the planner gave no decision, when it named a reason
                # (an LLM planner's "schema_failure:...", "empty_response",
                # "call_failure:..."): invalid_reason is then "not_a_decision".
                "planner_reason": (given if isinstance(given, str) else
                                   f"call_failure:{type(given).__name__}"
                                   if isinstance(given, Exception) else None),
            })
        # Published only after every decision this epoch is made: next epoch's
        # observations see them, this epoch's never do.
        self.red_published_intents = {j: (d, int(step)) for j, d in self.red_decisions.items()
                                      if self.red_outcomes[j] == "active"}

    def _advance_intruder_phases(self, step):
        from control.gates import INTERIOR, advance_phase

        for j in range(self.num_real_reds):
            if self.red_outcomes[j] != "active":
                continue
            d = self.red_decisions.get(j)
            if d is not None and d.gate_id is not None:
                self._selected_gate[j] = d.gate_id
            before = self.red_phase[j]
            self.red_phase[j] = advance_phase(
                before, self._selected_gate.get(j),
                self.red_state[j * self.red_nx : j * self.red_nx + 2], self.gates)
            if before != self.red_phase[j]:
                self.intruder_log.append({"step": int(step), "red": j, "event": "phase",
                                          "from": before, "to": self.red_phase[j],
                                          "gate": self._selected_gate.get(j)})
        goals = {}
        for j, phase in self.red_phase.items():
            gate = self._selected_gate.get(j)
            if phase != INTERIOR and gate is not None:
                goals[j] = self.gates[gate].position
        self.gate_attack_controller.goals = goals

    def _intruder_evade_beliefs(self, red_believed_blues):
        """EVADE(blue): the evading intruder's avoidance term sees only the
        blue it chose; other blues stay in its safety filter."""
        out = dict(red_believed_blues)
        for j, d in self.red_decisions.items():
            if d.role == "evade" and j in out:
                out[j] = {b: s for b, s in out[j].items() if b == d.evade_blue_id}
        return out

    def _blue_blue_collision_status(self):
        """Blue-blue hard-contact check only. Blue-red contact is handled
        separately by _resolve_active_reds, since for NEUTRALIZE-role
        contact with red is a resolution event, not a failure -- unlike
        blue-blue contact, which is always a genuine loss.
        """
        for i in range(self.num_blues):
            p_i = self.x_current[i * self.blue_nx : i * self.blue_nx + 2]
            for j in range(i + 1, self.num_blues):
                p_j = self.x_current[j * self.blue_nx : j * self.blue_nx + 2]
                dist_pp = float(np.linalg.norm(p_i - p_j))
                if dist_pp <= self.contact_tolerance:
                    return True, (i, j), dist_pp
        return False, None, None

    def _announce_resolution(self, step, team, idx, event):
        """resolution_info="public": every opposing agent's store forgets the
        resolved agent, permanently (LocalBeliefStore.mark_resolved)."""
        if self.resolution_info != "public":
            return
        stores = (self.belief_stores if team == "red" else
                  getattr(self, "red_belief_stores", None) or {})
        for store in stores.values():
            store.mark_resolved(idx)
        self.resolution_log.append({"step": int(step), "team": team, "agent": int(idx),
                                    "event": event})

    # -- non-hostile contacts: birds and civilian drones among the contacts ---------

    DECOY_MOTIONS = ("sector_waypoints.1",)
    DECOY_STREAM = 7717             # np.random.default_rng([seed, DECOY_STREAM])
    DECOY_SPAWN_DIST = (55.0, 80.0)  # from the defended centre
    DECOY_FIRST_WP_DIST = (20.0, 35.0)
    DECOY_WP_DIST = (20.0, 70.0)
    DECOY_MIN_CENTRE_DIST = 16.0     # no waypoint leg passes closer to the centre
    DECOY_SPEED_FRAC = (0.6, 1.0)    # cruise speed / the reds' per-axis v_max
    DECOY_ARRIVE = 3.0
    DECOY_TAU = 0.5                  # s, velocity tracking time constant
    DECOY_ARENA_MARGIN = 3.0

    @staticmethod
    def _check_decoys(spec, num_contact_slots):
        """None (off) or a validated copy of the non-hostile spec:
        {"birds": int, "civilians": int, "motion": "sector_waypoints.1",
        "images": {red-array index: image path} (optional)}."""
        if spec is None:
            return None
        if int(num_contact_slots) > 0:
            raise ValueError("decoys and num_contact_slots are two different decoy mechanics; "
                             "use one")
        spec = dict(spec)
        unknown = sorted(set(spec) - {"birds", "civilians", "motion", "images"})
        if unknown:
            raise ValueError(f"unknown decoy spec keys {unknown}")
        birds, civilians = int(spec.get("birds", 0)), int(spec.get("civilians", 0))
        if birds < 0 or civilians < 0 or birds + civilians == 0:
            raise ValueError(f"decoys needs birds + civilians >= 1, got {birds} + {civilians}")
        motion = spec.get("motion", "sector_waypoints.1")
        if motion not in Simulation.DECOY_MOTIONS:
            raise ValueError(f"unknown decoy motion {motion!r}; expected {Simulation.DECOY_MOTIONS}")
        images = {int(k): str(v) for k, v in (spec.get("images") or {}).items()}
        for path in images.values():
            if not os.path.isfile(path):
                raise ValueError(f"decoy-run contact image not found: {path}")
        return {"birds": birds, "civilians": civilians, "motion": motion, "images": images}

    def is_decoy(self, idx) -> bool:
        return int(idx) in self.decoy_indices

    def _sector_point(self, rng, dist_range):
        from coordination.coverage_value import PRIOR_SECTOR_DEG

        lo, hi = PRIOR_SECTOR_DEG
        m = self.DECOY_ARENA_MARGIN
        for _ in range(1000):
            b = np.deg2rad(rng.uniform(lo, hi))
            d = rng.uniform(*dist_range)
            pt = self.defense_center + d * np.array([np.cos(b), np.sin(b)])
            if m <= pt[0] <= self.arena_max - m and m <= pt[1] <= self.arena_max - m:
                return pt
        raise RuntimeError("no approach-sector point inside the arena")

    def _leg_clear(self, a, b):
        """The straight leg a->b stays at least DECOY_MIN_CENTRE_DIST from the
        defended centre (so a non-hostile never enters the defended region)."""
        c = self.defense_center
        ab = b - a
        t = 0.0 if float(ab @ ab) < 1e-12 else float(np.clip((c - a) @ ab / (ab @ ab), 0.0, 1.0))
        return float(np.linalg.norm(a + t * ab - c)) >= self.DECOY_MIN_CENTRE_DIST

    def _next_decoy_waypoint(self, j, pos, first=False):
        rng = self._decoy_rng
        for _ in range(200):
            wp = self._sector_point(rng, self.DECOY_FIRST_WP_DIST if first else self.DECOY_WP_DIST)
            if self._leg_clear(pos, wp):
                return wp
        # Fallback, deterministic: straight out along the current bearing.
        out = pos - self.defense_center
        n = float(np.linalg.norm(out))
        return self.defense_center + (out / n if n > 1e-9 else np.array([-1.0, -1.0]) / np.sqrt(2)) * 40.0

    def _spawn_decoys(self):
        """One state row per non-hostile contact (red-array order after the real reds): an
        approach-sector start, inbound toward a first waypoint near the region
        at a cruise speed drawn per non-hostile contact, heading toward the defended centre
        as reds spawn. Own stream: no other draw in the episode moves."""
        self._decoy_rng = np.random.default_rng([int(self.seed), self.DECOY_STREAM])
        self._decoy_wp, self._decoy_speed = {}, {}
        rows = []
        for j in range(self.num_real_reds, self.num_reds):
            pos = self._sector_point(self._decoy_rng, self.DECOY_SPAWN_DIST)
            speed = float(self._decoy_rng.uniform(*self.DECOY_SPEED_FRAC)) * self.red_v_max
            wp = self._next_decoy_waypoint(j, pos, first=True)
            self._decoy_wp[j], self._decoy_speed[j] = wp, speed
            v = self._decoy_desired_velocity(j, pos)
            row = [pos[0], pos[1], v[0], v[1]]
            if self.red.include_heading:
                to_c = self.defense_center - pos
                row.append(float(np.arctan2(to_c[1], to_c[0])))
            rows.append(row)
        return rows

    def _decoy_desired_velocity(self, j, pos):
        d = self._decoy_wp[j] - pos
        n = float(np.linalg.norm(d))
        v = np.zeros(2) if n < 1e-9 else d / n * self._decoy_speed[j]
        return np.clip(v, -self.red_v_max, self.red_v_max)

    def _decoy_control(self, j):
        """Scripted control for non-hostile j: track the desired velocity toward the
        current waypoint with bounded acceleration; turn the (unused) sensor
        heading toward the velocity. Waypoints advance on arrival."""
        st = self.red_state[j * self.red_nx : (j + 1) * self.red_nx]
        pos, vel = st[0:2], st[2:4]
        if float(np.linalg.norm(self._decoy_wp[j] - pos)) <= self.DECOY_ARRIVE:
            self._decoy_wp[j] = self._next_decoy_waypoint(j, pos)
        a = np.clip((self._decoy_desired_velocity(j, pos) - vel) / self.DECOY_TAU,
                    -self.red_a_max, self.red_a_max)
        if not self.red.include_heading:
            return a
        omega_max = float(getattr(self.red, "omega_max", self.omega_max))
        speed = float(np.linalg.norm(vel))
        if speed > self.RED_SENSOR_MIN_SPEED:
            err = float(np.arctan2(np.sin(np.arctan2(vel[1], vel[0]) - st[4]),
                                   np.cos(np.arctan2(vel[1], vel[0]) - st[4])))
            omega = float(np.clip(err / self.dt, -omega_max, omega_max))
        else:
            omega = 0.0
        return np.concatenate([a, [omega]])

    def _resolve_decoy(self, step, j):
        """A non-hostile is not an intruder: contact with an active blue destroys
        both (the blue is disabled as for a red, with no score); leaving the
        arena removes it quietly; it never breaches. The win condition counts
        real reds only."""
        p = self.red_state[j * self.red_nx : j * self.red_nx + 2]
        active_blues = [i for i in range(self.num_blues) if not self.blue_disabled[i]]
        if active_blues:
            dist = {i: float(np.linalg.norm(self.x_current[i * self.blue_nx : i * self.blue_nx + 2] - p))
                    for i in active_blues}
            i = min(dist, key=dist.get)
            if dist[i] <= self.contact_tolerance:
                self.red_outcomes[j] = "decoy_destroyed"
                self._announce_resolution(step, "red", j, "decoy_destroyed")
                self._announce_resolution(step, "blue", i, "disabled")
                self.blue_disabled[i] = True
                self.decoy_events.append({"step": int(step), "decoy": int(j), "kind": self.red_kinds[j],
                                          "event": "destroyed", "blue": int(i),
                                          "distance": round(dist[i], 4)})
                return
        if not (self.arena_min <= p[0] <= self.arena_max and self.arena_min <= p[1] <= self.arena_max):
            self.red_outcomes[j] = "decoy_left"
            self._announce_resolution(step, "red", j, "decoy_left")
            self.decoy_events.append({"step": int(step), "decoy": int(j), "kind": self.red_kinds[j],
                                      "event": "left_arena"})

    def _resolve_red_neutralized(self, step, red_idx, blue_idx, distance):
        self.red_outcomes[red_idx] = "neutralized"
        self._announce_resolution(step, "red", red_idx, "neutralized")
        self._announce_resolution(step, "blue", blue_idx, "disabled")
        self.num_neutralized += 1
        self.blue_disabled[blue_idx] = True
        self.blue_score += self.neutralize_points
        self.neutralized = True
        if self.neutralize_step is None:
            self.neutralize_step = step
            self.neutralize_agent = blue_idx
        print(
            f"Red {red_idx + 1} neutralized (contact) at step {step} by blue {blue_idx + 1} "
            f"(distance={distance:.3f}). Blue score += {self.neutralize_points} "
            f"(total={self.blue_score:.1f}) -- now permanently disabled: {blue_idx + 1}."
        )

    def _resolve_active_reds(self, step):
        """Per-red resolution: a single (non-disabled) blue making hard
        contact resolves a red as neutralized immediately -- that blue is
        permanently disabled afterward and the kill is worth
        neutralize_points. Leaving the arena resolves a red as escaped;
        entering the defended region is an immediate breach that ends the
        whole episode. Returns True if a breach occurred this call (caller
        should stop the simulation).
        """
        for j in range(self.num_reds):
            if self.red_outcomes[j] != "active":
                continue
            if j in self.decoy_indices:
                self._resolve_decoy(step, j)
                continue
            p_e = self.red_state[j * self.red_nx : j * self.red_nx + 2]

            active_blues = [i for i in range(self.num_blues) if not self.blue_disabled[i]]
            if active_blues:
                blue_distances = {
                    i: float(np.linalg.norm(self.x_current[i * self.blue_nx : i * self.blue_nx + 2] - p_e))
                    for i in active_blues
                }
                closest_agent = min(blue_distances, key=blue_distances.get)
                closest_dist = blue_distances[closest_agent]

                if closest_dist <= self.contact_tolerance:
                    self._resolve_red_neutralized(step, j, closest_agent, closest_dist)
                    continue

            if self._point_in_polygon(p_e, self.defended_shape_points):
                self.touchdown = True
                self.touchdown_step = step
                self.touchdown_red_index = j
                print(
                    f"Perimeter breached at step {step} by red {j + 1} at "
                    f"({p_e[0]:.3f}, {p_e[1]:.3f}). Blue loses."
                )
                return True

            red_x, red_y = float(p_e[0]), float(p_e[1])
            if not (
                self.arena_min <= red_x <= self.arena_max
                and self.arena_min <= red_y <= self.arena_max
            ):
                self.red_outcomes[j] = "escaped"
                self._announce_resolution(step, "red", j, "escaped")
                self.num_escaped += 1
                self.blue_score += self.escape_points
                self.escaped = True
                if self.escape_step is None:
                    self.escape_step = step
                print(
                    f"Red {j + 1} escaped the arena at step {step} "
                    f"(position=({red_x:.3f}, {red_y:.3f})). Blue score += "
                    f"{self.escape_points} (total={self.blue_score:.1f})."
                )

        return False

    def run(self):
        self.setup_agents()
        self.solver.reset_warm_starts()
        self.solver.set_fixed_adjacency_from_state(self.x_current, self.red_state)
        self.history_blues = [self.x_current.copy()]
        self.local_cbf_log = []
        # Red-red contact is not a simulator event (no termination, no
        # resolution); it is only counted here, after the reds step, among
        # reds still active. Instrumentation: nothing reads it back.
        self.red_red_separation = {
            "min_distance": float("inf"),
            "pair_steps_below_d_safe": 0,
            "pair_steps_at_contact": 0,
            "contact_events": [],  # (step, j, k, distance), distance <= contact_tolerance
        }
        self.history_red = [self.red_state.copy()]
        self.history_birds = [self.bird_state.copy()]
        self.history_civilians = [self.civilian_state.copy()]
        self.history_cbf_slack = []
        self.belief_stores = {
            i: LocalBeliefStore(i, n_regions=len(self.coverage_tracker.centers))
            for i in range(self.num_blues)
        }
        self._reset_falsified_reports()
        # The intruder-team rule backstop (tests/test_red_team_information.py): under "legacy"
        # no red-side store exists at all, so nothing red can start gating
        # on blue belief modes.
        if self.intruder_info == "shared_beliefs":
            self.red_belief_stores = {j: LocalBeliefStore(j) for j in range(self.num_real_reds)}
        # Built only for the networked mode, so every other mode is provably
        # untouched by M4: there is no network object to have side effects.
        self.network = (
            Network(
                seed=self.seed,
                link_model=DiskLinkModel(r_comm=self.r_comm),
                loss_model=self.comm_loss_model,
            )
            if self.belief_mode == "networked"
            else None
        )
        self.comm_stats = self._fresh_comm_stats()
        # Reported, never inferred: with symmetric attrition at 0.08-0.18 on 4
        # blues, losing the commander is not rare, and every the command-delivery conditions result has to
        # state how often it happened and when.
        self.commander_lost_step = None
        self.planning_source = self.commander_id
        self._reset_command_state()
        self.unavailable_target_log = []
        self._unavailable_logged = {}
        self.resolution_log = []
        self.recorder = (
            EpisodeRecorder(
                config=describe_config(self._init_args),
                seed=self.seed,
                rng_mode="legacy_global" if self.rng.legacy_global else "isolated",
                num_blues=self.num_blues,
                num_reds=self.num_reds,
                blue_nx=self.blue_nx,
                red_nx=self.red_nx,
                blue_nu=self.blue.nu_single,
            )
            if self.record_episode
            else None
        )
        self.role_assignment = {}
        self.planner_output = PlannerOutput()
        self.blue_target_red_idx = {}
        self.blue_last_detected_red_pos = {}
        self.blue_steps_since_detected = {}
        self._red_last_seen_blue_pos = {}
        self._red_steps_since_seen = {}
        self.red_sensor_log = []
        self.red_role_assignment = {}
        self.role_assignment_history = [dict(self.role_assignment)]
        self.red_role_assignment_history = [dict(self.red_role_assignment)]
        self.neutralized = False
        self.neutralize_step = None
        self.neutralize_agent = None
        self.touchdown = False
        self.touchdown_step = None
        self.touchdown_red_index = None
        self.escaped = False
        self.escape_step = None
        self.collided = False
        self.collision_step = None
        self.collision_agents = None
        self.collision_kind = None
        self.defenders_won = False
        self.defenders_win_step = None
        self.defenders_win_agents = None
        self.red_outcomes = ["active"] * self.num_reds
        self.decoy_events = []  #: one entry per non-hostile destroyed or gone
        self.num_neutralized = 0
        self.num_escaped = 0
        self.blue_score = 0.0
        self.blue_disabled = [False] * self.num_blues
        self.blue_disabled_history = [list(self.blue_disabled)]
        self.solver_failed = False
        self.solver_failure_step = None
        self.num_bird_engagements = 0
        self.num_civilian_engagements = 0
        self.target_override = {}
        self.target_override_ids = {}
        self.target_override_applied = {}
        self.recon_target_override = {}
        self.defend_target_override = {}
        if self.cbaa_runtime is not None:
            self._reset_allocation()
        if self.commander_task_model:
            self._reset_commander_task_model()
        if self.intruder_team is not None:
            self._reset_intruder_team()

        print("Running decentralized simulation...")
        for step in range(self.sim_steps):
            blue_collided, blue_collision_agents, blue_collision_dist = (
                self._blue_blue_collision_status()
            )
            if blue_collided:
                self.collided = True
                self.collision_step = step
                self.collision_agents = blue_collision_agents
                self.collision_kind = "blue_blue"
                print(
                    f"Swarm lost before solve at step {step} "
                    f"(blue-blue collision, agents={blue_collision_agents}, "
                    f"distance={blue_collision_dist:.3f})."
                )
                break
            if self._resolve_active_reds(step):
                break
            if self.num_neutralized + self.num_escaped >= self.num_real_reds:
                self.defenders_won = True
                self.defenders_win_step = step
                print(
                    f"All reds resolved before solve at step {step}: "
                    f"neutralized={self.num_neutralized}, escaped={self.num_escaped}, "
                    f"score={self.blue_score:.1f}. Blue wins."
                )
                break

            # Computed fresh every step (cheap) since EvadeController's
            # reactive avoidance below should use the freshest possible
            # sensing, not a stale once-per-planning-cycle snapshot --
            # mirroring how CBF safety filtering is also always
            # every-step/full-ground-truth, distinct from the slower
            # strategic replanning cadence.
            detected_red_indices, detected_blue_indices, detected_contact_indices, per_blue_detected_reds = (
                self._compute_detected_indices(step)
            )
            # Every step, not gated by planning_interval_steps, same as
            # detection itself -- a blue should turn to face a red the
            # instant its own sensor detects it, not wait for the next
            # planning cycle.
            bearing_target_override = self._resolve_bearing_targets(per_blue_detected_reds)
            # Beliefs advance every step, at the same cadence as sensing -- an
            # agent knows what it just saw immediately, independently of the
            # slower planning clock.
            self._update_beliefs(
                step,
                per_blue_detected_reds,
                blue_observers=None,
                blue_bearings=self._compute_bearings(
                    self.x_current, self.num_blues, self.blue_nx
                ),
            )
            red_believed_blues = None
            if self.intruder_info == "shared_beliefs":
                self._update_red_beliefs(step)
                red_believed_blues = self._red_believed_blues(step)
            if (
                self.cbaa_runtime is not None
                and self.cbaa_runtime.transport != "network"
                and step % self.comm_interval_steps == 0
            ):
                # Same point in the step as the networked round, which runs
                # inside _update_beliefs above -- so "ideal" differs from a
                # networked R=inf auction in graph and losses, not in timing.
                changed = self.cbaa_runtime.local_round(
                    [i for i in range(self.num_blues) if not self.blue_disabled[i]]
                )
                if changed:
                    self._apply_cbaa(step, changed)
            believed_targets = self._believed_targets(step)
            if self.commander_task_model and self.commander_pending:
                self._activate_ready_pending(step, believed_targets)
            if step % self.planning_interval_steps == 0:
                # Detection-gated: a contact (and its image) only enters
                # the observation once some blue's FOV has actually
                # detected it -- Experiment II's vision planner must not
                # see the full, unfiltered contact roster any more than
                # Experiment I's text planner sees every red unconditionally.
                visible_contact_states = {
                    cid: s
                    for cid, s in self._active_contact_states().items()
                    if cid in detected_contact_indices
                }
                obs = PlannerObservation(
                    x_current=self.x_current,
                    red_states=self.red_state,
                    num_blues=self.num_blues,
                    num_reds=self.num_reds,
                    sim_step=step,
                    arena_size=self.arena_size,
                    defense_center=self.defense_center,
                    disabled_blues=[i for i in range(self.num_blues) if self.blue_disabled[i]],
                    detected_red_indices=sorted(detected_red_indices),
                    detected_blue_indices=sorted(detected_blue_indices),
                    detected_contact_ids=sorted(visible_contact_states.keys()),
                    coverage_regions=self.coverage_tracker.summarize(step),
                    contact_states=visible_contact_states,
                    contact_image_paths=self._active_contact_images(visible_contact_states.keys()),
                )
                # One object carries roles AND targets out of the planner (see
                # planning.planner.PlannerOutput).
                if self.cbaa_runtime is not None:
                    planned = self._plan_cbaa(step)
                elif self.per_agent_planning:
                    planned = self._plan_per_agent(step)
                elif self.centralized_from_belief:
                    planned = self._plan_centralized_from_belief(step)
                else:
                    planned = plan_full_from(self.role_planner, obs)
                # None means the commander is gone and the failure rule is
                # active: surviving agents hold their previous commitment
                # The safety filter keeps running on ground
                # truth, and no further planning happens this episode.
                generated = planned is not None
                if planned is None:
                    if self.commander_lost_step is None:
                        self.commander_lost_step = step
                    planned = self.planner_output
                # Red plans from `obs` -- the ground-truth, detection-gated
                # observation -- in EVERY condition. Feeding it a blue-belief
                # observation would make red's own positions blue's guesses at
                # them, and would make red's information depend on belief_mode,
                # which the intruder-team rule forbids. Pinned by
                # tests/test_red_team_information.py.
                if self.intruder_team is not None:
                    self._plan_intruders(step)
                else:
                    self.red_role_assignment = self.red_role_planner.plan(obs)
                if self.deliver_assignments_over_network:
                    # Condition B: the commander's plan does not become the
                    # team's assignment here. It is issued as one command per
                    # subordinate and reaches them over the same network the
                    # beliefs travel on. What IS in force is the accumulation of
                    # commands each agent has accepted, which is what
                    # _issue_commands applies.
                    planned = self._issue_commands(
                        planned, step, believed_targets, generated=generated
                    )
                    # Into the round already open at this step, if there is
                    # one: planning runs after the comm round, so without this
                    # every command would wait for the next round.
                    self._flush_outbound_commands(step)
                self._apply_assignment(planned, step, believed_targets)
                target_contact_ids = planned.target_contact_ids
                target_red_ids = planned.target_red_ids
                target_region_ids = planned.recon_target_regions
                roles_log = {i: r.value for i, r in self.role_assignment.items()}
                red_roles_log = {j: r.value for j, r in self.red_role_assignment.items()}
                print(
                    f"[planner] step {step}: roles = {roles_log}, "
                    f"red_roles = {red_roles_log}, targets = {target_contact_ids}, "
                    f"target_red_ids = {target_red_ids}, "
                    f"recon_target_regions = {target_region_ids}"
                )

            self._log_falsified_commitments(step)
            if self.intruder_team is not None:
                self._advance_intruder_phases(step)
                red_believed_blues = self._intruder_evade_beliefs(red_believed_blues)
            u_p_first, X_p_plans, u_e_first, _ = self.solver.solve_decentralized(
                x_current=self.x_current,
                red_states=self.red_state,
                role_assignment=self.role_assignment,
                red_role_assignment=self.red_role_assignment,
                target_override=self._neutralize_aim_points(step, believed_targets),
                target_override_ids=self.target_override_ids,
                recon_target_override=self.recon_target_override,
                detected_blue_indices=detected_blue_indices,
                bearing_target_override=bearing_target_override,
                believed_targets=believed_targets,
                defend_target_override=self.defend_target_override,
                # The pursuer sensor-heading rule's inputs; read by the solver only under
                # neutralize_heading="pursued_contact".
                own_detected_reds=per_blue_detected_reds,
                pursued_belief_keys={
                    blue_id: key
                    for blue_id, (_, key) in self.target_override_applied.items()
                    if key is not None
                },
                red_believed_blues=red_believed_blues,
            )
            solve_failed = u_p_first is None
            x_p_next_nominal = None if solve_failed else X_p_plans[:, 1]

            if solve_failed:
                self.solver_failed = True
                self.solver_failure_step = step
                print(f"Minimax solve failed at step {step}")
                break

            x_current_before_step = self.x_current.copy()
            red_state_before_step = self.red_state.copy()
            local_snap = local_step_log = None
            if self.safety_filter == "local":
                local_snap = self._local_safety_snapshot(
                    x_current_before_step, red_state_before_step
                )
                local_step_log = {"step": step, "version": LOCAL_CBF_VERSION,
                                  "pursued": dict(local_snap["pursued"]),
                                  "blue": {}, "red": {}}
            # The engagement-authority rule: the single bit the safety supervisor receives from the
            # decision layer. Derived from the committed role assignment, and
            # logged every step whether or not the filter is enabled, so the
            # claim "the CBF's inputs are states plus this bit" is auditable
            # from the record rather than by reading the code.
            engagement_authorized = [
                self.role_assignment.get(i) == BlueRole.NEUTRALIZE
                for i in range(self.num_blues)
            ]
            u_requested_step = u_p_first.copy()
            u_executed_step = u_p_first.copy()
            cbf_slack_step = None
            cbf_active_step = False
            if self.solver_supports_cbf and (
                self.enable_blue_blue_cbf
                or self.enable_blue_red_cbf
                or self.enable_convex_hull_containment
            ):
                blue_red_cbf_exempt = {
                    i for i, r in self.role_assignment.items() if r == BlueRole.NEUTRALIZE
                }
                if self.blue.include_heading:
                    # The CBF filter only ever adjusts [ax, ay] -- yaw rate
                    # has no bearing on collision safety, so it's extracted
                    # here, passed through unfiltered, and stitched back in
                    # below rather than threaded through the CBF-QP at all.
                    blue_nu = self.blue.nu_single
                    u_trans_des = np.concatenate(
                        [u_p_first[i * blue_nu : i * blue_nu + 2] for i in range(self.num_blues)]
                    )
                else:
                    u_trans_des = u_p_first
                if self.safety_filter == "local":
                    u_safe = np.zeros(2 * self.num_blues)
                    for i in range(self.num_blues):
                        if local_snap["blue_disabled"][i]:
                            continue  # held stationary below regardless
                        p_i, v_i = self._snap_pv(local_snap, "blue", i)
                        result = filter_agent(
                            p_i, v_i, u_trans_des[i * 2 : i * 2 + 2],
                            self._local_blue_neighbors(i, local_snap), self.local_cbf_blue_cfg,
                        )
                        u_safe[i * 2 : i * 2 + 2] = result.u
                        local_step_log["blue"][i] = self._local_cbf_summary(result)
                    eps_used = max(
                        (e["max_slack"] for e in local_step_log["blue"].values()), default=0.0
                    )
                else:
                    u_safe, eps_used = self.solver.one_step_cbf_filter(
                        x_current=self.x_current,
                        red_state=self.red_state,
                        u_des=u_trans_des,
                        blue_red_cbf_exempt=blue_red_cbf_exempt,
                    )
                if self.blue.include_heading:
                    u_apply = u_p_first.copy()
                    for i in range(self.num_blues):
                        u_apply[i * blue_nu : i * blue_nu + 2] = u_safe[i * 2 : i * 2 + 2]
                        # u_apply[i*blue_nu+2] (omega) stays the nominal
                        # MPC's own value -- unfiltered pass-through.
                else:
                    u_apply = u_safe
                self.x_current = self.solver.step_blue_dynamics(self.x_current, u_apply)
                self.history_cbf_slack.append(eps_used)
                u_executed_step = u_apply.copy()
                cbf_slack_step = eps_used
                cbf_active_step = True
            else:
                self.x_current = x_p_next_nominal
            for i in range(self.num_blues):
                if self.blue_disabled[i]:
                    self.x_current[i * self.blue_nx : i * self.blue_nx + self.blue_nx] = (
                        x_current_before_step[i * self.blue_nx : i * self.blue_nx + self.blue_nx]
                    )
            if self.recorder is not None:
                self.recorder.record_step(
                    step=step,
                    blue_state=x_current_before_step,
                    red_state=red_state_before_step,
                    u_requested=u_requested_step,
                    u_executed=u_executed_step,
                    cbf_slack=cbf_slack_step,
                    cbf_active=cbf_active_step,
                    engagement_authorized=engagement_authorized,
                    blue_disabled=self.blue_disabled,
                    blue_roles={i: r.value for i, r in self.role_assignment.items()},
                    red_roles={j: r.value for j, r in self.red_role_assignment.items()},
                    detected_red_union=detected_red_indices,
                    per_blue_detected_reds=per_blue_detected_reds,
                    target_red_ids=self.planner_output.target_red_ids,
                    recon_target_regions=self.planner_output.recon_target_regions,
                    defend_targets=self.defend_target_override,
                    red_outcomes=self.red_outcomes,
                    effective_targets=self.solver.effective_targets,
                    beliefs={
                        i: {
                            cid: belief.age(step)
                            for cid, belief in store.targets.items()
                        }
                        for i, store in self.belief_stores.items()
                    },
                )

            self.history_blues.append(self.x_current.copy())
            self.role_assignment_history.append(dict(self.role_assignment))
            self.blue_disabled_history.append(list(self.blue_disabled))

            red_nu = self.red.nu_single
            red_omegas = (self._red_sensor_omegas(step, red_state_before_step, x_current_before_step)
                          if self.red_sensor_control == "track_or_velocity" else {})
            for j in range(self.num_reds):
                if self.red_outcomes[j] != "active":
                    continue
                if j in self.decoy_indices:
                    # Scripted motion, never the red solver or red filter.
                    self.red_state[j * self.red_nx : j * self.red_nx + self.red_nx] = self.red.step_single(
                        self.red_state[j * self.red_nx : j * self.red_nx + self.red_nx],
                        self._decoy_control(j))
                    continue
                u_e_j_full = u_e_first[j * red_nu : j * red_nu + red_nu]
                u_e_j_trans = u_e_j_full[0:2]
                red_state_j = self.red_state[j * self.red_nx : j * self.red_nx + self.red_nx]
                if self.solver_supports_cbf and self.enable_red_cbf and local_snap is not None:
                    p_j, v_j = self._snap_pv(local_snap, "red", j)
                    result = filter_agent(
                        p_j, v_j, u_e_j_trans,
                        self._local_red_neighbors(j, local_snap), self.local_cbf_red_cfg,
                    )
                    u_e_safe_trans = result.u
                    local_step_log["red"][j] = self._local_cbf_summary(result)
                elif self.solver_supports_cbf and self.enable_red_cbf:
                    u_e_safe_trans, _ = self.solver.one_step_red_cbf_filter(
                        x_current=(
                            x_current_before_step
                            if self.safety_state_sync == "frozen"
                            else self.x_current
                        ),
                        red_state=red_state_j,
                        u_des=u_e_j_trans,
                    )
                else:
                    u_e_safe_trans = u_e_j_trans
                if self.red.include_heading:
                    # Omega (u_e_j_full[2]) bypasses the CBF filter
                    # entirely, same convention as blue.
                    omega_j = u_e_j_full[2:3]
                    if self.red_sensor_control == "track_or_velocity":
                        omega_j = np.array([red_omegas[j]])
                    u_e_safe = np.concatenate([u_e_safe_trans, omega_j])
                else:
                    u_e_safe = u_e_safe_trans
                self.red_state[j * self.red_nx : j * self.red_nx + self.red_nx] = self.red.step_single(
                    red_state_j, u_e_safe
                )
            self.history_red.append(self.red_state.copy())
            if local_step_log is not None:
                self.local_cbf_log.append(local_step_log)
            self._record_red_red_separation(step)
            self.red_role_assignment_history.append(dict(self.red_role_assignment))

            self._step_birds()
            self._step_civilians()
            self.history_birds.append(self.bird_state.copy())
            self.history_civilians.append(self.civilian_state.copy())

            blue_collided, blue_collision_agents, blue_collision_dist = (
                self._blue_blue_collision_status()
            )
            if blue_collided:
                self.collided = True
                self.collision_step = step
                self.collision_agents = blue_collision_agents
                self.collision_kind = "blue_blue"
                print(
                    f"Swarm lost at step {step} "
                    f"(blue-blue collision, agents={blue_collision_agents}, "
                    f"distance={blue_collision_dist:.3f})."
                )
                break
            if self._resolve_active_reds(step):
                break
            if self.num_neutralized + self.num_escaped >= self.num_real_reds:
                self.defenders_won = True
                self.defenders_win_step = step
                print(
                    f"All reds resolved at step {step}: "
                    f"neutralized={self.num_neutralized}, escaped={self.num_escaped}, "
                    f"score={self.blue_score:.1f}. Blue wins."
                )
                break

        self.history_blues = np.array(self.history_blues)
        self.history_red = np.array(self.history_red)
        self.history_birds = np.array(self.history_birds)
        self.history_civilians = np.array(self.history_civilians)
        if self.recorder is not None:
            self.recorder.finalize(self.episode_outcome())
        print("Simulation complete.")

    def episode_outcome(self):
        """The episode's terminal state as a flat, serializable dict.

        Every mission metric in is derivable from this. Note
        blue_score counts an escaped red the same as a neutralized one
        (escape_points defaults to escape_points == neutralize_points), which
        is deliberate and declared -- see the escape-scoring rule -- so blue_score and
        breach rate are reported as separate primary outcomes rather than one
        standing in for the other.

        **Attrition is an identity, not a correlate.**
        ``_resolve_red_neutralized`` is the only site that ever sets
        ``blue_disabled``, and it sets exactly one entry per neutralization;
        a disabled blue leaves ``active_blues`` and can never resolve a second
        red. So ``n_blues_disabled == num_neutralized`` holds *by
        construction*, in every episode -- not as the 30/30 empirical
        regularity the Exp 0 pilot first recorded it as. Two consequences,
        both of which have already caused a misreading once:

        * kills, defender losses and (absent escapes) ``blue_score`` are **one
          number reported three times**. They can never corroborate each other.
        * the exchange ratio is identically 1.0, so it cannot discriminate
          between conditions and is deliberately not reported.

        What attrition adds is therefore not new information but the missing
        *sign*: ``blue_score`` credits +1 for a kill and charges nothing for
        the defender spent, so it systematically favours whichever arm trades
        more. ``attrition_rate`` is reported so both sides of the same event
        are visible. No net score is computed here -- weighting a defender
        against a kill is a free parameter that the escape-scoring rule has not settled."""
        n_blues_disabled = int(sum(bool(x) for x in self.blue_disabled))
        return {
            "touchdown": bool(self.touchdown),
            "touchdown_step": self.touchdown_step,
            "touchdown_red_index": self.touchdown_red_index,
            "num_neutralized": int(self.num_neutralized),
            "num_escaped": int(self.num_escaped),
            "escaped": bool(self.escaped),
            "escape_step": self.escape_step,
            "blue_score": float(self.blue_score),
            "neutralize_points": float(self.neutralize_points),
            "escape_points": float(self.escape_points),
            "solver_failed": bool(self.solver_failed),
            "solver_failure_step": self.solver_failure_step,
            "defenders_win_step": getattr(self, "defenders_win_step", None),
            "blue_disabled": [bool(x) for x in self.blue_disabled],
            "n_blues_disabled": n_blues_disabled,
            "attrition_rate": (
                n_blues_disabled / self.num_blues if self.num_blues else 0.0
            ),
            # Intruders only: non-hostile contacts are reported under "non-hostile contacts".
            "red_outcomes": list(self.red_outcomes[: self.num_real_reds]),
            "n_trajectory_samples": int(len(self.history_blues)),
            # command-delivery condition A and B only; None in every other configuration,
            # including when the commander happens to survive.
            "commander_id": self.commander_id,
            "commander_lost_step": self.commander_lost_step,
            "commander_lost": self.commander_lost_step is not None,
            # Condition B only; all zeros elsewhere, because nothing issues a
            # command unless assignments are delivered over the network. The
            # contract Sec 3 limitation -- B - A mixes inbound information
            # latency with outbound assignment latency, and this design cannot
            # separate them -- is only *observable* if the outbound half is
            # reported, so it is reported with every episode rather than
            # extracted on demand.
            "commands": self.command_metrics(),
            # The LLM planner. Empty for every rule-planner episode, so no existing result's
            # schema moves -- and never omitted, so "were there fallbacks?" is
            # answerable from the outcome alone rather than by knowing which
            # planner ran.
            "planner_fallbacks": self.planner_fallback_events(),
            # The auction. Empty unless allocation="cbaa". Convergence, churn, logged
            # conflicts, and the size of the channel the rule that auction traffic never creates a belief refused.
            "allocation": (
                self.cbaa_runtime.metrics() if self.cbaa_runtime is not None else {}
            ),
            "planner_diagnostics": self.planner_diagnostics(),
            **({"decoys": {
                "kinds": {int(j): self.red_kinds[j] for j in sorted(self.decoy_indices)},
                "outcomes": {int(j): self.red_outcomes[j] for j in sorted(self.decoy_indices)},
                "events": list(self.decoy_events),
                "n_destroyed": sum(1 for e in self.decoy_events if e["event"] == "destroyed"),
                "images": {int(j): os.path.basename(p) for j, p in sorted(self.contact_image_paths.items())},
                "motion": self.decoy_spec["motion"],
            }} if self.decoy_spec is not None else {}),
        }

    def planner_fallback_events(self):
        """Every planning call that silently became DEFEND, as a flat list.

        **LLM failure is silent to the run by design** -- a schema-invalid
        response falls back to DEFEND for the affected agent and the episode
        continues, because a 400-step run must not die on one bad token. The
        cost of that design is that a badly-behaved model produces a *complete,
        plausible-looking episode* that is really the rule fallback wearing an
        LLM's name.

        Surfacing the events here is what makes that cost payable: no result
        from an LLM planner may be read before this list has been.
        """
        planner = self.role_planner
        events = [dict(e) for e in (getattr(planner, "fallback_events", None) or [])]
        # The auction bridge: under CBAA the LLM chooses bid modifiers, not roles, and its
        # dropped fields and total failures must be just as loud.
        source = getattr(self.cbaa_runtime, "modifier_source", None)
        if source is not None:
            events.extend(dict(e) for e in source.fallback_events)
        return events

    def planner_diagnostics(self):
        """Model, call counts, cache hits and fallbacks, when the planner has them.

        Duck-typed on purpose: this codebase's planners are duck-typed
        throughout (``scripts/`` passes plain wrapper classes), so requiring a
        base class here would break shipped experiment scripts to report a
        diagnostic. A planner with nothing to say returns ``{}``.
        """
        diagnostics = getattr(self.role_planner, "diagnostics", None)
        out = dict(diagnostics()) if callable(diagnostics) else {}
        source = getattr(self.cbaa_runtime, "modifier_source", None)
        if source is not None:
            out["modifier_source"] = source.diagnostics()
        return out

    def generate_animation(self, video_filename=None):
        if len(self.history_blues) <= 1:
            return

        fig, ax = plt.subplots(figsize=(10, 10))
        label_fontsize = 12
        tick_fontsize = 10
        legend_fontsize = 10
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlabel("x", fontsize=label_fontsize)
        ax.set_ylabel("y", fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        plot_max = max(self.arena_max, 105.0)
        ax.set_xlim(self.arena_min, plot_max)
        ax.set_ylim(self.arena_min, plot_max)
        arena_border = plt.Rectangle(
            (self.arena_min, self.arena_min),
            self.arena_size,
            self.arena_size,
            fill=False,
            edgecolor="k",
            linewidth=2,
            linestyle="-",
            alpha=0.8,
        )
        ax.add_patch(arena_border)

        red_lines = []
        red_points = []
        red_role_texts = []
        for j in range(self.num_reds):
            line, = ax.plot([], [], "r--", linewidth=1.5, alpha=0.6)
            label = "Red" if self.num_reds == 1 else f"Red {j + 1}"
            point, = ax.plot([], [], "rX", markersize=12, label=label)
            red_lines.append(line)
            red_points.append(point)
            red_role_texts.append(
                ax.text(0, 0, "", fontsize=8, color="darkred", fontweight="bold")
            )

        bird_points = []
        for _ in range(self.num_birds):
            point, = ax.plot(
                [], [], marker="^", color="0.5", linestyle="None", markersize=7,
                label="Bird" if not bird_points else None,
            )
            bird_points.append(point)

        polygon_closed = np.vstack([self.defense_polygon, self.defense_polygon[0]])
        target_line, = ax.plot(
            polygon_closed[:, 0],
            polygon_closed[:, 1],
            color="k",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="Defense Polygon",
        )
        shape_closed = np.vstack([self.defended_shape_points, self.defended_shape_points[0]])
        true_shape_line, = ax.plot(
            shape_closed[:, 0],
            shape_closed[:, 1],
            color="0.35",
            linestyle="-",
            linewidth=1.5,
            alpha=0.8,
            label="Defended Shape",
        )

        blue_lines = []
        blue_points = []
        blue_role_texts = []
        colors = ["b", "g", "c", "m"]

        for i in range(self.num_blues):
            color = colors[i % len(colors)]
            line, = ax.plot([], [], f"{color}-", linewidth=2, alpha=0.7)
            point, = ax.plot([], [], f"{color}o", markersize=8, label=f"Blue {i + 1}")
            blue_lines.append(line)
            blue_points.append(point)
            blue_role_texts.append(
                ax.text(0, 0, "", fontsize=8, color=color, fontweight="bold")
            )

        ax.legend(loc="upper right", fontsize=legend_fontsize)

        label_offset = 0.015 * self.arena_size

        def update(frame):
            red_roles_now = self.red_role_assignment_history[
                min(frame, len(self.red_role_assignment_history) - 1)
            ]
            for j in range(self.num_reds):
                idx = j * self.red_nx
                x, y = self.history_red[frame, idx], self.history_red[frame, idx + 1]
                red_lines[j].set_data(
                    self.history_red[:frame + 1, idx],
                    self.history_red[:frame + 1, idx + 1],
                )
                red_points[j].set_data([x], [y])
                role = red_roles_now.get(j)
                red_role_texts[j].set_position((x + label_offset, y + label_offset))
                red_role_texts[j].set_text(role.value.upper() if role is not None else "")

            blue_roles_now = self.role_assignment_history[
                min(frame, len(self.role_assignment_history) - 1)
            ]
            blue_disabled_now = self.blue_disabled_history[
                min(frame, len(self.blue_disabled_history) - 1)
            ]
            for i in range(self.num_blues):
                idx = i * self.blue_nx
                x, y = self.history_blues[frame, idx], self.history_blues[frame, idx + 1]
                blue_lines[i].set_data(
                    self.history_blues[:frame + 1, idx],
                    self.history_blues[:frame + 1, idx + 1],
                )
                blue_points[i].set_data([x], [y])
                blue_role_texts[i].set_position((x + label_offset, y + label_offset))
                if blue_disabled_now[i]:
                    blue_role_texts[i].set_text("DISABLED")
                    blue_points[i].set_alpha(0.3)
                    blue_lines[i].set_alpha(0.15)
                else:
                    role = blue_roles_now.get(i)
                    blue_role_texts[i].set_text(role.value.upper() if role is not None else "")
                    blue_points[i].set_alpha(1.0)
                    blue_lines[i].set_alpha(0.7)

            bird_frame = min(frame, len(self.history_birds) - 1)
            for b in range(self.num_birds):
                idx = b * 4
                bx, by = self.history_birds[bird_frame, idx], self.history_birds[bird_frame, idx + 1]
                bird_points[b].set_data([bx], [by])

            artists = (
                red_lines + red_points + red_role_texts
                + blue_lines + blue_points + blue_role_texts + bird_points
                + [arena_border, target_line, true_shape_line]
            )
            return artists

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(self.history_red),
            interval=50,
            blit=True,
        )
        if video_filename is None:
            video_filename = f"decentralized_pursuit_evasion_{self.num_blues}.gif"
        print(f"Saving video to {video_filename}...")
        anim.save(video_filename, writer="pillow", fps=20)
        print(f"Video saved successfully: {video_filename}")

        base_filename = video_filename.rsplit(".", 1)[0]
        final_frame_idx = len(self.history_red) - 1
        update(final_frame_idx)
        png_filename = base_filename + ".png"
        fig.savefig(png_filename, dpi=300, bbox_inches="tight")
        print(f"Final frame saved successfully: {png_filename}")

        # Close the figure to free up memory
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a single simulation and save its animation."
    )
    parser.add_argument("--sim-steps", type=int, default=200, help="Number of simulation steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-reds", type=int, default=1, help="Number of red agents.")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use LLM-backed role planners (via OpenRouter; requires "
        "OPENROUTER_API_KEY) instead of the rule-based stubs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek/deepseek-chat",
        help="OpenRouter model slug to use with --llm.",
    )
    args = parser.parse_args()

    role_planner = LLMRolePlanner(model=args.model) if args.llm else None
    red_role_planner = LLMRedRolePlanner(model=args.model) if args.llm else None
    simulation = Simulation(
        num_blues=4,
        num_reds=args.num_reds,
        sim_steps=args.sim_steps,
        horizon=20,
        enable_blue_blue_cbf=True,
        enable_blue_red_cbf=True,
        role_planner=role_planner,
        red_role_planner=red_role_planner,
        seed=args.seed,
    )

    simulation.run()
    simulation.generate_animation()
