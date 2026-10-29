"""Per-step orchestration for the decentralized planning strategy: each blue
runs its own local controller for its currently assigned BlueRole, each red runs
its own local controller for its currently assigned RedRole, and the
resulting nominal plans are passed through control.cbf's CBF-QP safety
filter (inherited unchanged). Which controller *class* realizes a given role
is a planning-layer decision -- see planning.dispatch -- this module only
constructs those classes' instances for the current run's config and
dispatches to them each step.

Lives in simulation/, not control/: it has no cost function or Opti of its
own, just role->controller dispatch and CBF-filter invocation -- i.e. it's
the piece that runs the simulation bed's per-step control cycle, not a
controller itself.
"""

import numpy as np

from control.blue_controllers import DefendController, NeutralizeController, ReconController
from control.cbf import CBFFilterMixin
from control.red_controllers import AttackController, EvadeController, RedReconController
from planning.dispatch import BLUE_ROLE_CONTROLLER_CLASSES, RED_ROLE_CONTROLLER_CLASSES
from planning.roles import RedRole, BlueRole


class DecentralizedPursuitEvasionSolver(CBFFilterMixin):
    """Per-blue decentralized MPC: each blue runs its own local controller
    for its currently assigned BlueRole (DEFEND or NEUTRALIZE) instead of a joint
    QP across all blues. CBF safety filtering across blues/red is
    still applied centrally (inherited unchanged from CBFFilterMixin) --
    only the nominal per-agent planning is decentralized.
    """

    supports_one_step_cbf = True

    def __init__(
        self,
        blue_agent,
        red_agent,
        horizon,
        arena_size=100.0,
        defense_center=None,
        defense_polygon=None,
        defended_shape_points=None,
        enable_blue_blue_cbf=True,
        enable_blue_red_cbf=True,
        enable_convex_hull_containment=False,
        patrol_rng=None,
        **kwargs,
    ):
        del defended_shape_points
        super().__init__(
            blue_agent=blue_agent,
            red_agent=red_agent,
            horizon=horizon,
            enable_blue_blue_cbf=enable_blue_blue_cbf,
            enable_blue_red_cbf=enable_blue_red_cbf,
            enable_convex_hull_containment=enable_convex_hull_containment,
            **kwargs,
        )
        self.defense_center = (
            None
            if defense_center is None
            else np.asarray(defense_center, dtype=float).reshape(2)
        )
        self.defense_polygon = (
            None
            if defense_polygon is None
            else np.asarray(defense_polygon, dtype=float).reshape(-1, 2)
        )
        self.defend_stations = self._build_defend_stations()
        # Red-array indices the Simulation moves by script (non-hostile contacts).
        self.scripted_red_ids = frozenset()

        # Threaded to every controller so each one's own state/control width
        # (and, when present, its yaw-rate bound) matches the Agent it steers
        # -- see CBFFilterMixin.__init__ for why CBF machinery is unaffected.
        blue_heading_kwargs = (
            {"include_heading": True, "omega_max": self.blue.omega_max}
            if self.blue.include_heading else {}
        )
        red_heading_kwargs = (
            {"include_heading": True, "omega_max": self.red.omega_max}
            if self.red.include_heading else {}
        )

        self.defend_controller = DefendController(
            horizon=self.N,
            dt=self.dt,
            a_max=self.a_max,
            v_max=self.v_max,
            defense_polygon=self.defense_polygon,
            arena_size=arena_size,
            **blue_heading_kwargs,
        )
        self.neutralize_controller = NeutralizeController(
            horizon=self.N,
            dt=self.dt,
            a_max=self.a_max,
            v_max=self.v_max,
            **blue_heading_kwargs,
        )
        self.recon_controller = ReconController(
            horizon=self.N,
            dt=self.dt,
            a_max=self.a_max,
            v_max=self.v_max,
            arena_size=arena_size,
            # The episode's "patrol" RNG stream (simulation/rng.py). Passed in
            # rather than drawn from np.random inside the controller so patrol
            # waypoints are reproducible independently of every other source of
            # randomness in the episode.
            rng=patrol_rng,
            **blue_heading_kwargs,
        )
        self.evade_controller = EvadeController(
            horizon=self.N,
            dt=self.dt,
            a_max=self.red_a_max,
            v_max=self.red_v_max,
            n_blues=self.N_p,
            blue_nx=self.blue_nx,
            **red_heading_kwargs,
        )
        self.attack_controller = AttackController(
            horizon=self.N,
            dt=self.dt,
            a_max=self.red_a_max,
            v_max=self.red_v_max,
            defense_center=self.defense_center,
            arena_size=arena_size,
            **red_heading_kwargs,
        )
        self.red_recon_controller = RedReconController(
            horizon=self.N,
            dt=self.dt,
            a_max=self.red_a_max,
            v_max=self.red_v_max,
            **red_heading_kwargs,
        )

        # BlueRole -> instance dispatch, built from planning.dispatch's role ->
        # controller-*class* registry plus this run's already-constructed
        # instances -- keeps "which class realizes a role" a planning-layer
        # fact (single source of truth in planning/dispatch.py) while this
        # solver only owns instance construction/config.
        blue_instances_by_class = {
            DefendController: self.defend_controller,
            NeutralizeController: self.neutralize_controller,
            ReconController: self.recon_controller,
        }
        red_instances_by_class = {
            AttackController: self.attack_controller,
            RedReconController: self.red_recon_controller,
            EvadeController: self.evade_controller,
        }
        self._blue_role_controllers = {
            role: blue_instances_by_class[cls]
            for role, cls in BLUE_ROLE_CONTROLLER_CLASSES.items()
        }
        self._red_role_controllers = {
            role: red_instances_by_class[cls]
            for role, cls in RED_ROLE_CONTROLLER_CLASSES.items()
        }
        # Which role's controller an instance realizes, so solve_decentralized
        # can name the controller that actually ran. A NEUTRALIZE agent with no
        # realizable target is handed the DEFEND controller, and the log has to
        # say DEFEND there or it describes the assignment instead of the motion.
        self._controller_role_names = {
            id(instance): role.value
            for role, instance in self._blue_role_controllers.items()
        }

        # {blue_id: {"source", "contact_id", "controller"}} for the last solve.
        # A *diagnostic*, never read back into a decision: the assignment the
        # planner recorded and the contact a blue is actually flying at diverge
        # whenever a frozen NEUTRALIZE target goes inactive
        # And without this
        # the commander-loss analysis cannot tell commitment persistence from
        # actual pursuit. Empty until the first solve.
        self.effective_targets = {}
        # The station-heading rule (). "inherited" -- the default -- is today's
        # behaviour: a blue holding a station with nothing detected keeps
        # whatever heading its last motion left it with, because no heading
        # term is active. "outward" gives the explicit DEFEND(station) posture
        # a surveillance heading: the direction from the defended centre
        # through the assigned station. A BEHAVIOUR CHANGE motivated by the
        # The auction smoke, not a neutral one; the default keeps the goldens, it does
        # not make the posture free. Set by Simulation.
        self.station_heading = "inherited"
        # "configured": {station id: heading} set by Simulation.
        self.configured_station_headings = None
        # {blue_id: "detection" | "station_outward" | None} for the last solve:
        # which heading target, if any, the controller was given. Diagnostic.
        self.bearing_sources = {}
        # The three-task rule. True -- the default --
        # keeps every historical DEFEND branch, including the implicit screen
        # of the agent's nearest believed contact. False (set by Simulation in
        # The three-task mode) realises EVERY DEFEND as an explicit station
        # hold: its own station without an override, and a refusal for a
        # contact override or an unknown station, so no path -- no override, a
        # dropped or unbelieved screen target, a missing role entry -- can
        # reach the screening geometry.
        self.defend_screening = True
        # The pursuer sensor-heading rule (a policy change; adopted for NEW experiments via
        # scripts/experiment_policy.py). "inherited" -- the default, kept for
        # historical reproduction -- is the original behaviour: a NEUTRALIZE agent gets a heading target only
        # from its own detection (or the detection grace hold), so it can pursue
        # a contact it believes in, via sharing or extrapolation, with its
        # sensor pointed elsewhere. "pursued_contact" points it at its effective
        # contact's believed position, ahead of any other detection; see
        # solve_decentralized. Set by Simulation.
        self.neutralize_heading = "inherited"

    def _station_id_of(self, station):
        """The id of the station at position ``station``."""
        p = np.asarray(station, dtype=float).reshape(2)
        for k, v in self.defend_stations.items():
            if float(np.linalg.norm(np.asarray(v, dtype=float).reshape(2) - p)) < 1e-9:
                return int(k)
        raise ValueError(f"no station at {p}")

    def _outward_unit(self, station):
        """Unit vector from the defended centre through ``station``.
        The centre is the declared defense_center, else the polygon centroid."""
        center = self.defense_center
        if center is None:
            center = np.asarray(self.defense_polygon, dtype=float).mean(axis=0)
        v = np.asarray(station, dtype=float).reshape(2) - np.asarray(center, dtype=float).reshape(2)
        n = float(np.linalg.norm(v))
        if n <= 1e-9:
            raise ValueError("a station at the defended centre has no outward direction")
        return v / n

    @staticmethod
    def _nearest_believed(own_pos, believed, red_states, red_nx):
        """``(contact_id, state)`` for the nearest contact in ``believed``.

        Returns ``(None, None)`` if the agent believes in none.

        **The id is returned, not only the state**, because the assignment the
        planner recorded and the contact the controller actually flies at can
        differ -- a NEUTRALIZE agent whose frozen target has gone inactive
        silently falls back to this contact (see ``solve_decentralized``).
        Telling those two apart is what
 requires before the
        commander-loss analysis can be read, and it needs the id.

        ``believed is None`` selects the pre-M2 behavior -- nearest actual red
        from ground truth -- and exists only so a direct caller of this solver
        (verification scripts, tests constructing it by hand) is not silently
        broken. Simulation always passes a belief, so the ground-truth branch is
        dead on the research path. Deliberately noisy to reason about: if this
        branch ever runs during an experiment, the experiment is leaking.
        """
        if believed is None:
            n_red = len(red_states) // red_nx
            if n_red == 0:
                return None, None
            idx = int(np.argmin([
                np.linalg.norm(own_pos - red_states[j * red_nx : j * red_nx + 2])
                for j in range(n_red)
            ]))
            return idx, red_states[idx * red_nx : idx * red_nx + red_nx]

        if not believed:
            return None, None
        best = min(
            believed,
            key=lambda cid: (
                float(np.linalg.norm(own_pos - np.asarray(believed[cid], dtype=float).ravel()[0:2])),
                cid,  # stable tiebreak
            ),
        )
        return best, np.asarray(believed[best], dtype=float).ravel()

    def _build_defend_stations(self):
        """One fixed guard station per blue, evenly spaced around the defended
        perimeter.

        A DEFEND agent that believes in no contact has to go somewhere, and
        "the true nearest red" is not available to it (that would be ground truth
        leaking into a believed-information agent). Fixed
        stations give it a defensive posture that requires no contact to exist,
        and give the allocator a genuine persistent DEFEND(station_k) task to
        bid on, which is also what's coverage-floor constraint needs
        something to attach to.

        Deterministic and stable across a run, so an agent does not migrate
        between stations for reasons unrelated to its decisions.
        """
        if self.defense_polygon is None or len(self.defense_polygon) < 3:
            return {}
        n_vertices = len(self.defense_polygon)
        return {
            i: self.defense_polygon[int(round(i * n_vertices / self.N_p)) % n_vertices].copy()
            for i in range(self.N_p)
        }

    def _believed_blue_trajectories(self, believed, blue_nx):
        """Constant-velocity predictions over the horizon from one red's
        beliefs {blue_id: state at now}, laid out like X_p_plans (only
        positions are read downstream). Unbelieved blues stay zero and are
        masked out by the returned id list."""
        trajs = np.zeros((blue_nx * self.N_p, self.N + 1))
        k = np.arange(self.N + 1) * self.dt
        for i, state in believed.items():
            state = np.asarray(state, dtype=float).ravel()
            trajs[i * blue_nx, :] = state[0] + k * state[2]
            trajs[i * blue_nx + 1, :] = state[1] + k * state[3]
        return trajs, sorted(believed)

    def solve_decentralized(
        self, x_current, red_states, role_assignment, red_role_assignment=None,
        target_override=None, recon_target_override=None, detected_blue_indices=None,
        bearing_target_override=None, believed_targets=None, target_override_ids=None,
        defend_target_override=None, own_detected_reds=None, pursued_belief_keys=None,
        red_believed_blues=None,
    ):
        """Dispatches each blue to its assigned role's controller, then
        dispatches each red to its assigned RedRole controller against the
        resulting blue plans. Returns (u_p_first, X_p_plans, u_e_first,
        X_e_plans), or all-None on any sub-solve failure (matching
        solve_minimax's failure-signaling convention).

        NEUTRALIZE blues are exempt from the blue-red CBF filter
        (CBFFilterMixin.one_step_cbf_filter), allowing them to make contact
        with their target.

        `target_override`: optional {blue_idx: 4-vector state} for NEUTRALIZE
        agents, e.g. from a vision planner's explicit target_contact_id
        (which may resolve to a red OR a non-hostile bird's state -- this method has
        no notion of "contact"/"bird", it just steers toward whatever state
        it's given). Falls back to the nearest-red heuristic for any
        NEUTRALIZE agent with no override entry (logged, since a
        planner that's supposed to always supply one -- e.g. LLMVisionRolePlanner
        -- silently missing an entry usually signals a bug upstream). DEFEND
        always uses its own nearest-red heuristic regardless of overrides,
        since guarding a perimeter point isn't a single-target action.

        `recon_target_override`: optional {blue_idx: 2-vector waypoint} for
        RECON agents, e.g. a coverage region's center chosen by a role planner
        (see perception.coverage.CoverageTracker / planning.planner.RuleBasedBlueRolePlanner).
        Unlike target_override this is a plain waypoint, not an opposing-team
        state -- ReconController has no nearest-red fallback, so a RECON
        agent with no entry here just keeps its existing random-patrol
        behavior.

        `target_override_ids`: optional {blue_idx: contact_id} naming the
        contact each `target_override` entry came from. Labels only -- it never
        changes what is solved. It exists because this method receives override
        *positions* and so cannot otherwise name what an agent was steered at,
        and `self.effective_targets` is worth nothing if its assigned branch
        cannot say which contact it means.

        `defend_target_override`: optional {blue_idx: ("contact", contact_id)
        | ("station", station_id)} for DEFEND agents.
        **None -- the default -- leaves DEFEND exactly as it was**: screen this
        agent's own nearest believed contact, or hold its own station when it
        believes in none. With an entry, the choice is made by an allocator
        instead of by the agent's nearest-contact heuristic:

        * ("contact", c) -- SCREEN: interpose between contact c and the
          perimeter, using THIS agent's belief about c. Same controller, same
          cost; only who picks the contact changes. If the agent no longer
          believes in c, it falls back to the default behaviour rather than
          invent a target -- the L1 rule.
        * ("station", k) -- hold station k even while contacts are believed,
          which the default cannot do, since any believed contact overrides the
          station there.

        With ``self.defend_screening`` False none of
        the above screens: a DEFEND agent with no entry holds its own station,
        and a ("contact", c) entry or an unknown station raises.

        `bearing_target_override`: optional {blue_idx: 2-vector position} --
        a red currently detected by that specific blue's own sensor (see
        Simulation._resolve_bearing_targets), passed straight through to
        every role's controller.plan(bearing_target=...) regardless of role,
        since bearing-tracking is handled uniformly in
        BaseBlueRoleController, not per-role. A blue with no entry here
        detected nothing this step and keeps its role's own default heading
        behavior.

        `own_detected_reds` ({blue_idx: set of red ids its OWN sensor detects
        this step}) and `pursued_belief_keys` ({blue_idx: belief key} for
        `target_override` entries resolved from that agent's belief) are read
        only under ``self.neutralize_heading == "pursued_contact"``: the
        first labels the heading source, the second names which belief an
        assigned aim point came from. An Experiment II contact-roster id is not
        a belief key and has no entry.
        """
        red_role_assignment = red_role_assignment or {}
        target_override = target_override or {}
        target_override_ids = target_override_ids or {}
        # Rebuilt every solve, never accumulated: a stale entry for an agent
        # that was not solved this step would be read as a current pursuit.
        self.effective_targets = {}
        self.bearing_sources = {}
        # {blue_id: {contact_id: predicted state vector}} -- what THAT agent
        # believes, already extrapolated to now by the belief layer. None means
        # no belief layer is in play, and the pre-M2 ground-truth heuristic is
        # used; Simulation always supplies it.
        believed_targets = {} if believed_targets is None else believed_targets
        recon_target_override = recon_target_override or {}
        bearing_target_override = bearing_target_override or {}
        defend_target_override = defend_target_override or {}
        own_detected_reds = own_detected_reds or {}
        pursued_belief_keys = pursued_belief_keys or {}
        blue_nx, blue_nu = self.blue_nx, self.blue.nu_single
        red_nx, red_nu = self.red_nx, self.red.nu_single
        x_current = np.asarray(x_current, dtype=float).reshape(blue_nx * self.N_p)
        red_states = np.asarray(red_states, dtype=float).reshape(red_nx * self.N_red)

        u_p_first = np.zeros(blue_nu * self.N_p)
        X_p_plans = np.zeros((blue_nx * self.N_p, self.N + 1))

        for i in range(self.N_p):
            own_state = x_current[i * blue_nx : i * blue_nx + blue_nx]
            other_states = (
                np.concatenate(
                    [x_current[j * blue_nx : j * blue_nx + blue_nx] for j in range(self.N_p) if j != i]
                )
                if self.N_p > 1
                else np.zeros(0)
            )

            own_pos = own_state[0:2]
            # The nearest contact THIS AGENT BELIEVES IN. Before M2 this was the
            # nearest *actual* red, computed from ground-truth red_states for
            # every role including DEFEND -- so agents closed on reds nobody had
            # detected, and the fog of war applied to the planner but not to
            # behavior.
            nearest_believed_id, nearest_believed_state = self._nearest_believed(
                own_pos, believed_targets.get(i), red_states, red_nx
            )

            role = role_assignment.get(i, BlueRole.DEFEND)
            controller = self._blue_role_controllers[role]
            held_station = None  # set only by the explicit DEFEND(station) posture
            directed_target = recon_target_override.get(i)
            # What this agent is ACTUALLY steered at this step, recorded
            # alongside every branch below rather than reconstructed from them
            # afterwards. See self.effective_targets.
            effective_source, effective_contact = "station", None
            # The pursuer sensor-heading rule: the belief key of the contact this agent is pursuing, when it
            # pursues one. Only read under the "pursued_contact" heading policy.
            pursued_key = None
            if role == BlueRole.NEUTRALIZE:
                if i in target_override:
                    target_state = target_override[i]
                    effective_source = "assigned"
                    effective_contact = target_override_ids.get(i)
                    pursued_key = pursued_belief_keys.get(i)
                elif nearest_believed_state is not None:
                    print(
                        f"[solve_decentralized] blue {i} has role {role.value} but no "
                        "target_override entry -- falling back to nearest believed contact."
                    )
                    target_state = nearest_believed_state
                    effective_source = "nearest_believed"
                    effective_contact = nearest_believed_id
                    pursued_key = nearest_believed_id
                else:
                    # Assigned to intercept while believing in no contact. The
                    # role is unrealizable this step; hold station rather than
                    # invent a target.
                    target_state = np.zeros(red_nx)
                    directed_target = self.defend_stations.get(i, own_pos.copy())
                    controller = self._blue_role_controllers[BlueRole.DEFEND]
            elif role == BlueRole.DEFEND and not self.defend_screening:
                kind, ident = defend_target_override.get(i, ("station", i))
                if kind != "station":
                    raise ValueError(
                        f"blue {i}: DEFEND override {(kind, ident)!r} would screen, "
                        "and screening is disabled (task set without SCREEN)"
                    )
                if ident not in self.defend_stations:
                    raise ValueError(
                        f"blue {i}: no station {ident!r}; with screening disabled an "
                        "unknown station is refused rather than screened"
                    )
                target_state = np.zeros(red_nx)
                directed_target = self.defend_stations[ident]
                held_station = self.defend_stations[ident]
            elif role == BlueRole.DEFEND and i in defend_target_override:
                kind, ident = defend_target_override[i]
                screened = (
                    believed_targets.get(i, {}).get(ident) if kind == "contact" else None
                )
                if kind == "station" and ident in self.defend_stations:
                    target_state = np.zeros(red_nx)
                    directed_target = self.defend_stations[ident]
                    held_station = self.defend_stations[ident]
                elif screened is not None:
                    target_state = np.asarray(screened, dtype=float).ravel()
                    effective_source = "screen"
                    effective_contact = ident
                elif nearest_believed_state is not None:
                    # Assigned to screen a contact this agent no longer believes
                    # in: fall back exactly as an un-overridden DEFEND would.
                    target_state = nearest_believed_state
                    effective_source = "nearest_believed"
                    effective_contact = nearest_believed_id
                else:
                    target_state = np.zeros(red_nx)
                    directed_target = self.defend_stations.get(i, own_pos.copy())
            elif role == BlueRole.DEFEND:
                if nearest_believed_state is not None:
                    target_state = nearest_believed_state
                    effective_source = "nearest_believed"
                    effective_contact = nearest_believed_id
                else:
                    # No believed contact: guard the assigned station. See
                    # _build_defend_stations.
                    target_state = np.zeros(red_nx)
                    directed_target = self.defend_stations.get(i, own_pos.copy())
            else:
                # RECON ignores target_state entirely -- its cost depends only
                # on own_state and its directed waypoint. Labelled as such so a
                # reader cannot mistake the nearest-believed state passed below
                # for something this agent is pursuing.
                effective_source = "recon_waypoint"
                target_state = (
                    nearest_believed_state
                    if nearest_believed_state is not None
                    else np.zeros(red_nx)
                )
            self.effective_targets[i] = {
                "source": effective_source,
                "contact_id": (
                    None if effective_contact is None else int(effective_contact)
                ),
                "controller": self._controller_role_names[id(controller)],
            }
            # Detection-driven tracking always takes precedence. Only an agent
            # realising an explicit DEFEND(station k) with nothing detected gets
            # the outward posture, and it reaches the heading through the SAME
            # bearing-tracking term and yaw-rate bound detection uses. The
            # target is own position + the outward unit vector, so the heading
            # target is exactly outward wherever the agent is. Like detection
            # tracking, the target angle is atan2-wrapped and the heading state
            # is not, so a first turn can take the long way round; that is the
            # existing yaw behaviour, deliberately not changed here.
            bearing_target = bearing_target_override.get(i)
            bearing_source = None if bearing_target is None else "detection"
            if (
                bearing_target is None
                and held_station is not None
                and self.station_heading == "outward"
            ):
                bearing_target = own_pos + self._outward_unit(held_station)
                bearing_source = "station_outward"
            elif (
                bearing_target is None
                and held_station is not None
                and self.station_heading == "configured"
            ):
                # The station's static configured heading, through the
                # same bearing-tracking term and yaw-rate bound.
                h = self.configured_station_headings[self._station_id_of(held_station)]
                bearing_target = own_pos + np.array([np.cos(h), np.sin(h)])
                bearing_source = "station_configured"
            # The pursuer sensor-heading rule, "pursued_contact": an agent flying the NEUTRALIZE controller
            # points its sensor at the contact it is pursuing -- its OWN live
            # belief about it (already predicted to now; when its own sensor sees
            # the contact this step, that belief is the observation), else the
            # translational aim point -- ahead of any other detection and of the
            # grace hold. Never true state. Nothing is deleted when the believed
            # point is in view and nothing is detected there. The turn takes the
            # nearest equivalent angle. A NEUTRALIZE agent holding station
            # through the DEFEND controller (it believes in nothing) is untouched.
            plan_extra = {}
            if (
                self.neutralize_heading == "pursued_contact"
                and role == BlueRole.NEUTRALIZE
                and controller is self.neutralize_controller
            ):
                believed_pursued = (
                    None if pursued_key is None
                    else believed_targets.get(i, {}).get(pursued_key)
                )
                if believed_pursued is not None:
                    bearing_target = np.asarray(believed_pursued, dtype=float).ravel()[0:2].copy()
                    bearing_source = (
                        "pursued_detection"
                        if pursued_key in own_detected_reds.get(i, ())
                        else "pursued_belief"
                    )
                else:
                    bearing_target = np.asarray(target_state, dtype=float).ravel()[0:2].copy()
                    bearing_source = "pursued_aim"
                plan_extra["bearing_nearest_turn"] = True
            self.bearing_sources[i] = bearing_source
            u0, X_plan, _, ok = controller.plan(
                own_state, target_state, other_states, agent_id=i,
                directed_target=directed_target,
                bearing_target=bearing_target,
                **plan_extra,
            )
            if not ok:
                return None, None, None, None

            u_p_first[i * blue_nu : i * blue_nu + blue_nu] = u0
            X_p_plans[i * blue_nx : i * blue_nx + blue_nx, :] = X_plan

        u_e_first = np.zeros(red_nu * self.N_red)
        X_e_plans = np.zeros((red_nx * self.N_red, self.N + 1))

        for j in range(self.N_red):
            red_state_j = red_states[j * red_nx : j * red_nx + red_nx]
            if j in self.scripted_red_ids:
                # non-hostile contacts: moved by the Simulation's script, never
                # planned here. Plan = constant velocity, control = 0.
                X_e_plans[j * red_nx : j * red_nx + red_nx, :] = np.stack(
                    [red_state_j] * (self.N + 1), axis=1)
                X_e_plans[j * red_nx : j * red_nx + 2, :] += np.outer(
                    red_state_j[2:4], np.arange(self.N + 1) * self.dt)
                continue
            red_role = red_role_assignment.get(j, RedRole.EVADE)
            controller = self._red_role_controllers[red_role]
            if red_believed_blues is None:
                # Historical: the blues' actual nominal plans, gated by this
                # step's team-wide detection.
                blue_trajs, blue_ids = X_p_plans, detected_blue_indices
            else:
                blue_trajs, blue_ids = self._believed_blue_trajectories(
                    red_believed_blues.get(j, {}), blue_nx
                )
            u_e0, X_e_plan, _, ok_e = controller.plan(
                red_state_j, blue_trajs, detected_blue_indices=blue_ids,
                # Every historical red controller discards agent_id; the gate
                # attack controller keys its per-intruder goal on it.
                agent_id=j,
            )
            if not ok_e:
                return None, None, None, None

            u_e_first[j * red_nu : j * red_nu + red_nu] = u_e0
            X_e_plans[j * red_nx : j * red_nx + red_nx, :] = X_e_plan

        return u_p_first, X_p_plans, u_e_first, X_e_plans
