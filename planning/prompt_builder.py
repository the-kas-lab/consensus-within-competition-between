"""The user message one blue's LLM planner sees, built from its belief alone.

**The rule this module exists to enforce:** the prompt is a pure function of
``(LocalWorldView, PromptContext)``, where the context carries *static scenario
constants only*. Nothing else is in scope. See


This is's L1/L2 discipline applied
to the prompt. L1 and L2 were ground-truth leaks into the *controller*: agents
chased reds nobody had detected, and a committed interceptor tracked the real
red however falsified its belief was. The same leak into the *prompt* would make
any later LLM result an artifact in exactly the same way, and would be harder to
notice, because a prompt is read by a model rather than by a solver.

Two leaks the shipped team prompt has, which this one deliberately does not
inherit. Neither is a defect there -- a team planner is told the team's
composition by construction:

* **Team size.** ``_format_observation`` emits ``blue_agents (active=N/M)``,
  where M counts the disabled. A LocalWorldView lists only *active* blues
  ("liveness is part of the channel", ``perception/belief/view.py``), so this
  prompt states the active set and **no total**.
* **Red count.** The team observation carries ``num_reds`` beside its FOV-gated
  list. This prompt states **believed contacts only**, each with its age, and
  says plainly that more may exist.

Floats are formatted at fixed precision. Not cosmetic: the cache key is a hash
of this string, so an unstable repr would silently defeat the cache.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Bumped whenever the text below changes in any way that could change a model's
# answer. It is part of the cache key, so a bump retires every stale entry
# instead of replaying answers to a question no longer being asked.
PROMPT_BUILDER_VERSION = "m7.2"


@dataclass(frozen=True)
class PromptContext:
    """Static scenario constants the agent is entitled to know.

    The rules of the game and the shape of the board -- never information about
    the adversary, and never anything that varies step to step. ``LocalWorldView``
    carries none of these, which is why they need a home: the LLM planner spec's
    "from LocalWorldView only" read literally would forbid telling an agent how
    big the arena is or what a kill is worth, which makes the prompt worse
    without making it more honest.

    Frozen, and built once at planner construction. ``defense_center`` is a
    plain tuple rather than an ndarray so the frozenness is real: a frozen
    dataclass wrapping a mutable array is frozen in name only.

    Precedent for treating fixed geometry as non-information:
    ``tests/test_per_agent_planning.py::test_region_centers_are_static_geometry_not_information``.
    """

    arena_size: float
    defense_center: Optional[Tuple[float, float]] = None
    neutralize_points: float = 1.0
    # The episode length.
    horizon_steps: Optional[int] = None
    dt: Optional[float] = None

    @staticmethod
    def from_simulation(sim, neutralize_points: Optional[float] = None):
        """Build the context off a live Simulation.

        Preferred over hand-constructing one, per the standing rule about taking
        objects off a live Simulation: a hand-built context can drift from what
        the episode actually applies, and the scoring line would then describe a
        game the agent is not playing.
        """
        center = getattr(sim, "defense_center", None)
        if center is not None:
            center = tuple(float(v) for v in np.asarray(center, dtype=float).reshape(2))
        return PromptContext(
            arena_size=float(sim.arena_size),
            defense_center=center,
            neutralize_points=float(
                sim.neutralize_points if neutralize_points is None else neutralize_points
            ),
            horizon_steps=int(sim.sim_steps),
            dt=float(sim.dt),
        )


def _vec(values) -> str:
    v = np.asarray(values, dtype=float).ravel()
    return "[" + ", ".join(f"{x:.2f}" for x in v) + "]"


def build_system_prompt(ctx: PromptContext) -> str:
    """The role definitions and the scoring rule.

    Depends on the context only, never on the view, so it is identical across
    every agent and every epoch of a run. That is deliberate: it keeps the
    per-call difference confined to the user message, which is where the
    information axis lives and where an audit should have to look.
    """
    return f"""You are ONE defending agent (BLUE) in a 2D pursuit-evasion simulation. Blue defends a region against RED intruders. You decide your OWN role for this planning cycle, for yourself only -- you are not commanding the team, and no one is commanding you.

Choose exactly one role:

- DEFEND: hold a guard station on the perimeter of the defended region.
- NEUTRALIZE: close to physical contact with one specific contact you believe in. Contact is intended and no safety margin is applied. Consequence: you are PERMANENTLY DISABLED afterward and take no further part, and the kill is worth {ctx.neutralize_points} points. It is a trade of yourself for one intruder -- weigh it as one.
- RECON: patrol toward a coverage region you choose, sweeping your sensor, to re-establish eyes on ground that has not been swept recently. Not a way to park yourself idle.

What you know is what YOU believe, and it is incomplete. Your sensor is a cone ahead of your heading. The contacts listed below are the ones you currently believe in, each with the age in simulation steps of the observation it rests on -- an age of 0 is something seen this step, a large age is a guess extrapolated from an old sighting. **Contacts you cannot see are not listed, and an empty or short list does NOT mean the region is safe.** You are not told how many intruders exist.

Your teammates' positions are known to you exactly; their intentions are not, and you cannot talk to them this cycle. Other agents are deciding at the same moment on their own beliefs, so a contact you leave alone may or may not be covered by someone else.

The coverage list shows, for a fixed grid of regions, how many steps it has been since your knowledge of that region was last refreshed (-1 means never). Higher means an intruder could have been sitting there unseen for longer, and it is the main signal for where RECON is worth sending.

If you choose NEUTRALIZE you must set target_contact_id to one of the contact_id values listed below -- you cannot pursue something you do not believe in. Otherwise set it to -1.
If you choose RECON you must set target_region_id to one of the region_id values listed below. Otherwise set it to -1.
Give one sentence of reasoning that refers to the actual picture -- distances, ages, coverage staleness, what you are trading -- not a restatement of the role definition.

Respond only with a valid JSON object matching the required schema. The "role" value must be exactly one of the lowercase strings "defend", "neutralize" or "recon" -- not uppercase, and not the name of the role as written above. Always include all four fields, using -1 for whichever target does not apply."""


# Sensor geometry and angle convention shown to every heading-aware prompt
# version (m8b.3, commander.3, intruder.3). The numbers are perception.sensing's
# defaults; runners that use heading-aware prompts check the episode uses them.
SENSOR_GEOMETRY_TEXT = (
    "Headings: every agent's sensor points along its heading and sees a 60-degree "
    "wedge (30 degrees either side) out to 35 units; nothing outside it is "
    "detected. Angles are in degrees, counter-clockwise from the +x axis "
    "(0 = +x, 90 = +y, 180 = -x, 270 = -y). Steering and sensing are "
    "independent: an agent can move one way while looking another. A heading "
    "listed for an opponent is the one seen at that sighting and is NOT updated "
    "since (agents can turn up to 360 degrees per second)."
)


def heading_deg(theta) -> int:
    """A heading in whole degrees in [0, 360), counter-clockwise from +x
    (rounded BEFORE wrapping, so a hair below 0 prints 0, never 360)."""
    return int(round(float(np.degrees(float(theta))))) % 360


def build_user_message(view, ctx: PromptContext, max_age=None, with_headings: bool = False) -> str:
    """One agent's situation, entirely from its own frozen snapshot.

    ``max_age`` is the consumer's staleness limit and matches what the rule
    planner is given (``Simulation.belief_validity_age_steps``). ``None`` means
    the agent reasons over everything it knows, stale included -- the store's
    own semantics -- and the ages are printed either way so the model can
    discount an old sighting itself rather than have it silently withheld.
    """
    lines = [
        f"sim_step: {int(view.now)}",
        f"arena_size: {ctx.arena_size:.2f}",
    ]
    if ctx.defense_center is not None:
        lines.append(f"defense_center: {_vec(ctx.defense_center)}")

    me = np.asarray(view.self_state, dtype=float).ravel()
    lines.append(f"you are blue agent_id={int(view.agent_id)}")
    lines.append(f"  your_position: {_vec(me[0:2])}")
    lines.append(f"  your_velocity: {_vec(me[2:4])}")
    if me.size >= 5:
        # Heading is what the sensor cone points along, so it is the difference
        # between "I could look there" and "I am looking there".
        if with_headings:
            lines.append(f"  your_heading_deg: {heading_deg(me[4]):.0f}")
        else:
            lines.append(f"  your_heading_rad: {me[4]:.2f}")

    # Active teammates only, and no total: the view lists only active blues by
    # design, and a total would be a team-level fact this agent has no channel
    # for. Sorted so the string is stable, which the cache key depends on.
    lines.append(
        "active_teammates (positions known exactly; intentions unknown, no "
        "communication this cycle):"
    )
    if not view.teammates:
        lines.append("  (none -- every other defender is out of action)")
    for agent_id in sorted(view.teammates):
        s = np.asarray(view.teammates[agent_id], dtype=float).ravel()
        heading = (f" heading_deg={heading_deg(s[4]):.0f}"
                   if with_headings and s.size >= 5 else "")
        lines.append(
            f"  agent_id={int(agent_id)} pos={_vec(s[0:2])} vel={_vec(s[2:4])}{heading}"
        )

    believed = view.fresh_targets(max_age)
    lines.append(
        "contacts you believe in (YOUR belief only -- others may exist that you "
        "cannot see; age_steps is how old the underlying observation is):"
    )
    if not believed:
        lines.append(
            "  (none -- you currently believe in no contact. This does NOT mean "
            "there is none.)"
        )
    for contact_id in sorted(believed):
        belief = believed[contact_id]
        # The *believed* position, extrapolated from the observation the agent
        # actually holds -- never the contact's true current state. This line is
        # the single place a ground-truth leak would enter the prompt.
        # Heading as OBSERVED (never extrapolated), only when the belief
        # record carries one -- i.e. it came from a sighting.
        st = np.asarray(belief.state, dtype=float).ravel()
        heading = (f" observed_heading_deg={heading_deg(st[4]):.0f}"
                   if with_headings and st.size >= 5 else "")
        lines.append(
            f"  contact_id={int(contact_id)} "
            f"believed_position={_vec(belief.predicted_position(view.now, view.dt))} "
            f"believed_velocity={_vec(belief.velocity)}{heading} "
            f"age_steps={int(belief.age(view.now))}"
        )

    regions = view.coverage_regions()
    if regions:
        lines.append(
            "coverage (steps since your knowledge of each region was refreshed; "
            "-1 = never):"
        )
        for region in regions:
            c = np.asarray(region["center"], dtype=float).ravel()
            lines.append(
                f"  region_id={int(region['region_id'])} center={_vec(c)} "
                f"steps_since_seen={int(region['steps_since_seen'])}"
            )

    return "\n".join(lines)
