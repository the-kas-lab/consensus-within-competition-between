"""The objective-only mission briefs the LLM prompts state.

The blue brief (defender and commander prompts) and the red brief (intruder
prompt) give the objective and the factual mechanics, and prescribe nothing.
They state the same five facts. The episode length comes from the run
configuration and is checked against it by the caller. "Reds your team has not
detected may exist" is in the blue brief only, because the intruder prompt
already tells each red that unseen blues may exist.
"""


def _episode_length(horizon_steps, dt) -> str:
    if horizon_steps is None or dt is None:
        raise ValueError("the mission brief states the episode length; pass the run's "
                         "horizon_steps and dt")
    steps, dt = int(horizon_steps), float(dt)
    return f"The episode lasts {steps} steps (dt = {dt:g} s, {steps * dt:g} s)."


def blue_brief_v1(horizon_steps, dt) -> str:
    return (
        "Mission (the same for every blue team): your team's objective is to prevent any red "
        "from entering the defended region before the episode ends. "
        + _episode_length(horizon_steps, dt)
        + " Any entry by a red is a breach and ends the episode. A blue that touches a red "
        "destroys it and is permanently disabled. A red that leaves the arena is removed and "
        "can no longer breach. The episode also ends, without a breach, once every red has been "
        "destroyed or has left the arena. Reds can change direction at any time, and reds your "
        "team has not detected may exist.")


def red_brief_v1(horizon_steps, dt) -> str:
    return (
        "Your team's objective is for at least one red to enter the defended region before the "
        "episode ends. "
        + _episode_length(horizon_steps, dt)
        + " Any entry by a red is a breach: your team succeeds and the episode ends. A red that "
        "leaves the arena is removed and can no longer breach. The episode also ends, without a "
        "breach, once every red has been destroyed or has left the arena.")
