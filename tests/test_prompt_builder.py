"""The prompt carries the agent's belief, and nothing else.

's information-isolation discipline applied to the prompt, per
 The leaks catalogued in
01-plan-vs-repo.md Sec B (L1, L2) were ground truth reaching the *controller*;
the same leak reaching the *prompt* would make any LLM result an artifact in
the same way and would be harder to spot, because a prompt is read by a model
rather than by a solver.

The decisive tests here are the ones that drive the builder with a belief that
*contradicts* ground truth: they establish that the truth cannot appear, rather
than observing that it happens not to.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.belief import LocalBeliefStore  # noqa: E402
from planning.prompt_builder import (  # noqa: E402
    PROMPT_BUILDER_VERSION,
    PromptContext,
    build_system_prompt,
    build_user_message,
)

CTX = PromptContext(arena_size=100.0, defense_center=(50.0, 50.0), neutralize_points=1.0)


def _view(agent_id=0, now=10, contacts=None, teammates=None, n_regions=0):
    """A view off the shipped store, never a hand-rolled stand-in."""
    store = LocalBeliefStore(agent_id, n_regions=n_regions)
    for contact_id, (state, observed_at) in (contacts or {}).items():
        store.observe_target(contact_id, np.asarray(state, dtype=float), observed_at)
    return store.snapshot(
        now=now,
        dt=0.1,
        self_state=np.array([10.0, 10.0, 1.0, 0.0, 0.0]),
        teammates=teammates or {},
    )


# --- the decisive leak tests ----------------------------------------------

def test_the_prompt_states_the_belief_not_the_truth():
    """The decisive test, shaped like test_information_isolation.py's L2.

    One agent, one contact it saw 30 steps ago at a known place. The contact's
    *true* position now is somewhere else entirely. If ground truth can reach
    the prompt, the true coordinates appear in the string.
    """
    view = _view(now=40, contacts={0: ([20.0, 20.0, 0.0, 0.0, 0.0], 10)})
    # Where the red actually is now -- a number this builder is never given and
    # must therefore be incapable of printing.
    true_now = "[88.00, 88.00]"

    prompt = build_user_message(view, CTX)

    assert "contact_id=0" in prompt
    assert "believed_position=[20.00, 20.00]" in prompt
    assert "age_steps=30" in prompt
    assert true_now not in prompt


def test_a_contact_the_agent_does_not_believe_in_is_absent():
    """Fog of war at the prompt: an undetected contact appears nowhere."""
    view = _view(contacts={3: ([20.0, 20.0, 0.0, 0.0, 0.0], 10)})
    prompt = build_user_message(view, CTX)

    assert "contact_id=3" in prompt
    assert "contact_id=7" not in prompt


def test_an_empty_belief_says_so_without_implying_safety():
    """The prompt must not let 'nothing listed' read as 'nothing there' --
    that reading is exactly what fog of war makes false."""
    prompt = build_user_message(_view(contacts={}), CTX)

    assert "you currently believe in no contact" in prompt
    assert "does NOT mean" in prompt


def test_the_prompt_states_no_team_total_and_no_red_count():
    """The two leaks the shipped team prompt has and this one must not inherit
    (design Sec 3b). A LocalWorldView lists only active blues, so a total would
    be a fact with no channel to arrive through."""
    view = _view(
        teammates={1: np.array([30.0, 30.0, 0.0, 0.0, 0.0])},
        contacts={0: ([20.0, 20.0, 0.0, 0.0, 0.0], 10)},
    )
    prompt = build_user_message(view, CTX)

    assert "active=" not in prompt
    assert "num_reds" not in prompt
    assert "num_blues" not in prompt
    # It says what it *does* know: this agent plus the teammates it can see.
    assert "agent_id=1" in prompt


def test_belief_age_reaches_the_prompt():
    """Staleness is the information axis of this entire workstream. A planner
    that cannot see an observation's age cannot reason about it."""
    view = _view(now=100, contacts={0: ([20.0, 20.0, 0.0, 0.0, 0.0], 55)})
    assert "age_steps=45" in build_user_message(view, CTX)


def test_max_age_withholds_a_stale_contact_exactly_as_it_does_for_the_solver():
    """The staleness limit the rule planner is given must mean the same thing
    here, or the two planners would be reasoning over different contact sets at
    identical settings."""
    view = _view(now=100, contacts={0: ([20.0, 20.0, 0.0, 0.0, 0.0], 55)})

    assert "contact_id=0" in build_user_message(view, CTX, max_age=None)
    assert "contact_id=0" in build_user_message(view, CTX, max_age=50)
    assert "contact_id=0" not in build_user_message(view, CTX, max_age=10)


# --- purity, which is what the cache key rests on -------------------------

def test_the_prompt_is_a_pure_function_of_view_and_context():
    """Two views built from identical beliefs must give byte-identical strings.

    The cache key is a hash of this string, so any instability here is a cache
    miss at best and a wrong answer replayed at worst.
    """
    args = dict(now=40, contacts={0: ([20.0, 20.0, 1.5, -0.5, 0.0], 10)})
    assert build_user_message(_view(**args), CTX) == build_user_message(
        _view(**args), CTX
    )


def test_floats_are_fixed_precision():
    """An unstable repr would silently defeat the cache."""
    view = _view(contacts={0: ([20.123456789, 20.0, 0.0, 0.0, 0.0], 10)})
    prompt = build_user_message(view, CTX)

    assert "20.12" in prompt
    assert "20.123456789" not in prompt


def test_the_prompt_changes_when_the_belief_changes():
    """Guards the pair above against passing for the wrong reason: a builder
    that ignored the view entirely would satisfy purity perfectly."""
    near = _view(contacts={0: ([20.0, 20.0, 0.0, 0.0, 0.0], 10)})
    far = _view(contacts={0: ([80.0, 80.0, 0.0, 0.0, 0.0], 10)})
    assert build_user_message(near, CTX) != build_user_message(far, CTX)


def test_context_is_frozen_and_carries_no_step_varying_state():
    """A frozen dataclass wrapping a mutable array is frozen in name only, so
    defense_center is a tuple. Checked because the whole argument for letting
    these fields into the prompt is that they cannot vary."""
    import dataclasses

    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        CTX.arena_size = 200.0
    assert isinstance(CTX.defense_center, tuple)


def test_context_off_a_live_simulation_matches_the_episode_it_describes():
    """Taken off the object rather than hand-built, per the standing rule: a
    hand-built context can drift, and the scoring line would then describe a
    game the episode is not playing."""
    from conftest import build_simulation

    sim = build_simulation(neutralize_points=3.0)
    ctx = PromptContext.from_simulation(sim)

    assert ctx.arena_size == sim.arena_size
    assert ctx.neutralize_points == 3.0
    assert f"worth {3.0} points" in build_system_prompt(ctx)


def test_system_prompt_does_not_depend_on_the_view():
    """Keeps the per-call difference confined to the user message, which is
    where the information axis lives and where an audit has to look."""
    a = build_system_prompt(CTX)
    b = build_system_prompt(CTX)
    assert a == b
    assert "agent_id" not in a


def test_the_builder_version_is_declared():
    assert isinstance(PROMPT_BUILDER_VERSION, str) and PROMPT_BUILDER_VERSION


def test_the_position_shown_is_the_extrapolated_belief_not_the_raw_sighting():
    """Sharpens the leak test above, which uses a stationary contact and so
    could not tell ``position`` from ``predicted_position``.

    A 30-step-old sighting of something moving at 1.0/step is shown where the
    agent *believes* it now is, which is what the solver acts on
    (``_believed_targets``). Printing the raw sighting instead would describe a
    world neither the agent nor the controller is operating in.
    """
    view = _view(now=40, contacts={0: ([20.0, 20.0, 1.0, 0.0, 0.0], 10)})
    prompt = build_user_message(view, CTX)

    # 20.0 + 1.0 * (30 steps * dt 0.1) = 23.0
    assert "believed_position=[23.00, 20.00]" in prompt
    assert "believed_position=[20.00, 20.00]" not in prompt
