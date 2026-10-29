"""The configuration entry point for new experiments (constructor level, no
episode): it selects the pursuer aim-point rule "live" and the pursuer sensor-heading rule "pursued_contact", reaches the
Simulation and its solver through build_episode, and leaves every historical
runner configuration and Simulation default untouched."""

import pytest

from conftest import build_simulation
from scripts import arm_variants, network_conditions
from scripts.experiment_policy import NEW_EXPERIMENT_PURSUIT
from scripts.episode_builder import build_episode


def test_the_new_experiment_selection_is_live_aim_and_pursued_heading():
    assert NEW_EXPERIMENT_PURSUIT == {"neutralize_aim": "live",
                                      "neutralize_heading": "pursued_contact"}


def test_it_reaches_the_simulation_through_build_episode():
    belief, cfg = network_conditions.episode_cfg("multi_bearing", "REF", "C")
    sim = build_episode(2001, belief, dict(cfg, **NEW_EXPERIMENT_PURSUIT))
    assert sim.neutralize_aim == "live"
    assert sim.neutralize_heading == sim.solver.neutralize_heading == "pursued_contact"


def test_historical_defaults_and_runner_configs_are_unchanged():
    sim = build_simulation()
    assert (sim.neutralize_aim, sim.neutralize_heading) == ("frozen", "inherited")
    exp4 = dict(arm_variants.BASE_CFG, **arm_variants.ARMS[2])
    _, exp5 = network_conditions.episode_cfg("multi_bearing", "REF", "C")
    for cfg in (exp4, exp5, network_conditions.BASE_CFG):
        assert not set(NEW_EXPERIMENT_PURSUIT) & set(cfg)


def test_the_competing_teams_safety_configuration_filters_reds_explicitly():
    from scripts.experiment_policy import NEW_EXPERIMENT_INTRUDER_INFO, NEW_EXPERIMENT_SAFETY

    belief, cfg = network_conditions.episode_cfg("multi_bearing", "REF", "C")
    sim = build_episode(2001, belief, dict(cfg, **NEW_EXPERIMENT_SAFETY,
                                           **NEW_EXPERIMENT_INTRUDER_INFO))
    assert sim.safety_filter == "local" and sim.safety_state_sync == "frozen"
    assert sim.enable_red_cbf is True and sim.local_cbf_red_red is True
    assert sim.local_cbf_red_blue is False  # safety.2: no red-side blue rows
    assert sim.local_cbf_blue_cfg.gamma == sim.local_cbf_red_cfg.gamma == 0.05
    assert sim.local_cbf_blue_cfg.dt == sim.dt == 0.1
    assert sim.intruder_info == "shared_beliefs"
    # Historical runner configurations never pick these up.
    exp4 = dict(arm_variants.BASE_CFG, **arm_variants.ARMS[2])
    _, exp5 = network_conditions.episode_cfg("multi_bearing", "REF", "C")
    for c in (exp4, exp5):
        assert not (set(NEW_EXPERIMENT_SAFETY) | set(NEW_EXPERIMENT_INTRUDER_INFO)) & set(c)


@pytest.mark.parametrize("arm", ["C", "I"])
def test_new_experiments_disable_screen_explicitly_and_implicitly(arm):
    from scripts.experiment_policy import NEW_EXPERIMENT_TASKS

    belief, cfg = network_conditions.episode_cfg("multi_bearing", "REF", arm)
    sim = build_episode(2001, belief, dict(cfg, **NEW_EXPERIMENT_TASKS, **NEW_EXPERIMENT_PURSUIT))
    assert sim.cbaa_task_set == "three_task"
    assert sim.solver.defend_screening is False
    assert sim.neutralize_aim == "live" and sim.neutralize_heading == "pursued_contact"


def test_every_new_experiment_setting_reaches_the_simulation_through_build_episode():
    """build_episode forwards cfg keys from an explicit list, so a setting
    added to a policy dict but not to that list is silently dropped (found
 with local_cbf_red_blue). Every policy key must arrive."""
    from scripts import experiment_policy as ep

    belief, cfg = network_conditions.episode_cfg("multi_bearing", "REF", "C")
    policy = dict(ep.NEW_EXPERIMENT_COMMON, **ep.NEW_EXPERIMENT_TASKS)
    assert set(ep.NEW_EXPERIMENT_INFORMATION) <= set(policy)
    sim = build_episode(2001, belief, dict(cfg, **policy))
    for key, value in policy.items():
        assert sim._init_args[key] == value, key

