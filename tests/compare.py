import numpy as np
from openlock.common import Action
from openlock.envs.openlock_env import OpenLockEnv
from openlock.logger_env import ActionLog
from openlock.scenario import ScenarioInterface
from openlock.scenarios.CC3 import CommonCause3Scenario
from openlock.scenarios.CC3D import CommonCause3DelayScenario


def test_equiv():
    old = OpenLockEnv()
    old.use_physics = False
    old.initialize_for_scenario("CC3")
    old.setup_trial(scenario_name="CC3", action_limit=3, attempt_limit=10)
    assert old.scenario is not None
    old.reset()

    new = OpenLockEnv()
    new.use_physics = False
    new.initialize_for_scenario("CC3D")
    new.setup_trial(scenario_name="CC3D", action_limit=4, attempt_limit=10)
    assert new.scenario is not None
    new.reset()

    # print(old.observation_space.create_discrete_observation(old))
    # print(new.observation_space.create_discrete_observation(new))

    # assert old.action_space == new.action_space
    # assert old.observation_space == new.observation_space
    old_state, old_labels = old.observation_space.create_discrete_observation(old)
    new_state, new_labels = new.observation_space.create_discrete_observation(new)
    for i in range(len(old_state)):
        label = old_labels[i]
        value = old_state[i]
        if "inactive" in label:
            label = "inactive_" + label[8:]
        new_index = new_labels.index(label)
        assert value == new_state[new_index], label
