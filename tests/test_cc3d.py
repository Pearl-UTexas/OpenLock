import logging
from itertools import chain
from typing import List, Sequence, TypeVar

import numpy as np
from openlock.common import Action
from openlock.envs.openlock_env import OpenLockEnv
from openlock.logger_env import ActionLog
from openlock.scenarios.CC3D import CommonCause3DelayScenario


def test_constructor():
    scenario = CommonCause3DelayScenario(
        use_physics=False, active_effect_probability=0.7
    )
    assert scenario.levers == list()
    assert scenario.obj_map == dict()


def test_env():
    env = OpenLockEnv()
    env.use_physics = False
    env.initialize_for_scenario("CC3D")
    env.setup_trial(scenario_name="CC3D", action_limit=5, attempt_limit=10)
    assert env.scenario is not None
    env.reset()
    env.step(Action(name="push", obj="l0", params=None))
    assert env.scenario.get_state()["OBJ_STATES"]["l0"] == 1
    env.step(Action(name="push", obj="inactive_0", params=None))


def log_to_action(action_log: ActionLog) -> Action:
    assert action_log.name is not None
    name = action_log.name.split("_")[0]
    target = "_".join(action_log.name.split("_")[1:])
    return Action(name=name, obj=target, params=None)


def fill_wildcard(
    solution: Sequence[ActionLog], actions: Sequence[str]
) -> List[List[ActionLog]]:
    out: List[List[ActionLog]] = [[]]
    for action_log in solution:
        if action_log.name == "*":
            tmp = out
            out = list()
            for wildcard_action in actions:
                for partial_solution in tmp:
                    out.append(
                        partial_solution
                        + [ActionLog(name=wildcard_action, start_time=None)]
                    )
        else:
            out = [partial_solution + [action_log] for partial_solution in out]
    return out


T = TypeVar("T")


def flatten(ll: List[List[T]]) -> List[T]:
    out: List[T] = list()
    for l in ll:
        for item in l:
            out.append(item)

    return out


def test_solutions():
    logging.basicConfig(level="DEBUG")
    env = OpenLockEnv()
    env.use_physics = False
    env.initialize_for_scenario("CC3D")
    env.setup_trial(scenario_name="CC3D", action_limit=5, attempt_limit=10)
    assert env.scenario is not None
    env.reset()
    assert env.action_space is not None

    solutions = flatten(
        [
            fill_wildcard(solution, env.action_space)
            for solution in env.scenario.SOLUTIONS
        ]
    )
    for solution in solutions:
        env.reset()
        logging.debug("New solution")
        for action_log in solution:
            env.step(action=log_to_action(action_log))
            logging.debug(action_log)
            logging.debug(env.scenario._pushed)
            logging.debug(env.scenario._locked)
            logging.debug(env.scenario._timers)
        assert env.get_state()["OBJ_STATES"]["door"] == 1, env.get_state()["OBJ_STATES"]

