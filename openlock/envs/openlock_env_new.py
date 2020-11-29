import copy
import re
import time
from glob import glob
from typing import Optional

import gym  # type: ignore
import numpy as np
import openlock.common as common
from gym.spaces import MultiDiscrete
from openlock.logger_env import ActionLog, TrialLog
from openlock.rewards import RewardStrategy
from openlock.settings_scenario import select_scenario
from openlock.settings_trial import get_trial, select_trial

# TODO(joschnei): All I've really done is cut out the use_physics stuff. I didn't need to do that.
# I'm not about to start supporting it, but I think we can run with the original codebase for now.
# The only reason to use this is if the main version calls something that is only needed when
# you use physics without checking for that.


class ActionSpace:
    def __init__(self):
        pass

    @staticmethod
    def create_action_space(env, obj_map):
        num_levers = len(obj_map.keys()) - 2
        push_action_space = [None] * num_levers
        pull_action_space = [None] * num_levers
        door_action_space = []
        action_map = dict()
        action_map_external_role = dict()
        action_map_role_external = dict()
        for obj, val in list(obj_map.items()):
            if "button" not in obj and "door" not in obj:
                if env.lever_index_mode == "position":
                    name = val.position.name
                else:
                    name = obj
                role = obj
                # use position to map to an integer index
                twod_config = val.position.config
                lever_idx = env.config_to_idx[twod_config]

                # TODO(mjedmonds): refactor this, three mappings is complicated
                name_push = "push_{}".format(name)
                name_pull = "pull_{}".format(name)
                role_push = "push_{}".format(role)
                role_pull = "pull_{}".format(role)

                push_action_space[lever_idx] = name_push
                pull_action_space[lever_idx] = name_pull

                role_push_action = common.Action("push", role, 4)
                role_pull_action = common.Action("pull", role, 4)

                name_push_action = common.Action("push", name, 4)
                name_pull_action = common.Action("pull", name, 4)

                action_map[name_push] = name_push_action
                action_map[name_pull] = name_pull_action

                # role based mapping from external names to internal
                action_map_external_role[name_push] = role_push_action
                action_map_external_role[name_pull] = role_pull_action

                action_map_role_external[role_push] = name_push_action
                action_map_role_external[role_pull] = name_pull_action

            if "button" not in obj and "door" in obj and "door_lock" != obj:
                name_push = "push_{}".format(obj)
                name_action = common.Action("push", obj, 4)

                door_action_space.append(name_push)

                action_map[name_push] = name_action
                action_map_external_role[name_push] = name_action
                action_map_role_external[name_push] = name_action

        action_space = push_action_space + pull_action_space + door_action_space

        return (
            action_space,
            action_map,
            action_map_external_role,
            action_map_role_external,
        )


class ObservationSpace:
    def __init__(self, num_levers, append_solutions_remaining=False):
        self.append_solutions_remaining = append_solutions_remaining
        self.solutions_found = [0, 0, 0]
        self.labels = ["sln0", "sln1", "sln2"]

        if self.append_solutions_remaining:
            self.multi_discrete = self.create_observation_space(
                num_levers, len(self.solutions_found)
            )
        else:
            self.multi_discrete = self.create_observation_space(num_levers)
        self.num_levers = num_levers
        self.state = None
        self.state_labels = None
        self.external_to_role_mapping = None
        self.role_to_external_mapping = None

    @property
    def shape(self):
        return self.multi_discrete.shape

    @staticmethod
    def create_observation_space(num_levers, num_solutions=0):
        discrete_space = []
        num_lever_states = 2
        num_lever_colors = 2
        num_door_states = 2
        num_door_lock_states = 2
        # first num_levers represent the state of the levers
        for i in range(num_levers):
            discrete_space.append(num_lever_states)
        # second num_levers represent the colors of the levers
        for i in range(num_levers):
            discrete_space.append(num_lever_colors)
        discrete_space.append(num_door_lock_states)  # door lock
        discrete_space.append(num_door_states)  # door open
        # solutions appended
        for i in range(num_solutions):
            # solutions can only be found or not found
            discrete_space.append(2)
        discrete_space = np.array(discrete_space)
        multi_discrete = MultiDiscrete(discrete_space)
        return multi_discrete

    def create_internal_state_external_state_mappings(self, env):
        # TODO(mjedmonds): refactor this into a more coherent state/action conversion
        external_to_internal_action_map = env.action_map_external_role
        internal_state_external_state_mapping = dict()
        external_state_internal_state_mapping = dict()
        for (
            external_action_name,
            internal_action,
        ) in external_to_internal_action_map.items():
            external_state_name = external_action_name.split("_", 1)[1]
            internal_state_name = internal_action.obj
            internal_state_external_state_mapping[
                internal_state_name
            ] = external_state_name
            external_state_internal_state_mapping[
                external_state_name
            ] = internal_state_name
        return (
            internal_state_external_state_mapping,
            external_state_internal_state_mapping,
        )

    def create_discrete_observation(self, env):
        # create mapping from internal simulator state to external state
        if self.role_to_external_mapping or self.external_to_role_mapping is None:
            (
                self.role_to_external_mapping,
                self.external_to_role_mapping,
            ) = self.create_internal_state_external_state_mappings(env)
        discrete_state, discrete_labels = self.create_discrete_observation_from_fsm(env)
        # convert internal state labels to external labels
        # TODO(mjedmonds): refactor this, this is a very brittle way of doing this mapping
        for i in range(len(discrete_labels)):
            if discrete_labels[i] in self.role_to_external_mapping.keys():
                discrete_labels[i] = self.role_to_external_mapping[discrete_labels[i]]
            # active indicates color (grey = active. white = inactive) see ENTITY_STATES for values
            if discrete_labels[i].endswith("_active"):
                base_label = discrete_labels[i].split("_", 1)[0]
                if base_label in self.role_to_external_mapping.keys():
                    base_label = self.role_to_external_mapping[base_label]
                discrete_labels[i] = base_label + "_active"
        return discrete_state, discrete_labels

    def create_discrete_observation_from_fsm(self, env):
        """
        constructs a discrete observation from the underlying FSM
        Used when the physics simulator is being bypassed
        :param fsmm:
        :return:
        """
        levers = env.scenario.levers
        self.num_levers = len(levers)
        scenario_state = env.scenario.get_state()

        # need one element for state and color of each lock, need two addition for door lock status and door status
        state = [None] * (self.num_levers * 2 + 2)
        state_labels = [None] * (self.num_levers * 2 + 2)

        # lever states
        for lever in levers:
            lever_idx = env.config_to_idx[lever.position.config]

            # inactive lever, state is constant
            if re.search(common.INACTIVE_LOCK_REGEX_STR, lever.name):
                lever_active = np.int8(common.ENTITY_STATES["LEVER_INACTIVE"])
            else:
                lever_active = np.int8(common.ENTITY_STATES["LEVER_ACTIVE"])

            lever_state = np.int8(scenario_state["OBJ_STATES"][lever.name])

            state_labels[lever_idx] = lever.name
            state[lever_idx] = lever_state

            state_labels[lever_idx + self.num_levers] = lever.name + "_active"
            state[lever_idx + self.num_levers] = lever_active

        # update door state
        door_lock_name = "door_lock"
        door_lock_state = np.int8(scenario_state["OBJ_STATES"][door_lock_name])

        # TODO(mjedmonds): this is a hack to get whether or not the door is actually open; it should be part of the FSM
        door_name = "door"
        door_state = np.int8(scenario_state["OBJ_STATES"][door_name])

        state_labels[-1] = door_name
        state[-1] = door_state
        state_labels[-2] = door_lock_name
        state[-2] = door_lock_state

        return state, state_labels


class OpenLockEnv(gym.Env):
    # Set this in SOME subclasses
    metadata = {"render.modes": ["human"]}  # TODO what does this do?

    def __init__(self):
        self.viewer = None

        # handle to the scenario, defined by the scenario
        self.scenario = None

        self.i = 0
        self.clock = 0
        self.save_path = "../OpenLockResults/"

        self.col_label = []
        self.index_map = None
        self.results = None

        self.attempt_count = 0  # keeps track of the number of attempts
        self.action_count = 0  # keeps track of the number of actions executed
        self.action_limit = None
        self.attempt_limit = None

        self.full_attempt_limit = False

        self.action_executing = False  # used to disable action preemption
        self.pausing = False

        self.human_agent = True
        self.reward_mode = "basic"

        self.lever_index_mode = "role"  # controls whether or not to build action_map based on lever role or position
        self.observation_space = None
        self.action_space = None
        self.action_map = None
        # internal action map to go from external to internal latent action
        self.action_map_external_role = None
        # external action map to go from internal to external action
        self.action_map_role_external = None

        self.reward_strategy = RewardStrategy()
        self.reward_range = (
            self.reward_strategy.REWARD_IMMOVABLE,
            self.reward_strategy.REWARD_OPEN,
        )

        self.effect_probabilities = None

        self.states = []
        self.config_to_idx = dict()
        self.position_to_idx = dict()
        self.idx_to_position = dict()
        self.attribute_order = []
        self.attribute_labels = dict()
        self.attribute_function_map = {
            "position": self.get_obj_position_name,
            "color": self.get_obj_color,
        }
        # current trial to keep track of progress through this trial
        self.cur_trial = None
        # keeps track of current state. TODO(mjedmonds): can this safely be removed
        self.cur_state = None
        self.prev_state = None
        # keeps track of which trials have been completed this execution
        self.completed_trials = []

    def initialize_for_scenario(self, scenario_name):
        self._set_scenario(scenario_name)

        _, lever_configs = get_trial(scenario_name)

        self._set_lever_configs(lever_configs)
        self.config_to_idx = {
            lever_configs[i].LeverPosition.config: i for i in range(len(lever_configs))
        }
        self.position_to_idx = {
            lever_configs[i].LeverPosition.name: i for i in range(len(lever_configs))
        }
        self.idx_to_position = {
            i: lever_configs[i].LeverPosition.name for i in range(len(lever_configs))
        }
        # TODO(mjedmonds): elegantly include door; at this stage of initialization we don't have access to obj_map
        door_idx = len(self.config_to_idx.keys())
        self.config_to_idx[common.ObjectPositionEnum.DOOR.config] = door_idx
        self.position_to_idx["door"] = door_idx
        self.idx_to_position[door_idx] = "door"

        self.states = list(self.position_to_idx.keys())
        self.attribute_order = ["position", "color"]
        self.attribute_labels = {
            "color": common.COLOR_LABELS,
            "position": list(self.position_to_idx.keys()),
        }

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
                space.
        """
        if self.scenario is None:
            print("WARNING: resetting environment with no scenario")

        self.clock = 0
        self._seed()

        # initialize obj_map for scenario
        self.scenario.init_scenario_env()

        (
            self.action_space,
            self.action_map,
            self.action_map_external_role,
            self.action_map_role_external,
        ) = ActionSpace.create_action_space(self, self.obj_map)
        self.observation_space = ObservationSpace(len(self.levers))

        # reset results (must be after world_def exists and action space has been created)
        self._reset_results()

        # reset the finite state machine
        self.scenario.reset()
        self.action_count = 0
        self.cur_trial.add_attempt()

        self.cur_state = self.get_state()
        # append initial observation
        # self._print_observation(state, self.action_count)
        self._append_result(self._create_state_entry())

        self.update_state_machine()

        if self.observation_space is not None:
            (
                discrete_state,
                discrete_labels,
            ) = self.observation_space.create_discrete_observation(self)
            return np.array(discrete_state)
        else:
            raise ValueError(
                "Attempting to reset environment with no observation space. Cannot return state."
            )

    def step(self, action):
        """Run one __timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
                action (Action): desired Action
        Returns:
                observation (dict): END_EFFECTOR_POS : current end effector position
                                          LOCK_STATE : true if door is locked
                info (dict): CONVERGED : whether algorithm succesfully coverged on action
        """
        # save a copy of the current state
        self.prev_state = self.get_state()

        # rendering step
        if not action:
            self.cur_state = self.get_state()
            self.cur_state["SUCCESS"] = False
            self.update_state_machine()
            # no action, return nothing to indicate no reward possible
            return None
        # change to simple "else:" to enable action preemption
        elif self.action_executing is False and self.pausing is False:
            self.action_executing = True
            self.i += 1
            reset = False
            action_success = False
            attempt_success = False
            trial_success = False
            reward = None
            done = False
            observable_action = self._create_pre_obs_entry(action)
            if observable_action:
                # ack is used by manager to determine if the action needs to be logged in the agent's logger
                if self.cur_trial.cur_attempt is None:
                    print("problem")
                self.cur_trial.cur_attempt.add_action(str(action))

            # convert external action to internal action
            if str(action) in self.action_map_external_role.keys():
                action_role = self.action_map_external_role[str(action)]
            else:
                action_role = action
            # execute action
            action_success = self.execute_action(action_role)

            self.i += 1

            self.cur_state = self.get_state()
            self.cur_state["SUCCESS"] = action_success

            if observable_action:
                reward = self.finish_action(action)

                # if 10 < self.cur_trial.cur_attempt.reward < 50:
                #     print('reward is 20...')
                #     reward, _ = self.reward_strategy.determine_reward(self, action, self.reward_mode)

            if self.determine_attempt_finished():
                done = True
                attempt_success = self.determine_unique_solution()

            discrete_state, discrete_labels = self.get_discrete_state()

            self.action_executing = False

            return (
                discrete_state,
                reward,
                done,
                {
                    "action_success": action_success,
                    "attempt_success": attempt_success,
                    "results": self.results,
                    "state_labels": discrete_labels,
                },
            )
        else:
            self.cur_state = self.get_state()
            self.update_state_machine()
            return None
            # return self.state, 0, False, {}

    def _seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

            Note:
                    Some environments use multiple pseudorandom number generators.
                    We want to capture all such seeds used in order to ensure that
                    there aren't accidental correlations between multiple generators.

            Returns:
                    list<bigint>: Returns the list of seeds used in this env's random
                        number generators. The first value in the list should be the
                        "main" seed, or the value which a reproducer should pass to
                        'seed'. Often, the main seed equals the provided 'seed', but
                        this won't be true if seed=None, for example.
            """
        pass

    @property
    def obj_map(self):
        return self.scenario.obj_map

    @property
    def levers(self):
        return self.scenario.levers

    # code to run before human and computer trials
    def setup_trial(
        self,
        scenario_name,
        action_limit,
        attempt_limit,
        specified_trial=None,
        multiproc=False,
    ):
        """
        Set the env class variables and select a trial (specified if provided, otherwise a random trial from the scenario name).

        This method should be called before running human and computer trials.
        Returns the trial selected (string).

        :param scenario_name: name of scenario (e.g. those defined in settings_trial.PARAMS)
        :param action_limit: number of actions permitted
        :param attempt_limit: number of attempts permitted
        :param specified_trial: optional specified trial. If none, get_trial is used to select trial
        :param multiproc: disables printing if running in multiple threads
        :return state: state of env after reset
        :return trial_selected: the selected_trial as returned by get_trial or select_trial
        """
        self._set_scenario(scenario_name)
        # set limits
        self.attempt_count = 0
        self.attempt_limit = attempt_limit
        self.action_limit = action_limit
        # select trial
        if specified_trial is None:
            trial_selected, lever_configs = get_trial(
                scenario_name, self.completed_trials
            )
            if trial_selected is None:
                if not multiproc:
                    print(
                        "WARNING: no more trials available. Resetting completed_trials."
                    )
                    print(self.completed_trials)
                self.completed_trials = []
                trial_selected, lever_configs = get_trial(
                    scenario_name, self.completed_trials
                )
        else:
            trial_selected, lever_configs = select_trial(specified_trial)

        self._set_lever_configs(lever_configs)
        self.observation_space = ObservationSpace(len(self.scenario.levers))

        self.scenario.init_scenario_env(
            world_def=None, effect_probabilities=self.effect_probabilities
        )

        obj_map = self.scenario.obj_map
        (_, _, _, action_map_role_external,) = ActionSpace.create_action_space(
            self, obj_map
        )

        external_solutions = [
            [
                action_map_role_external[str(solution_action)]
                for solution_action in solution
            ]
            for solution in self.scenario.SOLUTIONS
        ]

        self.cur_trial = TrialLog(
            trial_selected, scenario_name, external_solutions, time.time()
        )

        if not multiproc:
            print(
                "INFO: New trial {}. There are {} unique solutions remaining.".format(
                    trial_selected, len(self.scenario.SOLUTIONS)
                )
            )

        return trial_selected

    def finish_trial(self, trial_selected):
        self.completed_trials.append(trial_selected)

    def finish_attempt(self):
        self.attempt_count += 1

        # stores whether or not this attempt executed a unique solution
        action_seq = self.get_current_action_seq(convert_to_action=True)
        attempt_success = self.cur_trial.finish_attempt(self.results, action_seq)

        self.pausing = self.update_user(attempt_success)

        self.cur_trial.add_attempt()
        self.pausing = False

    def finish_action(self, action):
        self.action_count += 1

        # self._print_observation(self.state, self.action_count)
        self._append_result(self._create_state_entry())
        # self.results.append(self._create_state_entry(self.state, self.action_count))

        # must finish action before computing reward
        self.cur_trial.cur_attempt.finish_action(self.results)

        reward, _ = self.reward_strategy.determine_reward(
            self, action, self.reward_mode
        )

        # add reward to current attempt
        self.cur_trial.cur_attempt.add_reward(reward)

        return reward

    def execute_action(self, action_role):
        failure_probability = np.random.sample()
        action_success = self._execute_fsm_action(
            action_role, failure_probability=failure_probability
        )
        return action_success

    def set_effect_probabilities(self, effect_probabilities):
        self.effect_probabilities = effect_probabilities

    def _set_scenario(self, scenario_name):
        # update scenario if needed
        if self.scenario is None or scenario_name is not self.scenario.NAME:
            self.scenario = select_scenario(scenario_name, use_physics=False)

    def _set_lever_configs(self, lever_configs):
        self.scenario.set_lever_configs(lever_configs)

    def _reset_results(self):
        # setup .csv headers
        self.col_label = []
        self.col_label.append("frame")
        discrete_states, discrete_labels = self.get_discrete_state()
        for col_name in discrete_labels:
            self.col_label.append(col_name)
        self.col_label.append("agent")
        for col_name in self.action_space:
            self.col_label.append(col_name)

        self.index_map = {name: idx for idx, name in enumerate(self.col_label)}

        self.results = [self.col_label]

    def get_actions(self):
        return list(self.action_map.keys())

    def get_discrete_state(self):
        (
            discrete_state,
            discrete_labels,
        ) = self.observation_space.create_discrete_observation(self)
        return np.array(discrete_state), discrete_labels

    def _create_state_entry(self):
        frame = self.action_count
        discrete_state, discrete_labels = self.get_discrete_state()
        entry = [0] * len(self.col_label)
        entry[0] = frame
        for name, val in zip(discrete_labels, discrete_state):
            entry[self.index_map[name]] = int(val)

        return entry

    def _create_pre_obs_entry(self, action):
        # create pre-observation entry
        entry = [0] * len(self.col_label)
        entry[0] = self.action_count
        # copy over previous state
        entry[1 : self.index_map["agent"] + 1] = copy.copy(
            self.results[-1][1 : self.index_map["agent"] + 1]
        )

        # mark action idx
        if type(action.obj) is str:
            col = "{}_{}".format(action.name, action.obj)
        else:
            col = action.name

        observable_action = col in self.index_map

        if observable_action:
            entry[self.index_map[col]] = 1
            # append pre-observation entry
            self._append_result(entry)

        return observable_action

    def update_state_machine(self, action=None):
        self.scenario.update_state_machine(action)

    def _print_observation(self, state, count):
        print(str(count) + ": " + str(state["OBJ_STATES"]))
        print(str(count) + ": " + str(state["_FSM_STATE"]))

    def _append_result(self, cur_result):
        self.results.append(cur_result)
        # if len(self.results) > 2:
        #     prev_result = self.results[-1]
        #     # remove frame
        #     differences = [x != y for (x, y) in zip(prev_result[1:], cur_result[1:])]
        #     changes = differences.count(True)
        #     if changes > 2:
        #         print 'WARNING: More than 2 changes between observations'
        #     self.results.append(cur_result)
        # else:
        #     self.results.append(cur_result)

    def update_user(self, attempt_success, multithreaded=False):
        """
        Print update to the user.
        Either all solutions have been found, there are solutions remaining, or the user has
        reached the attempt limit and the trial is over without finding all solutions.

        :param attempt_success:
        :param multithreaded:
        :return: two booleans, the first representing whether the all solutions have been found (trial is finished), the second representing whether the simulator should pause (for when the user opened the door).
        """
        pause = False
        completed_solutions = self.get_completed_solutions()
        num_solutions_remaining = self.get_num_solutions_remaining()
        # continue or end trial
        if self.get_trial_success():
            if not multithreaded:
                print("INFO: You found all of the solutions. ")
            pause = True  # pause if they open the door
        elif self.attempt_count < self.attempt_limit:
            # alert user to the number of solutions remaining
            if attempt_success is True:
                if not multithreaded:
                    print(
                        "INFO: You found a solution. There are {} unique solutions remaining.".format(
                            num_solutions_remaining
                        )
                    )
                pause = True  # pause if they open the door
            else:
                if not multithreaded and self.human_agent:
                    print(
                        "INFO: Ending attempt. Action limit reached. There are {} unique solutions remaining. You have {} attempts remaining.".format(
                            num_solutions_remaining,
                            self.attempt_limit - self.attempt_count,
                        )
                    )
                # pause if the door lock is missing and the agent is a human
                if self.human_agent and self.determine_door_unlocked():
                    pause = True
        else:
            if not multithreaded:
                print(
                    "INFO: Ending trial. Attempt limit reached. You found {} unique solutions".format(
                        len(completed_solutions)
                    )
                )

        return pause

    def get_state(self):
        return self.get_fsm_state()

    def get_fsm_state(self):
        return self.scenario.get_state()

    def get_num_solutions_remaining(self):
        return len(self.get_solutions()) - len(self.get_completed_solutions())

    def get_internal_variable_name(self, obj_name):
        # need to convert to internal object name
        if obj_name in self.observation_space.external_to_role_mapping.keys():
            obj_name = self.observation_space.external_to_role_mapping[obj_name]
        return obj_name

    def get_internal_action_name(self, action_str):
        action_name, obj_name = action_str.split("_", 1)
        obj_name = self.get_internal_variable_name(obj_name)
        return action_name + "_" + obj_name

    def get_obj_color(self, obj_name):
        obj_name = self.get_internal_variable_name(obj_name)
        # TODO(mjedmonds): this is hacky, refactor, but doors and door_locks have no color attribute
        if obj_name == "door_lock" or obj_name == "door":
            return "GREY"
        obj = self.scenario.obj_map[obj_name]
        color = common.COLOR_TO_COLOR_NAME[obj.color]
        return color

    def get_obj_position_name(self, obj_name):
        obj_name = self.get_internal_variable_name(obj_name)
        obj = self.scenario.obj_map[obj_name]
        return obj.position.name

    def get_obj_attributes(self, obj_name):
        """
        returns dict of attribute values for obj_name in the simulator.
        :param obj_name:
        :return:
        """
        obj_name = self.get_internal_variable_name(obj_name)
        obj_attributes = dict()
        for attribute_name in self.attribute_order:
            obj_attributes[attribute_name] = self.attribute_function_map[
                attribute_name
            ](obj_name)
        return obj_attributes

    def get_trial_success(self):
        return self.cur_trial.success

    def get_current_action_seq(
        self,
        convert_to_str=False,
        get_internal_action_seq=False,
        convert_to_action=False,
    ):
        cur_action_sequence = self.cur_trial.cur_attempt.action_seq
        if get_internal_action_seq and self.lever_index_mode != "role":
            cur_action_sequence = [
                ActionLog(self.get_internal_action_name(x.name), x.start_time)
                for x in cur_action_sequence
            ]
        if convert_to_str:
            cur_action_sequence = [str(x) for x in cur_action_sequence]
        if convert_to_action:
            cur_action_sequence = [
                common.Action(a.name.split("_")[0], a.name.split("_")[1], None)
                for a in cur_action_sequence
            ]
        return cur_action_sequence

    def get_completed_solutions(self, convert_to_str=False):
        completed_solutions = self.cur_trial.completed_solutions
        if convert_to_str:
            completed_solutions = [str(x) for x in completed_solutions]
        return completed_solutions

    def get_solutions(self, convert_to_str=False):
        solutions = self.cur_trial.solutions
        if convert_to_str:
            solutions = [str(x) for x in solutions]
        return solutions

    def get_num_solutions(self):
        return len(self.cur_trial.solutions)

    def determine_attempt_finished(self):
        if self.action_count >= self.action_limit:
            return True
        else:
            return False

    def determine_door_unlocked(self):
        return (
            self.get_state()["OBJ_STATES"]["door_lock"]
            == common.ENTITY_STATES["DOOR_UNLOCKED"]
        )

    def determine_door_seq(self):
        # we want the last action to always be push the door, the agent will be punished if the last action is not push the door.
        cur_action_seq = self.get_current_action_seq(convert_to_str=True)
        if len(cur_action_seq) == 3:
            door_act = ActionLog("push_door", None)
            if cur_action_seq[-1] == door_act:
                return 1
            else:
                return -1
        return 0

    # this function also determines if the action sequence is a duplicate to unlock the door, not just open the door
    def determine_unique_solution(self):
        cur_action_seq = self.get_current_action_seq(convert_to_str=True)
        solutions = self.get_solutions(convert_to_str=True)
        # TODO(mjedmonds): need more robust way - assumes solutions are all the same length
        if len(cur_action_seq) != len(solutions[0]):
            return False

        completed_solutions = self.get_completed_solutions(convert_to_str=True)
        # if this is a complete action sequence and it is not a solution, return false
        # full action sequence
        # solution is unique if it is in the list of solutions and not in the solutions found
        if cur_action_seq in solutions and cur_action_seq not in completed_solutions:
            return True
        else:
            return False

    def determine_partial_solution(self):
        """
        Determines if the current action sequence is part of a solution
        :return: True if the current action sequence is part of a solution, False otherwise
        """
        cur_action_seq = self.get_current_action_seq(convert_to_str=True)
        if cur_action_seq in [
            x[: len(cur_action_seq)] for x in self.get_solutions(convert_to_str=True)
        ]:
            return True
        else:
            return False

    def determine_unique_partial_solution(self):
        cur_action_seq = self.get_current_action_seq(convert_to_str=True)
        completed_solutions = self.get_completed_solutions(convert_to_str=True)
        for completed_solution in completed_solutions:
            if cur_action_seq == completed_solution[: len(cur_action_seq)]:
                return False
        # if the partial sequence is not in the completed solutions, just check if the partial sequence is
        # part of the solutions at all
        return self.determine_partial_solution()

    def determine_fluent_change(self):
        prev_fluent_state = self.prev_state["OBJ_STATES"]
        cur_fluent = self.cur_state["OBJ_STATES"]
        return prev_fluent_state != cur_fluent

    def determine_repeated_action(self):
        cur_action_seq = self.get_current_action_seq(convert_to_str=True)
        if len(cur_action_seq) >= 2 and cur_action_seq[-2] == cur_action_seq[-1]:
            return True
        return False

    def determine_moveable_action(self, action):
        """
        determines if the action is movable. Treats all active levers as movable, regardless of FSM
        If you need to detect if the action will cause an effect, negative the determine_fluent_change function
        :param action:
        :return:
        """
        state, labels = self.observation_space.create_discrete_observation_from_fsm(
            self
        )
        obj_name = action.obj
        if obj_name == "door":
            # door being movable depends on door lock
            if state[labels.index("door_lock")] == 1:
                return False
            else:
                return True
        active = state[labels.index(obj_name + "_active")]
        if active:
            return True
        else:
            return False

    def determine_obj_locked(self, obj):
        return self.obj_map[obj].locked

    def _lock_obj(self, obj):
        self.obj_map[obj].lock()

    def _unlock_obj(self, obj):
        self.obj_map[obj].unlock()

    def lock_door(self):
        self._lock_obj("door")

    def unlock_door(self):
        self._unlock_obj("door")

    def lock_lever(self, lever):
        self._lock_obj(lever)

    def unlock_lever(self, lever):
        self._unlock_obj(lever)

    def get_effect_probability(self, obj):
        return self.obj_map[obj].effect_probability

    def _export_results(self):
        save_count = len(glob(self.save_path + "results[0-9]*.csv"))
        np.savetxt(
            self.save_path + "results{}.csv".format(save_count),
            self.results,
            delimiter=",",
            fmt="%s",
        )

    def _execute_fsm_action(self, action, failure_probability):
        action_failed_probabilistically = (
            failure_probability > self.get_effect_probability(action.obj)
        )
        # execute the action if it did not fail probabilistically
        action_success = False
        if not action_failed_probabilistically:
            self.scenario.execute_fsm_action(action)
            action_success = True
        return action_success


def main():
    env = OpenLockEnv()


if __name__ == "__main__":
    main()
