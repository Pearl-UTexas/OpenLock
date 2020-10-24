from __future__ import annotations

import logging
from itertools import product
from typing import Dict, List, Sequence

from transitions import Machine  # type: ignore

from openlock.scenarios.scenario import Scenario


def cartesian_product(*lists: Sequence[str]) -> List[str]:
    """ Given a number of sequences of tokens, returns all strings from the product of tokens."""
    out = list()
    for choice in product(*lists):
        out += ["".join(choice)]
    return out


class FiniteStateMachine:
    def __init__(
        self,
        fsm_manager: FiniteStateMachineManager,
        name: str,
        vars: List[str],
        states: List[str],
        initial_state: str,
    ):
        self.fsm_manager = fsm_manager
        self.name = name
        self.vars = vars
        self.state_permutations = self._permute_states(vars, states)
        self.initial_state = initial_state

        self.machine = Machine(
            model=self,
            states=self.state_permutations,
            initial=self.initial_state,
            ignore_invalid_triggers=True,
            auto_transitions=False,
        )

    @staticmethod
    def _permute_states(vars: Sequence[str], states: Sequence[str]) -> List[str]:
        """ Returns all possible sequences of all variables in all states. """
        assert len(vars) > 0
        assert len(states) > 0

        v_list: List[str] = list()
        for var in vars:
            v_list = cartesian_product(v_list, cartesian_product(var, states))

        return v_list

    def reset(self) -> None:
        self.machine.set_state(self.initial_state)

    def update_manager(self) -> None:
        """
        Tells FSM manager to update the other FSM (latent/observable) based on the changes this FSM (obserable/latent) made
        :return:
        """
        self.fsm_manager.update(self)


class FiniteStateMachineManager:
    """
    Manages the observable and latent finite state machines
    """

    def __init__(
        self,
        scenario: Scenario,
        o_states: List[str],
        o_vars: List[str],
        o_initial: str,
        l_states: List[str],
        l_vars: List[str],
        l_initial: str,
        actions: List[str],
    ):
        self.scenario = scenario
        self.observable_states = o_states
        self.observable_vars = o_vars
        self.observable_initial_state = o_initial

        self.latent_states = l_states
        self.latent_vars = l_vars
        self.latent_initial_state = l_initial

        self.actions = actions

        self.observable_fsm = FiniteStateMachine(
            fsm_manager=self,
            name="observable",
            vars=self.observable_vars,
            states=self.observable_states,
            initial_state=self.observable_initial_state,
        )

        self.latent_fsm = FiniteStateMachine(
            fsm_manager=self,
            name="latent",
            vars=self.latent_vars,
            states=self.latent_states,
            initial_state=self.latent_initial_state,
        )

    def reset(self) -> None:
        """
        resets both the observable fsm and latent fsm
        :return:
        """
        self.observable_fsm.reset()
        self.latent_fsm.reset()

    def get_latent_states(self) -> Dict[str, str]:
        """
        extracts latent variables and their state into a dictonary. key: variable. value: variable state
        :return: dictionary of variables to their corresponding variable state
        """
        latent_states = dict()
        for latent_var in self.latent_vars:
            latent_states[latent_var] = self.extract_entity_state(
                self.latent_fsm.state, latent_var
            )
        return latent_states

        # parses out the state of a specified object from a full state string

    def get_observable_states(self) -> Dict[str, str]:
        """
        extracts observable variables and their state into a dictonary. key: variable. value: variable state
        :return: dictionary of variables to their corresponding variable state
        """
        observable_states = dict()
        for observable_var in self.observable_vars:
            observable_states[observable_var] = self.extract_entity_state(
                self.observable_fsm.state, observable_var
            )
        return observable_states

    def get_internal_state(self) -> str:
        return self.observable_fsm.state + self.latent_fsm.state

    def update(self, messenger: FiniteStateMachine) -> None:
        """ Updates the state space based"""
        if messenger.name == "observable":
            self.update_latent()
        elif messenger.name == "latent":
            self.update_observable()
        else:
            logging.warning(
                f"Asked to update by FSM with name={messenger.name} which is neither observable nor latent."
            )

    def update_latent(self) -> None:
        """
        updates the latent state space according to the scenario
        :return:
        """
        self.scenario.update_latent()

    def update_observable(self) -> None:
        """
        updates the observable state space according to the scenario
        :return:
        """
        self.scenario.update_observable()

    def execute_action(self, action: str) -> None:
        if action in self.actions:
            # changes in observable FSM will trigger a callback to update the latent FSM if needed
            self.observable_fsm.trigger(action)
        else:
            # TODO(mjedmonds): dirty hack to get door pushing action
            if action == "push_door:":
                self.scenario.push_door()
            else:
                raise ValueError(f"unknown action '{action}'")

    @staticmethod
    def extract_entity_state(state: str, obj: str) -> str:
        obj_start_idx = state.find(obj)
        # extract object name + state
        obj_str = state[obj_start_idx : state.find(",", obj_start_idx) + 1]
        # extract state up to next ',', inlcuding the ','
        obj_state = obj_str[obj_str.find(":") + 1 : obj_str.find(",") + 1]
        return obj_state

    # changes obj's state in state (full state) to next_obj_state
    @staticmethod
    def change_entity_state(state: str, entity, next_obj_state: str) -> str:
        # TODO(joschnei): What is an entity? This function is never called.
        # TODO(joschnei): Delete this function as deadcode if I don't find a use for it soon.
        tokens = state.split(",")
        tokens.pop(len(tokens) - 1)  # remove empty string at end of array
        for i in range(len(tokens)):
            token = tokens[i]
            token_lock = token[: token.find(":") + 1]
            # update this token's state
            if token_lock == entity:
                tokens[i] = entity + next_obj_state
            else:
                tokens[
                    i
                ] += ","  # next_obj_state should contain ',', but split removes ',' from all others
        new_state = "".join(tokens)
        return new_state
