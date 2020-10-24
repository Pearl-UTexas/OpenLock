from typing import List, Optional

from openlock.envs.world_defs.openlock_def import ArmLockDef
from openlock.finite_state_machine import FiniteStateMachineManager
from openlock.logger_env import ActionLog
from openlock.scenario import Scenario
from typing_extensions import Final


class CommonCause3Scenario(Scenario):

    NAME: Final[str] = "CC3"

    OBSERVABLE_STATES: Final[List[str]] = [
        "pulled,",
        "pushed,",
    ]  # '+' -> locked/pulled, '-' -> unlocked/pushed

    # TODO(mjedmonds): make names between obj_map in env consistent with names in FSM (extra ':' in FSM)
    OBSERVABLE_VARS: Final[List[str]] = ["l0:", "l1:", "l2:"]
    OBSERVABLE_INITIAL_STATE: Final[str] = "l0:pulled,l1:pulled,l2:pulled,"

    LATENT_STATES: Final = ["unlocked,", "locked,"]  # '+' -> open, '-' -> closed
    LATENT_VARS: Final = ["door:"]
    LATENT_INITIAL_STATE: Final = "door:locked,"

    ACTIONS: Final = (
        ["nothing"]
        + ["pull_{}".format(lock) for lock in OBSERVABLE_VARS]
        + ["push_{}".format(lock) for lock in OBSERVABLE_VARS]
    )

    # lists of actions that represent solution sequences
    SOLUTIONS: Final = [
        [
            ActionLog("push_l0", None),
            ActionLog("push_l1", None),
            ActionLog("push_door", None),
        ],
        [
            ActionLog("push_l0", None),
            ActionLog("push_l2", None),
            ActionLog("push_door", None),
        ],
    ]

    def __init__(self, use_physics: bool = True):
        super(CommonCause3Scenario, self).__init__(use_physics=use_physics)

        self.fsmm = FiniteStateMachineManager(
            scenario=self,
            o_states=self.OBSERVABLE_STATES,
            o_vars=self.OBSERVABLE_VARS,
            o_initial=self.OBSERVABLE_INITIAL_STATE,
            l_states=self.LATENT_STATES,
            l_vars=self.LATENT_VARS,
            l_initial=self.LATENT_INITIAL_STATE,
            actions=self.ACTIONS,
        )

        self.lever_configs = None
        self.lever_opt_params = None

        # define observable states that trigger changes in the latent space;
        # this is the clue between the two machines.
        # Here we assume if the observable case is in any criteria than those listed, the door is locked
        self.door_unlock_criteria = [
            s
            for s in self.fsmm.observable_fsm.state_permutations
            if "l1:pushed," in s or "l2:pushed," in s
        ]

        # add unlock/lock transition for every lock
        for lock in self.fsmm.observable_fsm.vars:
            if lock == "l1:" or lock == "l2:":
                pulled = [
                    s
                    for s in self.fsmm.observable_fsm.state_permutations
                    if lock + "pulled," in s and "l0:pushed," in s
                ]
                pushed = [
                    s
                    for s in self.fsmm.observable_fsm.state_permutations
                    if lock + "pushed," in s and "l0:pushed," in s
                ]
                super(CommonCause3Scenario, self).add_no_ops(lock, pushed, pulled)
            else:
                pulled = [
                    s
                    for s in self.fsmm.observable_fsm.state_permutations
                    if lock + "pulled," in s
                ]
                pushed = [
                    s
                    for s in self.fsmm.observable_fsm.state_permutations
                    if lock + "pushed," in s
                ]
            for pulled_state, pushed_state in zip(pulled, pushed):
                # these transitions need to change the latent FSM, so we update the manager after executing them
                self.fsmm.observable_fsm.machine.add_transition(
                    trigger=f"pull_{lock}",
                    source=pushed_state,
                    dest=pulled_state,
                    after="update_manager",
                )
                self.fsmm.observable_fsm.machine.add_transition(
                    trigger=f"push_{lock}",
                    source=pulled_state,
                    dest=pushed_state,
                    after="update_manager",
                )

        super(CommonCause3Scenario, self).add_nothing_transition()

        super(CommonCause3Scenario, self).add_door_transitions()

    def update_latent(self) -> None:
        """
        logic to transition in the latent state space based on the observable state space, if needed
        """
        super(CommonCause3Scenario, self).update_latent()

    def update_observable(self) -> None:
        """
        updates observable fsm based on some change in the observable fsm, if needed
        """
        super(CommonCause3Scenario, self).update_observable()

    def update_state_machine(self, action: Optional[str] = None) -> None:
        """
        Updates the finite state machines according to object status in the Box2D environment
        """
        super(CommonCause3Scenario, self).update_state_machine(action)

    def init_scenario_env(
        self, world_def: Optional[ArmLockDef] = None, effect_probabilities=None
    ) -> None:
        """
        initializes the scenario-specific components of the box2d world (e.g. levers)
        :return:
        """

        super(CommonCause3Scenario, self).init_scenario_env(
            world_def, effect_probabilities
        )

        if self.use_physics:
            self.obj_map["l1"].lock()  # initially lock l1
            self.obj_map["l2"].lock()  # initially lock l2

    def _update_env(self) -> None:
        """
        updates the Box2D environment based on the state of the finite state machine
        """
        super(CommonCause3Scenario, self)._update_env()

    def _update_latent_objs(self) -> None:
        """
        updates latent objects in the Box2D environment based on state of the latent finite state machine
        """
        super(CommonCause3Scenario, self)._update_latent_objs()

    def _update_observable_objs(self) -> None:
        """
        updates observable objects in the Box2D environment based on the observable state of the finite state machine
        """
        observable_states = self.fsmm.get_observable_states()
        for observable_var in observable_states.keys():
            # ---------------------------------------------------------------
            # add code to change part of the environment based on the state of an observable variable here
            # ---------------------------------------------------------------
            if observable_var == "l1:" or observable_var == "l2:":
                var = observable_var[:-1]  # Strip colon
                # l1 and l2 unlock if l0 is pushed
                if "l0:pushed," in self.fsmm.observable_fsm.state:
                    self.obj_map[var].unlock()
                else:
                    self.obj_map[var].lock()

