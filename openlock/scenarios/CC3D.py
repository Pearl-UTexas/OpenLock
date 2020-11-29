from openlock.logger_env import ActionLog
from openlock.scenario import NoFsmScenario
from typing_extensions import Final


class CommonCause3DelayScenario(NoFsmScenario):
    _UNLOCKS: Final = {
        "l0": [("l1", 2), ("l2", 2)],
        "l1": [("door", 1)],
        "l2": [("door", 1)],
    }

    _INIT_LOCKED: Final = {"l0": False, "l1": True, "l2": True, "door": True}
    _INIT_PUSHED: Final = {"l0": False, "l1": False, "l2": False, "door": False}

    NAME: Final[str] = "CC3D"

    SOLUTIONS: Final = [
        [
            ActionLog("push_l0", None),
            ActionLog("*", None),
            ActionLog("push_l1", None),
            ActionLog("push_door", None),
        ],
        [
            ActionLog("push_l0", None),
            ActionLog("*", None),
            ActionLog("push_l2", None),
            ActionLog("push_door", None),
        ],
    ]

