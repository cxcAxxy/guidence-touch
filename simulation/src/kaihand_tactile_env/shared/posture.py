"""One neutral, folded starting posture for every task.

Angles follow the existing seven-joint arm order. Joint 1/3/5/7 change sign
under left/right reflection; the model's calibrated hand mounts are unchanged.
Task approach and grasp waypoints belong to executors, never to reset().
"""

from types import MappingProxyType

import numpy as np

_left = np.deg2rad([112.0, -75.0, -80.0, -120.0, 150.0, 0.0, 0.0])
_right = _left * np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
_left.setflags(write=False)
_right.setflags(write=False)
ARM_HOME = MappingProxyType({"left": _left, "right": _right})
