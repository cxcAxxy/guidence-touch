"""Actuator-only travel from shared home to a calibrated approach waypoint."""

import numpy as np

from .simulation import HAND_JOINT_NAMES


def approach_waypoint(simulation, arm, hand, *, phase, seconds=3.0):
  """Yield each 100 Hz observation while moving; never write robot state."""
  names = HAND_JOINT_NAMES["right"]
  start = simulation.arm_goal["right"].copy()
  start_hand = np.array([simulation._hand_targets["right"][n] for n in names])
  count = round(seconds / 0.01)
  for i in range(1, count + 1):
    u = i / count
    alpha = 10 * u**3 - 15 * u**4 + 6 * u**5
    simulation.set_arm_joint_goal("right", start + alpha * (arm - start))
    simulation.set_hand_joint_targets(names, start_hand + alpha * (hand - start_hand))
    simulation.phase = phase
    simulation.step(round(0.01 / simulation.timestep))
    if not np.isfinite(simulation.data.qpos).all() or any(
      warning.number for warning in simulation.data.warning
    ):
      raise RuntimeError(f"{phase}: unstable travel from shared home")
    yield
