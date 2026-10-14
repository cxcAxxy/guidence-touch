"""Task-independent joint-space trajectory value objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointWaypoint:
  """One timed arm waypoint with its corresponding Cartesian target."""

  phase: str
  joint_positions: np.ndarray
  duration: float
  end_effector_position: np.ndarray
  end_effector_quaternion_wxyz: np.ndarray


__all__ = ["JointWaypoint"]
