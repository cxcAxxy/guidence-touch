"""Calibrated configuration for the pick-and-place task."""

from __future__ import annotations

from ...shared.posture import ARM_HOME as ARM_HOME

SCENE_NAME = "pick-place"
OBJECT_NAMES = ("cylinder",)
SCENE_GEOM_NAMES = (
  "cylinder_geom",
  "box_bottom",
  "box_wall_x_pos",
  "box_wall_x_neg",
  "box_wall_y_pos",
  "box_wall_y_neg",
)
OBJECT_STABILIZERS = {
  ("right", "cylinder"): "right_grasp_stabilizer",
}
DEFAULT_PLAYBACK_SPEED = 1.0
DEFAULT_XY_JITTER = 0.01
DEFAULT_YAW_JITTER = 0.05

_FINGERTIP_LINKS = (
  ("thumb", "link6"),
  ("index", "link4"),
  ("middle", "link4"),
  ("ring", "link4"),
  ("pinky", "link4"),
)
_FREE_CLOSE_FORCE_LIMIT = 0.82
_FREE_CLOSE_CUTOFF_M = 0.002
_CLOSE_HOLD_OFFSET_RAD = {
  "thumb": 0.024,
  "index": 0.012,
  "middle": 0.016,
  "ring": 0.040,
  "pinky": 0.040,
}

__all__ = [
  "ARM_HOME",
  "DEFAULT_PLAYBACK_SPEED",
  "DEFAULT_XY_JITTER",
  "DEFAULT_YAW_JITTER",
  "OBJECT_NAMES",
  "OBJECT_STABILIZERS",
  "SCENE_GEOM_NAMES",
  "SCENE_NAME",
]
