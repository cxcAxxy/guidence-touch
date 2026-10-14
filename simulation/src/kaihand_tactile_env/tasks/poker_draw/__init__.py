"""Poker-draw task package.

The implementation is intentionally not imported here so shared model selection
can load :mod:`.config` without creating a task/controller import cycle.
"""

from .config import (
  ARM_HOME,
  DEFAULT_PLAYBACK_SPEED,
  DEFAULT_XY_JITTER,
  DEFAULT_YAW_JITTER,
  OBJECT_NAMES,
  OBJECT_STABILIZERS,
  SCENE_GEOM_NAMES,
  SCENE_NAME,
)

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
