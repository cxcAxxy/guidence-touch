"""Explicit contracts for additional, independently identified tasks."""

from __future__ import annotations

import numpy as np

VARIANT_TASKS = (
  "usb-unplug", "whiteboard-wipe-horizontal", "bulb-unscrew",
  "whiteboard-wipe-horizontal-line",
)
WHITEBOARD_VARIANT_INK_COUNTS = {
  "whiteboard-wipe-horizontal": 25,
  "whiteboard-wipe-horizontal-line": 8,
}
VARIANT_SUCCESS_FIELDS = {
  "usb-unplug": (
    "initially_seated_verified", "grasp_verified", "fully_extracted",
    "placed_on_table", "released", "hand_retreated",
  ),
  "bulb-unscrew": (
    "initially_installed_verified", "grasp_verified", "thread_disengaged",
    "fully_extracted", "placed_on_table", "released", "hand_retreated",
  ),
  "whiteboard-wipe-horizontal": (
    "pickup_verified", "released_on_table", "home_return_verified",
  ),
  "whiteboard-wipe-horizontal-line": (
    "pickup_verified", "released_on_table", "home_return_verified",
  ),
}


def validate_variant_outcome(task: str, outcome: dict) -> None:
  """Reject a bare success flag or an insertion/tightening outcome relabelled."""
  if task not in VARIANT_TASKS:
    raise ValueError(f"not an additional task: {task!r}")
  if not isinstance(outcome, dict) or outcome.get("success") is not True:
    raise ValueError(f"{task}: task outcome is not successful")
  if "task" in outcome and outcome["task"] != task:
    raise ValueError(f"{task}: mismatched task outcome: {outcome['task']!r}")
  for name in VARIANT_SUCCESS_FIELDS[task]:
    if outcome.get(name) is not True:
      raise ValueError(f"{task}: physical success gate failed: {name}")
  if task in WHITEBOARD_VARIANT_INK_COUNTS:
    remaining = outcome.get("ink_remaining")
    if (
      not isinstance(remaining, list)
      or len(remaining) != WHITEBOARD_VARIANT_INK_COUNTS[task]
      or any(isinstance(value, bool) for value in remaining)
    ):
      raise ValueError(f"{task}: invalid ink state")
    try:
      values = np.asarray(remaining, dtype=float)
    except (TypeError, ValueError) as error:
      raise ValueError(f"{task}: invalid ink state") from error
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1e-9):
      raise ValueError(f"{task}: ink is not completely cleaned")


def current_variant_environment(task: str) -> dict[str, str]:
  from .config import default_model_path, model_fingerprint, task_config

  if task not in VARIANT_TASKS:
    raise ValueError(f"not an additional task: {task!r}")
  config = task_config(task)
  return {
    "mechanics_version": getattr(config, "MECHANICS_VERSION", f"{task}-v1"),
    "model_fingerprint": model_fingerprint(default_model_path(task)),
  }


def create_variant_expert(task: str, seed: int | None = None):
  """Lazy imports keep collection plans and capability queries model-free."""
  if task == "whiteboard-wipe-horizontal-line":
    from ..tasks.whiteboard_wipe_horizontal_line.execution import (
      HorizontalLineWhiteboardWipeExecutor,
    )
    from ..tasks.whiteboard_wipe_horizontal_line.task import (
      HorizontalLineWhiteboardWipeSimulation,
    )

    sim = HorizontalLineWhiteboardWipeSimulation(ink_seed=seed)
    return sim, HorizontalLineWhiteboardWipeExecutor(sim)
  if task == "whiteboard-wipe-horizontal":
    from ..tasks.whiteboard_wipe_horizontal.execution import (
      HorizontalWhiteboardWipeExecutor,
    )
    from ..tasks.whiteboard_wipe_horizontal.task import (
      HorizontalWhiteboardWipeSimulation,
    )

    sim = HorizontalWhiteboardWipeSimulation(ink_seed=seed)
    return sim, HorizontalWhiteboardWipeExecutor(sim)
  if task == "usb-unplug":
    from ..tasks.usb_unplug.execution import UsbUnplugExecutor
    from ..tasks.usb_unplug.setup import UsbUnplugSimulation

    sim = UsbUnplugSimulation(seed=0 if seed is None else seed)
    return sim, UsbUnplugExecutor(sim)
  if task == "bulb-unscrew":
    from ..tasks.bulb_unscrew.execution import BulbUnscrewExecutor
    from ..tasks.bulb_unscrew.task import BulbUnscrewSimulation

    sim = BulbUnscrewSimulation(position_seed=seed)
    return sim, BulbUnscrewExecutor(sim)
  raise ValueError(f"not an additional task: {task!r}")
