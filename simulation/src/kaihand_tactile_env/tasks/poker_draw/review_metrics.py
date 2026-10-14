"""Card-specific metrics plugin for the common policy evaluation video."""

from __future__ import annotations

from typing import Any

import numpy as np


class PokerReviewMetrics:
  """Track robotward card displacement without changing simulation state."""

  name = "poker-card-displacement-v1"
  EDGE_THRESHOLD_M = 0.05

  def __init__(self, initial_card_x_m: float, outcome_monitor: Any | None = None) -> None:
    if not np.isfinite(initial_card_x_m):
      raise ValueError("initial card x must be finite")
    self.initial_card_x_m = float(initial_card_x_m)
    self.outcome_monitor = outcome_monitor
    self.current_displacement_m = 0.0
    self.peak_displacement_m = 0.0

  def update(self, simulation: Any) -> None:
    current_x = float(simulation.object_pose("card")[0])
    if not np.isfinite(current_x):
      raise ValueError("card x must be finite")
    self.current_displacement_m = self.initial_card_x_m - current_x
    self.peak_displacement_m = max(
      self.peak_displacement_m, self.current_displacement_m, 0.0
    )

  def snapshot(self, simulation: Any) -> dict[str, float | bool | str | None]:
    current_x = float(simulation.object_pose("card")[0])
    current = self.initial_card_x_m - current_x
    result = {
      "card_displacement_now_mm": current * 1000.0,
      "card_displacement_peak_mm": self.peak_displacement_m * 1000.0,
      "edge_threshold_mm": self.EDGE_THRESHOLD_M * 1000.0,
      "edge_distance_reached": self.peak_displacement_m >= self.EDGE_THRESHOLD_M,
    }
    monitor = self.outcome_monitor
    if monitor is not None:
      if monitor.success:
        progress_stage = "success"
      elif monitor.lifted:
        progress_stage = "lifted"
      elif monitor.edge_reached:
        progress_stage = "edge_reached"
      elif monitor.contacted:
        progress_stage = "contacted"
      else:
        progress_stage = "awaiting_contact"
      result.update(
        success=bool(monitor.success),
        success_stage=progress_stage,
        failure_stage=None if monitor.success else monitor.failure_stage(),
      )
    return result

  def metadata(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "initial_card_x_m": self.initial_card_x_m,
      "direction": "robotward positive: initial_x - current_x",
      "peak_sampling": "update at every physics step, display at video frame rate",
      "edge_threshold_m": self.EDGE_THRESHOLD_M,
      "outcome_stages_in_frames": self.outcome_monitor is not None,
    }
