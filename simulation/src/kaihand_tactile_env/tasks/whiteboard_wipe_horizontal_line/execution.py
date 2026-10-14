"""Actuator-only top-line wiping with the existing pickup, release and home."""

import numpy as np

from ..whiteboard_wipe_horizontal.execution import HorizontalWhiteboardWipeExecutor


class HorizontalLineWhiteboardWipeExecutor(HorizontalWhiteboardWipeExecutor):
  """Replace only the loaded tangential path, not the contact or cleaning model."""

  position_above_ink_phase = "position_above_line"

  def build_wipe_path(self, first_line_start, *, ink_y, stroke_center_limit):
    # Keep physical loaded overtravel past the last ink patch. At randomization
    # limits the felt must remain inside the side frame, as in the original Z.
    end = np.array(
      [self.config.INK_LINE_VERTICES[1, 0], max(-0.100, -stroke_center_limit - ink_y)]
    )
    target_force = self.config.BOARD_TARGET_FORCE_N
    return [
      (first_line_start, 2.0, 0.0, 0.0, 0.0, target_force, "load_board"),
      (end, 5.5, 0.0, 0.0, target_force, target_force, "wipe"),
    ]

  def run(self):
    result = super().run()
    result["wipe_direction"] = "board_local_y_positive_to_negative"
    result["wipe_pattern"] = self.config.INK_PATTERN
    result["wipe_vertices_board_m"] = self.config.INK_LINE_VERTICES.tolist()
    result["ink_count"] = self.config.INK_COUNT
    result["wipe_stroke_count"] = 1
    return result


def run_episode(simulation, observer=None):
  return HorizontalLineWhiteboardWipeExecutor(simulation, observer=observer).run()
