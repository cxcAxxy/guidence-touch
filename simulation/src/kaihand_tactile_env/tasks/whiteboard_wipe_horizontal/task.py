"""Horizontal board variant sharing the same contact-driven cleaning model."""

from ..whiteboard_wipe.task import WhiteboardWipeSimulation
from . import config


class HorizontalWhiteboardWipeSimulation(WhiteboardWipeSimulation):
  """A distinct scene and per-instance board frame, never a global override."""

  def __init__(self, *, task_config=config, **kwargs):
    self.post_physics_observer = None
    super().__init__(task_config=task_config, **kwargs)

  def step(self, steps=1):
    """Expose a read-only recorder hook after each integrated physics step."""
    force_sum = 0.0
    for _ in range(steps):
      super().step()
      force_sum += self.board_force
      if self.post_physics_observer is not None:
        self.post_physics_observer(self)
    self.mean_board_force = force_sum / max(steps, 1)
