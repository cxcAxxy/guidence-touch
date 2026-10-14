"""Eight actual ink patches and the original horizontal board physics."""

from ..whiteboard_wipe_horizontal.task import HorizontalWhiteboardWipeSimulation
from . import config


class HorizontalLineWhiteboardWipeSimulation(HorizontalWhiteboardWipeSimulation):
  """Use an independent scene/config while retaining both physical-clock hooks."""

  def __init__(self, **kwargs):
    super().__init__(task_config=config, **kwargs)
