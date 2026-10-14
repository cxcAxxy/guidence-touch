"""Read-only full SE(3) samples from the same cache used by RGB rendering."""

from __future__ import annotations

import numpy as np

SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "little")


class TaskspaceCapture:
  """Native robot site frames; does not invoke forward dynamics or mutate data."""

  def __init__(self, simulation):
    self.sim = simulation
    if not hasattr(simulation, "observation_time"):
      raise ValueError(
        "taskspace capture requires an explicit cached-observation clock"
      )
    model = simulation.model
    self.wrist_names = [f"hand_{side}_base_link_site" for side in ("l", "r")]
    self.finger_names = [
      [
        f"hand_{side}_{'pinky' if finger == 'little' else finger}_link{6 if finger == 'thumb' else 4}_site"
        for finger in FINGERS
      ]
      for side in ("l", "r")
    ]
    self.wrist_ids = np.array([model.site(name).id for name in self.wrist_names])
    self.finger_ids = np.array(
      [[model.site(name).id for name in names] for names in self.finger_names]
    )

  def read(self):
    data = self.sim.data

    def poses(ids):
      result = np.broadcast_to(np.eye(4), (*ids.shape, 4, 4)).copy()
      result[..., :3, :3] = data.site_xmat[ids].reshape(*ids.shape, 3, 3)
      result[..., :3, 3] = data.site_xpos[ids]
      return result

    return (
      float(self.sim.observation_time),
      poses(self.wrist_ids),
      poses(self.finger_ids),
    )
