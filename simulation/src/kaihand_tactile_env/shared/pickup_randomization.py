"""Small, seeded pickup translations, applied only during scene reset."""

import mujoco
import numpy as np


class PickupPositionRandomization:
  def __init__(self, simulation, object_name, *, support_body=None):
    self.object_name = object_name
    self.nominal_pose = simulation._initial_object_pose[object_name].copy()
    self.support_body = support_body
    self.support_id = (
      simulation.model.body(support_body).id if support_body is not None else None
    )
    self.support_position = (
      simulation.model.body_pos[self.support_id].copy()
      if self.support_id is not None
      else None
    )

  def apply(self, simulation, seed):
    if simulation.data.time != 0:
      raise RuntimeError("Pickup randomization is only allowed at reset")
    half_range_m = 0.002
    if seed is not None:
      offset = np.r_[
        np.random.default_rng(seed).uniform(-half_range_m, half_range_m, 2), 0.0
      ]
      simulation.set_object_pose(
        self.object_name, self.nominal_pose[:3] + offset, self.nominal_pose[3:]
      )
    pose = simulation.object_pose(self.object_name).copy()
    offset = pose[:3] - self.nominal_pose[:3]
    support_position = None
    if self.support_id is not None:
      support_position = self.support_position + offset
      simulation.model.body_pos[self.support_id] = support_position
      mujoco.mj_forward(simulation.model, simulation.data)
    return {
      "version": "pickup_xy_v1",
      "enabled": seed is not None,
      "seed": int(seed) if seed is not None else None,
      "object": self.object_name,
      "xy_half_range_m": half_range_m if seed is not None else 0.0,
      "offset_xyz_m": offset.tolist(),
      "initial_pose_xyz_wxyz": pose.tolist(),
      "support_body": self.support_body,
      "support_body_position_m": (
        support_position.tolist() if support_position is not None else None
      ),
    }
