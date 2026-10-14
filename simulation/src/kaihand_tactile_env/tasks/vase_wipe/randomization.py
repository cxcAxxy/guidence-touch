"""Episode-local visual stain geometry; never modifies scene files or contacts."""

import numpy as np

from . import config


class StainLayout:
  def __init__(self, model):
    self.ids = np.array(
      [
        model.geom(f"stain_{i}").id
        for i in range(config.PATCH_ROWS * config.PATCH_COLUMNS)
      ]
    )
    self.position = model.geom_pos[self.ids].copy()
    self.quaternion = model.geom_quat[self.ids].copy()
    self.size = model.geom_size[self.ids].copy()
    self.rbound = model.geom_rbound[self.ids].copy()

  def apply(self, model, seed):
    position = self.position.copy()
    quaternion = self.quaternion.copy()
    size = self.size.copy()
    spacing = np.ones(2)
    offset = np.zeros(2)
    scales = np.ones((len(self.ids), 2))
    if seed is not None:
      rng = np.random.default_rng(seed)
      spacing = rng.uniform(0.92, 1.08, 2)
      offset = rng.uniform([-0.0015, -0.001], [0.0015, 0.001])
      scales = rng.uniform(0.90, 1.10, (len(self.ids), 2))
      z = np.mean(config.STAIN_HEIGHTS)
      position[:, 2] = z + (position[:, 2] - z) * spacing[1] + offset[1]
      angle = np.arctan2(position[:, 1], position[:, 0])
      angle = config.STAIN_ANGLE_RAD + (angle - config.STAIN_ANGLE_RAD) * spacing[0]
      angle += offset[0] / config.inner_radius(z)
      radius = np.array([config.inner_radius(h) - 0.0002 for h in position[:, 2]])
      position[:, 0] = radius * np.cos(angle)
      position[:, 1] = radius * np.sin(angle)
      quaternion[:] = 0
      quaternion[:, 0] = np.cos(angle / 2)
      quaternion[:, 3] = np.sin(angle / 2)
      size[:, 1:] *= scales
    model.geom_pos[self.ids] = position
    model.geom_quat[self.ids] = quaternion
    model.geom_size[self.ids] = size
    model.geom_rbound[self.ids] = size.max(axis=1) if seed is not None else self.rbound
    area = scales.prod(axis=1)
    return area, {
      "version": "vase_stains_v1",
      "enabled": seed is not None,
      "seed": seed,
      "spacing_scale_y_z": spacing.tolist(),
      "offset_tangential_vertical_m": offset.tolist(),
      "patch_size_scale_y_z": scales.tolist(),
      "patch_area_scale": area.tolist(),
      "geom_pos_vase_local_m": position.tolist(),
      "geom_quat_wxyz": quaternion.tolist(),
      "geom_size_half_axes_m": size.tolist(),
    }
