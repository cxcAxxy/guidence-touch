"""Local friction-work cleaning law; no contact/time-only disappearance."""

import numpy as np

from . import config


class CleaningProgress:
  def __init__(self, count, *, area_scale=None):
    self.area_scale = (
      np.ones(count)
      if area_scale is None
      else np.asarray(area_scale, dtype=float).copy()
    )
    if (
      self.area_scale.shape != (count,)
      or not np.isfinite(self.area_scale).all()
      or np.any(self.area_scale <= 0)
    ):
      raise ValueError("area_scale must contain one finite positive value per patch")
    self.required_work_j = config.PATCH_WORK_REQUIRED_J * self.area_scale
    self.required_stroke_m = config.PATCH_STROKE_REQUIRED_M * np.sqrt(self.area_scale)
    self.work_j = np.zeros(count)
    self.stroke_m = np.zeros(count)
    self.loaded_time_s = np.zeros(count)
    self.remaining = np.ones(count)

  def update(
    self, normal_n, tangent_n, local_force_n, local_power_w, local_speed_m_s, dt
  ):
    # Thresholds are on measured solver forces, not commanded forces or mu*Fn.
    if not (
      config.MIN_WALL_NORMAL_N <= normal_n <= config.MAX_CLEAN_NORMAL_N
      and tangent_n >= config.MIN_WALL_FRICTION_N
    ):
      return
    active = (
      (local_force_n >= config.MIN_PATCH_FRICTION_N)
      & (local_speed_m_s >= config.MIN_WIPE_SPEED_M_S)
      & (local_power_w > 0)
    )
    self.work_j[active] += local_power_w[active] * dt
    self.stroke_m[active] += local_speed_m_s[active] * dt
    self.loaded_time_s[active] += dt
    progress = np.minimum.reduce(
      [
        self.work_j / self.required_work_j,
        self.stroke_m / self.required_stroke_m,
        self.loaded_time_s / config.PATCH_DWELL_REQUIRED_S,
      ]
    )
    self.remaining = 1 - np.clip(progress, 0, 1)

  @property
  def success(self):
    return bool(
      self.remaining.mean() <= config.MAX_MEAN_DIRT
      and np.all(self.remaining <= config.MAX_PATCH_DIRT)
    )

  def report(self):
    return {
      "law": "thresholded_local_friction_work_area_v2",
      "patch_area_scale": self.area_scale.tolist(),
      "required_work_j": self.required_work_j.tolist(),
      "required_stroke_m": self.required_stroke_m.tolist(),
      "mean_remaining": float(self.remaining.mean()),
      "worst_remaining": float(self.remaining.max()),
      "clean_patch_fraction": float(np.mean(self.remaining <= config.MAX_PATCH_DIRT)),
      "work_j": self.work_j.tolist(),
      "stroke_m": self.stroke_m.tolist(),
      "loaded_time_s": self.loaded_time_s.tolist(),
      "thresholds": {
        key: getattr(config, key)
        for key in (
          "MIN_WALL_NORMAL_N",
          "MAX_CLEAN_NORMAL_N",
          "MIN_WALL_FRICTION_N",
          "MIN_PATCH_FRICTION_N",
          "MIN_WIPE_SPEED_M_S",
          "PATCH_WORK_REQUIRED_J",
          "PATCH_STROKE_REQUIRED_M",
          "PATCH_DWELL_REQUIRED_S",
          "MAX_MEAN_DIRT",
          "MAX_PATCH_DIRT",
        )
      },
    }
