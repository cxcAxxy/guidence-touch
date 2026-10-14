"""Inspect the far wall using actual shared HEAD RGB and metric depth images.

The fixed vase pose supplies only a search region. No stain alpha, cleaning
progress, object segmentation IDs or privileged inside-camera image is read.
"""

import numpy as np

from kaihand_tactile_env.shared.config import CameraConfig
from kaihand_tactile_env.shared.rendering import WorkcellRenderer

from . import config


class WallInspection:
  def __init__(self, model, *, stain_padding_m=0.007):
    self.stain_padding_m = stain_padding_m
    self.camera = CameraConfig(
      "head", width=960, height=720, depth=True, segmentation=False
    )
    self.renderer = WorkcellRenderer(
      model, (self.camera,), visible_geom_groups=(0, 1, 2)
    )
    self.reference_depth = None
    self.reference_mask = None
    self.bounds = None
    self.observations = []
    self.images = []

  def close(self):
    self.renderer.close()

  def inspect(self, data):
    capture = self.renderer.capture(data, self.camera)
    rgb, depth = capture["rgb"], capture["depth"]
    if self.bounds is None:
      calibration = self.renderer.calibration(data, self.camera)
      xyz = np.array(
        [
          config.VASE_CENTER
          + [config.inner_radius(z) * np.cos(a), config.inner_radius(z) * np.sin(a), z]
          for z in (
            min(config.STAIN_HEIGHTS) - self.stain_padding_m,
            max(config.STAIN_HEIGHTS) + self.stain_padding_m,
          )
          for a in (-0.34, 0.34)
        ]
      )
      camera_xyz = (
        xyz - calibration.world_from_camera[:3, 3]
      ) @ calibration.world_from_camera[:3, :3]
      uv = camera_xyz[:, :2] / -camera_xyz[:, 2:3]
      uv[:, 1] *= -1
      uv = uv * calibration.intrinsic[0, 0] + calibration.intrinsic[:2, 2]
      lo, hi = (
        np.floor(uv.min(axis=0) - 2).astype(int),
        np.ceil(uv.max(axis=0) + 3).astype(int),
      )
      self.bounds = (
        max(0, lo[0]),
        max(0, lo[1]),
        min(rgb.shape[1], hi[0]),
        min(rgb.shape[0], hi[1]),
      )
    x0, y0, x1, y1 = self.bounds
    roi = rgb[y0:y1, x0:x1].astype(float)
    r, g, b = roi.transpose(2, 0, 1)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    # The light-black pigment has a subtle cool cast. Requiring both low
    # luminance and b > r separates it from warm ivory even in wall shadows.
    stain = (luminance < 150) & (b > r + 3) & (b >= g - 2)
    current_depth = depth[y0:y1, x0:x1]
    if self.reference_mask is None:
      if stain.sum() < 40:
        raise RuntimeError("Initial HEAD inspection cannot resolve the far-wall stains")
      self.reference_mask = stain.copy()
      self.reference_depth = current_depth.copy()
    visible = np.isfinite(current_depth) & (
      np.abs(current_depth - self.reference_depth) < 0.006
    )
    visibility = float(visible[self.reference_mask].mean())
    # Track the initial stained pixels, including faint residuals made visible
    # by the task's nonlinear pigment-opacity mapping. Depth guards against
    # interpreting sponge occlusion as a clean surface.
    count = int((stain & self.reference_mask).sum())
    valid = visibility >= 0.98
    result = {
      "time_s": float(data.time),
      "camera": "head",
      "valid": valid,
      "visible_fraction": visibility,
      "stain_pixels": count,
      "initial_stain_pixels": int(self.reference_mask.sum()),
      "stain_pixel_fraction": count / int(self.reference_mask.sum()),
      # Compatibility aliases for existing result consumers. They now count
      # the configured stain mask rather than asserting a red color semantic.
      "red_pixels": count,
      "initial_red_pixels": int(self.reference_mask.sum()),
      "red_pixel_fraction": count / int(self.reference_mask.sum()),
      "visually_clean": bool(valid and count == 0),
      "decision": "clean"
      if valid and count == 0
      else "rewipe"
      if valid
      else "reinspect",
      "roi_xyxy": [int(v) for v in self.bounds],
    }
    selected = stain & self.reference_mask & visible
    if selected.any():
      yy, xx = np.nonzero(selected)
      calibration = self.renderer.calibration(data, self.camera)
      f = calibration.intrinsic[0, 0]
      d = current_depth[selected]
      xyz = np.column_stack(
        (
          (xx + x0 - calibration.intrinsic[0, 2]) * d / f,
          -(yy + y0 - calibration.intrinsic[1, 2]) * d / f,
          -d,
        )
      )
      world = (
        xyz @ calibration.world_from_camera[:3, :3].T
        + calibration.world_from_camera[:3, 3]
      )
      result["residual_center_world_m"] = world.mean(axis=0).tolist()
      result["residual_bounds_world_m"] = [
        world.min(axis=0).tolist(),
        world.max(axis=0).tolist(),
      ]
      # Connected red regions keep a split residual from averaging into an
      # already clean gap. This uses camera pixels, never patch identifiers.
      labels = np.zeros(selected.shape, dtype=int)
      count_regions = 0
      for y, x in zip(*np.nonzero(selected), strict=True):
        if labels[y, x]:
          continue
        count_regions += 1
        pending = [(int(y), int(x))]
        labels[y, x] = count_regions
        while pending:
          py, px = pending.pop()
          for ny, nx in ((py - 1, px), (py + 1, px), (py, px - 1), (py, px + 1)):
            if (
              0 <= ny < selected.shape[0]
              and 0 <= nx < selected.shape[1]
              and selected[ny, nx]
              and not labels[ny, nx]
            ):
              labels[ny, nx] = count_regions
              pending.append((ny, nx))
      regions = []
      for region_id in range(1, count_regions + 1):
        member = labels[selected] == region_id
        if member.sum() < 3:
          continue
        region = world[member]
        # A connected residual can wrap around a clean hole. Target its most
        # populated 9 mm height band, then snap onto a real observed red pixel.
        heights = region[:, 2]
        neighbours = np.abs(heights[:, None] - heights[None, :]) <= 0.0045
        band = region[neighbours[neighbours.sum(axis=1).argmax()]]
        target = band[np.linalg.norm(band - band.mean(axis=0), axis=1).argmin()]
        regions.append(
          {
            "pixel_count": int(member.sum()),
            "center_world_m": target.tolist(),
            "extent_world_m": np.ptp(region, axis=0).tolist(),
          }
        )
      result["residual_regions"] = sorted(
        regions, key=lambda region: region["pixel_count"], reverse=True
      )
    self.observations.append(result)
    self.images.append(rgb)
    return result
