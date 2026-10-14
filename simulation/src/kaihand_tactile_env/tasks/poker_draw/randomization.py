"""Bounded, reproducible card-pose variation at reset, never during motion.

The parameter ceiling is a geometry/scope guard, not a measured success claim.
The zero-noise middle preset and shared robot reset defaults remain unchanged.
"""

from __future__ import annotations

from itertools import product

import numpy as np

RANDOMIZED_PRESET = "middle-force-randomized-v1"
RANDOMIZATION_SCHEMA = "poker-initial-card-randomization-v1"
DEFAULT_XY_JITTER_M = 0.002
MAX_XY_JITTER_M = 0.005
MAX_YAW_JITTER_RAD = float(np.deg2rad(1.0))
MIN_TABLE_MARGIN_M = 0.00025


def validate_randomization_bounds(xy_jitter_m: float, yaw_jitter_rad: float) -> None:
  values = np.asarray([xy_jitter_m, yaw_jitter_rad], dtype=float)
  if not np.all(np.isfinite(values)) or np.any(values < 0):
    raise ValueError("card randomization bounds must be finite and nonnegative")
  if xy_jitter_m > MAX_XY_JITTER_M or yaw_jitter_rad > MAX_YAW_JITTER_RAD:
    raise ValueError(
      "card randomization is limited to 5 mm per XY axis and 1 degree yaw; limits do not guarantee task success"
    )


def sample_card_offsets(seed: int, xy_jitter_m: float, yaw_jitter_rad: float):
  """Match shared reset draw order exactly, including skipped zero-width draws."""
  validate_randomization_bounds(xy_jitter_m, yaw_jitter_rad)
  if (
    isinstance(seed, (bool, np.bool_))
    or not isinstance(seed, (int, np.integer))
    or seed < 0
  ):
    raise ValueError("randomization seed must be a nonnegative integer")
  rng = np.random.default_rng(seed)
  xy = rng.uniform(-xy_jitter_m, xy_jitter_m, size=2) if xy_jitter_m else np.zeros(2)
  yaw = float(rng.uniform(-yaw_jitter_rad, yaw_jitter_rad)) if yaw_jitter_rad else 0.0
  return xy, yaw


def offset_card_pose(nominal_pose_wxyz, offset_xy_m, yaw_offset_rad):
  nominal = np.asarray(nominal_pose_wxyz, dtype=float)
  xy = np.asarray(offset_xy_m, dtype=float)
  if (
    nominal.shape != (7,)
    or xy.shape != (2,)
    or not np.all(np.isfinite(nominal))
    or not np.all(np.isfinite(xy))
    or not np.isfinite(yaw_offset_rad)
  ):
    raise ValueError("invalid card pose or offset")
  if not np.isclose(np.linalg.norm(nominal[3:]), 1.0, atol=1e-9):
    raise ValueError("nominal quaternion must have unit norm")
  result = nominal.copy()
  result[:2] += xy
  # World-Z yaw left-multiplies the nominal quaternion; height/tilt preserved.
  c, s = np.cos(yaw_offset_rad / 2), np.sin(yaw_offset_rad / 2)
  w, x, y, z = nominal[3:]
  result[3:] = [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w]
  return result


def validate_recorded_randomization(
  metadata, seed, xy_jitter_m, yaw_jitter_rad
) -> bool:
  """Pure reproducibility check for resume; no simulator or mutable RNG state."""
  try:
    xy, yaw = sample_card_offsets(seed, xy_jitter_m, yaw_jitter_rad)
    if not isinstance(metadata, dict):
      return False
    if any(
      key not in metadata or metadata.get(key) != value
      for key, value in {
        "schema_version": RANDOMIZATION_SCHEMA,
        "mode": "seeded_uniform",
        "distribution": "independent_uniform",
        "seed": seed,
        "xy_jitter_m": xy_jitter_m,
        "yaw_jitter_rad": yaw_jitter_rad,
        "perturbation_scope": "reset_only_card_xy_yaw",
        "height_unchanged": True,
        "tilt_unchanged": True,
        "observation_noise": None,
        "action_noise": None,
      }.items()
    ):
      return False
    expected = offset_card_pose(metadata["nominal_pose_wxyz"], xy, yaw)
    actual = np.asarray(metadata["sampled_pose_wxyz"], dtype=float)
    if actual.shape != (7,) or not np.all(np.isfinite(actual)):
      return False
    support = metadata["table_support"]
    return bool(
      np.allclose(metadata["sampled_offset_xy_m"], xy, atol=1e-12, rtol=0)
      and np.isclose(metadata["sampled_yaw_offset_rad"], yaw, atol=1e-12, rtol=0)
      and np.allclose(actual[:3], expected[:3], atol=1e-12, rtol=0)
      and min(
        np.linalg.norm(actual[3:] - expected[3:]),
        np.linalg.norm(actual[3:] + expected[3:]),
      )
      < 1e-10
      and support["valid"] is True
      and support["minimum_corner_margin_xy_m"] >= MIN_TABLE_MARGIN_M
      and 0 <= support["bottom_gap_m"] <= 0.001
    )
  except (KeyError, ValueError, TypeError, OverflowError):
    return False


def _table_support(simulation):
  """Check all core-box corners in the actual tabletop frame before stepping."""
  model, data = simulation.model, simulation.data
  card = model.geom("card_core_geom").id
  table = model.geom("poker_table_top").id
  corners = np.array(list(product((-1.0, 1.0), repeat=3))) * model.geom_size[card]
  card_rotation = data.geom_xmat[card].reshape(3, 3)
  table_rotation = data.geom_xmat[table].reshape(3, 3)
  world = corners @ card_rotation.T + data.geom_xpos[card]
  local = (world - data.geom_xpos[table]) @ table_rotation
  margin = float(np.min(model.geom_size[table, :2] - np.abs(local[:, :2])))
  gap = float(local[:, 2].min() - model.geom_size[table, 2])
  valid = margin >= MIN_TABLE_MARGIN_M and -1e-12 <= gap <= 0.001
  if not valid:
    raise ValueError(
      f"randomized card is not fully supported at reset: XY margin={margin:.6g} m, bottom gap={gap:.6g} m"
    )
  return {
    "valid": True,
    "minimum_corner_margin_xy_m": margin,
    "bottom_gap_m": max(0.0, gap),
  }


def reset_randomized_card(
  simulation, seed: int, xy_jitter_m: float, yaw_jitter_rad: float, *, fixed_offset=None
):
  """Sample once, reset once, validate once; never silently resample failures.

  fixed_offset=(dx,dy,yaw) is for named boundary checks only, not random dataset
  collection. It uses the reset-only pose setter before the first physics step.
  """
  xy, yaw = sample_card_offsets(seed, xy_jitter_m, yaw_jitter_rad)
  if simulation.scene != "poker-draw":
    raise ValueError("card randomization requires the isolated poker-draw scene")
  nominal = np.asarray(simulation._initial_object_pose["card"], dtype=float).copy()
  if fixed_offset is None:
    simulation.reset(
      seed=seed, object_xy_jitter=xy_jitter_m, object_yaw_jitter=yaw_jitter_rad
    )
    mode, distribution = "seeded_uniform", "independent_uniform"
  else:
    fixed = np.asarray(fixed_offset, dtype=float)
    if (
      fixed.shape != (3,)
      or not np.all(np.isfinite(fixed))
      or np.any(np.abs(fixed[:2]) > xy_jitter_m + 1e-12)
      or abs(fixed[2]) > yaw_jitter_rad + 1e-12
    ):
      raise ValueError("fixed validation offset lies outside the declared bounds")
    xy, yaw = fixed[:2], float(fixed[2])
    simulation.reset(seed=seed, object_xy_jitter=0.0, object_yaw_jitter=0.0)
    pose = offset_card_pose(nominal, xy, yaw)
    simulation.set_object_pose("card", pose[:3], pose[3:])
    mode, distribution = "fixed_validation_offset", "none"
  # A reused simulation previously observed its last integration step. The
  # reset forward pass has now established a new, exact t=0 cache.
  simulation._observation_time = float(simulation.data.time)
  actual = simulation.object_pose("card")
  expected = offset_card_pose(nominal, xy, yaw)
  if (
    not np.allclose(actual[:3], expected[:3], atol=1e-12, rtol=0)
    or min(
      np.linalg.norm(actual[3:] - expected[3:]),
      np.linalg.norm(actual[3:] + expected[3:]),
    )
    > 1e-10
  ):
    raise RuntimeError("shared reset no longer matches the documented card sampler")
  return {
    "schema_version": RANDOMIZATION_SCHEMA,
    "mode": mode,
    "distribution": distribution,
    "seed": int(seed),
    "xy_jitter_m": float(xy_jitter_m),
    "yaw_jitter_rad": float(yaw_jitter_rad),
    "nominal_pose_wxyz": nominal.tolist(),
    "sampled_pose_wxyz": actual.tolist(),
    "sampled_offset_xy_m": xy.tolist(),
    "sampled_yaw_offset_rad": yaw,
    "perturbation_scope": "reset_only_card_xy_yaw",
    "height_unchanged": True,
    "tilt_unchanged": True,
    "table_support": _table_support(simulation),
    "observation_noise": None,
    "action_noise": None,
    "acceptance_scope": "geometric reset check only; full-task success requires execution and original gates",
  }
