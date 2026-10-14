"""Coordinate adapter between the KaiHand workcell and EgoSteer.

The converted training data has two representations of the same robot pose:

* the raw 48-D representation stores both canonical wrist poses and all ten
  fingertips in the MuJoCo world frame; and
* the model 48-D representation stores wrists in the *current* OpenCV camera
  frame and fingertips in their corresponding canonical wrist frames.

This module mirrors the contracts in ``scripts/workcell/convert_to_egosteer.py``
and EgoSteer's ``process_state_action``/``get_absolute_action`` without taking a
runtime dependency on EgoSteer, PyTorch, or its geometry helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
STATE_DIM = 48
WRIST_DIM = 18

# A MuJoCo camera looks along -Z with +Y up.  OpenCV uses +Z forward and +Y
# down.  This proper half-turn about X maps MuJoCo camera coordinates to the
# OpenCV RDF convention used in the converted WebDataset.
CV_FROM_MUJOCO_CAMERA = np.diag((1.0, -1.0, -1.0, 1.0))

# Columns are EgoSteer/MANO canonical wrist axes expressed in the physical
# KaiHand base-link site frame.  These values must remain identical to the
# converter constants: the two hands are mirrored.
SITE_FROM_EGOSTEER_WRIST = {
  "left": np.array(
    (
      (0.0, 0.0, 1.0),
      (0.0, 1.0, 0.0),
      (-1.0, 0.0, 0.0),
    ),
    dtype=np.float64,
  ),
  "right": np.array(
    (
      (0.0, 0.0, 1.0),
      (0.0, -1.0, 0.0),
      (1.0, 0.0, 0.0),
    ),
    dtype=np.float64,
  ),
}

# EgoSteer's homogeneous-to-Cartesian helper divides by w + 1e-6.  Keeping
# this tiny factor makes live inputs and decoded fingertip targets agree with
# the exact preprocessing/visualization path used by the trained model.
EGOSTEER_HOMOGENEOUS_EPS = 1.0e-6
_ROTATION_EPS = 1.0e-12

_TRANSLATION_SLICES = {"left": slice(0, 3), "right": slice(3, 6)}
_ROT6D_SLICES = {"left": slice(6, 12), "right": slice(12, 18)}
_FINGERTIP_SLICES = {"left": slice(18, 33), "right": slice(33, 48)}


@dataclass(frozen=True)
class SideActionTargets:
  """One side of a decoded action chunk, ready for workcell controllers."""

  site_positions_world: np.ndarray
  site_rotations_world: np.ndarray
  site_quaternions_wxyz: np.ndarray
  fingertips_world: np.ndarray
  # EgoSteer's legacy 48-D action contains fingertip XYZ only, while
  # EgoTouch's 108-D task-space action contains complete fingertip SE(3).
  # Keeping this optional preserves the position-only EgoSteer contract and
  # lets the shared executor opt into pose-aware hand IK when rotations exist.
  fingertip_rotations_world: np.ndarray | None = None

  @property
  def fingertip_poses_world(self) -> np.ndarray | None:
    """Return homogeneous fingertip targets when orientation is available."""
    if self.fingertip_rotations_world is None:
      return None
    positions = np.asarray(self.fingertips_world)
    rotations = np.asarray(self.fingertip_rotations_world)
    if rotations.shape != positions.shape[:-1] + (3, 3):
      raise ValueError(
        "fingertip rotations/positions have incompatible shapes: "
        f"{rotations.shape} and {positions.shape}"
      )
    poses = np.zeros(positions.shape[:-1] + (4, 4), dtype=np.result_type(
      positions, rotations
    ))
    poses[..., :3, :3] = rotations
    poses[..., :3, 3] = positions
    poses[..., 3, 3] = 1.0
    return poses


@dataclass(frozen=True)
class ActionTargets:
  """Decoded per-side targets for an entire ``H``-step action chunk."""

  left: SideActionTargets
  right: SideActionTargets

  def for_side(self, side: str) -> SideActionTargets:
    """Return targets for ``left`` or ``right`` with a useful error."""
    if side not in SIDES:
      raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    return self.left if side == "left" else self.right


def _finite_array(value: Any, *, name: str) -> np.ndarray:
  array = np.asarray(value)
  if not np.issubdtype(array.dtype, np.number):
    raise ValueError(f"{name} must be numeric")
  if not np.all(np.isfinite(array)):
    raise ValueError(f"{name} contains non-finite values")
  return array


def _validate_rigid_transform(value: Any, *, name: str) -> np.ndarray:
  transform = _finite_array(value, name=name).astype(np.float64, copy=False)
  if transform.shape != (4, 4):
    raise ValueError(f"{name} must have shape (4, 4), got {transform.shape}")
  if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-6):
    raise ValueError(f"{name} has an invalid homogeneous final row")
  rotation = transform[:3, :3]
  if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5):
    raise ValueError(f"{name} rotation is not orthonormal")
  if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-5):
    raise ValueError(f"{name} rotation is not proper")
  return transform


def _rot6d_from_rotation(rotation: np.ndarray) -> np.ndarray:
  """Return first rotation column followed by the second, as EgoSteer does."""
  return np.concatenate((rotation[..., :, 0], rotation[..., :, 1]), axis=-1)


def _rotation_from_rot6d(rot6d: np.ndarray) -> np.ndarray:
  """Reproduce EgoSteer's Gram-Schmidt 6-D rotation decoding in NumPy."""
  value = np.asarray(rot6d, dtype=np.float64)
  if value.shape[-1:] != (6,):
    raise ValueError(f"rot6d must end in dimension 6, got {value.shape}")
  first = value[..., :3]
  first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
  if np.any(first_norm <= _ROTATION_EPS):
    raise ValueError("rot6d has a degenerate first column")
  first = first / np.maximum(first_norm, _ROTATION_EPS)
  second = value[..., 3:]
  second = second - np.sum(first * second, axis=-1, keepdims=True) * first
  second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
  if np.any(second_norm <= _ROTATION_EPS):
    raise ValueError("rot6d has collinear first and second columns")
  second = second / np.maximum(second_norm, _ROTATION_EPS)
  third = np.cross(first, second, axis=-1)
  return np.stack((first, second, third), axis=-1)


def _pose_from_trans_rot6d(translation: np.ndarray, rot6d: np.ndarray) -> np.ndarray:
  translation = np.asarray(translation, dtype=np.float64)
  rotation = _rotation_from_rot6d(rot6d)
  if translation.shape[:-1] != rotation.shape[:-2] or translation.shape[-1:] != (3,):
    raise ValueError("translation and rot6d leading dimensions do not match")
  pose = np.zeros(translation.shape[:-1] + (4, 4), dtype=np.float64)
  pose[..., :3, :3] = rotation
  pose[..., :3, 3] = translation
  pose[..., 3, 3] = 1.0
  return pose


def _store_pose(state: np.ndarray, side: str, pose: np.ndarray) -> None:
  state[..., _TRANSLATION_SLICES[side]] = pose[..., :3, 3]
  state[..., _ROT6D_SLICES[side]] = _rot6d_from_rotation(pose[..., :3, :3])


def _wrist_pose(state: np.ndarray, side: str) -> np.ndarray:
  return _pose_from_trans_rot6d(
    state[..., _TRANSLATION_SLICES[side]],
    state[..., _ROT6D_SLICES[side]],
  )


def _transform_points_like_egosteer(
  points: np.ndarray, transform: np.ndarray
) -> np.ndarray:
  """Apply one aligned transform per leading index using EgoSteer's epsilon."""
  points = np.asarray(points, dtype=np.float64)
  transform = np.asarray(transform, dtype=np.float64)
  if points.shape[-1:] != (3,) or transform.shape[-2:] != (4, 4):
    raise ValueError("points/transform must end in (3,) and (4, 4)")
  if points.shape[:-2] != transform.shape[:-2]:
    raise ValueError("points and transform leading dimensions do not match")
  transformed = np.einsum("...ij,...pj->...pi", transform[..., :3, :3], points)
  transformed += transform[..., None, :3, 3]
  return transformed / (1.0 + EGOSTEER_HOMOGENEOUS_EPS)


def _site_id(simulation: Any, name: str) -> int:
  try:
    return int(simulation.model.site(name).id)
  except (KeyError, ValueError) as error:
    raise ValueError(f"MuJoCo model is missing required site {name!r}") from error


def _named_site_pose(simulation: Any, name: str) -> np.ndarray:
  site_id = _site_id(simulation, name)
  pose = np.eye(4, dtype=np.float64)
  pose[:3, :3] = np.asarray(simulation.data.site_xmat[site_id]).reshape(3, 3)
  pose[:3, 3] = np.asarray(simulation.data.site_xpos[site_id])
  return pose


def _base_site_name(side: str) -> str:
  return f"hand_{side[0]}_base_link_site"


def _fingertip_site_name(side: str, finger: str) -> str:
  link = "link6" if finger == "thumb" else "link4"
  return f"hand_{side[0]}_{finger}_{link}_site"


def camera_from_world_opencv(world_from_camera: Any) -> np.ndarray:
  """Convert a MuJoCo camera pose to OpenCV ``world -> camera`` extrinsic.

  ``world_from_camera`` is the matrix exposed by :class:`WorkcellRenderer` and
  uses MuJoCo camera axes.  The returned row-major 4x4 matrix is exactly the
  ``extrinsic`` stored by ``convert_to_egosteer.py``.
  """
  world_from_camera = _validate_rigid_transform(
    world_from_camera, name="world_from_camera"
  )
  return CV_FROM_MUJOCO_CAMERA @ np.linalg.inv(world_from_camera)


def raw_unified_from_simulation(simulation: Any) -> np.ndarray:
  """Read the current MuJoCo FK result into raw unified world-frame 48-D.

  The caller must ensure MuJoCo forward kinematics is current.  The returned
  layout is ``[left_xyz, right_xyz, left_rot6d, right_rot6d,
  left_fingertips(5x3), right_fingertips(5x3)]`` with fingertip order
  thumb, index, middle, ring, pinky.
  """
  state = np.empty(STATE_DIM, dtype=np.float64)
  for side in SIDES:
    world_from_site = _named_site_pose(simulation, _base_site_name(side))
    world_from_wrist = world_from_site.copy()
    world_from_wrist[:3, :3] = world_from_site[:3, :3] @ SITE_FROM_EGOSTEER_WRIST[side]
    _store_pose(state, side, world_from_wrist)
    fingertips = np.stack(
      [
        _named_site_pose(simulation, _fingertip_site_name(side, finger))[:3, 3]
        for finger in FINGERS
      ]
    )
    state[_FINGERTIP_SLICES[side]] = fingertips.reshape(-1)
  return state.astype(np.float32)


def model_state_history(raw_states: Any, current_camera_from_world: Any) -> np.ndarray:
  """Convert raw unified history to the exact 48-D EgoSteer model frame.

  Args:
    raw_states: Array ``[T, 48]`` produced by
      :func:`raw_unified_from_simulation` (or the converter's raw lowdim
      state).  Wrist poses and fingertips are in the MuJoCo world frame.
    current_camera_from_world: OpenCV ``world -> current camera`` 4x4
      extrinsic.  The same current-frame matrix intentionally transforms every
      historical wrist pose, matching EgoSteer's sample preprocessing.

  Returns:
    Float32 array ``[T, 48]``.  Wrist poses are in the current camera frame;
    each hand's fingertips are in that frame's corresponding canonical wrist
    coordinate system.
  """
  raw = _finite_array(raw_states, name="raw_states").astype(np.float64, copy=False)
  if raw.ndim != 2 or raw.shape[1] != STATE_DIM or raw.shape[0] == 0:
    raise ValueError(f"raw_states must have non-empty shape (T, 48), got {raw.shape}")
  camera_from_world = _validate_rigid_transform(
    current_camera_from_world, name="current_camera_from_world"
  )

  processed = np.empty_like(raw)
  for side in SIDES:
    world_from_wrist = _wrist_pose(raw, side)
    camera_from_wrist = camera_from_world @ world_from_wrist
    _store_pose(processed, side, camera_from_wrist)

    fingertips_world = raw[:, _FINGERTIP_SLICES[side]].reshape(-1, 5, 3)
    wrist_from_world = np.linalg.inv(world_from_wrist)
    fingertips_wrist = _transform_points_like_egosteer(
      fingertips_world, wrist_from_world
    )
    processed[:, _FINGERTIP_SLICES[side]] = fingertips_wrist.reshape(-1, 15)
  return processed.astype(np.float32)


def relative_to_absolute(current_model_state: Any, relative_actions: Any) -> np.ndarray:
  """Invert EgoSteer's relative-action encoding for one ``[H, 48]`` chunk."""
  state = _finite_array(current_model_state, name="current_model_state").astype(
    np.float64, copy=False
  )
  relative = _finite_array(relative_actions, name="relative_actions").astype(
    np.float64, copy=False
  )
  if state.shape != (STATE_DIM,):
    raise ValueError(f"current_model_state must have shape (48,), got {state.shape}")
  if relative.ndim != 2 or relative.shape[1] != STATE_DIM or relative.shape[0] == 0:
    raise ValueError(
      f"relative_actions must have non-empty shape (H, 48), got {relative.shape}"
    )

  absolute = relative.copy()
  for side in SIDES:
    state_from_relative = _wrist_pose(relative, side)
    camera_from_state = _wrist_pose(state, side)
    camera_from_action = camera_from_state @ state_from_relative
    _store_pose(absolute, side, camera_from_action)
  absolute[..., WRIST_DIM:] = relative[..., WRIST_DIM:] + state[WRIST_DIM:]
  return absolute.astype(np.float32)


def _quaternion_wxyz_from_rotation(rotation: np.ndarray) -> np.ndarray:
  """Convert a batch of proper rotations to normalized, canonical wxyz."""
  rotations = np.asarray(rotation, dtype=np.float64)
  if rotations.shape[-2:] != (3, 3):
    raise ValueError(f"rotation must end in shape (3, 3), got {rotations.shape}")
  flat = rotations.reshape(-1, 3, 3)
  quaternions = np.empty((flat.shape[0], 4), dtype=np.float64)
  for index, matrix in enumerate(flat):
    trace = float(np.trace(matrix))
    if trace > 0.0:
      scale = 2.0 * np.sqrt(trace + 1.0)
      quat = np.array(
        (
          0.25 * scale,
          (matrix[2, 1] - matrix[1, 2]) / scale,
          (matrix[0, 2] - matrix[2, 0]) / scale,
          (matrix[1, 0] - matrix[0, 1]) / scale,
        )
      )
    else:
      axis = int(np.argmax(np.diag(matrix)))
      if axis == 0:
        scale = 2.0 * np.sqrt(
          max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        )
        quat = np.array(
          (
            (matrix[2, 1] - matrix[1, 2]) / scale,
            0.25 * scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
          )
        )
      elif axis == 1:
        scale = 2.0 * np.sqrt(
          max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        )
        quat = np.array(
          (
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            0.25 * scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
          )
        )
      else:
        scale = 2.0 * np.sqrt(
          max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        )
        quat = np.array(
          (
            (matrix[1, 0] - matrix[0, 1]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            0.25 * scale,
          )
        )
    norm = float(np.linalg.norm(quat))
    if norm <= _ROTATION_EPS:
      raise ValueError("rotation produced a degenerate quaternion")
    quat /= norm
    if quat[0] < 0.0:
      quat *= -1.0
    quaternions[index] = quat
  return quaternions.reshape(rotations.shape[:-2] + (4,))


def _current_control_site_from_wrist(simulation: Any, side: str) -> np.ndarray:
  """Get fixed ``ArmHandSimulation control-site -> canonical wrist`` pose."""
  control_position, control_rotation = simulation.current_pose_matrix(side)
  world_from_control = np.eye(4, dtype=np.float64)
  world_from_control[:3, :3] = np.asarray(control_rotation, dtype=np.float64)
  world_from_control[:3, 3] = np.asarray(control_position, dtype=np.float64)

  world_from_base = _named_site_pose(simulation, _base_site_name(side))
  world_from_wrist = world_from_base.copy()
  world_from_wrist[:3, :3] = world_from_base[:3, :3] @ SITE_FROM_EGOSTEER_WRIST[side]
  return np.linalg.inv(world_from_control) @ world_from_wrist


def decode_action_targets(
  simulation: Any,
  absolute_model_actions: Any,
  current_camera_from_world: Any,
) -> ActionTargets:
  """Decode absolute model-frame actions into workcell world-frame targets.

  ``absolute_model_actions`` must first be produced by
  :func:`relative_to_absolute` when the policy was trained with relative
  actions.  Wrist poses are mapped from current OpenCV camera coordinates to
  world coordinates, then from the canonical EgoSteer wrist to the actual
  ``left_ee_site``/``right_ee_site`` controlled by
  :meth:`ArmHandSimulation.set_pose_target`.  Fingertips are mapped from their
  predicted local canonical wrist frames directly to world coordinates.
  """
  actions = _finite_array(absolute_model_actions, name="absolute_model_actions").astype(
    np.float64, copy=False
  )
  if actions.ndim != 2 or actions.shape[1] != STATE_DIM or actions.shape[0] == 0:
    raise ValueError(
      f"absolute_model_actions must have non-empty shape (H, 48), got {actions.shape}"
    )
  camera_from_world = _validate_rigid_transform(
    current_camera_from_world, name="current_camera_from_world"
  )
  world_from_camera = np.linalg.inv(camera_from_world)

  decoded: dict[str, SideActionTargets] = {}
  for side in SIDES:
    camera_from_wrist = _wrist_pose(actions, side)
    world_from_wrist = world_from_camera @ camera_from_wrist

    control_from_wrist = _current_control_site_from_wrist(simulation, side)
    world_from_control = world_from_wrist @ np.linalg.inv(control_from_wrist)
    site_positions = world_from_control[..., :3, 3].copy()
    site_rotations = world_from_control[..., :3, :3].copy()
    site_quaternions = _quaternion_wxyz_from_rotation(site_rotations)

    fingertips_wrist = actions[:, _FINGERTIP_SLICES[side]].reshape(-1, 5, 3)
    fingertips_world = _transform_points_like_egosteer(
      fingertips_wrist, world_from_wrist
    )
    decoded[side] = SideActionTargets(
      site_positions_world=site_positions,
      site_rotations_world=site_rotations,
      site_quaternions_wxyz=site_quaternions,
      fingertips_world=fingertips_world,
    )

  return ActionTargets(left=decoded["left"], right=decoded["right"])


__all__ = [
  "ActionTargets",
  "CV_FROM_MUJOCO_CAMERA",
  "EGOSTEER_HOMOGENEOUS_EPS",
  "FINGERS",
  "SITE_FROM_EGOSTEER_WRIST",
  "SideActionTargets",
  "camera_from_world_opencv",
  "decode_action_targets",
  "model_state_history",
  "raw_unified_from_simulation",
  "relative_to_absolute",
]
