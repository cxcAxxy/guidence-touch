"""Export recorded robot observations to the PDF's EgoTouch T-ICT release.

This module has no MuJoCo/renderer dependency: full 6-DoF site poses must have
been recorded alongside the pixels.  It deliberately does not manufacture the
undocumented ICT/finger tokens or claim compatibility with an absent trainer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "little")
CHANNELS = ("normal", "tangent_x", "tangent_y")
SOURCE_SCHEMA = "kaihand_tactile_episode_v1"
SIDECAR_SCHEMA = "egotouch-fingertip-tactile-sidecar-v1"
CONTRACT_VERSION = "kaihand-tict-release-v1"
CV_FROM_MUJOCO_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])


def _text(value: Any) -> str:
  return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _json_attribute(group: Any, key: str) -> dict[str, Any]:
  if key not in group.attrs:
    raise ValueError(f"source is missing JSON attribute {key}")
  result = json.loads(_text(group.attrs[key]))
  if not isinstance(result, dict):
    raise ValueError(f"{key} must contain a JSON object")
  return result


def sha256_file(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _write_json(path: Path, data: Any) -> None:
  path.write_text(
    json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    + "\n",
    encoding="utf-8",
  )


def require_se3(transforms: np.ndarray, name: str) -> None:
  """Reject invalid rotations instead of treating XYZ plus identity as 6-DoF."""
  values = np.asarray(transforms)
  if values.shape[-2:] != (4, 4) or not np.isfinite(values).all():
    raise ValueError(f"{name} must be finite SE(3) matrices")
  rotation = values[..., :3, :3]
  if not np.allclose(values[..., 3, :], [0, 0, 0, 1], atol=1e-7, rtol=0):
    raise ValueError(f"{name} has invalid homogeneous last rows")
  if not np.allclose(
    np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=1e-6, rtol=0
  ) or not np.allclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0):
    raise ValueError(f"{name} has non-SO(3) rotations")


def invert_se3(transforms: np.ndarray) -> np.ndarray:
  values = np.asarray(transforms, dtype=np.float64)
  result = np.zeros_like(values)
  result[..., :3, :3] = np.swapaxes(values[..., :3, :3], -1, -2)
  result[..., :3, 3] = -np.einsum(
    "...ij,...j->...i", result[..., :3, :3], values[..., :3, 3]
  )
  result[..., 3, 3] = 1
  return result


def pose9(transforms: np.ndarray) -> np.ndarray:
  """XYZ followed by complete first and second rotation columns (not rows)."""
  values = np.asarray(transforms)
  return np.concatenate(
    [values[..., :3, 3], values[..., :3, 0], values[..., :3, 1]], axis=-1
  )


def _source_object(metadata: dict[str, Any], outcome: dict[str, Any]) -> str:
  """Select the recorded task object without inferring it from stale streams."""
  objects = {"poker-draw": "card", "usb-insert": "usb_plug"}
  scene = metadata.get("scene")
  if scene not in objects:
    raise ValueError("T-ICT export supports recorded poker-draw or usb-insert scenes")
  object_name = objects[scene]
  if outcome.get("object_name", object_name) != object_name:
    raise ValueError("source outcome object_name disagrees with its scene")
  if scene == "usb-insert":
    if metadata.get("recording_contract") != "usb_insert_taskspace_raw_v1":
      raise ValueError("USB source must declare usb_insert_taskspace_raw_v1")
    if metadata.get("observation_clock") != "post_step_forward_v1":
      raise ValueError(
        "USB source must declare observation_clock=post_step_forward_v1; "
        "legacy pre-step labels are not accepted for training export"
      )
    if outcome.get("object_name") != object_name:
      raise ValueError("USB source must explicitly identify usb_plug")
    if not all(outcome.get(key) is True for key in ("released", "grasp_verified")):
      raise ValueError("USB source must have verified grasp and successful release")
    insertion = outcome.get("insertion", {})
    if not all(insertion.get(key) is True for key in ("success", "seated")):
      raise ValueError("USB source must have explicit successful stable seating")
  return object_name


def _pose_wxyz_to_se3(pose: np.ndarray, object_name: str = "card") -> np.ndarray:
  pose = np.asarray(pose, dtype=np.float64)
  if pose.shape != (7,) or not np.isfinite(pose).all():
    raise ValueError(
      f"initial {object_name} pose must contain finite xyz + quaternion wxyz"
    )
  w, x, y, z = pose[3:]
  if not np.isclose(np.linalg.norm(pose[3:]), 1, atol=1e-7, rtol=0):
    raise ValueError(f"initial {object_name} quaternion must be unit length")
  result = np.eye(4)
  result[:3, 3] = pose[:3]
  result[:3, :3] = [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ]
  require_se3(result, f"initial {object_name} static anchor")
  return result


def build_action_window(
  world_from_wrist: np.ndarray,
  fingertip_to_wrist: np.ndarray,
  world_from_camera: np.ndarray,
  start: int,
  horizon: int = 50,
) -> np.ndarray:
  """Build the specified raw 108-D target for audit, without normalizing it.

  Wrist targets use the observation-start camera, even when the camera moves.
  Fingertip targets use their own future wrist, not the observation wrist.
  """
  if start < 0 or horizon < 1 or start + horizon >= len(world_from_wrist):
    raise ValueError("action window must contain start and all future frames")
  future = slice(start + 1, start + horizon + 1)
  wrist = invert_se3(world_from_camera[start]) @ world_from_wrist[future]
  slots = np.concatenate(
    [pose9(wrist)[..., None, :], pose9(fingertip_to_wrist[future])], axis=2
  )
  return slots.reshape(horizon, 108).astype(np.float32)


def _timestamps_ns(values: np.ndarray, name: str, *, strict: bool) -> np.ndarray:
  values = np.asarray(values, dtype=np.float64)
  if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
    raise ValueError(f"{name} must be a nonempty finite timestamp vector")
  if np.any(values < 0) or np.max(values) >= np.iinfo(np.int64).max / 1e9:
    raise ValueError(f"{name} is outside nonnegative int64 nanosecond range")
  result = np.rint(values * 1e9).astype(np.int64)
  differences = np.diff(result)
  if np.any(differences <= 0 if strict else differences < 0):
    raise ValueError(
      f"{name} timestamps are not {'strictly ' if strict else ''}ordered"
    )
  return result


def _finger_link_names() -> tuple[str, ...]:
  return tuple(
    f"hand_{side[0]}_{'pinky' if finger == 'little' else finger}_"
    f"{'link6' if finger == 'thumb' else 'link4'}"
    for side in SIDES
    for finger in FINGERS
  )


def _sync_tactile(
  force: h5py.Group, timestamps: np.ndarray, max_error: int
) -> dict[str, np.ndarray]:
  force_times = _timestamps_ns(force["timestamp"][:], "tactile", strict=False)
  names = tuple(_text(item) for item in force["link_names"][:])
  if len(set(names)) != len(names):
    raise ValueError("duplicate tactile link names")
  mapping = {name: index for index, name in enumerate(names)}
  expected = _finger_link_names()
  normal = force["normal_taxel_force_n"]
  shear = force["tangent_taxel_force_n"]
  if normal.shape != (len(force_times), len(names), 7, 5):
    raise ValueError("normal taxel stream has an incompatible shape")
  if shear.shape != (len(force_times), len(names), 7, 5, 2):
    raise ValueError("signed tangent taxel stream has an incompatible shape")
  indices = np.searchsorted(force_times, timestamps, side="right") - 1
  errors = np.full(len(timestamps), -1, dtype=np.int64)
  causal = indices >= 0
  errors[causal] = timestamps[causal] - force_times[indices[causal]]
  valid = causal & (errors >= 0) & (errors <= max_error)
  means = np.zeros((len(timestamps), 10, 3), dtype=np.float32)
  masks = np.zeros_like(means, dtype=np.bool_)
  for frame in np.flatnonzero(valid):
    source_index = int(indices[frame])
    normal_frame = np.asarray(normal[source_index], dtype=np.float64)
    shear_frame = np.asarray(shear[source_index], dtype=np.float64)
    if not np.isfinite(normal_frame).all() or not np.isfinite(shear_frame).all():
      raise ValueError(f"nonfinite source taxel forces at {source_index}")
    if np.any(normal_frame < -1e-12):
      raise ValueError(f"negative normal taxel force at {source_index}")
    for target, link_name in enumerate(expected):
      if link_name not in mapping:
        continue
      link = mapping[link_name]
      means[frame, target, 0] = normal_frame[link].mean()
      means[frame, target, 1:] = shear_frame[link].mean(axis=(0, 1))
      masks[frame, target] = True
  # No suitable source has no source index. Stale sources retain their actual
  # index/error for diagnosis but carry false masks and zero values.
  return {
    "tactile_mean": means.reshape(-1, 2, 5, 3),
    "tactile_channel_mask": masks.reshape(-1, 2, 5, 3),
    "tactile_frame_valid": valid & np.any(masks, axis=(1, 2)),
    "tactile_source_index": indices.astype(np.int64),
    "tactile_sync_error_ns": errors,
  }


def _statistics(values: np.ndarray) -> dict[str, Any]:
  values = np.asarray(values, dtype=np.float64)
  if values.size == 0:
    return {"count": 0, "mean": None, "std": None}
  return {
    "count": len(values),
    "mean": values.mean(axis=0).tolist(),
    "std": values.std(axis=0).tolist(),
  }


def _training_stats(
  wrist: np.ndarray,
  finger_relative: np.ndarray,
  cameras: np.ndarray,
  tactile: dict[str, np.ndarray],
  horizon: int,
  session_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
  wrist_positions, finger_positions = [], []
  action_min, action_max = None, None
  for start in range(len(wrist) - horizon):
    action = build_action_window(wrist, finger_relative, cameras, start, horizon)
    if not np.isfinite(action).all():
      raise ValueError("nonfinite 108-D target")
    slots = action.reshape(horizon, 2, 6, 9)
    wrist_positions.append(slots[:, :, 0, :3].reshape(-1, 3))
    finger_positions.append(slots[:, :, 1:, :3].reshape(-1, 3))
    minimum, maximum = action.min(axis=0), action.max(axis=0)
    action_min = minimum if action_min is None else np.minimum(action_min, minimum)
    action_max = maximum if action_max is None else np.maximum(action_max, maximum)
  values = tactile["tactile_mean"]
  masks = tactile["tactile_channel_mask"]
  stats = {
    "schema_version": "kaihand-tict-train-statistics-v1",
    "sessions": [session_id],
    "split": "train",
    "position_sampling": "all published future action slots; window overlap retained",
    "wrist_xyz": _statistics(np.concatenate(wrist_positions)),
    "shared_ten_finger_xyz": _statistics(np.concatenate(finger_positions)),
    "tactile_sampling": "each recorded frame once, only true channel masks",
    "tactile_channels": {
      channel: _statistics(values[..., index][masks[..., index]])
      for index, channel in enumerate(CHANNELS)
    },
    "rotation_6d_normalization": "none",
    "rgb_normalization": "divide uint8 by 255 only; no ImageNet normalization",
    "raw_release_is_normalized": False,
    "zero_std_policy": "reported literally; downstream must choose and freeze epsilon",
    "upstream_stats_implementation_verified": False,
  }
  audit = {
    "shape_per_window": [horizon, 108],
    "valid_action_dimensions": 108,
    "rotation_6d": "first column then second column",
    "nonzero_action": bool(
      np.any(np.maximum(np.abs(action_min), np.abs(action_max)) > 0)
    ),
    "varying_action_dimensions": int(np.count_nonzero(action_max - action_min > 1e-7)),
    "translation_unit": "metre",
  }
  return stats, audit


def _data_contract(
  camera: str, horizon: int, session_id: str, object_name: str = "card"
) -> str:
  observation_clock = (
    "USB observation_clock is post_step_forward_v1: after each integration step,\n"
    "the controller refreshes forward dynamics before observation. State, solver\n"
    "force and indexed RGB pose/acquisition timestamps are the same post-step time."
    if object_name == "usb_plug"
    else "Camera pose_timestamp is the actual cached FK/render epoch, not the post-step\n"
    "qpos timestamp."
  )
  return f"""# KaiHand T-ICT release v1

This is a local, versioned interpretation of test.pdf Parts 1–7, not proof of
execution by the remote EgoTouch DataLoader. Foreign paths and Part 11 are ignored.
The PDF supplies exact JSON/NPZ fields, but not the remote manifest schemas,
ICT entity-slot packing, x_finger 18-D packing, or training launcher implementation.
Those tokens are not invented or precomputed. Upstream integration remains a gate.

## Frames and task labels

One complete source episode is one session, `{session_id}`, assigned to train only.
Validation/test are empty: this is a single-session format/overfit smoke release,
not a generalization benchmark. Frame names are consecutive five-digit numbers.
RGB is `{camera}` camera, uint8 320×240, lossless PNG. All actual captured frames
are retained, including an off-grid terminal frame if present. Use recorded
timestamps, not a silently assumed 30-Hz clock; horizon {horizon} means frames.
Timestamp origin is simulation episode time, not a claimed wall-clock epoch.
{observation_clock} is_finished is false before the final recorded frame and true on
that frame, using the successful terminal episode label. It is not an inferred
per-stage grasp label. Wrist grasp probabilities are not measured and are omitted.

## Geometry and action

Every transform is finite proper SE(3), translation in metres. c2w maps OpenCV
camera coordinates (right/down/forward) to world. Original MuJoCo right/up/back
c2w is right-multiplied by diag(1,-1,-1,1). metadata.world_transforms.cam0 equals
that frame's c2w, consistently in every frame/session of this contract. For a
sample starting at t, freeze **cam0 at t** for all future wrist transforms:
`inv(c2w[t]) @ world_from_wrist[t+k]`, k=1..{horizon}. Do not rebase each future
wrist in its moving future camera. No wrist canonicalization is silently applied:
wrist origin/orientation is the actual hand_l/r_base_link_site robot frame.

Fingertips use the complete distal link site transforms (thumb link6, other four
link4). The source name pinky maps to release little. Their relative transform is
`inv(world_from_wrist[t+k]) @ world_from_fingertip[t+k]`: the corresponding **future**
wrist, not the wrist at t. Rotations are actual FK, not identity placeholders.
These are known robot site frames, not MANUS/MANO/HaWoR anatomical calibration.
Embedding them under hands_hawor_v3 is a schema adapter, not a claim of HaWoR data.
pose9 is xyz followed by R[:,0] and R[:,1], i.e. column-concatenated rotation-6D.
The 108-D order is left wrist + left thumb/index/middle/ring/little, then right
wrist + right thumb/index/middle/ring/little. Every action is an absolute future
pose in its specified reference, not a pose delta. Objects are left empty; the
explicit virtual_static_anchor is the known initial {object_name} pose, held fixed for
the entire session, not a proxy dynamic {object_name} Object6D trajectory. Disable
object/region attention for the initial hand-only smoke.

## Tactile

Channels are normal, **signed** tangent_x, **signed** tangent_y; no tangent
magnitude is substituted. Each value is the arithmetic mean of all 35 taxels,
including zero taxels, in newtons (sum/35, not per-finger total and not pressure).
Source is solver contact force on the tactile pad from {object_name} only, spatially
allocated over a 7×5 grid: simulated forces, not calibrated hardware skin truth.
x follows increasing taxel column and y increasing row. These stable pad chart
axes are stored in source_metadata_json and are **not** assumed to be the same
as the distal-site XYZ axes or to form a proper rotation with the normal.
Use latest source force timestamp <= rendered image timestamp, never a future
sample; reject temporal validity beyond max_sync_error_ns. Measured no-contact
zeros are valid observations, including the inactive left hand. Missing/stale
channels are zeros with mask=false. Absent causal source uses index/error -1;
a stale causal source retains its nonnegative index/error but false validity.
The current tactile observation is not a future action supervision target.

## Windows, manifests, statistics, handoff

Every published start has all {horizon + 1} JSON/RGB/sidecar records t..t+{horizon}
within this one episode, known future done, valid current wrist/finger and future
actions. The final {horizon} frames are excluded as starts. Split is by session.
The selector contains final local absolute PNG paths, byte counts, SHA-256 and
relative JSON paths. Moving the release requires regenerating selector paths.
split_manifest, selector_manifest, window_starts, dataset_audit and train_statistics
use explicitly named **kaihand local v1 schemas**; the absent upstream repo may
need an adapter for their exact container format. No unsupported remote paths
are retained as executable configuration.

train_statistics.json uses train sessions only: wrist xyz in each sample's frozen
camera over all future slots, ten-finger shared xyz in future wrist frames, and
tactile channel statistics over active masks (each frame once). It contains raw
population statistics, including genuine zero std; no token is pre-normalized.
Rotation6D is not z-scored. RGB is /255 only. These statistics are auditable local
definitions, not a claim to have reproduced unknown upstream stats internals.

For upstream integration set img_name=rgb.png and supply these production,
sidecar, selector, split and window files through a verified T-ICT launcher.
Keep random resized crop and region attention disabled until anchor transforms
are verified. No remote trainer, run manifest, gradients, or overfit is claimed.
"""


def export_tict_episode(
  input_path: str | Path,
  output_dir: str | Path,
  *,
  session_id: str,
  camera: str = "head",
  horizon: int = 50,
  max_sync_error_ns: int = 20_000_000,
  source_sha256: str | None = None,
  source_sha256_verification: str | None = None,
  source_size_bytes: int | None = None,
  source_mtime_ns: int | None = None,
) -> dict[str, Any]:
  """Publish one successful episode without modifying source or existing output.

  Incomplete legacy files lacking actual full-pose/cached-clock streams are
  rejected. A failed build retains its unique sibling staging directory for
  diagnosis and never publishes a completed output directory.
  """
  source = Path(input_path).expanduser().resolve()
  destination = Path(output_dir).expanduser().resolve()
  if not source.is_file() or source.suffix != ".h5":
    raise ValueError("--input must name a finalized HDF5 .h5 episode")
  if destination.exists():
    raise FileExistsError(f"output already exists: {destination}")
  if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", session_id):
    raise ValueError("session_id must be a safe single path component")
  if not re.fullmatch(r"[A-Za-z0-9_-]+", camera):
    raise ValueError("camera must be a safe HDF5 group name")
  if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon != 50:
    raise ValueError("the PDF release contract requires horizon=50")
  if (
    isinstance(max_sync_error_ns, bool)
    or not isinstance(max_sync_error_ns, int)
    or not 0 <= max_sync_error_ns <= 20_000_000
  ):
    raise ValueError("max_sync_error_ns must be an integer in [0,20000000]")
  source_stat = source.stat()
  if (
    source_size_bytes is not None
    and source_stat.st_size != source_size_bytes
  ) or (
    source_mtime_ns is not None
    and source_stat.st_mtime_ns != source_mtime_ns
  ):
    raise RuntimeError("source changed after the batch identity snapshot")
  if source_sha256 is None:
    source_hash = sha256_file(source)
    source_hash_verification = "recomputed"
  else:
    source_hash = str(source_sha256).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", source_hash):
      raise ValueError("source_sha256 must be a lowercase hexadecimal SHA-256")
    source_hash_verification = (
      source_sha256_verification or "trusted_capture_sidecar"
    )
    if source_hash_verification not in {
      "trusted_capture_sidecar",
      "recomputed_by_batch_runner",
    }:
      raise ValueError("unsupported source_sha256_verification value")
  destination.parent.mkdir(parents=True, exist_ok=True)
  staging = Path(
    tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
  )
  try:
    with h5py.File(source, "r") as file:
      if _text(file.attrs.get("schema_version", "")) != SOURCE_SCHEMA:
        raise ValueError("unsupported source recording schema")
      outcome = _json_attribute(file, "outcome_json")
      if outcome.get("success") is not True:
        raise ValueError("source episode must have an explicit successful outcome")
      metadata = _json_attribute(file, "metadata_json")
      object_name = _source_object(metadata, outcome)
      object_pose_path = f"objects/{object_name}/pose_wxyz"
      required = [
        f"cameras/{camera}/{key}"
        for key in (
          "rgb",
          "pose_timestamp",
          "world_from_camera",
          "world_from_wrist",
          "world_from_fingertip",
          "intrinsic",
          "timestamp",
          "state_index",
        )
      ] + [
        "tactile_contact_force/timestamp",
        "state/timestamp",
        "commands/phase",
        object_pose_path,
      ]
      if object_name == "usb_plug":
        required.append("physics/solver_timestamp")
      missing = [key for key in required if key not in file]
      if missing:
        raise ValueError(f"source lacks recorded full-pose/clock streams: {missing}")
      frames = file[f"cameras/{camera}"]
      if (
        _text(frames.attrs.get("taskspace_schema", "")) != "kaihand-native-site-se3-v1"
      ):
        raise ValueError(
          "source camera must declare kaihand-native-site-se3-v1 taskspace"
        )
      for attribute, expected in (
        ("side_names_json", list(SIDES)),
        ("finger_names_json", list(FINGERS)),
      ):
        if json.loads(_text(frames.attrs.get(attribute, "null"))) != expected:
          raise ValueError(f"source taskspace ordering mismatch: {attribute}")
      timestamps = _timestamps_ns(
        frames["pose_timestamp"][:], "render pose", strict=True
      )
      count = len(timestamps)
      if count <= horizon or count > 100_000:
        raise ValueError(
          "episode needs > horizon frames and at most 100000 five-digit frames"
        )
      if frames["rgb"].shape != (count, 240, 320, 3) or frames["rgb"].dtype != np.uint8:
        raise ValueError("T-ICT RGB must have shape [N,240,320,3] and dtype uint8")
      state_times = _timestamps_ns(file["state/timestamp"][:], "state", strict=True)
      acquisition_times = _timestamps_ns(
        frames["timestamp"][:], "camera acquisition", strict=True
      )
      camera_state_indices = np.asarray(frames["state_index"][:])
      if (
        len(acquisition_times) != count
        or camera_state_indices.shape != (count,)
        or camera_state_indices.dtype.kind not in "iu"
        or np.any(camera_state_indices < 0)
        or np.any(camera_state_indices >= len(state_times))
        or camera_state_indices[-1] != len(state_times) - 1
        or acquisition_times[-1] != state_times[-1]
        or len(file["commands/phase"]) != len(state_times)
        or _text(file["commands/phase"][-1]) != "terminal_settle"
      ):
        raise ValueError(
          "camera stream lacks an exact recorded terminal_settle state; done label is unknown"
        )
      if np.any(acquisition_times < timestamps):
        raise ValueError("acquisition times must not precede rendered-pose times")
      object_poses = file[object_pose_path]
      if object_poses.shape != (len(state_times), 7):
        raise ValueError("object pose stream must align with the recorded state clock")
      if object_name == "usb_plug" and state_times[0] != 0:
        raise ValueError(
          "USB static anchor requires the exact initialized time-zero pose"
        )
      if object_name == "usb_plug":
        force_times = _timestamps_ns(
          file["tactile_contact_force/timestamp"][:], "USB tactile", strict=False
        )
        physics_times = _timestamps_ns(
          file["physics/solver_timestamp"][:], "USB physics solver", strict=False
        )
        if not np.array_equal(force_times, state_times):
          raise ValueError(
            "USB tactile solver clock must equal post-step state timestamps"
          )
        if not np.array_equal(physics_times, state_times):
          raise ValueError(
            "USB physics solver clock must equal post-step state timestamps"
          )
        if not np.array_equal(timestamps, acquisition_times) or not np.array_equal(
          timestamps, state_times[camera_state_indices]
        ):
          raise ValueError(
            "USB camera pose/acquisition clocks must equal indexed post-step state timestamps"
          )
      initial_object_anchor = _pose_wxyz_to_se3(object_poses[0], object_name)
      wrist = np.asarray(frames["world_from_wrist"][:], dtype=np.float64)
      fingertips = np.asarray(frames["world_from_fingertip"][:], dtype=np.float64)
      cameras = np.asarray(frames["world_from_camera"][:], dtype=np.float64)
      if (
        wrist.shape != (count, 2, 4, 4)
        or fingertips.shape != (count, 2, 5, 4, 4)
        or cameras.shape != (count, 4, 4)
      ):
        raise ValueError("recorded wrist/fingertip/camera pose shapes disagree")
      for name, poses in (
        ("wrist", wrist),
        ("fingertips", fingertips),
        ("camera", cameras),
      ):
        require_se3(poses, name)
      cameras = cameras @ CV_FROM_MUJOCO_CAMERA
      relative = invert_se3(wrist)[:, :, None] @ fingertips
      require_se3(relative, "fingertip-to-wrist")
      intrinsic = np.asarray(frames["intrinsic"][:], dtype=np.float64)
      if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or intrinsic[0, 0] <= 0
        or intrinsic[1, 1] <= 0
      ):
        raise ValueError("camera intrinsics must be a finite calibrated 3x3 matrix")
      force = file["tactile_contact_force"]
      if _text(force.attrs.get("force_unit", "")) not in ("N", "newton"):
        raise ValueError("source solver force stream must explicitly declare newtons")
      tactile = _sync_tactile(force, timestamps, max_sync_error_ns)
      names = np.asarray([f"{index:05d}" for index in range(count)])
      relative_root = (
        Path("production") / session_id / "09_humanego_adapter/preprocess/all_data"
      )
      data_root = staging / relative_root
      data_root.mkdir(parents=True)
      selectors = []
      for index, name in enumerate(names):
        frame_dir = data_root / str(name)
        frame_dir.mkdir()
        document = {
          "metadata": {
            "idx": index,
            "timestamp_ns": int(timestamps[index]),
            "is_finished": index == count - 1,
            "anchor_key": "virtual_static_anchor",
            "c2w": cameras[index].tolist(),
            "k": intrinsic.ravel().tolist(),
            "w": 320,
            "h": 240,
            "world_transforms": {
              "cam0": cameras[index].tolist(),
              "virtual_static_anchor": initial_object_anchor.tolist(),
            },
          },
          "obs": {},
          "entities": {
            "hands_hawor_v3": {
              side: {"T_hand_to_world": wrist[index, side_index].tolist()}
              for side_index, side in enumerate(SIDES)
            },
            "objects": {},
          },
        }
        _write_json(frame_dir / "training_data.json", document)
        image_path = frame_dir / "rgb.png"
        Image.fromarray(frames["rgb"][index]).save(image_path, format="PNG")
        selectors.append(
          {
            "session_id": session_id,
            "frame_name": str(name),
            "training_data_path": str(relative_root / str(name) / "training_data.json"),
            "rgb_path": str(destination / relative_root / str(name) / "rgb.png"),
            "rgb_bytes": image_path.stat().st_size,
            "rgb_sha256": sha256_file(image_path),
          }
        )
      source_metadata = {
        "source_schema": SOURCE_SCHEMA,
        "source_hdf5_path": str(source),
        "source_hdf5_sha256": source_hash,
        "source_recording_metadata": metadata,
        "source_outcome": outcome,
        "object_name": object_name,
        "object_pose_source": object_pose_path,
        "camera": camera,
        "pose_source": "recorded actual robot wrist and distal-link site FK at rendered epoch",
        "camera_reference": "current_observation_camera_frozen_at_window_start",
        "world_from_camera_conversion": "MuJoCo c2w @ diag(1,-1,-1,1)",
        "tactile_reduction": "arithmetic_mean_of_all_35_taxels_including_zeros",
        "tactile_source": _text(file.attrs.get("contact_force_source", "")),
        "force_calibration": "physical MuJoCo solver newtons, not real hardware calibration",
        "force_is_spatial_estimate": bool(force.attrs.get("is_spatial_estimate", True)),
        "force_action": f"force acting on tactile pad from {object_name} contact",
        "tangent_axis_semantics": ["grid_column_positive", "grid_row_positive"],
        "tactile_source_link_names": [_text(name) for name in force["link_names"][:]],
        "done_source": "successful episode terminal sample; earlier frame labels false",
        "grasp_label_available": False,
        "static_anchor_source": f"initial recorded {object_name} pose held fixed, not dynamic Object6D",
        "static_anchor_state_index": 0,
        "static_anchor_timestamp_ns": int(state_times[0]),
      }
      for key in ("tangent_basis_local", "normal_axis_local"):
        if key in force:
          source_metadata[key] = force[key][:].tolist()
      sidecar_relative = Path("tict_sidecars") / session_id / "fingertip_tactile_v1.npz"
      sidecar_path = staging / sidecar_relative
      sidecar_path.parent.mkdir(parents=True)
      np.savez_compressed(
        sidecar_path,
        schema_version=np.array(SIDECAR_SCHEMA),
        frame_names=names,
        timestamps_ns=timestamps,
        side_names=np.array(SIDES),
        finger_names=np.array(FINGERS),
        T_fingertip_to_wrist=relative,
        finger_valid=np.ones((count, 2, 5), dtype=np.bool_),
        **tactile,
        tactile_channel_names=np.array(CHANNELS),
        translation_unit=np.array("metre"),
        force_unit=np.array("newton"),
        max_sync_error_ns=np.array(max_sync_error_ns, dtype=np.int64),
        source_metadata_json=np.array(
          json.dumps(source_metadata, sort_keys=True, allow_nan=False)
        ),
      )
      stats, action_audit = _training_stats(
        wrist, relative, cameras, tactile, horizon, session_id
      )
      _write_json(staging / "train_statistics.json", stats)
      _write_json(
        staging / "split_manifest.json",
        {
          "schema_version": "kaihand-tict-split-v1",
          "split_unit": "session",
          "splits": {"train": [session_id], "validation": [], "test": []},
        },
      )
      _write_json(
        staging / "selector_manifest.json",
        {
          "schema_version": "kaihand-tict-selector-v1",
          "img_name": "rgb.png",
          "records": selectors,
        },
      )
      _write_json(
        staging / "window_starts.json",
        {
          "schema_version": "kaihand-tict-window-starts-v1",
          "horizon": horizon,
          "sessions": {session_id: list(range(count - horizon))},
        },
      )
      (staging / "DATA_CONTRACT.md").write_text(
        _data_contract(camera, horizon, session_id, object_name), encoding="utf-8"
      )
      valid_errors = tactile["tactile_sync_error_ns"][tactile["tactile_frame_valid"]]
      audit = {
        "schema_version": "kaihand-tict-audit-v1",
        "contract_version": CONTRACT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "validation_scope": "local export preflight and all action windows",
        "upstream_loader_verified": False,
        "sessions": [
          {
            "session_id": session_id,
            "split": "train",
            "frame_count": count,
            "window_count": count - horizon,
            "sidecar_path": str(sidecar_relative),
            "sidecar_sha256": sha256_file(sidecar_path),
          }
        ],
        "source": {
          "path": str(source),
          "sha256": source_hash,
          "sha256_verification": source_hash_verification,
          "bytes": source_stat.st_size,
          "successful_outcome": True,
        },
        "source_clock": {
          "image": "cameras/<camera>/pose_timestamp",
          "tactile": "tactile_contact_force/timestamp",
          "median_frame_interval_ns": int(np.median(np.diff(timestamps))),
          "min_frame_interval_ns": int(np.min(np.diff(timestamps))),
          "max_frame_interval_ns": int(np.max(np.diff(timestamps))),
          "max_poststep_to_render_offset_ns": int(
            np.max(acquisition_times - timestamps)
          ),
          "max_valid_tactile_sync_error_ns": int(valid_errors.max())
          if len(valid_errors)
          else None,
        },
        "coverage": {
          "finger_pose_valid_fraction": 1.0,
          "tactile_frame_valid_fraction": float(tactile["tactile_frame_valid"].mean()),
          "tactile_channel_valid_fraction": tactile["tactile_channel_mask"]
          .mean(axis=0)
          .tolist(),
          "tactile_nonzero_count": np.count_nonzero(
            tactile["tactile_mean"], axis=0
          ).tolist(),
        },
        "action": action_audit,
        "limitations": [
          "single session; no held-out validation or test",
          "source perturbations preserved as recorded; converter adds no noise",
          "robot-site frames, not anatomical MANO calibration",
          "local manifest schemas; target loader/launcher not executed",
        ],
      }
      _write_json(staging / "dataset_audit.json", audit)
    final_stat = source.stat()
    if (source_stat.st_size, source_stat.st_mtime_ns) != (
      final_stat.st_size,
      final_stat.st_mtime_ns,
    ):
      raise RuntimeError("source episode changed during export")
    if destination.exists():
      raise FileExistsError(
        f"refusing to replace output created concurrently: {destination}"
      )
    os.rename(staging, destination)
    return audit
  except Exception as error:
    _write_json(
      staging / "FAILED.json",
      {"error": str(error), "source": str(source), "output_published": False},
    )
    raise
