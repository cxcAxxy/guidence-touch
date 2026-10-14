"""Recorded-pose and optional tactile extensions for head-only EgoSteer shards.

These sidecars do not change the upstream 116-D lowdim or 48-D action schema.
No simulation, model reconstruction, image interpolation, or force smoothing is
performed here. Source HDF5 files are always opened read-only by the caller.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import h5py
import numpy as np

ARCHIVE_SCHEMA = "kaihand-egosteer-episode-observations-v1"
FINGERS = ("thumb", "index", "middle", "ring", "little")


def text(value) -> str:
  return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def proper_se3(values: np.ndarray, label: str) -> None:
  if values.shape[-2:] != (4, 4) or not np.isfinite(values).all():
    raise ValueError(f"{label}: expected finite SE(3) matrices")
  rotations = values[..., :3, :3]
  if not (
    np.allclose(values[..., 3, :], (0, 0, 0, 1), atol=1e-7)
    and np.allclose(np.swapaxes(rotations, -1, -2) @ rotations, np.eye(3), atol=1e-6)
    and np.allclose(np.linalg.det(rotations), 1.0, atol=1e-6)
  ):
    raise ValueError(f"{label}: improper SE(3)")


def increasing_times(values: np.ndarray, label: str) -> None:
  if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
    raise ValueError(f"{label}: expected finite nonempty timestamp vector")
  if np.any(np.diff(values) <= 0):
    raise ValueError(f"{label}: timestamps must strictly increase")


def validate_poker_outcome(metadata: dict, outcome: dict) -> None:
  from kaihand_tactile_env.tasks.poker_draw.acceptance import (
    STRICT_FORCE_POLICY,
    accept_edge,
    validate_recorded_acceptance,
  )
  if metadata.get("scene") != "poker-draw" or metadata.get("object") != "card":
    raise ValueError("recorded-pose poker export requires scene=poker-draw, object=card")
  if outcome.get("success") is not True or outcome.get("object_name") != "card":
    raise ValueError("poker source is not a successful card episode")
  validate_recorded_acceptance(metadata, outcome)
  for name in ("terminal_pinch", "sustained_pinch"):
    if outcome.get(name) is not True:
      raise ValueError(f"poker outcome lacks original success gate {name}")
  if str(metadata.get("preset", "")).startswith("middle-force"):
    edge = outcome.get("edge_outcome") or {}
    handoff = outcome.get("handoff_outcome") or {}
    if not all(edge.get(key) is True for key in (
      "target_reached", "held_at_edge"
    )) or not accept_edge(edge, metadata.get("acceptance_policy", STRICT_FORCE_POLICY)) or handoff.get("completed") is not True:
      raise ValueError("poker source lacks qualified full slide and handoff")


def archived_motion(
  file: h5py.File,
  canonical: dict,
  *,
  required_terminal_phase: str | None = "terminal_settle",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  camera = file["cameras/head"]
  if text(camera.attrs.get("taskspace_schema", "")) != "kaihand-native-site-se3-v1":
    raise ValueError("poker requires archived native-site taskspace poses")
  for name, expected in (("side_names_json", ["left", "right"]),
                         ("finger_names_json", list(FINGERS))):
    if json.loads(text(camera.attrs.get(name, "null"))) != expected:
      raise ValueError(f"poker camera ordering mismatch: {name}")
  acquisition = np.asarray(camera["timestamp"], dtype=np.float64)
  times = np.asarray(camera["pose_timestamp"], dtype=np.float64)
  increasing_times(acquisition, "camera acquisition")
  increasing_times(times, "render pose")
  count = len(times)
  if acquisition.shape != times.shape:
    raise ValueError("camera acquisition/pose length mismatch")
  dt = 1.0 / int(file.attrs["physics_hz"])
  if np.any(acquisition < times - 1e-10) or np.any(acquisition - times > dt + 1e-8):
    raise ValueError("render pose clock must be within one preceding physics step")
  wrists = np.asarray(camera["world_from_wrist"], dtype=np.float64)
  tips = np.asarray(camera["world_from_fingertip"], dtype=np.float64)
  if wrists.shape != (count, 2, 4, 4) or tips.shape != (count, 2, 5, 4, 4):
    raise ValueError("archived full-pose shape mismatch")
  proper_se3(wrists, "wrist")
  proper_se3(tips, "fingertips")
  proper_se3(np.asarray(camera["world_from_camera"]), "camera")
  state_time = np.asarray(file["state/timestamp"], dtype=np.float64)
  increasing_times(state_time, "state")
  state_indices = np.asarray(camera["state_index"])
  if (state_indices.shape != (count,) or state_indices.dtype.kind not in "iu"
      or np.any(state_indices < 0) or np.any(state_indices >= len(state_time))
      or state_indices[-1] != len(state_time) - 1
      or abs(acquisition[-1] - state_time[-1]) > 1e-9):
    raise ValueError("task requires an exact archived terminal camera/state")
  if (
    required_terminal_phase is not None
    and text(file["commands/phase"][-1]) != required_terminal_phase
  ):
    raise ValueError(
      f"task requires terminal phase {required_terminal_phase!r}"
    )
  # Preserve the established MANO-compatible wrist-axis convention. Fingertip
  # world positions need no canonicalization; the upstream loader localizes them.
  rotations = wrists[..., :3, :3].copy()
  for side_index, side in enumerate(("left", "right")):
    rotations[:, side_index] = rotations[:, side_index] @ canonical[side]
  wrist_values = np.concatenate((
    wrists[:, :, :3, 3].reshape(count, 6),
    np.concatenate((rotations[..., 0], rotations[..., 1]), axis=-1).reshape(count, 12),
  ), axis=-1).astype(np.float32)
  hand_values = tips[:, :, :, :3, 3].reshape(count, 30).astype(np.float32)
  return times, wrist_values, hand_values


def observation_archive(file: h5py.File, *, episode_index: int, prefix: int,
                        pose_times: np.ndarray, source_sha256: str,
                        include_tactile: bool) -> dict[str, np.ndarray]:
  """Keep all frames, including action-only and off-grid terminal observations."""
  camera = file["cameras/head"]
  count = len(pose_times)
  keys = np.asarray([f"episode_{episode_index:06d}_frame_{i:06d}" for i in range(count)])
  values = {
    "schema_version": np.asarray(ARCHIVE_SCHEMA),
    "source_hdf5_sha256": np.asarray(source_sha256),
    "episode_index": np.asarray(episode_index, dtype=np.int64),
    "sample_keys": keys,
    "source_frame_index": np.arange(count, dtype=np.int64),
    "acquisition_time_s": np.asarray(camera["timestamp"], dtype=np.float64),
    "pose_time_s": np.asarray(pose_times, dtype=np.float64),
    "state_index": np.asarray(camera["state_index"], dtype=np.int64),
    "training_sample_mask": np.arange(count) < prefix - 1,
    "strict_grid_mask": np.arange(count) < prefix,
    "terminal_mask": np.arange(count) == count - 1,
    "tactile_included": np.asarray(include_tactile, dtype=np.bool_),
    "outcome_json": np.asarray(text(file.attrs["outcome_json"])),
    "source_metadata_json": np.asarray(text(file.attrs["metadata_json"])),
  }
  if not include_tactile:
    return values
  group = file["tactile_contact_force"]
  if text(group.attrs.get("force_unit", "")) != "N":
    raise ValueError("tactile archive requires calibrated simulation force units N")
  names = [text(name) for name in group["link_names"][:]]
  expected = [f"hand_{side}_{finger}_{'link6' if finger == 'thumb' else 'link4'}"
              for side in ("l", "r")
              for finger in ("thumb", "index", "middle", "ring", "pinky")]
  if len(set(names)) != len(names) or any(name not in names for name in expected):
    raise ValueError("tactile source must identify both hands and all five fingers")
  order = np.asarray([names.index(name) for name in expected])
  force_times = np.asarray(group["timestamp"], dtype=np.float64)
  increasing_times(force_times, "tactile")
  source_index = np.searchsorted(force_times, pose_times, side="right") - 1
  source_time = np.full(count, -1.0)
  errors = np.full(count, -1.0)
  causal = source_index >= 0
  source_time[causal] = force_times[source_index[causal]]
  errors[causal] = pose_times[causal] - source_time[causal]
  valid = causal & (errors >= 0) & (errors <= 0.020)
  taxels = np.zeros((count, 2, 5, 7, 5, 3), dtype=np.float64)
  normal = group["normal_taxel_force_n"]
  tangent = group["tangent_taxel_force_n"]
  if normal.shape != (len(force_times), len(names), 7, 5) or tangent.shape != (*normal.shape, 2):
    raise ValueError("tactile taxel source shape mismatch")
  for target in np.flatnonzero(valid):
    source = int(source_index[target])
    taxels[target, ..., 0] = np.asarray(normal[source])[order].reshape(2, 5, 7, 5)
    taxels[target, ..., 1:] = np.asarray(tangent[source])[order].reshape(2, 5, 7, 5, 2)
  if not np.isfinite(taxels).all() or np.any(taxels[..., 0] < -1e-10):
    raise ValueError("invalid source tactile force")
  values.update({
    "tactile_source_index": source_index.astype(np.int64),
    "tactile_source_time_s": source_time,
    "tactile_source_timestamps_s": force_times,
    "tactile_sync_error_s": errors,
    "max_sync_error_s": np.asarray(0.020),
    "tactile_frame_valid": valid,
    "tactile_channel_mask": np.broadcast_to(valid[:, None, None, None], (count, 2, 5, 3)).copy(),
    "tactile_taxel_force_n": taxels,
    "tactile_force_n": taxels.sum(axis=(3, 4)),
    "tactile_mean_n": taxels.mean(axis=(3, 4)),
    "side_names": np.asarray(("left", "right")),
    "finger_names": np.asarray(FINGERS),
    "channel_names": np.asarray(("normal", "tangent_x", "tangent_y")),
    "force_unit": np.asarray("N"),
    "force_metadata_json": np.asarray(text(group.attrs.get("metadata_json", "{}"))),
  })
  return values


def safe_sidecar_path(root: Path, value: object) -> Path:
  if not isinstance(value, str) or "\\" in value:
    raise ValueError("sidecar path must be a relative POSIX path")
  path = PurePosixPath(value)
  if path.is_absolute() or not path.parts or any(part in (".", "..") for part in path.parts):
    raise ValueError("unsafe sidecar path")
  target = root.joinpath(*path.parts)
  if target.is_symlink() or any(parent.is_symlink() for parent in target.parents if parent != root and root in parent.parents):
    raise ValueError("sidecar symlinks are not permitted")
  if not target.resolve().is_relative_to(root.resolve()):
    raise ValueError("sidecar path escapes dataset")
  return target


def validate_observation_archive(values, *, episode_index: int, samples: int,
                                 source_frames: int, prefix: int, source_sha256: str,
                                 include_tactile: bool) -> None:
  def require(name, shape=None):
    array = np.asarray(values[name])
    if array.dtype.kind == "O" or (shape is not None and array.shape != shape):
      raise ValueError(f"invalid archive field {name}")
    return array

  if str(require("schema_version", ())) != ARCHIVE_SCHEMA:
    raise ValueError("unsupported observation archive schema")
  if int(require("episode_index", ())) != episode_index or str(require("source_hdf5_sha256", ())) != source_sha256:
    raise ValueError("observation archive source identity mismatch")
  count = source_frames
  frames = np.arange(count)
  expected_keys = np.asarray([f"episode_{episode_index:06d}_frame_{i:06d}" for i in frames])
  if not np.array_equal(require("sample_keys", (count,)), expected_keys):
    raise ValueError("observation archive sample keys mismatch")
  for name, expected in (("source_frame_index", frames), ("training_sample_mask", frames < samples),
                         ("strict_grid_mask", frames < prefix), ("terminal_mask", frames == count - 1)):
    if not np.array_equal(require(name, (count,)), expected):
      raise ValueError(f"observation archive {name} mismatch")
  acquisition = require("acquisition_time_s", (count,))
  pose = require("pose_time_s", (count,))
  increasing_times(acquisition, "archive acquisition")
  increasing_times(pose, "archive pose")
  if np.any(pose > acquisition + 1e-10):
    raise ValueError("archive pose is later than acquisition")
  if bool(require("tactile_included", ())) != include_tactile:
    raise ValueError("archive tactile inclusion mismatch")
  if not include_tactile:
    return
  for name, expected in (("side_names", ["left", "right"]), ("finger_names", list(FINGERS)),
                         ("channel_names", ["normal", "tangent_x", "tangent_y"])):
    if require(name).tolist() != expected:
      raise ValueError(f"archive tactile {name} mismatch")
  if str(require("force_unit", ())) != "N" or float(require("max_sync_error_s", ())) != 0.020:
    raise ValueError("archive tactile units/sync limit mismatch")
  force_times = require("tactile_source_timestamps_s")
  increasing_times(force_times, "archive force")
  indices = require("tactile_source_index", (count,))
  expected_indices = np.searchsorted(force_times, pose, side="right") - 1
  if indices.dtype.kind not in "iu" or not np.array_equal(indices, expected_indices):
    raise ValueError("archive tactile is not latest-not-future")
  source_time = np.full(count, -1.0)
  error = np.full(count, -1.0)
  causal = indices >= 0
  source_time[causal] = force_times[indices[causal]]
  error[causal] = pose[causal] - source_time[causal]
  valid = causal & (error >= 0) & (error <= 0.020)
  for name, expected in (("tactile_source_time_s", source_time), ("tactile_sync_error_s", error)):
    if not np.allclose(require(name, (count,)), expected, atol=1e-12, rtol=0):
      raise ValueError(f"archive {name} mismatch")
  mask = np.broadcast_to(valid[:, None, None, None], (count, 2, 5, 3))
  if not np.array_equal(require("tactile_frame_valid", (count,)), valid) or not np.array_equal(require("tactile_channel_mask", mask.shape), mask):
    raise ValueError("archive tactile mask mismatch")
  taxels = require("tactile_taxel_force_n", (count, 2, 5, 7, 5, 3))
  if not np.isfinite(taxels).all() or np.any(taxels[..., 0] < -1e-10) or np.any(taxels[~valid] != 0):
    raise ValueError("archive tactile values invalid")
  for name, expected in (("tactile_force_n", taxels.sum(axis=(3, 4))),
                         ("tactile_mean_n", taxels.mean(axis=(3, 4)))):
    if not np.allclose(require(name, expected.shape), expected, atol=1e-9, rtol=1e-9):
      raise ValueError(f"archive tactile force conservation failed: {name}")
