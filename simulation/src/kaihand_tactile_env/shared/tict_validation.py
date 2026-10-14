"""Local checks of the documented T-ICT release contract, not a remote loader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "little")


def check_se3(value, label="transform"):
  value = np.asarray(value, dtype=np.float64)
  if value.shape[-2:] != (4, 4) or not np.all(np.isfinite(value)):
    raise ValueError(f"{label}: expected finite SE(3)")
  rotation = value[..., :3, :3]
  if not (
    np.allclose(value[..., 3, :], [0, 0, 0, 1], atol=1e-6)
    and np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=1e-5)
    and np.allclose(np.linalg.det(rotation), 1, atol=1e-5)
  ):
    raise ValueError(f"{label}: invalid rigid transform (including reflection)")
  return value


def pose9(transform):
  transform = check_se3(transform)
  return np.concatenate(
    (transform[..., :3, 3], transform[..., :3, 0], transform[..., :3, 1]), axis=-1
  )


def build_action_window(world_from_wrists, fingertip_to_wrist, reference_c2w):
  """Future wrists in observation-t camera; fingers in their FUTURE wrist.

  Input [H,2,4,4], [H,2,5,4,4]. Each side contributes wrist then five fingers.
  No future RGB, camera, or tactile observation is read to form the reference.
  """
  wrists = check_se3(world_from_wrists, "future wrists")
  fingers = check_se3(fingertip_to_wrist, "future fingertips")
  reference = check_se3(reference_c2w, "frozen reference camera")
  if wrists.ndim != 4 or wrists.shape[1:] != (2, 4, 4):
    raise ValueError("future wrists must have shape [H,2,4,4]")
  if fingers.shape != (len(wrists), 2, 5, 4, 4) or reference.shape != (4, 4):
    raise ValueError("invalid action window inputs")
  wrist9 = pose9(np.linalg.inv(reference) @ wrists)
  finger9 = pose9(fingers)
  return np.concatenate((wrist9[:, :, None, :], finger9), axis=2).reshape(
    len(wrists), 108
  )


def _json(path):
  result = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(result, dict):
    raise ValueError(f"{path.name}: expected JSON object")
  return result


def _hash(path):
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def validate_tict_release(root):
  """Return a non-mutating exhaustive local report, including every H50 window."""
  root = Path(root).expanduser().resolve()
  errors = []
  session_reports = []
  try:
    split = _json(root / "split_manifest.json")
    selector = _json(root / "selector_manifest.json")
    windows = _json(root / "window_starts.json")
    audit = _json(root / "dataset_audit.json")
    if not (root / "DATA_CONTRACT.md").is_file():
      raise ValueError("missing DATA_CONTRACT.md")
    if (
      split.get("schema_version") != "kaihand-tict-split-v1"
      or split.get("split_unit") != "session"
    ):
      raise ValueError("unknown split contract")
    if (
      selector.get("schema_version") != "kaihand-tict-selector-v1"
      or selector.get("img_name") != "rgb.png"
    ):
      raise ValueError("selector must explicitly select rgb.png")
    if (
      windows.get("schema_version") != "kaihand-tict-window-starts-v1"
      or windows.get("horizon") != 50
    ):
      raise ValueError("expected documented H50 window contract")
    partitions = split["splits"]
    if set(partitions) != {"train", "validation", "test"}:
      raise ValueError("split partition names differ from contract")
    sessions = sum((partitions[name] for name in ("train", "validation", "test")), [])
    if not sessions or len(sessions) != len(set(sessions)):
      raise ValueError("empty sessions or train/validation/test session leakage")
    if len(sessions) == 1 and partitions["train"] != sessions:
      raise ValueError("single-session acceptance release must be train-only")
    if set(windows["sessions"]) != set(sessions):
      raise ValueError("window sessions differ from split sessions")
    stats = _json(root / "train_statistics.json")
    if (
      stats.get("schema_version") != "kaihand-tict-train-statistics-v1"
      or stats.get("split") != "train"
      or stats.get("sessions") != partitions["train"]
      or stats.get("rotation_6d_normalization") != "none"
      or stats.get("raw_release_is_normalized") is not False
    ):
      raise ValueError(
        "statistics must use train sessions only and preserve raw rotations"
      )
    if audit.get("upstream_loader_verified") is not False:
      raise ValueError(
        "release must not claim unperformed upstream loader verification"
      )
    records = selector["records"]
    record_map = {(r["session_id"], r["frame_name"]): r for r in records}
    if len(record_map) != len(records) or any(
      key[0] not in sessions for key in record_map
    ):
      raise ValueError("duplicate or unlisted selector records")
    expected_record_count = 0
    for session in sessions:
      if (
        not isinstance(session, str)
        or Path(session).name != session
        or session in {".", ".."}
      ):
        raise ValueError("unsafe session id")
      directory = (
        root / "production" / session / "09_humanego_adapter/preprocess/all_data"
      )
      frame_dirs = sorted(p.name for p in directory.iterdir() if p.is_dir())
      n = len(frame_dirs)
      if n < 51 or frame_dirs != [f"{i:05d}" for i in range(n)]:
        raise ValueError(
          f"{session}: requires at least 51 consecutive five-digit frames"
        )
      expected_record_count += n
      sidecar_path = root / "tict_sidecars" / session / "fingertip_tactile_v1.npz"
      with np.load(sidecar_path, allow_pickle=False) as archive:
        sidecar = {name: archive[name] for name in archive.files}
      for name, value in {
        "schema_version": "egotouch-fingertip-tactile-sidecar-v1",
        "translation_unit": "metre",
      }.items():
        if sidecar[name].shape != () or sidecar[name].item() != value:
          raise ValueError(f"{session}: invalid {name}")
      if sidecar["force_unit"].shape != () or sidecar["force_unit"].item() not in {
        "N",
        "newton",
      }:
        raise ValueError("simulated calibrated force must declare newtons")
      for name, expected in {
        "frame_names": frame_dirs,
        "side_names": list(SIDES),
        "finger_names": list(FINGERS),
        "tactile_channel_names": ["normal", "tangent_x", "tangent_y"],
      }.items():
        if sidecar[name].tolist() != expected:
          raise ValueError(f"{session}: invalid {name} order")
      shapes = {
        "timestamps_ns": (n,),
        "T_fingertip_to_wrist": (n, 2, 5, 4, 4),
        "finger_valid": (n, 2, 5),
        "tactile_mean": (n, 2, 5, 3),
        "tactile_channel_mask": (n, 2, 5, 3),
        "tactile_frame_valid": (n,),
        "tactile_source_index": (n,),
        "tactile_sync_error_ns": (n,),
      }
      for name, shape in shapes.items():
        if sidecar[name].shape != shape:
          raise ValueError(
            f"{session}: {name} has shape {sidecar[name].shape}, expected {shape}"
          )
      for name in ("finger_valid", "tactile_channel_mask", "tactile_frame_valid"):
        if sidecar[name].dtype != np.bool_:
          raise ValueError(f"{name} must have boolean dtype")
      for name in ("timestamps_ns", "tactile_source_index", "tactile_sync_error_ns"):
        if sidecar[name].dtype != np.int64:
          raise ValueError(f"{name} must have int64 dtype")
      times = sidecar["timestamps_ns"]
      if np.any(np.diff(times) <= 0) or np.any(times < 0):
        raise ValueError("timestamps must increase strictly")
      valid = sidecar["tactile_frame_valid"]
      error_ns = sidecar["tactile_sync_error_ns"]
      maximum = sidecar["max_sync_error_ns"]
      if (
        maximum.shape != ()
        or maximum.dtype != np.int64
        or not 0 <= maximum.item() <= 20_000_000
      ):
        raise ValueError("invalid maximum tactile age")
      source_indices = sidecar["tactile_source_index"]
      if (
        np.any(source_indices[valid] < 0)
        or np.any(error_ns[valid] < 0)
        or np.any(error_ns[valid] > maximum)
      ):
        raise ValueError("tactile sync is noncausal, stale, or lacks a source index")
      if np.any(np.diff(source_indices[valid]) < 0):
        raise ValueError("tactile source indices move backwards")
      if np.any(sidecar["tactile_channel_mask"][~valid]):
        raise ValueError("invalid tactile frame has valid channels")
      means = sidecar["tactile_mean"]
      if not np.all(np.isfinite(means)) or np.any(
        means[~sidecar["tactile_channel_mask"]] != 0
      ):
        raise ValueError("tactile values must be finite; missing channels must be zero")
      if np.any(means[..., 0] < -1e-10):
        raise ValueError("normal tactile force cannot be negative")
      source_metadata = json.loads(sidecar["source_metadata_json"].item())
      if not isinstance(source_metadata, dict):
        raise ValueError("source_metadata_json must be a JSON object")
      finger_valid = sidecar["finger_valid"]
      check_se3(sidecar["T_fingertip_to_wrist"][finger_valid], "valid fingertips")
      wrists = np.broadcast_to(np.eye(4), (n, 2, 4, 4)).copy()
      wrist_valid = np.zeros((n, 2), dtype=bool)
      cameras = []
      done = []
      session_anchor = None
      for i, frame in enumerate(frame_dirs):
        json_path = directory / frame / "training_data.json"
        data = _json(json_path)
        metadata = data["metadata"]
        if metadata["idx"] != i or metadata["timestamp_ns"] != times[i]:
          raise ValueError(
            f"{session}/{frame}: mismatched JSON frame index or timestamp"
          )
        if not isinstance(metadata["is_finished"], bool):
          raise ValueError("is_finished must be boolean")
        done.append(metadata["is_finished"])
        c2w = check_se3(metadata["c2w"], "c2w")
        cam0 = check_se3(metadata["world_transforms"]["cam0"], "cam0")
        if c2w.shape != (4, 4) or not np.allclose(c2w, cam0, atol=1e-8):
          raise ValueError(
            "this contract uses current camera as frozen sample-start frame"
          )
        cameras.append(cam0)
        if metadata["anchor_key"] != "virtual_static_anchor":
          raise ValueError("this contract requires an explicit virtual_static_anchor")
        anchor = check_se3(
          metadata["world_transforms"]["virtual_static_anchor"], "anchor"
        )
        if anchor.shape != (4, 4):
          raise ValueError("static anchor must have shape [4,4]")
        if session_anchor is None:
          session_anchor = anchor
        elif not np.allclose(anchor, session_anchor, atol=1e-9, rtol=0):
          raise ValueError(
            "virtual_static_anchor must remain fixed for the entire session"
          )
        intrinsic = np.asarray(metadata["k"])
        if (
          intrinsic.shape != (9,)
          or not np.all(np.isfinite(intrinsic))
          or intrinsic[0] <= 0
          or intrinsic[4] <= 0
        ):
          raise ValueError("invalid camera intrinsic")
        if (metadata["w"], metadata["h"]) != (320, 240):
          raise ValueError("document RGB shape is 320x240")
        if not isinstance(data["obs"], dict) or not isinstance(
          data["entities"]["objects"], dict
        ):
          raise ValueError("obs and objects must be dictionaries")
        hands = data["entities"]["hands_hawor_v3"]
        for side_id, side in enumerate(SIDES):
          if side in hands:
            wrists[i, side_id] = check_se3(
              hands[side]["T_hand_to_world"], f"{side} wrist"
            )
            wrist_valid[i, side_id] = True
        record = record_map[(session, frame)]
        image_path = json_path.with_name("rgb.png")
        if record["training_data_path"] != str(json_path.relative_to(root)) or record[
          "rgb_path"
        ] != str(image_path):
          raise ValueError("selector paths do not resolve to expected frame files")
        if (
          image_path.stat().st_size != record["rgb_bytes"]
          or _hash(image_path) != record["rgb_sha256"]
        ):
          raise ValueError("RGB hash/byte-count mismatch")
        with Image.open(image_path) as image:
          image.load()
          if image.size != (320, 240) or image.mode != "RGB":
            raise ValueError("RGB is not a 320x240 three-channel image")
      if not done[-1] or any(done[:-1]):
        raise ValueError(
          "acceptance release requires exact final-frame terminal label only"
        )
      starts = windows["sessions"][session]
      if starts != list(range(n - 50)):
        raise ValueError(
          "window starts must cover all eligible 51-frame spans; no tail starts"
        )
      for start in starts:
        stop = start + 51
        if not np.any(wrist_valid[start, :, None] & finger_valid[start]):
          raise ValueError("window has no valid current wrist+finger")
        future = slice(start + 1, stop)
        valid_fingers = finger_valid[future] & wrist_valid[future, :, None]
        slot_masks = np.concatenate(
          (wrist_valid[future, :, None], valid_fingers), axis=2
        )
        action_mask = np.repeat(slot_masks[..., None], 9, axis=-1).reshape(50, 108)
        if not action_mask.any():
          raise ValueError("window has no valid future action slot")
        finger_poses = sidecar["T_fingertip_to_wrist"][future].copy()
        finger_poses[~valid_fingers] = np.eye(4)
        action = build_action_window(wrists[future], finger_poses, cameras[start])
        if action.shape != (50, 108) or not np.all(np.isfinite(action[action_mask])):
          raise ValueError("invalid reconstructed H50 action targets")
      audit_session = next(s for s in audit["sessions"] if s["session_id"] == session)
      if audit_session["frame_count"] != n or audit_session["window_count"] != len(
        starts
      ):
        raise ValueError("audit counts differ from actual release")
      if audit_session["sidecar_sha256"] != _hash(sidecar_path):
        raise ValueError("sidecar digest mismatch")
      session_reports.append(
        {
          "session_id": session,
          "frame_count": n,
          "window_count": len(starts),
          "max_sync_error_ns": int(error_ns[valid].max(initial=0)),
          "tactile_valid_frames": int(valid.sum()),
          "action_shape": [50, 108],
        }
      )
    if len(records) != expected_record_count:
      raise ValueError("selector contains missing or extra frame records")
  except (KeyError, ValueError, TypeError, OSError, StopIteration, IndexError) as error:
    errors.append(f"{type(error).__name__}: {error}")
  return {
    "schema_version": "kaihand-tict-local-validation-v1",
    "valid": not errors,
    "errors": errors,
    "sessions": session_reports,
    "upstream_loader_verified": False,
  }
