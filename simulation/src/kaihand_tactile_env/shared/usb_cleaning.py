"""Read-only acceptance checks for complete USB raw recordings.

This module imports neither MuJoCo nor the recorder.  It checks the archived
500-Hz transition stream independently, using the recorded model coordinate
names and the implicitfast position-integration identity.  It does not claim
that temporal alignment alone proves a controller or its dynamics correct.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class CleaningThresholds:
  """Conservative corruption screens, not robot hardware safety limits."""

  joint_speed_rad_s: float = 30.0
  object_linear_speed_m_s: float = 2.0
  object_angular_speed_rad_s: float = 30.0
  joint_acceleration_warning_rad_s2: float = 500.0
  integration_tolerance_rad: float = 1e-8
  object_cache_tolerance_m: float = 1e-9
  object_cache_tolerance_rad: float = 1e-6
  duplicate_image_motion_m: float = 0.001
  duplicate_image_motion_rad: float = 0.01


def _text(value):
  return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _names(dataset):
  return [_text(value) for value in dataset[:]]


def _metadata(file, name):
  value = json.loads(_text(file.attrs.get(name, "{}")))
  if not isinstance(value, dict):
    raise ValueError(f"{name} must contain a JSON object")
  return value


def audit_usb_insertion_physics(file, metadata=None, outcome=None):
  """Audit versioned seating evidence without a monitor or dynamics replay.

  Raw transport integrity and recorded physical criteria are separate gates.
  Legacy recordings retain their original checks, but never receive a claim
  that their insertion friction or bottom contact was verified.
  """
  metadata = _metadata(file, "metadata_json") if metadata is None else metadata
  outcome = _metadata(file, "outcome_json") if outcome is None else outcome
  version = metadata.get("contact_model_version")
  pipeline_errors, physics_errors = [], []
  pipeline = {"checked": False, "valid": None, "errors": pipeline_errors}
  physics = {"checked": False, "valid": None, "errors": physics_errors}
  report = {
    "contact_model_version": version,
    "pipeline_integrity": pipeline,
    "recorded_physics_gate": physics,
    "errors": [],
    "valid": True,
    "limitations": (
      "Checks archived monitor metrics against seating criteria and recorded "
      "object velocities; it does not recompute contact wrenches or certify "
      "real-connector force calibration."
    ),
  }
  if version is None:
    if "usb_insertion" in file:
      pipeline_errors.append(
        "USB insertion stream exists but metadata contact_model_version is missing"
      )
      pipeline.update(checked=True, valid=False)
      report.update(valid=False, errors=pipeline_errors.copy())
      return report
    report["status"] = "legacy_physics_gate_not_available"
    return report

  from kaihand_tactile_env.tasks.usb_insert import config

  if version not in (
    "passive_spring_shoes_bottom_out_v2",
    config.CONTACT_MODEL_VERSION,
  ):
    pipeline_errors.append(f"unsupported USB contact_model_version: {version}")
    pipeline.update(checked=True, valid=False)
    report.update(valid=False, errors=pipeline_errors.copy())
    return report

  bottom_min_force, bottom_hold = (
    (0.8, 0.1)
    if version == "passive_spring_shoes_bottom_out_v2"
    else (config.BOTTOM_OUT_MIN_FORCE_N, config.BOTTOM_OUT_HOLD_S)
  )
  pipeline["checked"] = True
  metrics = (
    "insertion_depth_m",
    "axial_resistance_n",
    "backstop_axial_resistance_n",
    "spring_axial_resistance_n",
    "spring_normal_load_n",
    "wall_normal_load_n",
    "backstop_normal_load_n",
    "axial_speed_m_s",
    "linear_speed_m_s",
    "angular_speed_rad_s",
    "maximum_socket_penetration_m",
    "orientation_error_rad",
  )
  flags = (
    "seated",
    "success",
    "shell_fits_aperture",
    "backstop_contact",
    "bottom_out_confirmed",
  )
  try:
    times = np.asarray(file["state/timestamp"][:], dtype=float)
    n = len(times)
    if (
      n < 2
      or not np.isfinite(times).all()
      or not np.allclose(np.diff(times), 0.002, rtol=0, atol=1e-10)
    ):
      raise ValueError("v2 insertion evidence needs complete 500-Hz state rows")
    group = file["usb_insertion"]
    if not isinstance(group, h5py.Group):
      raise ValueError("usb_insertion must be a stream group")
    for key, expected in (
      ("schema_version", "usb_insertion_monitor_v1"),
      ("contact_model_version", version),
      ("state_index_reference", "/state/timestamp"),
      ("observation_clock", "post_step_forward_v1"),
    ):
      if _text(group.attrs.get(key, "")) != expected:
        pipeline_errors.append(f"usb_insertion {key} does not match its contract")
    required = ("timestamp", "state_index", *metrics, *flags)
    for name in required:
      if name not in group or group[name].shape != (n,):
        raise ValueError(f"usb_insertion/{name} requires exactly {n} scalar rows")
    for name, dataset in group.items():
      if not isinstance(dataset, h5py.Dataset) or dataset.shape[:1] != (n,):
        raise ValueError(f"usb_insertion/{name} is not a complete state-aligned stream")
      values = dataset[:]
      if values.dtype.kind not in "fiub" or not np.isfinite(values).all():
        raise ValueError(f"usb_insertion/{name} contains nonnumeric or nonfinite data")
    if not np.array_equal(group["timestamp"][:], times):
      pipeline_errors.append("usb_insertion timestamps differ from same-row state")
    if group["state_index"].dtype.kind not in "iu" or not np.array_equal(
      group["state_index"][:], np.arange(n)
    ):
      pipeline_errors.append(
        "usb_insertion state indices are missing, duplicated or shifted"
      )
    if any(group[name].dtype.kind != "b" for name in flags):
      pipeline_errors.append("usb_insertion flags must be boolean streams")
    values = {name: np.asarray(group[name][:]) for name in (*metrics, *flags)}
    phases = np.asarray(_names(file["commands/phase"]))
    twists = np.asarray(file["objects/usb_plug/twist_linear_angular"][:], dtype=float)
    if phases.shape != (n,) or twists.shape != (n, 6) or not np.isfinite(twists).all():
      raise ValueError("v2 insertion phase and object velocity streams must align")
    pipeline.update(state_samples=n, observation_clock="post_step_forward_v1")
  except (KeyError, ValueError, TypeError, IndexError) as error:
    pipeline_errors.append(f"USB insertion stream integrity: {error}")
  pipeline["valid"] = not pipeline_errors
  if pipeline_errors:
    report.update(valid=False, errors=pipeline_errors.copy())
    return report

  physics["checked"] = True
  dt = 0.002
  tolerance = 1e-10
  for name in metrics:
    if name not in ("insertion_depth_m", "axial_speed_m_s") and np.any(
      values[name] < 0
    ):
      physics_errors.append(f"usb_insertion/{name} contains a negative magnitude")
  linear_speed = np.linalg.norm(twists[:, :3], axis=1)
  angular_speed = np.linalg.norm(twists[:, 3:], axis=1)
  for name, actual in (
    ("linear_speed_m_s", linear_speed),
    ("angular_speed_rad_s", angular_speed),
    ("axial_speed_m_s", -twists[:, 2]),
  ):
    if not np.allclose(values[name], actual, rtol=0, atol=tolerance):
      physics_errors.append(
        f"usb_insertion/{name} differs from recorded object velocity"
      )
  loaded_bottom = (
    values["backstop_normal_load_n"] >= config.SEATED_BACKSTOP_MIN_LOAD_N
  ) & (values["backstop_axial_resistance_n"] >= config.SEATED_BACKSTOP_MIN_LOAD_N)
  if not np.array_equal(values["backstop_contact"], loaded_bottom):
    physics_errors.append(
      "usb_insertion backstop_contact disagrees with positive bottom load"
    )
  near_bottom = (
    (
      values["insertion_depth_m"]
      >= config.BACKSTOP_DEPTH_M - config.SEATED_DEPTH_TOLERANCE_M
    )
    & (
      values["insertion_depth_m"]
      <= config.BACKSTOP_DEPTH_M + config.MAX_SOCKET_PENETRATION_M
    )
    & values["shell_fits_aperture"]
    & (values["orientation_error_rad"] <= np.deg2rad(3.0))
    & (values["maximum_socket_penetration_m"] <= config.MAX_SOCKET_PENETRATION_M)
  )
  stationary = (linear_speed <= config.SEATED_LINEAR_SPEED_M_S) & (
    angular_speed <= 0.05
  )
  active_bottom = (
    near_bottom
    & stationary
    & loaded_bottom
    & (values["backstop_axial_resistance_n"] >= bottom_min_force)
  )
  expected_latch = np.zeros(n, dtype=bool)
  retained, active_duration = False, 0.0
  for index in range(n):
    elapsed = 0.0 if index == 0 else float(times[index] - times[index - 1])
    if elapsed < -1e-12 or elapsed > 1.5 * dt or not near_bottom[index]:
      retained, active_duration = False, 0.0
      elapsed = 0.0
    if active_bottom[index]:
      if elapsed > 1e-12:
        active_duration += elapsed
      if active_duration >= bottom_hold - 1e-12:
        retained = True
    else:
      active_duration = 0.0
    expected_latch[index] = retained
  latch_matches = bool(np.array_equal(values["bottom_out_confirmed"], expected_latch))
  if not latch_matches:
    physics_errors.append(
      "usb_insertion bottom_out_confirmed disagrees with reconstructed contact history"
    )
  expected_seated = near_bottom & stationary & (loaded_bottom | expected_latch)
  if not np.array_equal(values["seated"], expected_seated):
    physics_errors.append(
      "usb_insertion seated flags disagree with physical seating criteria"
    )
  release_indices = np.flatnonzero(phases == "release")
  first_release = int(release_indices[0]) if len(release_indices) else n
  if not len(release_indices):
    physics_errors.append("USB bottom-out evidence has no subsequent release phase")
  bottom_phase = phases == "bottom_out"
  if np.any(bottom_phase[first_release:]):
    physics_errors.append("USB bottom_out phase must occur before the first release")
  qualified = (
    bottom_phase
    & expected_seated
    & (values["backstop_axial_resistance_n"] >= bottom_min_force)
    & (np.arange(n) < first_release)
  )
  longest, run = 0, 0
  terminal_seated_run = 0
  expected_success = np.zeros(n, dtype=bool)
  for index in range(n):
    run = run + 1 if qualified[index] else 0
    longest = max(longest, run)
    terminal_seated_run = terminal_seated_run + 1 if expected_seated[index] else 0
    # The monitor's first seated snapshot starts a timer at that instant.
    expected_success[index] = (
      max(0, terminal_seated_run - 1) * dt >= config.SEATED_DWELL_S - tolerance
    )
  if not np.array_equal(values["success"], expected_success):
    physics_errors.append(
      "usb_insertion success flags disagree with continuous seating dwell"
    )
  # The executor counts each qualified post-step observation as one dt of
  # applied bottom preload; the recorded model version selects the required force and duration.
  maximum_bottom_hold = longest * dt
  if maximum_bottom_hold < bottom_hold - tolerance:
    physics_errors.append(
      f"USB bottom_out lacks continuous pre-release bottom force >={bottom_min_force} N for {bottom_hold} s"
    )
  if outcome.get("active_bottom_out_confirmed") is not True:
    physics_errors.append("USB outcome lacks active_bottom_out_confirmed")
  declared_hold = outcome.get("bottom_out_hold_s")
  if (
    not isinstance(declared_hold, (int, float))
    or not np.isfinite(declared_hold)
    or declared_hold < bottom_hold - tolerance
    or declared_hold > maximum_bottom_hold + tolerance
  ):
    physics_errors.append(
      "USB bottom_out_hold_s is not supported by recorded consecutive force samples"
    )
  insertion = outcome.get("insertion", {})
  if not isinstance(insertion, dict):
    physics_errors.append("USB terminal insertion outcome must be a JSON object")
    insertion = {}
  for name in (*metrics, *flags):
    declared = insertion.get(name)
    actual = values[name][-1]
    if name in flags:
      matches = isinstance(declared, bool) and declared == bool(actual)
    else:
      matches = (
        isinstance(declared, (int, float))
        and np.isfinite(declared)
        and np.isclose(declared, actual, rtol=0, atol=tolerance)
      )
    if not matches:
      physics_errors.append(
        f"USB terminal insertion/{name} differs from final monitor row"
      )
  if not expected_seated[-1] or not expected_success[-1]:
    physics_errors.append(
      "USB final monitor row lacks stationary seating with current bottom support or retained confirmation"
    )
  if not expected_latch[-1]:
    physics_errors.append(
      "USB final monitor row does not retain its verified bottom-out history"
    )
  physics.update(
    valid=not physics_errors,
    bottom_force_threshold_n=bottom_min_force,
    required_bottom_hold_s=bottom_hold,
    maximum_continuous_bottom_hold_s=maximum_bottom_hold,
    bottom_out_latch_reconstructed=latch_matches,
    bottom_out_confirmation_retained_at_end=bool(expected_latch[-1]),
    bottom_out_retention_semantics=(
      "bottom contact confirmation retained after unloading while depth/alignment/penetration "
      "stay valid; reset, clock discontinuity or leaving that region clears history"
    ),
    bottom_hold_sample_duration_semantics="consecutive qualified samples multiplied by 0.002 s",
    first_release_state_index=first_release if len(release_indices) else None,
    terminal_continuous_seated_s=max(0, terminal_seated_run - 1) * dt,
    terminal_backstop_axial_resistance_n=float(
      values["backstop_axial_resistance_n"][-1]
    ),
    minimum_seated_depth_m=config.BACKSTOP_DEPTH_M - config.SEATED_DEPTH_TOLERANCE_M,
    maximum_seated_depth_m=config.BACKSTOP_DEPTH_M + config.MAX_SOCKET_PENETRATION_M,
    maximum_seated_linear_speed_m_s=config.SEATED_LINEAR_SPEED_M_S,
  )
  report.update(
    valid=not physics_errors,
    errors=physics_errors.copy(),
    status="v2_recorded_physics_checked",
  )
  return report


def _time_ns(values, label):
  values = np.asarray(values, dtype=np.float64)
  if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
    raise ValueError(f"{label}: expected nonempty finite timestamp vector")
  if np.any(values < 0) or np.any(values >= np.iinfo(np.int64).max / 1e9):
    raise ValueError(f"{label}: timestamp outside nonnegative int64 range")
  return np.rint(values * 1_000_000_000).astype(np.int64)


def _indices(mask, limit=24):
  result = np.flatnonzero(mask)
  return {"count": int(len(result)), "first_indices": result[:limit].tolist()}


def _peak(values, times, names=None, destination_offset=0):
  values = np.asarray(values)
  if not values.size:
    return {"maximum": 0.0, "state_index": None, "time_s": None}
  if not np.isfinite(values).all():
    return {
      "maximum": None,
      "state_index": None,
      "time_s": None,
      "nonfinite_derived_values": int(np.count_nonzero(~np.isfinite(values))),
    }
  location = np.unravel_index(int(np.argmax(np.abs(values))), values.shape)
  state_index = int(location[0]) + destination_offset
  result = {
    "maximum": float(np.abs(values[location])),
    "state_index": state_index,
    "time_s": float(times[state_index]),
  }
  if names is not None and len(location) > 1:
    result["coordinate"] = names[location[1]]
  return result


def _quaternion_distance(a, b):
  """SO(3) angle; opposite quaternion signs represent the same orientation."""
  a = np.asarray(a, dtype=np.float64)
  b = np.asarray(b, dtype=np.float64)
  a = a / np.linalg.norm(a, axis=-1, keepdims=True)
  b = b / np.linalg.norm(b, axis=-1, keepdims=True)
  # atan2 is much more accurate than acos for nearly identical quaternions.
  sign = np.where(np.sum(a * b, axis=-1, keepdims=True) < 0, -1.0, 1.0)
  difference = np.linalg.norm(a - sign * b, axis=-1)
  summation = np.linalg.norm(a + sign * b, axis=-1)
  return 4 * np.arctan2(difference, summation)


def _rotation_distance(a, b):
  relative = np.swapaxes(a, -1, -2) @ b
  cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1) / 2
  return np.arccos(np.clip(cosine, -1, 1))


def _validate_streams(file, n, errors):
  """Check every declared append-only stream without loading it wholesale."""
  streams = []
  required = {
    "state": (
      "timestamp",
      "qpos",
      "qvel",
      "robot_joint_position",
      "robot_joint_velocity",
      "robot_joint_effort",
    ),
    "commands": ("arm_joint_target", "hand_joint_target", "phase", "actuator_control"),
    "physics": (
      "solver_timestamp",
      "qacc",
      "qfrc_applied",
      "xfrc_applied",
      "actuator_force",
      "noslip_iterations",
    ),
    "tactile_contact_force": (
      "timestamp",
      "normal_force_n",
      "normal_force_world_n",
      "force_world_n",
      "tangent_force_world_n",
      "tangent_force_n",
      "tangent_load_n",
      "normal_taxel_force_n",
      "tangent_taxel_force_n",
      "tangent_taxel_load_n",
      "tangent_basis_world",
      "normal_axis_world",
      "contact_count",
    ),
  }
  for group_name in ("state", "commands", "physics", "tactile_contact_force"):
    if group_name not in file:
      errors.append(f"missing required group: {group_name}")
      continue
    for name in required[group_name]:
      if name not in file[group_name]:
        errors.append(f"missing required stream: {group_name}/{name}")
    for dataset in file[group_name].values():
      if isinstance(dataset, h5py.Dataset) and dataset.maxshape[0] is None:
        streams.append(dataset)
  if "objects" in file:
    for group in file["objects"].values():
      streams.extend(
        dataset for dataset in group.values() if isinstance(dataset, h5py.Dataset)
      )
  for path in ("contacts/frame_start", "contacts/frame_count"):
    if path in file:
      streams.append(file[path])
    else:
      errors.append(f"missing required stream: {path}")
  for dataset in streams:
    if dataset.shape[0] != n:
      errors.append(f"{dataset.name}: {dataset.shape[0]} rows, expected {n}")
      continue
    if dataset.dtype.kind not in "fc":
      continue
    for start in range(0, n, 256):
      block = dataset[start : start + 256]
      if not np.isfinite(block).all():
        errors.append(
          f"{dataset.name}: nonfinite values in rows {start}:{start + len(block)}"
        )
        break
  return {"checked_streams": len(streams), "state_rows": n}


def _contacts(file, n, errors):
  starts = np.asarray(file["contacts/frame_start"], dtype=np.int64)
  counts = np.asarray(file["contacts/frame_count"], dtype=np.int64)
  if starts.shape != (n,) or counts.shape != (n,):
    return
  expected_starts = np.r_[0, np.cumsum(counts[:-1])]
  if np.any(counts < 0) or not np.array_equal(starts, expected_starts):
    errors.append("contacts: noncontiguous event ranges or negative frame count")
    return
  total = int(np.sum(counts))
  events = file["contacts/events"]
  for dataset in events.values():
    if dataset.shape[0] != total:
      errors.append(f"{dataset.name}: event count differs from frame ranges")
  event_state = events["state_index"]
  if event_state.shape != (total,):
    return
  # A bounded memory check of all event-to-state associations.
  for first in range(0, n, 256):
    last = min(n, first + 256)
    start = int(starts[first])
    stop = int(starts[last - 1] + counts[last - 1])
    expected = np.repeat(np.arange(first, last), counts[first:last])
    if not np.array_equal(event_state[start:stop], expected):
      errors.append(f"contacts: event state indices disagree with rows {first}:{last}")
      break


def _noise_indices(file, state_times, errors):
  path = "control/precontact_noise"
  if path not in file:
    errors.append("missing precontact command issue trace")
    return {"available": False}
  group = file[path]
  issue = np.asarray(group["time_s"], dtype=float)
  if not np.isfinite(issue).all() or np.any(np.diff(issue) < 0):
    errors.append("precontact command issue times are nonfinite or reversed")
    return {"available": True, "commands": len(issue)}
  following = np.searchsorted(state_times, issue, side="right")
  previous = following - 1
  future = np.where(following < len(state_times), following, -1)
  if not np.array_equal(group["state_index_at_or_before_issue"][:], previous):
    errors.append("precontact issue trace has wrong at-or-before state indices")
  if not np.array_equal(group["first_future_state_index"][:], future):
    errors.append("precontact issue trace has wrong future state indices")
  if len(issue) and (issue[0] < state_times[0] or issue[-1] > state_times[-1]):
    errors.append("precontact issue time lies outside recorded episode")
  for dataset in group.values():
    if dataset.shape[0] != len(issue):
      errors.append(f"{dataset.name}: count differs from command issue timestamps")
  return {"available": True, "commands": len(issue), "state_brackets_checked": True}


def _cameras(file, state_ns, solver_ns, thresholds, errors, warnings):
  result = {}
  camera_hz = float(file.attrs["camera_hz"])
  if not np.isfinite(camera_hz) or camera_hz <= 0:
    raise ValueError("camera_hz must be finite and positive")
  period_ns = int(round(1e9 / camera_hz))
  for name, group in file["cameras"].items():
    times = _time_ns(group["timestamp"][:], f"camera {name}")
    indices = np.asarray(group["state_index"], dtype=np.int64)
    pose_times = _time_ns(group["pose_timestamp"][:], f"camera {name} pose")
    count = len(times)
    if group["state_index"].dtype.kind not in "iu":
      errors.append(f"camera {name}: state indices must use integer dtype")
    for dataset in group.values():
      if (
        isinstance(dataset, h5py.Dataset)
        and dataset.maxshape[0] is None
        and dataset.shape[0] != count
      ):
        errors.append(f"{dataset.name}: frame count differs from camera timestamp")
    if indices.shape != times.shape or pose_times.shape != times.shape:
      errors.append(f"camera {name}: frame index/pose timestamp shape mismatch")
      continue
    if np.any(indices < 0) or np.any(indices >= len(state_ns)):
      errors.append(f"camera {name}: state index outside episode")
      continue
    if not np.array_equal(times, state_ns[indices]):
      errors.append(
        f"camera {name}: acquisition timestamps differ from referenced state"
      )
    if not np.array_equal(pose_times, solver_ns[indices]):
      errors.append(
        f"camera {name}: rendered pose clock differs from same-row solver clock"
      )
    if np.any(pose_times > times):
      errors.append(f"camera {name}: pose appears after acquisition/state timestamp")
    for pose_name in ("world_from_camera", "world_from_wrist", "world_from_fingertip"):
      pose = np.asarray(group[pose_name])
      if (
        pose.shape[0] != count
        or pose.shape[-2:] != (4, 4)
        or not np.isfinite(pose).all()
        or np.any(np.abs(pose[..., :3, 3]) > 1e6)
      ):
        raise ValueError(
          f"camera {name}/{pose_name}: invalid finite rigid transform stream"
        )
      rotation = pose[..., :3, :3]
      if not (
        np.allclose(pose[..., 3, :], [0, 0, 0, 1], atol=1e-7, rtol=0)
        and np.allclose(
          np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=1e-6, rtol=0
        )
        and np.allclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)
      ):
        raise ValueError(f"camera {name}/{pose_name}: rotation is not SO(3)")
    if camera_hz == 30:
      # USB's 30 Hz observations occur on actual 500 Hz states: 0,34,66,100
      # ms, not synthetic 33.333 ms timestamps. Match the nearest physics tick
      # while independently checking every deadline and the exact terminal.
      step_ns = int(round(1e9 / float(file.attrs["physics_hz"])))
      last_tick = int(round((state_ns[-1] - state_ns[0]) / step_ns))
      count_regular = (
        int(np.floor((last_tick + 0.5) * camera_hz / float(file.attrs["physics_hz"])))
        + 1
      )
      ticks = np.floor(
        np.arange(count_regular) * float(file.attrs["physics_hz"]) / camera_hz + 0.5
      ).astype(np.int64)
      regular = state_ns[0] + ticks[ticks <= last_tick] * step_ns
    else:
      regular = np.arange(state_ns[0], state_ns[-1] + 1, period_ns, dtype=np.int64)
    expected = regular if regular[-1] == state_ns[-1] else np.r_[regular, state_ns[-1]]
    duplicate = _indices(np.diff(times) == 0)
    reversed_frames = _indices(np.diff(times) < 0)
    missing = np.setdiff1d(expected, times)
    unexpected = np.setdiff1d(times, expected)
    if duplicate["count"] or reversed_frames["count"]:
      errors.append(f"camera {name}: duplicated or reversed timestamps")
    if len(missing) or len(unexpected) or len(times) != len(expected):
      errors.append(
        f"camera {name}: scheduled or exact terminal RGB frame missing/extra"
      )
    if times[0] != state_ns[0] or times[-1] != state_ns[-1]:
      errors.append(f"camera {name}: missing initial or exact terminal frame")
    duplicate_pairs, suspicious_pairs = [], []
    digests, first_hash_index, previous_rgb = [], {}, None
    repeated_nonadjacent = []
    # Read one image at a time: an episode never needs to reside in RAM.
    for k in range(count):
      rgb = np.asarray(group["rgb"][k])
      if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        errors.append(f"camera {name}: frame {k} is not uint8 HxWx3 RGB")
        break
      digest = hashlib.sha256(rgb.tobytes()).hexdigest()
      digests.append(digest)
      if digest in first_hash_index and first_hash_index[digest] < k - 1:
        repeated_nonadjacent.append([first_hash_index[digest], k])
      first_hash_index.setdefault(digest, k)
      if previous_rgb is not None and np.array_equal(previous_rgb, rgb):
        entry = {"before_frame": k - 1, "after_frame": k}
        # Cached task poses are a better visible-motion signal than post-step qpos.
        motion_m, motion_rad = 0.0, 0.0
        for pose_name in (
          "world_from_camera",
          "world_from_wrist",
          "world_from_fingertip",
        ):
          pose = np.asarray(group[pose_name][k - 1 : k + 1])
          motion_m = max(
            motion_m,
            float(
              np.max(np.linalg.norm(pose[1, ..., :3, 3] - pose[0, ..., :3, 3], axis=-1))
            ),
          )
          motion_rad = max(
            motion_rad,
            float(
              np.max(_rotation_distance(pose[0, ..., :3, :3], pose[1, ..., :3, :3]))
            ),
          )
        entry.update(
          task_pose_translation_m=motion_m, task_pose_rotation_rad=motion_rad
        )
        duplicate_pairs.append(entry)
        if (
          motion_m > thresholds.duplicate_image_motion_m
          or motion_rad > thresholds.duplicate_image_motion_rad
        ):
          suspicious_pairs.append(entry)
      previous_rgb = rgb
    if suspicious_pairs:
      warnings.append(
        f"camera {name}: {len(suspicious_pairs)} identical adjacent RGB pairs despite recorded task motion; inspect visually"
      )
    result[name] = {
      "frames": count,
      "expected_frames": len(expected),
      "nominal_period_s": period_ns / 1e9,
      "last_interval_s": float(times[-1] - times[-2]) / 1e9 if count > 1 else None,
      "terminal_short_interval_is_expected": bool(len(expected) > len(regular)),
      "duplicate_timestamps": duplicate,
      "reversed_timestamps": reversed_frames,
      "missing_timestamp_ns": missing[:24].tolist(),
      "unexpected_timestamp_ns": unexpected[:24].tolist(),
      "identical_adjacent_rgb_count": len(duplicate_pairs),
      "identical_adjacent_rgb_first_pairs": duplicate_pairs[:24],
      "identical_nonadjacent_rgb_count": len(repeated_nonadjacent),
      "suspicious_identical_rgb_pairs": suspicious_pairs[:24],
      "pixel_sequence_sha256": hashlib.sha256("".join(digests).encode()).hexdigest(),
      "same_pixel_values_alone_do_not_imply_dropped_frames": True,
      "maximum_capture_minus_render_s": float(np.max(times - pose_times)) / 1e9,
    }
  if not result:
    errors.append("no camera streams")
  return result


def audit_usb_cleaning(input_path, thresholds=None):
  """Check one HDF5 episode without changing it; malformed input fails closed."""
  thresholds = thresholds or CleaningThresholds()
  for name, value in asdict(thresholds).items():
    if not np.isfinite(value) or value <= 0:
      raise ValueError(f"threshold {name} must be finite and positive")
  path = Path(input_path).resolve()
  errors, warnings, checks = [], [], {}
  report = {
    "schema": "usb_raw_cleaning_v2",
    "input": str(path),
    "valid": False,
    "errors": errors,
    "warnings": warnings,
    "thresholds": asdict(thresholds),
    "checks": checks,
    "limitations": [
      "This checks recorded causal indexing and kinematic integration, not a full dynamics replay.",
      "Raw actuator controls are physical actions; T-ICT future-pose labels are derived target states, not torque/control commands.",
      "Trajectory thresholds screen gross data corruption and are not hardware safety ratings.",
      "Visual/tactile contact meaning still requires review of event-boundary images.",
    ],
  }
  try:
    with h5py.File(path, "r") as file:
      metadata = _metadata(file, "metadata_json")
      outcome = _metadata(file, "outcome_json")
      report["episode"] = {
        key: metadata.get(key)
        for key in (
          "episode_index",
          "root_seed",
          "object_seed",
          "noise_seed",
          "motion_profile",
          "controller_source_sha256",
        )
      }
      report["recorded_success"] = outcome.get("success")
      if metadata.get("recording_contract") != "usb_insert_taskspace_raw_v1":
        errors.append("unsupported USB raw recording contract")
      if metadata.get("observation_clock") != "post_step_forward_v1":
        errors.append(
          "USB observation_clock must be post_step_forward_v1; legacy recordings "
          "mislabel the refreshed post-step cache"
        )
      if outcome.get("success") is not True:
        errors.append("episode is not marked successfully completed")
      state_times = np.asarray(file["state/timestamp"], dtype=float)
      state_ns = _time_ns(state_times, "state")
      n = len(state_times)
      if n < 2:
        raise ValueError("at least two physical state samples required")
      physics_hz = float(file.attrs["physics_hz"])
      if physics_hz != 500 or float(file.attrs["control_hz"]) != 500:
        errors.append("USB contract requires complete 500-Hz state/physics capture")
      period_ns = 2_000_000
      dt = 0.002
      differences = np.diff(state_ns)
      bad_step = differences != period_ns
      if state_ns[0] != 0 or np.any(bad_step):
        errors.append(
          "state timestamps must start at reset zero and advance exactly one 2-ms physics step"
        )
      solver_ns = _time_ns(file["physics/solver_timestamp"][:], "solver")
      touch_ns = _time_ns(file["tactile_contact_force/timestamp"][:], "tactile")
      expected_solver = state_ns
      if not np.array_equal(solver_ns, expected_solver):
        errors.append(
          "solver timestamps must equal the same-row post-step state after monitor mj_forward"
        )
      if not np.array_equal(touch_ns, solver_ns):
        errors.append("tactile timestamps differ from same-row solver timestamps")
      checks["clock"] = {
        "state_samples": n,
        "duration_s": float(state_times[-1] - state_times[0]),
        "expected_period_s": dt,
        "bad_state_intervals": _indices(bad_step),
        "duplicate_state_timestamps": _indices(differences == 0),
        "missing_state_steps": int(np.maximum(differences // period_ns - 1, 0).sum()),
        "reset_solver_timestamp_duplicate_is_expected": False,
        "observation_clock": metadata.get("observation_clock"),
        "clock_contract": "row k: monitor mj_forward refreshes solver/tactile/pose at state[k].time; applied actuator_control[k] was used from state[k-1] to state[k] and is logged at state[k].time",
      }
      checks["stream_counts_and_finite"] = _validate_streams(file, n, errors)
      checks["insertion_physics"] = audit_usb_insertion_physics(file, metadata, outcome)
      errors.extend(checks["insertion_physics"]["errors"])
      _contacts(file, n, errors)
      qpos = np.asarray(file["state/qpos"], dtype=float)
      qvel = np.asarray(file["state/qvel"], dtype=float)
      position = np.asarray(file["state/robot_joint_position"], dtype=float)
      velocity = np.asarray(file["state/robot_joint_velocity"], dtype=float)
      control = np.asarray(file["commands/actuator_control"], dtype=float)
      pose = np.asarray(file["objects/usb_plug/pose_wxyz"], dtype=float)
      if any(
        len(array) != n or not np.isfinite(array).all()
        for array in (qpos, qvel, position, velocity, control, pose)
      ):
        raise ValueError(
          "required state/action/USB pose arrays have invalid count or nonfinite values"
        )
      joints = _names(file["state/joint_names"])
      qpos_names = _names(file["state/full_qpos_names"])
      qvel_names = _names(file["state/full_qvel_names"])
      if (
        qpos.shape != (n, len(qpos_names))
        or qvel.shape != (n, len(qvel_names))
        or position.shape != (n, len(joints))
        or velocity.shape != position.shape
        or control.ndim != 2
        or control.shape[1] == 0
        or not joints
      ):
        raise ValueError(
          "state/action matrix shapes disagree with recorded coordinate names"
        )
      if len(set(qpos_names)) != len(qpos_names) or len(set(qvel_names)) != len(
        qvel_names
      ):
        errors.append("full coordinate names are not unique")
      pi = [qpos_names.index(name) for name in joints]
      vi = [qvel_names.index(name) for name in joints]
      if not np.array_equal(position, qpos[:, pi]):
        errors.append("robot joint positions disagree with named full qpos coordinates")
      if not np.array_equal(velocity, qvel[:, vi]):
        errors.append(
          "robot joint velocities disagree with named full qvel coordinates"
        )
      residual = np.diff(position, axis=0) - dt * velocity[1:]
      if np.max(np.abs(residual)) > thresholds.integration_tolerance_rad:
        errors.append(
          "robot position transition disagrees with implicitfast destination velocity integration"
        )
      free_columns = [
        qpos_names.index(f"usb_plug_freejoint/{suffix}")
        for suffix in ("x", "y", "z", "qw", "qx", "qy", "qz")
      ]
      cached_free_pose = qpos[:, free_columns]
      if pose.shape != (n, 7) or np.any(
        np.abs(np.linalg.norm(pose[:, 3:], axis=1) - 1) > 1e-7
      ):
        raise ValueError(
          "USB cached poses must have finite XYZ and unit WXYZ quaternion"
        )
      if np.any(np.abs(np.linalg.norm(cached_free_pose[:, 3:], axis=1) - 1) > 1e-7):
        raise ValueError("USB freejoint qpos quaternion is not unit length")
      translation_cache_error = np.linalg.norm(
        pose[:, :3] - cached_free_pose[:, :3], axis=1
      )
      rotation_cache_error = _quaternion_distance(pose[:, 3:], cached_free_pose[:, 3:])
      if (
        np.max(translation_cache_error) > thresholds.object_cache_tolerance_m
        or np.max(rotation_cache_error) > thresholds.object_cache_tolerance_rad
      ):
        errors.append(
          "USB object cache does not match same-row freejoint pose at declared post-step epoch"
        )
      free_velocity_columns = [
        qvel_names.index(f"usb_plug_freejoint/{suffix}")
        for suffix in ("vx", "vy", "vz")
      ]
      free_translation_residual = (
        np.diff(qpos[:, free_columns[:3]], axis=0)
        - dt * qvel[1:][:, free_velocity_columns]
      )
      if (
        np.max(np.abs(free_translation_residual)) > thresholds.object_cache_tolerance_m
      ):
        errors.append(
          "USB freejoint translation transition disagrees with destination velocity integration"
        )
      checks["transitions"] = {
        "count": n - 1,
        "state_t_row": "k, 0 <= k < N-1",
        "action_t_source": "/commands/actuator_control[k+1]",
        "action_t_applied_interval": "(state/timestamp[k], state/timestamp[k+1]]",
        "action_logged_at": "state/timestamp[k+1] (same latest time as resulting state)",
        "state_t_plus_1_row": "k+1",
        "initial_control_row_excluded": True,
        "goal_stream_note": "arm/hand targets are requested goals; a post-step contact latch may update goals before the observer, so same-row goals cannot substitute for applied ctrl",
        "joint_position_integration_residual_rad": _peak(
          residual, state_times, joints, 1
        ),
        "usb_translation_integration_residual_m": _peak(
          free_translation_residual, state_times, ["x", "y", "z"], 1
        ),
        "usb_cached_pose_position_error_m": _peak(translation_cache_error, state_times),
        "usb_cached_pose_orientation_error_rad": _peak(
          rotation_cache_error, state_times
        ),
        "command_issue_trace": _noise_indices(file, state_times, errors),
      }
      joint_step = np.diff(position, axis=0)
      joint_rate = joint_step / dt
      acceleration = np.diff(velocity, axis=0) / dt
      # The insertion monitor refreshes FK after each step, so object poses
      # advance with the same-row post-step qpos, including the first step.
      object_step = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1)
      object_rotation = _quaternion_distance(pose[:-1, 3:], pose[1:, 3:])
      if (
        np.max(np.abs(joint_rate)) > thresholds.joint_speed_rad_s
        or np.max(np.abs(velocity)) > thresholds.joint_speed_rad_s
      ):
        errors.append(
          "robot joint speed exceeds configured trajectory corruption screen"
        )
      if np.max(object_step / dt) > thresholds.object_linear_speed_m_s:
        errors.append(
          "USB translation jump exceeds configured trajectory corruption screen"
        )
      if np.max(object_rotation / dt) > thresholds.object_angular_speed_rad_s:
        errors.append(
          "USB orientation jump exceeds configured trajectory corruption screen"
        )
      if np.max(np.abs(acceleration)) > thresholds.joint_acceleration_warning_rad_s2:
        warnings.append(
          "large joint acceleration samples require contextual review; contact can produce physical impulses"
        )
      same_state = np.all(np.diff(qpos, axis=0) == 0, axis=1) & np.all(
        np.diff(qvel, axis=0) == 0, axis=1
      )
      suspicious_state = same_state & np.any(np.abs(velocity[1:]) > 1e-8, axis=1)
      if np.any(suspicious_state):
        errors.append(
          "identical consecutive full states despite nonzero robot velocity"
        )
      checks["trajectory"] = {
        "joint_step_rad": _peak(joint_step, state_times, joints, 1),
        "joint_speed_rad_s": _peak(velocity, state_times, joints),
        "joint_acceleration_rad_s2": _peak(acceleration, state_times, joints, 1),
        "usb_translation_step_m": _peak(object_step, state_times, destination_offset=1),
        "usb_orientation_step_rad": _peak(
          object_rotation, state_times, destination_offset=1
        ),
        "identical_adjacent_full_states": _indices(same_state),
        "suspicious_identical_full_states": _indices(suspicious_state),
        "static_repeated_states_are_allowed": True,
        "quaternion_sign_flips_do_not_count_as_rotation_jumps": True,
      }
      checks["cameras"] = _cameras(
        file, state_ns, solver_ns, thresholds, errors, warnings
      )
      # An episode fingerprint supports detection of accidentally submitted copies.
      fingerprint = hashlib.sha256()
      for array in (state_ns, qpos, qvel, control):
        fingerprint.update(np.ascontiguousarray(array).tobytes())
      report["state_action_fingerprint_sha256"] = fingerprint.hexdigest()
  except (OSError, KeyError, ValueError, TypeError, IndexError) as error:
    errors.append(f"cannot complete cleaning checks: {type(error).__name__}: {error}")
  report["valid"] = not errors
  report["review_required"] = bool(warnings)
  return report


def write_transition_indices(input_path, output_path):
  """Save explicit raw state/action transition rows; source HDF5 is unchanged.

  This is an index sidecar, not a training action tensor.  The action command
  is archived at the *destination* row because observers run after mj_step.
  No observation timestamps are rewritten or moved into the future.
  """
  output_path = Path(output_path)
  if output_path.exists():
    raise FileExistsError(output_path)
  with h5py.File(input_path, "r") as file:
    if (
      _metadata(file, "metadata_json").get("observation_clock")
      != "post_step_forward_v1"
    ):
      raise ValueError(
        "transition sidecar requires the verified post_step_forward_v1 observation clock"
      )
    times = _time_ns(file["state/timestamp"][:], "state")
    if len(times) < 2 or np.any(np.diff(times) != 2_000_000):
      raise ValueError("transition sidecar requires complete 500-Hz state timestamps")
    rows = np.arange(len(times) - 1, dtype=np.int64)
    arrays = {
      "state_t_index": rows,
      "actuator_control_index": rows + 1,
      "state_t_plus_1_index": rows + 1,
      "state_t_timestamp_ns": times[:-1],
      "actuator_control_logged_timestamp_ns": times[1:],
      "actuator_control_interval_start_timestamp_ns": times[:-1],
      "actuator_control_interval_end_timestamp_ns": times[1:],
      "state_t_plus_1_timestamp_ns": times[1:],
    }
  with output_path.open("xb") as stream:
    np.savez_compressed(stream, **arrays)
