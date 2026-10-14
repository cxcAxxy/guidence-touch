"""Offline USB actuator/state consistency, without stepping or rendering physics.

Uses the compiled MJCF's affine actuator equations and forward kinematics to
independently reconstruct actuator forces and camera-epoch site poses from the
explicitly selected recorded state epoch. This checks alignment; it is not a replay
of constrained multibody dynamics.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def affine_actuator_force(
  control, position, velocity, gain, bias, force_range, *, control_range=None
):
  """Stateless fixed-gain, affine-bias scalar joint actuator force."""
  if control_range is not None:
    control = np.clip(control, control_range[:, 0], control_range[:, 1])
  return np.clip(
    control * gain + bias[:, 0] + position * bias[:, 1] + velocity * bias[:, 2],
    force_range[:, 0],
    force_range[:, 1],
  )


def force_alignment(
  control,
  position,
  velocity,
  force,
  gain,
  bias,
  force_range,
  tolerance=1e-8,
  *,
  control_range=None,
  state_epoch="pre",
):
  """Verify the force cache at a declared pre-step or refreshed post-step state.

  Post-step mj_forward can reevaluate force under the held control. That cache
  is not the actuator force used by the preceding integration step.
  """
  if state_epoch not in ("pre", "post"):
    raise ValueError("state_epoch must be 'pre' or 'post'; it is never inferred")
  arrays = [np.asarray(a, dtype=float) for a in (control, position, velocity, force)]
  control, position, velocity, force = arrays
  if (
    control.ndim != 2
    or len(control) < 2
    or control.shape[1] == 0
    or any(a.shape != control.shape or not np.isfinite(a).all() for a in arrays)
  ):
    raise ValueError(
      "control/position/velocity/force require matching finite [N,A] arrays"
    )
  gain, bias, force_range = (
    np.asarray(a, dtype=float) for a in (gain, bias, force_range)
  )
  count = control.shape[1]
  if (
    gain.shape != (count,)
    or bias.shape != (count, 3)
    or not np.isfinite(gain).all()
    or not np.isfinite(bias).all()
  ):
    raise ValueError("gain/bias require finite [A] and [A,3] arrays")
  if (
    force_range.shape != (count, 2)
    or np.isnan(force_range).any()
    or np.any(force_range[:, 0] > force_range[:, 1])
  ):
    raise ValueError("force ranges require ordered [A,2] bounds without NaN")
  if control_range is not None:
    control_range = np.asarray(control_range, dtype=float)
    if (
      control_range.shape != (count, 2)
      or np.isnan(control_range).any()
      or np.any(control_range[:, 0] > control_range[:, 1])
    ):
      raise ValueError("control ranges require ordered [A,2] bounds without NaN")
  if not np.isfinite(tolerance) or tolerance < 0:
    raise ValueError("force tolerance must be finite and nonnegative")
  reference = slice(None, -1) if state_epoch == "pre" else slice(1, None)
  opposite = slice(1, None) if state_epoch == "pre" else slice(None, -1)
  predicted = affine_actuator_force(
    control[1:],
    position[reference],
    velocity[reference],
    gain,
    bias,
    force_range,
    control_range=control_range,
  )
  residual = np.abs(force[1:] - predicted)
  wrong_time = affine_actuator_force(
    control[1:],
    position[opposite],
    velocity[opposite],
    gain,
    bias,
    force_range,
    control_range=control_range,
  )
  indices = np.argwhere(residual > tolerance)
  row, column = np.unravel_index(np.argmax(residual), residual.shape)
  return {
    "valid": bool(not len(indices)),
    "transitions": len(control) - 1,
    "actuators": control.shape[1],
    "state_epoch": state_epoch,
    "force_semantics": (
      "pre-integration actuator evaluation using state[k-1] and ctrl[k]"
      if state_epoch == "pre"
      else "post-integration mj_forward reevaluation using state[k] and held ctrl[k]; integration force was not archived"
    ),
    "force_tolerance": tolerance,
    "maximum_force_residual": float(residual[row, column]),
    "maximum_residual_record_row": int(row + 1),
    "maximum_residual_actuator_index": int(column),
    "mismatch_count": len(indices),
    "first_mismatches_record_row_actuator": (indices[:20] + [1, 0]).tolist(),
    f"incorrect_{'post' if state_epoch == 'pre' else 'pre'}_step_state_maximum_residual": float(
      np.max(np.abs(force[1:] - wrong_time))
    ),
    "initial_row_excluded": "reset forward evaluation; no preceding integration step",
  }


def pose_alignment(recorded, reconstructed, *, tolerance=1e-9):
  """Compare an ordered movie of archived SE(3) poses with independent FK."""
  recorded, reconstructed = (
    np.asarray(a, dtype=float) for a in (recorded, reconstructed)
  )
  if (
    recorded.ndim < 3
    or recorded.shape[-2:] != (4, 4)
    or recorded.size == 0
    or reconstructed.shape != recorded.shape
    or not np.isfinite(recorded).all()
    or not np.isfinite(reconstructed).all()
  ):
    raise ValueError("poses require matching nonempty finite [N,...,4,4] arrays")
  if not np.isfinite(tolerance) or tolerance < 0:
    raise ValueError("pose tolerance must be finite and nonnegative")
  translation_error = np.linalg.norm(
    recorded[..., :3, 3] - reconstructed[..., :3, 3],
    axis=-1,
  )
  rotation_error = np.max(
    np.abs(recorded[..., :3, :3] - reconstructed[..., :3, :3]),
    axis=(-2, -1),
  )
  homogeneous_error = np.max(
    np.abs(recorded[..., 3, :] - reconstructed[..., 3, :]),
    axis=-1,
  )
  mismatch = (
    (translation_error > tolerance)
    | (rotation_error > tolerance)
    | (homogeneous_error > tolerance)
  )
  return {
    "valid": bool(not np.any(mismatch)),
    "frames_checked": len(recorded),
    "poses_checked": int(mismatch.size),
    "translation_tolerance_m": tolerance,
    "rotation_matrix_tolerance": tolerance,
    "maximum_translation_error_m": float(np.max(translation_error)),
    "maximum_rotation_matrix_error": float(np.max(rotation_error)),
    "maximum_homogeneous_row_error": float(np.max(homogeneous_error)),
    "mismatch_count": int(np.count_nonzero(mismatch)),
    "first_mismatches_frame_pose_indices": np.argwhere(mismatch)[:20].tolist(),
  }


def _joint_qpos_labels(name, kind):
  """Normalize NumPy joint scalars before comparing MuJoCo enum values."""
  import mujoco

  kind = int(kind)
  if kind == int(mujoco.mjtJoint.mjJNT_FREE):
    return [f"{name}/{axis}" for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")]
  if kind == int(mujoco.mjtJoint.mjJNT_BALL):
    return [f"{name}/{i}" for i in range(4)]
  if kind in (int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE)):
    return [name]
  raise ValueError("unsupported base-model qpos joint type")


def audit_camera_kinematics(file, model, *, tolerance=1e-9, state_epoch="pre"):
  """Recompute archived camera/site poses without steps, dynamics or rendering.

  The caller must verify the model/controller fingerprints before calling.
  Genesis probe joints are intentionally absent from this static base model;
  matching each named base-model qpos coordinate avoids shifted joint offsets.
  """
  import mujoco

  if state_epoch not in ("pre", "post"):
    raise ValueError("state_epoch must be 'pre' or 'post'; it is never inferred")
  raw_names = file["state/full_qpos_names"].asstr()[:].tolist()
  raw_qpos = file["state/qpos"]
  times = np.asarray(file["state/timestamp"])
  if (
    len(set(raw_names)) != len(raw_names)
    or raw_qpos.ndim != 2
    or raw_qpos.shape != (len(times), len(raw_names))
    or len(times) == 0
    or times.ndim != 1
    or not np.isfinite(times).all()
    or np.any(np.diff(times) <= 0)
  ):
    raise ValueError("FK requires unique qpos names and complete increasing state rows")
  name_to_column = {name: index for index, name in enumerate(raw_names)}
  columns = np.full(model.nq, -1, dtype=int)
  for joint_id in range(model.njnt):
    name = model.joint(joint_id).name
    if not name:
      raise ValueError("FK audit requires named base-model joints")
    labels = _joint_qpos_labels(name, model.jnt_type[joint_id])
    if any(label not in name_to_column for label in labels):
      raise ValueError(f"recording is missing base-model joint {name}")
    address = int(model.jnt_qposadr[joint_id])
    columns[address : address + len(labels)] = [
      name_to_column[label] for label in labels
    ]
  if np.any(columns < 0) or len(set(columns)) != model.nq:
    raise ValueError("base-model qpos coordinates are not covered exactly")

  wrist_names = [f"hand_{side}_base_link_site" for side in ("l", "r")]
  finger_names = [
    [
      f"hand_{side}_{finger}_link{6 if finger == 'thumb' else 4}_site"
      for finger in ("thumb", "index", "middle", "ring", "pinky")
    ]
    for side in ("l", "r")
  ]
  wrist_ids = np.array([model.site(name).id for name in wrist_names])
  finger_ids = np.array(
    [[model.site(name).id for name in names] for names in finger_names]
  )

  def reject_mocap_ancestor(body_id):
    while body_id > 0:
      if model.body_mocapid[body_id] >= 0:
        raise ValueError("FK audit cannot infer unrecorded mocap ancestors")
      body_id = int(model.body_parentid[body_id])

  for body_id in model.site_bodyid[np.r_[wrist_ids, finger_ids.ravel()]]:
    reject_mocap_ancestor(int(body_id))
  data = mujoco.MjData(model)
  reports, errors = {}, []
  configured = json.loads(file["cameras"].attrs["configured_names_json"])
  if (
    not configured
    or len(set(configured)) != len(configured)
    or set(configured) != set(file["cameras"].keys())
  ):
    raise ValueError("FK audit requires exactly the configured recorded cameras")
  for camera in configured:
    frames = file[f"cameras/{camera}"]
    camera_id = model.camera(camera).id
    # COM tracking requires mj_comPos, which this kinematics-only audit avoids.
    if int(model.cam_mode[camera_id]) != int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED):
      raise ValueError("FK audit currently supports body-fixed cameras only")
    reject_mocap_ancestor(int(model.cam_bodyid[camera_id]))
    if (
      json.loads(frames.attrs["wrist_sites_json"]) != wrist_names
      or json.loads(frames.attrs["fingertip_sites_json"]) != finger_names
    ):
      raise ValueError(
        "recorded wrist/fingertip names do not match canonical pose order"
      )
    indices = np.asarray(frames["state_index"])
    capture_times = np.asarray(frames["timestamp"])
    pose_times = np.asarray(frames["pose_timestamp"])
    if (
      indices.ndim != 1
      or len(indices) == 0
      or indices.dtype.kind not in "iu"
      or np.any(indices < 0)
      or np.any(indices >= len(times))
      or capture_times.shape != indices.shape
      or pose_times.shape != indices.shape
      or not np.isfinite(capture_times).all()
      or not np.isfinite(pose_times).all()
    ):
      raise ValueError(
        "camera FK requires finite camera clocks and valid integer state indices"
      )
    source_indices = np.maximum(
      indices.astype(np.int64) - (1 if state_epoch == "pre" else 0),
      0,
    )
    capture_error = float(np.max(np.abs(capture_times - times[indices])))
    pose_time_error = float(np.max(np.abs(pose_times - times[source_indices])))
    if capture_error > tolerance or pose_time_error > tolerance:
      errors.append(
        f"{camera}: recorded camera epochs do not match indexed state clocks"
      )
    count = len(indices)
    expected = {
      "world_from_wrist": np.broadcast_to(np.eye(4), (count, 2, 4, 4)).copy(),
      "world_from_fingertip": np.broadcast_to(np.eye(4), (count, 2, 5, 4, 4)).copy(),
      "world_from_camera": np.broadcast_to(np.eye(4), (count, 4, 4)).copy(),
    }
    for frame, state_index in enumerate(source_indices):
      source = np.asarray(raw_qpos[int(state_index)])[columns]
      if not np.isfinite(source).all():
        raise ValueError("camera FK source state contains nonfinite qpos")
      data.qpos[:] = source
      mujoco.mj_kinematics(model, data)
      mujoco.mj_camlight(model, data)
      for name, ids in (
        ("world_from_wrist", wrist_ids),
        ("world_from_fingertip", finger_ids),
      ):
        expected[name][frame, ..., :3, :3] = data.site_xmat[ids].reshape(
          *ids.shape, 3, 3
        )
        expected[name][frame, ..., :3, 3] = data.site_xpos[ids]
      expected["world_from_camera"][frame, :3, :3] = data.cam_xmat[camera_id].reshape(
        3, 3
      )
      expected["world_from_camera"][frame, :3, 3] = data.cam_xpos[camera_id]
    pose_reports = {
      name: pose_alignment(np.asarray(frames[name]), poses, tolerance=tolerance)
      for name, poses in expected.items()
    }
    for name, report in pose_reports.items():
      if not report["valid"]:
        errors.append(
          f"{camera}/{name}: archived pose differs from declared {state_epoch}-step FK"
        )
    reports[camera] = {
      "frames_checked": count,
      "maximum_capture_clock_error_s": capture_error,
      "maximum_pose_clock_error_s": pose_time_error,
      "poses": pose_reports,
    }
  all_poses = [r for camera in reports.values() for r in camera["poses"].values()]
  return {
    "valid": not errors,
    "errors": errors,
    "cameras": reports,
    "frames_checked": sum(camera["frames_checked"] for camera in reports.values()),
    "poses_checked": sum(report["poses_checked"] for report in all_poses),
    "maximum_translation_error_m": max(
      r["maximum_translation_error_m"] for r in all_poses
    ),
    "maximum_rotation_matrix_error": max(
      r["maximum_rotation_matrix_error"] for r in all_poses
    ),
    "state_epoch": state_epoch,
    "state_mapping": (
      "named base-model qpos from max(camera.state_index - 1, 0)"
      if state_epoch == "pre"
      else "named base-model qpos from camera.state_index"
    ),
    "operations": ["mj_kinematics", "mj_camlight"],
    "limitations": "Verifies archived camera/site poses from source joint state; does not rerender RGB pixels or run dynamics. Supports body-fixed cameras without mocap ancestors.",
  }


def arm_slew_alignment(control, initial_position, timestamps, *, speed_limit=3.1416):
  """Check real arm commands against the reset arm command, not placeholder ctrl[0]."""
  control, initial_position, timestamps = (
    np.asarray(a, dtype=float) for a in (control, initial_position, timestamps)
  )
  if (
    control.ndim != 2
    or len(control) < 2
    or control.shape[1] == 0
    or initial_position.shape != control.shape[1:]
    or timestamps.shape != (len(control),)
    or not all(np.isfinite(a).all() for a in (control, initial_position, timestamps))
    or np.any(np.diff(timestamps) <= 0)
    or not np.isfinite(speed_limit)
    or speed_limit <= 0
  ):
    raise ValueError(
      "arm slew requires finite command/reset positions and increasing state times"
    )
  # Verified USB reset sets _arm_command to measured home joint positions.
  # The raw initial data.ctrl row can remain zero until the first servo step.
  previous = np.concatenate((initial_position[None, :], control[1:-1]), axis=0)
  delta = np.abs(control[1:] - previous)
  dt = np.diff(timestamps)
  excess = float(np.max(delta - speed_limit * dt[:, None]))
  return {
    "valid": excess <= 1e-9,
    "transitions": len(control) - 1,
    "maximum_slew_rad_s": float(np.max(delta / dt[:, None])),
    "slew_limit_rad_s": speed_limit,
    "maximum_delta_excess_rad": excess,
    "initial_control_row_excluded": "reset placeholder; first real control is compared with initial measured arm qpos",
  }


def audit_usb_actions(path):
  # The one verified model serves affine force equations and kinematics only.
  import mujoco

  from .tict_source_audit import _known_usb_source_paths, _mjcf_source_fingerprint

  path = Path(path).resolve()
  errors = []
  cache_state_epoch = "post"
  source_paths, scene = _known_usb_source_paths()
  import hashlib

  with h5py.File(path, "r") as f:
    metadata = json.loads(f.attrs["metadata_json"])
    if metadata.get("observation_clock") != "post_step_forward_v1":
      errors.append(
        "missing verified post_step_forward_v1 observation clock; legacy cache labels are unsupported"
      )
    expected_sources = metadata.get("controller_source_sha256", {})
    current_sources = {
      n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in source_paths.items()
    }
    if expected_sources != current_sources:
      errors.append("controller source drift: cannot assume current servo equations")
    if metadata.get("base_model_fingerprint") != _mjcf_source_fingerprint(scene):
      errors.append("MJCF source drift: current actuator parameters are not verified")
    if errors:
      return {"valid": False, "source": str(path), "errors": errors}
    model = mujoco.MjModel.from_xml_path(str(scene))
    names = f["commands/actuator_names"].asstr()[:].tolist()
    ids = np.array([model.actuator(n).id for n in names])
    if len(ids) != model.nu or len(set(ids)) != model.nu:
      raise ValueError("recorded actuator names do not cover the model exactly")
    if not (
      np.all(model.actuator_trntype[ids] == mujoco.mjtTrn.mjTRN_JOINT)
      and np.all(model.actuator_gaintype[ids] == mujoco.mjtGain.mjGAIN_FIXED)
      and np.all(model.actuator_biastype[ids] == mujoco.mjtBias.mjBIAS_AFFINE)
      and np.all(model.actuator_dyntype[ids] == mujoco.mjtDyn.mjDYN_NONE)
    ):
      raise ValueError("unsupported actuator equation; audit must not guess")
    joint_ids = model.actuator_trnid[ids, 0]
    if not np.all(
      np.isin(
        model.jnt_type[joint_ids],
        [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE],
      )
    ):
      raise ValueError("only scalar joint transmissions are supported")
    joint_names = [model.joint(int(i)).name for i in joint_ids]
    qp_names = f["state/full_qpos_names"].asstr()[:].tolist()
    qv_names = f["state/full_qvel_names"].asstr()[:].tolist()
    qp = np.asarray(f["state/qpos"])[:, [qp_names.index(n) for n in joint_names]]
    qv = np.asarray(f["state/qvel"])[:, [qv_names.index(n) for n in joint_names]]
    ctrl = np.asarray(f["commands/actuator_control"])
    gear = model.actuator_gear[ids, 0]
    ranges = model.actuator_forcerange[ids].copy()
    ranges[model.actuator_forcelimited[ids] == 0] = [-np.inf, np.inf]
    control_ranges = model.actuator_ctrlrange[ids].copy()
    control_ranges[model.actuator_ctrllimited[ids] == 0] = [-np.inf, np.inf]
    if model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL:
      control_ranges[:] = [-np.inf, np.inf]
    if model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_ACTUATION:
      raise ValueError("disabled actuation is outside this servo audit contract")
    gain, bias = model.actuator_gainprm[ids, 0], model.actuator_biasprm[ids, :3]
    force = force_alignment(
      ctrl,
      qp * gear,
      qv * gear,
      np.asarray(f["physics/actuator_force"]),
      gain,
      bias,
      ranges,
      control_range=control_ranges,
      state_epoch=cache_state_epoch,
    )
    if not force["valid"]:
      errors.append(
        f"actuator force cache does not match held control and declared {cache_state_epoch}-step state"
      )
    force["maximum_residual_actuator"] = names[force["maximum_residual_actuator_index"]]
    control_violations = (ctrl < control_ranges[:, 0] - 1e-9) | (
      ctrl > control_ranges[:, 1] + 1e-9
    )
    if np.any(control_violations):
      errors.append("recorded servo command exceeds actuator control limits")
    # Hand target is unchanged by the post-step tactile latch (only arm IK changes).
    hand_names = f["commands/hand_joint_names"].asstr()[:].tolist()
    hand_cols = [names.index(n) for n in hand_names]
    hand_ids = ids[hand_cols]
    hand_target = np.asarray(f["commands/hand_joint_target"])
    if (
      not hand_cols
      or len(set(hand_cols)) != len(hand_cols)
      or hand_target.shape != (len(ctrl), len(hand_cols))
      or not np.isfinite(hand_target).all()
    ):
      raise ValueError(
        "hand targets require unique joints and matching finite [N,H] rows"
      )
    hand_expected = np.clip(
      24.0 * (hand_target[1:] - qp[:-1, hand_cols]),
      model.actuator_ctrlrange[hand_ids, 0],
      model.actuator_ctrlrange[hand_ids, 1],
    )
    hand_residual = np.abs(ctrl[1:, hand_cols] - hand_expected)
    hand_max = float(np.max(hand_residual))
    if hand_max > 1e-9:
      errors.append(
        "hand velocity command does not match target and preceding position"
      )
    arm_names = f["commands/arm_joint_names"].asstr()[:].tolist()
    arm_cols = [names.index(n) for n in arm_names]
    timestamps = np.asarray(f["state/timestamp"])
    if (
      not arm_cols
      or len(set(arm_cols)) != len(arm_cols)
      or timestamps.shape != (len(ctrl),)
      or not np.isfinite(timestamps).all()
      or np.any(np.diff(timestamps) <= 0)
    ):
      raise ValueError(
        "arm joints must be unique and state timestamps finite and increasing"
      )
    arm_slew = arm_slew_alignment(ctrl[:, arm_cols], qp[0, arm_cols], timestamps)
    if not arm_slew["valid"]:
      errors.append("arm actuator command exceeds recorded controller slew limit")
    camera_kinematics = audit_camera_kinematics(f, model, state_epoch=cache_state_epoch)
    errors.extend(camera_kinematics["errors"])
    return {
      "schema_version": "usb_actuator_alignment_v1",
      "source": str(path),
      "valid": not errors,
      "errors": errors,
      "actuator_force": force,
      "hand_servo_maximum_residual_rad_s": hand_max,
      "control_limit_violation_count": int(np.count_nonzero(control_violations)),
      "arm_servo_maximum_slew_rad_s": arm_slew["maximum_slew_rad_s"],
      "arm_servo_slew_limit_rad_s": 3.1416,
      "arm_servo_slew": arm_slew,
      "source_identity_matches": True,
      "observation_clock": metadata["observation_clock"],
      "cache_state_epoch": cache_state_epoch,
      "camera_kinematics": camera_kinematics,
      "equation": (
        "clip(gain*clip(ctrl[k], enabled_control_range) + b0 + b1*gear*qpos[s] + b2*gear*qvel[s], force_range); "
        + (
          "s=k-1"
          if cache_state_epoch == "pre"
          else "s=k (refreshed force cache, not integration force)"
        )
      ),
      "transition": "state[k-1] -> commands/actuator_control[k] -> state[k]",
      "time_semantics": "action availability at the preceding state; logged at the resulting state; neither command targets nor future pose labels are measured actuator force",
      "limitations": "Checks servo equations, force response, cached pose FK and indices; does not replay constrained multibody dynamics or rerender images. Use alongside trajectory integration and task-outcome checks.",
    }
