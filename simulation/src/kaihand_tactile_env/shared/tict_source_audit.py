"""Independent, read-only cross-check of a T-ICT release against its HDF5 source.

No simulator, FK recomputation, renderer, exporter helper, or full taxel array is
loaded. Checks use recorded physical observations; actuator force-limit checks
cover recorded samples, not unrecorded intermediate 500-Hz integration steps.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "little")
FORCE_CAP_TOLERANCE_N = 1e-5
RANDOMIZED_PRESET = "middle-force-randomized-v1"
PRECONTACT_PRESET = "middle-force-precontact-v1"
CARD_RANDOMIZED_PRESETS = (RANDOMIZED_PRESET, PRECONTACT_PRESET)


def _text(value):
  return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _sha256(path):
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    while block := stream.read(1024 * 1024):
      digest.update(block)
  return digest.hexdigest()


def _json(path):
  value = json.loads(Path(path).read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: JSON must contain an object")
  return value


def _attr_json(group, name):
  value = json.loads(_text(group.attrs[name]))
  if not isinstance(value, dict):
    raise ValueError(f"{name}: JSON must contain an object")
  return value


def _ns(values, label, strict=True):
  values = np.asarray(values, dtype=np.float64)
  if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
    raise ValueError(f"{label}: timestamps must be a nonempty finite vector")
  if np.any(values < 0) or np.any(values >= np.iinfo(np.int64).max / 1e9):
    raise ValueError(f"{label}: timestamps outside nonnegative int64 range")
  result = np.rint(values * 1_000_000_000).astype(np.int64)
  differences = np.diff(result)
  if np.any(differences <= 0 if strict else differences < 0):
    raise ValueError(f"{label}: timestamps out of order")
  return result


def _audit_camera_state_clock(
  state_times, force_times, capture_times, render_times, camera_indices, errors
):
  """Check state links without assuming RGB and state have identical rates.

  A 30 Hz image can fall between 100 Hz state/force samples. Its stored FK
  still describes the exact image epoch; tactile is independently matched to
  the latest nonfuture force below. Requiring equal epochs here incorrectly
  rejects valid 30 Hz recordings that have a 0--8 ms tactile age.
  All inputs are validated integer nanosecond clocks.
  """
  expected = np.searchsorted(state_times, capture_times, side="right") - 1
  if not np.array_equal(camera_indices, expected):
    errors.append("camera state_index must identify the latest nonfuture state")
  if np.any(force_times[camera_indices] > render_times):
    errors.append("indexed state solver epoch is later than the rendered image")


def _se3(value, label):
  value = np.asarray(value, dtype=np.float64)
  if value.shape[-2:] != (4, 4) or not np.isfinite(value).all():
    raise ValueError(f"{label}: expected finite SE(3)")
  rotation = value[..., :3, :3]
  if not (
    np.allclose(value[..., 3, :], [0, 0, 0, 1], atol=1e-7, rtol=0)
    and np.allclose(
      np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=1e-6, rtol=0
    )
    and np.allclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)
  ):
    raise ValueError(f"{label}: invalid rigid transform")
  return value


def _card_pose_transform(value, object_name="card"):
  """Independent WXYZ-to-matrix check for the release's per-session anchor."""
  pose = np.asarray(value, dtype=np.float64)
  if pose.shape != (7,) or not np.isfinite(pose).all():
    raise ValueError(
      f"initial {object_name} pose must contain seven finite XYZ/WXYZ values"
    )
  if not np.isclose(np.linalg.norm(pose[3:]), 1, atol=1e-9, rtol=0):
    raise ValueError(f"initial {object_name} quaternion must be unit length")
  w, x, y, z = pose[3:]
  result = np.eye(4)
  result[:3, :3] = [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ]
  result[:3, 3] = pose[:3]
  return _se3(result, f"initial {object_name} anchor")


def _known_usb_source_paths():
  shared = Path(__file__).resolve().parent
  repository = shared.parents[2]
  names = ["scripts/workcell/record_usb_dataset.py"]
  names += [
    f"src/kaihand_tactile_env/tasks/usb_insert/{name}.py"
    for name in (
      "config",
      "setup",
      "grasp",
      "execution",
      "motion",
      "task",
      "precontact_noise",
      "recording",
      "buffered_h5",
      "batch",
    )
  ]
  names += [
    f"src/kaihand_tactile_env/shared/{name}.py"
    for name in (
      "simulation",
      "recording",
      "taskspace_recording",
      "contact_tactile",
      "tactile",
      "config",
      "cameras",
      "rendering",
    )
  ]
  return {name: repository / name for name in names}, (
    shared.parent / "tasks/usb_insert/scene.xml"
  )


def _mjcf_source_fingerprint(scene):
  """Independently hash the recorder's versioned recursive MJCF byte contract."""
  scene = Path(scene).resolve()
  digest = hashlib.sha256(b"kaihand-mjcf-includes-v1\0")
  seen, active = set(), set()

  def visit(path):
    path = path.resolve()
    if path in active:
      raise ValueError("cyclic MJCF include in USB base scene")
    if path in seen:
      return
    active.add(path)
    contents = path.read_bytes()
    digest.update(os.path.relpath(path, scene.parent).encode("utf-8") + b"\0")
    digest.update(len(contents).to_bytes(8, "big"))
    digest.update(contents)
    for include in ET.fromstring(contents).iter("include"):
      visit(path.parent / include.attrib["file"])
    active.remove(path)
    seen.add(path)

  visit(scene)
  return digest.hexdigest()


def _usb_physics_and_identity(file, metadata, outcome, errors):
  """USB gates use its own recorder/controller provenance, never poker settings."""
  from .usb_cleaning import audit_usb_insertion_physics

  def check(condition, message):
    if not condition:
      errors.append(message)

  check(metadata.get("scene") == "usb-insert", "source scene is not usb-insert")
  check(
    metadata.get("recording_contract") == "usb_insert_taskspace_raw_v1",
    "unknown USB raw recording contract",
  )
  check(
    metadata.get("observation_clock") == "post_step_forward_v1",
    "USB observation_clock must be post_step_forward_v1; legacy pre-step labels are rejected",
  )
  check(outcome.get("object_name") == "usb_plug", "USB outcome must name usb_plug")
  for name in ("success", "released", "grasp_verified", "source_files_unchanged"):
    check(outcome.get(name) is True, f"USB source gate failed: {name}")
  insertion = outcome.get("insertion", {})
  for name in ("success", "seated"):
    check(insertion.get(name) is True, f"USB insertion gate failed: {name}")
  stable_duration = insertion.get("stable_duration_s")
  check(
    isinstance(stable_duration, (int, float))
    and np.isfinite(stable_duration)
    and stable_duration >= 0.1 - 1e-9,
    "USB seating dwell is not at least 0.1 seconds",
  )
  check(
    metadata.get("motion_profile") in {"baseline", "fast"}
    and metadata.get("motion_profile") == outcome.get("motion_profile"),
    "USB motion profile provenance mismatch",
  )
  for name in ("physics_hz", "control_hz"):
    check(file.attrs.get(name) == 500, f"USB {name} must record every 500-Hz step")
  check(
    _text(file.attrs.get("contact_force_source", ""))
    == "solver_contact_distributed_taxel_v1",
    "USB tactile source must be the signed solver contact stream",
  )
  force = file["tactile_contact_force"]
  force_metadata = _attr_json(force, "metadata_json")
  targets = force_metadata.get("target_geom_names", [])
  check(
    isinstance(targets, list)
    and bool(targets)
    and all(isinstance(name, str) and name.startswith("usb_plug_") for name in targets),
    "USB tactile target set must contain only USB plug geometries",
  )
  check(
    _text(force.attrs.get("force_unit", "")) in {"N", "newton"},
    "USB force unit is not newtons",
  )
  paths, scene = _known_usb_source_paths()
  recorded = metadata.get("controller_source_sha256", {})
  check(
    isinstance(recorded, dict) and set(recorded) == set(paths),
    "USB controller source hash list incomplete or unexpected",
  )
  check(
    outcome.get("controller_source_sha256_at_end") == recorded,
    "USB controller sources changed during capture",
  )
  hashes, drift = {}, []
  for name, path in paths.items():
    current = _sha256(path)
    matched = current == recorded.get(name)
    hashes[name] = {
      "recorded": recorded.get(name),
      "current": current,
      "matches": matched,
    }
    if not matched:
      drift.append(name)
      errors.append(f"current-code drift: USB recorded source differs: {name}")
  scene_hash = _sha256(scene)
  check(metadata.get("base_model_sha256") == scene_hash, "USB base scene SHA mismatch")
  fingerprint = _mjcf_source_fingerprint(scene)
  check(
    metadata.get("base_model_fingerprint") == fingerprint,
    "USB base scene/include fingerprint mismatch",
  )
  for name in ("model_sha256", "model_fingerprint"):
    check(
      bool(re.fullmatch(r"[0-9a-f]{64}", _text(file.attrs.get(name, "")))),
      f"USB missing or invalid source {name}",
    )
  insertion_physics = audit_usb_insertion_physics(file, metadata, outcome)
  errors.extend(insertion_physics["errors"])
  return {
    "scene": "usb-insert",
    "object_name": "usb_plug",
    "motion_profile": metadata.get("motion_profile"),
    "released": outcome.get("released"),
    "grasp_verified": outcome.get("grasp_verified"),
    "stable_seating_duration_s": stable_duration,
    "insertion_physics": insertion_physics,
    "base_scene_sha256": scene_hash,
    "base_model_fingerprint": fingerprint,
    "controller_sources": hashes,
    "controller_source_status": "current_code_drift"
    if drift
    else "matches_current_code",
    "drift_source_names": drift,
    "source_identity_scope": "strict comparison with current code; drift alone does not establish malformed recorded bytes",
    "model_verification_scope": "known scene/controller bytes and recorded limits; compiled model and FK are not reconstructed",
  }


def _usb_recorded_motion(file, state_times, outcome, errors):
  """Check raw every-step joints, contacts and terminal velocity without simulation."""
  n = len(state_times)
  phases = np.asarray([_text(value) for value in file["commands/phase"][:]])
  if (
    phases.shape != (n,)
    or state_times[0] != 0
    or not np.allclose(np.diff(state_times), 2_000_000, atol=1, rtol=0)
  ):
    raise ValueError(
      "USB state/phase stream must include reset and every 2-ms physics step"
    )
  arm_names = [_text(value) for value in file["model/arm_joint_names"][:]]
  expected_names = [
    f"{side}_arm_joint{i}" for side in ("left", "right") for i in range(1, 8)
  ]
  command_names = [_text(value) for value in file["commands/arm_joint_names"][:]]
  if arm_names != expected_names or command_names != expected_names:
    raise ValueError(
      "USB recorded arm/command order must be left seven then right seven"
    )
  limits = np.asarray(file["model/arm_joint_limits_rad"][:], dtype=float)
  indices = np.asarray(file["model/arm_joint_qpos_indices"][:])
  qpos = file["state/qpos"]
  goals = file["commands/arm_joint_target"]
  qpos_names = [_text(value) for value in file["state/full_qpos_names"][:]]
  if (
    limits.shape != (14, 2)
    or not np.isfinite(limits).all()
    or np.any(limits[:, 1] <= limits[:, 0])
    or indices.shape != (14,)
    or indices.dtype.kind not in "iu"
    or len(set(indices.tolist())) != 14
    or qpos.ndim != 2
    or qpos.shape[0] != n
    or np.any(indices < 0)
    or np.any(indices >= qpos.shape[1])
    or goals.shape != (n, 14)
  ):
    raise ValueError(
      "USB recorded joint limits, qpos mapping or arm goal shapes are invalid"
    )
  if (
    len(qpos_names) != qpos.shape[1]
    or [qpos_names[index] for index in indices] != arm_names
  ):
    raise ValueError("USB arm qpos indices disagree with the full state joint names")
  minimum_actual = np.full(7, np.inf)
  minimum_goal = np.full(7, np.inf)
  required_margin = np.full(n, 15.0)
  parameters = outcome.get("motion_parameters", {})
  if (
    parameters.get("align_clearance_m") == 0.10
    and parameters.get("align_hold_s") == 0.6
    and "usb_insertion/insertion_depth_m" in file
  ):
    depth = np.asarray(file["usb_insertion/insertion_depth_m"][:], dtype=float)
    if depth.shape != (n,) or not np.isfinite(depth).all():
      raise ValueError("USB hover depth must be finite and state-aligned")
    # Match the verified 100 mm controller's high-hover posture. All other
    # phases, legacy profiles and the final insertion keep the 15-degree gate.
    required_margin[(phases == "align") & (depth < -0.03)] = 10.0
  margin_failures = set()
  for start in range(0, n, 1024):
    raw = qpos[start : start + 1024]
    goal = goals[start : start + 1024]
    if not np.isfinite(raw).all() or not np.isfinite(goal).all():
      raise ValueError("USB raw qpos or arm goals contain nonfinite values")
    actual = raw[:, indices[7:]]
    for name, values, result in (
      ("actual", actual, minimum_actual),
      ("goal", goal[:, 7:], minimum_goal),
    ):
      margins = np.rad2deg(np.minimum(values - limits[7:, 0], limits[7:, 1] - values))
      np.minimum(result, margins.min(axis=0), out=result)
      required = required_margin[start : start + len(raw), None]
      if np.any(margins < required - 1e-6):
        margin_failures.add(name)
  for name in sorted(margin_failures):
    errors.append(
      f"USB right arm {name} joint margin below phase-specific limit (15 degrees; 10 only during declared 100 mm high alignment)"
    )
  physics = file["physics"]
  qvel_names = [_text(value) for value in file["state/full_qvel_names"][:]]
  if len(set(qvel_names)) != len(qvel_names) or not set(arm_names).issubset(qvel_names):
    raise ValueError(
      "USB full state velocity names must identify fourteen unique arm DOFs"
    )
  arm_dofs = [qvel_names.index(name) for name in arm_names]
  object_dof_names = [
    f"usb_plug_freejoint/{name}" for name in ("vx", "vy", "vz", "wx", "wy", "wz")
  ]
  if len(set(qvel_names)) != len(qvel_names) or not set(object_dof_names).issubset(
    qvel_names
  ):
    raise ValueError("USB full qvel names must identify all six plug freejoint DOFs")
  non_arm_dofs = np.ones(len(qvel_names), dtype=bool)
  non_arm_dofs[arm_dofs] = False
  for name in ("qfrc_applied", "xfrc_applied"):
    stream = physics[name]
    if stream.shape[0] != n or (
      name == "qfrc_applied" and stream.shape != (n, len(qvel_names))
    ):
      raise ValueError(f"USB {name} must align with every physics step")
    for start in range(0, n, 1024):
      values = stream[start : start + 1024]
      # Arm gravity/Coriolis bias feedforward is part of the original servo.
      # All other DOFs, including the six plug DOFs, must remain unforced.
      unassisted = values[:, non_arm_dofs] if name == "qfrc_applied" else values
      if not np.isfinite(values).all() or np.any(unassisted != 0):
        errors.append(
          f"USB raw {name} contains object force assistance or nonfinite values"
        )
        break

  contacts = file["contacts"]
  starts, counts = contacts["frame_start"][:], contacts["frame_count"][:]
  events = contacts["events"]
  if (
    starts.shape != (n,)
    or counts.shape != (n,)
    or starts.dtype.kind not in "iu"
    or counts.dtype.kind not in "iu"
    or np.any(counts < 0)
    or not np.array_equal(starts, np.r_[0, np.cumsum(counts[:-1], dtype=np.int64)])
  ):
    raise ValueError(
      "USB contact event ranges do not cover every recorded physics step"
    )
  event_count = int(np.sum(counts, dtype=np.int64))
  event_indices = np.asarray(events["state_index"][:])
  if not np.array_equal(event_indices, np.repeat(np.arange(n), counts)):
    raise ValueError("USB contact event state indices do not match their frame ranges")
  bodies = [_text(value) for value in file["model/body_names"][:]]
  first, second = events["body1_id"][:], events["body2_id"][:]
  if any(
    value.shape != (event_count,)
    or value.dtype.kind not in "iu"
    or np.any(value < 0)
    or np.any(value >= len(bodies))
    for value in (first, second)
  ):
    raise ValueError("USB contact event body identifiers are invalid")
  robot = np.array(
    [
      name.startswith(("right_arm_", "left_arm_", "hand_r_", "hand_l_"))
      or name == "torso"
      for name in bodies
    ]
  )
  obstacle = np.array(
    [
      name in {"table", "usb_fixture", "usb_socket"} or name.startswith("usb_socket_")
      for name in bodies
    ]
  )
  forbidden = (robot[first] & obstacle[second]) | (robot[second] & obstacle[first])
  if forbidden.any():
    errors.append(
      "USB raw contact stream contains robot/table, fixture or socket contacts"
    )

  poses = np.asarray(file["objects/usb_plug/pose_wxyz"][:], dtype=float)
  twists = np.asarray(file["objects/usb_plug/twist_linear_angular"][:], dtype=float)
  if (
    poses.shape != (n, 7)
    or twists.shape != (n, 6)
    or not np.isfinite(poses).all()
    or not np.isfinite(twists).all()
    or not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1, atol=1e-7, rtol=0)
  ):
    raise ValueError(
      "USB object pose/twist must be finite, normalized and aligned to state"
    )
  for name, actual in (
    ("final_object_pose", poses[-1]),
    ("final_object_twist", twists[-1]),
  ):
    recorded = np.asarray(outcome.get(name, []), dtype=float)
    if recorded.shape != actual.shape or not np.allclose(
      recorded, actual, atol=1e-9, rtol=0
    ):
      errors.append(f"USB outcome {name} differs from final raw state")
  terminal = state_times >= state_times[-1] - 100_000_000
  speeds = np.column_stack(
    (np.linalg.norm(twists[:, :3], axis=1), np.linalg.norm(twists[:, 3:], axis=1))
  )
  if terminal.sum() < 50 or np.any(speeds[terminal] >= [0.02, 0.2]):
    errors.append("USB final 0.1-second raw velocity window is not stationary")
  stable = outcome.get("terminal_stability", {})
  for name, expected in (
    ("linear_speed", speeds[-1, 0]),
    ("angular_speed", speeds[-1, 1]),
  ):
    value = stable.get(name)
    if (
      not isinstance(value, (float, int))
      or not np.isfinite(value)
      or not np.isclose(value, expected, atol=1e-9, rtol=0)
    ):
      errors.append(f"USB terminal_stability {name} differs from recorded velocity")
  duration = stable.get("stable_seconds")
  if (
    not isinstance(duration, (float, int))
    or not np.isfinite(duration)
    or duration < 0.1 - 1e-9
  ):
    errors.append("USB terminal stability metadata does not establish 0.1-second dwell")
  initialization = _attr_json(file, "metadata_json").get(
    "initial_pose_randomization", {}
  )
  declared_initial = np.asarray(
    initialization.get("initial_pose_wxyz", []), dtype=float
  )
  if declared_initial.shape != (7,) or not np.allclose(
    declared_initial, poses[0], atol=1e-9, rtol=0
  ):
    errors.append(
      "USB initial pose provenance differs from exact time-zero object pose"
    )
  return phases, {
    "recorded_samples": n,
    "all_physics_steps_recorded": True,
    "minimum_actual_joint_margin_deg": minimum_actual.tolist(),
    "minimum_goal_joint_margin_deg": minimum_goal.tolist(),
    "joint_margin_policy": {
      "default_minimum_deg": 15.0,
      "high_alignment_minimum_deg": 10.0,
      "high_alignment_samples": int(np.count_nonzero(required_margin == 10.0)),
      "scope": "10 degrees only for align above 30 mm in the declared 100 mm / 0.6 s hover profile; all other samples retain 15 degrees",
    },
    "robot_obstacle_contact_events": int(forbidden.sum()),
    "terminal_velocity_samples": int(terminal.sum()),
    "terminal_linear_speed_m_s": float(speeds[-1, 0]),
    "terminal_angular_speed_rad_s": float(speeds[-1, 1]),
    "external_force_scope": "zero non-arm generalized force (including all six plug DOFs) and zero body external wrench; existing fourteen-arm-DOF gravity/Coriolis feedforward is allowed without comparing different cached epochs",
    "sampling_scope": "every recorded 2-ms step; actual qpos, arm goals, raw contact events and terminal velocity",
    "limitations": "seating uses the recorded task monitor outcome; palm/elbow FK and compiled collision geometry are not reconstructed",
  }


def _known_source_paths(preset="middle-force-v1"):
  shared = Path(__file__).resolve().parent
  task = shared.parent / "tasks/poker_draw"
  repository = shared.parents[2]
  paths = {
    "record_dataset.py": repository / "scripts/workcell/record_dataset.py",
    **{
      f"poker_draw/{name}": task / name
      for name in (
        "mid_full.py",
        "pressure_window.py",
        "friction.py",
        "task.py",
        "config.py",
        "press_control.py",
        "acceptance.py",
      )
    },
    **{
      f"shared/{name}": shared / name
      for name in (
        "recording.py",
        "taskspace_recording.py",
        "simulation.py",
        "config.py",
        "tactile.py",
        "contact_tactile.py",
      )
    },
  }
  if preset in CARD_RANDOMIZED_PRESETS:
    paths["poker_draw/randomization.py"] = task / "randomization.py"
  if preset == PRECONTACT_PRESET:
    paths["poker_draw/precontact_noise.py"] = task / "precontact_noise.py"
  return paths, task / "scene.xml"


def _xml_pose(element):
  if element is None or any(
    key in element.attrib for key in ("euler", "axisangle", "xyaxes", "zaxis")
  ):
    raise ValueError("card randomization audit needs the known direct XYZ/WXYZ scene")
  pose = np.fromstring(
    element.get("pos", "0 0 0") + " " + element.get("quat", "1 0 0 0"), sep=" "
  )
  _card_pose_transform(pose)
  return pose


def _audit_initial_randomization(file, metadata, scene):
  """Reproduce initial sampling independently of the capture/simulation helper."""
  variation = metadata.get("initial_card_randomization")
  if metadata.get("preset") not in CARD_RANDOMIZED_PRESETS:
    if variation is not None:
      raise ValueError(
        "zero middle preset must not contain card randomization metadata"
      )
    return {"enabled": False, "scope": "original zero-jitter preset"}
  if not isinstance(variation, dict):
    raise ValueError("randomized preset lacks initial_card_randomization metadata")
  for key, expected in (
    ("schema_version", "poker-initial-card-randomization-v1"),
    ("mode", "seeded_uniform"),
    ("distribution", "independent_uniform"),
    ("perturbation_scope", "reset_only_card_xy_yaw"),
  ):
    if variation.get(key) != expected:
      raise ValueError(f"card randomization mismatch: {key}")
  for key in ("height_unchanged", "tilt_unchanged"):
    if variation.get(key) is not True:
      raise ValueError(f"card randomization must explicitly preserve {key}")
  for key in ("observation_noise", "action_noise"):
    if key not in variation or variation[key] is not None:
      raise ValueError(f"card randomization must disable {key}")
  seed = variation.get("seed")
  if (
    isinstance(seed, bool)
    or not isinstance(seed, int)
    or seed < 0
    or seed != metadata.get("seed")
  ):
    raise ValueError("card randomization seed must match the nonnegative episode seed")
  settings = metadata["preset_settings"]
  if settings.get("robot_initial_state_noise", "missing") is not None:
    raise ValueError("randomized card preset must disable robot initial state noise")
  bounds = []
  for key, root_key, settings_key, maximum in (
    ("xy_jitter_m", "object_xy_jitter", "object_xy_jitter_m", 0.005),
    ("yaw_jitter_rad", "object_yaw_jitter", "object_yaw_jitter_rad", np.pi / 180),
  ):
    value = variation.get(key)
    if (
      isinstance(value, bool)
      or not isinstance(value, (float, int))
      or not np.isfinite(value)
      or not 0 <= value <= maximum
      or value != metadata.get(root_key)
      or value != settings.get(settings_key)
    ):
      raise ValueError(f"card randomization bounds mismatch/out of scope: {key}")
    bounds.append(value)
  rng = np.random.default_rng(seed)
  xy = rng.uniform(-bounds[0], bounds[0], size=2) if bounds[0] else np.zeros(2)
  yaw = float(rng.uniform(-bounds[1], bounds[1])) if bounds[1] else 0.0
  offset = np.asarray(variation.get("sampled_offset_xy_m"), dtype=float)
  if offset.shape != (2,) or not np.allclose(offset, xy, atol=1e-12, rtol=0):
    raise ValueError("card randomization XY offsets do not reproduce from episode seed")
  recorded_yaw = variation.get("sampled_yaw_offset_rad")
  if not isinstance(recorded_yaw, (int, float)) or not np.isclose(
    recorded_yaw, yaw, atol=1e-12, rtol=0
  ):
    raise ValueError("card randomization yaw does not reproduce from episode seed")

  # The current isolated scene declares these bodies directly in worldbody.
  # No external meshes, robot model, FK, or physics step is needed for this check.
  root = ET.parse(scene).getroot()
  card = root.find("./worldbody/body[@name='card']")
  nominal = _xml_pose(card)
  nominal_matrix = _card_pose_transform(nominal)
  recorded_nominal = _card_pose_transform(variation.get("nominal_pose_wxyz"))
  if not np.allclose(recorded_nominal, nominal_matrix, atol=1e-12, rtol=0):
    raise ValueError("card randomization nominal pose differs from known scene")
  rotation = np.eye(4)
  rotation[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
  expected = nominal_matrix.copy()
  expected[:3, :3] = rotation[:3, :3] @ nominal_matrix[:3, :3]
  expected[:2, 3] += xy
  sampled = _card_pose_transform(variation.get("sampled_pose_wxyz"))
  original = _card_pose_transform(file["objects/card/pose_wxyz"][0])
  if not np.allclose(sampled, expected, atol=1e-10, rtol=0):
    raise ValueError("card sampled pose differs from independently reconstructed pose")
  if not np.allclose(original, expected, atol=1e-10, rtol=0):
    raise ValueError("recorded first card pose differs from sampled reset pose")
  if float(file["state/timestamp"][0]) != 0:
    raise ValueError("card randomization source must include the exact reset sample")

  core = card.find("geom[@name='card_core_geom']")
  table = root.find("./worldbody/body[@name='poker_table']")
  top = table.find("geom[@name='poker_table_top']") if table is not None else None
  for geometry in (core, top):
    if geometry is None or geometry.get("type") != "box":
      raise ValueError("card randomization support audit expects known box geometries")
  card_size, table_size = (
    np.fromstring(geometry.get("size", ""), sep=" ") for geometry in (core, top)
  )
  if any(
    size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0)
    for size in (card_size, table_size)
  ):
    raise ValueError("card randomization support audit found invalid box sizes")
  world_from_card = original @ _card_pose_transform(_xml_pose(core))
  world_from_top = _card_pose_transform(_xml_pose(table)) @ _card_pose_transform(
    _xml_pose(top)
  )
  corners = np.array(list(product((-1, 1), repeat=3))) * card_size
  world_corners = corners @ world_from_card[:3, :3].T + world_from_card[:3, 3]
  local = (world_corners - world_from_top[:3, 3]) @ world_from_top[:3, :3]
  margin = float(np.min(table_size[:2] - np.abs(local[:, :2])))
  bottom_gap = float(local[:, 2].min() - table_size[2])
  support = variation.get("table_support", {})
  if (
    not isinstance(support, dict)
    or support.get("valid") is not True
    or margin < 0.00025
    or not -1e-12 <= bottom_gap <= 0.001
    or not np.isclose(
      support.get("minimum_corner_margin_xy_m", np.nan), margin, atol=1e-10, rtol=0
    )
    or not np.isclose(
      support.get("bottom_gap_m", np.nan), max(0, bottom_gap), atol=1e-10, rtol=0
    )
  ):
    raise ValueError("card randomization table support is not independently confirmed")
  return {
    "enabled": True,
    "mode": "seeded_uniform",
    "seed": seed,
    "xy_jitter_m": bounds[0],
    "yaw_jitter_rad": bounds[1],
    "sampled_offset_xy_m": xy.tolist(),
    "sampled_yaw_offset_rad": yaw,
    "first_raw_pose_verified": True,
    "nominal_scene_pose_verified": True,
    "minimum_corner_margin_xy_m": margin,
    "bottom_gap_m": max(0, bottom_gap),
    "scope": "initial card pose only; reset does not perturb observations, controls, or robot initial state",
  }


def audit_precontact_noise(file, metadata=None, outcome=None):
  """Read-only 500-Hz control evidence check, also usable by collection resume.

  This never constructs a simulator or trusts a claimed final latch without
  checking the full saved trace. Missing evidence is a failure, not a warning.
  """
  report = {"valid": False, "enabled": False, "errors": []}
  try:
    metadata = _attr_json(file, "metadata_json") if metadata is None else metadata
    outcome = _attr_json(file, "outcome_json") if outcome is None else outcome
    if not isinstance(metadata, dict) or not isinstance(outcome, dict):
      raise ValueError("source metadata and outcome must be JSON objects")
    if metadata.get("preset") != PRECONTACT_PRESET:
      if (
        metadata.get("precontact_noise") is not None
        or outcome.get("precontact_noise") is not None
        or (isinstance(file, h5py.Group) and "control/precontact_noise" in file)
      ):
        raise ValueError(
          "non-precontact presets must not contain precontact noise evidence"
        )
      report.update(valid=True, scope="no precontact perturbation in this preset")
      return report
    report["enabled"] = True
    report.update(_audit_precontact_present(file, metadata, outcome))
    report["valid"] = True
  except (
    AttributeError,
    IndexError,
    KeyError,
    OSError,
    TypeError,
    ValueError,
    OverflowError,
  ) as error:
    report["errors"].append(f"precontact noise evidence: {error}")
  return report


_PRECONTACT_LINKS = [
  f"hand_r_{finger}_link{6 if finger == 'thumb' else 4}"
  for finger in ("thumb", "index", "middle", "ring", "pinky")
]
_PRECONTACT_ACTUATORS = [f"right_arm_joint{i}" for i in range(1, 8)]


def _precontact_settings(value):
  if not isinstance(value, dict):
    raise ValueError("precontact action_noise settings must be explicit")
  sigma = value.get("std_rad")
  if (
    isinstance(sigma, bool)
    or not isinstance(sigma, (int, float))
    or not np.isfinite(sigma)
    or not 0 <= sigma <= np.deg2rad(0.05)
  ):
    raise ValueError("precontact sigma must be finite and between 0 and 0.05 degree")
  expected = {
    "schema_version": "poker-precontact-arm-noise-settings-v1",
    "std_rad": sigma,
    "correlation_time_s": 0.15,
    "maximum_offset_rad": 3 * sigma,
    "maximum_offset_rate_rad_s": float(np.deg2rad(0.5)),
    "distribution": "bounded_ornstein_uhlenbeck_gaussian",
    "tactile_source": "solver_contact_proxy_v1",
    "contact_link_names": _PRECONTACT_LINKS,
    "contact_stop_rule": "first_any_right_fingertip_contact",
    "actuator_scope": "right_arm_seven_position_servo_controls",
    "permanent_until_reset": True,
  }
  if value != expected or value.get("permanent_until_reset") is not True:
    raise ValueError("precontact noise parameters differ from the fixed contract")
  return expected


def _precontact_control_limits():
  """Read actual known robot actuator ranges without loading the robot model."""
  control = Path(__file__).resolve().parent / "mjcf/control.xml"
  root = ET.parse(control).getroot()
  limits = []
  for name in _PRECONTACT_ACTUATORS:
    node = root.find(f"./actuator/position[@name='{name}']")
    if node is None or node.get("joint") != name:
      raise ValueError("unknown right-arm actuator definition")
    limits.append(np.fromstring(node.get("ctrlrange", ""), sep=" "))
  limits = np.asarray(limits)
  if (
    limits.shape != (7, 2)
    or not np.isfinite(limits).all()
    or np.any(limits[:, 0] >= limits[:, 1])
  ):
    raise ValueError("invalid known right-arm actuator control ranges")
  robot = ET.parse(control.with_name("robot.xml")).getroot()
  if robot.find("compiler").get("angle") != "radian":
    raise ValueError("known robot joint limits must be declared in radians")
  joint_limits = []
  for name in _PRECONTACT_ACTUATORS:
    node = robot.find(f".//joint[@name='{name}']")
    if node is None:
      raise ValueError("unknown right-arm joint definition")
    joint_limits.append(np.fromstring(node.get("range", ""), sep=" "))
  joint_limits = np.asarray(joint_limits)
  if joint_limits.shape != (7, 2) or not np.isfinite(joint_limits).all():
    raise ValueError("invalid known right-arm joint limits")
  effective = np.column_stack(
    (
      np.maximum(limits[:, 0], joint_limits[:, 0]),
      np.minimum(limits[:, 1], joint_limits[:, 1]),
    )
  )
  return limits, effective


def _audit_precontact_present(file, metadata, outcome):
  settings = _precontact_settings(
    metadata.get("preset_settings", {}).get("action_noise")
  )
  seed = metadata.get("seed")
  if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
    raise ValueError("precontact episode seed must be a nonnegative integer")
  initial, final = metadata.get("precontact_noise"), outcome.get("precontact_noise")
  if not isinstance(initial, dict) or not isinstance(final, dict):
    raise ValueError("missing initial/final precontact metadata")
  for snapshot in (initial, final):
    for name, expected in (
      ("schema_version", "poker-precontact-arm-noise-v1"),
      ("settings", settings),
      ("seed", seed),
      ("configured", True),
      ("stream_tag", 0x5052434E),
      ("right_arm_actuator_names", _PRECONTACT_ACTUATORS),
      ("timestep_s", 0.002),
    ):
      if (
        name not in snapshot
        or snapshot[name] != expected
        or (isinstance(expected, bool) and snapshot[name] is not expected)
      ):
        raise ValueError(f"precontact initial/final metadata mismatch: {name}")
  limits, effective = _precontact_control_limits()
  for snapshot in (initial, final):
    recorded_limits = np.asarray(snapshot.get("right_arm_ctrlrange_rad"), dtype=float)
    if recorded_limits.shape != (7, 2) or not np.allclose(
      recorded_limits, limits, atol=1e-12, rtol=0
    ):
      raise ValueError("precontact recorded control ranges differ from known robot")
    recorded_effective = np.asarray(
      snapshot.get("right_arm_effective_control_bounds_rad"), dtype=float
    )
    if recorded_effective.shape != (7, 2) or not np.allclose(
      recorded_effective, effective, atol=1e-12, rtol=0
    ):
      raise ValueError(
        "precontact effective control bounds differ from known joint/actuator intersection"
      )
    if snapshot.get("right_arm_max_velocity_rad_s") != [3.1416] * 7:
      raise ValueError("precontact actuator speed limit differs from known controller")
  initial_command = np.asarray(initial.get("initial_arm_command_rad"), dtype=float)
  if initial_command.shape != (7,) or not np.isfinite(initial_command).all():
    raise ValueError("missing finite initial arm command")
  if not np.array_equal(
    initial_command, np.asarray(final.get("initial_arm_command_rad"))
  ):
    raise ValueError("final precontact metadata changed its initial arm command")
  group = file["control/precontact_noise"]
  if _attr_json(group, "metadata_json") != final:
    raise ValueError("precontact trace metadata differs from outcome metadata")
  columns = {
    name: np.asarray(group[name][:])
    for name in (
      "time_s",
      "tactile_time_s",
      "contact",
      "normal_force_n",
      "latched_before_step",
      "noise_offset_rad",
      "nominal_ctrl_rad",
      "actual_ctrl_rad",
      "control_mode",
    )
  }
  times = _ns(columns["time_s"], "precontact physical step")
  tactile_times = _ns(columns["tactile_time_s"], "precontact tactile evaluation")
  n = len(times)
  if not np.array_equal(times, np.arange(n, dtype=np.int64) * 2_000_000):
    raise ValueError(
      "precontact trace must cover every 500-Hz physical step from reset"
    )
  if not np.array_equal(tactile_times, times):
    raise ValueError("precontact tactile cache must correspond to the same solver step")
  for name, shape, kind in (
    ("contact", (n, 5), "b"),
    ("normal_force_n", (n, 5), "f"),
    ("latched_before_step", (n,), "b"),
    ("noise_offset_rad", (n, 7), "f"),
    ("nominal_ctrl_rad", (n, 7), "f"),
    ("actual_ctrl_rad", (n, 7), "f"),
    ("control_mode", (n,), "iu"),
  ):
    values = columns[name]
    if (
      values.shape != shape
      or values.dtype.kind not in kind
      or not np.isfinite(values).all()
    ):
      raise ValueError(f"precontact trace has invalid shape, dtype, or values: {name}")
  contact, force = columns["contact"], columns["normal_force_n"]
  latch, offset = columns["latched_before_step"], columns["noise_offset_rad"]
  nominal, actual = columns["nominal_ctrl_rad"], columns["actual_ctrl_rad"]
  mode = columns["control_mode"]
  if np.any(force < 0) or np.any(force[~contact] != 0):
    raise ValueError("precontact normal force is incompatible with the contact mask")
  if np.any((mode != 0) & (mode != 1)) or np.any((mode == 1) & ~latch):
    raise ValueError(
      "precontact control mode is invalid or Cartesian drive precedes contact"
    )
  if not np.allclose(actual - nominal, offset, atol=1e-12, rtol=0):
    raise ValueError(
      "precontact actual command does not equal nominal plus recorded offset"
    )
  if np.any(offset[latch] != 0) or np.any(offset[mode == 1] != 0):
    raise ValueError("precontact random input must be exactly zero after contact latch")
  if np.any(actual < limits[:, 0] - 1e-12) or np.any(actual > limits[:, 1] + 1e-12):
    raise ValueError("precontact actual controls exceeded known actuator limits")

  proxy = file["tactile_proxy"]
  names = [_text(value) for value in proxy["link_names"][:]]
  if len(names) != len(set(names)) or not set(_PRECONTACT_LINKS).issubset(names):
    raise ValueError("raw tactile proxy lacks the five distinct contact-stop links")
  if _text(proxy.attrs.get("source", "")) != "solver_contact_proxy_v1":
    raise ValueError("precontact stop evidence is not solver contact proxy")
  mapping = [names.index(name) for name in _PRECONTACT_LINKS]
  source_contact = np.asarray(proxy["contact"][:])[:, mapping]
  source_force = np.asarray(proxy["normal_force"][:])[:, mapping]
  state_times = _ns(file["state/timestamp"][:], "precontact source state")
  if state_times[0] != 0 or state_times[-1] != times[-1] + 2_000_000:
    raise ValueError(
      "precontact trace and original reset/terminal state coverage differ"
    )
  if source_contact.shape != (len(state_times), 5) or source_contact.dtype != np.bool_:
    raise ValueError("raw tactile proxy contact does not align with source state")
  if source_force.shape != source_contact.shape or not np.isfinite(source_force).all():
    raise ValueError("raw tactile proxy force does not align with source state")
  initial_contact = bool(source_contact[0].any())
  for snapshot in (initial, final):
    contact_snapshot = np.asarray(snapshot.get("initial_contact"))
    force_snapshot = np.asarray(snapshot.get("initial_normal_force_n"), dtype=float)
    if contact_snapshot.dtype != np.bool_ or not np.array_equal(
      contact_snapshot, source_contact[0]
    ):
      raise ValueError(
        "precontact initial contact metadata differs from raw reset tactile"
      )
    if force_snapshot.shape != (5,) or not np.allclose(
      force_snapshot, source_force[0], atol=1e-10, rtol=0
    ):
      raise ValueError(
        "precontact initial force metadata differs from raw reset tactile"
      )
  expected_latch = np.maximum.accumulate(np.r_[initial_contact, contact.any(axis=1)])[
    :-1
  ]
  if not np.array_equal(latch, expected_latch):
    raise ValueError(
      "precontact latch violates first-contact causality or is not permanent"
    )
  if not initial_contact and not contact.any():
    raise ValueError("successful precontact task has no contact-stop evidence")
  indices = np.searchsorted(times + 2_000_000, state_times[1:])
  if np.any(indices >= n) or not np.array_equal(
    times[indices] + 2_000_000, state_times[1:]
  ):
    raise ValueError("source state samples do not match physical integration endpoints")
  if not np.array_equal(source_contact[1:], contact[indices]) or not np.allclose(
    source_force[1:], force[indices], atol=1e-10, rtol=0
  ):
    raise ValueError(
      "precontact contact/force trace disagrees with recorded tactile proxy"
    )
  command_names = [_text(value) for value in file["commands/actuator_names"][:]]
  if len(command_names) != len(set(command_names)) or not set(
    _PRECONTACT_ACTUATORS
  ).issubset(command_names):
    raise ValueError("source controls lack distinct right-arm actuator names")
  command_ids = [command_names.index(name) for name in _PRECONTACT_ACTUATORS]
  source_controls = np.asarray(file["commands/actuator_control"][:])[:, command_ids]
  if (
    source_controls.shape != (len(state_times), 7)
    or not np.isfinite(source_controls).all()
  ):
    raise ValueError("source actuator controls do not align with state clock")
  if not np.allclose(source_controls[1:], actual[indices], atol=1e-12, rtol=0):
    raise ValueError(
      "precontact actual controls disagree with original 100-Hz controls"
    )
  # OU generation, handoff and summary counters are checked independently below.
  counters = _audit_precontact_generation(
    columns, settings, initial, final, initial_contact
  )
  return {
    "physics_step_count": n,
    "state_samples_matched": len(state_times) - 1,
    "source_tactile_contact_and_force_matched": True,
    "source_actuator_controls_matched": True,
    "initial_contact": initial_contact,
    "postcontact_offset_exactly_zero": True,
    "robot_actuator_limits_verified": True,
    **counters,
  }


def _audit_precontact_generation(columns, settings, initial, final, initial_contact):
  """Recreate RNG draws and bound projection, independently of the controller."""
  dt = 0.002
  latch = columns["latched_before_step"]
  actual, nominal = columns["actual_ctrl_rad"], columns["nominal_ctrl_rad"]
  offsets = columns["noise_offset_rad"]
  sigma = settings["std_rad"]
  maximum = 3 * sigma
  offset_delta = float(np.deg2rad(0.5)) * dt
  command_delta = 3.1416 * dt
  effective = np.asarray(initial["right_arm_effective_control_bounds_rad"], dtype=float)
  previous_actual = np.asarray(initial["initial_arm_command_rad"], dtype=float)
  previous_offset = np.zeros(7)
  ou = np.zeros(7)
  rng = np.random.default_rng(np.random.SeedSequence([initial["seed"], 0x5052434E]))
  alpha = np.exp(-dt / 0.15)
  scale = sigma * np.sqrt(-np.expm1(-2 * dt / 0.15))
  draws = 0
  max_control_delta = 0.0
  for i in range(len(latch)):
    if not latch[i] and sigma > 0:
      ou = alpha * ou + scale * rng.normal(size=7)
      draws += 1
      lower = np.maximum.reduce(
        (
          np.full(7, -maximum),
          previous_offset - offset_delta,
          effective[:, 0] - nominal[i],
          previous_actual - command_delta - nominal[i],
        )
      )
      upper = np.minimum.reduce(
        (
          np.full(7, maximum),
          previous_offset + offset_delta,
          effective[:, 1] - nominal[i],
          previous_actual + command_delta - nominal[i],
        )
      )
      if np.any(lower > upper + 1e-12):
        raise ValueError(
          "precontact nominal command creates incompatible noise/control limits"
        )
      expected_offset = np.clip(
        np.clip(ou, -maximum, maximum), np.minimum(lower, upper), upper
      )
      # The controller stores the actually rounded additive command, not the
      # pre-addition offset. Reproduce that rounding before the next slew bound.
      expected_actual = nominal[i] + expected_offset
      expected_offset = expected_actual - nominal[i]
      if not np.allclose(offsets[i], expected_offset, atol=1e-12, rtol=0):
        raise ValueError(
          f"precontact seeded OU/projection mismatch at physical step {i}"
        )
      if not np.allclose(actual[i], expected_actual, atol=1e-12, rtol=0):
        raise ValueError(
          f"precontact projected actual control mismatch at physical step {i}"
        )
      delta = float(np.max(np.abs(actual[i] - previous_actual)))
      max_control_delta = max(max_control_delta, delta)
      if delta > command_delta + 1e-12:
        raise ValueError("precontact active-noise command exceeds actuator slew limit")
      if np.max(np.abs(offsets[i] - previous_offset)) > offset_delta + 1e-12:
        raise ValueError("precontact random offset exceeds configured slew limit")
      previous_offset = expected_offset
    elif np.any(offsets[i] != 0):
      raise ValueError("precontact offset must be zero when sampling is disabled")
    previous_actual = actual[i]
  if np.any(np.abs(offsets) > maximum + 1e-12):
    raise ValueError("precontact offset exceeds 3-sigma bound")

  first_contact = (
    None if initial_contact else int(np.flatnonzero(columns["contact"].any(axis=1))[0])
  )
  detection_time = (
    0.0 if initial_contact else float(columns["time_s"][first_contact] + dt)
  )
  tactile_time = (
    0.0 if initial_contact else float(columns["tactile_time_s"][first_contact])
  )
  handoffs = int(first_contact is not None and np.any(offsets[first_contact] != 0))
  common = {
    "postcontact_random_sample_count": 0,
    "failed": False,
    "trace_time_semantics": "time_s is control start; tactile is completed-step solver cache",
    "control_mode_names": {"0": "joint_position_servo", "1": "bounded_cartesian"},
    "noise_offset_semantics": "actual additive random input; zero after latch; nominal may retain servo recovery",
  }
  expected_initial = {
    **common,
    "contact_latched": initial_contact,
    "contact_detected_time_s": 0.0 if initial_contact else None,
    "contact_tactile_time_s": 0.0 if initial_contact else None,
    "stop_reason": "first_right_fingertip_contact" if initial_contact else None,
    "random_sample_count": 0,
    "physics_step_count": 0,
    "postcontact_noise_disabled": initial_contact,
    "command_handoff_count": 0,
  }
  expected_final = {
    **common,
    "contact_latched": True,
    "contact_detected_time_s": detection_time,
    "contact_tactile_time_s": tactile_time,
    "stop_reason": "first_right_fingertip_contact",
    "random_sample_count": draws,
    "physics_step_count": len(latch),
    "postcontact_noise_disabled": True,
    "command_handoff_count": handoffs,
  }
  for label, snapshot, expected in (
    ("initial", initial, expected_initial),
    ("final", final, expected_final),
  ):
    for key, value in expected.items():
      recorded = snapshot.get(key)
      if isinstance(value, float):
        matched = (
          not isinstance(recorded, bool)
          and isinstance(recorded, (float, int))
          and np.isfinite(recorded)
          and abs(recorded - value) <= 1e-12
        )
      elif isinstance(value, bool):
        matched = recorded is value
      elif isinstance(value, int):
        matched = (
          isinstance(recorded, int)
          and not isinstance(recorded, bool)
          and recorded == value
        )
      else:
        matched = key in snapshot and recorded == value
      if not matched:
        raise ValueError(f"precontact {label} metadata disagrees with trace: {key}")
  return {
    "seeded_noise_projection_verified": True,
    "independent_seed_stream_tag": 0x5052434E,
    "random_sample_count": draws,
    "contact_detected_time_s": detection_time,
    "contact_tactile_time_s": tactile_time,
    "command_handoff_count": handoffs,
    "max_active_noise_control_delta_rad": max_control_delta,
    "actuator_slew_verification_scope": "steps with active precontact noise; later original joint/Cartesian control is not reinterpreted as perturbation",
  }


def _physics_and_identity(
  file, metadata, outcome, errors, *, source_code_policy="current"
):
  if source_code_policy not in {"current", "recorded"}:
    raise ValueError("unknown source code policy")

  def check(condition, message):
    if not condition:
      errors.append(message)

  check(outcome.get("success") is True, "source full-task success is not true")
  check(metadata.get("scene") == "poker-draw", "source scene is not poker-draw")
  preset = metadata.get("preset")
  check(
    preset in ("middle-force-v1", *CARD_RANDOMIZED_PRESETS),
    "source preset is not a supported middle-force preset",
  )
  settings = metadata.get("preset_settings", {})
  pressure = settings.get("pressure_window", {})
  contacts = metadata.get("contact_model", {})
  for value, expected, label in (
    (metadata.get("press_force_per_finger_n"), 0.5, "normal-force target"),
    (settings.get("press_force_per_finger_n"), 0.5, "preset normal-force target"),
    (pressure.get("table_friction"), 1.0, "table/card friction"),
    (pressure.get("drive_limit_n"), 4.0, "horizontal servo limit"),
    (pressure.get("slide_speed_m_s"), 0.005, "slide speed"),
    (contacts.get("physics_timestep_used_s"), 0.002, "physics timestep"),
    (
      contacts.get("contact_friction_impedance_ratio_used"),
      100,
      "contact impedance ratio",
    ),
  ):
    check(
      isinstance(value, (int, float))
      and np.isfinite(value)
      and np.isclose(value, expected, atol=1e-12, rtol=0),
      f"middle preset mismatch: {label}",
    )
  if preset not in CARD_RANDOMIZED_PRESETS:
    for key in ("object_xy_jitter", "object_yaw_jitter"):
      value = metadata.get(key)
      check(
        isinstance(value, (float, int)) and np.isfinite(value) and value == 0,
        f"middle zero preset mismatch: {key}",
      )
  check(
    settings.get("observation_noise", "missing") is None,
    "observation noise is not explicitly disabled",
  )
  if preset != PRECONTACT_PRESET:
    check(
      settings.get("action_noise", "missing") is None,
      "action noise is not explicitly disabled",
    )
  check(
    contacts.get("add_genesis_probes") is False,
    "accepted contact model must not add Genesis probes",
  )
  friction = np.asarray(contacts.get("table_card_pair_friction", []))
  check(
    friction.shape == (5,)
    and np.allclose(friction, [1, 1, 0.005, 0.0005, 0.0005], atol=1e-12, rtol=0),
    "runtime table/card pair friction does not match middle preset",
  )
  for key in ("table_card_pair_solref_used", "card_geom_solref_used"):
    value = np.asarray(contacts.get(key, []))
    check(
      value.shape == (2,) and np.allclose(value, [0.01, 1], atol=1e-12, rtol=0),
      f"runtime contact mismatch: {key}",
    )
  check(float(file.attrs.get("physics_hz", 0)) == 500, "recorded physics_hz is not 500")
  edge = outcome.get("edge_outcome", {})
  from kaihand_tactile_env.tasks.poker_draw.acceptance import (
    STRICT_FORCE_POLICY,
    accept_edge,
    validate_recorded_acceptance,
  )

  try:
    validate_recorded_acceptance(metadata, outcome)
    check(
      accept_edge(edge, metadata.get("acceptance_policy", STRICT_FORCE_POLICY)),
      "source middle edge acceptance failed",
    )
  except ValueError as error:
    errors.append(str(error))
  for name in ("target_reached", "held_at_edge"):
    check(edge.get(name) is True, f"source middle edge gate failed: {name}")
  handoff = outcome.get("handoff_outcome", {})
  check(handoff.get("completed") is True, "source bounded-servo handoff incomplete")
  transition = handoff.get("transition", {})
  for name in ("maximum_qpos_change", "maximum_qvel_change", "control_jump_rad"):
    value = transition.get(name)
    check(
      isinstance(value, (int, float)) and abs(value) <= 1e-12,
      f"handoff is not continuous: {name}",
    )
  check(
    transition.get("object_state_modified") is False,
    "handoff must explicitly avoid object state changes",
  )
  check(
    transition.get("physics_parameters_modified") is False,
    "handoff must not change physical parameters",
  )
  paths, scene = _known_source_paths(metadata.get("preset"))
  randomization = None
  try:
    randomization = _audit_initial_randomization(file, metadata, scene)
  except (AttributeError, KeyError, TypeError, ValueError, ET.ParseError) as error:
    errors.append(f"initial card randomization: {error}")
  precontact = audit_precontact_noise(file, metadata, outcome)
  errors.extend(precontact["errors"])
  recorded_hashes = settings.get("controller_source_sha256", {})
  allowed_sets = [set(paths)]
  # The original strict-force recordings predate the separate acceptance module.
  if (
    source_code_policy == "recorded"
    and metadata.get("acceptance_policy", "strict-force-v1") == "strict-force-v1"
  ):
    allowed_sets.append(set(paths) - {"poker_draw/acceptance.py"})
  check(
    isinstance(recorded_hashes, dict) and set(recorded_hashes) in allowed_sets,
    "controller source hash list incomplete or unexpected",
  )
  if not isinstance(recorded_hashes, dict):
    recorded_hashes = {}
  check(
    bool(recorded_hashes)
    and all(
      isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
      for value in recorded_hashes.values()
    ),
    "recorded controller source hashes missing or malformed",
  )
  hash_results = {}
  drift_source_names = []
  for name, path in paths.items():
    actual = _sha256(path)
    matched = actual == recorded_hashes.get(name)
    hash_results[name] = {
      "recorded": recorded_hashes.get(name),
      "current": actual,
      "matches": matched,
    }
    if not matched:
      drift_source_names.append(name)
    if source_code_policy == "current":
      check(
        matched,
        f"current-code drift: recorded controller source differs from current known file: {name}",
      )
  base_hash = _sha256(scene)
  check(
    metadata.get("base_model_sha256") == base_hash,
    "recorded base scene SHA differs from current known scene",
  )
  for name in ("model_sha256", "model_fingerprint"):
    check(
      bool(re.fullmatch(r"[0-9a-f]{64}", _text(file.attrs.get(name, "")))),
      f"missing or invalid source {name}",
    )
  return {
    "preset": metadata.get("preset"),
    "initial_card_randomization": randomization,
    "precontact_noise": precontact,
    "edge_gate": edge,
    "handoff_completed": handoff.get("completed"),
    "base_scene_sha256": base_hash,
    "controller_sources": hash_results,
    "controller_source_status": "current_code_drift"
    if drift_source_names
    else "matches_current_code",
    "drift_source_names": drift_source_names,
    "source_code_policy": source_code_policy,
    "source_identity_scope": (
      "archival data audit: recorded hash provenance preserved; historical controller source bytes are not independently verified"
      if source_code_policy == "recorded"
      else "strict comparison with current code; code drift alone does not establish that recorded data bytes are malformed"
    ),
    "recorded_model_sha256": _text(file.attrs.get("model_sha256", "")),
    "recorded_model_fingerprint": _text(file.attrs.get("model_fingerprint", "")),
    "temporary_wrapper_bytes_verified": False,
    "model_verification_scope": "known base scene and recorded runtime contact settings; controller byte verification follows source_code_policy; temporary wrapper bytes and compiled model not reconstructed",
  }


def _drive_and_phases(file, state_times, errors):
  command = file["commands"]
  n = len(state_times)
  phase = np.asarray([_text(item) for item in command["phase"][:]])
  active = np.asarray(command["drive_budget_active"][:])
  actual = np.asarray(command["drive_actual_fx_n"][:], dtype=np.float64)
  requested = np.asarray(command["drive_requested_fx_n"][:], dtype=np.float64)
  limit = np.asarray(command["drive_limit_n"][:], dtype=np.float64)
  if (
    any(value.shape != (n,) for value in (phase, active, actual, requested, limit))
    or active.dtype != np.bool_
  ):
    raise ValueError(
      "recorded drive streams must align with state clock and have boolean active mask"
    )
  if not np.isfinite(np.column_stack((actual, requested, limit))).all():
    raise ValueError("nonfinite recorded drive force or limit")
  if not active.any():
    errors.append("bounded Cartesian servo was never active")
  if np.any(np.abs(actual[active]) > 4 + FORCE_CAP_TOLERANCE_N):
    errors.append("recorded actual active world-X servo wrench exceeded 4 N")
  if np.any(np.abs(limit[active] - 4) > 1e-12):
    errors.append("active servo limit differs from 4 N")
  if np.any(limit[~active] != 0) or np.any(actual[~active] != 0):
    errors.append("inactive drive diagnostics are not explicitly zero")
  active_phases = sorted(set(phase[active]))
  if set(active_phases) != {"slide_card", "edge_hold"}:
    errors.append(
      f"force limit scope differs from slide_card/edge_hold: {active_phases}"
    )
  indices = np.flatnonzero(active)
  if len(indices) and not np.all(np.diff(indices) == 1):
    errors.append("drive budget active samples are not one contiguous interval")
  controls = command["actuator_control"]
  if controls.shape != (n, len(command["actuator_names"])):
    raise ValueError("actual actuator controls/names do not align with state samples")
  for start in range(0, n, 1024):
    if not np.isfinite(controls[start : start + 1024]).all():
      errors.append("actual actuator control contains nonfinite values")
      break
  return phase, {
    "recorded_samples": n,
    "active_samples": int(active.sum()),
    "active_phases": active_phases,
    "max_abs_active_actual_fx_n": float(np.max(np.abs(actual[active]), initial=0)),
    "max_abs_active_requested_fx_n": float(
      np.max(np.abs(requested[active]), initial=0)
    ),
    "cap_tolerance_n": FORCE_CAP_TOLERANCE_N,
    "first_active_state_time_s": float(state_times[indices[0]] / 1e9)
    if len(indices)
    else None,
    "last_active_state_time_s": float(state_times[indices[-1]] / 1e9)
    if len(indices)
    else None,
    "sampling_scope": "recorded state samples only; no assertion about unrecorded intermediate physics steps",
  }


def _phase_force_statistics(force, phases, mapping, *, include_thumb=False):
  normal = force["normal_force_n"]
  tangent = force["tangent_force_n"]
  if normal.shape != (len(phases), len(mapping)) or tangent.shape != (*normal.shape, 2):
    raise ValueError("force aggregate streams are not aligned with recorded phases")
  fingers = (
    ("thumb", "index", "middle", "ring", "pinky")
    if include_thumb
    else ("index", "middle", "ring", "pinky")
  )
  right = [
    mapping[f"hand_r_{finger}_link{6 if finger == 'thumb' else 4}"]
    for finger in fingers
  ]
  width = len(right)
  accumulators = {}
  for start in range(0, len(phases), 256):
    stop = min(len(phases), start + 256)
    fn = normal[start:stop][:, right]
    ft = tangent[start:stop][:, right]
    if not np.isfinite(fn).all() or not np.isfinite(ft).all():
      raise ValueError("nonfinite recorded aggregate forces")
    for label in set(phases[start:stop]):
      select = phases[start:stop] == label
      selected_fn, selected_ft = fn[select], ft[select]
      aggregate = accumulators.setdefault(
        str(label),
        {
          "count": 0,
          "fn_sum": np.zeros(width),
          "ft_min": np.full((width, 2), np.inf),
          "ft_max": np.full((width, 2), -np.inf),
          "all_four_positive": 0,
        },
      )
      aggregate["count"] += int(select.sum())
      aggregate["fn_sum"] += selected_fn.sum(axis=0)
      aggregate["ft_min"] = np.minimum(aggregate["ft_min"], selected_ft.min(axis=0))
      aggregate["ft_max"] = np.maximum(aggregate["ft_max"], selected_ft.max(axis=0))
      aggregate["all_four_positive"] += int(np.all(selected_fn > 0, axis=1).sum())
  return {
    label: {
      "samples": value["count"],
      ("right_finger_order" if include_thumb else "right_four_finger_order"): [
        "little" if finger == "pinky" else finger for finger in fingers
      ],
      "normal_total_per_finger_mean_n": (value["fn_sum"] / value["count"]).tolist(),
      "signed_tangent_total_per_finger_xy_min_n": value["ft_min"].tolist(),
      "signed_tangent_total_per_finger_xy_max_n": value["ft_max"].tolist(),
      (
        "all_fingers_strictly_positive_fraction"
        if include_thumb
        else "all_four_strictly_positive_fraction"
      ): value["all_four_positive"] / value["count"],
    }
    for label, value in accumulators.items()
  }


def audit_tict_source(
  source, release_root, session_id, camera="head", *, source_code_policy="current"
) -> dict[str, Any]:
  """Cross-check all exported frames and the source task's own physical gates."""
  errors, warnings = [], []
  report = {
    "schema_version": "kaihand-tict-source-audit-v1",
    "valid": False,
    "errors": errors,
    "warnings": warnings,
    "upstream_loader_verified": False,
    "source_code_policy": source_code_policy,
  }
  try:
    if source_code_policy not in {"current", "recorded"}:
      raise ValueError("unknown source code policy")
    source, root = (
      Path(source).expanduser().resolve(),
      Path(release_root).expanduser().resolve(),
    )
    if source.suffix != ".h5" or not source.is_file():
      raise ValueError("input must be a finalized .h5, never .partial")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", session_id):
      raise ValueError("unsafe session id")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", camera):
      raise ValueError("unsafe camera name")
    recorded_stat = source.stat()
    source_hash = _sha256(source)
    release_audit = _json(root / "dataset_audit.json")
    sidecar_path = root / "tict_sidecars" / session_id / "fingertip_tactile_v1.npz"
    report["session_id"] = session_id
    report["release"] = {
      "path": str(root),
      "sidecar_path": str(sidecar_path.relative_to(root)),
      "sidecar_sha256": _sha256(sidecar_path),
      "dataset_audit_sha256": _sha256(root / "dataset_audit.json"),
    }
    with np.load(
      sidecar_path,
      allow_pickle=False,
    ) as archive:
      sidecar = {name: archive[name] for name in archive.files}
    sidecar_source = json.loads(sidecar["source_metadata_json"].item())
    if (
      release_audit["source"]["sha256"] != source_hash
      or sidecar_source["source_hdf5_sha256"] != source_hash
    ):
      errors.append(
        "release/sidecar source SHA256 does not match the supplied source bytes"
      )
    report["source"] = {
      "path": str(source),
      "sha256": source_hash,
      "bytes": recorded_stat.st_size,
    }
    with h5py.File(source, "r") as file:
      if _text(file.attrs.get("schema_version", "")) != "kaihand_tactile_episode_v1":
        raise ValueError("unknown original recording schema")
      metadata, outcome = (
        _attr_json(file, "metadata_json"),
        _attr_json(file, "outcome_json"),
      )
      if (
        sidecar_source["source_recording_metadata"] != metadata
        or sidecar_source["source_outcome"] != outcome
      ):
        errors.append(
          "release source metadata/outcome differs from actual HDF5 attributes"
        )
      usb_source = metadata.get("scene") == "usb-insert"
      object_name = "usb_plug" if usb_source else "card"
      if usb_source:
        if source_code_policy != "current":
          raise ValueError(
            "archival source code policy is currently supported only for poker"
          )
        identity = _usb_physics_and_identity(file, metadata, outcome, errors)
      else:
        identity = _physics_and_identity(
          file, metadata, outcome, errors, source_code_policy=source_code_policy
        )
      report["physical_gates_and_identity"] = identity
      if source_code_policy == "recorded":
        warnings.append(
          "Archival audit: historical controller bytes were not independently verified; recorded provenance and current-code drift are reported separately from data integrity."
        )
      state_times = _ns(file["state/timestamp"][:], "state")
      if usb_source:
        phases, report["recorded_motion"] = _usb_recorded_motion(
          file, state_times, outcome, errors
        )
      else:
        phases, report["drive"] = _drive_and_phases(file, state_times, errors)
      frames = file[f"cameras/{camera}"]
      render_times = _ns(frames["pose_timestamp"][:], "render pose")
      capture_times = _ns(frames["timestamp"][:], "camera capture")
      force = file["tactile_contact_force"]
      force_times = _ns(force["timestamp"][:], "solver force", strict=False)
      if usb_source and not np.array_equal(
        _ns(file["physics/solver_timestamp"][:], "USB physics solver", strict=False),
        force_times,
      ):
        errors.append("USB raw physics and tactile solver clocks disagree")
      if usb_source:
        if not np.array_equal(force_times, state_times):
          errors.append(
            "USB tactile solver clock must equal post-step state timestamps"
          )
        if not np.array_equal(
          _ns(file["physics/solver_timestamp"][:], "USB physics solver", strict=False),
          state_times,
        ):
          errors.append(
            "USB physics solver clock must equal post-step state timestamps"
          )
        if not np.array_equal(render_times, capture_times):
          errors.append(
            "USB camera pose clock must equal its post-step acquisition clock"
          )
      if len(force_times) != len(state_times) or np.any(force_times > state_times):
        raise ValueError(
          "solver force clock must align with and never follow post-step state clock"
        )
      cached_offset = state_times - force_times
      if np.any(cached_offset > 2_000_001):
        errors.append(
          "cached solver timestamp is more than one physics step behind state"
        )
      n = len(render_times)
      if len(capture_times) != n or not np.array_equal(
        sidecar["timestamps_ns"], render_times
      ):
        raise ValueError("sidecar/image/cached-pose timestamp mismatch")
      if np.any(capture_times < render_times) or np.any(
        capture_times - render_times > 2_000_001
      ):
        errors.append("camera/render clock offset is outside zero to one physics step")
      frame_names = [f"{i:05d}" for i in range(n)]
      if sidecar["frame_names"].tolist() != frame_names:
        raise ValueError(
          "sidecar frame names do not enumerate every source camera frame"
        )
      for key, expected in (
        ("side_names", list(SIDES)),
        ("finger_names", list(FINGERS)),
        ("tactile_channel_names", ["normal", "tangent_x", "tangent_y"]),
      ):
        if sidecar[key].tolist() != expected:
          raise ValueError(f"source/release channel ordering mismatch: {key}")
      camera_indices = np.asarray(frames["state_index"][:])
      if (
        camera_indices.shape != (n,)
        or camera_indices.dtype.kind not in "iu"
        or np.any(camera_indices < 0)
        or np.any(camera_indices >= len(state_times))
        or camera_indices[-1] != len(state_times) - 1
        or capture_times[-1] != state_times[-1]
        or phases[-1] != "terminal_settle"
      ):
        raise ValueError(
          "source camera is not synchronized to the complete terminal state"
        )
      _audit_camera_state_clock(
        state_times, force_times, capture_times, render_times, camera_indices, errors
      )
      if usb_source and not np.array_equal(render_times, state_times[camera_indices]):
        errors.append(
          "USB camera pose clock must equal indexed post-step state timestamps"
        )
      link_names = [_text(item) for item in force["link_names"][:]]
      mapping = {name: i for i, name in enumerate(link_names)}
      expected_links = [
        f"hand_{side[0]}_{'pinky' if finger == 'little' else finger}_link{6 if finger == 'thumb' else 4}"
        for side in SIDES
        for finger in FINGERS
      ]
      if len(mapping) != len(link_names) or set(mapping) != set(expected_links):
        raise ValueError(
          "bilateral force source must contain all ten distinct expected links"
        )
      source_order = [mapping[name] for name in expected_links]
      if force["normal_taxel_force_n"].shape != (len(force_times), 10, 7, 5) or force[
        "tangent_taxel_force_n"
      ].shape != (len(force_times), 10, 7, 5, 2):
        raise ValueError("source force taxel shape mismatch")
      matched_indices = np.searchsorted(force_times, render_times, side="right") - 1
      max_age = int(sidecar["max_sync_error_ns"].item())
      if not 0 <= max_age <= 20_000_000:
        raise ValueError("maximum allowed tactile age differs from PDF contract")
      mismatches = {
        name: 0
        for name in (
          "index",
          "age",
          "validity",
          "mean",
          "geometry",
          "json",
          "pixels",
          "conservation",
        )
      }
      geometry_error = mean_error = conservation_error = 0.0
      nonzero = np.zeros((2, 5, 3), dtype=np.int64)
      channel_frames = np.zeros((2, 5, 3), dtype=np.int64)
      source_extrema_min = np.full((2, 5, 3), np.inf)
      source_extrema_max = np.full((2, 5, 3), -np.inf)
      original_intrinsic = np.asarray(frames["intrinsic"][:])
      object_pose_path = f"objects/{object_name}/pose_wxyz"
      initial_object_anchor = _card_pose_transform(
        file[object_pose_path][0], object_name
      )
      if usb_source and (
        sidecar_source.get("object_name") != object_name
        or sidecar_source.get("object_pose_source") != object_pose_path
        or sidecar_source.get("static_anchor_state_index") != 0
        or sidecar_source.get("static_anchor_timestamp_ns") != 0
        or sidecar_source.get("force_action")
        != "force acting on tactile pad from usb_plug contact"
      ):
        errors.append("USB sidecar object/anchor/force provenance mismatch")
      data_root = (
        root / "production" / session_id / "09_humanego_adapter/preprocess/all_data"
      )
      for index in range(n):
        source_index = int(matched_indices[index])
        age = (
          int(render_times[index] - force_times[source_index])
          if source_index >= 0
          else -1
        )
        valid = source_index >= 0 and 0 <= age <= max_age
        expected_mean = np.zeros((2, 5, 3), dtype=np.float64)
        if valid:
          # Each read is one ten-fingertip sample, never the full force movie.
          fn = np.asarray(
            force["normal_taxel_force_n"][source_index], dtype=np.float64
          )[source_order]
          ft = np.asarray(
            force["tangent_taxel_force_n"][source_index], dtype=np.float64
          )[source_order]
          if (
            not np.isfinite(fn).all()
            or not np.isfinite(ft).all()
            or np.any(fn < -1e-12)
          ):
            raise ValueError(
              f"invalid original taxel force at source sample {source_index}"
            )
          expected_mean.reshape(10, 3)[:, 0] = np.sum(fn, axis=(1, 2)) / 35.0
          expected_mean.reshape(10, 3)[:, 1:] = np.sum(ft, axis=(1, 2)) / 35.0
          actual_fn = np.asarray(force["normal_force_n"][source_index])[source_order]
          actual_ft = np.asarray(force["tangent_force_n"][source_index])[source_order]
          difference = max(
            float(np.max(np.abs(fn.sum(axis=(1, 2)) - actual_fn))),
            float(np.max(np.abs(ft.sum(axis=(1, 2)) - actual_ft))),
          )
          conservation_error = max(conservation_error, difference)
          mismatches["conservation"] += int(difference > 1e-9)
        mismatches["index"] += int(
          sidecar["tactile_source_index"][index] != source_index
        )
        mismatches["age"] += int(sidecar["tactile_sync_error_ns"][index] != age)
        mismatches["validity"] += int(
          bool(sidecar["tactile_frame_valid"][index]) != valid
          or not np.all(sidecar["tactile_channel_mask"][index] == valid)
        )
        difference = float(
          np.max(np.abs(sidecar["tactile_mean"][index] - expected_mean))
        )
        mean_error = max(mean_error, difference)
        mismatches["mean"] += int(
          not np.allclose(
            sidecar["tactile_mean"][index], expected_mean, atol=1e-8, rtol=2e-6
          )
        )
        nonzero += expected_mean != 0
        channel_frames += int(valid)
        source_extrema_min = np.minimum(source_extrema_min, expected_mean)
        source_extrema_max = np.maximum(source_extrema_max, expected_mean)
        wrist = _se3(frames["world_from_wrist"][index], "original wrist")
        fingertips = _se3(frames["world_from_fingertip"][index], "original fingertip")
        # General matrix solve is independent of exporter's custom rigid inverse.
        relative = np.empty((2, 5, 4, 4))
        for side in range(2):
          for finger in range(5):
            relative[side, finger] = np.linalg.solve(
              wrist[side], fingertips[side, finger]
            )
        difference = float(
          np.max(np.abs(relative - sidecar["T_fingertip_to_wrist"][index]))
        )
        geometry_error = max(geometry_error, difference)
        mismatches["geometry"] += int(
          difference > 1e-9 or not sidecar["finger_valid"][index].all()
        )
        camera_pose = _se3(frames["world_from_camera"][index], "original camera").copy()
        camera_pose[:3, 1:3] *= -1
        document = _json(data_root / frame_names[index] / "training_data.json")
        m = document["metadata"]
        correct = (
          m["idx"] == index
          and m["timestamp_ns"] == int(render_times[index])
          and m["is_finished"] is (index == n - 1)
          and np.allclose(m["c2w"], camera_pose, atol=1e-10, rtol=0)
          and np.allclose(
            m["world_transforms"]["cam0"], camera_pose, atol=1e-10, rtol=0
          )
          and np.allclose(
            m["world_transforms"]["virtual_static_anchor"],
            initial_object_anchor,
            atol=1e-10,
            rtol=0,
          )
          and np.allclose(
            np.asarray(m["k"]).reshape(3, 3), original_intrinsic, atol=1e-10, rtol=0
          )
        )
        for side_id, side in enumerate(SIDES):
          correct = correct and np.allclose(
            document["entities"]["hands_hawor_v3"][side]["T_hand_to_world"],
            wrist[side_id],
            atol=1e-10,
            rtol=0,
          )
        mismatches["json"] += int(not correct)
        with Image.open(data_root / frame_names[index] / "rgb.png") as image:
          pixels = np.asarray(image.convert("RGB"))
        mismatches["pixels"] += int(not np.array_equal(pixels, frames["rgb"][index]))
      for name, count in mismatches.items():
        if count:
          errors.append(f"source/release mismatch: {name} in {count} frame(s)")
      report["cross_check"] = {
        "frames_checked": n,
        "mismatch_counts": mismatches,
        "maximum_relative_transform_element_error": geometry_error,
        "maximum_tactile_mean_error_n": mean_error,
        "maximum_taxel_sum_vs_aggregate_error_n": conservation_error,
        "tactile_nonzero_frames_per_channel": nonzero.tolist(),
        "tactile_valid_frames_per_channel": channel_frames.tolist(),
        "source_mean_channel_min_n": source_extrema_min.tolist(),
        "source_mean_channel_max_n": source_extrema_max.tolist(),
        "side_order": list(SIDES),
        "finger_order": list(FINGERS),
        "channel_order": ["normal", "tangent_x", "tangent_y"],
        "force_mean_semantics": "each channel sum over 35 taxels / 35; signed xy retained",
      }
      report["clock"] = {
        "max_state_minus_solver_ns": int(cached_offset.max()),
        "max_capture_minus_render_ns": int(np.max(capture_times - render_times)),
        "max_tactile_age_ns": int(np.max(sidecar["tactile_sync_error_ns"])),
        "state_samples": len(state_times),
        "solver_samples": len(force_times),
        "rgb_samples": n,
        "median_rgb_interval_ns": int(np.median(np.diff(render_times))),
      }
      if usb_source:
        report["clock"]["observation_clock"] = metadata.get("observation_clock")
      report["phase_forces"] = _phase_force_statistics(
        force, phases, mapping, include_thumb=usb_source
      )
      warnings.append(
        "No remote T-ICT loader, training, or actuator replay was executed."
      )
      warnings.append(
        "Temporary wrapper bytes/compiled model and FK are not reconstructed; model gate checks known scene/controller bytes and recorded runtime evidence."
      )
    final_stat = source.stat()
    if (recorded_stat.st_size, recorded_stat.st_mtime_ns) != (
      final_stat.st_size,
      final_stat.st_mtime_ns,
    ):
      errors.append("source file changed during read-only audit")
  except (
    AttributeError,
    OSError,
    KeyError,
    ValueError,
    TypeError,
    IndexError,
    ET.ParseError,
    np.linalg.LinAlgError,
  ) as error:
    errors.append(f"{type(error).__name__}: {error}")
  report["valid"] = not errors
  return report
