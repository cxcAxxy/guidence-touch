"""Refresh the single bulb acceptance example from one successful episode."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from ...shared.config import TRAINING_CAMERA_NAMES, CameraConfig, WorkcellConfig
from ...shared.contact_tactile import SolverDistributedTactileProvider
from ...shared.poker_review import (
  _probe_video,
  _validate_probe,
  compose_review_frame,
  plan_review_frames,
)
from ...shared.recording import EpisodeRecorder, _append, _stream, validate_episode
from ...shared.tactile import RIGHT_FINGERTIP_LINK_NAMES
from ...shared.task_video import _FfmpegPipeWriter
from . import config
from .execution import BulbScrewExecutor
from .task import BulbScrewSimulation

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _json(path, value):
  path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


class BulbExampleRecorder(EpisodeRecorder):
  """Shared raw schema plus task-local seating and tightening measurements."""

  def __init__(self, path, executor, capture_config, *, buffer_rows=128):
    self.executor = executor
    super().__init__(
      path,
      executor.sim,
      capture_config,
      metadata={
        "scene": config.SCENE_NAME,
        "initial_position_randomization": executor.sim.initial_position_randomization,
        "mechanics_version": config.MECHANICS_VERSION,
        "rotation_driver": config.ROTATION_DRIVER,
      },
      capture_taskspace=True,
      buffer_rows=buffer_rows,
    )

  def _values(self):
    e, data = self.executor, self.sim.data
    wrist, rotation = self.sim.current_pose_matrix("right")
    return {
      **asdict(e._state),
      "hand_clockwise_torque_nm": e._hand_torque_nm,
      "grip_goal_n": e._grip_goal_n,
      "tightening_verified": e._tightening_verified,
      "finger_drive_active": e._finger_drive_active,
      "right_arm_joint_goal": self.sim.arm_goal["right"].copy(),
      "right_wrist_position_m": wrist,
      "right_wrist_rotation": rotation,
      "right_finger_joint_position": data.qpos[
        [self.sim._qpos_address[n] for n in e._names]
      ],
      "actuator_control": data.ctrl.copy(),
      "qfrc_applied": data.qfrc_applied.copy(),
      "xfrc_applied": data.xfrc_applied.copy(),
      "eq_active": data.eq_active.copy(),
      "eq_data": self.sim.model.eq_data.copy(),
    }

  def _initialize(self, metadata):
    self.contact_force_provider = SolverDistributedTactileProvider(
      self.sim.model,
      self.sim.genesis_probe_layout,
      link_names=RIGHT_FINGERTIP_LINK_NAMES,
    )
    super()._initialize(metadata)
    group = self._file.create_group("bulb_screw")
    for name, value in self._values().items():
      array = np.asarray(value)
      _stream(group, name, array.shape, array.dtype)

  def _record_state(self, phase):
    super()._record_state(phase)
    for name, value in self._values().items():
      _append(self._file[f"bulb_screw/{name}"], value)


def _forces(file):
  group = file["tactile_contact_force"]
  normal = group["normal_taxel_force_n"][:].sum(axis=(-2, -1))
  xy = group["tangent_taxel_force_n"][:].sum(axis=(-3, -2))
  np.testing.assert_allclose(normal, group["normal_force_n"][:], atol=1e-10)
  np.testing.assert_allclose(xy, group["tangent_force_n"][:], atol=1e-10)
  return group["timestamp"][:], normal, xy, np.linalg.norm(xy, axis=-1)


def _force_variation(t, fn, ft, mask, selection):
  """Describe unfiltered samples; never difference across excluded intervals."""
  adjacent = mask[1:] & mask[:-1]
  if not adjacent.any():
    return None
  return {
    "sample_interval_s": float(np.median(np.diff(t))),
    "samples": int(mask.sum()),
    "selection": selection,
    **{
      f"{name}_{stat}_n": value.tolist()
      for name, force in (("fn", fn), ("ft", ft))
      for stat, value in (
        ("std", force[mask].std(axis=0)),
        (
          "adjacent_rms",
          np.sqrt(np.mean(np.diff(force, axis=0)[adjacent] ** 2, axis=0)),
        ),
      )
    },
  }


def _verify_approach_alignment(phase, pose):
  """Reject a capture whose bulb is off-axis while entering the funnel."""
  entering = (phase == "align_thread") & (pose[:, 2] <= config.SOCKET_FUNNEL_TOP_Z_M)
  if not entering.any():
    raise ValueError("example never descended through the funnel opening")
  lateral = np.linalg.norm(
    pose[entering, :2] - config.SOCKET_MOUTH_POSITION_M[:2], axis=1
  )
  quat = pose[entering, 3:]
  tilt = np.arccos(np.clip(1 - 2 * np.sum(quat[:, 1:3] ** 2, axis=1), -1, 1))
  if not np.isfinite(lateral).all() or lateral.max() > config.CAPTURE_LATERAL_M:
    raise ValueError("bulb is laterally misaligned during example descent")
  if not np.isfinite(tilt).all() or tilt.max() > 0.01:
    raise ValueError("bulb is tilted during example descent")


def _verify_no_interarm_contact(file):
  """The return path must not use the passive left hand as a stop."""
  names = file["model/body_names"].asstr()[:]
  left = np.asarray(
    [name.startswith(("left_arm", "left_hand", "hand_l")) for name in names]
  )
  right = np.asarray(
    [name.startswith(("right_arm", "right_hand", "hand_r")) for name in names]
  )
  events = file["contacts/events"]
  body1, body2 = events["body1_id"][:], events["body2_id"][:]
  if np.any((left[body1] & right[body2]) | (right[body1] & left[body2])):
    raise ValueError("inter-arm contact in bulb example")


def _audit(file, result):
  t, fn, xy, ft = _forces(file)
  phase = file["commands/phase"].asstr()[:]
  task = file["bulb_screw"]
  turns = task["clockwise_turns"][:]
  pose = file["objects/bulb/pose_wxyz"][:]
  _verify_approach_alignment(phase, pose)
  quat = pose[:, 3:]
  w, x, y, z = quat.T
  actual_angle = -np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
  loose = (
    (phase == "turn")
    & (turns > 0.15 * config.TARGET_TURNS)
    & (turns < 0.8 * config.TARGET_TURNS)
  )
  tight = phase == "tighten"
  if not loose.any() or not tight.any():
    raise ValueError("missing loose or tightening measurements")
  metrics = {
    "loose_mean_fn_n": fn[loose].mean(axis=0).tolist(),
    "tighten_mean_fn_n": fn[tight].mean(axis=0).tolist(),
    "loose_mean_ft_n": ft[loose].mean(axis=0).tolist(),
    "tighten_mean_ft_n": ft[tight].mean(axis=0).tolist(),
    "loose_mean_torque_nm": float(task["hand_clockwise_torque_nm"][:][loose].mean()),
    "tighten_mean_torque_nm": float(task["hand_clockwise_torque_nm"][:][tight].mean()),
    "tighten_start_s": float(t[tight][0]),
    "tighten_end_s": float(t[tight][-1]),
    "release_start_s": float(t[phase == "release"][0]),
    "state_samples": len(t),
  }
  # Measure chatter only inside loose turning strokes: intentional unloading
  # during finger exchange and the rising seat load are different phenomena.
  turning = phase == "turn"
  starts = np.flatnonzero(turning & np.r_[True, ~turning[:-1]])
  ends = np.flatnonzero(turning & np.r_[~turning[1:], True])
  core = np.zeros(len(t), dtype=bool)
  for start, end in zip(starts, ends, strict=True):
    core |= (t >= t[start] + 0.2) & (t <= t[end] - 0.1)
  core &= loose
  variation = _force_variation(
    t,
    fn,
    ft,
    core,
    "turn, 15–80% insertion; exclude first 0.2 s and last 0.1 s per stroke",
  )
  if variation is not None:
    metrics["turn_force_variation"] = variation
  transport = {}
  for name in ("lift", "transfer", "align_thread"):
    indices = np.flatnonzero(phase == name)
    if not len(indices):
      continue
    core = (
      (phase == name)
      & (t >= t[indices[0]] + 0.2)
      & (t <= t[indices[-1]] - 0.3)
      & ~task["engaged"][:].astype(bool)
    )
    variation = _force_variation(
      t,
      fn,
      ft,
      core,
      f"{name}, before thread capture; exclude first 0.2 s and last 0.3 s",
    )
    if variation is not None:
      transport[name] = variation
  if transport:
    metrics["transport_force_variation"] = transport
  for values in (t, fn, xy, ft, file["state/qpos"][:], file["state/qvel"][:]):
    if not np.isfinite(values).all():
      raise ValueError("non-finite recorded values")
  if not (np.diff(t) > 0).all() or np.any(fn < -1e-12):
    raise ValueError("invalid force clock or normal forces")
  if not np.all(fn[tight].mean(axis=0) > fn[loose].mean(axis=0)):
    raise ValueError("not all five normal forces increased during tightening")
  if not np.all(ft[tight].mean(axis=0) > ft[loose].mean(axis=0)):
    raise ValueError("not all five tangential forces increased during tightening")
  if np.max(fn[-1]) > 0.01 or np.max(ft[-1]) > 0.01:
    raise ValueError("fingers remain loaded after release")
  if np.any(task["xfrc_applied"][:]):
    raise ValueError("unexpected external object wrench")
  if not result.success or not result.tightening_verified or result.elapsed_s >= 60:
    raise ValueError(
      "example must complete with verified tightening in under 60 sim seconds"
    )
  # Independently recover a loaded, almost stationary window from recorded data.
  torque = task["hand_clockwise_torque_nm"][:]
  found = False
  for end in np.flatnonzero(tight):
    start = np.searchsorted(t, t[end] - config.TIGHTENING_HOLD_S - 1e-9)
    window = slice(start, end + 1)
    if (
      t[end] - t[start] >= config.TIGHTENING_HOLD_S - 1e-8
      and tight[window].all()
      and np.ptp(actual_angle[window]) <= config.TIGHTENING_MAX_ROTATION_RAD
      and torque[window].min() >= config.TIGHTENING_MIN_TORQUE_NM
      and fn[window].min() >= 4
      and task["seated"][window].all()
    ):
      metrics["loaded_stall_window_s"] = [float(t[start]), float(t[end])]
      metrics["stall_rotation_degrees"] = float(
        np.rad2deg(np.ptp(actual_angle[window]))
      )
      metrics["stall_min_torque_nm"] = float(torque[window].min())
      found = True
  if not found:
    raise ValueError("raw data did not independently confirm loaded angular stall")
  return metrics


def _finger_motion(file):
  task = file["bulb_screw"]
  active = task["finger_drive_active"][:].astype(bool)
  if not active.any():
    raise ValueError("missing finger-driven motion")
  goals = task["right_arm_joint_goal"][:][active]
  position = task["right_wrist_position_m"][:][active]
  rotation = task["right_wrist_rotation"][:][active]
  relative = rotation @ rotation[0].T
  wrist_angle = np.rad2deg(
    np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1))
  )
  wrist_distance = np.linalg.norm(position - position[0], axis=1)
  joints = task["right_finger_joint_position"][:][active].reshape(-1, 5, 4)
  finger_range = np.rad2deg(np.ptp(joints, axis=0).max(axis=1))
  if np.max(np.abs(goals - goals[0])) > 1e-12:
    raise ValueError("arm targets moved during finger-driven screwing")
  if wrist_angle.max() > 1 or wrist_distance.max() > 0.004:
    raise ValueError("excessive wrist motion during finger-driven screwing")
  if np.min(finger_range) < 5:
    raise ValueError("not all five fingers moved through a useful range")
  return {
    "rotation_driver": config.ROTATION_DRIVER,
    "arm_goal_max_change_rad": float(np.max(np.abs(goals - goals[0]))),
    "recorded_wrist_rotation_deg": float(wrist_angle.max()),
    "recorded_wrist_displacement_mm": float(wrist_distance.max() * 1000),
    "finger_joint_range_deg": finger_range.tolist(),
  }


def _curves(file, output, metrics):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  output.mkdir(exist_ok=True)
  t, fn, _, ft = _forces(file)
  fig, axes = plt.subplots(5, 2, figsize=(14, 11), sharex=True)
  for i, finger in enumerate(FINGERS):
    for col, (values, label) in enumerate(((fn, "Fn (N)"), (ft, "|Ft| (N)"))):
      ax = axes[i, col]
      ax.plot(t, values[:, i], lw=0.65)
      ax.axvspan(
        metrics["tighten_start_s"], metrics["tighten_end_s"], alpha=0.17, color="orange"
      )
      ax.set_ylabel(f"{finger}\n{label}")
      ax.grid(alpha=0.25)
  for ax in axes[-1]:
    ax.set_xlabel("Simulation time (s)")
  fig.suptitle(
    "Light bulb: raw right-hand contact forces (100 Hz); orange = tightening"
  )
  fig.tight_layout()
  fig.savefig(output / "right_hand_force_curves.png", dpi=150)
  plt.close(fig)


def _videos(file, output):
  output.mkdir()
  np.testing.assert_array_equal(
    file["cameras/left_wrist/timestamp"][:], file["cameras/head/timestamp"][:]
  )
  frames = plan_review_frames(file, fps=10, second_camera="right_wrist")
  force = file["tactile_contact_force"]
  writer = _FfmpegPipeWriter(output / "review.mp4", fps=10, width=1280, height=720)
  try:
    for frame in frames:
      i, k = frame.camera_index, frame.tactile_index
      phase = file["commands/phase"].asstr()[k]
      composite, _ = compose_review_frame(
        file["cameras/head/rgb"][i],
        file["cameras/right_wrist/rgb"][i],
        force["normal_taxel_force_n"][k],
        force["tangent_taxel_force_n"][k],
        frame=frame,
        width=1280,
        height=720,
        phase=phase,
        heading="LIGHT BULB",
        second_camera_label="RIGHT WRIST",
        normal_scale_n=1.0,
        tangent_scale_n=0.8,
        caption="Raw solver forces; signed Ft in HDF5. Fixed heatmap scale may saturate; curves retain full values.",
      )
      writer.write(np.asarray(composite))
  finally:
    writer.finish()
  _validate_probe(
    _probe_video(output / "review.mp4"),
    count=len(frames),
    fps=10,
    width=1280,
    height=720,
  )
  return {
    "frames": len(frames),
    "fps": 10,
    "maximum_tactile_age_s": max(f.tactile_age_s for f in frames),
  }


def refresh_example(
  output: Path,
  *,
  position_seed=None,
  camera_hz: int = 30,
  cameras: tuple[str, ...] = TRAINING_CAMERA_NAMES,
):
  """Build privately, validate, then replace the current example without history."""
  output = output.resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  if output.exists():
    marker = output / "raw/light_bulb_000000.h5"
    if not marker.is_file():
      raise ValueError(f"refusing to replace an unrelated directory: {output}")
    with h5py.File(marker, "r") as previous:
      scene = json.loads(previous.attrs["metadata_json"]).get("scene")
    if scene != config.SCENE_NAME:
      raise ValueError(f"refusing to replace an unrelated directory: {output}")
  started = time.monotonic()
  with tempfile.TemporaryDirectory(
    prefix=".light_bulb_", dir=output.parent
  ) as temporary:
    staging = Path(temporary) / "example"
    (staging / "raw").mkdir(parents=True)
    sim = BulbScrewSimulation(position_seed=position_seed)
    executor = BulbScrewExecutor(sim, grasp_mode="five-finger", speed="fast")
    capture = WorkcellConfig(
      model_path=sim.model_path,
      control_hz=100,
      camera_hz=camera_hz,
      cameras=tuple(CameraConfig(n, depth=False, segmentation=False) for n in cameras),
    )
    raw = staging / "raw/light_bulb_000000.h5"
    with BulbExampleRecorder(raw, executor, capture) as recorder:
      last_progress = 0.0

      def observe(simulation, phase):
        nonlocal last_progress
        recorder.observe(simulation, phase)
        if simulation.data.time - last_progress >= 5:
          print(
            f"recording {simulation.data.time:.1f} sim s | {phase} | {time.monotonic() - started:.0f} wall s",
            flush=True,
          )
          last_progress = simulation.data.time

      executor.observer = observe
      recorder.record_initial()
      result = executor.execute()
      recorder.record_terminal("verify_seated")
      recorder.set_outcome(asdict(result))
    if not result.success:
      raise RuntimeError(result.reason)
    report = validate_episode(raw)
    if not report.valid:
      raise ValueError(report.errors)
    print("Episode complete; checking forces and exporting curves/videos", flush=True)
    with h5py.File(raw, "r") as file:
      metrics = _audit(file, result)
      _verify_no_interarm_contact(file)
      metrics.update(_finger_motion(file))
      lit = file["bulb_screw/bulb_lit"][:].astype(bool)
      verified = file["bulb_screw/tightening_verified"][:].astype(bool)
      if not lit.any() or not lit[-1] or np.any(lit & ~verified):
        raise ValueError("completion light must follow verified tightening")
      first_lit = int(np.flatnonzero(lit)[0])
      if not lit[first_lit:].all():
        raise ValueError("completion light did not remain on after tightening")
      metrics["bulb_light_on_s"] = float(
        file["tactile_contact_force/timestamp"][first_lit]
      )
      phase = file["commands/phase"].asstr()[:]
      lit_tighten = lit & np.isin(phase, ("tighten", "hold_tight"))
      if phase[first_lit] != "tighten" or not lit_tighten.any():
        raise ValueError("bulb must light while tightening, before release")
      time_values = file["tactile_contact_force/timestamp"][:]
      metrics["bulb_lit_before_release_s"] = float(
        time_values[phase == "release"][0] - time_values[first_lit]
      )
      if metrics["bulb_lit_before_release_s"] < config.TIGHTENING_LIGHT_HOLD_S - 0.01:
        raise ValueError("missing loaded hold between illumination and release")
      if (
        file["tactile_contact_force/normal_force_n"][:][lit_tighten].min() < 4
        or file["bulb_screw/hand_clockwise_torque_nm"][:][lit_tighten].min()
        < config.TIGHTENING_MIN_TORQUE_NM
      ):
        raise ValueError("bulb must illuminate with the fingers still loaded")
      _curves(file, staging / "curves", metrics)
      metrics["video"] = _videos(file, staging / "review")
    metrics.update(
      {
        "scene": config.SCENE_NAME,
        "mechanics_version": config.MECHANICS_VERSION,
        "result": asdict(result),
        "validation": asdict(report),
        "initial_position_randomization": sim.initial_position_randomization,
        "refresh_wall_seconds": time.monotonic() - started,
        "source_sha256": {
          p.name: hashlib.sha256(p.read_bytes()).hexdigest()
          for p in Path(__file__).parent.glob("*.py")
        },
      }
    )
    metrics["source_sha256"]["scene.xml"] = hashlib.sha256(
      Path(__file__).with_name("scene.xml").read_bytes()
    ).hexdigest()
    # The compact example keeps the audit alongside the standard outcome.
    # The top-level outcome keys remain usable by the collection adapter.
    _json(
      raw.with_suffix(".result.json"),
      {**asdict(result), "example_validation": metrics},
    )
    sidecar = raw.with_suffix(".json")
    manifest = json.loads(sidecar.read_text())
    manifest["validation"] = asdict(report)
    _json(sidecar, manifest)
    old = Path(temporary) / "previous"
    if output.exists():
      output.rename(old)
    try:
      staging.rename(output)
    except BaseException:
      if old.exists():
        old.rename(output)
      raise
    if old.exists():
      shutil.rmtree(old)
  print(f"Updated {output}", flush=True)
  return metrics


def record_raw_episode(
  output: Path,
  *,
  position_seed=None,
  buffer_rows=128,
  camera_hz: int = 30,
  cameras: tuple[str, ...] = TRAINING_CAMERA_NAMES,
):
  """Record one validated training episode without review derivatives."""
  output = output.resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  if output.exists() or output.is_symlink():
    raise FileExistsError(f"raw-only output must be new: {output}")
  started = time.monotonic()
  with tempfile.TemporaryDirectory(
    prefix=".light_bulb_raw_", dir=output.parent
  ) as temporary:
    staging = Path(temporary) / "episode"
    (staging / "raw").mkdir(parents=True)
    sim = BulbScrewSimulation(position_seed=position_seed)
    executor = BulbScrewExecutor(sim, grasp_mode="five-finger", speed="fast")
    capture = WorkcellConfig(
      model_path=sim.model_path,
      control_hz=100,
      camera_hz=camera_hz,
      cameras=tuple(
        CameraConfig(name, depth=False, segmentation=False) for name in cameras
      ),
    )
    raw = staging / "raw/light_bulb_000000.h5"
    with BulbExampleRecorder(
      raw, executor, capture, buffer_rows=buffer_rows
    ) as recorder:
      executor.observer = lambda simulation, phase: recorder.observe(simulation, phase)
      recorder.record_initial()
      result = executor.execute()
      recorder.record_terminal("verify_seated")
      recorder.set_outcome(asdict(result))
    if not result.success:
      raise RuntimeError(result.reason)
    _json(raw.with_suffix(".result.json"), asdict(result))
    validation = validate_episode(raw)
    if not validation.valid:
      raise ValueError(validation.errors)
    with h5py.File(raw, "r") as file:
      audit = _audit(file, result)
    validation_sidecar = raw.with_suffix(".json")
    sidecar = json.loads(validation_sidecar.read_text())
    sidecar.update(
      validation=asdict(validation),
      task_audit_passed=True,
      recording_wall_seconds=time.monotonic() - started,
    )
    _json(validation_sidecar, sidecar)
    staging.rename(output)
  print(f"Saved validated raw-only bulb episode: {output}", flush=True)
  return {"result": asdict(result), "validation": asdict(validation), "audit": audit}
