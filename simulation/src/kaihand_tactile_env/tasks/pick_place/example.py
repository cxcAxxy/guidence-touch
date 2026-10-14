"""Record and publish one pick/place episode with an offline video/force review."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import h5py
import mujoco
import numpy as np

from ...shared.config import (
  TRAINING_CAMERA_NAMES,
  CameraConfig,
  WorkcellConfig,
  model_fingerprint,
)
from ...shared.contact_tactile import SolverDistributedTactileProvider
from ...shared.poker_review import (
  _probe_video,
  _sha256,
  _validate_probe,
  compose_review_frame,
  plan_review_frames,
)
from ...shared.recording import (
  EpisodeRecorder,
  _qpos_names,
  validate_episode,
  wait_until_object_stable,
)
from ...shared.render_backend import prepare_render_backend
from ...shared.rendering import WorkcellRenderer
from ...shared.simulation import ArmHandSimulation
from ...shared.tactile import RIGHT_FINGERTIP_LINK_NAMES
from ...shared.task_video import _FfmpegPipeWriter
from .task import KnownStateGraspPlanner, PickPlaceExecutor, cylinder_is_in_box

FINGERS = ("thumb", "index", "middle", "ring", "little")
CAMERAS = TRAINING_CAMERA_NAMES
RECORDING_CONTRACT = "pick_place_stable_upright_v5"
INSTRUCTION = "Put the light grey cylinder into the white box."


def _json(path, value):
  path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class PickPlaceExampleSimulation(ArmHandSimulation):
  """Expose the existing executor's cached solver clock without extra dynamics."""

  @property
  def observation_time(self):
    # This task steps with mj_step and does not refresh with mj_forward after
    # integration. RGB, FK and contact forces therefore describe t - dt.
    return max(0.0, float(self.data.time) - self.timestep)


class PickPlaceExampleRecorder(EpisodeRecorder):
  """Add measured pad forces without changing the production controller clock."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, capture_taskspace=True, buffer_rows=128, **kwargs)

  def _initialize(self, metadata):
    self.contact_force_provider = SolverDistributedTactileProvider(
      self.sim.model,
      self.sim.genesis_probe_layout,
      link_names=RIGHT_FINGERTIP_LINK_NAMES,
    )
    super()._initialize(metadata)
    self._progress_time = 0.0

  def observe(self, simulation, phase):
    super().observe(simulation, phase)
    if simulation.data.time >= self._progress_time:
      print(f"recording t={simulation.data.time:.2f}s phase={phase}", flush=True)
      self._progress_time = simulation.data.time + 2.0


def _curves(file, destination):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  destination.mkdir()
  group = file["tactile_contact_force"]
  t = group["timestamp"][:]
  fn = group["normal_taxel_force_n"][:].sum(axis=(-2, -1))
  xy = group["tangent_taxel_force_n"][:].sum(axis=(-3, -2))
  ft = np.linalg.norm(xy, axis=-1)
  np.testing.assert_allclose(fn, group["normal_force_n"][:], atol=1e-10)
  np.testing.assert_allclose(xy, group["tangent_force_n"][:], atol=1e-10)
  if not np.isfinite(fn).all() or not np.isfinite(xy).all():
    raise ValueError("nonfinite force samples")
  phases = file["commands/phase"].asstr()[:]
  fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
  changes = np.flatnonzero(phases[1:] != phases[:-1]) + 1
  for j, finger in enumerate(FINGERS):
    for column, (values, label) in enumerate(((fn, "Normal"), (ft, "Tangential"))):
      ax = axes[j, column]
      ax.plot(t, values[:, j], linewidth=0.8)
      ax.set_ylabel(f"{finger} (N)")
      ax.grid(alpha=0.25)
      for k in changes:
        ax.axvline(t[k], color="grey", linewidth=0.5, alpha=0.4)
      if j == 0:
        ax.set_title(label)
      if j == 4:
        ax.set_xlabel("Simulation time (s)")
  fig.suptitle("Pick and place | right fingertip contact forces | 100 Hz, unfiltered")
  fig.tight_layout()
  fig.savefig(destination / "right_hand_force_curves.png", dpi=150)
  plt.close(fig)
  stats = {
    "sample_count": len(t),
    "force_hz": 100,
    "fingers": FINGERS,
    "source": str(file.attrs["contact_force_source"]),
    "normal_peak_n": fn.max(axis=0).tolist(),
    "tangent_peak_n": ft.max(axis=0).tolist(),
    "normal_mean_n": fn.mean(axis=0).tolist(),
    "tangent_mean_n": ft.mean(axis=0).tolist(),
    "phase_start_s": [
      {"phase": phases[k], "time_s": float(t[k])} for k in np.r_[0, changes]
    ],
    "tangent_definition": "norm of sum of signed local tangential force vectors",
    "filter": "none",
  }
  return stats


def _global_video(file, review, frames):
  """Display saved qpos with a global camera; never run another physical task."""
  path = Path(__file__).with_name("scene.xml")
  if file.attrs["model_fingerprint"] != model_fingerprint(path):
    raise ValueError("scene fingerprint changed since capture")
  model = mujoco.MjModel.from_xml_path(str(path))
  if tuple(file["state/full_qpos_names"].asstr()[:]) != tuple(_qpos_names(model)):
    raise ValueError("global replay joint mapping differs from raw")
  data = mujoco.MjData(model)
  camera = mujoco.MjvCamera()
  camera.lookat[:] = (0.25, 0, 0.86)
  camera.distance, camera.azimuth, camera.elevation = 2.25, 135, -20
  option = mujoco.MjvOption()
  option.geomgroup[:] = 0
  option.geomgroup[[0, 1, 2]] = 1
  times = file["state/timestamp"][:]
  sensor_times = np.array([frame.camera_pose_timestamp_s for frame in frames])
  upper = np.searchsorted(times, sensor_times).clip(0, len(times) - 1)
  lower = np.maximum(upper - 1, 0)
  indices = np.where(
    abs(times[lower] - sensor_times) <= abs(times[upper] - sensor_times), lower, upper
  )
  prepare_render_backend()
  destination = review / "robot_global_short_path.mp4"
  writer = _FfmpegPipeWriter(destination, fps=10, width=960, height=540)
  try:
    with mujoco.Renderer(model, width=960, height=540) as renderer:
      renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
      for index in indices:
        data.qpos[:] = file["state/qpos"][index]
        data.time = times[index]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        renderer.update_scene(data, camera=camera, scene_option=option)
        writer.write(renderer.render().copy())
    writer.finish()
  except BaseException:
    writer.abort()
    raise
  probe = _probe_video(destination)
  _validate_probe(probe, count=len(frames), fps=10, width=960, height=540)
  return {
    "mode": "saved_qpos_kinematic_visualization",
    "physics_steps": 0,
    "training_rgb_replaced": False,
    "state_indices": indices.tolist(),
    "maximum_state_vs_sensor_time_error_s": float(
      np.max(abs(times[indices] - sensor_times))
    ),
    "probe": probe,
  }


def export_example(source: Path, root: Path):
  """Publish only the same two MP4s and one force PNG as the poker example."""
  review = root / "review"
  review.mkdir()
  with h5py.File(source, "r") as file:
    stats = _curves(file, root / "curves")
    frames = plan_review_frames(file, fps=10, second_camera="right_wrist")
    force = file["tactile_contact_force"]
    normal_scale = max(0.1, float(force["normal_taxel_force_n"][:].max()))
    tangent_scale = max(
      0.02, float(np.linalg.norm(force["tangent_taxel_force_n"][:], axis=-1).max())
    )
    destination = review / "review.mp4"
    writer = _FfmpegPipeWriter(destination, fps=10, width=1280, height=720)
    try:
      for frame in frames:
        k, i = frame.tactile_index, frame.camera_index
        composite, _ = compose_review_frame(
          file["cameras/head/rgb"][i],
          file["cameras/right_wrist/rgb"][i],
          force["normal_taxel_force_n"][k],
          force["tangent_taxel_force_n"][k],
          frame=frame,
          width=1280,
          height=720,
          phase=file["commands/phase"].asstr()[k],
          heading="PICK AND PLACE",
          second_camera_label="RIGHT WRIST",
          normal_scale_n=normal_scale,
          tangent_scale_n=tangent_scale,
        )
        writer.write(np.asarray(composite))
      writer.finish()
    except BaseException:
      writer.abort()
      raise
    probe = _probe_video(destination)
    _validate_probe(probe, count=len(frames), fps=10, width=1280, height=720)
    summary = {
      "source_sha256": _sha256(source),
      "frame_count": len(frames),
      "video_fps": 10,
      "cameras": CAMERAS,
      "normal_taxel_scale_n": normal_scale,
      "tangent_taxel_scale_n": tangent_scale,
      "frame_alignment": [asdict(frame) for frame in frames],
      "review_video": probe,
      "global_video": _global_video(file, review, frames),
    }
  return stats, summary


def record_example(output: Path, *, seed: int = 0, overwrite: bool = False):
  """Validate a fresh capture before replacing the maintained example."""
  output = output.expanduser().resolve()
  if output.exists():
    if not overwrite:
      raise FileExistsError(f"Use --overwrite to replace {output}")
    marker = output / "manifest.json"
    sidecar = output / "raw/episode_000000_cylinder_right.json"
    legacy = marker.is_file() and json.loads(marker.read_text()).get("schema") in (
      "pick_place_example_v1",
      "pick_place_example_v2",
    )
    current = (
      sidecar.is_file()
      and json.loads(sidecar.read_text()).get("example", {}).get("scene")
      == "pick-place"
    )
    if not (legacy or current):
      raise ValueError(f"Refusing to replace an unrelated directory: {output}")
  output.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix=".pick_place_", dir=output.parent) as tmp:
    partial = Path(tmp) / "example"
    _record_example(partial, seed=seed)
    old = Path(tmp) / "previous"
    if output.exists():
      output.rename(old)
    try:
      partial.rename(output)
    except BaseException:
      if old.exists():
        old.rename(output)
      raise
    if old.exists():
      shutil.rmtree(old)
  print(f"Published {output}", flush=True)


def _record_example(partial: Path, *, seed: int):
  simulation = PickPlaceExampleSimulation(scene="pick-place")
  simulation.reset(seed=seed, object_xy_jitter=0.0, object_yaw_jitter=0.0)
  plan = KnownStateGraspPlanner(simulation).plan_pick_and_place("right")
  capture = WorkcellConfig(
    model_path=simulation.model_path,
    camera_hz=30,
    cameras=tuple(
      CameraConfig(name, depth=False, segmentation=False) for name in CAMERAS
    ),
  )
  (partial / "raw").mkdir(parents=True)
  source = partial / "raw/episode_000000_cylinder_right.h5"
  with (
    WorkcellRenderer(simulation.model, capture.cameras, shadows=False) as renderer,
    PickPlaceExampleRecorder(
      source,
      simulation,
      capture,
      renderer=renderer,
      metadata={
        "recording_contract": RECORDING_CONTRACT,
        "instruction": INSTRUCTION,
        "motion_profile": "compact_quintic_upright_place_v2",
        "seed": seed,
        "episode_index": 0,
        "object": "cylinder",
        "side": "right",
        "object_xy_jitter": 0.0,
        "object_yaw_jitter": 0.0,
        "controller": "KnownStateGraspPlanner + PickPlaceExecutor",
        "grasp_stabilizer": (
          "compliant carry constraint; cylinder upright correction before descent; "
          "disabled before release"
        ),
        "render_shadows": False,
        "initial_posture": "shared.posture.ARM_HOME",
      },
    ) as recorder,
  ):
    recorder.record_initial()
    result = PickPlaceExecutor(simulation, observer=recorder.observe).execute(plan)
    stability = wait_until_object_stable(
      simulation, "cylinder", observer=recorder.observe
    )
    recorder.record_terminal()
    pose = simulation.object_pose("cylinder")
    placed = cylinder_is_in_box(simulation, pose)
    outcome = {
      **asdict(result),
      "success": placed,
      "placed_in_box": placed,
      "final_object_pose": pose.tolist(),
      "box_center": result.box_center.tolist(),
      "final_object_twist": simulation.object_twist("cylinder").tolist(),
      "phases": [*result.phases, "terminal_settle"],
      "terminal_stability": asdict(stability),
    }
    if not placed:
      raise RuntimeError("Cylinder did not finish inside the box")
    recorder.set_outcome(outcome)
  report = validate_episode(source)
  if not report.valid:
    raise ValueError(report.errors)
  print(
    f"Recorded successful episode, {report.duration_seconds:.2f}s; exporting review",
    flush=True,
  )
  stats, summary = export_example(source, partial)
  sidecar = source.with_suffix(".json")
  manifest = json.loads(sidecar.read_text())
  manifest["example"] = {
    "scene": "pick-place",
    "schema": "pick_place_compact_example_v4",
    "recording_contract": RECORDING_CONTRACT,
    "instruction": INSTRUCTION,
    "validation": asdict(report),
    "forces": stats,
    "review": summary,
    "physics_hz": capture.physics_hz,
    "state_hz": capture.control_hz,
    "camera_hz": capture.camera_hz,
    "camera_resolution": [320, 240],
    "source_code_sha256": {
      p.name: _sha256(p)
      for p in (
        Path(__file__),
        Path(__file__).with_name("task.py"),
        simulation.model_path,
        Path(__file__).parents[2] / "shared/posture.py",
        Path(__file__).parents[2] / "shared/mjcf/robot.xml",
        Path(__file__).parents[2] / "shared/recording.py",
      )
    },
    "full_model_policy_loop_verified": False,
  }
  _json(sidecar, manifest)
