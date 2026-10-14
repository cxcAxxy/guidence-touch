"""Shared Raw and detailed review capture for whiteboard wiping."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import h5py
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from ...shared.config import CameraConfig, WorkcellConfig
from ...shared.poker_review import (
  _force_layout,
  _probe_video,
  _validate_probe,
  compose_review_frame,
  plan_review_frames,
)
from ...shared.recording import (
  EpisodeRecorder,
  WristWrenchSample,
  WristWrenchSensor,
  _append,
  _stream,
  create_wrist_wrench_group,
  validate_episode,
)
from ...shared.render_backend import prepare_render_backend
from ...shared.rendering import WorkcellRenderer
from ...shared.tactile import SolverContactTactileProvider
from ...shared.task_video import _FfmpegPipeWriter
from . import config

FINGERS = ("thumb", "index", "middle", "ring", "little")
SCHEMA = "erase_whiteboard_example_v2"
FORCE_HZ = 500


def _json(path, value):
  Path(path).write_text(
    json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    encoding="utf-8",
  )


def _sha256(path):
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _force_maps(sample, tangent=False):
  """Use the vase example's exact panel layout, labels and fixed color scales."""
  import matplotlib

  values = (
    np.linalg.norm(sample["tangent_taxel_force_n"], axis=-1)
    if tangent
    else sample["normal_taxel_force_n"]
  )
  limit = 0.05 if tangent else 0.4
  cmap = matplotlib.colormaps["magma" if tangent else "viridis"]
  panel = Image.new("RGB", (640, 360), "#101923")
  draw = ImageDraw.Draw(panel)
  draw.text(
    (15, 12),
    "RIGHT HAND | " + ("TANGENTIAL" if tangent else "NORMAL") + " TAXEL FORCE (N)",
    fill="white",
  )
  for i, finger in enumerate(FINGERS):
    colors = (cmap(np.clip(values[i] / limit, 0, 1))[..., :3] * 255).astype("uint8")
    panel.paste(
      Image.fromarray(colors).resize((90, 196), Image.Resampling.NEAREST),
      (15 + i * 125, 62),
    )
    draw.text((15 + i * 125, 40), finger, fill="white")
    fn, xy = sample["normal_force_n"][i], sample["tangent_force_n"][i]
    total = np.linalg.norm(xy) if tangent else fn
    draw.text((15 + i * 125, 270), f"{total:.3f} N", fill="white")
    if tangent:
      draw.text((15 + i * 125, 292), f"Fx {xy[0]:+.3f}", fill="white")
      draw.text((15 + i * 125, 310), f"Fy {xy[1]:+.3f}", fill="white")
  draw.text(
    (15, 342),
    f"Color scale: 0 .. {limit} N/taxel; raw data never clipped",
    fill="#bac8d7",
  )
  return panel


class WhiteboardEpisodeRecorder(EpisodeRecorder):
  """Shared Raw schema plus whiteboard-specific cleaning and contact state."""

  def __init__(self, path, simulation, capture_config, *, buffer_rows=128):
    super().__init__(
      path,
      simulation,
      capture_config,
      metadata={
        "scene": config.SCENE_NAME,
        "recording_contract": "whiteboard_wipe_shared_raw_v1",
        "ink_randomization": simulation.ink_randomization,
      },
      capture_taskspace=True,
      buffer_rows=buffer_rows,
    )

  def _values(self):
    sim = self.sim
    cleaning = sim.cleaning
    return {
      "qfrc_applied": sim.data.qfrc_applied.copy(),
      "xfrc_applied": sim.data.xfrc_applied.copy(),
      "ink_remaining": sim.remaining.copy(),
      "ink_rgba": sim.model.geom_rgba[sim.ink_ids].copy(),
      "patch_work_j": cleaning.work_j.copy(),
      "patch_stroke_m": cleaning.stroke_m.copy(),
      "patch_loaded_time_s": cleaning.loaded_time_s.copy(),
      "patch_contact_tangent_load_n": sim.patch_tangent_load.copy(),
      "patch_sliding_speed_m_s": sim.patch_speed.copy(),
      "patch_friction_power_w": sim.patch_power.copy(),
      "board_force_n": sim.board_force,
      "board_tangent_force_n": sim.board_tangent_force,
      "board_contact_center_world_m": sim.board_contact_center.copy(),
      "board_contact_torque_world_nm": sim.board_contact_torque.copy(),
      "board_contact_count": sim.board_contact_count,
      "table_support_force_n": sim.table_force,
      "direct_hand_board_force_n": sim.direct_hand_board_force,
    }

  def _initialize(self, metadata):
    # Preserve the task's measured ten-finger force maps in the common group.
    self.contact_force_provider = self.sim.forces
    super()._initialize(metadata)
    group = self._file.create_group("whiteboard_wipe")
    group.attrs["force_source"] = self.sim.forces.source
    group.attrs["ink_randomization_json"] = json.dumps(
      self.sim.ink_randomization, sort_keys=True
    )
    for name, value in self._values().items():
      array = np.asarray(value)
      _stream(group, name, array.shape, array.dtype)

  def _record_state(self, phase):
    super()._record_state(phase)
    for name, value in self._values().items():
      _append(self._file[f"whiteboard_wipe/{name}"], value)


class RawCapture:
  """Three-camera shared HDF5 without MP4, frames, curves, CSV or HTML."""

  def __init__(
    self,
    simulation,
    directory,
    *,
    camera_hz=30,
    cameras=("head", "left_wrist", "right_wrist"),
    buffer_rows=128,
    episode_stem="episode",
    result_relative="result.json",
  ):
    self.sim = simulation
    self.directory = Path(directory)
    self.directory.mkdir(parents=True, exist_ok=False)
    (self.directory / "raw").mkdir()
    capture = WorkcellConfig(
      model_path=simulation.model_path,
      physics_hz=round(1.0 / simulation.timestep),
      # Match the shared training Raw clock used by the existing semantic
      # adapters. The dedicated review recorder below remains native 500 Hz.
      control_hz=100,
      camera_hz=camera_hz,
      cameras=tuple(
        CameraConfig(name, width=320, height=240, depth=False, segmentation=False)
        for name in cameras
      ),
      tactile_provider=SolverContactTactileProvider.source,
    )
    self.result_path = self.directory / result_relative
    self.result_path.parent.mkdir(parents=True, exist_ok=True)
    self.raw_recorder = WhiteboardEpisodeRecorder(
      self.directory / f"raw/{episode_stem}.h5",
      simulation,
      capture,
      buffer_rows=buffer_rows,
    )
    self.raw_recorder.record_initial("tabletop_ready")
    simulation.physics_observer = self.record

  def record(self, simulation):
    self.raw_recorder.observe(
      simulation, getattr(simulation, "phase", "tabletop_ready")
    )

  def observe(self, simulation, phase):
    simulation.phase = phase

  def finish(self, result):
    self.raw_recorder.record_terminal("terminal_settle")
    self.raw_recorder.set_outcome(result)
    self.raw_recorder.close()
    validation = validate_episode(self.raw_recorder.path)
    validation_result = {
      "valid": validation.valid,
      "errors": list(validation.errors),
      "warnings": list(validation.warnings),
      "state_samples": validation.state_samples,
      "camera_samples": validation.camera_samples,
    }
    _json(self.result_path, result)
    validation_sidecar = self.raw_recorder.path.with_suffix(".json")
    sidecar = json.loads(validation_sidecar.read_text(encoding="utf-8"))
    sidecar.update(
      validation=validation_result,
      task_audit_passed=result.get("success") is True,
      ink_randomization=self.sim.ink_randomization,
    )
    _json(validation_sidecar, sidecar)
    if not validation.valid:
      raise ValueError(f"shared raw validation failed: {validation.errors}")
    return validation_result

  def close(self):
    self.sim.physics_observer = None
    if not self.raw_recorder._closed:
      self.raw_recorder.close(finalize=False)


def _compact_curves(file, output):
  """Plot unfiltered recorded right-hand totals on the shared 100 Hz clock."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.patches import Patch

  indices = _force_layout(file)
  force = file["tactile_contact_force"]
  time_s = force["timestamp"][:]
  normal = force["normal_force_n"][:, indices]
  tangent = np.linalg.norm(force["tangent_force_n"][:, indices], axis=-1)
  phases = file["commands/phase"].asstr()[:]
  boundaries = time_s[np.flatnonzero(phases[1:] != phases[:-1]) + 1]
  fig, axes = plt.subplots(5, 2, figsize=(14, 11), sharex=True)
  for finger, row in enumerate(axes):
    for axis, values, label in zip(
      row, (normal, tangent), ("Fn (N)", "|Ft| (N)"), strict=True
    ):
      axis.plot(time_s, values[:, finger], lw=0.75)
      axis.set_ylabel(f"{FINGERS[finger]}\n{label}")
      axis.grid(alpha=0.25)
      for phase, color in (("load_board", "#F4A340"), ("wipe", "#5B8FF9")):
        active = phases == phase
        if np.any(active):
          changes = np.diff(np.r_[False, active, False].astype(np.int8))
          for start, end in zip(
            np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True
          ):
            axis.axvspan(time_s[start], time_s[end - 1], color=color, alpha=0.18)
      for boundary in boundaries:
        axis.axvline(boundary, color="gray", lw=0.35, alpha=0.45)
  for axis in axes[-1]:
    axis.set_xlabel("Simulation time (s)")
  fig.suptitle(
    "Whiteboard wiping: recorded right-hand solver forces (100 Hz, unfiltered)"
  )
  output.mkdir(parents=True, exist_ok=True)
  fig.legend(
    handles=[
      Patch(color="#F4A340", alpha=0.35, label="Moving load transfer"),
      Patch(color="#5B8FF9", alpha=0.35, label="Stable wipe"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.975),
    ncol=2,
  )
  fig.tight_layout(rect=(0, 0, 1, 0.94))
  fig.savefig(output / "right_hand_force_curves.png", dpi=150)
  plt.close(fig)


def _compact_review(file, output):
  """Encode head, right wrist, and synchronized normal/tangent tactile panels."""
  frames = plan_review_frames(file, fps=10, second_camera="right_wrist")
  right = _force_layout(file)
  force = file["tactile_contact_force"]
  writer = _FfmpegPipeWriter(output / "review.mp4", fps=10, width=1280, height=720)
  try:
    for frame in frames:
      camera_index, tactile_index = frame.camera_index, frame.tactile_index
      normal = force["normal_taxel_force_n"][tactile_index][right]
      tangent = force["tangent_taxel_force_n"][tactile_index][right]
      phase = file["commands/phase"].asstr()[tactile_index]
      composite, _ = compose_review_frame(
        file["cameras/head/rgb"][camera_index],
        file["cameras/right_wrist/rgb"][camera_index],
        normal,
        tangent,
        frame=frame,
        width=1280,
        height=720,
        phase=phase,
        heading="WHITEBOARD WIPING",
        second_camera_label="RIGHT WRIST",
      )
      writer.write(np.asarray(composite))
    writer.finish()
  except BaseException:
    writer.abort()
    raise
  probe = _probe_video(output / "review.mp4")
  _validate_probe(probe, count=len(frames), fps=10, width=1280, height=720)
  return {
    "source": "../raw/erase_whiteboard_000000.h5",
    "frame_count": len(frames),
    "fps": 10,
    "camera_names": ["head", "right_wrist"],
    "tactile_panels": ["right_hand_normal", "right_hand_tangent"],
    "maximum_tactile_age_s": max(frame.tactile_age_s for frame in frames),
    "video_validation": probe,
  }


def _compact_global_review(file, output):
  """Render the exact saved states from a fixed presentation camera."""
  from .task import WhiteboardWipeSimulation

  sim = WhiteboardWipeSimulation()
  layout = json.loads(file["whiteboard_wipe"].attrs["ink_randomization_json"])
  sim.set_ink_layout(layout["geom_positions_board_m"], seed=layout["seed"])
  frames = plan_review_frames(file, fps=10, second_camera="right_wrist")
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.lookat[:] = (0.25, -0.03, 0.85)
  camera.distance = 2.25
  camera.azimuth = 135
  camera.elevation = -22
  option = mujoco.MjvOption()
  option.geomgroup[3:] = 0
  prepare_render_backend()
  path = output / "robot_global_short_path.mp4"
  writer = _FfmpegPipeWriter(path, fps=10, width=960, height=540)
  try:
    with mujoco.Renderer(sim.model, width=960, height=540) as renderer:
      for frame in frames:
        index = int(file["cameras/head/state_index"][frame.camera_index])
        sim.data.time = float(file["state/timestamp"][index])
        sim.data.qpos[:] = file["state/qpos"][index]
        sim.data.qvel[:] = file["state/qvel"][index]
        sim.data.ctrl[:] = file["commands/actuator_control"][index]
        sim.model.geom_rgba[sim.ink_ids] = file["whiteboard_wipe/ink_rgba"][index]
        mujoco.mj_forward(sim.model, sim.data)
        renderer.update_scene(sim.data, camera=camera, scene_option=option)
        writer.write(np.asarray(renderer.render()))
    writer.finish()
  except BaseException:
    writer.abort()
    raise
  probe = _probe_video(path)
  _validate_probe(probe, count=len(frames), fps=10, width=960, height=540)
  return {
    "source": "../raw/erase_whiteboard_000000.h5",
    "method": "offline rendering of exact saved states without physics stepping",
    "frame_count": len(frames),
    "video_validation": probe,
  }


def _compact_force_analysis(file):
  """Measure phase-wise levels and changes; fail publication on implausible loads."""
  right = _force_layout(file)
  force = file["tactile_contact_force"]
  time_s = force["timestamp"][:]
  phases = file["commands/phase"].asstr()[:]
  normal = force["normal_force_n"][:, right]
  tangent = np.linalg.norm(force["tangent_force_n"][:, right], axis=-1)
  board = file["whiteboard_wipe/board_force_n"][:]
  names = list(FINGERS)

  def changes(values):
    delta = np.abs(np.diff(values, axis=0))
    flat = int(np.argmax(delta))
    row, column = np.unravel_index(flat, delta.shape)
    return {
      "maximum_10ms_step_n": float(delta[row, column]),
      "time_s": float(time_s[row + 1]),
      "finger": names[column],
      "phase_before": str(phases[row]),
      "phase_after": str(phases[row + 1]),
    }

  def phase_changes(values, phase):
    # Compare only adjacent samples in one continuous phase. The two wipe
    # strokes are separated by an unloaded row change and must not be joined.
    adjacent = (phases[:-1] == phase) & (phases[1:] == phase)
    if not np.any(adjacent):
      return 0.0
    return float(np.abs(np.diff(values, axis=0)[adjacent]).max())

  def window(center_s, half_width_s=1.0):
    selected = np.flatnonzero(np.abs(time_s - center_s) <= half_width_s)
    if not len(selected):
      return {
        "available": False,
        "reason": f"episode ends at {time_s[-1]:.2f} s",
      }
    return {
      "available": True,
      "range_s": [float(time_s[selected[0]]), float(time_s[selected[-1]])],
      "phases": list(dict.fromkeys(str(value) for value in phases[selected])),
      "peak_board_normal_n": float(board[selected].max()),
      "right_fn_peak_to_peak_n": np.ptp(normal[selected], axis=0).tolist(),
      "right_ft_peak_to_peak_n": np.ptp(tangent[selected], axis=0).tolist(),
    }

  phase_summary = {}
  for phase in dict.fromkeys(phases):
    selected = np.flatnonzero(phases == phase)
    phase_summary[phase] = {
      "sample_count": int(len(selected)),
      "duration_s": float(len(selected) / float(file.attrs["control_hz"])),
      "peak_board_normal_n": float(board[selected].max()),
      "peak_right_fn_n": normal[selected].max(axis=0).tolist(),
      "peak_right_ft_n": tangent[selected].max(axis=0).tolist(),
    }
  wipe = phases == "wipe"
  load = phases == "load_board"
  pregrasp = np.isin(
    phases, ("open_hand", "move_above_eraser", "descend_to_eraser")
  )
  grasp = np.flatnonzero(phases == "grasp")
  release = np.flatnonzero(phases == "release")
  expected_pickup = (
    "open_hand", "move_above_eraser", "descend_to_eraser", "grasp", "lift"
  )
  phase_transitions = tuple(dict.fromkeys(phases))
  pickup_start = (
    phase_transitions.index("open_hand")
    if "open_hand" in phase_transitions else -1
  )
  taxel_normal = force["normal_taxel_force_n"][:]
  taxel_totals_match = np.allclose(
    taxel_normal.sum(axis=(-2, -1)), force["normal_force_n"][:],
    rtol=0, atol=1e-9,
  )
  direct_hand_board = file["whiteboard_wipe/direct_hand_board_force_n"][:]
  wipe_loaded = board[wipe]
  loaded_start = np.flatnonzero(wipe_loaded > 0.5)
  loaded_fraction = (
    float(np.mean(wipe_loaded[loaded_start[0]:] > 0.01))
    if len(loaded_start) else 0.0
  )
  wipe_fn_step = phase_changes(normal, "wipe")
  wipe_ft_step = phase_changes(tangent, "wipe")
  load_fn_step = phase_changes(normal, "load_board")
  load_ft_step = phase_changes(tangent, "load_board")
  criteria = {
    "pickup_phase_order": bool(
      pickup_start >= 0
      and phase_transitions[pickup_start:pickup_start + 5] == expected_pickup
    ),
    "pregrasp_has_no_fingertip_contact": bool(
      np.any(pregrasp) and np.max(normal[pregrasp]) < 0.05
    ),
    "grasp_has_opposed_fingertip_contact": bool(
      len(grasp)
      and normal[grasp[-1], 0] >= 0.05
      and normal[grasp[-1], 1:].sum() >= 0.15
    ),
    "wipe_retains_opposed_fingertip_contact": bool(
      np.any(wipe)
      and np.mean(
        (normal[wipe, 0] >= 0.05)
        & (normal[wipe, 1:].sum(axis=1) >= 0.15)
      ) > 0.99
    ),
    "wipe_board_contact_continuity": bool(loaded_fraction > 0.99),
    "no_direct_hand_board_contact": bool(np.max(direct_hand_board) < 1e-9),
    "release_unloads_fingertips": bool(
      len(release) and normal[release[-1]].sum() < 0.05
    ),
    "taxel_normal_force_conservation": bool(taxel_totals_match),
    "episode_fingertip_peak_below_6_n": bool(normal.max() < 6.0),
    "wipe_board_peak_below_3_n": bool(board[wipe].max() < 3.0),
    "load_board_peak_below_5_n": bool(board[load].max() < 5.0),
    "load_board_fn_10ms_step_below_0_5_n": bool(load_fn_step < 0.5),
    "load_board_ft_10ms_step_below_0_5_n": bool(load_ft_step < 0.5),
    "wipe_fn_10ms_step_below_0_5_n": bool(wipe_fn_step < 0.5),
    "wipe_ft_10ms_step_below_0_5_n": bool(wipe_ft_step < 0.5),
  }
  return {
    "source": "unfiltered solver-contact forces on the shared 100 Hz state clock",
    "phase_summary": phase_summary,
    "right_fn_change": changes(normal),
    "right_ft_change": changes(tangent),
    "phase_maximum_10ms_step_n": {
      "moving_load_transfer": {
        "normal": load_fn_step,
        "tangent": load_ft_step,
      },
      "stable_wipe": {
        "normal": wipe_fn_step,
        "tangent": wipe_ft_step,
      },
    },
    "inspection_windows": {
      "40s_plus_minus_1s": window(40.0),
      "52s_plus_minus_1s": window(52.0),
      "60s_plus_minus_1s": window(60.0),
    },
    "peak_board_normal_n": float(board.max()),
    "pregrasp_peak_fingertip_force_n": normal[pregrasp].max(axis=0).tolist(),
    "grasp_final_fingertip_force_n": normal[grasp[-1]].tolist(),
    "wipe_loaded_fraction_after_first_contact": loaded_fraction,
    "release_final_fingertip_force_n": normal[release[-1]].tolist(),
    "criteria": criteria,
    "reasonable": all(criteria.values()),
    "assessment": (
      "Pickup, grasp, wiping and release follow the measured contact sequence; "
      "taxel forces conserve the solver totals and board/fingertip loads pass bounds."
      if all(criteria.values())
      else "One or more measured action or contact criteria failed; do not publish."
    ),
  }


def record_compact_example(
  output,
  *,
  replace_existing=False,
  camera_hz=30,
  cameras=("head", "left_wrist", "right_wrist"),
  buffer_rows=128,
  ink_seed=None,
):
  """Record and atomically publish the light-bulb-style five-file example."""
  from .execution import WhiteboardWipeExecutor
  from .task import WhiteboardWipeSimulation

  output = Path(output).expanduser().absolute()
  work = output.with_name(output.name + ".rendering")
  replaced = output.with_name(output.name + ".replaced")
  for path in (output, work, replaced):
    if path.is_symlink():
      raise ValueError(f"refusing a symlink output: {path}")
  if work.exists() or replaced.exists():
    raise FileExistsError("whiteboard staging or replacement path already exists")
  if output.exists() and not replace_existing:
    raise FileExistsError("output exists; explicit replacement is required")
  sim = WhiteboardWipeSimulation(ink_seed=ink_seed)
  recorder = RawCapture(
    sim,
    work,
    camera_hz=camera_hz,
    cameras=cameras,
    buffer_rows=buffer_rows,
    episode_stem="erase_whiteboard_000000",
    result_relative="raw/erase_whiteboard_000000.result.json",
  )
  try:
    result = WhiteboardWipeExecutor(sim, recorder.observe).run()
    validation = recorder.finish(result)
  finally:
    recorder.close()
  try:
    if not result.get("success"):
      raise ValueError(f"whiteboard rollout failed: {result.get('reason')}")
    raw_path = work / "raw/erase_whiteboard_000000.h5"
    (work / "review").mkdir()
    with h5py.File(raw_path, "r") as file:
      _compact_curves(file, work / "curves")
      review = _compact_review(file, work / "review")
      force_analysis = _compact_force_analysis(file)
    if not force_analysis["reasonable"]:
      failed = [
        name for name, passed in force_analysis["criteria"].items() if not passed
      ]
      raise ValueError(f"{force_analysis['assessment']} Failed: {failed}")
    result_path = work / "raw/erase_whiteboard_000000.result.json"
    saved_result = json.loads(result_path.read_text(encoding="utf-8"))
    saved_result["example_validation"] = {
      "raw": validation,
      "force_analysis": force_analysis,
      "review": review,
      "raw_sha256": _sha256(raw_path),
      "layout": "light_bulb_example_raw_review_curves",
    }
    _json(result_path, saved_result)
    expected = {
      "curves/right_hand_force_curves.png",
      "raw/erase_whiteboard_000000.h5",
      "raw/erase_whiteboard_000000.json",
      "raw/erase_whiteboard_000000.result.json",
      "review/review.mp4",
    }
    actual = {
      str(path.relative_to(work)) for path in work.rglob("*") if path.is_file()
    }
    if actual != expected:
      raise ValueError(f"compact layout mismatch: {sorted(actual ^ expected)}")
    existed = output.exists()
    if existed:
      output.rename(replaced)
    try:
      work.rename(output)
    except BaseException:
      if existed:
        replaced.rename(output)
      raise
    if existed:
      shutil.rmtree(replaced)
    return saved_result
  except BaseException:
    if work.exists():
      shutil.rmtree(work)
    raise


class WhiteboardRecorder:
  """Capture once at the physics hook, then render without advancing physics."""

  def __init__(self, simulation, directory, *, video_fps=10):
    if video_fps <= 0 or FORCE_HZ % video_fps:
      raise ValueError("video_fps must be a positive divisor of 500")
    self.previous_observer = getattr(simulation, "physics_observer", None)
    if self.previous_observer is not None:
      raise ValueError("a physics observer is already attached")
    self.video_fps = int(video_fps)
    self.directory = Path(directory)
    self.directory.mkdir(parents=True, exist_ok=False)
    for name in (
      "curves",
      "raw",
      "review/raw",
      "review/inspection",
      "review/frames/head",
      "review/frames/right_wrist",
      "review/frames/board_overview",
      "review/frames/normal",
      "review/frames/tangent",
      "review/frames/composite",
    ):
      (self.directory / name).mkdir(parents=True, exist_ok=True)
    self.sim = simulation
    self.next_sample = float(simulation.data.time)
    self.buffers, self.frame_rows = [], []
    self.datasets = {}
    self.sample_count = 0
    self.source_hashes = {
      path.name: hashlib.sha256(path.read_bytes()).hexdigest()
      for path in sorted(Path(__file__).parent.iterdir())
      if path.is_file() and path.suffix in (".py", ".xml", ".png", ".obj")
    }
    self.h5 = h5py.File(self.directory / "raw/episode.h5", "x")
    self.h5.attrs.update(
      schema=SCHEMA,
      ink_randomization_json=json.dumps(simulation.ink_randomization),
      scene="whiteboard-wipe",
      physics_hz=1 / simulation.timestep,
      control_hz=100,
      force_hz=FORCE_HZ,
      record_dt_s=1 / FORCE_HZ,
      camera_hz=self.video_fps,
      force_source=simulation.forces.source,
      finger_order=json.dumps(FINGERS),
      right_hand_indices=json.dumps(list(range(5, 10))),
      capture_method="one native physical rollout; saved-state RGB rendering",
      force_processing="raw solver forces; no filtering, clipping or interpolation",
      force_evaluation="before integration, after control update; terminal sample evaluated without stepping",
    )
    self.wrist_wrench_sensor = WristWrenchSensor(simulation.model)
    create_wrist_wrench_group(self.h5, h5py.string_dtype(encoding="utf-8"))
    initial = simulation.forces.read(simulation.data)
    group = self.h5.require_group("tactile_contact_force")
    group.create_dataset("link_names", data=np.asarray(initial.link_names, dtype="S"))
    group.create_dataset("normal_axis_local", data=initial.normal_axis_local)
    group.create_dataset("tangent_basis_local", data=initial.tangent_basis_local)
    simulation.physics_observer = self.record
    simulation.refresh_observation()
    self.record(simulation)

  def record(self, sim):
    """Record current state/control/forces before their integration step."""
    timestamp = float(sim.data.time)
    if timestamp + 1e-9 < self.next_sample:
      return
    if timestamp > self.next_sample + 1e-6:
      raise RuntimeError("500 Hz recorder missed its physics callback")
    self.next_sample += 1 / FORCE_HZ
    tactile = sim.forces.read(sim.data)
    wrist_wrench = self.wrist_wrench_sensor.read(sim.data)
    cleaning = getattr(sim, "cleaning", None)
    zeros = np.zeros_like(sim.remaining)
    sample = {
      "time_s": timestamp,
      "phase": getattr(sim, "phase", "observe"),
      **{
        name: getattr(sim.data, name).copy()
        for name in ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied")
      },
      "ink_remaining": sim.remaining.copy(),
      "ink_sliding_distance": sim.sliding_distance.copy(),
      "ink_rgba": sim.model.geom_rgba[sim.ink_ids].copy(),
      "patch_work_j": np.asarray(getattr(cleaning, "work_j", zeros)).copy(),
      "patch_loaded_time_s": np.asarray(
        getattr(cleaning, "loaded_time_s", zeros)
      ).copy(),
      "patch_contact_tangent_load_n": np.asarray(
        getattr(sim, "patch_tangent_load", zeros)
      ).copy(),
      "patch_sliding_speed_m_s": np.asarray(getattr(sim, "patch_speed", zeros)).copy(),
      "patch_friction_power_w": np.asarray(getattr(sim, "patch_power", zeros)).copy(),
      "board_force_n": sim.board_force,
      "board_tangent_force_n": getattr(sim, "board_tangent_force", 0.0),
      "board_contact_center_world_m": sim.board_contact_center.copy(),
      "board_contact_torque_world_nm": sim.board_contact_torque.copy(),
      "board_contact_count": sim.board_contact_count,
      "table_support_force_n": sim.table_force,
      "direct_hand_board_force_n": getattr(sim, "direct_hand_board_force", 0.0),
      "normal_taxel_force_n": tactile.normal_taxel_force_n[5:].copy(),
      "tangent_taxel_force_n": tactile.tangent_taxel_force_n[5:].copy(),
      "normal_force_n": tactile.normal_force_n[5:].copy(),
      "tangent_force_n": tactile.tangent_force_n[5:].copy(),
      "tangent_contact_load_n": tactile.tangent_load_n[5:].copy(),
      "pad_contact_count": tactile.contact_count[5:].copy(),
      "pad_force_world_n": tactile.force_world_n[5:].copy(),
      "wrist_wrench/timestamp": timestamp,
      **{
        f"wrist_wrench/{name}": getattr(wrist_wrench, name)
        for name in WristWrenchSample.__dataclass_fields__
      },
    }
    for name in (
      "normal_taxel_force_n",
      "tangent_taxel_force_n",
      "normal_force_n",
      "tangent_force_n",
      "tangent_load_n",
      "tangent_taxel_load_n",
      "contact_count",
      "force_world_n",
      "normal_force_world_n",
      "tangent_force_world_n",
      "normal_axis_world",
      "tangent_basis_world",
    ):
      sample[f"tactile_contact_force/{name}"] = getattr(tactile, name).copy()
    self.buffers.append(sample)
    self.sample_count += 1
    if len(self.buffers) >= 128:
      self._flush()

  def observe(self, sim, phase):
    """Keep the executor observer API; sampling itself uses the physics hook."""
    sim.phase = phase

  def _flush(self):
    if not self.buffers:
      return
    for name in self.buffers[0]:
      values = np.asarray([sample[name] for sample in self.buffers])
      if name == "phase":
        values = values.astype("S64")
      if name not in self.datasets:
        self.datasets[name] = self.h5.create_dataset(
          name,
          shape=(0, *values.shape[1:]),
          maxshape=(None, *values.shape[1:]),
          chunks=(128, *values.shape[1:]),
          dtype=values.dtype,
          compression="gzip",
          compression_opts=1,
          shuffle=True,
        )
      dataset = self.datasets[name]
      start = dataset.shape[0]
      dataset.resize(start + len(values), axis=0)
      dataset[start:] = values
    self.buffers.clear()
    self.h5.flush()

  def finish(self, result, *, render=True):
    self.sim.refresh_observation()
    self.record(self.sim)
    self._flush()
    if self.sample_count < 2:
      raise ValueError("at least two native force samples are required")
    result = self._recorded_result(result)
    self.sim.physics_observer = self.previous_observer
    self.h5.attrs["success"] = bool(result.get("success", False))
    self.h5.attrs["source_sha256_json"] = json.dumps(self.source_hashes)
    # Root right-hand fields match vase; the complete bilateral group matches USB.
    self.h5["tactile_contact_force/timestamp"] = self.h5["time_s"]
    self.h5.require_group("state")["timestamp"] = self.h5["time_s"]
    self.h5["state/qpos"] = self.h5["qpos"]
    self.h5["state/qvel"] = self.h5["qvel"]
    self.h5.require_group("commands")["phase"] = self.h5["phase"]
    self.h5["commands/actuator_control"] = self.h5["ctrl"]
    self.h5.attrs["capture_complete"] = True
    _json(self.directory / "result.json", result)
    times = self.h5["time_s"][:]
    if not np.allclose(np.diff(times), 1 / FORCE_HZ, rtol=0, atol=1e-9):
      raise ValueError("recorded force clock is not native 500 Hz")
    if not render:
      _json(
        self.directory / "manifest.json",
        {
          "schema": SCHEMA,
          "capture_complete": True,
          "ink_randomization": self.sim.ink_randomization,
          "render_complete": False,
          "physics_hz": 1 / self.sim.timestep,
          "force_hz": FORCE_HZ,
          "video_fps": self.video_fps,
          "sample_count": self.sample_count,
          "source_sha256": self.source_hashes,
          "success": bool(result.get("success", False)),
        },
      )
      self.close()
      return
    self._render()
    self._curves()
    self._documents(result)
    self.h5.flush()
    self.close()

  def _recorded_result(self, result):
    """Label controller summaries separately from the native recorded peaks."""
    result = dict(result)
    result.setdefault(
      "controller_peak_fingertip_force_n", result.get("peak_fingertip_force_n")
    )
    result["controller_peak_fingertip_force_sampling_hz"] = 100
    result["peak_fingertip_force_n"] = self.h5["normal_force_n"][:].max(axis=0).tolist()
    result["peak_fingertip_force_sampling_hz"] = FORCE_HZ
    result["peak_board_force_sampling_hz"] = 1 / self.sim.timestep
    return result

  @classmethod
  def render_existing(cls, simulation, source_directory, directory, *, video_fps=10):
    """Render an existing capture into a new directory, preserving its source."""
    source = Path(source_directory)
    with h5py.File(source / "raw/episode.h5", "r") as h5:
      if h5.attrs.get("schema") != SCHEMA or not h5.attrs.get(
        "capture_complete", False
      ):
        raise ValueError("source must be a completed native whiteboard capture")
      hashes = json.loads(h5.attrs["source_sha256_json"])
      layout = json.loads(h5.attrs.get("ink_randomization_json", "null"))
    if layout is not None:
      simulation.set_ink_layout(layout["geom_positions_board_m"], seed=layout["seed"])
    for name, expected in hashes.items():
      # Replay uses saved states, but geometry and material definitions must agree.
      if Path(name).suffix in (".xml", ".png", ".obj") or name in (
        "geometry.py",
        "config.py",
      ):
        path = Path(__file__).parent / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
          raise ValueError(f"render source changed since capture: {name}")
    result = json.loads((source / "result.json").read_text())
    recorder = cls(simulation, directory, video_fps=video_fps)
    recorder.close()
    shutil.copyfile(source / "raw/episode.h5", recorder.directory / "raw/episode.h5")
    recorder.h5 = h5py.File(recorder.directory / "raw/episode.h5", "r+")
    recorder.datasets = {}
    recorder.source_hashes = hashes
    recorder.sample_count = len(recorder.h5["time_s"])
    result = recorder._recorded_result(result)
    recorder.h5.attrs["camera_hz"] = video_fps
    if "cameras" in recorder.h5:
      del recorder.h5["cameras"]
    _json(recorder.directory / "result.json", result)
    try:
      recorder._render()
      recorder._curves()
      recorder._documents(result)
    finally:
      recorder.close()
    return result

  def _render(self):
    cameras = tuple(
      CameraConfig(name, width=640, height=360, depth=False, segmentation=False)
      for name in ("head", "right_wrist", "board_overview")
    )
    times = self.h5["time_s"][:]
    phases = self.h5["phase"].asstr()[:]
    indices = np.arange(0, len(times), FORCE_HZ // self.video_fps, dtype=int)
    if indices[-1] != len(times) - 1:
      indices = np.r_[indices, len(times) - 1]
    replay = mujoco.MjData(self.sim.model)
    original_rgba = self.sim.model.geom_rgba[self.sim.ink_ids].copy()
    camera_groups = {}
    for camera in cameras:
      group = self.h5.require_group(f"cameras/{camera.name}")
      group.create_dataset(
        "rgb",
        shape=(len(indices), 360, 640, 3),
        dtype="u1",
        chunks=(1, 360, 640, 3),
        compression="gzip",
        compression_opts=1,
      )
      group.create_dataset("timestamp", data=times[indices])
      group.create_dataset("state_index", data=indices)
      group.create_dataset("world_from_camera", shape=(len(indices), 4, 4), dtype="f8")
      camera_groups[camera.name] = group
    renderer = writer = scene_writer = None
    snapshots = {}
    try:
      renderer = WorkcellRenderer(
        self.sim.model, cameras, visible_geom_groups=(0, 1, 2), shadows=False
      )
      writer = _FfmpegPipeWriter(
        self.directory / "review/review.mp4", fps=self.video_fps, width=1280, height=720
      )
      scene_writer = _FfmpegPipeWriter(
        self.directory / "review/scene.mp4", fps=self.video_fps, width=1280, height=400
      )
      print(
        f"Rendering {len(indices)} native frames from {len(times)} saved force samples",
        flush=True,
      )
      for frame_index, state_index in enumerate(indices):
        for name in ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied"):
          getattr(replay, name)[:] = self.h5[name][state_index]
        replay.time = times[state_index]
        self.sim.model.geom_rgba[self.sim.ink_ids] = self.h5["ink_rgba"][state_index]
        mujoco.mj_forward(self.sim.model, replay)
        panels = {}
        for camera in cameras:
          rgb = renderer.capture(replay, camera)["rgb"]
          group = camera_groups[camera.name]
          group["rgb"][frame_index] = rgb
          calibration = renderer.calibration(replay, camera)
          group["world_from_camera"][frame_index] = calibration.world_from_camera
          if frame_index == 0:
            group.create_dataset("intrinsic", data=calibration.intrinsic)
            group.attrs["fovy_degrees"] = calibration.fovy_degrees
          panels[camera.name] = Image.fromarray(rgb)
        sample = {
          name: self.h5[name][state_index]
          for name in (
            "normal_taxel_force_n",
            "tangent_taxel_force_n",
            "normal_force_n",
            "tangent_force_n",
          )
        }
        panels["normal"] = _force_maps(sample)
        panels["tangent"] = _force_maps(sample, True)
        composite = Image.new("RGB", (1280, 720), "#101923")
        for name, xy in (
          ("head", (0, 0)),
          ("right_wrist", (0, 360)),
          ("normal", (640, 0)),
          ("tangent", (640, 360)),
        ):
          composite.paste(panels[name], xy)
        draw = ImageDraw.Draw(composite)
        draw.text(
          (8, 8),
          f"HEAD | {phases[state_index]} | {times[state_index]:.3f}s",
          fill="white",
          stroke_width=1,
          stroke_fill="black",
        )
        draw.text(
          (8, 368), "RIGHT_WRIST", fill="white", stroke_width=1, stroke_fill="black"
        )
        panels["composite"] = composite
        for name, panel in panels.items():
          panel.save(self.directory / f"review/frames/{name}/{frame_index:06d}.png")
        fields = (
          "normal_taxel_force_n",
          "tangent_taxel_force_n",
          "normal_force_n",
          "tangent_force_n",
          "tangent_contact_load_n",
          "pad_contact_count",
        )
        np.savez_compressed(
          self.directory / f"review/raw/{frame_index:06d}.npz",
          time_s=times[state_index],
          state_index=state_index,
          **{name: self.h5[name][state_index] for name in fields},
        )
        writer.write(np.asarray(composite))
        scene = Image.new("RGB", (1280, 400), "#101923")
        scene.paste(panels["board_overview"], (0, 0))
        scene.paste(panels["head"], (640, 0))
        draw = ImageDraw.Draw(scene)
        for x, label in ((8, "BOARD OVERVIEW"), (648, "HEAD")):
          draw.text((x, 8), label, fill="white", stroke_width=1, stroke_fill="black")
        cleared = 1 - self.h5["ink_remaining"][state_index].mean()
        board_fn = self.h5["board_force_n"][state_index]
        board_ft = self.h5["board_tangent_force_n"][state_index]
        draw.text(
          (12, 372),
          f"{phases[state_index]} | {times[state_index]:.3f}s | cleared {cleared:.1%} | board Fn {board_fn:.2f} N / Ft {board_ft:.2f} N",
          fill="white",
        )
        scene_writer.write(np.asarray(scene))
        if frame_index in (0, len(indices) - 1):
          inspection_index = 0 if frame_index == 0 else 1
          panels["head"].save(
            self.directory / f"review/inspection/{inspection_index:02d}.png"
          )
        if phases[state_index] not in snapshots:
          snapshots[phases[state_index]] = scene.copy()
        playback = frame_index / self.video_fps + times[0]
        self.frame_rows.append(
          (
            frame_index,
            times[state_index],
            int(state_index),
            times[state_index],
            phases[state_index],
            playback,
            playback - times[state_index],
          )
        )
        if frame_index % 50 == 0 or frame_index == len(indices) - 1:
          print(f"Rendered {frame_index + 1}/{len(indices)} frames", flush=True)
    finally:
      self.sim.model.geom_rgba[self.sim.ink_ids] = original_rgba
      for resource in (writer, scene_writer):
        if resource is not None:
          resource.finish()
      if renderer is not None:
        renderer.close()
    with (self.directory / "review/frames.csv").open("w", newline="") as stream:
      csv.writer(stream).writerows(
        [
          (
            "frame_index",
            "time_s",
            "state_index",
            "force_time_s",
            "phase",
            "playback_time_s",
            "playback_time_error_s",
          ),
          *self.frame_rows,
        ]
      )
    chosen = [
      snapshots[phase]
      for phase in ("observe", "lift", "wipe", "verify")
      if phase in snapshots
    ]
    if not chosen:
      chosen = list(snapshots.values())[:4]
    preview = Image.new("RGB", (1280, 400 * len(chosen)), "#101923")
    for i, panel in enumerate(chosen):
      preview.paste(panel, (0, 400 * i))
    preview.save(self.directory / "review/preview.png")
    _json(
      self.directory / "review/review.json",
      {
        "fps": self.video_fps,
        "resolution": [1280, 720],
        "frame_count": len(indices),
        "camera_names": [c.name for c in cameras],
        "native_camera_resolution": [640, 360],
        "evaluation_video": "scene.mp4",
        "camera_source": "native RGB from this rollout's exact saved qpos; no physics stepping",
        "time_alignment": "frames.csv maps each image to the exact saved 500 Hz force sample",
        "normal_color_max_n": 0.4,
        "tangent_color_max_n": 0.05,
        "raw_clipped": False,
        "max_force_sync_error_s": 0.0,
        "max_playback_time_error_s": max(abs(row[-1]) for row in self.frame_rows),
      },
    )

  def _curves(self):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = self.h5["time_s"][:]
    phases = self.h5["phase"].asstr()[:]
    fn, xy = self.h5["normal_force_n"][:], self.h5["tangent_force_n"][:]
    ft = np.linalg.norm(xy, axis=-1)
    loads, counts = (
      self.h5["tangent_contact_load_n"][:],
      self.h5["pad_contact_count"][:],
    )
    normal_error = tangent_error = 0.0
    for start in range(0, len(times), 512):
      stop = min(start + 512, len(times))
      normal_error = max(
        normal_error,
        float(
          np.max(
            np.abs(
              self.h5["normal_taxel_force_n"][start:stop].sum(axis=(2, 3))
              - fn[start:stop]
            )
          )
        ),
      )
      tangent_error = max(
        tangent_error,
        float(
          np.max(
            np.abs(
              self.h5["tangent_taxel_force_n"][start:stop].sum(axis=(2, 3))
              - xy[start:stop]
            )
          )
        ),
      )
    if max(normal_error, tangent_error) > 1e-9:
      raise ValueError("taxel sums disagree with recorded fingertip forces")
    with (self.directory / "curves/right_hand_forces.csv").open(
      "w", newline=""
    ) as stream:
      writer = csv.writer(stream)
      writer.writerow(
        ["time_s", "phase"]
        + [
          f"{finger}_{value}"
          for finger in FINGERS
          for value in (
            "Fn_N",
            "Fx_N",
            "Fy_N",
            "Ft_resultant_N",
            "Ft_contact_load_N",
            "contact_count",
          )
        ]
      )
      for i, timestamp in enumerate(times):
        writer.writerow(
          [timestamp, phases[i]]
          + [
            value
            for j in range(5)
            for value in (fn[i, j], *xy[i, j], ft[i, j], loads[i, j], int(counts[i, j]))
          ]
        )
    fig, axes = plt.subplots(
      2, 1, figsize=(12, 7), sharex=True, constrained_layout=True
    )
    for j, finger in enumerate(FINGERS):
      axes[0].plot(times, fn[:, j], label=finger, lw=0.8)
      axes[1].plot(times, ft[:, j], label=finger, lw=0.8)
    axes[0].set_ylabel("Normal force Fn (N)")
    axes[1].set_ylabel("Tangential resultant |sum(Fx,Fy)| (N)")
    axes[1].set_xlabel("Simulation time (s)")
    axes[0].legend(ncol=5)
    for ax in axes:
      ax.grid(alpha=0.25)
    for extension in ("png", "pdf"):
      fig.savefig(
        self.directory / f"curves/right_hand_force_curves.{extension}", dpi=160
      )
    plt.close(fig)
    board_fn, board_ft = (
      self.h5["board_force_n"][:],
      self.h5["board_tangent_force_n"][:],
    )
    table = self.h5["table_support_force_n"][:]
    ink = self.h5["ink_remaining"][:]
    mean, worst = 1 - ink.mean(axis=1), 1 - ink.max(axis=1)
    with (self.directory / "curves/contact_and_cleaning.csv").open(
      "w", newline=""
    ) as stream:
      writer = csv.writer(stream)
      writer.writerow(
        (
          "time_s",
          "phase",
          "board_Fn_N",
          "board_Ft_load_N",
          "table_support_N",
          "mean_cleared",
          "worst_patch_cleared",
        )
      )
      writer.writerows(
        zip(times, phases, board_fn, board_ft, table, mean, worst, strict=True)
      )
    fig, axes = plt.subplots(
      2, 1, figsize=(12, 7), sharex=True, constrained_layout=True
    )
    axes[0].plot(times, board_fn, label="Board normal force", lw=0.8)
    axes[0].plot(times, board_ft, label="Board tangential contact load", lw=0.8)
    axes[0].plot(times, table, label="Table support", lw=0.8)
    axes[0].set_ylabel("Force (N)")
    axes[1].plot(times, mean * 100, label="Mean cleared")
    axes[1].plot(times, worst * 100, label="Worst patch cleared")
    axes[1].set_ylabel("Ink cleared (%)")
    axes[1].set_xlabel("Simulation time (s)")
    for ax in axes:
      ax.legend()
      ax.grid(alpha=0.25)
    for extension in ("png", "pdf"):
      fig.savefig(self.directory / f"curves/contact_and_cleaning.{extension}", dpi=160)
    plt.close(fig)
    phase_statistics = {}
    for phase in dict.fromkeys(phases):
      selected = phases == phase
      if selected.sum() < 2:
        continue
      normal_steps = np.diff(fn[selected], axis=0)
      tangent_steps = np.diff(ft[selected], axis=0)
      vector_steps = np.linalg.norm(np.diff(xy[selected], axis=0), axis=-1)
      phase_statistics[phase] = {
        "sample_count": int(selected.sum()),
        "normal_step_rms_n": np.sqrt(np.mean(normal_steps**2, axis=0)).tolist(),
        "tangent_resultant_step_rms_n": np.sqrt(
          np.mean(tangent_steps**2, axis=0)
        ).tolist(),
        "tangent_vector_step_rms_n": np.sqrt(np.mean(vector_steps**2, axis=0)).tolist(),
        "normal_step_p99_n": np.quantile(abs(normal_steps), 0.99, axis=0).tolist(),
        "tangent_vector_step_p99_n": np.quantile(vector_steps, 0.99, axis=0).tolist(),
      }
    contact_continuity = {}
    if "board_contact_center_world_m" in self.h5:
      active = (phases == "wipe") & (board_fn > 0.5)
      adjacent = active[1:] & active[:-1]
      contact_continuity["loaded_adjacent_samples"] = int(adjacent.sum())
      if adjacent.any():
        for field, name in (
          ("board_contact_center_world_m", "center_step_rms_m"),
          ("board_contact_torque_world_nm", "torque_step_rms_nm"),
        ):
          steps = np.diff(self.h5[field][:], axis=0)[adjacent]
          contact_continuity[name] = float(np.sqrt(np.mean(np.sum(steps**2, axis=1))))
    _json(
      self.directory / "curves/force_statistics.json",
      {
        "sample_count": len(times),
        "force_hz": FORCE_HZ,
        "step_interval_s": 1 / FORCE_HZ,
        "finger_order": FINGERS,
        "phase_statistics": phase_statistics,
        "board_contact_continuity": contact_continuity,
        "normal_sum_error_n": normal_error,
        "tangent_sum_error_n": tangent_error,
        "definitions": {
          "Fn": "sum of 35 normal taxels",
          "Fx_Fy": "signed pad-local taxel sums",
          "Ft_resultant": "hypot(Fx,Fy)",
          "Ft_contact_load": "sum of individual contact tangential magnitudes",
        },
        "per_finger": {
          finger: {
            "normal_max_n": float(fn[:, j].max()),
            "normal_mean_n": float(fn[:, j].mean()),
            "tangent_max_n": float(ft[:, j].max()),
          }
          for j, finger in enumerate(FINGERS)
        },
      },
    )

  def _documents(self, result):
    documentation_link = Path(
      os.path.relpath(
        Path(__file__).resolve().parents[4] / "docs/tasks/whiteboard_wipe.md",
        self.directory,
      )
    ).as_posix()
    duration = float(self.h5["time_s"][-1] - self.h5["time_s"][0])
    cleared = 1 - float(self.h5["ink_remaining"][-1].mean())
    _json(
      self.directory / "manifest.json",
      {
        "schema": SCHEMA,
        "ink_randomization": self.sim.ink_randomization,
        "physics_hz": 1 / self.sim.timestep,
        "force_hz": FORCE_HZ,
        "video_fps": self.video_fps,
        "fingers": FINGERS,
        "force_source": self.sim.forces.source,
        "source_sha256": self.source_hashes,
        "success": bool(result.get("success", False)),
        "sample_count": self.sample_count,
        "frame_count": len(self.frame_rows),
        "camera_source": "native 640x360 RGB from this one rollout's saved states",
        "camera_hdf5_groups": [
          "cameras/head",
          "cameras/right_wrist",
          "cameras/board_overview",
        ],
        "force_processing": "no smoothing, clipping, interpolation or resampling",
        "force_evaluation": "before integration, after control update; terminal sample evaluated without stepping",
        "bilateral_group": "tactile_contact_force",
        "root_tactile_fields": "right hand only, same convention as vase_wipe_example",
      },
    )
    verdict = (
      "已完成抓起、擦净并放回松手"
      if result.get("success")
      else "本次未通过完整任务验收"
    )
    (self.directory / "README.md").write_text(
      f"""# Erase whiteboard example

本目录来自一次新的原生物理执行：{duration:.3f} s，{self.sample_count} 个 500 Hz 状态/触觉样本。{verdict}。

- `review/review.mp4`：共享 head / right_wrist 和右手五指 7×5 法向/切向热图，1280×720，{self.video_fps} fps。
- `review/scene.mp4`：board_overview / head 与板面接触力、清洁进度，1280×400。
- `review/frames/{{head,right_wrist,board_overview}}`：原生 640×360 无损 PNG，与 HDF5 RGB 完全对应。
- `review/frames/{{normal,tangent,composite}}`：热图和合成帧；`review/raw/*.npz` 保存右手原始有符号触觉。
- `review/frames.csv`：图像时刻、原始样本索引、触觉时刻和固定帧率播放误差。最后一帧保留精确终态。
- `curves/right_hand_force_curves.{{png,pdf}}`、`right_hand_forces.csv`：500 Hz 未滤波五指力，Fn/Fx/Fy/Ft_resultant/Ft_contact_load/contact_count。
- `curves/contact_and_cleaning.{{png,pdf,csv}}`：板面法向/摩擦力、桌面支撑与清除比例。
- `curves/force_statistics.json`：各动作阶段的原始力相邻采样差分 RMS/P99，以及板面受力中心、合力矩的连续性。
- `raw/episode.h5`：完整原始状态、接触力、清洁进度、原生 RGB 与相机内外参；`result.json` 是执行结果。

目录、视频四面板和两行五指力曲线遵循 `vase_wipe_example`；力数据与 `usb_insert_example` 同为 500 Hz。视频默认与花瓶相同为 10 fps（USB 为 30 fps）。机器人头部和腕部相机定义来自 shared，另加 `board_overview` 验收视角；640×360 是原生渲染分辨率。

HDF5 根字段 `time_s/phase/qpos/qvel/ctrl/qfrc_applied` 对齐每个 500 Hz 时刻。根 `normal_taxel_force_n [T,5,7,5]`、`tangent_taxel_force_n [T,5,7,5,2]` 及指尖力保存右手，与花瓶一致。完整双手另存 `tactile_contact_force/*`（10 指，前五左手、后五右手）；`link_names` 保存顺序。`state/timestamp` 与 `commands/phase` 是原始字段的硬链接。

Fn 为 35 个法向触元之和；Fx/Fy 为有符号分量之和；Ft_resultant = hypot(Fx,Fy)，Ft_contact_load 单独保存逐接触切向模长之和。单位 N；原始力不平滑、不裁剪、不插值，仅热图固定颜色量程可饱和。

`result.json` 的指尖峰值来自 500 Hz 原始样本；另存控制器 100 Hz 统计及其采样率。板面峰值在 1000 Hz 物理更新中统计。清洁每 1 ms 更新，数据每 2 ms 保存，逐帧记录不能穷尽证明中间物理步的清洁门限。

状态、控制和触觉在控制量更新后、积分前同时保存；清洁使用该步实际积分的接触力，终态单独求值而不推进时间。随后对保存的 qpos/qvel/ctrl/qfrc_applied 和笔迹 RGBA 调用 mj_forward 原生渲染；渲染阶段不推进物理、不重算或替换已保存触觉。每路 PNG 与 HDF5 RGB 同源；MP4 仅为有损展示。初末图像供人工检查，当前控制器仍使用已知场景和工具状态。

实际法向/切向载荷、逐块滑动距离/摩擦功/有效加载时间保留在 HDF5。参数与局限见 [任务说明]({documentation_link})。

HDF5 的 `board_contact_center_world_m` 是法向载荷加权的板面受力中心，`board_contact_torque_world_nm` 是板面对板擦质心的合力矩，均为世界坐标；无接触时置零，另有 `board_contact_count`。这些字段用于检查总载荷平稳时是否仍存在接触点重排。

本任务通过零接触 margin 和较高 CCD 精度稳定板擦接触点，并采用较柔顺的右手速度伺服及每毫秒连续的位置目标。几何、质量、摩擦系数和需要受力滑动的清洁门槛不变；这些修改只作用于当前任务实例，原始触觉未滤波。

笔迹覆盖约 14×16 cm，分为间距约 14 cm 的两条、25 个独立笔画单元。控制器逐条覆盖，长笔画完整落入毛毡范围后才累计清洁量；清洁门槛不变。双臂仍从 shared 对称收臂姿势开始。

贴板深度控制使用低增益反馈和 2.2 N 目标；换条时先卸载、离板横移、再随擦拭运动建立载荷，实际载荷上升时减慢擦拭路径。触觉与力曲线继续保存未滤波的真实求解器载荷。

放回时先以 2 mm/s 接近桌面，再根据实际支撑力卸载至板擦自重；释放时逐指降低实测法向载荷，并在完全张手前持续消除多余下压力。全部曲线仍来自未经平滑的物理接触解。

原位更新：`pixi run record-whiteboard-example --overwrite`。只更新 `datasets/erase_whiteboard_example`，不创建备份；不加 `--overwrite` 时保护已有输出。
""",
      encoding="utf-8",
    )
    rows_json = json.dumps(
      [{"time": row[1], "phase": row[4]} for row in self.frame_rows]
    )
    (self.directory / "index.html").write_text(
      f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>45° 白板擦拭</title>
<style>body{{max-width:1280px;margin:32px auto;padding:0 24px;background:#101923;color:#e7eef5;font:16px/1.7 system-ui}}video,img,input{{width:100%}}video,img{{background:#080d14;border-radius:10px}}a{{color:#75d7c7}}.inspections{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}}h3{{font-size:16px}}</style>
<h1>45° 白板擦拭</h1><p>{verdict} · {duration:.2f} s · 平均清除 {cleared:.1%}</p>
<p>两条大范围笔迹，覆盖约 14×16 cm，中心间距约 14 cm；从共享双臂初始姿势出发，逐条擦净后放回板擦。</p>
<p>本次原生记录：500 Hz 状态与触觉，{self.video_fps} fps 相机；头部和右腕原图均为 640×360。</p>
<h2>初始与完成后</h2><div class="inspections"><section><h3>初始黑色笔迹</h3><img src="review/inspection/00.png"></section><section><h3>执行完成后</h3><img src="review/inspection/01.png"></section></div>
<h2>机器人相机与触觉</h2><video src="review/review.mp4" controls preload="metadata"></video><p>左上 head，左下 right_wrist；右側为右手五指法向与切向触元力，对应同一时刻的 500 Hz 原始样本。</p>
<h2>逐帧查看</h2><input id="slider" type="range" min="0" max="{len(self.frame_rows) - 1}" value="0"><p id="meta"></p><img id="frame" src="review/frames/composite/000000.png">
<h2>场景与接触</h2><video src="review/scene.mp4" controls preload="metadata"></video>
<h2>右手五指力曲线</h2><img src="curves/right_hand_force_curves.png"><p><a href="curves/right_hand_force_curves.pdf">PDF</a> · <a href="curves/right_hand_forces.csv">500 Hz CSV</a> · <a href="curves/force_statistics.json">力统计</a></p>
<h2>板面摩擦力与清洁进度</h2><img src="curves/contact_and_cleaning.png"><p><a href="curves/contact_and_cleaning.pdf">PDF</a> · <a href="curves/contact_and_cleaning.csv">500 Hz CSV</a></p>
<h2>阶段预览</h2><img src="review/preview.png">
<p><a href="raw/episode.h5">原始 HDF5</a> · <a href="result.json">执行结果</a> · <a href="review/frames.csv">帧与样本对应</a> · <a href="manifest.json">数据清单</a> · <a href="README.md">数据说明</a> · <a href="{documentation_link}">清洁模型参数</a></p>
<script id="rows" type="application/json">{rows_json}</script><script>const rows=JSON.parse(document.getElementById('rows').textContent),slider=document.getElementById('slider');function show(i){{document.getElementById('frame').src='review/frames/composite/'+String(i).padStart(6,'0')+'.png';document.getElementById('meta').textContent=`帧 ${{i}}/${{rows.length-1}} · ${{rows[i].time.toFixed(3)}} 秒 · ${{rows[i].phase}}`;}}slider.addEventListener('input',()=>show(Number(slider.value)));show(0);</script></html>""",
      encoding="utf-8",
    )

  def close(self):
    if getattr(self.sim, "physics_observer", None) == self.record:
      self.sim.physics_observer = self.previous_observer
    if self.h5 is not None:
      self._flush()
      self.h5.close()
      self.h5 = None
