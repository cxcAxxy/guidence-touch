"""Record one RAM installation and derive review artifacts from its saved data."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import h5py
import mujoco
import numpy as np
from PIL import Image

from ...shared.config import (
  TRAINING_CAMERA_NAMES,
  CameraConfig,
  WorkcellConfig,
  model_fingerprint,
)
from ...shared.contact_tactile import SolverDistributedTactileProvider
from ...shared.poker_review import (
  _probe_video,
  _validate_probe,
  compose_review_frame,
  plan_review_frames,
)
from ...shared.recording import EpisodeRecorder, _append, _stream, validate_episode
from ...shared.rendering import WorkcellRenderer
from ...shared.tactile import RIGHT_FINGERTIP_LINK_NAMES, default_fingertip_layout_path
from ...shared.task_video import _FfmpegPipeWriter
from . import config
from .diagnostics import ForceTraceRecorder, summarize_force_trace
from .force_analysis import (
  _noise_metrics,
  write_force_analysis,
  write_physics_force_comparison,
)
from .global_review import export_global_review
from .task import RamInstallSimulation

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
CAMERAS = TRAINING_CAMERA_NAMES


def _json(path: Path, value) -> None:
  path.write_text(
    json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


class RamExampleRecorder(EpisodeRecorder):
  """Shared schema, plus measured installation state and applied-force audit."""

  def __init__(self, path, executor, capture_config, *, buffer_rows=64):
    self.executor = executor
    self.state_write_seconds = 0.0
    super().__init__(
      path,
      executor.sim,
      capture_config,
      metadata={
        "scene": config.SCENE_NAME,
        "initial_position_randomization": executor.sim.initial_position_randomization,
        "episode_index": 0,
        "recording_contract": "install_ram_example_v4",
        "cast_shadows": False,
        "shadow_policy": "task_local_lights_no_cast_shadows",
        "mechanics_version": config.MECHANICS_VERSION,
      },
      capture_taskspace=True,
      buffer_rows=buffer_rows,
    )

  def _values(self):
    executor, data = self.executor, self.sim.data
    state = executor.state if hasattr(executor, "state") else executor._state
    object_dof = self.sim._object_dofs["ram"]
    values = {
      **asdict(state),
      "actuator_control": data.ctrl.copy(),
      "qfrc_applied": data.qfrc_applied.copy(),
      "xfrc_applied": data.xfrc_applied.copy(),
      "ram_qfrc_applied": data.qfrc_applied[object_dof : object_dof + 6].copy(),
      "wrist_target_position_m": np.asarray(
        getattr(executor, "_command_position", self.sim.current_pose_matrix("right")[0])
      ).copy(),
      "wrist_target_rotation": np.asarray(
        getattr(executor, "_command_rotation", self.sim.current_pose_matrix("right")[1])
      ).copy(),
      "right_hand_joint_target": np.array(
        [
          self.sim._hand_targets["right"][name]
          for name in self.sim._hand_joint_names["right"]
        ]
      ),
    }
    if data.eq_active.size:
      values["eq_active"] = data.eq_active.copy()
    return values

  def _initialize(self, metadata):
    self.contact_force_provider = SolverDistributedTactileProvider(
      self.sim.model,
      self.sim.genesis_probe_layout,
      link_names=RIGHT_FINGERTIP_LINK_NAMES,
    )
    super()._initialize(metadata)
    task = self._file.create_group("install_ram")
    for name, value in self._values().items():
      array = np.asarray(value)
      if array.dtype.kind not in "biuf":
        raise TypeError(f"task measurement {name!r} must be numeric")
      _stream(task, name, array.shape, array.dtype)

  def _record_state(self, phase):
    started = time.monotonic()
    super()._record_state(phase)
    for name, value in self._values().items():
      _append(self._file[f"install_ram/{name}"], value)
    self.state_write_seconds += time.monotonic() - started

  def close(self, *, finalize=True):
    if not self._closed:
      self._file.attrs["recording_timing_json"] = json.dumps(
        {
          "state_write_seconds": self.state_write_seconds,
          "camera_write_seconds": self._camera_write_seconds,
          "recording_wall_seconds": time.monotonic() - self._recording_started,
        }
      )
    super().close(finalize=finalize)


def _forces(file):
  group = file["tactile_contact_force"]
  names = tuple(group["link_names"].asstr()[:])
  if names != RIGHT_FINGERTIP_LINK_NAMES:
    raise ValueError("right-finger ordering differs from the shared tactile contract")
  normal = group["normal_taxel_force_n"][:].sum(axis=(-2, -1))
  tangent = group["tangent_taxel_force_n"][:].sum(axis=(-3, -2))
  np.testing.assert_allclose(normal, group["normal_force_n"][:], atol=1e-10)
  np.testing.assert_allclose(tangent, group["tangent_force_n"][:], atol=1e-10)
  return group["timestamp"][:], normal, tangent, np.linalg.norm(tangent, axis=-1)


def _audit_physics_trace(file, path: Path):
  """Independently verify the loaded hold and the raw 500/100 Hz force identity."""
  with np.load(path) as trace:
    timestamp = trace["time"]
    phase = trace["phase"]
    state_time = file["state/timestamp"][:]
    indices = np.searchsorted(timestamp, state_time)
    if np.any(indices >= len(timestamp)):
      raise ValueError("500 Hz trace does not cover every HDF5 sample")
    np.testing.assert_array_equal(timestamp[indices], state_time)
    normal = file["tactile_contact_force/normal_force_n"][:]
    tangent = file["tactile_contact_force/tangent_force_n"][:]
    np.testing.assert_allclose(trace["fn"][indices], normal, rtol=0, atol=1e-10)
    np.testing.assert_allclose(trace["ft"][indices], tangent, rtol=0, atol=1e-10)
    active = np.isin(
      phase,
      ("lift", "transfer", "transport", "align", "insert", "seat", "bottom_press"),
    )
    minimum_palm = float(trace["palm_down"][active].min())
    if minimum_palm < 0.35:
      raise ValueError("palm is not downward throughout transport and insertion")
    names = list(trace["socket_columns"])
    depth = trace["socket"][:, names.index("insertion_depth_m")]
    backstop = trace["socket"][:, names.index("backstop_load_n")]
    loaded = (
      (phase == "bottom_press")
      & (backstop >= config.BOTTOM_OUT_MIN_FORCE_N)
      & (
        np.abs(depth - config.TARGET_INSERTION_DEPTH_M)
        <= config.SEATED_DEPTH_TOLERANCE_M
      )
      & (trace["linear_speed_m_s"] < config.SEATED_LINEAR_SPEED_M_S)
      & (trace["angular_speed_rad_s"] < 0.05)
      & trace["aperture_fits"]
      & (trace["orientation_error_rad"] <= np.deg2rad(1.0))
      & (trace["maximum_socket_penetration_m"] <= config.MAX_SOCKET_PENETRATION_M)
    )
    selected = np.flatnonzero(loaded)
    intervals = (
      np.split(selected, np.flatnonzero(np.diff(selected) != 1) + 1)
      if len(selected)
      else []
    )
    windows = [
      run
      for run in intervals
      if timestamp[run[-1]] - timestamp[run[0]] >= config.BOTTOM_OUT_HOLD_S - 1e-8
    ]
    if not windows or not bool(trace["bottom_out_confirmed"][-1]):
      raise ValueError("500 Hz trace has no independent loaded bottom-out hold")
    window = windows[-1]
    friction = (phase == "insert") & (depth >= 0.0035) & (depth <= 0.0055)
    magnitude = np.linalg.norm(trace["ft"], axis=-1)
    noise = _noise_metrics(timestamp, magnitude, friction)
    if friction.sum() < 500 or noise["spectrum"] is None:
      raise ValueError("insufficient raw friction plateau for low-frequency audit")
    fluctuation = magnitude[friction].std(axis=0)
    if np.any(fluctuation[:2] > 0.015) or np.any(
      np.asarray(noise["spectrum"]["band_2_to_10hz_rms_n"])[:2] > 0.005
    ):
      raise ValueError(
        "raw insertion-friction forces still show excessive low-frequency variation"
      )
    return {
      "source_npz": "../raw/force_trace_500hz.npz",
      "same_execution_as_hdf5": True,
      "insertion_friction_noise": {
        "depth_window_m": [0.0035, 0.0055],
        "std_ft_n": fluctuation.tolist(),
        "std_limit_n": 0.015,
        "band_2_to_10hz_rms_limit_n": 0.005,
        **noise,
      },
      "sample_count": len(timestamp),
      "sample_hz": 1 / float(np.median(np.diff(timestamp))),
      "hdf5_samples_matched": len(indices),
      "maximum_fn_difference_n": float(np.max(np.abs(trace["fn"][indices] - normal))),
      "maximum_signed_ft_difference_n": float(
        np.max(np.abs(trace["ft"][indices] - tangent))
      ),
      "minimum_active_palm_down_cosine": minimum_palm,
      "loaded_bottom_window_s": [
        float(timestamp[window[0]]),
        float(timestamp[window[-1]]),
      ],
      "minimum_backstop_load_n": float(backstop[window].min()),
      "mean_backstop_load_n": float(backstop[window].mean()),
      "maximum_linear_speed_m_s": float(trace["linear_speed_m_s"][window].max()),
      "maximum_angular_speed_rad_s": float(trace["angular_speed_rad_s"][window].max()),
      "final_phase_semantics": "Trace retains actual verify phase; HDF5 marks that same final sample terminal_settle without duplicating its timestamp.",
    }


def _audit(file, result):
  timestamps, normal, tangent, magnitude = _forces(file)
  task = file["install_ram"]
  for name, dataset in task.items():
    if len(dataset) != len(timestamps) or not np.isfinite(dataset[:]).all():
      raise ValueError(f"invalid task stream {name!r}")
  np.testing.assert_array_equal(timestamps, file["state/timestamp"][:])
  if not np.all(np.diff(timestamps) > 0) or np.any(normal < -1e-12):
    raise ValueError("invalid tactile times or negative normal forces")
  if not all(np.isfinite(array).all() for array in (normal, tangent, magnitude)):
    raise ValueError("non-finite solver forces")
  if np.any(task["ram_qfrc_applied"][:]) or np.any(task["xfrc_applied"][:]):
    raise ValueError("installation used an externally applied object force")
  if not result.success:
    raise ValueError(f"installation failed: {result.reason}")
  if "seated" not in task or not bool(task["seated"][-1]):
    raise ValueError("terminal recorded geometry is not seated")
  twist = file["objects/ram/twist_linear_angular"][-1]
  if np.linalg.norm(twist[:3]) >= 0.02 or np.linalg.norm(twist[3:]) >= 0.2:
    raise ValueError("RAM is moving at the terminal frame")
  if normal.max() <= 0:
    raise ValueError("installation has no recorded fingertip/RAM contact")
  if normal[-1].max() > 0.01:
    raise ValueError("fingers remain loaded at terminal release")
  # Shared home may point the palm upward during the free-space approach.
  # The forehand requirement applies once the RAM is held and transported.
  phases = file["commands/phase"].asstr()[:]
  active = np.isin(phases, ("lift", "transfer", "align", "insert", "bottom_press"))
  if "palm_down_cosine" in task and np.any(task["palm_down_cosine"][:][active] < 0.35):
    raise ValueError("recorded palm is not downward during transport/insertion")
  if "bottom_out_confirmed" in task and not bool(task["bottom_out_confirmed"][-1]):
    raise ValueError("terminal result lacks recorded loaded bottom-out confirmation")
  phases = file["commands/phase"].asstr()[:]
  axial = np.isin(phases, ("insert", "bottom_press"))
  axial_target_errors = {}
  for name in (
    "wrist_target_position_m",
    "wrist_target_rotation",
    "right_hand_joint_target",
  ):
    values = task[name][:][axial]
    if name == "wrist_target_position_m":
      values = values[:, :2]
    error = float(np.max(np.abs(values - values[0])))
    if error > 1e-12:
      raise ValueError(f"axial insertion changed its frozen target: {name}")
    axial_target_errors[name] = error
  phase_intervals = []
  starts = np.r_[0, np.flatnonzero(phases[1:] != phases[:-1]) + 1]
  for start, stop in zip(starts, np.r_[starts[1:], len(phases)], strict=True):
    phase_intervals.append(
      {
        "phase": phases[start],
        "start_s": float(timestamps[start]),
        "end_s": float(timestamps[stop - 1]),
        "state_samples": int(stop - start),
      }
    )
  return {
    "state_samples": len(timestamps),
    "duration_seconds": float(timestamps[-1] - timestamps[0]),
    "fingers": list(FINGERS),
    "maximum_fn_n": normal.max(axis=0).tolist(),
    "maximum_ft_n": magnitude.max(axis=0).tolist(),
    "final_fn_n": normal[-1].tolist(),
    "final_ft_n": magnitude[-1].tolist(),
    "final_linear_speed_m_s": float(np.linalg.norm(twist[:3])),
    "final_angular_speed_rad_s": float(np.linalg.norm(twist[3:])),
    "external_object_forces_zero": True,
    "axial_insertion_frozen_target_maximum_errors": axial_target_errors,
    "phase_intervals": phase_intervals,
  }


def _audit_force_stages(report):
  friction = report["stages"]["insertion_friction"]
  press = report["stages"]["bottom_press"]
  if not friction["available"] or not press["available"]:
    raise ValueError("missing measured insertion-friction or loaded-press plateau")
  if min(friction["samples"], press["samples"]) < 5:
    raise ValueError("force plateau contains too few original samples")
  spring_friction = (
    friction["loads"].get("spring_axial_resistance_n", {}).get("mean_n", 0)
  )
  if spring_friction <= 0.05:
    raise ValueError("insertion has no measured positive spring friction resistance")
  bottom_difference = (
    press["loads"]["backstop_load_n"]["mean_n"]
    - friction["loads"]["backstop_load_n"]["mean_n"]
  )
  if bottom_difference < 0.5:
    raise ValueError("bottom pressing is not distinct from insertion friction")
  difference = np.asarray(press["mean_ft_n"]) - friction["mean_ft_n"]
  ratio = np.divide(
    press["mean_ft_n"],
    friction["mean_ft_n"],
    out=np.zeros(5),
    where=np.asarray(friction["mean_ft_n"]) > 1e-10,
  )
  if np.any(difference[:2] <= 0.05) or np.any(ratio[:2] <= 1.10):
    raise ValueError(
      "thumb/index raw tangential forces did not rise during verified bottom pressing"
    )
  return {
    "spring_friction_mean_n": spring_friction,
    "bottom_load_increase_n": bottom_difference,
    "mean_ft_increase_n": difference.tolist(),
    "mean_ft_ratio": [
      float(value) if base > 1e-10 else None
      for value, base in zip(ratio, friction["mean_ft_n"], strict=True)
    ],
    "mean_fn_increase_n": (
      np.asarray(press["mean_fn_n"]) - friction["mean_fn_n"]
    ).tolist(),
    "acceptance": "For actual thumb/index contact only: bottom-hold mean |Ft| > insertion plateau by 0.05 N and 10%; bottom-load mean rises >=0.5 N. Fn changes reported without imposing monotonicity; inactive fingers not required to gain force.",
  }


def _curves(file, output: Path):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  output.mkdir()
  timestamps, normal, tangent, magnitude = _forces(file)
  phases = file["commands/phase"].asstr()[:]
  with (output / "right_hand_forces.csv").open(
    "w", newline="", encoding="utf-8"
  ) as stream:
    writer = csv.writer(stream)
    writer.writerow(
      [
        "time_s",
        "phase",
        *[
          f"{finger}_{quantity}_n"
          for finger in FINGERS
          for quantity in ("fn", "ft_x", "ft_y", "ft")
        ],
      ]
    )
    for i, timestamp in enumerate(timestamps):
      writer.writerow(
        [
          timestamp,
          phases[i],
          *np.column_stack((normal[i], tangent[i], magnitude[i])).ravel(),
        ]
      )
  fig, axes = plt.subplots(5, 2, figsize=(14, 11), sharex=True)
  boundaries = timestamps[np.flatnonzero(phases[1:] != phases[:-1]) + 1]
  for finger, row in enumerate(axes):
    for axis, values, label in zip(
      row, (normal, magnitude), ("Fn (N)", "|Ft| (N)"), strict=True
    ):
      axis.plot(timestamps, values[:, finger], lw=0.7)
      axis.set_ylabel(f"{FINGERS[finger]}\n{label}")
      axis.grid(alpha=0.25)
      for phase, color in (("insert", "#F4A340"), ("bottom_press", "#E76F7A")):
        indices = np.flatnonzero(phases == phase)
        if len(indices):
          end = min(int(indices[-1]) + 1, len(timestamps) - 1)
          axis.axvspan(timestamps[indices[0]], timestamps[end], color=color, alpha=0.23)
      for boundary in boundaries:
        axis.axvline(boundary, color="gray", lw=0.4, alpha=0.5)
  for axis in axes[-1]:
    axis.set_xlabel("Simulation time (s)")
  fig.suptitle(
    "RAM installation: recorded right-hand solver forces (100 Hz, unfiltered)"
  )
  from matplotlib.patches import Patch

  fig.legend(
    handles=[
      Patch(color="#F4A340", alpha=0.23, label="Insertion"),
      Patch(color="#E76F7A", alpha=0.23, label="Bottom press"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.975),
    ncol=2,
  )
  fig.tight_layout(rect=(0, 0, 1, 0.95))
  fig.savefig(output / "right_hand_force_curves.png", dpi=150)
  fig.savefig(output / "right_hand_force_curves.pdf")
  plt.close(fig)


def _review(file, output: Path):
  """Use the shared layout and clock matching without the poker-only exporter."""
  for modality in (*CAMERAS, "normal", "tangent", "composite"):
    (output / "frames" / modality).mkdir(parents=True)
  (output / "raw").mkdir()
  frames = plan_review_frames(file, fps=10, second_camera="right_wrist")
  force = file["tactile_contact_force"]
  writer = _FfmpegPipeWriter(output / "review.mp4", fps=10, width=1280, height=720)
  try:
    with (output / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
      table = csv.DictWriter(stream, fieldnames=[*asdict(frames[0]), "phase"])
      table.writeheader()
      for frame in frames:
        i, k = frame.camera_index, frame.tactile_index
        rgb = {name: file[f"cameras/{name}/rgb"][i] for name in CAMERAS}
        normal = force["normal_taxel_force_n"][k]
        tangent = force["tangent_taxel_force_n"][k]
        phase = file["commands/phase"].asstr()[k]
        composite, panels = compose_review_frame(
          rgb["head"],
          rgb["right_wrist"],
          normal,
          tangent,
          frame=frame,
          width=1280,
          height=720,
          phase=phase,
          heading="RAM INSTALLATION",
          second_camera_label="RIGHT WRIST",
        )
        stem = f"{frame.output_index:06d}"
        for name, array in rgb.items():
          Image.fromarray(array).save(output / "frames" / name / f"{stem}.png")
        for name, panel in {**panels, "composite": composite}.items():
          panel.save(output / "frames" / name / f"{stem}.png")
        np.savez_compressed(
          output / "raw" / f"{stem}.npz",
          normal_taxel_force_n=normal,
          tangent_taxel_force_n=tangent,
          tangent_magnitude_n=np.linalg.norm(tangent, axis=-1),
          right_fingertip_link_names=np.asarray(RIGHT_FINGERTIP_LINK_NAMES),
          **asdict(frame),
        )
        table.writerow({**asdict(frame), "phase": phase})
        writer.write(np.asarray(composite))
    writer.finish()
  except BaseException:
    writer.abort()
    raise
  probe = _probe_video(output / "review.mp4")
  _validate_probe(probe, count=len(frames), fps=10, width=1280, height=720)
  # Check all exported RGB and signed-force artifacts against the saved source.
  for frame in frames:
    stem = f"{frame.output_index:06d}"
    for camera in CAMERAS:
      with Image.open(output / "frames" / camera / f"{stem}.png") as image:
        np.testing.assert_array_equal(
          np.asarray(image), file[f"cameras/{camera}/rgb"][frame.camera_index]
        )
    with np.load(output / "raw" / f"{stem}.npz") as raw:
      for quantity in ("normal_taxel_force_n", "tangent_taxel_force_n"):
        np.testing.assert_array_equal(
          raw[quantity], force[quantity][frame.tactile_index]
        )
  summary = {
    "schema_version": "install_ram_offline_review_v1",
    "completed": True,
    "source_hdf5": "../raw/install_ram_000000.h5",
    "source_sha256": _sha256(Path(file.filename)),
    "camera_names": list(CAMERAS),
    "output_frame_count": len(frames),
    "fps": 10,
    "maximum_tactile_age_s": max(frame.tactile_age_s for frame in frames),
    "last_camera_pose_timestamp_s": frames[-1].camera_pose_timestamp_s,
    "constant_fps_duration_s": len(frames) / 10,
    "last_frame_playback_time_error_s": frames[-1].playback_time_error_s,
    "source_pixels_and_signed_taxels_verified": True,
    "rgb_temporal_interpolation_or_repetition": False,
    "video_validation": probe,
  }
  _json(output / "review.json", summary)
  return summary


def _closeups(file, output: Path):
  """Render two auxiliary closeups from saved states without advancing physics."""
  simulation = RamInstallSimulation()
  metadata = json.loads(file.attrs.get("metadata_json", "{}"))
  layout = metadata.get("initial_position_randomization", {})
  if layout.get("support_body") == "ram_presentation_stand":
    simulation.model.body_pos[simulation.model.body("ram_presentation_stand").id] = (
      layout["support_body_position_m"]
    )
  camera = CameraConfig(
    "ram_closeup", width=960, height=720, depth=False, segmentation=False
  )
  frames = {}
  with WorkcellRenderer(simulation.model, (camera,)) as renderer:
    for label, index in (("initial", 0), ("final", len(file["state/timestamp"]) - 1)):
      data = simulation.data
      data.time = float(file["state/timestamp"][index])
      data.qpos[:] = file["state/qpos"][index]
      data.qvel[:] = file["state/qvel"][index]
      data.ctrl[:] = file["install_ram/actuator_control"][index]
      if "eq_active" in file["install_ram"]:
        data.eq_active[:] = file["install_ram/eq_active"][index]
      mujoco.mj_forward(simulation.model, data)
      filename = f"ram_closeup_{label}.png"
      Image.fromarray(renderer.capture(data, camera)["rgb"]).save(output / filename)
      frames[label] = {
        "file": filename,
        "source_state_index": index,
        "timestamp_s": data.time,
      }
  _json(
    output / "closeups.json",
    {
      "source": "../raw/install_ram_000000.h5",
      "camera": camera.name,
      "method": "Auxiliary offline rendering from saved qpos/qvel/ctrl/eq_active with mj_forward; no physics step.",
      "frames": frames,
    },
  )


def _documentation(output: Path, summary: dict, *, camera_hz: int):
  metrics, result = summary["metrics"], summary["result"]
  dimensions = getattr(config, "DIMENSIONS_M", {})
  dimension_text = "\n".join(
    f"- `{key}`: {value} m" for key, value in dimensions.items()
  )
  names = {
    "free_transport": "自由搬运",
    "insertion_friction": "插入摩擦平台",
    "bottom_press": "到底稳定承压",
  }
  phase_table = "| 阶段 | 样本数 | 拇指 Fn/Ft 均值 (N) | 食指 Fn/Ft 均值 (N) | 底挡均力 (N) |\n| --- | ---: | ---: | ---: | ---: |\n"
  for name, values in summary["metrics"]["force_stages"]["stages"].items():
    if values["available"]:
      phase_table += f"| {names[name]} | {values['samples']} | {values['mean_fn_n'][0]:.4f} / {values['mean_ft_n'][0]:.4f} | {values['mean_fn_n'][1]:.4f} / {values['mean_ft_n'][1]:.4f} | {values['loads']['backstop_load_n']['mean_n']:.4f} |\n"
  (output / "README.md").write_text(
    "# 插内存条示例\n\n"
    f"本目录来自一次实际仿真运行，任务成功：`{result['success']}`；"
    f"录制时长 {metrics['duration_seconds']:.3f} 秒。\n\n"
    "初态为内存条放在被动上料支架，双手在 home 位置张开平放；右手通过执行器运动到支架上方，再下降抓取。"
    "录制包含接近、闭指、抬起、对准、插入和松手。本例只验收标称初始姿态。\n\n"
    "本任务关闭灯光投射阴影，消除腕部近景中的移动阴影斑点；保留共享相机标定、光照和几何遮挡，接触物理不变。\n\n"
    "- [顶部相机与右腕相机、触觉视频](review/review.mp4) · [机器人全局视频](review/global.mp4) · [浏览页](index.html)\n"
    "- [初态近景](review/ram_closeup_initial.png) · [安装终态近景](review/ram_closeup_final.png)\n"
    "- [右手五指 Fn/Ft 曲线](curves/right_hand_force_curves.png) · [PDF](curves/right_hand_force_curves.pdf) · [CSV](curves/right_hand_forces.csv)\n"
    "- [分阶段力/底挡载荷/插深](curves/insertion_force_stages.png) · [阶段统计](curves/force_stages.json) · [原始切向力前后对照](curves/force_comparison.json)\n"
    + (
      "- [500 Hz 插入稳定性对照](curves/insertion_stability_comparison.pdf) · [2–10 Hz 原始力统计](curves/insertion_stability_comparison.json)\n"
      if "physics_force_comparison" in metrics
      else ""
    )
    + "- [原始 HDF5](raw/install_ram_000000.h5) · [原始清单](raw/install_ram_000000.json) · [验证结果](summary.json)\n"
    "- [同一次运行的 500 Hz 原始力](raw/force_trace_500hz.npz) · [500 Hz 高频/加载审计](processing/raw_force_diagnostics_500hz.json) · [100 Hz 有符号力诊断](processing/raw_force_diagnostics_100hz.json)\n"
    "- [逐帧时间和索引](review/frames.csv) · [交付清单](delivery_manifest.json)\n\n"
    "- [原始格式验证](processing/validation.json) · [任务力学校验](processing/physics_audit.json) · [源码溯源](provenance/source_manifest.json)\n\n"
    f"机械臂、手、相机及触觉布局复用 shared 配置。head/left_wrist/right_wrist 原始 RGB 为 320×240、{camera_hz} Hz；复核视频为 10 Hz；"
    "状态与触觉为 100 Hz，物理 500 Hz。review/frames 保存 head、right_wrist、normal、tangent、composite 五组 PNG；"
    "review/raw 保存逐帧 NPZ，保留有符号切向两轴。相机 PNG 与 NPZ 已逐帧精确比对原始 HDF5。\n\n"
    "触觉力来自 MuJoCo 实际接触求解；7×5 单元是保守空间分配，未经真实皮肤标定。"
    "Fn 是单指 35 单元法向力总和，|Ft| 是有符号切向两轴各自求和后的向量模。"
    "数据不平滑、不裁剪、不插值；热图沿用共享固定色标，颜色可能饱和。"
    "任务监测器在积分后刷新 mj_forward，记录状态、触觉及两相机采用同一仿真时刻。"
    "视频保留末帧，恒定帧率可能导致少量播放时长差异。\n\n"
    "另保存同一次运行每个物理步的 500 Hz 原始 Fn、两轴有符号 Ft、接触计数/几何掩码、卡槽载荷和掌心方向，"
    "已按精确时间戳逐点对齐主 HDF5 的所有 100 Hz 力，绝对误差阈值 1e-10 N。"
    "到底承压需在原始 500 Hz 数据上独立满足近底、低速且底挡载荷 ≥1.2 N 连续 ≥0.4 秒；搬运/插入掌心朝下余弦至少 0.35。\n\n"
    "先在槽外完成夹持和对准，再保持腕部 XY、腕部方向和手指目标不变，仅沿 Z 插入和承压；原始存档已逐点核验这些目标恒定。"
    "插入深度 3.5–5.5 mm 内，500 Hz 原始 Ft 的标准差上限为 0.015 N，2–10 Hz 频带 RMS 上限为 0.005 N；统计仅针对实际承载的拇指与食指。\n\n"
    "两张 ram_closeup 辅助近景由原始 HDF5 的初态/终态 qpos、qvel 等数据离线重绘，不推进仿真；"
    "它们与录制的共享相机 RGB 分开标注，详情见 review/closeups.json。\n\n"
    "内存条和卡槽使用近似真实尺度的碰撞几何；端部卡扣固定在打开状态，"
    "不模拟扣合、电气导通及全部 288 根弹片力。尺寸依据 Kingston 标准 DDR4 UDIMM 图纸及 TE DDR4 插槽资料；"
    "详细建模约定见仓库 docs/tasks/install_ram_dimensions.md。"
    "HDF5/install_ram 保存实际任务测量、执行器控制和外力审计；"
    "本示例已检查 RAM 自由关节外加力和 xfrc_applied 为零、终态就位且右手卸载。\n\n"
    "正手方向以原始 palm_down_cosine 核验；实际求解的弹性触片载荷、轴向摩擦阻力、固定侧壁力和底挡载荷分别保存。"
    "弹片滑动摩擦阶段采用实际插深 3.5–5.5 mm；到底阶段采用 bottom_press 最后连续底挡载荷 ≥1.2 N 的至多 0.4 秒。"
    "全程曲线保留所有采样；阶段选择只用于统计。\n\n"
    + phase_table
    + "\n"
    + "前后对照使用相同 100 Hz 原始采样，按阶段报告相邻差、二阶差和 15–50 Hz 能量；频谱计算的去均值/Hann 窗不改变保存或绘制的原始力。"
    "更早版本若没有独立承压保持，其 legacy 底挡接触段不等价于加载保持；轨迹和物理模型也已变化，不能把所有差别归为单一因素。"
    "未接触手指仍为实际零值。更新时，旧例的原始 HDF5、曲线、摘要及源码保存于 provenance/previous_example。\n\n"
    + (
      "旧模型的 500 Hz 严格重放和 11 组单项排查保存在 [独立力诊断](provenance/force_diagnosis/README.md)；"
      "它们是独立重跑，未混入本次原始样本；基线与改版前 HDF5 在共同采样点的 Fn/Ft 完全一致。\n\n"
      if summary["independent_force_diagnostics_archived"]
      else ""
    )
    + (f"模型尺寸（米）：\n\n{dimension_text}\n\n" if dimension_text else "")
    + "重新生成到一个新目录：\n\n"
    "```bash\npixi run record-install-ram-example -- --output-dir datasets/install_ram_example_new\n```\n\n"
    "更新现有同类示例时加 `--replace-existing`，仅验证成功后替换；失败保留旧例和带 .partial 后缀的诊断目录。"
    "HDF5 使用 shared 无损写缓存，原始位值和时钟已与即时写入逐项对照。\n",
    encoding="utf-8",
  )
  (output / "index.html").write_text(
    '<!doctype html><html lang="zh"><meta charset="utf-8"><title>插内存条示例</title>'
    "<style>body{max-width:1280px;margin:24px auto;background:#15171b;color:#eee;font:18px sans-serif}video,img{width:100%}a{color:#8bd}</style>"
    '<h1>插内存条示例</h1><p><a href="README.md">数据说明</a> · <a href="summary.json">验证结果</a></p>'
    '<h2>顶部相机与右腕相机、触觉</h2><video controls src="review/review.mp4"></video>'
    '<h2>机器人全局视角</h2><video controls src="review/global.mp4" poster="review/global_initial.png"></video>'
    '<p>原始分阶段力、卡槽载荷和插入深度</p><img src="curves/insertion_force_stages.png">'
    '<p>阶段原始切向力前后对照</p><a href="curves/force_comparison.json">统计报告</a>'
    + (
      '<img src="curves/tangential_force_comparison.png">'
      if summary["previous_example_archived"]
      else ""
    )
    + '<p>初态辅助近景（从原始状态离线重绘）</p><img src="review/ram_closeup_initial.png">'
    '<p>安装终态辅助近景（从原始状态离线重绘）</p><img src="review/ram_closeup_final.png">'
    '<p>原始右手五指法向力与切向合力</p><img src="curves/right_hand_force_curves.png"></html>',
    encoding="utf-8",
  )


def _provenance(output: Path, model_path: Path):
  """Snapshot task code, shared observation/control code and recursive MJCF."""
  repository = Path(__file__).resolve().parents[4]
  shared = Path(__file__).resolve().parents[2] / "shared"
  sources = set(Path(__file__).parent.glob("*.py"))
  sources.update(
    shared / f"{name}.py"
    for name in (
      "config",
      "simulation",
      "posture",
      "taskspace_recording",
      "pickup_randomization",
      "recording",
      "buffered_h5",
      "tactile",
      "contact_tactile",
      "cameras",
      "rendering",
      "render_backend",
      "task_video",
      "poker_review",
    )
  )
  sources.add(default_fingertip_layout_path())
  sources.add(shared.parent / "tactile/layout.py")
  sources.add(repository / "scripts/workcell/record_install_ram_example.py")
  for name in (
    "install_ram.md",
    "install_ram_dimensions.md",
    "install_ram_force_diagnosis.md",
  ):
    path = repository / "docs" / name
    if path.exists():
      sources.add(path)
  visited = set()

  def include(path):
    path = path.resolve()
    if path in visited:
      return
    visited.add(path)
    sources.add(path)
    for child in ET.fromstring(path.read_bytes()).iter("include"):
      include(path.parent / child.attrib["file"])

  include(model_path)
  records = {}
  for path in sorted(sources):
    relative = path.resolve().relative_to(repository)
    destination = output / "sources" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(path.read_bytes())
    records[str(relative)] = {
      "sha256": _sha256(destination),
      "bytes": destination.stat().st_size,
    }
  manifest = {
    "schema_version": "install_ram_source_provenance_v1",
    "model_fingerprint": model_fingerprint(model_path),
    "source_files": records,
    "dimension_sources": list(getattr(config, "DIMENSION_SOURCES", ())),
    "scope": "Task/shared Python, recursive MJCF, tactile layout and task docs; mesh binaries remain repository assets.",
  }
  _json(output / "source_manifest.json", manifest)
  return manifest


def _archive_previous(output: Path, destination: Path):
  """Retain original raw observations, force curves and their actual source code."""
  summary = json.loads((output / "summary.json").read_text())
  if summary.get("scene") != config.SCENE_NAME or summary.get("completed") is not True:
    raise ValueError(
      f"refusing to replace an unrelated or incomplete dataset: {output}"
    )
  required = ("raw/install_ram_000000.h5", "summary.json", "delivery_manifest.json")
  if not all((output / name).is_file() for name in required):
    raise ValueError(
      "previous RAM example is missing its raw source or completion manifest"
    )
  original_manifest_hash = _sha256(output / "delivery_manifest.json")
  names = [
    output / "summary.json",
    *sorted((output / "raw").glob("*")),
    *sorted((output / "curves").glob("*")),
    *sorted((output / "processing").rglob("*")),
    *sorted((output / "provenance/sources").rglob("*")),
  ]
  for name in ("provenance/source_manifest.json",):
    if (output / name).is_file():
      names.append(output / name)
  # Preserve earlier revisions and diagnostic evidence when replacing an
  # already revised example, rather than losing the first-generation baseline.
  for name in ("previous_example", "force_diagnosis"):
    names.extend(sorted((output / "provenance" / name).rglob("*")))
  files = {}
  for source in names:
    if not source.is_file():
      continue
    relative = source.relative_to(output)
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    digest = _sha256(source)
    if _sha256(target) != digest:
      raise ValueError(f"previous example archive failed byte verification: {relative}")
    files[str(relative)] = {"sha256": digest, "bytes": target.stat().st_size}
  _json(
    destination / "archive_manifest.json",
    {
      "source_dataset": output.name,
      "original_delivery_manifest_sha256": original_manifest_hash,
      "scope": "Unmodified previous raw HDF5/JSON, original curves, task summary and source snapshot; video/PNG frame derivatives can be recreated from raw RGB.",
      "files": files,
    },
  )
  return original_manifest_hash


def _publish_example(partial: Path, output: Path, previous_manifest_hash: str | None):
  """Swap only complete examples; restore the previous directory on rename failure."""
  if previous_manifest_hash is None:
    if output.exists() or output.is_symlink():
      raise FileExistsError(f"output appeared during recording: {output}")
    partial.rename(output)
    return
  if (
    output.is_symlink()
    or _sha256(output / "delivery_manifest.json") != previous_manifest_hash
  ):
    raise RuntimeError(
      "previous example changed during recording; refusing to replace it"
    )
  backup = output.with_name(output.name + ".replaced")
  if backup.exists() or backup.is_symlink():
    raise FileExistsError(f"replacement backup already exists: {backup}")
  output.rename(backup)
  try:
    partial.rename(output)
  except BaseException:
    backup.rename(output)
    raise
  # The archived raw observations and sources are now inside the validated new example.
  try:
    shutil.rmtree(backup)
  except OSError as error:
    print(
      f"New example published; previous backup retained at {backup}: {error}",
      flush=True,
    )


def record_raw_episode(
  output: Path,
  *,
  position_seed=None,
  buffer_rows=64,
  camera_hz: int = 30,
  cameras: tuple[str, ...] = CAMERAS,
):
  """Record one validated RAM episode without videos, frames or plots."""
  from .execution import RamInstallExecutor

  output = output.expanduser().absolute()
  partial = output.with_name(output.name + ".partial")
  if output.exists() or output.is_symlink() or partial.exists() or partial.is_symlink():
    raise FileExistsError(f"raw-only output or partial already exists: {output}")
  output.parent.mkdir(parents=True, exist_ok=True)
  partial.mkdir()
  started = time.monotonic()
  try:
    (partial / "raw").mkdir()
    sim = RamInstallSimulation(position_seed=position_seed)
    executor = RamInstallExecutor(sim)
    executor.prepare()
    capture = WorkcellConfig(
      model_path=sim.model_path,
      control_hz=100,
      camera_hz=camera_hz,
      cameras=tuple(
        CameraConfig(name, depth=False, segmentation=False) for name in cameras
      ),
    )
    raw = partial / "raw/install_ram_000000.h5"
    with RamExampleRecorder(
      raw, executor, capture, buffer_rows=buffer_rows
    ) as recorder:

      def executor_observer(simulation, phase):
        recorder.observe(simulation, phase)

      recorder.record_initial()
      result = executor.run(observer=executor_observer)
      recorder.record_terminal("terminal_settle")
      outcome = {**asdict(result), "object_name": "ram"}
      recorder.set_outcome(outcome)
    _json(raw.with_suffix(".result.json"), outcome)
    validation = validate_episode(raw)
    if not validation.valid:
      raise ValueError(f"raw episode validation failed: {validation.errors}")
    with h5py.File(raw, "r") as file:
      audit = _audit(file, result)
      timings = json.loads(file.attrs["recording_timing_json"])
    validation_sidecar = raw.with_suffix(".json")
    sidecar = json.loads(validation_sidecar.read_text())
    sidecar.update(
      validation=asdict(validation),
      task_audit_passed=True,
      recording_wall_seconds=time.monotonic() - started,
      recorder_timing=timings,
    )
    _json(validation_sidecar, sidecar)
    _publish_example(partial, output, None)
    print(f"Saved validated raw-only RAM episode: {output}", flush=True)
    return {"result": outcome, "validation": asdict(validation), "audit": audit}
  except BaseException as error:
    _json(
      partial / "failure.json",
      {
        "completed": False,
        "scene": config.SCENE_NAME,
        "error_type": type(error).__name__,
        "error": str(error),
      },
    )
    raise


def _archive_diagnostics(source: Path, destination: Path):
  """Keep independently rerun diagnostics distinct from the new recorded episode."""
  source = source.expanduser().resolve()
  manifest = json.loads((source / "manifest.json").read_text())
  destination.mkdir(parents=True)
  for name, expected in manifest["files"].items():
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
      raise ValueError("diagnostic manifest has an unsafe relative path")
    path = source / relative
    if path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]:
      raise ValueError(f"diagnostic source differs from its manifest: {name}")
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
  shutil.copy2(source / "manifest.json", destination / "manifest.json")
  _json(
    destination / "recording_relationship.json",
    {
      "method": "Independent diagnostic reruns; not additional camera/force samples from the new example episode.",
      "previous_source_hdf5": "../previous_example/raw/install_ram_000000.h5",
      "previous_source_sha256": manifest.get("baseline_h5_sha256"),
      "baseline_matches_old_hdf5": manifest.get("baseline_h5_fn_max_delta_n") == 0
      and manifest.get("baseline_h5_signed_ft_max_delta_n") == 0,
    },
  )


def record_example(
  output: Path,
  *,
  replace_existing=False,
  position_seed=None,
  buffer_rows=64,
  diagnostics_dir: Path | None = None,
  camera_hz: int = 30,
  cameras: tuple[str, ...] = CAMERAS,
):
  """Publish a validated example; optionally replace only a known completed RAM example."""
  from .execution import RamInstallExecutor

  output = output.expanduser().absolute()
  partial = output.with_name(output.name + ".partial")
  if output.is_symlink() or partial.exists() or partial.is_symlink():
    raise FileExistsError(
      f"output or partial already exists: {output}; choose a new directory"
    )
  if output.exists() and not replace_existing:
    raise FileExistsError(
      "output already exists; use --replace-existing for a completed RAM example"
    )
  output.parent.mkdir(parents=True, exist_ok=True)
  partial.mkdir()
  started = time.monotonic()
  try:
    previous_manifest_hash = None
    previous_raw = None
    if output.exists():
      previous = partial / "provenance/previous_example"
      previous_manifest_hash = _archive_previous(output, previous)
      previous_raw = previous / "raw/install_ram_000000.h5"
    if diagnostics_dir is not None:
      _archive_diagnostics(diagnostics_dir, partial / "provenance/force_diagnosis")
    (partial / "raw").mkdir()
    sim = RamInstallSimulation(position_seed=position_seed)
    executor = RamInstallExecutor(sim)
    executor.prepare()
    trace = ForceTraceRecorder(sim)
    capture = WorkcellConfig(
      model_path=sim.model_path,
      control_hz=100,
      camera_hz=camera_hz,
      cameras=tuple(
        CameraConfig(name, depth=False, segmentation=False) for name in cameras
      ),
    )
    raw = partial / "raw/install_ram_000000.h5"
    with RamExampleRecorder(
      raw, executor, capture, buffer_rows=buffer_rows
    ) as recorder:
      last_progress = float(sim.data.time)

      def observe(simulation, phase):
        nonlocal last_progress
        trace.record(phase, executor.state)
        recorder.observe(simulation, phase)
        if simulation.data.time - last_progress >= 5:
          recorder._file.flush()
          print(
            f"recording {simulation.data.time:.1f} sim s | {phase} | {time.monotonic() - started:.0f} wall s | state writes {recorder.state_write_seconds:.1f}s | cameras {recorder._camera_write_seconds:.1f}s",
            flush=True,
          )
          last_progress = simulation.data.time

      trace.record("reset", executor.state)
      recorder.record_initial()
      result = executor.run(observer=observe)
      recorder.record_terminal("terminal_settle")
      outcome = {**asdict(result), "object_name": "ram"}
      recorder.set_outcome(outcome)
    trace_path = partial / "raw/force_trace_500hz.npz"
    trace_summary = trace.save(trace_path)
    _json(raw.with_suffix(".result.json"), outcome)
    validation = validate_episode(raw)
    if not validation.valid:
      raise ValueError(f"raw episode validation failed: {validation.errors}")
    print(
      "Episode complete; validating and exporting saved RGB and tactile forces",
      flush=True,
    )
    with h5py.File(raw, "r") as file:
      metrics = _audit(file, result)
      trace_audit = _audit_physics_trace(file, trace_path)
      metrics["physics_trace"] = trace_audit
      _curves(file, partial / "curves")
      if previous_raw is not None:
        with h5py.File(previous_raw, "r") as previous_file:
          stage_report, comparison = write_force_analysis(
            file, partial / "curves", previous_file
          )
      else:
        stage_report, comparison = write_force_analysis(file, partial / "curves")
      metrics["force_stages"] = stage_report
      metrics["stage_force_acceptance"] = _audit_force_stages(stage_report)
      if previous_raw is not None:
        previous_trace = previous_raw.with_name("force_trace_500hz.npz")
        if previous_trace.exists():
          metrics["physics_force_comparison"] = write_physics_force_comparison(
            trace_path, previous_trace, partial / "curves"
          )
      group = file["tactile_contact_force"]
      force_diagnostics = summarize_force_trace(
        group["timestamp"][:],
        file["commands/phase"].asstr()[:],
        group["normal_force_n"][:],
        group["tangent_force_n"][:],
        contact_count=group["contact_count"][:],
      )
      force_diagnostics["source_hdf5"] = "../raw/install_ram_000000.h5"
      force_diagnostics["source_sha256"] = _sha256(raw)
      timings = json.loads(file.attrs["recording_timing_json"])
      review = _review(file, partial / "review")
      review["global"] = export_global_review(file, partial / "review")
      _closeups(file, partial / "review")
    processing = partial / "processing"
    processing.mkdir()
    _json(processing / "validation.json", asdict(validation))
    _json(processing / "physics_audit.json", metrics)
    _json(processing / "raw_force_diagnostics_100hz.json", force_diagnostics)
    _json(
      processing / "raw_force_diagnostics_500hz.json",
      {**trace_summary, "hdf5_alignment_and_bottom_hold": trace_audit},
    )
    provenance = _provenance(partial / "provenance", sim.model_path)
    summary = {
      "scene": config.SCENE_NAME,
      "completed": True,
      "result": outcome,
      "validation": asdict(validation),
      "metrics": metrics,
      "review": review,
      "initial_position_randomization": sim.initial_position_randomization,
      "recording_wall_seconds": time.monotonic() - started,
      "recorder_timing": timings,
      "previous_example_archived": previous_raw is not None,
      "independent_force_diagnostics_archived": diagnostics_dir is not None,
      "model_fingerprint": provenance["model_fingerprint"],
      "source_sha256": {
        str(path.relative_to(Path(__file__).parent)): _sha256(path)
        for path in sorted(Path(__file__).parent.iterdir())
        if path.suffix in (".py", ".xml")
      },
    }
    _json(partial / "summary.json", summary)
    _documentation(partial, summary, camera_hz=camera_hz)
    manifest = {
      "schema_version": "install_ram_example_manifest_v1",
      "completed": True,
      "scene": config.SCENE_NAME,
      "source_hdf5": "raw/install_ram_000000.h5",
      "files": {
        str(path.relative_to(partial)): {
          "sha256": _sha256(path),
          "bytes": path.stat().st_size,
        }
        for path in sorted(partial.rglob("*"))
        if path.is_file()
      },
    }
    _json(partial / "delivery_manifest.json", manifest)
    _publish_example(partial, output, previous_manifest_hash)
    print(f"Saved validated RAM example: {output}", flush=True)
    return summary
  except BaseException as error:
    _json(
      partial / "failure.json",
      {
        "completed": False,
        "scene": config.SCENE_NAME,
        "error_type": type(error).__name__,
        "error": str(error),
      },
    )
    raise


def record_compact_example(output: Path, *, replace_existing=False, **kwargs):
  """Publish the USB-style raw/review/curves layout without retaining old data.

  Replacement is explicit and only happens after the complete new episode and
  both videos pass validation. Failures before publication leave old data intact.
  """
  output = output.expanduser().absolute()
  work = output.with_name(output.name + ".rendering")
  staged = output.with_name(output.name + ".compact")
  backup = output.with_name(output.name + ".replaced")
  for path in (output, work, staged, backup):
    if path.is_symlink():
      raise ValueError(f"refusing a symlink output: {path}")
  if any(path.exists() for path in (work, staged, backup)):
    raise FileExistsError("RAM compact staging or replacement path already exists")
  if output.exists() and not replace_existing:
    raise FileExistsError("output exists; explicit --replace-existing is required")
  summary = record_example(work, **kwargs)
  files = {
    "raw/install_ram_000000.h5": "raw/install_ram_000000.h5",
    "raw/install_ram_000000.json": "raw/install_ram_000000.json",
    "raw/install_ram_000000.result.json": "raw/install_ram_000000.result.json",
    "review/review.mp4": "review/review.mp4",
    "review/global.mp4": "review/robot_global_short_path.mp4",
    "curves/right_hand_force_curves.png": "curves/right_hand_force_curves.png",
  }
  for source, target in files.items():
    destination = staged / target
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(work / source, destination)
    if _sha256(destination) != _sha256(work / source):
      raise ValueError(f"compact export differs from verified source: {source}")
  # Retain audit results inside the existing outcome sidecar, without extra
  # version folders, frame dumps, or archived examples in the delivery.
  outcome_path = staged / "raw/install_ram_000000.result.json"
  outcome = json.loads(outcome_path.read_text())
  outcome["example_validation"] = {
    "raw": summary["validation"],
    "physics": summary["metrics"],
    "review": summary["review"],
    "model_fingerprint": summary["model_fingerprint"],
    "source_sha256": summary["source_sha256"],
    "layout": "usb_example_raw_review_curves",
  }
  _json(outcome_path, outcome)
  existed = output.exists()
  if existed:
    output.rename(backup)
  try:
    staged.rename(output)
  except BaseException:
    if existed:
      backup.rename(output)
    raise
  if existed:
    shutil.rmtree(backup)
  shutil.rmtree(work)
  print(f"Saved compact RAM example without historical datasets: {output}", flush=True)
  return summary
