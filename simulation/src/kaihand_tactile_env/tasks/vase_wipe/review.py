"""Synchronized task-local review, matching the USB/poker example layout."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from kaihand_tactile_env.shared.config import (
  TRAINING_CAMERA_NAMES,
  CameraConfig,
  WorkcellConfig,
)
from kaihand_tactile_env.shared.recording import (
  EpisodeRecorder,
  _append,
  _stream,
  validate_episode,
)
from kaihand_tactile_env.shared.rendering import WorkcellRenderer
from kaihand_tactile_env.shared.tactile import SolverContactTactileProvider
from kaihand_tactile_env.shared.task_video import _FfmpegPipeWriter

from . import config

FINGERS = ("thumb", "index", "middle", "ring", "little")


class VaseEpisodeRecorder(EpisodeRecorder):
  """Shared raw episode plus vase-specific deformation and cleaning state."""

  def __init__(self, path, sim, capture_config, *, buffer_rows=128):
    super().__init__(
      path,
      sim,
      capture_config,
      metadata={
        "scene": config.SCENE_NAME,
        "recording_contract": "vase_wipe_shared_raw_v1",
        "stain_randomization": sim.stain_randomization,
      },
      capture_taskspace=True,
      buffer_rows=buffer_rows,
    )

  def _values(self):
    sim = self.sim
    return {
      "minimum_sponge_contact_distance_m": (
        np.nan
        if sim.minimum_sponge_contact_distance_m is None
        else sim.minimum_sponge_contact_distance_m
      ),
      "minimum_element_volume_ratio": sim.minimum_element_volume_ratio,
      "qfrc_applied": sim.data.qfrc_applied.copy(),
      "wrist_force_world_n": sim.wrist_force_world().copy(),
      "native_wrist_force_world_n": sim.native_wrist_force_world().copy(),
      "wrist_free_baseline_x_n": sim._wrist_force_baseline,
      "grasp_rolling_torque_world_n_m": sim.grasp_patch.torque_world_n_m.copy(),
      "grasp_patch_normal_load_n": sim.grasp_patch.normal_load_n,
      "flex_vertices_m": sim.data.flexvert_xpos.copy(),
      "remaining_dirt": sim.dirt.copy(),
      "patch_work_j": sim.cleaning.work_j.copy(),
      "patch_contact_tangent_load_n": sim.patch_contact_tangent_load_n.copy(),
      "patch_sliding_speed_m_s": sim.patch_sliding_speed_m_s.copy(),
      "patch_friction_power_w": sim.patch_friction_power_w.copy(),
      "patch_stroke_m": sim.cleaning.stroke_m.copy(),
      "patch_loaded_time_s": sim.cleaning.loaded_time_s.copy(),
      "wall_force_n": np.array([sim.wall_normal_force_n, sim.wall_tangent_force_n]),
      "table_support_force_n": sim.table_support_force_n,
    }

  def _initialize(self, metadata):
    # The flex-aware provider preserves the exact task force maps in the shared
    # tactile_contact_force group.  The aggregate link stream remains the
    # standard shared solver-contact view.
    self.contact_force_provider = self.sim.forces
    super()._initialize(metadata)
    group = self._file.create_group("vase_wipe")
    group.attrs["force_source"] = self.sim.forces.source
    group.attrs["wrist_force_source"] = "native_flange_plus_flex_contact_correction_v1"
    for name, value in self._values().items():
      array = np.asarray(value)
      _stream(group, name, array.shape, array.dtype)

  def _record_state(self, phase):
    super()._record_state(phase)
    for name, value in self._values().items():
      _append(self._file[f"vase_wipe/{name}"], value)


class RawCapture:
  """Three-camera HDF5 capture without any rendered review derivatives."""

  def __init__(
    self,
    sim,
    directory,
    *,
    camera_hz=30,
    cameras=TRAINING_CAMERA_NAMES,
    buffer_rows=128,
  ):
    self.directory = Path(directory)
    self.directory.mkdir(parents=True, exist_ok=False)
    (self.directory / "raw").mkdir()
    capture = WorkcellConfig(
      model_path=sim.model_path,
      physics_hz=round(1.0 / sim.timestep),
      control_hz=500,
      camera_hz=camera_hz,
      cameras=tuple(
        CameraConfig(name, width=320, height=240, depth=False, segmentation=False)
        for name in cameras
      ),
      tactile_provider=SolverContactTactileProvider.source,
    )
    self.sim = sim
    self.raw_recorder = VaseEpisodeRecorder(
      self.directory / "raw/episode.h5",
      sim,
      capture,
      buffer_rows=buffer_rows,
    )
    self.raw_recorder.record_initial("tabletop_ready")
    sim.physics_observer = self.record

  def record(self, sim):
    self.raw_recorder.observe(sim, getattr(sim, "phase", "tabletop_ready"))

  def observe(self, sim, phase, state):
    del sim, phase, state

  def finish(self, result):
    result["stain_randomization"] = self.sim.stain_randomization
    result["contact_integrity"] = self.sim.contact_integrity_report()
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
    (self.directory / "result.json").write_text(
      json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    validation_sidecar = self.raw_recorder.path.with_suffix(".json")
    sidecar = json.loads(validation_sidecar.read_text())
    sidecar.update(
      validation=validation_result,
      task_audit_passed=result.get("success") is True,
    )
    validation_sidecar.write_text(json.dumps(sidecar, indent=2, allow_nan=False) + "\n")
    if not validation.valid:
      raise ValueError(f"shared raw validation failed: {validation.errors}")

  def close(self):
    self.sim.physics_observer = None
    if not self.raw_recorder._closed:
      self.raw_recorder.close(finalize=False)


class Review:
  def __init__(
    self,
    sim,
    directory,
    *,
    camera_hz=30,
    cameras=TRAINING_CAMERA_NAMES,
  ):
    self.directory = Path(directory)
    self.directory.mkdir(parents=True, exist_ok=False)
    for name in ("curves", "raw", "review"):
      (self.directory / name).mkdir(parents=True, exist_ok=True)
    self.cameras = tuple(
      CameraConfig(n, width=640, height=360, depth=False, segmentation=False)
      for n in ("head", "right_wrist", "vase_closeup")
    )
    self.renderer = WorkcellRenderer(
      sim.model, self.cameras, visible_geom_groups=(0, 1, 2), shadows=False
    )
    self.writer = _FfmpegPipeWriter(
      self.directory / "review/review.mp4", fps=10, width=1280, height=720
    )
    self.global_writer = _FfmpegPipeWriter(
      self.directory / "review/global.mp4", fps=10, width=1280, height=720
    )
    self.frame_rows, self.samples = [], []
    self.next_sample = 0.002
    self.next_frame = 0.01
    self.sim = sim
    sources = (
      sorted(Path(__file__).parent.glob("*.py"))
      + sorted(Path(__file__).parent.glob("*.xml"))
      + sorted(Path(__file__).parent.glob("*.png"))
    )
    self.source_hashes = {
      p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }
    raw_cameras = tuple(
      CameraConfig(name, width=320, height=240, depth=False, segmentation=False)
      for name in cameras
    )
    capture = WorkcellConfig(
      model_path=sim.model_path,
      physics_hz=round(1.0 / sim.timestep),
      control_hz=500,
      camera_hz=camera_hz,
      cameras=raw_cameras,
      tactile_provider=SolverContactTactileProvider.source,
    )
    self.raw_recorder = VaseEpisodeRecorder(
      self.directory / "raw/episode.h5", sim, capture
    )
    self.raw_recorder.record_initial("tabletop_ready")
    sim.physics_observer = self.record

  def record(self, sim):
    self.raw_recorder.observe(sim, getattr(sim, "phase", "tabletop_ready"))
    if sim.data.time + 1e-9 < self.next_sample:
      return
    self.next_sample += 0.002
    sample = sim.forces.read(sim.data)
    self.samples.append(
      {
        "time_s": float(sim.data.time),
        "phase": getattr(sim, "phase", "tabletop_ready"),
        "minimum_sponge_contact_distance_m": sim.minimum_sponge_contact_distance_m,
        "minimum_element_volume_ratio": sim.minimum_element_volume_ratio,
        "qpos": sim.data.qpos.copy(),
        "qvel": sim.data.qvel.copy(),
        "ctrl": sim.data.ctrl.copy(),
        "qfrc_applied": sim.data.qfrc_applied.copy(),
        "normal_taxel_force_n": sample.normal_taxel_force_n[5:].copy(),
        "tangent_taxel_force_n": sample.tangent_taxel_force_n[5:].copy(),
        "normal_force_n": sample.normal_force_n[5:].copy(),
        "pad_contact_count": sample.contact_count[5:].copy(),
        "pad_force_world_n": sample.force_world_n[5:].copy(),
        "wrist_force_world_n": sim.wrist_force_world().copy(),
        "native_wrist_force_world_n": sim.native_wrist_force_world().copy(),
        "wrist_free_baseline_x_n": sim._wrist_force_baseline,
        "grasp_rolling_torque_world_n_m": sim.grasp_patch.torque_world_n_m.copy(),
        "grasp_patch_normal_load_n": sim.grasp_patch.normal_load_n,
        "tangent_force_n": sample.tangent_force_n[5:].copy(),
        "tangent_contact_load_n": sample.tangent_load_n[5:].copy(),
        "flex_vertices_m": sim.data.flexvert_xpos.copy(),
        "remaining_dirt": sim.dirt.copy(),
        "patch_work_j": sim.cleaning.work_j.copy(),
        "patch_contact_tangent_load_n": sim.patch_contact_tangent_load_n.copy(),
        "patch_sliding_speed_m_s": sim.patch_sliding_speed_m_s.copy(),
        "patch_friction_power_w": sim.patch_friction_power_w.copy(),
        "patch_stroke_m": sim.cleaning.stroke_m.copy(),
        "patch_loaded_time_s": sim.cleaning.loaded_time_s.copy(),
        "wall_force_n": np.array([sim.wall_normal_force_n, sim.wall_tangent_force_n]),
        "table_support_force_n": sim.table_support_force_n,
      }
    )

  @staticmethod
  def _maps(sample, tangent=False):
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
      fn = sample["normal_force_n"][i]
      xy = sample["tangent_force_n"][i]
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

  def observe(self, sim, phase, state):
    if sim.data.time + 1e-9 < self.next_frame:
      return
    self.next_frame += 0.1
    sample = self.samples[-1]
    panels = {}
    for camera in self.cameras:
      panels[camera.name] = Image.fromarray(
        self.renderer.capture(sim.data, camera)["rgb"]
      )
    panels["normal"] = self._maps(sample)
    panels["tangent"] = self._maps(sample, True)
    composite = Image.new("RGB", (1280, 720), "#101923")
    for name, xy in [
      ("head", (0, 0)),
      ("right_wrist", (0, 360)),
      ("normal", (640, 0)),
      ("tangent", (640, 360)),
    ]:
      composite.paste(panels[name], xy)
    draw = ImageDraw.Draw(composite)
    draw.text(
      (8, 8),
      f"HEAD | {phase} | {state.timestamp:.2f}s",
      fill="white",
      stroke_width=1,
      stroke_fill="black",
    )
    draw.text(
      (8, 368), "RIGHT_WRIST", fill="white", stroke_width=1, stroke_fill="black"
    )
    self.writer.write(np.asarray(composite))
    scene = panels["vase_closeup"].resize((1280, 720), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(scene)
    draw.rectangle((0, 0, 1280, 34), fill="#101923")
    draw.rectangle((0, 686, 1280, 720), fill="#101923")
    draw.text((12, 9), "GLOBAL ROBOT VIEW", fill="white")
    draw.text(
      (12, 695),
      f"{phase} | {state.timestamp:.2f}s | cleared {state.cleaned_fraction:.1%} | wall Fn {state.wall_normal_force_n:.2f} N / Ft {state.wall_tangent_force_n:.2f} N | deformation {state.deformation_mm:.1f} mm",
      fill="white",
    )
    self.global_writer.write(np.asarray(scene))
    self.frame_rows.append(
      (
        len(self.frame_rows),
        state.timestamp,
        len(self.samples) - 1,
        sample["time_s"],
        phase,
      )
    )

  def finish(self, result):
    result["stain_randomization"] = self.sim.stain_randomization
    result["contact_integrity"] = self.sim.contact_integrity_report()

    self.raw_recorder.record_terminal("terminal_settle")
    self.raw_recorder.set_outcome(result)
    self.raw_recorder.close()
    validation = validate_episode(self.raw_recorder.path)
    result["raw_validation"] = {
      "valid": validation.valid,
      "errors": list(validation.errors),
      "warnings": list(validation.warnings),
      "state_samples": validation.state_samples,
      "camera_samples": validation.camera_samples,
    }
    if not validation.valid:
      raise ValueError(f"shared raw validation failed: {validation.errors}")
    self.writer.finish()
    self.global_writer.finish()
    directory = self.directory
    inspection = getattr(self.sim, "inspection", None)
    if inspection is not None:
      result["visual_inspections"] = inspection.observations
    data = (
      {key: np.asarray([s[key] for s in self.samples]) for key in self.samples[0]}
      if self.samples
      else {}
    )
    if data:
      self._curves(data)
    self._json("result.json", result)
    self._json(
      "manifest.json",
      {
        "schema": "kaihand_tactile_episode_v1+vase_wipe_shared_raw_v1",
        "physics_hz": 1 / self.sim.timestep,
        "force_hz": 500,
        "video_fps": 10,
        "fingers": FINGERS,
        "force_source": self.sim.forces.source,
        "wrist_force_source": "native_flange_plus_flex_contact_correction_v1",
        "source_sha256": self.source_hashes,
        "stain_randomization": self.sim.stain_randomization,
        "success": result.get("success", False),
        "sample_count": len(self.samples),
        "frame_count": len(self.frame_rows),
      },
    )
    status = (
      "达到清洁标准" if result.get("success") else "未达到清洁标准（保留真实残留）"
    )
    audit = result["contact_integrity"]
    minimum_distance = audit["minimum_sponge_contact_distance_m"]
    distance_label = (
      "尚无接触记录"
      if minimum_distance is None
      else f"{1000 * minimum_distance:.6f} mm"
    )
    (directory / "index.html").write_text(
      f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>软海绵擦花瓶内壁</title>
<style>body{{max-width:1280px;margin:32px auto;padding:0 24px;background:#101923;color:#e7eef5;font:16px/1.7 system-ui}}video,img{{width:100%;background:#080d14;border-radius:10px}}a{{color:#75d7c7}}p{{max-width:1000px}}</style>
<h1>软海绵擦花瓶内壁</h1><p>{status} · 平均清除 {result.get("state", {}).get("cleaned_fraction", 0):.1%} · 合格污渍单元 {result.get("cleaned_patch_count", 0)}/{result.get("patch_count", config.PATCH_ROWS * config.PATCH_COLUMNS)}</p>
<p>动作记录：{"已完成桌面抓取、入瓶、擦拭和退出" if result.get("motion_completed") else result.get("error", "未完成")}</p>
<p>象牙白花瓶的远侧内壁带浅黑色污渍。机器人从共享初始姿势沿短逆时针腕部路径接近桌面上的扁平软海绵，完成抓取、擦拭、抬手复查和退出。</p>
<h2>顶部相机、右腕相机与触觉</h2><video src="review/review.mp4" controls preload="metadata"></video><p>左：共享 head / right_wrist；右：五指 7×5 法向与切向力。原始数据不截断、不平滑。</p>
<h2>机器人全局视角</h2><video src="review/global.mp4" controls preload="metadata"></video>
<h2>右手五指法向与切向力</h2><a href="curves/right_hand_force_curves.pdf"><img src="curves/right_hand_force_curves.png" alt="右手五指法向与切向力曲线"></a><p><a href="curves/right_hand_force_curves.pdf">PDF</a> · <a href="raw/episode.h5">原始 HDF5</a> · <a href="result.json">清洁结果</a></p>
<h2>接触与形变验收</h2><p>每个物理步检查海绵接触、手与花瓶/桌面的接触，以及弹性单元翻转。穿入计数：{audit["penetration_count"]}；海绵最小接触距离：{distance_label}；最小单元体积比：{audit["minimum_element_volume_ratio"]:.4f}。任意负接触距离或单元翻转都会终止执行，不用裁剪状态或平滑力曲线掩盖失败。详细记录见 result.json 的 contact_integrity。</p>
<h2>清洁标准</h2><p>内壁法向力 {config.MIN_WALL_NORMAL_N:g}–{config.MAX_CLEAN_NORMAL_N:g} N、总摩擦力 ≥{config.MIN_WALL_FRICTION_N:.2f} N；每块污渍分配到的摩擦力 ≥{config.MIN_PATCH_FRICTION_N:g} N、实际滑动速度 ≥{1000 * config.MIN_WIPE_SPEED_M_S:g} mm/s 才累计。标准面积污渍需摩擦功 {1000 * config.PATCH_WORK_REQUIRED_J:g} mJ、滑动距离 {1000 * config.PATCH_STROKE_REQUIRED_M:g} mm、有效时间 {config.PATCH_DWELL_REQUIRED_S:g} s；随机大小时，摩擦功按面积比例、滑动距离按面积比例平方根调整，有效时间不变。进度取三项中最小值。平均残留 ≤{100 * config.MAX_MEAN_DIRT:g}%，且每块残留 ≤{100 * config.MAX_PATCH_DIRT:g}%，抬手视觉复查已净并退出后才判定完成。仅接触、静压或低摩擦不清除。这些是可调任务参数，未做真实材料标定。</p></html>""",
      encoding="utf-8",
    )
    (directory / "README.md").write_text(
      """# Vase wipe example

This directory contains one physical rollout in the requested four-part format:

1. `raw/episode.h5`: shared Raw state, action, three-camera, task-space and tactile streams. `result.json` and `manifest.json` describe validation and outcome.
2. `review/review.mp4`: HEAD + RIGHT_WRIST + five-finger normal/tangential tactile maps, 1280×720 at 10 fps.
3. `review/global.mp4`: global robot view from the same timestamps, 1280×720 at 10 fps.
4. `curves/right_hand_force_curves.{png,pdf}`: unfiltered 500 Hz five-finger normal and tangential resultant curves.

The sponge is a 45 × 62 × 110 mm elastic cuboid. Its 62 × 110 mm +X cleaning face is dark green and the other five faces are yellow. The ivory vase carries cool light-black stains on the far inner wall. The right arm starts from shared home and uses the short counterclockwise pickup solution. No frame dumps, per-frame NPZ files, extra diagnostic plots, or duplicate force CSV are included; all numeric signals remain in Raw HDF5.
""",
      encoding="utf-8",
    )

  def _json(self, name, value):
    (self.directory / name).write_text(
      json.dumps(value, indent=2, allow_nan=False) + "\n"
    )

  def _curves(self, data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = data["time_s"]
    fn = data["normal_force_n"]
    xy = data["tangent_force_n"]
    ft = np.linalg.norm(xy, axis=-1)
    fig, axes = plt.subplots(
      2, 1, figsize=(12, 7), sharex=True, constrained_layout=True
    )
    for j, name in enumerate(FINGERS):
      axes[0].plot(times, fn[:, j], label=name, lw=0.8)
      axes[1].plot(times, ft[:, j], label=name, lw=0.8)
    axes[0].set_ylabel("Normal force Fn (N)")
    axes[1].set_ylabel("Tangential resultant |sum(Fx,Fy)| (N)")
    axes[1].set_xlabel("Simulation time (s)")
    axes[0].legend(ncol=5)
    for ax in axes:
      ax.grid(alpha=0.25)
    for ext in ("png", "pdf"):
      fig.savefig(self.directory / f"curves/right_hand_force_curves.{ext}", dpi=160)
    plt.close(fig)

  def close(self):
    self.sim.physics_observer = None
    try:
      if not self.raw_recorder._closed:
        self.raw_recorder.close(finalize=False)
      self.writer.close()
      self.global_writer.close()
    finally:
      self.renderer.close()
