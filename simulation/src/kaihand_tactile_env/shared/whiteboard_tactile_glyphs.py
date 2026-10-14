"""Offline five-finger force-change glyphs on the unchanged expert clock.

Circle area encodes current normal force, not physical contact area. Arrows
encode signed net tangential *increments*, not current force or center motion.
No interpolation, extrapolation, zero padding, or simulation stepping occurs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .whiteboard_contact_intensity import FINGER_LABELS, FINGERS, _chinese_plot_style
from .whiteboard_tactile_comparison import (
  CONTACT_THRESHOLD_N,
  TactileTrace,
  _compatible,
  _durations,
  _json_safe,
  _model_end,
  derive_tactile_trace,
)

SCHEMA_VERSION = "kaihand-whiteboard-tactile-overall-change-v1"
ARTIFACT_NAMES = {
  "metrics": "tactile_overall_change.json",
  "trace": "tactile_overall_change.npz",
  "snapshot_plot": "tactile_overall_change.png",
  "video": "tactile_overall_change.mp4",
}


def _inclusive_valid(
  mask: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
  """Return whether every native observation in each inclusive window is valid."""
  invalid = np.vstack((np.zeros((1, 5), dtype=int), np.cumsum(~mask, axis=0)))
  return invalid[end + 1] - invalid[start] == 0


def _sample_trace(
  trace: TactileTrace, times: np.ndarray, window: float, threshold: float
):
  derived = derive_tactile_trace(trace, contact_threshold_n=threshold)
  _, max_age = _durations(trace)

  def matched(query):
    # Native cumulative physics clocks may differ from a decimal display time
    # by ~1e-12 s. This tolerance corrects rounding, not real future sampling.
    index = np.searchsorted(trace.time_s, query + 1e-9, side="right") - 1
    safe = np.clip(index, 0, len(trace.time_s) - 1)
    valid = (
      (index >= 0)
      & (query <= trace.time_s[-1] + 1e-9)
      & (query - trace.time_s[safe] <= max_age + 1e-9)
    )
    return safe, valid

  current, observed = matched(times)
  previous, previous_observed = matched(times - window)
  gaps = np.r_[False, np.diff(trace.time_s) > max_age + 1e-9]
  gap_prefix = np.cumsum(gaps)
  continuous = (
    observed & previous_observed & (gap_prefix[current] - gap_prefix[previous] == 0)
  )
  normal_valid = observed[:, None] & derived["normal_valid"][current]
  normal = derived["normal_force_n"][current].copy()
  normal[~normal_valid] = np.nan
  center = derived["center_local_mm"][current].copy()
  center[~normal_valid] = np.nan
  contact = normal_valid & derived["contact"][current]
  tangent_native = np.sum(trace.tangent_taxel_force_n, axis=(-3, -2))
  tangent_valid = normal_valid & derived["tangent_valid"][current]
  tangent = tangent_native[current].copy()
  tangent[~tangent_valid] = np.nan
  normal_change_valid = continuous[:, None] & _inclusive_valid(
    derived["normal_valid"], previous, current
  )
  tangent_change_valid = normal_change_valid & _inclusive_valid(
    derived["tangent_valid"], previous, current
  )
  normal_change = (
    derived["normal_force_n"][current] - derived["normal_force_n"][previous]
  )
  normal_change[~normal_change_valid] = np.nan
  tangent_change = tangent_native[current] - tangent_native[previous]
  tangent_change[~tangent_change_valid] = np.nan
  reason = np.zeros((len(times), 5), dtype=np.int8)
  reason[~normal_valid] = 4
  reason[times < trace.time_s[0]] = 1
  reason[times > trace.time_s[-1]] = 2
  reason[(times >= trace.time_s[0]) & (times <= trace.time_s[-1]) & ~observed] = 3
  sampled_time = trace.time_s[current].copy()
  sampled_time[~observed] = np.nan
  previous_time = trace.time_s[previous].copy()
  previous_time[~previous_observed] = np.nan
  return {
    "sample_time_s": sampled_time,
    "previous_sample_time_s": previous_time,
    "normal_force_n": normal,
    "center_local_mm": center,
    "contact": contact,
    "normal_valid": normal_valid,
    "tangent_force_local_n": tangent,
    "tangent_valid": tangent_valid,
    "normal_change_n": normal_change,
    "normal_change_valid": normal_change_valid,
    "tangent_change_local_n": tangent_change,
    "tangent_change_valid": tangent_change_valid,
    "window_continuous": continuous,
    "missing_reason": reason,
    "native_normal_force_n": derived["normal_force_n"],
    "taxel_xy_mm": derived["taxel_xy_mm"],
  }


def _finite_max(values: list[np.ndarray], *, absolute: bool = False) -> float:
  pooled = np.concatenate([np.asarray(value).reshape(-1) for value in values])
  pooled = pooled[np.isfinite(pooled)]
  if absolute:
    pooled = np.abs(pooled)
  return float(np.max(pooled)) if len(pooled) else 0.0


def build_glyph_samples(
  rollout: TactileTrace,
  reference: TactileTrace,
  display_times_s: np.ndarray,
  *,
  window_s: float = 0.1,
  contact_threshold_n: float = CONTACT_THRESHOLD_N,
  arrow_display_cap_n: float = 2.0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
  """Sample both traces without using future samples or bridging missing windows."""
  times = np.asarray(display_times_s, dtype=float)
  if (
    times.ndim != 1
    or not len(times)
    or not np.all(np.isfinite(times))
    or np.any(np.diff(times) <= 0)
    or np.any(times < reference.time_s[0])
    or np.any(times > reference.time_s[-1])
  ):
    raise ValueError("展示时间必须有限、严格递增，且位于专家实际时间范围内")
  if not np.isfinite(window_s) or window_s <= 0:
    raise ValueError("力变化窗口必须有限且大于零")
  if not np.isfinite(arrow_display_cap_n) or arrow_display_cap_n <= 0:
    raise ValueError("箭头显示上限必须有限且大于零")
  _compatible(rollout, reference)
  sampled = {
    "rollout": _sample_trace(rollout, times, window_s, contact_threshold_n),
    "reference": _sample_trace(reference, times, window_s, contact_threshold_n),
  }
  archive = {"display_time_s": times.copy()}
  for source, values in sampled.items():
    archive.update({f"{source}_{key}": value for key, value in values.items()})
  force_max = _finite_max(
    [value["native_normal_force_n"] for value in sampled.values()]
  )
  delta_normal_max = _finite_max(
    [value["normal_change_n"] for value in sampled.values()], absolute=True
  )
  delta_tangent_max = _finite_max(
    [
      np.linalg.norm(value["tangent_change_local_n"], axis=-1)
      for value in sampled.values()
    ]
  )
  xy = sampled["reference"]["taxel_xy_mm"]
  archive["pad_center_local_mm"] = xy.mean(axis=1)
  pad_span = max(float(np.max(np.ptp(xy, axis=1))), 1.0)
  max_radius = 0.17 * pad_span
  max_arrow = 0.30 * pad_span
  summary = {
    "schema_version": SCHEMA_VERSION,
    "language": "zh-CN",
    "window_s": float(window_s),
    "contact_threshold_n": float(contact_threshold_n),
    "definitions": {
      "clock": "专家原始仿真时间；双方各按自身已过仿真时间对照，不拉伸或时间规整。",
      "matching": "向左取最近原始样本；样本年龄最多为本轨迹名义间隔的 1.5 倍，不外推。",
      "center": "当前每指法向力加权的指尖局部受力中心，单位 mm；无接触时中心不存在。",
      "panel_origin": "各指图形以本指网格位置的均值为原点，仅平移，不旋转或翻转。归档 center_local_mm 保留原始局部坐标。",
      "circle": "圆面积与当前每指总法向力成正比，半径与力平方根成正比；不是实际接触面积。",
      "color": "红色表示法向力增加，蓝色表示减少；灰色表示变化未知。",
      "arrow": "切向力向量增量 ΔFt=Ft(t)-Ft(t-窗口)，单位 N；不是当前力、中心位移或 N/s。",
      "tangent": "逐触觉单元的有符号切向分量先求和；相反方向可能抵消，不能当作总摩擦载荷。",
      "window": "固定物理时间窗口；保留两端实际取样时刻。不跨越采样缺口或该指无效观测。",
      "zero": "真实无接触与缺失不同：无接触不画受力中心及箭头；未知不补零。",
      "pressure": "网格为仿真接触力的核分配估计，不是柔性皮肤压力真值，也不能单独判定打滑。",
    },
    "missing_reason_codes": {
      0: "已观测",
      1: "尚未观测",
      2: "观测已结束",
      3: "采样缺口",
      4: "该指法向力无效",
    },
    "scales": {
      "normal_force_max_n": force_max,
      "normal_force_scale_source": "双方全部原始有效法向力样本的最大值，不仅是视频抽帧。",
      "normal_change_abs_max_n": delta_normal_max,
      "tangent_change_max_n": delta_tangent_max,
      "max_circle_radius_mm": max_radius,
      "circle_radius_squared_mm2_per_n": max_radius**2 / max(force_max, 1e-12),
      "arrow_mm_per_n": max_arrow / arrow_display_cap_n,
      "arrow_display_cap_n": float(arrow_display_cap_n),
      "arrow_clipping": "超过显示上限时箭头以虚线截顶，旁边仍显示实际完整增量 N；数据不截断。",
      "shared": "五指、双方及所有展示时刻统一尺度；不自动逐帧缩放，法向力圆面积不截断。",
    },
    "rollout": {"end_time_s": _model_end(rollout), "metadata": rollout.metadata},
    "reference": {
      "end_time_s": float(reference.time_s[-1]),
      "metadata": reference.metadata,
    },
    "display_samples": len(times),
  }
  return _json_safe(summary), archive


def _panel_setup(axis, xy, extent, title):
  from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

  low, high = xy.min(axis=0), xy.max(axis=0)
  axis.add_patch(
    Rectangle(low, *(high - low), facecolor="#f1f5f9", edgecolor="#cbd5e1")
  )
  axis.scatter(xy[:, 0], xy[:, 1], s=5, color="#cbd5e1", zorder=1)
  axis.set_xlim(extent[0], extent[1])
  axis.set_ylim(extent[2], extent[3])
  axis.set_aspect("equal", adjustable="box")
  axis.set_title(title, fontsize=10)
  axis.tick_params(labelsize=7)
  axis.grid(alpha=0.12)
  circle = Circle(
    (0, 0), radius=0, edgecolor="#475569", linewidth=0.7, alpha=0.83, zorder=3
  )
  axis.add_patch(circle)
  arrow = FancyArrowPatch(
    (0, 0),
    (0, 0),
    arrowstyle="-|>",
    mutation_scale=1e-4,
    shrinkA=0,
    shrinkB=0,
    color="#111827",
    linewidth=1.5,
    zorder=4,
  )
  axis.add_patch(arrow)
  status = axis.text(
    0.5,
    0.53,
    "",
    ha="center",
    va="center",
    transform=axis.transAxes,
    fontsize=9,
    color="#64748b",
    zorder=5,
  )
  label = axis.text(
    0.02,
    0.025,
    "",
    transform=axis.transAxes,
    fontsize=7,
    color="#334155",
    va="bottom",
    zorder=5,
  )
  return circle, arrow, status, label


def _draw_sample(panel, source, finger, index, archive, summary, color_map, color_norm):
  circle, arrow, status, label = panel
  force = archive[f"{source}_normal_force_n"][index, finger]
  known = archive[f"{source}_normal_valid"][index, finger]
  contact = archive[f"{source}_contact"][index, finger]
  delta = archive[f"{source}_normal_change_n"][index, finger]
  vector = archive[f"{source}_tangent_change_local_n"][index, finger]
  circle.set_visible(bool(contact))
  arrow.set_visible(False)
  status.set_text("")
  if not known:
    reason = int(archive[f"{source}_missing_reason"][index, finger])
    status.set_text(summary["missing_reason_codes"][str(reason)])
    label.set_text("未知；不补零")
    return
  delta_label = f"ΔFn {delta:+.3f} N" if np.isfinite(delta) else "ΔFn 未知"
  if not contact:
    status.set_text("无接触")
    label.set_text(f"Fn {force:.3f} N\n{delta_label}；无受力中心")
    return
  center = (
    archive[f"{source}_center_local_mm"][index, finger]
    - archive["pad_center_local_mm"][finger]
  )
  circle.center = center
  circle.radius = float(
    np.sqrt(force * summary["scales"]["circle_radius_squared_mm2_per_n"])
  )
  circle.set_facecolor(
    color_map(color_norm(delta)) if np.isfinite(delta) else "#94a3b8"
  )
  if np.all(np.isfinite(vector)):
    magnitude = float(np.linalg.norm(vector))
    if magnitude > 1e-12:
      cap = summary["scales"]["arrow_display_cap_n"]
      factor = min(1.0, cap / magnitude)
      arrow.set_positions(
        center, center + vector * factor * summary["scales"]["arrow_mm_per_n"]
      )
      # A fixed-size arrowhead would turn tiny increments into a large triangle
      # and undermine the shared magnitude scale. Head size scales with the
      # visible vector, and no default endpoint shrinking can reverse it.
      visible_fraction = min(1.0, magnitude / cap)
      arrow.set_mutation_scale(max(1e-4, 11.0 * visible_fraction))
      arrow.set_linewidth(max(1e-4, 1.5 * np.sqrt(visible_fraction)))
      arrow.set_linestyle("--" if magnitude > cap else "-")
      arrow.set_visible(True)
    tangent_label = f"|ΔFt| {magnitude:.3f} N" + (
      "（箭头截顶）" if magnitude > summary["scales"]["arrow_display_cap_n"] else ""
    )
  else:
    tangent_label = "ΔFt 未知"
  label.set_text(f"Fn {force:.3f} N\n{delta_label}\n{tangent_label}")


def _figure_setup(archive, summary, pairs, *, movie=False):
  import matplotlib.pyplot as plt
  from matplotlib.cm import ScalarMappable
  from matplotlib.colors import Normalize

  rows = len(pairs)
  figure, axes = plt.subplots(rows, 5, figsize=(15.5, 3.0 * rows + 1.3), squeeze=False)
  xy = archive["reference_taxel_xy_mm"] - archive["pad_center_local_mm"][:, None]
  scale = summary["scales"]
  max_arrow = scale["arrow_mm_per_n"] * min(
    scale["tangent_change_max_n"], scale["arrow_display_cap_n"]
  )
  margin = scale["max_circle_radius_mm"] + max_arrow + 0.4
  low, high = xy.min(axis=(0, 1)), xy.max(axis=(0, 1))
  extent = (low[0] - margin, high[0] + margin, low[1] - margin, high[1] + margin)
  panels = []
  for row, (source, time) in enumerate(pairs):
    source_label = "专家" if source == "reference" else "模型"
    for finger, name in enumerate(FINGERS):
      title = f"{source_label} · {FINGER_LABELS[name]}"
      if time is not None:
        title += f" · {time:.2f} s"
      panels.append(
        (source, finger, _panel_setup(axes[row, finger], xy[finger], extent, title))
      )
      if finger == 0:
        axes[row, finger].set_ylabel("局部行方向 / mm", fontsize=8)
      if row == rows - 1:
        axes[row, finger].set_xlabel("局部列方向 / mm", fontsize=8)
  delta_limit = max(scale["normal_change_abs_max_n"], 1e-6)
  norm = Normalize(-delta_limit, delta_limit)
  cmap = plt.get_cmap("RdBu_r")
  figure.subplots_adjust(
    left=0.055,
    right=0.90,
    bottom=0.12 if movie else 0.08,
    top=0.87 if movie else 0.94,
    wspace=0.28,
    hspace=0.32,
  )
  figure.colorbar(
    ScalarMappable(norm=norm, cmap=cmap),
    cax=figure.add_axes((0.93, 0.26, 0.012, 0.48)),
    label="法向力增量 ΔFn / N（蓝减、红增）",
  )
  title = figure.suptitle(
    "五指整体触觉变化｜圆为当前力，箭头为力变化", fontsize=16, y=0.985
  )
  figure.text(
    0.035,
    0.018,
    f"固定窗口 {summary['window_s']:.2f} s；圈面积 ∝ 当前 Fn，不是实际接触面积；圆心为法向力加权中心。\n"
    f"箭头 ΔFt 为切向力增量，不是当前力或中心运动；1 N → {scale['arrow_mm_per_n']:.2f} mm；超过 {scale['arrow_display_cap_n']:g} N 虚线截顶，数值仍为真值。\n"
    "灰色为变化未知；无接触不画中心；模型结束后留空。五指、双方和全程统一尺度；时间差包含动作进度差异。",
    fontsize=8 if movie else 9,
    color="#475569",
  )
  return figure, panels, cmap, norm, title


def _plot_snapshots(path, archive, summary, times):
  import matplotlib.pyplot as plt

  pairs = [(source, time) for time in times for source in ("reference", "rollout")]
  figure, panels, cmap, norm, _ = _figure_setup(archive, summary, pairs)
  for row, (_, time) in enumerate(pairs):
    index = int(np.searchsorted(archive["display_time_s"], time))
    for source, finger, panel in panels[row * 5 : (row + 1) * 5]:
      _draw_sample(panel, source, finger, index, archive, summary, cmap, norm)
  figure.savefig(path, dpi=130)
  plt.close(figure)


def _plot_movie(path, archive, summary, frame_times, video_fps):
  import matplotlib.pyplot as plt
  from matplotlib.animation import FFMpegWriter

  figure, panels, cmap, norm, title = _figure_setup(
    archive, summary, [("reference", None), ("rollout", None)], movie=True
  )
  writer = FFMpegWriter(
    fps=video_fps, codec="libx264", bitrate=2200, extra_args=["-pix_fmt", "yuv420p"]
  )
  try:
    with writer.saving(figure, str(path), dpi=100):
      for time in frame_times:
        index = int(np.searchsorted(archive["display_time_s"], time))
        suffix = (
          "｜模型已结束，后续不补零" if time > summary["rollout"]["end_time_s"] else ""
        )
        title.set_text(f"五指整体触觉变化｜专家时间线 {time:.2f} s {suffix}")
        for source, finger, panel in panels:
          _draw_sample(panel, source, finger, index, archive, summary, cmap, norm)
        writer.grab_frame()
  finally:
    plt.close(figure)


def export_tactile_glyphs(
  rollout: TactileTrace,
  reference: TactileTrace,
  output_dir: Path,
  *,
  snapshot_times_s=(20.0, 28.3, 34.0),
  window_s: float = 0.1,
  contact_threshold_n: float = CONTACT_THRESHOLD_N,
  frame_interval_s: float = 0.1,
  include_video: bool = True,
  video_fps: float = 10.0,
  arrow_display_cap_n: float = 2.0,
) -> dict[str, Any]:
  """Create additive Chinese snapshots/video, never overwrite prior artifacts."""
  snapshots = np.asarray(snapshot_times_s, dtype=float)
  if (
    snapshots.ndim != 1
    or not len(snapshots)
    or not np.all(np.isfinite(snapshots))
    or np.any(np.diff(snapshots) <= 0)
    or np.any(snapshots < reference.time_s[0])
    or np.any(snapshots > reference.time_s[-1])
  ):
    raise ValueError("关键时刻必须有限、严格递增，且位于专家实际时间范围内")
  if not np.isfinite(frame_interval_s) or frame_interval_s <= 0:
    raise ValueError("视频抽帧间隔必须有限且大于零")
  if not np.isfinite(video_fps) or video_fps <= 0:
    raise ValueError("视频帧率必须有限且大于零")
  if include_video and not np.isclose(frame_interval_s * video_fps, 1.0):
    raise ValueError("视频帧率与抽帧间隔必须对应原始仿真时间，不允许隐式加速")
  output_dir = Path(output_dir)
  paths = {
    key: output_dir / name
    for key, name in ARTIFACT_NAMES.items()
    if include_video or key != "video"
  }
  for path in paths.values():
    if path.exists():
      raise FileExistsError(f"整体触觉变化文件已存在：{path}")
  if include_video:
    from matplotlib.animation import FFMpegWriter

    if not FFMpegWriter.isAvailable():
      raise RuntimeError("生成触觉变化视频需要 ffmpeg；可关闭视频，仅导出图片和数据")
  frames = np.minimum(
    np.arange(reference.time_s[0], reference.time_s[-1] + 1e-9, frame_interval_s),
    reference.time_s[-1],
  )
  # The nonuniform final endpoint is archived but not appended as an extra movie
  # frame: movie cadence remains exactly its declared physical sampling period.
  display = np.unique(np.r_[frames, snapshots, reference.time_s[-1]])
  summary, archive = build_glyph_samples(
    rollout,
    reference,
    display,
    window_s=window_s,
    contact_threshold_n=contact_threshold_n,
    arrow_display_cap_n=arrow_display_cap_n,
  )
  summary["snapshot_times_s"] = snapshots.tolist()
  summary["video"] = {
    "generated": bool(include_video),
    "frame_interval_s": float(frame_interval_s),
    "fps": float(video_fps),
    "frames": len(frames) if include_video else 0,
    "last_frame_time_s": float(frames[-1]) if include_video else None,
    "speed": "原始仿真时间，不加速；末尾不足一帧的实际终点保留在数据中。",
  }
  archive["video_frame_time_s"] = frames if include_video else np.array([], dtype=float)
  output_dir.mkdir(parents=True, exist_ok=True)
  with _chinese_plot_style():
    _plot_snapshots(paths["snapshot_plot"], archive, summary, snapshots)
    if include_video:
      _plot_movie(paths["video"], archive, summary, frames, video_fps)
  summary["artifacts"] = {key: str(path) for key, path in paths.items()}
  with paths["trace"].open("xb") as stream:
    np.savez_compressed(stream, **archive)
  with paths["metrics"].open("x", encoding="utf-8") as stream:
    json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write("\n")
  return {"summary": summary, "artifacts": summary["artifacts"]}
