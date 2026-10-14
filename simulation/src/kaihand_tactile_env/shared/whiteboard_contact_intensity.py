"""Time-weighted whiteboard contact diagnostics and single-expert comparisons.

Forces are nonnegative contact loads in N. ``duration_s`` supplies each
sample's integration weight: solver samples may represent a physical step,
whereas archived observations are only a sampled approximation. Neither
missing channels nor invalid samples are silently replaced by zero.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CONTACT_THRESHOLD_N = 0.05
PHASES = ("pre_contact", "grasp_transport", "board_contact", "post_board")
MAIN_CHANNELS = ("grip_normal_n", "board_normal_n", "hand_board_normal_n")
SAFETY_THRESHOLDS_N = {"board_normal_n": 8.0, "hand_board_normal_n": 0.5}
CHANNEL_LABELS = {
  "grip_normal_n": "右手五指合计抓握法向力",
  "grip_tangent_load_n": "右手五指合计抓握切向载荷",
  "board_normal_n": "板擦与白板的法向接触力",
  "board_tangent_load_n": "板擦与白板的切向载荷",
  "hand_board_normal_n": "手与白板的直接碰撞力",
}
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGER_LABELS = dict(
  zip(FINGERS, ("拇指", "食指", "中指", "无名指", "小指"), strict=True)
)
PHASE_LABELS = {
  "pre_contact": "接触前",
  "grasp_transport": "抓持 / 搬运",
  "board_contact": "板擦接触白板",
  "post_board": "板面接触后",
  "unknown": "状态未知",
}
SCHEMA_VERSION = "kaihand-whiteboard-contact-intensity-v2"


@dataclass
class ContactTrace:
  time_s: np.ndarray
  duration_s: np.ndarray
  channels: dict[str, np.ndarray]
  phases: np.ndarray
  metadata: dict[str, Any]
  signals: dict[str, np.ndarray] = field(default_factory=dict)

  def __post_init__(self) -> None:
    self.time_s = np.asarray(self.time_s, dtype=np.float64)
    self.duration_s = np.asarray(self.duration_s, dtype=np.float64)
    self.phases = np.asarray(self.phases, dtype=str)
    if self.time_s.ndim != 1:
      raise ValueError("contact timestamps must be one-dimensional")
    shape = self.time_s.shape
    if self.duration_s.shape != shape or self.phases.shape != shape:
      raise ValueError(
        "contact timestamps, durations and phases must have the same shape"
      )
    self.channels = {
      str(name): np.asarray(values, dtype=np.float64)
      for name, values in self.channels.items()
    }
    for name, values in self.channels.items():
      if values.shape != shape:
        raise ValueError(f"contact channel {name!r} must have shape {shape}")
    self.signals = {
      str(name): np.asarray(values, dtype=np.float64)
      for name, values in self.signals.items()
    }
    for name, values in self.signals.items():
      if values.shape != shape:
        raise ValueError(f"contact signal {name!r} must have shape {shape}")
    finite_times = self.time_s[np.isfinite(self.time_s)]
    if np.any(np.diff(finite_times) < 0):
      raise ValueError("finite contact timestamps must be nondecreasing")


def contact_phases(channels: dict[str, np.ndarray]) -> np.ndarray:
  """Infer physical contact phases; unavailable evidence stays ``unknown``.

  A board contact takes precedence over grip contact. Once board contact was
  observed, intervals with known absent board contact are ``post_board``.
  This is contact-state segmentation, not the expert controller's task stage.
  """
  if not channels:
    return np.empty(0, dtype="U20")
  first = np.asarray(next(iter(channels.values())))
  if first.ndim != 1 or any(
    np.asarray(v).shape != first.shape for v in channels.values()
  ):
    raise ValueError("phase channels must be one-dimensional with matching shapes")
  count = len(first)
  grip = np.asarray(channels.get("grip_normal_n", np.full(count, np.nan)))
  board = np.asarray(channels.get("board_normal_n", np.full(count, np.nan)))
  phases = np.full(count, "unknown", dtype="U20")
  seen_board = False
  for index in range(count):
    if not np.isfinite(board[index]) or board[index] < 0:
      continue
    if board[index] > CONTACT_THRESHOLD_N:
      phases[index] = "board_contact"
      seen_board = True
    elif seen_board:
      phases[index] = "post_board"
    elif np.isfinite(grip[index]) and grip[index] >= 0:
      phases[index] = (
        "grasp_transport" if grip[index] > CONTACT_THRESHOLD_N else "pre_contact"
      )
  return phases


def _time_mask(trace: ContactTrace) -> np.ndarray:
  return (
    np.isfinite(trace.time_s) & np.isfinite(trace.duration_s) & (trace.duration_s >= 0)
  )


def _weighted_quantile(
  values: np.ndarray, weights: np.ndarray, quantile: float
) -> float | None:
  positive = weights > 0
  if not np.any(positive):
    return None
  values, weights = values[positive], weights[positive]
  order = np.argsort(values, kind="stable")
  cumulative = np.cumsum(weights[order])
  index = min(
    int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left")),
    len(order) - 1,
  )
  return float(values[order[index]])


def _reported_peak(trace: ContactTrace, name: str) -> float | None:
  value = trace.metadata.get("reported_peaks_n", {}).get(name)
  if value is None:
    return None
  try:
    number = float(value)
  except (TypeError, ValueError):
    return None
  return number if np.isfinite(number) and number >= 0 else None


def _exceedance(
  values: np.ndarray, weights: np.ndarray, threshold: float | None
) -> dict | None:
  if threshold is None:
    return None
  duration = float(np.sum(weights))
  seconds = float(np.sum(weights[values > threshold]))
  contact_duration = float(np.sum(weights[values > CONTACT_THRESHOLD_N]))
  return {
    "threshold_n": float(threshold),
    "duration_s": seconds,
    "fraction_of_valid_time": seconds / duration if duration > 0 else None,
    "fraction_of_contact_time": seconds / contact_duration
    if contact_duration > 0
    else None,
  }


def _channel_stats(
  trace: ContactTrace,
  name: str,
  *,
  phase: str | None = None,
  safety_threshold: float | None = None,
  expert_threshold: float | None = None,
) -> dict[str, Any]:
  selection = (
    np.ones(len(trace.time_s), dtype=bool) if phase is None else trace.phases == phase
  )
  known_durations = selection & np.isfinite(trace.duration_s) & (trace.duration_s >= 0)
  input_duration = float(np.sum(trace.duration_s[known_durations]))
  present = name in trace.channels
  raw = trace.channels.get(name, np.full(len(trace.time_s), np.nan))
  valid = selection & _time_mask(trace) & np.isfinite(raw) & (raw >= 0)
  values, weights = raw[valid], trace.duration_s[valid]
  valid_duration = float(np.sum(weights))
  contact = values > CONTACT_THRESHOLD_N
  contact_duration = float(np.sum(weights[contact]))
  peak_index = int(np.argmax(values)) if len(values) else None
  integral = float(np.sum(values * weights)) if len(values) else None
  return {
    "availability": "available"
    if len(values)
    else "no_valid_samples"
    if present
    else "channel_unavailable",
    "samples": int(np.count_nonzero(selection)),
    "valid_samples": int(np.count_nonzero(valid)),
    "excluded_samples": int(np.count_nonzero(selection & ~valid)),
    "input_duration_s": input_duration,
    "valid_duration_s": valid_duration,
    "coverage_fraction": valid_duration / input_duration
    if input_duration > 0
    else None,
    "sampled_peak_n": float(values[peak_index]) if peak_index is not None else None,
    "sampled_peak_time_s": float(trace.time_s[valid][peak_index])
    if peak_index is not None
    else None,
    "reported_peak_n": _reported_peak(trace, name) if phase is None else None,
    "mean_n": integral / valid_duration if valid_duration > 0 else None,
    "p95_n": _weighted_quantile(values, weights, 0.95),
    "contact_mean_n": float(np.sum(values[contact] * weights[contact]))
    / contact_duration
    if contact_duration > 0
    else None,
    "contact_p95_n": _weighted_quantile(values[contact], weights[contact], 0.95),
    "contact_duration_s": contact_duration if len(values) else None,
    "contact_fraction": contact_duration / valid_duration
    if valid_duration > 0
    else None,
    "force_time_integral_n_s": integral,
    "safety_exceedance": _exceedance(values, weights, safety_threshold)
    if len(values)
    else None,
    "expert_p95_exceedance": _exceedance(values, weights, expert_threshold)
    if len(values)
    else None,
  }


def _comparison(rollout: dict, reference: dict | None) -> dict:
  metrics = (
    "sampled_peak_n",
    "reported_peak_n",
    "mean_n",
    "p95_n",
    "contact_mean_n",
    "contact_p95_n",
    "contact_duration_s",
    "valid_duration_s",
    "force_time_integral_n_s",
  )
  result = {}
  for metric in metrics:
    model_value = rollout[metric]
    expert_value = reference[metric] if reference is not None else None
    available = model_value is not None and expert_value is not None
    result[metric] = {
      "rollout": model_value,
      "reference": expert_value,
      "delta": model_value - expert_value if available else None,
      "ratio": model_value / expert_value if available and expert_value > 0 else None,
    }
  return result


def _phase_names(trace: ContactTrace) -> set[str]:
  return set(trace.phases[_time_mask(trace) & (trace.duration_s > 0)]) & set(PHASES)


def _trace_description(trace: ContactTrace) -> dict:
  valid = _time_mask(trace)
  attempted = trace.metadata.get("attempted_steps")
  valid_steps = trace.metadata.get("valid_steps")
  return {
    "samples": len(trace.time_s),
    "valid_time_samples": int(np.count_nonzero(valid)),
    "total_duration_s": float(np.sum(trace.duration_s[valid])),
    "channels": list(trace.channels),
    "signals": list(trace.signals),
    "capture_coverage_fraction": valid_steps / attempted
    if attempted and valid_steps is not None
    else None,
    "phase_duration_s": {
      phase: float(np.sum(trace.duration_s[valid & (trace.phases == phase)]))
      for phase in (*PHASES, "unknown")
    },
    "metadata": trace.metadata,
  }


def _json_safe(value: Any) -> Any:
  if isinstance(value, dict):
    return {str(key): _json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple, np.ndarray)):
    return [_json_safe(item) for item in value]
  if isinstance(value, np.generic):
    return _json_safe(value.item())
  if isinstance(value, float) and not np.isfinite(value):
    return None
  if isinstance(value, Path):
    return str(value)
  return value


def _save_trace_payload(payload: dict, prefix: str, trace: ContactTrace) -> None:
  valid = _time_mask(trace)
  payload[f"{prefix}_time_s"] = trace.time_s[valid]
  payload[f"{prefix}_duration_s"] = trace.duration_s[valid]
  payload[f"{prefix}_phases"] = trace.phases[valid]
  for name, values in trace.channels.items():
    samples = values[valid].copy()
    # NaN is an explicit missing marker in the numeric archive, never a
    # measured zero. JSON metrics exclude these samples and report coverage.
    samples[~np.isfinite(samples) | (samples < 0)] = np.nan
    payload[f"{prefix}_{name}"] = samples
  for name, values in trace.signals.items():
    samples = values[valid].copy()
    samples[~np.isfinite(samples)] = np.nan
    payload[f"{prefix}_signal_{name}"] = samples


def _display(value: float | None) -> str:
  if value is None:
    return "缺失"
  return f"{value:.3g}"


def _plot_indices(values: np.ndarray) -> np.ndarray:
  """Bound line complexity while retaining peaks, troughs and missing gaps."""
  count = len(values)
  if count <= 6000:
    return np.arange(count)
  finite = np.isfinite(values)
  indices = set(np.flatnonzero(np.diff(finite.astype(np.int8))).tolist())
  indices.update(index + 1 for index in tuple(indices))
  boundaries = np.linspace(0, count, 1501, dtype=int)
  for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
    indices.update((start, stop - 1))
    local = np.flatnonzero(finite[start:stop]) + start
    if len(local):
      indices.update(
        (int(local[np.argmin(values[local])]), int(local[np.argmax(values[local])]))
      )
  return np.asarray(sorted(indices), dtype=int)


def _sampling_label(trace: ContactTrace | None) -> str:
  if trace is None:
    return "参考缺失"
  step = trace.metadata.get("physics_timestep_s")
  physics_steps = trace.metadata.get("sampling") == "physics_step"
  hz = trace.metadata.get("observed_hz")
  if physics_steps and step is not None and np.isfinite(step) and step > 0:
    hz = 1.0 / step
  if hz is None:
    intervals = np.diff(trace.time_s[np.isfinite(trace.time_s)])
    intervals = intervals[intervals > 0]
    hz = 1.0 / float(np.median(intervals)) if len(intervals) else None
  frequency = f"{hz:.4g} Hz" if hz is not None and np.isfinite(hz) else "频率未知"
  return f"{frequency} {'物理步采样' if physics_steps else '已记录样本'}"


@contextmanager
def _chinese_plot_style():
  import matplotlib as mpl
  from matplotlib import font_manager

  available = {font.name for font in font_manager.fontManager.ttflist}
  family = next(
    (
      name
      for name in (
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "SimHei",
      )
      if name in available
    ),
    None,
  )
  if family is None:
    raise RuntimeError("中文接触图需要中文字体，例如文泉驿微米黑或 Noto Sans CJK SC")
  with mpl.rc_context({"font.family": family, "axes.unicode_minus": False}):
    yield


def _episode_end(trace: ContactTrace) -> float:
  recorded = trace.metadata.get("trial_sim_seconds")
  if recorded is not None and np.isfinite(recorded) and recorded >= 0:
    origin = trace.metadata.get("source_start_time_s", 0.0)
    return max(0.0, float(recorded) - float(origin))
  valid = _time_mask(trace)
  if not np.any(valid):
    return 0.0
  origin = float(trace.time_s[valid][0])
  return float(np.max(trace.time_s[valid] + trace.duration_s[valid]) - origin)


def _outcome_label(trace: ContactTrace) -> str:
  success = trace.metadata.get("task_success", trace.metadata.get("success"))
  if success is True:
    return "任务成功"
  reason = str(
    trace.metadata.get("termination_reason") or trace.metadata.get("stop_reason") or ""
  )
  reasons = (
    (("hand struck", "direct_hand_board"), "手碰白板"),
    (("time_limit", "time limit"), "达到时间上限"),
    (("board_force", "board normal force"), "板擦接触力超限"),
    (("nonfinite", "solver_warning", "physics", "numerical"), "数值故障"),
    (("drop",), "物体掉落"),
  )
  for words, label in reasons:
    if any(word in reason.lower() for word in words):
      return f"任务未成功：{label}"
  if success is False:
    return "任务未成功" + ("（停止原因见统计文件）" if reason else "")
  return "任务结果未记录"


def _mark_model_end(axis: Any, rollout: ContactTrace, *, label: bool = False) -> None:
  end = _episode_end(rollout)
  axis.axvline(
    end,
    color="#b42318",
    ls="--",
    lw=0.9,
    alpha=0.7,
    label=f"模型结束 {end:.2f} s" if label else None,
  )


def _channel_label(name: str) -> str:
  for finger in FINGERS:
    if name == f"finger_{finger}_normal_n":
      return f"右手{FINGER_LABELS[finger]}法向接触力"
  return CHANNEL_LABELS.get(name, name)


def _plot_channel(
  axis: Any,
  name: str,
  rollout: ContactTrace,
  reference: ContactTrace | None,
  item: dict,
) -> None:
  drawn = False
  for trace, label, color, style in (
    (reference, "专家", "#2563a6", "-"),
    (rollout, "模型", "#e87924", "--"),
  ):
    if trace is None or name not in trace.channels:
      continue
    valid = _time_mask(trace)
    values = trace.channels[name].copy()
    valid &= np.isfinite(values) & (values >= 0)
    if not np.any(valid):
      continue
    values[~valid] = np.nan
    finite_times = trace.time_s[np.isfinite(trace.time_s)]
    origin = finite_times[0] if len(finite_times) else 0.0
    indices = _plot_indices(values)
    axis.plot(
      trace.time_s[indices] - origin,
      values[indices],
      color=color,
      ls=style,
      lw=1.15,
      alpha=0.9,
      label=label,
    )
    drawn = True
  safety = item["thresholds"]["safety_n"]
  if safety is not None:
    axis.axhline(
      safety, color="#b42318", ls=":", lw=1, label=f"仿真保护阈值 {safety:g} N"
    )
  expert_threshold = item["thresholds"]["expert_contact_p95_n"]
  if expert_threshold is not None:
    axis.axhline(
      expert_threshold,
      color="#2563a6",
      ls=":",
      lw=0.85,
      alpha=0.65,
      label="专家接触期 P95（参考）",
    )
  axis.set_title(
    _channel_label(name),
    loc="left",
    fontsize=11,
    fontweight="normal",
  )
  axis.set_ylabel("接触力（N）")
  axis.set_xlabel("各自轨迹的运行时间（s）")
  axis.set_ylim(bottom=0)
  axis.grid(alpha=0.17)
  axis.spines[["top", "right"]].set_visible(False)
  _mark_model_end(axis, rollout, label=name == "grip_normal_n")
  if drawn:
    axis.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
  missing = []
  for trace, label in ((reference, "专家"), (rollout, "模型")):
    if trace is not None and (
      name not in trace.channels
      or not np.any(np.isfinite(trace.channels[name]) & (trace.channels[name] >= 0))
    ):
      peak = _reported_peak(trace, name)
      suffix = f"；已记录物理峰值 {_display(peak)} N" if peak is not None else ""
      missing.append(f"{label}接触曲线缺失{suffix}")
      if label == "模型" and peak is not None and safety is not None and peak > safety:
        missing.append(f"已超过 {safety:g} N 仿真保护阈值；碰撞持续时间未知")
  if missing:
    axis.text(
      0.01,
      0.72 if name == "hand_board_normal_n" else 0.93,
      "\n".join(missing),
      transform=axis.transAxes,
      va="top",
      fontsize=11 if name == "hand_board_normal_n" else 8,
      color="#b42318" if name == "hand_board_normal_n" else "#475569",
      bbox={"facecolor": "#fff1f0", "edgecolor": "#f1b5b0", "alpha": 0.95},
    )
  if not drawn and not missing:
    axis.text(0.5, 0.5, "没有有效接触力样本", transform=axis.transAxes, ha="center")


def _plot_cleaning_progress(
  axis: Any, summary: dict, rollout: ContactTrace, reference: ContactTrace | None
) -> None:
  for trace, label, color, style in (
    (reference, "专家", "#2563a6", "-"),
    (rollout, "模型", "#e87924", "--"),
  ):
    if trace is None:
      continue
    values = trace.signals.get("cleaned_fraction")
    if values is None or not np.any(np.isfinite(values)):
      axis.text(
        0.02,
        0.8 if label == "专家" else 0.63,
        f"{label}清洁进度曲线缺失；终态见图上方",
        transform=axis.transAxes,
        color=color,
        fontsize=9,
      )
      continue
    values = values.copy()
    valid = _time_mask(trace) & np.isfinite(values) & (values >= 0) & (values <= 1)
    if not np.any(valid):
      axis.text(
        0.02,
        0.8 if label == "专家" else 0.63,
        f"{label}清洁进度没有有效时间样本",
        transform=axis.transAxes,
        color=color,
        fontsize=9,
      )
      continue
    values[~valid] = np.nan
    indices = _plot_indices(values)
    times = trace.time_s - trace.time_s[np.isfinite(trace.time_s)][0]
    if trace.metadata.get("cleaned_fraction_clock") == "interval_end":
      times = times + trace.duration_s
    axis.plot(
      times[indices], values[indices] * 100, color=color, ls=style, lw=1.4, label=label
    )
  _mark_model_end(axis, rollout)
  effect = summary["quality"]["effect"]["rollout"]
  no_progress = effect.get("contact_without_progress_duration_s")
  if no_progress is not None and no_progress > 0:
    axis.text(
      0.02,
      0.12,
      f"模型：接触期间未观察到清洁进展 {no_progress:.2f} s",
      transform=axis.transAxes,
      color="#b42318",
      fontsize=9,
      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )
  axis.set_title(
    "清洁进度：接触是否产生了擦除效果", loc="left", fontsize=11, fontweight="normal"
  )
  axis.set_ylabel("已擦除比例（%）")
  axis.set_xlabel("各自轨迹的运行时间（s）")
  axis.set_ylim(-3, 105)
  axis.set_xlim(
    0, max(_episode_end(rollout), _episode_end(reference) if reference else 0, 1) * 1.04
  )
  axis.grid(alpha=0.17)
  axis.spines[["top", "right"]].set_visible(False)
  if axis.get_legend_handles_labels()[0]:
    axis.legend(loc="upper left", fontsize=8)


def _plot_report(
  path: Path, summary: dict, rollout: ContactTrace, reference: ContactTrace | None
) -> None:
  from matplotlib.backends.backend_agg import FigureCanvasAgg
  from matplotlib.figure import Figure

  figure = Figure(figsize=(13, 15), layout="constrained", facecolor="white")
  FigureCanvasAgg(figure)
  axes = figure.subplots(5, 1, gridspec_kw={"height_ratios": [1, 1, 0.7, 0.85, 0.8]})
  figure.suptitle(
    "擦白板接触评估：用力、碰撞与清洁效果"
    if reference is not None
    else "擦白板接触评估：专家参考缺失",
    fontsize=17,
    fontweight="normal",
  )
  effect = summary["quality"]["effect"]
  for side, trace, label, y in (
    ("rollout", rollout, "模型", 0.947),
    ("reference", reference, "专家", 0.928),
  ):
    if trace is None:
      continue
    cleaned = effect[side].get("final_cleaned_fraction")
    percent = f"{cleaned:.1%}" if cleaned is not None else "未知"
    figure.text(
      0.5,
      y,
      f"{label}：{_outcome_label(trace)}  |  结束 {_episode_end(trace):.2f} s  |  最终清洁进度 {percent}",
      ha="center",
      fontsize=10,
      color="#b42318" if side == "rollout" else "#2563a6",
    )
  for axis, name in zip(axes[:3], MAIN_CHANNELS, strict=True):
    _plot_channel(axis, name, rollout, reference, summary["channels"][name])
  _plot_cleaning_progress(axes[3], summary, rollout, reference)
  time_limit = (
    max(_episode_end(rollout), _episode_end(reference) if reference else 0, 1) * 1.04
  )
  for time_axis in axes[:4]:
    time_axis.set_xlim(0, time_limit)
  axis = axes[4]
  axis.axis("off")
  rows = []
  for name, short in zip(
    MAIN_CHANNELS, ("五指抓握", "板擦与白板", "手与白板"), strict=True
  ):
    item = summary["channels"][name]
    model, expert = item["rollout"], item["reference"] or {}
    for metric, label in (
      ("sampled_peak_n", "采样峰值（N）"),
      ("contact_p95_n", "接触期 P95（N）"),
    ):
      comparison = item["comparison"][metric]
      rows.append(
        [
          short,
          label,
          _display(expert.get(metric)),
          _display(model[metric]),
          _display(comparison["delta"]),
          _display(comparison["ratio"]),
        ]
      )
    if (
      model["reported_peak_n"] is not None or expert.get("reported_peak_n") is not None
    ):
      comparison = item["comparison"]["reported_peak_n"]
      rows.append(
        [
          short,
          "已记录物理峰值（N）",
          _display(expert.get("reported_peak_n")),
          _display(model["reported_peak_n"]),
          _display(comparison["delta"]),
          _display(comparison["ratio"]),
        ]
      )
  table = axis.table(
    cellText=rows,
    colLabels=[
      "接触对象",
      "统计量",
      "专家",
      "模型",
      "模型 − 专家",
      "模型 / 专家",
    ],
    colWidths=[0.17, 0.25, 0.14, 0.14, 0.16, 0.14],
    cellLoc="left",
    loc="center",
  )
  table.auto_set_font_size(False)
  table.set_fontsize(8)
  table.scale(1, 1.45)
  for (row, _), cell in table.get_celld().items():
    cell.set_edgecolor("#e0e6ee")
    if row == 0:
      cell.set_facecolor("#edf2f7")
      cell.set_text_props(weight="normal")
    elif row % 2 == 0:
      cell.set_facecolor("#f8fafc")
  phases_note = (
    "阶段强度、时长与五指接触占比见阶段对比图"
    if "phases_plot" in summary["artifacts"]
    else "阶段对比缺少共同接触状态"
  )
  figure.text(
    0.5,
    0.006,
    f"采样：模型 {_sampling_label(rollout)}  |  专家 {_sampling_label(reference)}；低频采样可能遗漏冲击峰值。\n"
    "红色竖线为模型结束时刻；横轴为各自时间，未进行轨迹对齐。专家 P95 是用力参考，红色横线是仿真保护阈值。\n"
    f"接触期统计按 >0.05 N 的持续时长加权；缺失值不补零；单条专家仅供参考。{phases_note}。",
    ha="center",
    va="bottom",
    fontsize=8,
    color="#475569",
  )
  figure.get_layout_engine().set(rect=(0, 0.06, 1, 0.88))
  with path.open("xb") as stream:
    figure.savefig(stream, format="png", dpi=160)
  figure.clear()


def _plot_phases(
  path: Path, summary: dict, rollout: ContactTrace, reference: ContactTrace
) -> None:
  from matplotlib.backends.backend_agg import FigureCanvasAgg
  from matplotlib.figure import Figure
  from matplotlib.patches import Patch

  phases = summary["common_phases"]
  labels = PHASE_LABELS
  x = np.arange(len(phases), dtype=float)
  figure = Figure(figsize=(14, 11), layout="constrained", facecolor="white")
  FigureCanvasAgg(figure)
  axes = figure.subplots(3, 2, squeeze=False)
  figure.suptitle(
    "擦白板阶段对比：接触强度与接触持续性",
    fontsize=17,
    fontweight="normal",
  )
  for row, name in enumerate(MAIN_CHANNELS[:2]):
    phase_stats = summary["channels"][name]["phases"]
    for column, (metric, label) in enumerate(
      (("contact_mean_n", "接触期均值"), ("contact_p95_n", "接触期 P95"))
    ):
      axis = axes[row, column]
      axis.set_title(
        f"{CHANNEL_LABELS[name]} | {label}",
        loc="left",
        fontsize=10,
        fontweight="normal",
      )
      axis.set_xticks(x, [labels[phase] for phase in phases], fontsize=9)
      axis.set_xlim(-0.6, len(phases) - 0.4)
      axis.set_ylabel("接触期间的力（N）")
      axis.spines[["top", "right"]].set_visible(False)
      axis.grid(axis="y", alpha=0.17)
      axis.set_axisbelow(True)
      for side, offset, color in (
        ("reference", -0.19, "#2563a6"),
        ("rollout", 0.19, "#e87924"),
      ):
        for index, phase in enumerate(phases):
          item = phase_stats.get(phase)
          value = item[side][metric] if item is not None else None
          if value is None:
            continue
          bars = axis.bar(index + offset, value, 0.35, color=color, alpha=0.9)
          axis.bar_label(bars, labels=[_display(value)], fontsize=8, padding=3)
      for index, phase in enumerate(phases):
        item = phase_stats.get(phase)
        if item is None or all(
          item[side][metric] is None for side in ("reference", "rollout")
        ):
          axis.text(
            index,
            0.03,
            "缺少可比数据" if item is None else "未接触",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#64748b",
          )
      axis.set_ylim(bottom=0)
      axis.margins(y=0.2)
  for row_axes in axes[:2]:
    # Shared axes can stop autoscaling after the first panel sets its lower
    # bound; derive the common range from bar data, not stale view limits.
    upper = max(
      axis.dataLim.ymax if np.isfinite(axis.dataLim.ymax) else 0.0 for axis in row_axes
    )
    for axis in row_axes:
      axis.set_ylim(0, max(upper * 1.2, 0.1))
  duration_axis, finger_axis = axes[2]
  duration_axis.set_title(
    "各接触状态的持续时间：保留未完成阶段", loc="left", fontsize=10, fontweight="normal"
  )
  duration_axis.set_xticks(
    np.arange(len(PHASES)), [PHASE_LABELS[phase] for phase in PHASES], fontsize=9
  )
  duration_axis.set_ylabel("已观测持续时间（s）")
  for side, offset, color in (
    ("reference", -0.19, "#2563a6"),
    ("rollout", 0.19, "#e87924"),
  ):
    durations = summary[side]["phase_duration_s"]
    values = [durations[phase] for phase in PHASES]
    bars = duration_axis.bar(
      np.arange(len(PHASES)) + offset, values, 0.35, color=color, alpha=0.9
    )
    duration_axis.bar_label(
      bars,
      labels=[f"{value:.2f}" if value > 0 else "未观测" for value in values],
      fontsize=8,
      padding=3,
    )
  duration_axis.margins(y=0.25)
  finger_axis.set_title(
    "板擦接触白板期间：各指接触占比", loc="left", fontsize=10, fontweight="normal"
  )
  finger_axis.set_xticks(
    np.arange(len(FINGERS)), [FINGER_LABELS[finger] for finger in FINGERS]
  )
  finger_axis.set_ylabel("该指与板擦接触时间占比（%）")
  for side, offset, color in (
    ("reference", -0.19, "#2563a6"),
    ("rollout", 0.19, "#e87924"),
  ):
    data = summary["quality"]["continuity"][side]["finger_board_contact"]
    for index, finger in enumerate(FINGERS):
      value = data.get(finger, {}).get("contact_fraction")
      if value is None:
        finger_axis.text(
          index + offset, 1, "缺失", ha="center", fontsize=8, color=color
        )
        continue
      bars = finger_axis.bar(index + offset, value * 100, 0.35, color=color, alpha=0.9)
      coverage = data.get(finger, {}).get("coverage_fraction")
      text = f"{value:.1%}"
      if coverage is not None and coverage < 0.999:
        text += f"\n覆盖 {coverage:.0%}"
      finger_axis.bar_label(bars, labels=[text], fontsize=8, padding=3)
  finger_axis.set_ylim(0, 123)
  for axis in axes[2]:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.17)
    axis.set_axisbelow(True)
  figure.legend(
    handles=[
      Patch(color="#2563a6", label="专家"),
      Patch(color="#e87924", label="模型"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.952),
    ncol=2,
    frameon=False,
    fontsize=10,
  )
  figure.text(
    0.5,
    0.008,
    "板擦接触状态仅表示法向力 >0.05 N，不能据此认定有效擦拭；上方用力统计仅比较双方共有状态。\n"
    "接触期均值 / P95 按时长加权；各指占比以该指已知样本覆盖的板面接触时长为分母。未观测阶段不代表更高效率；缺失值不补零。\n"
    f"模型：{_outcome_label(rollout)}，{_episode_end(rollout):.2f} s；专家：{_outcome_label(reference)}，{_episode_end(reference):.2f} s。\n"
    f"采样：模型 {_sampling_label(rollout)}  |  专家 {_sampling_label(reference)}；单条专家仅供参考，初态与覆盖范围见统计文件。",
    ha="center",
    va="bottom",
    fontsize=8,
    color="#475569",
  )
  figure.get_layout_engine().set(rect=(0, 0.075, 1, 0.88))
  with path.open("xb") as stream:
    figure.savefig(stream, format="png", dpi=160)
  figure.clear()


def _plot_fingers(
  path: Path,
  names: list[str],
  summary: dict,
  rollout: ContactTrace,
  reference: ContactTrace | None,
) -> None:
  from matplotlib.backends.backend_agg import FigureCanvasAgg
  from matplotlib.figure import Figure

  figure = Figure(
    figsize=(13, 2.45 * len(names)), layout="constrained", facecolor="white"
  )
  FigureCanvasAgg(figure)
  axes = np.atleast_1d(figure.subplots(len(names), 1))
  figure.suptitle(
    "右手五指法向接触力：查看各指受力与接触持续性", fontsize=16, fontweight="normal"
  )
  for axis, name in zip(axes, names, strict=True):
    _plot_channel(axis, name, rollout, reference, summary["channels"][name])
    finger = name.removeprefix("finger_").removesuffix("_normal_n")
    parts = []
    for side, label in (("rollout", "模型"), ("reference", "专家")):
      continuity = summary["quality"]["continuity"][side]
      value = (
        continuity["finger_board_contact"].get(finger, {}).get("contact_fraction")
        if continuity
        else None
      )
      text = f"{label} {value:.1%}" if value is not None else f"{label}缺失"
      coverage = (
        continuity["finger_board_contact"].get(finger, {}).get("coverage_fraction")
        if continuity
        else None
      )
      if coverage is not None and coverage < 0.999:
        text += f"（覆盖 {coverage:.0%}）"
      parts.append(text)
    axis.text(
      0.015,
      0.86,
      "板面接触期间，该指接触占比：" + " / ".join(parts),
      transform=axis.transAxes,
      fontsize=8,
      color="#475569",
      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )
  figure.text(
    0.5,
    0.005,
    f"模型 {_sampling_label(rollout)}；专家 {_sampling_label(reference)}。红色竖线为模型结束时刻；接触占比门限 >0.05 N。\n"
    "合计力接近不能证明各指受力相同；不同抓法需结合清洁效果与碰撞情况评价。缺失值不补零。",
    ha="center",
    fontsize=8,
    color="#475569",
  )
  figure.get_layout_engine().set(rect=(0, 0.045, 1, 0.96))
  with path.open("xb") as stream:
    figure.savefig(stream, format="png", dpi=150)
  figure.clear()


def export_contact_intensity(
  output: Path,
  rollout: ContactTrace,
  reference: ContactTrace | None = None,
  *,
  reference_error: str | None = None,
) -> dict[str, Any]:
  """Write additive diagnostics beside a trial summary; never overwrite files.

  Returned metadata is JSON-serializable and may be included in the trial's
  summary. ``phases`` only compares phases observed in both trajectories;
  each phase uses its own expert contact P95, without temporal warping.
  """
  from .whiteboard_contact_quality import build_contact_quality

  output = Path(output)
  names = list(
    dict.fromkeys(
      (
        *MAIN_CHANNELS,
        *rollout.channels,
        *(reference.channels if reference else {}),
        *rollout.metadata.get("reported_peaks_n", {}),
        *(reference.metadata.get("reported_peaks_n", {}) if reference else {}),
      )
    )
  )
  fingers = [
    f"finger_{finger}_normal_n"
    for finger in FINGERS
    if f"finger_{finger}_normal_n" in names
  ]
  artifacts = {
    "metrics": output / "contact_intensity.json",
    "trace": output / "contact_intensity.npz",
    "plot": output / "contact_intensity.png",
  }
  if len(fingers) >= 2:
    artifacts["fingers_plot"] = output / "contact_intensity_fingers.png"
  common_phases = (
    _phase_names(rollout) & _phase_names(reference) if reference else set()
  )
  if common_phases:
    artifacts["phases_plot"] = output / "contact_intensity_phases.png"
  # Reserve optional names too: an earlier report must never be partially
  # mixed with new artifacts, even if the available channel set has changed.
  for path in (
    *artifacts.values(),
    output / "contact_intensity_fingers.png",
    output / "contact_intensity_phases.png",
  ):
    if path.exists():
      raise FileExistsError(f"contact intensity artifact already exists: {path}")
  channels = {}
  thresholds = dict(SAFETY_THRESHOLDS_N)
  thresholds.update(rollout.metadata.get("safety_thresholds_n", {}))
  for name in names:
    safety = thresholds.get(name)
    if safety is not None and (not np.isfinite(safety) or safety < 0):
      raise ValueError(f"invalid safety threshold for {name}")
    expert_stats = (
      _channel_stats(reference, name, safety_threshold=safety) if reference else None
    )
    expert_p95 = expert_stats["contact_p95_n"] if expert_stats else None
    model_stats = _channel_stats(
      rollout, name, safety_threshold=safety, expert_threshold=expert_p95
    )
    phase_stats = {}
    for phase in PHASES:
      if phase not in common_phases:
        continue
      expert_phase = _channel_stats(
        reference, name, phase=phase, safety_threshold=safety
      )
      model_phase = _channel_stats(
        rollout,
        name,
        phase=phase,
        safety_threshold=safety,
        expert_threshold=expert_phase["contact_p95_n"],
      )
      if expert_phase["valid_duration_s"] > 0 and model_phase["valid_duration_s"] > 0:
        phase_stats[phase] = {
          "rollout": model_phase,
          "reference": expert_phase,
          "comparison": _comparison(model_phase, expert_phase),
        }
    channels[name] = {
      "rollout": model_stats,
      "reference": expert_stats,
      "comparison": _comparison(model_stats, expert_stats),
      "phases": phase_stats,
      "thresholds": {"safety_n": safety, "expert_contact_p95_n": expert_p95},
    }
  partial = (
    bool(rollout.metadata.get("capture_error"))
    or bool(rollout.metadata.get("rejected_samples", 0))
    or rollout.metadata.get("physics_valid") is False
    or any(
      item["rollout"]["excluded_samples"]
      for name, item in channels.items()
      if name in rollout.channels
    )
  )
  summary = _json_safe(
    {
      "schema_version": SCHEMA_VERSION,
      "language": "zh-CN",
      "status": "partial"
      if partial
      else "no_valid_samples"
      if not np.any(_time_mask(rollout))
      else "ok"
      if reference is not None
      else "reference_unavailable",
      "reference_status": "available" if reference is not None else "unavailable",
      "reference_error": reference_error,
      "contact_threshold_n": CONTACT_THRESHOLD_N,
      "rollout": _trace_description(rollout),
      "reference": _trace_description(reference) if reference else None,
      "common_phases": [phase for phase in PHASES if phase in common_phases],
      "channels": channels,
      "quality": build_contact_quality(rollout, reference),
      "definitions": {
        "weighting": "按样本覆盖时长 duration_s 加权；P95 使用加权经验分布的逆累积分布",
        "contact": "力 >0.05 N；没有已测接触时长时，接触期统计为 null",
        "phase": "根据物理接触推断状态；不表示有效擦拭或任务成功；用力仅比较双方共有状态",
        "expert_p95_exceedance": "超过专家参考用力水平的时长；不等于保护超限，也不据此判定失败",
        "reported_peak_n": "独立记录的物理峰值；没有原始时间证据时不虚构峰值时刻或曲线",
        "sampled_peak_n": "可用轨迹样本的最大值；历史低频样本可能漏掉短暂冲击",
        "missing": "缺失通道、非有限或负的力样本均排除，不补成测量零值",
        "ratio": "模型 / 专家；专家为零或任一方缺失时为 null",
        "reference_scope": "单条成功专家仅供参考；无专家分布区间，无跨种子结论，无综合评分",
        "units": {"force": "N", "time": "s", "force_time_integral": "N*s"},
      },
      "artifacts": artifacts,
    }
  )
  output.mkdir(parents=True, exist_ok=True)
  payload: dict[str, Any] = {}
  _save_trace_payload(payload, "rollout", rollout)
  if reference is not None:
    _save_trace_payload(payload, "reference", reference)
  payload["metadata_json"] = np.array(json.dumps(summary, allow_nan=False))
  with artifacts["trace"].open("xb") as stream:
    np.savez_compressed(stream, **payload)
  with _chinese_plot_style():
    _plot_report(artifacts["plot"], summary, rollout, reference)
    if "fingers_plot" in artifacts:
      _plot_fingers(artifacts["fingers_plot"], fingers, summary, rollout, reference)
    if "phases_plot" in artifacts:
      _plot_phases(artifacts["phases_plot"], summary, rollout, reference)
  with artifacts["metrics"].open("x", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
    stream.write("\n")
  return summary
