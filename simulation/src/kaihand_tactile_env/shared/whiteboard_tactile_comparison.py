"""Read-only fingertip tactile comparisons on the original expert clock.

Taxel grids are solver-distributed forces, not measured skin pressure. Normal
force determines the contact center and normalized spatial distribution. Signed
tangential vectors are summed before taking their norm; this is a net force,
not the non-cancelling contact load. Missing observations stay missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .whiteboard_contact_intensity import (
  FINGER_LABELS,
  FINGERS,
  _chinese_plot_style,
)

SCHEMA_VERSION = "kaihand-whiteboard-tactile-comparison-v1"
CONTACT_THRESHOLD_N = 0.05
ARTIFACT_NAMES = {
  "metrics": "tactile_contact_comparison.json",
  "trace": "tactile_contact_comparison.npz",
  "timeline_plot": "tactile_contact_timeline.png",
  "centers_plot": "tactile_contact_centers.png",
  "heatmaps_plot": "tactile_contact_heatmaps.png",
}


@dataclass
class TactileTrace:
  time_s: np.ndarray
  normal_taxel_force_n: np.ndarray
  tangent_taxel_force_n: np.ndarray
  taxel_positions_local_m: np.ndarray
  tangent_basis_local: np.ndarray
  metadata: dict[str, Any]

  def __post_init__(self) -> None:
    self.time_s = np.asarray(self.time_s, dtype=float)
    self.normal_taxel_force_n = np.asarray(self.normal_taxel_force_n, dtype=float)
    self.tangent_taxel_force_n = np.asarray(self.tangent_taxel_force_n, dtype=float)
    self.taxel_positions_local_m = np.asarray(self.taxel_positions_local_m, dtype=float)
    self.tangent_basis_local = np.asarray(self.tangent_basis_local, dtype=float)
    if self.time_s.ndim != 1 or not len(self.time_s):
      raise ValueError("触觉时间戳必须是非空的一维数组")
    if not np.all(np.isfinite(self.time_s)) or np.any(np.diff(self.time_s) <= 0):
      raise ValueError("触觉时间戳必须有限且严格递增")
    count = len(self.time_s)
    expected = {
      "normal_taxel_force_n": (count, 5, 7, 5),
      "tangent_taxel_force_n": (count, 5, 7, 5, 2),
      "taxel_positions_local_m": (5, 35, 3),
      "tangent_basis_local": (5, 2, 3),
    }
    for name, shape in expected.items():
      if getattr(self, name).shape != shape:
        raise ValueError(f"{name} 的形状应为 {shape}")
    if not np.all(np.isfinite(self.taxel_positions_local_m)) or not np.all(
      np.isfinite(self.tangent_basis_local)
    ):
      raise ValueError("指尖坐标与切向基必须有限")
    gram = np.einsum("fkc,flc->fkl", self.tangent_basis_local, self.tangent_basis_local)
    if not np.allclose(gram, np.eye(2)[None], atol=1e-6, rtol=0):
      raise ValueError("指尖切向基必须正交且单位化")


def _cadence(trace: TactileTrace) -> float:
  for key in ("nominal_interval_s", "sample_period_s", "nominal_sample_period_s"):
    value = trace.metadata.get(key)
    if value is not None and np.isfinite(value) and value > 0:
      return float(value)
  return float(np.median(np.diff(trace.time_s))) if len(trace.time_s) > 1 else 0.0


def _durations(trace: TactileTrace) -> tuple[np.ndarray, float]:
  """Left-hold observations have no invented terminal interval or gap span."""
  cadence = _cadence(trace)
  max_age = 1.5 * cadence
  duration = np.r_[np.diff(trace.time_s), 0.0]
  # Do not count an entire sparse gap as observed contact duration.
  duration[duration > max_age + 1e-9] = 0.0
  return duration, max_age


def derive_tactile_trace(
  trace: TactileTrace, *, contact_threshold_n: float = CONTACT_THRESHOLD_N
) -> dict[str, np.ndarray]:
  if not np.isfinite(contact_threshold_n) or contact_threshold_n < 0:
    raise ValueError("接触阈值必须有限且非负")
  normal = trace.normal_taxel_force_n
  normal_valid = np.all(np.isfinite(normal) & (normal >= 0), axis=(-2, -1))
  force = np.sum(np.where(np.isfinite(normal), normal, 0), axis=(-2, -1))
  force[~normal_valid] = np.nan
  contact = normal_valid & (force > contact_threshold_n)
  distribution = np.full(normal.shape, np.nan)
  np.divide(
    normal,
    force[..., None, None],
    out=distribution,
    where=contact[..., None, None],
  )
  taxel_xy_mm = (
    np.einsum("fpc,fkc->fpk", trace.taxel_positions_local_m, trace.tangent_basis_local)
    * 1000
  )
  centers = np.einsum(
    "tfp,fpk->tfk", distribution.reshape(len(trace.time_s), 5, 35), taxel_xy_mm
  )
  tangent_valid = np.all(np.isfinite(trace.tangent_taxel_force_n), axis=(-3, -2, -1))
  tangent_net = np.linalg.norm(
    np.sum(trace.tangent_taxel_force_n, axis=(-3, -2)), axis=-1
  )
  tangent_net[~tangent_valid] = np.nan
  duration, _ = _durations(trace)
  return {
    "normal_force_n": force,
    "normal_valid": normal_valid,
    "contact": contact,
    "normalized_normal_distribution": distribution,
    "center_local_mm": centers,
    "tangent_net_force_n": tangent_net,
    "tangent_valid": tangent_valid,
    "duration_s": duration,
    "taxel_xy_mm": taxel_xy_mm,
  }


def _weighted(values: np.ndarray, weights: np.ndarray) -> dict[str, float | None]:
  valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
  values, weights = values[valid], weights[valid]
  if not len(values):
    return {"mean": None, "p95": None}
  order = np.argsort(values, kind="stable")
  cumulative = np.cumsum(weights[order])
  index = min(np.searchsorted(cumulative, 0.95 * cumulative[-1]), len(order) - 1)
  return {
    "mean": float(np.sum(values * weights) / np.sum(weights)),
    "p95": float(values[order[index]]),
  }


def _native_stats(trace: TactileTrace, derived: dict, finger: int) -> dict:
  weights = derived["duration_s"]
  normal_valid = derived["normal_valid"][:, finger]
  contact = derived["contact"][:, finger]
  force = derived["normal_force_n"][:, finger]
  normal_stats = _weighted(force, weights)
  contact_stats = _weighted(force, weights * contact)
  tangent_stats = _weighted(derived["tangent_net_force_n"][:, finger], weights)
  valid_duration = float(np.sum(weights * normal_valid))
  span = float(trace.time_s[-1] - trace.time_s[0])
  return {
    "valid_samples": int(np.count_nonzero(normal_valid)),
    "valid_duration_s": valid_duration,
    "coverage_fraction": valid_duration / span if span > 0 else None,
    "contact_duration_s": float(np.sum(weights * contact)),
    "normal_mean_n": normal_stats["mean"],
    "normal_p95_n": normal_stats["p95"],
    "normal_sampled_peak_n": float(np.max(force[normal_valid]))
    if np.any(normal_valid)
    else None,
    "contact_normal_mean_n": contact_stats["mean"],
    "contact_normal_p95_n": contact_stats["p95"],
    "tangent_net_mean_n": tangent_stats["mean"],
    "tangent_net_p95_n": tangent_stats["p95"],
  }


def _compatible(rollout: TactileTrace, reference: TactileTrace) -> None:
  for name in ("taxel_positions_local_m", "tangent_basis_local"):
    if not np.allclose(
      getattr(rollout, name), getattr(reference, name), atol=1e-8, rtol=0
    ):
      raise ValueError(f"专家与模型的 {name} 不一致，不能比较指尖物理坐标")


def _matched_indices(
  rollout: TactileTrace, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  indices = np.searchsorted(rollout.time_s, times, side="right") - 1
  safe_indices = np.clip(indices, 0, len(rollout.time_s) - 1)
  _, max_age = _durations(rollout)
  valid = (
    (indices >= 0)
    & (times <= rollout.time_s[-1])
    & (times - rollout.time_s[safe_indices] <= max_age + 1e-9)
  )
  return safe_indices, valid


def _json_safe(value: Any) -> Any:
  if isinstance(value, dict):
    return {str(k): _json_safe(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(v) for v in value]
  if isinstance(value, np.ndarray):
    return _json_safe(value.tolist())
  if isinstance(value, np.generic):
    return _json_safe(value.item())
  if isinstance(value, float) and not np.isfinite(value):
    return None
  if isinstance(value, Path):
    return str(value)
  return value


def build_tactile_comparison(
  rollout: TactileTrace,
  reference: TactileTrace,
  *,
  contact_threshold_n: float = CONTACT_THRESHOLD_N,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
  """Return JSON-safe metrics and raw/derived arrays without modifying inputs."""
  _compatible(rollout, reference)
  model = derive_tactile_trace(rollout, contact_threshold_n=contact_threshold_n)
  expert = derive_tactile_trace(reference, contact_threshold_n=contact_threshold_n)
  indices, temporal_valid = _matched_indices(rollout, reference.time_s)
  _, max_age = _durations(rollout)
  pair_duration = np.minimum(
    expert["duration_s"],
    np.maximum(
      0,
      np.minimum(rollout.time_s[-1], rollout.time_s[indices] + max_age)
      - reference.time_s,
    ),
  )
  pair_duration[~temporal_valid] = 0.0
  paired_valid = (
    temporal_valid[:, None] & model["normal_valid"][indices] & expert["normal_valid"]
  )
  both_contact = paired_valid & model["contact"][indices] & expert["contact"]
  missed = paired_valid & ~model["contact"][indices] & expert["contact"]
  extra = paired_valid & model["contact"][indices] & ~expert["contact"]
  center_distance = np.linalg.norm(
    model["center_local_mm"][indices] - expert["center_local_mm"], axis=-1
  )
  spatial_difference = 0.5 * np.sum(
    np.abs(
      model["normalized_normal_distribution"][indices]
      - expert["normalized_normal_distribution"]
    ),
    axis=(-2, -1),
  )
  center_distance[~both_contact] = np.nan
  spatial_difference[~both_contact] = np.nan
  force_difference = model["normal_force_n"][indices] - expert["normal_force_n"]
  force_difference[~paired_valid] = np.nan
  fingers = {}
  for finger, name in enumerate(FINGERS):
    weights = pair_duration * paired_valid[:, finger]
    expert_contact_duration = float(np.sum(weights * expert["contact"][:, finger]))
    expert_no_contact_duration = float(np.sum(weights * ~expert["contact"][:, finger]))
    known_duration = float(np.sum(weights))
    expert_valid_duration = float(
      np.sum(expert["duration_s"] * expert["normal_valid"][:, finger])
    )
    missed_duration = float(np.sum(pair_duration * missed[:, finger]))
    extra_duration = float(np.sum(pair_duration * extra[:, finger]))
    fingers[name] = {
      "label_zh": FINGER_LABELS[name],
      "rollout": _native_stats(rollout, model, finger),
      "reference": _native_stats(reference, expert, finger),
      "comparison": {
        "paired_valid_samples": int(np.count_nonzero(paired_valid[:, finger])),
        "paired_valid_duration_s": known_duration,
        "coverage_of_reference_valid_time": known_duration / expert_valid_duration
        if expert_valid_duration > 0
        else None,
        "both_contact_duration_s": float(
          np.sum(pair_duration * both_contact[:, finger])
        ),
        "reference_contact_duration_in_overlap_s": expert_contact_duration,
        "reference_no_contact_duration_in_overlap_s": expert_no_contact_duration,
        "missed_contact_duration_s": missed_duration,
        "missed_contact_fraction_of_reference_contact_in_overlap": missed_duration
        / expert_contact_duration
        if expert_contact_duration > 0
        else None,
        "extra_contact_duration_s": extra_duration,
        "extra_contact_fraction_of_reference_no_contact_in_overlap": extra_duration
        / expert_no_contact_duration
        if expert_no_contact_duration > 0
        else None,
        "normal_force_absolute_error_n": _weighted(
          np.abs(force_difference[:, finger]), weights
        ),
        "normal_force_signed_error_n": _weighted(force_difference[:, finger], weights),
        "center_distance_mm": _weighted(center_distance[:, finger], pair_duration),
        "normalized_spatial_difference": _weighted(
          spatial_difference[:, finger], pair_duration
        ),
      },
    }
  summary = {
    "schema_version": SCHEMA_VERSION,
    "language": "zh-CN",
    "contact_threshold_n": float(contact_threshold_n),
    "reference": {
      "start_time_s": float(reference.time_s[0]),
      "last_observation_time_s": float(reference.time_s[-1]),
      "metadata": reference.metadata,
    },
    "rollout": {
      "start_time_s": float(rollout.time_s[0]),
      "last_observation_time_s": float(rollout.time_s[-1]),
      "metadata": rollout.metadata,
    },
    "definitions": {
      "reference_clock": "固定使用 task_demo 专家原始仿真秒数；模型按自身实际仿真秒数比较，不做轨迹时间拉伸或阶段对齐。",
      "normal_force_n": "每指 7×5 网格法向力之和；超过接触阈值才定义受力中心与归一化分布。",
      "center_local_mm": "按法向力加权的受力中心，投影至同一指尖局部稳定切向基；轴 0 对应网格列，轴 1 对应网格行，单位毫米。",
      "normalized_spatial_difference": "双方均接触时，归一化法向力空间分布的总变差距离，即 0.5×各格绝对差之和；0 表示相同，1 表示完全不重叠。",
      "tangent_net_force_n": "先求和网格内有符号二维切向力，再取向量模长；这是净切向力，不是不会相互抵消的切向载荷。",
      "matching": "在专家时间戳上使用模型最近的左侧样本；仅在模型实际观测范围内且样本年龄不超过 1.5 倍标称采样间隔时匹配。",
      "duration_weighting": "原生时间间隔左保持积分；末个样本权重为零，过大采样空隙不计为完整观测区间，配对区间在模型末次观测或样本有效年龄处截断。",
      "contact_mismatch": "漏接触：专家接触但模型无接触；额外接触：专家无接触但模型接触。比例仅在双方可观测的重叠时间内计算，未录到的末段不算无接触。",
      "invalid": "单指网格含非有限或负法向力时该指法向数据未知；无接触与未知不同，中心/分布均不以原点或零补齐。",
      "scope": "这是单条专家轨迹下的抓持触觉对比，不是任务成功评分，也不是柔性皮肤压力真值；同一时刻差异包含动作进度不同的影响。",
    },
    "fingers": fingers,
  }
  archive = {
    "reference_time_s": reference.time_s,
    "rollout_time_s": rollout.time_s,
    "matched_rollout_indices": np.where(temporal_valid, indices, -1),
    "matched_time_valid": temporal_valid,
    "paired_duration_s": pair_duration,
    "paired_normal_valid": paired_valid,
    "both_contact": both_contact,
    "missed_contact": missed,
    "extra_contact": extra,
    "center_distance_mm": center_distance,
    "normalized_spatial_difference": spatial_difference,
    "normal_force_signed_error_n": force_difference,
  }
  for prefix, trace, derived in (
    ("rollout", rollout, model),
    ("reference", reference, expert),
  ):
    archive[f"{prefix}_normal_taxel_force_n"] = trace.normal_taxel_force_n
    archive[f"{prefix}_tangent_taxel_force_n"] = trace.tangent_taxel_force_n
    archive[f"{prefix}_taxel_positions_local_m"] = trace.taxel_positions_local_m
    archive[f"{prefix}_tangent_basis_local"] = trace.tangent_basis_local
    archive.update({f"{prefix}_{name}": values for name, values in derived.items()})
  return _json_safe(summary), archive


def _model_end(trace: TactileTrace) -> float:
  value = trace.metadata.get("trial_sim_seconds")
  if value is not None and np.isfinite(value) and value >= 0:
    source_start = trace.metadata.get("source_start_time_s")
    if source_start is not None and np.isfinite(source_start):
      return float(value) - float(source_start) + float(trace.time_s[0])
    return float(value)
  return float(trace.time_s[-1])


def _gap_curve(
  trace: TactileTrace, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  _, max_age = _durations(trace)
  gaps = np.flatnonzero(np.diff(trace.time_s) > max_age + 1e-9)
  times = trace.time_s.copy()
  result = values.copy()
  for index in reversed(gaps):
    times = np.insert(times, index + 1, (times[index] + times[index + 1]) / 2)
    result = np.insert(result, index + 1, np.nan)
  return times, result


def _plot_timeline(
  path: Path,
  rollout: TactileTrace,
  reference: TactileTrace,
  summary: dict,
  archive: dict,
) -> None:
  import matplotlib.pyplot as plt

  figure, axes = plt.subplots(5, 3, figsize=(16.8, 14), sharex=True)
  for finger, name in enumerate(FINGERS):
    force_axis, center_axis, spatial_axis = axes[finger]
    for prefix, trace, color, label in (
      ("reference", reference, "#2563a6", "专家"),
      ("rollout", rollout, "#e87924", "模型"),
    ):
      for channel, linestyle, suffix, alpha in (
        ("normal_force_n", "-", "法向", 1),
        ("tangent_net_force_n", "--", "净切向", 0.55),
      ):
        times, values = _gap_curve(trace, archive[f"{prefix}_{channel}"][:, finger])
        force_axis.plot(
          times,
          values,
          color=color,
          lw=1,
          ls=linestyle,
          alpha=alpha,
          label=label + suffix,
        )
    center_axis.plot(
      reference.time_s, archive["center_distance_mm"][:, finger], color="#6d28d9", lw=1
    )
    spatial_axis.plot(
      reference.time_s,
      archive["normalized_spatial_difference"][:, finger],
      color="#00856a",
      lw=1,
    )
    spatial_axis.set_ylim(-0.04, 1.16)
    metrics = summary["fingers"][name]["comparison"]
    coverage = metrics["coverage_of_reference_valid_time"]
    text = f"配对覆盖 {coverage:.1%}" if coverage is not None else "配对覆盖未知"
    text += f"；漏接触 {metrics['missed_contact_duration_s']:.2f} s；额外接触 {metrics['extra_contact_duration_s']:.2f} s"
    spatial_axis.text(
      0.02,
      0.84,
      text,
      transform=spatial_axis.transAxes,
      va="top",
      fontsize=8,
      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )
    # Separate contact-state markers remain visible even where center/maps are undefined.
    for mask, color, label, y in (
      ("missed_contact", "#b42318", "漏接触", 1.06),
      ("extra_contact", "#cf8500", "额外接触", 1.1),
    ):
      selected = archive[mask][:, finger]
      spatial_axis.scatter(
        reference.time_s[selected],
        np.full(np.count_nonzero(selected), y),
        marker="|",
        s=9,
        color=color,
        label=label,
      )
    force_axis.set_ylabel(f"{FINGER_LABELS[name]}\n力 / N")
    center_axis.set_ylabel("中心距离 / mm")
    spatial_axis.set_ylabel("分布差异 / 0–1")
    for axis in axes[finger]:
      axis.axvline(_model_end(rollout), color="#b42318", ls=":", lw=1.1)
      axis.set_xlim(min(0, reference.time_s[0]), reference.time_s[-1])
      axis.grid(alpha=0.2)
  for axis, title in zip(
    axes[0],
    (
      "五指法向力与净切向力",
      "双方均接触时：局部受力中心偏移",
      "双方均接触时：归一化空间分布差异",
    ),
    strict=True,
  ):
    axis.set_title(title, fontsize=11)
  axes[0, 0].legend(fontsize=8, ncol=2, loc="upper right")
  axes[0, 2].legend(fontsize=8, loc="lower right")
  for axis in axes[-1]:
    axis.set_xlabel("专家原始仿真时间 / s（模型使用自身同一秒数）")
  figure.suptitle(
    f"擦白板指尖触觉接触对比｜红线：模型结束 {_model_end(rollout):.2f} s",
    fontsize=17,
    y=0.989,
  )
  figure.text(
    0.03,
    0.014,
    "模型结束后的区域留空，不补零、不拉伸时间；中心与分布仅在双方接触时比较，接触缺失另列。\n"
    f"原生采样：专家 {_sampling_label(reference)}；模型 {_sampling_label(rollout)}。按实际时间加权；同一秒数的差异也可能来自动作进度不同。\n"
    + _coordinate_note(rollout),
    fontsize=9,
    color="#475569",
  )
  figure.tight_layout(rect=(0, 0.065, 1, 0.965), h_pad=1.1)
  figure.savefig(path, dpi=150)
  plt.close(figure)


def _sampling_label(trace: TactileTrace) -> str:
  cadence = _cadence(trace)
  return f"{1 / cadence:.1f} Hz" if cadence > 0 else "频率未知"


def _coordinate_note(trace: TactileTrace) -> str:
  if trace.metadata.get("coordinate_verification"):
    return "历史模型局部坐标：冻结网格与生成代码 SHA 校验通过后，采用与专家一致的坐标布局；证明见统计文件。"
  source = trace.metadata.get("coordinate_source")
  return (
    f"局部坐标来源：{source}"
    if source
    else "模型与专家必须使用一致的指尖网格位置和局部切向基；坐标依据见统计文件。"
  )


def _plot_centers(
  path: Path, rollout: TactileTrace, reference: TactileTrace, archive: dict
) -> None:
  import matplotlib.pyplot as plt
  from matplotlib.colors import Normalize

  figure, axes = plt.subplots(2, 5, figsize=(17, 8.5))
  norm = Normalize(vmin=float(reference.time_s[0]), vmax=float(reference.time_s[-1]))
  scatter = None
  for finger, name in enumerate(FINGERS):
    positions = archive["reference_taxel_xy_mm"][finger]
    bounds = np.stack((positions.min(axis=0), positions.max(axis=0)))
    margin = np.maximum(np.diff(bounds, axis=0)[0] * 0.1, 0.2)
    for row, prefix, trace, label in (
      (0, "reference", reference, "专家"),
      (1, "rollout", rollout, "模型"),
    ):
      axis = axes[row, finger]
      xy = archive[f"{prefix}_center_local_mm"][:, finger]
      known = np.all(np.isfinite(xy), axis=-1)
      axis.scatter(positions[:, 0], positions[:, 1], s=7, color="#cbd5e1", alpha=0.75)
      scatter = axis.scatter(
        xy[known, 0],
        xy[known, 1],
        c=trace.time_s[known],
        cmap="viridis",
        norm=norm,
        s=5,
        alpha=0.7,
      )
      axis.set_xlim(bounds[0, 0] - margin[0], bounds[1, 0] + margin[0])
      axis.set_ylim(bounds[0, 1] - margin[1], bounds[1, 1] + margin[1])
      axis.set_aspect("equal", adjustable="box")
      axis.set_title(f"{label} · {FINGER_LABELS[name]}")
      axis.set_xlabel("指尖局部列方向 / mm")
      axis.set_ylabel("指尖局部行方向 / mm")
      axis.grid(alpha=0.2)
      if not np.any(known):
        axis.text(
          0.5,
          0.5,
          "无有效接触中心",
          transform=axis.transAxes,
          ha="center",
          color="#b42318",
        )
  figure.subplots_adjust(
    left=0.065, right=0.93, bottom=0.13, top=0.9, wspace=0.36, hspace=0.28
  )
  color_axis = figure.add_axes((0.95, 0.2, 0.013, 0.62))
  figure.colorbar(
    scatter, cax=color_axis, label="原始仿真时间 / s（专家与模型共用色标）"
  )
  figure.suptitle("五指局部受力中心轨迹｜法向力加权中心，不是接触几何中心", fontsize=17)
  figure.text(
    0.04,
    0.03,
    "灰点为同一指尖的 7×5 触觉单元位置；上下两行共用各指坐标范围与专家全程时间色标。\n只展示真实接触样本；模型结束后的轨迹不延长。空间分布形状需结合下一张网格图判断。",
    fontsize=10,
    color="#475569",
  )
  figure.savefig(path, dpi=150)
  plt.close(figure)


def _snapshots(reference: TactileTrace, archive: dict) -> list[int]:
  provided = reference.metadata.get("snapshot_times_s", [])
  if len(provided):
    values = np.asarray(provided, dtype=float)
    if (
      values.ndim != 1
      or not np.all(np.isfinite(values))
      or np.any(values < reference.time_s[0])
      or np.any(values > reference.time_s[-1])
    ):
      raise ValueError("热图关键时刻必须有限且位于专家实际观测时间范围内")
    indices = [int(np.searchsorted(reference.time_s, value)) for value in values]
    return list(dict.fromkeys(indices))[:3]
  eligible = np.flatnonzero(np.any(archive["both_contact"], axis=1))
  if not len(eligible):
    eligible = np.flatnonzero(archive["matched_time_valid"])
  if not len(eligible):
    eligible = np.arange(len(reference.time_s))
  return list(
    dict.fromkeys(
      eligible[
        np.linspace(0, len(eligible) - 1, min(3, len(eligible))).astype(int)
      ].tolist()
    )
  )


def _plot_heatmaps(
  path: Path,
  rollout: TactileTrace,
  reference: TactileTrace,
  archive: dict,
  *,
  contact_threshold_n: float,
) -> list[dict]:
  import matplotlib.pyplot as plt

  snapshots = _snapshots(reference, archive)
  cols = 3 * len(snapshots)
  figure, axes = plt.subplots(
    5, cols, figsize=(5.25 * len(snapshots), 11.5), squeeze=False
  )
  grids, records = [], []
  for index in snapshots:
    model_index = int(archive["matched_rollout_indices"][index])
    expert_grid = reference.normal_taxel_force_n[index].copy()
    model_grid = (
      rollout.normal_taxel_force_n[model_index].copy()
      if model_index >= 0
      else np.full((5, 7, 5), np.nan)
    )
    expert_grid[~archive["reference_normal_valid"][index]] = np.nan
    if model_index >= 0:
      model_grid[~archive["rollout_normal_valid"][model_index]] = np.nan
    grids.append((expert_grid, model_grid))
    records.append(
      {
        "reference_time_s": float(reference.time_s[index]),
        "rollout_time_s": float(rollout.time_s[model_index])
        if model_index >= 0
        else None,
      }
    )
  finite_forces = np.concatenate(
    [grid[np.isfinite(grid)] for pair in grids for grid in pair]
  )
  force_max = max(float(np.max(finite_forces)) if len(finite_forces) else 0, 1e-6)
  finite_difference = np.concatenate([(b - a)[np.isfinite(b - a)] for a, b in grids])
  difference_max = max(
    float(np.max(np.abs(finite_difference))) if len(finite_difference) else 0, 1e-6
  )
  force_image = difference_image = None
  for group, (pair, record) in enumerate(zip(grids, records, strict=True)):
    for finger, name in enumerate(FINGERS):
      expert_grid, model_grid = pair[0][finger], pair[1][finger]
      for offset, grid, title in (
        (0, expert_grid, "专家"),
        (1, model_grid, "模型"),
        (2, model_grid - expert_grid, "模型 − 专家"),
      ):
        axis = axes[finger, group * 3 + offset]
        if offset == 2:
          difference_image = axis.imshow(
            grid,
            origin="lower",
            cmap="RdBu_r",
            vmin=-difference_max,
            vmax=difference_max,
          )
        else:
          force_image = axis.imshow(
            grid, origin="lower", cmap="magma", vmin=0, vmax=force_max
          )
          if np.all(np.isfinite(grid)) and float(np.sum(grid)) > contact_threshold_n:
            row_grid, column_grid = np.indices(grid.shape)
            total = float(np.sum(grid))
            column_center = float(np.sum(grid * column_grid) / total)
            row_center = float(np.sum(grid * row_grid) / total)
            # The index-chart center is a visual annotation, not a substitute
            # for the physical millimetre center stored in the comparison.
            axis.scatter(
              column_center,
              row_center,
              marker="o",
              s=42,
              facecolors="none",
              edgecolors="black",
              linewidths=2,
            )
            axis.scatter(
              column_center,
              row_center,
              marker="o",
              s=42,
              facecolors="none",
              edgecolors="white",
              linewidths=1,
            )
        if not np.any(np.isfinite(grid)):
          axis.text(
            0.5, 0.5, "未知 / 未观测", transform=axis.transAxes, ha="center", fontsize=9
          )
        elif offset != 2 and float(np.sum(grid)) <= contact_threshold_n:
          axis.text(
            0.5,
            0.5,
            f"{title}无接触",
            transform=axis.transAxes,
            ha="center",
            fontsize=8,
            color="white",
            bbox={"facecolor": "#475569", "edgecolor": "none", "alpha": 0.85},
          )
        axis.set_xticks((0, 2, 4))
        axis.set_yticks((0, 3, 6))
        axis.tick_params(labelsize=7)
        if finger == 0:
          model_time = record["rollout_time_s"]
          times = f"专家 {record['reference_time_s']:.2f} s"
          times += (
            f"\n模型 {model_time:.2f} s" if model_time is not None else "\n模型未观测"
          )
          axis.set_title(f"{title}\n{times}", fontsize=9)
        if offset == 0:
          axis.set_ylabel(f"{FINGER_LABELS[name]}\n网格行", fontsize=9)
        if finger == 4:
          axis.set_xlabel("网格列", fontsize=8)
  figure.subplots_adjust(
    left=0.06, right=0.88, bottom=0.115, top=0.87, wspace=0.27, hspace=0.16
  )
  figure.colorbar(
    force_image,
    cax=figure.add_axes((0.9, 0.53, 0.015, 0.3)),
    label="法向力 / N（每触觉单元，统一色标）",
  )
  figure.colorbar(
    difference_image,
    cax=figure.add_axes((0.9, 0.18, 0.015, 0.3)),
    label="法向力差 / N（每触觉单元）",
  )
  figure.suptitle("专家关键时刻的指尖法向力空间分布｜实际力值与差值", fontsize=17)
  figure.text(
    0.04,
    0.025,
    "白圈：法向力加权的网格中心（行列坐标）；力值色标统一。网格为核分配估计，不是柔性皮肤压力真值。\n零表示已观测的零力，未知不补零；图中列出双方实际取样时间，不插值、不拉伸。",
    fontsize=10,
    color="#475569",
  )
  figure.savefig(path, dpi=150)
  plt.close(figure)
  return records


def export_tactile_comparison(
  rollout: TactileTrace,
  reference: TactileTrace,
  output_dir: Path,
  *,
  contact_threshold_n: float = CONTACT_THRESHOLD_N,
) -> dict[str, Any]:
  """Create an additive Chinese report, refusing to overwrite any artifact."""
  output_dir = Path(output_dir)
  paths = {key: output_dir / name for key, name in ARTIFACT_NAMES.items()}
  for path in paths.values():
    if path.exists():
      raise FileExistsError(f"触觉对比文件已存在：{path}")
  summary, archive = build_tactile_comparison(
    rollout, reference, contact_threshold_n=contact_threshold_n
  )
  _snapshots(reference, archive)
  output_dir.mkdir(parents=True, exist_ok=True)
  with _chinese_plot_style():
    _plot_timeline(paths["timeline_plot"], rollout, reference, summary, archive)
    _plot_centers(paths["centers_plot"], rollout, reference, archive)
    summary["heatmap_snapshots"] = _plot_heatmaps(
      paths["heatmaps_plot"],
      rollout,
      reference,
      archive,
      contact_threshold_n=contact_threshold_n,
    )
  summary["artifacts"] = {key: str(path) for key, path in paths.items()}
  with paths["trace"].open("xb") as stream:
    np.savez_compressed(stream, **archive)
  with paths["metrics"].open("x", encoding="utf-8") as stream:
    json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write("\n")
  return {"artifacts": summary["artifacts"], "summary": summary}
