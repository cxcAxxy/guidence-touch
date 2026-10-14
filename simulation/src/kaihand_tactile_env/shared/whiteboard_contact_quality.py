"""Observed cleaning, safety and contact-continuity diagnostics for whiteboard.

All durations use the trace's physical interval weights. Unknown measurements
stay unknown; these descriptors are not an aggregate task-quality score.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .whiteboard_contact_intensity import (
  CONTACT_THRESHOLD_N,
  FINGERS,
  PHASES,
  SAFETY_THRESHOLDS_N,
)

PROGRESS_EPSILON = 1e-9


def _number(value: Any, *, maximum: float | None = None) -> float | None:
  try:
    number = float(value)
  except (TypeError, ValueError):
    return None
  if not np.isfinite(number) or number < 0:
    return None
  return number if maximum is None or number <= maximum else None


def _array(trace: Any, name: str, *, signal: bool = False) -> np.ndarray:
  source = getattr(trace, "signals", {}) if signal else trace.channels
  result = np.asarray(source.get(name, np.full(len(trace.time_s), np.nan)), dtype=float)
  if result.shape != np.asarray(trace.time_s).shape:
    raise ValueError(f"{name!r} must have the same shape as contact timestamps")
  return result


def _clock(trace: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  times = np.asarray(trace.time_s, dtype=float)
  durations = np.asarray(trace.duration_s, dtype=float)
  valid = np.isfinite(times) & np.isfinite(durations) & (durations > 0)
  return times, durations, valid


def _consecutive(
  times: np.ndarray, durations: np.ndarray, left: int, right: int
) -> bool:
  return right == left + 1 and bool(
    np.isfinite(times[left])
    and np.isfinite(times[right])
    and np.isfinite(durations[left])
    and durations[left] > 0
    and np.isclose(times[left] + durations[left], times[right], rtol=0, atol=1e-8)
  )


def _weighted_stats(values: np.ndarray, durations: np.ndarray) -> dict:
  total = float(np.sum(durations))
  contact = values > CONTACT_THRESHOLD_N
  contact_duration = float(np.sum(durations[contact]))

  def moments(mask: np.ndarray) -> tuple[float | None, float | None, float | None]:
    weights = durations[mask]
    selected = values[mask]
    denominator = float(np.sum(weights))
    if denominator <= 0:
      return None, None, None
    mean = float(np.sum(selected * weights) / denominator)
    std = float(np.sqrt(np.sum((selected - mean) ** 2 * weights) / denominator))
    return mean, std, std / mean if mean > 0 else None

  mean, std, cv = moments(np.ones(len(values), dtype=bool))
  contact_mean, contact_std, contact_cv = moments(contact)
  return {
    "valid_duration_s": total,
    "mean_n": mean,
    "std_n": std,
    "cv": cv,
    "contact_mean_n": contact_mean,
    "contact_std_n": contact_std,
    "contact_cv": contact_cv,
    "contact_duration_s": contact_duration if total > 0 else None,
    "contact_fraction": contact_duration / total if total > 0 else None,
  }


def _effect(trace: Any) -> dict:
  times, durations, time_valid = _clock(trace)
  clean = _array(trace, "cleaned_fraction", signal=True)
  present = "cleaned_fraction" in getattr(trace, "signals", {})
  clean_valid = np.isfinite(clean) & (clean >= 0) & (clean <= 1)
  sampled_final = (
    float(clean[-1])
    if len(clean) and clean_valid[-1] and np.isfinite(times[-1])
    else None
  )
  final = _number(trace.metadata.get("final_cleaned_fraction"), maximum=1)
  final_source = "metadata.final_cleaned_fraction" if final is not None else None
  if final is None and "final_cleaned_fraction" not in trace.metadata:
    final = _number(trace.metadata.get("cleaned_fraction"), maximum=1)
    final_source = "metadata.cleaned_fraction" if final is not None else None
  # An explicitly unknown final state must not be replaced with a stale sample.
  if final is None and not any(
    key in trace.metadata for key in ("final_cleaned_fraction", "cleaned_fraction")
  ):
    final = sampled_final
    final_source = "last_recorded_sample" if final is not None else None
  observed = np.flatnonzero(clean_valid & np.isfinite(times))
  last_observed = int(observed[-1]) if len(observed) else None
  valid_duration = float(np.sum(durations[time_valid & clean_valid]))
  input_duration = float(np.sum(durations[time_valid]))
  board = _array(trace, "board_normal_n")
  board_valid = time_valid & np.isfinite(board) & (board >= 0)
  contact = board_valid & (board > CONTACT_THRESHOLD_N)
  board_duration = float(np.sum(durations[contact])) if np.any(board_valid) else None

  deltas = np.full(len(times), np.nan)
  end_clock = trace.metadata.get("cleaned_fraction_clock") == "interval_end"
  for index in range(len(times)):
    if not time_valid[index] or not clean_valid[index]:
      continue
    if end_clock:
      if index == 0:
        previous = _number(trace.metadata.get("initial_cleaned_fraction"), maximum=1)
        if previous is not None:
          deltas[index] = clean[index] - previous
      elif clean_valid[index - 1] and _consecutive(times, durations, index - 1, index):
        deltas[index] = clean[index] - clean[index - 1]
    elif index + 1 < len(times):
      if clean_valid[index + 1] and _consecutive(times, durations, index, index + 1):
        deltas[index] = clean[index + 1] - clean[index]
  progress_valid = np.isfinite(deltas) & (deltas >= -PROGRESS_EPSILON)
  observed_contact = contact & progress_valid
  progress_duration = float(np.sum(durations[observed_contact]))
  no_progress_duration = float(
    np.sum(durations[observed_contact & (deltas <= PROGRESS_EPSILON)])
  )
  effect_known = present and board_duration is not None
  return {
    "availability": "available"
    if present and np.any(time_valid & clean_valid)
    else "no_valid_samples"
    if present
    else "signal_unavailable",
    "final_cleaned_fraction": final,
    "final_source": final_source,
    "sampled_final_cleaned_fraction": sampled_final,
    "last_observed_cleaned_fraction": float(clean[last_observed])
    if last_observed is not None
    else None,
    "last_observed_time_s": float(times[last_observed])
    if last_observed is not None
    else None,
    "signal_valid_duration_s": valid_duration if present else None,
    "signal_coverage_fraction": valid_duration / input_duration
    if present and input_duration > 0
    else None,
    "board_contact_observed_duration_s": board_duration,
    "contact_progress_observed_duration_s": progress_duration if effect_known else None,
    "contact_progress_coverage_fraction": progress_duration / board_duration
    if effect_known and board_duration > 0
    else None,
    "contact_without_progress_duration_s": no_progress_duration
    if effect_known and progress_duration > 0
    else None,
    "contact_without_progress_fraction": no_progress_duration / progress_duration
    if effect_known and progress_duration > 0
    else None,
    "contact_with_progress_duration_s": progress_duration - no_progress_duration
    if effect_known and progress_duration > 0
    else None,
    "unknown_contact_progress_duration_s": board_duration - progress_duration
    if effect_known
    else None,
    "cleaned_fraction_clock": "interval_end" if end_clock else "sample_time",
    "progress_epsilon": PROGRESS_EPSILON,
  }


def _safety(trace: Any) -> dict:
  times, durations, time_valid = _clock(trace)
  result = {}
  for channel, default_threshold in SAFETY_THRESHOLDS_N.items():
    threshold = _number(trace.metadata.get("safety_thresholds_n", {}).get(channel))
    if threshold is None:
      threshold = default_threshold
    values = _array(trace, channel)
    sample_valid = (
      np.isfinite(times)
      & np.isfinite(durations)
      & (durations >= 0)
      & np.isfinite(values)
      & (values >= 0)
    )
    duration_valid = sample_valid & time_valid
    sampled_peak = float(np.max(values[sample_valid])) if np.any(sample_valid) else None
    reported_peak = _number(trace.metadata.get("reported_peaks_n", {}).get(channel))
    sampled_exceeded = sampled_peak > threshold if sampled_peak is not None else None
    reported_exceeded = reported_peak > threshold if reported_peak is not None else None
    known_exceedance = [
      value for value in (sampled_exceeded, reported_exceeded) if value is not None
    ]
    total_duration = float(np.sum(durations[duration_valid]))
    exceeded_duration = float(np.sum(durations[duration_valid & (values > threshold)]))
    input_duration = float(np.sum(durations[time_valid]))
    result[channel] = {
      "availability": "available"
      if np.any(sample_valid)
      else "no_valid_samples"
      if channel in trace.channels
      else "channel_unavailable",
      "threshold_n": threshold,
      "sampled_peak_n": sampled_peak,
      "reported_peak_n": reported_peak,
      "sampled_exceeded": sampled_exceeded,
      "reported_exceeded": reported_exceeded,
      "exceeded": any(known_exceedance) if known_exceedance else None,
      "valid_duration_s": total_duration if np.any(duration_valid) else None,
      "coverage_fraction": total_duration / input_duration
      if channel in trace.channels and input_duration > 0
      else None,
      "exceeded_duration_s": exceeded_duration if np.any(duration_valid) else None,
      "exceeded_fraction": exceeded_duration / total_duration
      if total_duration > 0
      else None,
    }
  return result


def _run_count(mask: np.ndarray, times: np.ndarray, durations: np.ndarray) -> int:
  indices = np.flatnonzero(mask)
  return sum(
    offset == 0
    or not _consecutive(times, durations, int(indices[offset - 1]), int(index))
    for offset, index in enumerate(indices)
  )


def _covered_duration(
  mask: np.ndarray, times: np.ndarray, durations: np.ndarray, start: float, end: float
) -> float:
  """Measure the union of observed intervals, avoiding overlapping coverage."""
  covered = 0.0
  cursor = start
  for index in np.flatnonzero(mask):
    left = max(start, float(times[index]))
    right = min(end, float(times[index] + durations[index]))
    if right > max(cursor, left):
      covered += right - max(cursor, left)
    cursor = max(cursor, right)
  return covered


def _continuity(trace: Any) -> dict:
  times, durations, time_valid = _clock(trace)
  board = _array(trace, "board_normal_n")
  known = time_valid & np.isfinite(board) & (board >= 0)
  contact = known & (board > CONTACT_THRESHOLD_N)
  contact_duration = float(np.sum(durations[contact]))
  contacts = np.flatnonzero(contact)
  span = np.zeros(len(times), dtype=bool)
  span_duration = None
  internal_breaks = None
  break_duration = None
  unknown_duration = None
  discontinuities = None
  if len(contacts):
    first, last = int(contacts[0]), int(contacts[-1])
    span[first : last + 1] = True
    span_duration = float(times[last] + durations[last] - times[first])
    absent = span & known & ~contact
    internal_breaks = _run_count(absent, times, durations)
    break_duration = float(np.sum(durations[absent]))
    unknown_duration = max(
      0.0,
      span_duration
      - _covered_duration(
        span & known,
        times,
        durations,
        float(times[first]),
        float(times[last] + durations[last]),
      ),
    )
    discontinuities = 0
    for index in range(first, last):
      if not _consecutive(times, durations, index, index + 1):
        discontinuities += 1
  elif np.any(known):
    span_duration = break_duration = unknown_duration = 0.0
    internal_breaks = discontinuities = 0

  fingers = {}
  for finger in FINGERS:
    channel = f"finger_{finger}_normal_n"
    values = _array(trace, channel)
    selected = contact & np.isfinite(values) & (values >= 0)
    stats = _weighted_stats(values[selected], durations[selected])
    stats.update(
      fraction_scope="valid_finger_samples_during_board_contact",
      availability="available"
      if np.any(selected)
      else "channel_unavailable"
      if channel not in trace.channels
      else "board_unavailable"
      if not np.any(known)
      else "no_board_contact"
      if not np.any(contact)
      else "no_valid_samples",
      coverage_fraction=stats["valid_duration_s"] / contact_duration
      if contact_duration > 0
      else None,
    )
    fingers[finger] = stats
  return {
    "availability": "available"
    if np.any(known)
    else "no_valid_duration"
    if "board_normal_n" in trace.channels
    else "channel_unavailable",
    "board_contact_duration_s": contact_duration if np.any(known) else None,
    "board_contact_episode_count": _run_count(contact, times, durations)
    if np.any(known)
    else None,
    "observed_span_s": span_duration,
    "internal_break_count": internal_breaks,
    "internal_break_duration_s": break_duration,
    "unknown_duration_s": unknown_duration,
    "clock_discontinuity_count": discontinuities,
    "finger_board_contact": fingers,
  }


def _phase_variability(trace: Any) -> dict:
  _, durations, time_valid = _clock(trace)
  phases = np.asarray(trace.phases, dtype=str)
  result = {}
  for phase in (*PHASES, "unknown"):
    selection = time_valid & (phases == phase)
    if not np.any(selection):
      continue
    channels = {}
    for name in trace.channels:
      values = _array(trace, name)
      valid = selection & np.isfinite(values) & (values >= 0)
      channels[name] = _weighted_stats(values[valid], durations[valid])
      channels[name]["coverage_fraction"] = channels[name]["valid_duration_s"] / float(
        np.sum(durations[selection])
      )
    result[phase] = channels
  return result


def build_contact_quality(rollout: Any, reference: Any = None) -> dict:
  """Describe observed contact behavior without assigning a total score."""
  traces = {"rollout": rollout, "reference": reference}
  result = {
    "schema_version": "kaihand-whiteboard-contact-quality-v1",
    "contact_threshold_n": CONTACT_THRESHOLD_N,
    "definitions": {
      "effect": "接触但未观察到清洁进展：仅按相邻有效清洁比例判断，包含擦完后继续接触；不能等同于无效擦拭。",
      "progress_fraction": "无进展比例的分母为能够观测清洁增量的板接触时长；缺失或不连续的区间单列未知时长。",
      "final_cleaned_fraction": "优先使用终态记录；明确未知的终态不会被较早的有效采样覆盖。",
      "safety": "原生采样峰值与任务维护的物理峰值分别保留；缺少碰撞曲线时超阈时长未知。",
      "continuity": "接触阈值为0.05 N；接触片段在未知样本或时间缝隙处断开，零时长末帧不算接触片段。",
      "internal_break": "首次至末次观测接触之间的已知断触区间；未知区间另列，不能当作已知断触。",
      "finger_mean": "板接触期间的均值包含已知零值；接触均值只统计该手指大于接触阈值的时长。",
      "finger_contact_fraction": "手指接触比例以板面接触期间该指有效已知样本的时长为分母；缺失情况另看coverage_fraction，未知样本不能当作零接触。",
      "variation": "标准差与CV按时长加权，缓慢受力变化和快速波动都可能贡献，不能直接称为抓握稳定性。",
      "cv": "CV为时长加权标准差除以对应均值；均值为零或缺失时返回空值。",
    },
  }
  for key, builder in (
    ("effect", _effect),
    ("safety", _safety),
    ("continuity", _continuity),
    ("phase_variability", _phase_variability),
  ):
    result[key] = {
      label: builder(trace) if trace is not None else None
      for label, trace in traces.items()
    }
  result["trial"] = {
    label: {
      "success": trace.metadata.get("task_success", trace.metadata.get("success")),
      "stop_reason": trace.metadata.get(
        "stop_reason", trace.metadata.get("termination_reason")
      ),
      "physics_valid": trace.metadata.get("physics_valid"),
      "capture_error": trace.metadata.get("capture_error"),
    }
    if trace is not None
    else None
    for label, trace in traces.items()
  }
  return result
