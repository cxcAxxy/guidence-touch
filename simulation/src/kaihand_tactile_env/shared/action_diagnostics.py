"""Per-joint and per-task-stage diagnostics for bounded policy actions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


class ActionClipDiagnostics:
  """Accumulate where absolute joint actions exceed simulator limits."""

  def __init__(
    self,
    joint_names: Iterable[str],
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    tolerance: float = 1.0e-9,
  ):
    self.joint_names = tuple(joint_names)
    self.lower = np.asarray(lower, dtype=np.float64)
    self.upper = np.asarray(upper, dtype=np.float64)
    self.tolerance = float(tolerance)
    shape = (len(self.joint_names),)
    if (
      not self.joint_names
      or len(set(self.joint_names)) != len(self.joint_names)
      or self.lower.shape != shape
      or self.upper.shape != shape
      or not np.isfinite(self.lower).all()
      or not np.isfinite(self.upper).all()
      or np.any(self.lower >= self.upper)
      or not np.isfinite(self.tolerance)
      or self.tolerance < 0
    ):
      raise ValueError("invalid joint names, limits, or clipping tolerance")
    self._steps = 0
    self._lower_counts = np.zeros(shape, dtype=np.int64)
    self._upper_counts = np.zeros(shape, dtype=np.int64)
    self._max_excess = np.zeros(shape, dtype=np.float64)
    self._stage_steps: dict[str, int] = defaultdict(int)
    self._stage_lower: dict[str, np.ndarray] = {}
    self._stage_upper: dict[str, np.ndarray] = {}

  def record(self, action: np.ndarray, stage: str) -> dict[str, Any]:
    """Record one action and return its JSON-serializable clipping event."""
    raw = np.asarray(action, dtype=np.float64)
    if raw.shape != self.lower.shape or not np.isfinite(raw).all():
      raise ValueError(
        f"action must be finite with shape {self.lower.shape}, got {raw.shape}"
      )
    stage = str(stage)
    applied = np.clip(raw, self.lower, self.upper)
    lower_mask = raw < self.lower - self.tolerance
    upper_mask = raw > self.upper + self.tolerance
    mask = lower_mask | upper_mask
    excess = np.maximum(self.lower - raw, raw - self.upper)
    excess = np.maximum(excess, 0.0)

    self._steps += 1
    self._lower_counts += lower_mask
    self._upper_counts += upper_mask
    self._max_excess = np.maximum(self._max_excess, excess)
    self._stage_steps[stage] += 1
    if stage not in self._stage_lower:
      self._stage_lower[stage] = np.zeros_like(self._lower_counts)
      self._stage_upper[stage] = np.zeros_like(self._upper_counts)
    self._stage_lower[stage] += lower_mask
    self._stage_upper[stage] += upper_mask

    return {
      "raw_action": raw.tolist(),
      "applied_action": applied.tolist(),
      "clip_mask": mask.tolist(),
      "clipped_values": int(mask.sum()),
      "clipped_joint_names": [
        name for name, clipped in zip(self.joint_names, mask, strict=True) if clipped
      ],
      "maximum_abs_excess_rad": float(excess.max(initial=0.0)),
    }

  def summary(self) -> dict[str, Any]:
    total_values = self._steps * len(self.joint_names)
    counts = self._lower_counts + self._upper_counts

    by_joint = {}
    for index, name in enumerate(self.joint_names):
      clipped = int(counts[index])
      by_joint[name] = {
        "action_steps": self._steps,
        "clipped_values": clipped,
        "clipped_fraction": clipped / self._steps if self._steps else 0.0,
        "lower_clips": int(self._lower_counts[index]),
        "upper_clips": int(self._upper_counts[index]),
        "maximum_abs_excess_rad": float(self._max_excess[index]),
      }

    by_stage = {}
    for stage in sorted(self._stage_steps):
      steps = self._stage_steps[stage]
      lower = self._stage_lower[stage]
      upper = self._stage_upper[stage]
      stage_counts = lower + upper
      clipped = int(stage_counts.sum())
      by_stage[stage] = {
        "action_steps": steps,
        "total_values": steps * len(self.joint_names),
        "clipped_values": clipped,
        "clipped_fraction": clipped / (steps * len(self.joint_names)),
        "clipped_values_by_joint": {
          name: int(stage_counts[index])
          for index, name in enumerate(self.joint_names)
          if stage_counts[index]
        },
        "lower_clips_by_joint": {
          name: int(lower[index])
          for index, name in enumerate(self.joint_names)
          if lower[index]
        },
        "upper_clips_by_joint": {
          name: int(upper[index])
          for index, name in enumerate(self.joint_names)
          if upper[index]
        },
      }

    clipped_values = int(counts.sum())
    return {
      "joint_names": list(self.joint_names),
      "joint_lower_rad": self.lower.tolist(),
      "joint_upper_rad": self.upper.tolist(),
      "action_steps": self._steps,
      "total_values": total_values,
      "clipped_values": clipped_values,
      "clipped_fraction": clipped_values / total_values if total_values else 0.0,
      "by_joint": by_joint,
      "by_stage": by_stage,
    }


def merge_action_clip_summaries(summaries: Iterable[dict[str, Any]]) -> dict[str, Any]:
  """Merge summaries produced for trials with the same joint contract."""
  summaries = list(summaries)
  if not summaries:
    return {
      "trial_count": 0,
      "action_steps": 0,
      "total_values": 0,
      "clipped_values": 0,
      "clipped_fraction": 0.0,
      "by_joint": {},
      "by_stage": {},
    }
  names = summaries[0]["joint_names"]
  if any(summary.get("joint_names") != names for summary in summaries):
    raise ValueError("cannot merge action clipping summaries with different joints")

  joint = {
    name: {
      "action_steps": 0,
      "clipped_values": 0,
      "lower_clips": 0,
      "upper_clips": 0,
      "maximum_abs_excess_rad": 0.0,
    }
    for name in names
  }
  stage: dict[str, dict[str, Any]] = {}
  action_steps = total_values = clipped_values = 0
  for summary in summaries:
    action_steps += int(summary["action_steps"])
    total_values += int(summary["total_values"])
    clipped_values += int(summary["clipped_values"])
    for name in names:
      source = summary["by_joint"][name]
      target = joint[name]
      for key in ("action_steps", "clipped_values", "lower_clips", "upper_clips"):
        target[key] += int(source[key])
      target["maximum_abs_excess_rad"] = max(
        target["maximum_abs_excess_rad"],
        float(source["maximum_abs_excess_rad"]),
      )
    for name, source in summary["by_stage"].items():
      target = stage.setdefault(
        name,
        {
          "action_steps": 0,
          "total_values": 0,
          "clipped_values": 0,
          "clipped_values_by_joint": defaultdict(int),
          "lower_clips_by_joint": defaultdict(int),
          "upper_clips_by_joint": defaultdict(int),
        },
      )
      for key in ("action_steps", "total_values", "clipped_values"):
        target[key] += int(source[key])
      for key in (
        "clipped_values_by_joint",
        "lower_clips_by_joint",
        "upper_clips_by_joint",
      ):
        for joint_name, count in source[key].items():
          target[key][joint_name] += int(count)

  for values in joint.values():
    values["clipped_fraction"] = (
      values["clipped_values"] / values["action_steps"]
      if values["action_steps"]
      else 0.0
    )
  for values in stage.values():
    values["clipped_fraction"] = (
      values["clipped_values"] / values["total_values"]
      if values["total_values"]
      else 0.0
    )
    for key in (
      "clipped_values_by_joint",
      "lower_clips_by_joint",
      "upper_clips_by_joint",
    ):
      values[key] = dict(sorted(values[key].items()))
  return {
    "trial_count": len(summaries),
    "joint_names": names,
    "action_steps": action_steps,
    "total_values": total_values,
    "clipped_values": clipped_values,
    "clipped_fraction": clipped_values / total_values if total_values else 0.0,
    "by_joint": joint,
    "by_stage": dict(sorted(stage.items())),
  }
