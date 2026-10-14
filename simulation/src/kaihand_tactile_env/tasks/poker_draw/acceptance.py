"""Versioned task admission, independent of the unchanged force-quality metric."""

from __future__ import annotations

import math

STRICT_FORCE_POLICY = "strict-force-v1"
TASK_COMPLETION_POLICY = "task-completion-v1"
ACCEPTANCE_POLICIES = (STRICT_FORCE_POLICY, TASK_COMPLETION_POLICY)
# Reuse the existing 0.30 s loss-of-control timescale, not a new millisecond
# force-tracking tolerance. Only the opt-in training policy uses this guard.
MINIMUM_LOADED_FINGERS = 3
MAXIMUM_MULTI_FINGER_LOW_LOAD_S = 0.30
# Freeze historical contracts rather than using today's pose config to
# silently reclassify old datasets. Only draw contact changed in tip-pad-v7.
RECORDED_DRAW_ANGLE_LIMITS = {"legacy-flat-v1": 21.0, "tip-pad-v7": 50.0}


def recorded_draw_angle_limit(outcome: dict) -> float:
  version = outcome.get("draw_posture_version", "legacy-flat-v1")
  if not isinstance(version, str) or version not in RECORDED_DRAW_ANGLE_LIMITS:
    raise ValueError("unknown recorded draw posture version")
  expected = RECORDED_DRAW_ANGLE_LIMITS[version]
  declared = outcome.get("slide_fingertip_plane_angle_limit_degrees", 21.0)
  if (isinstance(declared, bool) or not isinstance(declared, (int, float))
      or not math.isfinite(declared) or declared != expected):
    raise ValueError("recorded draw posture/angle limit mismatch")
  return expected


def check_policy(policy: str) -> str:
  if policy not in ACCEPTANCE_POLICIES:
    raise ValueError(f"unknown poker acceptance policy: {policy!r}")
  return policy


def accept_task(completed: bool, strict_force_quality: bool, policy: str) -> bool:
  check_policy(policy)
  return bool(completed and (strict_force_quality or policy == TASK_COMPLETION_POLICY))


def pressure_quality_label(completed: bool, strict_force_quality: bool) -> str:
  if not completed:
    return "incomplete"
  return "stable" if strict_force_quality else "completed_with_force_variation"


def edge_task_completed(edge: dict | None) -> bool:
  return bool(
    edge
    and all(edge.get(name) is True for name in (
      "target_reached", "held_at_edge", "slide_geometry_qualified"
    ))
    and edge.get("terminal_reason") == "edge_reached"
  )


def accept_edge(edge: dict | None, policy: str) -> bool:
  check_policy(policy)
  if policy == STRICT_FORCE_POLICY:
    return bool(edge and edge.get("full_slide_qualified") is True)
  return bool(edge_task_completed(edge) and edge.get("slide_task_completed") is True)


def validate_recorded_acceptance(metadata: dict, outcome: dict) -> None:
  """Fail closed on unknown/mismatched policies; never relabel legacy records."""
  policy = check_policy(metadata.get("acceptance_policy", STRICT_FORCE_POLICY))
  if outcome.get("acceptance_policy", STRICT_FORCE_POLICY) != policy:
    raise ValueError("metadata/outcome acceptance policy mismatch")
  draw_angle_limit = recorded_draw_angle_limit(outcome)
  if policy == STRICT_FORCE_POLICY:
    if outcome.get("slide_press_control_qualified") is not True:
      raise ValueError("strict source did not pass force quality")
    return
  if not all(outcome.get(name) is True for name in (
    "success", "task_completed", "terminal_pinch", "sustained_pinch",
    "retained_at_end", "half_overhang_reached", "simultaneous_four_finger_contact",
  )):
    raise ValueError("training source lacks complete physical task/terminal gates")
  if not accept_edge(outcome.get("edge_outcome"), policy):
    raise ValueError("training source lacks a completed supported edge stage")
  if (outcome.get("handoff_outcome") or {}).get("completed") is not True:
    raise ValueError("training source has no completed handover")
  quality = outcome.get("slide_press_control_qualified")
  if not isinstance(quality, bool) or outcome.get("pressure_quality") != pressure_quality_label(True, quality):
    raise ValueError("pressure quality label is inconsistent")
  for name, bound, lower in (
    ("maximum_slide_fingertip_plane_angle_degrees", draw_angle_limit, False),
    ("inspection_face_alignment", .80, True),
    ("inspection_face_robot_alignment", .80, True),
    ("inspection_position_error", .03, False),
    ("minimum_supported_card_clearance", -.0006, True),
  ):
    value = outcome.get(name)
    if (not isinstance(value, (int, float)) or not math.isfinite(value)
        or (value < bound if lower else value > bound)):
      raise ValueError(f"training source failed physical gate: {name}")
