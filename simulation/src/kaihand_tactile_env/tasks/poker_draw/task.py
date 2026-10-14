"""Deterministic four-finger slide and thumb-pinch card-draw task."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import mujoco
import numpy as np

from kaihand_tactile_env.shared.simulation import (
  ArmHandSimulation,
  IkResult,
  _rotation_vector_world,
  _wxyz_from_matrix,
)
from kaihand_tactile_env.shared.tactile import load_fingertip_layout
from kaihand_tactile_env.shared.trajectory import JointWaypoint

from .acceptance import (
  STRICT_FORCE_POLICY,
  accept_task,
  check_policy,
  pressure_quality_label,
)
from .config import (
  _DRAW_FINGER_DEGREES,
  _FACE_NORMAL_COSINE,
  _FINGER_ABDUCTION_DEGREES,
  _FINGERS,
  _FLAT_DRAW_FINGER_DEGREES,
  _FLAT_DRAW_MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES,
  _FLAT_DRAW_POSTURE_VERSION,
  _FLAT_DRAW_PRECONTACT_OFFSET,
  _FLAT_DRAW_THUMB_DEGREES,
  _FLAT_DRAW_WRIST_PITCH_DEGREES,
  _FLAT_DRAW_WRIST_YAW_DEGREES,
  _FLAT_PINCH_CONTACT_OFFSET,
  _FLAT_PINCH_FINGER_DEGREES,
  _FLAT_PINCH_THUMB_DEGREES,
  _FLAT_PINCH_WRIST_PITCH_DEGREES,
  _FLAT_PINCH_WRIST_YAW_DEGREES,
  _INSPECTION_TARGET_CARD_POSITION,
  _MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES,
  _MINIMUM_FINGER_PAD_ALIGNMENT,
  _MINIMUM_PINCH_NORMAL_FORCE,
  _MINIMUM_THUMB_PAD_ALIGNMENT,
  _OPEN_THUMB_DEGREES,
  _PRESS_FORCE_CONTACT_RATIO,
  _PRESS_FORCE_CONTACT_RECOVERY_RATE_RAD_S,
  _PRESS_FORCE_CONTROL_PERIOD,
  _PRESS_FORCE_DEADBAND_N,
  _PRESS_FORCE_ESTABLISH_TIMEOUT,
  _PRESS_FORCE_ESTABLISH_TOLERANCE_N,
  _PRESS_FORCE_FILTER_TIME_CONSTANT,
  _PRESS_FORCE_INTEGRAL_GAIN_RAD_PER_N_S,
  _PRESS_FORCE_MAXIMUM_CONTACT_GAP,
  _PRESS_FORCE_MAXIMUM_OFFSET_RAD,
  _PRESS_FORCE_MAXIMUM_OFFSET_RATE_RAD_S,
  _PRESS_FORCE_MEAN_TOLERANCE_N,
  _PRESS_FORCE_MINIMUM_ALL_TARGET_BAND_FRACTION,
  _PRESS_FORCE_MINIMUM_CONTACT_FRACTION,
  _PRESS_FORCE_MINIMUM_TARGET_BAND_FRACTION,
  _PRESS_FORCE_RECOVERY_STABLE_DURATION,
  _PRESS_FORCE_RECOVERY_TIMEOUT,
  _PRESS_FORCE_SLIDE_ACCELERATION_DURATION,
  _PRESS_FORCE_SLIDE_SPEED_M_S,
  _PRESS_FORCE_STABLE_DURATION,
  _PRESS_FORCE_TARGET_TOLERANCE_N,
  _SIDE,
  DEFAULT_PRESS_FORCE_PER_FINGER_N,
)
from .press_control import (
  FourFingerForceController,
  PressForceMonitor,
  PressForceSummary,
)

PokerStepObserver = Callable[[ArmHandSimulation, str], None]

# This range is used only while selecting the task-specific approach IK
# branch.  The live MuJoCo joint range is restored before execution.  It keeps
# right joint 5 on the negative branch without placing the commanded approach
# on the real -178 degree mechanical stop.
_APPROACH_JOINT5_IK_RANGE_DEGREES = (-165.0, -100.0)


def _tilt_in_world_xy(
  rotation: np.ndarray,
  x_degrees: float,
  y_degrees: float,
) -> np.ndarray:
  """Apply a small world-frame tilt to a hand orientation."""
  x_angle = np.deg2rad(x_degrees)
  y_angle = np.deg2rad(y_degrees)
  rotate_x = np.array(
    [
      [1.0, 0.0, 0.0],
      [0.0, np.cos(x_angle), -np.sin(x_angle)],
      [0.0, np.sin(x_angle), np.cos(x_angle)],
    ]
  )
  rotate_y = np.array(
    [
      [np.cos(y_angle), 0.0, np.sin(y_angle)],
      [0.0, 1.0, 0.0],
      [-np.sin(y_angle), 0.0, np.cos(y_angle)],
    ]
  )
  return rotate_y @ rotate_x @ rotation


@dataclass(frozen=True)
class PokerDrawPlan:
  """Known-state plan up to the contact-driven slide and pinch stages."""

  side: str
  object_name: str
  initial_card_pose: np.ndarray
  table_edge_x: float
  end_effector_rotation: np.ndarray
  pinch_end_effector_rotation: np.ndarray
  waypoints: tuple[JointWaypoint, ...]


@dataclass(frozen=True)
class PokerSlideResult:
  """Outcome and measured per-finger pressure for the draw-to-edge phase."""

  success: bool
  error: str | None
  object_name: str
  side: str
  initial_card_pose: np.ndarray
  edge_card_pose: np.ndarray
  table_edge_x: float
  maximum_overhang_fraction: float
  toward_robot_displacement: float
  lateral_card_displacement: float
  press_force_fingers: tuple[str, ...]
  press_force_target_per_finger_n: float
  established_press_normal_forces_n: tuple[float, ...]
  slide_finger_normal_force_means_n: tuple[float, ...]
  slide_finger_normal_force_peaks_n: tuple[float, ...]
  slide_finger_contact_fractions: tuple[float, ...]
  slide_finger_target_band_fractions: tuple[float, ...]
  slide_finger_maximum_contact_gaps_s: tuple[float, ...]
  slide_four_finger_contact_fraction: float
  slide_four_finger_target_band_fraction: float
  slide_force_control_recovery_count: int
  slide_press_control_qualified: bool
  maximum_slide_fingertip_plane_angle_degrees: float
  maximum_slide_end_effector_z_error_m: float
  minimum_supported_card_clearance: float
  minimum_card_back_clearance: float
  phases: tuple[str, ...]
  draw_posture_version: str = "legacy-flat-v1"
  slide_fingertip_plane_angle_limit_degrees: float = 21.0


@dataclass(frozen=True)
class PokerDrawResult:
  """Physical outcome and event checks for one card-draw episode."""

  success: bool
  object_name: str
  side: str
  initial_card_pose: np.ndarray
  edge_card_pose: np.ndarray
  preinspection_card_pose: np.ndarray
  final_card_pose: np.ndarray
  inspection_target_card_position: np.ndarray
  table_edge_x: float
  maximum_overhang_fraction: float
  maximum_card_height: float
  maximum_card_tilt_degrees: float
  maximum_preinspection_card_tilt_degrees: float
  draw_fingers_contacted: tuple[str, ...]
  simultaneous_four_finger_contact: bool
  press_force_fingers: tuple[str, ...]
  press_force_target_per_finger_n: float
  established_press_normal_forces_n: tuple[float, ...]
  slide_finger_normal_force_means_n: tuple[float, ...]
  slide_finger_normal_force_peaks_n: tuple[float, ...]
  slide_finger_contact_fractions: tuple[float, ...]
  slide_finger_target_band_fractions: tuple[float, ...]
  slide_finger_maximum_contact_gaps_s: tuple[float, ...]
  slide_four_finger_contact_fraction: float
  slide_four_finger_target_band_fraction: float
  slide_force_control_recovery_count: int
  slide_press_control_qualified: bool
  maximum_slide_fingertip_plane_angle_degrees: float
  maximum_slide_end_effector_z_error_m: float
  half_overhang_reached: bool
  thumb_face_contact: bool
  sustained_pinch: bool
  lift_opposition_fraction: float
  lift_four_finger_fraction: float
  maximum_lift_opposition_gap: float
  hold_opposition_fraction: float
  hold_four_finger_fraction: float
  inspection_rotation_degrees: float
  inspection_face_alignment: float
  inspection_face_robot_alignment: float
  inspection_position_error: float
  inspection_opposition_fraction: float
  inspection_four_finger_fraction: float
  maximum_inspection_opposition_gap: float
  inspection_hold_opposition_fraction: float
  inspection_hold_four_finger_fraction: float
  minimum_inspection_card_height: float
  wrist_inward_turn_degrees: float
  terminal_pinch: bool
  terminal_grip_fingers: tuple[str, ...]
  terminal_thumb_normal_force: float
  terminal_finger_normal_force: float
  minimum_terminal_finger_pad_alignment: float
  terminal_thumb_pad_alignment: float
  maximum_terminal_fingertip_angle_to_card_plane_degrees: float
  retained_at_end: bool
  maximum_palm_normal_z: float
  toward_robot_displacement: float
  lateral_card_displacement: float
  minimum_supported_card_clearance: float
  minimum_card_back_clearance: float
  phases: tuple[str, ...]
  acceptance_policy: str = STRICT_FORCE_POLICY
  task_completed: bool = False
  pressure_quality: str = "incomplete"
  draw_posture_version: str = "legacy-flat-v1"
  slide_fingertip_plane_angle_limit_degrees: float = 21.0


class PokerDrawPlanner:
  """Build a palm-down approach with four curved, distal-pad contact fingers."""

  def __init__(self, simulation: ArmHandSimulation) -> None:
    self.sim = simulation

  def plan(
    self, side: str = _SIDE, *, legacy_flat_approach: bool = False
  ) -> PokerDrawPlan:
    if self.sim.scene != "poker-draw":
      raise ValueError("PokerDrawPlanner requires scene='poker-draw'")
    if side != _SIDE:
      raise ValueError("the calibrated poker-draw task currently uses the right hand")

    card_pose = self.sim.object_pose("card")
    table = self.sim.model.body("poker_table")
    tabletop = self.sim.model.geom("poker_table_top")
    # The robot torso is at x=0 and faces the workcell along +X, so the near
    # edge is the table's minimum-X edge.  Sliding along -X moves the card
    # toward the robot rather than sideways toward the right arm.
    table_edge_x = float(table.pos[0] - tabletop.size[0])

    _, home_ee_rotation = self.sim.current_pose_matrix(side)
    hand_id = self.sim.model.body("hand_r_base_link").id
    home_hand_rotation = self.sim.data.xmat[hand_id].reshape(3, 3).copy()
    ee_to_hand = home_ee_rotation.T @ home_hand_rotation
    # The draw uses a downward-facing, pitched palm and curved MCP/PIP joints.
    # Its distal pads contact near their tips. After the card reaches
    # the edge, the clear hand pitches into an anatomical pinch: proximal
    # joints curl but each distal tactile surface lies flat on the card.  This
    # puts the thumb under the index pad (about 9 mm away in-plane) instead of
    # at the diagonally opposite corner, removing the unphysical lift torque.
    palm_down_rotation = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    # The raised mini-table provides clearance for the PINCH pose below;
    # its distal axes are within 20 degrees of the card (unlike the draw).
    pitch = np.deg2rad(_FLAT_PINCH_WRIST_PITCH_DEGREES)
    world_y_pitch = np.array(
      [
        [np.cos(pitch), 0.0, np.sin(pitch)],
        [0.0, 1.0, 0.0],
        [-np.sin(pitch), 0.0, np.cos(pitch)],
      ]
    )
    yaw = np.deg2rad(_FLAT_PINCH_WRIST_YAW_DEGREES)
    world_z_yaw = np.array(
      [
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
      ]
    )
    pinch_hand_rotation = world_z_yaw @ world_y_pitch @ palm_down_rotation
    draw_hand_rotation = _tilt_in_world_xy(
      palm_down_rotation,
      0.0,
      0.0 if legacy_flat_approach else _FLAT_DRAW_WRIST_PITCH_DEGREES,
    )
    if not legacy_flat_approach:
      draw_yaw = np.deg2rad(_FLAT_DRAW_WRIST_YAW_DEGREES)
      draw_hand_rotation = (
        np.array(
          [
            [np.cos(draw_yaw), -np.sin(draw_yaw), 0.0],
            [np.sin(draw_yaw), np.cos(draw_yaw), 0.0],
            [0.0, 0.0, 1.0],
          ]
        )
        @ draw_hand_rotation
      )
    ee_rotation = draw_hand_rotation @ ee_to_hand.T
    pinch_ee_rotation = pinch_hand_rotation @ ee_to_hand.T

    card_position = card_pose[:3]
    # The clearance waypoint is calibrated in the workspace, not relative to
    # the folded idle hand. Evaluate its reference without changing live state.
    reference = mujoco.MjData(self.sim.model)
    reference.qpos[:] = self.sim.data.qpos
    reference.qpos[self.sim._arm_qpos[side]] = np.deg2rad(
      [-55, -65, 70, -60, 120, 0, 0]
    )
    mujoco.mj_kinematics(self.sim.model, reference)
    current_ee_position = reference.site(f"{side}_ee_site").xpos.copy()
    targets = (
      (
        "clear_card",
        np.array([current_ee_position[0], current_ee_position[1], 1.00]),
        3.00,
      ),
      ("ready_card", card_position + np.array([-0.144, 0.004, 0.27825]), 0.45),
      ("hover_card", card_position + np.array([-0.144, 0.004, 0.09825]), 0.55),
      (
        "precontact_card",
        card_position + np.array([-0.144, 0.004, 0.06825]),
        0.25,
      ),
    )
    if not legacy_flat_approach:
      # Preserve the approach clearances measured from the new curled pad
      # shape, rather than lowering a changed hand toward stale wrist targets.
      targets = (
        targets[0],
        *(
          (
            phase,
            card_position
            + _FLAT_DRAW_PRECONTACT_OFFSET
            + np.array([0.011, 0.0, height]),
            duration,
          )
          for phase, height, duration in (
            ("ready_card", 0.070, 0.40),
            ("hover_card", 0.025, 0.40),
            ("precontact_card", 0.010, 0.25),
          )
        ),
      )
      # One clearance above the card, followed only by descent. The old
      # clear/ready pair first lowered the hand then lifted it by 14 cm.
      targets = (
        (
          "clear_card",
          card_position + _FLAT_DRAW_PRECONTACT_OFFSET + np.array([0.011, 0.0, 0.095]),
          2.60,
        ),
        *targets[1:],
      )
    # Select the negative joint-5 IK branch.  The previous +120 degree seed
    # made joint 5 travel from the shared -150 degree home through +123
    # degrees: a visible 273 degree clockwise sweep.  This branch reaches the
    # same Cartesian targets by moving joint 5 farther negative instead, so
    # the opening gesture is a bounded counterclockwise segment.  It remains
    # an IK reference only; reset still starts at shared ARM_HOME.
    seed = np.deg2rad([-55, -65, 70, -60, -150, 0, 0])
    waypoints: list[JointWaypoint] = []
    joint5_id = self.sim.model.joint(f"{side}_arm_joint5").id
    original_joint5_range = self.sim.model.jnt_range[joint5_id].copy()
    self.sim.model.jnt_range[joint5_id] = np.deg2rad(_APPROACH_JOINT5_IK_RANGE_DEGREES)
    try:
      for phase, position, duration in targets:
        result = self.sim.solve_ik(
          side,
          position,
          ee_rotation,
          seed=seed,
          max_iterations=1500,
          position_tolerance=0.0005,
          orientation_tolerance=0.02,
          posture_weight=0.001,
        )
        _require_ik(result, phase)
        waypoints.append(
          JointWaypoint(
            phase=phase,
            joint_positions=result.joint_positions,
            duration=duration,
            end_effector_position=position,
            end_effector_quaternion_wxyz=_wxyz_from_matrix(ee_rotation),
          )
        )
        seed = result.joint_positions
    finally:
      self.sim.model.jnt_range[joint5_id] = original_joint5_range
    return PokerDrawPlan(
      side=side,
      object_name="card",
      initial_card_pose=card_pose,
      table_edge_x=table_edge_x,
      end_effector_rotation=ee_rotation,
      pinch_end_effector_rotation=pinch_ee_rotation,
      waypoints=tuple(waypoints),
    )


class PokerDrawExecutor:
  """Slide, physically pinch, lift, and turn a card toward the robot."""

  # The unmodified scene contacts are stiffer than the compliant collection
  # preset. A 3 N opposed jaw maintains pad contact without geometry locking.
  _pinch_force_targets_n = np.array([0.75, 0.75, 0.75, 0.75, 3.0])
  # The default viewer has much stiffer contacts than the compliant capture
  # preset. Lower integral gain prevents force chatter in a stationary pinch.
  _pinch_force_integral_gain = 0.035
  _pinch_force_maximum_joint_rate_degrees_s = 0.75
  _pinch_force_maximum_offset_degrees = 0.50
  _pinch_upper_preload_degrees = 0.15
  _prelift_settle_seconds = 0.35
  _transfer_lift_duration_seconds = 0.52
  _main_lift_duration_seconds = 1.00

  def __init__(
    self,
    simulation: ArmHandSimulation,
    *,
    observer: PokerStepObserver | None = None,
    press_force_per_finger_n: float | None = None,
    acceptance_policy: str = STRICT_FORCE_POLICY,
  ) -> None:
    self.sim = simulation
    self.observer = observer
    self.acceptance_policy = check_policy(acceptance_policy)
    target_force = (
      DEFAULT_PRESS_FORCE_PER_FINGER_N
      if press_force_per_finger_n is None
      else float(press_force_per_finger_n)
    )
    if not np.isfinite(target_force) or target_force <= 0.0:
      raise ValueError("press_force_per_finger_n must be finite and positive")
    self.press_force_per_finger_n = target_force
    self._tactile_layout = load_fingertip_layout()
    self._draw_contacts: set[str] = set()
    self._simultaneous_four_finger_contact = False
    self._thumb_face_contact = False
    self._maximum_overhang_fraction = 0.0
    self._maximum_card_height = float("-inf")
    self._maximum_card_tilt_degrees = 0.0
    self._maximum_palm_normal_z = float("-inf")
    self._maximum_task_palm_normal_z = float("-inf")
    self._minimum_supported_card_clearance = float("inf")
    self._minimum_card_back_clearance = float("inf")
    self._sustained_pinch = False
    self._lift_contact_frames = 0
    self._lift_opposed_frames = 0
    self._lift_four_finger_frames = 0
    self._lift_current_gap_frames = 0
    self._lift_maximum_gap_frames = 0
    self._hold_contact_frames = 0
    self._hold_opposed_frames = 0
    self._hold_four_finger_frames = 0
    self._inspection_contact_frames = 0
    self._inspection_opposed_frames = 0
    self._inspection_four_finger_frames = 0
    self._inspection_current_gap_frames = 0
    self._inspection_maximum_gap_frames = 0
    self._inspection_hold_contact_frames = 0
    self._inspection_hold_opposed_frames = 0
    self._inspection_hold_four_finger_frames = 0
    self._minimum_inspection_card_height = float("inf")
    self._pinch_flexion_targets: np.ndarray | None = None
    self._pinch_thumb_joint5_target = np.deg2rad(
      _FLAT_PINCH_THUMB_DEGREES["thumb_joint5"]
    )
    self._pinch_force_filtered: np.ndarray | None = None
    self._pinch_force_reference: np.ndarray | None = None
    self._pinch_force_steps = 0
    self._press_controller = FourFingerForceController(
      len(_FINGERS),
      target_force_n=target_force,
      timestep=self.sim.timestep,
      update_period_s=_PRESS_FORCE_CONTROL_PERIOD,
      filter_time_constant_s=_PRESS_FORCE_FILTER_TIME_CONSTANT,
      integral_gain_rad_per_n_s=_PRESS_FORCE_INTEGRAL_GAIN_RAD_PER_N_S,
      maximum_offset_rad=_PRESS_FORCE_MAXIMUM_OFFSET_RAD,
      maximum_offset_rate_rad_s=_PRESS_FORCE_MAXIMUM_OFFSET_RATE_RAD_S,
      contact_force_n=target_force * _PRESS_FORCE_CONTACT_RATIO,
      contact_recovery_rate_rad_s=_PRESS_FORCE_CONTACT_RECOVERY_RATE_RAD_S,
      force_deadband_n=_PRESS_FORCE_DEADBAND_N,
    )
    self._slide_force_monitor = PressForceMonitor(
      len(_FINGERS),
      target_force_n=target_force,
      timestep=self.sim.timestep,
      contact_force_ratio=_PRESS_FORCE_CONTACT_RATIO,
      target_tolerance_n=_PRESS_FORCE_TARGET_TOLERANCE_N,
    )
    self._press_controller_active = False
    self._record_slide_force_metrics = False
    self._press_joint2_base = np.zeros(len(_FINGERS), dtype=float)
    self._press_joint3_base = np.zeros(len(_FINGERS), dtype=float)
    self._press_linkage_directions = np.ones(len(_FINGERS), dtype=float)
    self._slide_force_control_recovery_count = 0
    self._maximum_slide_fingertip_plane_angle_degrees = 0.0
    self._maximum_slide_end_effector_z_error_m = 0.0
    self._press_slide_reference_z = 0.0
    self._press_slide_rotation = np.eye(3)
    self._press_slide_command_time_s = 0.0
    self._established_press_normal_forces_n = tuple(0.0 for _ in _FINGERS)

  def _validate_plan(self, plan: PokerDrawPlan) -> None:
    if self.sim.scene != "poker-draw" or plan.object_name != "card":
      raise ValueError("PokerDrawExecutor requires a poker-draw card plan")
    if plan.side != _SIDE:
      raise ValueError("the calibrated poker-draw task currently uses the right hand")

  def _reset_episode_state(self, plan: PokerDrawPlan) -> None:
    self._draw_contacts.clear()
    self._simultaneous_four_finger_contact = False
    self._thumb_face_contact = False
    self._maximum_overhang_fraction = self._overhang_fraction(plan.table_edge_x)
    self._maximum_card_height = float(self.sim.object_pose("card")[2])
    self._maximum_card_tilt_degrees = self._card_tilt_degrees()
    self._maximum_palm_normal_z = self._palm_normal_z()
    self._maximum_task_palm_normal_z = float("-inf")
    self._minimum_supported_card_clearance = self._card_table_clearance()
    self._minimum_card_back_clearance = self._card_back_top_clearance()
    self._sustained_pinch = False
    self._lift_contact_frames = 0
    self._lift_opposed_frames = 0
    self._lift_four_finger_frames = 0
    self._lift_current_gap_frames = 0
    self._lift_maximum_gap_frames = 0
    self._hold_contact_frames = 0
    self._hold_opposed_frames = 0
    self._hold_four_finger_frames = 0
    self._inspection_contact_frames = 0
    self._inspection_opposed_frames = 0
    self._inspection_four_finger_frames = 0
    self._inspection_current_gap_frames = 0
    self._inspection_maximum_gap_frames = 0
    self._inspection_hold_contact_frames = 0
    self._inspection_hold_opposed_frames = 0
    self._inspection_hold_four_finger_frames = 0
    self._minimum_inspection_card_height = float("inf")
    self._pinch_flexion_targets = None
    self._pinch_thumb_joint5_target = np.deg2rad(
      _FLAT_PINCH_THUMB_DEGREES["thumb_joint5"]
    )
    self._pinch_force_filtered = None
    self._pinch_force_reference = None
    self._pinch_force_steps = 0
    self._press_controller_active = False
    self._record_slide_force_metrics = False
    self._press_controller.reset()
    self._slide_force_monitor.reset()
    self._slide_force_control_recovery_count = 0
    self._maximum_slide_fingertip_plane_angle_degrees = 0.0
    self._maximum_slide_end_effector_z_error_m = 0.0
    self._press_slide_reference_z = 0.0
    self._press_slide_rotation = np.eye(3)
    self._press_slide_command_time_s = 0.0
    self._established_press_normal_forces_n = tuple(0.0 for _ in _FINGERS)

  def execute_slide_only(self, plan: PokerDrawPlan) -> PokerSlideResult:
    """Run the production force-controlled draw through the edge hold only.

    Expected friction failures are returned in ``error`` instead of being
    raised, which makes this entry point suitable for bounded material tests.
    Invalid scene/plan inputs still raise ``ValueError``.
    """
    self._validate_plan(plan)
    self._reset_episode_state(plan)
    phases: list[str] = []
    failure: str | None = None
    try:
      self._prepare_and_slide(plan, phases)
    except RuntimeError as error:
      failure = str(error)
    return self._make_slide_result(plan, phases, failure)

  def execute(self, plan: PokerDrawPlan) -> PokerDrawResult:
    self._validate_plan(plan)

    self._reset_episode_state(plan)
    phases: list[str] = []
    seed, slide_position, edge_pose = self._prepare_and_slide(plan, phases)
    # Fold the thumb below the exposed edge while all five pads are still
    # clear.  Form the terminal hand shape outside the card, then translate it
    # inward in 0.5 mm increments.  This keeps the distal links extended so
    # their tactile faces, rather than their edges, form the upper jaw.
    pinch_position, _ = self.sim.current_pose_matrix(plan.side)
    release_position = pinch_position + np.array([0.0, 0.0, 0.015])
    seed = self._move_pose_linear(
      plan,
      release_position,
      seed,
      duration=0.10,
      phase="thumb_face_press",
    )
    safe_outer_position = edge_pose[:3] + _FLAT_PINCH_CONTACT_OFFSET + [0, 0, 0.050]
    # Form the jaw only after the open hand reaches the collision-free outer
    # pose.  Starting this curl during the short release can let the long
    # fingers sweep the half-supported card at the table edge.
    self._set_flat_pinch_pose(include_thumb=False)
    # Its low-torque velocity servos keep converging while the arm pitches;
    # the gate below prevents descent until every joint is within half a
    # degree, so the transient can never become the card-contact posture.
    seed = self._move_pose_linear(
      plan,
      safe_outer_position,
      seed,
      duration=0.65,
      phase="thumb_face_press",
      end_effector_rotation=plan.pinch_end_effector_rotation,
    )
    self._advance_until_arm_settled(
      0.90,
      "thumb_face_press",
      plan.side,
      plan.table_edge_x,
    )
    finger_names = tuple(
      f"hand_r_{finger}_joint{joint}" for finger in _FINGERS for joint in (1, 2, 3, 4)
    )
    self._advance_until_hand_targets_settled(
      2.50,
      "thumb_face_press",
      finger_names,
      plan.table_edge_x,
    )
    seed = self.sim.hold_current_arm_position(plan.side)
    # The open jaw approaches the exposed edge from above. Its thumb then
    # scoops the printed face upward into the four pads; do not insist on a
    # world-horizontal pinch or push all four pads down before thumb support.
    flat_hover_position = edge_pose[:3] + _FLAT_PINCH_CONTACT_OFFSET + [0, 0, 0.020]
    seed = self._move_pose_linear(
      plan,
      flat_hover_position,
      seed,
      duration=0.35,
      phase="thumb_face_press",
      end_effector_rotation=plan.pinch_end_effector_rotation,
    )
    self._advance_until_arm_settled(
      0.60,
      "thumb_face_press",
      plan.side,
      plan.table_edge_x,
    )
    seed = self.sim.hold_current_arm_position(plan.side)
    near_contact_position = edge_pose[:3] + _FLAT_PINCH_CONTACT_OFFSET + [0, 0, 0.0012]
    seed = self._move_pose_linear(
      plan,
      near_contact_position,
      seed,
      duration=0.30,
      phase="thumb_face_press",
      end_effector_rotation=plan.pinch_end_effector_rotation,
    )
    self._advance_until_arm_settled(
      0.45,
      "thumb_face_press",
      plan.side,
      plan.table_edge_x,
    )
    seed = self.sim.hold_current_arm_position(plan.side)
    seed = self._approach_flat_pinch(plan, seed)
    # Establish force-supported opposition before accelerating the card.
    # Both the legacy and middle-pressure executors use the same bounded
    # tactile jaw loop, instead of a legacy binary-contact flexion reflex.
    for _ in range(
      max(1, round(self._prelift_settle_seconds / self.sim.timestep))
    ):
      self._step("thumb_face_press", plan.table_edge_x)
      self._maintain_dynamic_pinch(*self._current_card_face_contact_state())
    seed = self.sim.arm_goal[plan.side].copy()
    pinch_position, _ = self.sim.current_pose_matrix(plan.side)
    phases.append("thumb_face_press")

    # Unload the tabletop through a short guarded rise before the faster main
    # lift.  Progress pauses if the physical two-sided pinch opens, allowing
    # the compliant thumb/card system to catch up without curling the fingers.
    transfer_rotation = plan.pinch_end_effector_rotation
    transfer_position = pinch_position + np.array([0.0, 0.0, 0.004])
    seed = self._move_pose_with_guarded_pinch(
      plan,
      transfer_position,
      seed,
      duration=self._transfer_lift_duration_seconds,
      phase="lift_card",
      end_effector_rotation=transfer_rotation,
    )
    self._advance_fixed(0.10, "lift_card", plan.table_edge_x)
    lift_rotation = plan.pinch_end_effector_rotation
    lift_position = pinch_position + np.array([0.0, 0.0, 0.025])
    seed = self._move_pose_with_guarded_pinch(
      plan,
      lift_position,
      seed,
      duration=self._main_lift_duration_seconds,
      phase="lift_card",
      end_effector_rotation=lift_rotation,
    )
    self._advance_fixed(0.15, "hold_card", plan.table_edge_x)
    phases.extend(("lift_card", "hold_card"))

    # Bring the card to the head with one coordinated shoulder/elbow/wrist
    # gesture.  Both halves move all seven joints, so the arm supports the
    # inward wrist turn instead of freezing rigidly while joint7 rotates.
    maximum_preinspection_card_tilt_degrees = self._maximum_card_tilt_degrees
    seed = self.sim.arm_goal[plan.side].copy()
    preinspection_pose = self.sim.object_pose("card")
    preinspection_rotation = self._card_rotation()
    wrist_turn_start = self._arm_joint_degrees(7)
    tool_position, tool_rotation = self.sim.current_pose_matrix(plan.side)
    local_position = tool_rotation.T @ (preinspection_pose[:3] - tool_position)
    local_rotation = tool_rotation.T @ preinspection_rotation
    # Begin the inward turn while the elbow raises the card. A vertical-only
    # raise followed by a diagonal roll drives wrist joint6 into its limit.
    view_raise_rotation = _tilt_in_world_xy(np.eye(3), 0.0, 40.0) @ local_rotation.T
    view_raise_position = (
      np.array([0.490, -0.150, 1.000]) - view_raise_rotation @ local_position
    )
    seed = self._move_pose_with_guarded_pinch(
      plan,
      view_raise_position,
      seed,
      duration=1.60,
      phase="raise_card_to_view",
      end_effector_rotation=view_raise_rotation,
    )
    seed = self.sim.arm_goal[plan.side].copy()
    tool_position, tool_rotation = self.sim.current_pose_matrix(plan.side)
    card_position = self.sim.object_pose("card")[:3]
    local_position = tool_rotation.T @ (card_position - tool_position)
    local_rotation = tool_rotation.T @ self._card_rotation()
    # Retarget from the measured physical grasp, not the old positive-wrist
    # joint calibration. Only arm commands change; the card stays free.
    # Local Y is the 88 mm LONG axis, X the 63 mm short edge. Put Y upright
    # while turning the printed (-Z) face toward the robot. Use the +90 degree
    # portrait branch: the opposite branch lifts the elbow above the head.
    # A small oblique viewing yaw keeps the shoulder low while continuing
    # the inward wrist turn. The short edge remains horizontal in world space.
    portrait_rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    yaw = np.deg2rad(15.0)
    head_yaw = np.array(
      [
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
      ]
    )
    desired_card_rotation = (
      head_yaw @ _tilt_in_world_xy(np.eye(3), 0.0, 115.0) @ portrait_rotation
    )
    desired_tool_rotation = desired_card_rotation @ local_rotation.T
    seed = self._move_pose_with_guarded_pinch(
      plan,
      _INSPECTION_TARGET_CARD_POSITION - desired_tool_rotation @ local_position,
      seed,
      duration=2.4,
      phase="turn_card_inward",
      end_effector_rotation=desired_tool_rotation,
    )
    del seed
    self._advance_fixed(0.15, "turn_card_inward", plan.table_edge_x)
    if not self._advance_until_sustained_pinch(
      0.80, 0.10, "turn_card_inward", plan.table_edge_x
    ):
      raise RuntimeError("four-finger opposition did not settle at the viewing pose")
    self._advance_fixed(0.25, "inspect_card", plan.table_edge_x)
    wrist_turn_end = self._arm_joint_degrees(7)
    phases.extend(("raise_card_to_view", "turn_card_inward", "inspect_card"))

    final_pose = self.sim.object_pose("card")
    final_rotation = self._card_rotation()
    half_overhang = self._maximum_overhang_fraction >= 0.49
    retained = float(final_pose[2]) >= float(edge_pose[2]) + 0.08
    toward_robot_displacement = float(plan.initial_card_pose[0] - edge_pose[0])
    lateral_card_displacement = abs(float(edge_pose[1] - plan.initial_card_pose[1]))
    contacted = tuple(finger for finger in _FINGERS if finger in self._draw_contacts)
    lift_opposition_fraction = self._opposition_fraction(
      self._lift_opposed_frames,
      self._lift_contact_frames,
    )
    lift_four_finger_fraction = self._opposition_fraction(
      self._lift_four_finger_frames,
      self._lift_contact_frames,
    )
    hold_opposition_fraction = self._opposition_fraction(
      self._hold_opposed_frames,
      self._hold_contact_frames,
    )
    hold_four_finger_fraction = self._opposition_fraction(
      self._hold_four_finger_frames,
      self._hold_contact_frames,
    )
    inspection_opposition_fraction = self._opposition_fraction(
      self._inspection_opposed_frames,
      self._inspection_contact_frames,
    )
    inspection_four_finger_fraction = self._opposition_fraction(
      self._inspection_four_finger_frames,
      self._inspection_contact_frames,
    )
    inspection_hold_opposition_fraction = self._opposition_fraction(
      self._inspection_hold_opposed_frames,
      self._inspection_hold_contact_frames,
    )
    inspection_hold_four_finger_fraction = self._opposition_fraction(
      self._inspection_hold_four_finger_frames,
      self._inspection_hold_contact_frames,
    )
    maximum_lift_opposition_gap = self._lift_maximum_gap_frames * self.sim.timestep
    maximum_inspection_opposition_gap = (
      self._inspection_maximum_gap_frames * self.sim.timestep
    )
    inspection_rotation_degrees = self._rotation_angle_degrees(
      final_rotation,
      preinspection_rotation,
    )
    inspection_face_alignment = self._card_face_to_head_cosine()
    inspection_face_robot_alignment = self._card_face_to_robot_cosine()
    inspection_target_position = _INSPECTION_TARGET_CARD_POSITION.copy()
    inspection_position_error = float(
      np.linalg.norm(final_pose[:3] - inspection_target_position)
    )
    (
      top_contacts,
      bottom_contacts,
      terminal_finger_normal_force,
      terminal_thumb_normal_force,
    ) = self._current_card_face_contact_state()
    terminal_grip_fingers = tuple(
      finger for finger in _FINGERS if finger in top_contacts
    )
    pad_alignments = self._card_pad_alignments()
    minimum_terminal_finger_pad_alignment = min(
      pad_alignments[finger] for finger in _FINGERS
    )
    terminal_thumb_pad_alignment = pad_alignments["thumb"]
    maximum_fingertip_plane_angle = self._maximum_fingertip_plane_angle_degrees()
    wrist_inward_turn_degrees = wrist_turn_start - wrist_turn_end
    slide_force_summary = self._slide_force_monitor.summary()
    slide_press_control_qualified = self._slide_press_control_qualified(
      slide_force_summary
    )
    terminal_pinch = bool(
      "thumb" in bottom_contacts
      and set(_FINGERS).issubset(top_contacts)
      and terminal_thumb_normal_force >= _MINIMUM_PINCH_NORMAL_FORCE
      and terminal_finger_normal_force >= _MINIMUM_PINCH_NORMAL_FORCE
      and minimum_terminal_finger_pad_alignment >= _MINIMUM_FINGER_PAD_ALIGNMENT
      and terminal_thumb_pad_alignment >= _MINIMUM_THUMB_PAD_ALIGNMENT
      and maximum_fingertip_plane_angle <= _MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES
    )
    task_completed = bool(
      half_overhang
      and contacted == _FINGERS
      and self._simultaneous_four_finger_contact
      and self._maximum_slide_fingertip_plane_angle_degrees
      <= _FLAT_DRAW_MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES
      and self._thumb_face_contact
      and self._sustained_pinch
      and lift_opposition_fraction >= 0.95
      and maximum_lift_opposition_gap <= 0.03
      and hold_opposition_fraction >= 0.95
      and maximum_preinspection_card_tilt_degrees < 35.0
      and inspection_rotation_degrees > 80.0
      and wrist_inward_turn_degrees > 80.0
      and inspection_face_alignment >= 0.80
      and inspection_face_robot_alignment >= 0.80
      and inspection_position_error < 0.03
      and inspection_opposition_fraction >= 0.90
      and inspection_four_finger_fraction >= 0.55
      and maximum_inspection_opposition_gap <= 0.03
      and inspection_hold_opposition_fraction >= 0.95
      and inspection_hold_four_finger_fraction >= 0.95
      and self._minimum_inspection_card_height >= float(edge_pose[2]) + 0.005
      and terminal_pinch
      and retained
      and np.isfinite(self._maximum_task_palm_normal_z)
      and self._maximum_task_palm_normal_z < -0.95
      and toward_robot_displacement > 0.08
      and lateral_card_displacement < 0.015
      and self._minimum_supported_card_clearance >= -0.0006
      and self._minimum_card_back_clearance > 0.0
    )
    policy = getattr(self, "acceptance_policy", STRICT_FORCE_POLICY)
    success = accept_task(task_completed, slide_press_control_qualified, policy)
    return PokerDrawResult(
      draw_posture_version=_FLAT_DRAW_POSTURE_VERSION,
      slide_fingertip_plane_angle_limit_degrees=_FLAT_DRAW_MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES,
      success=success,
      acceptance_policy=policy,
      task_completed=task_completed,
      pressure_quality=pressure_quality_label(
        task_completed, slide_press_control_qualified
      ),
      object_name="card",
      side=plan.side,
      initial_card_pose=plan.initial_card_pose.copy(),
      edge_card_pose=edge_pose.copy(),
      preinspection_card_pose=preinspection_pose,
      final_card_pose=final_pose,
      inspection_target_card_position=inspection_target_position,
      table_edge_x=plan.table_edge_x,
      maximum_overhang_fraction=self._maximum_overhang_fraction,
      maximum_card_height=self._maximum_card_height,
      maximum_card_tilt_degrees=self._maximum_card_tilt_degrees,
      maximum_preinspection_card_tilt_degrees=(maximum_preinspection_card_tilt_degrees),
      draw_fingers_contacted=contacted,
      simultaneous_four_finger_contact=self._simultaneous_four_finger_contact,
      press_force_fingers=_FINGERS,
      press_force_target_per_finger_n=self.press_force_per_finger_n,
      established_press_normal_forces_n=self._established_press_normal_forces_n,
      slide_finger_normal_force_means_n=slide_force_summary.force_means_n,
      slide_finger_normal_force_peaks_n=slide_force_summary.force_peaks_n,
      slide_finger_contact_fractions=slide_force_summary.contact_fractions,
      slide_finger_target_band_fractions=slide_force_summary.target_band_fractions,
      slide_finger_maximum_contact_gaps_s=(slide_force_summary.maximum_contact_gaps_s),
      slide_four_finger_contact_fraction=(
        slide_force_summary.four_finger_contact_fraction
      ),
      slide_four_finger_target_band_fraction=(
        slide_force_summary.four_finger_target_band_fraction
      ),
      slide_force_control_recovery_count=(self._slide_force_control_recovery_count),
      slide_press_control_qualified=slide_press_control_qualified,
      maximum_slide_fingertip_plane_angle_degrees=(
        self._maximum_slide_fingertip_plane_angle_degrees
      ),
      maximum_slide_end_effector_z_error_m=(self._maximum_slide_end_effector_z_error_m),
      half_overhang_reached=half_overhang,
      thumb_face_contact=self._thumb_face_contact,
      sustained_pinch=self._sustained_pinch,
      lift_opposition_fraction=lift_opposition_fraction,
      lift_four_finger_fraction=lift_four_finger_fraction,
      maximum_lift_opposition_gap=maximum_lift_opposition_gap,
      hold_opposition_fraction=hold_opposition_fraction,
      hold_four_finger_fraction=hold_four_finger_fraction,
      inspection_rotation_degrees=inspection_rotation_degrees,
      inspection_face_alignment=inspection_face_alignment,
      inspection_face_robot_alignment=inspection_face_robot_alignment,
      inspection_position_error=inspection_position_error,
      inspection_opposition_fraction=inspection_opposition_fraction,
      inspection_four_finger_fraction=inspection_four_finger_fraction,
      maximum_inspection_opposition_gap=maximum_inspection_opposition_gap,
      inspection_hold_opposition_fraction=inspection_hold_opposition_fraction,
      inspection_hold_four_finger_fraction=(inspection_hold_four_finger_fraction),
      minimum_inspection_card_height=self._minimum_inspection_card_height,
      wrist_inward_turn_degrees=wrist_inward_turn_degrees,
      terminal_pinch=terminal_pinch,
      terminal_grip_fingers=terminal_grip_fingers,
      terminal_thumb_normal_force=terminal_thumb_normal_force,
      terminal_finger_normal_force=terminal_finger_normal_force,
      minimum_terminal_finger_pad_alignment=(minimum_terminal_finger_pad_alignment),
      terminal_thumb_pad_alignment=terminal_thumb_pad_alignment,
      maximum_terminal_fingertip_angle_to_card_plane_degrees=(
        maximum_fingertip_plane_angle
      ),
      retained_at_end=retained,
      maximum_palm_normal_z=self._maximum_palm_normal_z,
      toward_robot_displacement=toward_robot_displacement,
      lateral_card_displacement=lateral_card_displacement,
      minimum_supported_card_clearance=self._minimum_supported_card_clearance,
      minimum_card_back_clearance=self._minimum_card_back_clearance,
      phases=tuple(phases),
    )

  def _prepare_and_slide(
    self,
    plan: PokerDrawPlan,
    phases: list[str],
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approach, establish measured pressure, and slide to the near edge."""
    seed = self._execute_approach(plan)
    phases.extend(waypoint.phase for waypoint in plan.waypoints)

    # Curled joints and the compensating wrist pitch need their own pad-height
    # precontact waypoint. The legacy preload approach remains explicit.
    flat_precontact_position = (
      self.sim.object_pose("card")[:3] + _FLAT_DRAW_PRECONTACT_OFFSET
    )
    seed = self._move_pose_linear(
      plan,
      flat_precontact_position,
      seed,
      duration=0.24,
      phase="precontact_card",
    )

    seed, slide_position = self._press_until_four_contacts(plan, seed)
    # Continue from the FK pose of the arm command that carried the stable
    # preload, not from its elastically deflected measured pose.  Using the
    # latter as a fresh IK target silently removes the arm-servo contribution
    # to normal support on the first slide command.
    slide_position, self._press_slide_rotation = self._arm_goal_pose_matrix(plan.side)
    seed = self.sim.arm_goal[plan.side]
    self._press_slide_reference_z = float(slide_position[2])
    phases.append("four_finger_press")
    self._slide_force_monitor.reset()
    self._record_slide_force_metrics = True
    try:
      for _ in range(520):
        overhang = self._overhang_fraction(plan.table_edge_x)
        if overhang >= 0.49:
          break
        remaining_distance = (0.49 - overhang) * self._projected_card_length_x()
        slide_step = min(0.0005, max(0.00015, 0.8 * remaining_distance))
        seed, slide_position = self._move_force_guarded_slide_increment(
          plan,
          slide_position,
          seed,
          slide_step=slide_step,
          duration=max(0.010, slide_step / _PRESS_FORCE_SLIDE_SPEED_M_S),
        )
      else:
        raise RuntimeError("card did not reach half overhang during the slide")
    finally:
      self._record_slide_force_metrics = False
    phases.append("slide_card")

    # Contact relaxation can carry the card another fraction of a millimetre
    # after the commanded slide stops.  Keep the force loop active while
    # drawing it back just enough that its centre of mass remains supported
    # during the unconstrained hand-shape transition.
    self._record_slide_force_metrics = True
    self._press_slide_command_time_s = 0.0
    try:
      self._advance_fixed(0.01, "edge_hold", plan.table_edge_x)
      for _ in range(12):
        if self._overhang_fraction(plan.table_edge_x) <= 0.475:
          break
        seed, slide_position = self._move_force_guarded_slide_increment(
          plan,
          slide_position,
          seed,
          slide_step=0.0005,
          duration=0.0005 / _PRESS_FORCE_SLIDE_SPEED_M_S,
          direction=1.0,
          phase="edge_hold",
        )
      self._advance_fixed(0.02, "edge_hold", plan.table_edge_x)
    finally:
      self._record_slide_force_metrics = False
    edge_pose = self.sim.object_pose("card")

    if not self._simultaneous_four_finger_contact:
      missing = sorted(set(_FINGERS) - self._draw_contacts)
      detail = f"; never contacted: {missing}" if missing else ""
      raise RuntimeError(
        f"four-finger press never established simultaneous contact{detail}"
      )
    phases.append("edge_hold")
    self._press_controller_active = False
    # Pressure stability is a quality label under task-completion-v1. The
    # per-step pause/recovery, real contacts, edge and pinch guards still apply.
    if (
      self.acceptance_policy == STRICT_FORCE_POLICY
      and not self._slide_press_control_qualified(self._slide_force_monitor.summary())
    ):
      raise RuntimeError(
        "card reached edge but continuous four-finger force tracking did not qualify"
      )
    return seed, slide_position, edge_pose

  def _slide_press_control_qualified(self, summary: PressForceSummary) -> bool:
    return bool(
      summary.sample_count > 0
      and min(summary.contact_fractions, default=0.0)
      >= _PRESS_FORCE_MINIMUM_CONTACT_FRACTION
      and min(summary.target_band_fractions, default=0.0)
      >= _PRESS_FORCE_MINIMUM_TARGET_BAND_FRACTION
      and summary.four_finger_contact_fraction >= _PRESS_FORCE_MINIMUM_CONTACT_FRACTION
      and summary.four_finger_target_band_fraction
      >= _PRESS_FORCE_MINIMUM_ALL_TARGET_BAND_FRACTION
      and all(
        abs(force - self.press_force_per_finger_n) <= _PRESS_FORCE_MEAN_TOLERANCE_N
        for force in summary.force_means_n
      )
      and max(summary.maximum_contact_gaps_s, default=float("inf"))
      <= _PRESS_FORCE_MAXIMUM_CONTACT_GAP
      and self._maximum_slide_fingertip_plane_angle_degrees
      <= _FLAT_DRAW_MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES
    )

  def _make_slide_result(
    self,
    plan: PokerDrawPlan,
    phases: list[str],
    failure: str | None,
  ) -> PokerSlideResult:
    edge_pose = self.sim.object_pose("card")
    summary = self._slide_force_monitor.summary()
    qualified = self._slide_press_control_qualified(summary)
    half_overhang = self._maximum_overhang_fraction >= 0.49
    return PokerSlideResult(
      draw_posture_version=_FLAT_DRAW_POSTURE_VERSION,
      slide_fingertip_plane_angle_limit_degrees=_FLAT_DRAW_MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES,
      success=bool(failure is None and half_overhang and qualified),
      error=failure,
      object_name="card",
      side=plan.side,
      initial_card_pose=plan.initial_card_pose.copy(),
      edge_card_pose=edge_pose,
      table_edge_x=plan.table_edge_x,
      maximum_overhang_fraction=self._maximum_overhang_fraction,
      toward_robot_displacement=float(plan.initial_card_pose[0] - edge_pose[0]),
      lateral_card_displacement=abs(float(edge_pose[1] - plan.initial_card_pose[1])),
      press_force_fingers=_FINGERS,
      press_force_target_per_finger_n=self.press_force_per_finger_n,
      established_press_normal_forces_n=self._established_press_normal_forces_n,
      slide_finger_normal_force_means_n=summary.force_means_n,
      slide_finger_normal_force_peaks_n=summary.force_peaks_n,
      slide_finger_contact_fractions=summary.contact_fractions,
      slide_finger_target_band_fractions=summary.target_band_fractions,
      slide_finger_maximum_contact_gaps_s=summary.maximum_contact_gaps_s,
      slide_four_finger_contact_fraction=summary.four_finger_contact_fraction,
      slide_four_finger_target_band_fraction=(summary.four_finger_target_band_fraction),
      slide_force_control_recovery_count=self._slide_force_control_recovery_count,
      slide_press_control_qualified=qualified,
      maximum_slide_fingertip_plane_angle_degrees=(
        self._maximum_slide_fingertip_plane_angle_degrees
      ),
      maximum_slide_end_effector_z_error_m=(self._maximum_slide_end_effector_z_error_m),
      minimum_supported_card_clearance=self._minimum_supported_card_clearance,
      minimum_card_back_clearance=self._minimum_card_back_clearance,
      phases=tuple(phases),
    )

  def refresh_terminal_result(self, result: PokerDrawResult) -> PokerDrawResult:
    """Re-evaluate the physical pinch and viewing pose after recorder settling."""
    if result.object_name != "card" or result.side != _SIDE:
      raise ValueError("terminal poker result must describe the right-hand card")
    final_pose = self.sim.object_pose("card")
    (
      top_contacts,
      bottom_contacts,
      finger_normal_force,
      thumb_normal_force,
    ) = self._current_card_face_contact_state()
    terminal_pinch = bool(
      "thumb" in bottom_contacts
      and set(_FINGERS).issubset(top_contacts)
      and thumb_normal_force >= _MINIMUM_PINCH_NORMAL_FORCE
      and finger_normal_force >= _MINIMUM_PINCH_NORMAL_FORCE
    )
    face_alignment = self._card_face_to_head_cosine()
    face_robot_alignment = self._card_face_to_robot_cosine()
    terminal_grip_fingers = tuple(
      finger for finger in _FINGERS if finger in top_contacts
    )
    pad_alignments = self._card_pad_alignments()
    minimum_finger_pad_alignment = min(pad_alignments[finger] for finger in _FINGERS)
    thumb_pad_alignment = pad_alignments["thumb"]
    maximum_fingertip_plane_angle = self._maximum_fingertip_plane_angle_degrees()
    terminal_pinch = bool(
      terminal_pinch
      and minimum_finger_pad_alignment >= _MINIMUM_FINGER_PAD_ALIGNMENT
      and thumb_pad_alignment >= _MINIMUM_THUMB_PAD_ALIGNMENT
      and maximum_fingertip_plane_angle <= _MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES
    )
    position_error = float(
      np.linalg.norm(final_pose[:3] - result.inspection_target_card_position)
    )
    retained = bool(final_pose[2] >= result.edge_card_pose[2] + 0.08)
    terminal_success = bool(
      terminal_pinch
      and retained
      and face_alignment >= 0.80
      and face_robot_alignment >= 0.80
      and position_error < 0.03
    )
    return replace(
      result,
      success=bool(result.success and terminal_success),
      task_completed=bool(result.task_completed and terminal_success),
      pressure_quality=pressure_quality_label(
        bool(result.task_completed and terminal_success),
        result.slide_press_control_qualified,
      ),
      final_card_pose=final_pose,
      inspection_face_alignment=face_alignment,
      inspection_face_robot_alignment=face_robot_alignment,
      inspection_position_error=position_error,
      terminal_pinch=terminal_pinch,
      terminal_grip_fingers=terminal_grip_fingers,
      terminal_thumb_normal_force=thumb_normal_force,
      terminal_finger_normal_force=finger_normal_force,
      minimum_terminal_finger_pad_alignment=minimum_finger_pad_alignment,
      terminal_thumb_pad_alignment=thumb_pad_alignment,
      maximum_terminal_fingertip_angle_to_card_plane_degrees=(
        maximum_fingertip_plane_angle
      ),
      retained_at_end=retained,
    )

  def _move_pose(
    self,
    plan: PokerDrawPlan,
    position: np.ndarray,
    seed: np.ndarray,
    *,
    duration: float,
    phase: str,
    end_effector_rotation: np.ndarray | None = None,
  ) -> np.ndarray:
    rotation = (
      plan.end_effector_rotation
      if end_effector_rotation is None
      else end_effector_rotation
    )
    result = self.sim.solve_ik(
      plan.side,
      position,
      rotation,
      seed=seed,
      max_iterations=1000,
      position_tolerance=0.0005,
      orientation_tolerance=0.02,
      posture_weight=0.001,
    )
    _require_ik(result, phase)
    self.sim.set_arm_joint_goal(plan.side, result.joint_positions)
    self._advance_fixed(duration, phase, plan.table_edge_x)
    return result.joint_positions

  def _move_pose_linear(
    self,
    plan: PokerDrawPlan,
    position: np.ndarray,
    seed: np.ndarray,
    *,
    duration: float,
    phase: str,
    end_effector_rotation: np.ndarray | None = None,
    position_tolerance: float = 0.0005,
    orientation_tolerance: float = 0.02,
  ) -> np.ndarray:
    """Track a Cartesian line in 20 ms pieces for contact-safe motion."""
    rotation = (
      plan.end_effector_rotation
      if end_effector_rotation is None
      else end_effector_rotation
    )
    start_position, start_rotation = self.sim.current_pose_matrix(plan.side)
    rotation_vector = _rotation_vector_world(rotation, start_rotation)
    angle = float(np.linalg.norm(rotation_vector))
    axis = rotation_vector / max(angle, 1e-12)
    skew = np.array(
      [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    segment_count = max(1, int(round(duration / 0.02)))
    segment_duration = duration / segment_count
    for segment in range(1, segment_count + 1):
      unit = segment / segment_count
      # A minimum-jerk Cartesian profile is important at lift-off: a linear
      # first command used to pull the upper pads away from the light card
      # before the opposed thumb had accelerated it with the hand.
      alpha = 10.0 * unit**3 - 15.0 * unit**4 + 6.0 * unit**5
      target_position = (1.0 - alpha) * start_position + alpha * position
      target_rotation = (
        np.eye(3)
        + np.sin(alpha * angle) * skew
        + (1.0 - np.cos(alpha * angle)) * skew @ skew
      ) @ start_rotation
      result = self.sim.solve_ik(
        plan.side,
        target_position,
        target_rotation,
        seed=seed,
        max_iterations=1000,
        position_tolerance=position_tolerance,
        orientation_tolerance=orientation_tolerance,
        posture_weight=0.001,
      )
      _require_ik(result, phase)
      seed = result.joint_positions
      self._advance_arm_segment(plan, seed, segment_duration, phase)
    return seed

  def _advance_arm_segment(
    self, plan: PokerDrawPlan, target: np.ndarray, duration: float, phase: str
  ) -> None:
    """Execute each IK segment at physics rate, not as a 50 Hz position step.

    The surrounding Cartesian trajectory already supplies minimum-jerk timing.
    Linear interpolation here preserves its velocity across segment boundaries;
    restarting a smoothstep inside every segment would reintroduce force pulses.
    """
    start = self.sim.arm_goal[plan.side].copy()
    steps = max(1, round(duration / self.sim.timestep))
    for step in range(1, steps + 1):
      self.sim.set_arm_joint_goal(plan.side, start + (step / steps) * (target - start))
      self._step(phase, plan.table_edge_x)

  def _move_arm_joints_linear(
    self,
    plan: PokerDrawPlan,
    target: np.ndarray,
    seed: np.ndarray,
    *,
    duration: float,
    phase: str,
  ) -> np.ndarray:
    """Interpolate a calibrated joint gesture without changing its branch."""
    start = np.asarray(seed, dtype=float).copy()
    target = np.asarray(target, dtype=float)
    if start.shape != (7,) or target.shape != (7,):
      raise ValueError("arm joint gestures require seven positions")
    segment_count = max(1, int(round(duration / self.sim.timestep)))
    for segment in range(1, segment_count + 1):
      unit = segment / segment_count
      alpha = 10.0 * unit**3 - 15.0 * unit**4 + 6.0 * unit**5
      command = (1.0 - alpha) * start + alpha * target
      self.sim.set_arm_joint_goal(plan.side, command)
      self._step(phase, plan.table_edge_x)
    return target.copy()

  def _execute_approach(self, plan: PokerDrawPlan) -> np.ndarray:
    """Coordinate all arm/finger targets, including the real recorded start."""
    names = tuple(self.sim._hand_targets[plan.side])
    hand_start = np.array(
      [self.sim.data.qpos[self.sim.model.joint(name).qposadr[0]] for name in names]
    )
    self._set_flat_draw_pose()
    hand_end = np.array([self.sim._hand_targets[plan.side][name] for name in names])
    self.sim.set_hand_joint_targets(names, hand_start)
    seed = self.sim.arm_goal[plan.side].copy()
    for waypoint_index, waypoint in enumerate(plan.waypoints):
      if waypoint_index == 0:
        seed = self._execute_right_side_clearance_approach(
          plan, waypoint, seed, names, hand_start, hand_end
        )
        continue
      start = seed.copy()
      steps = max(1, round(waypoint.duration / self.sim.timestep))
      for step in range(1, steps + 1):
        u = step / steps
        alpha = u**3 * (10.0 + u * (-15.0 + 6.0 * u))
        command = start + alpha * (waypoint.joint_positions - start)
        self.sim.set_arm_joint_goal(plan.side, command)
        self._step(waypoint.phase, plan.table_edge_x)
      if self.sim.arm_goal_error(plan.side) > 0.08:
        raise RuntimeError(f"{waypoint.phase} arm did not settle")
      seed = waypoint.joint_positions.copy()
    return seed

  def _execute_right_side_clearance_approach(
    self,
    plan: PokerDrawPlan,
    waypoint: JointWaypoint,
    seed: np.ndarray,
    hand_names: tuple[str, ...],
    hand_start: np.ndarray,
    hand_end: np.ndarray,
  ) -> np.ndarray:
    """Move from shared home to the card without sweeping across the torso."""
    start_position, start_rotation = self.sim.current_pose_matrix(plan.side)
    target_position = waypoint.end_effector_position
    rotation_vector = _rotation_vector_world(plan.end_effector_rotation, start_rotation)
    rotation_angle = float(np.linalg.norm(rotation_vector))
    rotation_axis = rotation_vector / max(rotation_angle, 1.0e-12)
    axis_skew = np.array(
      [
        [0.0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0.0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0.0],
      ]
    )
    segment_count = max(1, round(waypoint.duration / 0.020))
    segment_duration = waypoint.duration / segment_count
    joint5_id = self.sim.model.joint(f"{plan.side}_arm_joint5").id
    original_joint5_range = self.sim.model.jnt_range[joint5_id].copy()
    self.sim.model.jnt_range[joint5_id] = np.deg2rad(_APPROACH_JOINT5_IK_RANGE_DEGREES)
    try:
      for segment in range(1, segment_count + 1):
        unit = segment / segment_count
        alpha = unit**3 * (10.0 + unit * (-15.0 + 6.0 * unit))
        # Stay on the robot's right side (negative world Y) while rotating the
        # palm down. The small lift/outward arc vanishes smoothly at both ends.
        clearance_arc = np.sin(np.pi * alpha) * np.array([0.0, -0.055, 0.035])
        position = (
          (1.0 - alpha) * start_position + alpha * target_position + clearance_arc
        )
        angle = alpha * rotation_angle
        rotation = (
          np.eye(3)
          + np.sin(angle) * axis_skew
          + (1.0 - np.cos(angle)) * axis_skew @ axis_skew
        ) @ start_rotation
        result = self.sim.solve_ik(
          plan.side,
          position,
          rotation,
          seed=seed,
          max_iterations=1200,
          position_tolerance=0.0005,
          orientation_tolerance=0.02,
          posture_weight=0.001,
        )
        _require_ik(result, waypoint.phase)
        arm_start = self.sim.arm_goal[plan.side].copy()
        substeps = max(1, round(segment_duration / self.sim.timestep))
        for substep in range(1, substeps + 1):
          fraction = substep / substeps
          self.sim.set_arm_joint_goal(
            plan.side,
            arm_start + fraction * (result.joint_positions - arm_start),
          )
          progress = ((segment - 1) + fraction) / segment_count
          hand_alpha = progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))
          self.sim.set_hand_joint_targets(
            hand_names, hand_start + hand_alpha * (hand_end - hand_start)
          )
          self._step(waypoint.phase, plan.table_edge_x)
        seed = result.joint_positions
    finally:
      self.sim.model.jnt_range[joint5_id] = original_joint5_range
    if self.sim.arm_goal_error(plan.side) > 0.08:
      raise RuntimeError(f"{waypoint.phase} arm did not settle")
    return seed

  def _move_pose_with_guarded_pinch(
    self,
    plan: PokerDrawPlan,
    position: np.ndarray,
    seed: np.ndarray,
    *,
    duration: float,
    phase: str,
    end_effector_rotation: np.ndarray,
  ) -> np.ndarray:
    """Lift on a smooth Cartesian path, pausing briefly if the pinch opens."""
    start_position, start_rotation = self.sim.current_pose_matrix(plan.side)
    rotation_vector = _rotation_vector_world(
      end_effector_rotation,
      start_rotation,
    )
    rotation_angle = float(np.linalg.norm(rotation_vector))
    rotation_axis = (
      rotation_vector / rotation_angle
      if rotation_angle > 1.0e-9
      else np.array([1.0, 0.0, 0.0])
    )
    axis_skew = np.array(
      [
        [0.0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0.0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0.0],
      ]
    )
    segment_count = max(1, int(round(duration / 0.02)))
    segment_duration = duration / segment_count
    completed_segments = 0
    stalled_time = 0.0
    while completed_segments < segment_count:
      top, bottom, top_force, bottom_force = self._current_card_face_contact_state()
      lift_supported = bool(
        "thumb" in bottom
        and len(top.intersection(_FINGERS)) >= 3
        and top_force >= _MINIMUM_PINCH_NORMAL_FORCE
        and bottom_force >= _MINIMUM_PINCH_NORMAL_FORCE
      )
      if not lift_supported:
        # Freeze the last commanded pose, not the load-deflected measurement.
        # Rebaselining on every contact dip erased servo preload and generated
        # abrupt wrist commands, turning a brief contact fluctuation into loss.
        seed = self.sim.arm_goal[plan.side].copy()
        self._advance_fixed(segment_duration, phase, plan.table_edge_x)
        stalled_time += segment_duration
        if stalled_time > 1.00:
          raise RuntimeError("physical card pinch did not recover during lift")
        continue

      stalled_time = 0.0
      completed_segments += 1
      unit = completed_segments / segment_count
      alpha = 10.0 * unit**3 - 15.0 * unit**4 + 6.0 * unit**5
      target_position = (1.0 - alpha) * start_position + alpha * position
      interpolated_angle = alpha * rotation_angle
      delta_rotation = (
        np.eye(3)
        + np.sin(interpolated_angle) * axis_skew
        + (1.0 - np.cos(interpolated_angle)) * (axis_skew @ axis_skew)
      )
      target_rotation = delta_rotation @ start_rotation
      result = self.sim.solve_ik(
        plan.side,
        target_position,
        target_rotation,
        seed=seed,
        max_iterations=1000,
        position_tolerance=0.0005,
        orientation_tolerance=0.02,
        posture_weight=0.001,
      )
      _require_ik(result, phase)
      seed = result.joint_positions
      self._advance_arm_segment(plan, seed, segment_duration, phase)
    return seed

  def _approach_flat_pinch(
    self,
    plan: PokerDrawPlan,
    seed: np.ndarray,
  ) -> np.ndarray:
    """Route the thumb around the edge and close its pad onto the card face."""
    required_top = set(_FINGERS)
    finger_names = tuple(f"hand_r_{finger}_joint2" for finger in _FINGERS)
    # A small common preload preserves all four upper contacts while the
    # thumb routes underneath.  Lift-time corrections remain deliberately
    # small so these distal links stay parallel to the card rather than curl
    # around its back.
    finger_targets = np.asarray(
      [
        float(
          self.sim.data.qpos[self.sim.model.joint(name).qposadr[0]]
          + np.deg2rad(self._pinch_upper_preload_degrees)
        )
        for name in finger_names
      ]
    )
    if self.sim.set_hand_joint_targets(finger_names, finger_targets) != len(_FINGERS):
      raise RuntimeError("four-finger pinch preload contains unavailable joints")
    self._pinch_flexion_targets = finger_targets

    thumb_stages = tuple(
      (f"hand_r_{joint}", target, duration)
      for (joint, target), duration in zip(
        _FLAT_PINCH_THUMB_DEGREES.items(),
        (0.16, 0.28, 0.40, 0.80),
        strict=True,
      )
    )
    required_contact_steps = max(1, int(round(0.060 / self.sim.timestep)))
    consecutive_contact_steps = 0
    for thumb_joint, target_degrees, duration in thumb_stages:
      qpos_address = self.sim.model.joint(thumb_joint).qposadr[0]
      start_value = float(self.sim.data.qpos[qpos_address])
      target_value = float(np.deg2rad(target_degrees))
      # The last two joints physically scoop/support the free card. Approach
      # twice as gently, updating the jaw at physics rate rather than steps.
      segment_count = max(1, int(round(duration / self.sim.timestep)))
      for segment in range(1, segment_count + 1):
        unit = segment / segment_count
        alpha = 10.0 * unit**3 - 15.0 * unit**4 + 6.0 * unit**5
        value = (1.0 - alpha) * start_value + alpha * target_value
        if self.sim.set_hand_joint_targets((thumb_joint,), (value,)) != 1:
          raise RuntimeError("thumb face routing joint is unavailable")
        for _ in range(max(1, round(duration / segment_count / self.sim.timestep))):
          self._step("thumb_face_press", plan.table_edge_x)
          top, bottom, top_force, bottom_force = self._current_card_face_contact_state()
          opposed = bool(
            required_top.issubset(top)
            and "thumb" in bottom
            and top_force >= _MINIMUM_PINCH_NORMAL_FORCE
            and bottom_force >= _MINIMUM_PINCH_NORMAL_FORCE
          )
          consecutive_contact_steps = consecutive_contact_steps + 1 if opposed else 0
          if consecutive_contact_steps >= required_contact_steps:
            self._thumb_face_contact = True
            self._sustained_pinch = True
            # Preserve the first stable, force-supported pinch instead of
            # driving all the way into the geometric target.  The full route
            # produced several newtons of unnecessary squeeze on a four-gram
            # card, deflecting the low-torque finger servos and making their
            # otherwise-flat pads look curled during lift-off.
            thumb_names = tuple(
              f"hand_r_{joint}" for joint in _FLAT_PINCH_THUMB_DEGREES
            )
            thumb_values = tuple(
              float(self.sim.data.qpos[self.sim.model.joint(name).qposadr[0]])
              for name in thumb_names
            )
            self.sim.set_hand_joint_targets(thumb_names, thumb_values)
            self._pinch_thumb_joint5_target = thumb_values[-1]
            return self.sim.hold_current_arm_position(plan.side)
    # The hand actuators deliberately have compliant dynamics, so the final
    # thumb flexion trails its command by several tenths of a second.  Keep
    # the arm fixed while the thumb pad finishes approaching the exposed
    # underside; stop immediately once genuine two-sided contact is stable.
    card_geom_id = self.sim.model.geom("card_core_geom").id
    pad_geom_ids = tuple(
      self.sim.model.geom(f"hand_r_{finger}_link4_tactile_pad_col").id
      for finger in _FINGERS
    )
    finger_limits = finger_targets + np.deg2rad(3.0)
    update_period = max(1, int(round(0.020 / self.sim.timestep)))
    minimum_settle_steps = max(1, int(round(0.30 / self.sim.timestep)))
    # Contacts detected during thumb interpolation are still carrying the
    # transition impulse.  Re-establish the debounce from zero and require a
    # settled interval; otherwise the routine can return on its very first
    # frame while the middle pads are about to spring clear.
    consecutive_contact_steps = 0
    for settle_step in range(max(1, int(round(1.30 / self.sim.timestep)))):
      self._step("thumb_face_press", plan.table_edge_x)
      top, bottom, top_force, bottom_force = self._current_card_face_contact_state()
      opposed = bool(
        required_top.issubset(top)
        and "thumb" in bottom
        and top_force >= _MINIMUM_PINCH_NORMAL_FORCE
        and bottom_force >= _MINIMUM_PINCH_NORMAL_FORCE
      )
      consecutive_contact_steps = consecutive_contact_steps + 1 if opposed else 0
      if (
        settle_step >= minimum_settle_steps
        and consecutive_contact_steps >= required_contact_steps
      ):
        self._thumb_face_contact = True
        self._sustained_pinch = True
        return self.sim.hold_current_arm_position(plan.side)
      if settle_step % update_period == 0:
        for index, finger in enumerate(_FINGERS):
          if finger in top:
            continue
          distance = float(
            mujoco.mj_geomDistance(
              self.sim.model,
              self.sim.data,
              card_geom_id,
              pad_geom_ids[index],
              0.01,
              None,
            )
          )
          if distance > 0.00019:
            finger_targets[index] = min(
              finger_targets[index] + np.deg2rad(0.02),
              finger_limits[index],
            )
        self.sim.set_hand_joint_targets(finger_names, finger_targets)
    raise RuntimeError(
      "thumb route did not establish four fingertip pads over and thumb below"
    )

  def _press_until_four_contacts(
    self,
    plan: PokerDrawPlan,
    seed: np.ndarray,
    *,
    distal_offset_degrees: float | None = None,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Lower until four contacts, then establish measured per-pad pressure.

    Production calls leave ``distal_offset_degrees`` unset and use the real
    force loop.  Passing a number explicitly preserves the historical joint-4
    preload protocol for already-recorded friction comparisons only.
    """
    if distal_offset_degrees is not None and not np.isfinite(distal_offset_degrees):
      raise ValueError("distal_offset_degrees must be finite")
    position = (
      plan.waypoints[-1].end_effector_position.copy()
      if distal_offset_degrees is not None
      else self.sim.current_pose_matrix(plan.side)[0]
    )
    required_contacts = set(_FINGERS)
    for _ in range(65):
      position[2] -= 0.0005
      result = self.sim.solve_ik(
        plan.side,
        position,
        plan.end_effector_rotation,
        seed=seed,
        max_iterations=1000,
        position_tolerance=0.00035,
        orientation_tolerance=0.02,
      )
      _require_ik(result, "four_finger_press")
      seed = result.joint_positions
      self.sim.set_arm_joint_goal(plan.side, seed)
      maximum_steps = max(1, int(round(0.04 / self.sim.timestep)))
      for _ in range(maximum_steps):
        self._step("four_finger_press", plan.table_edge_x)
        top_contacts, _ = self._current_card_face_contacts()
        if required_contacts.issubset(top_contacts):
          return self._finish_four_finger_press(
            plan,
            seed,
            distal_offset_degrees,
          )
    # The joint servo can trail the final sequence of 0.5 mm IK commands by a
    # few millimetres.  Keep the last reachable goal fixed and detect contact
    # while the physical arm catches up; do not continue commanding a deeper
    # pose that could drive the card through the table.
    for _ in range(max(1, int(round(0.60 / self.sim.timestep)))):
      self._step("four_finger_press", plan.table_edge_x)
      top_contacts, _ = self._current_card_face_contacts()
      if required_contacts.issubset(top_contacts):
        return self._finish_four_finger_press(
          plan,
          seed,
          distal_offset_degrees,
        )
    missing = sorted(required_contacts - self._draw_contacts)
    raise RuntimeError(
      "four-finger press did not establish simultaneous light contact; "
      f"never contacted: {missing}"
    )

  def _finish_four_finger_press(
    self,
    plan: PokerDrawPlan,
    seed: np.ndarray,
    distal_offset_degrees: float | None,
  ) -> tuple[np.ndarray, np.ndarray]:
    seed = self.sim.hold_current_arm_position(plan.side)
    if distal_offset_degrees is not None:
      # Explicit legacy mode: retain the old joint-4 offset and 40 ms settle
      # exactly.  The new production path never controls force through joint4.
      self._set_hand_pose(
        _DRAW_FINGER_DEGREES,
        _OPEN_THUMB_DEGREES,
        distal_offset_degrees=distal_offset_degrees,
      )
      self._advance_fixed(0.04, "four_finger_press", plan.table_edge_x)
      self._established_press_normal_forces_n = tuple(
        float(value) for value in self._current_card_finger_normal_forces()
      )
      actual_position, _ = self.sim.current_pose_matrix(plan.side)
    else:
      self._activate_press_force_controller()
      self._stabilize_four_finger_press(plan)
      # Preserve the first-contact arm goal after pressure builds.  Capturing
      # the load-deflected measured joints here would erase arm-servo preload.
      seed = self.sim.arm_goal[plan.side]
      actual_position, _ = self._arm_goal_pose_matrix(plan.side)
    return seed, actual_position

  def _arm_goal_pose_matrix(self, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Return FK of the commanded arm joints without changing live state."""
    np.copyto(self.sim.ik_data.qpos, self.sim.data.qpos)
    np.copyto(self.sim.ik_data.qvel, self.sim.data.qvel)
    self.sim.ik_data.qpos[self.sim._arm_qpos[side]] = self.sim.arm_goal[side]
    mujoco.mj_forward(self.sim.model, self.sim.ik_data)
    site_id = self.sim._site_id[side]
    return (
      self.sim.ik_data.site_xpos[site_id].copy(),
      self.sim.ik_data.site_xmat[site_id].reshape(3, 3).copy(),
    )

  def _activate_press_force_controller(self) -> None:
    """Bind the force loop to an orientation-preserving finger linkage."""
    joint2_names = tuple(f"hand_r_{finger}_joint2" for finger in _FINGERS)
    joint3_names = tuple(f"hand_r_{finger}_joint3" for finger in _FINGERS)
    self._press_joint2_base = np.asarray(
      [self.sim._hand_targets[_SIDE][name] for name in joint2_names],
      dtype=float,
    )
    self._press_joint3_base = np.asarray(
      [self.sim._hand_targets[_SIDE][name] for name in joint3_names],
      dtype=float,
    )

    # Determine which equal-and-opposite q2/q3 perturbation lowers each pad in
    # the current palm-down pose.  Since the two axes in a finger are parallel,
    # this changes pad height while preserving its distal orientation.
    directions: list[float] = []
    for finger in _FINGERS:
      jacobian = np.zeros((3, self.sim.model.nv), dtype=float)
      rotational = np.zeros((3, self.sim.model.nv), dtype=float)
      site_id = self.sim.model.site(f"hand_r_{finger}_link4_site").id
      mujoco.mj_jacSite(
        self.sim.model,
        self.sim.data,
        jacobian,
        rotational,
        site_id,
      )
      joint2_dof = self.sim.model.joint(f"hand_r_{finger}_joint2").dofadr[0]
      joint3_dof = self.sim.model.joint(f"hand_r_{finger}_joint3").dofadr[0]
      vertical_sensitivity = float(jacobian[2, joint2_dof] - jacobian[2, joint3_dof])
      directions.append(-1.0 if vertical_sensitivity > 0.0 else 1.0)
    self._press_linkage_directions = np.asarray(directions, dtype=float)
    forces = self._current_card_finger_normal_forces()
    self._press_controller.reset(forces)
    self._press_controller_active = True

  def _apply_press_force_offsets(self, offsets_rad: np.ndarray) -> None:
    signed_offsets = self._press_linkage_directions * offsets_rad
    names: list[str] = []
    values: list[float] = []
    for index, finger in enumerate(_FINGERS):
      names.extend(
        (
          f"hand_r_{finger}_joint2",
          f"hand_r_{finger}_joint3",
        )
      )
      values.extend(
        (
          float(self._press_joint2_base[index] + signed_offsets[index]),
          float(self._press_joint3_base[index] - signed_offsets[index]),
        )
      )
    if self.sim.set_hand_joint_targets(names, values) != len(names):
      raise RuntimeError("four-finger force controller contains unavailable joints")

  def _stabilize_four_finger_press(self, plan: PokerDrawPlan) -> None:
    # The ordinary viewer uses the same pressure gate as before, but does
    # not need the old six-second low-gain preload. Restore slide tuning even
    # if contact establishment fails; this only changes the approach phase.
    controller = self._press_controller
    original_gain = controller.integral_gain
    controller.integral_gain *= 2.0
    try:
      self._wait_for_press_stability(plan)
    finally:
      controller.integral_gain = original_gain

  def _wait_for_press_stability(self, plan: PokerDrawPlan) -> None:
    required_steps = max(
      1,
      int(round(_PRESS_FORCE_STABLE_DURATION / self.sim.timestep)),
    )
    maximum_steps = max(
      1,
      int(round(_PRESS_FORCE_ESTABLISH_TIMEOUT / self.sim.timestep)),
    )
    contact_threshold = self.press_force_per_finger_n * _PRESS_FORCE_CONTACT_RATIO
    consecutive_steps = 0
    stable_force_sum = np.zeros(len(_FINGERS), dtype=float)
    last_forces = np.zeros(len(_FINGERS), dtype=float)
    for _ in range(maximum_steps):
      self._step("four_finger_press", plan.table_edge_x)
      last_forces = self._current_card_finger_normal_forces()
      filtered = self._press_controller.filtered_forces_n
      stable = bool(
        np.all(last_forces >= contact_threshold)
        and np.all(
          np.abs(filtered - self.press_force_per_finger_n)
          <= _PRESS_FORCE_ESTABLISH_TOLERANCE_N
        )
      )
      if stable:
        consecutive_steps += 1
        stable_force_sum += last_forces
      else:
        consecutive_steps = 0
        stable_force_sum.fill(0.0)
      if consecutive_steps >= required_steps:
        self._established_press_normal_forces_n = tuple(
          float(value / consecutive_steps) for value in stable_force_sum
        )
        return
    raise RuntimeError(
      "four-finger force press did not stabilize at target; "
      f"target={self.press_force_per_finger_n:.3f} N, "
      f"measured={tuple(float(value) for value in last_forces)}"
    )

  def _press_is_load_bearing(self) -> bool:
    threshold = self.press_force_per_finger_n * _PRESS_FORCE_CONTACT_RATIO
    return bool(np.all(self._current_card_finger_normal_forces() >= threshold))

  def _press_is_target_ready(self) -> bool:
    forces = self._current_card_finger_normal_forces()
    return bool(
      np.all(forces >= self.press_force_per_finger_n * _PRESS_FORCE_CONTACT_RATIO)
      and np.all(
        np.abs(forces - self.press_force_per_finger_n)
        <= _PRESS_FORCE_TARGET_TOLERANCE_N
      )
      and np.all(
        np.abs(self._press_controller.filtered_forces_n - self.press_force_per_finger_n)
        <= _PRESS_FORCE_TARGET_TOLERANCE_N
      )
    )

  def _recover_four_finger_press(
    self,
    plan: PokerDrawPlan,
    position: np.ndarray,
    *,
    phase: str = "slide_card",
  ) -> tuple[np.ndarray, np.ndarray]:
    """Pause new slide commands until every pad again bears normal force."""
    # Keep the already commanded Cartesian reference untouched.  In
    # particular, never recapture the load-deflected measured joints here:
    # doing so clears arm-servo preload and makes recovery self-defeating.
    required_steps = max(
      1,
      int(round(_PRESS_FORCE_RECOVERY_STABLE_DURATION / self.sim.timestep)),
    )
    maximum_steps = max(
      1,
      int(round(_PRESS_FORCE_RECOVERY_TIMEOUT / self.sim.timestep)),
    )
    consecutive_steps = 0
    for _ in range(maximum_steps):
      self._step(phase, plan.table_edge_x)
      ready = self._press_is_target_ready()
      consecutive_steps = consecutive_steps + 1 if ready else 0
      if consecutive_steps >= required_steps:
        return self.sim.arm_goal[plan.side], position
    forces = self._current_card_finger_normal_forces()
    raise RuntimeError(
      "four-finger pressure did not recover while tangential slide was paused; "
      f"forces={tuple(float(value) for value in forces)}"
    )

  def _move_force_guarded_slide_increment(
    self,
    plan: PokerDrawPlan,
    position: np.ndarray,
    seed: np.ndarray,
    *,
    slide_step: float,
    duration: float,
    direction: float = -1.0,
    phase: str = "slide_card",
  ) -> tuple[np.ndarray, np.ndarray]:
    """Advance one small tangential increment, pausing on any pad force loss."""
    if not self._press_is_load_bearing():
      self._slide_force_control_recovery_count += 1
      seed, position = self._recover_four_finger_press(plan, position, phase=phase)

    target_position = position.copy()
    target_position[0] += direction * slide_step
    result = self.sim.solve_ik(
      plan.side,
      target_position,
      self._press_slide_rotation,
      seed=seed,
      max_iterations=1000,
      position_tolerance=0.00002,
      orientation_tolerance=0.001,
      posture_weight=0.001,
    )
    _require_ik(result, "slide_card")
    # Solve only the 0.5 mm endpoint, then stream its joint displacement in
    # roughly 50 micrometre increments at the physics rate.  A direct endpoint
    # command produced a load/unload impact every 18 ms even though its IK was
    # exact.  This interpolation is continuous across endpoints and avoids
    # multiplying the IK cost by ten.
    start_goal = self.sim.arm_goal[plan.side]
    target_goal = result.joint_positions
    nominal_speed = slide_step / duration
    progressed = 0.0
    maximum_steps = max(
      1,
      int(
        np.ceil(
          (duration + _PRESS_FORCE_SLIDE_ACCELERATION_DURATION) / self.sim.timestep
        )
      ),
    )
    for _ in range(maximum_steps):
      self._press_slide_command_time_s += self.sim.timestep
      acceleration_fraction = min(
        1.0,
        self._press_slide_command_time_s / _PRESS_FORCE_SLIDE_ACCELERATION_DURATION,
      )
      incremental_distance = min(
        slide_step - progressed,
        nominal_speed * acceleration_fraction * self.sim.timestep,
      )
      progressed += incremental_distance
      unit = progressed / slide_step
      commanded_goal = (1.0 - unit) * start_goal + unit * target_goal
      commanded_position = (1.0 - unit) * position + unit * target_position
      self.sim.set_arm_joint_goal(plan.side, commanded_goal)
      self._step(phase, plan.table_edge_x)
      if not self._press_is_load_bearing():
        self._slide_force_control_recovery_count += 1
        return self._recover_four_finger_press(plan, commanded_position, phase=phase)
      if progressed >= slide_step - 1.0e-12:
        return target_goal, target_position
    raise RuntimeError("slide command did not finish its bounded interpolation")

  def _advance_fixed(self, duration: float, phase: str, table_edge_x: float) -> None:
    steps = max(1, int(round(duration / self.sim.timestep)))
    for _ in range(steps):
      self._step(phase, table_edge_x)

  def _advance_until_sustained_pinch(
    self,
    maximum_duration: float,
    required_duration: float,
    phase: str,
    table_edge_x: float,
  ) -> bool:
    """Require continuous thumb-under/finger-over tactile-pad opposition."""
    maximum_steps = max(1, int(round(maximum_duration / self.sim.timestep)))
    required_steps = max(1, int(round(required_duration / self.sim.timestep)))
    consecutive_steps = 0
    for _ in range(maximum_steps):
      self._step(phase, table_edge_x)
      opposed = self._pinch_is_force_supported()
      consecutive_steps = consecutive_steps + 1 if opposed else 0
      if opposed:
        self._thumb_face_contact = True
      if consecutive_steps >= required_steps:
        self._sustained_pinch = True
        return True
    return False

  def _advance_until_arm_settled(
    self,
    maximum_duration: float,
    phase: str,
    side: str,
    table_edge_x: float,
  ) -> None:
    stable_steps = 0
    maximum_steps = max(1, int(round(maximum_duration / self.sim.timestep)))
    for _ in range(maximum_steps):
      self._step(phase, table_edge_x)
      settled = (
        self.sim.arm_goal_error(side) < 0.012 and self.sim.arm_velocity(side) < 0.08
      )
      stable_steps = stable_steps + 1 if settled else 0
      if stable_steps >= 5:
        return
    raise RuntimeError(f"{phase} did not settle within {maximum_duration:.2f} s")

  def _advance_until_hand_targets_settled(
    self,
    maximum_duration: float,
    phase: str,
    joint_names: tuple[str, ...],
    table_edge_x: float,
  ) -> None:
    """Wait for selected low-torque hand servos, ignoring the open thumb."""
    qpos_addresses = np.asarray(
      [self.sim.model.joint(name).qposadr[0] for name in joint_names],
      dtype=int,
    )
    dof_addresses = np.asarray(
      [self.sim.model.joint(name).dofadr[0] for name in joint_names],
      dtype=int,
    )
    targets = np.asarray(
      [self.sim._hand_targets[_SIDE][name] for name in joint_names],
      dtype=float,
    )
    maximum_steps = max(1, int(round(maximum_duration / self.sim.timestep)))
    required_stable_steps = max(1, int(round(0.05 / self.sim.timestep)))
    stable_steps = 0
    for _ in range(maximum_steps):
      self._step(phase, table_edge_x)
      error = float(np.max(np.abs(self.sim.data.qpos[qpos_addresses] - targets)))
      velocity = float(np.max(np.abs(self.sim.data.qvel[dof_addresses])))
      settled = error < np.deg2rad(0.5) and velocity < 0.05
      stable_steps = stable_steps + 1 if settled else 0
      if stable_steps >= required_stable_steps:
        return
    raise RuntimeError(
      f"{phase} finger servos did not settle within {maximum_duration:.2f} s"
    )

  def _step(self, phase: str, table_edge_x: float) -> None:
    self.sim.step()
    if self._press_controller_active and phase in {
      "four_finger_press",
      "slide_card",
      "edge_hold",
    }:
      finger_forces = self._current_card_finger_normal_forces()
      offsets, updated = self._press_controller.observe(finger_forces)
      if updated:
        self._apply_press_force_offsets(offsets)
      if phase in {"slide_card", "edge_hold"} and self._record_slide_force_metrics:
        self._slide_force_monitor.observe(finger_forces)
        self._maximum_slide_fingertip_plane_angle_degrees = max(
          self._maximum_slide_fingertip_plane_angle_degrees,
          self._maximum_fingertip_plane_angle_degrees(),
        )
        current_position, _ = self.sim.current_pose_matrix(_SIDE)
        self._maximum_slide_end_effector_z_error_m = max(
          self._maximum_slide_end_effector_z_error_m,
          abs(float(current_position[2]) - self._press_slide_reference_z),
        )
    if phase in {"four_finger_press", "slide_card"}:
      current_contacts = self._current_card_finger_contacts()
      self._draw_contacts.update(current_contacts)
      self._draw_contacts.discard("thumb")
      self._simultaneous_four_finger_contact |= set(_FINGERS).issubset(current_contacts)
    if phase == "thumb_face_press":
      self._thumb_face_contact |= self._thumb_is_on_card_face()
    if phase in {
      "lift_card",
      "hold_card",
      "raise_card_to_view",
      "turn_card_inward",
      "inspect_card",
    }:
      top, bottom, top_force, bottom_force = self._current_card_face_contact_state()
      opposed = bool(
        "thumb" in bottom
        and top.intersection(_FINGERS)
        and top_force >= _MINIMUM_PINCH_NORMAL_FORCE
        and bottom_force >= _MINIMUM_PINCH_NORMAL_FORCE
      )
      all_four = bool("thumb" in bottom and set(_FINGERS).issubset(top))
      if phase == "lift_card":
        self._lift_contact_frames += 1
        self._lift_opposed_frames += int(opposed)
        self._lift_four_finger_frames += int(all_four)
        self._lift_current_gap_frames = (
          0 if opposed else self._lift_current_gap_frames + 1
        )
        self._lift_maximum_gap_frames = max(
          self._lift_maximum_gap_frames,
          self._lift_current_gap_frames,
        )
      elif phase == "hold_card":
        self._hold_contact_frames += 1
        self._hold_opposed_frames += int(opposed)
        self._hold_four_finger_frames += int(all_four)
      elif phase in {"raise_card_to_view", "turn_card_inward"}:
        self._inspection_contact_frames += 1
        self._inspection_opposed_frames += int(opposed)
        self._inspection_four_finger_frames += int(all_four)
        self._inspection_current_gap_frames = (
          0 if opposed else self._inspection_current_gap_frames + 1
        )
        self._inspection_maximum_gap_frames = max(
          self._inspection_maximum_gap_frames,
          self._inspection_current_gap_frames,
        )
      else:
        self._inspection_hold_contact_frames += 1
        self._inspection_hold_opposed_frames += int(opposed)
        self._inspection_hold_four_finger_frames += int(all_four)
      self._maintain_dynamic_pinch(
        top,
        bottom,
        top_force,
        bottom_force,
      )
      if phase in {"raise_card_to_view", "turn_card_inward", "inspect_card"}:
        self._minimum_inspection_card_height = min(
          self._minimum_inspection_card_height,
          float(self.sim.object_pose("card")[2]),
        )
    if phase in {
      "four_finger_press",
      "slide_card",
      "edge_hold",
      "thumb_face_press",
    }:
      self._maximum_overhang_fraction = max(
        self._maximum_overhang_fraction,
        self._overhang_fraction(table_edge_x),
      )
    if phase in {
      "clear_card",
      "ready_card",
      "hover_card",
      "precontact_card",
      "four_finger_press",
      "slide_card",
      "edge_hold",
      "thumb_face_press",
    }:
      back_clearance = self._card_back_top_clearance()
      self._minimum_card_back_clearance = min(
        self._minimum_card_back_clearance,
        back_clearance,
      )
      if back_clearance <= 0.0:
        raise RuntimeError(
          "card back dropped below the raised tabletop: "
          f"top_clearance={back_clearance:.6f} m"
        )
    if phase in {"four_finger_press", "slide_card"}:
      clearance = self._card_table_clearance()
      overhang = self._overhang_fraction(table_edge_x)
      if overhang < 0.02:
        self._minimum_supported_card_clearance = min(
          self._minimum_supported_card_clearance,
          clearance,
        )
        if clearance < -0.0006:
          raise RuntimeError(
            "card penetrated the raised tabletop during the light press: "
            f"clearance={clearance:.6f} m"
          )
    self._maximum_card_height = max(
      self._maximum_card_height,
      float(self.sim.object_pose("card")[2]),
    )
    self._maximum_card_tilt_degrees = max(
      self._maximum_card_tilt_degrees,
      self._card_tilt_degrees(),
    )
    if phase in {
      "clear_card",
      "ready_card",
      "hover_card",
      "precontact_card",
      "four_finger_press",
      "slide_card",
      "edge_hold",
    }:
      self._maximum_palm_normal_z = max(
        self._maximum_palm_normal_z,
        self._palm_normal_z(),
      )
      # The common idle pose faces inward. Enforce palm-down task work after
      # the initial clear_card reorientation; retain the full recorded maximum.
      if phase != "clear_card":
        self._maximum_task_palm_normal_z = max(
          self._maximum_task_palm_normal_z, self._palm_normal_z()
        )
    if self.observer is not None:
      self.observer(self.sim, phase)

  def _current_card_finger_contacts(self) -> set[str]:
    top_contacts, bottom_contacts = self._current_card_face_contacts()
    return top_contacts | bottom_contacts

  def _thumb_is_on_card_face(self) -> bool:
    _, bottom_contacts = self._current_card_face_contacts()
    return "thumb" in bottom_contacts

  def _current_card_face_contacts(self) -> tuple[set[str], set[str]]:
    """Return exact tactile pads contacting the card's back and face."""
    top_contacts, bottom_contacts, _, _ = self._current_card_face_contact_state()
    return top_contacts, bottom_contacts

  def _current_card_face_contact_state(
    self,
  ) -> tuple[set[str], set[str], float, float]:
    """Return exact face contacts and summed top/bottom normal forces."""
    top_contacts, bottom_contacts, normal_forces = (
      self._current_card_face_contact_details()
    )
    return (
      top_contacts,
      bottom_contacts,
      sum(normal_forces[finger] for finger in _FINGERS),
      normal_forces["thumb"],
    )

  def _current_card_finger_normal_forces(self) -> np.ndarray:
    """Return card-back normal force for index through pinky, in newtons."""
    _, _, normal_forces = self._current_card_face_contact_details()
    return np.asarray([normal_forces[finger] for finger in _FINGERS], dtype=float)

  def _current_card_face_contact_details(
    self,
  ) -> tuple[set[str], set[str], dict[str, float]]:
    """Return face-classified tactile-pad contacts and force per pad."""
    card_geom_id = self.sim.model.geom("card_core_geom").id
    card_body_id = self.sim.model.body("card").id
    card_position = self.sim.data.xpos[card_body_id]
    card_rotation = self.sim.data.xmat[card_body_id].reshape(3, 3)
    card_half_size = self.sim.model.geom_size[card_geom_id]
    pad_fingers = {
      "hand_r_thumb_link6_tactile_pad_col": "thumb",
      **{f"hand_r_{finger}_link4_tactile_pad_col": finger for finger in _FINGERS},
    }
    top_contacts: set[str] = set()
    bottom_contacts: set[str] = set()
    normal_forces = {finger: 0.0 for finger in (*_FINGERS, "thumb")}
    for contact_index, contact in enumerate(self.sim.data.contact):
      if card_geom_id not in (int(contact.geom1), int(contact.geom2)):
        continue
      other_geom = int(
        contact.geom2 if contact.geom1 == card_geom_id else contact.geom1
      )
      geom_name = self.sim.model.geom(other_geom).name or ""
      finger = pad_fingers.get(geom_name)
      if finger is None:
        continue
      local_contact = card_rotation.T @ (contact.pos - card_position)
      local_normal = card_rotation.T @ contact.frame[:3]
      if abs(float(local_normal[2])) < _FACE_NORMAL_COSINE:
        continue
      # Reject edge contacts even if numerical penetration places their point
      # slightly above or below the centre plane.
      if abs(float(local_contact[0])) > float(card_half_size[0]) + 0.0003:
        continue
      if abs(float(local_contact[1])) > float(card_half_size[1]) + 0.0003:
        continue
      wrench = np.zeros(6, dtype=float)
      mujoco.mj_contactForce(
        self.sim.model,
        self.sim.data,
        contact_index,
        wrench,
      )
      normal_force = max(0.0, float(wrench[0]))
      if float(local_contact[2]) > 0.0 and finger in _FINGERS:
        top_contacts.add(finger)
        normal_forces[finger] += normal_force
      elif float(local_contact[2]) < 0.0 and finger == "thumb":
        bottom_contacts.add(finger)
        normal_forces[finger] += normal_force
    return top_contacts, bottom_contacts, normal_forces

  def _pinch_is_force_supported(self) -> bool:
    (
      top_contacts,
      bottom_contacts,
      top_normal_force,
      bottom_normal_force,
    ) = self._current_card_face_contact_state()
    return bool(
      "thumb" in bottom_contacts
      and set(_FINGERS).issubset(top_contacts)
      and top_normal_force >= _MINIMUM_PINCH_NORMAL_FORCE
      and bottom_normal_force >= _MINIMUM_PINCH_NORMAL_FORCE
    )

  def _maintain_dynamic_pinch(
    self,
    top_contacts: set[str],
    bottom_contacts: set[str],
    top_force: float,
    bottom_force: float,
  ) -> None:
    """Bounded low-bandwidth jaw force feedback, never modifying sensor data."""
    del top_contacts, bottom_contacts, top_force, bottom_force
    if self._pinch_flexion_targets is None:
      return
    _, _, forces = self._current_card_face_contact_details()
    measured = np.asarray([forces[f] for f in (*_FINGERS, "thumb")])
    if not np.all(np.isfinite(measured)) or np.any(measured < 0):
      raise RuntimeError("invalid pinch contact force feedback")
    if self._pinch_force_filtered is None:
      self._pinch_force_filtered = measured.copy()
      self._pinch_force_reference = np.r_[
        self._pinch_flexion_targets, self._pinch_thumb_joint5_target
      ]
    alpha = self.sim.timestep / (0.020 + self.sim.timestep)
    self._pinch_force_filtered += alpha * (measured - self._pinch_force_filtered)
    self._pinch_force_steps += 1
    period = max(1, round(0.010 / self.sim.timestep))
    if self._pinch_force_steps % period:
      return
    elapsed = period * self.sim.timestep
    target = self._pinch_force_targets_n
    error = target - self._pinch_force_filtered
    error[np.abs(error) <= 0.05] = 0.0
    rate = np.deg2rad(self._pinch_force_maximum_joint_rate_degrees_s) * elapsed
    delta = np.clip(self._pinch_force_integral_gain * error * elapsed, -rate, rate)
    current = np.r_[self._pinch_flexion_targets, self._pinch_thumb_joint5_target]
    reference = self._pinch_force_reference
    updated = np.clip(
      current + delta,
      reference - np.deg2rad(self._pinch_force_maximum_offset_degrees),
      reference + np.deg2rad(self._pinch_force_maximum_offset_degrees),
    )
    self._pinch_flexion_targets[:] = updated[:4]
    self._pinch_thumb_joint5_target = float(updated[4])
    self.sim.set_hand_joint_targets(
      tuple(f"hand_r_{f}_joint2" for f in _FINGERS), updated[:4]
    )
    self.sim.set_hand_joint_targets(("hand_r_thumb_joint5",), (updated[4],))

  @staticmethod
  def _opposition_fraction(opposed_frames: int, total_frames: int) -> float:
    if total_frames <= 0:
      return 0.0
    return opposed_frames / total_frames

  def _card_tilt_degrees(self) -> float:
    card_normal_z = float(self._card_rotation()[2, 2])
    return float(np.rad2deg(np.arccos(np.clip(card_normal_z, -1.0, 1.0))))

  def _card_rotation(self) -> np.ndarray:
    card_id = self.sim.model.body("card").id
    return self.sim.data.xmat[card_id].reshape(3, 3).copy()

  def _card_face_to_head_cosine(self) -> float:
    card_id = self.sim.model.body("card").id
    card_position = self.sim.data.xpos[card_id]
    head_camera_id = self.sim.model.camera("head").id
    to_head = self.sim.data.cam_xpos[head_camera_id] - card_position
    distance = float(np.linalg.norm(to_head))
    if distance <= 1.0e-9:
      return -1.0
    to_head /= distance
    # card local -Z is the visible face; +Z is the red back.
    face_normal = -self._card_rotation()[:, 2]
    return float(np.clip(face_normal @ to_head, -1.0, 1.0))

  def _card_face_to_robot_cosine(self) -> float:
    """Return alignment of the printed face with the robot-facing -X axis."""
    face_normal = -self._card_rotation()[:, 2]
    return float(np.clip(-face_normal[0], -1.0, 1.0))

  @staticmethod
  def _rotation_angle_degrees(target: np.ndarray, source: np.ndarray) -> float:
    relative_rotation = target @ source.T
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine)))

  def _overhang_fraction(self, table_edge_x: float) -> float:
    pose = self.sim.object_pose("card")
    projected_half_length = 0.5 * self._projected_card_length_x()
    if projected_half_length <= 1.0e-9:
      return 0.0
    overhang_length = table_edge_x - (float(pose[0]) - projected_half_length)
    return float(np.clip(overhang_length / (2.0 * projected_half_length), 0.0, 1.0))

  def _projected_card_length_x(self) -> float:
    """Return the card's current full extent along the robot-facing X axis."""
    pose = self.sim.object_pose("card")
    rotation = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(rotation, pose[3:])
    rotation = rotation.reshape(3, 3)
    half_size = self.sim.model.geom("card_core_geom").size
    projected_half_length = abs(float(rotation[0, 0])) * float(half_size[0]) + abs(
      float(rotation[0, 1])
    ) * float(half_size[1])
    return 2.0 * projected_half_length

  def _card_table_clearance(self) -> float:
    card_id = self.sim.model.body("card").id
    card_rotation = self.sim.data.xmat[card_id].reshape(3, 3)
    card_half_size = self.sim.model.geom("card_core_geom").size
    card_half_height = float(np.abs(card_rotation[2]) @ card_half_size)
    card_bottom = float(self.sim.data.xpos[card_id, 2] - card_half_height)

    table_geom_id = self.sim.model.geom("poker_table_top").id
    table_rotation = self.sim.data.geom_xmat[table_geom_id].reshape(3, 3)
    table_half_size = self.sim.model.geom_size[table_geom_id]
    table_half_height = float(np.abs(table_rotation[2]) @ table_half_size)
    table_top = float(self.sim.data.geom_xpos[table_geom_id, 2] + table_half_height)
    return card_bottom - table_top

  def _palm_normal_z(self) -> float:
    hand_id = self.sim.model.body("hand_r_base_link").id
    hand_rotation = self.sim.data.xmat[hand_id].reshape(3, 3)
    # The anatomical right-palm normal is local +Y.
    return float(hand_rotation[2, 1])

  def _arm_joint_degrees(self, joint_number: int) -> float:
    if not 1 <= joint_number <= 7:
      raise ValueError("arm joint number must be in [1, 7]")
    qpos_address = self.sim._arm_qpos[_SIDE][joint_number - 1]
    return float(np.rad2deg(self.sim.data.qpos[qpos_address]))

  def _card_back_top_clearance(self) -> float:
    back_geom_id = self.sim.model.geom("card_back_visual").id
    back_rotation = self.sim.data.geom_xmat[back_geom_id].reshape(3, 3)
    back_half_height = float(
      np.abs(back_rotation[2]) @ self.sim.model.geom_size[back_geom_id]
    )
    back_top = float(self.sim.data.geom_xpos[back_geom_id, 2] + back_half_height)

    table_geom_id = self.sim.model.geom("poker_table_top").id
    table_rotation = self.sim.data.geom_xmat[table_geom_id].reshape(3, 3)
    table_half_height = float(
      np.abs(table_rotation[2]) @ self.sim.model.geom_size[table_geom_id]
    )
    table_top = float(self.sim.data.geom_xpos[table_geom_id, 2] + table_half_height)
    return back_top - table_top

  def _card_pad_alignments(self) -> dict[str, float]:
    """Return alignment of the taxels that physically touch each card face."""
    card_geom_id = self.sim.model.geom("card_core_geom").id
    card_back_normal = self._card_rotation()[:, 2]
    pad_fingers = {
      "hand_r_thumb_link6_tactile_pad_col": "thumb",
      **{f"hand_r_{finger}_link4_tactile_pad_col": finger for finger in _FINGERS},
    }
    alignments = {finger: 0.0 for finger in (*_FINGERS, "thumb")}
    layout_names = np.asarray(self._tactile_layout.body_names)
    for contact in self.sim.data.contact:
      if card_geom_id not in (int(contact.geom1), int(contact.geom2)):
        continue
      other_geom = int(
        contact.geom2 if contact.geom1 == card_geom_id else contact.geom1
      )
      finger = pad_fingers.get(self.sim.model.geom(other_geom).name or "")
      if finger is None:
        continue
      body_id = int(self.sim.model.geom_bodyid[other_geom])
      body_name = self.sim.model.body(body_id).name or ""
      taxel_indices = np.flatnonzero(layout_names == body_name)
      if not taxel_indices.size:
        continue
      body_rotation = self.sim.data.xmat[body_id].reshape(3, 3)
      local_contact = body_rotation.T @ (contact.pos - self.sim.data.xpos[body_id])
      local_positions = self._tactile_layout.local_pos[taxel_indices]
      nearest = int(
        taxel_indices[
          np.argmin(np.linalg.norm(local_positions - local_contact, axis=1))
        ]
      )
      taxel_normal = body_rotation @ self._tactile_layout.local_normal[nearest]
      # Finger-pad normals point into the back (+Z side), while the opposed
      # thumb-pad normal points into the printed face (-Z side).  Preserve the
      # sign so a pad facing away from the card cannot pass this check.
      sign = -1.0 if finger in _FINGERS else 1.0
      alignment = sign * float(taxel_normal @ card_back_normal)
      alignments[finger] = max(alignments[finger], alignment)
    return alignments

  def _maximum_fingertip_plane_angle_degrees(self) -> float:
    """Return the worst distal-finger longitudinal angle to the card plane."""
    card_normal = self._card_rotation()[:, 2]
    angles: list[float] = []
    for finger in _FINGERS:
      body_id = self.sim.model.body(f"hand_r_{finger}_link4").id
      body_rotation = self.sim.data.xmat[body_id].reshape(3, 3)
      longitudinal_axis = body_rotation @ np.array([0.0, 0.0, -1.0])
      sine = abs(float(longitudinal_axis @ card_normal))
      angles.append(float(np.rad2deg(np.arcsin(np.clip(sine, 0.0, 1.0)))))
    return max(angles)

  def _set_flat_pinch_pose(self, *, include_thumb: bool = True) -> None:
    names: list[str] = []
    values: list[float] = []
    for finger, joint_degrees in zip(
      _FINGERS,
      _FLAT_PINCH_FINGER_DEGREES,
      strict=True,
    ):
      for joint, degrees in zip((1, 2, 3, 4), joint_degrees, strict=True):
        names.append(f"hand_r_{finger}_joint{joint}")
        values.append(float(np.deg2rad(degrees)))
    if include_thumb:
      for suffix, degrees in _FLAT_PINCH_THUMB_DEGREES.items():
        names.append(f"hand_r_{suffix}")
        values.append(float(np.deg2rad(degrees)))
    accepted = self.sim.set_hand_joint_targets(names, values)
    if accepted != len(names):
      raise RuntimeError("flat poker pinch contains unavailable joints")

  def _set_flat_draw_pose(self) -> None:
    """Curl MCP/PIP joints for raised-palm, tip-side tactile-pad contact."""
    names: list[str] = []
    values: list[float] = []
    for finger, joint_degrees in zip(
      _FINGERS,
      _FLAT_DRAW_FINGER_DEGREES,
      strict=True,
    ):
      for joint, degrees in zip((1, 2, 3, 4), joint_degrees, strict=True):
        names.append(f"hand_r_{finger}_joint{joint}")
        values.append(float(np.deg2rad(degrees)))
    for suffix, degrees in _FLAT_DRAW_THUMB_DEGREES.items():
      names.append(f"hand_r_{suffix}")
      values.append(float(np.deg2rad(degrees)))
    if self.sim.set_hand_joint_targets(names, values) != len(names):
      raise RuntimeError("flat poker draw pose contains unavailable joints")

  def _set_hand_pose(
    self,
    finger_degrees: tuple[float, float, float, float],
    thumb_degrees: dict[str, float],
    *,
    distal_offset_degrees: float = 0.0,
  ) -> None:
    names: list[str] = []
    values: list[float] = []
    for finger, flexion, abduction in zip(
      _FINGERS,
      finger_degrees,
      _FINGER_ABDUCTION_DEGREES,
      strict=True,
    ):
      for joint, degrees in (
        (1, abduction),
        (2, flexion),
        (3, flexion),
        (4, 0.5 * flexion + distal_offset_degrees),
      ):
        names.append(f"hand_r_{finger}_joint{joint}")
        values.append(float(np.deg2rad(degrees)))
    for suffix, degrees in thumb_degrees.items():
      names.append(f"hand_r_{suffix}")
      values.append(float(np.deg2rad(degrees)))
    accepted = self.sim.set_hand_joint_targets(names, values)
    if accepted != len(names):
      raise RuntimeError("poker-draw hand pose contains unavailable joints")


def _require_ik(result: IkResult, phase: str) -> None:
  if result.success:
    return
  raise RuntimeError(
    f"IK failed for {phase}: position_error={result.position_error:.4f}, "
    f"orientation_error={result.orientation_error:.4f}"
  )
