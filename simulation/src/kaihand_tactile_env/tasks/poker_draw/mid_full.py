"""Middle-pressure draw with compliant contact and bounded tactile pinch.

Neither the shared robot nor production task defaults are changed. The world-X
servo budget applies through the supported edge handover, not during pickup.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import mujoco
import numpy as np

from ...shared.physics_integrity import (
  PHYSICS_INTEGRITY_VERSION,
  check_physics_clock,
  check_physics_state,
)
from .acceptance import (
  MAXIMUM_MULTI_FINGER_LOW_LOAD_S,
  MINIMUM_LOADED_FINGERS,
  STRICT_FORCE_POLICY,
  TASK_COMPLETION_POLICY,
  accept_edge,
  check_policy,
)
from .config import (
  _FINGERS,
  _FLAT_DRAW_MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES,
  _PRESS_FORCE_CONTACT_RATIO,
)
from .pressure_window import (
  ForceLimitedPokerSimulation,
  PressureWindowExecutor,
  PressureWindowSettings,
)
from .task import PokerDrawPlan

MID_FORCE_PER_FINGER_N = 0.50
MID_FORCE_SETTINGS = PressureWindowSettings(
  goal="table-edge",
  table_friction=1.0,
  drive_limit_n=4.0,
  # 12 mm/s shortens the long supported draw while preserving the same
  # pressure target, force budget, final slow-down and edge dwell.
  slide_speed_m_s=0.012,
  # Over-damp the two compliant card interfaces. This keeps the raw signal
  # physical while suppressing the table-edge contact/recontact limit cycle.
  # Scale time constant inversely with damping ratio so effective stiffness
  # and static penetration remain approximately equal to v2 (.010 s, 1.0).
  contact_time_constant_s=0.010 / 1.5,
  contact_damping_ratio=1.5,
)

CONTACT_MODEL_VERSION = "poker-compliant-contact-v3"
# Distinct compliant interfaces: the felt support is softer than the tactile
# pads. The old near-rigid .98/.995 impedance made contact-manifold changes
# unload three fingers in a single 2 ms step. Preserve friction, margins,
# solref, masses and solver clocks; never change parameters mid-episode.
MID_TABLE_CONTACT_IMPEDANCE = (0.90, 0.95, 0.0005, 0.5, 2.0)
MID_CARD_CONTACT_IMPEDANCE = (0.95, 0.98, 0.0005, 0.5, 2.0)


def configure_poker_physics_integrity(
  simulation: MidForcePokerSimulation,
) -> dict[str, object]:
  """Use implicit damping for the light free card, without changing its physics.

  In MuJoCo 3.8.1 implicitfast's unconstrained free-body midpoint branch can
  explicitly integrate angular damping and amplify this card's angular speed.
  Keep the 2 ms clock, damping, mass, contacts and actuator gains unchanged.
  This override is poker-local, fixed before stepping, and part of the protocol.
  """
  original = mujoco.mjtIntegrator(simulation.model.opt.integrator).name
  simulation.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
  return {
    "physics_integrity_version": PHYSICS_INTEGRITY_VERSION,
    "physics_integrator_original": original,
    "physics_integrator_used": mujoco.mjtIntegrator(
      simulation.model.opt.integrator
    ).name,
    "mujoco_version": mujoco.__version__,
    "physics_integrity_checks": "every physical step: numerical warnings, finite state, exact clock advance",
    "physics_error_policy": "abort invalid trial; no auto-reset catch-up or state repair",
  }


def configure_middle_contact_impedance(
  simulation: MidForcePokerSimulation,
) -> dict[str, object]:
  """Apply and record only the two poker contact interfaces at construction."""
  from .friction import TABLE_CARD_PAIR_NAME

  model = simulation.model
  pair = model.pair(TABLE_CARD_PAIR_NAME).id
  card = model.geom("card_core_geom").id
  metadata = {
    "contact_model_version": CONTACT_MODEL_VERSION,
    "table_card_pair_solimp_original": model.pair_solimp[pair].tolist(),
    "card_geom_solimp_original": model.geom_solimp[card].tolist(),
  }
  model.pair_solimp[pair] = MID_TABLE_CONTACT_IMPEDANCE
  model.geom_solimp[card] = MID_CARD_CONTACT_IMPEDANCE
  metadata.update(
    table_card_pair_solimp_used=model.pair_solimp[pair].tolist(),
    card_geom_solimp_used=model.geom_solimp[card].tolist(),
    contact_impedance_scope="table-card pair and priority-selected card-pad contact; fixed for entire episode",
    tactile_temporal_smoothing=False,
  )
  return metadata


@contextmanager
def middle_force_simulation(
  model_path: str | Path,
  *,
  simulation_type: type[MidForcePokerSimulation] | None = None,
) -> Iterator[tuple[MidForcePokerSimulation, dict]]:
  """Reconstruct the accepted preset without changing the production scene.

  The include wrapper lives through recording. Runtime contact overrides are
  deliberately explicit because the wrapper hash alone cannot describe them.
  """
  from .friction import TABLE_CARD_PAIR_NAME, model_with_table_card_friction

  with model_with_table_card_friction(
    model_path, MID_FORCE_SETTINGS.table_friction
  ) as path:
    constructor = (
      MidForcePokerSimulation if simulation_type is None else simulation_type
    )
    sim = constructor(path, scene="poker-draw", add_genesis_probes=False)
    model = sim.model
    pair = model.pair(TABLE_CARD_PAIR_NAME).id
    card = model.geom("card_core_geom").id
    metadata = {
      "physics_timestep_original_s": float(model.opt.timestep),
      "contact_friction_impedance_ratio_original": float(model.opt.impratio),
      "table_card_pair_solref_original": model.pair_solref[pair].tolist(),
      "card_geom_solref_original": model.geom_solref[card].tolist(),
    }
    settings = MID_FORCE_SETTINGS
    if settings.physics_timestep_s is not None:
      model.opt.timestep = settings.physics_timestep_s
      sim.timestep = settings.physics_timestep_s
    model.opt.impratio = settings.contact_friction_impedance_ratio
    if settings.finger_servo_velocity_gain is not None:
      raise RuntimeError("middle-force-v1 does not modify finger servo gains")
    if settings.contact_time_constant_s is not None:
      model.pair_solref[pair, 0] = settings.contact_time_constant_s
      model.geom_solref[card, 0] = settings.contact_time_constant_s
    if settings.contact_damping_ratio is not None:
      model.pair_solref[pair, 1] = settings.contact_damping_ratio
      model.geom_solref[card, 1] = settings.contact_damping_ratio
    metadata.update(
      table_card_pair_friction=model.pair_friction[pair].tolist(),
      physics_timestep_used_s=float(model.opt.timestep),
      contact_friction_impedance_ratio_used=float(model.opt.impratio),
      table_card_pair_solref_used=model.pair_solref[pair].tolist(),
      card_geom_solref_used=model.geom_solref[card].tolist(),
      card_geom_friction=model.geom_friction[card].tolist(),
      card_geom_priority=int(model.geom_priority[card]),
      experimental_right_finger_servo_gains={},
      add_genesis_probes=False,
    )
    metadata.update(configure_middle_contact_impedance(sim))
    metadata.update(configure_poker_physics_integrity(sim))
    yield sim, metadata


class MidForcePokerSimulation(ForceLimitedPokerSimulation):
  """Allow an explicit, command-continuous transition after a qualified draw."""

  @property
  def observation_time(self) -> float:
    """Time of cached FK/render/solver data, not post-integration qpos."""
    return min(
      float(self.data.time), getattr(self, "_observation_time", float(self.data.time))
    )

  def step(self, steps: int = 1) -> None:
    for _ in range(max(1, int(steps))):
      check_physics_state(self.data, stage="before_step")
      evaluated_at = float(self.data.time)
      super().step(1)
      check_physics_state(self.data, stage="after_step")
      check_physics_clock(evaluated_at, float(self.data.time), timestep=self.timestep)
      self._observation_time = evaluated_at

  def end_cartesian_drive(self) -> dict[str, object]:
    if self.drive_limit_n is None or getattr(self, "_cartesian_drive", None) is None:
      raise RuntimeError("middle-pressure handover requires an active Cartesian drive")
    ids = self._arm_actuators["right"]
    control = self.data.ctrl[ids].copy()
    qpos_before = self.data.qpos.copy()
    qvel_before = self.data.qvel.copy()
    limits = self.model.actuator_ctrlrange[ids]
    if (
      not np.all(np.isfinite(control))
      or np.any(control < limits[:, 0])
      or np.any(control > limits[:, 1])
    ):
      raise RuntimeError("invalid bounded servo command at handover")
    outcome = {
      "time_s": float(self.data.time),
      "from": "bounded_cartesian_world_x",
      "to": "original_joint_position_servo",
      "previous_drive_limit_n": float(self.drive_limit_n),
      "previous_drive_state": self.drive_state(),
      "discarded_arm_goal_rad": self._arm_goal["right"].tolist(),
      "preserved_actuator_control_rad": control.tolist(),
      "control_jump_rad": 0.0,
      "object_state_modified": False,
      "physics_parameters_modified": False,
    }
    # Carry the actual executed control, NOT the stale IK target or measured q.
    # This preserves the settled PD preload and removes accumulated path error.
    self._arm_goal["right"] = control.copy()
    self._arm_command["right"] = control.copy()
    self.__dict__.pop("_cartesian_drive", None)
    self.__dict__.pop("_drive_state", None)
    self.drive_limit_n = None
    outcome.update(
      qpos_before=qpos_before.tolist(),
      qvel_before=qvel_before.tolist(),
      maximum_qpos_change=float(np.max(np.abs(self.data.qpos - qpos_before))),
      maximum_qvel_change=float(np.max(np.abs(self.data.qvel - qvel_before))),
      control_jump_rad=float(np.max(np.abs(self.data.ctrl[ids] - control))),
    )
    return outcome


class MidForcePokerExecutor(PressureWindowExecutor):
  """Reuse pickup/view paths and success gates, with load-based jaw feedback."""

  _pinch_force_targets_n = np.array([0.25, 0.25, 0.25, 0.25, 1.0])
  _pinch_force_integral_gain = 0.060
  _pinch_force_maximum_joint_rate_degrees_s = 1.00
  _pinch_force_maximum_offset_degrees = 0.75
  _prelift_settle_seconds = 0.45
  _transfer_lift_duration_seconds = 0.65
  _main_lift_duration_seconds = 1.25

  def __init__(
    self,
    simulation: MidForcePokerSimulation,
    *,
    acceptance_policy: str = STRICT_FORCE_POLICY,
    **kwargs: object,
  ) -> None:
    if not isinstance(simulation, MidForcePokerSimulation):
      raise TypeError("middle full task requires MidForcePokerSimulation")
    if "press_force_per_finger_n" in kwargs:
      raise ValueError("middle full preset fixes pressure to 0.50 N per finger")
    super().__init__(
      simulation, press_force_per_finger_n=MID_FORCE_PER_FINGER_N, **kwargs
    )
    self.acceptance_policy = check_policy(acceptance_policy)
    self._multi_low_load_duration_s = 0.0
    self._maximum_multi_low_load_duration_s = 0.0
    self.handoff_outcome: dict[str, object] | None = None
    self._lift_reference: tuple[np.ndarray, np.ndarray] | None = None
    self._lift_compensation: list[dict[str, object]] = []

  def control_metadata(self) -> dict[str, object]:
    return {
      **super().control_metadata(),
      "preset": "middle_force_full_task_v1",
      "contact_model_version": CONTACT_MODEL_VERSION,
      "motion_profile_version": "right-side-low-chatter-v4",
      "approach_path": "right-side Cartesian 55 mm lateral / 35 mm lift arc",
      "arm_interpolation": "physics-rate linear segments; minimum-jerk Cartesian path",
      "inspection_card_long_axis": "local Y upright; short edge down",
      "press_establish_controller": {
        "integral_gain_rad_per_n_s": 0.15,
        "maximum_joint_rate_deg_s": 2.5,
        "scope": "four_finger_press only; original slide force loop restored",
      },
      "acceptance_policy": getattr(self, "acceptance_policy", STRICT_FORCE_POLICY),
      "training_safety": {
        "minimum_loaded_fingers": MINIMUM_LOADED_FINGERS,
        "maximum_multi_finger_low_load_s": MAXIMUM_MULTI_FINGER_LOW_LOAD_S,
        "maximum_observed_multi_low_load_s": getattr(
          self, "_maximum_multi_low_load_duration_s", 0.0
        ),
        "scope": "task-completion-v1 slide/edge only; other physical guards unchanged",
      },
      "object_motion_or_slip_used_for_control": True,
      "geometry_usage": (
        "draw endpoint, supported edge retreat, and measured card/tool lever-arm "
        "compensation of pickup waypoints; never object-state writes"
      ),
      "slip_feedback_used_for_pressure_adjustment": False,
      "raise_card_to_view_duration_scale": 1.0,
      "lift_card_duration_scale": 1.0,
      "turn_card_inward_duration_scale": 1.0,
      "turn_card_inward_duration_s": 2.4,
      "turn_card_inward_path": "guarded Cartesian interpolation from measured grasp",
      "pickup_controller": "bounded tactile normal-force pinch v2",
      "pinch_force_control": {
        "target_normal_n": [0.25, 0.25, 0.25, 0.25, 1.0],
        "finger_order": [*_FINGERS, "thumb"],
        "update_period_s": 0.010,
        "feedback_filter_s": 0.020,
        "integral_gain_rad_per_n_s": 0.060,
        "maximum_joint_rate_deg_s": 1.00,
        "maximum_offset_deg": 0.75,
        "prelift_settle_s": 0.45,
        "upper_finger_preload_deg": 0.15,
        "raw_tactile_filtered": False,
      },
      "force_limit_scope": "slide_card and edge_hold; disabled by explicit handover",
      "contact_model_scope": "unchanged throughout the whole episode",
      "lift_waypoint_compensation": self._lift_compensation,
    }

  def _stabilize_four_finger_press(self, plan: PokerDrawPlan) -> None:
    # Establish preload faster without changing the pressure target, the
    # contact debounce, or the normal-force controller used during sliding.
    controller = self._press_controller
    gain, rate = controller.integral_gain, controller.maximum_offset_rate
    controller.integral_gain = 0.15
    controller.maximum_offset_rate = np.deg2rad(2.5)
    try:
      super()._stabilize_four_finger_press(plan)
    finally:
      controller.integral_gain, controller.maximum_offset_rate = gain, rate

  def _prepare_and_slide(
    self, plan: PokerDrawPlan, phases: list[str]
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    self.handoff_outcome = None
    self._lift_reference = None
    self._lift_compensation = []
    self._multi_low_load_duration_s = 0.0
    self._maximum_multi_low_load_duration_s = 0.0
    outcome = self.draw(plan, MID_FORCE_SETTINGS)
    policy = getattr(self, "acceptance_policy", STRICT_FORCE_POLICY)
    if not accept_edge(outcome, policy):
      raise RuntimeError("middle-pressure draw did not qualify; pickup is prohibited")
    phases.extend(waypoint.phase for waypoint in plan.waypoints)
    phases.extend(("four_finger_press", "slide_card", "edge_hold"))
    self.handoff_outcome = {"completed": False, "stage": "supported_edge_retreat"}
    # The original pickup releases/reforms the hand at the edge. Restore its
    # support margin by physically pushing the card inward, still at 4 N and
    # with the same per-finger Fn feedback. Never move the card state directly.
    self._supported_edge_retreat(plan)
    quality = self._slide_force_monitor.summary()
    if policy == STRICT_FORCE_POLICY and not self._slide_press_control_qualified(
      quality
    ):
      raise RuntimeError("middle-pressure edge handover lost continuous force quality")
    if (
      policy == TASK_COMPLETION_POLICY
      and self._maximum_slide_fingertip_plane_angle_degrees
      > _FLAT_DRAW_MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES
    ):
      raise RuntimeError("middle-pressure edge handover exceeded draw contact angle")
    edge_pose = self.sim.object_pose("card").copy()
    slide_position, _ = self.sim.current_pose_matrix(plan.side)
    transition = self.sim.end_cartesian_drive()
    self._press_controller_active = False
    self.handoff_outcome.update(
      completed=True,
      stage="joint_servo_ready",
      transition=transition,
      edge_card_pose=edge_pose.tolist(),
      slide_and_handoff_force_quality=asdict(quality),
      limit_scope="slide_card and edge_hold only; original servo during pickup/view",
    )
    return self.sim.arm_goal[plan.side], slide_position, edge_pose

  def _step(self, phase: str, table_edge_x: float) -> None:
    super()._step(phase, table_edge_x)
    if (
      getattr(self, "acceptance_policy", STRICT_FORCE_POLICY) != TASK_COMPLETION_POLICY
    ):
      return
    if phase not in {"slide_card", "edge_hold"}:
      self._multi_low_load_duration_s = 0.0
      return
    forces = self._current_card_finger_normal_forces()
    loaded = np.count_nonzero(
      forces >= self.press_force_per_finger_n * _PRESS_FORCE_CONTACT_RATIO
    )
    self._multi_low_load_duration_s = (
      self._multi_low_load_duration_s + self.sim.timestep
      if loaded < MINIMUM_LOADED_FINGERS
      else 0.0
    )
    self._maximum_multi_low_load_duration_s = max(
      self._maximum_multi_low_load_duration_s, self._multi_low_load_duration_s
    )
    if self._multi_low_load_duration_s >= MAXIMUM_MULTI_FINGER_LOW_LOAD_S - 1e-10:
      raise RuntimeError(
        "multiple draw fingers remained below load threshold for 0.30s"
      )

  def _supported_edge_retreat(self, plan: PokerDrawPlan) -> None:
    sim = self.sim
    start, _ = sim.current_pose_matrix(plan.side)
    commanded = 0.0
    # A tip-side draw stores more elastic deflection than the old flat pads.
    # At 800 N/m the unchanged 4 N budget can consume 5 mm of reference error
    # before the card moves; leave room for the ~1 mm supported-edge retreat.
    # This is a command bound, never extra force or a card-state correction.
    maximum_travel = 0.008
    target = 0.475
    started = float(sim.data.time)
    self._record_slide_force_metrics = True
    try:
      while self._overhang_fraction(plan.table_edge_x) > target:
        if float(sim.data.time) - started >= 2.0 or commanded >= maximum_travel:
          raise RuntimeError("card could not regain supported edge margin under 4 N")
        commanded += min(0.005 * sim.timestep, maximum_travel - commanded)
        reference = start.copy()
        reference[0] += commanded
        sim.set_cartesian_drive_position(reference)
        self._step("edge_hold", plan.table_edge_x)
      current, _ = sim.current_pose_matrix(plan.side)
      reference = start.copy()
      reference[0] = current[0]
      sim.set_cartesian_drive_position(reference)
      self._advance_fixed(0.10, "edge_hold", plan.table_edge_x)
      overhang = self._overhang_fraction(plan.table_edge_x)
      if not 0.44 <= overhang <= 0.48:
        raise RuntimeError("card left the supported edge handover geometry")
      self.handoff_outcome.update(
        retreat_duration_s=float(sim.data.time) - started,
        retreat_commanded_travel_m=commanded,
        overhang_fraction=overhang,
      )
    finally:
      self._record_slide_force_metrics = False

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
    if phase == "lift_card":
      # Tilting the wrist also translates a card ~15 cm ahead of the tool.
      # Compensate that lever arm so a 4/25 mm rise means a CARD rise, instead
      # of rotating the fingertips down into the still-supporting tabletop.
      current, rotation = self.sim.current_pose_matrix(plan.side)
      card = self.sim.object_pose("card")[:3].copy()
      if self._lift_reference is None:
        self._lift_reference = (current.copy(), card.copy())
      origin_tool, origin_card = self._lift_reference
      desired_card = origin_card + (position - origin_tool)
      local_offset = rotation.T @ (card - current)
      compensated = desired_card - end_effector_rotation @ local_offset
      if np.linalg.norm(compensated - position) > 0.04:
        raise RuntimeError("pickup lever-arm compensation exceeded 40 mm bound")
      self._lift_compensation.append(
        {
          "time_s": float(self.sim.data.time),
          "original_tool_target_m": np.asarray(position).tolist(),
          "compensated_tool_target_m": compensated.tolist(),
          "desired_card_center_m": desired_card.tolist(),
          "measured_local_card_offset_m": local_offset.tolist(),
        }
      )
      position = compensated
    # Both task entrances now share the smooth, contact-guarded timings.
    return super()._move_pose_with_guarded_pinch(
      plan,
      position,
      seed,
      duration=duration,
      phase=phase,
      end_effector_rotation=end_effector_rotation,
    )

  def _move_arm_joints_linear(
    self,
    plan: PokerDrawPlan,
    target: np.ndarray,
    seed: np.ndarray,
    *,
    duration: float,
    phase: str,
  ) -> np.ndarray:
    return super()._move_arm_joints_linear(
      plan,
      target,
      seed,
      duration=duration * (2.0 if phase == "turn_card_inward" else 1.0),
      phase=phase,
    )
