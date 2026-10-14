"""Opt-in pressure-window experiment; never changes the production controller.

The same seven-joint arm and four real fingertips are used. Only the active
right-arm servo wrench along world X is limited during the draw. No object
force, weld, or prescribed card trajectory is used. The table-edge mode uses
measured card geometry to slow/stop the hand, never slip-based pressure changes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from kaihand_tactile_env.shared.config import SIDES
from kaihand_tactile_env.shared.simulation import (
  ArmHandSimulation,
  _rotation_vector_world,
)

from .acceptance import MINIMUM_LOADED_FINGERS, TASK_COMPLETION_POLICY
from .config import (
  _FLAT_DRAW_MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES,
  _FLAT_DRAW_POSTURE_VERSION,
  _FLAT_DRAW_PRECONTACT_OFFSET,
)
from .press_control import FourFingerForceController
from .task import PokerDrawExecutor, PokerDrawPlan

_DRIVE_STIFFNESS = (800.0, 3000.0, 8000.0, 80.0, 80.0, 80.0)
_DRIVE_DAMPING = (80.0, 120.0, 200.0, 8.0, 8.0, 8.0)


def required_edge_travel_m(
  card_x: float,
  table_edge_x: float,
  projected_length: float,
  target_fraction: float = 0.49,
) -> float:
  """Unclipped geometry: initial distance includes travel before protrusion."""
  values = np.asarray([card_x, table_edge_x, projected_length, target_fraction])
  if (
    not np.all(np.isfinite(values))
    or projected_length <= 0
    or not 0 < target_fraction <= 0.49
  ):
    raise ValueError("finite geometry, positive length and target in (0, .49] required")
  target_x = table_edge_x + (0.5 - target_fraction) * projected_length
  return max(0.0, float(card_x - target_x))


def ideal_pressure_window(
  table_mu: float,
  finger_mu: float,
  mass_kg: float,
  gravity: float,
  drive_limit_n: float,
) -> tuple[float, float]:
  """Ideal TOTAL normal-force interval, ignoring arm losses and transients."""
  values = np.asarray([table_mu, finger_mu, mass_kg, gravity, drive_limit_n])
  if not np.all(np.isfinite(values)) or table_mu < 0 or np.any(values[1:] <= 0):
    raise ValueError(
      "finite nonnegative table friction and positive other values required"
    )
  if table_mu == 0:
    return 0.0, float("inf")
  if finger_mu <= table_mu:
    return float("inf"), float("-inf")
  weight = mass_kg * gravity
  lower = table_mu * weight / (finger_mu - table_mu)
  upper = drive_limit_n / table_mu - weight
  return (lower, upper) if upper > lower else (float("inf"), float("-inf"))


def limit_horizontal_wrench(
  jacobian: np.ndarray, torque: np.ndarray, limit_n: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Limit only Fx in a full-row-rank world-frame arm Jacobian.

  J maps joint velocity to [linear, angular] tool velocity. The torque
  nullspace and the other five wrench components are preserved exactly.
  This is a servo-wrench budget, NOT a bound on transient contact impulses.
  """
  jacobian = np.asarray(jacobian, dtype=float)
  torque = np.asarray(torque, dtype=float)
  if (
    jacobian.shape != (6, 7)
    or torque.shape != (7,)
    or not np.all(np.isfinite(jacobian))
    or not np.all(np.isfinite(torque))
    or not np.isfinite(limit_n)
    or limit_n <= 0
  ):
    raise ValueError("finite 6x7 Jacobian, seven torques and positive limit required")
  if np.linalg.cond(jacobian) > 1e6:
    raise ValueError(
      "arm Jacobian is singular or too ill-conditioned for a force budget"
    )
  requested = np.linalg.solve(jacobian @ jacobian.T, jacobian @ torque)
  limited = requested.copy()
  limited[0] = np.clip(limited[0], -limit_n, limit_n)
  limited_torque = torque + jacobian[0] * (limited[0] - requested[0])
  return limited_torque, requested, limited


@dataclass(frozen=True)
class PressureWindowSettings:
  table_friction: float = 1.30
  drive_limit_n: float = 4.0
  slide_distance_m: float = 0.030
  slide_speed_m_s: float = 0.005
  hold_seconds: float = 0.5
  contact_time_constant_s: float | None = 0.010
  contact_friction_impedance_ratio: float = 100.0
  finger_servo_velocity_gain: float | None = None
  physics_timestep_s: float | None = None
  goal: str = "short"
  max_slide_time_s: float = 30.0
  max_slide_travel_m: float = 0.16
  target_overhang_fraction: float = 0.49
  edge_dwell_s: float = 0.10
  contact_damping_ratio: float | None = None

  def __post_init__(self) -> None:
    if not isinstance(self.goal, str) or self.goal not in {"short", "table-edge"}:
      raise ValueError("goal must be short or table-edge")
    for name, value in asdict(self).items():
      if name == "goal":
        continue
      if value is None and name in {
        "contact_time_constant_s",
        "contact_damping_ratio",
        "finger_servo_velocity_gain",
        "physics_timestep_s",
      }:
        continue
      if value is None or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    if (
      self.physics_timestep_s is not None
      and not 0.0005 <= self.physics_timestep_s <= 0.002
    ):
      raise ValueError("experimental timestep must be between 0.0005 and 0.002 s")
    if (
      self.max_slide_time_s > 45.0
      or self.max_slide_travel_m > 0.20
      or self.target_overhang_fraction > 0.49
      or self.edge_dwell_s > 1.0
    ):
      raise ValueError(
        "edge bounds require <=45 s, <=200 mm, <=.49 overhang, <=1 s dwell"
      )
    if (
      self.slide_distance_m > 0.060 or self.slide_distance_m / self.slide_speed_m_s > 12
    ):
      raise ValueError(
        "bounded calibration requires <=60 mm and <=12 s commanded slide"
      )


class ForceLimitedPokerSimulation(ArmHandSimulation):
  """Experiment-local step variant; inactive stepping calls the original code."""

  drive_limit_n: float | None = None

  def reset(self, *args: object, **kwargs: object) -> None:
    # Reset is the only automatic release of the budget. Releasing it while a
    # stalled arm has a position error could cause an unintended strong pull.
    self.drive_limit_n = None
    self.__dict__.pop("_drive_state", None)
    self.__dict__.pop("_cartesian_drive", None)
    return super().reset(*args, **kwargs)

  def begin_cartesian_drive(self, limit_n: float) -> np.ndarray:
    """Decouple pose holding from a potentially large tangential goal error.

    Keeping the other five components of a *joint* PD wrench is insufficient:
    its off-diagonal Cartesian stiffness can turn an X tracking error into
    lift/rotation. Use independent task-space pose errors after switching.
    Preserve the settled preload wrench for a bumpless handover.
    """
    position, rotation = self.current_pose_matrix("right")
    self._cartesian_drive = {
      "position": position.copy(),
      "rotation": rotation.copy(),
      "preload": None,
    }
    self.drive_limit_n = float(limit_n)
    return position.copy()

  def set_cartesian_drive_position(self, position: np.ndarray) -> None:
    self._cartesian_drive["position"] = np.asarray(position, dtype=float).copy()

  def drive_state(self) -> dict[str, float]:
    return getattr(
      self,
      "_drive_state",
      {
        "drive_requested_fx_n": 0.0,
        "drive_limited_fx_n": 0.0,
        "drive_actual_fx_n": 0.0,
        "drive_saturated": 0.0,
        "drive_cap_error_n": 0.0,
        "drive_jacobian_condition": 0.0,
      },
    ).copy()

  def step(self, steps: int = 1) -> None:
    if self.drive_limit_n is None:
      return super().step(steps)
    for _ in range(max(1, int(steps))):
      # Same bias feedforward, goal slew and finger servos as shared.step.
      self.data.qfrc_applied.fill(0.0)
      for side in SIDES:
        dofs = self._arm_dofs[side]
        self.data.qfrc_applied[dofs] = self.data.qfrc_bias[dofs]
        error = self._arm_goal[side] - self._arm_command[side]
        delta = self.arm_speed_limit * self.timestep
        self._arm_command[side] += np.clip(error, -delta, delta)
        self.data.ctrl[self._arm_actuators[side]] = self._arm_command[side]
        for name, target in self._hand_targets[side].items():
          actuator = self._hand_actuators[side][name]
          velocity = self.hand_position_gain * (
            target - self.data.qpos[self._qpos_address[name]]
          )
          self.data.ctrl[actuator] = np.clip(
            velocity, *self.model.actuator_ctrlrange[actuator]
          )

      ids = self._arm_actuators["right"]
      dofs = self._arm_dofs["right"]
      kp = self.model.actuator_gainprm[ids, 0]
      kv = -self.model.actuator_biasprm[ids, 2]
      if not np.all(kp > 0) or not np.allclose(self.model.actuator_gear[ids, 0], 1):
        raise RuntimeError("force budget requires unit-gear position servos")
      q = self.data.qpos[self._arm_qpos["right"]]
      velocity = self.data.qvel[dofs]
      requested_torque = kp * (self.data.ctrl[ids] - q) - kv * velocity

      # Compute current kinematics on scratch data, never forward live contacts.
      self.ik_data.qpos[:] = self.data.qpos
      mujoco.mj_kinematics(self.model, self.ik_data)
      mujoco.mj_comPos(self.model, self.ik_data)
      jacp = np.zeros((3, self.model.nv))
      jacr = np.zeros_like(jacp)
      mujoco.mj_jacSite(self.model, self.ik_data, jacp, jacr, self._site_id["right"])
      jac = np.vstack((jacp[:, dofs], jacr[:, dofs]))
      cartesian = getattr(self, "_cartesian_drive", None)
      if cartesian is not None:
        baseline = np.linalg.solve(jac @ jac.T, jac @ requested_torque)
        if cartesian["preload"] is None:
          cartesian["preload"] = baseline.copy()
          cartesian["preload"][0] = 0.0
        site = self._site_id["right"]
        current = self.ik_data.site_xpos[site]
        rotation = self.ik_data.site_xmat[site].reshape(3, 3)
        error = np.concatenate(
          (
            cartesian["position"] - current,
            _rotation_vector_world(cartesian["rotation"], rotation),
          )
        )
        task_velocity = jac @ velocity
        stiffness = np.array(_DRIVE_STIFFNESS)
        damping = np.array(_DRIVE_DAMPING)
        wrench = cartesian["preload"] + stiffness * error - damping * task_velocity
        null_torque = requested_torque - jac.T @ baseline
        requested_torque = jac.T @ wrench + null_torque
      torque, requested, limited = limit_horizontal_wrench(
        jac, requested_torque, self.drive_limit_n
      )
      control = q + (torque + kv * velocity) / kp
      if np.any(control < self.model.actuator_ctrlrange[ids, 0]) or np.any(
        control > self.model.actuator_ctrlrange[ids, 1]
      ):
        raise RuntimeError(
          "force-budget control would exceed original joint command limits"
        )
      if np.any(torque < self.model.actuator_forcerange[ids, 0]) or np.any(
        torque > self.model.actuator_forcerange[ids, 1]
      ):
        raise RuntimeError("force-budget torque would exceed original actuator limits")
      self.data.ctrl[ids] = control
      mujoco.mj_step(self.model, self.data)
      actual = np.linalg.solve(jac @ jac.T, jac @ self.data.qfrc_actuator[dofs])
      cap_error = max(0.0, abs(float(actual[0])) - self.drive_limit_n)
      self._drive_state = {
        "drive_requested_fx_n": float(requested[0]),
        "drive_limited_fx_n": float(limited[0]),
        "drive_actual_fx_n": float(actual[0]),
        "drive_saturated": float(abs(requested[0]) > self.drive_limit_n),
        "drive_cap_error_n": cap_error,
        "drive_jacobian_condition": float(np.linalg.cond(jac)),
        "drive_ee_y_m": float(self.ik_data.site_xpos[self._site_id["right"], 1]),
        "drive_ee_z_m": float(self.ik_data.site_xpos[self._site_id["right"], 2]),
        "drive_pose_error_y_m": float(error[1]) if cartesian is not None else 0.0,
        "drive_pose_error_z_m": float(error[2]) if cartesian is not None else 0.0,
        "drive_rotation_error_rad": float(np.linalg.norm(error[3:]))
        if cartesian is not None
        else 0.0,
      }
      for index, component in enumerate(("fx", "fy", "fz", "tx", "ty", "tz")):
        self._drive_state[
          f"drive_requested_{component}_n"
          if index < 3
          else f"drive_requested_{component}_nm"
        ] = float(requested[index])
        self._drive_state[
          f"drive_actual_{component}_n" if index < 3 else f"drive_actual_{component}_nm"
        ] = float(actual[index])
      if cap_error > 1e-5:
        raise RuntimeError(
          f"actual arm servo Fx exceeded requested budget by {cap_error:.6g} N"
        )


class PressureWindowForceController(FourFingerForceController):
  """Filtered force servo without unconditional positive recontact increments.

  At small targets, repeatedly seeking contact despite an above-target filtered
  force creates a positive preload bias. Use the signed measured force error
  for both release and approach instead. This controller is experiment-only.
  """

  def observe(
    self, forces_n: np.ndarray | tuple[float, ...]
  ) -> tuple[np.ndarray, bool]:
    forces = self._validated_forces(forces_n)
    self.filtered_forces_n += self.filter_alpha * (forces - self.filtered_forces_n)
    self._steps_since_update += 1
    if self._steps_since_update < self.update_steps:
      return self.offsets_rad.copy(), False
    elapsed = self._steps_since_update * self.timestep
    self._steps_since_update = 0
    error = self.target_force_n - self.filtered_forces_n
    error = np.sign(error) * np.maximum(np.abs(error) - self.force_deadband_n, 0.0)
    delta = np.clip(
      self.integral_gain * error * elapsed,
      -self.maximum_offset_rate * elapsed,
      self.maximum_offset_rate * elapsed,
    )
    self.offsets_rad = np.clip(
      self.offsets_rad + delta, -self.maximum_offset, self.maximum_offset
    )
    return self.offsets_rad.copy(), True


class PressureWindowExecutor(PokerDrawExecutor):
  """Bounded continuous-drive test with normal feedback but no recovery pause.

  The production pause-on-low-Fn policy would stop a light-pressure failure
  before its natural slip can be observed. This distinct protocol never calls
  the production success gate and never labels low-pressure pause as a stall.
  """

  def __init__(self, simulation: ArmHandSimulation, **kwargs: object) -> None:
    super().__init__(simulation, **kwargs)
    self.edge_outcome: dict[str, object] | None = None
    self._press_controller = PressureWindowForceController(
      4,
      target_force_n=self.press_force_per_finger_n,
      timestep=simulation.timestep,
      update_period_s=simulation.timestep,
      filter_time_constant_s=0.010,
      integral_gain_rad_per_n_s=0.10,
      maximum_offset_rad=np.deg2rad(5.0),
      maximum_offset_rate_rad_s=np.deg2rad(1.5),
      contact_force_n=self.press_force_per_finger_n * 0.25,
      contact_recovery_rate_rad_s=0.01,  # unused by this signed-error controller
      force_deadband_n=min(0.005, self.press_force_per_finger_n * 0.05),
    )

  def control_metadata(self) -> dict[str, object]:
    controller = self._press_controller
    return {
      "normal_controller": type(controller).__name__,
      "normal_feedback": "per-finger measured Fn; signed filtered error integral",
      "target_force_per_finger_n": controller.target_force_n,
      "timestep_s": controller.timestep,
      "update_steps": controller.update_steps,
      "filter_alpha": controller.filter_alpha,
      "integral_gain_rad_per_n_s": controller.integral_gain,
      "maximum_offset_rad": controller.maximum_offset,
      "maximum_offset_rate_rad_s": controller.maximum_offset_rate,
      "force_deadband_n": controller.force_deadband_n,
      "unconditional_contact_seek": False,
      "pause_on_low_normal_force": False,
      "establish_tolerance_n": max(0.005, 0.10 * self.press_force_per_finger_n),
      "establish_stable_duration_s": 0.10,
      "establish_timeout_s": 12.0,
      "cartesian_axis_order": ["x", "y", "z", "rx", "ry", "rz"],
      "cartesian_stiffness_n_per_m_or_nm_per_rad": _DRIVE_STIFFNESS,
      "cartesian_damping_ns_per_m_or_nms_per_rad": _DRIVE_DAMPING,
      "object_motion_or_slip_used_for_control": False,
    }

  def _stabilize_four_finger_press(self, plan: PokerDrawPlan) -> None:
    tolerance = max(0.005, 0.10 * self.press_force_per_finger_n)
    stable = 0
    accumulated = np.zeros(4)
    required = max(1, int(round(0.10 / self.sim.timestep)))
    for _ in range(int(round(12.0 / self.sim.timestep))):
      self._step("four_finger_press", plan.table_edge_x)
      forces = self._current_card_finger_normal_forces()
      ready = np.all(forces >= self.press_force_per_finger_n * 0.25) and np.all(
        np.abs(self._press_controller.filtered_forces_n - self.press_force_per_finger_n)
        <= tolerance
      )
      if ready:
        stable += 1
        accumulated += forces
      else:
        stable = 0
        accumulated.fill(0.0)
      if stable >= required:
        self._established_press_normal_forces_n = tuple(accumulated / stable)
        return
    raise RuntimeError(
      "experimental relative-tolerance normal pressure did not stabilize"
    )

  def prepare(self, plan: PokerDrawPlan) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    self._validate_plan(plan)
    self._reset_episode_state(plan)
    seed = self._execute_approach(plan)
    seed = self._move_pose_linear(
      plan,
      self.sim.object_pose("card")[:3] + _FLAT_DRAW_PRECONTACT_OFFSET,
      seed,
      duration=0.24,
      phase="precontact_card",
    )
    self._press_until_four_contacts(plan, seed)
    position, rotation = self._arm_goal_pose_matrix(plan.side)
    return self.sim.arm_goal[plan.side], position, rotation

  def draw(
    self, plan: PokerDrawPlan, settings: PressureWindowSettings
  ) -> dict[str, object] | None:
    self.edge_outcome = None
    if settings.goal == "table-edge":
      return self._draw_to_edge(plan, settings)
    seed, start, rotation = self.prepare(plan)
    sim = self.sim
    if not isinstance(sim, ForceLimitedPokerSimulation):
      raise TypeError("pressure-window draw requires ForceLimitedPokerSimulation")
    actual_start = sim.begin_cartesian_drive(settings.drive_limit_n)
    self._press_slide_reference_z = float(start[2])
    self._record_slide_force_metrics = True
    duration = settings.slide_distance_m / settings.slide_speed_m_s
    segments = max(1, int(np.ceil(duration / 0.02)))
    steps_per_segment = max(1, int(round(duration / segments / sim.timestep)))
    try:
      for segment in range(segments):
        fraction = (segment + 1) / segments
        # Smooth start/end, fixed path for all pressure conditions.
        progress = fraction * fraction * (3 - 2 * fraction)
        position = start.copy()
        position[0] -= settings.slide_distance_m * progress
        result = sim.solve_ik(
          "right",
          position,
          rotation,
          seed=seed,
          max_iterations=200,
          position_tolerance=0.00002,
          orientation_tolerance=0.001,
          posture_weight=0.001,
        )
        if not result.success:
          raise RuntimeError("pressure-window slide IK failed")
        begin = sim.arm_goal["right"]
        for step in range(steps_per_segment):
          previous_fraction = segment / segments
          previous_progress = previous_fraction**2 * (3 - 2 * previous_fraction)
          interpolated_progress = previous_progress + (progress - previous_progress) * (
            (step + 1) / steps_per_segment
          )
          drive_position = actual_start.copy()
          drive_position[0] -= settings.slide_distance_m * interpolated_progress
          sim.set_cartesian_drive_position(drive_position)
          sim.set_arm_joint_goal(
            "right",
            begin + (result.joint_positions - begin) * ((step + 1) / steps_per_segment),
          )
          self._step("slide_card", plan.table_edge_x)
        seed = result.joint_positions
      self._advance_fixed(settings.hold_seconds, "edge_hold", plan.table_edge_x)
    finally:
      # Keep the budget latched for any post-trial observation steps. The next
      # explicit reset disables it before resetting state and actuator goals.
      self._record_slide_force_metrics = False

  def _draw_to_edge(
    self, plan: PokerDrawPlan, settings: PressureWindowSettings
  ) -> dict[str, object]:
    """Physically drag to the measured near edge; bounds are failures, not goals.

    The same speed and pressure controller applies to every pressure trial.
    Geometry only slows/stops the path near its endpoint; neither friction nor
    normal-force targets adapt to slip, and the free card is never driven.
    """
    self.edge_outcome = {
      "target_reached": False,
      "terminal_reason": "preparing",
      "full_slide_qualified": False,
    }
    try:
      seed, start, rotation = self.prepare(plan)
    except RuntimeError:
      self.edge_outcome["terminal_reason"] = "prepare_error"
      raise
    sim = self.sim
    if not isinstance(sim, ForceLimitedPokerSimulation):
      raise TypeError("pressure-window draw requires ForceLimitedPokerSimulation")
    actual_start = sim.begin_cartesian_drive(settings.drive_limit_n)
    initial_card_x = float(sim.object_pose("card")[0])
    required_distance = required_edge_travel_m(
      initial_card_x,
      plan.table_edge_x,
      self._projected_card_length_x(),
      settings.target_overhang_fraction,
    )
    self._press_slide_reference_z = float(start[2])
    self._slide_force_monitor.reset()
    self._record_slide_force_metrics = True
    started = float(sim.data.time)
    commanded = 0.0
    terminal_reason = "time_limit"
    reached = False
    geometric_dwell = 0.0
    strict_dwell = 0.0
    supported_hold_samples = 0
    geometric_hold_samples = 0
    minimum_loaded_fingers = (
      MINIMUM_LOADED_FINGERS
      if getattr(self, "acceptance_policy", None) == TASK_COMPLETION_POLICY
      else 4
    )
    all_unloaded_duration = 0.0
    # A diagnostic early stop requires a full second of measured near-stall,
    # saturated drive and continuous four-finger load, never a pressure label.
    stall_window: deque[tuple[float, float, float, float, bool]] = deque()
    try:
      while float(sim.data.time) - started < settings.max_slide_time_s:
        overhang = self._overhang_fraction(plan.table_edge_x)
        if overhang >= settings.target_overhang_fraction:
          reached, terminal_reason = True, "edge_reached"
          break
        if commanded >= settings.max_slide_travel_m - 1e-10:
          terminal_reason = "travel_limit"
          break
        elapsed = float(sim.data.time) - started
        remaining = required_edge_travel_m(
          float(sim.object_pose("card")[0]),
          plan.table_edge_x,
          self._projected_card_length_x(),
          settings.target_overhang_fraction,
        )
        # 0.5 s acceleration, with slow final 5 mm to retain table support.
        velocity = settings.slide_speed_m_s * min(1.0, (elapsed + 0.02) / 0.5)
        velocity *= min(1.0, max(0.10, remaining / 0.005))
        segment_steps = min(
          max(1, int(round(0.020 / sim.timestep))),
          max(1, int(np.ceil((settings.max_slide_time_s - elapsed) / sim.timestep))),
        )
        segment_delta = min(
          velocity * segment_steps * sim.timestep,
          settings.max_slide_travel_m - commanded,
        )
        endpoint = start.copy()
        endpoint[0] -= commanded + segment_delta
        ik = sim.solve_ik(
          "right",
          endpoint,
          rotation,
          seed=seed,
          max_iterations=200,
          position_tolerance=0.00002,
          orientation_tolerance=0.001,
          posture_weight=0.001,
        )
        if not ik.success:
          raise RuntimeError("pressure-window table-edge IK failed")
        begin = sim.arm_goal["right"]
        segment_start = commanded
        stop = False
        for step in range(segment_steps):
          fraction = (step + 1) / segment_steps
          commanded = segment_start + segment_delta * fraction
          reference = actual_start.copy()
          reference[0] -= commanded
          sim.set_cartesian_drive_position(reference)
          sim.set_arm_joint_goal(
            "right", begin + (ik.joint_positions - begin) * fraction
          )
          self._step("slide_card", plan.table_edge_x)
          if (
            self._overhang_fraction(plan.table_edge_x)
            >= settings.target_overhang_fraction
          ):
            reached, terminal_reason, stop = True, "edge_reached", True
            break
          forces = self._current_card_finger_normal_forces()
          all_unloaded_duration = (
            all_unloaded_duration + sim.timestep if np.all(forces <= 1e-5) else 0.0
          )
          if all_unloaded_duration >= 0.30:
            terminal_reason, stop = "contact_lost", True
            break
          now = float(sim.data.time)
          ee_position, _ = sim.current_pose_matrix("right")
          stall_window.append(
            (
              now,
              float(sim.object_pose("card")[0]),
              float(ee_position[0]),
              sim.drive_state()["drive_saturated"],
              bool(np.all(forces >= 0.25 * self.press_force_per_finger_n)),
            )
          )
          while len(stall_window) > 1 and stall_window[1][0] <= now - 1.0:
            stall_window.popleft()
          if stall_window and now - stall_window[0][0] >= 1.0 - sim.timestep * 0.1:
            window_time = now - stall_window[0][0]
            card_speed = abs(stall_window[-1][1] - stall_window[0][1]) / window_time
            hand_speed = abs(stall_window[-1][2] - stall_window[0][2]) / window_time
            if (
              card_speed < 0.0005
              and hand_speed < 0.0005
              and all(item[4] for item in stall_window)
              and sum(item[3] for item in stall_window) / len(stall_window) >= 0.95
            ):
              terminal_reason, stop = "sustained_drive_limit", True
              break
        seed = ik.joint_positions
        if stop:
          break
      if reached:
        # Freeze the actual hand X only, retaining orientation/height reference
        # and the budget. Do not release the cap or teleport/attach the card.
        current, _ = sim.current_pose_matrix("right")
        reference = actual_start.copy()
        reference[0] = current[0]
        sim.set_cartesian_drive_position(reference)
      hold_phase = "edge_hold" if reached else "final_hold"
      for _ in range(max(1, int(round(settings.hold_seconds / sim.timestep)))):
        self._step(hold_phase, plan.table_edge_x)
        forces = self._current_card_finger_normal_forces()
        geometry_held = bool(
          reached
          and self._overhang_fraction(plan.table_edge_x)
          >= settings.target_overhang_fraction - 0.015
        )
        loaded_fingers = int(
          np.count_nonzero(forces >= 0.25 * self.press_force_per_finger_n)
        )
        if geometry_held:
          geometric_dwell += sim.timestep
          geometric_hold_samples += 1
          supported_hold_samples += int(loaded_fingers >= minimum_loaded_fingers)
        else:
          geometric_dwell = 0.0
          geometric_hold_samples = 0
          supported_hold_samples = 0
        strict_held = bool(
          geometry_held and np.all(forces >= 0.25 * self.press_force_per_finger_n)
        )
        strict_dwell = strict_dwell + sim.timestep if strict_held else 0.0
    except (RuntimeError, ValueError):
      terminal_reason = "control_error"
      raise
    finally:
      summary = self._slide_force_monitor.summary()
      quality = self._slide_press_control_qualified(summary)
      support_fraction = (
        supported_hold_samples / geometric_hold_samples
        if geometric_hold_samples
        else 0.0
      )
      task_completion_dwell = bool(
        geometric_dwell >= settings.edge_dwell_s - 1e-9 and support_fraction >= 0.90
      )
      strict_completion_dwell = bool(strict_dwell >= settings.edge_dwell_s - 1e-9)
      policy_completion_dwell = (
        task_completion_dwell
        if getattr(self, "acceptance_policy", None) == TASK_COMPLETION_POLICY
        else strict_completion_dwell
      )
      accepted_dwell = geometric_dwell if policy_completion_dwell else 0.0
      self.edge_outcome = {
        "draw_posture_version": _FLAT_DRAW_POSTURE_VERSION,
        "slide_fingertip_plane_angle_limit_degrees": _FLAT_DRAW_MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES,
        "target_reached": reached,
        "held_at_edge": bool(reached and policy_completion_dwell),
        "terminal_reason": terminal_reason,
        "target_overhang_fraction": settings.target_overhang_fraction,
        "table_edge_x_m": float(plan.table_edge_x),
        "required_card_travel_m": required_distance,
        "commanded_travel_m": commanded,
        "actual_card_travel_m": initial_card_x - float(sim.object_pose("card")[0]),
        "final_overhang_fraction": self._overhang_fraction(plan.table_edge_x),
        "maximum_overhang_fraction": self._maximum_overhang_fraction,
        "edge_dwell_s": accepted_dwell,
        "geometric_edge_dwell_s": geometric_dwell,
        "strict_four_finger_edge_dwell_s": strict_dwell,
        "edge_support_minimum_loaded_fingers": minimum_loaded_fingers,
        "edge_support_sample_fraction": support_fraction,
        "edge_support_minimum_sample_fraction": 0.90,
        "slide_force_quality": asdict(summary),
        "full_slide_qualified": bool(
          reached
          and strict_dwell >= settings.edge_dwell_s - 1e-9
          and quality
          and terminal_reason == "edge_reached"
        ),
        "slide_geometry_qualified": bool(
          self._maximum_slide_fingertip_plane_angle_degrees
          <= _FLAT_DRAW_MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES
        ),
        "slide_task_completed": bool(
          reached
          and policy_completion_dwell
          and terminal_reason == "edge_reached"
          and self._maximum_slide_fingertip_plane_angle_degrees
          <= _FLAT_DRAW_MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES
        ),
        "geometry_feedback": "measured card geometry only for endpoint speed, stop and edge dwell; no slip-based pressure adjustment",
        "safety_stops": {
          "all_finger_contact_loss_s": 0.30,
          "near_stall_window_s": 1.0,
          "near_stall_speed_m_s": 0.0005,
          "minimum_drive_saturated_fraction": 0.95,
        },
      }
      self._record_slide_force_metrics = False
    return self.edge_outcome
