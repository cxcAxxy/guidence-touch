"""Known-state USB grasp and insertion using only the robot's actuators.

The policy reads simulated object pose as ground truth. Its baseline controller
supports optional precontact wrist noise with an independent seed. No object
pose, velocity, external wrench, collision mask or attachment constraint is
changed by execution.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable

import mujoco
import numpy as np

from kaihand_tactile_env.shared.config import FINGERTIP_LINK_NAMES
from kaihand_tactile_env.shared.simulation import ArmHandSimulation

from . import config
from .motion import MOTION_PROFILES
from .precontact_noise import NEAR_GRASP_SCALE, PrecontactMotionNoise
from .task import UsbInsertionMonitor, UsbInsertionState


@dataclass(frozen=True)
class UsbInsertionResult:
  success: bool
  phases: tuple[str, ...]
  failure_reason: str | None
  elapsed_simulation_s: float
  maximum_lift_m: float
  grasp_verified: bool
  released: bool
  active_bottom_out_confirmed: bool
  bottom_out_hold_s: float
  peak_backstop_axial_resistance_n: float
  peak_socket_normal_load_n: float
  peak_axial_resistance_n: float
  maximum_socket_penetration_m: float
  minimum_lift_grip_force_n: tuple[float, float]
  final_plug_pose_wxyz: tuple[float, ...]
  insertion: UsbInsertionState
  precontact_noise: dict
  motion_profile: str
  motion_parameters: dict


class _TaskFailure(RuntimeError):
  pass


def _rotation_vector_world(target: np.ndarray, current: np.ndarray) -> np.ndarray:
  """Quaternion logarithm keeps the relative axis signs correct at 180 degrees."""
  quaternion = np.empty(4)
  mujoco.mju_mat2Quat(quaternion, (target @ current.T).ravel())
  if quaternion[0] < 0:
    quaternion *= -1
  sine = float(np.linalg.norm(quaternion[1:]))
  if sine < 1e-12:
    return np.zeros(3)
  return quaternion[1:] * (2 * np.arctan2(sine, quaternion[0]) / sine)


def _rotation_step(vector: np.ndarray) -> np.ndarray:
  angle = float(np.linalg.norm(vector))
  if angle < 1e-12:
    return np.eye(3)
  quaternion = np.r_[np.cos(angle / 2), np.sin(angle / 2) * vector / angle]
  matrix = np.empty(9)
  mujoco.mju_quat2Mat(matrix, quaternion)
  return matrix.reshape(3, 3)


def _interpolate_rotation(
  start: np.ndarray, end: np.ndarray, alpha: float
) -> np.ndarray:
  return _rotation_step(alpha * _rotation_vector_world(end, start)) @ start


class UsbInsertionExecutor:
  """Execute one pose-aware right-hand episode; report physical failures explicitly."""

  def __init__(
    self,
    simulation: ArmHandSimulation,
    observer: Callable[[ArmHandSimulation, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    *,
    precontact_noise_std_m: float = 0.0,
    noise_seed: int | None = None,
    motion_profile: str = "fast",
  ) -> None:
    if simulation.scene != "usb-insert":
      raise ValueError("UsbInsertionExecutor requires scene='usb-insert'")
    self.sim = simulation
    if motion_profile not in MOTION_PROFILES:
      raise ValueError("USB motion profile must be 'fast' or 'baseline'")
    self.motion_profile = motion_profile
    self.motion = MOTION_PROFILES[motion_profile]
    self.observer = observer
    self.should_stop = should_stop
    self._noise = PrecontactMotionNoise(precontact_noise_std_m, noise_seed)
    self._noise_options = (precontact_noise_std_m, noise_seed)
    self._executed = False
    self._precontact_target = None
    self._next_noise_update_s = 0.0
    self.monitor = UsbInsertionMonitor(simulation)
    self._pad_ids = {
      simulation.model.geom(f"hand_r_{finger}_{link}_tactile_pad_col").id: i
      for i, (finger, link) in enumerate((("thumb", "link6"), ("index", "link4")))
    }
    self._tactile_pad_ids = {
      simulation.model.geom(f"{name}_tactile_pad_col").id: index
      for index, name in enumerate(FINGERTIP_LINK_NAMES)
    }
    self._tactile_forces = np.zeros(len(FINGERTIP_LINK_NAMES))
    plug_body = simulation.model.body("usb_plug").id
    self._plug_geoms = {
      i
      for i in range(simulation.model.ngeom)
      if int(simulation.model.geom_bodyid[i]) == plug_body
    }
    self._hand_names = simulation._hand_joint_names["right"]
    self._period_steps = max(1, round(0.02 / simulation.timestep))

  def _read_grip(self) -> np.ndarray:
    forces = np.zeros(2)
    self._tactile_forces[:] = 0.0
    wrench = np.empty(6)
    for i in range(self.sim.data.ncon):
      contact = self.sim.data.contact[i]
      g1, g2 = int(contact.geom1), int(contact.geom2)
      pad = g1 if g2 in self._plug_geoms else g2 if g1 in self._plug_geoms else None
      finger = self._pad_ids.get(pad)
      tactile_index = self._tactile_pad_ids.get(pad)
      if finger is not None or (self._noise.enabled and tactile_index is not None):
        mujoco.mj_contactForce(self.sim.model, self.sim.data, i, wrench)
        load = abs(float(wrench[0]))
        if finger is not None:
          forces[finger] += load
        if tactile_index is not None:
          self._tactile_forces[tactile_index] += load
    return forces

  def _step(self, phase: str, *, require_grip: bool = False) -> None:
    if self.should_stop is not None and self.should_stop():
      raise _TaskFailure("cancelled")
    if (
      self._precontact_target is not None
      and self.sim.data.time + 1e-10 >= self._next_noise_update_s
    ):
      position, rotation, scale, _ = self._precontact_target
      self._command_ee(position, rotation, precontact_scale=scale, noise_phase=phase)
    if not self._phases or self._phases[-1] != phase:
      self._phases.append(phase)
    if self._smooth_arm_target is not None:
      # Interpolate the robot command at the physics rate, not the force output.
      self._smooth_arm_elapsed = min(0.02, self._smooth_arm_elapsed + self.sim.timestep)
      blend = self._smooth_arm_elapsed / 0.02
      self.sim.set_arm_joint_goal(
        "right", (1 - blend) * self._smooth_arm_start + blend * self._smooth_arm_target
      )
    self.sim.step()
    if (
      not np.isfinite(self.sim.data.qpos).all()
      or not np.isfinite(self.sim.data.qvel).all()
    ):
      # Leave the last finite measurement available for a serializable failure.
      raise _TaskFailure("non-finite simulation state")
    self._state = self.monitor.update()
    if phase in {"bottom_out", "unload"}:
      self._bottom_force_window.append(self._state.backstop_axial_resistance_n)
    self._last_pose = self.sim.object_pose("usb_plug")
    self._grip = self._read_grip()
    if self._noise.enabled:
      finger = int(np.argmax(self._tactile_forces))
      latched = self._noise.latch_contact(
        float(self.sim.data.time),
        phase,
        f"{FINGERTIP_LINK_NAMES[finger]}_tactile_pad_col",
        self._tactile_forces[finger],
      )
      if latched and self._precontact_target is not None:
        # Stop randomness on this physics step. Keep the current target
        # continuous while deterministically blending its small bias to zero.
        position, rotation, scale, _ = self._precontact_target
        self._command_ee(position, rotation, precontact_scale=scale, noise_phase=phase)
    self._max_height = max(self._max_height, self.sim.object_pose("usb_plug")[2])
    self._peak_load = max(self._peak_load, self._state.socket_normal_load_n)
    self._peak_axial = max(self._peak_axial, self._state.axial_resistance_n)
    self._peak_backstop = max(
      self._peak_backstop, self._state.backstop_axial_resistance_n
    )
    self._max_penetration = max(
      self._max_penetration, self._state.maximum_socket_penetration_m
    )
    if self.observer is not None:
      self.observer(self.sim, phase)
    if require_grip:
      if phase == "lift":
        self._minimum_grip = np.minimum(self._minimum_grip, self._grip)
      self._grip_gap_s = (
        self._grip_gap_s + self.sim.timestep if np.min(self._grip) < 0.025 else 0.0
      )
      if self._grip_gap_s > 0.05:
        raise _TaskFailure(f"opposed fingertip contact lost during {phase}")
    else:
      self._grip_gap_s = 0.0
    if self._state.socket_normal_load_n > config.MAX_SOCKET_NORMAL_LOAD_N:
      raise _TaskFailure(f"socket total normal load exceeded limit during {phase}")
    if self._state.axial_resistance_n > config.MAX_SOCKET_AXIAL_FORCE_N:
      raise _TaskFailure(f"socket axial resistance exceeded limit during {phase}")
    if self._state.wall_normal_load_n > config.MAX_SOCKET_WALL_LOAD_N:
      raise _TaskFailure(f"socket side-wall load exceeded limit during {phase}")
    if self._state.maximum_socket_penetration_m > 0.0003:
      raise _TaskFailure(f"socket penetration exceeded 0.3 mm during {phase}")

  def _advance(
    self, duration: float, phase: str, *, require_grip: bool = False
  ) -> None:
    for _ in range(max(1, round(duration / self.sim.timestep))):
      self._step(phase, require_grip=require_grip)

  def _command_ee(
    self,
    position: np.ndarray,
    rotation: np.ndarray,
    *,
    precontact_scale: float | None = None,
    noise_phase: str | None = None,
    precise: bool = False,
  ) -> None:
    nominal_position = position.copy()
    noisy_command = self._noise.enabled and precontact_scale is not None
    if noisy_command:
      self._precontact_target = (
        nominal_position.copy(),
        rotation.copy(),
        precontact_scale,
        noise_phase,
      )
      offset = self._noise.sample(float(self.sim.data.time), precontact_scale)
      position = nominal_position + offset
      self._next_noise_update_s = (
        float(self.sim.data.time) + self._period_steps * self.sim.timestep
      )
      if self._noise.first_contact is not None and not np.any(offset):
        self._precontact_target = None
    else:
      self._precontact_target = None
    result = self.sim.solve_ik(
      "right",
      position,
      rotation,
      seed=self.sim.arm_goal["right"],
      max_iterations=400,
      position_tolerance=0.000001 if precise else 0.00002,
      orientation_tolerance=0.00005 if precise else 0.0003,
      posture_weight=0.0,
    )
    if not result.success:
      # Refine near the straight-arm region with lower numerical damping.
      # Keep the shared simulation's normal IK settings unchanged afterward.
      damping = self.sim.ik_damping
      try:
        self.sim.ik_damping = min(damping, 0.004)
        result = self.sim.solve_ik(
          "right",
          position,
          rotation,
          seed=result.joint_positions,
          max_iterations=400,
          position_tolerance=0.000001 if precise else 0.00002,
          orientation_tolerance=0.00005 if precise else 0.0003,
          posture_weight=0.0,
        )
      finally:
        self.sim.ik_damping = damping
    if not result.success:
      raise _TaskFailure(
        f"unreachable wrist target: position error {result.position_error:.6f} m, "
        f"orientation error {result.orientation_error:.5f} rad"
      )
    # The open insertion elbow is appropriate near the mouth. Applying it at
    # the 10 cm hover unnecessarily folds the wrist toward its joint limit.
    if (
      getattr(self, "_keep_elbow_open", False) and self._state.insertion_depth_m > -0.03
    ):
      result = self._open_elbow_solution(result, position, rotation, precise=precise)
    if (
      getattr(self, "_keep_elbow_open", False)
      and self._state.insertion_depth_m <= -0.03
    ):
      result = self._joint_margin_solution(result, position, rotation, precise=precise)
    if self._smooth_arm_enabled:
      # Respect the posture margin in the applied command. If constrained,
      # feed its actual FK back to the integrator to prevent windup.
      limits = self.sim.model.jnt_range[self.sim._arm_joint_ids["right"]]
      margin_deg = 10.01 if self._state.insertion_depth_m < -0.03 else 15.01
      bounded = np.clip(
        result.joint_positions,
        limits[:, 0] + np.deg2rad(margin_deg),
        limits[:, 1] - np.deg2rad(margin_deg),
      )
      if np.any(bounded != result.joint_positions):
        scratch = self.sim.ik_data
        scratch.qpos[self.sim._arm_qpos["right"]] = bounded
        mujoco.mj_kinematics(self.sim.model, scratch)
        site = self.sim.model.site("right_ee_site").id
        position = scratch.site_xpos[site].copy()
        rotation = scratch.site_xmat[site].reshape(3, 3).copy()
      self._smooth_arm_start = self.sim.arm_goal["right"].copy()
      self._smooth_arm_target = bounded.copy()
      self._smooth_arm_elapsed = 0.0
    else:
      self.sim.set_arm_joint_goal("right", result.joint_positions)
    self._ee_command_position = position.copy()
    self._ee_command_rotation = rotation.copy()
    if noisy_command:
      self._noise.commands.append(
        {
          "time_s": float(self.sim.data.time),
          "phase": noise_phase,
          "nominal_wrist_position_m": nominal_position.tolist(),
          "applied_wrist_position_m": position.tolist(),
          "random_offset_m": self._noise.random_offset_m.tolist(),
          "recovery_offset_m": self._noise.recovery_offset_m.tolist(),
          "arm_goal_rad": result.joint_positions.tolist(),
          "gaussian_knot_draws": self._noise.draw_count,
        }
      )

  def _joint_margin_solution(self, result, position, rotation, *, precise=False):
    """Keep the high hover away from joint limits without moving the wrist."""
    model, scratch = self.sim.model, self.sim.ik_data
    dofs = self.sim._arm_dofs["right"]
    qpos = self.sim._arm_qpos["right"]
    limits = model.jnt_range[self.sim._arm_joint_ids["right"]]
    wrist = model.site("right_ee_site").id
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(4):
      joints = result.joint_positions
      margins = np.minimum(joints - limits[:, 0], limits[:, 1] - joints)
      joint = int(np.argmin(margins))
      if margins[joint] >= np.deg2rad(20.0):
        break
      scratch.qpos[qpos] = joints
      mujoco.mj_kinematics(model, scratch)
      mujoco.mj_comPos(model, scratch)
      mujoco.mj_jacSite(model, scratch, jacp, jacr, wrist)
      null = np.linalg.svd(
        np.vstack((jacp[:, dofs], jacr[:, dofs])), full_matrices=True
      )[2][-1]
      if abs(null[joint]) < 1e-6:
        break
      target = np.clip(
        joints[joint],
        limits[joint, 0] + np.deg2rad(20.0),
        limits[joint, 1] - np.deg2rad(20.0),
      )
      seed = joints + null * np.clip(
        (target - joints[joint]) / null[joint], -0.02, 0.02
      )
      candidate = self.sim.solve_ik(
        "right",
        position,
        rotation,
        seed=seed,
        max_iterations=400,
        position_tolerance=0.000001 if precise else 0.00002,
        orientation_tolerance=0.00005 if precise else 0.0003,
        posture_weight=0.0,
      )
      candidate_margins = np.minimum(
        candidate.joint_positions - limits[:, 0],
        limits[:, 1] - candidate.joint_positions,
      )
      if not candidate.success or candidate_margins.min() <= margins.min():
        break
      result = candidate
    return result

  def _open_elbow_solution(self, result, position, rotation, *, precise=False):
    """Use the redundant arm direction to retain the open insertion elbow.

    This changes only the IK seed and actuator goal. The wrist target and all
    physical object/contact state remain untouched. Small nullspace steps
    prevent the extra final insertion travel from drawing the elbow inward.
    """
    model, scratch = self.sim.model, self.sim.ik_data
    dofs = self.sim._arm_dofs["right"]
    qpos = self.sim._arm_qpos["right"]
    elbow = model.body("right_arm_link4").id
    wrist = model.site("right_ee_site").id
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    elbow_jac = np.zeros((3, model.nv))
    for _ in range(4):
      scratch.qpos[qpos] = result.joint_positions
      mujoco.mj_kinematics(model, scratch)
      mujoco.mj_comPos(model, scratch)
      error = -0.1508 - float(scratch.xpos[elbow, 1])
      if error >= 0.0:
        break
      mujoco.mj_jacSite(model, scratch, jacp, jacr, wrist)
      jacobian = np.vstack((jacp[:, dofs], jacr[:, dofs]))
      null = np.linalg.svd(jacobian, full_matrices=True)[2][-1]
      mujoco.mj_jacBody(model, scratch, elbow_jac, None, elbow)
      derivative = float(elbow_jac[1, dofs] @ null)
      if abs(derivative) < 1e-6:
        break
      seed = result.joint_positions + null * np.clip(error / derivative, -0.01, 0.01)
      candidate = self.sim.solve_ik(
        "right",
        position,
        rotation,
        seed=seed,
        max_iterations=400,
        position_tolerance=0.000001 if precise else 0.00002,
        orientation_tolerance=0.00005 if precise else 0.0003,
        posture_weight=0.0,
      )
      if not candidate.success:
        break
      result = candidate
    return result

  def _move_ee(
    self,
    position: np.ndarray,
    rotation: np.ndarray,
    duration: float,
    phase: str,
    *,
    require_grip: bool = False,
  ) -> None:
    start_position, start_rotation = self.sim.current_pose_matrix("right")
    if phase == "approach" and self._noise.enabled:
      # The measured hover pose already follows the previous noisy target.
      # Remove that layer before adding the continuing process to the next
      # segment, so the handoff does not add the same bias a second time.
      start_position -= self._noise.random_offset_m + self._noise.recovery_offset_m
    count = max(1, round(duration / (self._period_steps * self.sim.timestep)))
    for i in range(1, count + 1):
      u = i / count
      alpha = 10 * u**3 - 15 * u**4 + 6 * u**5
      self._command_ee(
        start_position + alpha * (position - start_position),
        _interpolate_rotation(start_rotation, rotation, alpha),
        precontact_scale=1.0 - (1.0 - NEAR_GRASP_SCALE) * alpha
        if phase == "approach"
        else None,
        noise_phase=phase,
      )
      for _ in range(self._period_steps):
        self._step(phase, require_grip=require_grip)

  def _object_pose(self) -> tuple[np.ndarray, np.ndarray]:
    body = self.sim.model.body("usb_plug").id
    return self.sim.data.xpos[body].copy(), self.sim.data.xmat[body].reshape(
      3, 3
    ).copy()

  def _servo_object(
    self,
    position: np.ndarray,
    rotation: np.ndarray,
    *,
    inserting: bool = False,
    bottom_force_target_n: float | None = None,
  ) -> None:
    """Increment the wrist command from measured plug error, including arm bias."""
    actual, actual_rotation = self._object_pose()
    ee_position, ee_rotation = self.sim.current_pose_matrix("right")
    if bottom_force_target_n is not None:
      # Regulate axial preload while retaining lateral/rotational alignment.
      # Small bounded corrections prevent the compliant pinch from tilting
      # the plug as load transfers from sliding friction to the bottom stop.
      axis = self.sim.data.site_xmat[
        self.sim.model.site("usb_socket_mouth").id
      ].reshape(3, 3)[:, 0]
      force = (
        float(np.mean(self._bottom_force_window))
        if self._bottom_force_window
        else self._state.backstop_axial_resistance_n
      )
      error = bottom_force_target_n - force
      increment = (
        0.0 if abs(error) < 0.03 else float(np.clip(0.00008 * error, -0.00004, 0.00004))
      )
      if self._state.axial_resistance_n >= config.INSERTION_FORCE_LIMIT_N:
        increment = min(
          increment,
          -0.00004 * (self._state.axial_resistance_n - config.INSERTION_FORCE_LIMIT_N),
        )
      lateral = position - actual
      lateral -= axis * float(lateral @ axis)
      lateral = np.clip(0.3 * lateral, -0.00002, 0.00002)
      angular = 0.3 * _rotation_vector_world(rotation, actual_rotation)
      norm = np.linalg.norm(angular)
      if norm > 0.0006:
        angular *= 0.0006 / norm
      turn = _rotation_step(angular)
      command = (
        self._ee_command_position
        + axis * increment
        + lateral
        + (turn - np.eye(3)) @ (ee_position - actual)
      )
      lead = command - ee_position
      max_forward_lead = max(
        0.0, config.BOTTOM_OUT_MAX_COMMAND_DEPTH_M - self._state.insertion_depth_m
      )
      command -= axis * max(0.0, float(lead @ axis) - max_forward_lead)
      # Force control commands can be smaller than the transport IK's 20 um
      # tolerance; a 1 um solve avoids accumulating them into force steps.
      self._command_ee(command, turn @ self._ee_command_rotation, precise=True)
      return
    delta_limit = self.motion.insertion_servo_step_m if inserting else 0.0015
    delta = np.clip(0.3 * (position - actual), -delta_limit, delta_limit)
    if inserting:
      axis = self.sim.data.site_xmat[
        self.sim.model.site("usb_socket_mouth").id
      ].reshape(3, 3)[:, 0]
      axial_increment = float(delta @ axis)
      if self._state.axial_resistance_n >= config.INSERTION_FORCE_LIMIT_N:
        axial_increment = min(
          axial_increment,
          -0.00008 * (self._state.axial_resistance_n - config.INSERTION_FORCE_LIMIT_N),
        )
      if self._state.insertion_depth_m >= config.BOTTOM_OUT_MAX_COMMAND_DEPTH_M:
        axial_increment = min(0.0, axial_increment)
      delta += axis * (axial_increment - float(delta @ axis))
    angular = 0.3 * _rotation_vector_world(rotation, actual_rotation)
    norm = np.linalg.norm(angular)
    angular_limit = 0.0006 if inserting else 0.018
    if norm > angular_limit:
      angular *= angular_limit / norm
    turn = _rotation_step(angular)
    command_position = (
      self._ee_command_position + delta + (turn - np.eye(3)) @ (ee_position - actual)
    )
    command_rotation = turn @ self._ee_command_rotation
    # Bound accumulated servo bias, not just per-cycle increments.
    lead = command_position - ee_position
    if np.linalg.norm(lead) > 0.003:
      lead *= 0.003 / np.linalg.norm(lead)
    if inserting:
      max_forward_lead = max(
        0.0, config.BOTTOM_OUT_MAX_COMMAND_DEPTH_M - self._state.insertion_depth_m
      )
      lead -= axis * max(0.0, float(lead @ axis) - max_forward_lead)
    angle_lead = _rotation_vector_world(command_rotation, ee_rotation)
    if np.linalg.norm(angle_lead) > 0.04:
      angle_lead *= 0.04 / np.linalg.norm(angle_lead)
    self._command_ee(
      ee_position + lead,
      _rotation_step(angle_lead) @ ee_rotation,
      precise=inserting,
    )

  def _move_object(
    self,
    position: np.ndarray,
    rotation: np.ndarray,
    duration: float,
    phase: str,
  ) -> None:
    start_position, start_rotation = self._object_pose()
    count = max(1, round(duration / (self._period_steps * self.sim.timestep)))
    for i in range(1, count + 1):
      u = i / count
      alpha = 10 * u**3 - 15 * u**4 + 6 * u**5
      self._servo_object(
        start_position + alpha * (position - start_position),
        _interpolate_rotation(start_rotation, rotation, alpha),
      )
      for _ in range(self._period_steps):
        self._step(phase, require_grip=True)

  def _socket_target(self, depth: float) -> tuple[np.ndarray, np.ndarray]:
    site = self.sim.model.site("usb_socket_mouth").id
    rotation = self.sim.data.site_xmat[site].reshape(3, 3).copy()
    position = self.sim.data.site_xpos[site] + rotation @ (
      np.array([depth, 0.0, 0.0]) - config.PLUG_TIP_LOCAL_M
    )
    return position, rotation

  def _transfer_grasp(self) -> None:
    """Turn the held plug along a clear Cartesian path above the tabletop.

    The pinky-forward grasp needs a different transport path from the prior
    grasp. Joint interpolation between these endpoints sweeps fingers down
    into the fixture; measured-object feedback keeps the turn above its rim.
    """
    position, rotation = self._socket_target(-self.motion.align_clearance_m)
    self._move_object(position, rotation, self.motion.transfer_s, "rotate_and_transfer")

  def _align_above_socket(self) -> None:
    """Require a continuous, stationary alignment dwell clear of the mouth."""
    depth = -self.motion.align_clearance_m
    position, rotation = self._socket_target(depth)
    # Finish the transport at the same hover height before timing the dwell.
    self._move_object(
      position,
      rotation,
      self.motion.transfer_hold_s + 0.5 * self.motion.align_s,
      "align",
    )
    stable_s = 0.0
    for _ in range(round((3.0 + self.motion.align_hold_s) / 0.02)):
      self._servo_object(position, rotation)
      for _ in range(self._period_steps):
        self._step("align", require_grip=True)
        aligned = (
          np.linalg.norm(self._state.lateral_error_m) < 0.00008
          and self._state.orientation_error_rad < 0.004
          and abs(self._state.insertion_depth_m - depth) < 0.00015
          and self._state.linear_speed_m_s < 0.001
          and self._state.angular_speed_rad_s < 0.01
        )
        stable_s = stable_s + self.sim.timestep if aligned else 0.0
      if stable_s >= self.motion.align_hold_s - 1e-12:
        return
    raise _TaskFailure("plug did not settle and align above the socket")

  def execute(self) -> UsbInsertionResult:
    """Run once from the current reset state; never reset the object."""
    # A retry in the same physical episode must retain the tactile latch.
    # Only a reset (simulation time zero) may start a new noise process.
    if self._executed and self.sim.data.time == 0.0:
      self._noise = PrecontactMotionNoise(*self._noise_options)
    self._executed = True
    self._precontact_target = None
    self._keep_elbow_open = False
    self._phases: list[str] = []
    self.monitor.reset()
    self._state = self.monitor.update()
    self._last_pose = self.sim.object_pose("usb_plug")
    self._start_time = float(self.sim.data.time)
    self._start_height = float(self.sim.object_pose("usb_plug")[2])
    self._max_height = self._start_height
    self._grip = np.zeros(2)
    self._minimum_grip = np.full(2, np.inf)
    self._smooth_arm_enabled = False
    self._smooth_arm_target = None
    self._grip_gap_s = 0.0
    self._peak_load = self._peak_axial = self._max_penetration = 0.0
    self._peak_backstop = self._bottom_out_hold_s = 0.0
    self._bottom_force_window = deque(maxlen=self._period_steps)
    self._active_bottom_out_confirmed = False
    self._grasp_verified = self._released = False
    self._ee_command_position, self._ee_command_rotation = self.sim.current_pose_matrix(
      "right"
    )
    failure = None
    try:
      if self.should_stop is not None and self.should_stop():
        raise _TaskFailure("cancelled")
      self._execute_motion()
    except _TaskFailure as error:
      failure = str(error)
      if np.isfinite(self.sim.data.qpos).all():
        self.sim.hold_current_arm_position("right")
        self.sim.set_hand_joint_targets(
          self._hand_names, self.sim.data.qpos[self.sim._hand_qpos["right"]]
        )
    minimum = np.where(np.isfinite(self._minimum_grip), self._minimum_grip, 0.0)
    return UsbInsertionResult(
      success=bool(
        failure is None
        and self._state.success
        and self._released
        and self._active_bottom_out_confirmed
      ),
      phases=tuple(self._phases),
      failure_reason=failure,
      elapsed_simulation_s=float(self.sim.data.time) - self._start_time,
      maximum_lift_m=float(self._max_height - self._start_height),
      grasp_verified=self._grasp_verified,
      released=self._released,
      active_bottom_out_confirmed=self._active_bottom_out_confirmed,
      bottom_out_hold_s=self._bottom_out_hold_s,
      peak_backstop_axial_resistance_n=self._peak_backstop,
      peak_socket_normal_load_n=self._peak_load,
      peak_axial_resistance_n=self._peak_axial,
      maximum_socket_penetration_m=self._max_penetration,
      minimum_lift_grip_force_n=tuple(float(x) for x in minimum),
      final_plug_pose_wxyz=tuple(float(x) for x in self._last_pose),
      insertion=self._state,
      precontact_noise=self._noise.report(),
      motion_profile=self.motion_profile,
      motion_parameters=asdict(self.motion),
    )

  def _execute_motion(self) -> None:
    from .grasp import calibrated_grasp

    _, initial_rotation = self._object_pose()
    expected_rotation = np.empty(9)
    mujoco.mju_quat2Mat(expected_rotation, config.AUTO_PLUG_QUATERNION_WXYZ)
    expected_rotation = expected_rotation.reshape(3, 3)
    # A flat, mark-down plug may rotate freely about world Z at reset. Only
    # reject a flipped/tilted initial placement; grasp targets follow its yaw.
    tilt = np.arccos(np.clip(expected_rotation[:, 2] @ initial_rotation[:, 2], -1, 1))
    if tilt > 0.03:
      raise _TaskFailure(
        "this baseline requires a flat, mark-down --plug-for-insertion initial pose"
      )
    # A numerical friction post-solve prevents soft-contact torsional creep.
    # Mass, gravity, friction, joint limits and all contact geometry stay intact.
    self.sim.model.opt.noslip_iterations = 8
    self._advance(self.motion.settle_s, "settle")
    grasp = calibrated_grasp(
      self.sim, pinch_tilt_rad=0.05 if self.motion_profile == "baseline" else 0.06
    )
    self.sim.set_hand_joint_targets(self._hand_names, grasp.approach_hand)
    # Keep the pinch wide while travelling; only the unused fingers fold for
    # rim clearance. The narrow calibrated pinch is commanded after arrival.
    self._advance(self.motion.preshape_s, "preshape")
    hover = self.sim.solve_ik(
      "right",
      grasp.wrist_position + [0, 0, 0.07],
      grasp.wrist_rotation,
      seed=grasp.arm_seed,
      max_iterations=400,
      position_tolerance=0.00002,
      orientation_tolerance=0.0003,
      posture_weight=0.0,
    )
    if not hover.success:
      raise _TaskFailure("unreachable grasp hover")
    self.sim.set_arm_joint_goal("right", hover.joint_positions)
    if self._noise.enabled:
      self._command_ee(
        grasp.wrist_position + [0, 0, 0.07],
        grasp.wrist_rotation,
        precontact_scale=1.0,
        noise_phase="hover",
      )
    self._advance(self.motion.hover_s, "hover")
    self._move_ee(
      grasp.wrist_position, grasp.wrist_rotation, self.motion.approach_s, "approach"
    )
    self._advance(self.motion.approach_hold_s, "approach")
    wrist, rotation = self.sim.current_pose_matrix("right")
    if (
      np.linalg.norm(wrist - grasp.wrist_position) > 0.003
      or np.linalg.norm(_rotation_vector_world(grasp.wrist_rotation, rotation)) > 0.03
    ):
      raise _TaskFailure("pickup wrist did not arrive before finger closing")
    # Close the wide clearance only while stationary beside the plug. Retain
    # the original final pinch path and force verification below.
    count = max(1, round(0.6 * self.motion.grasp_ramp_s / self.sim.timestep))
    for i in range(1, count + 1):
      u = i / count
      alpha = 10 * u**3 - 15 * u**4 + 6 * u**5
      self.sim.set_hand_joint_targets(
        self._hand_names,
        (1 - alpha) * grasp.approach_hand + alpha * grasp.open_hand,
      )
      self._step("close")
    stable = 0.0
    for i in range(round(4.0 / self.sim.timestep)):
      alpha = min(1.0, i * self.sim.timestep / self.motion.grasp_ramp_s)
      self.sim.set_hand_joint_targets(
        self._hand_names, (1 - alpha) * grasp.open_hand + alpha * grasp.closed_hand
      )
      self._step("close")
      stable = stable + self.sim.timestep if np.min(self._grip) > 0.05 else 0.0
      if stable >= 0.15 and alpha == 1.0:
        break
    else:
      raise _TaskFailure("could not establish stable thumb/index grip")
    wrist, rotation = self.sim.current_pose_matrix("right")
    # Clear the tabletop before turning; a long vertical lift unnecessarily
    # folds the wrist while the plug is still horizontal.
    self._move_ee(
      wrist + [0, 0, 0.05], rotation, self.motion.lift_s, "lift", require_grip=True
    )
    if self.sim.object_pose("usb_plug")[2] < self._start_height + 0.03:
      raise _TaskFailure("plug did not lift with the hand")
    self._grasp_verified = True
    self._transfer_grasp()
    self._keep_elbow_open = True
    self._smooth_arm_enabled = True
    self._align_above_socket()
    position, rotation = self._socket_target(-0.002)
    self._move_object(position, rotation, self.motion.socket_approach_s, "align")
    for _ in range(150):
      self._servo_object(position, rotation)
      self._advance(0.02, "align", require_grip=True)
      if (
        np.linalg.norm(self._state.lateral_error_m) < 0.00008
        and self._state.orientation_error_rad < 0.004
        and abs(self._state.insertion_depth_m + 0.002) < 0.00015
        and self._state.linear_speed_m_s < 0.001
        and self._state.angular_speed_rad_s < 0.01
      ):
        break
    else:
      raise _TaskFailure("plug did not align with the socket")
    self._insertion_wrist_rotation = self.sim.current_pose_matrix("right")[1].copy()
    self._smooth_arm_enabled = True
    target_depth = -0.002
    for _ in range(1500):
      aligned = (
        np.linalg.norm(self._state.lateral_error_m) < 0.00015
        and self._state.orientation_error_rad < 0.005
      )
      if (
        aligned
        and self._state.axial_resistance_n < config.INSERTION_FORCE_LIMIT_N
        and self._state.insertion_depth_m > target_depth - 0.0002
      ):
        # Advance remains gated by alignment, force and physical tracking lag.
        target_depth = min(
          config.BACKSTOP_DEPTH_M + 0.00015,
          target_depth + self.motion.insertion_step_m,
        )
      position, rotation = self._socket_target(target_depth)
      self._servo_object(position, rotation, inserting=True)
      self._advance(0.02, "insert", require_grip=True)
      if (
        self._state.insertion_depth_m >= config.TARGET_INSERTION_DEPTH_M
        and self._state.backstop_axial_resistance_n >= 0.05
      ):
        break
    else:
      raise _TaskFailure("insertion timed out without physical bottom contact")
    # Geometric seating alone cannot trigger release. Verify a sustained
    # bottom-stop load while the pinch remains closed and the plug stops.
    position, rotation = self._socket_target(config.BACKSTOP_DEPTH_M)
    for _ in range(200):
      self._servo_object(
        position,
        rotation,
        inserting=True,
        bottom_force_target_n=config.BOTTOM_OUT_TARGET_FORCE_N,
      )
      for _ in range(self._period_steps):
        self._step("bottom_out", require_grip=True)
        self._bottom_out_hold_s = (
          self._bottom_out_hold_s + self.sim.timestep
          if self._state.seated
          and self._state.backstop_axial_resistance_n >= config.BOTTOM_OUT_MIN_FORCE_N
          else 0.0
        )
      if self._bottom_out_hold_s >= config.BOTTOM_OUT_HOLD_S - 1e-12:
        self._active_bottom_out_confirmed = True
        break
    else:
      raise _TaskFailure("bottom contact did not sustain controlled preload")
    # Hold the wrist and relax the pinch along the contact normals. Pulling
    # the wrist up against a still-closed pinch can increase the bottom load.
    directions = np.zeros((5, 3))
    for geom_id, finger in self._pad_ids.items():
      for contact_index in range(self.sim.data.ncon):
        contact = self.sim.data.contact[contact_index]
        if geom_id in (contact.geom1, contact.geom2) and (
          contact.geom1 in self._plug_geoms or contact.geom2 in self._plug_geoms
        ):
          directions[finger] = contact.frame[:3] * (
            1 if contact.geom2 == geom_id else -1
          )
          break
    seed = np.array([self.sim._hand_targets["right"][n] for n in self._hand_names])
    scratch = self.sim.ik_data
    scratch.qpos[:] = self.sim.data.qpos
    scratch.qpos[self.sim._hand_qpos["right"]] = seed
    scratch.qpos[self.sim._thumb_joint6_qpos["right"]] = seed[
      self.sim._thumb_joint5_index["right"]
    ]
    mujoco.mj_forward(self.sim.model, scratch)
    tips = scratch.site_xpos[self.sim._fingertip_site_ids["right"]].copy()
    quiet = 0.0
    displacement = 0.0
    for _ in range(150):
      # 1 mm/s pad separation, continuously commanded at each physics step.
      if quiet == 0.0:
        displacement += 0.00002
      result = self.sim.solve_hand_ik(
        "right",
        tips + displacement * directions,
        seed=seed,
        arm_joint_positions=self.sim.data.qpos[self.sim._arm_qpos["right"]],
        position_tolerance=0.000001,
        posture_weight=0.0,
        max_iterations=80,
      )
      if not result.success:
        raise _TaskFailure("finger separation IK failed")
      start = seed.copy()
      seed = result.joint_positions.copy()
      for step in range(self._period_steps):
        fraction = (step + 1) / self._period_steps
        self.sim.set_hand_joint_targets(
          self._hand_names, start + fraction * (seed - start)
        )
        self._step("unload")
        quiet = (
          quiet + self.sim.timestep
          if (
            self._state.backstop_axial_resistance_n <= 0.4
            # Only accelerate the final separation after the pad load is
            # nearly gone. A 0.4 N residual could rebound at the speed change.
            and max(self._grip) <= 0.1
            and self._state.linear_speed_m_s < 0.0005
          )
          else 0.0
        )
      if quiet >= 0.1:
        break
    else:
      raise _TaskFailure("unloading did not settle before release")
    # The longer low-load separation already clears the contact. A further
    # 2 mm gives retreat clearance without exhausting the finger IK workspace.
    for _ in range(20):
      displacement += 0.0001
      result = self.sim.solve_hand_ik(
        "right",
        tips + displacement * directions,
        seed=seed,
        arm_joint_positions=self.sim.data.qpos[self.sim._arm_qpos["right"]],
        position_tolerance=0.000001,
        posture_weight=0.0,
        max_iterations=80,
      )
      if not result.success:
        raise _TaskFailure("finger separation IK failed")
      start = seed.copy()
      seed = result.joint_positions.copy()
      for step in range(self._period_steps):
        fraction = (step + 1) / self._period_steps
        self.sim.set_hand_joint_targets(
          self._hand_names, start + fraction * (seed - start)
        )
        self._step("release")
    self._advance(0.2, "release")
    wrist, rotation = self.sim.current_pose_matrix("right")
    # Retreat upward and slightly forward: clear the released plug without
    # folding the wrist inward after the recalibrated finite-patch grasp.
    self._move_ee(
      wrist + [0.04, 0, 0.07],
      self._insertion_wrist_rotation,
      self.motion.retreat_s,
      "retreat",
    )
    self._advance(self.motion.verify_s, "verify")
    actual_wrist, _ = self.sim.current_pose_matrix("right")
    unloaded = 0.0
    for _ in range(round(0.2 / self.sim.timestep)):
      self._step("verify")
      touching = False
      for contact in self.sim.data.contact:
        g1, g2 = int(contact.geom1), int(contact.geom2)
        other = g2 if g1 in self._plug_geoms else g1 if g2 in self._plug_geoms else None
        if other is not None and self.sim.model.body(
          self.sim.model.geom_bodyid[other]
        ).name.startswith("hand_r_"):
          touching = True
          break
      unloaded = 0.0 if touching else unloaded + self.sim.timestep
    self._released = bool(unloaded >= 0.1 and actual_wrist[2] - wrist[2] > 0.05)
    if not self._released:
      raise _TaskFailure("hand did not release and retreat from the plug")
    if not self._state.success:
      raise _TaskFailure("plug was not stably seated after release")
