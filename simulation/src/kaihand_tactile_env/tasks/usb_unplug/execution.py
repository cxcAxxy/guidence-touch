"""Actuator-only expert for grasping, unplugging, and replacing a USB plug."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from kaihand_tactile_env.shared.simulation import ArmHandSimulation

from . import config
from .grasp import seated_grasp
from .motion import DEFAULT_MOTION, UsbUnplugMotion
from .setup import Simulation
from .task import UsbUnplugMonitor, UsbUnplugState


@dataclass(frozen=True)
class UsbUnplugResult:
  success: bool
  initially_seated_verified: bool
  grasp_verified: bool
  fully_extracted: bool
  placed_on_table: bool
  released: bool
  hand_retreated: bool
  failure_reason: str | None
  phases: tuple[str, ...]
  elapsed_simulation_s: float
  peak_withdrawal_resistance_n: float
  peak_socket_normal_load_n: float
  peak_wall_normal_load_n: float
  maximum_socket_penetration_m: float
  final_plug_pose_wxyz: tuple[float, ...]
  final_state: UsbUnplugState
  initialization: dict | None
  mechanics_version: str
  motion_parameters: dict


class _TaskFailure(RuntimeError):
  pass


def _rotation_vector_world(target: np.ndarray, current: np.ndarray) -> np.ndarray:
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
  start: np.ndarray, end: np.ndarray, fraction: float
) -> np.ndarray:
  return _rotation_step(fraction * _rotation_vector_world(end, start)) @ start


def _smoothstep(value: float) -> float:
  return 10 * value**3 - 15 * value**4 + 6 * value**5


class UsbUnplugExecutor:
  """Closed-loop nominal expert; all object motion comes from robot contact."""

  def __init__(
    self,
    simulation: ArmHandSimulation,
    observer: Callable[[ArmHandSimulation, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    *,
    motion: UsbUnplugMotion = DEFAULT_MOTION,
  ) -> None:
    if simulation.scene != config.SCENE_NAME:
      raise ValueError("UsbUnplugExecutor requires scene='usb-unplug'")
    self.sim = simulation
    # Public and intentionally writable: recorders may attach after creation.
    self.observer = observer
    self.should_stop = should_stop
    self.motion = motion
    self.monitor = UsbUnplugMonitor(simulation)
    self._hand_names = simulation._hand_joint_names["right"]
    self._period_steps = max(1, round(motion.control_period_s / simulation.timestep))
    self._plug_body = simulation.model.body("usb_plug").id
    joint = simulation.model.joint("usb_plug_freejoint")
    self._plug_dof = int(joint.dofadr[0])
    self._validate_no_plug_constraint()

  def _validate_no_plug_constraint(self) -> None:
    model = self.sim.model
    constrained = np.isin(
      model.eq_type, (mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT)
    ) & ((model.eq_obj1id == self._plug_body) | (model.eq_obj2id == self._plug_body))
    if np.any(constrained):
      raise ValueError("USB unplug task forbids plug weld/connect constraints")

  def _assert_unassisted(self) -> None:
    data = self.sim.data
    if np.any(data.xfrc_applied[self._plug_body]) or np.any(
      data.qfrc_applied[self._plug_dof : self._plug_dof + 6]
    ):
      raise _TaskFailure("unexpected external assistance on USB plug")

  def _step(self, phase: str, *, require_grip: bool = False) -> None:
    if self.should_stop is not None and self.should_stop():
      raise _TaskFailure("cancelled")
    self._assert_unassisted()
    self.sim.task_phase = phase
    self.sim.step()
    if (
      not np.isfinite(self.sim.data.qpos).all()
      or not np.isfinite(self.sim.data.qvel).all()
    ):
      raise _TaskFailure("non-finite simulation state")
    self._state = self.monitor.update()
    if not self._phases or self._phases[-1] != phase:
      self._phases.append(phase)
    self._peak_withdrawal = max(
      self._peak_withdrawal, self._state.withdrawal_resistance_n
    )
    self._peak_socket_load = max(
      self._peak_socket_load, self._state.socket_normal_load_n
    )
    self._peak_wall_load = max(self._peak_wall_load, self._state.wall_normal_load_n)
    self._maximum_penetration = max(
      self._maximum_penetration, self._state.maximum_socket_penetration_m
    )
    if self.observer is not None:
      self.observer(self.sim, phase)
    if require_grip:
      self._grip_gap_s = (
        self._grip_gap_s + self.sim.timestep
        if min(self._state.grip_force_n) < config.GRASP_MIN_FORCE_N / 2
        else 0.0
      )
      if self._grip_gap_s > config.GRIP_GAP_TIMEOUT_S:
        raise _TaskFailure(f"opposed fingertip contact lost during {phase}")
    else:
      self._grip_gap_s = 0.0
    if self._state.socket_normal_load_n > config.MAX_SOCKET_NORMAL_LOAD_N:
      raise _TaskFailure(f"socket total load exceeded limit during {phase}")
    if self._state.wall_normal_load_n > config.MAX_SOCKET_WALL_LOAD_N:
      raise _TaskFailure(f"socket wall load exceeded limit during {phase}")
    if self._state.withdrawal_resistance_n > config.MAX_WITHDRAWAL_RESISTANCE_N:
      raise _TaskFailure(f"withdrawal resistance exceeded limit during {phase}")
    if self._state.insertion_resistance_n > config.MAX_INSERTION_RESISTANCE_N:
      raise _TaskFailure(f"reverse insertion load exceeded limit during {phase}")
    if self._state.maximum_socket_penetration_m > config.MAX_SOCKET_PENETRATION_M:
      raise _TaskFailure(f"socket penetration exceeded limit during {phase}")

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
    seed: np.ndarray | None = None,
    precise: bool = False,
  ) -> None:
    result = self.sim.solve_ik(
      "right",
      position,
      rotation,
      seed=self.sim.arm_goal["right"] if seed is None else seed,
      max_iterations=500,
      position_tolerance=1e-6 if precise else 2e-5,
      orientation_tolerance=5e-5 if precise else 3e-4,
      posture_weight=0.0,
    )
    if not result.success:
      damping = self.sim.ik_damping
      try:
        self.sim.ik_damping = min(damping, 0.004)
        result = self.sim.solve_ik(
          "right",
          position,
          rotation,
          seed=result.joint_positions,
          max_iterations=500,
          position_tolerance=1e-6 if precise else 2e-5,
          orientation_tolerance=5e-5 if precise else 3e-4,
          posture_weight=0.0,
        )
      finally:
        self.sim.ik_damping = damping
    if not result.success:
      raise _TaskFailure(
        f"unreachable wrist target ({result.position_error:.6f} m, "
        f"{result.orientation_error:.6f} rad)"
      )
    self.sim.set_arm_joint_goal("right", result.joint_positions)
    self._ee_command_position = np.asarray(position, dtype=float).copy()
    self._ee_command_rotation = np.asarray(rotation, dtype=float).copy()

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
    count = max(1, round(duration / (self._period_steps * self.sim.timestep)))
    for index in range(1, count + 1):
      alpha = _smoothstep(index / count)
      self._command_ee(
        start_position + alpha * (position - start_position),
        _interpolate_rotation(start_rotation, rotation, alpha),
      )
      for _ in range(self._period_steps):
        self._step(phase, require_grip=require_grip)

  def _object_pose(self) -> tuple[np.ndarray, np.ndarray]:
    return (
      self.sim.data.xpos[self._plug_body].copy(),
      self.sim.data.xmat[self._plug_body].reshape(3, 3).copy(),
    )

  def _socket_object_target(self, depth: float) -> tuple[np.ndarray, np.ndarray]:
    mouth = self.sim.model.site("usb_socket_mouth").id
    rotation = self.sim.data.site_xmat[mouth].reshape(3, 3).copy()
    position = self.sim.data.site_xpos[mouth] + rotation @ (
      np.array([depth, 0.0, 0.0]) - config.PLUG_TIP_LOCAL_M
    )
    return position, rotation

  def _servo_object(
    self,
    position: np.ndarray,
    rotation: np.ndarray,
    *,
    extracting: bool = False,
  ) -> None:
    actual, actual_rotation = self._object_pose()
    ee_position, ee_rotation = self.sim.current_pose_matrix("right")
    limit = (
      self.motion.extract_servo_step_m if extracting else self.motion.carry_servo_step_m
    )
    delta = np.clip(0.30 * (position - actual), -limit, limit)
    if extracting:
      axis = self.sim.data.site_xmat[
        self.sim.model.site("usb_socket_mouth").id
      ].reshape(3, 3)[:, 0]
      axial = float(delta @ axis)
      control_limit = 0.8 * config.MAX_WITHDRAWAL_RESISTANCE_N
      if self._state.withdrawal_resistance_n > control_limit:
        # Withdrawal is local -X; a positive correction unloads the plug.
        axial = max(
          axial,
          0.00008 * (self._state.withdrawal_resistance_n - control_limit),
        )
      delta += axis * (axial - float(delta @ axis))
    angular = 0.30 * _rotation_vector_world(rotation, actual_rotation)
    angular_limit = 0.0006 if extracting else self.motion.angular_servo_step_rad
    norm = float(np.linalg.norm(angular))
    if norm > angular_limit:
      angular *= angular_limit / norm
    turn = _rotation_step(angular)
    command_position = (
      self._ee_command_position + delta + (turn - np.eye(3)) @ (ee_position - actual)
    )
    command_rotation = turn @ self._ee_command_rotation
    lead = command_position - ee_position
    if np.linalg.norm(lead) > 0.003:
      lead *= 0.003 / np.linalg.norm(lead)
    angle_lead = _rotation_vector_world(command_rotation, ee_rotation)
    if np.linalg.norm(angle_lead) > 0.04:
      angle_lead *= 0.04 / np.linalg.norm(angle_lead)
    self._command_ee(
      ee_position + lead,
      _rotation_step(angle_lead) @ ee_rotation,
      precise=extracting,
    )

  def _move_object(
    self,
    position: np.ndarray,
    rotation: np.ndarray,
    duration: float,
    phase: str,
    *,
    extracting: bool = False,
    require_grip: bool = True,
  ) -> None:
    start_position, start_rotation = self._object_pose()
    count = max(1, round(duration / (self._period_steps * self.sim.timestep)))
    for index in range(1, count + 1):
      alpha = _smoothstep(index / count)
      self._servo_object(
        start_position + alpha * (position - start_position),
        _interpolate_rotation(start_rotation, rotation, alpha),
        extracting=extracting,
      )
      for _ in range(self._period_steps):
        self._step(phase, require_grip=require_grip)

  def _close_hand(
    self, start: np.ndarray, target: np.ndarray, duration: float, phase: str
  ) -> None:
    count = max(1, round(duration / self.sim.timestep))
    for index in range(1, count + 1):
      alpha = _smoothstep(index / count)
      self.sim.set_hand_joint_targets(
        self._hand_names, start + alpha * (target - start)
      )
      self._step(phase)

  def _execute_motion(self) -> None:
    self._advance(self.motion.settle_s, "settle")
    if not self._state.initially_seated_verified:
      raise _TaskFailure("installed reset did not reach a stable seated state")

    grasp = seated_grasp(self.sim)
    self.sim.set_hand_joint_targets(self._hand_names, grasp.approach_hand)
    self._advance(self.motion.preshape_s, "preshape")
    hover_position = grasp.wrist_position + np.array([0.04, 0.0, 0.07])
    hover = self.sim.solve_ik(
      "right",
      hover_position,
      grasp.wrist_rotation,
      seed=grasp.arm_seed,
      max_iterations=500,
      position_tolerance=2e-5,
      orientation_tolerance=3e-4,
      posture_weight=0.0,
    )
    if not hover.success:
      raise _TaskFailure("unreachable seated-grasp hover")
    self.sim.set_arm_joint_goal("right", hover.joint_positions)
    self._advance(self.motion.hover_s, "approach_hover")
    self._move_ee(
      grasp.wrist_position,
      grasp.wrist_rotation,
      self.motion.approach_s,
      "approach",
    )
    wrist, rotation = self.sim.current_pose_matrix("right")
    if (
      np.linalg.norm(wrist - grasp.wrist_position) > 0.003
      or np.linalg.norm(_rotation_vector_world(grasp.wrist_rotation, rotation)) > 0.03
    ):
      raise _TaskFailure("wrist did not reach seated grasp")
    self._close_hand(
      grasp.approach_hand, grasp.closed_hand, self.motion.close_s, "grasp"
    )
    for _ in range(round(1.0 / self.sim.timestep)):
      self._step("grasp", require_grip=self._state.grasp_verified)
      if self._state.grasp_verified:
        break
    if not self._state.grasp_verified:
      raise _TaskFailure("could not establish stable thumb/index grip")

    self._ee_command_position, self._ee_command_rotation = self.sim.current_pose_matrix(
      "right"
    )
    position, rotation = self._socket_object_target(0.009)
    self._move_object(
      position,
      rotation,
      self.motion.breakaway_s,
      "breakaway",
      extracting=True,
    )
    position, rotation = self._socket_object_target(self.motion.post_extract_depth_m)
    self._move_object(
      position,
      rotation,
      self.motion.extraction_s,
      "extract",
      extracting=True,
    )
    for _ in range(round(1.0 / self.sim.timestep)):
      self._servo_object(position, rotation, extracting=True)
      self._step("socket_clear", require_grip=True)
      if self._state.fully_extracted:
        break
    if not self._state.fully_extracted:
      raise _TaskFailure("plug did not fully clear all socket contacts")

    position, rotation = self._socket_object_target(self.motion.socket_hover_depth_m)
    self._move_object(position, rotation, self.motion.socket_lift_s, "lift_clear")
    home_rotation = np.empty(9)
    mujoco.mju_quat2Mat(home_rotation, config.HOME_QUATERNION_WXYZ)
    home_rotation = home_rotation.reshape(3, 3)
    home_hover = config.HOME_POSITION_M + [0.0, 0.0, self.motion.home_hover_height_m]
    self._move_object(
      home_hover, home_rotation, self.motion.transfer_s, "transfer_home"
    )
    self._move_object(
      config.HOME_POSITION_M,
      home_rotation,
      self.motion.place_s,
      "place",
      require_grip=False,
    )
    self._advance(self.motion.place_hold_s, "place")
    if not self._state.placed_on_table:
      raise _TaskFailure("plug did not settle at tabletop home before release")

    current_hand = np.array(
      [self.sim._hand_targets["right"][name] for name in self._hand_names]
    )
    self._close_hand(
      current_hand, grasp.approach_hand, self.motion.release_s, "release"
    )
    self._advance(config.RELEASE_DWELL_S + 0.05, "release")
    if not self._state.released:
      raise _TaskFailure("plug did not physically release at tabletop home")
    wrist, rotation = self.sim.current_pose_matrix("right")
    self._move_ee(
      wrist + np.array([0.0, 0.0, 0.08]),
      rotation,
      self.motion.retreat_s,
      "retreat",
    )
    equivalent_home = config.ARM_HOME["right"].copy()
    if equivalent_home[4] < 0:
      equivalent_home[4] += 2 * np.pi
    self.sim.set_arm_joint_goal("right", equivalent_home)
    self.sim.set_hand_home("right")
    self._advance(self.motion.return_home_s, "return_home")
    self._advance(self.motion.verify_s, "verify")
    if not self._state.hand_retreated:
      raise _TaskFailure("hand did not retreat while plug remained at home")

  def run(self) -> UsbUnplugResult:
    self.monitor.reset()
    self._state = self.monitor.update()
    self._phases: list[str] = []
    self._grip_gap_s = 0.0
    self._peak_withdrawal = 0.0
    self._peak_socket_load = self._state.socket_normal_load_n
    self._peak_wall_load = self._state.wall_normal_load_n
    self._maximum_penetration = self._state.maximum_socket_penetration_m
    self._ee_command_position, self._ee_command_rotation = self.sim.current_pose_matrix(
      "right"
    )
    started = float(self.sim.data.time)
    failure = None
    try:
      self._execute_motion()
    except _TaskFailure as error:
      failure = str(error)
      if np.isfinite(self.sim.data.qpos).all():
        self.sim.hold_current_arm_position("right")
        self.sim.set_hand_joint_targets(
          self._hand_names, self.sim.data.qpos[self.sim._hand_qpos["right"]]
        )
    state = self._state
    success = bool(
      failure is None
      and state.initially_seated_verified
      and state.grasp_verified
      and state.fully_extracted
      and state.placed_on_table
      and state.released
      and state.hand_retreated
      and state.success
    )
    return UsbUnplugResult(
      success=success,
      initially_seated_verified=state.initially_seated_verified,
      grasp_verified=state.grasp_verified,
      fully_extracted=state.fully_extracted,
      placed_on_table=state.placed_on_table,
      released=state.released,
      hand_retreated=state.hand_retreated,
      failure_reason=failure,
      phases=tuple(self._phases),
      elapsed_simulation_s=float(self.sim.data.time) - started,
      peak_withdrawal_resistance_n=self._peak_withdrawal,
      peak_socket_normal_load_n=self._peak_socket_load,
      peak_wall_normal_load_n=self._peak_wall_load,
      maximum_socket_penetration_m=self._maximum_penetration,
      final_plug_pose_wxyz=tuple(
        float(value) for value in self.sim.object_pose("usb_plug")
      ),
      final_state=state,
      initialization=getattr(self.sim, "initialization", None),
      mechanics_version=config.MECHANICS_VERSION,
      motion_parameters=asdict(self.motion),
    )

  execute = run


def run_episode(
  *,
  seed: int = 0,
  simulation: ArmHandSimulation | None = None,
  observer: Callable[[ArmHandSimulation, str], None] | None = None,
  physics_observer: Callable[[ArmHandSimulation], None] | None = None,
  should_stop: Callable[[], bool] | None = None,
) -> UsbUnplugResult:
  """Run one nominal episode, constructing the consistent simulation if needed."""
  simulation = simulation or Simulation(seed=seed, add_genesis_probes=False)
  if hasattr(simulation, "physics_observer"):
    simulation.physics_observer = physics_observer
  return UsbUnplugExecutor(
    simulation,
    observer=observer,
    should_stop=should_stop,
  ).run()


__all__ = ["UsbUnplugExecutor", "UsbUnplugResult", "run_episode"]
