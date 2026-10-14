"""Known-state DIMM installation through shared robot actuators and contacts.

The example starts with both hands open at the shared home posture. The right
hand approaches the passive loading cradle through physical actuators; the free
RAM is never teleported, welded, or given an external force.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

from ...shared.simulation import _rotation_vector_world
from . import config
from .grasp import calibrated_grasp
from .task import RamInstallationMonitor, RamInstallationState


def _turn(vector):
  angle = float(np.linalg.norm(vector))
  if angle < 1e-12:
    return np.eye(3)
  x, y, z = vector / angle
  cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
  return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


@dataclass(frozen=True)
class RamInstallResult:
  success: bool
  reason: str
  elapsed_s: float
  final_state: RamInstallationState
  maximum_lift_m: float
  peak_fingertip_load_n: tuple[float, float]
  final_fingertip_load_n: tuple[float, float]
  phases: tuple[str, ...]
  peak_socket_normal_load_n: float
  peak_axial_resistance_n: float
  maximum_socket_penetration_m: float
  bottom_press_duration_s: float
  bottom_press_mean_load_n: float
  minimum_palm_down_cosine: float
  initialization: str = "open_hands_at_home_then_physical_approach"


class _TaskFailure(RuntimeError):
  pass


class RamInstallExecutor:
  def __init__(self, simulation):
    self.sim = simulation
    self.monitor = RamInstallationMonitor(simulation)
    self._state = self.monitor.measure()
    self._names = simulation._hand_joint_names["right"]
    self._body = simulation.model.body("ram").id
    self._ram_geoms = set(np.flatnonzero(simulation.model.geom_bodyid == self._body))
    self._pads = {
      simulation.model.geom(f"hand_r_{finger}_link{link}_tactile_pad_col").id: i
      for i, (finger, link) in enumerate((("thumb", 6), ("index", 4)))
    }
    self._grip = np.zeros(2)
    self._peak = np.zeros(2)
    self._phases = []
    self._peak_socket_load = 0.0
    self._peak_axial_load = 0.0
    self._max_socket_penetration = 0.0
    self._bottom_press_duration = 0.0
    self._bottom_press_loads = []
    self._minimum_palm_down_cosine = 1.0
    self._prepared = self._executed = False
    self.observer = None
    self._gap_s = 0.0
    self._aligned_duration = 0.0

  @property
  def state(self):
    return self._state

  def prepare(self):
    """Plan the grasp while preserving the fresh, open home configuration."""
    if self._prepared:
      return
    if self.sim.data.time > 1e-12:
      raise RuntimeError("RAM preparation requires a fresh simulation at time zero")
    self.grasp = calibrated_grasp(self.sim)
    self._command_position, self._command_rotation = self.sim.current_pose_matrix(
      "right"
    )
    self._state = self.monitor.measure()
    self._prepared = True

  def _read_grip(self):
    loads = np.zeros(2)
    wrench = np.zeros(6)
    for i, contact in enumerate(self.sim.data.contact):
      a, b = int(contact.geom1), int(contact.geom2)
      pad = a if b in self._ram_geoms else b if a in self._ram_geoms else -1
      if pad in self._pads:
        mujoco.mj_contactForce(self.sim.model, self.sim.data, i, wrench)
        loads[self._pads[pad]] += abs(float(wrench[0]))
    return loads

  def _step(self, phase, require_grip=False):
    self.sim.step()
    self._state = self.monitor.update()
    if phase == "align" and hasattr(self, "_insertion_hand"):
      aligned = (
        self._state.aperture_fits
        and self._state.orientation_error_rad < 0.002
        and max(abs(x) for x in self._state.lateral_error_m) < 0.00005
        and abs(self._state.insertion_depth_m + 0.0005) < 0.0001
        and self._state.linear_speed_m_s < 0.0002
        and self._state.angular_speed_rad_s < 0.005
      )
      self._aligned_duration = (
        self._aligned_duration + self.sim.timestep if aligned else 0.0
      )
    if phase in {"insert", "bottom_press"} and not self._state.aperture_fits:
      raise _TaskFailure("RAM lost aperture alignment during axial insertion")
    self._grip = self._read_grip()
    self._peak = np.maximum(self._peak, self._grip)
    self._peak_socket_load = max(
      self._peak_socket_load, self._state.socket_normal_load_n
    )
    self._peak_axial_load = max(self._peak_axial_load, self._state.axial_resistance_n)
    self._max_socket_penetration = max(
      self._max_socket_penetration, self._state.maximum_socket_penetration_m
    )
    self._maximum_height = max(self._maximum_height, self.sim.object_pose("ram")[2])
    palm = self.sim.data.xmat[self.sim.model.body("hand_r_base_link").id].reshape(3, 3)[
      :, 1
    ]
    self._minimum_palm_down_cosine = min(
      self._minimum_palm_down_cosine, -float(palm[2])
    )
    if (
      phase
      not in {"settle_cradle", "approach", "grasp", "preload", "retreat", "verify"}
      and palm[2] > -0.35
    ):
      raise _TaskFailure("palm-down grasp orientation was lost")
    if not self._phases or self._phases[-1] != phase:
      self._phases.append(phase)
    if self.observer:
      self.observer(self.sim, phase)
    if not (
      np.isfinite(self.sim.data.qpos).all() and np.isfinite(self.sim.data.qvel).all()
    ):
      raise _TaskFailure("non-finite physics state")
    if self.sim.data.time - self._start_time > 60:
      raise _TaskFailure("task exceeded 60 simulation seconds")
    if self._state.axial_resistance_n > config.INSERTION_FORCE_LIMIT_N:
      raise _TaskFailure(
        f"socket axial resistance exceeded {config.INSERTION_FORCE_LIMIT_N:g} N"
      )
    if self._state.socket_normal_load_n > 12:
      raise _TaskFailure("socket contact load exceeded 12 N")
    if self._state.maximum_socket_penetration_m > config.MAX_SOCKET_PENETRATION_M:
      raise _TaskFailure(
        "socket collision penetration exceeded the configured tolerance"
      )
    self._gap_s = (
      self._gap_s + self.sim.timestep
      if require_grip and min(self._grip) < 0.02
      else 0.0
    )
    if self._gap_s > 0.15:
      raise _TaskFailure(f"opposed fingertip contact lost during {phase}")
    if max(self._grip) > 20:
      raise _TaskFailure(f"fingertip load exceeded 20 N during {phase}")

  def _advance(self, seconds, phase, require_grip=False):
    for _ in range(max(1, round(seconds / self.sim.timestep))):
      self._step(phase, require_grip)

  def _hand_motion(self, target, seconds, phase):
    start = np.array([self.sim._hand_targets["right"][n] for n in self._names])
    count = max(1, round(seconds / self.sim.timestep))
    for i in range(1, count + 1):
      u = i / count
      alpha = 10 * u**3 - 15 * u**4 + 6 * u**5
      self.sim.set_hand_joint_targets(self._names, start + alpha * (target - start))
      self._step(phase)

  def _ramp_commands(self, arm, hand, phase, require_grip=False, seconds=0.02):
    """Interpolate physical actuator goals at 500 Hz between policy updates."""
    start_arm = self.sim.arm_goal["right"].copy()
    start_hand = np.array([self.sim._hand_targets["right"][n] for n in self._names])
    count = max(1, round(seconds / self.sim.timestep))
    for i in range(1, count + 1):
      alpha = i / count
      self.sim.set_arm_joint_goal("right", start_arm + alpha * (arm - start_arm))
      self.sim.set_hand_joint_targets(
        self._names, start_hand + alpha * (hand - start_hand)
      )
      self._step(phase, require_grip)

  def _regulate_grip(self, goal=2.5, opening=False):
    """Measured-force feedback moves only the two finger actuator goals."""
    target = np.array([self.sim._hand_targets["right"][n] for n in self._names])
    normal = self.sim.data.xmat[self._body].reshape(3, 3)[:, 1]
    for pad, i in self._pads.items():
      # The revised forehand pinch approaches the opposite PCB face.
      direction = -normal if i == 0 else normal
      point = self.sim.data.geom_xpos[pad]
      jac = np.zeros((3, self.sim.model.nv))
      mujoco.mj_jac(
        self.sim.model,
        self.sim.data,
        jac,
        None,
        point,
        int(self.sim.model.geom_bodyid[pad]),
      )
      selection = slice(4 * i, 4 * i + 4)
      matrix = jac[:, self.sim._hand_dofs["right"][selection]].copy()
      if i == 0:
        matrix[:, 3] += jac[:, self.sim._thumb_joint6_dof["right"]]
      displacement = direction * (
        -0.00002 if opening else 0.00002 * np.clip(goal - self._grip[i], -2.0, 2.0)
      )
      change = matrix.T @ np.linalg.solve(
        matrix @ matrix.T + np.eye(3) * 0.002**2, displacement
      )
      target[selection] += np.clip(change, -0.012, 0.012)
    bounds = self.sim.model.jnt_range[self.sim._hand_joint_ids["right"]]
    return np.clip(target, bounds[:, 0], bounds[:, 1])

  def _command_ee(self, position, rotation):
    damping = self.sim.ik_damping
    try:
      self.sim.ik_damping = 0.004
      result = self.sim.solve_ik(
        "right",
        position,
        rotation,
        seed=self.sim.arm_goal["right"],
        max_iterations=160,
        position_tolerance=1e-5,
        orientation_tolerance=2e-4,
        posture_weight=0.0,
      )
    finally:
      self.sim.ik_damping = damping
    if not result.success:
      raise _TaskFailure(f"unreachable wrist target ({result.position_error:.5g} m)")
    self._command_position, self._command_rotation = position.copy(), rotation.copy()
    return result.joint_positions

  def _servo(self, target, rotation, phase, require_grip=True, grip_goal=2.5):
    if phase == "insert":
      # The socket constrains lateral motion. Preserve the aligned wrist and
      # finger goals instead of correcting RAM rotation through those contacts.
      position = self._command_position.copy()
      position[:2] = self._insertion_xy
      desired_z = self._insertion_wrist_z + target[2] - self._insertion_module_z
      position[2] += np.clip(desired_z - position[2], -0.00002, 0.00002)
      arm = self._command_ee(position, self._insertion_rotation)
      self._ramp_commands(arm, self._insertion_hand, phase, require_grip)
      return
    actual = self.sim.data.xpos[self._body]
    actual_rotation = self.sim.data.xmat[self._body].reshape(3, 3)
    wrist, wrist_rotation = self.sim.current_pose_matrix("right")
    gain = 0.25
    position_step = 0.0005
    angle_step = 0.008
    delta = np.clip(gain * (target - actual), -position_step, position_step)
    angular = 0.25 * _rotation_vector_world(rotation, actual_rotation)
    magnitude = np.linalg.norm(angular)
    if magnitude > angle_step:
      angular *= angle_step / magnitude
    turn = _turn(angular)
    position = self._command_position + delta + (turn - np.eye(3)) @ (wrist - actual)
    rotation_goal = turn @ self._command_rotation
    lead = position - wrist
    magnitude = np.linalg.norm(lead)
    if magnitude > 0.003:
      position = wrist + 0.003 * lead / magnitude
    angular = _rotation_vector_world(rotation_goal, wrist_rotation)
    magnitude = np.linalg.norm(angular)
    if magnitude > 0.035:
      rotation_goal = _turn(angular * 0.035 / magnitude) @ wrist_rotation
    arm = self._command_ee(position, rotation_goal)
    hand = (
      self._insertion_hand
      if phase == "align" and hasattr(self, "_insertion_hand")
      else self._regulate_grip(goal=grip_goal)
    )
    self._ramp_commands(arm, hand, phase, require_grip)

  def _move_object(self, target, seconds, phase):
    start = self.sim.object_pose("ram")[:3].copy()
    for i in range(1, 1 + round(seconds / 0.02)):
      u = i * 0.02 / seconds
      alpha = 10 * u**3 - 15 * u**4 + 6 * u**5
      self._servo(start + alpha * (target - start), np.eye(3), phase)

  def _motion(self):
    self._advance(0.4, "settle_cradle")
    # Keep a wide pinch above the cradle and throughout the descent. The
    # narrow calibrated pinch is used only after the wrist reaches the DIMM.
    # All motion goes through the shared joint actuators.
    start_position, start_rotation = self.sim.current_pose_matrix("right")
    start_hand = np.array([self.sim._hand_targets["right"][n] for n in self._names])
    above = self.grasp.wrist_position + np.array([0.0, 0.0, 0.08])
    # Follow the short rotation in task space, rather than interpolating
    # unrelated IK branches (the old forearm target swept over 210 degrees).
    rotation_vector = _rotation_vector_world(self.grasp.wrist_rotation, start_rotation)
    for i in range(1, 201):
      u = i / 200
      alpha = 10 * u**3 - 15 * u**4 + 6 * u**5
      arm = self._command_ee(
        start_position + alpha * (above - start_position),
        _turn(alpha * rotation_vector) @ start_rotation,
      )
      hand = start_hand + alpha * (self.grasp.approach_hand - start_hand)
      self._ramp_commands(arm, hand, "approach")
    for i in range(1, 151):
      u = i / 150
      alpha = 10 * u**3 - 15 * u**4 + 6 * u**5
      arm = self._command_ee(
        above + alpha * (self.grasp.wrist_position - above),
        self.grasp.wrist_rotation,
      )
      self._ramp_commands(arm, self.grasp.approach_hand, "approach")
    self._advance(0.4, "settle_cradle")
    wrist, rotation = self.sim.current_pose_matrix("right")
    if (
      np.linalg.norm(wrist - self.grasp.wrist_position) > 0.003
      or np.linalg.norm(_rotation_vector_world(self.grasp.wrist_rotation, rotation))
      > 0.03
    ):
      raise _TaskFailure("pickup wrist did not arrive before finger closing")
    self._hand_motion(self.grasp.open_hand, 0.6, "grasp")
    self._hand_motion(self.grasp.contact_hand, 1.2, "grasp")
    for _ in range(100):
      self._ramp_commands(self.sim.arm_goal["right"], self._regulate_grip(), "preload")
    self._advance(0.2, "grasp_verified", True)
    lifted = self.sim.object_pose("ram")[:3].copy() + [0.0, 0.0, 0.03]
    self._move_object(lifted, 2.0, "lift")
    if self.sim.object_pose("ram")[2] < self._initial_height + 0.02:
      raise _TaskFailure("RAM did not lift at least 20 mm from the cradle")
    aligned = config.SOCKET_MOUTH_POSITION_M - config.RAM_BOTTOM_LOCAL_M
    transfer = aligned + [0.0, 0.0, 0.06]
    self._move_object(transfer, 3.0, "transfer")
    preinsert = aligned + [0.0, 0.0, 0.0005]
    self._move_object(preinsert, 3.0, "align")
    for _ in range(75):
      self._servo(preinsert, np.eye(3), "align")
    # Establish the final grip outside the socket, then settle alignment before
    # committing to the constrained, purely axial insertion.
    for i in range(1, 101):
      u = i / 100
      alpha = 10 * u**3 - 15 * u**4 + 6 * u**5
      self._servo(preinsert, np.eye(3), "align", grip_goal=2.5 + 1.5 * alpha)
    for _ in range(75):
      self._servo(preinsert, np.eye(3), "align", grip_goal=4.0)
    self._insertion_hand = np.array(
      [self.sim._hand_targets["right"][n] for n in self._names]
    )
    for _ in range(100):
      self._servo(preinsert, np.eye(3), "align", grip_goal=4.0)
    if not self._state.aperture_fits:
      raise _TaskFailure("full DIMM edge does not fit the socket aperture")
    if self._aligned_duration < 0.2 - 1e-10:
      raise _TaskFailure(
        "RAM did not settle aligned outside the socket before insertion"
      )
    self._insertion_xy = self._command_position[:2].copy()
    self._insertion_rotation = self._command_rotation.copy()
    self._insertion_wrist_z = float(self._command_position[2])
    self._insertion_module_z = float(self.sim.data.xpos[self._body, 2])
    self._insertion_hand = np.array(
      [self.sim._hand_targets["right"][n] for n in self._names]
    )
    seated = aligned - [0.0, 0.0, config.TARGET_INSERTION_DEPTH_M - 0.00015]
    self._move_object(seated, 6.0, "insert")
    for _ in range(200):
      if self._state.insertion_depth_m >= config.TARGET_INSERTION_DEPTH_M - 0.00025:
        break
      self._servo(seated, np.eye(3), "insert")
    else:
      raise _TaskFailure("RAM did not reach the bottom approach depth")
    self._press_bottom()
    # Unload slowly so a narrow, loosely fitting board does not spring sideways.
    for _ in range(100):
      self._ramp_commands(
        self.sim.arm_goal["right"], self._regulate_grip(opening=True), "release"
      )
    if max(self._grip) > 0.03:
      raise _TaskFailure("fingertips did not release the installed RAM")
    wrist, rotation = self.sim.current_pose_matrix("right")
    for i in range(1, 76):
      alpha = i / 75
      arm = self._command_ee(wrist + np.array([0.0, 0.0, 0.08]) * alpha, rotation)
      hand = np.array([self.sim._hand_targets["right"][n] for n in self._names])
      self._ramp_commands(arm, hand, "retreat")
    self._advance(0.8, "verify")
    for i, contact in enumerate(self.sim.data.contact):
      a, b = int(contact.geom1), int(contact.geom2)
      other = a if b in self._ram_geoms else b if a in self._ram_geoms else -1
      if other < 0:
        continue
      body_name = self.sim.model.body(self.sim.model.geom_bodyid[other]).name
      if body_name.startswith("hand_"):
        force = np.zeros(6)
        mujoco.mj_contactForce(self.sim.model, self.sim.data, i, force)
        if abs(force[0]) > 0.01:
          raise _TaskFailure("robot still supports the RAM after retreat")
    if not self._state.success:
      raise _TaskFailure("RAM did not remain stably seated after release")

  def _press_bottom(self):
    """Build and hold measured bottom resistance using a bounded wrist servo."""
    hand = self._insertion_hand
    position = self._command_position.copy()
    rotation = self._command_rotation.copy()
    start_z = float(position[2])
    load_estimate = self._state.backstop_load_n
    initial_load = load_estimate
    target_load = config.BOTTOM_OUT_TARGET_FORCE_N + 0.2
    stable = 0.0
    for step in range(400):
      # This filter belongs solely to feedback control. Recorded forces retain
      # every unfiltered 500 Hz solver measurement.
      load_estimate += 0.25 * (self._state.backstop_load_n - load_estimate)
      u = min((step + 1) * 0.02 / 0.8, 1.0)
      alpha = 10 * u**3 - 15 * u**4 + 6 * u**5
      ramped_load = initial_load + alpha * (target_load - initial_load)
      position[2] -= np.clip(0.00002 * (ramped_load - load_estimate), -0.00001, 0.00001)
      if start_z - position[2] > 0.003:
        raise _TaskFailure("bottom press exhausted its 3 mm wrist compliance budget")
      arm = self._command_ee(position, rotation)
      self._ramp_commands(arm, hand, "bottom_press", True)
      loaded = (
        self._state.backstop_load_n
        >= max(config.BOTTOM_OUT_MIN_FORCE_N, target_load - 0.2)
        and self._state.linear_speed_m_s < 0.001
        and self._state.angular_speed_rad_s < 0.05
        and abs(self._state.insertion_depth_m - config.TARGET_INSERTION_DEPTH_M)
        <= 0.0002
      )
      stable = stable + 0.02 if loaded else 0.0
      if loaded:
        self._bottom_press_loads.append(self._state.backstop_load_n)
      self._bottom_press_duration = max(self._bottom_press_duration, stable)
      if (
        stable >= max(0.5, config.BOTTOM_OUT_HOLD_S)
        and self._state.bottom_out_confirmed
      ):
        return
    raise _TaskFailure("bottom press did not establish sustained load")

  def run(self, observer: Callable | None = None) -> RamInstallResult:
    if self._executed:
      raise RuntimeError("create a fresh executor for each episode")
    self.prepare()
    self._executed = True
    self.observer = observer
    self._start_time = float(self.sim.data.time)
    self._initial_height = float(self.sim.object_pose("ram")[2])
    self._maximum_height = self._initial_height
    success, reason = False, "completed"
    try:
      self._motion()
      success = True
    except _TaskFailure as error:
      reason = str(error)
      self.sim.hold_current_arm_position("right")
    return RamInstallResult(
      success,
      reason,
      float(self.sim.data.time) - self._start_time,
      self._state,
      float(self._maximum_height - self._initial_height),
      tuple(float(x) for x in self._peak),
      tuple(float(x) for x in self._grip),
      tuple(self._phases),
      self._peak_socket_load,
      self._peak_axial_load,
      self._max_socket_penetration,
      self._bottom_press_duration,
      float(np.mean(self._bottom_press_loads)) if self._bottom_press_loads else 0.0,
      self._minimum_palm_down_cosine,
    )
