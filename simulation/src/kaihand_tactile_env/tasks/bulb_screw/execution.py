"""Known-state robot policy: pick, align, repeat grasp/turn/regrasp, then release.

All motion comes from shared arm/hand actuators. This policy never assigns the
free bulb state, applies a body wrench, changes thread coordinates or welds a
hand to the bulb. Task-local passive thread capture remains in task.py.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Callable

import mujoco
import numpy as np

from kaihand_tactile_env.shared.simulation import _rotation_vector_world

from . import config
from .grasp import calibrated_grasp
from .task import BulbScrewMonitor, BulbScrewSimulation, BulbScrewState


def _rotation_step(vector):
  angle = float(np.linalg.norm(vector))
  if angle < 1e-12:
    return np.eye(3)
  x, y, z = vector / angle
  skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
  return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew


def _bounded_finger_delta(jacobian, displacement, lower, upper):
  """Re-solve with saturated joints fixed instead of clipping away pad motion."""
  change = np.zeros(4)
  free = np.ones(4, dtype=bool)
  for _ in range(5):
    if not free.any():
      break
    jac = jacobian[:, free]
    residual = displacement - jacobian[:, ~free] @ change[~free]
    proposed = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(3) * 0.002**2, residual)
    indices = np.flatnonzero(free)
    change[free] = np.clip(proposed, lower[free], upper[free])
    saturated = (proposed < lower[free]) | (proposed > upper[free])
    if not saturated.any():
      break
    free[indices[saturated]] = False
  return change


@dataclass(frozen=True)
class _MotionTiming:
  open_s: float = 1.0
  clearance_m: float = 0.07
  clear_s: float = 1.0
  recover_s: float = 1.5
  approach_s: float = 1.5
  close_steps: int = 100
  close_increment: float = 0.015
  turn_s: float = 3.0
  angular_step_rad: float = 0.008
  pickup_scale: float = 1.0
  retreat_scale: float = 1.0


_TIMINGS = {
  "normal": _MotionTiming(),
  "fast": _MotionTiming(
    open_s=0.4,
    clear_s=0.7,
    recover_s=1.0,
    approach_s=1.2,
    close_steps=45,
    close_increment=0.035,
    turn_s=2.0,
    angular_step_rad=0.016,
  ),
}

_FIVE_FINGER_FAST_TIMING = _MotionTiming(
  angular_step_rad=0.016, pickup_scale=0.6, retreat_scale=0.7
)


@dataclass(frozen=True)
class BulbScrewResult:
  success: bool
  reason: str
  elapsed_s: float
  strokes: int
  maximum_lift_m: float
  peak_fingertip_load_n: tuple[float, ...]
  final_fingertip_load_n: tuple[float, ...]
  phases: tuple[str, ...]
  state: BulbScrewState
  grasp_mode: str
  speed: str
  fingertip_names: tuple[str, ...]
  turn_contact_fraction: tuple[float, ...]
  simultaneous_turn_contact_fraction: float
  phase_durations_s: dict[str, float]
  tightening_verified: bool
  tightening_peak_torque_nm: float
  tightening_stall_duration_s: float
  rotation_driver: str
  maximum_wrist_rotation_deg: float | None
  maximum_wrist_displacement_m: float | None


class _TaskFailure(RuntimeError):
  pass


class BulbScrewExecutor:
  """Bounded nominal right-hand policy using simulator truth and real contacts.

  ``observer`` runs at each physics step and can record the shared tactile/RGB
  streams. ``should_stop`` is checked before every step, including regrasp.
  """

  def __init__(
    self,
    simulation: BulbScrewSimulation,
    observer: Callable[[BulbScrewSimulation, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    *,
    max_strokes: int = 20,
    speed: str = "fast",
    grasp_mode: str = "five-finger",
  ):
    if not isinstance(simulation, BulbScrewSimulation):
      raise ValueError("BulbScrewExecutor requires BulbScrewSimulation")
    if (
      not isinstance(max_strokes, int)
      or isinstance(max_strokes, bool)
      or max_strokes < 1
    ):
      raise ValueError("max_strokes must be a positive integer")
    if speed not in _TIMINGS:
      raise ValueError("speed must be 'normal' or 'fast'")
    self.timing = _TIMINGS[speed]
    self.speed = speed
    if grasp_mode not in {"pinch", "five-finger"}:
      raise ValueError("grasp_mode must be 'pinch' or 'five-finger'")
    self.grasp_mode = grasp_mode
    self.finger_count = 5 if grasp_mode == "five-finger" else 2
    if speed == "fast" and self.finger_count == 5:
      self.timing = _FIVE_FINGER_FAST_TIMING
    self.sim = simulation
    self._home_wrist_position, self._home_wrist_rotation = (
      simulation.current_pose_matrix("right")
    )
    self.observer, self.should_stop = observer, should_stop
    self.max_strokes = max_strokes
    # This expert performs its own loaded-stall and final-release checks.
    self.monitor = BulbScrewMonitor(simulation, auto_confirm_tightening=False)
    self._names = simulation._hand_joint_names["right"]
    self._body = simulation.model.body("bulb").id
    self._pads = {
      simulation.model.geom(f"hand_r_{finger}_link{link}_tactile_pad_col").id: i
      for i, (finger, link) in enumerate(
        (("thumb", 6), ("index", 4), ("middle", 4), ("ring", 4), ("pinky", 4))[
          : self.finger_count
        ]
      )
    }
    self._object_geoms = {
      i
      for i in range(simulation.model.ngeom)
      if simulation.model.geom_bodyid[i] == self._body
    }
    self._executed = False
    self._command_position = self._command_rotation = None
    self._grip = np.zeros(self.finger_count)
    self._grip_tangent = np.zeros(self.finger_count)
    self._hand_torque_nm = 0.0
    self._grip_goal_n = 3.0
    self._tightening = False
    self._tightening_verified = False
    self._tightening_peak_torque = 0.0
    self._tightening_stall_s = 0.0
    self._tightening_history = deque()
    self._tightening_rotation = None
    self._fixed_arm_goal = None
    self._finger_reset_open = None
    self._finger_reset_ready = False
    self._finger_drive_active = False
    self._maximum_wrist_rotation = 0.0
    self._maximum_wrist_displacement = 0.0
    self._peak_grip = np.zeros(self.finger_count)
    self._gap_s = 0.0
    self._phases: list[str] = []
    self._strokes = 0
    self._state = self.monitor.measure()
    self._phase_durations: dict[str, float] = {}
    self._turn_steps = self._all_contact_steps = 0
    self._turn_contact_steps = np.zeros(self.finger_count, dtype=int)

  def _read_grip(self):
    force = np.zeros(self.finger_count)
    wrench = np.empty(6)
    tangent = np.zeros((self.finger_count, 3))
    torque = 0.0
    for i, contact in enumerate(self.sim.data.contact):
      g1, g2 = int(contact.geom1), int(contact.geom2)
      pad = g1 if g2 in self._object_geoms else g2 if g1 in self._object_geoms else None
      if pad in self._pads:
        mujoco.mj_contactForce(self.sim.model, self.sim.data, i, wrench)
        force[self._pads[pad]] += abs(float(wrench[0]))
        frame = np.asarray(contact.frame).reshape(3, 3)
        sign = 1 if g2 in self._object_geoms else -1
        world_force = sign * (frame.T @ wrench[:3])
        world_moment = sign * (frame.T @ wrench[3:])
        tangent[self._pads[pad]] += frame[1:].T @ wrench[1:3]
        torque -= float(
          (
            np.cross(contact.pos - self.sim.data.xpos[self._body], world_force)
            + world_moment
          )[2]
        )
    self._grip_tangent = np.linalg.norm(tangent, axis=1)
    self._hand_torque_nm = torque
    return force

  def _step(self, phase, *, require_grip=False, require_thread=False):
    if self.should_stop is not None and self.should_stop():
      raise _TaskFailure("cancelled")
    if self.sim.data.time < self._state.timestamp - 1e-12:
      raise _TaskFailure("simulation reset during task")
    if self.sim.data.time - self._start_time > 360:
      raise _TaskFailure("task exceeded 360 simulation seconds")
    if not self._phases or self._phases[-1] != phase:
      self._phases.append(phase)
    self.sim.step()
    if not (
      np.isfinite(self.sim.data.qpos).all() and np.isfinite(self.sim.data.qvel).all()
    ):
      raise _TaskFailure("non-finite simulation state")
    self._state = self.monitor.update()
    self._grip = self._read_grip()
    if self._finger_drive_active:
      if not np.array_equal(self.sim.arm_goal["right"], self._fixed_arm_goal):
        raise _TaskFailure("arm goal changed during finger-driven screwing")
      wrist, rotation = self.sim.current_pose_matrix("right")
      self._maximum_wrist_rotation = max(
        self._maximum_wrist_rotation,
        float(
          np.linalg.norm(_rotation_vector_world(rotation, self._fixed_wrist_rotation))
        ),
      )
      self._maximum_wrist_displacement = max(
        self._maximum_wrist_displacement,
        float(np.linalg.norm(wrist - self._fixed_wrist_position)),
      )
    self._peak_grip = np.maximum(self._peak_grip, self._grip)
    self._phase_durations[phase] = (
      self._phase_durations.get(phase, 0) + self.sim.timestep
    )
    if phase in {"turn", "tighten"}:
      contact = self._grip >= 0.025
      self._turn_steps += 1
      self._turn_contact_steps += contact
      self._all_contact_steps += int(np.all(contact))
    self._maximum_height = max(self._maximum_height, self.sim.object_pose("bulb")[2])
    if phase == "tighten" and self.finger_count == 5:
      self._check_tightening_stall()
    if self.observer is not None:
      self.observer(self.sim, phase)
    self._gap_s = (
      self._gap_s + self.sim.timestep
      if require_grip and np.min(self._grip) < 0.025
      else 0.0
    )
    if self._gap_s > 0.1:
      raise _TaskFailure(f"required fingertip contact lost during {phase}")
    if require_thread and not self._state.engaged:
      raise _TaskFailure(f"thread disengaged during {phase}")
    if np.max(self._grip) > config.MAX_FINGERTIP_LOAD_N:
      raise _TaskFailure(
        f"fingertip load exceeded {config.MAX_FINGERTIP_LOAD_N:g} N during {phase}"
      )

  def _advance(self, duration, phase, **checks):
    for _ in range(max(1, round(duration / self.sim.timestep))):
      self._step(phase, **checks)

  def _command_ee(self, position, rotation):
    damping = self.sim.ik_damping
    try:
      self.sim.ik_damping = min(damping, 0.004)
      result = self.sim.solve_ik(
        "right",
        position,
        rotation,
        seed=self.sim.arm_goal["right"],
        max_iterations=200,
        position_tolerance=2e-5,
        orientation_tolerance=3e-4,
        posture_weight=0.0,
      )
      if not result.success:
        # The shared folded start can reach a different redundant arm branch.
        # Continue convergence without relaxing the task's pose tolerances.
        result = self.sim.solve_ik(
          "right",
          position,
          rotation,
          seed=result.joint_positions,
          max_iterations=600,
          position_tolerance=2e-5,
          orientation_tolerance=3e-4,
          posture_weight=0.0,
        )
    finally:
      self.sim.ik_damping = damping
    if not result.success:
      raise _TaskFailure(
        f"unreachable wrist target: {result.position_error:.6f} m, "
        f"{result.orientation_error:.6f} rad"
      )
    self.sim.set_arm_joint_goal("right", result.joint_positions)
    self._command_position, self._command_rotation = position.copy(), rotation.copy()

  def _move_ee(self, position, rotation, duration, phase, **checks):
    start, start_rotation = self.sim.current_pose_matrix("right")
    vector = _rotation_vector_world(rotation, start_rotation)
    count = max(1, round(duration / 0.02))
    for i in range(1, count + 1):
      u = i / count
      alpha = 10 * u**3 - 15 * u**4 + 6 * u**5
      self._command_ee(
        start + alpha * (position - start),
        _rotation_step(alpha * vector) @ start_rotation,
      )
      self._advance(0.02, phase, **checks)

  def _object_pose(self):
    return (
      self.sim.data.xpos[self._body].copy(),
      self.sim.data.xmat[self._body].reshape(3, 3).copy(),
    )

  def _servo_object(self, position, rotation, phase, **checks):
    actual, actual_rotation = self._object_pose()
    if phase == "turn" and self.finger_count == 2:
      # Wrist rotation changes the opposed-pad geometry.  Preserve the actual
      # thumb/index preload at the same 50 Hz cadence as the legacy Cartesian
      # servo instead of relying on a hidden contact from a tucked finger.
      self._regulate_grip()
    wrist, wrist_rotation = self.sim.current_pose_matrix("right")
    delta = np.clip(0.3 * (position - actual), -0.0008, 0.0008)
    angular = 0.3 * _rotation_vector_world(rotation, actual_rotation)
    norm = np.linalg.norm(angular)
    if norm > self.timing.angular_step_rad:
      angular *= self.timing.angular_step_rad / norm
    turn = _rotation_step(angular)
    command = self._command_position + delta + (turn - np.eye(3)) @ (wrist - actual)
    command_rotation = turn @ self._command_rotation
    lead = command - wrist
    norm = np.linalg.norm(lead)
    if norm > 0.003:
      command = wrist + lead * 0.003 / norm
    angular_lead = _rotation_vector_world(command_rotation, wrist_rotation)
    norm = np.linalg.norm(angular_lead)
    if norm > 0.05:
      command_rotation = _rotation_step(angular_lead * 0.05 / norm) @ wrist_rotation
    arm_start = self.sim.arm_goal["right"].copy()
    self._command_ee(command, command_rotation)
    if self.finger_count == 5 and checks.get("require_grip"):
      self._advance_grip(0.02, phase, arm_start=arm_start, **checks)
    else:
      self._advance(0.02, phase, **checks)

  def _move_object(self, position, rotation, duration, phase, **checks):
    start, start_rotation = self._object_pose()
    vector = _rotation_vector_world(rotation, start_rotation)
    count = max(1, round(duration / 0.02))
    for i in range(1, count + 1):
      u = i / count
      alpha = 10 * u**3 - 15 * u**4 + 6 * u**5
      self._servo_object(
        start + alpha * (position - start),
        _rotation_step(alpha * vector) @ start_rotation,
        phase,
        **checks,
      )

  def _ramp_hand(self, target, duration, phase, **checks):
    """Advance actuator goals continuously at the physics rate, not in 20 ms jumps."""
    start = np.array([self.sim._hand_targets["right"][n] for n in self._names])
    count = max(1, round(duration / self.sim.timestep))
    for i in range(1, count + 1):
      self.sim.set_hand_joint_targets(self._names, start + (target - start) * i / count)
      self._step(phase, **checks)

  def _hand_motion(self, target, duration, phase, *, reset_group=None, **checks):
    start = np.array([self.sim._hand_targets["right"][n] for n in self._names])
    count = max(1, round(duration / 0.02))
    for i in range(1, count + 1):
      goal = start + (target - start) * i / count
      if self._finger_drive_active:
        if reset_group is not None:
          # Released fingers return to their starting angles while the other
          # group continues regulating its measured support load.
          support = self._finger_targets(active_fingers=())
          for finger in reset_group:
            selection = slice(4 * finger, 4 * finger + 4)
            support[selection] = goal[selection]
          goal = support
        self._ramp_hand(goal, 0.02, phase, **checks)
      else:
        self.sim.set_hand_joint_targets(self._names, goal)
        self._advance(0.02, phase, **checks)

  def _update_grip_goal(self, duration_s=0.02):
    seat_load = self._state.cushion_contact_load_n + self._state.shoulder_contact_load_n
    desired = 3.0 + min(3.0, 0.12 * seat_load) if self.sim.thread_engaged else 3.0
    if self._finger_drive_active:
      desired = 2.2 + min(3.8, 0.152 * seat_load)
    if self._tightening:
      desired = max(desired, 6.0)
    limit = 0.12 * duration_s / 0.02
    self._grip_goal_n += np.clip(desired - self._grip_goal_n, -limit, limit)

  def _finger_targets(
    self,
    angle_step=0.0,
    *,
    opening=False,
    active_fingers=None,
  ):
    """Compute the next 20 ms goal, including normal feedback for support fingers.

    ``active_fingers`` selects which pads turn or open. The remaining pads
    retain normal force regulation throughout the handoff and reset.
    """
    self._update_grip_goal()
    target = np.array([self.sim._hand_targets["right"][n] for n in self._names])
    bounds = self.sim.model.jnt_range[self.sim._hand_joint_ids["right"]]
    position, rotation = self._object_pose()
    center = position + rotation @ np.array([0, 0, 0.072])
    for pad, i in self._pads.items():
      active = active_fingers is None or i in active_fingers
      point = self.sim.data.geom_xpos[pad]
      normal = center - point
      normal /= np.linalg.norm(normal)
      if opening and active:
        delta = -normal * 0.0004
      else:
        angle = angle_step if active else 0.0
        delta = -np.cross([0, 0, 1], point - center) * angle
        delta[2] -= config.THREAD_PITCH_M * angle / (2 * np.pi)
        delta += normal * 0.0001 * np.clip(self._grip_goal_n - self._grip[i], -2, 3)
      jacobian = np.zeros((3, self.sim.model.nv))
      mujoco.mj_jac(
        self.sim.model,
        self.sim.data,
        jacobian,
        None,
        point,
        int(self.sim.model.geom_bodyid[pad]),
      )
      selection = slice(4 * i, 4 * i + 4)
      jac = jacobian[:, self.sim._hand_dofs["right"][selection]].copy()
      if i == 0:
        jac[:, 3] += jacobian[:, self.sim._thumb_joint6_dof["right"]]
      limit = 0.04 if opening and active else 0.025
      target[selection] += _bounded_finger_delta(
        jac,
        delta,
        np.maximum(-limit, bounds[selection, 0] - target[selection]),
        np.minimum(limit, bounds[selection, 1] - target[selection]),
      )
    return np.clip(target, bounds[:, 0], bounds[:, 1])

  def _finger_step(
    self,
    angle_step=0.0,
    *,
    opening=False,
    phase="turn",
    require_grip=True,
    active_fingers=None,
  ):
    target = self._finger_targets(
      angle_step, opening=opening, active_fingers=active_fingers
    )
    self._ramp_hand(
      target,
      0.02,
      phase,
      require_grip=require_grip and not opening,
      require_thread=True,
    )

  def _finger_regrasp(self):
    """Reset alternating finger groups while the other fingers hold the bulb."""
    groups = ((1, 2, 3), (0, 4))
    if self._finger_reset_open is None:
      self._finger_reset_open = np.array(
        [self.sim._hand_targets["right"][n] for n in self._names]
      )
    for group in groups:
      for i in range(30):
        self._finger_step(opening=True, phase="open_for_regrasp", active_fingers=group)
        if i >= 9 and np.max(self._grip[list(group)]) < 0.01:
          break
      if np.max(self._grip[list(group)]) > 0.1:
        raise _TaskFailure("finger group did not unload before resetting")
      target = np.array([self.sim._hand_targets["right"][n] for n in self._names])
      for i in group:
        selection = slice(4 * i, 4 * i + 4)
        if self._finger_reset_ready:
          target[selection] = self._finger_reset_open[selection]
        else:
          self._finger_reset_open[selection] = target[selection]
      if self._finger_reset_ready:
        self._hand_motion(
          target, 0.16, "reset_fingers", reset_group=group, require_thread=True
        )
      stable = 0
      for _ in range(40):
        self._finger_step(phase="regrasp", require_grip=False, active_fingers=group)
        forces = self._grip[list(group)]
        stable = (
          stable + 1
          if min(forces) > 1.5 and max(forces) < max(5.5, self._grip_goal_n + 1)
          else 0
        )
        if stable >= 3:
          break
      else:
        raise _TaskFailure("finger group did not restore stable contact")
    self._finger_reset_ready = True
    self._preload(0.4, "regrasp", require_thread=True)

  def _grip_targets(self, duration_s=0.02, *, radial_only=False):
    """Compute grip correction without immediately stepping actuator targets.

    The thumb supplies opposition during pickup and joins force regulation
    once threaded. Fix abduction so adjacent fingers cannot converge into one
    another. All updates are bounded actuator goals; no object forces or state
    changes are used. During transport, ``radial_only`` solves the complete
    pad displacement with a gentler correction, limiting tangential dragging
    that a scalar normal-force gradient cannot control.
    """
    target = np.array([self.sim._hand_targets["right"][n] for n in self._names])
    bounds = self.sim.model.jnt_range[self.sim._hand_joint_ids["right"]]
    position, rotation = self._object_pose()
    center = position + rotation @ np.array([0, 0, 0.072])
    # Increase actual actuator effort in response to measured seat resistance.
    # Raw recorded forces are never synthesized or rescaled.
    self._update_grip_goal(duration_s)
    for pad, i in self._pads.items():
      if i == 0 and not self.sim.thread_engaged:
        continue
      point = self.sim.data.geom_xpos[pad]
      direction = center - point
      direction /= max(np.linalg.norm(direction), 1e-9)
      jacobian = np.zeros((3, self.sim.model.nv))
      mujoco.mj_jac(
        self.sim.model,
        self.sim.data,
        jacobian,
        None,
        point,
        int(self.sim.model.geom_bodyid[pad]),
      )
      selection = slice(4 * i, 4 * i + 4)
      error = np.clip(self._grip_goal_n - self._grip[i], -2, 3)
      if radial_only:
        jac = jacobian[:, self.sim._hand_dofs["right"][selection]].copy()
        if i == 0:
          jac[:, 3] += jacobian[:, self.sim._thumb_joint6_dof["right"]]
        limit = 0.01 * duration_s / 0.02
        lower = np.maximum(-limit, bounds[selection, 0] - target[selection])
        upper = np.minimum(limit, bounds[selection, 1] - target[selection])
        lower[0] = upper[0] = 0.0
        target[selection] += _bounded_finger_delta(
          jac, direction * 0.000025 * error * duration_s / 0.02, lower, upper
        )
        continue
      gradient = direction @ jacobian[:, self.sim._hand_dofs["right"][selection]]
      if i == 0:
        gradient[3] += direction @ jacobian[:, self.sim._thumb_joint6_dof["right"]]
      gradient[0] = 0
      target[selection] += (
        duration_s
        / 0.02
        * np.clip(
          0.00010 * error * gradient / (gradient @ gradient + 1e-6),
          -0.01,
          0.01,
        )
      )
    return np.clip(target, bounds[:, 0], bounds[:, 1])

  def _regulate_grip(self):
    """Retain the calibrated contact preload and explicit legacy pinch policy."""
    self.sim.set_hand_joint_targets(self._names, self._grip_targets())

  def _advance_grip(self, duration, phase, *, arm_start, **checks):
    """Smooth arm motion and regulate the carried bulb at the physics rate.

    Arm IK still runs at 50 Hz. Advancing both arm and finger targets at 500 Hz
    avoids exciting contact forces with stepped motion of the entire grasp.
    """
    arm_target = self.sim.arm_goal["right"].copy()
    count = max(1, round(duration / self.sim.timestep))
    for i in range(1, count + 1):
      self.sim.set_arm_joint_goal(
        "right", arm_start + (arm_target - arm_start) * i / count
      )
      self.sim.set_hand_joint_targets(
        self._names, self._grip_targets(self.sim.timestep, radial_only=True)
      )
      self._step(phase, **checks)

  def _preload(self, duration, phase, **checks):
    for _ in range(round(duration / 0.02)):
      if self._finger_drive_active:
        self._ramp_hand(self._finger_targets(), 0.02, phase, **checks)
      else:
        self._regulate_grip()
        self._advance(0.02, phase, **checks)

  def _legacy_regrasp(self):
    self._hand_motion(
      self.grasp.open_hand, self.timing.open_s, "open_for_regrasp", require_thread=True
    )
    if np.max(self._grip) > 0.1:
      raise _TaskFailure("fingertips did not release before wrist recovery")
    wrist, rotation = self.sim.current_pose_matrix("right")
    self._move_ee(
      wrist + [0, 0, self.timing.clearance_m],
      rotation,
      self.timing.clear_s,
      "clear_bulb",
      require_thread=True,
    )
    target = (
      self.grasp.wrist_position
      + self.sim.object_pose("bulb")[:3]
      - self._initial_position
    )
    self._move_ee(
      target + [0, 0, self.timing.clearance_m],
      self.grasp.wrist_rotation,
      self.timing.recover_s,
      "recover_wrist",
      require_thread=True,
    )
    self._move_ee(
      target,
      self.grasp.wrist_rotation,
      self.timing.approach_s,
      "reapproach",
      require_thread=True,
    )
    # Independently stop each finger's closure ramp at measured contact load.
    # The free-space pickup uses the firmer calibrated grasp to carry gravity;
    # once threaded, the socket supports the bulb during these lighter grasps.
    alpha = np.zeros(self.finger_count)
    for _ in range(self.timing.close_steps):
      for i in range(self.finger_count):
        if self._grip[i] < 2:
          alpha[i] = min(1.0, alpha[i] + self.timing.close_increment)
        elif self._grip[i] > 5:
          alpha[i] = max(0.0, alpha[i] - 0.002)
      target = self.grasp.closed_hand.copy()
      for i in range(self.finger_count):
        selection = slice(4 * i, 4 * i + 4)
        target[selection] = (
          self.grasp.open_hand[selection]
          + alpha[i] * (self.grasp.closed_hand - self.grasp.open_hand)[selection]
        )
      self.sim.set_hand_joint_targets(self._names, target)
      self._advance(0.02, "regrasp", require_thread=True)
    if np.min(self._grip) < 0.1:
      raise _TaskFailure("regrasp did not establish opposed tactile contact")
    self._command_position, self._command_rotation = self.sim.current_pose_matrix(
      "right"
    )

  def _motion(self):
    if self.sim.thread_engaged:
      raise _TaskFailure("automatic task must start with the free tabletop bulb")
    if self.finger_count == 2:
      # The legacy wrist-turn mode needs its calibrated redundant branch.
      # Roll while the arm is folded, then extend above the bulb; every move
      # uses actuators and remains part of the recorded hover phase.
      reference = np.deg2rad([-55, -65, 70, -60, 120, 0, 0])
      folded = self.sim.arm_goal["right"].copy()
      folded[4] = reference[4]
      for target in (folded, reference):
        start = self.sim.arm_goal["right"].copy()
        count = round(3.0 / self.sim.timestep)
        for i in range(1, count + 1):
          u = i / count
          alpha = 10 * u**3 - 15 * u**4 + 6 * u**5
          self.sim.set_arm_joint_goal("right", start + alpha * (target - start))
          self._step("hover")
      self._advance(0.4, "hover")
    self.grasp = calibrated_grasp(self.sim, five_finger=self.finger_count == 5)
    pickup = self.timing.pickup_scale
    retreat = self.timing.retreat_scale
    self.sim.set_hand_joint_targets(self._names, self.grasp.open_hand)
    self._move_ee(
      self.grasp.wrist_position + [0, 0, 0.12],
      self.grasp.wrist_rotation,
      3 * pickup,
      "hover",
    )
    self._move_ee(
      self.grasp.wrist_position, self.grasp.wrist_rotation, 3 * pickup, "approach"
    )
    self._advance(0.5 * pickup, "approach")
    self._hand_motion(self.grasp.closed_hand, 2 * pickup, "grasp")
    if self.finger_count == 5:
      self._preload(1.0 * pickup, "preload")
    self._advance(0.3 * pickup, "grasp", require_grip=True)
    lifted = self._initial_position.copy()
    lifted[2] = 0.82
    self._move_object(lifted, np.eye(3), 4 * pickup, "lift", require_grip=True)
    if self.sim.object_pose("bulb")[2] < self._initial_position[2] + 0.06:
      raise _TaskFailure("bulb did not lift at least 60 mm")
    mouth = config.THREAD_ENTRY_POSITION_M
    socket = config.SOCKET_MOUTH_POSITION_M
    transfer_height = socket[2] + config.SOCKET_TRANSFER_CLEARANCE_M
    hover = socket + [0, 0, config.SOCKET_APPROACH_CLEARANCE_M]
    # Do not translate toward the socket until the actual bulb tip, rather
    # than just its commanded pose, has cleared the rim by at least 50 mm.
    for _ in range(150):
      if self._object_pose()[0][2] >= transfer_height - 0.002:
        break
      self._servo_object(lifted, np.eye(3), "lift", require_grip=True)
    else:
      raise _TaskFailure("bulb did not reach safe transfer height")
    across = socket.copy()
    across[2] = transfer_height
    self._move_object(across, np.eye(3), 3 * pickup, "transfer", require_grip=True)
    # Servo lag can leave the bulb centimetres short after a timed move. Hold
    # the high path until its real XY is above the bore, then descend to the
    # measured 50 mm waypoint before starting the insertion stroke.
    for _ in range(250):
      actual, rotation = self._object_pose()
      if (
        np.linalg.norm(actual[:2] - socket[:2]) < 0.0005
        and abs(actual[2] - hover[2]) < 0.001
        and np.linalg.norm(_rotation_vector_world(np.eye(3), rotation)) < 0.01
      ):
        break
      target = across if np.linalg.norm(actual[:2] - socket[:2]) >= 0.0005 else hover
      self._servo_object(target, np.eye(3), "transfer", require_grip=True)
    else:
      raise _TaskFailure("bulb did not align 50 mm above socket")
    self._move_object(mouth, np.eye(3), 4 * pickup, "align_thread", require_grip=True)
    for _ in range(100):
      if self.sim.thread_engaged:
        break
      self._servo_object(mouth, np.eye(3), "engage_thread", require_grip=True)
    if not self.sim.thread_engaged:
      raise _TaskFailure("aligned thread capture timed out")
    if self.finger_count == 5:
      self._screw_with_fingers()
    else:
      self._screw_with_legacy_wrist()
    self._hand_motion(self.grasp.open_hand, 1 * retreat, "release", require_thread=True)
    wrist, rotation = self.sim.current_pose_matrix("right")
    self._move_ee(
      wrist + [0, 0, 0.10], rotation, 1.5 * retreat, "retreat", require_thread=True
    )
    if self.speed == "fast" or self.finger_count == 5:
      self._move_ee(
        self.grasp.wrist_position + [0, 0, 0.18],
        self.grasp.wrist_rotation,
        2.0 * retreat,
        "clear_for_home",
        require_thread=True,
      )
    self.sim.set_hand_home("right")
    if self.finger_count == 5:
      # Keep the wrist on the right side while rotating into its home pose.
      # Direct joint interpolation from the socket-side pose sweeps the open
      # fingertips across the stationary left hand.
      self._move_ee(
        self._home_wrist_position + [0, 0, 0.18],
        self._home_wrist_rotation,
        2.0 * retreat,
        "return_home",
        require_thread=True,
      )
      self._move_ee(
        self._home_wrist_position,
        self._home_wrist_rotation,
        2.0 * retreat,
        "return_home",
        require_thread=True,
      )
    # Finish at the exact joint home configuration. The Cartesian approach
    # makes the remaining redundant-joint motion clear of the other hand.
    start = self.sim.data.qpos[self.sim._arm_qpos["right"]].copy()
    for i in range(1, 151):
      u = i / 150
      alpha = 10 * u**3 - 15 * u**4 + 6 * u**5
      self.sim.set_arm_joint_goal(
        "right", start + alpha * (config.ARM_HOME["right"] - start)
      )
      self._advance(0.02, "return_home", require_thread=True)
    self._advance(0.5, "return_home", require_thread=True)
    self._advance(0.5, "verify_seated", require_thread=True)
    if (
      not self._state.success
      or not self._state.released
      or not self._state.bulb_lit
      or not self._tightening_verified
      or np.max(self._grip) > 0.01
    ):
      raise _TaskFailure("bulb did not remain seated, lit, tightened, and released")

  def _screw_with_fingers(self):
    wrist, rotation = self.sim.current_pose_matrix("right")
    self._command_ee(wrist, rotation)
    self._fixed_arm_goal = self.sim.arm_goal["right"].copy()
    self._fixed_wrist_position, self._fixed_wrist_rotation = (
      wrist.copy(),
      rotation.copy(),
    )
    self._finger_drive_active = True
    self._finger_regrasp()
    while self._state.clockwise_turns < config.TARGET_TURNS - 0.002:
      if self._strokes >= self.max_strokes:
        raise _TaskFailure("maximum number of finger strokes reached")
      if self._strokes:
        self._finger_regrasp()
      start_turns = self._state.clockwise_turns
      target_turns = min(start_turns + config.FINGER_STROKE_TURNS, config.TARGET_TURNS)
      for _ in range(90 if self.speed == "fast" else 125):
        error = 2 * np.pi * (target_turns - self._state.clockwise_turns)
        self._finger_step(
          np.clip(0.12 * error, 0, 0.008 if self.speed == "fast" else 0.006)
        )
        if self._state.clockwise_turns >= target_turns - 0.4 / 360:
          break
      self._strokes += 1
      if self._state.clockwise_turns - start_turns < min(
        0.025, (target_turns - start_turns) / 2
      ):
        raise _TaskFailure("finger stroke did not rotate the bulb")
      if target_turns >= config.TARGET_TURNS - 1e-9:
        break
    self._tighten()
    self._finger_drive_active = False

  def _screw_with_legacy_wrist(self):
    """Explicit --grasp pinch compatibility mode; never the default driver."""
    mouth = config.THREAD_ENTRY_POSITION_M
    while self._state.clockwise_turns < config.TARGET_TURNS - 0.01:
      if self._strokes >= self.max_strokes:
        raise _TaskFailure("maximum number of turn/regrasp strokes reached")
      # The opposed pinch already holds the entry pose. Start its first turn
      # before releasing; brushing a barely engaged crest on reapproach can
      # otherwise turn it backwards past the release threshold.
      if self._strokes:
        self._legacy_regrasp()
      start_turns = self._state.clockwise_turns
      increment = min(1 / 9, config.TARGET_TURNS - start_turns)
      # Absorb a small final remainder into this grasp instead of another full
      # release/recovery cycle. Never extend a stroke beyond 45 degrees.
      if config.TARGET_TURNS - start_turns <= 1 / 8:
        increment = config.TARGET_TURNS - start_turns
      if increment <= 0:
        break
      _, rotation = self._object_pose()
      target_rotation = (
        _rotation_step(np.array([0, 0, -2 * np.pi * increment])) @ rotation
      )
      target = mouth.copy()
      target[2] -= config.THREAD_PITCH_M * (start_turns + increment)
      self._move_object(
        target,
        target_rotation,
        self.timing.turn_s,
        "turn",
        require_grip=True,
        require_thread=True,
      )
      if start_turns + increment >= config.TARGET_TURNS - 1e-9:
        for _ in range(25):
          if self._state.backstop_load_n > 0.2:
            break
          self._servo_object(
            target, target_rotation, "turn", require_grip=True, require_thread=True
          )
      self._strokes += 1
      if self._state.clockwise_turns - start_turns < min(0.04, increment / 2):
        raise _TaskFailure("turn did not advance the bulb; grasp may be slipping")
      if start_turns + increment >= config.TARGET_TURNS - 1e-9:
        break
    self._tighten()

  def _tighten(self):
    """Keep driving the fingertips and confirm load with almost no bulb motion."""
    # Restore finger range at the seat before applying the final effort.
    if self.finger_count == 5:
      self._finger_regrasp()
    self._tightening = True
    center, initial_rotation = self._object_pose()
    position = self._command_position.copy()
    rotation = self._command_rotation.copy()
    self._tightening_history.clear()
    self._tightening_rotation = initial_rotation
    duration_s = (
      config.TIGHTENING_DURATION_S + config.TIGHTENING_EFFORT_HOLD_S
      if self.finger_count == 5
      else 0.8
    )
    for i in range(round(duration_s / 0.02)):
      if self.finger_count == 5:
        if i < round(config.TIGHTENING_DURATION_S / 0.02):
          self._finger_step(
            config.TIGHTENING_COMMAND_RAD * 0.02 / config.TIGHTENING_DURATION_S,
            phase="tighten",
          )
        else:
          # Hold the loaded actuator targets at the stop so the final test
          # observes sustained effort after the finger motion has settled.
          self._advance(0.02, "tighten", require_grip=True, require_thread=True)
      else:
        angle = np.deg2rad(8.0) * min(1.0, (i + 1) * 0.02 / 0.8)
        turn = _rotation_step(np.array([0, 0, -angle]))
        if i < 40:
          self._regulate_grip()
        self._command_ee(center + turn @ (position - center), turn @ rotation)
        self._advance(0.02, "tighten", require_grip=True, require_thread=True)
      if self.finger_count != 5:
        self._check_tightening_stall()
    # A slightly different pickup can leave one pad unloaded after the fixed
    # effort hold. Restore normal load using the existing smooth finger servo,
    # without adding commanded rotation or weakening the measured stall gate.
    if self.finger_count == 5 and not self._tightening_verified:
      for _ in range(round(config.TIGHTENING_LOAD_RECOVERY_S / 0.02)):
        self._finger_step(phase="tighten")
        if self._tightening_verified:
          break
    if not self._tightening_verified:
      self._tightening = False
      raise _TaskFailure("tightening did not establish loaded angular stall")
    # Keep regulating the five-finger normal load after illumination. A fixed
    # target can relax one pad on the smaller globe even though the tightening
    # gate was satisfied on the preceding control tick.
    if self.finger_count == 5:
      self._preload(
        config.TIGHTENING_LIGHT_HOLD_S,
        "hold_tight",
        require_grip=True,
        require_thread=True,
      )
    else:
      self._advance(
        config.TIGHTENING_LIGHT_HOLD_S,
        "hold_tight",
        require_grip=True,
        require_thread=True,
      )
    self._tightening = False

  def _check_tightening_stall(self):
    """Accumulate load and angle samples; the five-finger driver calls at 500 Hz."""
    history = self._tightening_history
    t = float(self.sim.data.time)
    history.append(
      (
        t,
        float(
          _rotation_vector_world(self._object_pose()[1], self._tightening_rotation)[2]
        ),
        self._hand_torque_nm,
        float(min(self._grip)),
        self._state.seated,
        self._state.right_hand_nonpad_contact_count == 0
        and self._state.robot_contact_count == self._state.right_hand_contact_count,
      )
    )
    while len(history) > 1 and t - history[1][0] >= config.TIGHTENING_HOLD_S:
      history.popleft()
    self._tightening_peak_torque = max(
      self._tightening_peak_torque, self._hand_torque_nm
    )
    duration = t - history[0][0]
    displacement = max(h[1] for h in history) - min(h[1] for h in history)
    verified = bool(
      duration >= config.TIGHTENING_HOLD_S - 1e-9
      and displacement <= config.TIGHTENING_MAX_ROTATION_RAD
      and min(h[2] for h in history) >= config.TIGHTENING_MIN_TORQUE_NM
      and min(h[3] for h in history) >= 4.0
      and all(h[4] for h in history)
      and all(h[5] for h in history)
    )
    # Certification records that one complete strict window occurred.  Keep it
    # monotonic, matching BulbScrewMonitor and the already irreversible light:
    # a later transient while finishing the commanded effort must not erase a
    # physically valid earlier loaded stall.  Final release is still checked
    # independently over its own continuous all-robot-contact-free window.
    if verified:
      self._tightening_verified = True
      self._tightening_stall_s = max(self._tightening_stall_s, duration)
    if verified and not self.sim.bulb_lit:
      self.sim.confirm_tightening()
      self._state = replace(self._state, bulb_lit=True)

  def execute(self) -> BulbScrewResult:
    if self._executed:
      raise RuntimeError("create a fresh executor for each episode")
    self._executed = True
    self._start_time = float(self.sim.data.time)
    self._initial_position = self.sim.object_pose("bulb")[:3].copy()
    self._maximum_height = float(self._initial_position[2])
    reason = "completed"
    success = False
    try:
      self._motion()
      success = True
    except _TaskFailure as error:
      reason = str(error)
      self.sim.hold_current_arm_position("right")
    return BulbScrewResult(
      success,
      reason,
      float(self.sim.data.time) - self._start_time,
      self._strokes,
      float(self._maximum_height - self._initial_position[2]),
      tuple(float(v) for v in self._peak_grip),
      tuple(float(v) for v in self._grip),
      tuple(self._phases),
      self._state,
      self.grasp_mode,
      self.speed,
      ("thumb", "index", "middle", "ring", "pinky")[: self.finger_count],
      tuple(float(v) / max(1, self._turn_steps) for v in self._turn_contact_steps),
      self._all_contact_steps / max(1, self._turn_steps),
      dict(self._phase_durations),
      self._tightening_verified,
      self._tightening_peak_torque,
      self._tightening_stall_s,
      config.ROTATION_DRIVER if self.finger_count == 5 else "legacy_pinch_wrist",
      float(np.rad2deg(self._maximum_wrist_rotation))
      if self.finger_count == 5
      else None,
      self._maximum_wrist_displacement if self.finger_count == 5 else None,
    )
