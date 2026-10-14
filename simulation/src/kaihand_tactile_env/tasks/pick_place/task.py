"""Known-state grasp planning and deterministic execution for table objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

from kaihand_tactile_env.shared.config import SIDES
from kaihand_tactile_env.shared.simulation import (
  ArmHandSimulation,
  IkResult,
  _wxyz_from_matrix,
)
from kaihand_tactile_env.shared.trajectory import JointWaypoint

from .config import (
  _CLOSE_HOLD_OFFSET_RAD,
  _FINGERTIP_LINKS,
  _FREE_CLOSE_CUTOFF_M,
  _FREE_CLOSE_FORCE_LIMIT,
  ARM_HOME,
  OBJECT_NAMES,
)


@dataclass(frozen=True)
class GraspCandidate:
  side: str
  object_name: str
  label: str
  grasp_position: np.ndarray
  grasp_quaternion_wxyz: np.ndarray
  joint_positions: np.ndarray
  score: float


@dataclass(frozen=True)
class GraspPlan:
  side: str
  object_name: str
  initial_object_pose: np.ndarray
  candidate: GraspCandidate
  waypoints: tuple[JointWaypoint, ...]


@dataclass(frozen=True)
class ExecutionResult:
  success: bool
  object_name: str
  side: str
  initial_height: float
  maximum_height: float
  final_height: float
  retained_at_end: bool
  phases: tuple[str, ...]


@dataclass(frozen=True)
class PickPlaceResult:
  success: bool
  object_name: str
  side: str
  final_object_pose: np.ndarray
  box_center: np.ndarray
  placed_in_box: bool
  phases: tuple[str, ...]


StepObserver = Callable[[ArmHandSimulation, str], None]


class KnownStateGraspPlanner:
  """Generate top-grasp candidates from simulator-truth object poses."""

  def __init__(self, simulation: ArmHandSimulation) -> None:
    self.sim = simulation

  @staticmethod
  def _grasp_seed(side):
    # Grasp calibration is independent of the common idle/reset posture.
    sign = 1 if side == "left" else -1
    return np.deg2rad([sign * 55, -65, -sign * 70, -60, sign * 60, 0, 0])

  def candidates(self, object_name: str, side: str) -> tuple[GraspCandidate, ...]:
    _validate_object_and_side(object_name, side)
    object_position = self.sim.object_pose(object_name)[:3]
    scratch = mujoco.MjData(self.sim.model)
    scratch.qpos[:] = self.sim.data.qpos
    scratch.qpos[self.sim._arm_qpos[side]] = self._grasp_seed(side)
    mujoco.mj_kinematics(self.sim.model, scratch)
    home_rotation = scratch.site(f"{side}_ee_site").xmat.reshape(3, 3)
    hand_body_id = self.sim.model.body(f"hand_{side[0]}_base_link").id
    home_hand_rotation = scratch.xmat[hand_body_id].reshape(3, 3)
    ee_to_hand = home_rotation.T @ home_hand_rotation
    # At the calibrated reference pose, hand-local +Y points world +Z (palm up).
    # Rotate the right palm 90 degrees about world X so its normal points
    # horizontally toward +Y: palm vertical and facing the upright cylinder.
    # Mirror the rotation for the left hand.
    angle = -0.5 * np.pi if side == "right" else 0.5 * np.pi
    cosine, sine = np.cos(angle), np.sin(angle)
    world_x_quarter_turn = np.array(
      [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
    )
    desired_hand_rotation = world_x_quarter_turn @ home_hand_rotation
    desired_ee_rotation = desired_hand_rotation @ ee_to_hand.T
    lateral_sign = -1.0 if side == "left" else 1.0
    output: list[GraspCandidate] = []
    for local_x in (0.010, 0.020, 0.030):
      for local_z in (-0.090, -0.100, -0.110):
        object_in_hand = np.array([local_x, lateral_sign * 0.050, local_z])
        rotation = desired_ee_rotation
        position = object_position - desired_hand_rotation @ object_in_hand
        result = self.sim.solve_ik(
          side,
          position,
          rotation,
          seed=self._grasp_seed(side),
          max_iterations=300,
          position_tolerance=0.006,
          orientation_tolerance=0.04,
        )
        if not result.success:
          continue
        if not self.configuration_clear(
          side, result.joint_positions, allowed_object=object_name
        ):
          continue
        score = result.position_error + 0.2 * result.orientation_error
        score += 0.2 * abs(local_x - 0.030) + 0.2 * abs(local_z + 0.100)
        output.append(
          GraspCandidate(
            side=side,
            object_name=object_name,
            label=f"side_cup_local_x{local_x:+.3f}_z{local_z:+.3f}",
            grasp_position=position,
            grasp_quaternion_wxyz=_wxyz_from_matrix(rotation),
            joint_positions=result.joint_positions,
            score=float(score),
          )
        )
    return tuple(sorted(output, key=lambda item: item.score))

  def plan(self, object_name: str, side: str, *, compact: bool = False) -> GraspPlan:
    candidates = self.candidates(object_name, side)
    if not candidates:
      raise RuntimeError(f"no collision-free grasp candidate for {side} {object_name}")
    candidate = candidates[0]
    rotation = _matrix_from_result(candidate)
    object_pose = self.sim.object_pose(object_name)
    # Approach horizontally along the palm-to-object direction.  A vertical
    # pregrasp would press the upright cylinder from above and tip it over.
    approach_direction = object_pose[:3] - candidate.grasp_position
    approach_direction[2] = 0.0
    approach_direction /= np.linalg.norm(approach_direction)
    pregrasp_position = (
      candidate.grasp_position - (0.14 if compact else 0.16) * approach_direction
    )
    positions = (
      (
        "ready",
        pregrasp_position + np.array([0.0, 0.0, 0.04 if compact else 0.20]),
        1.25 if compact else 2.5,
      ),
      ("pregrasp", pregrasp_position, 0.35 if compact else 2.0),
      ("approach", candidate.grasp_position, 0.65 if compact else 2.0),
      # Stop while the object is still retained.  The executor captures the
      # physically achieved hand-to-object pose here for stable transport.
      (
        "lift",
        candidate.grasp_position + np.array([0.0, 0.0, 0.18 if compact else 0.30]),
        0.75 if compact else 1.4,
      ),
    )
    seed = self._grasp_seed(side)
    previous_waypoint = self.sim.arm_goal[side]
    waypoints: list[JointWaypoint] = []
    for phase, position, duration in positions:
      result = self.sim.solve_ik(
        side,
        position,
        rotation,
        seed=seed,
        max_iterations=300,
        position_tolerance=0.008,
        orientation_tolerance=0.05,
      )
      _require_ik(result, phase)
      if phase in {"ready", "pregrasp", "lift", "transfer"} and not self.path_clear(
        side,
        previous_waypoint,
        result.joint_positions,
        allowed_object=object_name if phase == "lift" else None,
      ):
        raise RuntimeError(f"planned path to {phase} is in collision")
      previous_waypoint = result.joint_positions
      waypoints.append(
        JointWaypoint(
          phase=phase,
          joint_positions=result.joint_positions,
          duration=duration,
          end_effector_position=position,
          end_effector_quaternion_wxyz=candidate.grasp_quaternion_wxyz,
        )
      )
      seed = result.joint_positions
    return GraspPlan(
      side=side,
      object_name=object_name,
      initial_object_pose=object_pose,
      candidate=candidate,
      waypoints=tuple(waypoints),
    )

  def plan_pick_and_place(self, side: str = "right") -> GraspPlan:
    """Plan a top pick followed by a vertical placement in the fixed box."""
    grasp_plan = self.plan("cylinder", side, compact=True)
    candidate = grasp_plan.candidate
    rotation = _matrix_from_result(candidate)
    object_position = grasp_plan.initial_object_pose[:3]
    box_center = self.sim.model.body("box").pos.copy()
    # Preserve the calibrated object-to-hand offset while translating the cylinder
    # first above the receptacle and then close to its inner floor.
    box_wall = self.sim.model.geom("box_wall_x_pos")
    box_wall_top = box_center[2] + float(box_wall.pos[2] + box_wall.size[2])
    cylinder_half_height = float(self.sim.model.geom("cylinder_geom").size[1])
    release_object_center = np.array(
      [
        box_center[0],
        box_center[1],
        box_wall_top + cylinder_half_height - 0.030,
      ]
    )
    delta_xy = box_center[:2] - object_position[:2]
    lift = grasp_plan.waypoints[-1]
    aligned_position = lift.end_effector_position + np.r_[delta_xy, 0.0]
    targets = (
      ("transfer", aligned_position, 0.85),
      (
        "place",
        candidate.grasp_position + (release_object_center - object_position),
        1.10,
      ),
    )
    seed = lift.joint_positions
    waypoints = list(grasp_plan.waypoints)
    for phase, position, duration in targets:
      result = self.sim.solve_ik(
        side,
        position,
        rotation,
        seed=seed,
        max_iterations=240,
        position_tolerance=0.012,
        orientation_tolerance=0.08 if phase.startswith("transfer") else 0.035,
      )
      _require_ik(result, phase)
      # The transfer joint target is executed directly and must have a clear
      # joint-space path.  The nominal place IK target is not: after transfer,
      # the executor measures the held object and descends in Cartesian segments.
      # Checking a transfer-to-place joint interpolation would therefore test an
      # unexecuted path that can spuriously clip a box wall.
      if phase == "transfer" and not self.path_clear(
        side, seed, result.joint_positions, allowed_object="cylinder"
      ):
        raise RuntimeError(f"planned path to {phase} is in collision")
      waypoints.append(
        JointWaypoint(
          phase=phase,
          joint_positions=result.joint_positions,
          duration=duration,
          end_effector_position=position,
          end_effector_quaternion_wxyz=candidate.grasp_quaternion_wxyz,
        )
      )
      seed = result.joint_positions
    return GraspPlan(
      side=side,
      object_name="cylinder",
      initial_object_pose=grasp_plan.initial_object_pose,
      candidate=candidate,
      waypoints=tuple(waypoints),
    )

  def configuration_clear(
    self,
    side: str,
    joint_positions: np.ndarray,
    *,
    penetration_tolerance: float = 0.004,
    allowed_object: str | None = None,
  ) -> bool:
    """Reject material robot/environment penetrations at an IK solution."""
    data = self.sim.ik_data
    np.copyto(data.qpos, self.sim.data.qpos)
    np.copyto(data.qvel, self.sim.data.qvel)
    data.qpos[self.sim._arm_qpos[side]] = joint_positions
    mujoco.mj_forward(self.sim.model, data)
    side_prefixes = (f"{side}_arm_", f"hand_{side[0]}_")
    for contact_id in range(data.ncon):
      item = data.contact[contact_id]
      if item.dist >= -penetration_tolerance:
        continue
      names = []
      for geom_id in (item.geom1, item.geom2):
        body_id = int(self.sim.model.geom_bodyid[geom_id])
        names.append(self.sim.model.body(body_id).name or "")
      robot_involved = [name.startswith(side_prefixes) for name in names]
      if allowed_object is not None and allowed_object in names:
        continue
      if any(robot_involved) and not all(robot_involved):
        return False
    return True

  def path_clear(
    self,
    side: str,
    start: np.ndarray,
    end: np.ndarray,
    *,
    samples: int = 16,
    allowed_object: str | None = None,
  ) -> bool:
    """Check an interpolated joint-space path at fixed resolution."""
    if samples < 2:
      raise ValueError("samples must be at least two")
    return all(
      self.configuration_clear(
        side,
        (1.0 - alpha) * start + alpha * end,
        allowed_object=allowed_object,
      )
      for alpha in np.linspace(0.0, 1.0, samples)
    )


class GraspExecutor:
  """Execute a plan with time-based arm and hand target interpolation."""

  def __init__(
    self,
    simulation: ArmHandSimulation,
    *,
    observer: StepObserver | None = None,
  ) -> None:
    self.sim = simulation
    self.observer = observer

  def execute(self, plan: GraspPlan) -> ExecutionResult:
    initial_height = float(self.sim.object_pose(plan.object_name)[2])
    maximum_height = initial_height
    phases: list[str] = []

    # Keep opening while the arm starts moving; there is no separate timed
    # "settle open" stage between reset and the first waypoint.
    self.sim.set_hand_home(plan.side)

    approach_index = next(
      index
      for index, waypoint in enumerate(plan.waypoints)
      if waypoint.phase == "approach"
    )
    for waypoint in plan.waypoints[: approach_index + 1]:
      self.sim.set_arm_joint_goal(plan.side, waypoint.joint_positions)
      settle = waypoint.phase in {"ready", "approach"}
      maximum_height = max(
        maximum_height,
        self._advance_until_goals(
          waypoint.duration,
          waypoint.phase,
          plan.object_name,
          plan.side,
          arm=True,
          hand=False,
          arm_tolerance=0.001
          if waypoint.phase == "ready"
          else (0.05 if not settle else 0.012),
          arm_velocity_tolerance=0.01 if waypoint.phase == "ready" else 0.08,
          settle=settle,
        ),
      )
      phases.append(waypoint.phase)

    maximum_height = max(
      maximum_height,
      self._close_until_fingertip_contacts(0.65, "close", plan.object_name, plan.side),
    )
    if plan.object_name == "cylinder":
      # Capture the relative pose produced by physical finger closure.  Do not
      # teleport the rigid object into the hand.
      self.sim.set_grasp_stabilizer(True)
    phases.append("close")

    remaining_waypoints = plan.waypoints[approach_index + 1 :]
    for index, waypoint in enumerate(remaining_waypoints):
      self.sim.set_arm_joint_goal(plan.side, waypoint.joint_positions)
      settle = index == len(remaining_waypoints) - 1
      maximum_height = max(
        maximum_height,
        self._advance_until_goals(
          waypoint.duration,
          waypoint.phase,
          plan.object_name,
          plan.side,
          arm=True,
          hand=False,
          arm_tolerance=0.05 if not settle else 0.012,
          arm_velocity_tolerance=0.01 if waypoint.phase == "ready" else 0.08,
          settle=settle,
        ),
      )
      phases.append(waypoint.phase)

    final_height = float(self.sim.object_pose(plan.object_name)[2])
    retained_at_end = final_height >= initial_height + 0.06
    return ExecutionResult(
      success=maximum_height >= initial_height + 0.06,
      object_name=plan.object_name,
      side=plan.side,
      initial_height=initial_height,
      maximum_height=maximum_height,
      final_height=final_height,
      retained_at_end=retained_at_end,
      phases=tuple(phases),
    )

  def _close_until_fingertip_contacts(
    self,
    maximum_duration: float,
    phase: str,
    object_name: str,
    side: str,
  ) -> float:
    """Close quickly in free space, then hold once all tactile pads contact."""
    maximum_steps = max(1, int(round(maximum_duration / self.sim.timestep)))
    maximum_height = float(self.sim.object_pose(object_name)[2])
    target_geom = self.sim.model.geom(f"{object_name}_geom").id
    pad_ids = np.array(
      [
        self.sim.model.geom(f"hand_{side[0]}_{finger}_{link}_tactile_pad_col").id
        for finger, link in _FINGERTIP_LINKS
      ],
      dtype=np.int32,
    )
    pad_to_finger = {int(pad_id): index for index, pad_id in enumerate(pad_ids)}
    actuator_ids_by_finger = tuple(
      tuple(
        actuator_id
        for name, actuator_id in self.sim._hand_actuators[side].items()
        if f"_{finger}_" in name
      )
      for finger, _link in _FINGERTIP_LINKS
    )
    actuator_ids = tuple(
      actuator_id for finger_ids in actuator_ids_by_finger for actuator_id in finger_ids
    )
    baseline_ranges = {
      actuator_id: self.sim.model.actuator_forcerange[actuator_id].copy()
      for actuator_id in actuator_ids
    }
    self.sim.set_hand_closure(side, 1.0)
    joint_names = tuple(self.sim._hand_targets[side])
    full_targets = np.array(
      [self.sim._hand_targets[side][name] for name in joint_names]
    )
    hold_limits = np.array(
      [
        _CLOSE_HOLD_OFFSET_RAD[
          next(finger for finger, _link in _FINGERTIP_LINKS if f"_{finger}_" in name)
        ]
        for name in joint_names
      ]
    )
    stable_contact_steps = 0
    contacted = np.zeros(len(_FINGERTIP_LINKS), dtype=bool)
    fromto = np.zeros(6, dtype=float)
    try:
      for _ in range(maximum_steps):
        gaps = np.array(
          [
            mujoco.mj_geomDistance(
              self.sim.model,
              self.sim.data,
              int(pad_id),
              target_geom,
              0.1,
              fromto,
            )
            for pad_id in pad_ids
          ]
        )
        for finger_index, finger_actuators in enumerate(actuator_ids_by_finger):
          for actuator_id in finger_actuators:
            baseline = baseline_ranges[actuator_id]
            baseline_force = float(np.max(np.abs(baseline)))
            force = (
              max(baseline_force, _FREE_CLOSE_FORCE_LIMIT)
              if gaps[finger_index] > _FREE_CLOSE_CUTOFF_M
              else baseline_force
            )
            self.sim.model.actuator_forcerange[actuator_id] = (-force, force)

        self.sim.step()
        maximum_height = max(
          maximum_height, float(self.sim.object_pose(object_name)[2])
        )
        if self.observer is not None:
          self.observer(self.sim, phase)

        contacted.fill(False)
        for contact_id in range(self.sim.data.ncon):
          item = self.sim.data.contact[contact_id]
          if target_geom not in (int(item.geom1), int(item.geom2)):
            continue
          other = int(item.geom2 if item.geom1 == target_geom else item.geom1)
          finger_index = pad_to_finger.get(other)
          if finger_index is not None:
            contacted[finger_index] = True
        stable_contact_steps = stable_contact_steps + 1 if np.all(contacted) else 0
        if stable_contact_steps >= 3:
          positions = np.array(
            [self.sim.data.qpos[self.sim._qpos_address[name]] for name in joint_names]
          )
          hold_targets = positions + np.clip(
            full_targets - positions, -hold_limits, hold_limits
          )
          self.sim.set_hand_joint_targets(joint_names, hold_targets)
          return maximum_height
    finally:
      for actuator_id, force_range in baseline_ranges.items():
        self.sim.model.actuator_forcerange[actuator_id] = force_range

    missing = [
      finger
      for is_contacting, (finger, _link) in zip(
        contacted, _FINGERTIP_LINKS, strict=True
      )
      if not is_contacting
    ]
    raise RuntimeError(
      f"{phase} did not establish all five fingertip contacts within "
      f"{maximum_duration:.2f} s; missing={missing}"
    )

  def _advance_until_goals(
    self,
    maximum_duration: float,
    phase: str,
    object_name: str,
    side: str,
    *,
    arm: bool,
    hand: bool,
    arm_tolerance: float = 0.012,
    arm_velocity_tolerance: float = 0.08,
    settle: bool = True,
  ) -> float:
    """Advance to a goal, optionally blending into the following waypoint."""
    maximum_steps = max(1, int(round(maximum_duration / self.sim.timestep)))
    stable_steps = 0
    maximum_height = float(self.sim.object_pose(object_name)[2])

    def goals_reached() -> bool:
      arm_reached = not arm or self.sim.arm_goal_error(side) < arm_tolerance
      hand_reached = not hand or self.sim.hand_goal_error(side) < 0.03
      if settle:
        arm_reached = arm_reached and (
          not arm or self.sim.arm_velocity(side) < arm_velocity_tolerance
        )
        hand_reached = hand_reached and (
          not hand or self.sim.hand_velocity(side) < 0.10
        )
      return arm_reached and hand_reached

    required_stable_steps = 5 if settle else 1
    if goals_reached():
      return maximum_height
    for _ in range(maximum_steps):
      self.sim.step()
      maximum_height = max(maximum_height, float(self.sim.object_pose(object_name)[2]))
      if self.observer is not None:
        self.observer(self.sim, phase)
      stable_steps = stable_steps + 1 if goals_reached() else 0
      if stable_steps >= required_stable_steps:
        return maximum_height
    raise RuntimeError(
      f"{phase} did not reach its {'settled' if settle else 'blend'} goal "
      f"within {maximum_duration:.2f} s"
    )

  def _advance_until_object_released(
    self,
    maximum_duration: float,
    phase: str,
    object_name: str,
    side: str,
  ) -> float:
    """Return as soon as gravity has taken over instead of waiting in place."""
    maximum_steps = max(1, int(round(maximum_duration / self.sim.timestep)))
    falling_steps = 0
    maximum_height = float(self.sim.object_pose(object_name)[2])
    actuator_ids = tuple(self.sim._hand_actuators[side].values())
    baseline_ranges = {
      actuator_id: self.sim.model.actuator_forcerange[actuator_id].copy()
      for actuator_id in actuator_ids
    }
    # Opening force only pulls fingers away from the object.  Boosting it here
    # shortens the asymmetric release impulse without increasing grasp pressure.
    for actuator_id, baseline in baseline_ranges.items():
      force = max(float(np.max(np.abs(baseline))), 0.7)
      self.sim.model.actuator_forcerange[actuator_id] = (-force, force)
    try:
      for _ in range(maximum_steps):
        self.sim.step()
        maximum_height = max(
          maximum_height, float(self.sim.object_pose(object_name)[2])
        )
        if self.observer is not None:
          self.observer(self.sim, phase)
        falling_steps = (
          falling_steps + 1 if self.sim.object_twist(object_name)[2] < -0.02 else 0
        )
        if falling_steps >= 2:
          return maximum_height
      raise RuntimeError(
        f"{object_name} did not start falling within {maximum_duration:.2f} s"
      )
    finally:
      for actuator_id, force_range in baseline_ranges.items():
        self.sim.model.actuator_forcerange[actuator_id] = force_range


class PickPlaceExecutor(GraspExecutor):
  """Follow a compact, time-parameterized pick/place path from shared home."""

  def _smooth_joint_path(self, side, targets, durations, phases):
    """C2 quintic segments with shared knot velocities and zero endpoint motion.

    Targets pass through the existing actuators at every physics step. Monotone
    knot velocities prevent joint overshoot at corners. Durations are actual
    motion times, not maximum deadlines for a full-speed goal chase.
    """
    points = np.vstack((self.sim.arm_goal[side], *targets))
    durations = np.asarray(durations, dtype=float)
    slopes = np.diff(points, axis=0) / durations[:, None]
    velocities = np.zeros_like(points)
    for i in range(1, len(points) - 1):
      before, after = slopes[i - 1], slopes[i]
      velocities[i] = np.where(
        before * after > 0,
        np.sign(before) * np.minimum(np.abs(before), np.abs(after)),
        0.0,
      )
    for i, (duration, phase) in enumerate(zip(durations, phases, strict=True)):
      steps = max(1, int(round(duration / self.sim.timestep)))
      duration = steps * self.sim.timestep
      q0, q1 = points[i : i + 2]
      v0, v1 = velocities[i : i + 2] * duration
      delta = q1 - q0 - v0
      velocity_delta = v1 - v0
      a3 = 10 * delta - 4 * velocity_delta
      a4 = -15 * delta + 7 * velocity_delta
      a5 = 6 * delta - 3 * velocity_delta
      for step in range(1, steps + 1):
        u = step / steps
        goal = q0 + v0 * u + a3 * u**3 + a4 * u**4 + a5 * u**5
        self.sim.set_arm_joint_goal(side, goal)
        self.sim.step()
        if self.observer is not None:
          self.observer(self.sim, phase)

  def execute(self, plan: GraspPlan) -> PickPlaceResult:
    self.sim.set_hand_home(plan.side)
    approach = plan.waypoints[:3]
    self._smooth_joint_path(
      plan.side,
      [w.joint_positions for w in approach],
      [w.duration for w in approach],
      [w.phase for w in approach],
    )
    self._advance_until_goals(
      0.5,
      "approach",
      plan.object_name,
      plan.side,
      arm=True,
      hand=False,
      arm_tolerance=0.006,
      arm_velocity_tolerance=0.04,
    )
    self._close_until_fingertip_contacts(0.65, "close", plan.object_name, plan.side)
    self.sim.set_grasp_stabilizer(True)
    carry = plan.waypoints[3:5]
    self._smooth_joint_path(
      plan.side,
      [w.joint_positions for w in carry],
      [w.duration for w in carry],
      [w.phase for w in carry],
    )
    self._place_with_cartesian_feedback(plan)
    self.sim.set_grasp_stabilizer(False)
    self.sim.set_hand_home(plan.side)
    self._open_and_clear(plan)

    # Retract a short distance away from the object before folding home, instead
    # of returning to the high transfer waypoint after release.
    position, rotation = self.sim.current_pose_matrix(plan.side)
    retreat = self.sim.solve_ik(
      plan.side,
      position + np.array([0.0, 0.0, 0.09]),
      rotation,
      seed=self.sim.arm_goal[plan.side],
      max_iterations=240,
      position_tolerance=0.003,
      orientation_tolerance=0.03,
    )
    _require_ik(retreat, "retreat")
    self._smooth_joint_path(
      plan.side,
      [retreat.joint_positions, ARM_HOME[plan.side]],
      [0.50, 1.25],
      ["retreat", "return_home"],
    )
    self._advance_until_goals(
      0.8,
      "return_home",
      plan.object_name,
      plan.side,
      arm=True,
      hand=False,
      arm_tolerance=0.001,
      arm_velocity_tolerance=0.05,
    )
    pose = self.sim.object_pose(plan.object_name)
    box_center = self.sim.model.body("box").pos.copy()
    placed = cylinder_is_in_box(self.sim, pose)
    return PickPlaceResult(
      success=placed,
      object_name=plan.object_name,
      side=plan.side,
      final_object_pose=pose,
      box_center=box_center,
      placed_in_box=placed,
      phases=(
        "ready",
        "pregrasp",
        "approach",
        "close",
        "lift",
        "transfer",
        "place",
        "release",
        "retreat",
        "return_home",
      ),
    )

  def _open_and_clear(self, plan):
    """Finish finger opening before the arm retreats and can snag the cylinder.

    Retain the existing release force limit for the whole short opening window,
    rather than dropping it as soon as the cylinder starts falling. The carry
    constraint is already off; the object moves only under gravity and contact.
    """
    actuator_ids = tuple(self.sim._hand_actuators[plan.side].values())
    baseline = self.sim.model.actuator_forcerange[list(actuator_ids)].copy()
    for actuator_id, limits in zip(actuator_ids, baseline, strict=True):
      force = max(float(np.max(np.abs(limits))), 0.7)
      self.sim.model.actuator_forcerange[actuator_id] = (-force, force)
    try:
      for _ in range(round(0.18 / self.sim.timestep)):
        self.sim.step()
        if self.observer is not None:
          self.observer(self.sim, "release")
    finally:
      self.sim.model.actuator_forcerange[list(actuator_ids)] = baseline

  def _place_with_cartesian_feedback(self, plan: GraspPlan) -> None:
    """Upright the held cylinder, then descend without asymmetric preload."""
    self._upright_held_cylinder(plan)
    self._descend_to_box(plan)

  def _upright_held_cylinder(self, plan: GraspPlan) -> None:
    """Apply the smallest hand rotation that makes the cylinder axis vertical."""
    object_pose = self.sim.object_pose(plan.object_name)
    object_rotation = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(object_rotation, object_pose[3:])
    object_axis = object_rotation.reshape(3, 3)[:, 2]
    vertical = np.array([0.0, 0.0, 1.0])
    cross = np.cross(object_axis, vertical)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(object_axis @ vertical, -1.0, 1.0))
    if sine < 1.0e-8:
      if cosine > 0.0:
        return
      raise RuntimeError("cannot upright an inverted cylinder")
    skew = np.array(
      [
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
      ]
    )
    correction_rotation = (
      np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)
    )
    hand_position, hand_rotation = self.sim.current_pose_matrix(plan.side)
    result = self.sim.solve_ik(
      plan.side,
      hand_position,
      correction_rotation @ hand_rotation,
      seed=self.sim.arm_goal[plan.side],
      max_iterations=240,
      position_tolerance=0.002,
      orientation_tolerance=0.015,
    )
    _require_ik(result, "upright before place")
    self._smooth_joint_path(
      plan.side,
      [result.joint_positions],
      [0.50],
      ["place"],
    )
    self._advance_until_goals(
      0.35,
      "place",
      plan.object_name,
      plan.side,
      arm=True,
      hand=False,
      arm_tolerance=0.008,
      arm_velocity_tolerance=0.05,
    )

  def _descend_to_box(self, plan: GraspPlan) -> None:
    """Execute the measured Cartesian correction used by the place phase."""
    object_pose = self.sim.object_pose(plan.object_name)
    box_center = self.sim.model.body("box").pos.copy()
    wall = self.sim.model.geom("box_wall_x_pos")
    wall_top = box_center[2] + float(wall.pos[2] + wall.size[2])
    cylinder = self.sim.model.geom("cylinder_geom")
    rotation = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(rotation, object_pose[3:])
    rotation = rotation.reshape(3, 3)
    radius, half_height = map(float, cylinder.size[:2])
    vertical_extent = radius * float(
      np.linalg.norm(rotation[2, :2])
    ) + half_height * abs(float(rotation[2, 2]))
    desired_object_position = np.array(
      [box_center[0], box_center[1], wall_top + vertical_extent - 0.030]
    )
    correction = desired_object_position - object_pose[:3]
    start_position, start_rotation = self.sim.current_pose_matrix(plan.side)
    # Solve dense Cartesian checkpoints, then traverse them continuously.
    # Avoid repeatedly accelerating toward coarse IK goals.
    segment_count = max(2, int(np.ceil(np.linalg.norm(correction) / 0.025)))
    seed = self.sim.arm_goal[plan.side]
    targets = []
    for index in range(1, segment_count + 1):
      target_position = start_position + (index / segment_count) * correction
      result = self.sim.solve_ik(
        plan.side,
        target_position,
        start_rotation,
        seed=seed,
        max_iterations=240,
        position_tolerance=0.001,
        orientation_tolerance=0.02,
      )
      _require_ik(result, f"place segment {index}/{segment_count}")
      targets.append(result.joint_positions)
      seed = result.joint_positions
    # Cosine knot times give spatially uniform knots a gradual speed envelope.
    times = np.arccos(1 - 2 * np.linspace(0, 1, segment_count + 1)) / np.pi
    self._smooth_joint_path(
      plan.side,
      targets,
      np.diff(times) * plan.waypoints[-1].duration,
      ["place"] * segment_count,
    )
    self._advance_until_goals(
      0.5,
      "place",
      plan.object_name,
      plan.side,
      arm=True,
      hand=False,
      arm_tolerance=0.015,
      arm_velocity_tolerance=0.10,
    )


def cylinder_is_in_box(
  simulation: ArmHandSimulation, pose_wxyz: np.ndarray | None = None
) -> bool:
  """Evaluate placement from a current or explicitly synchronized pose."""
  pose = (
    simulation.object_pose("cylinder")
    if pose_wxyz is None
    else np.asarray(pose_wxyz, dtype=float)
  )
  if pose.shape != (7,) or not np.all(np.isfinite(pose)):
    raise ValueError("cylinder pose must contain seven finite values")
  box_center = simulation.model.body("box").pos
  box_bottom = simulation.model.geom("box_bottom")
  floor_top = box_center[2] + float(box_bottom.pos[2] + box_bottom.size[2])
  cylinder = simulation.model.geom("cylinder_geom")
  cylinder_radius, half_height = map(float, cylinder.size[:2])
  expected_center_height = floor_top + half_height
  x_wall = simulation.model.geom("box_wall_x_pos")
  y_wall = simulation.model.geom("box_wall_y_pos")
  center_tolerance = np.array(
    [
      float(x_wall.pos[0] - x_wall.size[0]) - cylinder_radius - 0.002,
      float(y_wall.pos[1] - y_wall.size[1]) - cylinder_radius - 0.002,
    ]
  )
  return bool(
    np.all(np.abs(pose[:2] - box_center[:2]) < center_tolerance)
    and abs(float(pose[2]) - expected_center_height) < 0.05
  )


def _matrix_from_result(candidate: GraspCandidate) -> np.ndarray:
  quat = candidate.grasp_quaternion_wxyz
  matrix = np.empty(9, dtype=float)
  mujoco.mju_quat2Mat(matrix, quat)
  return matrix.reshape(3, 3)


def _require_ik(result: IkResult, phase: str) -> None:
  if not result.success:
    raise RuntimeError(
      f"IK failed for {phase}: position_error={result.position_error:.4g}, "
      f"orientation_error={result.orientation_error:.4g}"
    )


def _validate_object_and_side(object_name: str, side: str) -> None:
  if object_name not in OBJECT_NAMES:
    raise ValueError(f"object_name must be one of {OBJECT_NAMES}")
  if side not in SIDES:
    raise ValueError(f"side must be one of {SIDES}")
