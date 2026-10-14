"""Known-state actuator policy: grasp, lift, wipe, replace, and release.

No welds, object pose writes, or forces applied to the eraser during execution.
"""

from __future__ import annotations

import numpy as np

from ...shared.approach import approach_waypoint
from ...shared.simulation import HAND_JOINT_NAMES, _rotation_vector_world
from . import cleaning, grasp
from . import config as C

PICKUP_APPROACH_PHASES = (
  "open_hand",
  "move_above_eraser",
  "descend_to_eraser",
)


def turn(vector):
  angle = np.linalg.norm(vector)
  if angle < 1e-12:
    return np.eye(3)
  x, y, z = vector / angle
  skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
  return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew


class WhiteboardWipeExecutor:
  position_above_ink_phase = "position_above_z"

  def __init__(self, simulation, observer=None):
    self.sim = simulation
    self.config = getattr(simulation, "task_config", C)
    self.observer = observer
    self.position, self.rotation = simulation.current_pose_matrix("right")
    self.progress = np.zeros(5)
    self.holding = False
    self.closing = False
    self.phases = []
    self.peak_tactile = np.zeros(5)
    self.max_lift = 0.0
    self.pickup_verified = False
    self.peak_board_phase = None
    self.peak_board_time_s = None
    self.observed_peak_board_force = 0.0
    self.peak_board_eraser_position_board_m = None
    self.approach_times = []
    self.approach_joints = []
    self.phase_peak_tactile = {}
    self.phase_final_tactile = {}
    self.phase_peak_board_force = {}
    self.open_hand_target_error_rad = None
    self.above_wrist_position_error_m = None
    self.commanded_descent_world_m = None
    self.actual_descent_world_m = None
    self.actual_descent_lateral_m = None
    self.maximum_descent_lateral_deviation_m = 0.0
    self.descent_reference_xy = None
    self.pregrasp_eraser_displacement_m = None
    self.pregrasp_peak_tactile = np.zeros(5)
    self.grasp_tactile = np.zeros(5)

  def record_approach_state(self):
    """Retain the actual arm motion used to reject winding or IK branch flips."""
    time_s = float(self.sim.data.time)
    if self.approach_times and time_s <= self.approach_times[-1]:
      return
    self.approach_times.append(time_s)
    self.approach_joints.append(self.sim.data.qpos[self.sim._arm_qpos["right"]].copy())

  def approach_motion(self):
    joints = np.asarray(self.approach_joints)
    times = np.asarray(self.approach_times)
    velocity = np.diff(joints, axis=0) / np.diff(times)[:, None]
    acceleration = np.diff(velocity, axis=0) / np.diff(times)[1:, None]
    travel_deg = np.rad2deg(joints[-1] - joints[0])
    return {
      "sample_count": int(len(times)),
      "duration_s": float(times[-1] - times[0]),
      "joint_travel_deg": travel_deg.tolist(),
      "wrist_turn_joint5_deg": float(travel_deg[4]),
      "maximum_joint_speed_deg_s": np.rad2deg(
        np.max(np.abs(velocity), axis=0)
      ).tolist(),
      "maximum_joint_acceleration_deg_s2": np.rad2deg(
        np.max(np.abs(acceleration), axis=0)
      ).tolist(),
    }

  def record_completed_step(self, phase, *, approach=False):
    """Audit one completed control step, including tactile phase boundaries."""
    s = self.sim
    if approach:
      self.record_approach_state()
    if phase == "descend_to_eraser" and self.descent_reference_xy is not None:
      wrist = s.current_pose_matrix("right")[0]
      self.maximum_descent_lateral_deviation_m = max(
        self.maximum_descent_lateral_deviation_m,
        float(np.linalg.norm(wrist[:2] - self.descent_reference_xy)),
      )
    self.max_lift = max(
      self.max_lift, float(s.object_pose("eraser")[2] - s.pickup_center[2])
    )
    tactile = s.forces.read(s.data).normal_force_n[5:].copy()
    self.peak_tactile = np.maximum(self.peak_tactile, tactile)
    self.phase_peak_tactile[phase] = np.maximum(
      self.phase_peak_tactile.get(phase, np.zeros(5)), tactile
    )
    self.phase_final_tactile[phase] = tactile
    self.phase_peak_board_force[phase] = max(
      self.phase_peak_board_force.get(phase, 0.0), float(s.board_force)
    )
    if s.board_force > self.observed_peak_board_force:
      self.observed_peak_board_force = float(s.board_force)
      self.peak_board_phase = phase
      self.peak_board_time_s = float(s.data.time)
      self.peak_board_eraser_position_board_m = (
        self.config.BOARD_ROTATION.T
        @ (s.object_pose("eraser")[:3] - s.ink_surface_center)
      ).tolist()
    if not self.phases or self.phases[-1] != phase:
      self.phases.append(phase)
    if not np.isfinite(s.data.qpos).all() or any(w.number for w in s.data.warning):
      raise RuntimeError(f"{phase}: unstable physics")
    if self.holding:
      wrist, rot = s.current_pose_matrix("right")
      expected = wrist + rot @ self.tool_offset
      if np.linalg.norm(expected - s.object_pose("eraser")[:3]) > 0.045:
        raise RuntimeError(f"{phase}: eraser slipped from fingers")
    if self.observer:
      self.observer(s, phase)

  def tick(self, phase, *, period_s=None, arm_joint_goal=None):
    s = self.sim
    if period_s is None:
      period_s = self.config.CONTROL_PERIOD_S
    if self.closing:
      loads = s.forces.read(s.data).normal_force_n[5:]
      targets = np.array([2.0, 1.5, 1.2, 1.2, 0.8])
      rate = np.clip(1.2 * (targets - loads), -0.1, 0.35)
      self.progress = np.clip(self.progress + period_s * rate, 0, 1.2)
      s.set_hand_joint_targets(
        HAND_JOINT_NAMES["right"],
        s.open_grip + np.repeat(self.progress, 4) * (s.grip - s.open_grip),
      )
    if arm_joint_goal is None:
      ik = s.solve_ik(
        "right",
        self.position,
        self.rotation,
        seed=s.arm_goal["right"],
        max_iterations=140,
        position_tolerance=(
          0.000001
          if phase
          in (
            "wipe",
            "unload_board",
            "reposition",
            "approach_board",
            "load_board",
            "support",
            "release",
          )
          else 0.0003
        ),
        orientation_tolerance=0.004,
        posture_weight=0,
      )
      if ik.position_error > 0.005 or ik.orientation_error > 0.05:
        raise RuntimeError(f"{phase}: unreachable wrist ({ik.position_error:.4f} m)")
      arm_joint_goal = ik.joint_positions
    s.set_arm_joint_goal("right", arm_joint_goal)
    s.phase = phase
    s.step(round(period_s / s.timestep))
    self.record_completed_step(phase, approach=phase in PICKUP_APPROACH_PHASES)

  def move_wrist(self, target, seconds, phase, rotation=None):
    start, rot = self.position.copy(), self.rotation.copy()
    vector = (
      _rotation_vector_world(rotation, rot) if rotation is not None else np.zeros(3)
    )
    for i in range(round(seconds / 0.01)):
      u = (i + 1) / round(seconds / 0.01)
      u = u * u * (3 - 2 * u)
      self.position = start + u * (np.asarray(target) - start)
      self.rotation = turn(u * vector) @ rot
      self.tick(phase)

  def move_hand(self, target, seconds, phase):
    """Shape the hand smoothly while holding the wrist at its descended pose."""
    s = self.sim
    names = HAND_JOINT_NAMES["right"]
    start = np.array([s._hand_targets["right"][name] for name in names])
    count = round(seconds / self.config.CONTROL_PERIOD_S)
    for i in range(count):
      u = (i + 1) / count
      blend = 10 * u**3 - 15 * u**4 + 6 * u**5
      s.set_hand_joint_targets(names, start + blend * (target - start))
      self.tick(phase)

  def move_arm_joints(self, target, seconds, phase):
    """Move through one IK branch with a zero-velocity quintic joint path."""
    start = self.sim.arm_goal["right"].copy()
    count = round(seconds / self.config.CONTROL_PERIOD_S)
    times = [float(self.sim.data.time)]
    joints = [self.sim.data.qpos[self.sim._arm_qpos["right"]].copy()]
    for i in range(count):
      u = (i + 1) / count
      blend = 10 * u**3 - 15 * u**4 + 6 * u**5
      self.tick(phase, arm_joint_goal=start + blend * (target - start))
      times.append(float(self.sim.data.time))
      joints.append(self.sim.data.qpos[self.sim._arm_qpos["right"]].copy())
    self.position, self.rotation = self.sim.current_pose_matrix("right")
    values = np.asarray(joints)
    time_s = np.asarray(times)
    velocity = np.diff(values, axis=0) / np.diff(time_s)[:, None]
    acceleration = np.diff(velocity, axis=0) / np.diff(time_s)[1:, None]
    return {
      "duration_s": float(time_s[-1] - time_s[0]),
      "joint_travel_deg": np.rad2deg(values[-1] - values[0]).tolist(),
      "maximum_joint_speed_deg_s": float(np.rad2deg(np.max(np.abs(velocity)))),
      "maximum_joint_acceleration_deg_s2": float(
        np.rad2deg(np.max(np.abs(acceleration)))
      ),
    }

  def servo_tool(self, target, rotation, phase):
    s = self.sim
    actual = s.data.xpos[s.eraser_body]
    actual_rot = s.data.xmat[s.eraser_body].reshape(3, 3)
    wrist, wrist_rot = s.current_pose_matrix("right")
    gain = 0.06 if phase == "wipe" else 0.25
    delta = np.clip(gain * (target - actual), -0.001, 0.001)
    angular = gain * _rotation_vector_world(rotation, actual_rot)
    length = np.linalg.norm(angular)
    if length > 0.015:
      angular *= 0.015 / length
    change = turn(angular)
    self.position += delta + (change - np.eye(3)) @ (wrist - actual)
    self.rotation = change @ self.rotation
    lead = self.position - wrist
    if np.linalg.norm(lead) > 0.006:
      self.position = wrist + 0.006 * lead / np.linalg.norm(lead)
    self.tick(phase)

  def move_tool(self, target, seconds, phase, rotation):
    s = self.sim
    start = s.object_pose("eraser")[:3].copy()
    start_rot = s.data.xmat[s.eraser_body].reshape(3, 3).copy()
    vector = _rotation_vector_world(rotation, start_rot)
    for i in range(round(seconds / 0.01)):
      u = (i + 1) / round(seconds / 0.01)
      u = u * u * (3 - 2 * u)
      self.servo_tool(
        start + u * (np.asarray(target) - start), turn(u * vector) @ start_rot, phase
      )

  def place_eraser(self):
    """Transfer weight to the table, then unload opposing finger forces."""
    s = self.sim
    self.move_tool(
      s.pickup_center + [0, 0, 0.003], 3.0, "lower", self.config.INITIAL_ERASER_ROTATION
    )
    weight = float(s.model.body_mass[s.eraser_body] * -s.model.opt.gravity[2])
    for _ in range(500):
      if s.table_force >= 0.20:
        break
      self.position[2] -= 0.00002
      self.tick("support")
    else:
      raise RuntimeError("Table contact was not established before release")
    self.holding = False

    def unload(phase):
      self.position[2] += float(
        np.clip(0.00003 * (s.table_force - weight), -0.000015, 0.000015)
      )
      self.tick(phase)

    for _ in range(100):
      unload("support")
    if abs(s.table_force - weight) > 0.2:
      raise RuntimeError("Table has not taken the eraser weight before release")
    initial_loads = s.forces.read(s.data).normal_force_n[5:].copy()
    # Reduce all five measured normal loads toward zero before opening fully.
    for i in range(250):
      u = min((i + 1) / 150, 1.0)
      u = u * u * (3 - 2 * u)
      loads = s.forces.read(s.data).normal_force_n[5:]
      goal = (1 - u) * initial_loads
      rate = np.clip(0.08 * (goal - loads), -0.15, 0.04)
      self.progress = np.clip(self.progress + 0.01 * rate, 0, 1.2)
      s.set_hand_joint_targets(
        HAND_JOINT_NAMES["right"],
        s.open_grip + np.repeat(self.progress, 4) * (s.grip - s.open_grip),
      )
      unload("release")
    hand = np.array(
      [s._hand_targets["right"][name] for name in HAND_JOINT_NAMES["right"]]
    )
    for i in range(100):
      u = (i + 1) / 100
      s.set_hand_joint_targets(
        HAND_JOINT_NAMES["right"], hand + u * (s.open_grip - hand)
      )
      unload("release")

  def build_wipe_path(self, first_line_start, *, ink_y, stroke_center_limit):
    """Return the original loaded Z strokes; variants may replace only the path."""
    return [
      (
        first_line_start,
        2.0,
        0.0,
        0.0,
        0.0,
        self.config.BOARD_TARGET_FORCE_N,
        "load_board",
      ),
      (
        self.config.INK_Z_VERTICES[1],
        5.5,
        0.0,
        0.0,
        self.config.BOARD_TARGET_FORCE_N,
        self.config.BOARD_TARGET_FORCE_N,
        "wipe",
      ),
      (
        self.config.INK_Z_VERTICES[2],
        6.5,
        0.0,
        0.0,
        self.config.BOARD_TARGET_FORCE_N,
        self.config.BOARD_TARGET_FORCE_N,
        "wipe",
      ),
      (
        np.array([-0.070, max(-0.100, -stroke_center_limit - ink_y)]),
        5.5,
        0.0,
        0.0,
        self.config.BOARD_TARGET_FORCE_N,
        self.config.BOARD_TARGET_FORCE_N,
        "wipe",
      ),
    ]

  def run(self):
    s = self.sim
    reason = "completed"
    released = False
    approach_motion = None
    transfer_motion = None
    try:
      self.move_wrist(self.position, 0.4, "observe")
      if s.table_force < 0.2 or s.forces.read(s.data).normal_force_n[5:].sum() > 0.05:
        raise RuntimeError(
          "Eraser must initially rest on the table with no finger contact"
        )
      self.record_approach_state()
      for _ in approach_waypoint(
        s,
        s.arm_goal["right"],
        s.spread_grip,
        phase="open_hand",
        seconds=self.config.OPEN_HAND_SETTLE_S,
      ):
        self.record_completed_step("open_hand", approach=True)
      hand = np.array(
        [s.data.qpos[s._qpos_address[name]] for name in HAND_JOINT_NAMES["right"]]
      )
      self.open_hand_target_error_rad = float(np.max(np.abs(hand - s.spread_grip)))
      if self.open_hand_target_error_rad > self.config.OPEN_HAND_MAX_ERROR_RAD:
        raise RuntimeError("open_hand: spread pose did not settle")
      for _ in approach_waypoint(
        s,
        s.approach_arm,
        s.spread_grip,
        phase="move_above_eraser",
        seconds=self.config.MOVE_ABOVE_ERASER_S,
      ):
        self.record_completed_step("move_above_eraser", approach=True)
      self.position, self.rotation = s.current_pose_matrix("right")
      above_target = (
        s.pickup_center
        + s.wrist_offset
        + np.array([0.0, 0.0, self.config.PICKUP_CLEARANCE_M])
      )
      self.above_wrist_position_error_m = float(
        np.linalg.norm(self.position - above_target)
      )
      if self.above_wrist_position_error_m > 0.008:
        raise RuntimeError("move_above_eraser: wrist did not reach the overhead pose")
      above_actual = self.position.copy()
      self.descent_reference_xy = above_actual[:2].copy()
      eraser_before_descent = s.object_pose("eraser")[:3].copy()
      grasp_target = s.pickup_center + s.wrist_offset
      self.commanded_descent_world_m = (grasp_target - above_target).copy()
      self.move_wrist(
        grasp_target,
        self.config.DESCEND_TO_ERASER_S,
        "descend_to_eraser",
      )
      descended_actual = s.current_pose_matrix("right")[0]
      self.actual_descent_world_m = descended_actual - above_actual
      self.actual_descent_lateral_m = float(
        np.linalg.norm((descended_actual - above_actual)[:2])
      )
      self.pregrasp_eraser_displacement_m = float(
        np.linalg.norm(s.object_pose("eraser")[:3] - eraser_before_descent)
      )
      self.pregrasp_peak_tactile = np.maximum.reduce(
        [self.phase_peak_tactile[phase] for phase in PICKUP_APPROACH_PHASES]
      )
      if np.max(self.pregrasp_peak_tactile) > 0.05:
        raise RuntimeError("descend_to_eraser: finger contact occurred before grasp")
      if self.pregrasp_eraser_displacement_m > 0.002:
        raise RuntimeError("descend_to_eraser: open hand disturbed the eraser")
      approach_motion = self.approach_motion()
      if not -45.0 < approach_motion["wrist_turn_joint5_deg"] < 0.0:
        raise RuntimeError(
          "approach: wrist did not use the short counterclockwise branch"
        )
      if max(approach_motion["maximum_joint_speed_deg_s"]) >= 60.0:
        raise RuntimeError("approach: arm joint speed is not smooth")
      if max(approach_motion["maximum_joint_acceleration_deg_s2"]) >= 100.0:
        raise RuntimeError("approach: arm joint acceleration is not smooth")
      # Only now, with the spread hand stationary at the grasp pose, bend it
      # into the collision-free pre-grasp shape and enable tactile closure.
      self.move_hand(s.open_grip, self.config.SHAPE_GRASP_S, "grasp")
      self.closing = True
      self.move_wrist(self.position, self.config.TACTILE_GRASP_S, "grasp")
      loads = s.forces.read(s.data).normal_force_n[5:]
      self.grasp_tactile = loads.copy()
      if loads[0] < 0.05 or loads[1:].sum() < 0.15:
        raise RuntimeError(f"No opposed tactile grasp: {loads}")
      self.closing = False
      wrist, rot = s.current_pose_matrix("right")
      self.tool_offset = rot.T @ (s.object_pose("eraser")[:3] - wrist)
      self.tool_rotation_offset = rot.T @ s.data.xmat[s.eraser_body].reshape(3, 3)
      self.holding = True
      # Take the eraser's weight gradually before the main lift. This softens
      # the table-to-finger load transfer visible in the native force trace.
      self.move_tool(
        s.pickup_center + [0, 0, 0.025],
        2.0,
        "lift",
        self.config.INITIAL_ERASER_ROTATION,
      )
      self.move_tool(
        s.pickup_center + [0, 0, 0.15], 2.5, "lift", self.config.INITIAL_ERASER_ROTATION
      )
      if s.table_force > 0.01 or self.max_lift < 0.12:
        raise RuntimeError("Eraser did not lift off the table")
      self.pickup_verified = True
      # Tool bottom faces the board; the handle remains on the robot side.
      center = (
        s.ink_surface_center - self.config.BOARD_ROTATION @ self.config.PAD_BOTTOM
      )
      ink_y = s.ink_randomization["center_offset_board_m"][1]
      # Keep the 11.1 cm felt footprint inside the side frames, including
      # at the +/-6 cm randomization limits. Only the lead-in/out changes.
      stroke_center_limit = getattr(self.config, "BOARD_STROKE_CENTER_LIMIT_M", 0.140)
      first_line_start = np.array([0.070, min(0.100, stroke_center_limit - ink_y)])
      # Orient on the known short IK branch before moving up to the Z.
      # A direct joint-space transfer to its far corner sweeps the held tool
      # into the board; keep the subsequent Cartesian traverse well clear.
      transfer_start = np.array([-0.070, -0.120])
      first_line_clearance = (
        center
        + 0.070 * self.config.BOARD_NORMAL
        + self.config.BOARD_ROTATION[:, :2] @ transfer_start
      )
      target_wrist_rotation = self.config.WIPE_ROTATION @ self.tool_rotation_offset.T
      target_wrist_position = (
        first_line_clearance - target_wrist_rotation @ self.tool_offset
      )
      transfer_ik = s.solve_ik(
        "right",
        target_wrist_position,
        target_wrist_rotation,
        seed=grasp.BOARD_APPROACH_ARM_SEED,
        max_iterations=500,
        position_tolerance=0.0005,
        orientation_tolerance=0.006,
        posture_weight=0,
      )
      if not transfer_ik.success:
        raise RuntimeError("orient_and_transfer: short-branch target is unreachable")
      transfer_motion = self.move_arm_joints(
        transfer_ik.joint_positions, 5.0, "orient_and_transfer"
      )
      if transfer_motion["maximum_joint_speed_deg_s"] >= 70.0:
        raise RuntimeError("orient_and_transfer: arm joint speed is not smooth")
      if transfer_motion["maximum_joint_acceleration_deg_s2"] >= 100.0:
        raise RuntimeError("orient_and_transfer: arm joint acceleration is not smooth")
      self.move_tool(
        center
        + 0.070 * self.config.BOARD_NORMAL
        + self.config.BOARD_ROTATION[:, :2] @ first_line_start,
        4.0,
        self.position_above_ink_phase,
        self.config.WIPE_ROTATION,
      )
      self.move_tool(
        center
        + 0.0005 * self.config.BOARD_NORMAL
        + self.config.BOARD_ROTATION[:, :2] @ first_line_start,
        1.5,
        "approach_board",
        self.config.WIPE_ROTATION,
      )
      wipe_offset = self.position - s.object_pose("eraser")[:3]
      wipe_rotation = self.rotation.copy()
      depth = 0.0005
      # All loaded segments start/end at rest. The default remains the full Z;
      # independent variants can choose another path without changing feedback.
      path = self.build_wipe_path(
        first_line_start, ink_y=ink_y, stroke_center_limit=stroke_center_limit
      )
      offsets = []
      reliefs = []
      force_goals = []
      wipe_phases = []
      start = first_line_start
      for (
        end,
        seconds,
        relief_start,
        relief_end,
        force_start,
        force_end,
        phase,
      ) in path:
        # Every segment starts and ends at rest, with the same quintic blend
        # applied to lateral position, normal relief, and desired board load.
        delta = end - start
        for u in np.linspace(
          0, 1, round(seconds / self.config.BOARD_CONTROL_PERIOD_S) + 1
        )[1:]:
          blend = 10 * u**3 - 15 * u**4 + 6 * u**5
          offsets.append(start + blend * delta)
          reliefs.append(relief_start + blend * (relief_end - relief_start))
          force_goals.append(force_start + blend * (force_end - force_start))
          wipe_phases.append(phase)
        start = end
      normal_command = depth
      previous_force = s.mean_board_force
      cursor = 0.0
      contact_acquired = False
      path_speed = 0.0
      for step in range(8 * len(offsets)):
        # The normal admittance regulates load while the planned tangential
        # clock requests its nominal rate outside wiping. During wiping,
        # overload may pause tangential motion briefly while the normal servo
        # unloads; overload gating does not stall the initial force ramp.
        force = s.mean_board_force
        i = min(int(cursor), len(offsets) - 1)
        fraction = cursor - i
        following = min(i + 1, len(offsets) - 1)
        offset = (1 - fraction) * offsets[i] + fraction * offsets[following]
        relief = (1 - fraction) * reliefs[i] + fraction * reliefs[following]
        force_goal = (1 - fraction) * force_goals[i] + fraction * force_goals[following]
        regulated_force = force + 5.0 * max(0.0, force - previous_force)
        # Close a free-space gap quickly, then revert to the low-gain contact
        # admittance before the first measurable load.
        depth_gain = 0.000006 if force < 0.05 and force_goal > 0.2 else 0.000003
        depth = float(
          np.clip(
            depth + depth_gain * (regulated_force - force_goal),
            -0.012,
            0.005,
          )
        )
        previous_force = force
        desired_normal = depth + relief
        # Limit approach to 1 mm/s even when the relief trajectory and
        # admittance both request motion into the board.  Motion away from
        # the board remains immediate, so overloads are relieved promptly.
        normal_command = max(desired_normal, normal_command - 0.000005)
        target = (
          center
          + normal_command * self.config.BOARD_NORMAL
          + self.config.BOARD_ROTATION[:, :2] @ offset
        )
        self.position = target + wipe_offset
        self.rotation = wipe_rotation
        phase = wipe_phases[i]
        if phase == "wipe":
          contact_acquired |= force >= self.config.BOARD_PRELOAD_FORCE_N
          if not contact_acquired:
            phase = "load_board"
        if not contact_acquired and step * self.config.BOARD_CONTROL_PERIOD_S >= 8.0:
          raise RuntimeError("load_board: failed to establish preload within 8 s")
        self.tick(phase, period_s=self.config.BOARD_CONTROL_PERIOD_S)
        if step > 300 and np.max(s.remaining) <= 1e-9:
          break
        if cursor >= len(offsets) - 1:
          break
        requested_path_speed = 1.0
        if wipe_phases[i] == "wipe":
          # Slow on measured overload, not the derivative prediction used
          # by normal admittance. Gating both axes on that prediction caused
          # repeated stop/start friction transients at the Z's reversals.
          requested_path_speed = float(
            np.clip((force_goal + 0.50 - force) / 0.30, 0.0, 1.0)
          )
          if not contact_acquired:
            requested_path_speed = 0.0
        # Rate-limit the trajectory clock, not measured forces. Abruptly
        # stopping/restarting a loaded tool excites tangential stick-slip.
        path_speed += float(np.clip(requested_path_speed - path_speed, -0.01, 0.01))
        cursor = min(cursor + path_speed, len(offsets) - 1)
      self.move_tool(
        s.object_pose("eraser")[:3] + 0.07 * self.config.BOARD_NORMAL,
        1.5,
        "leave_board",
        self.config.WIPE_ROTATION,
      )
      self.move_tool(
        s.pickup_center + [0, 0, 0.15],
        3.0,
        "return",
        self.config.INITIAL_ERASER_ROTATION,
      )
      self.place_eraser()
      self.move_wrist(self.position + [0, 0, 0.16], 2.0, "retreat")
      self.move_wrist(self.position, 0.8, "verify")
      released = bool(
        s.table_force > 0.2
        and np.linalg.norm(s.object_twist("eraser")[:3]) < 0.01
        and s.forces.read(s.data).normal_force_n[5:].sum() < 0.05
      )
      if not released:
        reason = "Eraser did not settle freely on table"
      elif np.max(s.remaining) > 1e-9:
        reason = "Ink remains after bounded wiping"
    except RuntimeError as exc:
      reason = str(exc)
    grasp_peak = self.phase_peak_tactile.get("grasp", np.zeros(5))
    wipe_peak = self.phase_peak_tactile.get("wipe", np.zeros(5))
    release_final = self.phase_final_tactile.get("release", np.full(5, np.inf))
    tactile_criteria = {
      "pregrasp_finger_peak_below_0_05_n": bool(
        np.max(self.pregrasp_peak_tactile) < 0.05
      ),
      "opposed_grasp_detected": bool(
        self.grasp_tactile[0] >= 0.05 and self.grasp_tactile[1:].sum() >= 0.15
      ),
      "grasp_fingertip_peak_below_6_n": bool(np.max(grasp_peak) < 6.0),
      "wipe_fingertip_peak_below_6_n": bool(np.max(wipe_peak) < 6.0),
      "wipe_board_peak_below_3_n": bool(
        self.phase_peak_board_force.get("wipe", np.inf) < 3.0
      ),
      "no_direct_hand_board_contact": bool(s.peak_direct_hand_board_force < 1e-9),
      "released_fingertip_sum_below_0_05_n": bool(release_final.sum() < 0.05),
    }
    tactile_reasonable = all(tactile_criteria.values())
    required_pickup_phases = [*PICKUP_APPROACH_PHASES, "grasp", "lift"]
    actual_pickup_phases = self.phases[1:6]
    descent = self.commanded_descent_world_m
    pickup_sequence_valid = bool(
      actual_pickup_phases == required_pickup_phases
      and self.open_hand_target_error_rad is not None
      and self.open_hand_target_error_rad < self.config.OPEN_HAND_MAX_ERROR_RAD
      and self.above_wrist_position_error_m is not None
      and self.above_wrist_position_error_m < 0.008
      and descent is not None
      and np.linalg.norm(descent[:2]) < 1e-12
      and abs(descent[2] + self.config.PICKUP_CLEARANCE_M) < 1e-12
      and self.actual_descent_lateral_m is not None
      and self.actual_descent_lateral_m < 0.01
      and self.maximum_descent_lateral_deviation_m < 0.01
      and self.actual_descent_world_m is not None
      and abs(self.actual_descent_world_m[2] + self.config.PICKUP_CLEARANCE_M) < 0.01
      and self.pregrasp_eraser_displacement_m is not None
      and self.pregrasp_eraser_displacement_m < 0.002
    )
    if reason == "completed" and not pickup_sequence_valid:
      reason = "Pickup sequence validation failed"
    elif reason == "completed" and not tactile_reasonable:
      reason = "Tactile validation failed"
    pickup_sequence = {
      "required_phase_order": required_pickup_phases,
      "actual_phase_order": actual_pickup_phases,
      "valid": pickup_sequence_valid,
      "clearance_m": self.config.PICKUP_CLEARANCE_M,
      "open_hand_target_error_rad": self.open_hand_target_error_rad,
      "above_wrist_position_error_m": self.above_wrist_position_error_m,
      "commanded_descent_world_m": (
        None
        if self.commanded_descent_world_m is None
        else self.commanded_descent_world_m.tolist()
      ),
      "actual_descent_lateral_m": self.actual_descent_lateral_m,
      "actual_descent_world_m": (
        None
        if self.actual_descent_world_m is None
        else self.actual_descent_world_m.tolist()
      ),
      "maximum_descent_lateral_deviation_m": self.maximum_descent_lateral_deviation_m,
      "pregrasp_eraser_displacement_m": self.pregrasp_eraser_displacement_m,
      "pregrasp_peak_fingertip_force_n": self.pregrasp_peak_tactile.tolist(),
      "grasp_final_fingertip_force_n": self.grasp_tactile.tolist(),
    }
    return dict(
      success=bool(
        self.pickup_verified
        and released
        and np.max(s.remaining) <= 1e-9
        and pickup_sequence_valid
        and tactile_reasonable
        and reason == "completed"
      ),
      reason=reason,
      ink_randomization=s.ink_randomization,
      wipe_direction="top_to_bottom",
      wipe_vertices_board_m=self.config.INK_Z_VERTICES.tolist(),
      elapsed_s=float(s.data.time),
      pickup_verified=self.pickup_verified,
      released_on_table=released,
      ink_remaining=s.remaining.tolist(),
      maximum_lift_m=self.max_lift,
      peak_board_force_n=s.peak_board_force,
      peak_board_force_phase=self.peak_board_phase,
      peak_board_force_time_s=self.peak_board_time_s,
      peak_board_eraser_position_board_m=self.peak_board_eraser_position_board_m,
      peak_board_tangent_force_n=s.peak_board_tangent_force,
      peak_direct_hand_board_force_n=s.peak_direct_hand_board_force,
      cleaning=dict(
        minimum_normal_force_n=cleaning.MIN_NORMAL_FORCE_N,
        maximum_normal_force_n=None,
        board_normal_force_guard_threshold_n=(cleaning.BOARD_NORMAL_FORCE_GUARD_N),
        cleaning_model_has_upper_normal_force_gate=False,
        cleaning_can_accumulate_above_board_force_guard=True,
        board_force_guard_role=(
          "evaluation safety/out-of-distribution threshold, not an erasure "
          "efficacy ceiling"
        ),
        minimum_patch_tangent_load_n=cleaning.MIN_TANGENT_LOAD_N,
        minimum_sliding_speed_m_s=cleaning.MIN_SLIDING_SPEED_M_S,
        work_required_j=cleaning.REQUIRED_WORK_J,
        stroke_required_m=cleaning.REQUIRED_STROKE_M,
        loaded_time_required_s=cleaning.REQUIRED_LOADED_TIME_S,
        patch_work_j=s.cleaning.work_j.tolist(),
        patch_stroke_m=s.cleaning.stroke_m.tolist(),
        patch_loaded_time_s=s.cleaning.loaded_time_s.tolist(),
        remaining=s.remaining.tolist(),
      ),
      eraser_board_friction=s.model.pair_friction[
        s.model.pair("eraser_board_contact").id
      ].tolist(),
      eraser_board_solimp=s.model.pair_solimp[
        s.model.pair("eraser_board_contact").id
      ].tolist(),
      friction_impedance_ratio=float(s.model.opt.impratio),
      hand_velocity_gain=self.config.HAND_VELOCITY_GAIN,
      board_target_force_n=self.config.BOARD_TARGET_FORCE_N,
      board_preload_timeout_s=8.0,
      board_depth_gain_m_per_n_tick={
        "free_space": 0.000006,
        "in_contact": 0.000003,
      },
      maximum_board_approach_m_per_tick=0.000005,
      board_depth_control="integral_normal_admittance_with_bounded_wipe_slowdown",
      board_force_derivative_prediction=5.0,
      board_force_derivative_rectified=True,
      maximum_wipe_clock_acceleration_s_inv=2.0,
      minimum_wipe_path_speed_scale=0.0,
      wipe_slowdown_force_source="measured board force; derivative used only for normal admittance",
      wipe_slowdown_above_target_n=0.20,
      wipe_stop_above_target_n=0.50,
      contact_control_ik_tolerance_m=0.000001,
      table_approach_speed_m_s=0.002,
      table_target_support_n=float(
        s.model.body_mass[s.eraser_body] * -s.model.opt.gravity[2]
      ),
      release_force_ramp_s=1.5,
      command_interpolation_period_s=self.config.CONTROL_PERIOD_S,
      board_contact_control_period_s=self.config.BOARD_CONTROL_PERIOD_S,
      ccd_tolerance=float(s.model.opt.ccd_tolerance),
      ccd_iterations=int(s.model.opt.ccd_iterations),
      eraser_contact_margins_m={
        "handle": float(s.model.geom_margin[s.handle_id]),
        "felt": float(s.model.geom_margin[s.pad_id]),
        "board_pair": float(
          s.model.pair_margin[s.model.pair("eraser_board_contact").id]
        ),
      },
      peak_fingertip_force_n=self.peak_tactile.tolist(),
      pickup_sequence=pickup_sequence,
      tactile_validation={
        "source": "unfiltered MuJoCo solver contacts on the five right fingertip taxel pads",
        "criteria": tactile_criteria,
        "reasonable": tactile_reasonable,
        "phase_peak_fingertip_force_n": {
          phase: value.tolist() for phase, value in self.phase_peak_tactile.items()
        },
        "phase_peak_board_force_n": self.phase_peak_board_force,
      },
      initial_approach_motion=approach_motion,
      orient_transfer_motion=transfer_motion,
      phases=self.phases,
      solver_warnings=[int(w.number) for w in s.data.warning],
    )
