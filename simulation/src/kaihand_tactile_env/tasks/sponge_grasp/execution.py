"""Actuator-only right-hand pickup and placement into the right-side plate.

The sponge is never welded to the hand or repositioned after reset.  The
controller uses the shared arm/hand motors and measured finger-pad forces.
"""

from __future__ import annotations

import mujoco
import numpy as np

from kaihand_tactile_env.shared.approach import approach_waypoint
from kaihand_tactile_env.shared.simulation import HAND_JOINT_NAMES
from kaihand_tactile_env.tasks.vase_wipe.sponge import PIN

from . import config

CONTROL_PERIOD_S = 0.01
MAX_CARRY_SPONGE_DROP_M = 0.005
APPROACH_PHASES = ("open_hand", "move_above_sponge", "descend_to_sponge")
LOADED_GRASP_PHASES = ("lift", "carry_to_plate", "lower_into_plate")


class SpongeGraspExecutor:
  """Known tabletop pose for approach, tactile feedback for the actual grasp."""

  def __init__(self, simulation, observer=None):
    self.sim = simulation
    self.observer = observer
    self.target = simulation.current_pose_matrix("right")[0] - simulation.wrist_offset
    self.phase_times = {}
    self.phase_peak_tactile = {}
    self.phase_final_tactile = {}
    self.pregrasp_peak_tactile = np.zeros(5)
    self._grip_goal = None
    self._grip_progress = np.zeros(5)
    self._grip_force_filtered = np.zeros(5)
    self._grip_contact_time = np.full(5, np.nan)
    self._hold_progress_cap = None
    self._settled_grasp_check_start_s = None
    self._lift_start_s = None
    self._ring_goal_before_lift = None
    self.minimum_five_finger_force_n = np.full(5, np.inf)
    self.grasp_verified = False
    self.grasp_body = simulation.model.body("hand_r_base_link").id
    self.grasp_in_hand = np.zeros(3)
    self.max_grasp_drift_m = 0.0
    self.minimum_carry_clearance_m = np.inf
    self.maximum_carry_table_force_n = 0.0
    self.minimum_carry_thumb_force_n = np.inf
    self.minimum_carry_opposing_force_n = np.inf
    self.loaded_carry_steps = 0
    self.carry_steps = 0
    self.carry_reference_pin_z_m = None
    self.maximum_carry_sponge_drop_m = 0.0
    self.palm_down_cosine_before_descent = None
    self.maximum_approach_wrist_y_m = -np.inf
    self.minimum_approach_hand_separation_m = np.inf
    self._lower_plate_impulse_trace = []

  def _material_pin(self):
    sim = self.sim
    return sim.data.flexvert_xpos[sim._flex_start + PIN].copy()

  def _sponge_clearance(self):
    sim = self.sim
    vertices = sim.data.flexvert_xpos[
      sim._flex_start : sim._flex_start + sim._flex_count
    ]
    return float(vertices[:, 2].min() - sim.table_height)

  def _record(self, phase, state):
    sim = self.sim
    now = float(sim.data.time)
    if phase not in self.phase_times:
      self.phase_times[phase] = [now, now]
    else:
      self.phase_times[phase][1] = now
    loads = sim.forces.read(sim.data).normal_force_n[5:].copy()
    if phase in LOADED_GRASP_PHASES or (
      phase == "grasp"
      and self._settled_grasp_check_start_s is not None
      and now >= self._settled_grasp_check_start_s
    ):
      self.minimum_five_finger_force_n = np.minimum(
        self.minimum_five_finger_force_n, loads
      )
      if np.any(loads < config.MINIMUM_FIVE_FINGER_NORMAL_FORCE_N):
        raise RuntimeError(
          f"{phase} at {now:.2f}s: five-finger contact lost "
          f"({loads.round(3).tolist()} N)"
        )
    self.phase_peak_tactile[phase] = np.maximum(
      self.phase_peak_tactile.get(phase, np.zeros(5)), loads
    )
    self.phase_final_tactile[phase] = loads
    if phase in APPROACH_PHASES:
      self.pregrasp_peak_tactile = np.maximum(self.pregrasp_peak_tactile, loads)
    if phase == "move_above_sponge":
      wrist_y = float(sim.current_pose_matrix("right")[0][1])
      left_hand = sim.data.body("hand_l_base_link").xpos
      right_hand = sim.data.body("hand_r_base_link").xpos
      separation = float(np.linalg.norm(right_hand - left_hand))
      self.maximum_approach_wrist_y_m = max(self.maximum_approach_wrist_y_m, wrist_y)
      self.minimum_approach_hand_separation_m = min(
        self.minimum_approach_hand_separation_m, separation
      )
      if wrist_y > -0.08 or separation < 0.23:
        raise RuntimeError(
          f"move_above_sponge: right hand crossed toward left "
          f"(wrist y={wrist_y:.3f} m, hand separation={separation:.3f} m)"
        )
    if self.grasp_verified:
      hand_position = sim.data.xpos[self.grasp_body]
      hand_rotation = sim.data.xmat[self.grasp_body].reshape(3, 3)
      expected_pin = hand_position + hand_rotation @ self.grasp_in_hand
      drift = float(np.linalg.norm(self._material_pin() - expected_pin))
      self.max_grasp_drift_m = max(self.max_grasp_drift_m, drift)
      if drift > 0.045:
        raise RuntimeError(f"{phase}: sponge slipped from right hand ({drift:.3f} m)")
    if phase == "carry_to_plate":
      clearance = self._sponge_clearance()
      drop = float(self.carry_reference_pin_z_m - self._material_pin()[2])
      self.maximum_carry_sponge_drop_m = max(self.maximum_carry_sponge_drop_m, drop)
      self.minimum_carry_clearance_m = min(self.minimum_carry_clearance_m, clearance)
      self.maximum_carry_table_force_n = max(
        self.maximum_carry_table_force_n, float(sim.table_support_force_n)
      )
      self.minimum_carry_thumb_force_n = min(self.minimum_carry_thumb_force_n, loads[0])
      self.minimum_carry_opposing_force_n = min(
        self.minimum_carry_opposing_force_n, float(loads[1:].sum())
      )
      self.loaded_carry_steps += int(loads[0] > 0.05 and loads[1:].sum() > 0.15)
      self.carry_steps += 1
      if clearance < 0.08 or sim.table_support_force_n > 0.01:
        raise RuntimeError("carry_to_plate: sponge is not clear of the table")
      if drop > MAX_CARRY_SPONGE_DROP_M:
        raise RuntimeError(
          f"carry_to_plate at {now:.2f}s: sponge slipped downward by "
          f"{drop * 1000:.1f} mm; fingertip forces {loads.round(2).tolist()}"
        )
    if phase == "lower_into_plate":
      self._lower_plate_impulse_trace.append(
        (now, float(sim.plate_base_support_impulse_ns))
      )
    if self.observer is not None:
      self.observer(sim, phase, state)

  def _tick(self, phase):
    sim = self.sim
    if self._grip_goal is not None:
      if phase == "lift" and self._lift_start_s is None:
        self._lift_start_s = float(sim.data.time)
        self._ring_goal_before_lift = float(self._grip_goal[13])
      if self._lift_start_s is not None and phase in LOADED_GRASP_PHASES:
        u = np.clip(
          (sim.data.time - self._lift_start_s) / config.RING_LIFT_FOLLOW_SECONDS,
          0.0,
          1.0,
        )
        blend = 10 * u**3 - 15 * u**4 + 6 * u**5
        # Ring joint 2 closes only after the other pads have established a
        # grasp.  Following the retreating elastic face preserves contact
        # without the early over-compression caused by a static deep curl.
        self._grip_goal[13] = (
          self._ring_goal_before_lift
          + config.RING_LIFT_FOLLOW_EXTRA_RAD * blend
        )
      loads = sim.forces.read(sim.data).normal_force_n[5:]
      self._grip_force_filtered += 0.5 * (loads - self._grip_force_filtered)
      new_contact = np.isnan(self._grip_contact_time) & (loads > 0.03)
      self._grip_contact_time[new_contact] = sim.data.time
      touched = ~np.isnan(self._grip_contact_time)
      ramp = np.where(
        touched, np.clip((sim.data.time - self._grip_contact_time) / 0.5, 0, 1), 1.0
      )
      force_target = 0.1 + ramp * (config.GRASP_FORCE_TARGETS_N - 0.1)
      rate = np.clip(
        1.5 * (force_target - np.maximum(loads, self._grip_force_filtered)),
        -0.5,
        np.where(touched, 0.12, 0.5),
      )
      progress_cap = 1.0
      if self._hold_progress_cap is not None:
        rate = np.clip(
          1.5 * (
            config.GRASP_FORCE_TARGETS_N - np.maximum(loads, self._grip_force_filtered)
          ),
          0,
          0.04,
        )
        progress_cap = self._hold_progress_cap
      self._grip_progress = np.clip(
        self._grip_progress + rate * CONTROL_PERIOD_S, 0, progress_cap
      )
      sim.set_hand_joint_targets(
        HAND_JOINT_NAMES["right"],
        sim.open_grip
        + np.repeat(self._grip_progress, 4) * (self._grip_goal - sim.open_grip),
      )
    result = sim.solve_ik(
      "right",
      self.target + sim.wrist_offset,
      sim.wrist_rotation,
      seed=sim.arm_goal["right"],
      max_iterations=40,
    )
    if not result.success and (
      result.position_error > 0.004 or result.orientation_error > 0.05
    ):
      raise RuntimeError(f"{phase}: wrist IK error {result.position_error:.4f} m")
    sim.set_arm_joint_goal("right", result.joint_positions)
    before = float(sim.data.time)
    sim.phase = phase
    sim.step(round(CONTROL_PERIOD_S / sim.timestep))
    if (
      not np.isfinite(sim.data.qpos).all()
      or sim.data.time <= before
      or any(w.number for w in sim.data.warning)
    ):
      raise RuntimeError(f"{phase}: physics became unstable")
    state = sim.measure()
    self._record(phase, state)
    return state

  def _move(self, destination, seconds, phase):
    start = self.target.copy()
    count = max(1, round(seconds / CONTROL_PERIOD_S))
    destination = np.asarray(destination)
    for i in range(count):
      u = (i + 1) / count
      blend = 10 * u**3 - 15 * u**4 + 6 * u**5
      self.target = start + blend * (destination - start)
      self._tick(phase)

  def _shape_hand(self, destination, seconds, phase):
    sim = self.sim
    names = HAND_JOINT_NAMES["right"]
    start = np.array([sim._hand_targets["right"][name] for name in names])
    count = max(1, round(seconds / CONTROL_PERIOD_S))
    for i in range(count):
      u = (i + 1) / count
      blend = 10 * u**3 - 15 * u**4 + 6 * u**5
      sim.set_hand_joint_targets(names, start + blend * (destination - start))
      self._tick(phase)

  def _approach_from_right(self, seconds):
    """Follow a Cartesian right-side path while turning the spread palm down.

    Interpolating the seven home-to-hover joint angles directly sends the
    right hand through the robot midline.  A continuous wrist pose path keeps
    the hand on its own side and uses the same arm motors and IK as later moves.
    """
    sim = self.sim
    start_position, start_rotation = sim.current_pose_matrix("right")
    destination = sim.pickup_center + [0, 0, 0.16] + sim.wrist_offset
    q_start = np.empty(4)
    q_end = np.empty(4)
    mujoco.mju_mat2Quat(q_start, start_rotation.ravel())
    mujoco.mju_mat2Quat(q_end, sim.wrist_rotation.ravel())
    cosine = float(q_start @ q_end)
    if cosine < 0:
      q_end = -q_end
      cosine = -cosine
    angle = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    count = max(1, round(seconds / CONTROL_PERIOD_S))
    for i in range(count):
      u = (i + 1) / count
      blend = 10 * u**3 - 15 * u**4 + 6 * u**5
      position = start_position + blend * (destination - start_position)
      if angle < 1e-8:
        quaternion = q_start
      else:
        quaternion = (
          np.sin((1 - blend) * angle) * q_start
          + np.sin(blend * angle) * q_end
        ) / np.sin(angle)
      rotation_flat = np.empty(9)
      mujoco.mju_quat2Mat(rotation_flat, quaternion)
      result = sim.solve_ik(
        "right", position, rotation_flat.reshape(3, 3),
        seed=sim.arm_goal["right"], max_iterations=60,
      )
      if not result.success and (
        result.position_error > 0.004 or result.orientation_error > 0.05
      ):
        raise RuntimeError(
          f"move_above_sponge: wrist IK error {result.position_error:.4f} m"
        )
      sim.set_arm_joint_goal("right", result.joint_positions)
      sim.phase = "move_above_sponge"
      sim.step(round(CONTROL_PERIOD_S / sim.timestep))
      if not np.isfinite(sim.data.qpos).all() or any(
        warning.number for warning in sim.data.warning
      ):
        raise RuntimeError("move_above_sponge: physics became unstable")
      self._record("move_above_sponge", sim.measure())

  def run(self):
    sim = self.sim
    initial_loads = sim.forces.read(sim.data).normal_force_n[5:].copy()
    initial_bottom = self._sponge_clearance()
    vertices = sim.data.flexvert_xpos[
      sim._flex_start : sim._flex_start + sim._flex_count
    ]
    initial_height = float(np.ptp(vertices[:, 2]))
    if initial_loads.sum() > 0.05:
      raise RuntimeError("Sponge must start clear of the hand")
    if initial_bottom > 0.006 or initial_height < 0.09:
      raise RuntimeError("Sponge does not start upright on the tabletop")
    for _ in approach_waypoint(
      sim, sim.arm_goal["right"], sim.spread_grip,
      phase="open_hand", seconds=0.8,
    ):
      self._record("open_hand", sim.measure())
    initial_support = float(sim.table_support_force_n)
    if initial_support < 0.15:
      raise RuntimeError("Sponge must stand supported on the tabletop")
    names = HAND_JOINT_NAMES["right"]
    actual_hand = np.array([sim.data.qpos[sim._qpos_address[name]] for name in names])
    open_error = float(np.max(np.abs(actual_hand - sim.spread_grip)))
    if open_error > 0.08:
      raise RuntimeError(f"open_hand: pose did not settle ({open_error:.3f} rad)")
    self._approach_from_right(3.0)
    self.target = sim.current_pose_matrix("right")[0] - sim.wrist_offset
    approach_target_error = float(
      np.linalg.norm(self.target - (sim.pickup_center + [0, 0, 0.16]))
    )
    if approach_target_error > 0.01:
      raise RuntimeError("move_above_sponge: right hand missed overhead waypoint")
    # Still above the standing sponge, align the spread palm before lowering.
    self._move(sim.pickup_center + [0, 0, 0.04], 1.2, "move_above_sponge")
    palm_rotation = sim.data.xmat[self.grasp_body].reshape(3, 3)
    self.palm_down_cosine_before_descent = float(-palm_rotation[2, 1])
    if self.palm_down_cosine_before_descent < 0.98:
      raise RuntimeError("move_above_sponge: right palm is not facing down")
    before_descent = self._material_pin()
    wrist_before = sim.current_pose_matrix("right")[0]
    self._move(sim.pickup_center, 1.5, "descend_to_sponge")
    wrist_after = sim.current_pose_matrix("right")[0]
    descent_lateral_m = float(np.linalg.norm((wrist_after - wrist_before)[:2]))
    pregrasp_displacement_m = float(np.linalg.norm(self._material_pin() - before_descent))
    if self.pregrasp_peak_tactile.max() > 0.05:
      raise RuntimeError("A finger touched the sponge before grasp closure")
    if descent_lateral_m > 0.01:
      raise RuntimeError("descend_to_sponge: hand did not lower vertically")
    if pregrasp_displacement_m > 0.01:
      raise RuntimeError("Open hand displaced the sponge before grasp")
    # Only after reaching the grasp plane do the finger motors bend.
    self._shape_hand(sim.open_grip, 1.0, "grasp")
    grip = sim.grip.copy()
    grip[3] += 0.35
    grip[[6, 10, 14, 18]] += 0.12
    grip[[5, 9, 13, 17]] += 0.08
    # Seat the middle and ring pads against the sponge's far face.  In the
    # palm-down geometry their original second-joint goals left them almost
    # straight, making the sponge hang from thumb and little finger alone.
    grip[[9, 13]] += 0.20
    grip[13] += config.RING_GRASP_JOINT2_EXTRA_RAD
    grip[17] += config.PINKY_GRASP_JOINT2_EXTRA_RAD
    grip[18] += config.PINKY_GRASP_JOINT3_EXTRA_RAD
    self._grip_goal = grip
    self._settled_grasp_check_start_s = (
      float(sim.data.time)
      + config.GRASP_CLOSE_SECONDS
      - config.SETTLED_GRASP_CONTACT_SECONDS
    )
    for _ in range(round(config.GRASP_CLOSE_SECONDS / CONTROL_PERIOD_S)):
      self._tick("grasp")
    grasp_loads = sim.forces.read(sim.data).normal_force_n[5:].copy()
    if np.any(grasp_loads < config.MINIMUM_FIVE_FINGER_NORMAL_FORCE_N):
      raise RuntimeError(f"grasp: not all five fingertips contact {grasp_loads}")
    # Permit bounded preload recovery while taking the sponge's weight.
    self._hold_progress_cap = np.ones(5)
    sim.grasp_patch.enabled = True
    sim.wrist_offset = (
      sim.current_pose_matrix("right")[0] - sim.grasp_reference_position()
    )
    self.target = sim.grasp_reference_position().copy()
    hand_position = sim.data.xpos[self.grasp_body]
    hand_rotation = sim.data.xmat[self.grasp_body].reshape(3, 3)
    self.grasp_in_hand = hand_rotation.T @ (self._material_pin() - hand_position)
    self.grasp_verified = True
    lift_start_s = float(sim.data.time)
    self._move(self.target + [0, 0, 0.18], config.LIFT_SECONDS, "lift")
    lift_clearance = self._sponge_clearance()
    if lift_clearance < 0.12 or sim.table_support_force_n > 0.01:
      raise RuntimeError("lift: sponge did not leave the tabletop")
    # Hold the measured loaded hand shape while moving the sponge across the
    # tabletop.  Keep the transport level and genuinely supported by pads.
    self._hold_progress_cap = self._grip_progress.copy()
    self.carry_reference_pin_z_m = float(self._material_pin()[2])
    plate_overhead = np.r_[config.PLATE_TABLE_XY, self.target[2]]
    self._move(plate_overhead, config.CARRY_SECONDS, "carry_to_plate")
    loaded_fraction = self.loaded_carry_steps / max(self.carry_steps, 1)
    if loaded_fraction < 0.95:
      raise RuntimeError("carry_to_plate: opposing tactile grip was not sustained")
    # The sponge is compliant, so use its measured loaded bottom-to-reference
    # offset.  Command a shallow 2 mm compression into the plate, with contact
    # mechanics (not a pose reset) limiting the actual descent.
    loaded_bottom_z = float(
      sim.data.flexvert_xpos[
        sim._flex_start : sim._flex_start + sim._flex_count, 2
      ].min()
    )
    bottom_from_reference = loaded_bottom_z - sim.grasp_reference_position()[2]
    place_reference_z = (
      config.PLATE_INTERIOR_TOP_Z_M - bottom_from_reference - 0.002
    )
    if not config.PLATE_INTERIOR_TOP_Z_M + 0.065 < place_reference_z < config.PLATE_INTERIOR_TOP_Z_M + 0.12:
      raise RuntimeError("lower_into_plate: implausible loaded sponge height")
    self._move(
      np.r_[config.PLATE_TABLE_XY, place_reference_z],
      config.LOWER_TO_PLATE_SECONDS,
      "lower_into_plate",
    )
    lower_end_s = self._lower_plate_impulse_trace[-1][0]
    recent_lower = [
      sample for sample in self._lower_plate_impulse_trace
      if sample[0] >= lower_end_s - 0.25
    ]
    lower_plate_mean_support_n = (
      (recent_lower[-1][1] - recent_lower[0][1])
      / (recent_lower[-1][0] - recent_lower[0][0])
    )
    if lower_plate_mean_support_n < 0.08:
      raise RuntimeError(
        "lower_into_plate: sponge has not contacted the plate base "
        f"(last 0.25 s mean {lower_plate_mean_support_n:.3f} N)"
      )
    # Once the plate bears the sponge, open the fingers slowly while their
    # contact-dependent rolling-resistance couple naturally fades.  Cutting
    # that couple at full grip gave a larger thumb shear transient.
    self.grasp_verified = False
    self._grip_goal = None
    self._hold_progress_cap = None
    self._shape_hand(sim.spread_grip, config.RELEASE_SECONDS, "release")
    release_loads = sim.forces.read(sim.data).normal_force_n[5:]
    if release_loads.max() > 0.10:
      raise RuntimeError("release: fingers did not let go of the sponge")
    sim.grasp_patch.enabled = False
    pick_to_place_duration_s = float(sim.data.time - lift_start_s)
    if abs(pick_to_place_duration_s - config.PICK_TO_PLACE_SECONDS) > 0.02:
      raise RuntimeError("release: pick-to-place duration differs from schedule")
    self._move(self.target + [0, 0, 0.12], config.RETREAT_SECONDS, "retreat")
    settle_start_s = float(sim.data.time)
    settle_start_impulse_ns = float(sim.plate_base_support_impulse_ns)
    for _ in range(round(config.SETTLE_SECONDS / CONTROL_PERIOD_S)):
      self._tick("settle_in_plate")
    settle_duration_s = float(sim.data.time - settle_start_s)
    final_plate_mean_support_n = float(
      (sim.plate_base_support_impulse_ns - settle_start_impulse_ns)
      / settle_duration_s
    )
    final_loads = sim.forces.read(sim.data).normal_force_n[5:].copy()
    final_vertices = sim.data.flexvert_xpos[
      sim._flex_start : sim._flex_start + sim._flex_count
    ]
    final_sponge_xy = final_vertices[:, :2].mean(axis=0)
    final_plate_offset = float(np.linalg.norm(final_sponge_xy - config.PLATE_TABLE_XY))
    final_bottom_z = float(final_vertices[:, 2].min())
    if final_plate_offset > 0.035:
      raise RuntimeError("settle_in_plate: sponge missed plate center")
    if final_plate_mean_support_n < 0.20 or sim.table_support_force_n > 0.05:
      raise RuntimeError("settle_in_plate: sponge is not supported by the plate base")
    if final_loads.max() > 0.10:
      raise RuntimeError("settle_in_plate: right fingertips still hold the sponge")
    if not config.PLATE_INTERIOR_TOP_Z_M - 0.005 <= final_bottom_z <= config.PLATE_INTERIOR_TOP_Z_M + 0.015:
      raise RuntimeError("settle_in_plate: sponge bottom is not on the plate")
    return {
      "success": True,
      "phase_times_s": self.phase_times,
      "initial_table_support_force_n": initial_support,
      "initial_sponge_height_m": initial_height,
      "initial_sponge_bottom_clearance_m": initial_bottom,
      "initial_fingertip_normal_force_n": initial_loads.tolist(),
      "open_hand_target_error_rad": open_error,
      "overhead_target_error_m": approach_target_error,
      "descent_lateral_m": descent_lateral_m,
      "pregrasp_sponge_displacement_m": pregrasp_displacement_m,
      "pregrasp_peak_fingertip_normal_force_n": self.pregrasp_peak_tactile.tolist(),
      "grasp_fingertip_normal_force_n": grasp_loads.tolist(),
      "minimum_closed_grasp_to_placement_fingertip_force_n": (
        self.minimum_five_finger_force_n.tolist()
      ),
      "lift_clearance_m": lift_clearance,
      "pick_to_place_duration_s": pick_to_place_duration_s,
      "carry_samples": self.carry_steps,
      "carry_opposing_load_fraction": loaded_fraction,
      "carry_minimum_clearance_m": self.minimum_carry_clearance_m,
      "carry_maximum_sponge_drop_m": self.maximum_carry_sponge_drop_m,
      "carry_maximum_table_support_force_n": self.maximum_carry_table_force_n,
      "carry_minimum_thumb_force_n": float(self.minimum_carry_thumb_force_n),
      "carry_minimum_other_fingers_force_n": self.minimum_carry_opposing_force_n,
      "final_fingertip_normal_force_n": final_loads.tolist(),
      "lower_plate_base_mean_support_force_n": lower_plate_mean_support_n,
      "final_plate_base_mean_support_force_n": final_plate_mean_support_n,
      "final_plate_base_support_force_n": sim.plate_base_support_force_n,
      "final_sponge_plate_center_offset_m": final_plate_offset,
      "final_sponge_bottom_z_m": final_bottom_z,
      "maximum_grasp_drift_m": self.max_grasp_drift_m,
      "palm_down_cosine_before_descent": self.palm_down_cosine_before_descent,
      "maximum_approach_right_wrist_y_m": self.maximum_approach_wrist_y_m,
      "minimum_approach_hand_separation_m": self.minimum_approach_hand_separation_m,
      "phase_peak_fingertip_normal_force_n": {
        name: values.tolist() for name, values in self.phase_peak_tactile.items()
      },
    }
