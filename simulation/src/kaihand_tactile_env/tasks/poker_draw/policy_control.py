"""Contact-conditioned low-level control for model wrist/fingertip targets.

No expert trajectories, phase timestamps, source qpos, card teleports or welds.
Reuses the middle preset's Cartesian servo and signed normal-force integrator.
Mode transitions are inferred from live contact and the policy's target changes;
they are NOT claimed to reproduce the expert's internal state machine exactly.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ...shared.evaluation_guards import task_guard
from .mid_full import MID_FORCE_PER_FINGER_N, MID_FORCE_SETTINGS
from .pressure_window import PressureWindowForceController
from .task import PokerDrawExecutor

CONTROLLER_VERSION = "poker-contact-feedback-v2"
PENETRATION_GUARD_THRESHOLD_M = 0.0006
FINGERS = ("index", "middle", "ring", "pinky")


def make_force_controller(timestep):
  return PressureWindowForceController(
    4,
    target_force_n=MID_FORCE_PER_FINGER_N,
    timestep=timestep,
    update_period_s=timestep,
    filter_time_constant_s=0.010,
    integral_gain_rad_per_n_s=0.10,
    maximum_offset_rad=np.deg2rad(5.0),
    maximum_offset_rate_rad_s=np.deg2rad(1.5),
    contact_force_n=0.125,
    contact_recovery_rate_rad_s=0.01,
    force_deadband_n=0.005,
  )


class PokerPolicyController:
  """Owns feedback residuals, not the policy's trajectory or success label."""

  # Reuse read-only force/geometry extraction and the identical q2/-q3 linkage.
  _current_card_face_contact_details = (
    PokerDrawExecutor._current_card_face_contact_details
  )
  _current_card_finger_normal_forces = (
    PokerDrawExecutor._current_card_finger_normal_forces
  )
  _activate_press_force_controller = PokerDrawExecutor._activate_press_force_controller
  _apply_press_force_offsets = PokerDrawExecutor._apply_press_force_offsets
  _card_table_clearance = PokerDrawExecutor._card_table_clearance
  _card_rotation = PokerDrawExecutor._card_rotation

  def __init__(self, simulation, *, penetration_guard_enabled=True):
    if simulation.scene != "poker-draw" or simulation.drive_limit_n is not None:
      raise ValueError("fresh middle-force poker simulation required")
    self.sim = simulation
    self.penetration_guard_enabled = bool(penetration_guard_enabled)
    self.maximum_supported_table_penetration_m = 0.0
    self.first_penetration_limit_exceeded_s = None
    self.mode = "approach"
    self._press_controller = make_force_controller(simulation.timestep)
    self._press_controller_active = False
    self.base = None
    self.target_position = None
    self.target_rotation = None
    self.reference_position = None
    self.reference_rotation = None
    self.loaded_time = self.loss_time = self.near_time = 0.0
    self.max_low_load_s = 0.0
    self.transitions = []
    self.samples = 0
    self.peak_forces = np.zeros(5)
    self.force_sum = np.zeros(4)
    self.force_steps = 0
    self.near_diagnostic = {}
    self.pinch_residual = np.zeros(5)
    self.pinch_filtered_force = np.zeros(5)
    self.pinch_filter_initialized = False
    self.missing = np.zeros(5, dtype=int)
    self._card_id = simulation.model.geom("card_core_geom").id
    self._table_id = simulation.model.geom("poker_table_top").id
    self._pad_ids = [
      simulation.model.geom(f"hand_r_{finger}_link4_tactile_pad_col").id
      for finger in FINGERS
    ]

  def _transition(self, mode, reason):
    self.transitions.append(
      {
        "time": float(self.sim.data.time),
        "from": self.mode,
        "to": mode,
        "reason": reason,
      }
    )
    self.mode = mode

  def before_command(self):
    # Do not accumulate a residual into the next IK posture reference.
    if self.base is not None:
      self.sim.set_hand_joint_targets(tuple(self.base), tuple(self.base.values()))

  def after_command(self, targets, index):
    self.base = self.sim._hand_targets["right"].copy()
    self.target_position = targets.site_positions_world[index].copy()
    self.target_rotation = targets.site_rotations_world[index].copy()
    if self.mode in {"press", "slide"}:
      self._press_joint2_base = np.array(
        [self.base[f"hand_r_{f}_joint2"] for f in FINGERS]
      )
      self._press_joint3_base = np.array(
        [self.base[f"hand_r_{f}_joint3"] for f in FINGERS]
      )
      # Withdrawal/reorientation belongs to the model, never a recorded phase.
      angle = np.arccos(
        np.clip(
          (np.trace(self.reference_rotation.T @ self.target_rotation) - 1) / 2, -1, 1
        )
      )
      withdrawal = self.target_position[2] - self.reference_position[2] > 0.008
      if withdrawal or angle > np.deg2rad(25):
        if self.mode == "slide":
          self.sim.end_cartesian_drive()
        self._press_controller_active = False
        self._transition("manipulate", "policy withdrawal/reorientation")
      else:
        if self.mode == "slide":
          self.sim.set_cartesian_drive_position(self.target_position)
          self.sim._cartesian_drive["rotation"] = self.target_rotation.copy()
        self._apply_press_force_offsets(self._press_controller.offsets_rad)
    if self.mode == "pinch":
      self._apply_pinch()

  def _supported(self):
    for index, contact in enumerate(self.sim.data.contact):
      if {int(contact.geom1), int(contact.geom2)} == {self._card_id, self._table_id}:
        wrench = np.zeros(6)
        mujoco.mj_contactForce(self.sim.model, self.sim.data, index, wrench)
        if wrench[0] > 1e-5:
          return True
    return False

  def _near_flat_pads(self):
    # Proximity only arms the force servo; it does not fake contact or pressure.
    card_normal = self._card_rotation()[:, 2]
    if card_normal[2] < 0.94:
      return False
    distances = [
      mujoco.mj_geomDistance(
        self.sim.model, self.sim.data, pad, self._card_id, 0.004, None
      )
      for pad in self._pad_ids
    ]
    pad_axes = [
      self.sim.data.xmat[self.sim.model.body(f"hand_r_{f}_link4").id].reshape(3, 3)[
        :, 2
      ]
      for f in FINGERS
    ]
    flat = all(
      abs(float(np.dot(axis, card_normal))) < np.sin(np.deg2rad(21))
      for axis in pad_axes
    )
    self.near_diagnostic = {
      "pad_distance_m": [float(d) for d in distances],
      "pad_plane_angle_deg": [
        float(np.rad2deg(np.arcsin(np.clip(abs(axis @ card_normal), 0, 1))))
        for axis in pad_axes
      ],
    }
    return flat and max(distances) < 0.002

  def after_step(self):
    self.samples += 1
    top, bottom, forces = self._current_card_face_contact_details()
    fn = np.array([forces[f] for f in FINGERS])
    self.peak_forces = np.maximum(self.peak_forces, [forces["thumb"], *fn])
    supported = self._supported()
    supported_penetration_m = (
      max(0.0, -float(self._card_table_clearance())) if supported else 0.0
    )
    self.maximum_supported_table_penetration_m = max(
      self.maximum_supported_table_penetration_m, supported_penetration_m
    )
    if (
      supported_penetration_m > PENETRATION_GUARD_THRESHOLD_M
      and self.first_penetration_limit_exceeded_s is None
    ):
      self.first_penetration_limit_exceeded_s = float(self.sim.data.time)
    task_guard(
      self.sim,
      self.penetration_guard_enabled
      and supported_penetration_m > PENETRATION_GUARD_THRESHOLD_M,
      "supported_table_penetration",
      "card penetrated supported tabletop beyond 0.6 mm",
      measured=supported_penetration_m,
      threshold=PENETRATION_GUARD_THRESHOLD_M,
      recovery="continue_recording_without_state_repair",
    )
    if self.base is None:
      return
    if self.mode == "approach":
      self.near_time = (
        self.near_time + self.sim.timestep
        if supported and self._near_flat_pads()
        else 0.0
      )
      if self.near_time >= 0.020 - 1e-10:
        self._activate_press_force_controller()
        self.reference_position = self.target_position.copy()
        self.reference_rotation = self.target_rotation.copy()
        self._transition("press", "flat pads near supported card")
    if self.mode in {"press", "slide"}:
      offsets, updated = self._press_controller.observe(fn)
      if updated:
        self._apply_press_force_offsets(offsets)
      self.force_sum += fn
      self.force_steps += 1
      self.loaded_time = (
        self.loaded_time + self.sim.timestep if np.all(fn >= 0.125) else 0.0
      )
      if self.mode == "press" and self.loaded_time >= 0.080 - 1e-10:
        self.sim.begin_cartesian_drive(MID_FORCE_SETTINGS.drive_limit_n)
        self.sim.set_cartesian_drive_position(self.target_position)
        self.sim._cartesian_drive["rotation"] = self.target_rotation.copy()
        self._transition("slide", "four fingers loaded for 80 ms")
      if self.mode == "slide":
        self.loss_time = (
          self.loss_time + self.sim.timestep
          if np.count_nonzero(fn >= 0.125) < 3
          else 0.0
        )
        self.max_low_load_s = max(self.max_low_load_s, self.loss_time)
        task_guard(
          self.sim,
          self.loss_time >= 0.30 - 1e-10,
          "draw_contact_loss",
          "multiple draw fingers below load for 0.30 s",
          measured=float(np.count_nonzero(fn >= 0.125)),
          threshold=3.0,
          recovery="continue_recording_with_policy_and_force_controller",
        )
      else:
        press_elapsed = float(self.sim.data.time) - self.transitions[-1]["time"]
        task_guard(
          self.sim,
          press_elapsed > 12.0,
          "press_timeout",
          "policy press failed to establish four-finger load in 12 s",
          measured=press_elapsed,
          threshold=12.0,
          recovery="continue_recording_with_policy_and_force_controller",
        )
    opposed = bool(
      top and "thumb" in bottom and sum(fn) >= 0.05 and forces["thumb"] >= 0.05
    )
    if self.mode in {"approach", "manipulate"} and opposed:
      self._transition("pinch", "measured opposed face contacts")
    if self.mode == "pinch":
      measured = np.array([forces["thumb"], *fn])
      if not self.pinch_filter_initialized:
        self.pinch_filtered_force = measured.copy()
        self.pinch_filter_initialized = True
      alpha = self.sim.timestep / (0.050 + self.sim.timestep)
      self.pinch_filtered_force += alpha * (measured - self.pinch_filtered_force)
    if (
      self.mode == "pinch"
      and self.samples % max(1, round(0.020 / self.sim.timestep)) == 0
    ):
      present = np.array(["thumb" in bottom, *[f in top for f in FINGERS]])
      self.missing = np.where(present, 0, self.missing + 1)
      # Measured next-pose targets are not the expert's servo commands. A
      # one-sided close reflex can wind up against that moving IK reference.
      # Unload excess force as well as restoring contact, at the same slow
      # 0.02deg/20ms rate. Bands, not exact-force success criteria.
      measured = self.pinch_filtered_force
      lower_band = np.array([0.8, 0.2, 0.2, 0.2, 0.2])
      upper_band = np.array([1.6, 0.4, 0.4, 0.4, 0.4])
      error = np.maximum(lower_band - measured, 0.0) - np.maximum(
        measured - upper_band, 0.0
      )
      # Filter solver chatter, and do not close against a momentarily lost
      # contact until that absence persists for 60 ms.
      error[(~present) & (self.missing < 3) & (error > 0)] = 0.0
      self.pinch_residual += np.clip(
        0.05 * error * 0.020, -np.deg2rad(0.020), np.deg2rad(0.020)
      )
      self.pinch_residual = np.clip(
        self.pinch_residual,
        np.deg2rad([-5, -5, -5, -5, -5]),
        np.deg2rad([1, 3, 3, 3, 3]),
      )
      self._apply_pinch()

  def _apply_pinch(self):
    names = ["hand_r_thumb_joint5", *[f"hand_r_{f}_joint2" for f in FINGERS]]
    self.sim.set_hand_joint_targets(
      names, [self.base[n] + d for n, d in zip(names, self.pinch_residual, strict=True)]
    )

  def report(self):
    return {
      "version": CONTROLLER_VERSION,
      "mode": self.mode,
      "transitions": self.transitions,
      "pressure_target_per_finger_n": 0.5,
      "maximum_multi_low_load_s": self.max_low_load_s,
      "peak_force_thumb_first_n": self.peak_forces.tolist(),
      "press_mean_force_n": (self.force_sum / max(1, self.force_steps)).tolist(),
      "force_steps": self.force_steps,
      "near_diagnostic": self.near_diagnostic,
      "pinch_residual_degrees": np.rad2deg(self.pinch_residual).tolist(),
      "pinch_force_bands_n_thumb_first": [[0.8, 1.6], *[[0.2, 0.4]] * 4],
      "expert_trajectory_used": False,
      "source_phases_used": False,
      "full_training_controller_equivalent": False,
    }

  def terminal_metrics(self):
    """Read-only endpoint measurements; no full-task success assertion."""
    top, bottom, forces = self._current_card_face_contact_details()
    position = self.sim.object_pose("card")[:3]
    face_normal = -self._card_rotation()[:, 2]
    camera_id = self.sim.model.camera("head").id
    to_camera = self.sim.data.cam_xpos[camera_id] - position
    alignment = float(face_normal @ to_camera / max(1e-9, np.linalg.norm(to_camera)))
    return {
      "top_contacts": sorted(top),
      "bottom_contacts": sorted(bottom),
      "force_n": forces,
      "opposed_contact": bool(
        top
        and "thumb" in bottom
        and sum(forces[f] for f in FINGERS) >= 0.05
        and forces["thumb"] >= 0.05
      ),
      "four_finger_opposition": bool(set(FINGERS).issubset(top) and "thumb" in bottom),
      "face_to_head_cosine": alignment,
      "face_to_robot_cosine": float(-face_normal[0]),
      "inspection_position_error_m": float(
        np.linalg.norm(position - [0.55, -0.1, 1.05])
      ),
      "card_twist": self.sim.object_twist("card").tolist(),
      "scope": "instantaneous endpoint, not sustained task acceptance",
    }
