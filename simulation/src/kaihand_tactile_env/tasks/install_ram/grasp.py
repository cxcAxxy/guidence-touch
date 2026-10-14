"""Opposed thin-board pinch fitted to the shared thumb/index tactile surfaces."""

from dataclasses import dataclass

import mujoco
import numpy as np

from . import config


@dataclass(frozen=True)
class RamGrasp:
  wrist_position: np.ndarray
  wrist_rotation: np.ndarray
  arm_joints: np.ndarray
  open_hand: np.ndarray
  contact_hand: np.ndarray
  approach_hand: np.ndarray


def calibrated_grasp(simulation) -> RamGrasp:
  """Return robot targets around the stationary vertical DIMM in its cradle.

  The calibrated palm transform and eight finger angles place the opposing
  tactile surface centroids across the thin PCB/chip envelope. The remaining
  fingers curl away. Measured-force preload accommodates the task-configured chip thickness.
  These are robot configurations, never object constraints.
  """
  position, rotation = simulation.current_pose_matrix("right")
  hand = simulation.model.body("hand_r_base_link").id
  hand_rotation = simulation.data.xmat[hand].reshape(3, 3)
  ee_to_hand = rotation.T @ hand_rotation
  hand_offset = rotation.T @ (simulation.data.xpos[hand] - position)
  desired_hand = np.array(
    [
      [-0.83201658, 0.55475076, 0.0],
      [-0.24190668, -0.36281225, -0.89991579],
      [-0.49922897, -0.74874486, 0.43606372],
    ]
  )
  # Remove the final printed digits' roundoff before using it as a rotation.
  u, _, vt = np.linalg.svd(desired_hand)
  # Approach the opposite face with a forehand pinch. This calibrated
  # orientation admits a ~102-degree short turn from shared home, instead
  # of the old ~201-degree sweep through the other wrist IK branch.
  pitch = np.deg2rad(-80.0)
  cp, sp = np.cos(pitch), np.sin(pitch)
  desired_hand = (
    np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    @ np.diag([-1.0, -1.0, 1.0])
    @ (u @ vt)
  )
  target_rotation = desired_hand @ ee_to_hand.T
  # Pinch near the mass centre: an off-centre point accumulates gravity-driven
  # rotation about the opposing-pad axis and can turn the palm upwards.
  local_grasp = config.RAM_GRASP_LOCAL_M.copy()
  local_grasp[0] = 0.0
  ram_body = simulation.model.body("ram").id
  center = (
    simulation.data.xpos[ram_body]
    + simulation.data.xmat[ram_body].reshape(3, 3) @ local_grasp
  )
  local_center = np.array([0.03109731, 0.09336745, -0.09115383])
  target_position = center - desired_hand @ local_center - target_rotation @ hand_offset
  damping = simulation.ik_damping
  try:
    simulation.ik_damping = 0.004
    arm = simulation.solve_ik(
      "right",
      target_position,
      target_rotation,
      # Preserve the calibrated IK branch independently of shared reset pose.
      seed=np.deg2rad([-55, -65, 70, -60, 120, 0, 0]),
      max_iterations=500,
      position_tolerance=2e-5,
      orientation_tolerance=3e-4,
      posture_weight=0,
    )
  finally:
    simulation.ik_damping = damping
  if not arm.success:
    raise RuntimeError(f"RAM pickup wrist IK failed: {arm.position_error:.4g} m")
  contact = np.r_[
    [
      0.14358300671080104,
      1.2741323585345197,
      0.591170748459486,
      0.10267425249123445,
      -0.26,
      1.4636779833242912,
      0.2072302821599501,
      -0.175,
    ],
    [0.0, 0.5, 1.5, 1.5] * 3,
  ]
  scratch = simulation.ik_data
  scratch.qpos[:] = simulation.data.qpos
  scratch.qpos[simulation._arm_qpos["right"]] = arm.joint_positions
  scratch.qpos[simulation._hand_qpos["right"]] = contact
  scratch.qpos[simulation._thumb_joint6_qpos["right"]] = contact[3]
  mujoco.mj_forward(simulation.model, scratch)
  targets = scratch.site_xpos[simulation._fingertip_site_ids["right"]].copy()
  targets[0, 1] += 0.005
  targets[1, 1] -= 0.005
  opening = simulation.solve_hand_ik(
    "right",
    targets,
    seed=contact,
    arm_joint_positions=arm.joint_positions,
    max_iterations=160,
    position_tolerance=1e-5,
    posture_weight=0,
  )
  if opening.position_error > 0.001:
    raise RuntimeError(f"RAM pickup finger IK failed: {opening.position_error:.4g} m")
  # The old 5 mm-per-side opening is a fine pinch waypoint, not sufficient
  # travel clearance. Approach with 20 mm per side and close only at the DIMM.
  wide_targets = targets.copy()
  wide_targets[0, 1] += 0.015
  wide_targets[1, 1] -= 0.015
  wide = simulation.solve_hand_ik(
    "right",
    wide_targets,
    seed=opening.joint_positions,
    arm_joint_positions=arm.joint_positions,
    max_iterations=300,
    position_tolerance=1e-5,
    posture_weight=0,
  )
  if wide.position_error > 0.001:
    raise RuntimeError(f"RAM approach finger IK failed: {wide.position_error:.4g} m")
  return RamGrasp(
    target_position,
    target_rotation,
    arm.joint_positions,
    opening.joint_positions,
    contact,
    wide.joint_positions,
  )
