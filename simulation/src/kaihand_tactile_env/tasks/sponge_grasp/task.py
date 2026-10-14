"""Shared KaiHand moves the vase-wipe elastic sponge into a right-side plate.

The sponge is a free volumetric flex.  Only actuators and contact forces move it
after reset; there is no hand attachment, pose servo, or hidden support force.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from ...shared.evaluation_guards import (
  FatalSimulationError,
  is_observe_mode,
  mark_physics_invalid,
  record_guard_event,
  task_guard,
)
from ...shared.simulation import HAND_JOINT_NAMES, ArmHandSimulation
from ..vase_wipe.sponge import PIN, GraspPatchFriction, SpongeTactileProvider, points
from . import config


@dataclass(frozen=True)
class GraspState:
  timestamp: float
  sponge_height_m: float
  table_support_force_n: float
  deformation_mm: float
  fingertip_normal_force_n: tuple[float, ...]
  fingertip_tangent_force_n: tuple[float, ...]


def _external_flex_contact_mask(geom, elem):
  """Ignore MuJoCo's internal inversion-prevention tetrahedron constraints."""
  geom = np.asarray(geom)
  elem = np.asarray(elem)
  return ~(np.all(geom < 0, axis=1) & (np.sum(elem >= 0, axis=1) == 1))


class SpongeGraspSimulation(ArmHandSimulation):
  def __init__(
    self, *, position_seed: int | None = None, randomize_xy: bool = False, **kwargs
  ):
    if position_seed is not None and position_seed < 0:
      raise ValueError("position_seed must be nonnegative")
    if randomize_xy and position_seed is None:
      raise ValueError("position_seed is required for randomized sponge placement")
    self.position_seed = position_seed
    self.randomize_xy = bool(randomize_xy)
    self.xy_offset_low_m = (
      config.COLLECTION_XY_OFFSET_LOW_M.copy() if randomize_xy else np.zeros(2)
    )
    self.xy_offset_high_m = (
      config.COLLECTION_XY_OFFSET_HIGH_M.copy() if randomize_xy else np.zeros(2)
    )
    self.sponge_xy_offset_m = np.zeros(2)
    self.sponge_initial_xy_m = config.SPONGE_TABLE_XY.copy()
    super().__init__(scene=config.SCENE_NAME, **kwargs)
    self.model.opt.timestep = self.timestep = config.PHYSICS_TIMESTEP_S
    self.forces = SpongeTactileProvider(self.model)
    self.evaluation_tactile_provider = self.forces
    # All changes are applied to this task's compiled model, never shared MJCF.
    self.model.geom_solref[:] = self.model.flex_solref[:] = (
      config.SPONGE_CONTACT_TIME_S,
      1.0,
    )
    self.model.geom_solimp[:] = (
      config.SPONGE_CONTACT_IMPEDANCE,
      config.SPONGE_CONTACT_IMPEDANCE,
      0.001,
      0.5,
      2.0,
    )
    self.model.flex_solimp[:] = (
      config.SPONGE_SELF_CONTACT_IMPEDANCE,
      config.SPONGE_SELF_CONTACT_IMPEDANCE,
      0.001,
      0.5,
      2.0,
    )
    self.model.geom_priority[:] = 1
    self.model.flex_priority[:] = 0
    self.model.geom_friction[:, 0] = np.maximum(self.model.geom_friction[:, 0], 1.1)
    self.model.geom_margin[:] = config.SPONGE_CONTACT_MARGIN_M / 2
    self.model.flex_margin[:] = config.SPONGE_CONTACT_MARGIN_M / 2
    self.model.flex_damping[:] = config.SPONGE_ELASTIC_DAMPING_S
    self.model.flex_selfcollide[:] = mujoco.mjtFlexSelf.mjFLEXSELF_AUTO
    self.model.opt.tolerance = 1e-10
    # Soft flex-pad contacts otherwise creep at nearly constant speed under
    # the sponge's weight, despite a balanced vertical contact force.  The
    # task-local no-slip solve keeps the suspended grasp genuinely supported
    # by friction; it does not attach or reposition the sponge.
    self.model.opt.noslip_iterations = 20
    self._hand_geom_ids = {
      i
      for i in range(self.model.ngeom)
      if self.model.body(int(self.model.geom_bodyid[i])).name.startswith("hand_r_")
    }
    self.probes = None
    self.physics_observer = None
    self.grasp_patch = GraspPatchFriction(self.model, self._hand_geom_ids)

  @property
  def observation_time(self):
    return float(self.data.time)

  def reset(self, **kwargs):
    super().reset(**kwargs)
    self._flex_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_FLEX, "sponge_volume"
    )
    self._flex_start = int(self.model.flex_vertadr[self._flex_id])
    self._flex_count = int(self.model.flex_vertnum[self._flex_id])
    self._flex_bodies = self.model.flex_vertbodyid[
      self._flex_start : self._flex_start + self._flex_count
    ].copy()
    self._rest_points = points()
    adr = int(self.model.flex_elemdataadr[self._flex_id])
    count = int(self.model.flex_elemnum[self._flex_id]) * 4
    self._tetrahedra = self.model.flex_elem[adr : adr + count].reshape(-1, 4)
    rest = self._rest_points[self._tetrahedra]
    self._rest_volumes = np.linalg.det(rest[:, 1:] - rest[:, :1])
    self.minimum_element_volume_ratio = 1.0
    self.minimum_sponge_contact_distance_m = None
    self.minimum_hand_table_distance_m = None
    self.penetration_count = 0
    self.table_support_force_n = 0.0
    self.plate_support_force_n = 0.0
    self.plate_base_support_force_n = 0.0
    self.plate_base_support_impulse_ns = 0.0
    self._table_id = self.model.geom("tabletop").id
    self._plate_base_id = self.model.geom("plate_base").id
    self._plate_geom_ids = {
      self.model.geom(name).id for name in config.SCENE_GEOM_NAMES
    }
    self.table_height = float(
      self.data.geom_xpos[self._table_id, 2] + self.model.geom_size[self._table_id, 2]
    )
    self._hand_geom_mask = np.array(
      [
        self.model.body(int(b)).name.startswith("hand_r_")
        for b in self.model.geom_bodyid
      ]
    )
    self._set_tabletop_preset()
    # The shared hand-home thumb intersects its own palm slightly in this
    # contact-rich scene.  Without a task-local settled reset, physics pushes
    # both thumbs open visibly during the first few frames, even though no
    # thumb movement was commanded.  Begin at the measured unloaded posture
    # instead; arm home, all other hand joints, and the hand model stay shared.
    for side in ("left", "right"):
      names = HAND_JOINT_NAMES[side][:4]
      self.set_hand_joint_targets(names, (0.0, 0.65, 0.20, 0.0))
      for name, value in zip(names, (0.0, 0.68, 0.20, 0.0), strict=True):
        self.data.qpos[self._qpos_address[name]] = value
        self.data.qvel[self._qvel_address[name]] = 0.0
    mujoco.mj_forward(self.model, self.data)
    if getattr(self, "grasp_patch", None) is not None:
      self.grasp_patch.enabled = False
      self.grasp_patch.reference = None
      self.grasp_patch.torque_world_n_m[:] = 0
      self.grasp_patch.normal_load_n = 0.0

  def _set_tabletop_preset(self):
    position, rotation = self.current_pose_matrix("right")
    hand_id = self.model.body("hand_r_base_link").id
    hand_rotation = self.data.xmat[hand_id].reshape(3, 3)
    ee_to_hand = rotation.T @ hand_rotation
    hand_offset = rotation.T @ (self.data.xpos[hand_id] - position)
    angle = np.arctan2(0.604, 0.797)
    target_hand_rotation = np.array(
      [
        [0, -np.sin(angle), -np.cos(angle)],
        [1, 0, 0],
        [0, -np.cos(angle), np.sin(angle)],
      ]
    )
    # Local +Y is the palm-facing direction of the shared right hand.  Rotate
    # that axis exactly onto world -Z before moving over the upright sponge.
    # A small world yaw retains the reachable short right-arm IK branch.
    yaw = config.PICKUP_HAND_YAW_OFFSET_RAD
    world_yaw = np.array(
      [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    world_tilt = np.array(
      [[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]]
    )
    target_hand_rotation = world_yaw @ world_tilt @ target_hand_rotation
    self.nominal_sponge_rotation = np.eye(3)
    self.sponge_xy_offset_m = (
      np.random.default_rng(self.position_seed).uniform(
        self.xy_offset_low_m, self.xy_offset_high_m
      )
      if self.randomize_xy else np.zeros(2)
    )
    self.sponge_initial_xy_m = config.SPONGE_TABLE_XY + self.sponge_xy_offset_m
    centre = np.r_[
      self.sponge_initial_xy_m,
      self.table_height + 0.001 - self._rest_points[:, 2].min(),
    ]
    self.pickup_center = centre.copy()
    material_positions = self._rest_points + centre
    self.set_object_pose("sponge", material_positions[PIN])
    for i, point in enumerate(material_positions):
      if i == PIN:
        continue
      body_id = self._flex_bodies[i]
      for j in range(3):
        joint = self.model.joint(f"sponge_node_{i}_{j}")
        self.data.qpos[joint.qposadr[0]] = point[j] - self.model.body_pos[body_id, j]
    mujoco.mj_forward(self.model, self.data)
    self.wrist_rotation = target_hand_rotation @ ee_to_hand.T
    self.wrist_offset = (
      -target_hand_rotation @ np.array([-0.00254, 0.06557, -0.11947])
      - self.wrist_rotation @ hand_offset
    )
    solution = self.solve_ik(
      "right",
      centre + [0, 0, 0.16] + self.wrist_offset,
      self.wrist_rotation,
      seed=config.ARM_HOME["right"],
      max_iterations=300,
    )
    if not solution.success and (
      solution.position_error > 0.002 or solution.orientation_error > 0.05
    ):
      raise RuntimeError(f"sponge hover IK failed: {solution}")
    self.approach_arm = solution.joint_positions.copy()
    self.grip = np.array(
      [
        0.10848, 1.745, 0.698, 0.25062,
        -0.05, 0.41975, 1.12822, 0.29551,
        0, 0.06992, 1.30959, 0.12397,
        0, 0.00582, 1.35305, -0.03040,
        0.1, 0.51136, 0.63303, 0.50484,
      ]
    )
    self.open_grip = self.grip.copy()
    self.open_grip[2:4] = 0.05
    self.open_grip[[5, 9, 13, 17]] *= 0.5
    self.open_grip[[6, 10, 14, 18]] = 0.1
    self.open_grip[[7, 11, 15, 19]] = 0.0
    # The arm and hand still reset to the shared neutral pose. During the
    # explicit opening phase the thumb settles at its physically reachable
    # unloaded spread posture in this soft-contact scene; four fingers stay
    # straight and the sponge is still untouched.
    self.spread_grip = np.zeros(20)
    self.spread_grip[:4] = [0.0, 0.65, 0.20, 0.0]

  def _before_physics_step(self):
    if getattr(self, "grasp_patch", None) is not None:
      self.grasp_patch.apply(self.data)

  def step(self, steps=1):
    for _ in range(max(1, int(steps))):
      super().step()
      mujoco.mj_forward(self.model, self.data)
      self._update_support_and_integrity()
      if getattr(self, "physics_observer", None) is not None:
        self.physics_observer(self)

  def _update_support_and_integrity(self):
    contacts = self.data.contact
    wrench = np.zeros(6)
    self.table_support_force_n = 0.0
    self.plate_support_force_n = 0.0
    self.plate_base_support_force_n = 0.0
    for i, contact in enumerate(contacts):
      if contact.flex[0] == self._flex_id:
        rigid_geom = int(contact.geom2)
      elif contact.flex[1] == self._flex_id:
        rigid_geom = int(contact.geom1)
      else:
        continue
      if rigid_geom == self._table_id or rigid_geom in self._plate_geom_ids:
        mujoco.mj_contactForce(self.model, self.data, i, wrench)
        normal_force = max(0.0, float(wrench[0]))
        if rigid_geom == self._table_id:
          self.table_support_force_n += normal_force
        else:
          self.plate_support_force_n += normal_force
          if rigid_geom == self._plate_base_id:
            self.plate_base_support_force_n += normal_force
    # Accumulate at 4 kHz; an individual sample of a resting soft contact
    # can be zero even when its time-averaged force supports the full weight.
    self.plate_base_support_impulse_ns += (
      self.plate_base_support_force_n * self.timestep
    )
    verts = self.data.flexvert_xpos[
      self._flex_start : self._flex_start + self._flex_count
    ]
    tetrahedra = verts[self._tetrahedra]
    ratio = float(
      np.min(np.linalg.det(tetrahedra[:, 1:] - tetrahedra[:, :1]) / self._rest_volumes)
    )
    self.minimum_element_volume_ratio = min(self.minimum_element_volume_ratio, ratio)
    inverted = ratio <= 0
    if inverted and not is_observe_mode(self):
      record_guard_event(
        self,
        "sponge_integrity_guard",
        "Sponge element inverted",
        measured=ratio,
        threshold=0.0,
        recovery="stop_invalid_physics",
      )
      mark_physics_invalid(self)
      raise FatalSimulationError("Sponge element inverted")
    rigid = np.flatnonzero(np.all(contacts.geom >= 0, axis=1))
    if len(rigid):
      pairs = contacts.geom[rigid]
      mask = np.any(self._hand_geom_mask[pairs], axis=1) & np.any(
        pairs == self._table_id, axis=1
      )
      if np.any(mask):
        distance = float(np.min(contacts.dist[rigid[mask]]))
        self.minimum_hand_table_distance_m = (
          distance
          if self.minimum_hand_table_distance_m is None
          else min(distance, self.minimum_hand_table_distance_m)
        )
        if distance < 0:
          self.penetration_count += 1
        task_guard(
          self,
          distance < 0,
          "penetration_guard",
          "Hand/table penetration",
          measured=distance,
          threshold=0.0,
          recovery="continue_physics",
        )
    ids = np.flatnonzero(np.any(contacts.flex == self._flex_id, axis=1))
    if len(ids):
      ids = ids[_external_flex_contact_mask(contacts.geom[ids], contacts.elem[ids])]
    if len(ids):
      contact = contacts[int(ids[np.argmin(contacts.dist[ids])])]
      distance = float(contact.dist)
      self.minimum_sponge_contact_distance_m = (
        distance
        if self.minimum_sponge_contact_distance_m is None
        else min(distance, self.minimum_sponge_contact_distance_m)
      )
      if distance < 0:
        self.penetration_count += 1
      task_guard(
        self,
        distance < 0,
        "penetration_guard",
        "Sponge penetration",
        measured=distance,
        threshold=0.0,
        recovery="continue_physics",
        details={
          "geom": contact.geom.tolist(),
          "flex": contact.flex.tolist(),
          "elem": contact.elem.tolist(),
          "vert": contact.vert.tolist(),
        },
      )
    if inverted:
      self.penetration_count += 1
      record_guard_event(
        self,
        "sponge_integrity_guard",
        "Sponge element inverted",
        measured=ratio,
        threshold=0.0,
        recovery="stop_invalid_physics",
      )
      mark_physics_invalid(self)
      raise FatalSimulationError("Sponge element inverted")

  def deformation_mm(self):
    current = self.data.flexvert_xpos[
      self._flex_start : self._flex_start + self._flex_count
    ]
    a = self._rest_points - self._rest_points.mean(axis=0)
    b = current - current.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    if np.linalg.det(u @ vt) < 0:
      u[:, -1] *= -1
    return float(np.linalg.norm(b - a @ (u @ vt), axis=1).max() * 1000)

  def grasp_reference_position(self):
    return (
      self.data.flexvert_xpos[self._flex_start + PIN]
      - self.nominal_sponge_rotation @ self._rest_points[PIN]
    )

  def measure(self):
    sample = self.forces.read(self.data)
    return GraspState(
      float(self.data.time),
      float(self.data.flexvert_xpos[self._flex_start : self._flex_start + self._flex_count, 2].min()),
      self.table_support_force_n,
      self.deformation_mm(),
      tuple(sample.normal_force_n[5:].tolist()),
      tuple(np.linalg.norm(sample.tangent_force_n[5:], axis=1).tolist()),
    )

  def contact_integrity_report(self):
    return {
      "checked_every_physics_step": True,
      "minimum_sponge_contact_distance_m": self.minimum_sponge_contact_distance_m,
      "minimum_hand_table_distance_m": self.minimum_hand_table_distance_m,
      "minimum_element_volume_ratio": self.minimum_element_volume_ratio,
      "penetration_count": self.penetration_count,
      "contact_margin_m": config.SPONGE_CONTACT_MARGIN_M,
    }
