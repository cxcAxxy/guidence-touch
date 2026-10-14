"""Pick a volumetric elastic sponge up from the table, then wipe the vase.

Only reset places the hand and sponge. Motors, native contacts and finite
contact-patch rolling resistance move it; no weld or object pose servo is used.
Cleaning state is privileged evaluation state, never a controller observation.
"""

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from kaihand_tactile_env.shared.evaluation_guards import (
  FatalSimulationError,
  is_observe_mode,
  mark_physics_invalid,
  record_guard_event,
  task_guard,
)
from kaihand_tactile_env.shared.simulation import HAND_JOINT_NAMES, ArmHandSimulation

from . import config
from .cleaning import CleaningProgress
from .randomization import StainLayout
from .sponge import (
  PIN,
  WORKING_NODES,
  GraspPatchFriction,
  SpongeTactileProvider,
  points,
)


@dataclass(frozen=True)
class WipeState:
  timestamp: float
  cleaned_fraction: float
  wall_normal_force_n: float
  wall_tangent_force_n: float
  deformation_mm: float
  fingertip_normal_force_n: tuple[float, ...]
  fingertip_tangent_force_n: tuple[float, ...]
  tactile_wall_estimate_n: float
  wrist_wall_estimate_n: float


def _external_flex_contact_mask(geom, elem):
  """Exclude MuJoCo's per-tetrahedron inversion-prevention contacts.

  Those internal vertex-to-opposite-face constraints have no geom on either
  side and exactly one element id. External flex/rigid and true flex surface
  self-contacts remain in the audit.
  """
  geom = np.asarray(geom)
  elem = np.asarray(elem)
  internal = np.all(geom < 0, axis=1) & (np.sum(elem >= 0, axis=1) == 1)
  return ~internal


class VaseWipeSimulation(ArmHandSimulation):
  def __init__(self, *, stain_seed=None, **kwargs):
    self.stain_seed = stain_seed
    self._stain_layout = None
    super().__init__(scene=config.SCENE_NAME, **kwargs)
    self.model.opt.timestep = self.timestep = config.PHYSICS_TIMESTEP_S
    self.forces = SpongeTactileProvider(self.model)
    # The common evaluation video must use the flex-aware provider; the
    # default rigid-contact provider cannot see the sponge side of flex contacts.
    self.evaluation_tactile_provider = self.forces
    # These arrays belong to this compiled task, never to the shared XML.
    # Apply the same constraint response to both sides of flex contacts so
    # softer hand/table defaults cannot dilute the nonpenetrating contact.
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
    # External flex contacts use the rigid participant's 0.99 parameters;
    # the firmer flex parameters act only on sponge surface self-contact.
    self.model.geom_priority[:] = 1
    self.model.flex_priority[:] = 0
    self.model.geom_friction[:, 0] = np.maximum(self.model.geom_friction[:, 0], 1.1)
    # Split the buffer between participants: both flex/rigid and hand/vase
    # pairs get the same positive margin (MuJoCo adds the two margins).
    self.model.geom_margin[:] = config.SPONGE_CONTACT_MARGIN_M / 2
    self.model.flex_margin[:] = config.SPONGE_CONTACT_MARGIN_M / 2
    self.model.flex_damping[:] = config.SPONGE_ELASTIC_DAMPING_S
    self.model.flex_selfcollide[:] = mujoco.mjtFlexSelf.mjFLEXSELF_AUTO
    self.model.opt.tolerance = 1e-10
    self._hand_geom_ids = {
      i
      for i in range(self.model.ngeom)
      if self.model.body(int(self.model.geom_bodyid[i])).name.startswith("hand_r_")
    }
    self.probes = None  # Native flex contacts use shared force taxels, not rigid-geom distance probes.
    self.physics_observer = None
    self.grasp_patch = GraspPatchFriction(self.model, self._hand_geom_ids)

  @property
  def observation_time(self):
    """Time of the forward-evaluated state used by tactile and RGB capture."""
    return float(self.data.time)

  def reset(self, **kwargs):
    super().reset(**kwargs)
    if self._stain_layout is None:
      self._stain_layout = StainLayout(self.model)
    area, self.stain_randomization = self._stain_layout.apply(
      self.model, self.stain_seed
    )
    self.cleaning = CleaningProgress(len(area), area_scale=area)
    self.dirt = self.cleaning.remaining
    self.patch_contact_tangent_load_n = np.zeros_like(self.dirt)
    self.patch_sliding_speed_m_s = np.zeros_like(self.dirt)
    self.patch_friction_power_w = np.zeros_like(self.dirt)
    self._stain_ids = np.array(
      [self.model.geom(f"stain_{i}").id for i in range(len(self.dirt))]
    )
    self.model.geom_rgba[self._stain_ids, 3] = 1.0
    self._wall_ids = {
      self.model.geom(f"vase_wall_{i}").id
      for i in range(config.WALL_COUNT * (len(config.INNER_PROFILE) - 1))
    }
    self._flex_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_FLEX, "sponge_volume"
    )
    self._flex_start = int(self.model.flex_vertadr[self._flex_id])
    self._flex_count = int(self.model.flex_vertnum[self._flex_id])
    self._flex_bodies = self.model.flex_vertbodyid[
      self._flex_start : self._flex_start + self._flex_count
    ].copy()
    self._rest_points = points()
    self.minimum_sponge_contact_distance_m = None
    self.minimum_hand_environment_distance_m = None
    self.peak_hand_environment_force_n = 0.0
    self.closest_sponge_contact = None
    self.penetration_count = 0
    adr = int(self.model.flex_elemdataadr[self._flex_id])
    size = int(self.model.flex_elemnum[self._flex_id]) * 4
    self._tetrahedra = self.model.flex_elem[adr : adr + size].reshape(-1, 4)
    rest = self._rest_points[self._tetrahedra]
    self._rest_volumes = np.linalg.det(rest[:, 1:] - rest[:, :1])
    self.minimum_element_volume_ratio = 1.0
    self._audit_hand_geoms = np.array(
      [
        self.model.body(int(b)).name.startswith("hand_r_")
        for b in self.model.geom_bodyid
      ]
    )
    self._audit_environment_geoms = np.array(
      [
        self.model.geom(i).name.startswith("vase_")
        or self.model.geom(i).name == "tabletop"
        for i in range(self.model.ngeom)
      ]
    )
    self._table_id = self.model.geom("tabletop").id
    self.table_height = float(
      self.data.geom_xpos[self._table_id, 2] + self.model.geom_size[self._table_id, 2]
    )
    self.table_support_force_n = 0.0
    self.wall_normal_force_n = self.wall_tangent_force_n = 0.0
    self.peak_wall_force_n = 0.0
    self.tactile_wall_estimate_n = 0.0
    self.wrist_wall_estimate_n = 0.0
    self._wrist_force_baseline = 0.0
    self._force_baseline = 0.0
    self._last_force_time = 0.0
    self._tabletop_preset()
    if getattr(self, "grasp_patch", None) is not None:
      self.grasp_patch.enabled = False
      self.grasp_patch.reference = None
      self.grasp_patch.torque_world_n_m[:] = 0
      self.grasp_patch.normal_load_n = 0.0
    if hasattr(self, "probes") and self.probes is not None:
      self.probes.reset()

  def _tabletop_preset(self):
    # Five opposing pads in the shared hand frame; the continuous cleaning
    # end extends below the fingers into the shallow cavity.
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
    yaw = config.PICKUP_HAND_YAW_OFFSET_RAD
    target_hand_rotation = (
      np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
      )
      @ target_hand_rotation
    )
    # Stand on the broad cleaning end, supported only by the tabletop. The
    # open hand reaches the hover through the executor; no initial overlap
    # or constraint is introduced at reset.
    tilt_rotation = np.eye(3)
    self.nominal_sponge_rotation = tilt_rotation
    target_hand_rotation = tilt_rotation @ target_hand_rotation
    centre = np.r_[
      config.SPONGE_TABLE_XY,
      self.table_height + 0.001 - self._rest_points[:, 2].min(),
    ]
    self.pickup_centre = centre.copy()
    material_positions = self._rest_points @ tilt_rotation.T + centre
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
    result = self.solve_ik(
      "right",
      centre + [0, 0, 0.16] + self.wrist_offset,
      self.wrist_rotation,
      seed=config.ARM_HOME["right"],
      max_iterations=300,
    )
    if not result.success:
      raise RuntimeError(f"tabletop approach IK failed: {result}")
    self.approach_arm = result.joint_positions.copy()
    pickup_delta = self.approach_arm - config.ARM_HOME["right"]
    if pickup_delta[4] >= 0 or np.max(np.abs(pickup_delta)) > np.deg2rad(100):
      raise RuntimeError(
        "tabletop approach must use the short counterclockwise wrist path"
      )
    self.grip = np.array(
      [
        0.10848,
        1.745,
        0.698,
        0.25062,
        -0.05,
        0.41975,
        1.12822,
        0.29551,
        0,
        0.06992,
        1.30959,
        0.12397,
        0,
        0.00582,
        1.35305,
        -0.03040,
        0.1,
        0.51136,
        0.63303,
        0.50484,
      ]
    )
    self.open_grip = self.grip.copy()
    self.open_grip[2:4] = 0.05
    self.open_grip[[5, 9, 13, 17]] *= 0.5
    self.open_grip[[6, 10, 14, 18]] = 0.1
    self.open_grip[[7, 11, 15, 19]] = 0.0
    # Keep the shared open-hand home pose until the executor starts moving.

  def _before_physics_step(self):
    if getattr(self, "grasp_patch", None) is not None:
      self.grasp_patch.apply(self.data)

  def step(self, steps=1):
    for _ in range(max(1, int(steps))):
      super().step()
      # Account once per physics step, independent of render/observation rate.
      mujoco.mj_forward(self.model, self.data)
      self._check_contact_integrity()
      self._update_cleaning()
      self.peak_wall_force_n = max(self.peak_wall_force_n, self.wall_normal_force_n)
      if getattr(self, "physics_observer", None) is not None:
        self.physics_observer(self)

  def _check_contact_integrity(self):
    """Check every physics step, including steps omitted by the 500 Hz recorder.

    A failed trajectory is rejected, never repaired by moving material nodes,
    clipping contact distances, or smoothing the exported force.
    """
    contacts = self.data.contact
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
    pairs = contacts.geom[rigid]
    relevant = np.any(self._audit_hand_geoms[pairs], axis=1) & np.any(
      self._audit_environment_geoms[pairs], axis=1
    )
    if np.any(relevant):
      distance = float(np.min(contacts.dist[rigid[relevant]]))
      if (
        self.minimum_hand_environment_distance_m is None
        or distance < self.minimum_hand_environment_distance_m
      ):
        self.minimum_hand_environment_distance_m = distance
      if distance < 0:
        self.penetration_count += 1
      task_guard(
        self,
        distance < 0,
        "penetration_guard",
        "Hand/environment penetration",
        measured=distance,
        threshold=0.0,
        recovery="continue_physics",
      )
      wrench = np.zeros(6)
      load = 0.0
      for contact_id in rigid[relevant]:
        mujoco.mj_contactForce(self.model, self.data, int(contact_id), wrench)
        load += max(0.0, float(wrench[0]))
      self.peak_hand_environment_force_n = max(self.peak_hand_environment_force_n, load)
      task_guard(
        self,
        load > 2.0,
        "collision_guard",
        "Hand struck vase/table",
        measured=load,
        threshold=2.0,
        recovery="continue_physics",
      )
    ids = np.flatnonzero(np.any(contacts.flex == self._flex_id, axis=1))
    if len(ids):
      ids = ids[_external_flex_contact_mask(contacts.geom[ids], contacts.elem[ids])]
    if len(ids):
      i = int(ids[np.argmin(contacts.dist[ids])])
      contact = contacts[i]
      distance = float(contact.dist)
      if (
        self.minimum_sponge_contact_distance_m is None
        or distance < self.minimum_sponge_contact_distance_m
      ):
        self.minimum_sponge_contact_distance_m = distance
        self.closest_sponge_contact = {
          "time_s": float(self.data.time),
          "phase": getattr(self, "phase", "tabletop_ready"),
          "geom_ids": contact.geom.tolist(),
          "element_ids": contact.elem.tolist(),
        }
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

  def contact_integrity_report(self):
    return {
      "checked_every_physics_step": True,
      "minimum_sponge_contact_distance_m": self.minimum_sponge_contact_distance_m,
      "minimum_hand_environment_distance_m": self.minimum_hand_environment_distance_m,
      "peak_hand_environment_force_n": self.peak_hand_environment_force_n,
      "minimum_element_volume_ratio": self.minimum_element_volume_ratio,
      "closest_sponge_contact": self.closest_sponge_contact,
      "penetration_count": self.penetration_count,
      "contact_margin_m": config.SPONGE_CONTACT_MARGIN_M,
    }

  def _update_cleaning(self):
    self.wall_normal_force_n = self.wall_tangent_force_n = 0.0
    self.table_support_force_n = 0.0
    n = len(self.dirt)
    local_force, local_power, speed_force = np.zeros(n), np.zeros(n), np.zeros(n)
    verts = self.data.flexvert_xpos[
      self._flex_start : self._flex_start + self._flex_count
    ]
    cvel = self.data.cvel[self._flex_bodies]
    velocities = cvel[:, 3:] + np.cross(
      cvel[:, :3],
      verts - self.data.subtree_com[self.model.body_rootid[self._flex_bodies]],
    )
    wrench = np.zeros(6)
    for i, contact in enumerate(self.data.contact):
      if (contact.flex[0] == self._flex_id and contact.geom2 == self._table_id) or (
        contact.flex[1] == self._flex_id and contact.geom1 == self._table_id
      ):
        mujoco.mj_contactForce(self.model, self.data, i, wrench)
        self.table_support_force_n += max(0.0, float(wrench[0]))
        continue
      if contact.flex[0] == self._flex_id and int(contact.geom2) in self._wall_ids:
        side, wall_normal = 0, -contact.frame[:3]
      elif contact.flex[1] == self._flex_id and int(contact.geom1) in self._wall_ids:
        side, wall_normal = 1, contact.frame[:3]
      else:
        continue
      if contact.pos[2] >= config.VASE_CENTER[2] + config.INNER_PROFILE[-1][0] - 0.003:
        continue  # Lip/upper edge pressure is not inside-wall wiping.
      radial = contact.pos[:2] - config.VASE_CENTER[:2]
      if np.dot(wall_normal[:2], radial) >= -0.3 * np.linalg.norm(radial):
        continue
      if (
        abs(
          np.linalg.norm(radial)
          - config.inner_radius(contact.pos[2] - config.VASE_CENTER[2])
        )
        > 0.006
      ):
        continue
      mujoco.mj_contactForce(self.model, self.data, i, wrench)
      fn, ft = max(0.0, float(wrench[0])), float(np.linalg.norm(wrench[1:3]))
      self.wall_normal_force_n += fn
      self.wall_tangent_force_n += ft
      if contact.vert[side] >= 0:
        point_velocity = velocities[contact.vert[side]]
      else:
        adr = int(self.model.flex_elemdataadr[self._flex_id]) + 4 * int(
          contact.elem[side]
        )
        ids = self.model.flex_elem[adr : adr + 4]
        positions = verts[ids]
        bary = np.linalg.lstsq(
          (positions[1:] - positions[0]).T, contact.pos - positions[0], rcond=None
        )[0]
        weights = np.clip(np.r_[1 - bary.sum(), bary], 0, 1)
        weights /= max(weights.sum(), 1e-12)
        point_velocity = weights @ velocities[ids]
      tangent_velocity = point_velocity - wall_normal * np.dot(
        point_velocity, wall_normal
      )
      speed = float(np.linalg.norm(tangent_velocity))
      force_on_sponge = (1 if side == 1 else -1) * (
        contact.frame.reshape(3, 3).T @ np.r_[0.0, wrench[1:3]]
      )
      power = max(0.0, -float(np.dot(force_on_sponge, tangent_velocity)))
      # Exclude numerical impact impulses, and conserve work across patches.
      if speed > 0.15:
        continue
      weights = np.maximum(
        0.0,
        1
        - np.linalg.norm(self.data.geom_xpos[self._stain_ids] - contact.pos, axis=1)
        / (0.013 * np.sqrt(self.cleaning.area_scale)),
      )
      weights /= max(1.0, float(weights.sum()))
      local_force += weights * ft
      local_power += weights * power
      speed_force += weights * ft * speed
    local_speed = np.divide(
      speed_force, local_force, out=np.zeros(n), where=local_force > 1e-12
    )
    self.patch_contact_tangent_load_n = local_force
    self.patch_sliding_speed_m_s = local_speed
    self.patch_friction_power_w = local_power
    self.cleaning.update(
      self.wall_normal_force_n,
      self.wall_tangent_force_n,
      local_force,
      local_power,
      local_speed,
      self.timestep,
    )
    self.dirt = self.cleaning.remaining
    self.model.geom_rgba[self._stain_ids, 3] = self.dirt**config.STAIN_OPACITY_GAMMA

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
    # Track a material point at the grasp, expressed in the nominal tool frame.
    return (
      self.data.flexvert_xpos[self._flex_start + PIN]
      - self.nominal_sponge_rotation @ self._rest_points[PIN]
    )

  def measure(self):
    sample = self.forces.read(self.data)
    if self.probes is not None:
      self.probes.read(self.data)
    estimate = max(0.0, self._force_baseline - float(sample.force_world_n[5:, 0].sum()))
    dt = max(0.0, float(self.data.time) - self._last_force_time)
    self.tactile_wall_estimate_n += (1 - np.exp(-dt / 0.05)) * (
      estimate - self.tactile_wall_estimate_n
    )
    wrist_estimate = max(
      0.0, float(self.wrist_force_world()[0]) - self._wrist_force_baseline
    )
    self.wrist_wall_estimate_n += (1 - np.exp(-dt / 0.05)) * (
      wrist_estimate - self.wrist_wall_estimate_n
    )
    self._last_force_time = float(self.data.time)
    return WipeState(
      float(self.data.time),
      float(1 - self.dirt.mean()),
      self.wall_normal_force_n,
      self.wall_tangent_force_n,
      self.deformation_mm(),
      tuple(sample.normal_force_n[5:].tolist()),
      tuple(np.linalg.norm(sample.tangent_force_n[5:], axis=1).tolist()),
      self.tactile_wall_estimate_n,
      self.wrist_wall_estimate_n,
    )

  def native_wrist_force_world(self):
    sensor = self.model.sensor("vase_wrist_force")
    address = int(sensor.adr[0])
    force = self.data.sensordata[address : address + 3]
    return (
      self.data.site_xmat[self.model.site("right_ee_site").id].reshape(3, 3) @ force
    )

  def wrist_force_world(self):
    # mj_rnePostConstraint skips flex contacts. Restore their omitted force on
    # the hand subtree to obtain a flange reading that supports native flex.
    # Only hand contacts are read here, never the hidden ceramic contact list.
    force = self.native_wrist_force_world()
    wrench = np.zeros(6)
    for i, contact in enumerate(self.data.contact):
      if contact.flex[0] >= 0 and int(contact.geom2) in self._hand_geom_ids:
        sign = 1.0
      elif contact.flex[1] >= 0 and int(contact.geom1) in self._hand_geom_ids:
        sign = -1.0
      else:
        continue
      mujoco.mj_contactForce(self.model, self.data, i, wrench)
      force -= sign * (contact.frame.reshape(3, 3).T @ wrench[:3])
    return force


class VaseWipeExecutor:
  """Known-pose approach, then a bounded tactile normal-force feedback sweep."""

  def __init__(self, simulation, observer=None):
    self.sim = simulation
    self.observer = observer
    self.target = simulation.current_pose_matrix("right")[0] - simulation.wrist_offset
    self.grasp_verified = False
    self.pickup_report = {}
    self.working_offset = np.zeros(3)
    self.grasp_body = simulation.model.body("hand_r_base_link").id
    self._calibrate_grasp_guard()
    self.peak_deformation = 0.0
    self.max_grasp_drift = 0.0
    self._grip_progress = np.zeros(5)
    self._grip_force_filtered = np.zeros(5)
    self._grip_contact_time = np.full(5, np.nan)
    self._grip_goal = None
    self._hold_progress_cap = None

  def _calibrate_grasp_guard(self):
    sim = self.sim
    self.grasp_in_hand = sim.data.xmat[self.grasp_body].reshape(3, 3).T @ (
      sim.data.flexvert_xpos[sim._flex_start + PIN] - sim.data.xpos[self.grasp_body]
    )

  def _tick(self, phase):
    sim = self.sim
    if self._grip_goal is not None:
      # Build the grasp with independent pad feedback for each finger.
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
        # Restore a lost preload without opening against wall reaction.
        # Bound the extra closure so an unloaded pad cannot chase contact
        # indefinitely when another part of the finger is already loaded.
        rate = np.clip(
          1.5
          * (
            config.GRASP_FORCE_TARGETS_N - np.maximum(loads, self._grip_force_filtered)
          ),
          0,
          0.04,
        )
        progress_cap = self._hold_progress_cap
      self._grip_progress = np.clip(self._grip_progress + rate * 0.01, 0, progress_cap)
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
    sim.step(round(0.01 / sim.timestep))
    if (
      not np.all(np.isfinite(sim.data.qpos))
      or sim.data.time < before
      or any(w.number for w in sim.data.warning)
    ):
      raise RuntimeError(f"{phase}: physics became unstable")
    state = sim.measure()
    self.peak_deformation = max(self.peak_deformation, state.deformation_mm)
    # Evaluation-only drop guard in the actual palm frame, including rotations.
    expected_pin = (
      sim.data.xpos[self.grasp_body]
      + sim.data.xmat[self.grasp_body].reshape(3, 3) @ self.grasp_in_hand
    )
    drift = float(
      np.linalg.norm(sim.data.flexvert_xpos[sim._flex_start + PIN] - expected_pin)
    )
    if self.grasp_verified:
      self.max_grasp_drift = max(self.max_grasp_drift, drift)
    if self.grasp_verified and drift > 0.045:
      raise RuntimeError(f"{phase}: sponge slipped out of grasp ({drift:.3f} m)")
    if self.observer is not None:
      self.observer(sim, phase, state)
    return state

  def _move(self, destination, seconds, phase):
    start = self.target.copy()
    count = max(1, round(seconds / 0.01))
    for i in range(count):
      t = (i + 1) / count
      self.target = start + (3 * t * t - 2 * t * t * t) * (
        np.asarray(destination) - start
      )
      self._tick(phase)

  def _wall_x_for_tool(self, target):
    radius = config.inner_radius(
      target[2] + self.working_offset[2] - config.VASE_CENTER[2]
    )
    lateral = target[1] + self.working_offset[1] - config.VASE_CENTER[1]
    return float(np.sqrt(max(radius * radius - lateral * lateral, 0.001)))

  def _contact_target_x(self, target):
    """Known wall geometry and unloaded tool calibration, without contact truth."""
    return float(
      config.VASE_CENTER[0]
      + self._wall_x_for_tool(target)
      - self.working_offset[0]
      - self.sim.model.flex_radius[self.sim._flex_id]
      - config.SPONGE_CONTACT_MARGIN_M
    )

  @staticmethod
  def _scan_offsets(seconds, lateral_amplitude, vertical_amplitude):
    # Sweep both sides at one height before changing rows. Coupled sine waves
    # leave opposite corners with very different contact time.
    levels = (0.0, -1.0, 1.0, -1.0)
    row = min(int(seconds / 2.8), len(levels) - 1)
    previous = levels[max(0, row - 1)]
    blend = np.clip((seconds - row * 2.8) / 0.6, 0, 1)
    blend = blend * blend * (3 - 2 * blend)
    return (
      lateral_amplitude * np.sin(2 * np.pi * seconds / 2.8),
      vertical_amplitude * (previous + (levels[row] - previous) * blend),
    )

  def _pickup(self, grip):
    sim = self.sim
    from ...shared.approach import approach_waypoint

    for _ in approach_waypoint(
      sim, sim.approach_arm, sim.open_grip, phase="approach_sponge"
    ):
      if self.observer is not None:
        self.observer(sim, "approach_sponge", sim.measure())
    self.target = sim.current_pose_matrix("right")[0] - sim.wrist_offset
    self._move(self.target, 0.5, "observe_tabletop")
    initial_load = sim.table_support_force_n
    initial_pad_load = float(sim.forces.read(sim.data).normal_force_n[5:].sum())
    if initial_load < 0.15 or initial_pad_load > 0.05:
      raise RuntimeError("Sponge must initially rest on the table, clear of the hand")
    centre = sim.pickup_centre.copy()
    self._move(centre + [0, 0, 0.04], 1.2, "approach_sponge")
    self._move(centre, 0.8, "lower_to_grasp")
    close_steps = round(config.GRASP_CLOSE_SECONDS / 0.01)
    self._grip_goal = grip.copy()
    for _ in range(close_steps):
      self._tick("close_on_sponge")
    self._move(self.target, 0.4, "confirm_grasp")
    # Permit only a bounded, slow preload recovery during subsequent motion.
    self._hold_progress_cap = np.ones(5)
    loads = sim.forces.read(sim.data).normal_force_n[5:]
    if loads[0] < 0.05 or loads[1:].sum() < 0.15:
      raise RuntimeError("Pickup requires opposing thumb and finger contact")
    sim.grasp_patch.enabled = True
    sim.wrist_offset = (
      sim.current_pose_matrix("right")[0] - sim.grasp_reference_position()
    )
    self.target = sim.grasp_reference_position().copy()
    self._calibrate_grasp_guard()
    self.grasp_verified = True
    self._move([self.target[0], self.target[1], 1.055], 1.8, "lift_from_table")
    clearance = float(sim.data.flexvert_xpos[:, 2].min() - sim.table_height)
    if clearance < 0.12 or sim.table_support_force_n > 0.01:
      raise RuntimeError("Sponge did not lift clear of the tabletop")
    self.pickup_report = {
      "success": True,
      "initial_table_support_force_n": initial_load,
      "initial_fingertip_force_n": initial_pad_load,
      "grasp_fingertip_normal_force_n": loads.tolist(),
      "lift_clearance_m": clearance,
      "lift_time_s": float(sim.data.time),
    }
    # Rotate the real held sponge in free space while moving above the vase.
    # Only wrist motor goals change; material-node positions are never reset.
    start = self.target.copy()
    wrist_rotation, wrist_offset = sim.wrist_rotation.copy(), sim.wrist_offset.copy()
    nominal = sim.nominal_sponge_rotation.copy()
    material = sim.data.flexvert_xpos[sim.grasp_patch.ids]
    u, _, vt = np.linalg.svd(
      sim.grasp_patch.rest.T @ (material - material.mean(axis=0))
    )
    actual = (u @ vt).T
    pitch, roll = config.SPONGE_WIPE_PITCH_RAD, config.SPONGE_WIPE_ROLL_RAD
    desired = np.array(
      [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
    ) @ np.array(
      [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )
    yaw = config.SPONGE_WIPE_YAW_RAD
    desired = (
      np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
      )
      @ desired
    )
    quaternion, rotation_vector = np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(quaternion, (desired @ actual.T).ravel())
    mujoco.mju_quat2Vel(rotation_vector, quaternion, 1.0)
    angle = float(np.linalg.norm(rotation_vector))
    axis = rotation_vector / max(angle, 1e-12)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    for i in range(150):
      u = (i + 1) / 150
      u = u * u * (3 - 2 * u)
      rotation = (
        np.eye(3) + np.sin(angle * u) * skew + (1 - np.cos(angle * u)) * (skew @ skew)
      )
      sim.wrist_rotation = rotation @ wrist_rotation
      sim.wrist_offset = rotation @ wrist_offset
      sim.nominal_sponge_rotation = rotation @ nominal
      self.target = start + u * (np.array([0.60, -0.18, 1.04]) - start)
      self._tick("orient_for_wipe")

  def execute(self):
    from .vision import WallInspection

    self.sim.inspection = WallInspection(
      self.sim.model,
      stain_padding_m=0.011 if self.sim.stain_seed is not None else 0.007,
    )
    try:
      return self._execute_inspected()
    finally:
      self.sim.inspection.close()

  def _execute_inspected(self):
    sim = self.sim
    grip = sim.grip.copy()
    # Thumb joint3 is already at its upper limit. Close the available distal
    # joint5 instead, opposing the four fingers rather than overdriving a stop.
    grip[3] += 0.35
    grip[[6, 10, 14, 18]] += 0.12
    grip[[5, 9, 13, 17]] += 0.08
    self._pickup(grip)
    sim.grasp_patch.enabled = True
    sim.wrist_offset = (
      sim.current_pose_matrix("right")[0] - sim.grasp_reference_position()
    )
    self.target = sim.grasp_reference_position().copy()
    self._calibrate_grasp_guard()
    # Keep the cuboid oblique to the concave wall. A face parallel to the wall
    # contacts at both outer edges, leaving its centre away from the stains.
    patch = sim.grasp_patch
    material = sim.data.flexvert_xpos[patch.ids]
    u, _, vt = np.linalg.svd(patch.rest.T @ (material - material.mean(axis=0)))
    rotation = (u @ vt).T
    self.alignment_yaw = config.SPONGE_WIPE_YAW_RAD - float(
      np.arctan2(rotation[1, 0], rotation[0, 0])
    )
    c, sn = np.cos(self.alignment_yaw), np.sin(self.alignment_yaw)
    rotate = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
    sim.wrist_rotation = rotate @ sim.wrist_rotation
    sim.wrist_offset = rotate @ sim.wrist_offset
    sim.nominal_sponge_rotation = rotate @ sim.nominal_sponge_rotation
    self._move(self.target, 1.5, "align_sponge")
    sim.wrist_offset = (
      sim.current_pose_matrix("right")[0] - sim.grasp_reference_position()
    )
    self.target = sim.grasp_reference_position().copy()
    # One unloaded tool calibration after pregrasp. Wiping uses this fixed
    # offset, flange feedback and RGB/depth inspections, not hidden contacts.
    self.working_offset = (
      sim.data.flexvert_xpos[sim._flex_start + np.array(WORKING_NODES)].mean(axis=0)
      - self.target
    )
    # HEAD already sees the far wall from the aligned pose above the vase.
    # Let the held sponge settle in place, without a sideways observation detour.
    self._move(self.target, 0.2, "observe_before")
    inspection = sim.inspection.inspect(sim.data)
    completed_passes = 0
    self.contact_reacquisitions = 0
    for pass_index in range(1, config.MAX_WIPE_PASSES + 1):
      if inspection["visually_clean"]:
        break
      if not inspection["valid"]:
        raise RuntimeError("HEAD inspection remains occluded after lifting")
      bounds = np.array(inspection["residual_bounds_world_m"])
      centre = np.array(inspection["residual_center_world_m"])
      extent = bounds[1] - bounds[0]
      targeted = inspection["stain_pixels"] < 0.95 * inspection["initial_stain_pixels"]
      # Small pixel residuals still require motion beyond the elastic tool's
      # lost motion. Tiny sweeps can rub the neck while never loading a lower
      # stain. Keep a minimum physical stroke while fitting the whole cuboid
      # inside the mouth, including its trailing edge.
      lateral_amplitude = 0.012
      vertical_amplitude = float(np.clip(0.5 * extent[2] - 0.005, 0.010, 0.012))
      cy = float(np.clip(centre[1] - self.working_offset[1], -0.205, -0.125))
      # Bound the observed working-surface height, not a hard-coded grasp
      # height from the old 120 mm tool. The shorter tool must still reach
      # a low residual row identified by the camera.
      work_z = np.clip(
        centre[2],
        config.VASE_CENTER[2] + config.FLOOR_HEIGHT + 0.02,
        config.VASE_CENTER[2] + config.INNER_PROFILE[-1][0] - 0.015,
      )
      cz = float(work_z - self.working_offset[2])
      # Settle at the current observation pose before taring the unloaded
      # hand; lifting inertia must not become the wall-force baseline.
      self._move(self.target, 0.4, f"settle_before_{pass_index}")
      sim._force_baseline = float(sim.forces.read(sim.data).force_world_n[5:, 0].sum())
      sim.tactile_wall_estimate_n = 0.0
      sim._wrist_force_baseline = float(sim.wrist_force_world()[0])
      sim.wrist_wall_estimate_n = 0.0
      contact_x = self._contact_target_x(np.array([0.0, cy, cz]))
      retreat_x = contact_x - 0.012
      self._move([retreat_x, cy, config.ENTRY_HEIGHT_M], 0.8, f"approach_{pass_index}")
      self._move([retreat_x, cy, cz], 2.0, f"insert_{pass_index}")
      self._move([contact_x + 0.002, cy, cz], 1.0, f"seek_inner_wall_{pass_index}")
      scan_time = 0.0
      had_contact = False
      wipe_seconds = config.RESIDUAL_WIPE_SECONDS if targeted else config.WIPE_SECONDS
      # Allow time to reacquire the wall. Tangential travel stops while airborne;
      # only the measured flange reaction advances the scan and regulates X.
      for _ in range(round((wipe_seconds + 5.0) / 0.01)):
        state = sim.measure()
        # Hysteresis prevents tiny load changes from repeatedly stopping and
        # restarting tangential travel at the contact threshold.
        contact = state.wrist_wall_estimate_n >= (0.25 if had_contact else 0.35)
        if had_contact and not contact:
          self.contact_reacquisitions += 1
        had_contact = contact
        old_wall_x = self._wall_x_for_tool(self.target)
        if contact:
          scan_time += 0.01
          dy, dz = self._scan_offsets(scan_time, lateral_amplitude, vertical_amplitude)
          self.target[1] = cy + dy
          self.target[2] = cz + dz
        correction = np.clip(
          (config.TARGET_WALL_FORCE_N - state.wrist_wall_estimate_n) * 0.010,
          -0.008,
          0.008,
        )
        self.target[0] = np.clip(
          self.target[0]
          + correction * 0.01
          + self._wall_x_for_tool(self.target)
          - old_wall_x,
          self._contact_target_x(self.target) - 0.012,
          self._contact_target_x(self.target) + 0.025,
        )
        self._tick(
          f"tactile_wipe_{pass_index}" if contact else f"reacquire_wall_{pass_index}"
        )
        if scan_time + 1e-9 >= wipe_seconds:
          break
      completed_passes = pass_index
      # Release normal load before re-centering Y/Z. A diagonal retreat can
      # drag the wide sponge edge across the curved wall during unloading.
      self._move(
        [retreat_x, self.target[1], self.target[2]],
        0.6,
        f"unload_{pass_index}",
      )
      self._move([retreat_x, cy, cz + 0.006], 0.6, f"unload_{pass_index}")
      self._move(
        [retreat_x, cy, config.ENTRY_HEIGHT_M],
        2.0,
        f"lift_{pass_index}",
      )
      # The vertical lift already exposes the far-wall stains to HEAD.
      # Inspect here instead of carrying the sponge sideways and back again.
      inspection = sim.inspection.inspect(sim.data)
      if not inspection["valid"]:
        # A visibility fallback may lift further, but never changes XY.
        self._move(
          self.target + np.array([0.0, 0.0, 0.03]),
          0.8,
          f"clear_view_{pass_index}",
        )
        inspection = sim.inspection.inspect(sim.data)
    self._move(self.target, 0.3, "withdraw")
    state = sim.measure()
    return {
      "task": config.SCENE_NAME,
      "preset": "tabletop_pickup_compliant_contact_patch",
      "pickup": self.pickup_report,
      "contact_integrity": sim.contact_integrity_report(),
      "grasp_force_targets_n": config.GRASP_FORCE_TARGETS_N.tolist(),
      "grasp_contact_model": {
        "model": "finite_patch_rolling_resistance_v1",
        "stiffness_nm_rad": 0.8,
        "damping_nms_rad": 0.010,
        "rolling_radius_m": config.GRASP_PATCH_RADIUS_M,
        "absolute_torque_limit_nm": 0.15,
        "minimum_grasp_load_n": 0.4,
        "net_support_force_n": 0.0,
      },
      "success": bool(
        self.pickup_report.get("success")
        and sim.cleaning.success
        and inspection["visually_clean"]
      ),
      "visual_inspections": sim.inspection.observations,
      "completed_wipe_passes": completed_passes,
      "working_offset_from_grasp_m": self.working_offset.tolist(),
      "pregrasp_alignment_yaw_rad": self.alignment_yaw,
      "working_material_nodes": list(WORKING_NODES),
      "contact_reacquisitions": self.contact_reacquisitions,
      "motion_completed": True,
      "state": asdict(state),
      "peak_wall_force_n": sim.peak_wall_force_n,
      "peak_deformation_mm": self.peak_deformation,
      "cleaning": sim.cleaning.report(),
      "sponge_model": "volumetric_flex_72_nodes_168_tetrahedra",
      "max_grasp_drift_m": self.max_grasp_drift,
      "cleaned_patch_count": int((sim.dirt <= config.MAX_PATCH_DIRT).sum()),
      "patch_count": len(sim.dirt),
      "physics_warning_count": int(sum(w.number for w in sim.data.warning)),
      "remaining_dirt": sim.dirt.reshape(
        config.PATCH_ROWS, config.PATCH_COLUMNS
      ).tolist(),
    }
