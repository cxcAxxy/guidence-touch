"""Validated, compact example recording for the isolated sponge-to-plate task."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from ...shared.config import TRAINING_CAMERA_NAMES, CameraConfig, WorkcellConfig
from ...shared.poker_review import (
  _force_layout,
  _probe_video,
  _validate_probe,
  compose_review_frame,
  plan_review_frames,
)
from ...shared.recording import EpisodeRecorder, _append, _stream, validate_episode
from ...shared.tactile import SolverContactTactileProvider
from ...shared.task_video import _FfmpegPipeWriter
from ..vase_wipe.sponge import PIN
from . import config

FINGERS = ("thumb", "index", "middle", "ring", "little")
PICK_AND_PLACE_PHASES = (
  "open_hand",
  "move_above_sponge",
  "descend_to_sponge",
  "grasp",
  "lift",
  "carry_to_plate",
  "lower_into_plate",
  "release",
  "retreat",
  "settle_in_plate",
)
MINIMUM_CARRY_CLEARANCE_M = 0.08
MAXIMUM_INITIAL_THUMB_MOTION_RAD = 0.005
MINIMUM_PALM_DOWN_COSINE = 0.98
MAXIMUM_CARRY_PIN_DROP_M = 0.005
MAXIMUM_CONTACT_PAD_FORCE_N = 5.0
MAXIMUM_CONTACT_FORCE_JUMP_N = 0.6
MAXIMUM_CONTACT_TANGENT_FORCE_N = 3.0
MAXIMUM_CONTACT_TANGENT_JUMP_N = 0.6
MAXIMUM_RELEASE_TANGENT_JUMP_N = 0.8
# Small pose shifts change the exact solver-contact taxels.  Keep the fixed
# example's limits intact, but allow the measured transient spread in seeded
# collection without relaxing sustained contact, peak loads, or placement.
MAXIMUM_RANDOMIZED_CONTACT_FORCE_JUMP_N = 0.8
MAXIMUM_RANDOMIZED_CONTACT_TANGENT_JUMP_N = 0.8
MAXIMUM_RANDOMIZED_RELEASE_NORMAL_JUMP_N = 1.0
MAXIMUM_RANDOMIZED_RELEASE_TANGENT_JUMP_N = 1.1
MAXIMUM_GRASP_FORCE_JUMP_N = 1.0
MAXIMUM_SETTLE_HEIGHT_RANGE_M = 0.003
MAXIMUM_APPROACH_WRIST_Y_M = -0.08
MINIMUM_APPROACH_HAND_SEPARATION_M = 0.23


def _json(path: Path, value: dict) -> None:
  path.write_text(
    json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    encoding="utf-8",
  )


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


class SpongeGraspEpisodeRecorder(EpisodeRecorder):
  """Shared training Raw plus sponge deformation and support measurements."""

  def __init__(self, path, simulation, capture_config, *, buffer_rows=128):
    super().__init__(
      path,
      simulation,
      capture_config,
      metadata={
        "scene": config.SCENE_NAME,
        "recording_contract": "sponge_place_shared_raw_v3",
      },
      capture_taskspace=True,
      buffer_rows=buffer_rows,
    )

  def _values(self):
    sim = self.sim
    wrist, rotation = sim.current_pose_matrix("right")
    hand_rotation = sim.data.xmat[
      sim.model.body("hand_r_base_link").id
    ].reshape(3, 3)
    vertices = sim.data.flexvert_xpos[
      sim._flex_start : sim._flex_start + sim._flex_count
    ]
    return {
      "flex_vertices_m": vertices.copy(),
      "minimum_sponge_clearance_m": float(vertices[:, 2].min() - sim.table_height),
      "table_support_force_n": float(sim.table_support_force_n),
      "plate_support_force_n": float(sim.plate_support_force_n),
      "plate_base_support_force_n": float(sim.plate_base_support_force_n),
      "plate_base_support_impulse_ns": float(sim.plate_base_support_impulse_ns),
      "right_wrist_position_m": wrist.copy(),
      "right_wrist_rotation": rotation.copy(),
      "hand_base_separation_m": float(np.linalg.norm(
        sim.data.body("hand_r_base_link").xpos
        - sim.data.body("hand_l_base_link").xpos
      )),
      # Local +Y faces out of the palm in the shared KaiHand model.
      "right_palm_down_cosine": float(-hand_rotation[2, 1]),
    }

  def _initialize(self, metadata):
    # Flex contacts are absent from the ordinary rigid-body force provider.
    self.contact_force_provider = self.sim.forces
    super()._initialize(metadata)
    group = self._file.create_group("sponge_grasp")
    group.attrs["force_source"] = self.sim.forces.source
    group.attrs["table_height_m"] = float(self.sim.table_height)
    group.attrs["plate_center_xy_m"] = config.PLATE_TABLE_XY
    group.attrs["plate_interior_top_z_m"] = config.PLATE_INTERIOR_TOP_Z_M
    group.attrs["plate_interior_radius_m"] = config.PLATE_INTERIOR_RADIUS_M
    group.attrs["initial_sponge_xy_m"] = self.sim.sponge_initial_xy_m
    group.attrs["initial_sponge_xy_offset_m"] = self.sim.sponge_xy_offset_m
    group.attrs["initial_sponge_xy_offset_low_m"] = self.sim.xy_offset_low_m
    group.attrs["initial_sponge_xy_offset_high_m"] = self.sim.xy_offset_high_m
    group.attrs["initial_pose_randomized"] = self.sim.randomize_xy
    group.attrs["position_seed"] = (
      -1 if self.sim.position_seed is None else self.sim.position_seed
    )
    group.attrs["palm_down_definition"] = (
      "hand_r_base_link local +Y dotted with world -Z"
    )
    for name, value in self._values().items():
      array = np.asarray(value)
      _stream(group, name, array.shape, array.dtype)

  def _record_state(self, phase):
    super()._record_state(phase)
    for name, value in self._values().items():
      _append(self._file[f"sponge_grasp/{name}"], value)


def audit_recorded_grasp(file: h5py.File) -> dict:
  """Verify right-side approach, loaded carry, plate placement and tactile."""
  attrs = file["sponge_grasp"].attrs
  # Older Raw files have no explicit flag, but do record the sampled range.
  randomized = bool(attrs.get(
    "initial_pose_randomized",
    np.any(
      np.asarray(attrs.get("initial_sponge_xy_offset_high_m", [0, 0]))
      != np.asarray(attrs.get("initial_sponge_xy_offset_low_m", [0, 0]))
    ),
  ))
  initial_target_xy = np.asarray(
    attrs.get("initial_sponge_xy_m", config.SPONGE_TABLE_XY)
  )
  normal_jump_limit = (
    MAXIMUM_RANDOMIZED_CONTACT_FORCE_JUMP_N
    if randomized else MAXIMUM_CONTACT_FORCE_JUMP_N
  )
  loaded_tangent_jump_limit = (
    MAXIMUM_RANDOMIZED_CONTACT_TANGENT_JUMP_N
    if randomized else MAXIMUM_CONTACT_TANGENT_JUMP_N
  )
  release_tangent_jump_limit = (
    MAXIMUM_RANDOMIZED_RELEASE_TANGENT_JUMP_N
    if randomized else MAXIMUM_RELEASE_TANGENT_JUMP_N
  )
  time_s = file["state/timestamp"][:]
  phase = file["commands/phase"].asstr()[:]
  force = file["tactile_contact_force"]
  indices = _force_layout(file)
  normal = force["normal_force_n"][:, indices]
  contact_count = force["contact_count"][:, indices]
  tangent_vector = force["tangent_force_n"][:, indices]
  tangent = np.linalg.norm(tangent_vector, axis=-1)
  normal_taxels = force["normal_taxel_force_n"][:, indices]
  tangent_taxels = force["tangent_taxel_force_n"][:, indices]
  clearance = file["sponge_grasp/minimum_sponge_clearance_m"][:]
  table_support = file["sponge_grasp/table_support_force_n"][:]
  plate_support = file["sponge_grasp/plate_support_force_n"][:]
  plate_base_support = file["sponge_grasp/plate_base_support_force_n"][:]
  plate_base_impulse = file["sponge_grasp/plate_base_support_impulse_ns"][:]
  vertices = file["sponge_grasp/flex_vertices_m"][:]
  wrist = file["sponge_grasp/right_wrist_position_m"][:]
  hand_separation = file["sponge_grasp/hand_base_separation_m"][:]
  palm_down = file["sponge_grasp/right_palm_down_cosine"][:]
  sponge = file["objects/sponge/pose_wxyz"][:, :3]
  qpos_names = file["state/full_qpos_names"].asstr()[:]
  left_thumb_indices = [
    i for i, name in enumerate(qpos_names) if name.startswith("hand_l_thumb_joint")
  ]
  right_thumb_indices = [
    i for i, name in enumerate(qpos_names) if name.startswith("hand_r_thumb_joint")
  ]
  if len(left_thumb_indices) != 6 or len(right_thumb_indices) != 6:
    raise ValueError("Raw state does not contain all twelve thumb joint positions")
  thumb_indices = left_thumb_indices + right_thumb_indices
  thumb_position = file["state/qpos"][:, thumb_indices]
  if not all(
    len(value) == len(time_s)
    for value in (
      phase, normal, contact_count, tangent, clearance, table_support, plate_support,
      plate_base_support, plate_base_impulse, vertices, wrist, hand_separation, palm_down,
      sponge, thumb_position,
    )
  ):
    raise ValueError("Raw sponge, phase and force streams have different lengths")
  if not all(
    np.isfinite(value).all()
    for value in (
      time_s, normal, tangent, clearance, table_support, plate_support,
      plate_base_support, plate_base_impulse, vertices, wrist, hand_separation, palm_down,
      sponge, thumb_position,
    )
  ):
    raise ValueError("Raw sponge placement contains non-finite measurements")
  if np.any(np.diff(time_s) <= 0) or np.any(normal < -1e-9):
    raise ValueError("Invalid Raw clock or negative fingertip normal force")
  if np.any(np.diff(plate_base_impulse) < -1e-9):
    raise ValueError("Plate-base support impulse must be cumulative")
  taxels_conserve_force = bool(
    np.allclose(normal_taxels.sum(axis=(-2, -1)), normal, rtol=0, atol=1e-9)
    and np.allclose(
      tangent_taxels.sum(axis=(-3, -2)),
      force["tangent_force_n"][:, indices],
      rtol=0,
      atol=1e-9,
    )
  )
  transitions = tuple(dict.fromkeys(str(value) for value in phase))
  ordered = transitions == ("tabletop_ready", *PICK_AND_PLACE_PHASES)
  if not all(np.any(phase == name) for name in PICK_AND_PLACE_PHASES):
    missing = [name for name in PICK_AND_PLACE_PHASES if not np.any(phase == name)]
    raise ValueError(f"Raw placement missing phases: {missing}")
  pregrasp = np.isin(phase, PICK_AND_PLACE_PHASES[:3])
  open_hand = phase == "open_hand"
  move_above = phase == "move_above_sponge"
  descend = phase == "descend_to_sponge"
  grasp = phase == "grasp"
  lift = phase == "lift"
  carry = phase == "carry_to_plate"
  release = phase == "release"
  settle = phase == "settle_in_plate"
  contact_window = np.isin(
    phase, ("lift", "carry_to_plate", "lower_into_plate", "release")
  )
  loaded_motion = np.isin(
    phase, ("lift", "carry_to_plate", "lower_into_plate")
  )
  # Fingers cannot sense the sponge before they close onto it.  Once the
  # closed grasp has settled, however, require every real solver-contact pad
  # to stay loaded continuously until active release begins.
  settled_grasp_start = (
    time_s[np.flatnonzero(grasp)[-1]] - config.SETTLED_GRASP_CONTACT_SECONDS
  )
  five_finger_window = loaded_motion | (grasp & (time_s >= settled_grasp_start))
  five_finger_normal = normal[five_finger_window]
  five_finger_contacts = contact_count[five_finger_window]
  minimum_five_finger_normal = five_finger_normal.min(axis=0)
  minimum_five_finger_contacts = five_finger_contacts.min(axis=0)
  thumb_excursion = np.max(
    np.abs(thumb_position[open_hand] - thumb_position[0]), axis=0
  )
  left_thumb_excursion = float(np.max(thumb_excursion[:6]))
  right_thumb_excursion = float(np.max(thumb_excursion[6:]))
  overhead_palm_down = float(palm_down[np.flatnonzero(move_above)[-1]])
  minimum_descent_palm_down = float(np.min(palm_down[descend]))
  maximum_approach_wrist_y = float(np.max(wrist[move_above, 1]))
  minimum_approach_hand_separation = float(np.min(hand_separation[move_above]))

  carry_vertices = vertices[carry]
  carry_pin_height = carry_vertices[:, PIN, 2]
  carry_centroid_height = carry_vertices[:, :, 2].mean(axis=1)
  carry_bottom_height = carry_vertices[:, :, 2].min(axis=1)
  carry_pin_drop = float(carry_pin_height[0] - np.min(carry_pin_height))
  carry_centroid_drop = float(
    carry_centroid_height[0] - np.min(carry_centroid_height)
  )
  carry_bottom_drop = float(carry_bottom_height[0] - np.min(carry_bottom_height))
  carried_normal = normal[carry]
  opposed_fraction = float(np.mean(
    (carried_normal[:, 0] > 0.05)
    & (carried_normal[:, 1:].sum(axis=1) > 0.15)
  ))
  relative = sponge[carry] - wrist[carry]
  relative_drift = float(np.linalg.norm(relative - relative[0], axis=1).max())

  first_lift = int(np.flatnonzero(lift)[0])
  final_release = int(np.flatnonzero(release)[-1])
  control_period = 1.0 / float(file.attrs["control_hz"])
  pick_to_place_s = float(
    time_s[final_release] + control_period - time_s[first_lift]
  )
  contact_normal = normal[contact_window]
  contact_tangent = tangent[contact_window]
  contact_tangent_vector = tangent_vector[contact_window]
  grasp_normal = normal[grasp]
  grasp_tangent = tangent[grasp]
  grasp_tangent_vector = tangent_vector[grasp]
  grasp_normal_jump = float(np.max(np.abs(np.diff(grasp_normal, axis=0))))
  grasp_tangent_jump = float(np.max(
    np.linalg.norm(np.diff(grasp_tangent_vector, axis=0), axis=-1)
  ))
  grasp_peak_normal = float(np.max(grasp_normal))
  grasp_peak_tangent = float(np.max(grasp_tangent))
  normal_jump = float(np.max(np.abs(np.diff(contact_normal, axis=0))))
  normal_steps = np.abs(np.diff(normal, axis=0))
  loaded_motion_normal_jump = float(np.max(
    normal_steps[loaded_motion[1:]]
  ))
  release_normal_jump = float(np.max(normal_steps[release[1:]]))
  tangent_jump = float(
    np.max(np.linalg.norm(np.diff(contact_tangent_vector, axis=0), axis=-1))
  )
  tangent_steps = np.linalg.norm(np.diff(tangent_vector, axis=0), axis=-1)
  loaded_motion_tangent_jump = float(np.max(
    tangent_steps[loaded_motion[1:]]
  ))
  release_tangent_jump = float(np.max(tangent_steps[release[1:]]))
  peak_normal = float(np.max(contact_normal))
  peak_tangent = float(np.max(contact_tangent))
  final_vertices = vertices[-1]
  final_sponge_xy = final_vertices[:, :2].mean(axis=0)
  final_center_offset = float(np.linalg.norm(final_sponge_xy - config.PLATE_TABLE_XY))
  final_furthest_vertex_radius = float(np.linalg.norm(
    final_vertices[:, :2] - config.PLATE_TABLE_XY, axis=1
  ).max())
  final_bottom_z = float(final_vertices[:, 2].min())
  phase_boundaries = {
    name: float(time_s[np.flatnonzero(phase == name)[0]])
    for name in PICK_AND_PLACE_PHASES
  }
  release_final_loads = normal[final_release]
  settle_peak_loads = normal[settle].max(axis=0)
  settle_vertices = vertices[settle, :, 2]
  settle_pin_height_range = float(np.ptp(settle_vertices[:, PIN]))
  settle_centroid_height_range = float(np.ptp(settle_vertices.mean(axis=1)))
  settle_bottom_height_range = float(np.ptp(settle_vertices.min(axis=1)))
  settle_indices = np.flatnonzero(settle)
  settle_first, settle_last = int(settle_indices[0]), int(settle_indices[-1])
  settle_duration_s = float(time_s[settle_last] - time_s[settle_first])
  if settle_duration_s <= 0:
    raise ValueError("Raw settle has too few samples to assess plate support")
  settle_plate_base_mean_support_n = float(
    (plate_base_impulse[settle_last] - plate_base_impulse[settle_first])
    / settle_duration_s
  )
  criteria = {
    "sponge_starts_upright_on_table": bool(
      np.ptp(vertices[0, :, 2]) >= 0.09
      and clearance[0] <= 0.006
      and table_support[open_hand].max() >= 0.15
    ),
    "sponge_starts_at_right_shifted_target": bool(
      np.linalg.norm(vertices[0, :, :2].mean(axis=0) - initial_target_xy)
      < 0.015
      and vertices[0, :, 1].mean() < -0.03
    ),
    "pick_and_place_phase_order": ordered,
    "both_thumbs_stationary_during_open_hand": bool(
      left_thumb_excursion < MAXIMUM_INITIAL_THUMB_MOTION_RAD
      and right_thumb_excursion < MAXIMUM_INITIAL_THUMB_MOTION_RAD
    ),
    "right_wrist_stays_right_during_approach": bool(
      maximum_approach_wrist_y <= MAXIMUM_APPROACH_WRIST_Y_M
    ),
    "hands_stay_separated_during_approach": bool(
      minimum_approach_hand_separation >= MINIMUM_APPROACH_HAND_SEPARATION_M
    ),
    "palm_faces_down_at_overhead_waypoint": bool(
      overhead_palm_down >= MINIMUM_PALM_DOWN_COSINE
    ),
    "palm_faces_down_through_descent": bool(
      minimum_descent_palm_down >= MINIMUM_PALM_DOWN_COSINE
    ),
    "pregrasp_has_no_fingertip_contact": bool(
      normal[pregrasp].max() < 0.05
    ),
    "grasp_has_opposed_fingertip_contact": bool(
      normal[grasp][-1, 0] >= 0.05
      and normal[grasp][-1, 1:].sum() >= 0.15
    ),
    "five_fingers_continuously_sense_closed_grasp_and_loaded_motion": bool(
      np.all(minimum_five_finger_normal >= config.MINIMUM_FIVE_FINGER_NORMAL_FORCE_N)
      and np.all(minimum_five_finger_contacts > 0)
    ),
    "grasp_normal_force_below_5_n_per_pad": bool(
      grasp_peak_normal < MAXIMUM_CONTACT_PAD_FORCE_N
    ),
    "grasp_tangent_force_below_3_n_per_pad": bool(
      grasp_peak_tangent < MAXIMUM_CONTACT_TANGENT_FORCE_N
    ),
    "grasp_normal_force_jump_below_1_n": bool(
      grasp_normal_jump < MAXIMUM_GRASP_FORCE_JUMP_N
    ),
    "grasp_tangent_force_jump_below_1_n": bool(
      grasp_tangent_jump < MAXIMUM_GRASP_FORCE_JUMP_N
    ),
    "pick_to_place_duration_about_10_seconds": bool(
      abs(pick_to_place_s - config.PICK_TO_PLACE_SECONDS) <= 0.1
    ),
    "sponge_clear_of_table_during_carry": bool(
      np.min(clearance[carry]) >= MINIMUM_CARRY_CLEARANCE_M
    ),
    "table_does_not_support_sponge_during_carry": bool(
      np.max(table_support[carry]) < 0.05
    ),
    "sponge_pin_does_not_drop_during_level_carry": bool(
      carry_pin_drop <= MAXIMUM_CARRY_PIN_DROP_M
    ),
    "sponge_centroid_does_not_drop_during_level_carry": bool(
      carry_centroid_drop <= MAXIMUM_CARRY_PIN_DROP_M
    ),
    "sponge_bottom_does_not_drop_during_level_carry": bool(
      carry_bottom_drop <= MAXIMUM_CARRY_PIN_DROP_M
    ),
    "opposed_pad_contact_sustained_during_carry": bool(opposed_fraction >= 0.95),
    "sponge_remains_at_grasp_during_carry": bool(relative_drift < 0.05),
    "lift_to_release_normal_force_below_5_n_per_pad": bool(
      peak_normal < MAXIMUM_CONTACT_PAD_FORCE_N
    ),
    "lift_to_release_tangent_force_below_3_n_per_pad": bool(
      peak_tangent < MAXIMUM_CONTACT_TANGENT_FORCE_N
    ),
    f"loaded_motion_tangent_force_jump_below_{str(loaded_tangent_jump_limit).replace('.', '_')}_n": bool(
      loaded_motion_tangent_jump < loaded_tangent_jump_limit
    ),
    # A fingertip unloading from the grounded sponge can change shear faster
    # than a continuously loaded fingertip.  Keep this separate from the
    # stricter in-flight limit, and still require complete release below.
    f"release_tangent_force_jump_below_{str(release_tangent_jump_limit).replace('.', '_')}_n": bool(
      release_tangent_jump < release_tangent_jump_limit
    ),
    "right_fingertips_release_sponge": bool(
      release_final_loads.max() < 0.1 and settle_peak_loads.max() < 0.1
    ),
    "sponge_lands_fully_inside_plate": bool(
      final_center_offset <= 0.035
      and final_furthest_vertex_radius <= config.PLATE_INTERIOR_RADIUS_M
    ),
    "sponge_bottom_rests_on_plate": bool(
      config.PLATE_INTERIOR_TOP_Z_M - 0.005
      <= final_bottom_z
      <= config.PLATE_INTERIOR_TOP_Z_M + 0.015
    ),
    "sponge_height_stable_during_settle": bool(
      settle_pin_height_range <= MAXIMUM_SETTLE_HEIGHT_RANGE_M
      and settle_centroid_height_range <= MAXIMUM_SETTLE_HEIGHT_RANGE_M
      and settle_bottom_height_range <= MAXIMUM_SETTLE_HEIGHT_RANGE_M
    ),
    "plate_base_supports_sponge_at_end": bool(
      settle_plate_base_mean_support_n >= 0.20
    ),
    "table_does_not_support_sponge_at_end": bool(
      table_support[-1] < 0.05
    ),
    "taxel_force_conservation": taxels_conserve_force,
  }
  if randomized:
    criteria["loaded_motion_normal_force_jump_below_0_8_n"] = bool(
      loaded_motion_normal_jump < normal_jump_limit
    )
    criteria["release_normal_force_jump_below_1_0_n"] = bool(
      release_normal_jump < MAXIMUM_RANDOMIZED_RELEASE_NORMAL_JUMP_N
    )
  else:
    criteria["lift_to_release_normal_force_jump_below_0_6_n"] = bool(
      normal_jump < normal_jump_limit
    )
  return {
    "source": "unfiltered solver flex-contact taxels on the shared Raw clock",
    "randomized_initial_pose": randomized,
    "force_jump_limits_n": (
      {
        "loaded_motion_normal": normal_jump_limit,
        "release_normal": MAXIMUM_RANDOMIZED_RELEASE_NORMAL_JUMP_N,
        "loaded_motion_tangent": loaded_tangent_jump_limit,
        "release_tangent": release_tangent_jump_limit,
      }
      if randomized else {
        "lift_to_release_normal": normal_jump_limit,
        "loaded_motion_tangent": loaded_tangent_jump_limit,
        "release_tangent": release_tangent_jump_limit,
      }
    ),
    "pick_and_place_phase_sequence": list(transitions),
    "phase_boundaries_s": phase_boundaries,
    "initial_sponge_height_m": float(np.ptp(vertices[0, :, 2])),
    "initial_sponge_xy_m": vertices[0, :, :2].mean(axis=0).tolist(),
    "initial_target_sponge_xy_m": initial_target_xy.tolist(),
    "initial_sponge_clearance_m": float(clearance[0]),
    "left_thumb_maximum_open_hand_motion_rad": left_thumb_excursion,
    "right_thumb_maximum_open_hand_motion_rad": right_thumb_excursion,
    "maximum_approach_right_wrist_y_m": maximum_approach_wrist_y,
    "minimum_approach_hand_separation_m": minimum_approach_hand_separation,
    "overhead_palm_down_cosine": overhead_palm_down,
    "minimum_descent_palm_down_cosine": minimum_descent_palm_down,
    "pick_to_place_window_s": [float(time_s[first_lift]), float(time_s[final_release])],
    "pick_to_place_duration_s": pick_to_place_s,
    "carry_samples": int(np.count_nonzero(carry)),
    "minimum_clearance_during_carry_m": float(clearance[carry].min()),
    "carry_pin_initial_world_z_m": float(carry_pin_height[0]),
    "carry_pin_final_world_z_m": float(carry_pin_height[-1]),
    "maximum_carry_pin_world_z_drop_m": carry_pin_drop,
    "maximum_carry_centroid_world_z_drop_m": carry_centroid_drop,
    "maximum_carry_bottom_world_z_drop_m": carry_bottom_drop,
    "peak_table_support_during_carry_n": float(table_support[carry].max()),
    "opposed_contact_fraction_during_carry": opposed_fraction,
    "maximum_sponge_wrist_relative_drift_during_carry_m": relative_drift,
    "carry_mean_normal_fingertip_n": carried_normal.mean(axis=0).tolist(),
    "five_finger_contact_window_s": [
      float(time_s[np.flatnonzero(five_finger_window)[0]]),
      float(time_s[np.flatnonzero(five_finger_window)[-1]]),
    ],
    "minimum_closed_grasp_to_placement_normal_fingertip_n": (
      minimum_five_finger_normal.tolist()
    ),
    "minimum_closed_grasp_to_placement_contact_count": (
      minimum_five_finger_contacts.tolist()
    ),
    "grasp_peak_normal_fingertip_n": grasp_normal.max(axis=0).tolist(),
    "grasp_peak_tangent_fingertip_n": grasp_tangent.max(axis=0).tolist(),
    "grasp_maximum_normal_force_jump_n": grasp_normal_jump,
    "grasp_maximum_tangent_force_jump_n": grasp_tangent_jump,
    "lift_to_release_peak_normal_fingertip_n": contact_normal.max(axis=0).tolist(),
    "lift_to_release_peak_tangent_fingertip_n": contact_tangent.max(axis=0).tolist(),
    "lift_to_release_maximum_normal_force_jump_n": normal_jump,
    "loaded_motion_maximum_normal_force_jump_n": loaded_motion_normal_jump,
    "release_maximum_normal_force_jump_n": release_normal_jump,
    "lift_to_release_maximum_tangent_force_jump_n": tangent_jump,
    "loaded_motion_maximum_tangent_force_jump_n": loaded_motion_tangent_jump,
    "release_maximum_tangent_force_jump_n": release_tangent_jump,
    "release_final_normal_fingertip_n": release_final_loads.tolist(),
    "settle_peak_normal_fingertip_n": settle_peak_loads.tolist(),
    "final_sponge_plate_center_offset_m": final_center_offset,
    "final_sponge_maximum_plate_radius_m": final_furthest_vertex_radius,
    "final_sponge_bottom_z_m": final_bottom_z,
    "settle_pin_world_z_range_m": settle_pin_height_range,
    "settle_centroid_world_z_range_m": settle_centroid_height_range,
    "settle_bottom_world_z_range_m": settle_bottom_height_range,
    "final_plate_base_support_force_n": float(plate_base_support[-1]),
    "final_plate_base_support_impulse_ns": float(plate_base_impulse[-1]),
    "settle_plate_base_support_measurement_window_s": [
      float(time_s[settle_first]), float(time_s[settle_last])
    ],
    "settle_plate_base_support_measurement_duration_s": settle_duration_s,
    "settle_plate_base_mean_support_from_impulse_n": settle_plate_base_mean_support_n,
    "final_plate_support_force_n": float(plate_support[-1]),
    "final_table_support_force_n": float(table_support[-1]),
    "criteria": criteria,
    "passed": bool(all(criteria.values())),
  }


def _curves(file: h5py.File, output: Path, audit: dict) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  indices = _force_layout(file)
  force = file["tactile_contact_force"]
  time_s = force["timestamp"][:]
  normal = force["normal_force_n"][:, indices]
  tangent = np.linalg.norm(force["tangent_force_n"][:, indices], axis=-1)
  start, last = audit["pick_to_place_window_s"]
  output.mkdir()
  figure, axes = plt.subplots(5, 2, figsize=(14, 11), sharex=True)
  for finger, row in enumerate(axes):
    for axis, values, label in zip(
      row, (normal, tangent), ("Fn (N)", "|Ft| (N)"), strict=True
    ):
      axis.plot(time_s, values[:, finger], lw=0.75)
      axis.axvspan(start, last, color="orange", alpha=0.17)
      axis.set_ylabel(f"{FINGERS[finger]}\n{label}")
      axis.grid(alpha=0.25)
  for name in ("lift", "carry_to_plate", "lower_into_plate", "release"):
    boundary = audit["phase_boundaries_s"][name]
    for axis in axes.flat:
      axis.axvline(boundary, color="0.45", alpha=0.35, lw=0.65)
    axes[0, 0].text(
      boundary + 0.08, 0.96, name.replace("_", " "),
      transform=axes[0, 0].get_xaxis_transform(),
      fontsize=7, va="top", rotation=90,
    )
  for axis in axes[-1]:
    axis.set_xlabel("Simulation time (s)")
  figure.suptitle("Sponge to plate: recorded right-hand contact forces (100 Hz)")
  figure.tight_layout()
  figure.savefig(output / "right_hand_force_curves.png", dpi=150)
  plt.close(figure)


def _review(file: h5py.File, output: Path) -> dict:
  frames = plan_review_frames(file, fps=10, second_camera="right_wrist")
  indices = _force_layout(file)
  force = file["tactile_contact_force"]
  phases = file["commands/phase"].asstr()
  output.mkdir()
  video = output / "review.mp4"
  writer = _FfmpegPipeWriter(video, fps=10, width=1280, height=720)
  try:
    for frame in frames:
      i, k = frame.camera_index, frame.tactile_index
      composite, _ = compose_review_frame(
        file["cameras/head/rgb"][i],
        file["cameras/right_wrist/rgb"][i],
        force["normal_taxel_force_n"][k][indices],
        force["tangent_taxel_force_n"][k][indices],
        frame=frame,
        width=1280,
        height=720,
        phase=phases[k],
        heading="SPONGE TO PLATE",
        second_camera_label="RIGHT WRIST",
      )
      writer.write(np.asarray(composite))
    writer.finish()
  except BaseException:
    writer.abort()
    raise
  probe = _probe_video(video)
  _validate_probe(probe, count=len(frames), fps=10, width=1280, height=720)
  return {
    "source": "../raw/sponge_grasp_000000.h5",
    "frame_count": len(frames),
    "fps": 10,
    "camera_names": ["head", "right_wrist"],
    "tactile_panels": ["right_hand_normal", "right_hand_tangent"],
    "maximum_tactile_age_s": max(frame.tactile_age_s for frame in frames),
    "video_validation": probe,
  }


def record_compact_example(
  output: str | Path,
  *,
  replace_existing: bool = False,
  camera_hz: int = 10,
  buffer_rows: int = 128,
) -> dict:
  """Record one validated run, then publish the light-bulb five-file layout."""
  from .execution import SpongeGraspExecutor
  from .task import SpongeGraspSimulation

  if camera_hz < 10:
    raise ValueError("camera_hz must be at least 10 for the 10 Hz review")
  output = Path(output).expanduser().absolute()
  output.parent.mkdir(parents=True, exist_ok=True)
  if output.is_symlink():
    raise ValueError("refusing a symbolic-link example output")
  if output.exists():
    if not replace_existing:
      raise FileExistsError(output)
    marker = output / "raw/sponge_grasp_000000.h5"
    if not marker.is_file():
      raise ValueError("refusing to replace a directory without sponge Raw")
    with h5py.File(marker, "r") as previous:
      previous_scene = json.loads(previous.attrs["metadata_json"]).get("scene")
    if previous_scene != config.SCENE_NAME:
      raise ValueError("refusing to replace an unrelated example directory")

  with tempfile.TemporaryDirectory(prefix=".sponge_grasp_", dir=output.parent) as temp:
    staging = Path(temp) / "example"
    (staging / "raw").mkdir(parents=True)
    simulation = SpongeGraspSimulation()
    executor = SpongeGraspExecutor(simulation)
    capture = WorkcellConfig(
      model_path=simulation.model_path,
      physics_hz=round(1.0 / simulation.timestep),
      control_hz=100,
      camera_hz=camera_hz,
      cameras=tuple(
        CameraConfig(name, width=320, height=240, depth=False, segmentation=False)
        for name in TRAINING_CAMERA_NAMES
      ),
      tactile_provider=SolverContactTactileProvider.source,
    )
    raw = staging / "raw/sponge_grasp_000000.h5"
    with SpongeGraspEpisodeRecorder(
      raw, simulation, capture, buffer_rows=buffer_rows
    ) as recorder:
      executor.observer = lambda sim, phase, state: recorder.observe(sim, phase)
      recorder.record_initial("tabletop_ready")
      result = executor.run()
      if not result.get("success"):
        raise RuntimeError(f"sponge placement failed: {result.get('reason')}")
      result["contact_integrity"] = simulation.contact_integrity_report()
      # The episode ends with the released sponge supported by the plate.
      recorder.record_terminal("settle_in_plate")
      recorder.set_outcome(result)

    validation = validate_episode(raw)
    if not validation.valid:
      raise ValueError(f"shared Raw validation failed: {validation.errors}")
    with h5py.File(raw, "r") as file:
      audit = audit_recorded_grasp(file)
      if not audit["passed"]:
        failed = [name for name, passed in audit["criteria"].items() if not passed]
        raise ValueError(f"recorded sponge placement audit failed: {failed}")
      _curves(file, staging / "curves", audit)
      review = _review(file, staging / "review")
    outcome = {
      **result,
      "example_validation": {
        "raw": asdict(validation),
        "audit": audit,
        "review": review,
        "raw_sha256": _sha256(raw),
        "layout": "light_bulb_example_raw_review_curves",
      },
    }
    _json(raw.with_suffix(".result.json"), outcome)
    manifest_path = raw.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(validation=asdict(validation), task_audit_passed=True)
    _json(manifest_path, manifest)
    expected = {
      "raw/sponge_grasp_000000.h5",
      "raw/sponge_grasp_000000.json",
      "raw/sponge_grasp_000000.result.json",
      "review/review.mp4",
      "curves/right_hand_force_curves.png",
    }
    actual = {str(path.relative_to(staging)) for path in staging.rglob("*") if path.is_file()}
    if actual != expected:
      raise ValueError(f"compact example layout mismatch: {sorted(actual ^ expected)}")
    previous = Path(temp) / "previous"
    if output.exists():
      output.rename(previous)
    try:
      staging.rename(output)
    except BaseException:
      if previous.exists():
        previous.rename(output)
      raise
    if previous.exists():
      shutil.rmtree(previous)
  return outcome


def record_raw_episode(
  output: str | Path,
  *,
  seed: int | None = None,
  camera_hz: int = 30,
  cameras: tuple[str, ...] = TRAINING_CAMERA_NAMES,
  buffer_rows: int = 128,
) -> dict:
  """Record one audited three-camera Raw episode for the batch collector.

  Seeded runs translate the upright sponge by at most 1 mm per XY axis.
  The same seed reproduces the same pose; the controller and plate stay fixed.
  """
  from .execution import SpongeGraspExecutor
  from .task import SpongeGraspSimulation

  if seed is not None and seed < 0:
    raise ValueError("seed must be nonnegative")
  if camera_hz <= 0:
    raise ValueError("camera_hz must be positive")
  if tuple(cameras) != TRAINING_CAMERA_NAMES:
    raise ValueError("raw cameras must be head, left_wrist, right_wrist")
  if not 0 <= buffer_rows <= 256:
    raise ValueError("buffer_rows must be in 0..256")
  output = Path(output).expanduser().absolute()
  partial = output.with_name(output.name + ".partial")
  if output.exists() or output.is_symlink() or partial.exists() or partial.is_symlink():
    raise FileExistsError(f"raw output or partial already exists: {output}")
  output.parent.mkdir(parents=True, exist_ok=True)
  partial.mkdir()
  started = time.monotonic()
  try:
    (partial / "raw").mkdir()
    simulation = SpongeGraspSimulation(
      position_seed=seed,
      randomize_xy=seed is not None,
    )
    executor = SpongeGraspExecutor(simulation)
    capture = WorkcellConfig(
      model_path=simulation.model_path,
      physics_hz=round(1.0 / simulation.timestep),
      control_hz=100,
      camera_hz=camera_hz,
      cameras=tuple(
        CameraConfig(name, width=320, height=240, depth=False, segmentation=False)
        for name in cameras
      ),
      tactile_provider=SolverContactTactileProvider.source,
    )
    raw = partial / "raw/sponge_grasp_000000.h5"
    with SpongeGraspEpisodeRecorder(
      raw, simulation, capture, buffer_rows=buffer_rows
    ) as recorder:
      executor.observer = lambda sim, phase, state: recorder.observe(sim, phase)
      recorder.record_initial("tabletop_ready")
      result = executor.run()
      if not result.get("success"):
        raise RuntimeError(f"sponge placement failed: {result.get('reason')}")
      result["contact_integrity"] = simulation.contact_integrity_report()
      result["collection_seed"] = seed
      result["initial_condition_randomized"] = simulation.randomize_xy
      result["sponge_initial_xy_m"] = simulation.sponge_initial_xy_m.tolist()
      result["sponge_xy_offset_m"] = simulation.sponge_xy_offset_m.tolist()
      result["sponge_xy_offset_low_m"] = simulation.xy_offset_low_m.tolist()
      result["sponge_xy_offset_high_m"] = simulation.xy_offset_high_m.tolist()
      recorder.record_terminal("settle_in_plate")
      recorder.set_outcome(result)
    validation = validate_episode(raw)
    if not validation.valid:
      raise ValueError(f"shared Raw validation failed: {validation.errors}")
    with h5py.File(raw, "r") as file:
      audit = audit_recorded_grasp(file)
    if not audit["passed"]:
      failed = [name for name, passed in audit["criteria"].items() if not passed]
      raise ValueError(f"recorded sponge placement audit failed: {failed}")
    _json(raw.with_suffix(".result.json"), result)
    manifest_path = raw.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
      validation=asdict(validation),
      task_audit_passed=True,
      recording_wall_seconds=time.monotonic() - started,
    )
    _json(manifest_path, manifest)
    partial.rename(output)
    print(f"Saved validated raw-only sponge episode: {output}", flush=True)
    return {"result": result, "validation": asdict(validation), "audit": audit}
  except BaseException as error:
    _json(
      partial / "failure.json",
      {
        "completed": False,
        "scene": config.SCENE_NAME,
        "error_type": type(error).__name__,
        "error": str(error),
      },
    )
    raise
