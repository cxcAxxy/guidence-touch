"""Opt-in close-up cameras for the poker table-edge pressure experiment.

Only the recorder's free camera and this simulation instance's overhead camera
are changed. No renderer, physics step, contact update, or live ``mj_forward``
is needed. Call after reset, before the recorder observes its first frame.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import mujoco
import numpy as np


def _named_id(model: mujoco.MjModel, object_type: Any, name: str) -> int:
  object_id = mujoco.mj_name2id(model, object_type, name)
  if object_id < 0:
    raise ValueError(f"pressure-window video requires {name!r}")
  return object_id


def configure_pressure_window_video(video: Any, simulation: Any) -> dict[str, Any]:
  """Frame the starting card, right hand, and robot-side table edge.

  The returned JSON-safe metadata is also copied into the recorder's metadata.
  The generic recorder reports ``cameras.main='viewer'`` for any free camera;
  here it is a fixed experimental close-up, not a native viewer or GUI. The
  shared dashboard's ``FRONT`` caption denotes its main pane, not this camera's
  orientation. Other task instances and MJCF files remain unchanged.
  """
  if getattr(simulation, "scene", None) != "poker-draw":
    raise ValueError("pressure-window close-up is only for the poker-draw scene")
  if video.simulation is not simulation:
    raise ValueError("video recorder must belong to this simulation instance")

  model, data = simulation.model, simulation.data
  table_id = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "poker_table_top")
  card_id = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "card_core_geom")
  overhead_id = _named_id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
  if model.cam_bodyid[overhead_id] != 0:
    raise ValueError("pressure-window overhead camera must belong to the world body")
  if model.cam_mode[overhead_id] != mujoco.mjtCamLight.mjCAMLIGHT_FIXED:
    raise ValueError("pressure-window overhead camera must use fixed camera mode")
  if model.cam_projection[overhead_id] != mujoco.mjtProjection.mjPROJ_PERSPECTIVE:
    raise ValueError("pressure-window overhead camera must use perspective projection")
  if model.geom_type[table_id] != mujoco.mjtGeom.mjGEOM_BOX:
    raise ValueError("pressure-window table top must be a box")

  table_center = data.geom_xpos[table_id].copy()
  table_rotation = data.geom_xmat[table_id].reshape(3, 3)
  card_start = data.geom_xpos[card_id].copy()
  if not all(
    np.all(np.isfinite(value)) for value in (table_center, table_rotation, card_start)
  ):
    raise ValueError("pressure-window camera geometry must be finite")
  if not np.allclose(np.abs(table_rotation), np.eye(3), atol=1.0e-6):
    raise ValueError("pressure-window camera requires an axis-aligned table top")
  half_extent = np.abs(table_rotation) @ model.geom_size[table_id]
  edge_x = float(table_center[0] - half_extent[0])
  tabletop_z = float(table_center[2] + half_extent[2])
  if card_start[0] < edge_x:
    raise ValueError("configure pressure-window cameras before drawing the card")
  lookat = np.asarray(
    [0.5 * (card_start[0] + edge_x), card_start[1], tabletop_z],
    dtype=np.float64,
  )

  # At the current table this centers x=0.5225 m, rather than the robot torso.
  # The 11.4 cm draw is visible sideways in the oblique main view. The overhead
  # covers about 36 cm vertically, retaining the complete card and table edge.
  framing_distance = max(0.50, 3.0 * abs(float(card_start[0]) - edge_x))
  overhead_height = max(0.50, framing_distance)
  camera = mujoco.MjvCamera()
  mujoco.mjv_defaultCamera(camera)
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.lookat[:] = lookat
  camera.distance = framing_distance
  camera.azimuth = 135.0
  camera.elevation = -35.0

  previous = {
    "position_m": model.cam_pos[overhead_id].tolist(),
    "quaternion_wxyz": model.cam_quat[overhead_id].tolist(),
    "fovy_deg": float(model.cam_fovy[overhead_id]),
  }
  original = getattr(simulation, "_pressure_window_original_overhead_camera", previous)
  metadata = {
    "preset": "poker_table_edge_closeup_v1",
    "scope": "visual_cameras_only_on_this_simulation_instance",
    "main": {
      "mode": "fixed_free_camera_closeup",
      "lookat_world_m": lookat.tolist(),
      "distance_m": framing_distance,
      "azimuth_deg": float(camera.azimuth),
      "elevation_deg": float(camera.elevation),
      "recorder_mode_label": "viewer",
      "dashboard_pane_label": "FRONT",
      "native_viewer_opened": False,
    },
    "overhead": {
      "name": "overhead",
      "position_m": [float(lookat[0]), float(lookat[1]), tabletop_z + overhead_height],
      "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
      "fovy_deg": 40.0,
      "original_before_first_configuration": deepcopy(original),
      "previous_before_this_configuration": previous,
    },
    "geometry_at_configuration": {
      "card_center_world_m": card_start.tolist(),
      "table_robot_side_edge_x_m": edge_x,
      "tabletop_z_m": tabletop_z,
      "drag_direction_world": [-1.0, 0.0, 0.0],
    },
  }

  # This public recorder API retains the free camera itself. Validate/attach it
  # before changing model camera arrays so a closed recorder causes no mutation.
  video.follow_viewer_camera(camera)
  model.cam_pos[overhead_id] = metadata["overhead"]["position_m"]
  model.cam_quat[overhead_id] = metadata["overhead"]["quaternion_wxyz"]
  model.cam_fovy[overhead_id] = metadata["overhead"]["fovy_deg"]
  mujoco.mj_camlight(model, data)
  simulation._pressure_window_original_overhead_camera = deepcopy(original)
  video._metadata["pressure_window_camera"] = deepcopy(metadata)
  return metadata
