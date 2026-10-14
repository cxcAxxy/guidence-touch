"""Named-camera RGB, metric-depth and instance-segmentation capture."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import CameraConfig
from .render_backend import current_backend, prepare_render_backend


@dataclass(frozen=True)
class CameraCalibration:
  name: str
  width: int
  height: int
  fovy_degrees: float
  intrinsic: np.ndarray
  world_from_camera: np.ndarray


class WorkcellRenderer:
  """Reuse one MuJoCo offscreen renderer per requested image resolution."""

  def __init__(
    self,
    model: mujoco.MjModel,
    cameras: tuple[CameraConfig, ...],
    *,
    visible_geom_groups: tuple[int, ...] | None = None,
    shadows: bool = True,
  ) -> None:
    self.model = model
    self.cameras = tuple(cameras)
    self._scene_option: mujoco.MjvOption | None = None
    if visible_geom_groups is not None:
      invalid_groups = [
        group for group in visible_geom_groups if not 0 <= group < 6
      ]
      if invalid_groups:
        raise ValueError(f"geom groups must be in [0, 5], got {invalid_groups}")
      self._scene_option = mujoco.MjvOption()
      self._scene_option.geomgroup[:] = 0
      self._scene_option.geomgroup[list(visible_geom_groups)] = 1
    self._camera_ids = {
      camera.name: _require_camera(model, camera.name) for camera in self.cameras
    }
    self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
    self.backend_info: dict = {}
    prepare_render_backend()
    try:
      for camera in self.cameras:
        key = (camera.width, camera.height)
        if key not in self._renderers:
          self._renderers[key] = mujoco.Renderer(
            model, height=camera.height, width=camera.width
          )
          self.backend_info = current_backend()
          if not shadows:
            self._renderers[key].scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    except BaseException:
      self.close()
      raise
    if self.backend_info:
      print(f"OpenGL renderer: {self.backend_info['renderer']} "
            f"(requested={self.backend_info['requested']})", flush=True)

  def close(self) -> None:
    for renderer in self._renderers.values():
      renderer.close()
    self._renderers.clear()

  def __enter__(self) -> WorkcellRenderer:
    return self

  def __exit__(self, *_: object) -> None:
    self.close()

  def calibration(self, data: mujoco.MjData, camera: CameraConfig) -> CameraCalibration:
    camera_id = self._camera_ids[camera.name]
    fovy = float(self.model.cam_fovy[camera_id])
    focal_y = 0.5 * camera.height / np.tan(np.deg2rad(fovy) * 0.5)
    focal_x = focal_y
    intrinsic = np.array(
      [
        [focal_x, 0.0, (camera.width - 1.0) * 0.5],
        [0.0, focal_y, (camera.height - 1.0) * 0.5],
        [0.0, 0.0, 1.0],
      ],
      dtype=np.float64,
    )
    world_from_camera = np.eye(4, dtype=np.float64)
    world_from_camera[:3, :3] = data.cam_xmat[camera_id].reshape(3, 3)
    world_from_camera[:3, 3] = data.cam_xpos[camera_id]
    return CameraCalibration(
      name=camera.name,
      width=camera.width,
      height=camera.height,
      fovy_degrees=fovy,
      intrinsic=intrinsic,
      world_from_camera=world_from_camera,
    )

  def capture(self, data: mujoco.MjData, camera: CameraConfig) -> dict[str, np.ndarray]:
    renderer = self._renderers[(camera.width, camera.height)]
    output: dict[str, np.ndarray] = {}
    renderer.disable_depth_rendering()
    renderer.disable_segmentation_rendering()
    if camera.rgb:
      renderer.update_scene(
        data, camera=camera.name, scene_option=self._scene_option
      )
      output["rgb"] = np.asarray(renderer.render()).copy()
    if camera.depth:
      renderer.enable_depth_rendering()
      renderer.update_scene(
        data, camera=camera.name, scene_option=self._scene_option
      )
      output["depth"] = np.asarray(renderer.render()).copy()
      renderer.disable_depth_rendering()
    if camera.segmentation:
      renderer.enable_segmentation_rendering()
      renderer.update_scene(
        data, camera=camera.name, scene_option=self._scene_option
      )
      output["segmentation"] = np.asarray(renderer.render()).copy()
      renderer.disable_segmentation_rendering()
    return output


def _require_camera(model: mujoco.MjModel, name: str) -> int:
  camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
  if camera_id < 0:
    raise RuntimeError(f"MuJoCo model is missing camera {name!r}")
  return camera_id


def set_fixed_camera_lookat(
  model: mujoco.MjModel,
  name: str,
  *,
  position: np.ndarray,
  lookat: np.ndarray,
  up: np.ndarray | None = None,
  fovy_degrees: float | None = None,
) -> None:
  """Set a world-attached MJCF camera from an intuitive look-at definition."""
  position = np.asarray(position, dtype=float)
  lookat = np.asarray(lookat, dtype=float)
  up = np.asarray([0.0, 0.0, 1.0] if up is None else up, dtype=float)
  if position.shape != (3,) or lookat.shape != (3,) or up.shape != (3,):
    raise ValueError("position, lookat and up must have shape (3,)")
  camera_z = position - lookat
  distance = float(np.linalg.norm(camera_z))
  if distance < 1.0e-9:
    raise ValueError("camera position and lookat cannot coincide")
  camera_z /= distance
  camera_x = np.cross(up, camera_z)
  norm_x = float(np.linalg.norm(camera_x))
  if norm_x < 1.0e-9:
    fallback_up = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(fallback_up, camera_z))) > 0.99:
      fallback_up = np.array([1.0, 0.0, 0.0])
    camera_x = np.cross(fallback_up, camera_z)
    norm_x = float(np.linalg.norm(camera_x))
  camera_x /= norm_x
  camera_y = np.cross(camera_z, camera_x)
  world_from_camera = np.column_stack((camera_x, camera_y, camera_z))
  quaternion = np.empty(4, dtype=float)
  mujoco.mju_mat2Quat(quaternion, world_from_camera.reshape(9))
  camera_id = _require_camera(model, name)
  model.cam_pos[camera_id] = position
  model.cam_quat[camera_id] = quaternion
  if fovy_degrees is not None:
    if not 1.0 < fovy_degrees < 179.0:
      raise ValueError("fovy_degrees must be between 1 and 179")
    model.cam_fovy[camera_id] = fovy_degrees
