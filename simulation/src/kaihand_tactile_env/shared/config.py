"""Configuration and fixed naming contract for the KaiHand workcell."""

from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from kaihand_tactile_env.tasks.bulb_screw import config as bulb_screw_config
from kaihand_tactile_env.tasks.bulb_unscrew import config as bulb_unscrew_config
from kaihand_tactile_env.tasks.install_ram import config as install_ram_config
from kaihand_tactile_env.tasks.pick_place import config as pick_place_config
from kaihand_tactile_env.tasks.poker_draw import config as poker_draw_config
from kaihand_tactile_env.tasks.sponge_grasp import config as sponge_grasp_config
from kaihand_tactile_env.tasks.usb_insert import config as usb_insert_config
from kaihand_tactile_env.tasks.usb_unplug import config as usb_unplug_config
from kaihand_tactile_env.tasks.vase_wipe import config as vase_wipe_config
from kaihand_tactile_env.tasks.whiteboard_wipe import config as whiteboard_wipe_config
from kaihand_tactile_env.tasks.whiteboard_wipe_horizontal import (
  config as whiteboard_horizontal_config,
)
from kaihand_tactile_env.tasks.whiteboard_wipe_horizontal_line import (
  config as whiteboard_horizontal_line_config,
)

from .cameras import LEGACY_CAPTURE_CAMERA_NAMES
from .cameras import ROBOT_CAMERA_NAMES as ROBOT_CAMERA_NAMES
from .cameras import SHARED_CAMERA_NAMES as SHARED_CAMERA_NAMES
from .cameras import TRAINING_CAMERA_NAMES as TRAINING_CAMERA_NAMES
from .posture import ARM_HOME as ARM_HOME

SIDES = ("left", "right")
WRIST_FT_SITE_NAMES = tuple(f"{side}_wrist_ft_site" for side in SIDES)
WRIST_FORCE_SENSOR_NAMES = tuple(f"{side}_wrist_force" for side in SIDES)
WRIST_TORQUE_SENSOR_NAMES = tuple(f"{side}_wrist_torque" for side in SIDES)
TASK_CONFIGS = {
  pick_place_config.SCENE_NAME: pick_place_config,
  poker_draw_config.SCENE_NAME: poker_draw_config,
  usb_insert_config.SCENE_NAME: usb_insert_config,
  bulb_screw_config.SCENE_NAME: bulb_screw_config,
  vase_wipe_config.SCENE_NAME: vase_wipe_config,
  install_ram_config.SCENE_NAME: install_ram_config,
  whiteboard_wipe_config.SCENE_NAME: whiteboard_wipe_config,
  sponge_grasp_config.SCENE_NAME: sponge_grasp_config,
  usb_unplug_config.SCENE_NAME: usb_unplug_config,
  whiteboard_horizontal_config.SCENE_NAME: whiteboard_horizontal_config,
  whiteboard_horizontal_line_config.SCENE_NAME: whiteboard_horizontal_line_config,
  bulb_unscrew_config.SCENE_NAME: bulb_unscrew_config,
}
SCENE_NAMES = tuple(TASK_CONFIGS)
SCENE_OBJECTS = {scene: config.OBJECT_NAMES for scene, config in TASK_CONFIGS.items()}
OBJECT_NAMES = tuple(
  dict.fromkeys(name for names in SCENE_OBJECTS.values() for name in names)
)
TACTILE_PROVIDERS = ("solver_contact_proxy_v1", "genesis_probe_bimanual_clean_v1")
FINGERTIP_LINK_NAMES = tuple(
  f"hand_{side[0]}_{finger}_{link}"
  for side in SIDES
  for finger, link in (
    ("thumb", "link6"),
    ("index", "link4"),
    ("middle", "link4"),
    ("ring", "link4"),
    ("pinky", "link4"),
  )
)


def task_config(scene: str) -> ModuleType:
  """Return task-local settings without importing a task's executor."""
  if scene not in TASK_CONFIGS:
    raise ValueError(f"scene must be one of {SCENE_NAMES}, got {scene!r}")
  return TASK_CONFIGS[scene]


def default_model_path(scene: str = "pick-place") -> Path:
  """Select an independent scene; the historical default stays pick-place."""
  return Path(task_config(scene).__file__).resolve().with_name("scene.xml")


def legacy_model_path() -> Path:
  """Frozen combined scene used to interpret recordings from before the split."""
  return (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "workcell"
    / "mjcf"
    / "kaihand_dual_arm_workcell.xml"
  )


def model_fingerprint(model_path: str | Path) -> str:
  """Hash MJCF source and recursive includes, including shared robot settings.

  Relative dependency names keep the identity stable when the project moves.
  Mesh assets retain their separate asset-manifest identity.
  """
  root_path = Path(model_path).expanduser().resolve()
  digest = hashlib.sha256(b"kaihand-mjcf-includes-v1\0")
  visited: set[Path] = set()
  active: set[Path] = set()

  def visit(path: Path) -> None:
    path = path.resolve()
    if path in active:
      raise ValueError(f"cyclic MJCF include at {path}")
    if path in visited:
      return
    active.add(path)
    content = path.read_bytes()
    relative_name = os.path.relpath(path, root_path.parent)
    digest.update(relative_name.encode("utf-8") + b"\0")
    digest.update(len(content).to_bytes(8, "big"))
    digest.update(content)
    for include in ET.fromstring(content).iter("include"):
      visit(path.parent / include.attrib["file"])
    active.remove(path)
    visited.add(path)

  visit(root_path)
  return digest.hexdigest()


@dataclass(frozen=True)
class CameraConfig:
  """Runtime image settings for one named MJCF camera."""

  name: str
  width: int = 320
  height: int = 240
  rgb: bool = True
  depth: bool = True
  segmentation: bool = True

  def __post_init__(self) -> None:
    if self.width <= 0 or self.height <= 0:
      raise ValueError("camera width and height must be positive")
    if not (self.rgb or self.depth or self.segmentation):
      raise ValueError("at least one camera modality must be enabled")


@dataclass(frozen=True)
class WorkcellConfig:
  """Timing, scene and capture defaults shared by tools and datasets."""

  model_path: Path = field(default_factory=default_model_path)
  physics_hz: int = 500
  control_hz: int = 100
  camera_hz: int = 30
  cameras: tuple[CameraConfig, ...] = field(
    default_factory=lambda: tuple(
      CameraConfig(name) for name in LEGACY_CAPTURE_CAMERA_NAMES
    )
  )
  tactile_provider: str = "genesis_probe_bimanual_clean_v1"

  def __post_init__(self) -> None:
    if self.physics_hz <= 0 or self.control_hz <= 0 or self.camera_hz <= 0:
      raise ValueError("sample rates must be positive")
    if self.physics_hz % self.control_hz:
      raise ValueError("physics_hz must be an integer multiple of control_hz")
    if self.control_hz > self.physics_hz:
      raise ValueError("control_hz cannot exceed physics_hz")
    if len({camera.name for camera in self.cameras}) != len(self.cameras):
      raise ValueError("camera names must be unique")
    if self.tactile_provider not in TACTILE_PROVIDERS:
      raise ValueError(
        f"tactile_provider must be one of {TACTILE_PROVIDERS}, "
        f"got {self.tactile_provider!r}"
      )

  @property
  def control_decimation(self) -> int:
    return self.physics_hz // self.control_hz
