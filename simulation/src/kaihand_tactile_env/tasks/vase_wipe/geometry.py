"""Rebuild the task-local, genuinely hollow ceramic vase from its radial profile.

Run ``python -m kaihand_tactile_env.tasks.vase_wipe.geometry`` after editing the
profile. Convex sector meshes provide inner/outer/shoulder collision surfaces;
the detailed visual shell and deterministic glaze texture share that profile.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from . import config


def _numbers(values):
  return " ".join(f"{x:.8g}" for x in np.asarray(values).ravel())


def rebuild():
  path = Path(__file__).with_name("scene.xml")
  tree = ET.parse(path)
  root = tree.getroot()
  defaults = root.find("default")
  legacy_foam = defaults.find("default[@class='sponge_foam']")
  if legacy_foam is not None:
    defaults.remove(legacy_foam)
  sensors = root.find("sensor")
  if sensors is None:
    sensors = ET.SubElement(root, "sensor")
  if sensors.find("force[@name='vase_wrist_force']") is None:
    ET.SubElement(sensors, "force", name="vase_wrist_force", site="right_ee_site")
  vase = root.find(".//body[@name='vase']")
  vase.clear()
  vase.attrib.update(name="vase", pos=_numbers(config.VASE_CENTER))
  asset = root.find("asset")
  for child in list(asset):
    if child.get("name", "").startswith("vase_"):
      asset.remove(child)
  ET.SubElement(asset, "texture", name="vase_glaze", type="2d", file="vase_glaze.png")
  ET.SubElement(
    asset,
    "material",
    name="vase_glaze",
    texture="vase_glaze",
    texuniform="false",
    specular=".28",
    shininess=".25",
  )
  ET.SubElement(
    asset,
    "material",
    name="vase_clay",
    rgba=_numbers(config.VASE_CLAY_RGBA),
    specular=".1",
  )
  ET.SubElement(
    vase,
    "geom",
    name="vase_foot",
    type="cylinder",
    size=".051 .008",
    pos="0 0 .008",
    **{
      "class": "vase_ceramic",
      "material": "vase_clay",
      "rgba": _numbers(config.VASE_CLAY_RGBA),
    },
  )
  ET.SubElement(
    vase,
    "geom",
    name="vase_floor",
    type="cylinder",
    size=f"{config.inner_radius(config.FLOOR_HEIGHT)} .007",
    pos=f"0 0 {config.FLOOR_HEIGHT - 0.007}",
    **{
      "class": "vase_ceramic",
      "material": "vase_clay",
      "rgba": _numbers(config.VASE_CLAY_RGBA),
    },
  )
  cube_faces = [
    0,
    2,
    1,
    0,
    3,
    2,
    4,
    5,
    6,
    4,
    6,
    7,
    0,
    1,
    5,
    0,
    5,
    4,
    1,
    2,
    6,
    1,
    6,
    5,
    2,
    3,
    7,
    2,
    7,
    6,
    3,
    0,
    4,
    3,
    4,
    7,
  ]
  profile = np.array(config.INNER_PROFILE)
  for k in range(len(profile) - 1):
    z0, r0 = profile[k]
    z1, r1 = profile[k + 1]
    for i in range(config.WALL_COUNT):
      a, b = np.array([i - 0.5, i + 0.5]) * 2 * np.pi / config.WALL_COUNT
      # Each frustum sector has its own hull. Never make a hull spanning the bore.
      vertices = []
      for z, radius in [(z0, r0), (z1, r1)]:
        for rad, angle in [
          (radius, a),
          (radius, b),
          (radius + 0.007, b),
          (radius + 0.007, a),
        ]:
          vertices.append((rad * np.cos(angle), rad * np.sin(angle), z))
      name = f"vase_wall_{k * config.WALL_COUNT + i}"
      ET.SubElement(
        asset, "mesh", name=name, vertex=_numbers(vertices), face=_numbers(cube_faces)
      )
      ET.SubElement(
        vase,
        "geom",
        name=name,
        type="mesh",
        mesh=name,
        group="3",
        **{"class": "vase_ceramic"},
      )
  # Revolved ivory visual surface: horizontal throwing ridges and rolled lip.
  zvalues = np.linspace(profile[0, 0], profile[-1, 0], 70)
  inner = np.interp(zvalues, profile[:, 0], profile[:, 1])
  outer = inner + 0.007 + 0.00035 * np.sin(zvalues * 2 * np.pi / 0.006)
  n = 96
  vertices, texcoords, faces = [], [], []
  rings = [
    *(zip(zvalues, outer, strict=True)),
    *(zip(zvalues[::-1], inner[::-1], strict=True)),
  ]
  for z, radius in rings:
    for i in range(n + 1):
      angle = 2 * np.pi * i / n
      vertices.append([radius * np.cos(angle), radius * np.sin(angle), z])
      texcoords.append([i / n, (z - profile[0, 0]) / (profile[-1, 0] - profile[0, 0])])
  for k in range(len(rings)):
    for i in range(n):
      a = k * (n + 1) + i
      b = a + 1
      c = ((k + 1) % len(rings)) * (n + 1) + i + 1
      d = c - 1
      faces.extend([a, b, c, a, c, d])
  ET.SubElement(
    asset,
    "mesh",
    name="vase_shell_mesh",
    vertex=_numbers(vertices),
    face=_numbers(faces),
    texcoord=_numbers(texcoords),
  )
  ET.SubElement(
    vase,
    "geom",
    name="vase_visual",
    type="mesh",
    mesh="vase_shell_mesh",
    contype="0",
    conaffinity="0",
    mass="0",
    material="vase_glaze",
  )
  for row, z in enumerate(config.STAIN_HEIGHTS):
    radius = config.inner_radius(z) - 0.0002
    for col in range(config.PATCH_COLUMNS):
      angle = (
        config.STAIN_ANGLE_RAD
        + (col - (config.PATCH_COLUMNS - 1) / 2) * config.STAIN_ANGLE_SPACING_RAD
      )
      ET.SubElement(
        vase,
        "geom",
        name=f"stain_{row * config.PATCH_COLUMNS + col}",
        type="ellipsoid",
        pos=_numbers([radius * np.cos(angle), radius * np.sin(angle), z]),
        quat=_numbers([np.cos(angle / 2), 0, 0, np.sin(angle / 2)]),
        size=".0010 .0048 .0055",
        contype="0",
        conaffinity="0",
        mass="0",
        rgba=_numbers(config.STAIN_COLOR),
      )
  camera = root.find(".//camera[@name='vase_inside']")
  position = np.array([0.28, -0.40, 1.12])
  forward = np.array([0.645, -0.18, 0.854]) - position
  forward /= np.linalg.norm(forward)
  right = np.cross(forward, [0, 0, 1])
  right /= np.linalg.norm(right)
  up = np.cross(right, forward)
  camera.set("pos", _numbers(position))
  camera.set("xyaxes", _numbers(np.r_[right, up]))
  # The task overview includes the tabletop pickup and the vase together.
  overview = root.find(".//camera[@name='vase_closeup']")
  position = np.array([0.87, -0.70, 1.20])
  forward = np.array([0.52, -0.28, 0.88]) - position
  forward /= np.linalg.norm(forward)
  right = np.cross(forward, [0, 0, 1])
  right /= np.linalg.norm(right)
  up = np.cross(right, forward)
  overview.set("pos", _numbers(position))
  overview.set("xyaxes", _numbers(np.r_[right, up]))
  overview.set("fovy", "45")
  from .sponge import build

  for child in list(asset):
    if child.get("name", "").startswith("sponge_"):
      asset.remove(child)
  build(root, asset)
  ET.indent(tree, space="  ")
  tree.write(path, encoding="unicode")
  path.write_text(path.read_text() + "\n")
  # A seeded procedural glaze, not a projected photograph or an opaque bore cap.
  rng = np.random.default_rng(7)
  h, w = 512, 512
  noise = rng.normal(0, 1, (h, w))
  from PIL import ImageFilter

  cloudy = (
    np.asarray(
      Image.fromarray(np.uint8(np.clip(128 + noise * 50, 0, 255))).filter(
        ImageFilter.GaussianBlur(3)
      ),
      dtype=float,
    )
    - 128
  )
  y = np.arange(h)[:, None]
  bands = 7 * np.sin(y * 2 * np.pi / 31) + 4 * np.sin(y * 2 * np.pi / 9)
  color = (
    (255 * np.array(config.VASE_IVORY_RGB))[None, None, :]
    + cloudy[..., None] * 2
    + noise[..., None] * 3
    + 0.35 * bands[..., None]
  )
  dark = rng.random((h, w)) < 0.075
  color[dark] *= 0.90
  # Texture vertical coordinates run bottom to top; retain subtle warm ceramic
  # variation while keeping the whole vase in the ivory family.
  foot = int(h * 0.12)
  color[:foot] = np.array([206.0, 195.0, 166.0]) + noise[:foot, :, None] * 4
  color[-7:] = np.array([218.0, 207.0, 179.0]) + noise[-7:, :, None] * 4
  Image.fromarray(np.uint8(np.clip(color, 0, 255))).save(
    path.with_name("vase_glaze.png")
  )


if __name__ == "__main__":
  rebuild()
