"""Rebuild the task-local rounded white plastic and black felt board eraser.

Run ``python -m kaihand_tactile_env.tasks.whiteboard_wipe.geometry`` to regenerate
the inline MJCF meshes. The eraser is 115 x 56 x 30 mm, with a 6 mm felt layer.
Each physical component has one convex collision mesh. The shallow grip lines
are massless visual details and therefore do not add spurious contact points.
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from . import config as C


def _numbers(values):
  return " ".join(f"{value:.9g}" for value in values)


def _rounded_mesh(asset, name, half_length, half_width, radius, profile):
  """Make a watertight, convex rounded prism from (height, inset) rings."""
  vertices, faces = [], []
  corner_samples = 12
  for z, inset in profile:
    x = half_length - inset
    y = half_width - inset
    r = radius - inset
    for cx, cy, start in (
      (x - r, y - r, 0),
      (-x + r, y - r, 90),
      (-x + r, -y + r, 180),
      (x - r, -y + r, 270),
    ):
      for sample in range(corner_samples):
        angle = math.radians(start + sample * 90 / (corner_samples - 1))
        vertices.extend((cx + r * math.cos(angle), cy + r * math.sin(angle), z))
  ring_size = 4 * corner_samples
  for ring in range(len(profile) - 1):
    for i in range(ring_size):
      a = ring * ring_size + i
      b = ring * ring_size + (i + 1) % ring_size
      c = b + ring_size
      d = a + ring_size
      faces.extend((a, b, c, a, c, d))
  bottom_center = len(vertices) // 3
  vertices.extend((0, 0, profile[0][0]))
  top_center = len(vertices) // 3
  vertices.extend((0, 0, profile[-1][0]))
  top_ring = (len(profile) - 1) * ring_size
  for i in range(ring_size):
    following = (i + 1) % ring_size
    faces.extend((bottom_center, following, i))
    faces.extend((top_center, top_ring + i, top_ring + following))
  ET.SubElement(
    asset,
    "mesh",
    name=name,
    vertex=_numbers(vertices),
    face=" ".join(map(str, faces)),
    smoothnormal="true",
  )


def rebuild():
  path = Path(__file__).with_name("scene.xml")
  tree = ET.parse(path)
  root = tree.getroot()
  asset = root.find("asset")
  if asset is None:
    asset = ET.Element("asset")
    root.insert(1, asset)
  for child in list(asset):
    if child.get("name", "").startswith("eraser_"):
      asset.remove(child)
  ET.SubElement(
    asset,
    "material",
    name="eraser_white_plastic",
    rgba=".88 .89 .88 1",
    specular=".32",
    shininess=".25",
    reflectance="0",
  )
  ET.SubElement(
    asset,
    "material",
    name="eraser_grip_recess",
    rgba=".74 .76 .75 1",
    specular=".18",
    shininess=".15",
  )
  ET.SubElement(
    asset,
    "texture",
    name="eraser_felt_texture",
    type="2d",
    builtin="flat",
    width="128",
    height="128",
    rgb1=".025 .028 .03",
    rgb2=".025 .028 .03",
    mark="random",
    markrgb=".060 .065 .068",
    random=".42",
  )
  ET.SubElement(
    asset,
    "material",
    name="eraser_black_felt",
    texture="eraser_felt_texture",
    texuniform="true",
    texrepeat="16 16",
    specular=".02",
    shininess=".02",
  )
  # Quarter-circle edge profiles keep the collision hull convex. The molded
  # shell is gently rounded over the top and wraps down to the felt seam.
  shell_profile = []
  for degrees in (0, 30, 60, 90):
    angle = math.radians(degrees)
    shell_profile.append(
      (-0.007 - 0.002 * math.cos(angle), 0.002 * (1 - math.sin(angle)))
    )
  for degrees in (0, 22.5, 45, 67.5, 90):
    angle = math.radians(degrees)
    shell_profile.append(
      (0.012 + 0.003 * math.sin(angle), 0.003 * (1 - math.cos(angle)))
    )
  _rounded_mesh(asset, "eraser_shell_mesh", 0.0575, 0.028, 0.009, shell_profile)
  pad_profile = []
  for degrees in (0, 30, 60, 90):
    angle = math.radians(degrees)
    pad_profile.append(
      (-0.0142 - 0.0008 * math.cos(angle), 0.0008 * (1 - math.sin(angle)))
    )
  for degrees in (0, 30, 60, 90):
    angle = math.radians(degrees)
    pad_profile.append(
      (-0.0098 + 0.0008 * math.sin(angle), 0.0008 * (1 - math.cos(angle)))
    )
  _rounded_mesh(asset, "eraser_felt_mesh", 0.0555, 0.026, 0.008, pad_profile)
  eraser = root.find(".//body[@name='eraser']")
  if eraser is None:
    raise ValueError("The whiteboard scene has no eraser body")
  eraser.clear()
  yaw = math.atan2(C.INITIAL_ERASER_ROTATION[1, 0], C.INITIAL_ERASER_ROTATION[0, 0])
  eraser.attrib.update(name="eraser", pos=".43 -.40 .6955", euler=_numbers((0, 0, yaw)))
  ET.SubElement(eraser, "freejoint", name="eraser_freejoint")
  ET.SubElement(
    eraser,
    "geom",
    name="eraser_handle",
    type="mesh",
    mesh="eraser_shell_mesh",
    mass=".062",
    material="eraser_white_plastic",
    contype="3075",
    conaffinity="3075",
    friction="1.2 .003 .0003",
    condim="6",
    solref=".006 1",
    margin="0",
  )
  ET.SubElement(
    eraser,
    "geom",
    name="eraser_pad",
    type="mesh",
    mesh="eraser_felt_mesh",
    mass=".018",
    material="eraser_black_felt",
    contype="3075",
    conaffinity="3075",
    friction=".65 .003 .0003",
    condim="6",
    solref=".02 1",
    margin="0",
  )
  # Fine parallel grooves, aligned with the long axis, like molded desk erasers.
  # They project only 0.12 mm above the flat top, with no physical ridges.
  for i in range(-7, 8):
    y = i * 0.003
    ET.SubElement(
      eraser,
      "geom",
      name=f"eraser_grip_line_{i + 7}",
      type="capsule",
      fromto=_numbers((-0.047, y, 0.01497, 0.047, y, 0.01497)),
      size=".00015",
      material="eraser_grip_recess",
      mass="0",
      contype="0",
      conaffinity="0",
    )
  # A small smooth rectangular panel interrupts the grip pattern, without a
  # brand or a raised handle. It lies just over the decorative grooves.
  _rounded_mesh(
    asset,
    "eraser_label_mesh",
    0.014,
    0.0075,
    0.002,
    ((0.01510, 0.0001), (0.01518, 0)),
  )
  ET.SubElement(
    eraser,
    "geom",
    name="eraser_label_panel",
    type="mesh",
    mesh="eraser_label_mesh",
    material="eraser_white_plastic",
    mass="0",
    contype="0",
    conaffinity="0",
  )
  contact = root.find("contact")
  if contact is None:
    contact = ET.SubElement(root, "contact")
  for child in list(contact):
    if child.get("name", "") == "eraser_board_contact":
      contact.remove(child)
  ET.SubElement(
    contact,
    "pair",
    name="eraser_board_contact",
    geom1="eraser_pad",
    geom2="board_surface",
    condim="6",
    friction=".65 .65 .003 .0003 .0003",
    solref=".09 1",
    solimp="0 .95 .001 .5 2",
    margin="0",
  )
  ET.indent(tree, space="  ")
  tree.write(path, encoding="unicode")
  path.write_text(path.read_text() + "\n")


if __name__ == "__main__":
  rebuild()
