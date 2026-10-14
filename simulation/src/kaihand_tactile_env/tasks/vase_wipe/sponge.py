"""Volumetric elastic sponge and an opt-in shared-taxel flex contact adapter."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from kaihand_tactile_env.shared.contact_tactile import SolverDistributedTactileProvider

SHAPE = (3, 3, 8)
PIN = (1 * SHAPE[1] + 1) * SHAPE[2] + 6
# The leading corner of the flat face contacts the concave wall obliquely.
WORKING_NODES = ((2 * SHAPE[1]) * SHAPE[2],)
# At reset the material axes match the world axes: +X faces away from the robot.
SCOURING_PANEL = 1  # axis * 2 + side: X, positive side


def points():
  """Rest mesh: one exact 45 × 62 × 110 mm rectangular cuboid.

  The +X cleaning face is 62 × 110 mm while the side thickness is 45 mm.
  """
  return np.array(
    [
      (x, y, z)
      for x in np.linspace(-0.0245, 0.0205, SHAPE[0])
      for y in np.linspace(-0.031, 0.031, SHAPE[1])
      for z in (-0.080, -0.064, -0.048, -0.032, -0.016, 0.0, 0.015, 0.030)
    ]
  )


def _numbers(values):
  return " ".join(f"{x:.9g}" for x in np.asarray(values).ravel())


def build(root, asset):
  from . import config

  body = root.find(".//body[@name='sponge']")
  body.clear()
  xyz = points()
  # Match the independent tabletop reset, including the flex contact radius.
  table = root.find(".//body[@name='table']")
  if table is None:
    shared = ET.parse(Path(__file__).parents[2] / "shared/mjcf/robot.xml").getroot()
    table = shared.find(".//body[@name='table']")
  tabletop = table.find("geom[@name='tabletop']")
  table_height = float(table.get("pos").split()[2]) + float(
    tabletop.get("size").split()[2]
  )
  origin = np.r_[config.SPONGE_TABLE_XY, table_height + 0.001 - xyz[:, 2].min()]
  body.attrib.update(name="sponge", pos=_numbers(origin + xyz[PIN]))
  ET.SubElement(body, "freejoint", name="sponge_freejoint")
  world = root.find("worldbody")
  for child in list(world):
    if child.get("name", "").startswith("sponge_node_"):
      world.remove(child)
  # A free material reference point, never attached to the hand or world.
  # Its tiny sphere is one material node, not a rigid load-bearing core.
  ET.SubElement(
    body,
    "geom",
    name="sponge_reference",
    type="sphere",
    size=".0005",
    pos="0 0 0",
    mass=".001",
    rgba="0 0 0 0",
    contype="15363",
    conaffinity="15363",
    group="3",
  )
  names, vertices = [], []
  for i, point in enumerate(xyz):
    if i == PIN:
      names.append("sponge")
      vertices.append([0, 0, 0])
      continue
    name = f"sponge_node_{i}"
    # Independent translational material particles avoid the redundant freely
    # rotating parent frame of a free joint plus translational child nodes.
    node = ET.SubElement(world, "body", name=name, pos=_numbers(origin + point))
    ET.SubElement(
      node,
      "inertial",
      pos="0 0 0",
      mass=str(0.042 / (len(xyz) - 1)),
      diaginertia="1e-8 1e-8 1e-8",
    )
    for j in range(3):
      ET.SubElement(
        node,
        "joint",
        name=f"{name}_{j}",
        type="slide",
        axis=_numbers(np.eye(3)[j]),
        limited="false",
        damping=".005",
      )
    names.append(name)
    vertices.append([0, 0, 0])

  def index(i, j, k):
    return (i * SHAPE[1] + j) * SHAPE[2] + k

  elements = []
  for i in range(SHAPE[0] - 1):
    for j in range(SHAPE[1] - 1):
      for k in range(SHAPE[2] - 1):
        a, b, c, d, e, f, g, h = [
          index(i + dx, j + dy, k + dz)
          for dx, dy, dz in [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
          ]
        ]
        for tet in [
          [a, b, d, h],
          [a, d, c, h],
          [a, c, g, h],
          [a, g, e, h],
          [a, e, f, h],
          [a, f, b, h],
        ]:
          if np.linalg.det((xyz[tet[1:]] - xyz[tet[0]]).T) < 0:
            tet[1], tet[2] = tet[2], tet[1]
          elements.extend(tet)
  deform = root.find("deformable")
  if deform is None:
    deform = ET.SubElement(root, "deformable")
  for child in list(deform):
    if child.get("name", "").startswith("sponge"):
      deform.remove(child)
  flex = ET.SubElement(
    deform,
    "flex",
    name="sponge_volume",
    dim="3",
    radius=".001",
    body=" ".join(names),
    vertex=_numbers(vertices),
    element=_numbers(elements),
    rgba="0 0 0 0",
    group="3",
  )
  ET.SubElement(
    flex,
    "contact",
    contype="15363",
    conaffinity="15363",
    condim="3",
    friction="1.1 .003 .0001",
    solref=".005 1",
    solimp=".95 .99 .0005",
    selfcollide="none",
    internal="true",
  )
  ET.SubElement(
    flex, "elasticity", young=str(config.SPONGE_YOUNG_PA), poisson=".12", damping="0"
  )
  ET.SubElement(flex, "edge", damping=".0001")
  # Six UV panels and node-bound bones form one continuous visual surface.
  sv, uv, faces, node_indices = [], [], [], []
  for axis in range(3):
    others = [x for x in range(3) if x != axis]
    for side in range(2):
      panel = axis * 2 + side
      base = len(sv)
      for u in range(SHAPE[others[0]]):
        for v in range(SHAPE[others[1]]):
          ijk = [0, 0, 0]
          ijk[axis] = side * (SHAPE[axis] - 1)
          ijk[others[0]] = u
          ijk[others[1]] = v
          idx = index(*ijk)
          sv.append(origin + xyz[idx])
          node_indices.append(idx)
          uv.append(
            [
              ((panel % 3) + 0.01 + 0.98 * u / (SHAPE[others[0]] - 1)) / 3,
              ((panel // 3) + 0.01 + 0.98 * v / (SHAPE[others[1]] - 1)) / 2,
            ]
          )
      stride = SHAPE[others[1]]
      for u in range(SHAPE[others[0]] - 1):
        for v in range(stride - 1):
          a = base + u * stride + v
          b = a + stride
          c = b + 1
          d = a + 1
          for tri in [[a, b, c], [a, c, d]]:
            normal = np.cross(
              np.array(sv[tri[1]]) - sv[tri[0]], np.array(sv[tri[2]]) - sv[tri[0]]
            )
            if normal[axis] * (2 * side - 1) < 0:
              tri[1], tri[2] = tri[2], tri[1]
            faces.extend(tri)
  ET.SubElement(
    asset, "texture", name="sponge_surface", type="2d", file="sponge_surface.png"
  )
  ET.SubElement(
    asset,
    "material",
    name="sponge_surface",
    texture="sponge_surface",
    # Keep the five yellow sides recognisable under the workcell's directional
    # lighting; this only changes shading, never contact or elastic parameters.
    emission=".3",
    specular=".05",
    shininess=".05",
  )
  skin = ET.SubElement(
    deform,
    "skin",
    name="sponge_skin",
    material="sponge_surface",
    vertex=_numbers(sv),
    texcoord=_numbers(uv),
    face=_numbers(faces),
    inflate=".001",
    group="0",
  )
  for i, name in enumerate(names):
    vids = [j for j, n in enumerate(node_indices) if n == i]
    if vids:
      ET.SubElement(
        skin,
        "bone",
        body=name,
        bindpos=_numbers(origin + xyz[i]),
        bindquat="1 0 0 0",
        vertid=_numbers(vids),
        vertweight=_numbers(np.ones(len(vids))),
      )
  _texture()


def _texture():
  rng = np.random.default_rng(12)
  size = 256
  image = Image.new("RGB", (size * 3, size * 2))
  for panel in range(6):
    noise = rng.normal(0, 8, (size, size, 1))
    yellow = np.clip(np.array([255, 210, 32]) + noise, 0, 255).astype("uint8")
    green = np.clip(np.array([20, 55, 32]) + noise, 0, 255).astype("uint8")
    # Only the initial far (+X) face is dark green, with no wrap onto its edges.
    tile = Image.fromarray(green if panel == SCOURING_PANEL else yellow)
    draw = ImageDraw.Draw(tile)
    for _ in range(650):
      x, y = rng.integers(0, size, 2)
      r = int(rng.integers(1, 4))
      if panel == SCOURING_PANEL:
        draw.line(
          (int(x), int(y), int(x + r * 3), int(y + r)), fill=(12, 39, 21), width=1
        )
      else:
        draw.ellipse((int(x), int(y), int(x + r), int(y + r)), fill=(206, 155, 14))
    image.paste(tile, ((panel % 3) * size, (panel // 3) * size))
  image.save(Path(__file__).with_name("sponge_surface.png"))


class SpongeTactileProvider(SolverDistributedTactileProvider):
  """Use shared force-conserving 7x5 pad maps for native flex contacts."""

  def __init__(self, model):
    super().__init__(model)
    if (
      model.nflex != 1
      or mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, "sponge_volume") != 0
    ):
      raise ValueError("sponge adapter requires exactly the sponge flex")
    self.source = "solver_contact_distributed_taxel_flex_v1"

  def _select_contact(self, geom1, geom2):
    # This isolated model has one flex; -1 identifies its non-geom side.
    if geom1 == -1 and geom2 in self._pad_geom_to_link:
      return self._pad_geom_to_link[geom2], geom2, -1, 1.0
    if geom2 == -1 and geom1 in self._pad_geom_to_link:
      return self._pad_geom_to_link[geom1], geom1, -1, -1.0
    return super()._select_contact(geom1, geom2)


class GraspPatchFriction:
  """Compliant rolling resistance at the loaded grasp, using force couples.

  A flex particle has no material rotation DOF. Distributed couples represent
  the finite contact patch instead of locking the entire sponge to a frame.
  There is no net supporting force, and resistance vanishes without contact.
  """

  def __init__(self, model, hand_geom_ids):
    self.model = model
    self.hand_geom_ids = hand_geom_ids
    self.ids = np.flatnonzero(points()[:, 2] >= -0.005)
    self.rest = points()[self.ids]
    self.rest -= self.rest.mean(axis=0)
    self.hand = model.body("hand_r_base_link").id
    self.dofs = np.array(
      [
        [
          int(model.joint("sponge_freejoint").dofadr[0]) + j
          if i == PIN
          else int(model.joint(f"sponge_node_{i}_{j}").dofadr[0])
          for j in range(3)
        ]
        for i in self.ids
      ]
    )
    self.reference = None
    self.enabled = False
    self.torque_world_n_m = np.zeros(3)
    self.normal_load_n = 0.0

  def apply(self, data):
    from . import config

    self.torque_world_n_m[:] = 0
    if not self.enabled:
      return
    wrench = np.zeros(6)
    self.normal_load_n = 0.0
    for i, contact in enumerate(data.contact):
      if (contact.flex[0] >= 0 and int(contact.geom2) in self.hand_geom_ids) or (
        contact.flex[1] >= 0 and int(contact.geom1) in self.hand_geom_ids
      ):
        mujoco.mj_contactForce(self.model, data, i, wrench)
        self.normal_load_n += max(0.0, float(wrench[0]))
    if self.normal_load_n < 0.4:
      self.reference = None
      return
    p = data.flexvert_xpos[self.ids]
    r = p - p.mean(axis=0)
    u, _, vt = np.linalg.svd(self.rest.T @ r)
    if np.linalg.det(u @ vt) < 0:
      u[:, -1] *= -1
    rotation = (u @ vt).T
    hand_rotation = data.xmat[self.hand].reshape(3, 3)
    if self.reference is None:
      self.reference = hand_rotation.T @ rotation
    desired = hand_rotation @ self.reference
    error = 0.5 * np.cross(rotation.T, desired.T).sum(axis=0)
    moment = np.eye(3) * (r * r).sum() - r.T @ r
    velocities = data.qvel[self.dofs]
    omega = np.linalg.solve(
      moment, np.cross(r, velocities - velocities.mean(axis=0)).sum(axis=0)
    )
    torque = 0.8 * error - 0.010 * (omega - data.cvel[self.hand, :3])
    limit = min(0.15, config.GRASP_PATCH_RADIUS_M * self.normal_load_n)
    torque *= min(1.0, limit / max(float(np.linalg.norm(torque)), 1e-12))
    forces = np.cross(np.linalg.solve(moment, torque), r)
    data.qfrc_applied[self.dofs] += forces
    # Equal and opposite moment on the actual hand: no net force or torque is
    # added to the hand+sponge system. Positions remain entirely solver-driven.
    mujoco.mj_applyFT(
      self.model,
      data,
      np.zeros(3),
      -torque,
      data.xpos[self.hand],
      self.hand,
      data.qfrc_applied,
    )
    self.torque_world_n_m[:] = torque
