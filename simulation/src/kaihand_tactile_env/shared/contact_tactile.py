"""Distribute real MuJoCo fingertip contact forces over tactile taxel grids.

This module keeps two distinct quantities in every sample:

* per-fingertip forces are the actual constraint forces returned by
  :func:`mujoco.mj_contactForce`;
* per-taxel values are a conservative spatial allocation of those forces via
  a normalized distance kernel.  They are expressed in newtons allocated to
  each taxel, not pressure and not a simulated flexible-skin measurement.

Only contacts between an exact ``*_tactile_pad_col`` geometry and an enabled
task object (card or cylinder) are admitted.  Palm, table, arm and self
contacts can therefore never appear in these tactile maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from .config import FINGERTIP_LINK_NAMES, OBJECT_NAMES
from .tactile import load_fingertip_layout

GRID_SHAPE = (7, 5)
SOURCE = "solver_contact_distributed_taxel_v1"


@dataclass(frozen=True)
class ContactTaxelAssignment:
  """Diagnostic mapping from one solver contact to one fingertip grid."""

  contact_id: int
  link_index: int
  link_name: str
  pad_geom_id: int
  target_geom_id: int
  position_world_m: np.ndarray
  position_link_local_m: np.ndarray
  force_world_n: np.ndarray
  normal_force_n: float
  normal_force_world_n: np.ndarray
  tangent_force_world_n: np.ndarray
  tangent_force_n: np.ndarray
  tangent_load_n: float
  nearest_taxel_flat_index: int
  nearest_taxel_row_col: tuple[int, int]
  nearest_taxel_distance_m: float
  kernel_weights: np.ndarray

  def __post_init__(self) -> None:
    expected_shapes = {
      "position_world_m": (3,),
      "position_link_local_m": (3,),
      "force_world_n": (3,),
      "normal_force_world_n": (3,),
      "tangent_force_world_n": (3,),
      "tangent_force_n": (2,),
      "kernel_weights": GRID_SHAPE,
    }
    for name, expected in expected_shapes.items():
      value = np.asarray(getattr(self, name))
      if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
      if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.isfinite((self.normal_force_n, self.tangent_load_n)).all():
      raise ValueError("contact force magnitudes must be finite")
    if self.normal_force_n < 0.0 or self.tangent_load_n < 0.0:
      raise ValueError("contact force magnitudes must be nonnegative")
    if self.nearest_taxel_distance_m < 0.0:
      raise ValueError("nearest taxel distance must be nonnegative")
    if not 0 <= self.nearest_taxel_flat_index < GRID_SHAPE[0] * GRID_SHAPE[1]:
      raise ValueError("nearest taxel index is outside the 7x5 grid")
    if self.nearest_taxel_row_col != divmod(
      self.nearest_taxel_flat_index, GRID_SHAPE[1]
    ):
      raise ValueError("nearest taxel row/column disagrees with its flat index")
    if np.any(self.kernel_weights < 0.0):
      raise ValueError("contact kernel weights must be nonnegative")
    if not np.isclose(float(np.sum(self.kernel_weights)), 1.0, atol=1.0e-12):
      raise ValueError("contact kernel weights must sum to one")


@dataclass(frozen=True)
class SolverDistributedTactileSample:
  """One force-conserving bilateral fingertip tactile sample.

  ``tangent_force_n[..., 0:2]`` uses the stable pad basis stored in
  ``tangent_basis_local/world``.  Basis axis 0 follows increasing taxel grid
  columns and axis 1 follows increasing grid rows.  ``tangent_load_n`` is the
  non-cancelling sum of per-contact tangential magnitudes.
  """

  timestamp: float
  source: str
  link_names: tuple[str, ...]
  contact_count: np.ndarray
  normal_force_n: np.ndarray
  normal_force_world_n: np.ndarray
  force_world_n: np.ndarray
  tangent_force_world_n: np.ndarray
  tangent_force_n: np.ndarray
  tangent_load_n: np.ndarray
  normal_taxel_force_n: np.ndarray
  tangent_taxel_force_n: np.ndarray
  tangent_taxel_load_n: np.ndarray
  normal_axis_local: np.ndarray
  normal_axis_world: np.ndarray
  tangent_basis_local: np.ndarray
  tangent_basis_world: np.ndarray
  assignments: tuple[ContactTaxelAssignment, ...]

  def __post_init__(self) -> None:
    if not np.isfinite(self.timestamp):
      raise ValueError("timestamp must be finite")
    link_count = len(self.link_names)
    expected_shapes = {
      "contact_count": (link_count,),
      "normal_force_n": (link_count,),
      "normal_force_world_n": (link_count, 3),
      "force_world_n": (link_count, 3),
      "tangent_force_world_n": (link_count, 3),
      "tangent_force_n": (link_count, 2),
      "tangent_load_n": (link_count,),
      "normal_taxel_force_n": (link_count, *GRID_SHAPE),
      "tangent_taxel_force_n": (link_count, *GRID_SHAPE, 2),
      "tangent_taxel_load_n": (link_count, *GRID_SHAPE),
      "normal_axis_local": (link_count, 3),
      "normal_axis_world": (link_count, 3),
      "tangent_basis_local": (link_count, 2, 3),
      "tangent_basis_world": (link_count, 2, 3),
    }
    for name, expected in expected_shapes.items():
      value = np.asarray(getattr(self, name))
      if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
      if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(self.normal_force_n < 0.0) or np.any(self.tangent_load_n < 0.0):
      raise ValueError("force magnitudes must be nonnegative")
    if np.any(self.normal_taxel_force_n < 0.0) or np.any(
      self.tangent_taxel_load_n < 0.0
    ):
      raise ValueError("taxel force magnitudes must be nonnegative")
    if not np.allclose(
      self.normal_taxel_force_n.sum(axis=(1, 2)),
      self.normal_force_n,
      rtol=1.0e-12,
      atol=1.0e-12,
    ):
      raise ValueError("normal taxel allocation does not conserve force")
    if not np.allclose(
      self.tangent_taxel_force_n.sum(axis=(1, 2)),
      self.tangent_force_n,
      rtol=1.0e-12,
      atol=1.0e-12,
    ):
      raise ValueError("tangent taxel allocation does not conserve net force")
    if not np.allclose(
      self.tangent_taxel_load_n.sum(axis=(1, 2)),
      self.tangent_load_n,
      rtol=1.0e-12,
      atol=1.0e-12,
    ):
      raise ValueError("tangent taxel allocation does not conserve load")
    if not np.allclose(
      self.normal_force_world_n + self.tangent_force_world_n,
      self.force_world_n,
      rtol=1.0e-12,
      atol=1.0e-12,
    ):
      raise ValueError("normal and tangent components do not recover total force")


class SolverDistributedTactileProvider:
  """Map exact tactile-pad solver contacts onto the accepted 7x5 layouts."""

  source = SOURCE
  force_unit = "N"
  taxel_force_unit = "N"
  pressure_unit = None
  grid_shape = GRID_SHAPE
  taxel_force_semantics = (
    "normalized Gaussian allocation of real MuJoCo solver contact force; "
    "N allocated per taxel, not Pa and not flexible-skin ground truth"
  )
  tangent_basis_semantics = (
    "axis 0 follows increasing grid columns; axis 1 follows increasing grid rows"
  )

  def __init__(
    self,
    model: mujoco.MjModel,
    layout: Any | None = None,
    *,
    target_geom_names: tuple[str, ...] | None = None,
    link_names: tuple[str, ...] = FINGERTIP_LINK_NAMES,
    kernel_sigma_m: float = 0.003,
  ) -> None:
    if not np.isfinite(kernel_sigma_m) or kernel_sigma_m <= 0.0:
      raise ValueError("kernel_sigma_m must be finite and positive")
    if not link_names:
      raise ValueError("link_names cannot be empty")
    if len(set(link_names)) != len(link_names):
      raise ValueError("link_names must be unique")
    unknown_links = sorted(set(link_names) - set(FINGERTIP_LINK_NAMES))
    if unknown_links:
      raise ValueError(f"unsupported fingertip links: {unknown_links}")

    self.model = model
    self.layout = layout or load_fingertip_layout()
    self.link_names = tuple(link_names)
    self.kernel_sigma_m = float(kernel_sigma_m)
    self.target_geom_names = _resolve_target_geom_names(model, target_geom_names)
    self._target_geom_ids = frozenset(
      _require_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
      for name in self.target_geom_names
    )
    self.pad_geom_names = tuple(f"{name}_tactile_pad_col" for name in link_names)
    self._pad_geom_ids = np.asarray(
      [
        _require_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in self.pad_geom_names
      ],
      dtype=np.int32,
    )
    self._body_ids = np.asarray(
      [_require_id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in link_names],
      dtype=np.int32,
    )
    for link_name, body_id, geom_id in zip(
      self.link_names, self._body_ids, self._pad_geom_ids, strict=True
    ):
      if int(model.geom_bodyid[geom_id]) != int(body_id):
        raise RuntimeError(
          f"{link_name!r} does not own its exact tactile-pad collision geometry"
        )
    self._pad_geom_to_link = {
      int(geom_id): index for index, geom_id in enumerate(self._pad_geom_ids)
    }

    layout_body_names = np.asarray(self.layout.body_names)
    taxel_positions: list[np.ndarray] = []
    normal_axes: list[np.ndarray] = []
    tangent_bases: list[np.ndarray] = []
    self._layout_indices: list[np.ndarray] = []
    for link_name in self.link_names:
      indices = np.flatnonzero(layout_body_names == link_name)
      if len(indices) != GRID_SHAPE[0] * GRID_SHAPE[1]:
        raise RuntimeError(
          f"{link_name}: expected 35 tactile probes, got {len(indices)}"
        )
      if any(self.layout.grid_shape[int(index)] != GRID_SHAPE for index in indices):
        raise RuntimeError(f"{link_name}: expected one 7x5 tactile grid")
      positions = np.asarray(self.layout.local_pos[indices], dtype=np.float64)
      normals = np.asarray(self.layout.local_normal[indices], dtype=np.float64)
      normal_axis, tangent_basis = _stable_pad_basis(positions, normals)
      self._layout_indices.append(indices)
      taxel_positions.append(positions)
      normal_axes.append(normal_axis)
      tangent_bases.append(tangent_basis)
    self.taxel_positions_local_m = np.stack(taxel_positions)
    self.normal_axis_local = np.stack(normal_axes)
    self.tangent_basis_local = np.stack(tangent_bases)
    self.grid_basis_handedness = np.sign(
      np.einsum(
        "li,li->l",
        np.cross(
          self.tangent_basis_local[:, 0], self.tangent_basis_local[:, 1]
        ),
        self.normal_axis_local,
      )
    ).astype(np.int8)
    self._wrench = np.zeros(6, dtype=np.float64)

  @property
  def available(self) -> bool:
    return True

  def metadata(self) -> dict[str, Any]:
    """Return JSON-safe units and approximation semantics for recording."""
    return {
      "source": self.source,
      "algorithm": "normalized_gaussian_3d_taxel_distance",
      "force_unit": self.force_unit,
      "taxel_force_unit": self.taxel_force_unit,
      "pressure_unit": self.pressure_unit,
      "is_spatial_estimate": True,
      "grid_shape": list(self.grid_shape),
      "kernel": "normalized_gaussian_3d_taxel_distance",
      "kernel_sigma_m": self.kernel_sigma_m,
      "force_action": (
        "force acting on the tactile pad; mj_contactForce geom2 convention "
        "is sign-corrected when the pad is geom1"
      ),
      "taxel_force_semantics": self.taxel_force_semantics,
      "tangent_basis_semantics": self.tangent_basis_semantics,
      "tangent_basis": ["grid_col_positive", "grid_row_positive"],
      "conservation": (
        "per-link taxel normal sum equals solver Fn; signed tangent taxel "
        "vector sum equals net solver tangent projected into the stable pad basis"
      ),
      "grid_basis_handedness": self.grid_basis_handedness.tolist(),
      "basis_warning": (
        "grid row/column directions preserve chart indexing across mirrored hands; "
        "normal plus tangent axes must not be assumed to form an SO(3) rotation"
      ),
      "target_geom_names": list(self.target_geom_names),
      "pad_geom_names": list(self.pad_geom_names),
    }

  def read(self, data: mujoco.MjData) -> SolverDistributedTactileSample:
    """Read current constraint forces and return exact aggregates plus maps."""
    link_count = len(self.link_names)
    contact_count = np.zeros(link_count, dtype=np.int32)
    normal_force_n = np.zeros(link_count, dtype=np.float64)
    normal_force_world_n = np.zeros((link_count, 3), dtype=np.float64)
    force_world_n = np.zeros((link_count, 3), dtype=np.float64)
    tangent_force_world_n = np.zeros((link_count, 3), dtype=np.float64)
    tangent_force_n = np.zeros((link_count, 2), dtype=np.float64)
    tangent_load_n = np.zeros(link_count, dtype=np.float64)
    normal_taxel_force_n = np.zeros((link_count, *GRID_SHAPE), dtype=np.float64)
    tangent_taxel_force_n = np.zeros(
      (link_count, *GRID_SHAPE, 2), dtype=np.float64
    )
    tangent_taxel_load_n = np.zeros(
      (link_count, *GRID_SHAPE), dtype=np.float64
    )

    rotations = np.stack(
      [np.asarray(data.xmat[body_id]).reshape(3, 3) for body_id in self._body_ids]
    )
    normal_axis_world = np.einsum(
      "lij,lj->li", rotations, self.normal_axis_local
    )
    tangent_basis_world = np.einsum(
      "lij,lkj->lki", rotations, self.tangent_basis_local
    )
    assignments: list[ContactTaxelAssignment] = []

    for contact_id in range(int(data.ncon)):
      contact = data.contact[contact_id]
      selected = self._select_contact(int(contact.geom1), int(contact.geom2))
      if selected is None:
        continue
      link_index, pad_geom_id, target_geom_id, sign = selected
      self._wrench.fill(0.0)
      mujoco.mj_contactForce(self.model, data, contact_id, self._wrench)
      frame_world = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
      contact_force_world = sign * (frame_world.T @ self._wrench[:3])
      contact_normal_world = sign * (
        frame_world.T @ np.asarray([self._wrench[0], 0.0, 0.0])
      )
      contact_tangent_world = sign * (
        frame_world.T @ np.asarray([0.0, self._wrench[1], self._wrench[2]])
      )
      contact_tangent_pad = (
        tangent_basis_world[link_index] @ contact_tangent_world
      )
      normal_magnitude = abs(float(self._wrench[0]))
      tangent_magnitude = float(np.linalg.norm(self._wrench[1:3]))

      rotation = rotations[link_index]
      position_world = np.asarray(contact.pos, dtype=np.float64).copy()
      position_local = rotation.T @ (
        position_world - np.asarray(data.xpos[self._body_ids[link_index]])
      )
      weights, nearest_index, nearest_distance = self._kernel_weights(
        link_index, position_local
      )

      contact_count[link_index] += 1
      normal_force_n[link_index] += normal_magnitude
      normal_force_world_n[link_index] += contact_normal_world
      force_world_n[link_index] += contact_force_world
      tangent_force_world_n[link_index] += contact_tangent_world
      tangent_force_n[link_index] += contact_tangent_pad
      tangent_load_n[link_index] += tangent_magnitude
      normal_taxel_force_n[link_index] += weights * normal_magnitude
      tangent_taxel_force_n[link_index] += (
        weights[..., None] * contact_tangent_pad
      )
      tangent_taxel_load_n[link_index] += weights * tangent_magnitude
      assignments.append(
        ContactTaxelAssignment(
          contact_id=contact_id,
          link_index=link_index,
          link_name=self.link_names[link_index],
          pad_geom_id=pad_geom_id,
          target_geom_id=target_geom_id,
          position_world_m=position_world,
          position_link_local_m=position_local,
          force_world_n=contact_force_world,
          normal_force_n=normal_magnitude,
          normal_force_world_n=contact_normal_world,
          tangent_force_world_n=contact_tangent_world,
          tangent_force_n=contact_tangent_pad,
          tangent_load_n=tangent_magnitude,
          nearest_taxel_flat_index=nearest_index,
          nearest_taxel_row_col=divmod(nearest_index, GRID_SHAPE[1]),
          nearest_taxel_distance_m=nearest_distance,
          kernel_weights=weights.copy(),
        )
      )

    return SolverDistributedTactileSample(
      timestamp=float(data.time),
      source=self.source,
      link_names=self.link_names,
      contact_count=contact_count,
      normal_force_n=normal_force_n,
      normal_force_world_n=normal_force_world_n,
      force_world_n=force_world_n,
      tangent_force_world_n=tangent_force_world_n,
      tangent_force_n=tangent_force_n,
      tangent_load_n=tangent_load_n,
      normal_taxel_force_n=normal_taxel_force_n,
      tangent_taxel_force_n=tangent_taxel_force_n,
      tangent_taxel_load_n=tangent_taxel_load_n,
      normal_axis_local=self.normal_axis_local.copy(),
      normal_axis_world=normal_axis_world,
      tangent_basis_local=self.tangent_basis_local.copy(),
      tangent_basis_world=tangent_basis_world,
      assignments=tuple(assignments),
    )

  def _select_contact(
    self, geom1: int, geom2: int
  ) -> tuple[int, int, int, float] | None:
    link_index = self._pad_geom_to_link.get(geom1)
    if link_index is not None and geom2 in self._target_geom_ids:
      # mj_contactForce follows the repository convention of acting on geom2.
      return link_index, geom1, geom2, -1.0
    link_index = self._pad_geom_to_link.get(geom2)
    if link_index is not None and geom1 in self._target_geom_ids:
      return link_index, geom2, geom1, 1.0
    return None

  def _kernel_weights(
    self, link_index: int, position_local: np.ndarray
  ) -> tuple[np.ndarray, int, float]:
    delta = self.taxel_positions_local_m[link_index] - position_local
    squared_distance = np.einsum("ij,ij->i", delta, delta)
    nearest_index = int(np.argmin(squared_distance))
    nearest_squared = float(squared_distance[nearest_index])
    # Subtracting the nearest exponent is algebraically neutral after
    # normalization and prevents underflow for an outlying diagnostic point.
    weights = np.exp(
      -0.5 * (squared_distance - nearest_squared) / self.kernel_sigma_m**2
    )
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
      raise RuntimeError("could not normalize tactile contact kernel")
    weights = (weights / weight_sum).reshape(GRID_SHAPE)
    return weights, nearest_index, float(np.sqrt(nearest_squared))


def _stable_pad_basis(
  positions: np.ndarray, normals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  grid = np.asarray(positions, dtype=np.float64).reshape(*GRID_SHAPE, 3)
  mean_normal = np.mean(np.asarray(normals, dtype=np.float64), axis=0)
  normal_axis = _normalize(mean_normal, "mean tactile normal")
  column_direction = np.mean(grid[:, -1] - grid[:, 0], axis=0)
  row_direction = np.mean(grid[-1] - grid[0], axis=0)
  tangent_column = column_direction - normal_axis * np.dot(
    normal_axis, column_direction
  )
  tangent_column = _normalize(tangent_column, "tactile grid column direction")
  tangent_row = row_direction - normal_axis * np.dot(normal_axis, row_direction)
  tangent_row -= tangent_column * np.dot(tangent_column, tangent_row)
  if np.linalg.norm(tangent_row) < 1.0e-10:
    tangent_row = np.cross(normal_axis, tangent_column)
  tangent_row = _normalize(tangent_row, "tactile grid row direction")
  if np.dot(tangent_row, row_direction) < 0.0:
    tangent_row *= -1.0
  return normal_axis, np.stack((tangent_column, tangent_row))


def _normalize(vector: np.ndarray, description: str) -> np.ndarray:
  vector = np.asarray(vector, dtype=np.float64)
  norm = float(np.linalg.norm(vector))
  if not np.isfinite(norm) or norm < 1.0e-10:
    raise RuntimeError(f"cannot construct {description}")
  return vector / norm


def _resolve_target_geom_names(
  model: mujoco.MjModel, requested: tuple[str, ...] | None
) -> tuple[str, ...]:
  enabled_by_object: dict[str, tuple[str, ...]] = {}
  for object_name in OBJECT_NAMES:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    if body_id < 0:
      continue
    names = tuple(
      model.geom(geom_id).name
      for geom_id in range(model.ngeom)
      if int(model.geom_bodyid[geom_id]) == body_id
      and (model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0)
    )
    if names:
      enabled_by_object[object_name] = names
  if not enabled_by_object:
    raise RuntimeError("model has no enabled card or cylinder collision geometry")

  allowed = {
    name: object_name
    for object_name, names in enabled_by_object.items()
    for name in names
  }
  if requested is None:
    if len(enabled_by_object) != 1:
      raise RuntimeError(
        "multiple task objects have enabled collision geometry; explicitly select "
        "the current card or cylinder target geoms"
      )
    return next(iter(enabled_by_object.values()))
  if not requested or len(set(requested)) != len(requested):
    raise ValueError("target_geom_names must be non-empty and unique")
  invalid = [name for name in requested if name not in allowed]
  if invalid:
    raise ValueError(
      "target geoms must belong to an enabled card or cylinder: " f"{invalid}"
    )
  objects = {allowed[name] for name in requested}
  if len(objects) != 1:
    raise ValueError("target geoms must belong to exactly one active task object")
  return tuple(requested)


def _require_id(
  model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str
) -> int:
  object_id = mujoco.mj_name2id(model, object_type, name)
  if object_id < 0:
    raise RuntimeError(f"MuJoCo model is missing {name!r}")
  return object_id


__all__ = [
  "ContactTaxelAssignment",
  "GRID_SHAPE",
  "SOURCE",
  "SolverDistributedTactileProvider",
  "SolverDistributedTactileSample",
]
