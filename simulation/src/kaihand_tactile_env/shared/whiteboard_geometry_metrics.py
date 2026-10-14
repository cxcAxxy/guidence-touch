"""Read-only, finite-area whiteboard measurements for staged evaluation.

The felt is a rigid convex collision mesh, not a deformable pressure sensor.
``effective_pad_contact_area_ratio`` is therefore a *geometric estimate*: a
loaded pad/board solver contact, opposed surface normals, a small surface gap,
and overlap with the finite board are all required. Neither contact counts nor
forces are converted into square metres. The denominator is the true flat
bottom polygon of the compiled felt mesh (including its rounded corners).

Coverage uses a fixed raster of the *initial ink footprint union*, not the
already-clean whiteboard. Revisiting a cell never increases its contribution.
Geometry is sampled at 30 Hz by default; actual contact duration is integrated
at every observer update. Swept motion between geometry samples is deliberately
not interpolated: coverage is sampled coverage and can miss between-sample
contacts. Cleaning remains an independent dissipated-work task measurement.

No MuJoCo forward/step call, state assignment, or task implementation import is
performed here. Instantiate after reset/randomization and call ``update`` only
after the evaluation runner has accepted a physics step. Real whiteboard tasks
cache the integrated pad/board load in ``board_force`` before their final
``mj_forward``. That cached load gates interval-end geometry; refreshed solver
forces must not replace the load that actually participated in integration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import mujoco
import numpy as np

SCHEMA_VERSION = "kaihand-whiteboard-geometric-area-v1"


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
  """Return a counterclockwise convex polygon, without repeated vertices."""
  points = np.unique(np.asarray(points, dtype=np.float64), axis=0)
  if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
    raise ValueError("at least three two-dimensional footprint vertices required")
  points = points[np.lexsort((points[:, 1], points[:, 0]))]

  def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

  lower, upper = [], []
  for point in points:
    while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1e-14:
      lower.pop()
    lower.append(point)
  for point in points[::-1]:
    while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1e-14:
      upper.pop()
    upper.append(point)
  polygon = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
  if len(polygon) < 3:
    raise ValueError("felt bottom footprint is degenerate")
  return polygon


def _polygon_area(polygon: np.ndarray) -> float:
  following = np.roll(polygon, -1, axis=0)
  return float(abs(np.sum(polygon[:, 0] * following[:, 1]
                          - polygon[:, 1] * following[:, 0])) / 2)


def _inside_convex(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
  """Cell-centre inclusion in a counterclockwise polygon."""
  edges = np.roll(polygon, -1, axis=0) - polygon
  normals = np.column_stack((-edges[:, 1], edges[:, 0]))
  offsets = np.einsum("ij,ij->i", polygon, normals)
  return np.all(points @ normals.T >= offsets[None, :] - 1e-12, axis=1)


def _midpoint_grid(lower: np.ndarray, upper: np.ndarray, resolution: float):
  counts = np.maximum(1, np.ceil((upper - lower) / resolution).astype(int))
  spacing = (upper - lower) / counts
  if np.any(spacing <= 0):
    raise ValueError("raster footprint must have positive extent on both axes")
  axes = [lower[i] + (np.arange(counts[i]) + 0.5) * spacing[i] for i in range(2)]
  x, y = np.meshgrid(*axes, indexing="ij")
  return np.column_stack((x.ravel(), y.ravel())), spacing


def _capsule_membership(points: np.ndarray, start: np.ndarray,
                        end: np.ndarray, radius: float) -> np.ndarray:
  direction = end - start
  square_length = float(direction @ direction)
  parameter = np.zeros(len(points)) if square_length <= 1e-20 else np.clip(
    (points - start) @ direction / square_length, 0.0, 1.0
  )
  nearest = start + parameter[:, None] * direction
  return np.sum((points - nearest) ** 2, axis=1) <= radius * radius + 1e-15


def _body_vertices(simulation, geom_id: int, body_id: int) -> np.ndarray:
  """Account for MuJoCo's compiled mesh centring and principal-axis rotation."""
  model, data = simulation.model, simulation.data
  mesh_id = int(model.geom_dataid[geom_id])
  if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH) or mesh_id < 0:
    raise ValueError("eraser_pad must be the compiled felt collision mesh")
  start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
  vertices = np.asarray(model.mesh_vert[start:start + count], dtype=np.float64)
  world = vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
  return (world - data.xpos[body_id]) @ data.xmat[body_id].reshape(3, 3)


class WhiteboardGeometryMetrics:
  """Accumulate geometry/contact/cleaning diagnostics without changing physics.

  ``update(simulation, dt=None)`` returns a JSON-serializable snapshot. ``dt``
  defaults to the difference between successive simulation clocks; passing it
  is useful when the observer sees beginning-of-step cached poses. Zero-duration
  updates refresh the snapshot but do not accumulate area or contact time.
  """

  def __init__(
    self, simulation, *, geometry_sample_hz: float = 30.0,
    pad_grid_resolution_m: float = 0.001,
    ink_grid_resolution_m: float = 0.0005, max_gap_m: float = 0.002,
    max_penetration_m: float = 0.002, max_normal_angle_deg: float = 20.0,
    minimum_normal_force_n: float = 1e-8,
  ):
    values = (geometry_sample_hz, pad_grid_resolution_m, ink_grid_resolution_m,
              max_gap_m, max_penetration_m, minimum_normal_force_n)
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in values):
      raise ValueError("sampling rates, resolutions, gaps and force gate must be positive")
    if not math.isfinite(float(max_normal_angle_deg)) or not 0 < max_normal_angle_deg < 90:
      raise ValueError("normal angle must be finite and between zero and 90 degrees")
    self.simulation = simulation
    self.geometry_sample_hz = float(geometry_sample_hz)
    self.sample_period_s = 1.0 / self.geometry_sample_hz
    self.max_gap_m = float(max_gap_m)
    self.max_penetration_m = float(max_penetration_m)
    self.max_normal_angle_deg = float(max_normal_angle_deg)
    self.minimum_normal_force_n = float(minimum_normal_force_n)
    self._normal_cosine = math.cos(math.radians(self.max_normal_angle_deg))
    model, data = simulation.model, simulation.data
    self.pad_id = int(model.geom("eraser_pad").id)
    self.board_id = int(model.geom("board_surface").id)
    self.body_id = int(model.body("eraser").id)
    if int(model.geom_type[self.board_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
      raise ValueError("board_surface must be a finite box")
    self.board_half_sizes_m = np.asarray(model.geom_size[self.board_id], dtype=float).copy()
    body_vertices = _body_vertices(simulation, self.pad_id, self.body_id)
    self.pad_bottom_z_m = float(body_vertices[:, 2].min())
    bottom = body_vertices[np.abs(body_vertices[:, 2] - self.pad_bottom_z_m) <= 1e-7]
    self.pad_polygon = _convex_hull_2d(bottom[:, :2])
    self.full_pad_bottom_area_m2 = _polygon_area(self.pad_polygon)
    points, self.pad_raster_spacing_m = _midpoint_grid(
      self.pad_polygon.min(axis=0), self.pad_polygon.max(axis=0), pad_grid_resolution_m
    )
    points = points[_inside_convex(points, self.pad_polygon)]
    if not len(points) or self.full_pad_bottom_area_m2 <= 0:
      raise ValueError("felt bottom has no positive-area quadrature samples")
    self._pad_points_body = np.column_stack((points, np.full(len(points), self.pad_bottom_z_m)))
    # Equal midpoint weights are normalized to the exact polygon area, so a
    # completely flat, fully contained loaded pad always has ratio exactly one.
    self._pad_point_area_m2 = self.full_pad_bottom_area_m2 / len(points)
    self.ink_ids = np.asarray(simulation.ink_ids, dtype=int).copy()
    if not len(self.ink_ids):
      raise ValueError("initial ink geometry is required for target-union coverage")
    self._initialize_ink_union(simulation, float(ink_grid_resolution_m))
    self._start_time = self._last_time = float(data.time)
    self._last_geometry_time = -math.inf
    self._sampled_area_m2 = 0.0
    self._sampled_normal_alignment = 0.0
    self._current_contact = False
    self._prior_contact = False
    self._contact_force_source = (
      "task_cached_integrated_eraser_pad_board_force"
      if hasattr(simulation, "board_force") else "live_pad_board_solver_contact_force"
    )
    self._elapsed_s = 0.0
    self._contact_duration_s = 0.0
    self._effective_duration_s = 0.0
    self._area_integral_m2_s = 0.0
    self._peak_area_m2 = 0.0
    self._update_count = self._geometry_sample_count = 0
    self._max_geometry_interval_s = 0.0
    self._covered = np.zeros(len(self._ink_points_board), dtype=bool)
    self._cleaned_fraction = self._area_weighted_cleaned_fraction = 0.0
    self._cleaning_snapshot(simulation)

  def _initialize_ink_union(self, simulation, resolution: float) -> None:
    model, data = simulation.model, simulation.data
    board_rotation = data.geom_xmat[self.board_id].reshape(3, 3)
    centres = (data.geom_xpos[self.ink_ids] - data.geom_xpos[self.board_id]) @ board_rotation
    axes = data.geom_xmat[self.ink_ids].reshape(-1, 3, 3)[:, :, 2] @ board_rotation
    sizes = model.geom_size[self.ink_ids]
    if not np.all(model.geom_type[self.ink_ids] == int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
      raise ValueError("whiteboard initial ink footprints must be capsules")
    if np.any(np.abs(axes[:, 2]) > 1e-5):
      raise ValueError("ink capsule axes must lie in the board plane")
    self._ink_start = centres[:, :2] - axes[:, :2] * sizes[:, 1, None]
    self._ink_end = centres[:, :2] + axes[:, :2] * sizes[:, 1, None]
    self._ink_radii = np.asarray(sizes[:, 0], dtype=float).copy()
    lower = np.minimum(self._ink_start, self._ink_end) - self._ink_radii[:, None]
    upper = np.maximum(self._ink_start, self._ink_end) + self._ink_radii[:, None]
    lower = np.maximum(lower.min(axis=0), -self.board_half_sizes_m[:2])
    upper = np.minimum(upper.max(axis=0), self.board_half_sizes_m[:2])
    points, self.ink_raster_spacing_m = _midpoint_grid(lower, upper, resolution)
    membership = np.column_stack([
      _capsule_membership(points, start, end, radius)
      for start, end, radius in zip(self._ink_start, self._ink_end, self._ink_radii, strict=True)
    ])
    covered = np.any(membership, axis=1)
    self._ink_points_board = np.column_stack((
      points[covered], np.full(np.count_nonzero(covered), self.board_half_sizes_m[2])
    ))
    if not len(self._ink_points_board):
      raise ValueError("initial ink union raster has no samples inside the finite board")
    self._ink_point_area_m2 = float(np.prod(self.ink_raster_spacing_m))
    self.initial_ink_union_area_m2 = len(self._ink_points_board) * self._ink_point_area_m2
    membership = membership[covered]
    # Overlapping patch cells are split evenly between their owners, rather
    # than double-counted. These fixed weights add up to the target union area.
    shared_weights = membership / membership.sum(axis=1)[:, None]
    self._ink_patch_union_weights_m2 = shared_weights.sum(axis=0) * self._ink_point_area_m2

  def _loaded_contact(self, simulation) -> bool:
    if hasattr(simulation, "board_force"):
      # WhiteboardWipeSimulation._update_cleaning accumulates board_force only
      # for {eraser_pad, board_surface}; direct-hand and table loads have their
      # own fields. Its integrated-step value survives the final mj_forward.
      force = float(simulation.board_force)
      if not math.isfinite(force) or force < 0:
        raise ValueError("cached integrated pad/board load must be finite and nonnegative")
      self._contact_force_source = "task_cached_integrated_eraser_pad_board_force"
      return force > self.minimum_normal_force_n
    self._contact_force_source = "live_pad_board_solver_contact_force"
    force = 0.0
    wrench = np.zeros(6)
    for contact_index in range(int(simulation.data.ncon)):
      contact = simulation.data.contact[contact_index]
      if {int(contact.geom1), int(contact.geom2)} != {self.pad_id, self.board_id}:
        continue
      mujoco.mj_contactForce(simulation.model, simulation.data, contact_index, wrench)
      normal_force = float(wrench[0])
      if not math.isfinite(normal_force):
        raise ValueError("non-finite pad/board solver contact force")
      force += max(0.0, normal_force)
    if not math.isfinite(force):
      raise ValueError("non-finite pad/board solver contact force")
    return force > self.minimum_normal_force_n

  def _sample_geometry(self, simulation, now: float) -> None:
    data = simulation.data
    eraser_rotation = data.xmat[self.body_id].reshape(3, 3)
    board_rotation = data.geom_xmat[self.board_id].reshape(3, 3)
    board_origin = data.geom_xpos[self.board_id]
    eraser_origin = data.xpos[self.body_id]
    # Bottom outward normal (-eraser Z) must oppose the board top outward (+Z).
    self._sampled_normal_alignment = float(eraser_rotation[:, 2] @ board_rotation[:, 2])
    self._sampled_area_m2 = 0.0
    if self._current_contact and self._sampled_normal_alignment >= self._normal_cosine:
      world = self._pad_points_body @ eraser_rotation.T + eraser_origin
      board_local = (world - board_origin) @ board_rotation
      gaps = board_local[:, 2] - self.board_half_sizes_m[2]
      effective = (
        (np.abs(board_local[:, 0]) <= self.board_half_sizes_m[0])
        & (np.abs(board_local[:, 1]) <= self.board_half_sizes_m[1])
        & (gaps >= -self.max_penetration_m) & (gaps <= self.max_gap_m)
      )
      self._sampled_area_m2 = float(np.count_nonzero(effective) * self._pad_point_area_m2)
      # Test each ink-union cell against the same finite, loaded bottom plane.
      world_ink = self._ink_points_board @ board_rotation.T + board_origin
      body_ink = (world_ink - eraser_origin) @ eraser_rotation
      body_gap = self.pad_bottom_z_m - body_ink[:, 2]
      # Convert separation along pad normal to separation along board normal.
      board_gap = body_gap / self._sampled_normal_alignment
      touched = (
        _inside_convex(body_ink[:, :2], self.pad_polygon)
        & (board_gap >= -self.max_penetration_m) & (board_gap <= self.max_gap_m)
      )
      self._covered |= touched
    if math.isfinite(self._last_geometry_time):
      self._max_geometry_interval_s = max(self._max_geometry_interval_s,
                                          now - self._last_geometry_time)
    self._last_geometry_time = now
    self._geometry_sample_count += 1

  def _cleaning_snapshot(self, simulation) -> None:
    remaining = np.asarray(simulation.remaining, dtype=float)
    if remaining.shape != self.ink_ids.shape or not np.isfinite(remaining).all():
      raise ValueError("remaining ink must be a finite array matching the initial patches")
    remaining = np.clip(remaining, 0.0, 1.0)
    self._cleaned_fraction = float(1.0 - np.mean(remaining))
    self._area_weighted_cleaned_fraction = float(
      1.0 - remaining @ self._ink_patch_union_weights_m2 / self.initial_ink_union_area_m2
    )

  def update(self, simulation=None, dt: float | None = None) -> Mapping[str, object]:
    simulation = self.simulation if simulation is None else simulation
    now = float(simulation.data.time)
    duration = now - self._last_time if dt is None else float(dt)
    if math.isfinite(now) and now < self._last_time - 1e-12:
      raise ValueError("simulation was reset; instantiate fresh whiteboard metrics")
    if not math.isfinite(now) or not math.isfinite(duration) or duration < -1e-12:
      raise ValueError("observer clock/duration must be finite and nondecreasing")
    duration = max(0.0, duration)
    self._current_contact = self._loaded_contact(simulation)
    if (now - self._last_geometry_time >= self.sample_period_s - 1e-12
        or self._current_contact != self._prior_contact):
      self._sample_geometry(simulation, now)
    current_area = self._sampled_area_m2 if self._current_contact else 0.0
    self._elapsed_s += duration
    self._contact_duration_s += duration * self._current_contact
    self._effective_duration_s += duration * (current_area > 0.0)
    self._area_integral_m2_s += duration * current_area
    self._peak_area_m2 = max(self._peak_area_m2, current_area)
    self._last_time, self._prior_contact = now, self._current_contact
    self._update_count += 1
    self._cleaning_snapshot(simulation)
    return self.snapshot()

  def snapshot(self) -> dict[str, object]:
    area = self._sampled_area_m2 if self._current_contact else 0.0
    elapsed = self._elapsed_s
    contact_time = self._contact_duration_s
    coverage_area = float(np.count_nonzero(self._covered) * self._ink_point_area_m2)
    mean_area = self._area_integral_m2_s / elapsed if elapsed > 0 else 0.0
    contact_mean = self._area_integral_m2_s / contact_time if contact_time > 0 else 0.0
    return {
      "schema_version": SCHEMA_VERSION,
      "measurement_type": "loaded_rigid_mesh_geometric_area_estimate",
      "is_true_deformable_contact_area": False,
      "full_pad_bottom_area_m2": self.full_pad_bottom_area_m2,
      "effective_pad_contact_area_m2": area,
      "effective_pad_contact_area_ratio": area / self.full_pad_bottom_area_m2,
      "time_weighted_mean_effective_area_m2": mean_area,
      "time_weighted_mean_effective_area_ratio": mean_area / self.full_pad_bottom_area_m2,
      "contact_time_weighted_mean_effective_area_ratio": contact_mean / self.full_pad_bottom_area_m2,
      "peak_effective_area_ratio": self._peak_area_m2 / self.full_pad_bottom_area_m2,
      "initial_ink_union_area_m2": self.initial_ink_union_area_m2,
      "cumulative_covered_ink_union_area_m2": coverage_area,
      "cumulative_ink_coverage_ratio": coverage_area / self.initial_ink_union_area_m2,
      "cleaned_fraction": self._cleaned_fraction,
      "ink_union_area_weighted_cleaned_fraction": self._area_weighted_cleaned_fraction,
      "elapsed_s": elapsed,
      "pad_board_contact_duration_s": contact_time,
      "effective_area_contact_duration_s": self._effective_duration_s,
      "pad_board_contact_time_fraction": contact_time / elapsed if elapsed > 0 else 0.0,
      "effective_area_contact_time_fraction": self._effective_duration_s / elapsed if elapsed > 0 else 0.0,
      "pad_board_loaded_contact": self._current_contact,
      "contact_force_source": self._contact_force_source,
      "geometry_pose_source": "accepted_interval_end_forward_kinematics",
      "load_geometry_alignment": (
        "integrated pad-board load cached by task plus interval-end geometry"
        if self._contact_force_source == "task_cached_integrated_eraser_pad_board_force"
        else "live pad-board solver load plus currently supplied geometry"
      ),
      "sampled_normal_alignment_cosine": self._sampled_normal_alignment,
      "geometry_sample_hz_requested": self.geometry_sample_hz,
      "geometry_sample_count": self._geometry_sample_count,
      "physics_observer_update_count": self._update_count,
      "maximum_geometry_sample_interval_s": self._max_geometry_interval_s,
      "geometry_sample_age_s": max(0.0, self._last_time - self._last_geometry_time)
      if math.isfinite(self._last_geometry_time) else None,
      "pad_quadrature_point_count": len(self._pad_points_body),
      "pad_grid_spacing_m": self.pad_raster_spacing_m.tolist(),
      "ink_union_raster_point_count": len(self._ink_points_board),
      "ink_grid_spacing_m": self.ink_raster_spacing_m.tolist(),
      "area_denominator": "compiled_felt_mesh_flat_bottom_polygon_area",
      "coverage_denominator": "fixed_initial_ink_capsule_footprint_union_on_finite_board",
      "coverage_overlap_policy": "unique_initial_ink_union_cells_counted_once_across_all_passes",
      "area_weighted_cleaning_overlap_policy": "each_union_cell_shared_equally_between_overlapping_ink_patches",
      "area_integration": "physics_dt_weighted_zero_order_hold_of_sampled_geometry_with_physics_rate_loaded_contact_gate",
      "sampling_limitations": "rigid proxy; midpoint spatial quadrature; no swept interpolation between geometry samples; sampled coverage may miss brief contacts or motion between samples",
      "spatial_boundary_uncertainty_m": float(max(np.linalg.norm(self.pad_raster_spacing_m),
                                                   np.linalg.norm(self.ink_raster_spacing_m))),
      "thresholds": {
        "maximum_surface_gap_m": self.max_gap_m,
        "maximum_surface_penetration_m": self.max_penetration_m,
        "maximum_opposed_normal_error_deg": self.max_normal_angle_deg,
        "minimum_solver_normal_force_n": self.minimum_normal_force_n,
      },
    }

  def report(self) -> dict[str, object]:
    """Alias used by trial finalization; includes denominator/method metadata."""
    return self.snapshot()
