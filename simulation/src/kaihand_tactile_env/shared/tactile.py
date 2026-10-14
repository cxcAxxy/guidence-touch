"""Tactile providers for the pure-MuJoCo dual-arm workcell.

The solver-contact provider remains available for comparison.  The bilateral
Genesis-compatible provider uses spherical probe geometry, signed geometry
distance and a physical-contact gate.  Per-probe depth is exposed to the
dataset recorder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mujoco
import numpy as np

from kaihand_tactile_env.tactile.layout import TaxelLayout, load_taxel_layout

from .config import FINGERTIP_LINK_NAMES, OBJECT_NAMES

RIGHT_FINGERTIP_LINK_NAMES = tuple(
  name for name in FINGERTIP_LINK_NAMES if name.startswith("hand_r_")
)
GENESIS_PROBE_GEOM_PREFIX = "workcell_genesis_probe_"


def default_fingertip_layout_path() -> Path:
  return (
    Path(__file__).resolve().parents[1]
    / "tactile"
    / "taxels"
    / "kaihand_bimanual_fingertip_genesis.json"
  )


def load_fingertip_layout(path: str | Path | None = None) -> TaxelLayout:
  """Load an explicit or the accepted bilateral fingertip layout."""
  return load_taxel_layout(
    Path(path).expanduser().resolve()
    if path is not None
    else default_fingertip_layout_path()
  )


def compile_model_with_genesis_probes(
  model_path: str | Path,
  layout_path: str | Path | None = None,
) -> tuple[mujoco.MjModel, TaxelLayout]:
  """Compile a workcell model with non-colliding spherical probe geoms."""
  layout = load_fingertip_layout(layout_path)
  specification = mujoco.MjSpec.from_file(str(Path(model_path).resolve()))
  for index, (link_name, position, radius) in enumerate(
    zip(
      layout.body_names,
      layout.local_pos,
      layout.probe_radius,
      strict=True,
    )
  ):
    body = specification.body(link_name)
    if body is None:
      raise ValueError(f"Workcell model is missing probe owner {link_name!r}")
    body.add_geom(
      name=f"{GENESIS_PROBE_GEOM_PREFIX}{index:04d}",
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      pos=np.asarray(position, dtype=float),
      size=[max(float(radius), 1.0e-9), 0.0, 0.0],
      contype=0,
      conaffinity=0,
      group=5,
      rgba=[0.1, 0.5, 1.0, 0.0],
    )
  return specification.compile(), layout


def _default_target_geom_names(model: mujoco.MjModel) -> tuple[str, ...]:
  """Find enabled collision geoms belonging to the model's task objects.

  Independent scenes only contain their own object.  The legacy combined
  scene disables the other object's collision masks when it selects a task.
  Visual shells and fixed scene furniture are not tactile targets.
  """
  body_ids = {
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in OBJECT_NAMES
  } - {-1}
  names = tuple(
    model.geom(geom_id).name
    for geom_id in range(model.ngeom)
    if int(model.geom_bodyid[geom_id]) in body_ids
    and (model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0)
  )
  if not names:
    raise RuntimeError(
      "MuJoCo model has no enabled task-object collision geometry for tactile sensing; "
      "select/reset the scene first or supply target_geom_names explicitly"
    )
  return names


@dataclass(frozen=True)
class LinkTactileSample:
  """One synchronized aggregate sample for all ten fingertips.

  ``contact`` is the provider's thresholded and possibly debounced state; it
  is not a validity mask for ``centroid_world``.  A centroid is defined when
  the sample contains a strictly positive spatial weight and is otherwise
  represented by three NaNs.  Consumers should use :attr:`centroid_valid`
  before filling missing centroids for training.
  """

  timestamp: float
  source: str
  link_names: tuple[str, ...]
  contact: np.ndarray
  normal_force: np.ndarray
  contact_count: np.ndarray
  force_world: np.ndarray
  torque_world: np.ndarray
  force_local: np.ndarray
  centroid_world: np.ndarray

  def __post_init__(self) -> None:
    count = len(self.link_names)
    expected = {
      "contact": (count,),
      "normal_force": (count,),
      "contact_count": (count,),
      "force_world": (count, 3),
      "torque_world": (count, 3),
      "force_local": (count, 3),
      "centroid_world": (count, 3),
    }
    for name, shape in expected.items():
      if getattr(self, name).shape != shape:
        raise ValueError(f"{name} must have shape {shape}")

  @property
  def centroid_valid(self) -> np.ndarray:
    """Return the per-link mask for finite, spatially defined centroids."""
    return np.all(np.isfinite(self.centroid_world), axis=1)


class TactileProvider(Protocol):
  """Minimal provider API consumed by recording and comparison tools."""

  source: str
  link_names: tuple[str, ...]

  @property
  def available(self) -> bool: ...

  def read(self, data: mujoco.MjData) -> LinkTactileSample: ...


class SolverContactTactileProvider:
  """Aggregate MuJoCo constraint contact wrench on each fingertip link."""

  source = "solver_contact_proxy_v1"
  dataset_group = "tactile_proxy"
  is_genesis_probe_truth = False
  status = "temporary_solver_contact_baseline"
  force_unit = "N"

  def __init__(
    self,
    model: mujoco.MjModel,
    link_names: tuple[str, ...] = FINGERTIP_LINK_NAMES,
  ) -> None:
    self.model = model
    self.link_names = tuple(link_names)
    self._body_ids = np.array(
      [_require_id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.link_names],
      dtype=np.int32,
    )
    self._body_to_index = {
      int(body_id): index for index, body_id in enumerate(self._body_ids)
    }

  @property
  def available(self) -> bool:
    return True

  def read(self, data: mujoco.MjData) -> LinkTactileSample:
    link_count = len(self.link_names)
    contact = np.zeros(link_count, dtype=bool)
    normal_force = np.zeros(link_count, dtype=np.float64)
    contact_count = np.zeros(link_count, dtype=np.int32)
    force_world = np.zeros((link_count, 3), dtype=np.float64)
    torque_world = np.zeros((link_count, 3), dtype=np.float64)
    centroid_accumulator = np.zeros((link_count, 3), dtype=np.float64)
    centroid_weight = np.zeros(link_count, dtype=np.float64)
    contact_wrench = np.zeros(6, dtype=np.float64)

    for contact_id in range(data.ncon):
      item = data.contact[contact_id]
      # MuJoCo uses geom id -1 for the flex side of a flex/rigid contact.  Do
      # not accidentally index the last rigid geom when a task uses a deformable
      # object (for example vase wiping).
      geom1_body = (
        int(self.model.geom_bodyid[item.geom1]) if int(item.geom1) >= 0 else -1
      )
      geom2_body = (
        int(self.model.geom_bodyid[item.geom2]) if int(item.geom2) >= 0 else -1
      )
      indexed_sides: list[tuple[int, float]] = []
      if geom1_body in self._body_to_index:
        indexed_sides.append((self._body_to_index[geom1_body], -1.0))
      if geom2_body in self._body_to_index:
        indexed_sides.append((self._body_to_index[geom2_body], 1.0))
      if not indexed_sides:
        continue

      mujoco.mj_contactForce(self.model, data, contact_id, contact_wrench)
      frame = np.asarray(item.frame).reshape(3, 3)
      # mj_contactForce is expressed in the contact frame and acts on geom2.
      wrench_force_world = frame.T @ contact_wrench[:3]
      wrench_torque_world = frame.T @ contact_wrench[3:]
      weight = abs(float(contact_wrench[0]))
      for index, sign in indexed_sides:
        contact[index] = True
        contact_count[index] += 1
        normal_force[index] += weight
        force_world[index] += sign * wrench_force_world
        torque_world[index] += sign * wrench_torque_world
        centroid_accumulator[index] += weight * np.asarray(item.pos)
        centroid_weight[index] += weight

    nonzero = centroid_weight > 0.0
    centroid_world = np.full((link_count, 3), np.nan, dtype=np.float64)
    centroid_world[nonzero] = (
      centroid_accumulator[nonzero] / centroid_weight[nonzero, None]
    )
    force_local = np.zeros_like(force_world)
    for index, body_id in enumerate(self._body_ids):
      rotation_world_from_body = data.xmat[body_id].reshape(3, 3)
      force_local[index] = rotation_world_from_body.T @ force_world[index]

    return LinkTactileSample(
      timestamp=float(data.time),
      source=self.source,
      link_names=self.link_names,
      contact=contact,
      normal_force=normal_force,
      contact_count=contact_count,
      force_world=force_world,
      torque_world=torque_world,
      force_local=force_local,
      centroid_world=centroid_world,
    )


class GenesisProbeTactileProvider:
  """Bilateral clean Genesis ``depth``/``bool``/``agg_force`` provider.

  Probe spheres are non-colliding geoms injected by
  :func:`compile_model_with_genesis_probes`.  Distance queries are gated by an
  actual solver contact between the owning distal link and the candidate object,
  matching the anti-telepathy gate in Tactile Genesis.  ``probe_contact`` is
  thresholded, hysteretic and release-debounced, while
  ``probe_contact_instantaneous`` is the current-frame threshold mask.
  ``probe_depth`` always remains an instantaneous geometric quantity.
  Targets default to the enabled collision geometry of the selected task's
  objects; callers may supply explicit geometry names for custom models.
  """

  source = "genesis_probe_bimanual_clean_v1"
  dataset_group = "tactile_genesis"
  is_genesis_probe_truth = True
  status = "accepted"
  force_unit = "genesis_agg_force_depth_x_1e4"

  def __init__(
    self,
    model: mujoco.MjModel,
    layout: TaxelLayout | None = None,
    *,
    target_geom_names: tuple[str, ...] | None = None,
    contact_threshold_m: float = 5.0e-5,
    release_threshold_m: float = 2.5e-5,
    bool_count_threshold: int = 0,
    release_debounce_steps: int = 10,
    aggregate_force_scale: float = 1.0e4,
  ) -> None:
    self.model = model
    self.layout = layout or load_fingertip_layout()
    self.link_names = tuple(dict.fromkeys(self.layout.body_names))
    if self.link_names != FINGERTIP_LINK_NAMES:
      raise ValueError(
        "bilateral Genesis layout must contain all ten fingertip links in canonical order"
      )
    self._link_to_index = {name: index for index, name in enumerate(self.link_names)}
    self._probe_link_index = np.asarray(
      [self._link_to_index[name] for name in self.layout.body_names], dtype=np.int32
    )
    self._probe_indices_by_link = tuple(
      np.flatnonzero(self._probe_link_index == link_index)
      for link_index in range(len(self.link_names))
    )
    self._body_ids = np.asarray(
      [_require_id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.link_names],
      dtype=np.int32,
    )
    self._body_to_link = {
      int(body_id): index for index, body_id in enumerate(self._body_ids)
    }
    self._probe_geom_ids = np.asarray(
      [
        _require_id(
          model,
          mujoco.mjtObj.mjOBJ_GEOM,
          f"{GENESIS_PROBE_GEOM_PREFIX}{index:04d}",
        )
        for index in range(self.layout.count)
      ],
      dtype=np.int32,
    )
    if target_geom_names is None:
      target_geom_names = _default_target_geom_names(model)
    self._target_geom_ids = np.asarray(
      [
        _require_id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in target_geom_names
      ],
      dtype=np.int32,
    )
    self._target_geom_set = set(self._target_geom_ids.tolist())
    self.contact_threshold_m = float(contact_threshold_m)
    self.release_threshold_m = float(release_threshold_m)
    self.bool_count_threshold = int(bool_count_threshold)
    self.release_debounce_steps = int(release_debounce_steps)
    self.aggregate_force_scale = float(aggregate_force_scale)
    if self.contact_threshold_m < 0.0 or self.release_threshold_m < 0.0:
      raise ValueError("contact thresholds must be non-negative")
    if self.release_debounce_steps < 0:
      raise ValueError("release_debounce_steps cannot be negative")
    self.probe_contact = np.zeros(self.layout.count, dtype=bool)
    self.probe_contact_instantaneous = np.zeros(self.layout.count, dtype=bool)
    self._release_until = np.full(self.layout.count, -np.inf, dtype=np.float64)
    self.probe_depth = np.zeros(self.layout.count, dtype=np.float64)
    self.probe_target_geom_id = np.full(self.layout.count, -1, dtype=np.int32)
    self._fromto = np.zeros(6, dtype=np.float64)

  def reset(self) -> None:
    """Clear per-probe contact hysteresis after a simulator reset."""
    self.probe_contact.fill(False)
    self.probe_contact_instantaneous.fill(False)
    self._release_until.fill(-np.inf)
    self.probe_depth.fill(0.0)
    self.probe_target_geom_id.fill(-1)

  @property
  def available(self) -> bool:
    return True

  def _candidate_geoms(self, data: mujoco.MjData) -> list[set[int]]:
    candidates = [set() for _ in self.link_names]
    for contact_id in range(data.ncon):
      item = data.contact[contact_id]
      geom1 = int(item.geom1)
      geom2 = int(item.geom2)
      body1 = int(self.model.geom_bodyid[geom1])
      body2 = int(self.model.geom_bodyid[geom2])
      if body1 in self._body_to_link and geom2 in self._target_geom_set:
        candidates[self._body_to_link[body1]].add(geom2)
      if body2 in self._body_to_link and geom1 in self._target_geom_set:
        candidates[self._body_to_link[body2]].add(geom1)
    return candidates

  def read(self, data: mujoco.MjData) -> LinkTactileSample:
    candidates = self._candidate_geoms(data)
    depth = np.zeros(self.layout.count, dtype=np.float64)
    target_ids = np.full(self.layout.count, -1, dtype=np.int32)
    # Only links with a solver contact can pass the anti-telepathy gate.  Most
    # trajectory frames have no fingertip contact, so avoid walking all 350
    # probes in Python when their candidate set is empty.
    for link_index, target_geoms in enumerate(candidates):
      if not target_geoms:
        continue
      for probe_index in self._probe_indices_by_link[link_index]:
        best_distance = np.inf
        best_target = -1
        probe_geom = self._probe_geom_ids[probe_index]
        for target_geom in target_geoms:
          distance = mujoco.mj_geomDistance(
            self.model,
            data,
            int(probe_geom),
            target_geom,
            1.0,
            self._fromto,
          )
          # MuJoCo begins resolving contact inside a geom's positive margin.
          # Treat that margin as compliant tactile-pad travel: this aligns sensor
          # onset with the first physical contact constraint without requiring the
          # rigid visual meshes to interpenetrate first.
          distance -= float(self.model.geom_margin[target_geom])
          if distance < best_distance:
            best_distance = float(distance)
            best_target = target_geom
        if best_target >= 0:
          depth[probe_index] = max(-best_distance, 0.0)
          target_ids[probe_index] = best_target

    valid_probe = self.layout.probe_radius > 0.0
    activated = depth >= self.contact_threshold_m
    self.probe_contact_instantaneous = activated & valid_probe
    sustained = self.probe_contact & (depth > self.release_threshold_m)
    refreshed = activated | sustained
    debounce_seconds = self.release_debounce_steps * float(self.model.opt.timestep)
    self._release_until[refreshed] = float(data.time) + debounce_seconds
    # Base release hysteresis on simulation time, not read-call frequency.  A
    # recorder and the 500 Hz dashboard therefore observe the same 20 ms tail.
    # ``probe_contact`` intentionally remains true during this tail even when
    # the instantaneous geometric depth has returned to zero.  Consumers that
    # need a non-debounced label should use ``probe_contact_instantaneous``.
    decaying = ~refreshed & (float(data.time) < self._release_until)
    latched = refreshed | decaying
    self.probe_contact = latched & valid_probe
    self.probe_depth = depth
    self.probe_target_geom_id = target_ids

    link_count = len(self.link_names)
    contact_count = np.zeros(link_count, dtype=np.int32)
    np.add.at(
      contact_count, self._probe_link_index, self.probe_contact.astype(np.int32)
    )
    contact = contact_count > self.bool_count_threshold
    force_local = np.zeros((link_count, 3), dtype=np.float64)
    np.add.at(
      force_local,
      self._probe_link_index,
      depth[:, None] * self.layout.local_normal * self.aggregate_force_scale,
    )
    normal_force = np.linalg.norm(force_local, axis=1)
    force_world = np.zeros_like(force_local)
    torque_world = np.zeros_like(force_local)
    centroid_world = np.full((link_count, 3), np.nan, dtype=np.float64)
    centroid_sum = np.zeros((link_count, 3), dtype=np.float64)
    centroid_weight = np.zeros(link_count, dtype=np.float64)

    for link_index, body_id in enumerate(self._body_ids):
      rotation = data.xmat[body_id].reshape(3, 3)
      force_world[link_index] = rotation @ force_local[link_index]
    for probe_index, (link_index, body_id) in enumerate(
      zip(
        self._probe_link_index,
        self._body_ids[self._probe_link_index],
        strict=True,
      )
    ):
      if depth[probe_index] <= 0.0:
        continue
      rotation = data.xmat[body_id].reshape(3, 3)
      probe_position = (
        data.xpos[body_id] + rotation @ self.layout.local_pos[probe_index]
      )
      probe_force_world = (
        rotation
        @ (depth[probe_index] * self.layout.local_normal[probe_index])
        * self.aggregate_force_scale
      )
      torque_world[link_index] += np.cross(
        probe_position - data.xpos[body_id], probe_force_world
      )
      centroid_sum[link_index] += depth[probe_index] * probe_position
      centroid_weight[link_index] += depth[probe_index]
    active_links = centroid_weight > 0.0
    centroid_world[active_links] = (
      centroid_sum[active_links] / centroid_weight[active_links, None]
    )
    return LinkTactileSample(
      timestamp=float(data.time),
      source=self.source,
      link_names=self.link_names,
      contact=contact,
      normal_force=normal_force,
      contact_count=contact_count,
      force_world=force_world,
      torque_world=torque_world,
      force_local=force_local,
      centroid_world=centroid_world,
    )


def raw_contact_snapshot(
  model: mujoco.MjModel, data: mujoco.MjData
) -> dict[str, np.ndarray]:
  """Return all active contacts as flat arrays for a ragged episode stream."""
  count = int(data.ncon)
  result = {
    "geom1_id": np.empty(count, dtype=np.int32),
    "geom2_id": np.empty(count, dtype=np.int32),
    "body1_id": np.empty(count, dtype=np.int32),
    "body2_id": np.empty(count, dtype=np.int32),
    "distance": np.empty(count, dtype=np.float64),
    "position_world": np.empty((count, 3), dtype=np.float64),
    "frame_world": np.empty((count, 3, 3), dtype=np.float64),
    "wrench_contact_on_geom2": np.empty((count, 6), dtype=np.float64),
    "wrench_world_on_geom2": np.empty((count, 6), dtype=np.float64),
  }
  wrench = np.zeros(6, dtype=np.float64)
  for contact_id in range(count):
    item = data.contact[contact_id]
    frame = np.asarray(item.frame).reshape(3, 3)
    mujoco.mj_contactForce(model, data, contact_id, wrench)
    result["geom1_id"][contact_id] = item.geom1
    result["geom2_id"][contact_id] = item.geom2
    result["body1_id"][contact_id] = (
      model.geom_bodyid[item.geom1] if int(item.geom1) >= 0 else -1
    )
    result["body2_id"][contact_id] = (
      model.geom_bodyid[item.geom2] if int(item.geom2) >= 0 else -1
    )
    result["distance"][contact_id] = item.dist
    result["position_world"][contact_id] = item.pos
    result["frame_world"][contact_id] = frame
    result["wrench_contact_on_geom2"][contact_id] = wrench
    result["wrench_world_on_geom2"][contact_id, :3] = frame.T @ wrench[:3]
    result["wrench_world_on_geom2"][contact_id, 3:] = frame.T @ wrench[3:]
  return result


def compare_link_samples(
  reference: list[LinkTactileSample], candidate: list[LinkTactileSample]
) -> dict[str, float]:
  """Compare two aligned link-aggregate tactile streams."""
  if len(reference) != len(candidate) or not reference:
    raise ValueError("reference and candidate must be non-empty and equally sized")
  if reference[0].link_names != candidate[0].link_names:
    raise ValueError("tactile link order differs")
  reference_force = np.stack([sample.normal_force for sample in reference])
  candidate_force = np.stack([sample.normal_force for sample in candidate])
  reference_contact = np.stack([sample.contact for sample in reference])
  candidate_contact = np.stack([sample.contact for sample in candidate])
  error = candidate_force - reference_force
  true_positive = np.logical_and(reference_contact, candidate_contact).sum()
  false_positive = np.logical_and(~reference_contact, candidate_contact).sum()
  false_negative = np.logical_and(reference_contact, ~candidate_contact).sum()
  precision = float(true_positive / max(true_positive + false_positive, 1))
  recall = float(true_positive / max(true_positive + false_negative, 1))
  f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
  ref_flat = reference_force.ravel()
  cand_flat = candidate_force.ravel()
  correlation = 0.0
  if np.std(ref_flat) > 0.0 and np.std(cand_flat) > 0.0:
    correlation = float(np.corrcoef(ref_flat, cand_flat)[0, 1])
  return {
    "normal_force_rmse_n": float(np.sqrt(np.mean(error**2))),
    "normal_force_mae_n": float(np.mean(np.abs(error))),
    "normal_force_correlation": correlation,
    "contact_precision": precision,
    "contact_recall": recall,
    "contact_f1": f1,
  }


def _require_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
  object_id = mujoco.mj_name2id(model, object_type, name)
  if object_id < 0:
    raise RuntimeError(f"MuJoCo model is missing {name!r}")
  return object_id
