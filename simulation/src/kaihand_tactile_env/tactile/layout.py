"""Load fixed spherical-probe layouts from JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TaxelLayout:
  """Probe positions and normals expressed in their owning body frames."""

  body_names: tuple[str, ...]
  local_pos: np.ndarray
  local_normal: np.ndarray
  probe_radius: np.ndarray
  grid_shape: tuple[tuple[int, int] | None, ...]

  @property
  def count(self) -> int:
    return len(self.body_names)


def load_taxel_layout(path: str | Path) -> TaxelLayout:
  """Load and validate the per-link grid/list schema used by this environment."""
  layout_path = Path(path).expanduser().resolve()
  raw = json.loads(layout_path.read_text(encoding="utf-8"))
  if not isinstance(raw, list) or not raw:
    raise ValueError(f"{layout_path}: expected a non-empty list of probe groups")

  names: list[str] = []
  positions: list[list[float]] = []
  normals: list[list[float]] = []
  radii: list[float] = []
  shapes: list[tuple[int, int] | None] = []
  for group in raw:
    if not isinstance(group, dict) or "link_name" not in group:
      raise ValueError(f"{layout_path}: every group requires link_name")
    probes = group.get("probes")
    kind = str(group.get("kind", "list"))
    if not isinstance(probes, list) or not probes:
      raise ValueError(f"{layout_path}: probe group must be a non-empty list")
    if kind == "grid":
      if not isinstance(probes[0], list) or not probes[0]:
        raise ValueError(f"{layout_path}: grid rows must be non-empty lists")
      width = len(probes[0])
      if any(not isinstance(row, list) or len(row) != width for row in probes):
        raise ValueError(f"{layout_path}: grid must be rectangular")
      flat = [probe for row in probes for probe in row]
      shape: tuple[int, int] | None = (len(probes), width)
    elif kind == "list":
      flat = probes
      shape = None
    else:
      raise ValueError(f"{layout_path}: unsupported group kind {kind!r}")

    for probe in flat:
      if not isinstance(probe, dict):
        raise ValueError(f"{layout_path}: every probe must be an object")
      names.append(str(group["link_name"]))
      positions.append(probe["pos"])
      normals.append(probe["normal"])
      radii.append(float(probe["radius"]))
      shapes.append(shape)

  count = len(names)
  local_pos = np.asarray(positions, dtype=np.float32).reshape(count, 3)
  local_normal = np.asarray(normals, dtype=np.float32).reshape(count, 3)
  probe_radius = np.asarray(radii, dtype=np.float32)
  if not np.isfinite(local_pos).all() or not np.isfinite(local_normal).all():
    raise ValueError(f"{layout_path}: positions and normals must be finite")
  if not np.isfinite(probe_radius).all() or np.any(probe_radius <= 0.0):
    raise ValueError(f"{layout_path}: radii must be finite and positive")
  normal_norm = np.linalg.norm(local_normal, axis=1)
  if np.any(normal_norm < 1.0e-8):
    raise ValueError(f"{layout_path}: normals must be non-zero")
  local_normal /= normal_norm[:, None]
  return TaxelLayout(
    body_names=tuple(names),
    local_pos=local_pos,
    local_normal=local_normal,
    probe_radius=probe_radius,
    grid_shape=tuple(shapes),
  )
