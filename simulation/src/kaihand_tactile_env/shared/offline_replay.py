"""Task-neutral offline replay from saved RGB and tactile HDF5 streams.

The exporter never steps physics or changes the source episode.  It supports
the shared spatial contact-force schema, the PickPlace Genesis probe schema,
and a conservative aggregate-force fallback for older recordings.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import FINGERTIP_LINK_NAMES, SCENE_NAMES, TRAINING_CAMERA_NAMES

SCHEMA_VERSION = "kaihand-offline-multimodal-replay-v1"
FINGER_LABELS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
HAND_LABELS = ("LEFT", "RIGHT")
FORCE_CHANNEL_NAMES = ("Ft_col", "Ft_row", "Fn")
FORCE_CHANNEL_COLORS = ((52, 174, 255), (255, 174, 52), (86, 220, 116))


def _text(value: Any) -> str:
  return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _json_attr(file: h5py.File, name: str) -> dict[str, Any]:
  try:
    value = json.loads(_text(file.attrs.get(name, "{}")))
  except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(f"{name} must contain valid JSON") from error
  if not isinstance(value, dict):
    raise ValueError(f"{name} must contain a JSON object")
  return value


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _clock(values: Any, name: str, *, strict: bool) -> np.ndarray:
  times = np.asarray(values, dtype=np.float64)
  if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
    raise ValueError(f"{name} must be a nonempty finite timestamp vector")
  difference = np.diff(times)
  if np.any(difference <= 0.0 if strict else difference < 0.0):
    raise ValueError(f"{name} timestamps are not ordered")
  return times


def _available_rgb_cameras(file: h5py.File) -> tuple[str, ...]:
  if "cameras" not in file:
    return ()
  return tuple(
    name
    for name in file["cameras"]
    if f"cameras/{name}/rgb" in file
    and file[f"cameras/{name}/rgb"].ndim == 4
    and file[f"cameras/{name}/rgb"].shape[-1] == 3
    and file[f"cameras/{name}/rgb"].dtype == np.uint8
  )


def resolve_replay_cameras(
  file: h5py.File, requested: tuple[str, ...] | None
) -> tuple[str, ...]:
  """Resolve one to four stored RGB cameras in deterministic display order."""

  available = _available_rgb_cameras(file)
  if requested:
    if len(requested) != len(set(requested)):
      raise ValueError("replay cameras must not contain duplicates")
    missing = tuple(name for name in requested if name not in available)
    if missing:
      raise ValueError(
        f"episode has no saved RGB for {missing}; available cameras: {available}"
      )
    cameras = tuple(requested)
  else:
    cameras = tuple(name for name in TRAINING_CAMERA_NAMES if name in available)
    if not cameras:
      cameras = available[:1]
  if not cameras:
    raise ValueError("episode contains no usable saved RGB camera")
  if len(cameras) > 4:
    raise ValueError("multimodal replay supports at most four displayed cameras")
  return cameras


def select_replay_frames(camera_times: np.ndarray, fps: float) -> np.ndarray:
  """Choose distinct saved frames nearest a fixed-rate grid, including endpoints."""

  times = _clock(camera_times, "reference camera", strict=True)
  if not math.isfinite(fps) or fps <= 0.0:
    raise ValueError("replay fps must be finite and positive")
  targets = times[0] + np.arange(
    int(math.floor((times[-1] - times[0]) * fps + 1.0e-8)) + 1
  ) / fps
  right = np.searchsorted(times, targets, side="left").clip(0, len(times) - 1)
  left = np.maximum(0, right - 1)
  selected = np.where(
    np.abs(times[left] - targets) <= np.abs(times[right] - targets), left, right
  )
  return np.unique(np.r_[0, selected, len(times) - 1]).astype(np.int64)


def _nearest_indices(times: np.ndarray, targets: np.ndarray) -> np.ndarray:
  right = np.searchsorted(times, targets, side="left").clip(0, len(times) - 1)
  left = np.maximum(0, right - 1)
  return np.where(
    np.abs(times[left] - targets) <= np.abs(times[right] - targets), left, right
  ).astype(np.int64)


def _causal_indices(times: np.ndarray, targets: np.ndarray, name: str) -> np.ndarray:
  indices = np.searchsorted(times, targets, side="right") - 1
  if np.any(indices < 0):
    raise ValueError(f"some replay frames precede the first {name} sample")
  return indices.astype(np.int64)


def _canonical_link_destinations(names: tuple[str, ...]) -> np.ndarray:
  if len(names) != len(set(names)):
    raise ValueError("tactile link names must be unique")
  unknown = sorted(set(names) - set(FINGERTIP_LINK_NAMES))
  if unknown:
    raise ValueError(f"unknown tactile fingertip links: {unknown}")
  if not names:
    raise ValueError("tactile link names cannot be empty")
  return np.asarray([FINGERTIP_LINK_NAMES.index(name) for name in names], dtype=np.intp)


@dataclass(frozen=True)
class TactileReplaySource:
  kind: str
  group_name: str
  times: np.ndarray
  destinations: np.ndarray
  first_label: str
  second_label: str
  first_unit: str
  second_unit: str
  curve_unit: str
  curve_channels: tuple[str, str, str]
  probe_indices: np.ndarray | None = None


def _spatial_force_source(file: h5py.File) -> TactileReplaySource | None:
  name = "tactile_contact_force"
  if name not in file:
    return None
  group = file[name]
  required = ("link_names", "normal_taxel_force_n", "tangent_taxel_force_n")
  if any(item not in group for item in required):
    return None
  names = tuple(_text(value) for value in group["link_names"][:])
  destinations = _canonical_link_destinations(names)
  normal = group["normal_taxel_force_n"]
  tangent = group["tangent_taxel_force_n"]
  count = normal.shape[0]
  if normal.shape != (count, len(names), 7, 5):
    raise ValueError("normal taxel force must have shape [T,finger,7,5]")
  if tangent.shape != (count, len(names), 7, 5, 2):
    raise ValueError("tangent taxel force must have shape [T,finger,7,5,2]")
  if "timestamp" in group:
    times = _clock(group["timestamp"][:], f"{name}/timestamp", strict=False)
  else:
    times = _clock(file["state/timestamp"][:], "state/timestamp", strict=True)
  if len(times) != count:
    raise ValueError("spatial tactile clock and sample count differ")
  return TactileReplaySource(
    kind="spatial_force",
    group_name=name,
    times=times,
    destinations=destinations,
    first_label="Fn",
    second_label="|Ft|",
    first_unit="N/taxel",
    second_unit="N/taxel",
    curve_unit="N per fingertip",
    curve_channels=FORCE_CHANNEL_NAMES,
  )


def _genesis_probe_source(file: h5py.File) -> TactileReplaySource | None:
  name = "tactile_genesis"
  if name not in file:
    return None
  group = file[name]
  required = (
    "link_names",
    "probe_link_names",
    "probe_depth",
    "probe_contact_instantaneous",
    "force_local",
  )
  if any(item not in group for item in required):
    return None
  link_names = tuple(_text(value) for value in group["link_names"][:])
  destinations = _canonical_link_destinations(link_names)
  probe_names = np.asarray([_text(value) for value in group["probe_link_names"][:]])
  probe_indices = np.full((10, 35), -1, dtype=np.int64)
  for destination, canonical_name in enumerate(FINGERTIP_LINK_NAMES):
    matching = np.flatnonzero(probe_names == canonical_name)
    if len(matching):
      if len(matching) != 35:
        raise ValueError(
          f"Genesis fingertip {canonical_name!r} must contain exactly 35 probes"
        )
      probe_indices[destination] = matching
  count = group["probe_depth"].shape[0]
  if group["probe_depth"].shape != (count, len(probe_names)):
    raise ValueError("Genesis probe depth has an invalid shape")
  if group["probe_contact_instantaneous"].shape != (count, len(probe_names)):
    raise ValueError("Genesis probe contact has an invalid shape")
  if group["force_local"].shape != (count, len(link_names), 3):
    raise ValueError("Genesis local-force proxy has an invalid shape")
  times = _clock(file["state/timestamp"][:], "state/timestamp", strict=True)
  if len(times) != count:
    raise ValueError("Genesis tactile samples must align with state/timestamp")
  return TactileReplaySource(
    kind="genesis_probe",
    group_name=name,
    times=times,
    destinations=destinations,
    first_label="Depth",
    second_label="Contact",
    first_unit="mm",
    second_unit="0/1",
    curve_unit="Genesis local proxy; not N",
    curve_channels=("Fx", "Fy", "Fz"),
    probe_indices=probe_indices,
  )


def _aggregate_force_source(file: h5py.File) -> TactileReplaySource | None:
  for name in ("tactile_proxy", "tactile_genesis"):
    if name not in file:
      continue
    group = file[name]
    required = ("link_names", "normal_force", "force_local")
    if any(item not in group for item in required):
      continue
    names = tuple(_text(value) for value in group["link_names"][:])
    destinations = _canonical_link_destinations(names)
    count = group["normal_force"].shape[0]
    if group["normal_force"].shape != (count, len(names)):
      raise ValueError("aggregate normal force has an invalid shape")
    if group["force_local"].shape != (count, len(names), 3):
      raise ValueError("aggregate local force has an invalid shape")
    times = _clock(file["state/timestamp"][:], "state/timestamp", strict=True)
    if len(times) != count:
      raise ValueError("aggregate tactile samples must align with state/timestamp")
    unit = _text(group.attrs.get("force_unit", "unspecified"))
    return TactileReplaySource(
      kind="aggregate_force",
      group_name=name,
      times=times,
      destinations=destinations,
      first_label="Fn uniform",
      second_label="|Ft| uniform",
      first_unit=f"{unit}/35",
      second_unit=f"{unit}/35",
      curve_unit=f"aggregate local force [{unit}]",
      curve_channels=("Fx", "Fy", "Fz"),
    )
  return None


def resolve_tactile_source(file: h5py.File) -> TactileReplaySource:
  source = (
    _spatial_force_source(file)
    or _genesis_probe_source(file)
    or _aggregate_force_source(file)
  )
  if source is None:
    raise ValueError(
      "episode has no supported tactile_contact_force, Genesis probe, or aggregate tactile stream"
    )
  return source


def read_tactile_sample(
  file: h5py.File, source: TactileReplaySource, index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Return two canonical 10x7x5 maps and ten three-channel curve values."""

  first = np.zeros((10, 7, 5), dtype=np.float64)
  second = np.zeros((10, 7, 5), dtype=np.float64)
  curves = np.zeros((10, 3), dtype=np.float64)
  group = file[source.group_name]
  if source.kind == "spatial_force":
    normal = np.asarray(group["normal_taxel_force_n"][index], dtype=np.float64)
    tangent = np.asarray(group["tangent_taxel_force_n"][index], dtype=np.float64)
    if not np.isfinite(normal).all() or not np.isfinite(tangent).all():
      raise ValueError("spatial tactile sample contains nonfinite values")
    if np.any(normal < -1.0e-12):
      raise ValueError("spatial tactile normal force cannot be negative")
    first[source.destinations] = np.maximum(normal, 0.0)
    second[source.destinations] = np.linalg.norm(tangent, axis=-1)
    curves[source.destinations] = np.column_stack(
      (
        tangent[..., 0].sum(axis=(1, 2)),
        tangent[..., 1].sum(axis=(1, 2)),
        normal.sum(axis=(1, 2)),
      )
    )
  elif source.kind == "genesis_probe":
    assert source.probe_indices is not None
    depth = np.asarray(group["probe_depth"][index], dtype=np.float64)
    contact = np.asarray(
      group["probe_contact_instantaneous"][index], dtype=np.float64
    )
    for destination, indices in enumerate(source.probe_indices):
      if indices[0] >= 0:
        first[destination] = depth[indices].reshape(7, 5) * 1000.0
        second[destination] = contact[indices].reshape(7, 5)
    proxy = np.asarray(group["force_local"][index], dtype=np.float64)
    curves[source.destinations] = proxy
    if (
      not np.isfinite(first).all()
      or not np.isfinite(second).all()
      or not np.isfinite(curves).all()
      or np.any(first < 0.0)
    ):
      raise ValueError("Genesis tactile sample is invalid")
  else:
    normal = np.asarray(group["normal_force"][index], dtype=np.float64)
    local = np.asarray(group["force_local"][index], dtype=np.float64)
    if not np.isfinite(normal).all() or not np.isfinite(local).all():
      raise ValueError("aggregate tactile sample contains nonfinite values")
    first[source.destinations] = np.maximum(normal, 0.0)[:, None, None] / 35.0
    second[source.destinations] = (
      np.linalg.norm(local[:, :2], axis=-1)[:, None, None] / 35.0
    )
    curves[source.destinations] = local
  return first, second, curves


def _nice_positive_scale(value: float, floor: float) -> float:
  value = max(float(value), float(floor))
  exponent = 10.0 ** math.floor(math.log10(value))
  normalized = value / exponent
  step = next(item for item in (1.0, 2.0, 5.0, 10.0) if normalized <= item)
  return step * exponent


def tactile_scales(
  file: h5py.File,
  source: TactileReplaySource,
  indices: np.ndarray,
) -> tuple[float, float, float]:
  first_max = second_max = curve_max = 0.0
  for index in np.unique(indices):
    first, second, curves = read_tactile_sample(file, source, int(index))
    first_max = max(first_max, float(np.max(first)))
    second_max = max(second_max, float(np.max(second)))
    curve_max = max(curve_max, float(np.max(np.abs(curves))))
  if source.kind == "genesis_probe":
    return (
      _nice_positive_scale(first_max, 1.0),
      1.0,
      _nice_positive_scale(curve_max, 1.0),
    )
  return (
    _nice_positive_scale(first_max, 0.1),
    _nice_positive_scale(second_max, 0.02),
    _nice_positive_scale(curve_max, 0.5),
  )


@lru_cache(maxsize=None)
def _font(size: int) -> ImageFont.ImageFont:
  try:
    return ImageFont.truetype("DejaVuSans.ttf", size=size)
  except OSError:
    return ImageFont.load_default()


def _heatmap_lut() -> np.ndarray:
  positions = np.arange(256, dtype=np.float64)
  knots = np.asarray([0, 48, 112, 176, 224, 255], dtype=np.float64)
  red = np.interp(positions, knots, [5, 40, 150, 235, 255, 255])
  green = np.interp(positions, knots, [4, 8, 15, 45, 145, 245])
  blue = np.interp(positions, knots, [12, 75, 115, 75, 20, 220])
  return np.column_stack((red, green, blue)).astype(np.uint8)


_HEATMAP_LUT = _heatmap_lut()


def _paste_fit(
  target: Image.Image, pixels: np.ndarray, box: tuple[int, int, int, int]
) -> None:
  source = Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="RGB")
  left, top, right, bottom = box
  scale = min((right - left) / source.width, (bottom - top) / source.height)
  size = (
    max(1, int(round(source.width * scale))),
    max(1, int(round(source.height * scale))),
  )
  resized = source.resize(size, Image.Resampling.BILINEAR)
  target.paste(
    resized,
    (left + (right - left - size[0]) // 2, top + (bottom - top - size[1]) // 2),
  )


def _draw_camera(
  image: Image.Image,
  draw: ImageDraw.ImageDraw,
  pixels: np.ndarray,
  box: tuple[int, int, int, int],
  label: str,
) -> None:
  _paste_fit(image, pixels, box)
  draw.rectangle(box, outline=(78, 82, 94), width=2)
  label_width = min(box[2] - box[0], max(120, 12 * len(label)))
  draw.rectangle((box[0], box[1], box[0] + label_width, box[1] + 28), fill=(0, 0, 0))
  draw.text((box[0] + 7, box[1] + 4), label, fill=(245, 245, 248), font=_font(18))


def _draw_heatmap(
  image: Image.Image,
  draw: ImageDraw.ImageDraw,
  values: np.ndarray,
  box: tuple[int, int, int, int],
  label: str,
  maximum: float,
) -> None:
  left, top, right, bottom = box
  draw.rectangle(box, outline=(58, 61, 72), width=1)
  draw.text((left + 3, top + 3), label, fill=(225, 226, 232), font=_font(12))
  heat_top = top + 20
  available_width = max(1, right - left - 6)
  available_height = max(1, bottom - heat_top - 3)
  heat_width = min(available_width, int(available_height * 5 / 7))
  heat_height = min(available_height, int(heat_width * 7 / 5))
  normalized = np.clip(np.asarray(values, dtype=np.float64) / maximum, 0.0, 1.0)
  heatmap = Image.fromarray(
    _HEATMAP_LUT[np.rint(normalized * 255).astype(np.uint8)][::-1], mode="RGB"
  ).resize((max(1, heat_width), max(1, heat_height)), Image.Resampling.NEAREST)
  image.paste(
    heatmap,
    (
      left + (right - left - heatmap.width) // 2,
      heat_top + (bottom - heat_top - heatmap.height) // 2,
    ),
  )


def _draw_curve(
  draw: ImageDraw.ImageDraw,
  history: np.ndarray,
  finger_index: int,
  box: tuple[int, int, int, int],
  title: str,
  maximum: float,
  channel_names: tuple[str, str, str],
) -> None:
  left, top, right, bottom = box
  draw.rectangle(box, outline=(58, 61, 72), width=1)
  draw.text((left + 5, top + 3), title, fill=(230, 231, 236), font=_font(13))
  plot = (left + 34, top + 22, right - 7, bottom - 19)
  draw.rectangle(plot, outline=(88, 91, 102), width=1)
  zero_y = (plot[1] + plot[3]) // 2
  draw.line((plot[0], zero_y, plot[2], zero_y), fill=(92, 95, 105), width=1)
  count = len(history)
  saturated = False
  if count:
    x_values = np.linspace(plot[0], plot[2], count)
    for channel, color in enumerate(FORCE_CHANNEL_COLORS):
      values = history[:, finger_index, channel]
      saturated |= bool(np.any(np.abs(values) > maximum))
      clipped = np.clip(values, -maximum, maximum)
      y_values = zero_y - clipped / maximum * ((plot[3] - plot[1]) / 2)
      points = [
        (int(round(x)), int(round(y)))
        for x, y in zip(x_values, y_values, strict=True)
      ]
      if len(points) == 1:
        x, y = points[0]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
      else:
        draw.line(points, fill=color, width=2)
  legend_x = plot[0]
  for name, color in zip(channel_names, FORCE_CHANNEL_COLORS, strict=True):
    draw.line((legend_x, bottom - 10, legend_x + 12, bottom - 10), fill=color, width=2)
    draw.text((legend_x + 14, bottom - 16), name, fill=(190, 193, 202), font=_font(9))
    legend_x += max(50, (plot[2] - plot[0]) // 3)
  if saturated:
    draw.text((right - 31, top + 3), "SAT", fill=(255, 80, 80), font=_font(11))


def compose_multimodal_replay_frame(
  camera_frames: tuple[tuple[str, np.ndarray], ...],
  first_maps: np.ndarray,
  second_maps: np.ndarray,
  curve_history: np.ndarray,
  *,
  width: int,
  height: int,
  timestamp_s: float,
  phase: str,
  heading: str,
  first_label: str,
  second_label: str,
  first_unit: str,
  second_unit: str,
  curve_unit: str,
  curve_channels: tuple[str, str, str],
  first_maximum: float,
  second_maximum: float,
  curve_maximum: float,
) -> Image.Image:
  """Compose selected RGB, bilateral heatmaps and ten evolving curves."""

  if not 1 <= len(camera_frames) <= 4:
    raise ValueError("one to four camera frames are required")
  if first_maps.shape != (10, 7, 5) or second_maps.shape != (10, 7, 5):
    raise ValueError("tactile maps must have shape (10,7,5)")
  if curve_history.ndim != 3 or curve_history.shape[1:] != (10, 3):
    raise ValueError("curve history must have shape (T,10,3)")
  if not all(value > 0.0 for value in (first_maximum, second_maximum, curve_maximum)):
    raise ValueError("visualization scales must be positive")
  if not (
    np.isfinite(first_maps).all()
    and np.isfinite(second_maps).all()
    and np.isfinite(curve_history).all()
  ):
    raise ValueError("tactile visualization values must be finite")

  image = Image.new("RGB", (width, height), (14, 15, 19))
  draw = ImageDraw.Draw(image)
  draw.rectangle((0, 0, width, 70), fill=(27, 29, 35))
  draw.text(
    (12, 7),
    f"{heading} | t={timestamp_s:.3f}s | phase={phase}",
    fill=(246, 246, 249),
    font=_font(22),
  )
  draw.text(
    (12, 40),
    f"heatmaps: {first_label} [{first_unit}] / {second_label} [{second_unit}] | curves: {', '.join(curve_channels)} [{curve_unit}]",
    fill=(190, 193, 204),
    font=_font(15),
  )

  margin, top, bottom = 8, 78, height - 8
  left_end, middle_end = int(width * 0.34), int(width * 0.61)
  camera_gap = 6
  camera_height = (bottom - top - camera_gap * (len(camera_frames) - 1)) // len(camera_frames)
  for index, (name, pixels) in enumerate(camera_frames):
    camera_top = top + index * (camera_height + camera_gap)
    camera_bottom = bottom if index == len(camera_frames) - 1 else camera_top + camera_height
    _draw_camera(
      image,
      draw,
      pixels,
      (margin, camera_top, left_end - margin, camera_bottom),
      name.replace("_", " ").upper() + " / RECORDED",
    )

  tactile_left, tactile_right = left_end + margin, middle_end - margin
  tactile_row_height = (bottom - top) // 4
  for side_index, hand in enumerate(HAND_LABELS):
    offset = side_index * 5
    rows = (
      (first_maps[offset : offset + 5], first_maximum, first_label),
      (second_maps[offset : offset + 5], second_maximum, second_label),
    )
    for quantity_index, (maps, maximum, quantity) in enumerate(rows):
      row = side_index * 2 + quantity_index
      row_top = top + row * tactile_row_height
      row_bottom = bottom if row == 3 else row_top + tactile_row_height - 4
      cell_width = (tactile_right - tactile_left) // 5
      for finger_index, finger in enumerate(FINGER_LABELS):
        cell_left = tactile_left + finger_index * cell_width
        cell_right = tactile_right if finger_index == 4 else cell_left + cell_width - 2
        _draw_heatmap(
          image,
          draw,
          maps[finger_index],
          (cell_left, row_top, cell_right, row_bottom),
          f"{hand[0]} {quantity} {finger[:2]}",
          maximum,
        )

  curve_left, curve_right = middle_end + margin, width - margin
  curve_width = (curve_right - curve_left - 5) // 2
  curve_height = (bottom - top - 16) // 5
  plotted_history = curve_history
  if len(plotted_history) > curve_width:
    plotted_history = plotted_history[
      np.linspace(0, len(plotted_history) - 1, curve_width).astype(np.int64)
    ]
  for finger_index, finger in enumerate(FINGER_LABELS):
    row_top = top + finger_index * (curve_height + 4)
    for side_index, hand in enumerate(("L", "R")):
      box_left = curve_left + side_index * (curve_width + 5)
      _draw_curve(
        draw,
        plotted_history,
        side_index * 5 + finger_index,
        (box_left, row_top, box_left + curve_width, row_top + curve_height),
        f"{hand} {finger}",
        curve_maximum,
        curve_channels,
      )
  return image


def export_multimodal_replay(
  source_path: str | Path,
  output_dir: str | Path,
  *,
  cameras: tuple[str, ...] | None = None,
  fps: float = 10.0,
  width: int = 1920,
  height: int = 1080,
  maximum_tactile_age_s: float = 0.050,
) -> dict[str, Any]:
  """Export synchronized saved RGB, tactile heatmaps, and force curves."""

  source_path = Path(source_path).expanduser().resolve()
  output = Path(output_dir).expanduser().absolute()
  partial = output.with_name(output.name + ".partial")
  if source_path.suffix != ".h5" or not source_path.is_file():
    raise ValueError("replay source must be a completed .h5 episode")
  if output.exists() or output.is_symlink() or partial.exists() or partial.is_symlink():
    raise FileExistsError("replay output or partial directory already exists")
  if width < 960 or height < 540 or width % 2 or height % 2:
    raise ValueError("replay dimensions must be even and at least 960 by 540")
  if not math.isfinite(maximum_tactile_age_s) or not 0.0 <= maximum_tactile_age_s <= 1.0:
    raise ValueError("maximum tactile age must be between 0 and 1 second")

  from .task_video import _FfmpegPipeWriter, _find_ffmpeg_executable

  ffmpeg = _find_ffmpeg_executable()
  with h5py.File(source_path, "r") as file:
    metadata = _json_attr(file, "metadata_json")
    scene = metadata.get("scene", "pick-place")
    if scene not in SCENE_NAMES:
      raise ValueError(f"episode has unsupported scene {scene!r}")
    selected_cameras = resolve_replay_cameras(file, cameras)
    reference = file[f"cameras/{selected_cameras[0]}"]
    reference_times = _clock(
      reference["timestamp"][:], f"{selected_cameras[0]} camera", strict=True
    )
    if fps > float(file.attrs.get("camera_hz", 0.0)) + 1.0e-10:
      raise ValueError("replay fps cannot exceed the saved camera rate")
    reference_indices = select_replay_frames(reference_times, fps)
    frame_times = reference_times[reference_indices]
    pose_times = (
      _clock(reference["pose_timestamp"][:], "reference camera pose", strict=True)[
        reference_indices
      ]
      if "pose_timestamp" in reference
      else frame_times
    )
    camera_indices: dict[str, np.ndarray] = {}
    camera_times: dict[str, np.ndarray] = {}
    maximum_camera_skew = 0.0
    for name in selected_cameras:
      group = file[f"cameras/{name}"]
      times = _clock(group["timestamp"][:], f"{name} camera", strict=True)
      if group["rgb"].shape[0] != len(times):
        raise ValueError(f"{name} RGB and timestamp counts differ")
      indices = _nearest_indices(times, frame_times)
      skew = np.abs(times[indices] - frame_times)
      maximum_camera_skew = max(maximum_camera_skew, float(np.max(skew)))
      if np.any(skew > 0.5 / float(file.attrs.get("camera_hz", 1.0)) + 1.0e-8):
        raise ValueError(f"{name} has no RGB frame close to the replay clock")
      camera_indices[name] = indices
      camera_times[name] = times

    tactile = resolve_tactile_source(file)
    tactile_indices = _causal_indices(tactile.times, pose_times, "tactile")
    tactile_age = pose_times - tactile.times[tactile_indices]
    if np.any(tactile_age < -1.0e-10) or np.any(tactile_age > maximum_tactile_age_s):
      raise ValueError(
        f"tactile/RGB alignment exceeds {maximum_tactile_age_s * 1000:g} ms"
      )
    scales = tactile_scales(file, tactile, tactile_indices)
    state_times = _clock(file["state/timestamp"][:], "state/timestamp", strict=True)
    state_indices = _causal_indices(state_times, pose_times, "state")
    phases = file.get("commands/phase")
    if phases is None or len(phases) != len(state_times):
      raise ValueError("commands/phase must align with state/timestamp")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    writer = None
    last_image = None
    history: list[np.ndarray] = []
    try:
      writer = _FfmpegPipeWriter(
        partial / "review.mp4",
        fps=fps,
        width=width,
        height=height,
        executable=ffmpeg,
      )
      with (partial / "frames.jsonl").open("x", encoding="utf-8") as log:
        for output_index in range(len(reference_indices)):
          tactile_index = int(tactile_indices[output_index])
          first, second, curves = read_tactile_sample(file, tactile, tactile_index)
          history.append(curves)
          state_index = int(state_indices[output_index])
          phase = _text(phases[state_index])
          frames = tuple(
            (
              name,
              np.asarray(
                file[f"cameras/{name}/rgb"][camera_indices[name][output_index]]
              ),
            )
            for name in selected_cameras
          )
          image = compose_multimodal_replay_frame(
            frames,
            first,
            second,
            np.asarray(history),
            width=width,
            height=height,
            timestamp_s=float(pose_times[output_index]),
            phase=phase,
            heading=f"{str(scene).upper()} MULTIMODAL REPLAY",
            first_label=tactile.first_label,
            second_label=tactile.second_label,
            first_unit=tactile.first_unit,
            second_unit=tactile.second_unit,
            curve_unit=tactile.curve_unit,
            curve_channels=tactile.curve_channels,
            first_maximum=scales[0],
            second_maximum=scales[1],
            curve_maximum=scales[2],
          )
          writer.write(np.asarray(image))
          if output_index == 0:
            image.save(partial / "first_frame.png")
          last_image = image
          log.write(
            json.dumps(
              {
                "output_index": output_index,
                "playback_timestamp_s": output_index / fps,
                "camera_pose_timestamp_s": float(pose_times[output_index]),
                "camera_indices": {
                  name: int(camera_indices[name][output_index])
                  for name in selected_cameras
                },
                "camera_timestamps_s": {
                  name: float(camera_times[name][camera_indices[name][output_index]])
                  for name in selected_cameras
                },
                "state_index": state_index,
                "state_timestamp_s": float(state_times[state_index]),
                "tactile_index": tactile_index,
                "tactile_timestamp_s": float(tactile.times[tactile_index]),
                "tactile_age_s": float(tactile_age[output_index]),
                "phase": phase,
                "fingertip_order": list(FINGERTIP_LINK_NAMES),
                "curve_values": curves.tolist(),
              },
              ensure_ascii=False,
            )
            + "\n"
          )
      writer.finish()
      if last_image is None:
        raise ValueError("replay frame plan is empty")
      last_image.save(partial / "last_frame.png")
      report = {
        "schema_version": SCHEMA_VERSION,
        "completed": True,
        "task": scene,
        "source_hdf5": str(source_path),
        "source_sha256": _sha256(source_path),
        "source_episode_index": metadata.get("episode_index"),
        "camera_names": list(selected_cameras),
        "reference_camera": selected_cameras[0],
        "maximum_camera_skew_s": maximum_camera_skew,
        "tactile_source_kind": tactile.kind,
        "tactile_group": tactile.group_name,
        "measured_fingertips": [
          FINGERTIP_LINK_NAMES[int(index)] for index in tactile.destinations
        ],
        "tactile_heatmaps": [
          {"label": tactile.first_label, "unit": tactile.first_unit, "maximum": scales[0]},
          {"label": tactile.second_label, "unit": tactile.second_unit, "maximum": scales[1]},
        ],
        "curve_channels": list(tactile.curve_channels),
        "curve_unit": tactile.curve_unit,
        "curve_absolute_maximum": scales[2],
        "fingertip_order": list(FINGERTIP_LINK_NAMES),
        "fps": fps,
        "output_size": [width, height],
        "output_frame_count": len(reference_indices),
        "source_reference_frame_count": len(reference_times),
        "first_camera_pose_timestamp_s": float(pose_times[0]),
        "last_camera_pose_timestamp_s": float(pose_times[-1]),
        "maximum_tactile_age_s": float(np.max(tactile_age)),
        "video_duration_s": len(reference_indices) / fps,
        "source_modified": False,
        "physics_rerun": False,
      }
      (partial / "replay.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
      )
      if output.exists() or output.is_symlink():
        raise FileExistsError("replay output appeared while exporting")
      partial.rename(output)
      return report
    except BaseException as error:
      if writer is not None:
        try:
          writer.abort()
        except BaseException as abort_error:
          error.add_note(f"replay encoder cleanup failed: {abort_error}")
      try:
        (partial / "failure.json").write_text(
          json.dumps(
            {
              "completed": False,
              "source_hdf5": str(source_path),
              "error_type": type(error).__name__,
              "error": str(error),
            },
            indent=2,
          )
          + "\n",
          encoding="utf-8",
        )
      except BaseException as report_error:
        error.add_note(f"failure report could not be written: {report_error}")
      raise


__all__ = [
  "SCHEMA_VERSION",
  "TactileReplaySource",
  "compose_multimodal_replay_frame",
  "export_multimodal_replay",
  "read_tactile_sample",
  "resolve_replay_cameras",
  "resolve_tactile_source",
  "select_replay_frames",
  "tactile_scales",
]
