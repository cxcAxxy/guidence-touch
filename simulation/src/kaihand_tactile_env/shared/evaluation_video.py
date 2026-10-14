"""Task-neutral, read-only policy evaluation video with bilateral tactile force."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import FINGERTIP_LINK_NAMES
from .contact_tactile import SolverDistributedTactileProvider
from .render_backend import current_backend
from .task_video import _FfmpegPipeWriter

SCHEMA_VERSION = "kaihand-policy-evaluation-review-v3"
FINGER_LABELS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
HAND_LABELS = ("LEFT", "RIGHT")
CHANNEL_NAMES = ("Ft_col", "Ft_row", "Fn")
CHANNEL_COLORS = ((52, 174, 255), (255, 174, 52), (86, 220, 116))
NORMAL_TAXEL_MAX_N = 0.10
TANGENT_TAXEL_MAX_N = 0.02
CURVE_ABS_MAX_N_PER_TAXEL = 0.12
_EVALUATION_ARTIFACT_NAMES = (
  "review.mp4",
  "review.json",
  "frames.jsonl",
  "first_frame.png",
  "last_frame.png",
  "evaluation_rollout_trace.npz",
  "evaluation_comparison.json",
  "evaluation_right_wrist_state.png",
  "evaluation_right_hand_actuated_dof.png",
  "evaluation_right_fingertip_tactile.png",
  "evaluation_evaluation_plots.json",
)


def evaluation_artifact_directory(trial: Path) -> Path:
  """Find flat artifacts, falling back to the untouched legacy review directory.

  Prefer even partially written flat artifacts so that an old nested review
  cannot make an incomplete new recording look valid.
  """
  if any(
    (trial / name).exists() or (trial / name).is_symlink()
    for name in _EVALUATION_ARTIFACT_NAMES
  ):
    return trial
  legacy = trial / "review"
  return legacy if legacy.is_dir() else trial


def _prepare_evaluation_artifact_directory(output: Path) -> None:
  """Share the trial directory with policy logs without replacing artifacts."""
  output.mkdir(parents=True, exist_ok=True)
  for name in _EVALUATION_ARTIFACT_NAMES:
    artifact = output / name
    if artifact.exists() or artifact.is_symlink():
      raise FileExistsError(f"evaluation artifact already exists: {artifact}")


class EvaluationMetrics(Protocol):
  """Optional task-specific metrics; update it at the physics-step rate."""

  name: str

  def snapshot(self, simulation: Any) -> Mapping[str, Any]: ...

  def metadata(self) -> Mapping[str, Any]: ...


@lru_cache(maxsize=None)
def _font(size: int) -> ImageFont.ImageFont:
  try:
    return ImageFont.truetype("DejaVuSans.ttf", size=size)
  except OSError:
    return ImageFont.load_default()


def _json_safe(value: Any) -> Any:
  if isinstance(value, np.ndarray):
    return _json_safe(value.tolist())
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, Mapping):
    return {str(key): _json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(item) for item in value]
  if value is None or isinstance(value, (str, int, float, bool)):
    return value
  return str(value)


def _heatmap_lut() -> np.ndarray:
  positions = np.arange(256, dtype=np.float64)
  knots = np.asarray([0, 48, 112, 176, 224, 255], dtype=np.float64)
  red = np.interp(positions, knots, [5, 40, 150, 235, 255, 255])
  green = np.interp(positions, knots, [4, 8, 15, 45, 145, 245])
  blue = np.interp(positions, knots, [12, 75, 115, 75, 20, 220])
  return np.column_stack((red, green, blue)).astype(np.uint8)


_HEATMAP_LUT = _heatmap_lut()


def _paste_fit(
  target: Image.Image,
  pixels: np.ndarray,
  box: tuple[int, int, int, int],
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
  label_font = _font(18)
  label_width = draw.textbbox((0, 0), label, font=label_font)[2] + 14
  draw.rectangle(
    (box[0], box[1], min(box[2], box[0] + label_width), box[1] + 28),
    fill=(0, 0, 0),
  )
  draw.text((box[0] + 7, box[1] + 4), label, fill=(245, 245, 248), font=label_font)


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
  draw.text((left + 4, top + 3), label, fill=(225, 226, 232), font=_font(14))
  heat_top = top + 22
  available_width = max(1, right - left - 8)
  available_height = max(1, bottom - heat_top - 4)
  heat_width = min(available_width, int(available_height * 5 / 7))
  heat_height = min(available_height, int(heat_width * 7 / 5))
  normalized = np.clip(np.asarray(values, dtype=np.float64) / maximum, 0.0, 1.0)
  indices = np.rint(normalized * 255).astype(np.uint8)
  heatmap = Image.fromarray(_HEATMAP_LUT[np.flipud(indices)], mode="RGB").resize(
    (max(1, heat_width), max(1, heat_height)), Image.Resampling.NEAREST
  )
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
  channel_names: tuple[str, str, str] = CHANNEL_NAMES,
) -> None:
  left, top, right, bottom = box
  draw.rectangle(box, outline=(58, 61, 72), width=1)
  draw.text((left + 5, top + 3), title, fill=(230, 231, 236), font=_font(14))
  plot = (left + 34, top + 23, right - 7, bottom - 20)
  draw.rectangle(plot, outline=(88, 91, 102), width=1)
  zero_y = (plot[1] + plot[3]) // 2
  draw.line((plot[0], zero_y, plot[2], zero_y), fill=(92, 95, 105), width=1)
  draw.text((left + 3, plot[1] - 5), f"+{maximum:g}", fill=(145, 148, 158), font=_font(10))
  draw.text((left + 7, zero_y - 5), "0", fill=(145, 148, 158), font=_font(10))
  draw.text((left + 3, plot[3] - 7), f"-{maximum:g}", fill=(145, 148, 158), font=_font(10))
  count = len(history)
  saturated = False
  if count:
    x_values = np.linspace(plot[0], plot[2], count)
    for channel, color in enumerate(CHANNEL_COLORS):
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
  for name, color in zip(channel_names, CHANNEL_COLORS, strict=True):
    draw.line((legend_x, bottom - 11, legend_x + 13, bottom - 11), fill=color, width=2)
    draw.text((legend_x + 16, bottom - 17), name, fill=(190, 193, 202), font=_font(10))
    legend_x += (plot[2] - plot[0]) // 3
  if saturated:
    draw.text((right - 32, top + 3), "SAT", fill=(255, 80, 80), font=_font(12))


def _metrics_text(metrics: Mapping[str, Any]) -> str:
  if not metrics:
    return ""
  labels = {
    "card_displacement_now_mm": "card now",
    "card_displacement_peak_mm": "peak",
    "edge_threshold_mm": "edge",
    "success_stage": "stage",
    "failure_stage": "failure",
    "insertion_depth_mm": "depth",
    "peak_insertion_depth_mm": "peak depth",
    "lateral_error_mm": "lateral",
    "orientation_error_deg": "angle",
    "axial_resistance_n": "axial N",
    "socket_normal_load_n": "socket N",
    "socket_penetration_mm": "penetration",
  }
  parts = []
  for key, value in metrics.items():
    if isinstance(value, bool):
      parts.append(f"{labels.get(key, key)}={'YES' if value else 'NO'}")
    elif isinstance(value, (int, float)):
      unit = " mm" if key.endswith("_mm") else ""
      parts.append(f"{labels.get(key, key)}={value:.1f}{unit}")
    elif isinstance(value, str):
      parts.append(f"{labels.get(key, key)}={value}")
  return " | ".join(parts)


def compose_evaluation_frame(
  head_rgb: np.ndarray,
  second_rgb: np.ndarray,
  normal_taxel_force_n: np.ndarray,
  tangent_taxel_force_n: np.ndarray,
  force_history: np.ndarray,
  *,
  width: int = 1920,
  height: int = 1080,
  simulation_time_s: float,
  phase: str,
  second_camera_label: str,
  model_wrist_rgb: np.ndarray | None = None,
  review_wrist_rgb: np.ndarray | None = None,
  head_is_model_input: bool = True,
  metrics: Mapping[str, Any] | None = None,
  heading: str = "MODEL EVALUATION",
) -> Image.Image:
  """Compose two/three cameras and enlarged bilateral force maps.

  ``force_history`` remains part of the public call signature and is validated
  for compatibility with existing recorders, but it is no longer rendered in
  the evaluation video.
  """
  normal = np.asarray(normal_taxel_force_n, dtype=np.float64)
  tangent = np.asarray(tangent_taxel_force_n, dtype=np.float64)
  history = np.asarray(force_history, dtype=np.float64)
  if normal.shape != (10, 7, 5):
    raise ValueError(f"normal force must have shape (10, 7, 5), got {normal.shape}")
  if tangent.shape != (10, 7, 5, 2):
    raise ValueError(f"tangent force must have shape (10, 7, 5, 2), got {tangent.shape}")
  if history.ndim != 3 or history.shape[1:] != (10, 3):
    raise ValueError(f"force history must have shape (T, 10, 3), got {history.shape}")
  if not np.isfinite(normal).all() or not np.isfinite(tangent).all() or not np.isfinite(history).all():
    raise ValueError("force data must be finite")
  if np.any(normal < -1.0e-12):
    raise ValueError("normal force must be nonnegative")
  if model_wrist_rgb is not None and review_wrist_rgb is not None:
    raise ValueError("right wrist cannot be both a model and review-only panel")

  image = Image.new("RGB", (width, height), (14, 15, 19))
  draw = ImageDraw.Draw(image)
  draw.rectangle((0, 0, width, 68), fill=(27, 29, 35))
  draw.text(
    (12, 7),
    f"{heading} | t={simulation_time_s:.3f}s | phase={phase}",
    fill=(246, 246, 249),
    font=_font(22),
  )
  detail = "10 fingertips | heatmaps: bilateral Fn / |Ft| [N per taxel]"
  metric_text = _metrics_text(metrics or {})
  if metric_text:
    detail += " | " + metric_text
  draw.text((12, 38), detail, fill=(190, 193, 204), font=_font(16))

  margin = 8
  top, bottom = 76, height - margin
  left_end = int(width * 0.45)
  camera_gap = 8
  cameras = [(head_rgb, "HEAD / MODEL VIEW" if head_is_model_input else "HEAD / REVIEW ONLY")]
  if model_wrist_rgb is not None:
    cameras.append((model_wrist_rgb, "RIGHT WRIST / MODEL VIEW"))
  elif review_wrist_rgb is not None:
    cameras.append((review_wrist_rgb, "RIGHT WRIST / REVIEW ONLY"))
  cameras.append((second_rgb, second_camera_label))
  camera_height = (bottom - top - camera_gap * (len(cameras) - 1)) // len(cameras)
  for index, (pixels, label) in enumerate(cameras):
    camera_top = top + index * (camera_height + camera_gap)
    camera_bottom = bottom if index == len(cameras) - 1 else camera_top + camera_height
    _draw_camera(
      image,
      draw,
      pixels,
      (margin, camera_top, left_end - margin, camera_bottom),
      label,
    )

  tactile_left, tactile_right = left_end + margin, width - margin
  tactile_height = bottom - top
  tactile_row_height = tactile_height // 4
  tangent_norm = np.linalg.norm(tangent, axis=-1)
  for side_index, hand in enumerate(HAND_LABELS):
    offset = side_index * 5
    rows = (
      (normal[offset : offset + 5], NORMAL_TAXEL_MAX_N, f"{hand} Fn"),
      (tangent_norm[offset : offset + 5], TANGENT_TAXEL_MAX_N, f"{hand} |Ft|"),
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
          f"{quantity} {finger[:2]}",
          maximum,
        )

  return image


def _create_evaluation_spatial_provider(simulation: Any) -> Any:
  return SolverDistributedTactileProvider(
    simulation.model,
    getattr(simulation, "genesis_probe_layout", None),
    link_names=FINGERTIP_LINK_NAMES,
  )


class EvaluationVideo:
  """Read-only, task-neutral evaluation artifact recorder."""

  def __init__(
    self,
    simulation: Any,
    output: Path,
    *,
    fps: int = 10,
    width: int = 1920,
    height: int = 1080,
    render_width: int = 640,
    render_height: int = 480,
    second_camera: str = "global",
    include_model_wrist: bool = False,
    include_review_wrist: bool = False,
    comparison: bool = False,
    reference_dataset: Path | None = None,
    reference_episode_index: int = 0,
    tactile_provider: Any | None = None,
    metrics: EvaluationMetrics | None = None,
    heading: str = "MODEL EVALUATION",
    metadata: Mapping[str, Any] | None = None,
  ) -> None:
    if fps not in (5, 10):
      raise ValueError("evaluation video fps must be 5 or 10")
    if min(width, height, render_width, render_height) <= 0:
      raise ValueError("video and render dimensions must be positive")
    if width % 2 or height % 2:
      raise ValueError("video dimensions must be even for yuv420p")
    if second_camera not in ("global", "right_wrist", "overhead"):
      raise ValueError("second camera must be global, right_wrist or overhead")
    if include_model_wrist and second_camera == "right_wrist":
      raise ValueError("model wrist view would duplicate the second camera")
    if reference_episode_index < 0:
      raise ValueError("reference episode index must be nonnegative")
    _prepare_evaluation_artifact_directory(output)
    self.simulation = simulation
    self.output = output
    self.fps = fps
    self.width, self.height = width, height
    self.stride = 30 // fps
    self.second_camera_name = second_camera
    self.include_model_wrist = bool(include_model_wrist)
    self.include_review_wrist = bool(include_review_wrist)
    self.show_dedicated_wrist = (
      second_camera != "right_wrist"
      and (self.include_model_wrist or self.include_review_wrist)
    )
    self.metrics = metrics
    self.heading = heading
    self.count = 0
    self.first_time = self.last_time = None
    self.closed = False
    self.last_frame: Image.Image | None = None
    self.force_history: list[np.ndarray] = []
    self.renderer = self.writer = self.log = None
    self.provider = (
      tactile_provider
      if tactile_provider is not None
      else getattr(simulation, "evaluation_tactile_provider", None)
    )
    if self.provider is None:
      self.provider = _create_evaluation_spatial_provider(simulation)
    provider_links = tuple(self.provider.link_names)
    if set(provider_links) != set(FINGERTIP_LINK_NAMES) or len(provider_links) != 10:
      raise RuntimeError("evaluation tactile provider must identify all ten fingertips")
    self._source_indices = np.asarray(
      [provider_links.index(name) for name in FINGERTIP_LINK_NAMES], dtype=np.intp
    )
    supplied_metadata = _json_safe(dict(metadata or {}))
    observation_contract = supplied_metadata.get("observation_contract")
    server_metadata = supplied_metadata.get("server_metadata")
    if not isinstance(observation_contract, Mapping) and isinstance(server_metadata, Mapping):
      observation_contract = server_metadata.get("observation_contract")
    declared_cameras = (
      observation_contract.get("cameras")
      if isinstance(observation_contract, Mapping)
      else None
    )
    if isinstance(server_metadata, Mapping) and server_metadata.get("cameras") is not None:
      declared_cameras = server_metadata["cameras"]
    contract_cameras = (
      tuple(declared_cameras)
      if isinstance(declared_cameras, (list, tuple))
      else None
    )
    self.head_is_model_input = contract_cameras is None or "head" in contract_cameras
    self.wrist_is_model_input = (
      self.include_model_wrist
      or (contract_cameras is not None and "right_wrist" in contract_cameras)
    )
    wrist_displayed = self.show_dedicated_wrist or second_camera == "right_wrist"
    model_cameras_displayed = []
    if self.head_is_model_input:
      model_cameras_displayed.append("head")
    if wrist_displayed and self.wrist_is_model_input:
      model_cameras_displayed.append("right_wrist")
    review_only_cameras_displayed = []
    if not self.head_is_model_input:
      review_only_cameras_displayed.append("head")
    if second_camera != "right_wrist" or not self.wrist_is_model_input:
      review_only_cameras_displayed.append(second_camera)
    if self.show_dedicated_wrist and not self.wrist_is_model_input:
      review_only_cameras_displayed.append("right_wrist")
    reference_value = (
      reference_dataset
      if reference_dataset is not None
      else supplied_metadata.get("reference_dataset")
    )
    self.reference_dataset = Path(reference_value) if reference_value else None
    self.reference_episode_index = int(reference_episode_index)
    self.comparison_enabled = bool(comparison)
    self.comparison_trace = None
    self.comparison_error: str | None = None
    self._comparison_capture_enabled = False
    if self.comparison_enabled:
      try:
        from .evaluation_comparison import EvaluationComparisonTrace

        self.comparison_trace = EvaluationComparisonTrace(tactile_provider=self.provider)
        self._comparison_capture_enabled = True
      except Exception as exc:
        self.comparison_error = f"comparison recorder unavailable: {type(exc).__name__}: {exc}"
    self.camera = None
    if second_camera == "global":
      self.camera = mujoco.MjvCamera()
      self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
      self.camera.lookat[:] = [0.45, 0.0, 0.85]
      self.camera.distance = 1.65
      self.camera.azimuth = 135
      self.camera.elevation = -25
    self.metadata = {
      "task": getattr(simulation, "scene", "unknown"),
      "fps": fps,
      "output_size": [width, height],
      "review_render_size": [render_width, render_height],
      "layout": (
        "enlarged head/right-wrist/global-or-named RGB; enlarged bilateral Fn/|Ft| maps"
        if self.show_dedicated_wrist
        else "enlarged head/global-or-named RGB; enlarged bilateral Fn/|Ft| maps"
      ),
      "second_camera": second_camera,
      "head_camera_modified": False,
      "render_shadows": False,
      "physics_modified": False,
      "tactile_source": self.provider.source,
      "tactile_semantics": self.provider.taxel_force_semantics,
      "fingertip_order": list(FINGERTIP_LINK_NAMES),
      "force_mean_channel_order": list(CHANNEL_NAMES),
      "force_mean_channel_aliases": {
        "Fx(sensor_chart)": "Ft_col",
        "Fy(sensor_chart)": "Ft_row",
        "Fz(sensor_chart)": "Fn",
      },
      "force_mean_value": "arithmetic mean across each fingertip's 7x5 taxels",
      "force_mean_storage": (
        "frames.jsonl.force_mean_n_per_taxel; not rendered in review.mp4"
      ),
      "heatmap_scale": {
        "Fn_n_per_taxel": [0.0, NORMAL_TAXEL_MAX_N],
        "Ft_magnitude_n_per_taxel": [0.0, TANGENT_TAXEL_MAX_N],
        "fixed": True,
      },
      "tangent_display": (
        "heatmap is per-taxel |Ft|; signed components remain in frames.jsonl "
        "and are not rendered in review.mp4"
      ),
      "timing": "same cached FK and solver state for both cameras and tactile; recorder never steps/forwards physics",
      "video_clock": "simulation time; final off-grid frame allowed",
      "task_metrics": None if metrics is None else _json_safe(metrics.metadata()),
      **supplied_metadata,
      "schema": SCHEMA_VERSION,
      "time_series_displayed": False,
      "model_input_cameras_displayed": model_cameras_displayed,
      "review_only_cameras_displayed": review_only_cameras_displayed,
      "right_wrist_camera_role": (
        "model_input" if self.wrist_is_model_input else "review_only"
      ) if wrist_displayed else None,
      "comparison_requested": self.comparison_enabled,
      "reference_dataset": (
        str(self.reference_dataset) if self.reference_dataset is not None else None
      ),
      "reference_episode_index": self.reference_episode_index if self.comparison_enabled else None,
    }
    try:
      self.renderer = mujoco.Renderer(
        simulation.model, height=render_height, width=render_width
      )
      self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
      self.metadata["render_backend"] = current_backend()
      self.writer = _FfmpegPipeWriter(
        output / "review.mp4", fps=fps, width=width, height=height
      )
      self.log = (output / "frames.jsonl").open("x", encoding="utf-8")
    except BaseException:
      if self.writer is not None:
        abort = getattr(self.writer, "abort", None)
        if abort is not None:
          abort()
      if self.renderer is not None:
        self.renderer.close()
      raise

  def _sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample = self.provider.read(self.simulation.data)
    links = tuple(sample.link_names)
    if len(links) != len(set(links)) or set(FINGERTIP_LINK_NAMES) - set(links):
      raise RuntimeError("tactile sample must contain ten distinct canonical fingertips")
    indices = np.asarray([links.index(name) for name in FINGERTIP_LINK_NAMES], dtype=np.intp)
    normal = np.asarray(sample.normal_taxel_force_n, dtype=np.float64)
    tangent = np.asarray(sample.tangent_taxel_force_n, dtype=np.float64)
    if normal.shape != (len(links), 7, 5):
      raise RuntimeError(f"normal tactile shape must be ({len(links)}, 7, 5), got {normal.shape}")
    if tangent.shape != (len(links), 7, 5, 2):
      raise RuntimeError(f"tangent tactile shape must be ({len(links)}, 7, 5, 2), got {tangent.shape}")
    normal, tangent = normal[indices], tangent[indices]
    if not np.isfinite(normal).all() or not np.isfinite(tangent).all():
      raise RuntimeError("tactile sample contains non-finite force")
    if np.any(normal < -1.0e-12):
      raise RuntimeError("normal tactile force must be nonnegative")
    means = np.column_stack(
      (
        tangent[..., 0].mean(axis=(1, 2)),
        tangent[..., 1].mean(axis=(1, 2)),
        normal.mean(axis=(1, 2)),
      )
    )
    return np.maximum(normal, 0.0), tangent, means

  def _capture_comparison(
    self,
    normal: np.ndarray | None = None,
    tangent: np.ndarray | None = None,
  ) -> None:
    if not self._comparison_capture_enabled or self.comparison_trace is None:
      return
    try:
      if normal is None or tangent is None:
        self.comparison_trace.capture(self.simulation)
      else:
        self.comparison_trace.capture(
          self.simulation,
          normal_taxel_force_n=normal,
          tangent_taxel_force_n=tangent,
        )
    except Exception as exc:
      self.comparison_error = f"comparison capture failed: {type(exc).__name__}: {exc}"
      self._comparison_capture_enabled = False

  def capture(self, tick: int, phase: str, *, force: bool = False) -> None:
    if self.closed:
      raise RuntimeError("video already closed")
    timestamp = float(
      getattr(self.simulation, "observation_time", self.simulation.data.time)
    )
    video_due = (force or tick % self.stride == 0) and (
      self.last_time is None or timestamp > self.last_time + 1.0e-10
    )
    if not video_due:
      self._capture_comparison()
      return
    normal, tangent, means = self._sample()
    self._capture_comparison(normal, tangent)
    self.force_history.append(means.copy())
    assert self.renderer is not None and self.writer is not None and self.log is not None
    self.renderer.update_scene(self.simulation.data, camera="head")
    head = self.renderer.render().copy()
    second_camera = self.camera if self.camera is not None else self.second_camera_name
    self.renderer.update_scene(self.simulation.data, camera=second_camera)
    second = self.renderer.render().copy()
    wrist = None
    if self.show_dedicated_wrist:
      self.renderer.update_scene(self.simulation.data, camera="right_wrist")
      wrist = self.renderer.render().copy()
    second_camera_label = self.second_camera_name.replace("_", " ").upper()
    if self.second_camera_name == "right_wrist":
      second_camera_label += (
        " / MODEL VIEW" if self.wrist_is_model_input else " / REVIEW ONLY"
      )
    task_metrics = {} if self.metrics is None else dict(self.metrics.snapshot(self.simulation))
    if self.first_time is None:
      self.first_time = timestamp
    composite = compose_evaluation_frame(
      head,
      second,
      normal,
      tangent,
      np.asarray(self.force_history),
      width=self.width,
      height=self.height,
      simulation_time_s=float(self.simulation.data.time),
      phase=phase,
      second_camera_label=second_camera_label,
      model_wrist_rgb=wrist if self.wrist_is_model_input else None,
      review_wrist_rgb=wrist if not self.wrist_is_model_input else None,
      head_is_model_input=self.head_is_model_input,
      metrics=task_metrics,
      heading=self.heading,
    )
    self.writer.write(np.asarray(composite))
    row = {
      "frame": self.count,
      "control_tick": int(tick),
      "simulation_time_s": float(self.simulation.data.time),
      "camera_pose_time_s": timestamp,
      "tactile_time_s": timestamp,
      "phase": phase,
      "fingertip_order": list(FINGERTIP_LINK_NAMES),
      "force_mean_n_per_taxel": means.tolist(),
      "normal_taxel_force_n": normal.tolist(),
      "tangent_taxel_force_n": tangent.tolist(),
      "task_metrics": _json_safe(task_metrics),
    }
    self.log.write(json.dumps(row, ensure_ascii=False) + "\n")
    self.log.flush()
    if self.count == 0:
      composite.save(self.output / "first_frame.png")
    self.last_frame = composite
    self.last_time = timestamp
    self.count += 1

  def capture_due(self, phase: str, *, force: bool = False) -> None:
    """Capture from an arbitrary-rate task callback at the configured FPS."""
    timestamp = float(
      getattr(self.simulation, "observation_time", self.simulation.data.time)
    )
    if (
      not force
      and self.last_time is not None
      and timestamp < self.last_time + 1.0 / self.fps - 1.0e-10
    ):
      self._capture_comparison()
      return
    self.capture(self.count * self.stride, phase, force=True)

  def finish(self, *, status: str, evaluation: Any, error: str | None = None) -> dict[str, Any]:
    if self.closed:
      return self.metadata
    self.closed = True
    completed = False
    try:
      assert self.writer is not None
      self.writer.finish()
      if self.last_frame is not None:
        self.last_frame.save(self.output / "last_frame.png")
      completed = self.count > 0
    finally:
      if self.log is not None:
        self.log.close()
      if self.renderer is not None:
        self.renderer.close()
      comparison_summary = None
      if self.comparison_enabled:
        if self.comparison_trace is None:
          comparison_summary = {
            "status": "unavailable",
            "reason": self.comparison_error,
          }
        else:
          try:
            comparison_summary = self.comparison_trace.finish(
              self.output,
              reference_dataset=self.reference_dataset,
              reference_episode_index=self.reference_episode_index,
            )
          except Exception as exc:
            comparison_summary = {
              "status": "error",
              "reason": f"comparison export failed: {type(exc).__name__}: {exc}",
            }
          if self.comparison_error is not None:
            comparison_summary = {
              **comparison_summary,
              "status": "capture_error",
              "reason": self.comparison_error,
            }
      self.metadata.update(
        completed=completed,
        task_status=status,
        evaluation=_json_safe(evaluation),
        task_error=error,
        frame_count=self.count,
        first_pose_time_s=self.first_time,
        last_pose_time_s=self.last_time,
        playback_duration_s=self.count / self.fps,
      )
      if comparison_summary is not None:
        self.metadata["comparison_plots"] = _json_safe(comparison_summary)
      (self.output / "review.json").write_text(
        json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
      )
    if (
      comparison_summary is not None
      and comparison_summary.get("status") not in {"ok", "reference_unavailable"}
    ):
      raise RuntimeError(
        "evaluation comparison artifact failed: "
        f"{comparison_summary.get('reason', comparison_summary.get('status'))}"
      )
    return self.metadata
