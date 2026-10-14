"""Headless plotting for poker-draw friction telemetry CSV files."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

_FINGERS = ("index", "middle", "ring", "pinky")
_FINGER_COLORS = {
  "index": "#1f77b4",
  "middle": "#ff7f0e",
  "ring": "#2ca02c",
  "pinky": "#d62728",
}
_FINGER_METRICS = (
  "fn_n",
  "ft_x_n",
  "ft_y_n",
  "ft_n",
  "slip_speed_m_s",
  "cumulative_slip_m",
  "contact",
)
_BASE_COLUMNS = (
  "time_s",
  "phase",
  "card_x_m",
  "card_y_m",
  "card_z_m",
  "card_yaw_rad",
  "table_normal_force_n",
  "table_force_on_card_x_n",
  "table_force_on_card_y_n",
  "table_force_on_card_z_n",
)
_REQUIRED_COLUMNS = (
  *_BASE_COLUMNS,
  *(f"{finger}_{metric}" for finger in _FINGERS for metric in _FINGER_METRICS),
)
_NUMERIC_COLUMNS = tuple(
  column
  for column in _REQUIRED_COLUMNS
  if column != "phase" and not column.endswith("_contact")
)
_PHASE_STYLES = {
  "slide_card": ("#4c78a8", 0.075),
  "edge_hold": ("#f2cf5b", 0.12),
}


def plot_friction_trace(
  csv_path: Path,
  output_path: Path,
  *,
  target_force_n: float,
) -> Path:
  """Render a synchronized six-panel friction trace as a headless PNG.

  Per-taxel telemetry is deliberately outside this compact diagnostic.  The
  four-finger lines retain one color per finger across normal force,
  tangential force, slip speed and accumulated slip panels.
  """

  source = Path(csv_path).expanduser()
  destination = Path(output_path).expanduser()
  target_force = float(target_force_n)
  if not math.isfinite(target_force) or target_force <= 0.0:
    raise ValueError("target_force_n must be finite and positive")
  if destination.exists():
    raise FileExistsError(f"refusing to overwrite existing plot: {destination}")

  telemetry, phases = _read_trace(source)
  time = telemetry["time_s"]
  figure = Figure(figsize=(11.5, 9.0), dpi=120, constrained_layout=True)
  canvas = FigureCanvasAgg(figure)
  axes = np.asarray(figure.subplots(3, 2, sharex=True), dtype=object).reshape(-1)

  normal_axis, tangent_axis, speed_axis, slip_axis, card_axis, table_axis = axes
  for finger in _FINGERS:
    color = _FINGER_COLORS[finger]
    label = finger.capitalize()
    normal_axis.plot(
      time,
      telemetry[f"{finger}_fn_n"],
      color=color,
      linewidth=1.25,
      label=label,
    )
    tangent_axis.plot(
      time,
      telemetry[f"{finger}_ft_n"],
      color=color,
      linewidth=1.25,
      label=label,
    )
    speed_axis.plot(
      time,
      telemetry[f"{finger}_slip_speed_m_s"] * 1000.0,
      color=color,
      linewidth=1.25,
      label=label,
    )
    slip_axis.plot(
      time,
      telemetry[f"{finger}_cumulative_slip_m"] * 1000.0,
      color=color,
      linewidth=1.25,
      label=label,
    )

  normal_axis.axhline(
    target_force,
    color="#222222",
    linestyle="--",
    linewidth=1.1,
    label=f"Target {target_force:g} N",
  )
  normal_axis.set_title("Finger normal force")
  normal_axis.set_ylabel("Fn [N]")
  tangent_axis.set_title("Finger tangential resultant")
  tangent_axis.set_ylabel("Ft [N]")
  speed_axis.set_title("Fn-weighted contact slip speed")
  speed_axis.set_ylabel("Contact slip [mm/s]")
  slip_axis.set_title("Slide + edge accumulated slip")
  slip_axis.set_ylabel("Cumulative slip [mm]")

  card_x = telemetry["card_x_m"]
  finite_x = np.flatnonzero(np.isfinite(card_x))
  reference_x = card_x[finite_x[0]] if len(finite_x) else np.nan
  toward_robot_mm = (reference_x - card_x) * 1000.0
  displacement_line = card_axis.plot(
    time,
    toward_robot_mm,
    color="#222222",
    linewidth=1.4,
    label="Toward-robot displacement",
  )[0]
  card_axis.set_title("Card motion")
  card_axis.set_ylabel("Toward robot [mm]")
  yaw_axis = card_axis.twinx()
  yaw_line = yaw_axis.plot(
    time,
    np.rad2deg(telemetry["card_yaw_rad"]),
    color="#9467bd",
    linewidth=1.2,
    label="Card yaw",
  )[0]
  yaw_axis.set_ylabel("Yaw [deg]", color="#9467bd")
  yaw_axis.tick_params(axis="y", colors="#9467bd")
  card_axis.legend(
    (displacement_line, yaw_line),
    (displacement_line.get_label(), yaw_line.get_label()),
    loc="best",
    fontsize=7,
  )

  table_axis.plot(
    time,
    telemetry["table_normal_force_n"],
    color="#17becf",
    linewidth=1.35,
    label="Table normal force",
  )
  table_axis.plot(
    time,
    telemetry["table_force_on_card_x_n"],
    color="#8c564b",
    linewidth=1.25,
    label="Table force on card, X",
  )
  table_axis.axhline(0.0, color="#777777", linewidth=0.7)
  table_axis.set_title("Card/table contact force")
  table_axis.set_ylabel("Force [N]")

  for axis in axes:
    axis.grid(True, color="#d8d8d8", linewidth=0.55, alpha=0.8)
    axis.tick_params(labelsize=8)
  for axis in (normal_axis, tangent_axis, speed_axis, slip_axis, table_axis):
    axis.legend(loc="best", fontsize=7, ncols=2)
  for axis in axes[-2:]:
    axis.set_xlabel("Simulation time [s]")

  _mark_phases(axes, time, phases)
  figure.suptitle("Poker-draw friction telemetry", fontsize=14)

  destination.parent.mkdir(parents=True, exist_ok=True)
  created = False
  try:
    with destination.open("xb") as stream:
      created = True
      canvas.print_png(stream)
  except BaseException:
    if created:
      destination.unlink(missing_ok=True)
    raise
  finally:
    figure.clear()
  return destination


def _read_trace(path: Path) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
  try:
    stream = path.open("r", encoding="utf-8-sig", newline="")
  except OSError as error:
    raise OSError(f"could not open friction telemetry CSV {path}: {error}") from error

  values = {column: [] for column in _NUMERIC_COLUMNS}
  phases: list[str] = []
  with stream:
    reader = csv.DictReader(stream)
    if reader.fieldnames is None:
      raise ValueError("friction telemetry CSV has no header")
    missing = [
      column for column in _REQUIRED_COLUMNS if column not in reader.fieldnames
    ]
    if missing:
      raise ValueError(
        "friction telemetry CSV is missing columns: " + ", ".join(missing)
      )
    for line_number, row in enumerate(reader, start=2):
      phase = (row.get("phase") or "").strip()
      phases.append(phase)
      for column in _NUMERIC_COLUMNS:
        raw = row.get(column)
        try:
          value = float(raw) if raw is not None else math.nan
        except ValueError as error:
          raise ValueError(
            f"friction telemetry CSV line {line_number}: {column} is not numeric"
          ) from error
        if math.isinf(value):
          raise ValueError(
            f"friction telemetry CSV line {line_number}: {column} is infinite"
          )
        values[column].append(value)
      for finger in _FINGERS:
        _validate_contact(row.get(f"{finger}_contact"), line_number, finger)

  if not phases:
    raise ValueError("friction telemetry CSV contains no data rows")
  telemetry = {
    column: np.asarray(column_values, dtype=np.float64)
    for column, column_values in values.items()
  }
  if not np.all(np.isfinite(telemetry["time_s"])):
    raise ValueError("friction telemetry time_s must contain only finite values")
  if np.any(np.diff(telemetry["time_s"]) < 0.0):
    raise ValueError("friction telemetry time_s must be non-decreasing")
  return telemetry, tuple(phases)


def _validate_contact(raw: str | None, line_number: int, finger: str) -> None:
  text = "" if raw is None else raw.strip().lower()
  if text in {"true", "false"}:
    return
  try:
    value = float(text)
  except ValueError as error:
    raise ValueError(
      f"friction telemetry CSV line {line_number}: {finger}_contact is not boolean"
    ) from error
  if not math.isfinite(value) or value not in (0.0, 1.0):
    raise ValueError(
      f"friction telemetry CSV line {line_number}: {finger}_contact must be 0 or 1"
    )


def _mark_phases(
  axes: np.ndarray,
  time: np.ndarray,
  phases: tuple[str, ...],
) -> None:
  transitions = [0]
  transitions.extend(
    index for index in range(1, len(phases)) if phases[index] != phases[index - 1]
  )
  transitions.append(len(phases))
  label_axis = axes[0]
  label_row = 0
  for segment_index in range(len(transitions) - 1):
    start_index = transitions[segment_index]
    end_index = transitions[segment_index + 1]
    phase = phases[start_index]
    style = _PHASE_STYLES.get(phase)
    if style is None:
      continue
    color, alpha = style
    start_time = float(time[start_index])
    end_time = float(time[end_index]) if end_index < len(time) else float(time[-1])
    for axis in axes:
      axis.axvline(start_time, color=color, linewidth=0.8, linestyle=":", alpha=0.8)
      if end_time > start_time:
        axis.axvspan(start_time, end_time, color=color, alpha=alpha, linewidth=0.0)
    label_axis.text(
      start_time,
      0.98 - 0.12 * (label_row % 2),
      phase,
      color=color,
      fontsize=7,
      rotation=90,
      va="top",
      ha="right",
      transform=label_axis.get_xaxis_transform(),
    )
    label_row += 1


__all__ = ["plot_friction_trace"]
