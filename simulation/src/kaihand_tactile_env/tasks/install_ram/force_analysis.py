"""Unfiltered, phase-specific RAM force and insertion-load diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
STAGES = ("free_transport", "insertion_friction", "bottom_press")


def _runs(mask):
  indices = np.flatnonzero(mask)
  return (
    np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1) if len(indices) else []
  )


def stage_masks(phases, depth, backstop):
  """Keep transport, sliding before the stop, and loaded bottom pressing separate."""
  insertion = np.isin(phases, ("insert", "insertion", "insertion_friction", "seat"))
  bottom = np.isin(phases, ("bottom_press", "bottom_hold", "press", "hold_bottom"))
  # The prior example has no named bottom-press phase; classify its actual stop contact.
  legacy_bottom = insertion & (backstop > 0.1) & (depth > 0.0057)
  return {
    "free_transport": np.isin(phases, ("lift", "transfer", "transport", "align"))
    & (depth < 0),
    "insertion_friction": insertion
    & (depth >= 0.0035)
    & (depth <= 0.0055)
    & ~legacy_bottom,
    "bottom_press": bottom | legacy_bottom,
  }


def _noise_metrics(time, values, mask):
  runs = _runs(mask)
  first = [np.diff(values[run], axis=0) for run in runs if len(run) > 1]
  second = [np.diff(values[run], n=2, axis=0) for run in runs if len(run) > 2]
  result = {
    "adjacent_difference_rms_n": None,
    "adjacent_difference_p95_n": None,
    "second_difference_rms_n": None,
    "spectrum": None,
  }
  if first:
    difference = np.concatenate(first)
    result["adjacent_difference_rms_n"] = np.sqrt(
      np.mean(difference**2, axis=0)
    ).tolist()
    result["adjacent_difference_p95_n"] = np.quantile(
      np.abs(difference), 0.95, axis=0
    ).tolist()
  if second:
    result["second_difference_rms_n"] = np.sqrt(
      np.mean(np.concatenate(second) ** 2, axis=0)
    ).tolist()
  if runs:
    longest = max(runs, key=len)
    if len(longest) >= 32:
      times = time[longest]
      dt = float(np.median(np.diff(times)))
      if np.allclose(np.diff(times), dt, rtol=0, atol=1e-9):
        selected = values[longest]
        window = np.hanning(len(selected))
        spectrum = np.fft.rfft(
          (selected - selected.mean(axis=0)) * window[:, None], axis=0
        )
        power = np.abs(spectrum) ** 2 / (len(selected) * np.sum(window**2))
        power[1 : -1 if len(selected) % 2 == 0 else None] *= 2
        frequency = np.fft.rfftfreq(len(selected), dt)
        band = (frequency >= 15) & (frequency <= min(50, 0.5 / dt) + 1e-8)
        total = power.sum(axis=0)
        high = power[band].sum(axis=0)
        low_oscillation = power[(frequency >= 2) & (frequency <= 10)].sum(axis=0)
        result["spectrum"] = {
          "source_time_range_s": [float(times[0]), float(times[-1])],
          "samples": len(selected),
          "sample_hz": 1 / dt,
          "band_hz": [15, min(50, 0.5 / dt)],
          "band_rms_n": np.sqrt(high).tolist(),
          "band_2_to_10hz_rms_n": np.sqrt(low_oscillation).tolist(),
          "band_power_fraction": np.divide(
            high, total, out=np.zeros_like(high), where=total > 1e-20
          ).tolist(),
          "method": "Longest contiguous stage; subtract mean and Hann window for spectrum only; plotted/archived forces remain unfiltered.",
        }
  return result


def read_force_episode(file):
  force, task = file["tactile_contact_force"], file["install_ram"]
  values = {
    "time": force["timestamp"][:],
    "phase": file["commands/phase"].asstr()[:],
    "normal": force["normal_force_n"][:],
    "signed_tangent": force["tangent_force_n"][:],
    "depth": task["insertion_depth_m"][:],
    "loads": {
      name: value[:]
      for name, value in task.items()
      if value.ndim == 1 and name.endswith("_n")
    },
  }
  values["tangent"] = np.linalg.norm(values["signed_tangent"], axis=-1)
  values["stages"] = stage_masks(
    values["phase"], values["depth"], values["loads"]["backstop_load_n"]
  )
  explicit_press = np.isin(
    values["phase"], ("bottom_press", "bottom_hold", "press", "hold_bottom")
  )
  if explicit_press.any():
    loaded = explicit_press & (values["loads"]["backstop_load_n"] >= 1.2)
    intervals = _runs(loaded)
    stable = np.zeros(len(loaded), dtype=bool)
    if intervals:
      final = intervals[-1]
      stable[final[values["time"][final] >= values["time"][final[-1]] - 0.4 - 1e-8]] = (
        True
      )
    values["stages"]["bottom_press"] = stable
  return values


def force_stage_report(episode):
  time, normal, tangent = episode["time"], episode["normal"], episode["tangent"]
  report = {}
  for name, mask in episode["stages"].items():
    indices = np.flatnonzero(mask)
    if not len(indices):
      report[name] = {"samples": 0, "available": False}
      continue
    report[name] = {
      "available": True,
      "samples": len(indices),
      "intervals_s": [
        [float(time[run[0]]), float(time[run[-1]])] for run in _runs(mask)
      ],
      "depth_range_mm": [
        float(episode["depth"][mask].min() * 1000),
        float(episode["depth"][mask].max() * 1000),
      ],
      "mean_fn_n": normal[mask].mean(axis=0).tolist(),
      "mean_ft_n": tangent[mask].mean(axis=0).tolist(),
      "std_ft_n": tangent[mask].std(axis=0).tolist(),
      "peak_to_peak_ft_n": np.ptp(tangent[mask], axis=0).tolist(),
      "peak_fn_n": normal[mask].max(axis=0).tolist(),
      "peak_ft_n": tangent[mask].max(axis=0).tolist(),
      "mean_signed_ft_xy_n": episode["signed_tangent"][mask].mean(axis=0).tolist(),
      "loads": {
        name: {"mean_n": float(value[mask].mean()), "peak_n": float(value[mask].max())}
        for name, value in episode["loads"].items()
      },
      "ft_fluctuation": _noise_metrics(time, tangent, mask),
    }
  return {
    "fingers": list(FINGERS),
    "stages": report,
    "normal_definition": "Recorded per-fingertip solver normal-force aggregate; separately audited to equal the sum of 35 distributed taxels",
    "tangent_definition": "Magnitude of the recorded per-fingertip signed two-axis solver aggregate in the shared pad basis; separately audited against summed distributed taxels",
    "phase_classification": "Named transport phases before entry; insertion depth 3.5–5.5 mm before bottom load; last <=0.4 s of the final continuous explicit bottom_press segment with backstop >=1.2 N. Prior example without explicit press uses its legacy >0.1 N backstop at depth >5.7 mm and is not an equivalent loaded-hold condition.",
    "force_temporal_filtering": False,
  }


def write_force_analysis(file, output: Path, previous=None):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  output.mkdir(parents=True, exist_ok=True)
  episode = read_force_episode(file)
  report = force_stage_report(episode)
  (output / "force_stages.json").write_text(json.dumps(report, indent=2) + "\n")
  fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
  for index, finger in enumerate(FINGERS):
    axes[0].plot(episode["time"], episode["normal"][:, index], lw=0.8, label=finger)
    axes[1].plot(episode["time"], episode["tangent"][:, index], lw=0.8, label=finger)
  for name, load in episode["loads"].items():
    axes[2].plot(episode["time"], load, lw=0.8, label=name.removesuffix("_n"))
  axes[3].plot(episode["time"], episode["depth"] * 1000, color="black", lw=1)
  for axis, label in zip(
    axes,
    ("Fn (N)", "|Ft| (N)", "Measured socket load (N)", "Insertion depth (mm)"),
    strict=True,
  ):
    axis.set_ylabel(label)
    axis.grid(alpha=0.25)
    for stage, color in zip(STAGES, ("tab:blue", "tab:orange", "tab:red"), strict=True):
      for run in _runs(episode["stages"][stage]):
        axis.axvspan(
          episode["time"][run[0]], episode["time"][run[-1]], color=color, alpha=0.10
        )
  axes[0].legend(ncol=5)
  axes[2].legend(ncol=2, fontsize=8)
  axes[-1].set_xlabel(
    "Simulation time (s); blue: transport; orange: insertion friction; red: bottom press"
  )
  fig.suptitle(
    "RAM installation: raw fingertip forces, physical socket loads and depth"
  )
  fig.tight_layout()
  for suffix in ("png", "pdf"):
    fig.savefig(output / f"insertion_force_stages.{suffix}", dpi=150)
  plt.close(fig)
  comparison = {
    "current": report,
    "previous": None,
    "comparison_sample_hz": float(file.attrs["control_hz"]),
    "interpretation": "Compare stage-matched raw 100 Hz signals; differing trajectories/contact loads are not controlled experimental variables. Inactive fingers remain zero. No claim about frequencies above the saved Nyquist limit.",
  }
  if previous is not None:
    old = read_force_episode(previous)
    comparison["previous"] = force_stage_report(old)
    if not np.isclose(
      float(previous.attrs["control_hz"]), float(file.attrs["control_hz"])
    ):
      raise ValueError(
        "before/after force comparison requires equal raw sampling rates"
      )
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), squeeze=False)
    for column, (label, data) in enumerate((("Previous", old), ("Current", episode))):
      for row, stage in enumerate(STAGES):
        runs = _runs(data["stages"][stage])
        axis = axes[row, column]
        if runs:
          longest = max(runs, key=len)
          for i, finger in enumerate(FINGERS):
            axis.plot(
              data["time"][longest] - data["time"][longest[0]],
              data["tangent"][longest, i],
              lw=0.7,
              label=finger,
            )
        else:
          axis.text(
            0.5, 0.5, "No recorded stage", ha="center", transform=axis.transAxes
          )
        axis.set_title(f"{label}: {stage.replace('_', ' ')}")
        axis.set_ylabel("Raw |Ft| (N)")
        axis.set_xlabel("Time since selected stage segment began (s)")
        axis.grid(alpha=0.25)
    for row in axes:
      upper = max(axis.get_ylim()[1] for axis in row)
      for axis in row:
        axis.set_ylim(0, upper)
    axes[0, 0].legend(ncol=3)
    fig.suptitle(
      "Unfiltered tangential-force comparison; longest contiguous segment per stage; shared row scales"
    )
    fig.tight_layout()
    fig.savefig(output / "tangential_force_comparison.png", dpi=150)
    plt.close(fig)
  (output / "force_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
  return report, comparison


def write_physics_force_comparison(current: Path, previous: Path, output: Path):
  """Compare raw 500 Hz plateaus at matching insertion depth, not wall time."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharey="row")
  report = {"depth_window_m": [0.0035, 0.0055], "force_filtering": False}
  for column, (label, path) in enumerate(
    (("Previous", previous), ("Current", current))
  ):
    with np.load(path) as trace:
      time = trace["time"]
      names = list(trace["socket_columns"])
      depth = trace["socket"][:, names.index("insertion_depth_m")]
      selected = (trace["phase"] == "insert") & (depth >= 0.0035) & (depth <= 0.0055)
      magnitude = np.linalg.norm(trace["ft"], axis=-1)
      runs = _runs(selected)
      if not runs:
        raise ValueError("no friction plateau in physics comparison")
      run = max(runs, key=len)
      report[label.lower()] = {
        "source": str(path.name),
        "intervals_s": [[float(time[r[0]]), float(time[r[-1]])] for r in runs],
        "std_ft_n": magnitude[selected].std(axis=0).tolist(),
        "peak_to_peak_ft_n": np.ptp(magnitude[selected], axis=0).tolist(),
        **_noise_metrics(time, magnitude, selected),
      }
      for finger, row in enumerate(axes):
        row[column].plot(time[run] - time[run[0]], magnitude[run, finger], lw=0.7)
        row[column].set_title(f"{label}: {FINGERS[finger]} |Ft|")
        row[column].set_xlabel("Time since selected depth window began (s)")
        row[column].set_ylabel("Raw force (N)")
        row[column].grid(alpha=0.25)
  before = np.asarray(report["previous"]["spectrum"]["band_2_to_10hz_rms_n"])[:2]
  after = np.asarray(report["current"]["spectrum"]["band_2_to_10hz_rms_n"])[:2]
  report["thumb_index_2_to_10hz_rms_reduction_percent"] = (
    100 * (1 - after / before)
  ).tolist()
  report["interpretation"] = (
    "Same depth window, original 500 Hz samples; controller and task-local patch parameters both changed. Not a single-variable ablation. Earlier stages and all transients remain in the full trace."
  )
  fig.suptitle("Insertion friction: original 500 Hz forces at depth 3.5–5.5 mm")
  fig.tight_layout()
  for extension in ("png", "pdf"):
    fig.savefig(output / f"insertion_stability_comparison.{extension}", dpi=150)
  plt.close(fig)
  (output / "insertion_stability_comparison.json").write_text(
    json.dumps(report, indent=2) + "\n"
  )
  return report
