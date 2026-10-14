"""Measure raw solver-force variation without filtering or inventing samples.

Adjacent differences include legitimate impacts and load changes. Compare the
same physical phase and sampling interval, and inspect actual task success;
small differences alone do not establish a better mechanical model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def summarize_force_trace(
  timestamps: np.ndarray,
  phases: Sequence[str],
  normal_force_n: np.ndarray,
  tangent_force_n: np.ndarray,
  *,
  finger_names: Sequence[str] = ("thumb", "index", "middle", "ring", "pinky"),
  contact_count: np.ndarray | None = None,
  target_mask: np.ndarray | None = None,
) -> dict:
  """Summarize signed pad-frame Ft at its original physical sample interval.

  Inputs are N timestamps/phases, N x F normal forces and N x F x 2 signed
  tangential forces. Contact counts and optional integer target-geometry masks
  must be N x F. Repeated phases are split into contiguous intervals so a gap
  never becomes a fictitious adjacent-force difference. Spectra are diagnostic
  demeaned FFT powers; returned force statistics always use unfiltered values.
  """
  time = np.asarray(timestamps, dtype=float)
  labels = np.asarray(phases, dtype=str)
  normal = np.asarray(normal_force_n, dtype=float)
  tangent = np.asarray(tangent_force_n, dtype=float)
  fingers = tuple(str(name) for name in finger_names)
  if time.ndim != 1 or len(time) < 2:
    raise ValueError("timestamps require at least two one-dimensional samples")
  n, f = len(time), len(fingers)
  if labels.shape != (n,) or normal.shape != (n, f) or tangent.shape != (n, f, 2):
    raise ValueError("timestamp, phase and force shapes do not match")
  if not all(np.isfinite(a).all() for a in (time, normal, tangent)):
    raise ValueError("force diagnostics require finite raw samples")
  if np.any(np.diff(time) <= 0) or np.any(normal < -1e-12):
    raise ValueError("timestamps must increase and normal forces be nonnegative")
  if len(set(fingers)) != len(fingers):
    raise ValueError("finger_names must be unique")
  for array in (contact_count, target_mask):
    if array is not None and np.asarray(array).shape != (n, f):
      raise ValueError("contact count/mask shape must be N x F")
  dt = float(np.median(np.diff(time)))
  regular = bool(np.allclose(np.diff(time), dt, atol=1e-9, rtol=1e-6))
  report = {
    "schema_version": "install_ram_unfiltered_force_diagnostics_v1",
    "sample_count": n,
    "finger_names": list(fingers),
    "median_sample_interval_s": dt,
    "sample_hz": 1.0 / dt,
    "nyquist_frequency_hz": 0.5 / dt,
    "regular_sample_clock": regular,
    "force_semantics": "Actual solver forces; signed two-axis Ft in shared pad basis.",
    "variation_semantics": "Euclidean difference of consecutive signed Ft vectors within each contiguous phase; no smoothing or interpolation.",
    "phase_intervals": [],
  }
  starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
  for start, stop in zip(starts, np.r_[starts[1:], n], strict=True):
    a = tangent[start:stop]
    magnitude = np.linalg.norm(a, axis=-1)
    interval = {
      "phase": str(labels[start]),
      "start_s": float(time[start]),
      "end_s": float(time[stop - 1]),
      "samples": int(stop - start),
      "mean_fn_n": normal[start:stop].mean(axis=0).tolist(),
      "maximum_fn_n": normal[start:stop].max(axis=0).tolist(),
      "maximum_ft_n": magnitude.max(axis=0).tolist(),
      "ft_vector_rms_n": np.sqrt(np.mean(magnitude**2, axis=0)).tolist(),
    }
    if stop - start >= 2:
      delta = np.linalg.norm(np.diff(a, axis=0), axis=-1)
      interval["adjacent_ft_difference_p95_n"] = np.quantile(
        delta, 0.95, axis=0
      ).tolist()
      interval["adjacent_ft_difference_rms_n"] = np.sqrt(
        np.mean(delta**2, axis=0)
      ).tolist()
      change = np.zeros_like(delta, dtype=bool)
      for optional in (contact_count, target_mask):
        if optional is not None:
          values = np.asarray(optional)[start:stop]
          change |= values[1:] != values[:-1]
      if contact_count is not None or target_mask is not None:
        interval["contact_change_fraction"] = change.mean(axis=0).tolist()
        interval["unchanged_contact_difference_rms_n"] = [
          float(np.sqrt(np.mean(delta[~change[:, i], i] ** 2)))
          if (~change[:, i]).any()
          else None
          for i in range(f)
        ]
      interval_regular = np.allclose(
        np.diff(time[start:stop]), dt, atol=1e-9, rtol=1e-6
      )
      if interval_regular and stop - start >= 16:
        power = np.abs(np.fft.rfft(a - a.mean(axis=0), axis=0)) ** 2
        power = power.sum(axis=-1)
        frequency = np.fft.rfftfreq(stop - start, dt)
        energy = np.maximum(power.sum(axis=0), 1e-30)
        interval["spectral_power_above_100hz_fraction"] = (
          (power[frequency > 100].sum(axis=0) / energy).tolist()
          if 0.5 / dt > 100
          else None
        )
        interval["spectral_power_49_to_51hz_fraction"] = (
          power[(frequency >= 49) & (frequency <= 51)].sum(axis=0) / energy
        ).tolist()
        peaks = np.argmax(power, axis=0)
        interval["peak_non_dc_frequency_hz"] = [
          float(frequency[peaks[i]]) if power[:, i].sum() > 1e-25 else None
          for i in range(f)
        ]
    report["phase_intervals"].append(interval)
  return report


class ForceTraceRecorder:
  """Collect lightweight raw force aggregates from one live simulation run.

  Call after the monitor's post-integration ``mj_forward``. This reader never
  changes simulation state, advances physics, filters forces or creates taxels.
  Duplicate terminal timestamps are ignored; backwards or skipped physics
  timestamps raise an error so incomplete capture cannot masquerade as 500 Hz.
  """

  FINGERS = ("thumb", "index", "middle", "ring", "pinky")
  SOCKET_COLUMNS = (
    "insertion_depth_m",
    "axial_resistance_n",
    "backstop_load_n",
    "spring_normal_load_n",
  )

  def __init__(self, simulation):
    from ...shared.config import model_fingerprint
    from ...shared.contact_tactile import SolverDistributedTactileProvider
    from ...shared.tactile import RIGHT_FINGERTIP_LINK_NAMES

    self.sim = simulation
    self._provider = SolverDistributedTactileProvider(
      simulation.model, link_names=RIGHT_FINGERTIP_LINK_NAMES
    )
    self._palm_body = simulation.model.body("hand_r_base_link").id
    targets = sorted(self._provider._target_geom_ids)
    if len(targets) > 64:
      raise ValueError("force trace target geometry mask supports at most 64 geoms")
    self._target_bits = {geom: i for i, geom in enumerate(targets)}
    self._finger_count = len(self.FINGERS)
    self._wrench = np.zeros(6)
    self._rows: list[dict] = []
    self._model_fingerprint = model_fingerprint(simulation.model_path)
    self._timestep = float(simulation.model.opt.timestep)

  @property
  def sample_count(self) -> int:
    return len(self._rows)

  def record(self, phase: str, state) -> bool:
    import mujoco

    model, data = self.sim.model, self.sim.data
    timestamp = float(data.time)
    if not np.isclose(float(state.timestamp), timestamp, atol=1e-10, rtol=0):
      raise ValueError("force trace requires a monitor state at the current time")
    if self._rows:
      elapsed = timestamp - self._rows[-1]["time"]
      if abs(elapsed) <= 1e-12:
        return False
      if not np.isclose(elapsed, self._timestep, atol=1e-9, rtol=1e-6):
        raise ValueError("force trace skipped a physics step or time moved backwards")
    rotations = data.xmat[self._provider._body_ids].reshape(-1, 3, 3)
    basis = np.einsum("lij,lkj->lki", rotations, self._provider.tangent_basis_local)
    fn = np.zeros(self._finger_count)
    ft = np.zeros((self._finger_count, 2))
    count = np.zeros(self._finger_count, dtype=np.int32)
    mask = np.zeros(self._finger_count, dtype=np.uint64)
    for contact_id, contact in enumerate(data.contact):
      selected = self._provider._select_contact(int(contact.geom1), int(contact.geom2))
      if selected is None:
        continue
      finger, _, target, sign = selected
      mujoco.mj_contactForce(model, data, contact_id, self._wrench)
      frame = contact.frame.reshape(3, 3)
      tangent_world = sign * (
        frame.T @ np.array([0.0, self._wrench[1], self._wrench[2]])
      )
      fn[finger] += abs(float(self._wrench[0]))
      ft[finger] += basis[finger] @ tangent_world
      count[finger] += 1
      mask[finger] |= np.uint64(1) << np.uint64(self._target_bits[target])
    palm_down = -float(data.xmat[self._palm_body].reshape(3, 3)[2, 1])
    self._rows.append(
      {
        "time": timestamp,
        "phase": str(phase),
        "fn": fn,
        "ft": ft,
        "count": count,
        "mask": mask,
        "socket": np.array(
          [float(getattr(state, name, 0.0)) for name in self.SOCKET_COLUMNS]
        ),
        "palm_down": palm_down,
        "spring_axial_resistance_n": float(
          getattr(state, "spring_axial_resistance_n", 0.0)
        ),
        "socket_normal_load_n": float(state.socket_normal_load_n),
        "maximum_socket_penetration_m": float(state.maximum_socket_penetration_m),
        "bottom_out_duration_s": float(getattr(state, "bottom_out_duration_s", 0.0)),
        "bottom_out_confirmed": bool(getattr(state, "bottom_out_confirmed", False)),
        "linear_speed_m_s": float(state.linear_speed_m_s),
        "angular_speed_rad_s": float(state.angular_speed_rad_s),
        "aperture_fits": bool(state.aperture_fits),
        "orientation_error_rad": float(state.orientation_error_rad),
        "seated": bool(state.seated),
        "success": bool(state.success),
      }
    )
    return True

  def arrays(self) -> dict[str, np.ndarray]:
    if not self._rows:
      raise ValueError("force trace is empty")
    arrays = {
      key: np.asarray([row[key] for row in self._rows]) for key in self._rows[0]
    }
    arrays["finger_names"] = np.asarray(self.FINGERS)
    arrays["socket_columns"] = np.asarray(self.SOCKET_COLUMNS)
    arrays["palm"] = arrays["palm_down"].copy()
    return arrays

  def save(self, path) -> dict:
    """Write NPZ plus a same-stem JSON summary and return that summary."""
    import hashlib
    import json
    from pathlib import Path

    destination = Path(path).with_suffix(".npz")
    report_path = destination.with_suffix(".json")
    if destination.exists() or report_path.exists():
      raise FileExistsError("force trace output already exists")
    arrays = self.arrays()
    summary = summarize_force_trace(
      arrays["time"],
      arrays["phase"],
      arrays["fn"],
      arrays["ft"],
      contact_count=arrays["count"],
      target_mask=arrays["mask"],
    )
    summary.update(
      {
        "samples": self.sample_count,
        "source": "Live per-physics-step MuJoCo solver contacts; equality to an HDF5 is checked separately.",
        "source_model_fingerprint": self._model_fingerprint,
        "socket_columns": list(self.SOCKET_COLUMNS),
        "palm_semantics": "Cosine of right palm local +Y with world -Z; positive means palm down.",
        "target_geometry_bits": {
          str(bit): self.sim.model.geom(geom).name
          for geom, bit in self._target_bits.items()
        },
      }
    )
    for interval in summary["phase_intervals"]:
      selected = (arrays["time"] >= interval["start_s"]) & (
        arrays["time"] <= interval["end_s"]
      )
      interval["mean_ft_n"] = (
        np.linalg.norm(arrays["ft"][selected], axis=-1).mean(axis=0).tolist()
      )
      interval["mean_signed_ft_n"] = arrays["ft"][selected].mean(axis=0).tolist()
      interval["mean_socket_loads"] = arrays["socket"][selected].mean(axis=0).tolist()
      interval["minimum_palm_down_cosine"] = float(arrays["palm_down"][selected].min())
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    summary["npz_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    report_path.write_text(
      json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
