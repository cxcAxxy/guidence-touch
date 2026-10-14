"""Read-only, per-physics-step diagnostics for poker friction experiments.

Contact-point slip is a rigid-body solver diagnostic, not a controller input or
an extra success criterion.  In particular, object motion over the table must
not be confused with the unwanted motion of a fingertip over the card.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from kaihand_tactile_env.shared.contact_tactile import (
  SolverDistributedTactileProvider,
)

FINGERS = ("index", "middle", "ring", "pinky")
SLIDING_PHASES = frozenset(("slide_card", "edge_hold"))
CONTACT_FORCE_THRESHOLD_N = 1.0e-5
PRESSURE_TARGET_RATIO = 0.25
SLIP_SPEED_THRESHOLD_M_S = 0.001
SUSTAINED_SLIP_DURATION_S = 0.020
SOURCE = "poker_solver_contact_point_telemetry_v1"


def cached_point_velocity(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  body_id: int,
  point_world_m: np.ndarray,
) -> np.ndarray:
  """World linear velocity at a point using the cached solver-stage c-frame.

  MuJoCo cvel is world-oriented, centered at the root tree's subtree COM,
  angular first.  Reading it avoids combining the post-integration qvel with
  the pre-integration contact positions left behind by mj_step.
  """
  root_id = int(model.body_rootid[body_id])
  spatial_velocity = np.asarray(data.cvel[body_id])
  offset = np.asarray(point_world_m) - np.asarray(data.subtree_com[root_id])
  return spatial_velocity[3:] + np.cross(spatial_velocity[:3], offset)


@dataclass
class _FingerStatistics:
  samples: int = 0
  duration_s: float = 0.0
  force_time_n_s: float = 0.0
  tangent_time_n_s: float = 0.0
  peak_force_n: float = 0.0
  loaded_samples: int = 0
  pressure_samples: int = 0
  unloaded_time_s: float = 0.0
  low_pressure_time_s: float = 0.0
  contact_gap_s: float = 0.0
  pressure_gap_s: float = 0.0
  maximum_contact_gap_s: float = 0.0
  maximum_pressure_gap_s: float = 0.0
  contact_loss_events: int = 0
  cumulative_slip_m: float = 0.0
  cumulative_chart_slip_m: np.ndarray = field(default_factory=lambda: np.zeros(2))
  maximum_slip_speed_m_s: float = 0.0
  maximum_contact_slip_speed_m_s: float = 0.0
  slip_integration_time_s: float = 0.0
  above_slip_threshold_time_s: float = 0.0
  first_sustained_slip: dict[str, Any] | None = None
  _previous_loaded: bool = False
  _previous_speed: float = 0.0
  _previous_chart_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
  _slip_run_start: dict[str, Any] | None = None

  def add(
    self,
    *,
    time_s: float,
    dt_s: float,
    contiguous: bool,
    normal_force_n: float,
    tangent_force_n: float,
    loaded: bool,
    pressure_qualified: bool,
    slip_speed_m_s: float,
    slip_max_speed_m_s: float,
    chart_velocity_m_s: np.ndarray,
    card_position_m: np.ndarray,
  ) -> None:
    if not contiguous:
      self.contact_gap_s = 0.0
      self.pressure_gap_s = 0.0
      self._slip_run_start = None
      self._previous_loaded = False
    self.samples += 1
    self.duration_s += dt_s
    self.force_time_n_s += normal_force_n * dt_s
    self.tangent_time_n_s += tangent_force_n * dt_s
    self.peak_force_n = max(self.peak_force_n, normal_force_n)
    self.loaded_samples += int(loaded)
    self.pressure_samples += int(pressure_qualified)
    if loaded:
      self.contact_gap_s = 0.0
    else:
      if contiguous and self._previous_loaded:
        self.contact_loss_events += 1
      self.unloaded_time_s += dt_s
      self.contact_gap_s += dt_s
      self.maximum_contact_gap_s = max(self.maximum_contact_gap_s, self.contact_gap_s)
    if pressure_qualified:
      self.pressure_gap_s = 0.0
    else:
      self.low_pressure_time_s += dt_s
      self.pressure_gap_s += dt_s
      self.maximum_pressure_gap_s = max(
        self.maximum_pressure_gap_s, self.pressure_gap_s
      )
    if loaded:
      self.maximum_slip_speed_m_s = max(self.maximum_slip_speed_m_s, slip_speed_m_s)
      self.maximum_contact_slip_speed_m_s = max(
        self.maximum_contact_slip_speed_m_s, slip_max_speed_m_s
      )
      if contiguous and self._previous_loaded:
        self.cumulative_slip_m += 0.5 * (self._previous_speed + slip_speed_m_s) * dt_s
        self.cumulative_chart_slip_m += (
          0.5 * (self._previous_chart_velocity + chart_velocity_m_s) * dt_s
        )
        self.slip_integration_time_s += dt_s
      if slip_speed_m_s >= SLIP_SPEED_THRESHOLD_M_S:
        self.above_slip_threshold_time_s += dt_s
        if self._slip_run_start is None:
          self._slip_run_start = {
            "time_s": time_s,
            "card_position_m": card_position_m.tolist(),
          }
        if (
          self.first_sustained_slip is None
          and time_s - self._slip_run_start["time_s"]
          >= SUSTAINED_SLIP_DURATION_S - 1.0e-12
        ):
          self.first_sustained_slip = {
            **self._slip_run_start,
            "detected_time_s": time_s,
          }
      else:
        self._slip_run_start = None
      self._previous_speed = slip_speed_m_s
      self._previous_chart_velocity = chart_velocity_m_s.copy()
    else:
      self._slip_run_start = None
    self._previous_loaded = loaded

  def summary(self) -> dict[str, Any]:
    return {
      "sample_count": self.samples,
      "duration_s": self.duration_s,
      "normal_force_mean_n": (
        self.force_time_n_s / self.duration_s if self.duration_s else None
      ),
      "normal_force_peak_n": self.peak_force_n,
      "tangent_force_mean_n": (
        self.tangent_time_n_s / self.duration_s if self.duration_s else None
      ),
      "loaded_sample_fraction": (
        self.loaded_samples / self.samples if self.samples else None
      ),
      "pressure_qualified_sample_fraction": (
        self.pressure_samples / self.samples if self.samples else None
      ),
      "unloaded_time_s": self.unloaded_time_s,
      "low_pressure_time_s": self.low_pressure_time_s,
      "maximum_contact_gap_s": self.maximum_contact_gap_s,
      "maximum_pressure_gap_s": self.maximum_pressure_gap_s,
      "contact_loss_events": self.contact_loss_events,
      "cumulative_slip_m": self.cumulative_slip_m,
      "cumulative_rotating_chart_slip_m": self.cumulative_chart_slip_m.tolist(),
      "slip_integration_time_s": self.slip_integration_time_s,
      "maximum_slip_speed_m_s": self.maximum_slip_speed_m_s,
      "maximum_contact_slip_speed_m_s": self.maximum_contact_slip_speed_m_s,
      "above_slip_threshold_time_s": self.above_slip_threshold_time_s,
      "first_sustained_slip": self.first_sustained_slip,
    }


class PokerFrictionTelemetry:
  """Stream one flat row per completed physics step without changing state.

  Construct after reset, call ``sample(phase)`` once after each ``sim.step``.
  No taxel rendering, mj_forward, qpos/qvel mutation or controller feedback is
  performed.  Only a fixed amount of accumulators is retained in memory.
  """

  def __init__(self, simulation: Any, *, target_force_n: float) -> None:
    if not np.isfinite(target_force_n) or target_force_n <= 0.0:
      raise ValueError("target_force_n must be finite and positive")
    self.simulation = simulation
    self.model = simulation.model
    self.data = simulation.data
    self.target_force_n = float(target_force_n)
    self.timestep = float(self.model.opt.timestep)
    self._card_geom_id = self.model.geom("card_core_geom").id
    self._card_body_id = int(self.model.geom_bodyid[self._card_geom_id])
    self._table_geom_id = self.model.geom("poker_table_top").id
    self.link_names = tuple(f"hand_r_{finger}_link4" for finger in FINGERS)
    # Reuse the exact chart definition used by the Fn/Ft heatmaps, but do not
    # allocate or compute 7x5 taxel maps on each physics step.
    provider = SolverDistributedTactileProvider(
      self.model,
      target_geom_names=("card_core_geom",),
      link_names=self.link_names,
    )
    self._basis_local = provider.tangent_basis_local.copy()
    self._pad_ids = tuple(
      self.model.geom(f"{name}_tactile_pad_col").id for name in self.link_names
    )
    self._body_ids = tuple(self.model.body(name).id for name in self.link_names)
    self._pad_index = {geom_id: index for index, geom_id in enumerate(self._pad_ids)}
    self._wrench = np.zeros(6)
    self._phase_counts: Counter[str] = Counter()
    self._phase_duration: Counter[str] = Counter()
    self._statistics = {
      phase: {finger: _FingerStatistics() for finger in FINGERS}
      for phase in ("press", "slide_card", "edge_hold", "slide_and_edge")
    }
    self._previous_time: float | None = None
    self._previous_phase: str | None = None
    self._first_time: float | None = None
    self._maximum_dt_s = 0.0
    self._unobserved_time_s = 0.0
    self._sampling_complete = True

  def sample(self, phase: str) -> dict[str, float | int | str]:
    if not isinstance(phase, str) or not phase:
      raise ValueError("phase must be a nonempty string")
    data = self.data
    time_s = float(data.time)
    dt_s = (
      self.timestep if self._previous_time is None else time_s - self._previous_time
    )
    if not np.isfinite(time_s) or dt_s <= 0.0:
      raise ValueError("telemetry must be sampled once per increasing simulation time")
    self._maximum_dt_s = max(self._maximum_dt_s, dt_s)
    sampling_contiguous = bool(
      np.isclose(dt_s, self.timestep, rtol=1.0e-6, atol=1.0e-12)
    )
    self._sampling_complete &= sampling_contiguous
    missing_time_s = max(0.0, dt_s - self.timestep) if not sampling_contiguous else 0.0
    self._unobserved_time_s += missing_time_s
    observed_dt_s = dt_s - missing_time_s
    if self._first_time is None:
      self._first_time = time_s
    card_position = np.asarray(data.xpos[self._card_body_id]).copy()
    card_quaternion = np.asarray(data.xquat[self._card_body_id]).copy()
    card_rotation = np.asarray(data.xmat[self._card_body_id]).reshape(3, 3)
    card_velocity = cached_point_velocity(
      self.model, data, self._card_body_id, card_position
    )
    basis_world = np.stack(
      [
        self._basis_local[index] @ np.asarray(data.xmat[body_id]).reshape(3, 3).T
        for index, body_id in enumerate(self._body_ids)
      ]
    )
    counts = np.zeros(4, dtype=int)
    normal = np.zeros(4)
    tangent = np.zeros((4, 2))
    force_world = np.zeros((4, 3))
    tangent_load = np.zeros(4)
    velocity_weight = np.zeros(4)
    weighted_velocity = np.zeros((4, 2))
    weighted_world_velocity = np.zeros((4, 3))
    weighted_speed = np.zeros(4)
    maximum_speed = np.zeros(4)
    table_normal = 0.0
    table_force = np.zeros(3)
    table_contacts = 0
    for contact_id in range(int(data.ncon)):
      contact = data.contact[contact_id]
      geom1, geom2 = int(contact.geom1), int(contact.geom2)
      if self._card_geom_id not in (geom1, geom2):
        continue
      other = geom2 if geom1 == self._card_geom_id else geom1
      index = self._pad_index.get(other)
      if index is None and other != self._table_geom_id:
        continue
      self._wrench.fill(0.0)
      mujoco.mj_contactForce(self.model, data, contact_id, self._wrench)
      frame = np.asarray(contact.frame).reshape(3, 3)
      normal_n = abs(float(self._wrench[0]))
      if other == self._table_geom_id:
        sign_card = 1.0 if geom2 == self._card_geom_id else -1.0
        table_force += sign_card * (frame.T @ self._wrench[:3])
        table_normal += normal_n
        table_contacts += 1
        continue
      assert index is not None
      sign_pad = 1.0 if geom2 == other else -1.0
      force_world[index] += sign_pad * (frame.T @ self._wrench[:3])
      shear_world = sign_pad * (frame[1:].T @ self._wrench[1:3])
      counts[index] += 1
      normal[index] += normal_n
      tangent[index] += basis_world[index] @ shear_world
      tangent_load[index] += float(np.linalg.norm(self._wrench[1:3]))
      if normal_n <= 0.0:
        continue
      point = np.asarray(contact.pos)
      relative = cached_point_velocity(
        self.model, data, self._body_ids[index], point
      ) - cached_point_velocity(self.model, data, self._card_body_id, point)
      # Tangential to the actual contact plane first, then expressed in the
      # stable pad chart. Contact-normal compression is not counted as slip.
      relative_tangent = relative - np.dot(relative, frame[0]) * frame[0]
      speed = float(np.linalg.norm(relative_tangent))
      velocity_weight[index] += normal_n
      weighted_velocity[index] += normal_n * (basis_world[index] @ relative_tangent)
      weighted_world_velocity[index] += normal_n * relative_tangent
      weighted_speed[index] += normal_n * speed
      maximum_speed[index] = max(maximum_speed[index], speed)

    # Most production runs use implicitfast. The explicit label separates
    # step completion from the solver-stage measurements cached by mj_step.
    solver_time_s = (
      time_s - self.timestep
      if self.model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4
      else float("nan")
    )
    row: dict[str, float | int | str] = {
      "time_s": time_s,
      "solver_time_s": solver_time_s,
      "sample_dt_s": dt_s,
      "sampling_gap_s": missing_time_s,
      "phase": phase,
      "card_yaw_rad": float(np.arctan2(card_rotation[1, 0], card_rotation[0, 0])),
      "table_normal_force_n": table_normal,
      "table_contact_count": table_contacts,
    }
    for axis, position, velocity, omega, force in zip(
      "xyz",
      card_position,
      card_velocity,
      np.asarray(data.cvel[self._card_body_id])[:3],
      table_force,
      strict=True,
    ):
      row[f"card_{axis}_m"] = float(position)
      row[f"card_v{axis}_m_s"] = float(velocity)
      row[f"card_omega_{axis}_rad_s"] = float(omega)
      row[f"table_force_on_card_{axis}_n"] = float(force)
    for axis, value in zip("wxyz", card_quaternion, strict=True):
      row[f"card_q{axis}"] = float(value)
    self._phase_counts[phase] += 1
    self._phase_duration[phase] += observed_dt_s
    for index, finger in enumerate(FINGERS):
      loaded = normal[index] > CONTACT_FORCE_THRESHOLD_N
      qualified = normal[index] >= PRESSURE_TARGET_RATIO * self.target_force_n
      ft = float(np.linalg.norm(tangent[index]))
      if loaded:
        chart_velocity = weighted_velocity[index] / velocity_weight[index]
        world_velocity = weighted_world_velocity[index] / velocity_weight[index]
        speed = float(weighted_speed[index] / velocity_weight[index])
        max_speed = float(maximum_speed[index])
      else:
        chart_velocity = np.full(2, np.nan)
        world_velocity = np.full(3, np.nan)
        speed = max_speed = float("nan")
      groups = []
      if phase == "four_finger_press":
        groups.append(("press", self._previous_phase == phase))
      if phase in SLIDING_PHASES:
        groups.extend(
          (
            (phase, self._previous_phase == phase),
            ("slide_and_edge", self._previous_phase in SLIDING_PHASES),
          )
        )
      for group, contiguous in groups:
        self._statistics[group][finger].add(
          time_s=time_s,
          dt_s=observed_dt_s,
          contiguous=contiguous and sampling_contiguous,
          normal_force_n=float(normal[index]),
          tangent_force_n=ft,
          loaded=bool(loaded),
          pressure_qualified=bool(qualified),
          slip_speed_m_s=speed,
          slip_max_speed_m_s=max_speed,
          chart_velocity_m_s=chart_velocity,
          card_position_m=card_position,
        )
      cumulative = self._statistics["slide_and_edge"][finger]
      row.update(
        {
          f"{finger}_fn_n": float(normal[index]),
          f"{finger}_ft_x_n": float(tangent[index, 0]),
          f"{finger}_ft_y_n": float(tangent[index, 1]),
          f"{finger}_ft_n": ft,
          f"{finger}_ft_load_n": float(tangent_load[index]),
          f"{finger}_contact_count": int(counts[index]),
          f"{finger}_contact": int(loaded),
          f"{finger}_pressure_qualified": int(qualified),
          f"{finger}_slip_vx_m_s": float(chart_velocity[0]),
          f"{finger}_slip_vy_m_s": float(chart_velocity[1]),
          f"{finger}_slip_speed_m_s": speed,
          f"{finger}_slip_max_speed_m_s": max_speed,
          f"{finger}_cumulative_slip_m": cumulative.cumulative_slip_m,
          f"{finger}_cumulative_slip_x_m": float(cumulative.cumulative_chart_slip_m[0]),
          f"{finger}_cumulative_slip_y_m": float(cumulative.cumulative_chart_slip_m[1]),
          f"{finger}_contact_gap_s": (
            cumulative.contact_gap_s if phase in SLIDING_PHASES else 0.0
          ),
          f"{finger}_pressure_gap_s": (
            cumulative.pressure_gap_s if phase in SLIDING_PHASES else 0.0
          ),
        }
      )
      for axis, value in zip("xyz", world_velocity, strict=True):
        row[f"{finger}_slip_world_v{axis}_m_s"] = float(value)
      for axis, value in zip("xyz", force_world[index], strict=True):
        row[f"{finger}_force_world_{axis}_n"] = float(value)
    self._previous_time = time_s
    self._previous_phase = phase
    return row

  def metadata(self) -> dict[str, Any]:
    return {
      "source": SOURCE,
      "finger_order": list(FINGERS),
      "link_names": list(self.link_names),
      "target_geom": "card_core_geom",
      "table_geom": "poker_table_top",
      "units": {
        "force": "N",
        "position": "m",
        "velocity": "m/s",
        "time": "s",
        "angle": "rad",
      },
      "target_force_per_finger_n": self.target_force_n,
      "card_mass_kg": float(
        np.sum(self.model.body_mass[self.model.body_rootid == self._card_body_id])
      ),
      "gravity_world_m_s2": np.asarray(self.model.opt.gravity).tolist(),
      "tangent_basis_local": self._basis_local.tolist(),
      "tangent_basis": ["grid_col_positive", "grid_row_positive"],
      "force_sign": "Ft acts on pad, sign-corrected from mj_contactForce acting on geom2",
      "table_force_sign": "world-frame force exerted by poker_table_top on card_core_geom",
      "ft_n": "norm of the net signed tangent force projected into the pad chart",
      "ft_load_n": "sum of magnitudes of each solver-contact shear force, without cancellation",
      "slip_sign": "finger velocity minus card velocity at the same world contact point",
      "slip_velocity": (
        "cached cvel translation plus omega cross(point - subtree_com[rootid]); "
        "relative velocity projected onto each actual contact plane, then pad chart"
      ),
      "slip_speed": "Fn-weighted mean of per-contact tangent speed magnitudes; no vector cancellation",
      "slip_max_speed": "maximum speed across positive-Fn contacts of each finger",
      "unloaded_velocity": "NaN, not zero; no slip integration through separation or recontact",
      "contact_force_threshold_n": CONTACT_FORCE_THRESHOLD_N,
      "contact_rule": "summed Fn strictly greater than contact_force_threshold_n",
      "pressure_quality_threshold_n": PRESSURE_TARGET_RATIO * self.target_force_n,
      "pressure_quality_rule": "summed Fn at least 0.25 times target, separate from actual load-bearing contact",
      "sustained_slip_speed_threshold_m_s": SLIP_SPEED_THRESHOLD_M_S,
      "sustained_slip_duration_s": SUSTAINED_SLIP_DURATION_S,
      "sustained_slip_semantics": (
        "diagnostic heuristic, not a physical success gate; weighted mean speed "
        "at/above threshold for at least duration while continuously loaded; "
        "report onset sample time/card position and later detection time"
      ),
      "integration": (
        "trapezoidal integration only between consecutive loaded samples in a "
        "contiguous measured phase; slide_and_edge includes their transition; "
        "first sample and each recontact contribute no displacement; cumulative "
        "CSV values cover only slide_card plus edge_hold and freeze outside them"
      ),
      "signed_integral": (
        "rotating_chart_path_integral: sum of local pad-chart vx/vy times dt; "
        "not net displacement in a fixed world frame and not contact-position changes"
      ),
      "contact_gap": (
        "right-endpoint sample occupancy, duration accumulated while Fn below "
        "the relevant threshold; initial unloaded interval included in gaps but "
        "not counted as a loaded-to-unloaded loss event"
      ),
      "sampling": "call once immediately after every completed physics step; no mj_forward or state changes",
      "time_s": "step-completion data.time label, not pose/force evaluation time",
      "solver_time_s": (
        "data.time - model timestep for Euler/implicit/implicitfast cached solver "
        "stage; NaN for RK4 because a single stage timestamp is not asserted"
      ),
      "pose_velocity_force_alignment": (
        "pose uses cached xpos/xquat, velocity cached cvel/subtree_com, forces "
        "cached contact solver output; never combine cached contacts with post-step qvel"
      ),
      "sample_interval": "successive data.time differences; first sample uses model.opt.timestep",
      "missing_samples": (
        "interval differing from model timestep breaks continuity; no slip "
        "integration or sustained-slip detection across missing samples; excess "
        "time is unobserved, not classified as loaded/unloaded or assigned to phase duration"
      ),
      "model_timestep_s": self.timestep,
      "summary_groups": {
        "press": ["four_finger_press"],
        "slide_card": ["slide_card"],
        "edge_hold": ["edge_hold"],
        "slide_and_edge": ["slide_card", "edge_hold"],
      },
    }

  def summary(self) -> dict[str, Any]:
    return {
      "source": SOURCE,
      "sample_count": sum(self._phase_counts.values()),
      "phase_sample_counts": dict(self._phase_counts),
      "phase_duration_s": dict(self._phase_duration),
      "first_time_s": self._first_time,
      "last_time_s": self._previous_time,
      "maximum_sample_dt_s": self._maximum_dt_s,
      "unobserved_time_s": self._unobserved_time_s,
      "physics_step_sampling_complete": bool(self._phase_counts)
      and self._sampling_complete,
      **{
        group: {
          "sample_count": statistics[FINGERS[0]].samples,
          "duration_s": statistics[FINGERS[0]].duration_s,
          "fingers": {finger: value.summary() for finger, value in statistics.items()},
        }
        for group, statistics in self._statistics.items()
      },
    }
