"""USB insertion outcome and review-video metrics for model rollouts."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .task import UsbInsertionMonitor, UsbInsertionState


class UsbPolicyOutcome:
  """Continuously evaluate the canonical mechanical USB insertion criterion."""

  name = "usb-insertion-outcome-v1"

  def __init__(self, simulation: Any) -> None:
    self.monitor = UsbInsertionMonitor(simulation)
    self.initial_height_m = float(simulation.object_pose("usb_plug")[2])
    self.state = self.monitor.update()
    self.maximum_lift_m = 0.0
    self.maximum_insertion_depth_m = 0.0
    self.maximum_socket_penetration_m = self.state.maximum_socket_penetration_m
    self.peak_socket_normal_load_n = self.state.socket_normal_load_n
    self.peak_axial_resistance_n = self.state.axial_resistance_n
    self.first_times: dict[str, float] = {}

  def update(self, simulation: Any) -> UsbInsertionState:
    self.state = self.monitor.update()
    self.maximum_lift_m = max(
      self.maximum_lift_m,
      float(simulation.object_pose("usb_plug")[2]) - self.initial_height_m,
    )
    aligned = self._aligned_with_socket()
    if aligned:
      self.maximum_insertion_depth_m = max(
        self.maximum_insertion_depth_m, self.state.insertion_depth_m
      )
    self.maximum_socket_penetration_m = max(
      self.maximum_socket_penetration_m, self.state.maximum_socket_penetration_m
    )
    self.peak_socket_normal_load_n = max(
      self.peak_socket_normal_load_n, self.state.socket_normal_load_n
    )
    self.peak_axial_resistance_n = max(
      self.peak_axial_resistance_n, self.state.axial_resistance_n
    )
    now = float(self.state.timestamp)
    if self.maximum_lift_m >= 0.03:
      self.first_times.setdefault("lifted", now)
    if aligned and self.state.insertion_depth_m > 0.0:
      self.first_times.setdefault("entered_socket", now)
    if self.state.bottom_out_confirmed:
      self.first_times.setdefault("bottom_out_confirmed", now)
    if self.state.seated:
      self.first_times.setdefault("seated", now)
    if self.state.success:
      self.first_times.setdefault("success", now)
    return self.state

  @property
  def success(self) -> bool:
    return bool(self.state.success)

  def _aligned_with_socket(self) -> bool:
    return bool(
      self.state.shell_fits_aperture
      and self.state.orientation_error_rad <= np.deg2rad(5.0)
    )

  def stage(self) -> str:
    if self.state.success:
      return "success"
    if self.state.seated:
      return "seated"
    if self.state.bottom_out_confirmed:
      return "bottom_out"
    if self._aligned_with_socket() and self.state.insertion_depth_m > 0.0:
      return "inserting"
    if self.maximum_lift_m >= 0.03:
      return "lifted_or_aligning"
    return "pickup"

  def failure_stage(self) -> str | None:
    if self.success:
      return None
    if self.maximum_lift_m < 0.03:
      return "did_not_lift"
    if self.maximum_insertion_depth_m <= 0.0:
      return "did_not_enter_socket"
    if not self.state.bottom_out_confirmed:
      return "did_not_confirm_bottom_out"
    return "not_stably_seated"

  def snapshot(self, _simulation: Any) -> dict[str, float | bool | str | None]:
    lateral = float(np.linalg.norm(self.state.lateral_error_m))
    aligned = self._aligned_with_socket()
    return {
      "insertion_depth_mm": (
        self.state.insertion_depth_m * 1000.0 if aligned else 0.0
      ),
      "peak_insertion_depth_mm": self.maximum_insertion_depth_m * 1000.0,
      "lateral_error_mm": lateral * 1000.0,
      "orientation_error_deg": float(np.rad2deg(self.state.orientation_error_rad)),
      "aligned_with_socket": aligned,
      "axial_resistance_n": self.state.axial_resistance_n,
      "socket_normal_load_n": self.state.socket_normal_load_n,
      "socket_penetration_mm": self.state.maximum_socket_penetration_m * 1000.0,
      "seated": bool(self.state.seated),
      "success": self.success,
      "success_stage": self.stage(),
      "failure_stage": self.failure_stage(),
    }

  def metadata(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "criterion": (
        "canonical UsbInsertionMonitor: aligned near-bottom mechanical fit, "
        "physical bottom-out evidence, stationary, and continuous seated dwell"
      ),
      "sampling": "updated at every 500 Hz physics step",
    }

  def report(self) -> dict[str, Any]:
    return {
      "version": self.name,
      "success": self.success,
      "success_stage": self.stage(),
      "failure_stage": self.failure_stage(),
      "maximum_lift_m": self.maximum_lift_m,
      "maximum_insertion_depth_m": self.maximum_insertion_depth_m,
      "maximum_socket_penetration_m": self.maximum_socket_penetration_m,
      "peak_socket_normal_load_n": self.peak_socket_normal_load_n,
      "peak_axial_resistance_n": self.peak_axial_resistance_n,
      "first_times": self.first_times,
      "insertion": asdict(self.state),
    }
