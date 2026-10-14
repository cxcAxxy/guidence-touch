"""Nominal actuator-only USB unplugging schedule."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UsbUnplugMotion:
  settle_s: float = 0.30
  preshape_s: float = 0.60
  hover_s: float = 2.40
  approach_s: float = 1.40
  close_s: float = 1.30
  breakaway_s: float = 1.00
  extraction_s: float = 3.20
  socket_lift_s: float = 2.00
  transfer_s: float = 3.20
  place_s: float = 2.20
  place_hold_s: float = 0.50
  release_s: float = 1.20
  retreat_s: float = 1.00
  return_home_s: float = 2.50
  verify_s: float = 0.50
  control_period_s: float = 0.02
  extract_servo_step_m: float = 0.00008
  carry_servo_step_m: float = 0.0015
  angular_servo_step_rad: float = 0.018
  socket_hover_depth_m: float = -0.10
  post_extract_depth_m: float = -0.004
  home_hover_height_m: float = 0.050


DEFAULT_MOTION = UsbUnplugMotion()

__all__ = ["DEFAULT_MOTION", "UsbUnplugMotion"]
