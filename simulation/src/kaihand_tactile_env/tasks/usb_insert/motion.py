"""USB-only movement schedules; contact and success thresholds stay independent."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UsbMotionProfile:
  settle_s: float = 0.4
  preshape_s: float = 1.0
  # Shared folded home requires a longer wrist-roll travel before descent.
  hover_s: float = 2.1
  approach_s: float = 1.6
  approach_hold_s: float = 0.3
  grasp_ramp_s: float = 1.25
  lift_s: float = 1.3
  transfer_s: float = 3.2
  transfer_hold_s: float = 1.0
  align_s: float = 2.5
  align_clearance_m: float = 0.10
  align_hold_s: float = 0.6
  socket_approach_s: float = 3.0
  insertion_step_m: float = 0.00006
  insertion_servo_step_m: float = 0.00008
  retreat_s: float = 1.0
  verify_s: float = 0.5


MOTION_PROFILES = {
  "baseline": UsbMotionProfile(),
  "fast": UsbMotionProfile(
    settle_s=0.25,
    preshape_s=0.7,
    hover_s=2.1,
    approach_s=1.1,
    approach_hold_s=0.2,
    grasp_ramp_s=1.0,
    lift_s=0.9,
    transfer_s=2.6,
    transfer_hold_s=0.6,
    align_s=1.8,
    insertion_step_m=0.00006,
    insertion_servo_step_m=0.00008,
    retreat_s=0.75,
    verify_s=0.4,
  ),
}
