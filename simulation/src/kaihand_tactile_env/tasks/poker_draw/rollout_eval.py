"""Read-only task outcome monitor, independent of policy/controller phases."""

from __future__ import annotations

import numpy as np

EVALUATION_VERSION = "poker-draw-outcome-v1"


class PokerOutcomeMonitor:
  """Reached supported edge, lifted with opposition, then held facing robot.

  No exact per-finger pressure band or scripted timing is a success requirement.
    The edge threshold is >=40% overhang (10 percentage-point tolerance around
    half-card exposure for closed-loop tracking), while the
  card remains flat and near tabletop height; aerial motion cannot satisfy it.
  """

  def __init__(
    self, *, required_hold_seconds: float = 0.10,
    require_clearance_during_hold: bool = False,
  ):
    if not np.isfinite(required_hold_seconds) or required_hold_seconds <= 0:
      raise ValueError("required_hold_seconds must be positive and finite")
    self.required_hold_seconds = float(required_hold_seconds)
    self.require_clearance_during_hold = bool(require_clearance_during_hold)
    self.contacted = self.edge_reached = self.lifted = self.success = False
    self.hold_seconds = 0.0
    self.max_overhang = self.max_displacement = 0.0
    self.first_times = {}

  def observe(
    self,
    *,
    time_s,
    dt,
    contacted,
    supported,
    flat,
    overhang,
    displacement,
    clearance,
    opposed,
    face_head,
    face_robot,
    position_error,
    linear_speed,
    angular_speed,
  ):
    values = np.array(
      [
        time_s,
        dt,
        overhang,
        displacement,
        clearance,
        face_head,
        face_robot,
        position_error,
        linear_speed,
        angular_speed,
      ]
    )
    if not np.all(np.isfinite(values)) or dt <= 0:
      raise ValueError("nonfinite outcome metrics or invalid timestep")
    self.max_displacement = max(self.max_displacement, displacement)
    conditions = {
      "contacted": contacted,
      "edge_reached": self.contacted
      and supported
      and flat
      and overhang >= 0.40
      and displacement >= 0.05,
      "lifted": self.edge_reached and opposed and clearance >= 0.02,
    }
    if supported and flat:
      self.max_overhang = max(self.max_overhang, overhang)
    for stage, satisfied in conditions.items():
      if satisfied and not getattr(self, stage):
        setattr(self, stage, True)
        self.first_times[stage] = float(time_s)
    held = (
      self.lifted
      and opposed
      and (not self.require_clearance_during_hold or clearance >= 0.02)
      and face_head >= 0.8
      and face_robot >= 0.8
      and position_error <= 0.03
      and linear_speed < 0.02
      and angular_speed < 0.2
    )
    self.hold_seconds = self.hold_seconds + dt if held else 0.0
    if self.hold_seconds >= self.required_hold_seconds - 1e-10 and not self.success:
      self.success = True
      self.first_times["success"] = float(time_s)

  def failure_stage(self):
    if not self.contacted:
      return "no_card_contact"
    if not self.edge_reached:
      return "did_not_reach_edge"
    if not self.lifted:
      return "did_not_lift_with_opposition"
    return "did_not_hold_card_facing_robot"

  def report(self):
    return {
      "version": EVALUATION_VERSION,
      "success": self.success,
      "contacted": self.contacted,
      "edge_reached": self.edge_reached,
      "lifted": self.lifted,
      "hold_seconds": self.hold_seconds,
      "required_hold_seconds": self.required_hold_seconds,
      "require_clearance_during_hold": self.require_clearance_during_hold,
      "maximum_supported_overhang_fraction": self.max_overhang,
      "maximum_robotward_card_displacement_m": self.max_displacement,
      "first_times": self.first_times,
      "failure_stage": None if self.success else self.failure_stage(),
    }


def observe_simulation(monitor, controller, initial_x):
  """Sensors and object geometry only; no writes to sim or control state."""
  sim = controller.sim
  metrics = controller.terminal_metrics()
  pose = sim.object_pose("card")
  rotation = controller._card_rotation()
  half_size = sim.model.geom("card_core_geom").size
  half_x = float(np.abs(rotation[0]) @ half_size)
  table = sim.model.geom("poker_table_top").id
  table_edge = float(sim.data.geom_xpos[table, 0] - sim.model.geom_size[table, 0])
  overhang = float(np.clip((table_edge - pose[0] + half_x) / (2 * half_x), 0, 1))
  twist = np.asarray(metrics["card_twist"])
  monitor.observe(
    time_s=float(sim.data.time),
    dt=sim.timestep,
    contacted=any(force > 0.01 for force in metrics["force_n"].values()),
    supported=controller._supported(),
    flat=rotation[2, 2] >= 0.94,
    overhang=overhang,
    displacement=float(initial_x - pose[0]),
    clearance=controller._card_table_clearance(),
    opposed=metrics["opposed_contact"],
    face_head=metrics["face_to_head_cosine"],
    face_robot=metrics["face_to_robot_cosine"],
    position_error=metrics["inspection_position_error_m"],
    linear_speed=float(np.linalg.norm(twist[:3])),
    angular_speed=float(np.linalg.norm(twist[3:])),
  )
