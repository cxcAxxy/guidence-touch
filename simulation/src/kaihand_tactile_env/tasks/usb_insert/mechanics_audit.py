"""Independent USB contact audit; observes dynamics without changing the controller.

Force balance uses recorded post-forward qacc, NOT a finite difference of the
integrated velocity (a different solver epoch). Slip uses material-point relative
velocity at the contact, NOT movement of the geometrical contact point.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

HOLD_PHASES = ("insert", "bottom_out")
# Engineering acceptance for this simplified task, not real-USB specifications.
# Slip accumulated over the entire insertion must stay below 1 mm, small
# compared with the 14--21 mm pad dimensions. Sustained speed is checked over
# 100 ms, separately from transient speed and the existing 2 ms force gates.
LIMITS = {
  "force_residual_n": 1e-7,
  "torque_residual_nm": 1e-9,
  "hold_min_normal_n": 0.4,
  "hold_slip_path_mm": 1.0,
  "hold_slip_peak_mm_s": 10.0,
  "hold_slip_100ms_mean_mm_s": 2.0,
  "hold_relative_translation_mm": 2.0,
  "hold_relative_rotation_deg": 3.0,
  "hold_normal_step_n": 0.35,
  "hold_tangent_step_n": 0.2,
  "release_peak_tangent_n": 0.15,
  "unload_release_normal_step_n": 0.35,
  "friction_cone_utilization": 1.00001,
}


def digest(path):
  with Path(path).open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def balance_residual(force, torque, mass, gravity, acceleration, inertia, omega, alpha):
  """Newton--Euler residual in world axes, torque about the plug COM."""
  return (
    force + mass * gravity - mass * acceleration,
    torque - inertia @ alpha - np.cross(omega, inertia @ omega),
  )


def write_report(path, report):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("x") as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write("\n")


class ContactAudit:
  def __init__(self, model):
    self.model = model
    self.plug = model.body("usb_plug").id
    self.palm = model.body("hand_r_base_link").id
    self.dof = int(model.joint("usb_plug_freejoint").dofadr[0])
    self.pads = [
      model.geom(n).id
      for n in (
        "hand_r_thumb_link6_tactile_pad_col",
        "hand_r_index_link4_tactile_pad_col",
      )
    ]
    self.friction = [
      model.pair(n).friction.copy() for n in ("usb_thumb_grip", "usb_index_grip")
    ]
    if np.any(model.dof_damping[self.dof : self.dof + 6]) or np.any(
      model.dof_frictionloss[self.dof : self.dof + 6]
    ):
      raise ValueError("audit requires a free undamped plug")
    self.rows = []
    self.jac1 = np.empty((3, model.nv))
    self.jac2 = np.empty_like(self.jac1)

  def observe(self, data, phase, events, qacc=None):
    """events: (geom1, geom2, point_world, frame_world, wrench_on_geom2_local)."""
    m, b, j = self.model, self.plug, self.dof
    qacc = data.qacc if qacc is None else qacc
    rotation = data.xmat[b].reshape(3, 3)
    radius = rotation @ m.body_ipos[b]
    omega = rotation @ data.qvel[j + 3 : j + 6]
    alpha = rotation @ qacc[j + 3 : j + 6]
    acceleration = (
      qacc[j : j + 3]
      + np.cross(alpha, radius)
      + np.cross(omega, np.cross(omega, radius))
    )
    ri = data.ximat[b].reshape(3, 3)
    inertia = (ri * m.body_inertia[b]) @ ri.T
    total_force, total_torque = np.zeros(3), np.zeros(3)
    normal, tangent, slip, utilization, torque_ratio = [np.zeros(2) for _ in range(5)]
    count = np.zeros(2)
    for g1, g2, point, frame, local in events:
      b1, b2 = m.geom_bodyid[[g1, g2]]
      if b not in (b1, b2):
        continue
      sign = 1 if b2 == b else -1
      force, torque = sign * (frame.T @ local[:3]), sign * (frame.T @ local[3:])
      total_force += force
      total_torque += torque + np.cross(point - data.xipos[b], force)
      pad = g1 if g1 in self.pads else g2 if g2 in self.pads else None
      if pad is None:
        continue
      finger = self.pads.index(pad)
      if local[0] <= 1e-8:
        continue
      count[finger] += 1
      normal[finger] += local[0]
      tangent[finger] += np.linalg.norm(local[1:3])
      mujoco.mj_jac(m, data, self.jac1, None, point, b)
      mujoco.mj_jac(m, data, self.jac2, None, point, int(m.geom_bodyid[pad]))
      velocity = (self.jac1 - self.jac2) @ data.qvel
      velocity -= frame[0] * (frame[0] @ velocity)
      slip[finger] = max(slip[finger], np.linalg.norm(velocity) * 1000)
      normalized = local[1:] / self.friction[finger] / local[0]
      utilization[finger] = max(utilization[finger], np.linalg.norm(normalized))
      torque_ratio[finger] = max(torque_ratio[finger], np.linalg.norm(normalized[2:]))
    rf, rt = balance_residual(
      total_force,
      total_torque,
      m.body_mass[b],
      m.opt.gravity,
      acceleration,
      inertia,
      omega,
      alpha,
    )
    palm_rotation = data.xmat[self.palm].reshape(3, 3)
    self.rows.append(
      {
        "time": float(data.time),
        "phase": phase,
        "force_error": np.linalg.norm(rf),
        "torque_error": np.linalg.norm(rt),
        "normal": normal,
        "tangent": tangent,
        "count": count,
        "slip": slip,
        "cone": utilization,
        "torque_ratio": torque_ratio,
        "relative_position": palm_rotation.T @ (data.xpos[b] - data.xpos[self.palm]),
        "relative_rotation": palm_rotation.T @ rotation,
      }
    )

  def live_observer(self, simulation, phase):
    data = simulation.data
    events = []
    for i, contact in enumerate(data.contact):
      if self.plug not in self.model.geom_bodyid[[contact.geom1, contact.geom2]]:
        continue
      wrench = np.empty(6)
      mujoco.mj_contactForce(self.model, data, i, wrench)
      events.append(
        (contact.geom1, contact.geom2, contact.pos, contact.frame.reshape(3, 3), wrench)
      )
    # Monitor has already refreshed FK/solver at the post-step state.
    self.observe(data, phase, events)

  def summarize(self):
    rows = self.rows
    arrays = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
    phases, time = arrays["phase"], arrays["time"]
    hold = np.isin(phases, HOLD_PHASES)
    release = phases == "release"
    unload = np.isin(phases, ("unload", "release"))
    if not all(np.any(phases == p) for p in (*HOLD_PHASES, "unload", "release")):
      return {"passed": False, "error": "missing required phases"}, arrays
    dt = float(np.median(np.diff(time)))
    if not np.allclose(np.diff(time), dt, atol=1e-10, rtol=0):
      raise ValueError("audit requires consecutive constant-rate states")
    first = np.flatnonzero(hold)[0]
    position = arrays["relative_position"][hold]
    angle = np.rad2deg(
      np.arccos(
        np.clip(
          (
            np.einsum(
              "nij,ij->n",
              arrays["relative_rotation"][hold],
              arrays["relative_rotation"][first],
            )
            - 1
          )
          / 2,
          -1,
          1,
        )
      )
    )
    window = int(round(0.1 / dt))
    slip = arrays["slip"][hold]
    mean_slip = np.array(
      [
        np.convolve(slip[:, i], np.ones(window) / window, "valid").max()
        for i in range(2)
      ]
    )
    hold_steps = hold[1:] & hold[:-1]
    unload_steps = unload[1:] & unload[:-1]
    metrics = {
      "max_force_residual_n": float(arrays["force_error"].max()),
      "max_torque_residual_nm": float(arrays["torque_error"].max()),
      "hold_min_normal_n": arrays["normal"][hold].min(axis=0).tolist(),
      "hold_slip_path_mm": (slip.sum(axis=0) * dt).tolist(),
      "hold_slip_peak_mm_s": slip.max(axis=0).tolist(),
      "hold_slip_100ms_mean_mm_s": mean_slip.tolist(),
      "hold_relative_translation_mm": float(
        np.linalg.norm(position - position[0], axis=1).max() * 1000
      ),
      "hold_relative_rotation_deg": float(angle.max()),
      "hold_normal_step_n": abs(np.diff(arrays["normal"], axis=0))[hold_steps]
      .max(axis=0)
      .tolist(),
      "hold_tangent_step_n": abs(np.diff(arrays["tangent"], axis=0))[hold_steps]
      .max(axis=0)
      .tolist(),
      "release_peak_tangent_n": arrays["tangent"][release].max(axis=0).tolist(),
      "unload_release_normal_step_n": abs(np.diff(arrays["normal"], axis=0))[
        unload_steps
      ]
      .max(axis=0)
      .tolist(),
      "friction_cone_utilization": arrays["cone"][hold].max(axis=0).tolist(),
      "torsion_rolling_capacity_utilization": arrays["torque_ratio"][hold]
      .max(axis=0)
      .tolist(),
      "hold_contact_counts": [
        np.unique(arrays["count"][hold, i]).tolist() for i in range(2)
      ],
    }
    gates = {
      "force_balance": metrics["max_force_residual_n"] < LIMITS["force_residual_n"],
      "torque_balance": metrics["max_torque_residual_nm"]
      < LIMITS["torque_residual_nm"],
      "continuous_bilateral_contact": bool(np.all(arrays["count"][hold] == 1)),
      "adequate_grip": min(metrics["hold_min_normal_n"]) >= LIMITS["hold_min_normal_n"],
    }
    for key in LIMITS:
      if key not in ("force_residual_n", "torque_residual_nm", "hold_min_normal_n"):
        gates[key] = bool(np.max(metrics[key]) <= LIMITS[key])
    report = {
      "schema": "usb_contact_mechanics_v1",
      "passed": all(gates.values()),
      "gates": gates,
      "limits": LIMITS,
      "metrics": metrics,
      "samples": len(rows),
      "hold_interval_s": [float(time[hold][0]), float(time[hold][-1])],
      "finger_order": ["thumb", "index"],
      "notes": [
        "Newton-Euler uses all plug contacts including contact torques and lever arms about COM.",
        "Balance compares same-epoch solver qacc, not finite-difference integration acceleration.",
        "Slip is relative material velocity at contact; rolling of contact location alone is not slip.",
        "1 mm path / 2 mm relative translation / 3 degree rotation are declared engineering limits, not real-USB specifications.",
        "No requirement of constant Fn, equal finger load, monotonic Ft, or agreement with real hardware.",
      ],
    }
    return report, arrays


def audit_raw(raw):
  import h5py

  from kaihand_tactile_env.shared.config import model_fingerprint
  from kaihand_tactile_env.shared.simulation import ArmHandSimulation

  raw = Path(raw)
  original = digest(raw)
  simulation = ArmHandSimulation(scene="usb-insert", add_genesis_probes=True)
  m, d = simulation.model, simulation.data
  audit = ContactAudit(m)
  with h5py.File(raw) as f:
    fingerprint = model_fingerprint(Path(f.attrs["model_path"]))
    if fingerprint != f.attrs["model_fingerprint"]:
      raise ValueError("model does not match source")
    if not np.array_equal(
      f["model/geom_names"].asstr()[:], [m.geom(i).name for i in range(m.ngeom)]
    ):
      raise ValueError("compiled geometry IDs do not match source")
    times, phase = f["state/timestamp"][:], f["commands/phase"].asstr()[:]
    qpos, qvel, qacc = (f[n][:] for n in ("state/qpos", "state/qvel", "physics/qacc"))
    if not np.array_equal(times, f["physics/solver_timestamp"][:]):
      raise ValueError("solver clock mismatch")
    if np.any(f["physics/qfrc_applied"][:, audit.dof : audit.dof + 6]) or np.any(
      f["physics/xfrc_applied"][:, audit.plug]
    ):
      raise ValueError("unexpected external plug assistance")
    e = {k: v[:] for k, v in f["contacts/events"].items()}
    chosen = (m.geom_bodyid[e["geom1_id"]] == audit.plug) | (
      m.geom_bodyid[e["geom2_id"]] == audit.plug
    )
    e = {k: v[chosen] for k, v in e.items()}
    offsets = np.searchsorted(e["state_index"], np.arange(len(times) + 1))
    for i in range(len(times)):
      d.time, d.qpos[:], d.qvel[:] = times[i], qpos[i], qvel[i]
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
      events = [
        (
          int(e["geom1_id"][z]),
          int(e["geom2_id"][z]),
          e["position_world"][z],
          e["frame_world"][z],
          e["wrench_contact_on_geom2"][z],
        )
        for z in range(offsets[i], offsets[i + 1])
      ]
      audit.observe(d, str(phase[i]), events, qacc[i])
  report, arrays = audit.summarize()
  report.update(
    source_raw_sha256=original, source_raw_name=raw.name, model_fingerprint=fingerprint
  )
  if digest(raw) != original:
    raise ValueError("source changed during audit")
  return report, arrays
