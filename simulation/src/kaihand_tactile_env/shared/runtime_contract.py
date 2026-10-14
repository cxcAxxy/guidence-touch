"""Describe the compiled physics and tactile providers used by a recorder."""

from __future__ import annotations

import mujoco


def _provider_contract(provider, *, timestep: float) -> dict:
  result = {
    "source": getattr(provider, "source", None),
    "force_unit": getattr(provider, "force_unit", None),
    "status": getattr(provider, "status", None),
  }
  metadata = getattr(provider, "metadata", None)
  if callable(metadata):
    result["metadata"] = metadata()
  for name in (
    "contact_threshold_m", "release_threshold_m", "bool_count_threshold",
    "release_debounce_steps", "aggregate_force_scale", "kernel_sigma_m",
  ):
    if hasattr(provider, name):
      result[name] = getattr(provider, name)
  if "release_debounce_steps" in result:
    result["release_debounce_seconds"] = result["release_debounce_steps"] * timestep
  return result


def simulation_runtime_contract(simulation, *, tactile_provider, contact_force_provider=None) -> dict:
  """Snapshot actual model options, including task-side runtime modifications.

  Units come from each provider. In particular, depth-derived probe aggregates
  must not be relabelled as newtons. No physics or provider settings are changed.
  """
  model = simulation.model
  options = model.opt
  timestep = float(options.timestep)
  physics = {
    "timestep_s": timestep,
    "physics_hz": 1.0 / timestep,
    "integrator": mujoco.mjtIntegrator(int(options.integrator)).name,
    "solver": mujoco.mjtSolver(int(options.solver)).name,
    "cone": mujoco.mjtCone(int(options.cone)).name,
    "iterations": int(options.iterations),
    "tolerance": float(options.tolerance),
    "noslip_iterations": int(options.noslip_iterations),
    "noslip_tolerance": float(options.noslip_tolerance),
    "impratio": float(options.impratio),
    "gravity_m_s2": options.gravity.tolist(),
    "enableflags": int(options.enableflags),
    "disableflags": int(options.disableflags),
  }
  pairs = []
  for index in range(model.npair):
    pairs.append({
      "name": model.pair(index).name or None,
      "geom1": model.geom(int(model.pair_geom1[index])).name,
      "geom2": model.geom(int(model.pair_geom2[index])).name,
      "dim": int(model.pair_dim[index]),
      "margin_m": float(model.pair_margin[index]),
      "gap_m": float(model.pair_gap[index]),
      "friction": model.pair_friction[index].tolist(),
      "solref": model.pair_solref[index].tolist(),
      "solreffriction": model.pair_solreffriction[index].tolist(),
      "solimp": model.pair_solimp[index].tolist(),
    })
  result = {
    "schema": "kaihand_simulation_runtime_v1",
    "schema_version": 1,
    "captured_at": "recorder_initialization",
    "engine": "mujoco",
    "engine_version": mujoco.__version__,
    "task": simulation.scene,
    "physics": physics,
    "explicit_contact_pairs": pairs,
    "tactile": _provider_contract(tactile_provider, timestep=timestep),
  }
  if contact_force_provider is not None:
    result["contact_force"] = _provider_contract(contact_force_provider, timestep=timestep)
  return result
