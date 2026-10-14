"""Shared Raw recording for installed-object removal and face-up wiping."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

from .config import CameraConfig, WorkcellConfig
from .contact_tactile import SolverDistributedTactileProvider
from .recording import EpisodeRecorder, _append, _stream, validate_episode
from .tactile import SolverContactTactileProvider
from .task_variants import (
  WHITEBOARD_VARIANT_INK_COUNTS,
  current_variant_environment,
  validate_variant_outcome,
)


def _json(path, value):
  Path(path).write_text(
    json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    encoding="utf-8",
  )


class VariantEpisodeRecorder(EpisodeRecorder):
  def __init__(self, path, simulation, capture, *, executor, seed, buffer_rows=128):
    self.executor = executor
    self.group_name = simulation.scene.replace("-", "_")
    self._numeric_names = ()
    super().__init__(
      path, simulation, capture,
      metadata={
        "recording_contract": f"{simulation.scene}_shared_raw_v1",
        "episode_index": 0,
        "seed": seed,
        **current_variant_environment(simulation.scene),
        "initial_state": (
          "tabletop-eraser-and-face-up-board"
          if simulation.scene in WHITEBOARD_VARIANT_INK_COUNTS else "installed"
        ),
        **({
          "ink_pattern": getattr(simulation.task_config, "INK_PATTERN", "z"),
          "ink_patch_count": WHITEBOARD_VARIANT_INK_COUNTS[simulation.scene],
        } if simulation.scene in WHITEBOARD_VARIANT_INK_COUNTS else {}),
        "ink_randomization": getattr(simulation, "ink_randomization", None),
        "initial_position_randomization": getattr(
          simulation, "initial_position_randomization", None
        ),
      },
      capture_taskspace=True, buffer_rows=buffer_rows,
    )

  def _values(self):
    sim = self.sim
    monitor = getattr(self.executor, "monitor", None)
    state = monitor.measure() if monitor is not None else None
    values = asdict(state) if is_dataclass(state) else dict(state or {})
    values.update(
      qfrc_applied=sim.data.qfrc_applied.copy(),
      xfrc_applied=sim.data.xfrc_applied.copy(),
      equality_active=sim.data.eq_active.copy(),
    )
    if sim.scene in WHITEBOARD_VARIANT_INK_COUNTS:
      # Preserve the controller's last integrated cleaning-force caches.
      # Raw is a post-step forward-evaluated observation, not a second cleaning
      # update or a replacement of the controller's interval measurement.
      board_normal = board_tangent = table_support = direct_hand_board = 0.0
      wrench = np.zeros(6)
      for index, contact in enumerate(sim.data.contact):
        pair = {int(contact.geom1), int(contact.geom2)}
        if (
          {sim.board_id, sim.pad_id} <= pair
          or (sim.table_id in pair and pair & {sim.pad_id, sim.handle_id})
          or (sim.board_id in pair and pair & sim._hand_geom_ids)
        ):
          mujoco.mj_contactForce(sim.model, sim.data, index, wrench)
          normal = max(0.0, float(wrench[0]))
          if {sim.board_id, sim.pad_id} <= pair:
            board_normal += normal
            board_tangent += float(np.linalg.norm(wrench[1:3]))
          if sim.table_id in pair and pair & {sim.pad_id, sim.handle_id}:
            table_support += normal
          if sim.board_id in pair and pair & sim._hand_geom_ids:
            direct_hand_board += normal
      values.update(
        ink_remaining=sim.remaining.copy(),
        patch_work_j=sim.cleaning.work_j.copy(),
        patch_stroke_m=sim.cleaning.stroke_m.copy(),
        patch_loaded_time_s=sim.cleaning.loaded_time_s.copy(),
        board_normal_force_n=board_normal,
        board_tangent_force_n=board_tangent,
        table_support_force_n=table_support,
        direct_hand_board_force_n=direct_hand_board,
      )
    return values

  def _initialize(self, metadata):
    self.contact_force_provider = getattr(self.sim, "forces", None)
    if self.contact_force_provider is None:
      object_bodies = {self.sim._object_body_ids[name] for name in self.sim.object_names}
      target_names = tuple(
        self.sim.model.geom(i).name for i in range(self.sim.model.ngeom)
        if self.sim.model.geom_bodyid[i] in object_bodies
        and self.sim.model.geom(i).name
        and (self.sim.model.geom_contype[i] or self.sim.model.geom_conaffinity[i])
      )
      self.contact_force_provider = SolverDistributedTactileProvider(
        self.sim.model, target_geom_names=target_names,
      )
    super()._initialize(metadata)
    self._file["tactile_contact_force"].attrs["clock_semantics"] = (
      "forward-evaluated contacts at the matching recorded state timestamp"
    )
    group = self._file.create_group(self.group_name)
    group.attrs["timestamp_reference"] = "/state/timestamp"
    group.attrs["mechanics_version"] = metadata["mechanics_version"]
    group.attrs["history_clock_semantics"] = (
      "instantaneous measurements at state timestamp; expert-owned history "
      "latches may lag by at most one physics step; recorder never advances monitor"
    )
    numeric = []
    for name, value in self._values().items():
      array = np.asarray(value)
      if array.dtype.kind in "bifu" and array.size:
        _stream(group, name, array.shape, array.dtype)
        numeric.append(name)
    self._numeric_names = tuple(numeric)

  def _record_state(self, phase):
    super()._record_state(phase)
    values = self._values()
    for name in self._numeric_names:
      _append(self._file[f"{self.group_name}/{name}"], values[name])


def record_variant_episode(
  simulation, executor, output, *, seed=None, camera_hz=30,
  cameras=("head", "left_wrist", "right_wrist"), buffer_rows=128,
):
  """Never overwrite an episode; failed experts are saved but never accepted."""
  output = Path(output).expanduser().resolve()
  output.mkdir(parents=True, exist_ok=False)
  capture = WorkcellConfig(
    model_path=simulation.model_path,
    physics_hz=round(1.0 / simulation.timestep), control_hz=100,
    camera_hz=camera_hz,
    cameras=tuple(CameraConfig(name, depth=False, segmentation=False) for name in cameras),
    tactile_provider=SolverContactTactileProvider.source,
  )
  recorder = VariantEpisodeRecorder(
    output / "raw/episode.h5", simulation, capture,
    executor=executor, seed=seed, buffer_rows=buffer_rows,
  )
  physics_observer_attribute = (
    "post_physics_observer"
    if simulation.scene in WHITEBOARD_VARIANT_INK_COUNTS else "physics_observer"
  )
  previous_physics_observer = getattr(simulation, physics_observer_attribute, None)
  previous_observer = executor.observer
  phase = (
    "tabletop_ready" if simulation.scene in WHITEBOARD_VARIANT_INK_COUNTS
    else "installed_ready"
  )
  observed_steps = 0
  initial_time = float(simulation.data.time)
  peak_object_external_force = 0.0
  last_physics_state = None
  # Only the shared robot arm gravity compensation may use qfrc_applied.
  # Include passive socket springs and thread carriage DOFs: assisting them
  # can move an object indirectly even if its freejoint force remains zero.
  prohibited_qfrc_dofs = np.ones(simulation.model.nv, dtype=bool)
  for dofs in simulation._arm_dofs.values():
    prohibited_qfrc_dofs[dofs] = False

  def observe_phase(sim, label):
    nonlocal phase
    phase = label
    sim.phase = label
    if previous_observer is not None:
      previous_observer(sim, label)

  def observe_physics(sim):
    nonlocal observed_steps, peak_object_external_force, last_physics_state
    if previous_physics_observer is not None:
      previous_physics_observer(sim)
    observed_steps += 1
    value = max(
      float(np.max(np.abs(sim.data.xfrc_applied), initial=0.0)),
      float(np.max(np.abs(sim.data.qfrc_applied[prohibited_qfrc_dofs]), initial=0.0)),
    )
    peak_object_external_force = max(peak_object_external_force, value)
    if value > 1e-12:
      raise RuntimeError("additional-task object or mechanism force assistance is forbidden")
    last_physics_state = (
      sim.data.qpos.copy(), sim.data.qvel.copy(), sim.data.eq_active.copy(),
    )
    recorder.observe(sim, getattr(sim, "task_phase", getattr(sim, "phase", phase)))

  try:
    recorder.record_initial(phase)
    setattr(simulation, physics_observer_attribute, observe_physics)
    executor.observer = observe_phase
    result = executor.run()
    if is_dataclass(result):
      result = asdict(result)
    result = dict(result)
    expected_steps = round((float(simulation.data.time) - initial_time) / simulation.timestep)
    terminal_physics_consistent = bool(
      last_physics_state is not None and all(
        np.array_equal(actual, recorded) for actual, recorded in zip(
          (simulation.data.qpos, simulation.data.qvel, simulation.data.eq_active),
          last_physics_state, strict=True,
        )
      )
    )
    if observed_steps and not terminal_physics_consistent:
      raise RuntimeError("terminal state changed after the last observed physics step")
    physics_audit_passed = bool(
      observed_steps > 0 and observed_steps == expected_steps
      and peak_object_external_force <= 1e-12 and terminal_physics_consistent
    )
    result.update(
      task=simulation.scene,
      object_name=simulation.object_names[0],
      physics_recording_audit={
        "observed_physics_steps": observed_steps,
        "expected_physics_steps": expected_steps,
        "peak_object_external_wrench_or_generalized_force": peak_object_external_force,
        "force_audit_scope": "all body xfrc and all qfrc except shared robot arm gravity compensation",
        "terminal_physics_state_unchanged": terminal_physics_consistent,
        "passed": physics_audit_passed,
      },
    )
    if result.get("success") is True:
      validate_variant_outcome(simulation.scene, result)
      if not physics_audit_passed:
        raise RuntimeError("expert did not expose every physical-step recording clock")
    recorder.record_terminal("terminal_settle")
    recorder.set_outcome(result)
    recorder.close()
    validation = validate_episode(recorder.path)
    import h5py

    diagnostic_errors = []
    with h5py.File(recorder.path, "r") as file:
      group = file[recorder.group_name]
      for name in recorder._numeric_names:
        dataset = group[name]
        if len(dataset) != validation.state_samples:
          diagnostic_errors.append(f"{group.name}/{name}: state clock length mismatch")
        elif not np.isfinite(dataset[:]).all():
          diagnostic_errors.append(f"{group.name}/{name}: nonfinite task diagnostic")
    if diagnostic_errors:
      validation = replace(
        validation, valid=False, errors=validation.errors + tuple(diagnostic_errors),
      )
    sidecar_path = recorder.path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar.update(
      validation=asdict(validation),
      task_audit_passed=result.get("success") is True,
    )
    _json(sidecar_path, sidecar)
    _json(output / "result.json", result)
    if not validation.valid:
      raise ValueError(f"Raw validation failed: {validation.errors}")
    return result
  finally:
    setattr(simulation, physics_observer_attribute, previous_physics_observer)
    executor.observer = previous_observer
    recorder.close(finalize=False)
