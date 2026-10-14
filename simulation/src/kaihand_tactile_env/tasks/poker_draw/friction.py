"""Pair-local, headless friction experiments for the poker-draw scene.

This module deliberately leaves the accepted task model and full draw
controller unchanged.  An experiment-only wrapper adds one explicit MuJoCo
contact pair, allowing table-card sliding friction to vary without changing
card-pad or hand-table friction.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator
from xml.sax.saxutils import quoteattr

import mujoco
import numpy as np

from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.shared.tactile import GenesisProbeTactileProvider

from .config import _DRAW_FINGER_DEGREES, _FINGERS, _OPEN_THUMB_DEGREES
from .task import PokerDrawExecutor, PokerDrawPlanner

TABLE_CARD_PAIR_NAME = "poker_table_card_friction_pair"
CARD_GEOM_NAME = "card_core_geom"
TABLE_GEOM_NAME = "poker_table_top"
BASELINE_TABLE_CARD_FRICTION = (0.10, 0.10, 0.005, 0.0005, 0.0005)


def press_control_metadata() -> dict[str, object]:
  """Snapshot force/pose settings and controller sources for later comparisons."""
  from . import config

  settings = {}
  for name in dir(config):
    if (
      name.startswith(("_PRESS_", "_FLAT_DRAW_"))
      or name == "DEFAULT_PRESS_FORCE_PER_FINGER_N"
    ):
      value = getattr(config, name)
      settings[name] = value.tolist() if hasattr(value, "tolist") else value
  directory = Path(__file__).parent
  hashes = {
    name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
    for name in ("config.py", "task.py", "press_control.py")
  }
  return {"parameters": settings, "controller_source_sha256": hashes}


def _validate_friction(value: float) -> float:
  value = float(value)
  if not np.isfinite(value) or value < 0.0:
    raise ValueError("table-card sliding friction must be finite and non-negative")
  return value


@contextmanager
def model_with_table_card_friction(
  scene_path: str | Path,
  sliding_friction: float,
) -> Iterator[Path]:
  """Yield a temporary scene with a pair-local table-card friction override.

  The absolute include keeps all nested mesh/include resolution anchored at the
  installed poker scene.  The complete pair parameters reproduce the current
  priority-selected table contact except for the two sliding coefficients.
  """
  friction = _validate_friction(sliding_friction)
  scene = Path(scene_path).expanduser().resolve()
  if not scene.is_file():
    raise FileNotFoundError(f"poker scene does not exist: {scene}")
  mesh_directory = Path(__file__).resolve().parents[2] / "assets/workcell/meshes"
  if not mesh_directory.is_dir():
    raise FileNotFoundError(f"workcell mesh directory does not exist: {mesh_directory}")
  coefficients = (
    friction,
    friction,
    BASELINE_TABLE_CARD_FRICTION[2],
    BASELINE_TABLE_CARD_FRICTION[3],
    BASELINE_TABLE_CARD_FRICTION[4],
  )
  friction_text = " ".join(f"{value:.12g}" for value in coefficients)
  xml = (
    '<mujoco model="poker_table_card_friction_experiment">\n'
    f"  <include file={quoteattr(str(scene))}/>\n"
    f"  <compiler meshdir={quoteattr(str(mesh_directory))}/>\n"
    "  <contact>\n"
    f'    <pair name="{TABLE_CARD_PAIR_NAME}" '
    f'geom1="{CARD_GEOM_NAME}" geom2="{TABLE_GEOM_NAME}"\n'
    '          condim="3" '
    f'friction="{friction_text}" margin="0.00035" gap="0"\n'
    '          solref="0.002 1" '
    'solimp="0.98 0.995 0.0005 0.5 2"/>\n'
    "  </contact>\n"
    "</mujoco>\n"
  )
  with TemporaryDirectory(prefix="kaihand_poker_friction_") as directory:
    wrapper = Path(directory) / "scene.xml"
    wrapper.write_text(xml, encoding="utf-8")
    yield wrapper


def set_table_card_friction(model: mujoco.MjModel, sliding_friction: float) -> None:
  """Change only the explicit table-card pair's two sliding coefficients."""
  friction = _validate_friction(sliding_friction)
  pair_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, TABLE_CARD_PAIR_NAME)
  if pair_id < 0:
    raise RuntimeError(
      f"MuJoCo model is missing experimental pair {TABLE_CARD_PAIR_NAME!r}"
    )
  model.pair_friction[pair_id, :2] = friction


def theoretical_minimum_press_force(
  table_card_friction: float,
  finger_card_friction: float,
  card_mass_kg: float,
  gravity_m_s2: float,
) -> float:
  """Return the ideal Coulomb breakaway threshold for total finger normal N.

  The inequality is ``mu_finger*N >= mu_table*(m*g + N)``.  It is only a
  quasi-static reference; multi-point contact and servo transients determine
  the measured outcome.
  """
  table_mu = _validate_friction(table_card_friction)
  finger_mu = _validate_friction(finger_card_friction)
  mass = float(card_mass_kg)
  gravity = float(gravity_m_s2)
  if not np.isfinite(mass) or mass <= 0.0:
    raise ValueError("card_mass_kg must be finite and positive")
  if not np.isfinite(gravity) or gravity <= 0.0:
    raise ValueError("gravity_m_s2 must be finite and positive")
  if finger_mu <= table_mu:
    return float("inf")
  return table_mu * mass * gravity / (finger_mu - table_mu)


@dataclass(frozen=True)
class PokerFrictionTrial:
  """One deterministic short-slide condition."""

  table_card_friction: float
  press_distal_offset_degrees: float
  slide_distance_m: float = 0.135
  slide_step_m: float = 0.004
  slide_speed_m_s: float = 0.055
  settle_seconds: float = 0.08
  seed: int = 0

  def __post_init__(self) -> None:
    _validate_friction(self.table_card_friction)
    for name in (
      "press_distal_offset_degrees",
      "slide_distance_m",
      "slide_step_m",
      "slide_speed_m_s",
      "settle_seconds",
    ):
      value = float(getattr(self, name))
      if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if self.slide_distance_m <= 0.0:
      raise ValueError("slide_distance_m must be positive")
    if self.slide_step_m <= 0.0 or self.slide_step_m > self.slide_distance_m:
      raise ValueError("slide_step_m must be positive and no larger than the slide")
    if self.slide_speed_m_s <= 0.0 or self.settle_seconds < 0.0:
      raise ValueError("slide speed must be positive and settle time non-negative")


@dataclass(frozen=True)
class PokerFrictionResult:
  """Measured summary of a short table-card friction trial."""

  table_card_friction: float
  press_distal_offset_degrees: float
  finger_card_sliding_friction: float
  card_mass_kg: float
  theoretical_minimum_press_force_n: float
  initial_normal_force_mean_n: float
  initial_normal_force_peak_n: float
  slide_normal_force_mean_n: float
  slide_normal_force_peak_n: float
  table_normal_force_mean_n: float
  commanded_slide_distance_m: float
  actual_ee_displacement_m: float
  fingertip_displacement_m: float
  card_displacement_m: float
  fingertip_card_relative_slip_m: float
  card_motion_transfer_ratio: float
  lateral_card_displacement_m: float
  maximum_overhang_fraction: float
  final_overhang_fraction: float
  reached_half_overhang: bool
  four_finger_contact_fraction: float
  lost_all_finger_contact: bool
  maximum_genesis_probe_depth_m: float
  maximum_genesis_active_probes: int
  minimum_table_clearance_m: float
  minimum_supported_table_clearance_m: float
  minimum_edge_lowest_point_clearance_m: float
  minimum_table_contact_distance_m: float
  maximum_table_contact_penetration_m: float
  maximum_card_tilt_degrees: float
  observed_table_card_friction: float
  penetrated_table: bool
  outcome: str
  failure: str | None

  def as_dict(self) -> dict[str, object]:
    return asdict(self)


@dataclass
class _TrialSamples:
  press_normal: list[float]
  slide_normal: list[float]
  table_normal: list[float]
  all_four: list[bool]
  overhang: list[float]
  clearance: list[float]
  supported_clearance: list[float]
  table_contact_distance: list[float]
  tilt: list[float]
  observed_mu: list[float]
  maximum_probe_depth: float = 0.0
  maximum_active_probes: int = 0
  lost_contact_steps: int = 0

  @classmethod
  def empty(cls) -> _TrialSamples:
    return cls([], [], [], [], [], [], [], [], [], [])


class PokerFrictionExperiment:
  """Run several friction/preload trials on one headless MuJoCo model."""

  def __init__(self, simulation: ArmHandSimulation) -> None:
    if simulation.scene != "poker-draw":
      raise ValueError("PokerFrictionExperiment requires scene='poker-draw'")
    pair_id = mujoco.mj_name2id(
      simulation.model, mujoco.mjtObj.mjOBJ_PAIR, TABLE_CARD_PAIR_NAME
    )
    if pair_id < 0:
      raise ValueError("simulation was not built with model_with_table_card_friction")
    self.sim = simulation
    self._pair_id = pair_id
    self._card_body_id = simulation.model.body("card").id
    self._card_geom_id = simulation.model.geom(CARD_GEOM_NAME).id
    self._table_geom_id = simulation.model.geom(TABLE_GEOM_NAME).id
    self._pad_geom_ids = {
      simulation.model.geom(f"hand_r_{finger}_link4_tactile_pad_col").id: finger
      for finger in _FINGERS
    }
    self._pad_body_ids = np.asarray(
      [simulation.model.body(f"hand_r_{finger}_link4").id for finger in _FINGERS]
    )
    self._tactile = GenesisProbeTactileProvider(
      simulation.model,
      simulation.genesis_probe_layout,
      target_geom_names=(CARD_GEOM_NAME,),
    )

  def run(
    self, trials: tuple[PokerFrictionTrial, ...]
  ) -> tuple[PokerFrictionResult, ...]:
    if not trials:
      raise ValueError("at least one friction trial is required")
    return tuple(self.run_trial(trial) for trial in trials)

  def run_trial(self, trial: PokerFrictionTrial) -> PokerFrictionResult:
    set_table_card_friction(self.sim.model, trial.table_card_friction)
    self.sim.reset(seed=trial.seed, randomized_objects=())
    self._tactile.reset()
    samples = _TrialSamples.empty()
    collect_press = False
    collect_slide = False
    tactile_decimation = max(1, int(round(0.02 / self.sim.timestep)))
    observed_steps = 0
    executor: PokerDrawExecutor
    plan = PokerDrawPlanner(self.sim).plan("right", legacy_flat_approach=True)

    def observe(_: ArmHandSimulation, phase: str) -> None:
      nonlocal observed_steps
      if not ((collect_press and phase == "four_finger_press") or collect_slide):
        return
      observed_steps += 1
      (
        pad_normal,
        table_normal,
        contacts,
        contact_mu,
        table_contact_distance,
      ) = self._contact_state()
      if collect_slide:
        samples.slide_normal.append(pad_normal)
        samples.table_normal.append(table_normal)
        samples.lost_contact_steps = (
          samples.lost_contact_steps + 1 if not contacts else 0
        )
      else:
        samples.press_normal.append(pad_normal)
      samples.all_four.append(set(_FINGERS).issubset(contacts))
      overhang = executor._overhang_fraction(plan.table_edge_x)
      clearance = executor._card_table_clearance()
      samples.overhang.append(overhang)
      samples.clearance.append(clearance)
      if overhang < 0.02:
        samples.supported_clearance.append(clearance)
      if np.isfinite(table_contact_distance):
        samples.table_contact_distance.append(table_contact_distance)
      samples.tilt.append(executor._card_tilt_degrees())
      if np.isfinite(contact_mu):
        samples.observed_mu.append(contact_mu)
      if observed_steps % tactile_decimation == 0:
        self._tactile.read(self.sim.data)
        samples.maximum_probe_depth = max(
          samples.maximum_probe_depth,
          float(np.max(self._tactile.probe_depth, initial=0.0)),
        )
        samples.maximum_active_probes = max(
          samples.maximum_active_probes,
          int(np.count_nonzero(self._tactile.probe_contact)),
        )

    executor = PokerDrawExecutor(self.sim, observer=observe)
    start_card = self.sim.object_pose("card")
    start_ee, _ = self.sim.current_pose_matrix("right")
    start_pad_x = self._mean_pad_x()
    slide_position = start_ee.copy()
    seed = self.sim.arm_goal["right"]
    commanded = 0.0
    failure: str | None = None

    try:
      executor._set_hand_pose(_DRAW_FINGER_DEGREES, _OPEN_THUMB_DEGREES)
      for waypoint in plan.waypoints:
        self.sim.set_arm_joint_goal(plan.side, waypoint.joint_positions)
        executor._advance_fixed(waypoint.duration, waypoint.phase, plan.table_edge_x)
        if self.sim.arm_goal_error(plan.side) > 0.08:
          raise RuntimeError(f"{waypoint.phase} arm error remained above 0.08 rad")
        seed = waypoint.joint_positions
      seed, slide_position = executor._press_until_four_contacts(
        plan,
        seed,
        distal_offset_degrees=trial.press_distal_offset_degrees,
      )
      collect_press = True
      executor._advance_fixed(
        trial.settle_seconds,
        "four_finger_press",
        plan.table_edge_x,
      )
      collect_press = False

      start_card = self.sim.object_pose("card")
      start_ee, _ = self.sim.current_pose_matrix("right")
      start_pad_x = self._mean_pad_x()
      collect_slide = True
      lost_limit = max(1, int(round(0.05 / self.sim.timestep)))
      while commanded < trial.slide_distance_m:
        step = min(trial.slide_step_m, trial.slide_distance_m - commanded)
        slide_position[0] -= step
        seed = executor._move_pose(
          plan,
          slide_position,
          seed,
          duration=max(0.008, step / trial.slide_speed_m_s),
          phase="slide_card",
        )
        commanded += step
        if executor._overhang_fraction(plan.table_edge_x) >= 0.49:
          break
        if samples.lost_contact_steps >= lost_limit:
          break
    except (RuntimeError, ValueError) as error:
      failure = str(error)
    finally:
      collect_press = False
      collect_slide = False

    final_card = self.sim.object_pose("card")
    final_ee, _ = self.sim.current_pose_matrix("right")
    ee_displacement = max(0.0, float(start_ee[0] - final_ee[0]))
    pad_displacement = max(0.0, float(start_pad_x - self._mean_pad_x()))
    card_displacement = max(0.0, float(start_card[0] - final_card[0]))
    relative_slip = abs(pad_displacement - card_displacement)
    transfer = card_displacement / max(pad_displacement, 1.0e-9)
    final_overhang = executor._overhang_fraction(plan.table_edge_x)
    maximum_overhang = max(samples.overhang, default=final_overhang)
    edge_lowest_clearance = min(
      samples.clearance,
      default=executor._card_table_clearance(),
    )
    minimum_supported_clearance = min(
      samples.supported_clearance,
      default=edge_lowest_clearance,
    )
    minimum_contact_distance = min(samples.table_contact_distance, default=float("inf"))
    maximum_tilt = max(samples.tilt, default=executor._card_tilt_degrees())
    finger_mu = float(self.sim.model.geom_friction[self._card_geom_id, 0])
    card_mass = float(self.sim.model.body_mass[self._card_body_id])
    gravity = abs(float(self.sim.model.opt.gravity[2]))
    initial_force = (
      float(np.mean(samples.press_normal)) if samples.press_normal else 0.0
    )
    initial_peak = max(samples.press_normal, default=0.0)
    slide_force = float(np.mean(samples.slide_normal)) if samples.slide_normal else 0.0
    slide_peak = max(samples.slide_normal, default=0.0)
    table_force = float(np.mean(samples.table_normal)) if samples.table_normal else 0.0
    four_fraction = float(np.mean(samples.all_four)) if samples.all_four else 0.0
    observed_mu = (
      float(np.median(samples.observed_mu))
      if samples.observed_mu
      else float(self.sim.model.pair_friction[self._pair_id, 0])
    )
    lost_contact = samples.lost_contact_steps >= max(
      1, int(round(0.05 / self.sim.timestep))
    )
    reached_half = maximum_overhang >= 0.49
    card_thickness = 2.0 * float(self.sim.model.geom_size[self._card_geom_id, 2])
    # A negative lowest-corner clearance is expected when an overhanging card
    # tilts around the finite table edge.  Flag either the accepted controller's
    # fully-supported safety limit, or contact overlap exceeding the complete
    # rigid card thickness; always retain the raw contact distance alongside it.
    penetrated_table = bool(
      minimum_supported_clearance < -0.0006
      or minimum_contact_distance < -card_thickness
    )
    if failure is not None or penetrated_table:
      outcome = "failed"
    elif reached_half:
      outcome = "dragged_to_half_overhang"
    elif lost_contact and card_displacement < 0.25 * max(pad_displacement, 1.0e-9):
      outcome = "card_stuck_fingers_slipped_off"
    elif card_displacement < 0.002:
      outcome = "card_stuck"
    else:
      outcome = "partial_drag"

    return PokerFrictionResult(
      table_card_friction=trial.table_card_friction,
      press_distal_offset_degrees=trial.press_distal_offset_degrees,
      finger_card_sliding_friction=finger_mu,
      card_mass_kg=card_mass,
      theoretical_minimum_press_force_n=theoretical_minimum_press_force(
        trial.table_card_friction,
        finger_mu,
        card_mass,
        gravity,
      ),
      initial_normal_force_mean_n=initial_force,
      initial_normal_force_peak_n=initial_peak,
      slide_normal_force_mean_n=slide_force,
      slide_normal_force_peak_n=slide_peak,
      table_normal_force_mean_n=table_force,
      commanded_slide_distance_m=commanded,
      actual_ee_displacement_m=ee_displacement,
      fingertip_displacement_m=pad_displacement,
      card_displacement_m=card_displacement,
      fingertip_card_relative_slip_m=relative_slip,
      card_motion_transfer_ratio=transfer,
      lateral_card_displacement_m=abs(float(final_card[1] - start_card[1])),
      maximum_overhang_fraction=maximum_overhang,
      final_overhang_fraction=final_overhang,
      reached_half_overhang=reached_half,
      four_finger_contact_fraction=four_fraction,
      lost_all_finger_contact=lost_contact,
      maximum_genesis_probe_depth_m=samples.maximum_probe_depth,
      maximum_genesis_active_probes=samples.maximum_active_probes,
      minimum_table_clearance_m=minimum_supported_clearance,
      minimum_supported_table_clearance_m=minimum_supported_clearance,
      minimum_edge_lowest_point_clearance_m=edge_lowest_clearance,
      minimum_table_contact_distance_m=minimum_contact_distance,
      maximum_table_contact_penetration_m=max(0.0, -minimum_contact_distance),
      maximum_card_tilt_degrees=maximum_tilt,
      observed_table_card_friction=observed_mu,
      penetrated_table=penetrated_table,
      outcome=outcome,
      failure=failure,
    )

  def _mean_pad_x(self) -> float:
    return float(np.mean(self.sim.data.xpos[self._pad_body_ids, 0]))

  def _contact_state(self) -> tuple[float, float, set[str], float, float]:
    pad_normal = 0.0
    table_normal = 0.0
    contacts: set[str] = set()
    contact_friction: list[float] = []
    table_contact_distance: list[float] = []
    wrench = np.zeros(6, dtype=float)
    for contact_id in range(self.sim.data.ncon):
      contact = self.sim.data.contact[contact_id]
      geom1 = int(contact.geom1)
      geom2 = int(contact.geom2)
      pair = {geom1, geom2}
      if self._card_geom_id not in pair:
        continue
      other = geom2 if geom1 == self._card_geom_id else geom1
      mujoco.mj_contactForce(self.sim.model, self.sim.data, contact_id, wrench)
      normal = max(0.0, float(wrench[0]))
      if other == self._table_geom_id:
        table_normal += normal
        contact_friction.append(float(contact.friction[0]))
        table_contact_distance.append(float(contact.dist))
      elif other in self._pad_geom_ids:
        pad_normal += normal
        contacts.add(self._pad_geom_ids[other])
    observed_mu = (
      float(np.median(contact_friction)) if contact_friction else float("nan")
    )
    minimum_distance = (
      min(table_contact_distance) if table_contact_distance else float("inf")
    )
    return pad_normal, table_normal, contacts, observed_mu, minimum_distance


def default_friction_trials() -> tuple[PokerFrictionTrial, ...]:
  """Return the bounded six-condition experiment used by the CLI."""
  return tuple(
    PokerFrictionTrial(mu, preload)
    for mu in (0.10, 0.90, 1.15, 1.30)
    for preload in (0.0, 2.0)
  )


__all__ = [
  "BASELINE_TABLE_CARD_FRICTION",
  "PokerFrictionExperiment",
  "PokerFrictionResult",
  "PokerFrictionTrial",
  "TABLE_CARD_PAIR_NAME",
  "default_friction_trials",
  "model_with_table_card_friction",
  "set_table_card_friction",
  "theoretical_minimum_press_force",
]
