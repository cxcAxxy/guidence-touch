"""Queryable collection profiles for the independent KaiHand simulation tasks.

The public collector owns process isolation, deterministic episode numbering,
resume, and publication.  This module owns the task-specific recorder command
and the recorded-artifact success gate.  Keeping those two pieces together
makes the production profile inspectable without changing any task controller,
camera clock, tactile provider, or success rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from kaihand_tactile_env.pipeline.catalog import TASKS

COLLECTION_CAMERAS = ("head", "left_wrist", "right_wrist")
COLLECTION_CAMERA_HZ = 30
POKER_ACCEPTANCE_POLICY = "task-completion-v1"

DEFAULT_PROFILE = "production-v1"
PROFILE_ALIASES = {
  "production-v1": DEFAULT_PROFILE,
  "legacy-compatible": DEFAULT_PROFILE,
}
PROFILE_NAMES = tuple(PROFILE_ALIASES)

CommandBuilder = Callable[[Path, str, Path, int, bool], list[str]]
OutcomeValidator = Callable[[Path], bool]


def _read(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def _record_dataset_command(
  root: Path, python: str, output: Path, seed: int, _fixed_stains: bool, *, task: str
) -> list[str]:
  command = [
    python,
    str(root / "scripts/workcell/record_dataset.py"),
    "--scene",
    task,
    "--output-dir",
    str(output),
    "--episodes",
    "1",
    "--workers",
    "1",
    "--seed",
    str(seed),
    "--camera-hz",
    str(COLLECTION_CAMERA_HZ),
    "--cameras",
    *COLLECTION_CAMERAS,
    "--rgb-only",
  ]
  if task == "pick-place":
    return command + ["--object", "cylinder", "--side", "right"]
  return command + [
    "--preset",
    "middle-force-precontact-v1",
    "--tactile-links",
    "right",
    "--acceptance-policy",
    POKER_ACCEPTANCE_POLICY,
  ]


def _pick_place_command(
  root: Path, python: str, output: Path, seed: int, fixed_stains: bool
) -> list[str]:
  return _record_dataset_command(
    root, python, output, seed, fixed_stains, task="pick-place"
  )


def _poker_command(
  root: Path, python: str, output: Path, seed: int, fixed_stains: bool
) -> list[str]:
  return _record_dataset_command(
    root, python, output, seed, fixed_stains, task="poker-draw"
  )


def _usb_command(
  root: Path, python: str, output: Path, seed: int, _fixed_stains: bool
) -> list[str]:
  return [
    python,
    str(root / "scripts/workcell/record_usb_dataset.py"),
    "--output-dir",
    str(output),
    "--episodes",
    "1",
    "--workers",
    "1",
    "--seed",
    str(seed),
    "--motion-profile",
    "fast",
    "--camera-hz",
    str(COLLECTION_CAMERA_HZ),
    "--cameras",
    *COLLECTION_CAMERAS,
    "--xy-jitter-mm",
    "10",
    "--yaw-jitter-deg",
    "5",
    "--precontact-noise-mm",
    "0.5",
  ]


def _bulb_command(
  root: Path, python: str, output: Path, seed: int, _fixed_stains: bool
) -> list[str]:
  return [
    python,
    str(root / "scripts/workcell/refresh_light_bulb_example.py"),
    "--position-seed",
    str(seed),
    "--camera-hz",
    str(COLLECTION_CAMERA_HZ),
    "--buffer-rows",
    "128",
    "--cameras",
    *COLLECTION_CAMERAS,
    "--raw-only",
    "--output",
    str(output),
  ]


def _ram_command(
  root: Path, python: str, output: Path, seed: int, _fixed_stains: bool
) -> list[str]:
  return [
    python,
    str(root / "scripts/workcell/record_install_ram_example.py"),
    "--position-seed",
    str(seed),
    "--camera-hz",
    str(COLLECTION_CAMERA_HZ),
    "--buffer-rows",
    "64",
    "--cameras",
    *COLLECTION_CAMERAS,
    "--raw-only",
    "--output-dir",
    str(output),
  ]


def _whiteboard_command(
  root: Path, python: str, output: Path, seed: int, fixed_stains: bool
) -> list[str]:
  command = [
    python,
    str(root / "scripts/workcell/record_whiteboard_example.py"),
    "--output-dir",
    str(output),
    "--camera-hz",
    str(COLLECTION_CAMERA_HZ),
    "--buffer-rows",
    "128",
    "--cameras",
    *COLLECTION_CAMERAS,
    "--raw-only",
  ]
  return command if fixed_stains else command + ["--ink-seed", str(seed)]


def _vase_command(
  root: Path, python: str, output: Path, seed: int, fixed_stains: bool
) -> list[str]:
  command = [
    python,
    str(root / "scripts/workcell/view_vase_wipe.py"),
    "--headless",
    "--run-task",
    "--output-dir",
    str(output),
    "--camera-hz",
    str(COLLECTION_CAMERA_HZ),
    "--buffer-rows",
    "128",
    "--cameras",
    *COLLECTION_CAMERAS,
    "--raw-only",
  ]
  return command if fixed_stains else command + ["--stain-seed", str(seed)]


def _sponge_command(
  root: Path, python: str, output: Path, seed: int, _fixed_stains: bool
) -> list[str]:
  return [
    python,
    str(root / "scripts/workcell/record_sponge_grasp.py"),
    "--output-dir",
    str(output),
    "--seed",
    str(seed),
    "--camera-hz",
    str(COLLECTION_CAMERA_HZ),
    "--buffer-rows",
    "128",
    "--raw-only",
    "--cameras",
    *COLLECTION_CAMERAS,
  ]


def _new_task_command(root, python, output, seed, fixed_stains, *, task):
  command = [
    python, str(root / "scripts/workcell/record_task_variant.py"),
    "--task", task, "--output-dir", str(output),
    "--camera-hz", str(COLLECTION_CAMERA_HZ), "--buffer-rows", "128",
    "--cameras", *COLLECTION_CAMERAS,
  ]
  if not fixed_stains or task not in {
    "whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line",
  }:
    command += ["--seed", str(seed)]
  return command


def _unplug_command(root, python, output, seed, fixed_stains):
  return _new_task_command(root, python, output, seed, fixed_stains, task="usb-unplug")


def _horizontal_command(root, python, output, seed, fixed_stains):
  return _new_task_command(
    root, python, output, seed, fixed_stains, task="whiteboard-wipe-horizontal"
  )


def _horizontal_line_command(root, python, output, seed, fixed_stains):
  return _new_task_command(
    root, python, output, seed, fixed_stains, task="whiteboard-wipe-horizontal-line"
  )


def _unscrew_command(root, python, output, seed, fixed_stains):
  return _new_task_command(root, python, output, seed, fixed_stains, task="bulb-unscrew")


def _flat_episode_outcome(output: Path) -> bool:
  manifests = [path.with_suffix(".json") for path in output.glob("*.h5")]
  return (
    len(manifests) == 1
    and _read(manifests[0])["outcome"].get("success") is True
  )


def _sponge_outcome(output: Path) -> bool:
  raw = output / "raw/sponge_grasp_000000.h5"
  result = _read(raw.with_suffix(".result.json"))
  manifest = _read(raw.with_suffix(".json"))
  return (
    raw.is_file()
    and result.get("success") is True
    and manifest.get("validation", {}).get("valid") is True
    and manifest.get("task_audit_passed") is True
  )


def _shared_result_outcome(output: Path) -> bool:
  result = _read(output / "result.json")
  validation = _read(output / "raw/episode.json").get("validation", {})
  return result.get("success") is True and validation.get("valid") is True


def _variant_outcome(output: Path) -> bool:
  from kaihand_tactile_env.shared.task_variants import validate_variant_outcome

  result = _read(output / "result.json")
  try:
    validate_variant_outcome(result.get("task"), result)
  except (ValueError, TypeError):
    return False
  sidecar = _read(output / "raw/episode.json")
  return bool(
    sidecar.get("outcome") == result
    and sidecar.get("validation", {}).get("valid") is True
    and sidecar.get("task_audit_passed") is True
    and result.get("physics_recording_audit", {}).get("passed") is True
  )


def _usb_outcome(output: Path) -> bool:
  report = _read(output / "summary.json")
  return report.get("complete") is True and report.get("successful_episodes") == 1


def _raw_sidecar_outcome(output: Path) -> bool:
  raw = next(output.glob("raw/*.h5"))
  result = _read(raw.with_suffix(".result.json"))
  validation = _read(raw.with_suffix(".json")).get("validation", {})
  return result.get("success") is True and validation.get("valid") is True


@dataclass(frozen=True)
class CollectionSpec:
  """One task's immutable production command and recorded-outcome contract."""

  task: str
  backend: str
  command_builder: CommandBuilder
  outcome_validator: OutcomeValidator
  outcome_contract: str
  settings: Mapping[str, object]
  supports_fixed_stains: bool = False

  def build_command(
    self,
    root: Path,
    python: str,
    output: Path,
    seed: int,
    *,
    fixed_stains: bool = False,
    profile: str = DEFAULT_PROFILE,
  ) -> list[str]:
    canonical_profile(profile)
    return self.command_builder(root, python, output, seed, fixed_stains)

  def resolved_profile(
    self,
    *,
    profile: str = DEFAULT_PROFILE,
    fixed_stains: bool = False,
  ) -> dict[str, object]:
    profile_id = canonical_profile(profile)
    seed_mode = "episode-derived"
    if self.supports_fixed_stains and fixed_stains:
      seed_mode = "fixed-task-layout"
    return {
      "profile_id": profile_id,
      "compatibility_aliases": [
        name for name, target in PROFILE_ALIASES.items() if target == profile_id
      ],
      "task": self.task,
      "backend": self.backend,
      "cameras": list(COLLECTION_CAMERAS),
      "camera_hz": COLLECTION_CAMERA_HZ,
      "seed_mode": seed_mode,
      "supports_fixed_stains": self.supports_fixed_stains,
      "settings": dict(self.settings),
      "outcome_contract": self.outcome_contract,
      "validation_source": "recorded task outcome and Raw sidecar",
      "runtime_contract": {
        "authority": "episode-hdf5-root-attribute",
        "attribute": "runtime_contract_json",
        "missing_value": "unknown",
      },
    }

  def validate_outcome(self, output: Path) -> bool:
    # Do not manufacture a success from profile metadata.  Every accepted
    # attempt must contain Raw HDF5 and pass its recorder-owned result gate.
    if not any(output.rglob("*.h5")):
      return False
    return self.outcome_validator(output)


COLLECTION_SPECS = {
  "pick-place": CollectionSpec(
    "pick-place",
    "record_dataset.py",
    _pick_place_command,
    _flat_episode_outcome,
    "flat-hdf5-sidecar-outcome-v1",
    {"episodes": 1, "workers": 1, "rgb_only": True, "object": "cylinder", "side": "right"},
  ),
  "poker-draw": CollectionSpec(
    "poker-draw",
    "record_dataset.py",
    _poker_command,
    _flat_episode_outcome,
    "flat-hdf5-sidecar-outcome-v1",
    {
      "episodes": 1,
      "workers": 1,
      "rgb_only": True,
      "preset": "middle-force-precontact-v1",
      "tactile_links": "right",
      "acceptance_policy": POKER_ACCEPTANCE_POLICY,
    },
  ),
  "usb-insert": CollectionSpec(
    "usb-insert",
    "record_usb_dataset.py",
    _usb_command,
    _usb_outcome,
    "usb-summary-single-success-v1",
    {
      "episodes": 1,
      "workers": 1,
      "motion_profile": "fast",
      "xy_jitter_mm": 10,
      "yaw_jitter_deg": 5,
      "precontact_noise_mm": 0.5,
    },
  ),
  "bulb-screw": CollectionSpec(
    "bulb-screw",
    "refresh_light_bulb_example.py",
    _bulb_command,
    _raw_sidecar_outcome,
    "raw-result-sidecar-validation-v1",
    {"raw_only": True, "buffer_rows": 128, "position_seed": "episode-derived"},
  ),
  "vase-wipe": CollectionSpec(
    "vase-wipe",
    "view_vase_wipe.py",
    _vase_command,
    _shared_result_outcome,
    "shared-result-and-raw-validation-v1",
    {"raw_only": True, "buffer_rows": 128, "headless": True, "run_task": True},
    supports_fixed_stains=True,
  ),
  "install-ram": CollectionSpec(
    "install-ram",
    "record_install_ram_example.py",
    _ram_command,
    _raw_sidecar_outcome,
    "raw-result-sidecar-validation-v1",
    {"raw_only": True, "buffer_rows": 64, "position_seed": "episode-derived"},
  ),
  "whiteboard-wipe": CollectionSpec(
    "whiteboard-wipe",
    "record_whiteboard_example.py",
    _whiteboard_command,
    _shared_result_outcome,
    "shared-result-and-raw-validation-v1",
    {"raw_only": True, "buffer_rows": 128},
    supports_fixed_stains=True,
  ),
  "sponge-grasp": CollectionSpec(
    "sponge-grasp",
    "record_sponge_grasp.py",
    _sponge_command,
    _sponge_outcome,
    "sponge-task-audit-and-raw-validation-v1",
    {"raw_only": True, "buffer_rows": 128, "seed": "episode-derived"},
  ),
  "usb-unplug": CollectionSpec(
    "usb-unplug", "record_task_variant.py", _unplug_command,
    _variant_outcome, "unplug-physical-clear-place-release-v1",
    {"raw_only": True, "buffer_rows": 128, "initial_state": "installed"},
  ),
  "whiteboard-wipe-horizontal": CollectionSpec(
    "whiteboard-wipe-horizontal", "record_task_variant.py", _horizontal_command,
    _variant_outcome, "horizontal-clean-lift-table-release-v1",
    {"raw_only": True, "buffer_rows": 128, "board_pose": "horizontal-face-up"},
    supports_fixed_stains=True,
  ),
  "whiteboard-wipe-horizontal-line": CollectionSpec(
    "whiteboard-wipe-horizontal-line", "record_task_variant.py", _horizontal_line_command,
    _variant_outcome, "horizontal-line-clean-lift-table-release-v1",
    {"raw_only": True, "buffer_rows": 128, "board_pose": "horizontal-face-up",
     "ink_pattern": "single-top-horizontal-line", "ink_segments": 8},
    supports_fixed_stains=True,
  ),
  "bulb-unscrew": CollectionSpec(
    "bulb-unscrew", "record_task_variant.py", _unscrew_command,
    _variant_outcome, "unscrew-physical-disengage-table-release-v2",
    {"raw_only": True, "buffer_rows": 128, "initial_state": "installed",
     "placement": "ordinary-tabletop-without-cradle"},
  ),
}

if tuple(COLLECTION_SPECS) != TASKS:
  raise RuntimeError("collection spec order differs from the public task order")


def canonical_profile(profile: str) -> str:
  try:
    return PROFILE_ALIASES[profile]
  except KeyError as error:
    raise ValueError(
      f"unknown collection profile {profile!r}; expected one of {PROFILE_NAMES}"
    ) from error


def collection_spec(task: str) -> CollectionSpec:
  try:
    return COLLECTION_SPECS[task]
  except KeyError as error:
    raise ValueError(f"unknown collection task {task!r}; expected one of {TASKS}") from error


def build_collection_command(
  task: str,
  root: Path,
  python: str,
  output: Path,
  seed: int,
  *,
  fixed_stains: bool = False,
  profile: str = DEFAULT_PROFILE,
) -> list[str]:
  return collection_spec(task).build_command(
    root,
    python,
    output,
    seed,
    fixed_stains=fixed_stains,
    profile=profile,
  )


def validate_collection_outcome(task: str, output: Path) -> bool:
  return collection_spec(task).validate_outcome(output)


def recorded_runtime_contracts(output: Path) -> dict[str, object]:
  """Read recorder-authored runtime contracts without inferring physics values."""
  import h5py

  attribute = "runtime_contract_json"
  artifacts = {}
  for path in sorted(output.rglob("*.h5")):
    relative = str(path.relative_to(output))
    entry: dict[str, object] = {"status": "unknown"}
    try:
      with h5py.File(path, "r") as file:
        raw = file.attrs.get(attribute)
      if raw is not None:
        if isinstance(raw, bytes):
          raw = raw.decode("utf-8")
        payload = json.loads(str(raw))
        if isinstance(payload, dict):
          entry = {"status": "recorded", "contract": payload}
    except (OSError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
      # Historical or test artifacts may not carry the new root attribute.
      # Unknown is explicit and never replaced with guessed physics values.
      pass
    artifacts[relative] = entry
  return {
    "source": "episode-hdf5-root-attribute",
    "attribute": attribute,
    "artifacts": artifacts,
  }


def resolved_collection_profile(
  task: str,
  *,
  profile: str = DEFAULT_PROFILE,
  fixed_stains: bool = False,
) -> dict[str, object]:
  return collection_spec(task).resolved_profile(
    profile=profile, fixed_stains=fixed_stains
  )


def collection_catalog(
  *, profile: str = DEFAULT_PROFILE, fixed_stains: bool = False
) -> dict[str, object]:
  profile_id = canonical_profile(profile)
  return {
    "default_profile": DEFAULT_PROFILE,
    "profiles": [
      {"name": name, "resolves_to": target}
      for name, target in PROFILE_ALIASES.items()
    ],
    "tasks": [
      resolved_collection_profile(
        task, profile=profile_id, fixed_stains=fixed_stains
      )
      for task in TASKS
    ],
  }


def describe_collection_task(
  task: str,
  *,
  profile: str = DEFAULT_PROFILE,
  fixed_stains: bool = False,
) -> dict[str, object]:
  return resolved_collection_profile(
    task, profile=profile, fixed_stains=fixed_stains
  )
