"""Verify frozen pi0.5 inference bytes before a policy server starts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from .policy_tasks import TASK_INSTRUCTIONS
from .task_variants import VARIANT_TASKS

_SHA256 = re.compile(r"[0-9a-f]{64}")
SHARED_DEPLOYMENT_SCHEMA_V1 = "shared_task_pi05_deployment_v1"
SHARED_DEPLOYMENT_SCHEMA_V2 = "shared_task_pi05_deployment_v2"
POKER_DEPLOYMENT_SCHEMA_V1 = "poker_pi05_deployment_v1"
POKER_DEPLOYMENT_SCHEMA_V2 = "poker_pi05_deployment_v2"
POKER_INSTRUCTION = (
  "Slide the face-down card toward the table edge, pinch it between the fingers "
  "and thumb, lift it, and turn its face toward the robot to look at it."
)
TRAINING_PROVENANCE_SCHEMA = "kaihand_training_provenance_v1"
TRAINING_PROVENANCE_FILENAME = "training_provenance.json"
CONVERSION_SNAPSHOT_FILENAME = "conversion_manifest.snapshot.json"
LEGACY_ACTION_CONTRACT = "legacy-measured-arm-hybrid-v1"
RECORDED_TARGET_ACTION_CONTRACT = "recorded-controller-target-v1"
LEGACY_KAIHAND_CONFIG = "pi05_kaihand"
KAIHAND_H50_CONFIG = "pi05_kaihand_h50"
BULB_CONTROLLER_TARGET_CONFIG = "pi05_kaihand_bulb_controller_target_v1"
BULB_CONTROLLER_TARGET_H50_CONFIG = "pi05_kaihand_bulb_controller_target_h50_v1"
BULB_CONTROLLER_TARGET_CONFIGS = frozenset(
  {BULB_CONTROLLER_TARGET_CONFIG, BULB_CONTROLLER_TARGET_H50_CONFIG}
)
SHARED_KAIHAND_ACTION_HORIZONS = {
  LEGACY_KAIHAND_CONFIG: 30,
  KAIHAND_H50_CONFIG: 50,
  BULB_CONTROLLER_TARGET_CONFIG: 30,
  BULB_CONTROLLER_TARGET_H50_CONFIG: 50,
}
FORMAL_ACTION_INTERPOLATION = "linear"
_CONVERSION_MANIFEST_SCHEMAS = {
  "kaihand_fast_pi05_conversion.json": (
    "schema",
    "kaihand_fast_pi05_video_v2",
  ),
  "kaihand_card_conversion_manifest.json": (
    "schema_version",
    "kaihand_card_lerobot_v2_1_conversion_v2",
  ),
}
_FAST_METADATA_ARTIFACTS = frozenset(
  {
    "meta/episodes.jsonl",
    "meta/episodes_stats.jsonl",
    "meta/info.json",
    "meta/kaihand_source_episodes.jsonl",
    "meta/tasks.jsonl",
  }
)
SHARED_DEPLOYMENT_V1_ID_FIELDS = (
  "task",
  "model_family",
  "model_config_name",
  "normalizer_asset_id",
  "checkpoint_path",
  "checkpoint_sha256",
  "checkpoint_step",
  "prediction_horizon",
  "model_action_dim",
  "action_dim",
  "control_hz",
  "suggested_execute_steps",
  "joint_names",
  "instruction",
  "observation_contract",
  "action_representation",
  "model_project_path",
  "openpi_source_sha256",
  "evaluation_source_sha256",
)
SHARED_DEPLOYMENT_V2_ID_FIELDS = (
  *SHARED_DEPLOYMENT_V1_ID_FIELDS,
  "training_provenance",
  "required_action_interpolation",
  "diagnostic_only",
  "formal_metrics_valid",
)
# Historical callers import this name when constructing v1 fixtures.  Keep it
# as an explicit compatibility alias; new builders must use the v2 tuple.
SHARED_DEPLOYMENT_ID_FIELDS = SHARED_DEPLOYMENT_V1_ID_FIELDS
POKER_DEPLOYMENT_V1_ID_FIELDS = (
  "task", "model_family", "model_config_name", "normalizer_asset_id",
  "checkpoint_path", "checkpoint_sha256", "checkpoint_step",
  "prediction_horizon", "model_action_dim", "action_dim", "joint_names",
  "observation_contract", "action_representation",
)
POKER_DEPLOYMENT_V2_ID_FIELDS = (
  *POKER_DEPLOYMENT_V1_ID_FIELDS,
  "training_provenance", "control_hz", "suggested_execute_steps", "instruction",
  "model_project_path", "openpi_source_sha256", "evaluation_source_sha256",
  "required_action_interpolation", "diagnostic_only", "formal_metrics_valid",
)


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
  try:
    value = json.loads(payload)
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise RuntimeError(f"{label} is not valid JSON") from error
  if not isinstance(value, dict):
    raise RuntimeError(f"{label} must contain one JSON object")
  return value


def _require_sha256(value: object, *, label: str) -> str:
  if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
    raise RuntimeError(f"{label} must be a lowercase hexadecimal SHA-256")
  return value


def _normalizer_semantic_sha256(path: Path) -> str:
  """Reproduce OpenPI's value-level normalizer identity without importing it."""
  payload, _ = stable_file_bytes(path)
  document = _json_object(payload, label="normalizer statistics")
  norm_stats = document.get("norm_stats")
  if not isinstance(norm_stats, dict) or set(norm_stats) != {"state", "actions"}:
    raise RuntimeError("normalizer must contain exactly state/actions statistics")
  semantic: dict[str, Any] = {}
  for key in ("state", "actions"):
    source = norm_stats[key]
    if not isinstance(source, dict):
      raise RuntimeError(f"normalizer {key!r} statistics must be an object")
    semantic[key] = {}
    for statistic in ("mean", "std", "q01", "q99"):
      values = source.get(statistic)
      if (
        not isinstance(values, list)
        or len(values) != 27
        or any(
          isinstance(value, bool)
          or not isinstance(value, int | float)
          or not (-float("inf") < float(value) < float("inf"))
          for value in values
        )
      ):
        raise RuntimeError(
          f"normalizer {key}.{statistic} must contain 27 finite numbers"
        )
      semantic[key][statistic] = {
        "shape": [27],
        "values": values,
      }
  return _canonical_hash(semantic)


def _conversion_manifest_repo_id(manifest: dict[str, Any], filename: str) -> object:
  if filename == "kaihand_fast_pi05_conversion.json":
    return manifest.get("repo_id")
  lerobot = manifest.get("lerobot")
  return lerobot.get("repo_id") if isinstance(lerobot, dict) else None


def _snapshot_artifact_identity(value: object, *, label: str) -> dict[str, Any]:
  if not isinstance(value, dict):
    raise RuntimeError(f"{label} identity must be an object")
  size = value.get("size_bytes")
  digest = value.get("sha256")
  if not isinstance(size, int) or isinstance(size, bool) or size < 0:
    raise RuntimeError(f"{label} size_bytes must be a nonnegative integer")
  _require_sha256(digest, label=f"{label} SHA-256")
  return {"size_bytes": size, "sha256": digest}


def _fast_snapshot_artifact_aggregate(snapshot: dict[str, Any]) -> dict[str, Any]:
  """Recompute OpenPI's loader-visible dataset identity from its snapshot."""
  cameras = snapshot.get("cameras")
  if (
    not isinstance(cameras, list)
    or not cameras
    or len(set(cameras)) != len(cameras)
    or any(
      not isinstance(camera, str)
      or not camera
      or PurePosixPath(camera).name != camera
      or camera in {".", ".."}
      for camera in cameras
    )
  ):
    raise RuntimeError("fast conversion snapshot cameras are invalid")
  metadata = snapshot.get("metadata_artifacts")
  if not isinstance(metadata, dict) or set(metadata) != _FAST_METADATA_ARTIFACTS:
    raise RuntimeError("fast conversion snapshot metadata artifact set is incomplete")
  rows = [
    {
      "path": relative,
      **_snapshot_artifact_identity(
        metadata[relative], label=f"metadata artifact {relative}"
      ),
    }
    for relative in sorted(_FAST_METADATA_ARTIFACTS)
  ]
  episodes = snapshot.get("episodes")
  total_episodes = snapshot.get("total_episodes")
  if (
    not isinstance(episodes, list)
    or not isinstance(total_episodes, int)
    or isinstance(total_episodes, bool)
    or total_episodes <= 0
    or len(episodes) != total_episodes
  ):
    raise RuntimeError("fast conversion snapshot episode totals are invalid")
  contract = snapshot.get("action_contract")
  contract_id = contract.get("id") if isinstance(contract, dict) else None
  indices = []
  expected_names = {
    "data.parquet",
    "stats.json",
    *(f"{camera}.mp4" for camera in cameras),
  }
  for row_number, episode in enumerate(episodes):
    if not isinstance(episode, dict):
      raise RuntimeError(f"fast conversion episode {row_number} is invalid")
    index = episode.get("output_episode_index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
      raise RuntimeError(f"fast conversion episode {row_number} index is invalid")
    indices.append(index)
    if episode.get("action_contract") != contract_id:
      raise RuntimeError(f"fast conversion episode {index} action contract differs")
    artifacts = episode.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected_names:
      raise RuntimeError(f"fast conversion episode {index} artifact set differs")
    identities = {
      name: _snapshot_artifact_identity(
        identity, label=f"episode {index} artifact {name}"
      )
      for name, identity in artifacts.items()
    }
    chunk = index // 1000
    rows.append(
      {
        "path": f"data/chunk-{chunk:03d}/episode_{index:06d}.parquet",
        **identities["data.parquet"],
      }
    )
    for camera in cameras:
      rows.append(
        {
          "path": (
            f"videos/chunk-{chunk:03d}/observation.images.{camera}/"
            f"episode_{index:06d}.mp4"
          ),
          **identities[f"{camera}.mp4"],
        }
      )
  if indices != list(range(total_episodes)):
    raise RuntimeError("fast conversion episode indices are not contiguous and ordered")
  rows.sort(key=lambda row: row["path"])
  return {
    "algorithm": "sha256",
    "file_count": len(rows),
    "total_size_bytes": sum(row["size_bytes"] for row in rows),
    "canonical_sha256": _canonical_hash(rows),
  }


def current_bulb_environment() -> dict[str, str]:
  """Return the evaluation mechanics identity that formal Bulb must match."""
  from kaihand_tactile_env.shared.config import default_model_path, model_fingerprint
  from kaihand_tactile_env.tasks.bulb_screw import config

  return {
    "mechanics_version": config.MECHANICS_VERSION,
    "model_fingerprint": model_fingerprint(default_model_path("bulb-screw")),
  }


def action_representation_from_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
  """Return the inference action semantics authenticated by a checkpoint."""
  contract = provenance.get("action_contract")
  if not isinstance(contract, dict):
    raise RuntimeError("training provenance action_contract must be an object")
  values = {
    "contract_id": contract.get("id"),
    "arm_0_6": contract.get("arm_0_6"),
    "hand_7_26": contract.get("hand_7_26"),
    "policy_output_is_absolute": True,
  }
  if not all(
    isinstance(value, str) and value for value in values.values() if value is not True
  ):
    raise RuntimeError("training provenance action semantics are incomplete")
  return values


def load_checkpoint_training_provenance(
  checkpoint: Path,
) -> dict[str, Any]:
  """Authenticate the two checkpoint provenance assets and normalizer semantics."""
  root = checkpoint.expanduser().resolve(strict=True)
  assets = root / "assets"
  provenance_path = assets / TRAINING_PROVENANCE_FILENAME
  snapshot_path = assets / CONVERSION_SNAPSHOT_FILENAME
  provenance_bytes, provenance_sha256 = stable_file_bytes(provenance_path)
  snapshot_bytes, snapshot_sha256 = stable_file_bytes(snapshot_path)
  provenance = _json_object(provenance_bytes, label="training provenance")
  snapshot = _json_object(snapshot_bytes, label="conversion manifest snapshot")

  if provenance.get("schema") != TRAINING_PROVENANCE_SCHEMA:
    raise RuntimeError(
      "training provenance schema must be "
      f"{TRAINING_PROVENANCE_SCHEMA!r}, got {provenance.get('schema')!r}"
    )
  if provenance.get("binding_status") != "bound":
    raise RuntimeError("training provenance must declare binding_status='bound'")
  conversion = provenance.get("conversion_manifest")
  if not isinstance(conversion, dict):
    raise RuntimeError("training provenance conversion_manifest must be an object")
  filename = conversion.get("filename")
  if filename not in _CONVERSION_MANIFEST_SCHEMAS:
    raise RuntimeError(f"unsupported conversion manifest filename: {filename!r}")
  schema_key, expected_schema = _CONVERSION_MANIFEST_SCHEMAS[filename]
  if conversion.get("schema") != expected_schema:
    raise RuntimeError("training provenance conversion manifest schema is invalid")
  if snapshot.get(schema_key) != expected_schema:
    raise RuntimeError("conversion snapshot schema differs from training provenance")
  if conversion.get("size_bytes") != len(snapshot_bytes):
    raise RuntimeError("conversion snapshot size differs from training provenance")
  if conversion.get("sha256") != snapshot_sha256:
    raise RuntimeError("conversion snapshot SHA-256 differs from training provenance")

  task = provenance.get("task")
  config_name = provenance.get("config_name")
  if not isinstance(task, str) or not task:
    raise RuntimeError("training provenance task must be a nonempty string")
  if not isinstance(config_name, str) or not config_name:
    raise RuntimeError("training provenance config_name must be a nonempty string")
  if config_name not in SHARED_KAIHAND_ACTION_HORIZONS:
    raise RuntimeError(f"unsupported shared KaiHand training config: {config_name!r}")
  if snapshot.get("task") != task:
    raise RuntimeError("conversion snapshot task differs from training provenance")
  if task in VARIANT_TASKS:
    instruction_key = (
      "instruction"
      if filename == "kaihand_fast_pi05_conversion.json"
      else "task_instruction"
    )
    if snapshot.get(instruction_key) != TASK_INSTRUCTIONS[task]:
      raise RuntimeError(
        "additional-task conversion snapshot instruction differs from the "
        "frozen task prompt"
      )
  contract = provenance.get("action_contract")
  if not isinstance(contract, dict) or snapshot.get("action_contract") != contract:
    raise RuntimeError("conversion snapshot action contract is not exactly bound")
  if provenance.get("action_contract_sha256") != _canonical_hash(contract):
    raise RuntimeError("training provenance action contract SHA-256 is invalid")
  contract_id = contract.get("id")
  if contract_id not in {LEGACY_ACTION_CONTRACT, RECORDED_TARGET_ACTION_CONTRACT}:
    raise RuntimeError(f"unsupported checkpoint action contract: {contract_id!r}")
  if contract.get("legacy") is not (contract_id == LEGACY_ACTION_CONTRACT):
    raise RuntimeError("checkpoint action contract legacy flag disagrees with its id")
  if contract.get("dimensions") != 27:
    raise RuntimeError("checkpoint action contract must have 27 dimensions")
  joint_order = contract.get("joint_order")
  if (
    not isinstance(joint_order, list)
    or len(joint_order) != 27
    or len(set(joint_order)) != 27
    or not all(isinstance(name, str) and name for name in joint_order)
  ):
    raise RuntimeError("checkpoint action contract must have 27 unique joints")
  execution = contract.get("evaluation_execution")
  if not isinstance(execution, dict):
    raise RuntimeError("checkpoint action execution contract must be an object")
  expected_execution = {
    "endpoint_rate_hz": 30,
    "physical_substep_interpolation_required": FORMAL_ACTION_INTERPOLATION,
    "zero_order_hold_equivalent": False,
  }
  mismatches = {
    key: {"expected": value, "actual": execution.get(key)}
    for key, value in expected_execution.items()
    if execution.get(key) != value
  }
  if mismatches:
    raise RuntimeError(f"checkpoint action execution contract mismatch: {mismatches}")

  dataset = provenance.get("dataset")
  if not isinstance(dataset, dict):
    raise RuntimeError("training provenance dataset must be an object")
  manifest_repo_id = _conversion_manifest_repo_id(snapshot, filename)
  training_repo_id = dataset.get("repo_id")
  if (
    not isinstance(manifest_repo_id, str)
    or not manifest_repo_id
    or dataset.get("manifest_repo_id") != manifest_repo_id
    or not isinstance(training_repo_id, str)
    or not training_repo_id
  ):
    raise RuntimeError("conversion snapshot repo id differs from training provenance")
  dataset_artifacts = dataset.get("artifacts")
  if filename == "kaihand_fast_pi05_conversion.json":
    expected_artifacts = _fast_snapshot_artifact_aggregate(snapshot)
    if dataset_artifacts != expected_artifacts:
      raise RuntimeError(
        "fast dataset artifact identity differs from the conversion snapshot"
      )
  elif dataset_artifacts is not None:
    raise RuntimeError("card dataset provenance must not claim fast artifact binding")

  source_environment = provenance.get("source_environment")
  snapshot_environment = snapshot.get("source_environment")
  if filename == "kaihand_fast_pi05_conversion.json" and task == "bulb-screw":
    if snapshot.get("source_hash_verification") != "recomputed":
      raise RuntimeError(
        "Bulb conversion snapshot must bind recomputed source HDF5 hashes"
      )
    if (
      not isinstance(source_environment, dict)
      or source_environment != snapshot_environment
      or set(source_environment) != {"mechanics_version", "model_fingerprint"}
      or not isinstance(source_environment.get("mechanics_version"), str)
      or not source_environment["mechanics_version"]
    ):
      raise RuntimeError("conversion snapshot source environment is not exactly bound")
    _require_sha256(
      source_environment.get("model_fingerprint"),
      label="source model fingerprint",
    )
  elif filename == "kaihand_fast_pi05_conversion.json":
    if source_environment is not None or snapshot_environment is not None:
      if (
        not isinstance(source_environment, dict)
        or source_environment != snapshot_environment
        or set(source_environment) != {"mechanics_version", "model_fingerprint"}
        or not isinstance(source_environment.get("mechanics_version"), str)
        or not source_environment["mechanics_version"]
      ):
        raise RuntimeError(
          "optional source environment is not exactly bound for fast dataset"
        )
      _require_sha256(
        source_environment.get("model_fingerprint"),
        label="source model fingerprint",
      )
  elif source_environment is not None:
    raise RuntimeError(
      "card dataset provenance must not claim a fast source environment"
    )
  if task == "bulb-screw" and source_environment != current_bulb_environment():
    raise RuntimeError(
      "Bulb checkpoint source mechanics/model identity differs from evaluation"
    )
  if task in VARIANT_TASKS:
    from .task_variants import current_variant_environment

    if source_environment != current_variant_environment(task):
      raise RuntimeError(
        "additional-task checkpoint source mechanics/model identity differs from evaluation"
      )

  transform = provenance.get("training_action_transform")
  normalizer = provenance.get("normalizer")
  if not isinstance(normalizer, dict):
    raise RuntimeError("training provenance normalizer must be an object")
  asset_id = normalizer.get("asset_id")
  if (
    not isinstance(asset_id, str)
    or not asset_id
    or Path(asset_id).name != asset_id
    or asset_id in {".", ".."}
  ):
    raise RuntimeError("training provenance normalizer asset id is unsafe")
  if normalizer.get("keys") != ["actions", "state"]:
    raise RuntimeError("training provenance normalizer keys must be actions/state")
  if not isinstance(normalizer.get("use_quantile_norm"), bool):
    raise RuntimeError("training provenance normalizer must bind use_quantile_norm")
  normalizer_path = assets / asset_id / "norm_stats.json"
  normalizer_bytes, normalizer_sha256 = stable_file_bytes(normalizer_path)
  semantic_sha256 = _normalizer_semantic_sha256(normalizer_path)
  if normalizer.get("semantic_sha256") != semantic_sha256:
    raise RuntimeError("checkpoint normalizer semantic SHA-256 is invalid")
  stats_provenance = normalizer.get("stats_provenance")
  if not isinstance(stats_provenance, dict):
    raise RuntimeError("training provenance must bind normalization-statistics origin")
  if (
    stats_provenance.get("filename") != "norm_stats.provenance.json"
    or not isinstance(stats_provenance.get("size_bytes"), int)
    or isinstance(stats_provenance.get("size_bytes"), bool)
    or stats_provenance["size_bytes"] <= 0
  ):
    raise RuntimeError("normalization-statistics provenance asset identity is invalid")
  _require_sha256(
    stats_provenance.get("sha256"),
    label="normalization-statistics provenance SHA-256",
  )
  stats_record = stats_provenance.get("record")
  if not isinstance(stats_record, dict):
    raise RuntimeError("normalization-statistics provenance record is missing")
  expected_stats_conversion = {
    "filename": conversion.get("filename"),
    "sha256": conversion.get("sha256"),
    "size_bytes": conversion.get("size_bytes"),
  }
  expected_stats_parameters = {
    "prefix_frames": (
      transform.get("skip_episode_prefix_frames")
      if isinstance(transform, dict)
      else None
    ),
    "action_horizon": (
      transform.get("action_horizon") if isinstance(transform, dict) else None
    ),
    "arm_dim": 7,
  }
  expected_stats_file = {
    "filename": "norm_stats.json",
    "sha256": normalizer_sha256,
    "size_bytes": len(normalizer_bytes),
  }
  if (
    stats_record.get("schema") != "kaihand_norm_stats_provenance_v1"
    or stats_record.get("dataset_root") != dataset.get("root")
    or stats_record.get("conversion_manifest") != expected_stats_conversion
    or stats_record.get("parameters") != expected_stats_parameters
    or stats_record.get("norm_stats") != expected_stats_file
  ):
    raise RuntimeError(
      "normalization statistics are not bound to this dataset, manifest, "
      "transform, and checkpoint asset"
    )

  expected_transform = {
    "action_dim": 27,
    "model_action_dim": 32,
    "action_horizon": SHARED_KAIHAND_ACTION_HORIZONS[config_name],
    "arm_delta_mask": [True] * 7 + [False] * 20,
  }
  if not isinstance(transform, dict):
    raise RuntimeError("training action transform must be an object")
  transform_mismatches = {
    key: {"expected": value, "actual": transform.get(key)}
    for key, value in expected_transform.items()
    if transform.get(key) != value
  }
  if (
    not isinstance(transform.get("skip_episode_prefix_frames"), int)
    or isinstance(transform.get("skip_episode_prefix_frames"), bool)
    or transform["skip_episode_prefix_frames"] < 0
  ):
    transform_mismatches["skip_episode_prefix_frames"] = {
      "expected": "nonnegative integer",
      "actual": transform.get("skip_episode_prefix_frames"),
    }
  if transform_mismatches:
    raise RuntimeError(f"training action transform mismatch: {transform_mismatches}")

  return {
    "status": "bound",
    "provenance_asset": {
      "path": f"assets/{TRAINING_PROVENANCE_FILENAME}",
      "schema": TRAINING_PROVENANCE_SCHEMA,
      "size_bytes": len(provenance_bytes),
      "sha256": provenance_sha256,
    },
    "conversion_manifest_snapshot": {
      "path": f"assets/{CONVERSION_SNAPSHOT_FILENAME}",
      "source_filename": filename,
      "schema": expected_schema,
      "size_bytes": len(snapshot_bytes),
      "sha256": snapshot_sha256,
    },
    "record": provenance,
  }


def unbound_training_provenance(reason: str) -> dict[str, Any]:
  if not isinstance(reason, str) or not reason.strip():
    raise ValueError("unbound provenance reason must be a nonempty string")
  return {
    "status": "unbound",
    "reason": reason.strip(),
    "required_checkpoint_assets": [
      f"assets/{TRAINING_PROVENANCE_FILENAME}",
      f"assets/{CONVERSION_SNAPSHOT_FILENAME}",
    ],
  }


def deployment_diagnostic_reasons(
  manifest: dict[str, Any],
  *,
  action_interpolation: str | None = None,
) -> list[str]:
  """Return reasons a rollout cannot contribute to formal metrics."""
  reasons: list[str] = []
  if manifest.get("schema") in {
    SHARED_DEPLOYMENT_SCHEMA_V1, POKER_DEPLOYMENT_SCHEMA_V1,
  }:
    reasons.append("legacy_v1_unbound_deployment")
  elif manifest.get("diagnostic_only") is True:
    provenance = manifest.get("training_provenance")
    reason = provenance.get("reason") if isinstance(provenance, dict) else None
    reasons.append(
      reason
      if isinstance(reason, str) and reason
      else "deployment_manifest_diagnostic_only"
    )
  if action_interpolation is not None:
    required = manifest.get(
      "required_action_interpolation", FORMAL_ACTION_INTERPOLATION
    )
    if action_interpolation != required:
      reasons.append(
        f"action_interpolation_{action_interpolation}_differs_from_required_{required}"
      )
  return reasons


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  before = path.stat()
  with path.open("rb") as source:
    while chunk := source.read(16 * 1024 * 1024):
      digest.update(chunk)
  after = path.stat()
  identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
  identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
  if identity_before != identity_after:
    raise RuntimeError(f"file changed while hashing: {path}")
  return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
  encoded = json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
  ).encode()
  return hashlib.sha256(encoded).hexdigest()


def stable_file_bytes(path: Path) -> tuple[bytes, str]:
  """Read a small identity file once and reject replacement during the read."""
  resolved = path.expanduser().resolve(strict=True)
  before = resolved.stat()
  payload = resolved.read_bytes()
  after = resolved.stat()
  identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
  identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
  if identity_before != identity_after or len(payload) != after.st_size:
    raise RuntimeError(f"file changed while reading: {resolved}")
  return payload, hashlib.sha256(payload).hexdigest()


def shared_deployment_identity(manifest: dict[str, Any]) -> dict[str, Any]:
  """Return the exact identity payload used by the shared deployment builder."""
  schema = manifest.get("schema")
  if schema == SHARED_DEPLOYMENT_SCHEMA_V1:
    fields = SHARED_DEPLOYMENT_V1_ID_FIELDS
  elif schema == SHARED_DEPLOYMENT_SCHEMA_V2:
    fields = SHARED_DEPLOYMENT_V2_ID_FIELDS
  else:
    raise RuntimeError(f"unsupported shared deployment schema: {schema!r}")
  missing = [key for key in fields if key not in manifest]
  if missing:
    raise RuntimeError(f"shared deployment identity fields are missing: {missing}")
  return {key: manifest[key] for key in fields}


def poker_deployment_identity(manifest: dict[str, Any]) -> dict[str, Any]:
  """Authenticate both historical H30 and checkpoint-bound Poker manifests."""
  schema = manifest.get("schema")
  if schema == POKER_DEPLOYMENT_SCHEMA_V1:
    fields = POKER_DEPLOYMENT_V1_ID_FIELDS
  elif schema == POKER_DEPLOYMENT_SCHEMA_V2:
    fields = POKER_DEPLOYMENT_V2_ID_FIELDS
  else:
    raise RuntimeError(f"unsupported Poker deployment schema: {schema!r}")
  missing = [key for key in fields if key not in manifest]
  if missing:
    raise RuntimeError(f"Poker deployment identity fields are missing: {missing}")
  return {key: manifest[key] for key in fields}


def validate_poker_deployment_semantics(
  manifest: dict[str, Any], *, verify_checkpoint_assets: bool = True,
) -> None:
  """Validate H50 binding without upgrading missing source provenance to formal."""
  schema = manifest.get("schema")
  if schema == POKER_DEPLOYMENT_SCHEMA_V1:
    return
  if schema != POKER_DEPLOYMENT_SCHEMA_V2:
    raise RuntimeError(f"unsupported Poker deployment schema: {schema!r}")
  if manifest.get("task") != "poker-draw" or manifest.get("model_family") != "pi0.5":
    raise RuntimeError("Poker deployment requires poker-draw + pi0.5")
  if manifest.get("instruction") != POKER_INSTRUCTION:
    raise RuntimeError("Poker deployment instruction differs from the task prompt")
  observation = manifest.get("observation_contract")
  if not isinstance(observation, dict) or observation.get("cameras") != ["head", "right_wrist"] or observation.get("image_shape_hwc") != [240, 320, 3] or observation.get("tactile_sent_to_model") is not False:
    raise RuntimeError("Poker deployment observation contract mismatch")
  # The card conversion snapshot has no authenticated source environment.
  # Even bound training/normalization provenance is not proof of scene identity.
  if manifest.get("diagnostic_only") is not True or manifest.get("formal_metrics_valid") is not False:
    raise RuntimeError("Poker v2 deployment must remain diagnostic-only")
  if manifest.get("required_action_interpolation") != FORMAL_ACTION_INTERPOLATION:
    raise RuntimeError("Poker v2 requires linear physical-substep interpolation")
  if manifest.get("control_hz") != 30:
    raise RuntimeError("Poker deployment requires 30 Hz endpoint control")
  provenance = manifest.get("training_provenance")
  if not isinstance(provenance, dict) or provenance.get("status") != "bound":
    raise RuntimeError("Poker v2 requires bound checkpoint training provenance")
  record = provenance.get("record")
  if not isinstance(record, dict):
    raise RuntimeError("Poker checkpoint provenance record must be an object")
  if verify_checkpoint_assets:
    observed = load_checkpoint_training_provenance(Path(manifest["checkpoint_path"]))
    if observed != provenance:
      raise RuntimeError("Poker training provenance differs from checkpoint asset bytes")
  if record.get("task") != "poker-draw":
    raise RuntimeError("Poker checkpoint provenance task mismatch")
  config_name = record.get("config_name")
  if config_name not in {LEGACY_KAIHAND_CONFIG, KAIHAND_H50_CONFIG}:
    raise RuntimeError("Poker checkpoint requires a shared KaiHand config")
  if manifest.get("model_config_name") != config_name:
    raise RuntimeError("Poker deployment config differs from training provenance")
  contract = record.get("action_contract")
  transform = record.get("training_action_transform")
  normalizer = record.get("normalizer")
  if not isinstance(contract, dict) or not isinstance(transform, dict) or not isinstance(normalizer, dict):
    raise RuntimeError("Poker checkpoint action/transform/normalizer binding is missing")
  if contract.get("id") != LEGACY_ACTION_CONTRACT:
    raise RuntimeError("Poker checkpoint requires legacy-measured-arm-hybrid-v1 actions")
  execution = contract.get("evaluation_execution")
  if not isinstance(execution, dict) or execution.get("physical_substep_interpolation_required") != FORMAL_ACTION_INTERPOLATION:
    raise RuntimeError("Poker checkpoint requires linear physical-substep interpolation")
  expected = {
    "prediction_horizon": transform.get("action_horizon"),
    "action_dim": transform.get("action_dim"),
    "model_action_dim": transform.get("model_action_dim"),
    "joint_names": contract.get("joint_order"),
    "normalizer_asset_id": normalizer.get("asset_id"),
    "action_representation": action_representation_from_provenance(record),
  }
  mismatches = {key: {"expected": value, "actual": manifest.get(key)} for key, value in expected.items() if manifest.get(key) != value}
  if mismatches:
    raise RuntimeError(f"Poker deployment differs from checkpoint provenance: {mismatches}")
  if transform.get("action_horizon") != SHARED_KAIHAND_ACTION_HORIZONS[config_name] or transform.get("action_dim") != 27 or transform.get("model_action_dim") != 32 or transform.get("arm_delta_mask") != [True] * 7 + [False] * 20:
    raise RuntimeError("Poker checkpoint training action transform mismatch")
  if not isinstance(manifest.get("suggested_execute_steps"), int) or isinstance(manifest["suggested_execute_steps"], bool) or not 1 <= manifest["suggested_execute_steps"] <= transform["action_horizon"]:
    raise RuntimeError("Poker suggested execute steps must be within prediction horizon")


def validate_shared_deployment_semantics(
  manifest: dict[str, Any],
  *,
  verify_checkpoint_assets: bool = True,
) -> None:
  """Fail closed on v2 provenance and formal-evaluation claims.

  Historical v1 manifests remain structurally verifiable, but callers must use
  :func:`deployment_diagnostic_reasons` to label every v1 rollout diagnostic.
  """
  schema = manifest.get("schema")
  if schema == SHARED_DEPLOYMENT_SCHEMA_V1:
    return
  if schema != SHARED_DEPLOYMENT_SCHEMA_V2:
    raise RuntimeError(f"unsupported shared deployment schema: {schema!r}")
  diagnostic_only = manifest.get("diagnostic_only")
  formal_metrics_valid = manifest.get("formal_metrics_valid")
  if not isinstance(diagnostic_only, bool) or not isinstance(
    formal_metrics_valid, bool
  ):
    raise RuntimeError("v2 deployment validity flags must be booleans")
  if diagnostic_only is formal_metrics_valid:
    raise RuntimeError(
      "v2 deployment must be exactly one of diagnostic-only or formal-valid"
    )
  required_interpolation = manifest.get("required_action_interpolation")
  if required_interpolation != FORMAL_ACTION_INTERPOLATION:
    raise RuntimeError(
      "v2 shared deployment requires linear physical-substep interpolation"
    )
  provenance = manifest.get("training_provenance")
  if not isinstance(provenance, dict):
    raise RuntimeError("v2 deployment training_provenance must be an object")
  status = provenance.get("status")
  if status == "bound":
    record = provenance.get("record")
    if not isinstance(record, dict):
      raise RuntimeError("bound deployment provenance record must be an object")
    if verify_checkpoint_assets:
      observed = load_checkpoint_training_provenance(Path(manifest["checkpoint_path"]))
      if observed != provenance:
        raise RuntimeError(
          "deployment training provenance differs from checkpoint asset bytes"
        )
    contract = record.get("action_contract")
    execution = (
      contract.get("evaluation_execution") if isinstance(contract, dict) else None
    )
    if (
      not isinstance(execution, dict)
      or execution.get("physical_substep_interpolation_required")
      != required_interpolation
    ):
      raise RuntimeError(
        "deployment interpolation differs from checkpoint action contract"
      )
    expected_representation = action_representation_from_provenance(record)
    if manifest.get("action_representation") != expected_representation:
      raise RuntimeError(
        "deployment action representation differs from checkpoint provenance"
      )
    transform = record.get("training_action_transform")
    expected_dimensions = {
      "action_dim": transform.get("action_dim")
      if isinstance(transform, dict)
      else None,
      "model_action_dim": (
        transform.get("model_action_dim") if isinstance(transform, dict) else None
      ),
      "prediction_horizon": (
        transform.get("action_horizon") if isinstance(transform, dict) else None
      ),
    }
    dimension_mismatches = {
      key: {"expected": value, "actual": manifest.get(key)}
      for key, value in expected_dimensions.items()
      if manifest.get(key) != value
    }
    if dimension_mismatches:
      raise RuntimeError(
        f"deployment dimensions differ from training provenance: {dimension_mismatches}"
      )
    normalizer = record.get("normalizer")
    if not isinstance(normalizer, dict) or manifest.get(
      "normalizer_asset_id"
    ) != normalizer.get("asset_id"):
      raise RuntimeError(
        "deployment normalizer asset differs from checkpoint provenance"
      )
    contract_joints = (
      contract.get("joint_order") if isinstance(contract, dict) else None
    )
    if manifest.get("joint_names") != contract_joints:
      raise RuntimeError("deployment joint order differs from action contract")
    if manifest.get("task") != record.get("task"):
      raise RuntimeError("deployment task differs from checkpoint provenance")
    if manifest.get("model_config_name") != record.get("config_name"):
      raise RuntimeError("deployment config differs from checkpoint provenance")
    if manifest.get("task") == "bulb-screw":
      if record.get("config_name") not in BULB_CONTROLLER_TARGET_CONFIGS:
        raise RuntimeError(
          "bound bulb deployment requires the dedicated controller-target config"
        )
      if (
        not isinstance(contract, dict)
        or contract.get("id") != RECORDED_TARGET_ACTION_CONTRACT
      ):
        raise RuntimeError(
          "bound bulb deployment requires recorded-controller-target-v1 actions"
        )
    elif record.get("config_name") in BULB_CONTROLLER_TARGET_CONFIGS:
      raise RuntimeError("the dedicated Bulb config cannot deploy a non-Bulb task")
    if formal_metrics_valid is not True or diagnostic_only is not False:
      raise RuntimeError(
        "a fully bound v2 deployment must retain its formal-valid provenance claim"
      )
  elif status == "unbound":
    reason = provenance.get("reason")
    if not isinstance(reason, str) or not reason:
      raise RuntimeError("unbound training provenance must include an explicit reason")
    if diagnostic_only is not True or formal_metrics_valid is not False:
      raise RuntimeError("unbound checkpoint deployment must be diagnostic-only")
    if manifest.get("model_config_name") != LEGACY_KAIHAND_CONFIG:
      raise RuntimeError("unbound diagnostic deployment must use the legacy config")
    representation = manifest.get("action_representation")
    if (
      not isinstance(representation, dict)
      or representation.get("contract_id") != LEGACY_ACTION_CONTRACT
    ):
      raise RuntimeError(
        "unbound diagnostic deployment cannot claim corrected action semantics"
      )
  else:
    raise RuntimeError(f"unsupported deployment provenance status: {status!r}")


def openpi_runtime_source_paths(project: Path) -> list[Path]:
  """Enumerate the complete in-repository OpenPI server/client package trees."""
  root = project.expanduser().resolve(strict=True)
  package_roots = (
    root / "src/openpi",
    root / "packages/openpi-client/src/openpi_client",
  )
  missing = [path for path in package_roots if not path.is_dir()]
  if missing:
    raise RuntimeError(f"missing OpenPI runtime package directories: {missing}")
  paths = sorted(
    (
      path
      for package_root in package_roots
      for path in package_root.rglob("*")
      if path.is_file()
      and "__pycache__" not in path.parts
      and path.suffix.casefold() != ".pyc"
    ),
    key=lambda path: path.relative_to(root).as_posix(),
  )
  if not paths:
    raise RuntimeError("OpenPI runtime source closure is empty")
  if len(paths) != len(set(paths)):
    raise RuntimeError("duplicate path in OpenPI runtime source closure")
  return paths


def _openpi_inventory(project: Path) -> set[str]:
  root = project.expanduser().resolve(strict=True)
  return {
    path.relative_to(root).as_posix() for path in openpi_runtime_source_paths(root)
  }


def _verify_rows(
  root: Path, rows: object, aggregate: object, label: str
) -> tuple[set[str], list[dict[str, Any]]]:
  if not isinstance(rows, list) or not rows:
    raise RuntimeError(f"{label} file snapshot must be a nonempty list")
  if not isinstance(aggregate, str) or _SHA256.fullmatch(aggregate) is None:
    raise RuntimeError(f"{label} aggregate SHA-256 is invalid")
  root = root.expanduser().resolve(strict=True)
  observed = set()
  actual_rows = []
  for row in rows:
    if not isinstance(row, dict):
      raise RuntimeError(f"{label} file snapshot entries must be objects")
    relative = row.get("path")
    if not isinstance(relative, str) or not relative:
      raise RuntimeError(f"{label} file path must be a relative string")
    name = PurePosixPath(relative)
    if (
      "\\" in relative
      or name.is_absolute()
      or ".." in name.parts
      or name.as_posix() != relative
    ):
      raise RuntimeError(f"unsafe {label} file path: {relative!r}")
    if relative in observed:
      raise RuntimeError(f"duplicate {label} file path: {relative!r}")
    observed.add(relative)
    expected_size = row.get("size")
    expected_digest = row.get("sha256")
    if (
      not isinstance(expected_size, int)
      or isinstance(expected_size, bool)
      or expected_size < 0
    ):
      raise RuntimeError(f"invalid {label} file size: {relative!r}")
    if (
      not isinstance(expected_digest, str) or _SHA256.fullmatch(expected_digest) is None
    ):
      raise RuntimeError(f"invalid {label} file SHA-256: {relative!r}")
    path = (root / Path(*name.parts)).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
      raise RuntimeError(f"{label} file escapes snapshot root: {relative!r}")
    actual_size = path.stat().st_size
    actual_digest = _sha256_file(path)
    actual_rows.append({"path": relative, "size": actual_size, "sha256": actual_digest})
    if actual_size != expected_size or actual_digest != expected_digest:
      raise RuntimeError(f"{label} content SHA-256 mismatch: {path}")
  if _canonical_hash(rows) != aggregate:
    raise RuntimeError(f"{label} aggregate SHA-256 mismatch")
  return observed, actual_rows


def _source_report(
  root: Path,
  rows: object,
  aggregate: object,
  label: str,
) -> dict[str, Any]:
  report = {
    "status": "verified",
    "root": str(root.expanduser().resolve()),
    "expected_sha256": aggregate,
    "actual_sha256": None,
    "file_count": len(rows) if isinstance(rows, list) else None,
    "error": None,
  }
  try:
    _, actual_rows = _verify_rows(root, rows, aggregate, label)
    report["actual_sha256"] = _canonical_hash(actual_rows)
  except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
    report.update(status="mismatch", error=f"{type(error).__name__}: {error}")
  return report


def inspect_shared_pi05_runtime_integrity(
  manifest_path: Path,
  manifest: dict[str, Any],
  *,
  evaluation_root: Path,
  expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
  """Independently authenticate one child runner's manifest and source trees."""
  report: dict[str, Any] = {
    "status": "verified",
    "manifest": {
      "path": str(manifest_path.expanduser().resolve()),
      "expected_sha256": expected_manifest_sha256,
      "actual_sha256": None,
      "size": None,
    },
    "deployment_id": {
      "declared": manifest.get("deployment_id"),
      "recomputed": None,
    },
    "checkpoint_snapshot": {
      "expected_sha256": manifest.get("checkpoint_sha256"),
      "manifest_rows_sha256": None,
      "file_count": None,
    },
    "training_provenance": None,
    "openpi_source": None,
    "evaluation_source": None,
    "errors": [],
  }
  if manifest.get("schema") not in {
    SHARED_DEPLOYMENT_SCHEMA_V1,
    SHARED_DEPLOYMENT_SCHEMA_V2,
    POKER_DEPLOYMENT_SCHEMA_V2,
  }:
    report["errors"].append("manifest schema is not a supported shared-task schema")
  try:
    payload, payload_sha256 = stable_file_bytes(manifest_path)
    report["manifest"].update(actual_sha256=payload_sha256, size=len(payload))
    loaded = json.loads(payload)
    if loaded != manifest:
      report["errors"].append(
        "manifest content changed between structural load and integrity verification"
      )
  except (OSError, RuntimeError, UnicodeDecodeError, json.JSONDecodeError) as error:
    report["errors"].append(f"manifest read failed: {type(error).__name__}: {error}")

  if expected_manifest_sha256 is not None and (
    not isinstance(expected_manifest_sha256, str)
    or _SHA256.fullmatch(expected_manifest_sha256) is None
  ):
    report["errors"].append("expected manifest SHA-256 is invalid")
  elif (
    expected_manifest_sha256 is not None
    and report["manifest"]["actual_sha256"] != expected_manifest_sha256
  ):
    report["errors"].append("manifest content SHA-256 differs from batch snapshot")

  try:
    identity = (
      poker_deployment_identity(manifest)
      if manifest.get("schema") == POKER_DEPLOYMENT_SCHEMA_V2
      else shared_deployment_identity(manifest)
    )
    recomputed_id = _canonical_hash(identity)
    report["deployment_id"]["recomputed"] = recomputed_id
    declared_id = manifest.get("deployment_id")
    if not isinstance(declared_id, str) or _SHA256.fullmatch(declared_id) is None:
      report["errors"].append("deployment_id is not a lowercase hexadecimal SHA-256")
    elif declared_id != recomputed_id:
      report["errors"].append("deployment_id does not authenticate deployment identity")
  except (KeyError, RuntimeError, TypeError, ValueError) as error:
    report["errors"].append(
      f"deployment identity verification failed: {type(error).__name__}: {error}"
    )

  checkpoint_rows = manifest.get("checkpoint_files")
  if not isinstance(checkpoint_rows, list) or not checkpoint_rows:
    report["errors"].append("checkpoint_files must be a nonempty list")
  else:
    checkpoint_rows_sha256 = _canonical_hash(checkpoint_rows)
    report["checkpoint_snapshot"].update(
      manifest_rows_sha256=checkpoint_rows_sha256,
      file_count=len(checkpoint_rows),
    )
    if checkpoint_rows_sha256 != manifest.get("checkpoint_sha256"):
      report["errors"].append(
        "checkpoint_sha256 does not authenticate checkpoint_files"
      )

  if manifest.get("schema") in {
    SHARED_DEPLOYMENT_SCHEMA_V2, POKER_DEPLOYMENT_SCHEMA_V2,
  }:
    try:
      if manifest.get("schema") == POKER_DEPLOYMENT_SCHEMA_V2:
        validate_poker_deployment_semantics(manifest)
      else:
        validate_shared_deployment_semantics(manifest)
      report["training_provenance"] = {
        "status": "verified",
        "binding_status": manifest["training_provenance"]["status"],
      }
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
      report["training_provenance"] = {
        "status": "mismatch",
        "error": f"{type(error).__name__}: {error}",
      }
      report["errors"].append(
        f"training provenance verification failed: {type(error).__name__}: {error}"
      )
  elif manifest.get("schema") == SHARED_DEPLOYMENT_SCHEMA_V1:
    report["training_provenance"] = {
      "status": "unbound_historical_v1",
      "formal_metrics_valid": False,
    }

  try:
    project = Path(manifest["model_project_path"])
  except (KeyError, TypeError, ValueError) as error:
    report["openpi_source"] = {
      "status": "mismatch",
      "error": f"invalid model_project_path: {type(error).__name__}: {error}",
    }
  else:
    report["openpi_source"] = _source_report(
      project,
      manifest.get("openpi_source_files"),
      manifest.get("openpi_source_sha256"),
      "OpenPI source",
    )
    if report["openpi_source"]["status"] == "verified":
      try:
        declared_inventory = {row["path"] for row in manifest["openpi_source_files"]}
        actual_inventory = _openpi_inventory(project)
        if declared_inventory != actual_inventory:
          missing = sorted(actual_inventory - declared_inventory)
          unexpected = sorted(declared_inventory - actual_inventory)
          raise RuntimeError(
            "OpenPI source inventory changed: "
            f"missing={missing}, unexpected={unexpected}"
          )
        report["openpi_source"]["inventory_file_count"] = len(actual_inventory)
      except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        report["openpi_source"].update(
          status="mismatch",
          error=f"{type(error).__name__}: {error}",
        )
  report["evaluation_source"] = _source_report(
    evaluation_root,
    manifest.get("evaluation_source_files"),
    manifest.get("evaluation_source_sha256"),
    "evaluation source",
  )
  for name in ("openpi_source", "evaluation_source"):
    source = report[name]
    if source.get("status") != "verified":
      report["errors"].append(f"{name} snapshot verification failed: {source['error']}")
  if report["errors"]:
    report["status"] = "mismatch"
  return report


def verify_pi05_deployment(
  manifest: dict, *, evaluation_root: Path | None = None
) -> None:
  """Reject changed checkpoint or source code before advertising an identity."""
  if manifest.get("schema") in {
    SHARED_DEPLOYMENT_SCHEMA_V1,
    SHARED_DEPLOYMENT_SCHEMA_V2,
  }:
    recomputed_id = _canonical_hash(shared_deployment_identity(manifest))
    if manifest.get("deployment_id") != recomputed_id:
      raise RuntimeError("deployment_id does not authenticate deployment identity")
    validate_shared_deployment_semantics(manifest)
  elif manifest.get("schema") in {
    POKER_DEPLOYMENT_SCHEMA_V1, POKER_DEPLOYMENT_SCHEMA_V2,
  }:
    recomputed_id = _canonical_hash(poker_deployment_identity(manifest))
    if manifest.get("deployment_id") != recomputed_id:
      raise RuntimeError("deployment_id does not authenticate Poker deployment identity")
    validate_poker_deployment_semantics(manifest)
  checkpoint = Path(manifest["checkpoint_path"])
  observed, _ = _verify_rows(
    checkpoint,
    manifest.get("checkpoint_files"),
    manifest.get("checkpoint_sha256"),
    "checkpoint",
  )
  actual = {"_CHECKPOINT_METADATA"}
  for directory in (checkpoint / "params", checkpoint / "assets"):
    if not directory.is_dir():
      raise RuntimeError(f"missing checkpoint directory: {directory}")
    actual.update(
      path.relative_to(checkpoint).as_posix()
      for path in directory.rglob("*")
      if path.is_file()
    )
  if observed != actual:
    raise RuntimeError(
      f"checkpoint file inventory changed: missing={sorted(actual - observed)}, "
      f"unexpected={sorted(observed - actual)}"
    )

  project = Path(manifest["model_project_path"])
  if manifest.get("schema") == "pickplace_pi05_deployment_v1":
    source = project / "src/openpi/training/config.py"
    if _sha256_file(source) != manifest.get("model_source_sha256"):
      raise RuntimeError(f"OpenPI model source SHA-256 mismatch: {source}")
  else:
    source_observed, _ = _verify_rows(
      project,
      manifest.get("openpi_source_files"),
      manifest.get("openpi_source_sha256"),
      "OpenPI source",
    )
    if manifest.get("schema") in {
      SHARED_DEPLOYMENT_SCHEMA_V1,
      SHARED_DEPLOYMENT_SCHEMA_V2,
      POKER_DEPLOYMENT_SCHEMA_V2,
    }:
      source_actual = _openpi_inventory(project)
      if source_observed != source_actual:
        raise RuntimeError(
          "OpenPI source inventory changed: "
          f"missing={sorted(source_actual - source_observed)}, "
          f"unexpected={sorted(source_observed - source_actual)}"
        )
  if manifest.get("schema") in {
    SHARED_DEPLOYMENT_SCHEMA_V1,
    SHARED_DEPLOYMENT_SCHEMA_V2,
    POKER_DEPLOYMENT_SCHEMA_V2,
  }:
    if evaluation_root is None:
      raise RuntimeError("shared-task deployment requires evaluation_root")
    _verify_rows(
      evaluation_root,
      manifest.get("evaluation_source_files"),
      manifest.get("evaluation_source_sha256"),
      "evaluation source",
    )
