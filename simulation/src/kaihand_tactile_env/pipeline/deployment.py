"""Manifest-bound policy deployment orchestration.

This module deliberately stays model-free.  It describes the deployment
capabilities already implemented by ``scripts/workcell`` and starts those
entry points in their own Python environments.  Model-specific checkpoint,
source and training-provenance checks remain owned by the existing validators.
"""

from __future__ import annotations

import hashlib
import importlib
import ipaddress
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Sequence
from urllib.parse import urlparse

from kaihand_tactile_env.pipeline.catalog import (
  EVALUATION_RUNNERS,
  MODEL_FAMILIES,
  TASKS,
)
from kaihand_tactile_env.pipeline.runtime import RuntimeProfile

ROOT = Path(__file__).resolve().parents[3]
WORKCELL = ROOT / "scripts/workcell"
EVALUATE = ROOT / "scripts/evaluate/evaluate.py"

RuntimeMode = Literal["websocket-sidecar", "local-worker", "embedded"]


class DeploymentError(RuntimeError):
  """A requested deployment operation is invalid or unsupported."""


class CapabilityError(DeploymentError):
  """The task/model adapter does not implement the requested stage."""


class DeploymentInterrupted(DeploymentError):
  """The deployment supervisor received an interrupt or termination signal."""

  def __init__(self, signum: int):
    self.signum = signum
    try:
      name = signal.Signals(signum).name
    except ValueError:  # pragma: no cover - handlers only install known signals.
      name = str(signum)
    super().__init__(f"deployment interrupted by {name}")


@dataclass(frozen=True)
class DeploymentAdapter:
  task: str
  model_family: str
  runtime_mode: RuntimeMode
  evaluate_backend: str | None
  prepare_backend: str | None = None
  serve_backend: str | None = None
  verify_backend: str | None = None
  default_port: int | None = None
  bind_host: str | None = None
  manifest_schemas: tuple[str, ...] = ()

  @property
  def capabilities(self) -> dict[str, bool]:
    return {
      "prepare": self.prepare_backend is not None,
      "verify": self.verify_backend is not None,
      "serve": self.serve_backend is not None,
      "run": self.evaluate_backend is not None,
      "evaluate": self.evaluate_backend is not None,
    }


@dataclass(frozen=True)
class ProcessPlan:
  command: tuple[str, ...]
  cwd: Path
  environment: dict[str, str]
  endpoint: str | None = None

  def as_dict(self, *, include_environment: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
      "command": list(self.command),
      "cwd": str(self.cwd),
      "endpoint": self.endpoint,
    }
    if include_environment:
      result["environment"] = dict(sorted(self.environment.items()))
    return result


_PI05_PREPARE = {
  "pick-place": "prepare_pickplace_pi05_deployment.py",
  "poker-draw": "prepare_poker_pi05_deployment.py",
  "usb-insert": "prepare_usb_pi05_deployment.py",
  "bulb-screw": "prepare_shared_task_pi05_deployment.py",
  "install-ram": "prepare_shared_task_pi05_deployment.py",
  "vase-wipe": "prepare_shared_task_pi05_deployment.py",
  "whiteboard-wipe": "prepare_shared_task_pi05_deployment.py",
  "sponge-grasp": "prepare_shared_task_pi05_deployment.py",
  "usb-unplug": "prepare_shared_task_pi05_deployment.py",
  "whiteboard-wipe-horizontal": "prepare_shared_task_pi05_deployment.py",
  "whiteboard-wipe-horizontal-line": "prepare_shared_task_pi05_deployment.py",
  "bulb-unscrew": "prepare_shared_task_pi05_deployment.py",
}
_PI05_SERVE = {
  "pick-place": "serve_pickplace_pi05_policy.py",
  "poker-draw": "serve_poker_pi05_policy.py",
  "usb-insert": "serve_usb_pi05_policy.py",
  "bulb-screw": "serve_shared_task_pi05_policy.py",
  "install-ram": "serve_shared_task_pi05_policy.py",
  "vase-wipe": "serve_shared_task_pi05_policy.py",
  "whiteboard-wipe": "serve_shared_task_pi05_policy.py",
  "sponge-grasp": "serve_shared_task_pi05_policy.py",
  "usb-unplug": "serve_shared_task_pi05_policy.py",
  "whiteboard-wipe-horizontal": "serve_shared_task_pi05_policy.py",
  "whiteboard-wipe-horizontal-line": "serve_shared_task_pi05_policy.py",
  "bulb-unscrew": "serve_shared_task_pi05_policy.py",
}
_PI05_SCHEMAS = {
  "pick-place": ("pickplace_pi05_deployment_v1",),
  "poker-draw": ("poker_pi05_deployment_v1", "poker_pi05_deployment_v2"),
  "usb-insert": ("usb_pi05_deployment_v1", "usb_pi05_deployment_v2"),
  "bulb-screw": ("shared_task_pi05_deployment_v1", "shared_task_pi05_deployment_v2"),
  "install-ram": ("shared_task_pi05_deployment_v1", "shared_task_pi05_deployment_v2"),
  "vase-wipe": ("shared_task_pi05_deployment_v1", "shared_task_pi05_deployment_v2"),
  "whiteboard-wipe": ("shared_task_pi05_deployment_v1", "shared_task_pi05_deployment_v2"),
  "sponge-grasp": ("shared_task_pi05_deployment_v1", "shared_task_pi05_deployment_v2"),
  "usb-unplug": ("shared_task_pi05_deployment_v2",),
  "whiteboard-wipe-horizontal": ("shared_task_pi05_deployment_v2",),
  "whiteboard-wipe-horizontal-line": ("shared_task_pi05_deployment_v2",),
  "bulb-unscrew": ("shared_task_pi05_deployment_v2",),
}
_EGOSTEER_PREPARE = {
  "pick-place": "prepare_pickplace_egosteer_deployment.py",
  "poker-draw": "prepare_poker_egosteer_deployment.py",
  "usb-insert": "prepare_usb_egosteer_deployment.py",
}
_EGOSTEER_SCHEMAS = {
  "pick-place": ("pickplace_egosteer_deployment_v1",),
  "poker-draw": ("poker-egosteer-deployment-v1",),
  "usb-insert": ("usb-egosteer-deployment-v1",),
}
def _build_registry() -> dict[tuple[str, str], DeploymentAdapter]:
  registry: dict[tuple[str, str], DeploymentAdapter] = {}
  for task, family in EVALUATION_RUNNERS:
    if family == "pi05":
      adapter = DeploymentAdapter(
        task=task,
        model_family=family,
        runtime_mode="websocket-sidecar",
        evaluate_backend=EVALUATION_RUNNERS[(task, family)],
        prepare_backend=_PI05_PREPARE[task],
        serve_backend=_PI05_SERVE[task],
        verify_backend="pi05-integrity",
        default_port=18784 if task == "poker-draw" else 18783,
        bind_host="127.0.0.1" if task == "pick-place" else "0.0.0.0",
        manifest_schemas=_PI05_SCHEMAS[task],
      )
    elif family == "egosteer":
      prepared = task in _EGOSTEER_PREPARE
      adapter = DeploymentAdapter(
        task=task,
        model_family=family,
        runtime_mode="websocket-sidecar",
        evaluate_backend=EVALUATION_RUNNERS[(task, family)],
        prepare_backend=_EGOSTEER_PREPARE.get(task),
        serve_backend="egosteer-external" if prepared else None,
        verify_backend="egosteer-frozen" if prepared else None,
        default_port=18782,
        bind_host="127.0.0.1" if prepared else None,
        manifest_schemas=_EGOSTEER_SCHEMAS.get(task, ()),
      )
    elif family == "egotouch":
      pickplace = task == "pick-place"
      adapter = DeploymentAdapter(
        task=task,
        model_family=family,
        runtime_mode="local-worker",
        evaluate_backend=EVALUATION_RUNNERS[(task, family)],
        prepare_backend=(
          "prepare_pickplace_egotouch_deployment.py" if pickplace else None
        ),
        verify_backend=("egotouch-pickplace" if pickplace else None),
        manifest_schemas=(
          ("pickplace_egotouch_deployment_v1",) if pickplace else ()
        ),
      )
    elif family == "lingbot-vla2":
      adapter = DeploymentAdapter(
        task=task,
        model_family=family,
        runtime_mode="websocket-sidecar",
        evaluate_backend=EVALUATION_RUNNERS[(task, family)],
        prepare_backend="prepare_pickplace_lingbot_vla2_deployment.py",
        serve_backend="serve_pickplace_lingbot_vla2_policy.py",
        verify_backend="lingbot-validate-only",
        default_port=8006,
        manifest_schemas=("pickplace_lingbot_vla2_deployment_v1",),
      )
    elif family == "pi05+trex":
      # T-Rex currently freezes no combined deployment identity.  Its existing
      # runner validates the base manifest and expert artifacts at execution.
      adapter = DeploymentAdapter(
        task=task,
        model_family=family,
        runtime_mode="embedded",
        evaluate_backend=EVALUATION_RUNNERS[(task, family)],
        manifest_schemas=("poker_pi05_deployment_v1",),
      )
    else:  # pragma: no cover - catalog additions must choose an explicit mode.
      raise AssertionError(f"unclassified deployment family: {family}")
    registry[(task, family)] = adapter
  return registry


ADAPTERS = _build_registry()


def adapter_catalog() -> tuple[dict[str, Any], ...]:
  """Return stable, JSON-serializable task/model deployment capabilities."""
  rows = []
  for adapter in sorted(ADAPTERS.values(), key=lambda row: (row.task, row.model_family)):
    capabilities = adapter.capabilities
    rows.append({
      "task": adapter.task,
      "model_family": adapter.model_family,
      "runtime_mode": adapter.runtime_mode,
      "prepare_backend": adapter.prepare_backend,
      "serve_backend": adapter.serve_backend,
      "evaluate_backend": adapter.evaluate_backend,
      "verify_backend": adapter.verify_backend,
      "bind_host": adapter.bind_host,
      "manifest_schemas": list(adapter.manifest_schemas),
      "capabilities": capabilities,
      **capabilities,
    })
  return tuple(rows)


def adapter_for(task: str, model_family: str) -> DeploymentAdapter:
  try:
    return ADAPTERS[(task, model_family)]
  except KeyError as error:
    raise CapabilityError(
      f"no deployment adapter for {task!r} + {model_family!r}"
    ) from error


def normalize_model_family(value: object) -> str:
  normalized = str(value).strip().lower().replace(".", "")
  aliases = {
    "egosteer": "egosteer",
    "pi05": "pi05",
    "π05": "pi05",
    "egotouch": "egotouch",
    "lingbot-vla-20": "lingbot-vla2",
    "lingbot-vla2": "lingbot-vla2",
    "pi05+trex": "pi05+trex",
  }
  try:
    return aliases[normalized]
  except KeyError as error:
    raise DeploymentError(f"unsupported model_family in manifest: {value!r}") from error


def load_deployment_manifest(
  path: Path, *, model_family: str | None = None
) -> tuple[Path, dict[str, Any], DeploymentAdapter]:
  resolved = path.expanduser().resolve(strict=True)
  try:
    payload = json.loads(resolved.read_text(encoding="utf-8"))
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise DeploymentError(f"invalid deployment manifest JSON: {resolved}") from error
  if not isinstance(payload, dict):
    raise DeploymentError("deployment manifest must contain a JSON object")
  task = payload.get("task")
  if task not in TASKS:
    raise DeploymentError(f"unsupported deployment task: {task!r}")
  declared_family = normalize_model_family(payload.get("model_family"))
  selected_family = model_family or declared_family
  if selected_family not in MODEL_FAMILIES:
    raise DeploymentError(f"unsupported requested model family: {selected_family!r}")
  if selected_family == "pi05+trex":
    if declared_family != "pi05":
      raise DeploymentError("pi05+trex must extend a frozen pi0.5 base manifest")
  elif selected_family != declared_family:
    raise DeploymentError(
      f"deployment model family {declared_family!r} differs from {selected_family!r}"
    )
  adapter = adapter_for(task, selected_family)
  schema = payload.get("schema")
  if adapter.manifest_schemas and schema not in adapter.manifest_schemas:
    raise DeploymentError(
      f"{task} + {selected_family} requires schema in "
      f"{adapter.manifest_schemas!r}, got {schema!r}"
    )
  required = (
    "deployment_id", "checkpoint_path", "checkpoint_sha256",
    "prediction_horizon", "action_dim", "observation_contract",
    "action_representation",
  )
  missing = [name for name in required if name not in payload]
  if missing:
    raise DeploymentError(f"deployment manifest missing fields: {missing}")
  for name in ("deployment_id", "checkpoint_sha256"):
    value = payload[name]
    if not isinstance(value, str) or not value:
      raise DeploymentError(f"deployment {name} must be a nonempty string")
  if schema == "usb_pi05_deployment_v2":
    validator = _workcell_module("prepare_usb_pi05_deployment")
    try:
      validator.validate_deployment_identity(payload)
    except RuntimeError as error:
      raise DeploymentError(f"invalid USB v2 deployment identity: {error}") from error
  elif schema == "poker_pi05_deployment_v2":
    validator = _workcell_module("prepare_poker_pi05_deployment")
    try:
      validator.validate_deployment_identity(payload)
      validator.validate_manifest_training_binding(payload)
    except RuntimeError as error:
      raise DeploymentError(f"invalid Poker v2 deployment: {error}") from error
  return resolved, payload, adapter


def _has_option(arguments: Sequence[str], option: str) -> bool:
  return any(value == option or value.startswith(option + "=") for value in arguments)


def _reject_options(arguments: Sequence[str], options: Sequence[str]) -> None:
  conflicts = sorted(option for option in options if _has_option(arguments, option))
  if conflicts:
    raise DeploymentError(f"unified interface owns these options: {conflicts}")


def _python(profile: RuntimeProfile | None, fallback: object = None) -> Path:
  if profile is not None and profile.python is not None:
    return profile.python
  if isinstance(fallback, str) and fallback:
    return Path(fallback).expanduser().resolve()
  return Path(sys.executable).resolve()


def _project(profile: RuntimeProfile | None, fallback: object = None) -> Path | None:
  if profile is not None and profile.project_root is not None:
    return profile.project_root
  if isinstance(fallback, str) and fallback:
    return Path(fallback).expanduser().resolve()
  return None


def _manifest_project(
  profile: RuntimeProfile | None, manifest: dict[str, Any]
) -> Path | None:
  fallback = manifest.get("model_project_path") or manifest.get("lingbot_repo_path")
  if not fallback:
    source = manifest.get("model_source")
    if isinstance(source, dict):
      fallback = source.get("project")
  return _project(profile, fallback)


def _environment(
  profile: RuntimeProfile | None,
  *,
  model_family: str | None = None,
  project: Path | None = None,
) -> dict[str, str]:
  environment = dict(os.environ) if profile is None else profile.environment()
  python_paths = [ROOT / "src"]
  if model_family in {"pi05", "pi05+trex"} and project is not None:
    python_paths.extend((
      project / "src",
      project / "packages/openpi-client/src",
    ))
  declared = environment.get("PYTHONPATH")
  values = [str(path) for path in python_paths]
  if declared:
    values.append(declared)
  environment["PYTHONPATH"] = os.pathsep.join(values)
  return environment


def build_prepare_plan(
  task: str,
  model_family: str,
  *,
  checkpoint: Path,
  output_dir: Path,
  profile: RuntimeProfile | None = None,
  backend_args: Sequence[str] = (),
) -> ProcessPlan:
  adapter = adapter_for(task, model_family)
  if adapter.prepare_backend is None:
    raise CapabilityError(
      f"prepare is not implemented for {task} + {model_family}; "
      "provide an already frozen manifest from the model pipeline"
    )
  extras = tuple(backend_args)
  _reject_options(extras, ("--checkpoint", "--output-dir", "--task"))
  command = [
    str(_python(profile)),
    str(WORKCELL / adapter.prepare_backend),
    "--checkpoint", str(checkpoint.expanduser().resolve()),
    "--output-dir", str(output_dir.expanduser().resolve()),
  ]
  project = _project(profile)
  if model_family == "pi05":
    if adapter.prepare_backend == "prepare_shared_task_pi05_deployment.py":
      command.extend(("--task", task))
    if project is not None and not _has_option(extras, "--openpi-root"):
      command.extend(("--openpi-root", str(project)))
  elif model_family == "egosteer":
    if profile is not None and profile.python is not None and not _has_option(
      extras, "--model-python"
    ):
      command.extend(("--model-python", str(profile.python)))
    if project is not None and not _has_option(extras, "--model-project"):
      command.extend(("--model-project", str(project)))
    if not _has_option(extras, "--port"):
      command.extend(("--port", str(adapter.default_port)))
  elif model_family == "egotouch":
    if profile is not None and profile.python is not None and not _has_option(
      extras, "--model-python"
    ):
      command.extend(("--model-python", str(profile.python)))
    if project is not None and not _has_option(extras, "--model-project"):
      command.extend(("--model-project", str(project)))
  elif model_family == "lingbot-vla2":
    if project is not None and not _has_option(extras, "--lingbot-repo"):
      command.extend(("--lingbot-repo", str(project)))
  command.extend(extras)
  return ProcessPlan(
    tuple(command),
    ROOT,
    _environment(profile, model_family=model_family, project=project),
  )


def _workcell_module(name: str):
  value = str(WORKCELL)
  if value not in sys.path:
    sys.path.insert(0, value)
  return importlib.import_module(name)


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _verify_egosteer(
  manifest: dict[str, Any], profile: RuntimeProfile | None
) -> None:
  helpers = _workcell_module("prepare_poker_egosteer_deployment")
  checkpoint = Path(manifest["checkpoint_path"]).expanduser().resolve(strict=True)
  checkpoint_hash, checkpoint_files = helpers.checkpoint_identity(checkpoint)
  if checkpoint_hash != manifest["checkpoint_sha256"]:
    raise DeploymentError("EgoSteer checkpoint SHA-256 mismatch")
  if checkpoint_files != manifest.get("checkpoint_files"):
    raise DeploymentError("EgoSteer checkpoint file inventory changed")
  config = Path(manifest["model_config_path"]).expanduser().resolve(strict=True)
  normalizer = Path(manifest["normalizer_path"]).expanduser().resolve(strict=True)
  if helpers.sha256_file(config) != manifest.get("model_config_sha256"):
    raise DeploymentError("EgoSteer model config SHA-256 mismatch")
  if helpers.sha256_file(normalizer) != manifest.get("normalizer_sha256"):
    raise DeploymentError("EgoSteer normalizer SHA-256 mismatch")
  project = _manifest_project(profile, manifest)
  if project is None or not project.is_dir():
    raise DeploymentError("EgoSteer verification requires its model project")
  if helpers.source_identity(project) != manifest.get("model_source"):
    raise DeploymentError("EgoSteer model source identity changed")
  model_python = _python(profile, manifest.get("model_python"))
  camera_override = ",".join(manifest["observation_contract"]["cameras"])
  contract = helpers.parse_model_contract(config, model_python, camera_override)
  for key in (
    "prediction_horizon", "action_dim", "observation_contract",
    "action_representation",
  ):
    if contract[key] != manifest.get(key):
      raise DeploymentError(f"EgoSteer frozen model contract changed at {key}")
  identity_keys = (
    "task", "model_family", "checkpoint_path", "checkpoint_sha256",
    "model_config_path", "model_config_sha256", "normalizer_path",
    "normalizer_sha256", "prediction_horizon", "action_dim",
    "observation_contract", "action_representation", "model_source",
  )
  identity = {key: manifest[key] for key in identity_keys}
  recomputed = hashlib.sha256(
    json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
  ).hexdigest()
  if recomputed != manifest["deployment_id"]:
    raise DeploymentError("EgoSteer deployment_id does not authenticate identity")


def _verify_pickplace_egotouch(
  manifest: dict[str, Any], profile: RuntimeProfile | None
) -> None:
  batch = _workcell_module("evaluate_pickplace_policy_batch")
  requested_python = (
    profile.python if profile is not None and profile.python is not None else None
  )
  requested_project = (
    profile.project_root
    if profile is not None and profile.project_root is not None
    else None
  )
  batch._egotouch_runtime(
    SimpleNamespace(
      model_python=requested_python,
      model_project=requested_project,
    ),
    manifest,
  )


def build_verification_plan(
  manifest_path: Path,
  manifest: dict[str, Any],
  adapter: DeploymentAdapter,
  *,
  profile: RuntimeProfile | None = None,
  verify_hashes: bool = False,
) -> ProcessPlan | None:
  if adapter.verify_backend == "lingbot-validate-only":
    project = _project(profile, manifest.get("lingbot_repo_path"))
    if project is None:
      raise DeploymentError("LingBot verification requires its repository")
    command = [
      str(_python(profile)),
      str(WORKCELL / "serve_pickplace_lingbot_vla2_policy.py"),
      "--deployment-manifest", str(manifest_path),
      "--lingbot-repo", str(project),
      "--validate-only",
    ]
    if verify_hashes:
      command.append("--verify-hashes")
    return ProcessPlan(
      tuple(command),
      project,
      _environment(profile, model_family=adapter.model_family, project=project),
    )
  if adapter.verify_backend == "egotouch-pickplace":
    python = _python(profile, manifest.get("model_python"))
    project = _project(profile, manifest.get("model_project_path"))
    if project is None:
      raise DeploymentError("EgoTouch verification requires its model project")
    command = (
      str(python),
      str(WORKCELL / "egotouch_model_worker.py"),
      "--project", str(project),
      "--checkpoint", str(manifest["checkpoint_path"]),
      "--binding", str(manifest["binding_path"]),
      "--validate-only",
    )
    return ProcessPlan(
      command,
      project,
      _environment(profile, model_family=adapter.model_family, project=project),
    )
  return None


def verify_deployment(
  manifest_path: Path,
  *,
  model_family: str | None = None,
  profile: RuntimeProfile | None = None,
  execute_external: bool = True,
  verify_hashes: bool = False,
) -> dict[str, Any]:
  resolved, manifest, adapter = load_deployment_manifest(
    manifest_path, model_family=model_family
  )
  if adapter.verify_backend is None:
    raise CapabilityError(
      f"complete standalone verification is not implemented for "
      f"{adapter.task} + {adapter.model_family}; validation remains runner-owned"
    )
  if execute_external:
    if adapter.verify_backend == "pi05-integrity":
      from kaihand_tactile_env.shared.pi05_deployment_integrity import (
        verify_pi05_deployment,
      )

      verify_pi05_deployment(manifest, evaluation_root=ROOT)
    elif adapter.verify_backend == "egosteer-frozen":
      _verify_egosteer(manifest, profile)
    elif adapter.verify_backend == "egotouch-pickplace":
      _verify_pickplace_egotouch(manifest, profile)
  plan = build_verification_plan(
    resolved,
    manifest,
    adapter,
    profile=profile,
    verify_hashes=verify_hashes,
  )
  returncode = None
  if plan is not None and execute_external:
    returncode = _execute_owned_process(plan)
    if returncode:
      raise DeploymentError(
        f"{adapter.verify_backend} exited with status {returncode}"
      )
  return {
    "schema": "kaihand_deployment_verify_report_v1",
    "schema_version": 1,
    "stage": "deploy",
    "operation": "verify",
    "status": "verified" if execute_external else "planned",
    "task": adapter.task,
    "model_family": adapter.model_family,
    "deployment_id": manifest["deployment_id"],
    "deployment_manifest": str(resolved),
    "validator": adapter.verify_backend,
    "external_validation": None if plan is None else plan.as_dict(),
    "external_returncode": returncode,
  }


def _endpoint(host: str, port: int) -> str:
  if not 1 <= port <= 65535:
    raise DeploymentError("port must be in 1..65535")
  if not host or "://" in host:
    raise DeploymentError("host must be a hostname or address without a URI scheme")
  return f"ws://{host}:{port}"


def _endpoint_parts(value: str) -> tuple[str, int, str]:
  uri = value if value.startswith(("ws://", "wss://")) else f"ws://{value}"
  parsed = urlparse(uri)
  if parsed.scheme not in {"ws", "wss"} or not parsed.hostname or parsed.port is None:
    raise DeploymentError(f"invalid WebSocket endpoint: {value!r}")
  return parsed.hostname, parsed.port, uri


def _is_loopback_host(host: str) -> bool:
  if host.lower() == "localhost":
    return True
  try:
    return ipaddress.ip_address(host).is_loopback
  except ValueError:
    return False


def _add_endpoint_no_proxy(environment: dict[str, str], endpoint: str) -> None:
  """Keep local policy traffic off HTTP proxies without dropping proxy config."""
  host, _, _ = _endpoint_parts(endpoint)
  required = ("localhost", "127.0.0.1", "::1", host)
  for key in ("NO_PROXY", "no_proxy"):
    entries = [item.strip() for item in environment.get(key, "").split(",")]
    entries = [item for item in entries if item]
    for item in required:
      if item not in entries:
        entries.append(item)
    environment[key] = ",".join(entries)


def build_serve_plan(
  manifest_path: Path,
  manifest: dict[str, Any],
  adapter: DeploymentAdapter,
  *,
  profile: RuntimeProfile | None = None,
  host: str | None = None,
  port: int | None = None,
  endpoint: str | None = None,
  backend_args: Sequence[str] = (),
) -> ProcessPlan:
  if adapter.serve_backend is None:
    raise CapabilityError(
      f"serve is not applicable to {adapter.task} + {adapter.model_family} "
      f"({adapter.runtime_mode}); its runner owns model execution"
    )
  extras = tuple(backend_args)
  owned_options = ["--deployment-manifest", "--host", "--port"]
  if adapter.model_family == "lingbot-vla2":
    owned_options.append("--lingbot-repo")
  _reject_options(extras, owned_options)
  if adapter.bind_host is not None and host is not None and host != adapter.bind_host:
    raise CapabilityError(
      f"{adapter.task} + {adapter.model_family} server has fixed bind host "
      f"{adapter.bind_host}; {host!r} cannot be applied by its backend"
    )
  if endpoint is not None:
    endpoint_host, endpoint_port, normalized_endpoint = _endpoint_parts(endpoint)
    if normalized_endpoint.startswith("wss://"):
      raise CapabilityError(
        "managed serving uses the existing plaintext WebSocket backends; "
        "use ws:// or --service-mode external for a TLS endpoint"
      )
    if not _is_loopback_host(endpoint_host):
      raise CapabilityError(
        "managed serving starts a local sidecar and requires a loopback endpoint; "
        "use --service-mode external for a remote endpoint"
      )
    if port is not None and port != endpoint_port:
      raise DeploymentError("--port differs from --endpoint")
    port = endpoint_port
  else:
    port = adapter.default_port if port is None else port
    if port is None:  # pragma: no cover - every current sidecar has a port.
      raise DeploymentError("deployment adapter has no default port")
    connect_host = host or adapter.bind_host or "127.0.0.1"
    if connect_host == "0.0.0.0":
      connect_host = "127.0.0.1"
    normalized_endpoint = _endpoint(connect_host, port)
  python = _python(profile, manifest.get("model_python"))
  project = _manifest_project(profile, manifest)
  environment = _environment(
    profile, model_family=adapter.model_family, project=project
  )
  if adapter.serve_backend == "egosteer-external":
    frozen_endpoint = manifest.get("server")
    if isinstance(frozen_endpoint, str):
      _, frozen_port, _ = _endpoint_parts(frozen_endpoint)
      if port != frozen_port:
        raise DeploymentError(
          "EgoSteer port is frozen in inference.yaml; prepare a new runtime "
          "config instead of overriding it"
        )
    if project is None:
      raise DeploymentError("managed EgoSteer serving requires its model project")
    inference = manifest_path.parent / "inference.yaml"
    if not inference.is_file():
      raise DeploymentError(f"EgoSteer inference config is missing: {inference}")
    environment["EGOSTEER_INFERENCE_CONFIG"] = str(inference)
    command = [str(python), "-m", "src.serving.serve_policy", *extras]
    cwd = project
  else:
    command = [
      str(python),
      str(WORKCELL / adapter.serve_backend),
      "--deployment-manifest", str(manifest_path),
    ]
    if adapter.model_family == "lingbot-vla2":
      if project is None:
        raise DeploymentError("LingBot serving requires its repository")
      lingbot_host = endpoint_host if endpoint is not None else (host or "127.0.0.1")
      command.extend((
        "--lingbot-repo", str(project),
        "--host", lingbot_host,
      ))
      cwd = project
    else:
      cwd = project or ROOT
    command.extend(("--port", str(port), *extras))
  return ProcessPlan(tuple(command), cwd, environment, normalized_endpoint)


def validate_server_metadata(
  metadata: object,
  manifest: dict[str, Any],
  adapter: DeploymentAdapter,
) -> dict[str, Any]:
  if not isinstance(metadata, dict):
    raise DeploymentError("policy server metadata must be an object")
  expected = {
    "deployment_id": manifest["deployment_id"],
    "checkpoint_sha256": manifest["checkpoint_sha256"],
    "action_horizon": manifest["prediction_horizon"],
    "action_dim": manifest["action_dim"],
    "observation_contract": manifest["observation_contract"],
    "action_representation": manifest["action_representation"],
  }
  mismatches = {
    key: {"expected": value, "actual": metadata.get(key)}
    for key, value in expected.items()
    if metadata.get(key) != value
  }
  advertised_family = metadata.get("model_family")
  if advertised_family is not None:
    try:
      normalized = normalize_model_family(advertised_family)
    except DeploymentError:
      normalized = str(advertised_family)
    if normalized != adapter.model_family:
      mismatches["model_family"] = {
        "expected": adapter.model_family,
        "actual": advertised_family,
      }
  advertised_task = metadata.get("evaluation_task")
  if advertised_task is None and metadata.get("task") in TASKS:
    advertised_task = metadata["task"]
  if advertised_task is not None and advertised_task != adapter.task:
    mismatches["task"] = {
      "expected": adapter.task,
      "actual": advertised_task,
    }
  if mismatches:
    raise DeploymentError(f"policy server deployment identity mismatch: {mismatches}")
  return metadata


def probe_server(endpoint: str, *, timeout: float = 5.0) -> dict[str, Any]:
  """Perform the real WebSocket metadata handshake, not a TCP port probe."""
  if not math.isfinite(timeout) or timeout <= 0:
    raise ValueError("probe timeout must be positive")
  _, _, uri = _endpoint_parts(endpoint)
  from websockets.sync.client import connect

  from kaihand_tactile_env.shared.egosteer_client import unpack_payload

  with connect(
    uri,
    proxy=None,
    compression=None,
    max_size=None,
    open_timeout=timeout,
    close_timeout=timeout,
    ping_interval=None,
  ) as websocket:
    encoded = websocket.recv(timeout=timeout)
  if isinstance(encoded, str):
    raise DeploymentError(f"policy server returned text during metadata handshake: {encoded}")
  metadata = unpack_payload(encoded)
  if not isinstance(metadata, dict):
    raise DeploymentError("policy server metadata must decode to an object")
  return metadata


def wait_for_server(
  endpoint: str,
  manifest: dict[str, Any],
  adapter: DeploymentAdapter,
  *,
  timeout: float,
  process: subprocess.Popen | None = None,
) -> dict[str, Any]:
  if not math.isfinite(timeout) or timeout <= 0:
    raise ValueError("server timeout must be positive")
  from websockets.exceptions import ConnectionClosed, InvalidHandshake

  deadline = time.monotonic() + timeout
  last_error: BaseException | None = None
  while time.monotonic() < deadline:
    if process is not None and process.poll() is not None:
      raise DeploymentError(
        f"managed policy server exited during startup with status {process.returncode}"
      )
    remaining = deadline - time.monotonic()
    try:
      metadata = probe_server(endpoint, timeout=max(0.1, min(2.0, remaining)))
    except (ConnectionClosed, InvalidHandshake, OSError, TimeoutError) as error:
      last_error = error
      time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
      continue
    if process is not None and process.poll() is not None:
      raise DeploymentError(
        f"managed policy server exited during startup with status {process.returncode}"
      )
    # A reachable WebSocket with the wrong identity is not a startup transient.
    return validate_server_metadata(metadata, manifest, adapter)
  raise DeploymentError(
    f"policy server at {endpoint} did not pass metadata readiness within "
    f"{timeout:g}s: {last_error}"
  )


def _egotouch_evaluation_args(
  manifest: dict[str, Any], profile: RuntimeProfile | None
) -> list[str]:
  python = _python(profile, manifest.get("model_python"))
  project = _manifest_project(profile, manifest)
  if project is None:
    raise DeploymentError("EgoTouch run requires a model project runtime")
  return ["--model-python", str(python), "--model-project", str(project)]


def build_evaluation_plan(
  manifest_path: Path,
  manifest: dict[str, Any],
  adapter: DeploymentAdapter,
  *,
  output_dir: Path,
  service_mode: str,
  endpoint: str | None,
  profile: RuntimeProfile | None,
  num_trials: int,
  seed_start: int,
  video_count: int,
  execute_steps: int | str | None = None,
  smoke_seconds: float | None = None,
  max_sim_seconds: float = 90.0,
  guard_mode: str = "observe",
  evaluator_args: Sequence[str] = (),
) -> ProcessPlan:
  if adapter.evaluate_backend is None:
    raise CapabilityError(f"run is not implemented for {adapter.task} + {adapter.model_family}")
  if num_trials <= 0 or seed_start < 0:
    raise DeploymentError("num-trials must be positive and seed-start nonnegative")
  if not 0 <= video_count <= num_trials:
    raise DeploymentError("video-count must be between zero and num-trials")
  if smoke_seconds is not None and (
    not math.isfinite(smoke_seconds) or smoke_seconds <= 0
  ):
    raise DeploymentError("smoke-seconds must be finite and positive")
  if not math.isfinite(max_sim_seconds) or max_sim_seconds <= 0:
    raise DeploymentError("max-sim-seconds must be finite and positive")
  if guard_mode not in {"observe", "strict"}:
    raise DeploymentError("guard-mode must be observe or strict")
  extras = list(evaluator_args)
  _reject_options(
    extras,
    (
      "--task", "--model-family", "--deployment-manifest", "--output-dir",
      "--num-trials", "--seed-start", "--video-count", "--execute-steps",
      "--max-sim-seconds", "--guard-mode", "--server", "--model-python", "--model-project",
    ),
  )
  command = [
    sys.executable,
    str(EVALUATE),
    "--task", adapter.task,
    "--model-family", adapter.model_family,
    "--deployment-manifest", str(manifest_path),
    "--output-dir", str(output_dir),
    "--num-trials", str(num_trials),
    "--seed-start", str(seed_start),
    "--video-count", str(video_count),
  ]
  if execute_steps is not None:
    command.extend(("--execute-steps", str(execute_steps)))
  command.extend((
    "--max-sim-seconds", str(smoke_seconds if smoke_seconds is not None else max_sim_seconds),
    "--guard-mode", guard_mode,
  ))
  runner_args: list[str] = []
  if adapter.runtime_mode == "websocket-sidecar":
    if service_mode not in {"managed", "external"}:
      raise DeploymentError("websocket deployments require managed or external service mode")
    if endpoint is None:
      raise DeploymentError("websocket deployment run requires an endpoint")
    runner_args.extend(("--server", endpoint))
  elif service_mode == "external":
    raise DeploymentError(
      f"external service mode is not applicable to {adapter.runtime_mode} deployments"
    )
  elif endpoint is not None:
    raise DeploymentError(f"--endpoint is not applicable to {adapter.runtime_mode}")
  if adapter.runtime_mode == "local-worker":
    runner_args.extend(_egotouch_evaluation_args(manifest, profile))
  runner_args.extend(extras)
  if runner_args:
    command.extend(("--", *runner_args))
  if adapter.runtime_mode == "websocket-sidecar":
    # The evaluator is always the simulator Python. Model runtime overrides
    # belong to the sidecar and can make native simulator modules unimportable.
    environment = _environment(None)
  else:
    project = _manifest_project(profile, manifest)
    environment = _environment(
      profile, model_family=adapter.model_family, project=project
    )
  if adapter.runtime_mode == "websocket-sidecar" and endpoint is not None:
    _add_endpoint_no_proxy(environment, endpoint)
  return ProcessPlan(tuple(command), ROOT, environment)


def _write_json(path: Path, value: dict[str, Any]) -> None:
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(
    json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  temporary.replace(path)


def _signal_owned_process_group(
  process: subprocess.Popen, signum: signal.Signals
) -> bool:
  """Signal a session created by us, falling back for test/platform shims."""
  try:
    os.killpg(process.pid, signum)
    return True
  except ProcessLookupError:
    if process.poll() is not None:
      return False
    if signum == signal.SIGTERM:
      process.terminate()
    else:
      process.kill()
    return True
  except (AttributeError, PermissionError, OSError):
    if process.poll() is not None:
      return False
    if signum == signal.SIGTERM:
      process.terminate()
    else:
      process.kill()
    return True


def _stop_owned_process(process: subprocess.Popen, *, timeout: float = 15.0) -> None:
  if not math.isfinite(timeout) or timeout <= 0:
    raise ValueError("process cleanup timeout must be finite and positive")
  running = process.poll() is None
  signalled = _signal_owned_process_group(process, signal.SIGTERM)
  if not running and not signalled:
    return
  deadline = time.monotonic() + timeout
  try:
    process.wait(timeout=max(0.0, deadline - time.monotonic()))
  except subprocess.TimeoutExpired:
    pass

  while _owned_process_group_exists(process.pid) and time.monotonic() < deadline:
    time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
  if not _owned_process_group_exists(process.pid):
    return

  _signal_owned_process_group(process, signal.SIGKILL)
  kill_deadline = time.monotonic() + min(5.0, timeout)
  if process.poll() is None:
    try:
      process.wait(timeout=max(0.0, kill_deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
      pass
  while _owned_process_group_exists(process.pid) and time.monotonic() < kill_deadline:
    time.sleep(min(0.1, max(0.0, kill_deadline - time.monotonic())))
  if _owned_process_group_exists(process.pid):
    # Linux can retain killed descendants as zombies until their new parent
    # reaps them. They no longer execute or own model/GPU resources.
    return


def _owned_process_group_exists(process_group: int) -> bool:
  try:
    os.killpg(process_group, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    return True
  return True


def _assert_managed_endpoint_available(
  endpoint: str, *, bind_host: str | None
) -> None:
  """Refuse to launch when another process already owns the managed port."""
  endpoint_host, port, _ = _endpoint_parts(endpoint)
  host = bind_host or endpoint_host
  if host == "localhost":
    host = "127.0.0.1"
  family = socket.AF_INET6 if ":" in host else socket.AF_INET
  address: tuple[str, int] | tuple[str, int, int, int]
  address = (host, port, 0, 0) if family == socket.AF_INET6 else (host, port)
  with socket.socket(family, socket.SOCK_STREAM) as listener:
    try:
      listener.bind(address)
    except OSError as error:
      raise DeploymentError(
        f"refusing managed server start: {host}:{port} is already in use"
      ) from error


def _install_shutdown_handlers() -> dict[signal.Signals, Any]:
  previous: dict[signal.Signals, Any] = {}

  def interrupt(signum: int, _frame: object) -> None:
    raise DeploymentInterrupted(signum)

  for signum in (signal.SIGINT, signal.SIGTERM):
    try:
      old = signal.getsignal(signum)
      signal.signal(signum, interrupt)
    except ValueError:
      # signal.signal is only legal in the main interpreter thread.
      continue
    previous[signum] = old
  return previous


def _restore_shutdown_handlers(previous: dict[signal.Signals, Any]) -> None:
  for signum, handler in previous.items():
    signal.signal(signum, handler)


def _execute_owned_process(plan: ProcessPlan) -> int:
  """Run a foreground child in an owned session with signal-safe cleanup."""
  previous_handlers = _install_shutdown_handlers()
  process: subprocess.Popen | None = None
  try:
    process = subprocess.Popen(
      plan.command,
      cwd=plan.cwd,
      env=plan.environment,
      start_new_session=True,
    )
    return process.wait()
  finally:
    try:
      if process is not None:
        _stop_owned_process(process)
    finally:
      _restore_shutdown_handlers(previous_handlers)


def run_deployment(
  manifest_path: Path,
  *,
  output_dir: Path,
  model_family: str | None = None,
  profile: RuntimeProfile | None = None,
  service_mode: str = "managed",
  endpoint: str | None = None,
  num_trials: int = 1,
  seed_start: int = 0,
  video_count: int = 0,
  execute_steps: int | str | None = None,
  smoke_seconds: float | None = None,
  max_sim_seconds: float = 90.0,
  guard_mode: str = "observe",
  server_timeout: float = 600.0,
  evaluator_args: Sequence[str] = (),
  dry_run: bool = False,
) -> dict[str, Any]:
  resolved, manifest, adapter = load_deployment_manifest(
    manifest_path, model_family=model_family
  )
  if adapter.evaluate_backend is None:
    raise CapabilityError(f"run is not implemented for {adapter.task} + {adapter.model_family}")
  if not math.isfinite(server_timeout) or server_timeout <= 0:
    raise DeploymentError("server-timeout must be finite and positive")
  if profile is not None and not dry_run:
    if adapter.runtime_mode == "websocket-sidecar" and service_mode == "managed":
      profile.validate(
        require_python=True,
        require_project=adapter.model_family in {"egosteer", "lingbot-vla2"},
      )
    elif adapter.runtime_mode == "local-worker":
      profile.validate(require_python=True, require_project=True)
    elif adapter.runtime_mode == "embedded":
      profile.validate()
  serve_plan = None
  selected_endpoint = endpoint
  if adapter.runtime_mode == "websocket-sidecar":
    if service_mode == "managed":
      if adapter.serve_backend is None:
        raise CapabilityError(
          f"managed serving is unavailable for {adapter.task} + {adapter.model_family}; "
          "use --service-mode external --endpoint ..."
        )
      serve_plan = build_serve_plan(
        resolved,
        manifest,
        adapter,
        profile=profile,
        endpoint=endpoint,
      )
      selected_endpoint = serve_plan.endpoint
    elif service_mode == "external":
      if endpoint is None:
        raise DeploymentError("external service mode requires --endpoint")
      _, _, selected_endpoint = _endpoint_parts(endpoint)
    else:
      raise DeploymentError("service-mode must be managed or external")
  elif service_mode == "external":
    raise DeploymentError(
      f"external service mode is not applicable to {adapter.runtime_mode}"
    )
  elif endpoint is not None:
    raise DeploymentError(f"--endpoint is not applicable to {adapter.runtime_mode}")

  root = output_dir.expanduser().resolve()
  evaluation_dir = root / "evaluation"
  evaluation_plan = build_evaluation_plan(
    resolved,
    manifest,
    adapter,
    output_dir=evaluation_dir,
    service_mode=service_mode,
    endpoint=selected_endpoint,
    profile=profile,
    num_trials=num_trials,
    seed_start=seed_start,
    video_count=video_count,
    execute_steps=execute_steps,
    smoke_seconds=smoke_seconds,
    max_sim_seconds=max_sim_seconds,
    guard_mode=guard_mode,
    evaluator_args=evaluator_args,
  )
  record: dict[str, Any] = {
    "schema": "kaihand_deployment_run_v1",
    "schema_version": 1,
    "stage": "deploy",
    "operation": "run",
    "task": adapter.task,
    "model_family": adapter.model_family,
    "runtime_mode": adapter.runtime_mode,
    "service_mode": service_mode,
    "deployment_manifest": str(resolved),
    "deployment_id": manifest["deployment_id"],
    "deployment_manifest_rewritten": False,
    "runtime_profile": None if profile is None else profile.as_dict(),
    "endpoint": selected_endpoint,
    "managed_server": serve_plan is not None,
    "server": None if serve_plan is None else serve_plan.as_dict(),
    "evaluation": evaluation_plan.as_dict(),
    "evaluation_dir": str(evaluation_dir),
    "num_trials": num_trials,
    "seed_start": seed_start,
    "video_count": video_count,
    "execute_steps": execute_steps,
    "smoke_seconds": smoke_seconds,
    "max_sim_seconds": smoke_seconds if smoke_seconds is not None else max_sim_seconds,
    "guard_mode": guard_mode,
    "diagnostic_only": smoke_seconds is not None or guard_mode == "observe",
    "formal_metrics_requested": smoke_seconds is None and guard_mode == "strict",
    "formal_metrics_valid": False if smoke_seconds is not None or guard_mode == "observe" else None,
    "status": "planned" if dry_run else "starting",
  }
  if dry_run:
    return record
  if root.exists():
    raise FileExistsError(f"run output must be a new directory: {root}")
  if adapter.verify_backend is not None:
    verification_profile = (
      None
      if adapter.runtime_mode == "websocket-sidecar" and service_mode == "external"
      else profile
    )
    record["verification"] = verify_deployment(
      resolved,
      model_family=adapter.model_family,
      profile=verification_profile,
      execute_external=True,
    )
  else:
    record["verification"] = {
      "status": "runner-owned",
      "validator": None,
    }
  root.mkdir(parents=True, exist_ok=False)
  run_manifest = root / "run_manifest.json"
  record["started_at"] = datetime.now(timezone.utc).isoformat()
  _write_json(run_manifest, record)

  process: subprocess.Popen | None = None
  evaluation_process: subprocess.Popen | None = None
  server_log = None
  previous_handlers = _install_shutdown_handlers()
  try:
    if serve_plan is not None:
      if selected_endpoint is None:  # pragma: no cover - ensured by serve plan.
        raise DeploymentError("managed WebSocket deployment has no endpoint")
      _assert_managed_endpoint_available(
        selected_endpoint, bind_host=adapter.bind_host
      )
      server_log = (root / "server.log").open("xb")
      process = subprocess.Popen(
        serve_plan.command,
        cwd=serve_plan.cwd,
        env=serve_plan.environment,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
      )
      record["server_pid"] = process.pid
      metadata = wait_for_server(
        selected_endpoint,
        manifest,
        adapter,
        timeout=server_timeout,
        process=process,
      )
      record["server_metadata"] = metadata
    elif adapter.runtime_mode == "websocket-sidecar":
      record["server_metadata"] = wait_for_server(
        selected_endpoint,
        manifest,
        adapter,
        timeout=server_timeout,
      )
    record["status"] = "evaluating"
    _write_json(run_manifest, record)
    evaluation_process = subprocess.Popen(
      evaluation_plan.command,
      cwd=evaluation_plan.cwd,
      env=evaluation_plan.environment,
      start_new_session=True,
    )
    record["evaluation_pid"] = evaluation_process.pid
    evaluation_returncode = evaluation_process.wait()
    record["evaluation_returncode"] = evaluation_returncode
    record["status"] = "complete" if evaluation_returncode == 0 else "failed"
    return record
  except BaseException as error:
    record["status"] = "error"
    record["error"] = f"{type(error).__name__}: {error}"
    raise
  finally:
    if evaluation_process is not None:
      try:
        _stop_owned_process(evaluation_process)
        record["evaluation_process_stopped"] = True
      except BaseException as error:  # Preserve the evaluation exception/result.
        record["evaluation_process_stopped"] = False
        record["evaluation_cleanup_error"] = f"{type(error).__name__}: {error}"
    if process is not None:
      try:
        _stop_owned_process(process)
        record["managed_server_stopped"] = True
      except BaseException as error:  # Preserve the evaluation exception/result.
        record["managed_server_stopped"] = False
        record["server_cleanup_error"] = f"{type(error).__name__}: {error}"
    if server_log is not None:
      server_log.close()
    _restore_shutdown_handlers(previous_handlers)
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(run_manifest, record)


def execute_prepare(plan: ProcessPlan) -> int:
  return _execute_owned_process(plan)


def write_pipeline_deployment(
  output_dir: Path,
  *,
  task: str,
  model_family: str,
  prepare_plan: ProcessPlan,
  profile: RuntimeProfile | None,
  verification: dict[str, Any],
) -> Path:
  """Add orchestration provenance beside, never inside, the frozen manifest."""
  output = output_dir.expanduser().resolve(strict=True)
  manifest_path, manifest, adapter = load_deployment_manifest(
    output / "deployment_manifest.json", model_family=model_family
  )
  if adapter.task != task:
    raise DeploymentError(
      f"prepared deployment task {adapter.task!r} differs from {task!r}"
    )
  destination = output / "pipeline_deployment.json"
  if destination.exists():
    raise FileExistsError(f"refusing to overwrite pipeline record: {destination}")
  record = {
    "schema": "kaihand_pipeline_deployment_v1",
    "schema_version": 1,
    "stage": "deploy",
    "operation": "prepare",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "task": task,
    "model_family": model_family,
    "runtime_mode": adapter.runtime_mode,
    "prepare_command": list(prepare_plan.command),
    "prepare_cwd": str(prepare_plan.cwd),
    "runtime_profile": None if profile is None else profile.as_dict(),
    "deployment_manifest": manifest_path.name,
    "deployment_manifest_sha256": _sha256_file(manifest_path),
    "deployment_id": manifest["deployment_id"],
    "frozen_manifest_rewritten": False,
    "verification": verification,
  }
  destination.write_text(
    json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  return destination


def execute_serve(plan: ProcessPlan) -> int:
  return _execute_owned_process(plan)


__all__ = (
  "ADAPTERS",
  "CapabilityError",
  "DeploymentAdapter",
  "DeploymentError",
  "DeploymentInterrupted",
  "ProcessPlan",
  "adapter_catalog",
  "adapter_for",
  "build_evaluation_plan",
  "build_prepare_plan",
  "build_serve_plan",
  "execute_prepare",
  "execute_serve",
  "load_deployment_manifest",
  "normalize_model_family",
  "probe_server",
  "run_deployment",
  "validate_server_metadata",
  "verify_deployment",
  "wait_for_server",
  "write_pipeline_deployment",
)
