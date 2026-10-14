"""Named model runtimes, shared by conversion and deployment commands."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SECRET_PARTS = ("TOKEN", "PASSWORD", "SECRET", "PRIVATE_KEY", "API_KEY", "ACCESS_KEY")


@dataclass(frozen=True)
class RuntimeProfile:
  name: str
  python: Path | None
  project_root: Path | None
  env: dict[str, str]
  config_path: Path
  config_sha256: str

  def environment(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Apply only explicit overrides; never mutate the caller's environment."""
    return {**(os.environ if base is None else base), **self.env}

  def as_dict(self) -> dict:
    """Record the selected config identity without publishing credentials."""
    return {
      "name": self.name,
      "python": str(self.python) if self.python is not None else None,
      "project_root": str(self.project_root) if self.project_root is not None else None,
      "env": {
        key: "<redacted>" if any(part in key.upper() for part in _SECRET_PARTS) else value
        for key, value in sorted(self.env.items())
      },
      "config_path": str(self.config_path),
      "config_sha256": self.config_sha256,
    }

  def validate(self, *, require_python: bool = False, require_project: bool = False) -> None:
    """Preflight a local runtime when executing, rather than when planning."""
    if require_python and self.python is None:
      raise ValueError(f"runtime {self.name!r} must declare python")
    if self.python is not None and (
      not self.python.is_file() or not os.access(self.python, os.X_OK)
    ):
      raise ValueError(f"runtime Python is not executable: {self.python}")
    if require_project and self.project_root is None:
      raise ValueError(f"runtime {self.name!r} must declare project_root")
    if self.project_root is not None and not self.project_root.is_dir():
      raise ValueError(f"runtime project_root is not a directory: {self.project_root}")


def _path(value: object, *, field: str, base: Path) -> Path | None:
  if value is None:
    return None
  if not isinstance(value, str) or not value.strip():
    raise ValueError(f"runtime {field} must be a nonempty path string")
  result = Path(value).expanduser()
  if not result.is_absolute():
    result = base / result
  return result.resolve()


def load_runtime_profile(
  config: Path | None, name: str | None
) -> RuntimeProfile | None:
  """Read a TOML/JSON config; resolve paths relative to that config file.

  A missing config preserves the historical command defaults. If a config is
  supplied without a name, it must explicitly declare ``default_runtime``.
  Config files are data: shell expansion and executable expressions are never
  evaluated.
  """
  if config is None:
    if name is not None:
      raise ValueError("--runtime requires --runtime-config")
    return None
  path = Path(config).expanduser().resolve()
  contents = path.read_bytes()
  try:
    payload = (
      json.loads(contents) if path.suffix.lower() == ".json"
      else tomllib.loads(contents.decode("utf-8"))
    )
  except (ValueError, UnicodeDecodeError) as error:
    raise ValueError(f"invalid runtime configuration {path}: {error}") from error
  if not isinstance(payload, dict):
    raise ValueError("runtime configuration must contain an object/table")
  selected = name if name is not None else payload.get("default_runtime")
  if not isinstance(selected, str) or not selected:
    raise ValueError("select --runtime NAME or declare default_runtime in the config")
  runtimes = payload.get("runtimes")
  if not isinstance(runtimes, dict) or selected not in runtimes:
    available = ", ".join(sorted(runtimes)) if isinstance(runtimes, dict) else "none"
    raise ValueError(f"unknown runtime {selected!r}; available: {available}")
  row = runtimes[selected]
  if not isinstance(row, dict):
    raise ValueError(f"runtime {selected!r} must be an object/table")
  unknown = row.keys() - {"python", "project_root", "env"}
  if unknown:
    raise ValueError(f"unknown runtime fields: {', '.join(sorted(unknown))}")
  environment = row.get("env", {})
  if not isinstance(environment, dict):
    raise ValueError("runtime env must be an object/table of string overrides")
  for key, value in environment.items():
    if not isinstance(key, str) or _ENV_NAME.fullmatch(key) is None:
      raise ValueError(f"invalid environment variable name: {key!r}")
    if not isinstance(value, str) or "\x00" in value:
      raise ValueError(f"runtime env.{key} must be a string without NUL bytes")
  return RuntimeProfile(
    name=selected,
    python=_path(row.get("python"), field="python", base=path.parent),
    project_root=_path(row.get("project_root"), field="project_root", base=path.parent),
    env=dict(environment),
    config_path=path,
    config_sha256=hashlib.sha256(contents).hexdigest(),
  )
