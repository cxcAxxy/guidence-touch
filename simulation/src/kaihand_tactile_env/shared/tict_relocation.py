"""Relocate a T-ICT package using Python's standard library only.

This file is copied unchanged into delivered releases as rebase_paths.py. The
same implementation backs the project API, which adds full geometry validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SPLITS = ("train", "validation", "test")
INTEGRITY_SCHEMA = "kaihand-tict-transfer-integrity-v1"


def sha256_file(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path.name} must contain a JSON object")
  return value


def _write_json(path: Path, value: Any) -> None:
  path.write_text(
    json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    + "\n",
    encoding="utf-8",
  )


def _safe_session(value: Any) -> str:
  if not isinstance(value, str) or not re.fullmatch(
    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value
  ):
    raise ValueError("session_id must be a safe single path component")
  return value


def _local_file(root: Path, relative: str) -> Path:
  """Accept canonical relative files only; never follow payload symlinks."""
  if not isinstance(relative, str):
    raise ValueError(f"unsafe relative release path: {relative!r}")
  path = Path(relative)
  if (
    path.is_absolute()
    or not path.parts
    or any(part in {".", ".."} for part in path.parts)
    or str(path) != relative
  ):
    raise ValueError(f"unsafe relative release path: {relative!r}")
  candidate = root
  for part in path.parts:
    candidate /= part
    if candidate.is_symlink():
      raise ValueError(f"symlink is not allowed in release payload: {relative}")
  if not candidate.is_file() or not candidate.resolve().is_relative_to(root):
    raise ValueError(f"missing or escaping release file: {relative}")
  return candidate


def _digest(root: Path, relative: str) -> dict[str, Any]:
  path = _local_file(root, relative)
  return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _check_digest(root: Path, record: dict[str, Any]) -> None:
  actual = _digest(root, record["path"])
  if actual != record:
    raise ValueError(f"payload hash/byte-count mismatch: {record['path']}")


def _selector_for_root(root: Path, target: Path, *, verify_payloads=True) -> tuple[dict, list[str]]:
  """Reconstruct expected paths from session/frame identities, never old rgb_path."""
  for name in ("window_starts.json", "train_statistics.json", "DATA_CONTRACT.md"):
    _local_file(root, name)
  selector = _read_json(_local_file(root, "selector_manifest.json"))
  if (
    selector.get("schema_version") != "kaihand-tict-selector-v1"
    or selector.get("img_name") != "rgb.png"
  ):
    raise ValueError("unsupported selector manifest")
  split = _read_json(_local_file(root, "split_manifest.json"))
  if (
    split.get("schema_version") != "kaihand-tict-split-v1"
    or split.get("split_unit") != "session"
    or set(split.get("splits", {})) != set(SPLITS)
    or any(not isinstance(split["splits"][name], list) for name in SPLITS)
  ):
    raise ValueError("unsupported session split manifest")
  sessions = [_safe_session(s) for name in SPLITS for s in split["splits"][name]]
  if not sessions or len(sessions) != len(set(sessions)):
    raise ValueError("empty sessions or train/validation/test session leakage")
  seen = set()
  payload = []
  records = selector.get("records")
  if not isinstance(records, list):
    raise ValueError("selector records must be a list")
  for record in records:
    if not isinstance(record, dict):
      raise ValueError("each selector record must be an object")
    session = _safe_session(record.get("session_id"))
    frame = record.get("frame_name")
    if (
      session not in sessions
      or not isinstance(frame, str)
      or not re.fullmatch(r"[0-9]{5}", frame)
    ):
      raise ValueError("selector contains an unknown session or unsafe frame name")
    if (session, frame) in seen:
      raise ValueError("duplicate selector frame")
    seen.add((session, frame))
    relative = (
      Path("production") / session / "09_humanego_adapter/preprocess/all_data" / frame
    )
    document = str(relative / "training_data.json")
    rgb = str(relative / "rgb.png")
    if record.get("training_data_path") != document:
      raise ValueError("selector training_data_path is not the canonical frame path")
    _local_file(root, document)
    if verify_payloads:
      _check_digest(
        root,
        {"path": rgb, "bytes": record["rgb_bytes"], "sha256": record["rgb_sha256"]},
      )
    else:
      _local_file(root, rgb)
    record["rgb_path"] = str(target / rgb)
    payload.extend([document, rgb])
  audit = _read_json(_local_file(root, "dataset_audit.json"))
  audit_sessions = audit["sessions"]
  if len(audit_sessions) != len(sessions) or {
    item["session_id"] for item in audit_sessions
  } != set(sessions):
    raise ValueError("audit sessions differ from split sessions")
  for item in audit_sessions:
    session = item["session_id"]
    count = item["frame_count"]
    if (
      isinstance(count, bool)
      or not isinstance(count, int)
      or not 51 <= count <= 100_000
    ):
      raise ValueError("audit frame count must be an integer in [51, 100000]")
    expected = {(session, f"{i:05d}") for i in range(item["frame_count"])}
    if {key for key in seen if key[0] == session} != expected:
      raise ValueError("selector frames differ from audit frame count")
    relative = f"tict_sidecars/{session}/fingertip_tactile_v1.npz"
    if item["sidecar_path"] != relative:
      raise ValueError("audit sidecar path is not the canonical session path")
    if sha256_file(_local_file(root, relative)) != item["sidecar_sha256"]:
      raise ValueError("sidecar hash mismatch")
    payload.append(relative)
  return selector, payload


def rebase_tict_release(
  root: str | Path, *, validator=None, require_integrity: bool = False
) -> dict:
  """Rebuild selector paths at the receiving root after checking immutable bytes.

  Old absolute paths are treated as inert provenance; no file at those paths is
  read. A failed final contract check restores the exact previous selector.
  """
  root = Path(root).expanduser().resolve()
  selector, payload = _selector_for_root(root, root)
  integrity_path = root / "transfer_integrity.json"
  integrity = None
  if integrity_path.exists():
    integrity = _read_json(_local_file(root, "transfer_integrity.json"))
    if integrity.get("schema_version") != INTEGRITY_SCHEMA:
      raise ValueError("unsupported transfer integrity schema")
    paths = [item["path"] for item in integrity["files"]]
    if len(paths) != len(set(paths)) or not set(payload).issubset(paths):
      raise ValueError(
        "transfer integrity is duplicate or does not cover all payload files"
      )
    for item in integrity["files"]:
      _check_digest(root, item)
  if require_integrity and integrity is None:
    raise ValueError("standalone rebase requires transfer_integrity.json")
  before = [_digest(root, relative) for relative in payload]
  selector_path = _local_file(root, "selector_manifest.json")
  original = selector_path.read_bytes()
  history = root / "relocation_history"
  if history.is_symlink():
    raise ValueError("relocation history must not be a symlink")
  history.mkdir(exist_ok=True)
  record_dir = history / (
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
  )
  record_dir.mkdir()
  backup = record_dir / "selector_manifest.before.json"
  backup.write_bytes(original)
  temporary = record_dir / "selector_manifest.next.json"
  _write_json(temporary, selector)
  os.replace(temporary, selector_path)
  try:
    report = (
      validator(root)
      if validator is not None
      else {
        "valid": True,
        "validation_scope": "transfer hashes and canonical paths; publisher full validation preserved",
        "full_geometry_validation_rerun": False,
        "upstream_loader_verified": False,
      }
    )
    if not report["valid"]:
      raise ValueError(f"relocated release failed validation: {report['errors']}")
    for item in before:
      _check_digest(root, item)
    record = {
      "schema_version": "kaihand-tict-selector-rebase-v1",
      "created_utc": datetime.now(timezone.utc).isoformat(),
      "release_root": str(root),
      "previous_selector_sha256": sha256_file(backup),
      "new_selector_sha256": sha256_file(selector_path),
      "old_absolute_paths_used_for_io": False,
      "payload_files_verified_unchanged": len(before),
      "transfer_integrity_verified": integrity is not None,
      "validation": report,
    }
    _write_json(record_dir / "rebase_record.json", record)
    return record
  except Exception as error:
    temporary.write_bytes(original)
    os.replace(temporary, selector_path)
    _write_json(
      record_dir / "FAILED.json", {"error": str(error), "selector_restored": True}
    )
    raise


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--root",
    type=Path,
    default=Path(__file__).resolve().parent,
    help="received release root; defaults to this script's directory",
  )
  args = parser.parse_args()
  try:
    result = rebase_tict_release(args.root, require_integrity=True)
  except (ValueError, OSError, KeyError, TypeError) as error:
    parser.exit(1, f"{type(error).__name__}: {error}\n")
  print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
