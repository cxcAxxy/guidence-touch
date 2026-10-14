"""Verify the frozen source files that define a formal evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_hash(value: Any) -> str:
  encoded = json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
  ).encode()
  return hashlib.sha256(encoded).hexdigest()


def _stable_file_identity(path: Path) -> tuple[int, str] | None:
  """Return size and digest, or None when the file changes while hashing."""
  before = path.stat()
  digest = hashlib.sha256()
  with path.open("rb") as source:
    while chunk := source.read(16 * 1024 * 1024):
      digest.update(chunk)
  after = path.stat()
  identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
  identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
  if identity_before != identity_after:
    return None
  return after.st_size, digest.hexdigest()


def verify_evaluation_sources(deployment: dict[str, Any], root: Path) -> dict[str, Any]:
  """Compare the current source tree with a deployment's frozen file rows."""
  rows = deployment.get("evaluation_source_files")
  expected_hash = deployment.get("evaluation_source_sha256")
  base = root.resolve()
  report: dict[str, Any] = {
    "status": "verified",
    "expected_sha256": expected_hash,
    "manifest_rows_sha256": None,
    "actual_sha256": None,
    "file_count": 0,
    "changed_files": [],
    "errors": [],
  }
  if rows is None or expected_hash is None:
    report["status"] = "unverifiable"
    report["errors"].append(
      "deployment manifest lacks evaluation_source_files or evaluation_source_sha256"
    )
    return report
  if not isinstance(rows, list) or not rows:
    report["status"] = "mismatch"
    report["errors"].append("evaluation_source_files must be a nonempty list")
    return report
  if not isinstance(expected_hash, str) or _SHA256.fullmatch(expected_hash) is None:
    report["status"] = "mismatch"
    report["errors"].append(
      "evaluation_source_sha256 must be a lowercase hexadecimal SHA-256"
    )
    return report

  report["file_count"] = len(rows)
  report["manifest_rows_sha256"] = canonical_hash(rows)
  if report["manifest_rows_sha256"] != expected_hash:
    report["errors"].append(
      "evaluation_source_sha256 does not authenticate evaluation_source_files"
    )

  actual_rows: list[dict[str, Any]] = []
  seen: set[str] = set()
  for index, row in enumerate(rows):
    if not isinstance(row, dict):
      report["errors"].append(f"evaluation_source_files[{index}] is not an object")
      continue
    relative = row.get("path")
    size = row.get("size")
    digest = row.get("sha256")
    if (
      not isinstance(relative, str)
      or not relative
      or "\\" in relative
      or relative != PurePosixPath(relative).as_posix()
      or PurePosixPath(relative).is_absolute()
      or ".." in PurePosixPath(relative).parts
    ):
      report["errors"].append(
        f"evaluation_source_files[{index}] has an unsafe noncanonical path"
      )
      continue
    if relative in seen:
      report["errors"].append(f"duplicate evaluation source path: {relative}")
      continue
    seen.add(relative)
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
      report["errors"].append(f"invalid evaluation source size: {relative}")
      continue
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
      report["errors"].append(f"invalid evaluation source SHA-256: {relative}")
      continue

    try:
      path = (base / relative).resolve(strict=True)
      path.relative_to(base)
      if not path.is_file():
        raise FileNotFoundError(path)
      current = _stable_file_identity(path)
    except (OSError, ValueError):
      report["changed_files"].append(
        {"path": relative, "issues": ["missing_or_outside_source_root"]}
      )
      continue
    if current is None:
      report["changed_files"].append(
        {"path": relative, "issues": ["changed_while_hashing"]}
      )
      continue
    actual_size, actual_digest = current
    actual_rows.append({"path": relative, "size": actual_size, "sha256": actual_digest})
    issues = []
    if actual_size != size:
      issues.append("size")
    if actual_digest != digest:
      issues.append("sha256")
    if issues:
      report["changed_files"].append(
        {
          "path": relative,
          "issues": issues,
          "expected_size": size,
          "actual_size": actual_size,
          "expected_sha256": digest,
          "actual_sha256": actual_digest,
        }
      )

  if len(actual_rows) == len(rows):
    report["actual_sha256"] = canonical_hash(actual_rows)
  if report["errors"] or report["changed_files"]:
    report["status"] = "mismatch"
  elif report["actual_sha256"] != expected_hash:
    report["status"] = "mismatch"
    report["errors"].append("current evaluation source aggregate hash differs")
  return report
