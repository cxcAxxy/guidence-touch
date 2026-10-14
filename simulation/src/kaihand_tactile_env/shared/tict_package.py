"""Offline session-level assembly and safe relocation of local T-ICT releases.

Pixels, per-frame documents and sidecars are copied byte-for-byte. Only local
manifests/statistics are assembled; this does not invoke an upstream trainer.
"""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .tict_export import CHANNELS, SIDES, build_action_window, sha256_file
from .tict_relocation import (
  INTEGRITY_SCHEMA,
  SPLITS,
  _check_digest,
  _digest,
  _local_file,
  _read_json,
  _safe_session,
  _selector_for_root,
  _write_json,
)
from .tict_relocation import (
  rebase_tict_release as _rebase_release,
)
from .tict_validation import validate_tict_release

MANIFEST_SCHEMA = "kaihand-tict-package-input-v1"


class _Moments:
  """Population moments accumulated without retaining all overlapping windows."""

  def __init__(self):
    self.count = 0
    self.mean = None
    self.m2 = None

  def add(self, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
      return
    if not np.isfinite(values).all():
      raise ValueError("nonfinite training statistics input")
    count = len(values)
    mean = values.mean(axis=0)
    m2 = ((values - mean) ** 2).sum(axis=0)
    if self.count == 0:
      self.count, self.mean, self.m2 = count, mean, m2
      return
    delta = mean - self.mean
    total = self.count + count
    self.m2 += m2 + delta**2 * self.count * count / total
    self.mean += delta * count / total
    self.count = total

  def report(self) -> dict[str, Any]:
    if self.count == 0:
      return {"count": 0, "mean": None, "std": None}
    return {
      "count": self.count,
      "mean": self.mean.tolist(),
      "std": np.sqrt(np.maximum(self.m2 / self.count, 0)).tolist(),
    }


def _training_statistics(root: Path, sessions: list[str], windows: dict) -> dict:
  wrist_stats, finger_stats = _Moments(), _Moments()
  tactile_stats = {name: _Moments() for name in CHANNELS}
  for session in sessions:
    with np.load(
      root / "tict_sidecars" / session / "fingertip_tactile_v1.npz", allow_pickle=False
    ) as sidecar:
      relative = sidecar["T_fingertip_to_wrist"]
      finger_valid = sidecar["finger_valid"]
      names = sidecar["frame_names"].tolist()
      cameras = np.zeros((len(names), 4, 4))
      wrists = np.broadcast_to(np.eye(4), (len(names), 2, 4, 4)).copy()
      wrist_valid = np.zeros((len(names), 2), dtype=np.bool_)
      for index, name in enumerate(names):
        frame = _read_json(
          root
          / "production"
          / session
          / "09_humanego_adapter/preprocess/all_data"
          / name
          / "training_data.json"
        )
        cameras[index] = frame["metadata"]["c2w"]
        hands = frame["entities"]["hands_hawor_v3"]
        for side_index, side in enumerate(SIDES):
          if side in hands:
            wrists[index, side_index] = hands[side]["T_hand_to_world"]
            wrist_valid[index, side_index] = True
      for start in windows["sessions"][session]:
        horizon = windows["horizon"]
        future = slice(start + 1, start + horizon + 1)
        action = build_action_window(wrists, relative, cameras, start, horizon)
        slots = action.reshape(horizon, 2, 6, 9)
        wrist_stats.add(slots[:, :, 0, :3][wrist_valid[future]])
        valid_fingers = finger_valid[future] & wrist_valid[future, :, None]
        finger_stats.add(slots[:, :, 1:, :3][valid_fingers])
      for index, name in enumerate(CHANNELS):
        tactile_stats[name].add(
          sidecar["tactile_mean"][..., index][
            sidecar["tactile_channel_mask"][..., index]
          ]
        )
  return {
    "schema_version": "kaihand-tict-train-statistics-v1",
    "sessions": sessions,
    "split": "train",
    "position_sampling": "all published valid future action slots; window overlap retained",
    "wrist_xyz": wrist_stats.report(),
    "shared_ten_finger_xyz": finger_stats.report(),
    "tactile_sampling": "each recorded frame once, only true channel masks",
    "tactile_channels": {key: value.report() for key, value in tactile_stats.items()},
    "rotation_6d_normalization": "none",
    "rgb_normalization": "divide uint8 by 255 only; no ImageNet normalization",
    "raw_release_is_normalized": False,
    "zero_std_policy": "reported literally; downstream must choose and freeze epsilon",
    "upstream_stats_implementation_verified": False,
    "calculation": "recomputed from published train JSON/sidecar; population moments",
  }


def _copy_payload(source: Path, target: Path, relative: str, *, link_payloads=False) -> dict:
  before = _digest(source, relative)
  destination = target / relative
  destination.parent.mkdir(parents=True, exist_ok=True)
  if link_payloads:
    os.link(_local_file(source, relative), destination)
  else:
    shutil.copyfile(_local_file(source, relative), destination)
  _check_digest(target, before)
  return before


def package_tict_releases(manifest_path: str | Path, output_dir: str | Path, *, link_payloads=False, reuse_source_audits=False) -> dict:
  """Assemble explicitly assigned complete sessions into a new portable release."""
  manifest_path = Path(manifest_path).expanduser().resolve()
  manifest = _read_json(manifest_path)
  if manifest.get("schema_version") != MANIFEST_SCHEMA:
    raise ValueError(f"package input schema must be {MANIFEST_SCHEMA}")
  assignments = manifest.get("sessions")
  if not isinstance(assignments, list) or not assignments:
    raise ValueError("package input requires explicit session assignments")
  partitions = {name: [] for name in SPLITS}
  inputs = []
  seen, source_hashes = set(), set()
  for item in assignments:
    session = _safe_session(item.get("session_id"))
    split = item.get("split")
    if split not in SPLITS or session in seen:
      raise ValueError(
        "invalid split or duplicate session assignment; no session leakage"
      )
    seen.add(session)
    path = Path(item["release_dir"]).expanduser()
    source = (
      (manifest_path.parent / path).resolve()
      if not path.is_absolute()
      else path.resolve()
    )
    # Hash checks use canonical relative paths before invoking the regular validator.
    if reuse_source_audits and not item.get("source_audit_path"):
      raise ValueError("reuse_source_audits requires a passing independent source audit for every session")
    selector, payload = _selector_for_root(source, source, verify_payloads=not reuse_source_audits)
    report = ({"valid": True, "scope": "reuse independent source audit; full assembled validation follows",
               "reused_source_audit": True} if reuse_source_audits else validate_tict_release(source))
    if not report["valid"]:
      raise ValueError(f"input release {session} failed validation: {report['errors']}")
    audit = _read_json(_local_file(source, "dataset_audit.json"))
    if (
      audit.get("passed") is not True
      or len(audit["sessions"]) != 1
      or audit["sessions"][0]["session_id"] != session
    ):
      raise ValueError(
        "each input must be a verified single-session release matching session_id"
      )
    source_hash = audit["source"]["sha256"]
    if source_hash in source_hashes:
      raise ValueError(
        "duplicate source episode hash; renamed sessions must not leak across splits"
      )
    source_hashes.add(source_hash)
    partitions[split].append(session)
    source_audit = None
    if item.get("source_audit_path") is not None:
      path = Path(item["source_audit_path"]).expanduser()
      source_audit = (
        (manifest_path.parent / path).resolve()
        if not path.is_absolute()
        else path.resolve()
      )
      external_report = _read_json(source_audit)
      if (
        external_report.get("schema_version") != "kaihand-tict-source-audit-v1"
        or external_report.get("valid") is not True
        or external_report.get("session_id") != session
        or external_report.get("source", {}).get("sha256") != source_hash
        or external_report.get("release", {}).get("sidecar_sha256")
        != audit["sessions"][0]["sidecar_sha256"]
        or external_report.get("release", {}).get("dataset_audit_sha256")
        != sha256_file(_local_file(source, "dataset_audit.json"))
      ):
        raise ValueError(
          "source_audit_path must be a passing audit of this session, source HDF5 hash and exact release"
        )
    inputs.append(
      (source, session, split, selector, payload, audit, report, source_audit)
    )
    if reuse_source_audits and len(inputs) % 20 == 0:
      print(f"EgoTouch assembly: prepared {len(inputs)}/{len(assignments)} audited sessions", flush=True)
  if not partitions["train"]:
    raise ValueError("at least one explicit train session is required for statistics")
  destination = Path(output_dir).expanduser().resolve()
  if destination.exists():
    raise FileExistsError(f"output already exists: {destination}")
  if any(destination.is_relative_to(source) for source, *_ in inputs):
    raise ValueError("output must not be placed inside a source release")
  destination.parent.mkdir(parents=True, exist_ok=True)
  staging = Path(
    tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
  )
  try:
    selectors, audits, sources, files = [], [], [], []
    windows = {
      "schema_version": "kaihand-tict-window-starts-v1",
      "horizon": 50,
      "sessions": {},
    }
    for (
      source,
      session,
      split,
      selector,
      payload,
      audit,
      report,
      source_audit,
    ) in inputs:
      for relative in payload:
        if reuse_source_audits and link_payloads:
          destination_file = staging / relative
          destination_file.parent.mkdir(parents=True, exist_ok=True)
          os.link(_local_file(source, relative), destination_file)
        else:
          files.append(_copy_payload(source, staging, relative, link_payloads=link_payloads))
      for record in selector["records"]:
        record["rgb_path"] = str(
          staging / Path(record["training_data_path"]).with_name("rgb.png")
        )
      selectors.extend(selector["records"])
      input_windows = _read_json(_local_file(source, "window_starts.json"))
      windows["sessions"][session] = input_windows["sessions"][session]
      provenance = staging / "source_provenance" / session
      provenance.mkdir(parents=True)
      for filename in (
        "dataset_audit.json",
        "train_statistics.json",
        "split_manifest.json",
        "selector_manifest.json",
        "window_starts.json",
        "DATA_CONTRACT.md",
      ):
        original = _local_file(source, filename)
        destination_file = provenance / filename
        before = sha256_file(original)
        shutil.copyfile(original, destination_file)
        if sha256_file(destination_file) != before:
          raise ValueError(f"source changed while copying {filename}")
      _write_json(provenance / "local_validation.json", report)
      if source_audit is not None:
        before_hash = sha256_file(source_audit)
        shutil.copyfile(source_audit, provenance / "source_audit.json")
        if sha256_file(provenance / "source_audit.json") != before_hash:
          raise ValueError("source audit changed while copying")
      entry = copy.deepcopy(audit["sessions"][0])
      entry.update(
        split=split,
        export_audit_path=str((provenance / "dataset_audit.json").relative_to(staging)),
        source_audit_verified=source_audit is not None,
        source_audit_path=str((provenance / "source_audit.json").relative_to(staging))
        if source_audit is not None
        else None,
      )
      audits.append(entry)
      sources.append(
        {
          "session_id": session,
          "original_release_root": str(source),
          "source": audit["source"],
        }
      )
      if reuse_source_audits and len(audits) % 20 == 0:
        print(f"EgoTouch assembly: linked {len(audits)}/{len(inputs)} sessions", flush=True)
    _write_json(
      staging / "split_manifest.json",
      {
        "schema_version": "kaihand-tict-split-v1",
        "split_unit": "session",
        "splits": partitions,
      },
    )
    _write_json(
      staging / "selector_manifest.json",
      {
        "schema_version": "kaihand-tict-selector-v1",
        "img_name": "rgb.png",
        "records": selectors,
      },
    )
    _write_json(staging / "window_starts.json", windows)
    print("EgoTouch assembly: train-only statistics", flush=True)
    _write_json(
      staging / "train_statistics.json",
      _training_statistics(staging, partitions["train"], windows),
    )
    audit = {
      "schema_version": "kaihand-tict-audit-v1",
      "contract_version": "kaihand-tict-release-v1",
      "created_utc": datetime.now(timezone.utc).isoformat(),
      "passed": True,
      "validation_scope": ("reused independent source audits; one complete assembled validation" if reuse_source_audits else
                           "complete local validation of each source and assembled release"),
      "upstream_loader_verified": False,
      "sessions": audits,
      "sources": sources,
      "all_source_audits_verified": all(
        entry["source_audit_verified"] for entry in audits
      ),
      "split_assignment": "explicit input manifest; entire sessions only; unique source hashes",
      "limitations": ["local manifest schemas; upstream loader/launcher not executed"],
    }
    _write_json(staging / "dataset_audit.json", audit)
    _write_json(staging / "package_input_manifest.json", manifest)
    (staging / "DATA_CONTRACT.md").write_text(_PACKAGE_CONTRACT, encoding="utf-8")
    (staging / "TRANSFER_README.md").write_text(_TRANSFER_README, encoding="utf-8")
    shutil.copyfile(
      Path(__file__).with_name("tict_relocation.py"), staging / "rebase_paths.py"
    )
    print("EgoTouch assembly: one complete output validation", flush=True)
    report = validate_tict_release(staging)
    if not report["valid"]:
      raise ValueError(f"assembled release failed validation: {report['errors']}")
    _write_json(staging / "package_validation.json", report)
    # Selector absolute paths alone depend on the installation root. Everything
    # else is immutable under relocation and included in the transfer checksum.
    print("EgoTouch assembly: transfer checksums", flush=True)
    files = []
    for directory, subdirs, filenames in os.walk(staging, followlinks=False):
      for name in subdirs:
        if (Path(directory) / name).is_symlink():
          raise ValueError("unexpected symlink in assembled directory")
      for name in filenames:
        path = Path(directory) / name
        if path == staging / "selector_manifest.json":
          continue
        if path.is_symlink() or not path.is_file():
          raise ValueError("unexpected non-regular assembled payload")
        files.append({"path": str(path.relative_to(staging)), "bytes": path.stat().st_size,
                      "sha256": sha256_file(path)})
    files.sort(key=lambda item: item["path"])
    _write_json(
      staging / "transfer_integrity.json",
      {
        "schema_version": INTEGRITY_SCHEMA,
        "files": files,
        "mutable_files": ["selector_manifest.json", "relocation_history/*"],
        "selector_integrity": "RGB bytes/hash and canonical session/frame paths are checked on rebase",
      },
    )
    # The full validator just verified every canonical path and RGB digest.
    # Relocation changes only the selector prefix, not any payload bytes.
    relocated = {"schema_version": "kaihand-tict-selector-v1", "img_name": "rgb.png", "records": selectors}
    for record in selectors:
      record["rgb_path"] = str(destination / Path(record["training_data_path"]).with_name("rgb.png"))
    _write_json(staging / "selector_manifest.json", relocated)
    if destination.exists():
      raise FileExistsError(f"output created concurrently: {destination}")
    os.rename(staging, destination)
    return {
      "output": str(destination),
      "sessions": audits,
      "splits": partitions,
      "valid": True,
    }
  except Exception as error:
    _write_json(
      staging / "FAILED.json", {"error": str(error), "output_published": False}
    )
    raise


def rebase_tict_release(root: str | Path) -> dict:
  """Relocate with immutable byte checks plus the full local H50/SE3 validator."""
  return _rebase_release(root, validator=validate_tict_release)


_PACKAGE_CONTRACT = """# KaiHand assembled T-ICT release v1

This package preserves each source session's PNG, per-frame JSON and fingertip
tactile NPZ bytes exactly. See source_provenance/<session>/DATA_CONTRACT.md for
the geometry, force units, camera and raw recording conventions of each session.
The source contracts describe their original one-session train-only smoke split;
the authoritative split in this assembled package is split_manifest.json.

Sessions are assigned explicitly to train/validation/test before publication.
No frame-level random split is used; identical source HDF5 hashes are rejected
even under different session names. Every H50 window stays in its source session.
train_statistics.json is recomputed from the published TRAIN sessions only.
Positions use all valid future action slots (including window overlap); tactile
statistics use each frame once and only true channel masks. Standard deviations
are population std. Wrist positions use the camera frozen at observation t;
fingers use their corresponding future wrist. Rotation-6D is not z-scored.
Raw poses/forces remain unnormalized and RGB preprocessing remains uint8 / 255.

source_provenance preserves input export audits, manifests, statistics and local
validation. A supplied passing independent source audit is additionally copied
as source_audit.json and identified by source_audit_verified=true for its session.
Absence of this optional report is explicitly marked false; export preflight alone
is not an independent raw-to-release source audit. Old absolute source paths are
informational, not executable paths.
transfer_integrity.json checksums immutable files, including frame JSON/PNG and
sidecar NPZ. After moving the package run the rebase tool in TRANSFER_README.md.
Only selector absolute paths and relocation history may change during rebase.

These are versioned local manifest container schemas, not an assertion that an
unavailable upstream DataLoader or training launcher has accepted the package.
No remote machine path or screw-driving task is required by these tools.
"""

_TRANSFER_README = """# Receiving this T-ICT dataset

Copy the ENTIRE release directory. Keep production/, tict_sidecars/, manifests,
statistics, source_provenance/ and transfer_integrity.json together. Raw source
HDF5 files are not duplicated into this training package; their SHA-256 and
recording metadata remain in the source audit/NPZ for provenance.

On the receiving computer, use Python 3.11 or newer. No third-party packages or
original project checkout are needed for relocation. From the received directory:

```bash
python rebase_paths.py
```

The script defaults to its own directory, so it also works from any directory as
`python /your/received/release/rebase_paths.py`. An explicit `--root` is available.
Rebase verifies checksums, rebuilds selector absolute RGB paths from canonical
relative session/frame paths, and preserves the original selector plus a report
under relocation_history/. It never reads files via the old machine's paths or
modifies RGB, per-frame JSON, sidecar NPZ, split assignment or train statistics.
Repeat rebase after any later directory move. Treat source metadata paths as
provenance only. Distribute a checksum of the whole archive separately if you
also need transport authenticity; in-package hashes detect accidental corruption.

The standalone script verifies immutable hashes and safe manifest paths. The
publisher's full geometry/H50 validation is preserved in package_validation.json;
the standalone script does not rerun those NumPy/Pillow-dependent checks. If the
full project is available, optionally repeat all geometry/window/image checks:

```bash
pixi run python scripts/workcell/validate_tict.py /your/received/release
```

Training integration must use img_name=rgb.png and the published sidecar, split
and window manifests. Normalize positions/forces from train_statistics.json
only; do not fit statistics on held-out sessions or z-score rotation-6D. Read
DATA_CONTRACT.md and source contracts before mapping these local manifest
containers into the receiver's T-ICT loader. No remote launcher is included.
"""
