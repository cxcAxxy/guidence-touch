#!/usr/bin/env python3
"""Download and verify the pinned public DINOv2-Base CPM dependency.

Uses only Python's standard library. Downloads configuration and safetensors
bytes; it never imports ML libraries, instantiates a model, or starts training.
An existing published target is verified read-only and is never overwritten.
Failed downloads retain their isolated staging directory for inspection.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import UTC
from datetime import datetime
import errno
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

REPOSITORY = "facebook/dinov2-base"
REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
MODEL_SHA256 = "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841"
MODEL_BYTES = 346345912
EXPECTED_FILES = {
    "config.json": {
        "size_bytes": 548,
        "sha256": "f7ff4cfa73d2f70647dbf6950541ad25d73082d54c2e7e9bded160c7656b2a70",
    },
    "preprocessor_config.json": {
        "size_bytes": 436,
        "sha256": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
    },
    "model.safetensors": {"size_bytes": MODEL_BYTES, "sha256": MODEL_SHA256},
}
FILES = tuple(EXPECTED_FILES)
DEFAULT_TARGET = Path("/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/backbones/dinov2_base_f9e44c8")


def checksum(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_files(root: Path, records: dict | None = None) -> dict:
    observed = {}
    for name in FILES:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Dependency file missing or is a symlink: {path}")
        size, sha256 = path.stat().st_size, checksum(path)
        expected = EXPECTED_FILES[name]
        if size != expected["size_bytes"] or sha256 != expected["sha256"]:
            raise ValueError(f"Invalid pinned {name}: bytes={size}, sha256={sha256}")
        if name.endswith(".json"):
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError(f"Configuration must contain a JSON object: {path}")
        if records is not None:
            record = records[name]
            if record["size_bytes"] != size or record["sha256"] != sha256:
                raise ValueError(f"Existing dependency hash differs from its manifest: {path}")
        observed[name] = {"size_bytes": size, "sha256": sha256}
    return observed


def verify_existing(target: Path, *, published_target: Path | None = None) -> dict:
    if not target.is_dir() or target.is_symlink():
        raise FileExistsError(f"Target already exists and is not a plain directory: {target}")
    manifest_path = target / "backbone_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileExistsError(f"Existing target lacks its provenance manifest; refusing to overwrite: {target}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "touchguide_public_backbone_v1":
        raise FileExistsError(f"Existing target has an incompatible manifest schema: {target}")
    if manifest.get("repository") != REPOSITORY or manifest.get("revision") != REVISION:
        raise FileExistsError(f"Existing target is a different dependency version: {target}")
    destination = target if published_target is None else published_target
    if manifest.get("published_target") != str(destination):
        raise FileExistsError(f"Existing provenance refers to a different target: {target}")
    if not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != set(FILES):
        raise FileExistsError(f"Existing provenance must record all three pinned files: {target}")
    for name in FILES:
        record = manifest["files"][name]
        expected_url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}?download=true"
        if record.get("source_url") != expected_url:
            raise FileExistsError(f"Existing dependency source differs from the pinned public URL: {name}")
    verify_files(target, manifest["files"])
    return manifest


def download_file(staging: Path, name: str, timeout: float) -> dict:
    url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}?download=true"
    partial = staging / (name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "touchguide-pinned-dependency/1.0"})
    total = 0
    with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("xb") as output:
        if response.status != 200:
            raise RuntimeError(f"Unexpected download status {response.status} for {name}")
        expected_header = response.headers.get("Content-Length")
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            output.write(block)
            total += len(block)
            if name == "model.safetensors":
                print(f"Downloading {name}: {total:,} bytes", flush=True)
        output.flush()
        os.fsync(output.fileno())
    if expected_header is not None and total != int(expected_header):
        raise ValueError(f"Incomplete HTTP response for {name}: {total} != {expected_header}")
    partial.rename(staging / name)
    print(f"Downloaded {name}: {total:,} bytes", flush=True)
    # Keep the pinned public URL, not temporary signed CDN redirect URLs.
    return {"source_url": url}


def publish_without_overwrite(staging: Path, target: Path) -> None:
    # Linux renameat2(RENAME_NOREPLACE) atomically refuses even an empty target
    # directory created concurrently. A normal os.rename would replace it.
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, os.fsencode(staging), -100, os.fsencode(target), 1)
        if result == 0:
            return
        error = ctypes.get_errno()
        if error not in (errno.EINVAL, errno.ENOSYS, errno.ENOTSUP):
            raise OSError(error, os.strerror(error), str(target))
    # Some NAS mounts lack RENAME_NOREPLACE. mkdir atomically reserves a new
    # destination and fails if any target already exists. Replace only this
    # invocation's own empty reservation using the supported normal rename.
    target.mkdir(mode=0o700, exist_ok=False)
    reservation = target.stat(follow_symlinks=False)
    try:
        current = target.stat(follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (reservation.st_dev, reservation.st_ino):
            raise RuntimeError("Destination reservation changed; refusing to publish")
        if any(target.iterdir()):
            raise RuntimeError("Destination reservation is no longer empty; refusing to publish")
        os.rename(staging, target)
    except BaseException:
        try:
            current = target.stat(follow_symlinks=False)
            if (current.st_dev, current.st_ino) == (reservation.st_dev, reservation.st_ino):
                target.rmdir()  # Only removes our own still-empty reservation.
        except OSError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--publish-staging",
        type=Path,
        help="Publish a previously downloaded staging directory after rechecking all hashes",
    )
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("HTTP timeout must be positive")
    target = args.output_dir.expanduser().absolute()
    if target.exists() or target.is_symlink():
        verify_existing(target)
        print(
            json.dumps(
                {
                    "status": "verified_existing",
                    "target": str(target),
                    "revision": REVISION,
                    "model_sha256": MODEL_SHA256,
                }
            ),
            flush=True,
        )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.publish_staging is not None:
        staging = args.publish_staging.expanduser().absolute()
        manifest = verify_existing(staging, published_target=target)
        if manifest.get("published_target") != str(target):
            raise ValueError("Staging manifest refers to a different publication target")
        staging.chmod(0o755)
        publish_without_overwrite(staging, target)
        print(
            json.dumps(
                {
                    "status": "published_verified_staging",
                    "target": str(target),
                    "revision": REVISION,
                    "model_sha256": MODEL_SHA256,
                }
            ),
            flush=True,
        )
        return
    staging = Path(tempfile.mkdtemp(prefix=target.name + ".downloading.", dir=target.parent))
    print(f"Staging public dependency: {staging}", flush=True)
    try:
        sources = {name: download_file(staging, name, args.timeout_seconds) for name in FILES}
        observed = verify_files(staging)
        manifest = {
            "schema": "touchguide_public_backbone_v1",
            "repository": REPOSITORY,
            "revision": REVISION,
            "published_target": str(target),
            "prepared_at_utc": datetime.now(UTC).isoformat(),
            "files": {name: {**observed[name], **sources[name]} for name in FILES},
            "expected_model": {"size_bytes": MODEL_BYTES, "sha256": MODEL_SHA256},
            "expected_files": EXPECTED_FILES,
            "source_metadata_url": (f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true"),
            "model_loaded": False,
            "training_started": False,
        }
        manifest_path = staging / "backbone_manifest.json"
        with manifest_path.open("x") as output:
            output.write(json.dumps(manifest, indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
        staging.chmod(0o755)
        publish_without_overwrite(staging, target)
        print(
            json.dumps(
                {
                    "status": "downloaded_and_verified",
                    "target": str(target),
                    "revision": REVISION,
                    "model_sha256": MODEL_SHA256,
                }
            ),
            flush=True,
        )
    except BaseException:
        print(f"Dependency preparation failed. Staging retained: {staging}", flush=True)
        raise


if __name__ == "__main__":
    main()
