"""Read-only cloud preflight for CPM inputs and genuine NAS checkpoint storage.

Run ``python -m openpi.touchguide.paths --root DATA_ROOT --output-dir RUN_DIR``.
This module never loads torch/DINO, creates directories, starts training or W&B,
or follows provenance-only paths to original pi0.5 checkpoints/source datasets.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

import h5py

NAS_FILESYSTEMS = frozenset({"nfs", "nfs4", "cifs", "smb3"})
REQUIRED_FILES = (
    "config/cpm.json",
    "config/train.json",
    "config/val.json",
    "meta/manifest.json",
    "index/train.npz",
    "index/val.npz",
    "index/val_batches.npz",
    "meta/tactile_scale.json",
    "meta/action_norm_stats.json",
    "meta/entry_config.json",
    "meta/splits.json",
)


class PathCheckError(ValueError):
    """A required input or checkpoint storage location failed the preflight."""


@dataclass(frozen=True)
class Mount:
    mountpoint: Path
    filesystem: str
    source: str
    mount_options: frozenset[str]
    super_options: frozenset[str]
    device: str


def _unescape_mount_field(value: str) -> str:
    return re.sub(r"\\(040|011|012|134)", lambda match: chr(int(match[1], 8)), value)


def parse_mountinfo(contents: str) -> list[Mount]:
    """Parse Linux mountinfo, including spaces escaped in mountpoint/source."""
    mounts = []
    for line in contents.splitlines():
        left, separator, right = line.partition(" - ")
        before, after = left.split(), right.split()
        if not separator or len(before) < 6 or len(after) < 3:
            raise PathCheckError(f"Malformed /proc/self/mountinfo entry: {line}")
        mounts.append(
            Mount(
                mountpoint=Path(_unescape_mount_field(before[4])),
                filesystem=after[0],
                source=_unescape_mount_field(after[1]),
                mount_options=frozenset(before[5].split(",")),
                super_options=frozenset(after[2].split(",")),
                device=before[2],
            )
        )
    if not mounts:
        raise PathCheckError("No mounts found in /proc/self/mountinfo")
    return mounts


def mount_for_path(path: Path, mounts: list[Mount]) -> Mount:
    """Select the longest containing mountpoint, including nested local mounts."""
    candidates = [mount for mount in mounts if path.is_relative_to(mount.mountpoint)]
    if not candidates:
        raise PathCheckError(f"Cannot identify the filesystem mount for {path}")
    depth = max(len(mount.mountpoint.parts) for mount in candidates)
    candidates = [mount for mount in candidates if len(mount.mountpoint.parts) == depth]
    if len(candidates) > 1:
        device = path.stat().st_dev
        identifier = f"{os.major(device)}:{os.minor(device)}"
        visible = [mount for mount in candidates if mount.device == identifier]
        if not visible:
            raise PathCheckError(
                f"Cannot identify the visible filesystem mount for {path}"
            )
        candidates = visible
    return candidates[-1]


def require_network_nas_mount(path: Path, mounts: list[Mount]) -> Mount:
    if not path.is_relative_to(Path("/nas")):
        raise PathCheckError(f"Checkpoint path must resolve beneath /nas: {path}")
    mount = mount_for_path(path, mounts)
    if mount.filesystem not in NAS_FILESYSTEMS:
        allowed = ", ".join(sorted(NAS_FILESYSTEMS))
        raise PathCheckError(
            f"Checkpoint path {path} is on {mount.filesystem} at {mount.mountpoint}; "
            f"require a genuine NAS network mount ({allowed}), not a local directory named /nas"
        )
    if (
        "ro" in mount.mount_options
        or "ro" in mount.super_options
        or "rw" not in mount.mount_options
    ):
        raise PathCheckError(
            f"NAS checkpoint mount is not read-write: {mount.mountpoint}"
        )
    if os.statvfs(path).f_flag & os.ST_RDONLY:
        raise PathCheckError(
            f"NAS checkpoint filesystem is read-only: {mount.mountpoint}"
        )
    return mount


def check_output_directory(output_dir: Path, mounts: list[Mount]) -> dict[str, Any]:
    output = output_dir.expanduser().resolve(strict=False)
    if not output.is_relative_to(Path("/nas")):
        raise PathCheckError(f"Output directory must resolve beneath /nas: {output}")
    parent = output
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir():
        raise PathCheckError(
            f"Output directory or its nearest existing parent is not a directory: {parent}"
        )
    mount = require_network_nas_mount(parent, mounts)
    if not os.access(parent, os.W_OK | os.X_OK):
        raise PathCheckError(
            f"Nearest existing NAS output parent is not writable/searchable: {parent}"
        )
    return {
        "directory": str(output),
        "nearest_existing_parent": str(parent),
        "mountpoint": str(mount.mountpoint),
        "filesystem": mount.filesystem,
        "mount_source": mount.source,
        "mount_read_write": True,
        "parent_writable": True,
    }


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise PathCheckError(f"Missing {description}: {path}")
    if not os.access(path, os.R_OK):
        raise PathCheckError(f"Unreadable {description}: {path}")
    return path.resolve(strict=True)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise PathCheckError(
            f"Cannot read JSON configuration {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise PathCheckError(f"Expected a JSON object: {path}")
    return value


def video_path(episode: h5py.File, attribute: str, episode_path: Path) -> Path:
    if attribute not in episode.attrs:
        raise PathCheckError(f"Missing video attribute {attribute} in {episode_path}")
    value = episode.attrs[attribute]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value:
        raise PathCheckError(f"Invalid video attribute {attribute} in {episode_path}")
    path = Path(value).expanduser()
    return require_file(path, f"{attribute} video referenced by {episode_path.name}")


def check_paths(
    root: str | Path, output_dir: str | Path, resume: str | Path | None = None
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise PathCheckError(f"Dataset root does not exist: {root}")
    try:
        mounts = parse_mountinfo(Path("/proc/self/mountinfo").read_text())
    except OSError as error:
        raise PathCheckError(
            f"Cannot inspect Linux mounts via /proc/self/mountinfo: {error}"
        ) from error
    output = check_output_directory(Path(output_dir), mounts)
    for name in REQUIRED_FILES:
        require_file(root / name, "CPM runtime file")
    manifest = read_json(root / "meta/manifest.json")
    config = read_json(root / "config/cpm.json")
    if manifest.get("schema") != "touchguide_cpm_positive_entry_v1":
        raise PathCheckError(
            "Dataset manifest must use touchguide_cpm_positive_entry_v1"
        )
    if config.get("schema") != "openpi_touchguide_cpm_config_v2":
        raise PathCheckError(
            "CPM configuration must use openpi_touchguide_cpm_config_v2"
        )
    if "recovery" in config:
        for key in ("normalization", "negative_policy"):
            entry = config["recovery"][key]
            require_file(root / entry["file"], f"Recovery CPM {key}")
    for split in ("train", "val"):
        entry = read_json(root / f"config/{split}.json")
        kwargs = entry.get("reader_kwargs", {})
        if (
            not isinstance(kwargs, dict)
            or not isinstance(kwargs.get("root"), str)
            or not kwargs["root"]
        ):
            raise PathCheckError(
                f"config/{split}.json must specify a nonempty reader_kwargs.root"
            )
        configured_root = Path(kwargs["root"]).expanduser().resolve(strict=False)
        if configured_root != root or kwargs.get("split") != split:
            raise PathCheckError(
                f"config/{split}.json reader_kwargs must reference root {root} and split {split}; "
                f"configured root is {configured_root}"
            )
        if kwargs.get("include_negatives") or entry.get("noise_on_read"):
            raise PathCheckError(
                f"config/{split}.json must use clean positive samples without a legacy negative pool"
            )
        if split == "val":
            fixed_index = entry.get("batching", {}).get("index_file")
            if fixed_index != "index/val_batches.npz":
                raise PathCheckError("Validation must reference index/val_batches.npz")
    vision_setting = config.get("vision", {}).get("directory")
    if not isinstance(vision_setting, str) or not vision_setting:
        raise PathCheckError("CPM config must specify vision.directory")
    vision = Path(vision_setting).expanduser().resolve(strict=False)
    if not vision.is_dir():
        raise PathCheckError(f"DINO backbone directory does not exist: {vision}")
    for name in (
        "backbone_manifest.json",
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
    ):
        require_file(vision / name, "DINO backbone artifact")
    scale = config.get("tactile", {}).get("statistics_file")
    if not isinstance(scale, str) or not scale:
        raise PathCheckError("CPM config must specify tactile.statistics_file")
    require_file(root / scale, "train-only tactile scale")
    splits = manifest.get("splits", {})
    episode_ids = []
    for split in ("train", "val"):
        ids = splits.get(split, {}).get("episode_indices")
        if not isinstance(ids, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in ids
        ):
            raise PathCheckError(f"Manifest has invalid {split} episode indices")
        episode_ids.extend(ids)
    if len(episode_ids) != 200 or len(set(episode_ids)) != 200:
        raise PathCheckError(
            f"Expected exactly 200 distinct episode IDs across train/val; found {len(episode_ids)} entries"
        )
    virtual_source_files: set[str] = set()
    h5_roots: Counter[str] = Counter()
    video_roots: Counter[str] = Counter()
    videos: set[str] = set()
    non_nas: set[str] = set()
    for episode_id in sorted(episode_ids):
        episode_path = root / "episodes" / f"episode_{episode_id:06d}.h5"
        resolved = require_file(episode_path, "episode H5/link target")
        h5_roots[str(resolved.parent)] += 1
        if not resolved.is_relative_to(Path("/nas")):
            non_nas.add(str(resolved))
        try:
            with h5py.File(episode_path, "r") as episode:
                for name in ("actions", "state_raw", "tactile_map_raw"):
                    if episode[name].is_virtual:
                        for source in episode[name].virtual_sources():
                            dependency = Path(source.file_name)
                            if not dependency.is_absolute():
                                dependency = episode_path.parent / dependency
                            checked = require_file(dependency, "shared VDS data source")
                            virtual_source_files.add(str(checked))
                for attribute in ("head_video", "right_wrist_video"):
                    video = video_path(episode, attribute, episode_path)
                    video_roots[str(video.parent)] += 1
                    videos.add(str(video))
                    if not video.is_relative_to(Path("/nas")):
                        non_nas.add(str(video))
        except (OSError, ValueError, KeyError) as error:
            if isinstance(error, PathCheckError):
                raise
            raise PathCheckError(
                f"Cannot inspect episode {episode_path}: {error}"
            ) from error
    episodes = root / "episodes"
    result = {
        "status": "passed",
        "dataset_root": str(root),
        "runtime_metadata_files": len(REQUIRED_FILES),
        "episode_files": len(episode_ids),
        "episode_directory_is_symlink": episodes.is_symlink(),
        "episode_directory_link": str(episodes.readlink())
        if episodes.is_symlink()
        else None,
        "episode_directory_resolved": str(episodes.resolve(strict=True)),
        "h5_target_directories": dict(h5_roots),
        "video_references": sum(video_roots.values()),
        "unique_video_files": len(videos),
        "shared_virtual_source_files": len(virtual_source_files),
        "video_directories": dict(video_roots),
        "non_nas_input_paths": sorted(non_nas),
        "vision_directory": str(vision),
        "vision_files": 4,
        "output": output,
        "model_loaded": False,
        "training_started": False,
        "directories_created": False,
    }
    if resume is not None:
        checkpoint = require_file(Path(resume).expanduser(), "resume checkpoint")
        if checkpoint.parent != Path(output["directory"]):
            raise PathCheckError(
                f"Resume checkpoint must belong to the output directory: {checkpoint.parent} != {output['directory']}"
            )
        mount = require_network_nas_mount(checkpoint, mounts)
        result["resume"] = {
            "path": str(checkpoint),
            "filesystem": mount.filesystem,
            "mountpoint": str(mount.mountpoint),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    try:
        result = check_paths(args.root, args.output_dir, args.resume)
    except (PathCheckError, OSError) as error:
        parser.exit(2, f"CPM path preflight failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
