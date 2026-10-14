"""Export configuration-only OA deployment dependencies; never copy samples or weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def export(root: Path, output: Path, instruction: str | None = None) -> dict:
    root = root.expanduser().resolve(strict=True)
    settings = json.loads((root / "config/cpm.json").read_text())
    manifest = json.loads((root / "meta/manifest.json").read_text())
    names = ["config/cpm.json", "meta/tactile_scale.json", "meta/action_norm_stats.json", "meta/manifest.json"]
    if "recovery" in settings:
        names += [settings["recovery"][key]["file"] for key in ("normalization", "negative_policy")]
    if instruction is None:
        source = root / "meta/task_instruction.json"
        if source.exists():
            instruction = json.loads(source.read_text())["task"]
        else:
            source = Path(manifest["sources"]["pi05_root"]) / "meta/tasks.jsonl"
            tasks = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
            if len(tasks) != 1:
                raise ValueError("Expected a single task; pass --instruction explicitly")
            instruction = tasks[0]["task"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Instruction must be nonempty")
    for name in names:
        source = (root / name).resolve(strict=True)
        if not source.is_relative_to(root):
            raise ValueError("Metadata path leaves data root")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name in names:
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        (output / "meta/task_instruction.json").write_text(json.dumps({"task": instruction}, ensure_ascii=False) + "\n")
        result = {
            "schema": "oa_inference_metadata_v1",
            "source": str(root),
            "output": str(output),
            "contains_samples": False,
            "contains_weights": False,
            "sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in names},
        }
        (output / "bundle.json").write_text(json.dumps(result, indent=2) + "\n")
    except Exception:
        # Preserve an incomplete export for inspection; never remove caller data.
        (output / "EXPORT_INCOMPLETE").touch()
        raise
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instruction")
    args = parser.parse_args()
    print(json.dumps(export(args.data_root, args.output, args.instruction), indent=2))
