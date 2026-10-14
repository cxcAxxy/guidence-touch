#!/usr/bin/env bash
set -euo pipefail

# One-time network configuration preparation. No training is launched.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/path/to/touchguidance/bin/python}"
dataset_root="${DATASET_ROOT:-/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1}"
vision_dir="${VISION_DIR:-/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/backbones/dinov2_base_f9e44c8}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${repo_root}/src:${repo_root}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
"${python_bin}" "${repo_root}/scripts/touchguide/prepare_dinov2_base.py" --output-dir "${vision_dir}"
"${python_bin}" "${repo_root}/scripts/touchguide/prepare_cpm.py" \
    --dataset-root "${dataset_root}" --vision-directory "${vision_dir}" --upgrade-from-cnn-v1
