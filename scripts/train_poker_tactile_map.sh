#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/path/to/workspace/miniconda3/envs/trex/bin/python}"
data_root="${DATA_ROOT:-/path/to/datasets/sim/poker-draw/pi05_trex/0920_200}"
output_root="${OUTPUT_ROOT:-/path/to/datasets/sim/poker-draw/pi05_trex/map_runs}"
run_name="${RUN_NAME:-map_right_denoise256_$(date +%Y%m%d_%H%M%S)}"
output_dir="${OUTPUT_DIR:-${output_root}/${run_name}}"
wandb_entity="${WANDB_ENTITY:-xianchichen8-hunan-university-library}"
wandb_project="${WANDB_PROJECT:-poker-draw-tactile-map-ae}"
wandb_mode="${WANDB_MODE:-online}"

"${python_bin}" - <<'PY'
import h5py  # noqa: F401
import torch
import wandb  # noqa: F401

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to PyTorch; run from a GPU-enabled environment")
print("GPU:", torch.cuda.get_device_name(0))
PY

mkdir -p "${output_dir}"
export WANDB_DIR="${output_dir}"
cd "${repo_root}"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python_bin}" -m openpi.tactile_vqvae.train_tactile_map \
    --data-root "${data_root}" \
    --output-dir "${output_dir}" \
    --device cuda \
    --hand-mode right \
    --stride 4 \
    --val-ratio 0.1 \
    --token-dim 256 \
    --batch-size 128 \
    --epochs 30 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --grad-clip 1.0 \
    --mask-prob 0.10 \
    --contact-weight 3.0 \
    --num-workers 0 \
    --seed 42 \
    --wandb-mode "${wandb_mode}" \
    --wandb-entity "${wandb_entity}" \
    --wandb-project "${wandb_project}" \
    --wandb-run-name "${run_name}" \
    --wandb-log-interval 20
