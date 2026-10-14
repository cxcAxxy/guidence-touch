#!/usr/bin/env bash
set -euo pipefail

# Editable settings. Offline model verification only; no training or simulator run.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/path/to/touchguidance/bin/python}"
base_checkpoint="${BASE_CHECKPOINT:-/path/to/datasets/sim/poker-draw/pi05_trex/pi05_pytorch_20k}"
cpm_checkpoint="${CPM_CHECKPOINT:-/path/to/datasets/touchguidence/cpm_dino_b128_g8_20260928_173559/best.pt}"
data_root="${DATA_ROOT:-/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1}"
inputs="${INPUTS:-/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/baseline_20k/verification/fixed_inputs.npz}"
output_dir="${OUTPUT_DIR:-/path/to/datasets/touchguidence/inference_checks/sampler_$(date +%Y%m%d_%H%M%S)_$$}"
guidance_scale="${GUIDANCE_SCALE:-1.0}"
guidance_steps="${GUIDANCE_STEPS:-3}"
num_steps="${NUM_STEPS:-10}"
precision="${PRECISION:-bfloat16}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${repo_root}/src:${repo_root}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
cd "${repo_root}"

"${python_bin}" -u scripts/touchguide/verify_guided_sampler.py \
    --base-checkpoint "${base_checkpoint}" \
    --cpm-checkpoint "${cpm_checkpoint}" \
    --data-root "${data_root}" \
    --inputs "${inputs}" \
    --output-dir "${output_dir}" \
    --guidance-scale "${guidance_scale}" \
    --guidance-steps "${guidance_steps}" \
    --num-steps "${num_steps}" \
    --precision "${precision}"
