#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/path/to/touchguidance/bin/python}"
base_checkpoint="${BASE_CHECKPOINT:-/path/to/datasets/sim/poker-draw/pi05_trex/pi05_pytorch_20k}"
cpm_checkpoint="${CPM_CHECKPOINT:-/path/to/datasets/touchguidence/cpm_dino_b128_g8_20260928_173559/best.pt}"
data_root="${DATA_ROOT:-/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1}"
inputs="${INPUTS:-/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/baseline_20k/verification/fixed_inputs.npz}"
output_dir="${OUTPUT_DIR:-${repo_root}/tac_forward_guidance/artifacts/verification_$(date +%Y%m%d_%H%M%S)_$$}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${repo_root}:${repo_root}/src:${repo_root}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
cd "$repo_root"

"$python_bin" -u -m tac_forward_guidance.verify \
  --base-checkpoint "$base_checkpoint" --cpm-checkpoint "$cpm_checkpoint" \
  --data-root "$data_root" --inputs "$inputs" --output-dir "$output_dir" \
  --guidance-scale "${GUIDANCE_SCALE:-0.3}" --guidance-steps "${GUIDANCE_STEPS:-3}" \
  --gradient-mode "${GRADIENT_MODE:-full}" --max-update-rms "${MAX_UPDATE_RMS:-0.01}" \
  --num-steps "${NUM_STEPS:-10}" --precision "${PRECISION:-bfloat16}" --batch-size "${BATCH_SIZE:-1}"
