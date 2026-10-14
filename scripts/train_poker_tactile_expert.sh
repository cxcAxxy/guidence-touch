#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/path/to/simulator-env/bin/python}"
pi05_root="${PI05_ROOT:-/path/to/datasets/sim/poker-draw/pi05/0920_200}"
tactile_root="${TACTILE_ROOT:-/path/to/datasets/sim/poker-draw/pi05_trex/0920_200}"
jax_checkpoint="${JAX_CHECKPOINT:-${pi05_root}/checkpoints/pi05_kaihand/poker_draw_bs128_noskip_25k/24999}"
base_pytorch="${BASE_PYTORCH:-/path/to/datasets/sim/poker-draw/pi05_trex/pi05_pytorch_25k}"
f6_checkpoint="${F6_CHECKPOINT:-/path/to/datasets/sim/poker-draw/pi05_trex/vqvae_runs/f6_right_k128_20260923_101736/epoch_0021.pt}"
map_checkpoint="${MAP_CHECKPOINT:-/path/to/datasets/sim/poker-draw/pi05_trex/map_runs/map_right_denoise256_20260923_110027/best.pt}"
output_root="${OUTPUT_ROOT:-/path/to/datasets/sim/poker-draw/pi05_trex/tactile_expert_runs}"
run_name="${RUN_NAME:-late4_right_$(date +%Y%m%d_%H%M%S)}"
output_dir="${OUTPUT_DIR:-${output_root}/${run_name}}"
resume_args=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    resume_args+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi

export PYTHONPATH="${repo_root}/src:${repo_root}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${output_root}/huggingface_cache}"
export WANDB_DIR="${output_dir}"
export TOKENIZERS_PARALLELISM=false
# LeRobot's current video decoder emits this deprecation warning for every
# decoded frame. Hide only that warning so training progress remains readable.
torchvision_warning_filter="ignore:The video decoding and encoding capabilities of torchvision are deprecated:UserWarning"
export PYTHONWARNINGS="${PYTHONWARNINGS:+${PYTHONWARNINGS},}${torchvision_warning_filter}"
mkdir -p "${output_dir}" "${HF_HOME}"
cd "${repo_root}"

"${python_bin}" - <<'PY'
import h5py  # noqa: F401
import lerobot  # noqa: F401
import safetensors  # noqa: F401
import torch
import wandb  # noqa: F401

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to PyTorch; use a GPU-enabled environment")
print("GPU:", torch.cuda.get_device_name(0))
PY

if [[ ! -f "${base_pytorch}/model.safetensors" ]]; then
    echo "Converting the existing JAX pi0.5 checkpoint once..."
    "${python_bin}" examples/convert_jax_model_to_pytorch.py \
        --checkpoint_dir "${jax_checkpoint}" \
        --config_name pi05_poker_trex \
        --output_path "${base_pytorch}" \
        --precision bfloat16
fi
if [[ ! -r "${base_pytorch}/model.safetensors" ]]; then
    echo "Converted pi0.5 weights are not readable: ${base_pytorch}/model.safetensors" >&2
    echo "Set BASE_PYTORCH to a directory your training account can read, then run the script again." >&2
    exit 1
fi

"${python_bin}" -m openpi.tactile_expert.train \
    --pi05-root "${pi05_root}" \
    --tactile-root "${tactile_root}" \
    --norm-stats "${jax_checkpoint}/assets/normalizer/norm_stats.json" \
    --base-pytorch "${base_pytorch}" \
    --f6-checkpoint "${f6_checkpoint}" \
    --map-checkpoint "${map_checkpoint}" \
    --output-dir "${output_dir}" \
    --steps "${STEPS:-3000}" \
    --checkpoint-interval "${CHECKPOINT_INTERVAL:-0}" \
    "${resume_args[@]}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --stride "${STRIDE:-4}" \
    --val-stride "${VAL_STRIDE:-16}" \
    --val-ratio "${VAL_RATIO:-0.1}" \
    --lr "${LR:-1e-4}" \
    --num-workers "${NUM_WORKERS:-0}" \
    --eval-interval "${EVAL_INTERVAL:-250}" \
    --eval-batches "${EVAL_BATCHES:-20}" \
    --seed "${SEED:-42}" \
    --wandb-mode "${WANDB_MODE:-online}" \
    --wandb-entity "${WANDB_ENTITY:-xianchichen8-hunan-university-library}" \
    --wandb-project "${WANDB_PROJECT:-poker-draw-tactile-expert}" \
    --wandb-run-name "${run_name}"
