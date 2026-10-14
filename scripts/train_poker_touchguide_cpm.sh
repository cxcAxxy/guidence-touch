#!/usr/bin/env bash
set -euo pipefail

# 在这里修改配置；本脚本只训练 CPM，不加载或更新 pi0.5。
# DINOv2-Base 同时编码视觉与触觉并保持冻结，其他约 122M 参数参与训练。
# batch / epoch / LR / warmup 参考 TouchGuide Table XIV；其他值为本地设置。
PYTHON_BIN="${PYTHON_BIN:-/path/to/touchguidance/bin/python}"
DATA_ROOT="${DATA_ROOT:-/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/datasets/touchguidence}"
RUN_NAME="${RUN_NAME:-}"  # 留空时按实际global batch和GPU数自动命名。
MODEL_VARIANT="${MODEL_VARIANT:-cpm}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"  # 续训填原运行的 last.pt；保持下面训练参数一致。

NUM_GPUS="${NUM_GPUS:-16}"  # 单机16卡；其他配置请显式传 --num-gpus。
BATCH_SIZE="${BATCH_SIZE:-128}"  # 所有GPU合计的global batch；DDP每卡至少2条样本。
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-}"  # 新运行默认跟随训练batch；续训从原run配置恢复。
EPOCHS="${EPOCHS:-200}"
LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
PRECISION="${PRECISION:-bf16}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
WANDB_ENTITY="${WANDB_ENTITY:-xianchichen8-hunan-university-library}"
WANDB_PROJECT="${WANDB_PROJECT:-poker-draw-touchguide-cpm}"
WANDB_MODE="${WANDB_MODE:-online}"

CHECK_PATHS_ONLY=false
CHECK_GPU_ONLY=false

usage() {
    cat <<'HELP'
Train the poker TouchGuide CPM on CUDA GPUs. Checkpoints must be on NAS.
Usage: bash scripts/train_poker_touchguide_cpm.sh [options]
  --python-bin PATH       Python with the CPM dependencies (default: touchguidance)
  --data-root PATH        Prepared CPM dataset entry
  --output-root PATH      NAS output parent; each run gets its own directory
  --run-name NAME         Run name; default includes a timestamp
  --model-variant NAME    cpm (default), state-cpm, or recovery-state-cpm
  --num-gpus N            Single-node GPU count; default: 16
  --batch-size N          Global total batch across GPUs; default: 128
  --val-batch-size N      Validation candidate group size; new runs default to training batch
  --epochs N              Default: 200
  --lr FLOAT              Default: 1e-5
  --num-workers N         Default: 4
  --precision bf16|fp32   Default: bf16
  --save-every-epochs N   Save last/best every N epochs; default: 20
  --save-interval N       Optional step snapshots; 0 disables (default)
  --log-interval N        Default: 20 updates
  --seed N               Default: 42
  --wandb-enabled         Record to W&B (default)
  --wandb-mode MODE       online, offline or disabled
  --wandb-entity NAME     Default: xianchichen8-hunan-university-library
  --wandb-project NAME    Default: poker-draw-touchguide-cpm
  --resume PATH           Resume in the original NAS checkpoint directory
  --check-paths           Verify paths and NAS only; do not start training
  --check-only            Real GPU forward/backward check; no optimizer update
  -h, --help              Show this help
All settings can also be edited in the header or supplied as environment variables.
Multiple GPUs use DDP with global contrastive pairs. Duration uses --epochs.
Global batch must divide by GPU count and provide at least 2 samples per GPU.
HELP
}

while (( $# )); do
    case "$1" in
        --wandb-enabled) WANDB_MODE="online"; shift; continue ;;
        --check-paths) CHECK_PATHS_ONLY=true; shift; continue ;;
        --check-only) CHECK_GPU_ONLY=true; shift; continue ;;
        -h|--help) usage; exit 0 ;;
        --python-bin) setting="PYTHON_BIN" ;;
        --data-root) setting="DATA_ROOT" ;;
        --output-root) setting="OUTPUT_ROOT" ;;
        --run-name) setting="RUN_NAME" ;;
        --model-variant) setting="MODEL_VARIANT" ;;
        --num-gpus) setting="NUM_GPUS" ;;
        --batch-size) setting="BATCH_SIZE" ;;
        --val-batch-size) setting="VAL_BATCH_SIZE" ;;
        --epochs) setting="EPOCHS" ;;
        --lr) setting="LR" ;;
        --num-workers) setting="NUM_WORKERS" ;;
        --precision) setting="PRECISION" ;;
        --save-every-epochs) setting="SAVE_EVERY_EPOCHS" ;;
        --save-interval) setting="SAVE_INTERVAL" ;;
        --log-interval) setting="LOG_INTERVAL" ;;
        --seed) setting="SEED" ;;
        --wandb-mode) setting="WANDB_MODE" ;;
        --wandb-entity) setting="WANDB_ENTITY" ;;
        --wandb-project) setting="WANDB_PROJECT" ;;
        --resume) setting="RESUME_CHECKPOINT" ;;
        *) echo "Unknown option: $1; use --help for supported options" >&2; exit 2 ;;
    esac
    if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
        echo "Missing value for $1" >&2
        exit 2
    fi
    printf -v "${setting}" '%s' "$2"
    shift 2
done
if [[ "${MODEL_VARIANT}" != cpm && "${MODEL_VARIANT}" != state-cpm && "${MODEL_VARIANT}" != recovery-state-cpm ]]; then
    echo "--model-variant must be cpm, state-cpm or recovery-state-cpm" >&2
    exit 2
fi
if [[ -z "${RUN_NAME}" ]]; then
    run_prefix="cpm_dino"
    if [[ "${MODEL_VARIANT}" == state-cpm ]]; then
        run_prefix="state_cpm_dino"
    fi
    if [[ "${MODEL_VARIANT}" == recovery-state-cpm ]]; then
        run_prefix="recovery_state_cpm_$(basename "${DATA_ROOT}")"
    fi
    RUN_NAME="${run_prefix}_b${BATCH_SIZE}_g${NUM_GPUS}_$(date +%Y%m%d_%H%M%S)"
fi
if [[ "${RUN_NAME}" == */* || "${RUN_NAME}" == . || "${RUN_NAME}" == .. ]]; then
    echo "RUN_NAME must be a name, not a path; use --output-root for the NAS path" >&2
    exit 2
fi

if [[ ! "${SAVE_EVERY_EPOCHS}" =~ ^[1-9][0-9]*$ || ! "${SAVE_INTERVAL}" =~ ^[0-9]+$ ]]; then
    echo "--save-every-epochs must be positive and --save-interval must be nonnegative" >&2
    exit 2
fi

if [[ ! "${NUM_GPUS}" =~ ^[1-9][0-9]*$ || ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] ||
   (( BATCH_SIZE % NUM_GPUS != 0 || BATCH_SIZE / NUM_GPUS < 2 )); then
    echo "Global --batch-size must be divisible by --num-gpus, with at least 2 samples per GPU" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${OUTPUT_ROOT}/${RUN_NAME}"
resume_args=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    output_dir="$(cd "$(dirname "${RESUME_CHECKPOINT}")" && pwd)"
    resume_args=(--resume "${RESUME_CHECKPOINT}")
fi

# 新运行的验证候选组与训练global batch相同。旧run续训沿用其原始固定组。
if [[ -z "${VAL_BATCH_SIZE}" ]]; then
    if [[ -z "${RESUME_CHECKPOINT}" ]]; then
        VAL_BATCH_SIZE="${BATCH_SIZE}"
    elif [[ -f "${output_dir}/run_config.json" ]]; then
        VAL_BATCH_SIZE="$("${PYTHON_BIN}" - "${output_dir}/run_config.json" <<'PYJSON'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    value = json.load(stream).get("arguments", {}).get("val_batch_size")
print("" if value is None else value)
PYJSON
)"
    fi
fi
if [[ -n "${VAL_BATCH_SIZE}" && ! "${VAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--val-batch-size must be an integer >= 2" >&2
    exit 2
fi
if [[ -n "${VAL_BATCH_SIZE}" ]] && (( VAL_BATCH_SIZE < 2 )); then
    echo "--val-batch-size must be an integer >= 2" >&2
    exit 2
fi
val_batch_args=()
if [[ -n "${VAL_BATCH_SIZE}" ]]; then
    val_batch_args=(--val-batch-size "${VAL_BATCH_SIZE}")
fi

cd "${repo_root}"
export PYTHONPATH="${repo_root}/src:${repo_root}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
"${PYTHON_BIN}" -m openpi.touchguide.paths \
    --root "${DATA_ROOT}" --output-dir "${output_dir}" "${resume_args[@]}"
if [[ "${CHECK_PATHS_ONLY}" == true ]]; then
    exit 0
fi
echo "CPM 输出目录: ${output_dir}"
launch=("${PYTHON_BIN}" -u -m openpi.touchguide.train)
if (( NUM_GPUS > 1 )); then
    launch=("${PYTHON_BIN}" -u -m torch.distributed.run
        --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS}" --max-restarts=0
        -m openpi.touchguide.train_distributed)
fi
check_args=()
if [[ "${CHECK_GPU_ONLY}" == true ]]; then
    check_args=(--check-only)
fi
echo "GPU数量: ${NUM_GPUS}; global batch: ${BATCH_SIZE}; 每卡batch: $(( BATCH_SIZE / NUM_GPUS )); 验证batch: ${VAL_BATCH_SIZE:-原数据固定分组}"
exec "${launch[@]}" \
    --root "${DATA_ROOT}" \
    --model-variant "${MODEL_VARIANT}" \
    --output-dir "${output_dir}" \
    --run-name "${RUN_NAME}" \
    --batch-size "${BATCH_SIZE}" \
    "${val_batch_args[@]}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --min-lr-ratio "${MIN_LR_RATIO}" \
    --grad-clip "${GRAD_CLIP}" \
    --num-workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --precision "${PRECISION}" \
    --log-interval "${LOG_INTERVAL}" \
    --save-every-epochs "${SAVE_EVERY_EPOCHS}" \
    --save-interval "${SAVE_INTERVAL}" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-project "${WANDB_PROJECT}" \
    "${resume_args[@]}" "${check_args[@]}"
