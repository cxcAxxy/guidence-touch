#!/usr/bin/env bash
set -euo pipefail
# 可在这里直接改配置；也可以使用下方命令行参数覆盖。
AUGMENTATION_ROOT="${AUGMENTATION_ROOT:-/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_rules_v1_random10}"
RATIO="${RATIO:-1}"  # 1、5、10；all 按顺序训练全部三种比例。
export NUM_GPUS="${NUM_GPUS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${BATCH_SIZE}}"
export EPOCHS="${EPOCHS:-200}"
export SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-20}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/datasets/touchguidence}"
export MODEL_VARIANT="recovery-state-cpm"
usage() {
    cat <<'HELP'
Usage: bash scripts/train_poker_recovery_state_cpm.sh --ratio 1|5|10|all [training options]
  --ratio all   Sequentially train 1:1, 1:5, 1:10 on the same GPUs.
Each ratio starts a fresh model and receives its own timestamped NAS directory
and W&B run. If one run fails, the command stops before starting the next.
All other options are passed to train_poker_touchguide_cpm.sh.
For all, do not supply --resume, --run-name, --data-root or --model-variant.
Defaults: 8 GPUs, global batch 128, validation batch 128, 200 epochs per ratio,
BF16, checkpoint last/best every 20 epochs, W&B online.
--check-paths checks all selected ratios without starting training.
HELP
    bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/train_poker_touchguide_cpm.sh" --help
}
args=()
while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --ratio)
            if (( $# < 2 )); then echo "Missing value for --ratio" >&2; exit 2; fi
            RATIO="$2"; shift 2 ;;
        *) args+=("$1"); shift ;;
    esac
done
if [[ "${RATIO}" != 1 && "${RATIO}" != 5 && "${RATIO}" != 10 && "${RATIO}" != all ]]; then
    echo "--ratio must be 1, 5, 10 or all" >&2; exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ratios=("${RATIO}")
if [[ "${RATIO}" == all ]]; then
    ratios=(1 5 10)
    for arg in "${args[@]}"; do
        case "${arg}" in
            --resume|--run-name|--data-root|--model-variant)
                echo "${arg} applies to a single run; select --ratio 1, 5 or 10." >&2; exit 2 ;;
        esac
    done
    if [[ -n "${RUN_NAME:-}" || -n "${RESUME_CHECKPOINT:-}" ]]; then
        echo "For --ratio all, unset RUN_NAME and RESUME_CHECKPOINT so each ratio starts a separate run." >&2
        exit 2
    fi
fi
for selected_ratio in "${ratios[@]}"; do
    export DATA_ROOT="${AUGMENTATION_ROOT}/ratio_1_${selected_ratio}"
    echo "运行干净:增强=1:${selected_ratio}，数据入口: ${DATA_ROOT}"
    if bash "${repo_root}/scripts/train_poker_touchguide_cpm.sh" "${args[@]}"; then
        echo "比例 1:${selected_ratio} 已完成。"
    else
        result=$?
        echo "比例 1:${selected_ratio} 运行失败（退出码 ${result}），已停止后续比例。" >&2
        exit "${result}"
    fi
done
