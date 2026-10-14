#!/usr/bin/env bash
set -euo pipefail

# Edit these settings. Completed seeds are retained; interrupted seeds are archived and retried.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PYTHON="${MODEL_PYTHON:-/path/to/touchguidance/bin/python}"
SIM_PYTHON="${SIM_PYTHON:-/path/to/simulator-env/bin/python}"
SIM_ROOT="${SIM_ROOT:-/path/to/kaitactilesim}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/datasets/touchguidence/eval/pi05_vs_touchguidance_20261008_134126_1119009}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/path/to/datasets/sim/poker-draw/pi05_trex/pi05_pytorch_20k}"
CPM_CHECKPOINT="${CPM_CHECKPOINT:-/path/to/datasets/touchguidence/cpm_dino_b128_g8_20260928_173559/best.pt}"
DATA_ROOT="${DATA_ROOT:-/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-$HOME/.cache/openpi/big_vision/paligemma_tokenizer.model}"

SCALES=(5 10)
TARGET_TRIALS="${TARGET_TRIALS:-10}" # TOTAL per scale, including the three existing trials.
SEED_START="${SEED_START:-0}"       # Seeds 0..9; initially only seeds 3..9 will run.
GPU_ID="${GPU_ID:-0}"
SERVER_PORT="${SERVER_PORT:-18786}"
MAX_SIM_SECONDS="${MAX_SIM_SECONDS:-50}"
GUIDANCE_STEPS="${GUIDANCE_STEPS:-3}"
RECORD_FPS="${RECORD_FPS:-10}"
DRY_RUN="${DRY_RUN:-0}"             # 1 checks existing records and prints commands without starting inference.

[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo 'DRY_RUN must be 0 or 1' >&2; exit 2; }
extras=(--resume)
if [[ "$DRY_RUN" == 1 ]]; then extras+=(--dry-run); fi
export PYTHONDONTWRITEBYTECODE=1
cd "$repo_root"
for scale in "${SCALES[@]}"; do
  "$MODEL_PYTHON" -u scripts/touchguide/evaluate_poker_touchguide.py \
    --sim-root "$SIM_ROOT" --model-python "$MODEL_PYTHON" --sim-python "$SIM_PYTHON" \
    --base-checkpoint "$BASE_CHECKPOINT" --cpm-checkpoint "$CPM_CHECKPOINT" \
    --data-root "$DATA_ROOT" --tokenizer-model "$TOKENIZER_MODEL" \
    --output-root "$OUTPUT_ROOT" --run-name "guidance_scale_${scale}" \
    --variants pi05+touchguidance --num-trials "$TARGET_TRIALS" --seed-start "$SEED_START" \
    --gpu-id "$GPU_ID" --port "$SERVER_PORT" --server-wait-seconds 600 --trial-wall-limit 3600 \
    --max-sim-seconds "$MAX_SIM_SECONDS" --success-hold-seconds 5 \
    --record-fps "$RECORD_FPS" --video-mode head-only \
    --xy-jitter-mm 4 --yaw-jitter-deg 0.5 --guidance-scale "$scale" --guidance-steps "$GUIDANCE_STEPS" \
    --save-raw --record --save-policy-traces --compare-base --full-duration-evaluation "${extras[@]}"
done
