#!/usr/bin/env bash
set -euo pipefail

# Edit these settings, then run this script. Each run creates a new directory.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIM_ROOT="${SIM_ROOT:-/path/to/kaitactilesim}"
MODEL_PYTHON="${MODEL_PYTHON:-/path/to/touchguidance/bin/python}"
SIM_PYTHON="${SIM_PYTHON:-/path/to/simulator-env/bin/python}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/path/to/datasets/sim/poker-draw/pi05_trex/pi05_pytorch_20k}"
CPM_CHECKPOINT="${CPM_CHECKPOINT:-/path/to/datasets/touchguidence/cpm_dino_b128_g8_20260928_173559/best.pt}"
DATA_ROOT="${DATA_ROOT:-/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-$HOME/.cache/openpi/big_vision/paligemma_tokenizer.model}"
REFERENCE_DATASET="${REFERENCE_DATASET-/path/to/datasets/sim/poker-draw/pi05/0920_200}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/datasets/touchguidence/eval}"
RUN_NAME="${RUN_NAME:-pi05_vs_touchguidance_$(date +%Y%m%d_%H%M%S)_$$}"

VARIANTS=(pi05 pi05+touchguidance)   # Use (pi05+touchguidance) for a guided-only run.
NUM_TRIALS="${NUM_TRIALS:-10}"       # Per variant; both variants use the same seeds.
SEED_START="${SEED_START:-0}"
GPU_ID="${GPU_ID:-0}"
SERVER_PORT="${SERVER_PORT:-18786}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-600}"
TRIAL_WALL_LIMIT="${TRIAL_WALL_LIMIT:-3600}"
MAX_SIM_SECONDS="${MAX_SIM_SECONDS:-50}"
SUCCESS_HOLD_SECONDS="${SUCCESS_HOLD_SECONDS:-5}"
XY_JITTER_MM="${XY_JITTER_MM:-4.0}"
YAW_JITTER_DEG="${YAW_JITTER_DEG:-0.5}"
RECORD_FPS="${RECORD_FPS:-10}"
VIDEO_MODE="${VIDEO_MODE:-all}" # all, head-only, or overhead-only; does not change policy observations.
EXECUTE_STEPS="${EXECUTE_STEPS:-30}" # Historical comparison default; new OA launcher sets 5.
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
GUIDANCE_STEPS="${GUIDANCE_STEPS:-3}" # Last 3 of 10 denoising steps: t=0.3/0.2/0.1.
TACTILE_DISPLAY_SCALE="${TACTILE_DISPLAY_SCALE:-0.05}" # N per taxel; display only.
SAVE_RAW="${SAVE_RAW:-1}"
RECORD="${RECORD:-1}"
SAVE_POLICY_TRACES="${SAVE_POLICY_TRACES:-1}"
COMPARE_BASE="${COMPARE_BASE:-1}"   # Extra base-only denoising with identical inputs and noise.
FULL_DURATION_EVALUATION="${FULL_DURATION_EVALUATION:-1}" # Same diagnostic mode as previous rollouts.
DRY_RUN="${DRY_RUN:-0}"             # 1 prints commands, does not start models or simulation.

extras=()
if [[ -n "$REFERENCE_DATASET" ]]; then extras+=(--reference-dataset "$REFERENCE_DATASET"); fi
for key in SAVE_RAW RECORD SAVE_POLICY_TRACES COMPARE_BASE FULL_DURATION_EVALUATION; do
  value="${!key}"
  [[ "$value" == 0 || "$value" == 1 ]] || { echo "$key must be 0 or 1" >&2; exit 2; }
  flag="${key,,}"
  flag="${flag//_/-}"
  if [[ "$value" == 1 ]]; then extras+=("--$flag"); else extras+=("--no-$flag"); fi
done
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
if [[ "$DRY_RUN" == 1 ]]; then extras+=(--dry-run); fi

export PYTHONDONTWRITEBYTECODE=1
cd "$repo_root"
"$MODEL_PYTHON" -u scripts/touchguide/evaluate_poker_touchguide.py \
  --sim-root "$SIM_ROOT" --model-python "$MODEL_PYTHON" --sim-python "$SIM_PYTHON" \
  --base-checkpoint "$BASE_CHECKPOINT" --cpm-checkpoint "$CPM_CHECKPOINT" \
  --data-root "$DATA_ROOT" --tokenizer-model "$TOKENIZER_MODEL" \
  --output-root "$OUTPUT_ROOT" --run-name "$RUN_NAME" \
  --variants "${VARIANTS[@]}" --num-trials "$NUM_TRIALS" --seed-start "$SEED_START" \
  --gpu-id "$GPU_ID" --port "$SERVER_PORT" --server-wait-seconds "$SERVER_WAIT_SECONDS" \
  --trial-wall-limit "$TRIAL_WALL_LIMIT" --max-sim-seconds "$MAX_SIM_SECONDS" \
  --success-hold-seconds "$SUCCESS_HOLD_SECONDS" --record-fps "$RECORD_FPS" --video-mode "$VIDEO_MODE" \
  --xy-jitter-mm "$XY_JITTER_MM" --yaw-jitter-deg "$YAW_JITTER_DEG" \
  --execute-steps "$EXECUTE_STEPS" --guidance-scale "$GUIDANCE_SCALE" --guidance-steps "$GUIDANCE_STEPS" \
  --tactile-display-scale "$TACTILE_DISPLAY_SCALE" "${extras[@]}"
