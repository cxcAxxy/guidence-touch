#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_python="${MODEL_PYTHON:-/path/to/touchguidance/bin/python}"
sim_python="${SIM_PYTHON:-/path/to/simulator-env/bin/python}"
sim_root="${SIM_ROOT:-/path/to/kaitactilesim}"
base_checkpoint="${BASE_CHECKPOINT:-/path/to/datasets/sim/poker-draw/pi05_trex/pi05_pytorch_20k}"
cpm_checkpoint="${CPM_CHECKPOINT:-/path/to/datasets/touchguidence/cpm_dino_b128_g8_20260928_173559/best.pt}"
data_root="${DATA_ROOT:-/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1}"
tokenizer="${TOKENIZER_MODEL:-$HOME/.cache/openpi/big_vision/paligemma_tokenizer.model}"
output_root="${OUTPUT_ROOT:-${repo_root}/tac_forward_guidance/artifacts/eval}"
run_name="${RUN_NAME:-pi05_vs_tac_forward_guidance_$(date +%Y%m%d_%H%M%S)_$$}"
extras=()
for key in SAVE_RAW RECORD SAVE_POLICY_TRACES COMPARE_BASE FULL_DURATION_EVALUATION; do
  # Preserve the previous diagnostic protocol unless explicitly changed.
  value="${!key:-1}"
  [[ "$value" == 0 || "$value" == 1 ]] || { echo "$key must be 0 or 1" >&2; exit 2; }
  flag="${key,,}"
  flag="${flag//_/-}"
  if [[ "$value" == 1 ]]; then extras+=("--$flag"); else extras+=("--no-$flag"); fi
done
for key in DRY_RUN RESUME; do
  value="${!key:-0}"
  [[ "$value" == 0 || "$value" == 1 ]] || { echo "$key must be 0 or 1" >&2; exit 2; }
  flag="${key,,}"
  flag="${flag//_/-}"
  if [[ "$value" == 1 ]]; then extras+=("--$flag"); fi
done

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${repo_root}:${repo_root}/src:${repo_root}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$repo_root"

"$model_python" -u -m tac_forward_guidance.evaluate \
  --sim-root "$sim_root" --model-python "$model_python" --sim-python "$sim_python" \
  --base-checkpoint "$base_checkpoint" --cpm-checkpoint "$cpm_checkpoint" \
  --data-root "$data_root" --tokenizer-model "$tokenizer" \
  --reference-dataset "${REFERENCE_DATASET:-/path/to/datasets/sim/poker-draw/pi05/0920_200}" \
  --output-root "$output_root" --run-name "$run_name" \
  --variants pi05 pi05+tac_forward_guidance --num-trials "${NUM_TRIALS:-3}" --seed-start "${SEED_START:-0}" \
  --gpu-id "${GPU_ID:-0}" --port "${SERVER_PORT:-18787}" \
  --guidance-scale "${GUIDANCE_SCALE:-0.3}" --guidance-steps "${GUIDANCE_STEPS:-3}" \
  --gradient-mode "${GRADIENT_MODE:-full}" --max-update-rms "${MAX_UPDATE_RMS:-0.01}" \
  --num-steps "${NUM_STEPS:-10}" --precision "${PRECISION:-bfloat16}" \
  --max-sim-seconds "${MAX_SIM_SECONDS:-50}" --success-hold-seconds "${SUCCESS_HOLD_SECONDS:-5}" \
  --xy-jitter-mm "${XY_JITTER_MM:-4}" --yaw-jitter-deg "${YAW_JITTER_DEG:-0.5}" \
  --record-fps "${RECORD_FPS:-10}" --video-mode "${VIDEO_MODE:-all}" \
  --server-wait-seconds "${SERVER_WAIT_SECONDS:-600}" --trial-wall-limit "${TRIAL_WALL_LIMIT:-3600}" \
  "${extras[@]}"
