#!/usr/bin/env bash
set -euo pipefail

# state-CPM: current right-arm/hand state -> 27/512/1024 MLP -> one token.
# Reuse the original clean dataset and frozen DINO. Train new heads/fusion from
# the same seed as the baseline; do not load pi0.5 or the trained baseline CPM.
# Each run gets a new NAS directory and its own W&B run tagged state-CPM.
export PYTHON_BIN="${PYTHON_BIN:-/path/to/touchguidance/bin/python}"
export DATA_ROOT="${DATA_ROOT:-/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/datasets/touchguidence}"
export NUM_GPUS="${NUM_GPUS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-128}"  # global batch; 16 examples per GPU with 8 GPUs
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-}"  # follow the final training batch option
export EPOCHS="${EPOCHS:-400}"  # match the completed original CPM run
export SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-20}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
export WANDB_ENTITY="${WANDB_ENTITY:-xianchichen8-hunan-university-library}"
export WANDB_PROJECT="${WANDB_PROJECT:-poker-draw-touchguide-cpm}"
export WANDB_MODE="${WANDB_MODE:-online}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/train_poker_touchguide_cpm.sh" "$@" --model-variant state-cpm
