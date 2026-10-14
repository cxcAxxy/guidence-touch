#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_ROOT="${SOURCE_ROOT:?Set SOURCE_ROOT to the prepared original observation-action dataset}"
TACTILE_ROOT="${TACTILE_ROOT:?Set TACTILE_ROOT to the synchronized tactile sidecar root}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to the new augmentation directory outside this repository}"
MODEL_PATH="${MODEL_PATH:-${SIM_ROOT:-$repo_root/simulation}/src/kaihand_tactile_env/tasks/poker_draw/scene.xml}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-42}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON_BIN" -u "$repo_root/scripts/touchguide/prepare_contact_scales.py"   --source-root "$SOURCE_ROOT" --tactile-root "$TACTILE_ROOT" --model-path "$MODEL_PATH"   --output-dir "$OUTPUT_DIR" --sampling-policy vector_components --workers "$WORKERS" --seed "$SEED" "$@"
# Preparation only: fit scorer normalization and write training configurations; no training is started.
"$PYTHON_BIN" "$repo_root/scripts/touchguide/prepare_recovery_cpm.py" --data-root "$OUTPUT_DIR" --base-entry "$SOURCE_ROOT"
