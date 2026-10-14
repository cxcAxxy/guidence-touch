#!/usr/bin/env bash
set -euo pipefail
# Full contact-origin augmentation only; no training. Re-run to resume.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/path/to/touchguidance/bin/python}"
ANCHOR_ROOT="${ANCHOR_ROOT:-/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${ANCHOR_ROOT}/full_v1_30steps}"
WORKERS="${WORKERS:-8}"
SHARD_SIZE="${SHARD_SIZE:-64}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"${PYTHON_BIN}" -u "${repo_root}/scripts/touchguide/prepare_fingertip_full.py" \
  --anchor-root "${ANCHOR_ROOT}" --output-dir "${OUTPUT_DIR}" \
  --workers "${WORKERS}" --shard-size "${SHARD_SIZE}" "$@"
