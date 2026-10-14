#!/usr/bin/env bash
set -euo pipefail

# Step 1 only: index recorded contact anchors and geometry on NAS.
# Edit these settings, or supply environment variables/CLI overrides.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/path/to/touchguidance/bin/python}"
DATA_ROOT="${DATA_ROOT:-/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1}"
TACTILE_ROOT="${TACTILE_ROOT:-/path/to/datasets/sim/poker-draw/pi05_trex/0920_200}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_v1}"
STRIDE="${STRIDE:-4}"
WORKERS="${WORKERS:-8}"
CONTACT_THRESHOLD_N="${CONTACT_THRESHOLD_N:-auto}"
MINIMUM_NORMAL_COHERENCE="${MINIMUM_NORMAL_COHERENCE:-0.9}"
MAX_CLOCK_SKEW_S="${MAX_CLOCK_SKEW_S:-0.03333333333333333}"

export PYTHONDONTWRITEBYTECODE=1
"${PYTHON_BIN}" -u "${repo_root}/scripts/touchguide/prepare_contact_anchors.py" \
    --data-root "${DATA_ROOT}" --tactile-root "${TACTILE_ROOT}" \
    --output-dir "${OUTPUT_DIR}" --stride "${STRIDE}" --workers "${WORKERS}" \
    --contact-threshold-n "${CONTACT_THRESHOLD_N}" \
    --minimum-normal-coherence "${MINIMUM_NORMAL_COHERENCE}" \
    --max-clock-skew-s "${MAX_CLOCK_SKEW_S}" "$@"
