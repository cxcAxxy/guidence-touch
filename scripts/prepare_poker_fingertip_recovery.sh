#!/usr/bin/env bash
set -euo pipefail
# Build and inspect the existing static pilot. No training or dynamic rollout.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/path/to/touchguidance/bin/python}"
ANCHOR_ROOT="${ANCHOR_ROOT:-/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_v1}"
KINEMATIC_ROOT="${KINEMATIC_ROOT:-${ANCHOR_ROOT}/kinematics/pilot_v2_mixed}"
OUTPUT_DIR="${OUTPUT_DIR:-${ANCHOR_ROOT}/recovery/pilot_v1_30steps}"
RELEASE_DISTANCE_MM="${RELEASE_DISTANCE_MM:-1.0}"
MAX_FORCE_GAIN="${MAX_FORCE_GAIN:-3.0}"
MAX_ROTATION_DEG="${MAX_ROTATION_DEG:-15.0}"
export PYTHONDONTWRITEBYTECODE=1
"${PYTHON_BIN}" -u "${repo_root}/scripts/touchguide/prepare_fingertip_recovery.py" \
  --kinematic-root "${KINEMATIC_ROOT}" --output-dir "${OUTPUT_DIR}" \
  --release-distance-mm "${RELEASE_DISTANCE_MM}" --max-force-gain "${MAX_FORCE_GAIN}" \
  --max-rotation-deg "${MAX_ROTATION_DEG}" "$@"
"${PYTHON_BIN}" -u "${repo_root}/scripts/touchguide/validate_fingertip_recovery.py" --data-root "${OUTPUT_DIR}"
"${PYTHON_BIN}" -u "${repo_root}/scripts/touchguide/analyze_fingertip_recovery.py" --data-root "${OUTPUT_DIR}"
