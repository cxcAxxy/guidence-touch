#!/usr/bin/env bash
set -euo pipefail
# Stage 2 only: small static FK/IK pilot; no training or physics rollout.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/path/to/touchguidance/bin/python}"
ANCHOR_ROOT="${ANCHOR_ROOT:-/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${ANCHOR_ROOT}/kinematics/pilot_v2_mixed}"
SIM_ROOT="${SIM_ROOT:-/path/to/kaitactilesim}"
ANCHORS_PER_PHASE_SPLIT="${ANCHORS_PER_PHASE_SPLIT:-4}"
COMBINATION_MODE="${COMBINATION_MODE:-all_subsets_mixed}"
MIXED_PATTERNS_PER_SUBSET="${MIXED_PATTERNS_PER_SUBSET:-6}"
SELECTED_ANCHORS="${SELECTED_ANCHORS:-${ANCHOR_ROOT}/kinematics/pilot_v1/selected_anchors.parquet}"
anchor_args=()
if [[ -f "${SELECTED_ANCHORS}" ]]; then
  anchor_args=(--selected-anchors "${SELECTED_ANCHORS}")
fi
MAGNITUDES_MM="${MAGNITUDES_MM:-0.5 1.0 2.0}"
MAX_JOINT_DELTA_RAD="${MAX_JOINT_DELTA_RAD:-0.15}"
POSITION_TOLERANCE_MM="${POSITION_TOLERANCE_MM:-0.05}"
read -r -a magnitudes <<< "${MAGNITUDES_MM}"
export PYTHONDONTWRITEBYTECODE=1
"${PYTHON_BIN}" -u "${repo_root}/scripts/touchguide/prepare_fingertip_perturbations.py" \
  --anchor-root "${ANCHOR_ROOT}" --output-dir "${OUTPUT_DIR}" \
  --sim-root "${SIM_ROOT}" --anchors-per-phase-split "${ANCHORS_PER_PHASE_SPLIT}" \
  --magnitudes-mm "${magnitudes[@]}" --max-joint-delta-rad "${MAX_JOINT_DELTA_RAD}" \
  --position-tolerance-mm "${POSITION_TOLERANCE_MM}" \
  --combination-mode "${COMBINATION_MODE}" --mixed-patterns-per-subset "${MIXED_PATTERNS_PER_SUBSET}" \
  "${anchor_args[@]}" "$@"
