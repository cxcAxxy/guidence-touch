#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/path/to/touchguidance/bin/python}"
base_root="${BASE_ROOT:-/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_v3_independent10_preview}"
output_dir="${OUTPUT_DIR:-/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_rules_v1_preview}"
workers="${WORKERS:-2}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

"${python_bin}" -u "${repo_root}/scripts/touchguide/prepare_contact_scales.py" \
    --output-dir "${output_dir}" --reuse-base-root "${base_root}" \
    --sampling-policy vector_components --workers "${workers}" \
    --episode-ids 0 7 --max-origins-per-episode 8 --skip-ratio-views

"${python_bin}" -u "${repo_root}/scripts/touchguide/visualize_contact_scales.py" \
    --data-root "${output_dir}" --workers "${workers}" \
    --episode-ids 0 7 --anchors-per-episode 8

"${python_bin}" -u "${repo_root}/scripts/touchguide/review_finger_randomization.py" \
    --data-root "${output_dir}"
