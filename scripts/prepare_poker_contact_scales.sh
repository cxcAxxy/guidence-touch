#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/path/to/touchguidance/bin/python}"
output_dir="${OUTPUT_DIR:-/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_v3_independent10}"
# Set BASE_ROOT="" for an isolated preview that selects fewer source origins.
base_root="${BASE_ROOT-/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_v2_random10}"
base_args=()
if [[ -n "${base_root}" ]]; then
    base_args=(--reuse-base-root "${base_root}")
fi
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"${python_bin}" -u "${repo_root}/scripts/touchguide/prepare_contact_scales.py" \
    --output-dir "${output_dir}" --workers "${WORKERS:-8}" \
    --sampling-policy independent "${base_args[@]}" "$@"
