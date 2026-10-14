#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/path/to/touchguidance/bin/python}"
data_root="${DATA_ROOT:-/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_v2_random10}"
output_dir="${OUTPUT_DIR:-${data_root}/visualization}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"${python_bin}" -u "${repo_root}/scripts/touchguide/visualize_contact_scales.py" \
    --data-root "${data_root}" --output-dir "${output_dir}" \
    --workers "${WORKERS:-4}" --anchors-per-episode "${ANCHORS_PER_EPISODE:-3}" "$@"
