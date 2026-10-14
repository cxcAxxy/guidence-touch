#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/path/to/touchguidance/bin/python}"
data_root="${DATA_ROOT:-/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_rules_v1_preview}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"${python_bin}" -u "${repo_root}/scripts/touchguide/visualize_recovery_pair.py" \
  --data-root "${data_root}" --episode "${EPISODE:-7}" --row "${CLEAN_ROW:-3}" \
  --variant "${VARIANT:-9}" --variants "${VARIANTS:-2,4,9}" "$@"
