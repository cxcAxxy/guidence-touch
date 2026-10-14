#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/path/to/touchguidance/bin/python}"
base_root="${BASE_ROOT:-/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_v2_random10}"
output_dir="${OUTPUT_DIR:-/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_rules_v1_random10}"
workers="${WORKERS:-8}"
seed="${SEED:-42}"

mkdir -p "${output_dir}"
trap 'result=$?; printf "%s\n" "${result}" > "${output_dir}/exit_code"' EXIT
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Reuse all clean contact origins and geometry; generate a new augmented pool.
# The builder resumes completed episodes and then validates/publishes all three
# nested clean:augmented dataset entries (1:1, 1:5 and 1:10).
"${python_bin}" -u "${repo_root}/scripts/touchguide/prepare_contact_scales.py" \
    --output-dir "${output_dir}" --reuse-base-root "${base_root}" \
    --sampling-policy vector_components --workers "${workers}" --seed "${seed}"
