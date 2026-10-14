#!/usr/bin/env bash
set -euo pipefail

# Editable baseline settings. This script performs no training or simulator run.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/path/to/touchguidance/bin/python}"
jax_python="${JAX_PYTHON:-${python_bin}}"
jax_checkpoint="${JAX_CHECKPOINT:-/path/to/datasets/sim/poker-draw/pi05/0920_200/checkpoints/pi05_kaihand/poker_draw_bs128_noskip_25k/20000}"
torch_checkpoint="${TORCH_CHECKPOINT:-/path/to/datasets/sim/poker-draw/pi05_trex/pi05_pytorch_20k}"
verification_dir="${VERIFICATION_DIR:-/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/baseline_20k/verification}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${repo_root}/src:${repo_root}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "${repo_root}"
mkdir -p "${verification_dir}"

if [[ ! -f "${torch_checkpoint}/model.safetensors" ]]; then
    JAX_PLATFORMS=cpu "${python_bin}" -u examples/convert_jax_model_to_pytorch.py \
        --checkpoint-dir "${jax_checkpoint}" --config-name pi05_poker_trex \
        --output-path "${torch_checkpoint}" --precision bfloat16 \
        2>&1 | tee "${verification_dir}/conversion.log"
fi
"${python_bin}" - "${torch_checkpoint}" "${jax_checkpoint}" <<'PY'
import json
import sys
from pathlib import Path
converted, source = map(Path, sys.argv[1:])
cfg = json.loads((converted / "config.json").read_text())
assert Path(cfg["source_checkpoint"]).resolve() == source.resolve(), "Source checkpoint mismatch"
assert cfg["pi05"] and cfg["action_dim"] == 32 and cfg["action_horizon"] == 30
assert cfg["max_token_len"] == 200 and cfg["discrete_state_input"]
assert cfg["precision"] == "bfloat16", "Expected BF16 baseline"
assert not cfg["unexpected_keys"]
PY
if [[ ! -f "${verification_dir}/fixed_inputs.npz" ]]; then
    JAX_PLATFORMS=cpu "${python_bin}" -u scripts/touchguide/prepare_fixed_inputs.py \
        --output-dir "${verification_dir}" --checkpoint "${jax_checkpoint}" \
        2>&1 | tee "${verification_dir}/prepare_inputs.log"
fi
JAX_PLATFORMS=cuda "${jax_python}" -u scripts/touchguide/verify_jax_baseline.py \
    --inputs "${verification_dir}/fixed_inputs.npz" --checkpoint20k "${jax_checkpoint}" \
    --output "${verification_dir}/jax_outputs.npz" \
    2>&1 | tee "${verification_dir}/jax.log"
JAX_PLATFORMS=cpu "${python_bin}" -u scripts/touchguide/verify_torch_baseline.py \
    --inputs "${verification_dir}/fixed_inputs.npz" --checkpoint "${torch_checkpoint}" \
    --output "${verification_dir}/torch_outputs.npz" \
    2>&1 | tee "${verification_dir}/torch.log"
"${python_bin}" -u scripts/touchguide/compare_baselines.py \
    --directory "${verification_dir}" --jax-checkpoint "${jax_checkpoint}" \
    --torch-checkpoint "${torch_checkpoint}" \
    2>&1 | tee "${verification_dir}/comparison.log"
