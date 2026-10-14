#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Optional shell configuration, e.g. configs/oa_guidance.env (copy the example).
if (( $# > 1 )); then echo "Usage: bash $0 [configuration.env]" >&2; exit 2; fi
if (( $# == 1 )); then set -a; source "$1"; set +a; fi
export EXECUTE_STEPS="${EXECUTE_STEPS:-5}" # Infer 30 actions, execute a prefix, then refresh all observations.
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
export GUIDANCE_STEPS="${GUIDANCE_STEPS:-3}" # Final three of ten FLOW steps; independent of execution length.
export NUM_TRIALS="${NUM_TRIALS:-10}"
export VIDEO_MODE="${VIDEO_MODE:-head-only}"
export RUN_NAME="${RUN_NAME:-pi05_vs_oa_guidance_$(date +%Y%m%d_%H%M%S)_$$}"
export DATA_ROOT="${OA_DATA_ROOT:-/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_rules_v1_random10/ratio_1_1}"
export CPM_CHECKPOINT="${OA_CHECKPOINT:-}"
if [[ -z "$CPM_CHECKPOINT" || ! -f "$CPM_CHECKPOINT" ]]; then
    echo "Set OA_CHECKPOINT to a trained recovery-state-cpm best.pt or last.pt; weights are external to this repository." >&2
    exit 2
fi
if [[ -d "$repo_root/simulation/src" && -z "${SIM_ROOT:-}" ]]; then export SIM_ROOT="$repo_root/simulation"; fi
if [[ -z "${MODEL_PYTHON:-}" && ! -x /path/to/touchguidance/bin/python ]]; then export MODEL_PYTHON="$(command -v python)"; fi
export REFERENCE_DATASET="${REFERENCE_DATASET:-}" # Optional replay comparison; deployment does not need source videos.
"${MODEL_PYTHON:-/path/to/touchguidance/bin/python}" - "$DATA_ROOT" <<'CHECK'
import json, sys
from pathlib import Path
settings=json.loads((Path(sys.argv[1])/"config/cpm.json").read_text())
if settings.get("recovery",{}).get("model_variant") != "recovery-state-cpm":
    raise SystemExit("OA_DATA_ROOT must select a prepared recovery-state-cpm entry or exported inference metadata bundle")
CHECK
exec bash "$repo_root/scripts/run_poker_touchguide_comparison.sh"
