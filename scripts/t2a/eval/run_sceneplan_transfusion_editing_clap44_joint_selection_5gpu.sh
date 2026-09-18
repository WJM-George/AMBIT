#!/usr/bin/env bash
set -euo pipefail
REPO="${P10_REPO:-.}"
ROOT="${EDITING_DATA_ROOT:-${AMBIT_DATA_ROOT}/sceneplan_transfusion_editing_v1}"
RUN_DIR="${CLAP44_AR_RUN_DIR:?set the completed native CLAP44 full AR run directory}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo '[clap44-selection] set CUDA_VISIBLE_DEVICES to the GPUs for this job' >&2
    exit 2
fi
mkdir -p "$ROOT/materialized/locks"
exec 7>"$ROOT/materialized/locks/training-chain.lock"
flock -n 7 || { echo '[clap44-selection] another Editing chain is active; keep it running' >&2; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
cd "$REPO"
exec "$REPO/.venv/bin/torchrun" --standalone --nproc_per_node=5 \
    "$REPO/scripts/t2a/eval/select_sceneplan_transfusion_editing_clap44_joint.py" \
    --run-dir "$RUN_DIR" --preflight "$ROOT/contracts/full_training/PREFLIGHT.json" "$@"
