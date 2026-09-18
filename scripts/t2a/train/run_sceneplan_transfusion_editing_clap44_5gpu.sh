#!/usr/bin/env bash
set -euo pipefail
REPO="${P10_REPO:-.}"
ROOT="${EDITING_DATA_ROOT:-${AMBIT_DATA_ROOT}/sceneplan_transfusion_editing_v1}"
OUTPUT="${CLAP44_RUN_ROOT:-${AMBIT_CKPT_ROOT}/transfusion_editing/mainline/editing_clap44_v1_seed42}"
GPUS="${EDITING_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "$GPUS" ]]; then
    echo '[clap44] set EDITING_GPUS to the physical GPUs assigned to this job' >&2
    exit 2
fi
if [[ -z "${EDITING_GPU_LEASES:-}" ]]; then
    exec "$REPO/.venv/bin/python" "$REPO/scripts/t2a/train/editing_gpu_runtime.py" \
        --gpus "$GPUS" -- bash "$0" "$@"
fi
IFS=',' read -r -a GPU_IDS <<< "$EDITING_GPUS"
NPROC="${#GPU_IDS[@]}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
ulimit -Sn 65536
cd "$REPO"
resume_args=()
if [[ -r "$OUTPUT/TRAIN_CONTRACT.json" ]]; then
    RESUME_CHECKPOINT="$("$REPO/.venv/bin/python" - "$OUTPUT" <<'PY'
import sys
from stable_audio_tools.training.sceneplan_transfusion_editing_clap44_pretrain import resolve_training_resume
record=resolve_training_resume(sys.argv[1])
print('' if record is None else record['checkpoint'])
PY
)"
    if [[ -n "$RESUME_CHECKPOINT" ]]; then
        resume_args=(--resume "$RESUME_CHECKPOINT")
    fi
fi
exec "$REPO/.venv/bin/torchrun" --standalone --nproc_per_node="$NPROC" \
    "$REPO/scripts/t2a/train/train_sceneplan_transfusion_editing_clap44.py" \
    --index "$ROOT/training_index/train.sqlite" \
    --preflight "$ROOT/contracts/full_training/PREFLIGHT.json" \
    --config "${CLAP44_CONFIG:-$REPO/stable_audio_tools/configs/model_configs/txt2audio/t2a/editing_clap44_v1.json}" \
    --output "$OUTPUT" "${resume_args[@]}" "$@"
