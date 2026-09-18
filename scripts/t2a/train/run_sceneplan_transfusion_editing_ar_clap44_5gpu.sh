#!/usr/bin/env bash
set -euo pipefail
REPO="${P10_REPO:-.}"
ROOT="${EDITING_DATA_ROOT:-${AMBIT_DATA_ROOT}/sceneplan_transfusion_editing_v1}"
PHASE="${CLAP44_AR_PHASE:-pilot}"
VARIANT="${CLAP44_AR_VARIANT:-global_and_sequence}"
MODE="${EDITING_AR_MODE:-joint}"
GPUS="${EDITING_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "$GPUS" ]]; then
    echo '[clap44-ar] set EDITING_GPUS to the physical GPUs assigned to this job' >&2
    exit 2
fi
if [[ -z "${EDITING_GPU_LEASES:-}" ]]; then
    exec "$REPO/.venv/bin/python" "$REPO/scripts/t2a/train/editing_gpu_runtime.py" \
        --gpus "$GPUS" -- bash "$0" "$@"
fi
IFS=',' read -r -a GPU_IDS <<< "$EDITING_GPUS"
NPROC="${#GPU_IDS[@]}"
case "$MODE" in joint|ar_pretrain) ;; *) echo '[clap44-ar] invalid EDITING_AR_MODE' >&2; exit 2 ;; esac
case "$PHASE" in pilot|full) ;; *) echo '[clap44-ar] phase must be pilot or full' >&2; exit 2 ;; esac
case "$VARIANT" in latent_only|global_only|sequence_only|global_and_sequence) ;; *) echo '[clap44-ar] unknown feature variant' >&2; exit 2 ;; esac
RUN_DIR="${CLAP44_AR_RUN_DIR:-${AMBIT_CKPT_ROOT}/transfusion_editing/mainline/editing_ar_clap44_${MODE}_${VARIANT}_${PHASE}_seed42_v1}"
BASE_RUN="${BASE_DIT_RUN:-${AMBIT_CKPT_ROOT}/transfusion_editing/mainline/sceneplan_transfusion_editing_dit_full_seed42_v1}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
ulimit -Sn 65536
cd "$REPO"
mode_args=(--training-mode "$MODE")
if [[ "$MODE" == ar_pretrain ]]; then
    mode_args+=(--p10-checkpoint "${EDITING_P10_CHECKPOINT:?set the P10-v11 initialization checkpoint}")
    CONFIG="${CLAP44_AR_CONFIG:-$REPO/stable_audio_tools/configs/model_configs/txt2audio/t2a/editing_ar_clap44_pretrain_v1.json}"
else
    mode_args+=(--dit-selection "$BASE_RUN/evaluation/validation_20k_checkpoint_selection/SELECTED.json"
        --dit-gt-audio-gate "$BASE_RUN/evaluation/validation_1k_gt_audio/GATE.json")
    CONFIG="${CLAP44_AR_CONFIG:-$REPO/stable_audio_tools/configs/model_configs/txt2audio/t2a/editing_ar_clap44_joint_${PHASE}_v1.json}"
fi
if [[ "$VARIANT" != latent_only || -n "${CLAP44_CHECKPOINT:-}${CLAP44_VALIDATION_REPORT:-}" ]]; then
    mode_args+=(--clap-checkpoint "${CLAP44_CHECKPOINT:?set the trained encoder checkpoint}"
        --clap-validation-report "${CLAP44_VALIDATION_REPORT:?set its full validation report}")
fi
resume_args=()
if [[ -r "$RUN_DIR/RUN_CONTRACT.json" ]]; then
    RESUME_CHECKPOINT="$("$REPO/.venv/bin/python" - "$RUN_DIR" <<'PY'
import sys
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_joint_io import resolve_joint_resume
record=resolve_joint_resume(sys.argv[1])
print('' if record is None else record['checkpoint'])
PY
)"
    if [[ -n "$RESUME_CHECKPOINT" ]]; then
        resume_args=(--resume "$RESUME_CHECKPOINT")
    fi
fi
exec "$REPO/.venv/bin/torchrun" --standalone --nproc_per_node="$NPROC" \
    "$REPO/scripts/t2a/train/train_sceneplan_transfusion_editing_ar_clap44.py" \
    --run-dir "$RUN_DIR" \
    --config "$CONFIG" \
    --variant "$VARIANT" \
    --preflight "$ROOT/contracts/full_training/PREFLIGHT.json" \
    "${mode_args[@]}" "${resume_args[@]}" "$@"
