#!/usr/bin/env bash
set -euo pipefail
REPO="${P10_REPO:-.}"
ROOT="${EDITING_DATA_ROOT:-${AMBIT_DATA_ROOT}/sceneplan_transfusion_editing_v1}"
RUN_DIR="${CLAP44_AR_RUN_DIR:?set the completed native CLAP44 full AR run directory}"
SELECTION="$RUN_DIR/evaluation/validation_20k_clap44_joint_selection/SELECTED.json"
SELECTION_SHA="${CLAP44_JOINT_SELECTION_SHA256:?set the pinned native joint selection SHA256}"
PHASE="${CLAP44_AUDIO_PHASE:-calibration}"
case "$PHASE" in calibration|test) ;; *) echo '[clap44-audio] unknown phase' >&2; exit 2 ;; esac
calibration_args=()
if [[ "$PHASE" == test ]]; then
    CALIBRATION_SHA="${CLAP44_AUDIO_CALIBRATION_SHA256:?set the pinned passed native calibration RESULT.json SHA256}"
    calibration_args=(--calibration "$RUN_DIR/evaluation/clap44_audio_calibration/RESULT.json" --calibration-sha256 "$CALIBRATION_SHA")
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo '[clap44-audio] set CUDA_VISIBLE_DEVICES to the GPUs for this job' >&2
    exit 2
fi
mkdir -p "$ROOT/materialized/locks"
exec 7>"$ROOT/materialized/locks/training-chain.lock"
flock -n 7 || { echo '[clap44-audio] another Editing chain is active; keep it running' >&2; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
cd "$REPO"
exec "$REPO/.venv/bin/torchrun" --standalone --nproc_per_node=5 \
    "$REPO/scripts/t2a/eval/evaluate_sceneplan_transfusion_editing_clap44_audio.py" \
    --phase "$PHASE" --joint-selection "$SELECTION" --joint-selection-sha256 "$SELECTION_SHA" \
    --post-joint-gt-gate "$RUN_DIR/evaluation/validation_1k_clap44_post_joint_gt_audio/GATE.json" \
    --preflight "$ROOT/contracts/full_training/PREFLIGHT.json" \
    --output-dir "$RUN_DIR/evaluation/clap44_audio_${PHASE}" \
    --batch-size "${CLAP44_AUDIO_BATCH_SIZE:-2}" "${calibration_args[@]}" "$@"
