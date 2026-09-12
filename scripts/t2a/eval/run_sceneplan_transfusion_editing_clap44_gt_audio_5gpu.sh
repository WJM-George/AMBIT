#!/usr/bin/env bash
set -euo pipefail
REPO="${P10_REPO:-/mnt/sdc/stable-audio-tools-workspace}"
ROOT="${EDITING_DATA_ROOT:-/mnt/sdb/audio_dataset/sceneplan_transfusion_editing_v1}"
RUN_DIR="${CLAP44_AR_RUN_DIR:?set the completed native CLAP44 full AR run directory}"
SELECTION="$RUN_DIR/evaluation/validation_20k_clap44_joint_selection/SELECTED.json"
SELECTION_SHA="${CLAP44_JOINT_SELECTION_SHA256:?set the pinned native joint selection SHA256}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES// /}" != "3,4,5,6,7" ]]; then
    echo '[clap44-gt-audio] only physical Editing GPUs 3-7 are allowed' >&2
    exit 2
fi
mkdir -p "$ROOT/materialized/locks"
exec 7>"$ROOT/materialized/locks/training-chain.lock"
flock -n 7 || { echo '[clap44-gt-audio] another Editing chain is active; keep it running' >&2; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3,4,5,6,7
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
cd "$REPO"
exec "$REPO/.venv/bin/torchrun" --standalone --nproc_per_node=5 \
    "$REPO/scripts/t2a/eval/evaluate_sceneplan_transfusion_editing_clap44_gt_audio.py" \
    --joint-selection "$SELECTION" --joint-selection-sha256 "$SELECTION_SHA" \
    --output-dir "$RUN_DIR/evaluation/validation_1k_clap44_post_joint_gt_audio" "$@"
