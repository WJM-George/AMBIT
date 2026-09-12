#!/usr/bin/env bash
set -euo pipefail
REPO="${P10_REPO:-/mnt/sdc/stable-audio-tools-workspace}"
RUN_ROOT="${RUN_ROOT:-/mnt/sdb/model_archives/transfusion_editing/mainline/sceneplan_transfusion_editing_dit_full_seed42_v1}"
SELECTION="${DIT_CHECKPOINT_SELECTION:-$RUN_ROOT/evaluation/validation_20k_checkpoint_selection/SELECTED.json}"
OUTPUT="${GT_AUDIO_OUTPUT:-$RUN_ROOT/evaluation/validation_1k_gt_audio}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES// /}" != "3,4,5,6,7" ]]; then
    echo "[editing-gt-audio] only physical GPUs 3,4,5,6,7 are allowed" >&2
    exit 2
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3,4,5,6,7
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
post_args=()
if [[ -n "${GT_AUDIO_JOINT_SELECTION:-}" || -n "${GT_AUDIO_PRE_JOINT_GATE:-}" ]]; then
    if [[ -z "${GT_AUDIO_JOINT_SELECTION:-}" || -z "${GT_AUDIO_PRE_JOINT_GATE:-}" ]]; then
        echo "[editing-gt-audio] post-joint evaluation requires its selection and base audio gate" >&2
        exit 2
    fi
    post_args=(--joint-selection "$GT_AUDIO_JOINT_SELECTION" --pre-joint-gate "$GT_AUDIO_PRE_JOINT_GATE")
fi
mkdir -p "$OUTPUT/logs"
"$REPO/.venv/bin/torchrun" --standalone --nproc_per_node=5 \
    "$REPO/scripts/t2a/eval/evaluate_sceneplan_transfusion_editing_gt_audio.py" \
    --selection "$SELECTION" --output-dir "$OUTPUT" "${post_args[@]}" \
    2>&1 | tee -a "$OUTPUT/logs/evaluate.log"
"$REPO/.venv/bin/python" \
    "$REPO/scripts/t2a/eval/evaluate_sceneplan_transfusion_editing_gt_audio.py" \
    --selection "$SELECTION" --output-dir "$OUTPUT" "${post_args[@]}" --verify-only
