#!/usr/bin/env bash
set -euo pipefail

REPO="${P11_REPO:-/mnt/sdc/stable-audio-tools-workspace}"
PY="${P11_PYTHON:-$REPO/.venv/bin/python}"
RUN_NAME="${RUN_NAME:?set a unique RUN_NAME for the O(1) resume smoke}"
RUN_ROOT="${RUN_ROOT:-/mnt/sdc/ckpts/sceneplan_p11/$RUN_NAME}"
LAUNCHER="$REPO/scripts/t2a/train/run_sceneplan_p11.sh"
VALIDATOR="$REPO/scripts/t2a/test/validate_sceneplan_p11_o1_resume.py"
CURRICULUM="/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/p11_v4_curriculum/p11_train_269568_ddp8_rank_balanced_batch8_seed42_v2.sqlite"

case "$RUN_ROOT" in
    /mnt/sdc/ckpts/sceneplan_p11/*) ;;
    *)
        echo "resume-smoke RUN_ROOT must stay under /mnt/sdc/ckpts/sceneplan_p11" >&2
        exit 2
        ;;
esac
for required in "$PY" "$LAUNCHER" "$VALIDATOR" "$CURRICULUM"; do
    if [[ ! -r "$required" ]]; then
        echo "resume-smoke prerequisite is unreadable: $required" >&2
        exit 1
    fi
done
if [[ -e "$RUN_ROOT" ]]; then
    echo "refusing existing resume-smoke run root: $RUN_ROOT" >&2
    exit 1
fi
if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)" != "8" ]]; then
    echo "O(1) resume smoke requires exactly eight host GPUs" >&2
    exit 2
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | awk \
    'BEGIN { found=0 } /^[[:space:]]*[0-9]+[[:space:]]*$/ { found=1 } END { exit found ? 0 : 1 }'; then
    echo "refusing to share GPUs during the O(1) resume smoke" >&2
    exit 2
fi

cd "$REPO"
echo "[p11-resume] stage 1: fresh full-state run to optimizer step 8"
RUN_NAME="$RUN_NAME" RUN_ROOT="$RUN_ROOT" MAX_STEPS=8 \
    "$LAUNCHER" resume_smoke

STAGE1="$RUN_ROOT/checkpoints/epoch=0-step=8.ckpt"
if [[ ! -r "$STAGE1" || ! -r "$RUN_ROOT/checkpoints/last.ckpt" ]]; then
    echo "fresh resume-smoke stage did not publish its step-8 checkpoint" >&2
    exit 1
fi
# ModelCheckpoint(save_top_k=1) legitimately removes the step-8 numbered file
# after publishing step 16. Preserve the exact source inode under a name that
# Lightning does not manage, avoiding a second multi-GB checkpoint copy.
STAGE1_EVIDENCE="$RUN_ROOT/checkpoints/resume-source-step8.ckpt"
if ! ln -- "$STAGE1" "$STAGE1_EVIDENCE"; then
    echo "could not preserve the step-8 resume source as a hard link" >&2
    exit 1
fi
if [[ "$(stat -c '%d:%i' "$STAGE1")" != "$(stat -c '%d:%i' "$STAGE1_EVIDENCE")" ]]; then
    echo "step-8 resume evidence is not an exact hard link" >&2
    exit 1
fi

echo "[p11-resume] stage 2: restore optimizer/EMA/RNG/loader state and advance to step 16"
RUN_NAME="$RUN_NAME" RUN_ROOT="$RUN_ROOT" MAX_STEPS=16 \
    CKPT_PATH="$RUN_ROOT/checkpoints/last.ckpt" \
    "$LAUNCHER" resume_smoke

STAGE2="$RUN_ROOT/checkpoints/epoch=0-step=16.ckpt"
if [[ ! -r "$STAGE2" ]]; then
    echo "restored resume-smoke stage did not publish its step-16 checkpoint" >&2
    exit 1
fi

"$PY" "$VALIDATOR" \
    --curriculum "$CURRICULUM" \
    --stage1-checkpoint "$STAGE1_EVIDENCE" \
    --stage2-checkpoint "$STAGE2" \
    --log "$RUN_ROOT/train.log" \
    --output "$RUN_ROOT/o1_resume_report.json"

echo "P11_O1_RESUME_SMOKE_COMPLETE=$RUN_ROOT/o1_resume_report.json"
