#!/usr/bin/env bash
set -euo pipefail

TRAINING_LOCK="${TRAINING_LOCK:?set TRAINING_LOCK to the launch-wide flock file}"
TRAIN_LOG="${TRAIN_LOG:?set TRAIN_LOG to the training log}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT to the training directory}"
STEP0_CKPT="${STEP0_CKPT:?set STEP0_CKPT to the initialization checkpoint}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to the evaluation directory}"
POLL_SECONDS="${POLL_SECONDS:-30}"
TRAIN_START_TIMEOUT_SECONDS="${TRAIN_START_TIMEOUT_SECONDS:-300}"
FINAL_STEP="${FINAL_STEP:-300}"
SWEEP="${SWEEP:-/home/tanhe/dataset_storage/stable-audio-tools/scripts/t2a/eval/run_spatial_cot_sequential_family_sweep.sh}"

if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[post-train-sweep] POLL_SECONDS must be a positive integer" >&2
    exit 2
fi
if [[ ! "$TRAIN_START_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[post-train-sweep] TRAIN_START_TIMEOUT_SECONDS must be a positive integer" >&2
    exit 2
fi
if [[ ! "$FINAL_STEP" =~ ^[1-9][0-9]*$ ]]; then
    echo "[post-train-sweep] FINAL_STEP must be a positive integer" >&2
    exit 2
fi
if [[ ! -x "$SWEEP" ]]; then
    echo "[post-train-sweep] sweep is not executable: $SWEEP" >&2
    exit 2
fi

resolve_checkpoint() {
    local step="$1"
    local matches=()
    mapfile -t matches < <(
        find "$RUN_ROOT/checkpoints" -maxdepth 1 -type f \
            -name "*-step=${step}.ckpt" -print | sort
    )
    (( ${#matches[@]} == 1 )) || return 1
    printf '%s\n' "${matches[0]}"
}

# The watcher is normally launched immediately after the trainer.  The trainer
# may need a few milliseconds to acquire its launch-wide flock, so treating an
# initially free lock as "training finished" is a race.  First observe either
# the held lock or an already-complete checkpoint/gate pair, then wait for the
# observed training process to release the lock.
lock_observed=0
start_waited=0
while (( start_waited < TRAIN_START_TIMEOUT_SECONDS )); do
    if final_checkpoint="$(resolve_checkpoint "$FINAL_STEP" 2>/dev/null)" \
        && [[ -r "$final_checkpoint" ]] \
        && grep -Eq 'SAT_TRAINING_GATE_RESULT=.*"status": "PASS"' "$TRAIN_LOG" 2>/dev/null; then
        break
    fi
    if ! flock -n "$TRAINING_LOCK" -c true; then
        lock_observed=1
        break
    fi
    sleep "$POLL_SECONDS"
    start_waited=$(( start_waited + POLL_SECONDS ))
done
if (( ! lock_observed )) \
    && ! resolve_checkpoint "$FINAL_STEP" >/dev/null 2>&1; then
    echo "[post-train-sweep] training did not start within ${TRAIN_START_TIMEOUT_SECONDS}s" >&2
    exit 1
fi

while ! flock -n "$TRAINING_LOCK" -c true; do
    sleep "$POLL_SECONDS"
done

if ! final_checkpoint="$(resolve_checkpoint "$FINAL_STEP")" \
    || [[ ! -r "$final_checkpoint" ]]; then
    echo "[post-train-sweep] training ended without unique step${FINAL_STEP}" >&2
    exit 1
fi
if ! grep -Eq 'SAT_TRAINING_GATE_RESULT=.*"status": "PASS"' "$TRAIN_LOG"; then
    echo "[post-train-sweep] training gate did not pass" >&2
    exit 1
fi

echo "[post-train-sweep] training complete and gate=PASS; beginning ordered evaluation"
exec env \
    RUN_ROOT="$RUN_ROOT" \
    STEP0_CKPT="$STEP0_CKPT" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    EVAL_SEED=42 \
    "$SWEEP"
