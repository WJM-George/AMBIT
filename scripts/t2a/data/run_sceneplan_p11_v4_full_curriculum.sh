#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${P11_REPO:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
PY="${P11_PYTHON:-$REPO/.venv/bin/python}"
BUILDER="$REPO/scripts/t2a/data/build_sceneplan_p11_v4_full_curriculum.py"
VALIDATOR="$REPO/scripts/t2a/test/validate_sceneplan_p11_v4_full_curriculum.py"
BALANCE_AUDITOR="$REPO/scripts/t2a/test/audit_sceneplan_p11_v4_full_augmentation_balance.py"

ROOT="/mnt/sdb/audio_dataset/sceneplan_v2_1p124m"
P11_ROOT="$ROOT/p11_single_turn_15s_v2"
MANIFEST="${P11_MANIFEST:-$P11_ROOT/manifests/p11_train_4p8m_v6.sqlite}"
INDEX="${P11_INDEX:-$ROOT/revisions/speech_expansion_noalign_15s_v1/training_index/train.sqlite}"
CODEC="${P11_CODEC:-$P11_ROOT/model_sceneplan_codec_v4}"
CHALLENGE="${P11_HELDOUT_CHALLENGE:-$REPO/artifacts/sceneplan_p11/challenges/p11_v4_heldout_challenge_v1_20260901.sqlite}"
OUTPUT="${P11_OUTPUT:-$P11_ROOT/p11_v4_curriculum/p11_train_9p6m_ddp8_batch8_seed42_v2.sqlite}"
VALIDATION_REPORT="${P11_VALIDATION_REPORT:-${OUTPUT%.sqlite}.validation.json}"
BALANCE_REPORT="${P11_BALANCE_REPORT:-${OUTPUT%.sqlite}.augmentation_balance.json}"
NUM_PARTS="${P11_CURRICULUM_PARTS:-64}"
CPU_WORKERS="${P11_CURRICULUM_WORKERS:-8}"
KEEP_PARTS="${KEEP_PARTS:-0}"
MIN_FREE_DISK_GIB="${P11_MIN_FREE_DISK_GIB:-100}"

if [[ ! "$NUM_PARTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "P11_CURRICULUM_PARTS must be a positive integer" >&2
    exit 2
fi
if [[ ! "$CPU_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "P11_CURRICULUM_WORKERS must be a positive integer" >&2
    exit 2
fi
if (( CPU_WORKERS > NUM_PARTS )); then
    CPU_WORKERS="$NUM_PARTS"
fi
if [[ "$KEEP_PARTS" != "0" && "$KEEP_PARTS" != "1" ]]; then
    echo "KEEP_PARTS must be 0 or 1" >&2
    exit 2
fi
if [[ ! "$MIN_FREE_DISK_GIB" =~ ^[1-9][0-9]*$ ]]; then
    echo "P11_MIN_FREE_DISK_GIB must be a positive integer" >&2
    exit 2
fi
if [[ ! -x "$PY" || ! -r "$BUILDER" || ! -r "$VALIDATOR" || ! -r "$BALANCE_AUDITOR" || ! -r "$MANIFEST" || ! -r "$INDEX" || ! -r "$CHALLENGE" ]]; then
    echo "P11 full-curriculum prerequisites are incomplete" >&2
    exit 1
fi
if [[ ! -d "$CODEC" ]]; then
    echo "P11 codec directory is missing: $CODEC" >&2
    exit 1
fi
if [[ -e "$OUTPUT" ]]; then
    echo "refusing to overwrite canonical P11 full curriculum: $OUTPUT" >&2
    exit 1
fi
OUTPUT_PARENT="$(dirname -- "$OUTPUT")"
mkdir -p "$OUTPUT_PARENT"
free_kib="$(df -Pk "$OUTPUT_PARENT" | awk 'NR==2 {print $4}')"
required_kib="$((MIN_FREE_DISK_GIB * 1024 * 1024))"
if (( free_kib < required_kib )); then
    echo "only $((free_kib / 1024 / 1024)) GiB free at $OUTPUT_PARENT; need ${MIN_FREE_DISK_GIB} GiB" >&2
    exit 1
fi

MANIFEST_SHA256="$(sha256sum "$MANIFEST" | awk '{print $1}')"
INDEX_SHA256="$(sha256sum "$INDEX" | awk '{print $1}')"
CHALLENGE_SHA256="$(sha256sum "$CHALLENGE" | awk '{print $1}')"
"$PY" -u "$BALANCE_AUDITOR" \
    --manifest "$MANIFEST" \
    --sample-pairs 20000 \
    --seed 42 \
    --output "$BALANCE_REPORT"
PART_ROOT="${OUTPUT%.sqlite}.parts"
mkdir -p "$PART_ROOT"
parts=()
for (( part_index=0; part_index<NUM_PARTS; part_index++ )); do
    parts+=(
        "$PART_ROOT/part$(printf '%03d' "$part_index")-of-$(printf '%03d' "$NUM_PARTS").sqlite"
    )
done

run_worker() {
    local worker_index="$1"
    local part_index part log
    for (( part_index=worker_index; part_index<NUM_PARTS; part_index+=CPU_WORKERS )); do
        part="${parts[$part_index]}"
        if [[ -r "$part" ]]; then
            echo "[p11-full-data] reusing completed part $part_index: $part"
            continue
        fi
        log="$PART_ROOT/part$(printf '%03d' "$part_index").log"
        echo "[p11-full-data] starting part=$part_index worker=$worker_index log=$log"
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" -u "$BUILDER" part \
            --manifest "$MANIFEST" \
            --index "$INDEX" \
            --codec "$CODEC" \
            --heldout-challenge "$CHALLENGE" \
            --manifest-sha256 "$MANIFEST_SHA256" \
            --index-sha256 "$INDEX_SHA256" \
            --heldout-sha256 "$CHALLENGE_SHA256" \
            --part-index "$part_index" \
            --num-parts "$NUM_PARTS" \
            --seed 42 \
            --output "$part" \
            >"$log" 2>&1
        echo "[p11-full-data] completed part=$part_index worker=$worker_index"
    done
}

pids=()
for (( worker_index=0; worker_index<CPU_WORKERS; worker_index++ )); do
    run_worker "$worker_index" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
if (( failed )); then
    echo "one or more P11 full-curriculum workers failed; completed parts were retained" >&2
    exit 1
fi

"$PY" -u "$BUILDER" merge --inputs "${parts[@]}" --output "$OUTPUT"
"$PY" -u "$VALIDATOR" \
    --curriculum "$OUTPUT" \
    --output "$VALIDATION_REPORT"
if [[ "$KEEP_PARTS" == "0" ]]; then
    rm -f -- "${parts[@]}" "$PART_ROOT"/part*.log
    rmdir -- "$PART_ROOT"
    echo "[p11-full-data] removed merged part intermediates"
fi
echo "[p11-full-data] validation: $VALIDATION_REPORT"
echo "[p11-full-data] augmentation balance: $BALANCE_REPORT"
echo "[p11-full-data] complete: $OUTPUT"
