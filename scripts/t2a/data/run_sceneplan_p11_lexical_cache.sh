#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${P11_REPO:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
PY="${P11_PYTHON:-$REPO/.venv/bin/python}"
if (( $# != 1 )); then
    echo "usage: $0 {pilot|trial|heldout|train|validation|test}" >&2
    exit 2
fi
PROFILE="$1"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
KEEP_PARTS="${KEEP_PARTS:-0}"
MIN_FREE_DISK_GIB="${P11_MIN_FREE_DISK_GIB:-100}"
ROOT="/mnt/sdb/audio_dataset/sceneplan_v2_1p124m"
P11_ROOT="$ROOT/p11_single_turn_15s_v2"

case "$PROFILE" in
    pilot)
        INDEX="$ROOT/revisions/speech_expansion_noalign_15s_v1/training_index/train.sqlite"
        MANIFEST="$P11_ROOT/manifests/p11_train_pilot90_v6.sqlite"
        OUTPUT="$P11_ROOT/lexical_cache/p11_train_pilot90_distilwhisper_v1.sqlite"
        ;;
    trial)
        INDEX="$ROOT/revisions/speech_expansion_noalign_15s_v1/training_index/train.sqlite"
        MANIFEST="$P11_ROOT/manifests/p11_train_trial30k_owner_balanced_v1.sqlite"
        OUTPUT="$P11_ROOT/lexical_cache/p11_train_trial10k_owner_balanced_distilwhisper_v1.sqlite"
        ;;
    heldout)
        INDEX="$ROOT/revisions/speech_expansion_noalign_15s_v1/training_index/validation.sqlite"
        MANIFEST="$P11_ROOT/manifests/p11_validation_heldout900_v6.sqlite"
        OUTPUT="$P11_ROOT/lexical_cache/p11_validation_heldout900_distilwhisper_v1.sqlite"
        ;;
    train)
        INDEX="$ROOT/revisions/speech_expansion_noalign_15s_v1/training_index/train.sqlite"
        MANIFEST="$P11_ROOT/manifests/p11_train_4p8m_v6.sqlite"
        OUTPUT="$P11_ROOT/lexical_cache/p11_train_distilwhisper_v1.sqlite"
        ;;
    validation)
        INDEX="$ROOT/revisions/speech_expansion_noalign_15s_v1/training_index/validation.sqlite"
        MANIFEST="$P11_ROOT/manifests/p11_validation_96k_v6.sqlite"
        OUTPUT="$P11_ROOT/lexical_cache/p11_validation_distilwhisper_v1.sqlite"
        ;;
    test)
        INDEX="$ROOT/revisions/speech_expansion_noalign_15s_v1/training_index/test.sqlite"
        MANIFEST="$P11_ROOT/manifests/p11_test_24k_v6.sqlite"
        OUTPUT="$P11_ROOT/lexical_cache/p11_test_distilwhisper_v1.sqlite"
        ;;
    *)
        echo "usage: $0 {pilot|trial|heldout|train|validation|test}" >&2
        exit 2
        ;;
esac

INDEX="${P11_INDEX:-$INDEX}"
MANIFEST="${P11_MANIFEST:-$MANIFEST}"
OUTPUT="${P11_OUTPUT:-$OUTPUT}"
MANIFEST_STEM="$(basename -- "${MANIFEST%.sqlite}")"
INVENTORY="${P11_ORDINAL_INVENTORY:-$P11_ROOT/cache_indices/${MANIFEST_STEM}_ordinals_v1.sqlite}"
if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "GPU_IDS must be a comma-separated physical-GPU list" >&2
    exit 2
fi
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
GPU_COUNT="${#GPU_ARRAY[@]}"
declare -A SEEN_GPUS=()
for gpu_id in "${GPU_ARRAY[@]}"; do
    if [[ -n "${SEEN_GPUS[$gpu_id]:-}" ]]; then
        echo "GPU_IDS contains duplicate physical GPU $gpu_id" >&2
        exit 2
    fi
    if ! nvidia-smi -i "$gpu_id" --query-gpu=index \
        --format=csv,noheader,nounits >/dev/null 2>&1; then
        echo "GPU_IDS contains unavailable physical GPU $gpu_id" >&2
        exit 2
    fi
    SEEN_GPUS[$gpu_id]=1
done
if [[ "$PROFILE" == "train" ]]; then
    if [[ "$GPU_IDS" != "0,1,2,3,4,5,6,7" ]]; then
        echo "canonical full lexical-cache build requires physical GPUs 0..7" >&2
        exit 2
    fi
    # Consume the complete nvidia-smi stream; an early-exiting `rg -q` is
    # incompatible with pipefail because the producer receives SIGPIPE.
    if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        2>/dev/null | awk \
        'BEGIN { found=0 } /^[[:space:]]*[0-9]+[[:space:]]*$/ { found=1 } END { exit found ? 0 : 1 }'; then
        echo "refusing to share GPUs during the full lexical-cache build" >&2
        exit 2
    fi
fi
if [[ "$KEEP_PARTS" != "0" && "$KEEP_PARTS" != "1" ]]; then
    echo "KEEP_PARTS must be 0 or 1" >&2
    exit 2
fi
if [[ ! "$MIN_FREE_DISK_GIB" =~ ^[1-9][0-9]*$ ]]; then
    echo "P11_MIN_FREE_DISK_GIB must be a positive integer" >&2
    exit 2
fi
if [[ ! -x "$PY" || ! -r "$INDEX" || ! -r "$MANIFEST" ]]; then
    echo "P11 lexical cache prerequisites are incomplete" >&2
    exit 1
fi
if [[ -e "$OUTPUT" ]]; then
    echo "refusing to overwrite canonical lexical cache: $OUTPUT" >&2
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
if [[ -r "$INVENTORY" ]]; then
    "$PY" "$REPO/scripts/t2a/data/build_sceneplan_p11_cache_ordinal_inventory.py" \
        --manifest "$MANIFEST" --output "$INVENTORY" --validate-only
else
    "$PY" "$REPO/scripts/t2a/data/build_sceneplan_p11_cache_ordinal_inventory.py" \
        --manifest "$MANIFEST" --output "$INVENTORY"
fi

if [[ "$PROFILE" == "train" ]]; then
    DEFAULT_PARTS=64
else
    DEFAULT_PARTS="$GPU_COUNT"
fi
NUM_PARTS="${P11_CACHE_PARTS:-$DEFAULT_PARTS}"
if [[ ! "$NUM_PARTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "P11_CACHE_PARTS must be a positive integer" >&2
    exit 2
fi
if [[ "$PROFILE" == "train" ]] && (( NUM_PARTS % GPU_COUNT != 0 )); then
    echo "full lexical-cache parts must divide evenly across 8 GPUs" >&2
    exit 2
fi

PART_ROOT="${OUTPUT%.sqlite}.parts"
mkdir -p "$PART_ROOT"
parts=()
for (( shard_index=0; shard_index<NUM_PARTS; shard_index++ )); do
    parts+=(
        "$PART_ROOT/part$(printf '%03d' "$shard_index")-of-$(printf '%03d' "$NUM_PARTS").sqlite"
    )
done

run_worker() {
    local worker_index="$1"
    local gpu_id="$2"
    local shard_index part log
    for (( shard_index=worker_index; shard_index<NUM_PARTS; shard_index+=GPU_COUNT )); do
        part="${parts[$shard_index]}"
        if [[ -r "$part" ]]; then
            echo "[p11-lexical] reusing completed shard $shard_index: $part"
            continue
        fi
        log="$PART_ROOT/part$(printf '%03d' "$shard_index").log"
        echo "[p11-lexical] starting shard=$shard_index gpu=$gpu_id log=$log"
        CUDA_VISIBLE_DEVICES="$gpu_id" TOKENIZERS_PARALLELISM=false \
            OMP_NUM_THREADS=1 "$PY" -u \
            "$REPO/scripts/t2a/data/build_sceneplan_p11_lexical_cache.py" \
            --index "$INDEX" \
            --manifest "$MANIFEST" \
            --ordinal-inventory "$INVENTORY" \
            --output "$part" \
            --device cuda:0 \
            --compute-type "${P11_ASR_COMPUTE_TYPE:-float16}" \
            --batch-size 8 \
            --shard-index "$shard_index" \
            --num-shards "$NUM_PARTS" \
            >"$log" 2>&1
        echo "[p11-lexical] completed shard=$shard_index gpu=$gpu_id"
    done
}

pids=()
for worker_index in "${!GPU_ARRAY[@]}"; do
    run_worker "$worker_index" "${GPU_ARRAY[$worker_index]}" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
if (( failed )); then
    echo "one or more P11 lexical-cache workers failed; completed parts were retained" >&2
    exit 1
fi

"$PY" "$REPO/scripts/t2a/data/merge_sceneplan_p11_lexical_cache.py" \
    --inputs "${parts[@]}" \
    --output "$OUTPUT"
if [[ "$KEEP_PARTS" == "0" ]]; then
    rm -f -- "${parts[@]}" "$PART_ROOT"/part*.log
    rmdir -- "$PART_ROOT"
    echo "[p11-lexical] removed merged shard intermediates"
fi
echo "[p11-lexical] complete: $OUTPUT"
