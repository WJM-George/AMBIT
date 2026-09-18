#!/usr/bin/env bash
set -euo pipefail

# Canonical learned gate for the active audio-aware P11 pilot.  The 90 frozen
# rows are split across independent decoder workers on physical GPUs 0 and 1;
# GPU 2-7 are outside this launcher's authority.  Multiple workers are needed
# because prefix-recompute autoregression is latency-bound and one 0.8B model
# uses only a small fraction of a 48-GiB GPU.  EMA weights and canonical
# prefix-recompute decoding are enforced again by the merger.
REPO="${P11_REPO:-.}"
PY="${P11_PYTHON:-$REPO/.venv/bin/python}"
CHECKPOINT="${1:?usage: $0 CHECKPOINT OUTPUT_DIR}"
OUTPUT_DIR="${2:?usage: $0 CHECKPOINT OUTPUT_DIR}"
GPU_IDS="${P11_EVAL_GPU_IDS:-0,1}"
WORKERS_PER_GPU="${P11_EVAL_WORKERS_PER_GPU:-4}"

if [[ "$GPU_IDS" != "0,1" ]]; then
    echo "[p11-eval] P11_EVAL_GPU_IDS must be exactly 0,1; got '$GPU_IDS'" >&2
    exit 2
fi
if [[ ! "$WORKERS_PER_GPU" =~ ^[1-4]$ ]]; then
    echo "[p11-eval] P11_EVAL_WORKERS_PER_GPU must be an integer in 1..4" >&2
    exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[p11-eval] checkpoint does not exist: $CHECKPOINT" >&2
    exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "[p11-eval] output directory already exists: $OUTPUT_DIR" >&2
    exit 2
fi

MODEL_CONFIG="$REPO/stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/qwen35_0p8b_sceneplan_p11_audio_aware_v1.json"
DATASET_CONFIG="$REPO/stable_audio_tools/configs/dataset_configs/sceneplan_p11_audio_aware_v1_pilot90.json"
EVALUATOR="$REPO/scripts/t2a/eval/evaluate_sceneplan_p11_audio_aware_pilot.py"
MERGER="$REPO/scripts/t2a/eval/merge_sceneplan_p11_audio_aware_pilot.py"
EXPECTED_ROWS=90
TOTAL_WORKERS=$((2 * WORKERS_PER_GPU))

mkdir -p "$OUTPUT_DIR"
CHECKPOINT_SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
echo "[p11-eval] GPUs=0,1 workers/GPU=$WORKERS_PER_GPU total_workers=$TOTAL_WORKERS rows=$EXPECTED_ROWS"
echo "[p11-eval] checkpoint_sha256=$CHECKPOINT_SHA256"

common_args=(
    --checkpoint "$CHECKPOINT"
    --checkpoint-sha256 "$CHECKPOINT_SHA256"
    --model-config "$MODEL_CONFIG"
    --dataset-config "$DATASET_CONFIG"
    --device cuda:0
    --decode-mode prefix_recompute
    --seed 42
    --skip-interventions
)

declare -a worker_pids=()
declare -a worker_outputs=()
cleanup_workers() {
    for pid in "${worker_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup_workers EXIT INT TERM
for ((worker = 0; worker < TOTAL_WORKERS; worker++)); do
    physical_gpu=$((worker % 2))
    worker_ordinals=""
    for ((ordinal = worker; ordinal < EXPECTED_ROWS; ordinal += TOTAL_WORKERS)); do
        worker_ordinals="${worker_ordinals:+$worker_ordinals,}$ordinal"
    done
    worker_stem="$(printf 'worker_%02d_gpu%d' "$worker" "$physical_gpu")"
    worker_output="$OUTPUT_DIR/$worker_stem.json"
    CUDA_VISIBLE_DEVICES="$physical_gpu" \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
        "$PY" -u "$EVALUATOR" \
        "${common_args[@]}" \
        --ordinals "$worker_ordinals" \
        --output "$worker_output" \
        >"$OUTPUT_DIR/$worker_stem.log" 2>&1 &
    worker_pids+=("$!")
    worker_outputs+=("$worker_output")
done

status=0
for pid in "${worker_pids[@]}"; do
    if wait "$pid"; then
        worker_status=0
    else
        worker_status=$?
    fi
    if (( status == 0 && worker_status != 0 )); then
        status=$worker_status
    fi
done
if (( status != 0 )); then
    echo "[p11-eval] a baseline worker failed; inspect $OUTPUT_DIR/worker_*.log" >&2
    exit "$status"
fi
trap - EXIT INT TERM

"$PY" "$MERGER" \
    --inputs "${worker_outputs[@]}" \
    --expected-rows "$EXPECTED_ROWS" \
    --output "$OUTPUT_DIR/full_pilot90_prefix_ema.json"

# Intervention decoding is intentionally separate from the disjoint 90-row
# baseline gate.  It runs only after the full pilot has merged successfully.
CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
    "$PY" -u "$EVALUATOR" \
    --checkpoint "$CHECKPOINT" \
    --checkpoint-sha256 "$CHECKPOINT_SHA256" \
    --model-config "$MODEL_CONFIG" \
    --dataset-config "$DATASET_CONFIG" \
    --device cuda:0 \
    --decode-mode prefix_recompute \
    --rows-per-task 3 \
    --seed 42 \
    --output "$OUTPUT_DIR/interventions_3x3_prefix_ema.json" \
    >"$OUTPUT_DIR/interventions_3x3_prefix_ema.log" 2>&1

echo "P11_PILOT90_REPORT=$OUTPUT_DIR/full_pilot90_prefix_ema.json"
echo "P11_INTERVENTION_REPORT=$OUTPUT_DIR/interventions_3x3_prefix_ema.json"
echo "P11_CHECKPOINT_SHA256=$CHECKPOINT_SHA256"
