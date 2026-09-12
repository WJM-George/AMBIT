#!/usr/bin/env bash
set -euo pipefail

REPO=/home/tanhe/dataset_storage/stable-audio-tools
PY="$REPO/.venv/bin/python"
RUN_NAME="${RUN_NAME:?invoke this helper through run_t2a_spatial_chat_500m_8gpu.sh}"
RUN_LABEL="${RUN_LABEL:-t2a-spatial-chat}"
RUN_CATEGORY="${RUN_CATEGORY:-mainline}"
case "$RUN_CATEGORY" in
    mainline|pilots|preflights|benchmarks|probes|smoke|repairs) ;;
    *)
        echo "[$RUN_LABEL] invalid Spatial-CoT RUN_CATEGORY=$RUN_CATEGORY" >&2
        exit 2
        ;;
esac
RUN_ROOT="${RUN_ROOT:-/mnt/sdc/ckpts/spatial_cot/$RUN_CATEGORY/$RUN_NAME}"
CHECKPOINT_DIR="$RUN_ROOT/checkpoints"
LOG_DIR="$RUN_ROOT/logs"
WANDB_DIR="$RUN_ROOT/wandb"
# Keep this overrideable. Python multiprocessing creates additional socket
# names below TMPDIR and Linux limits AF_UNIX paths to roughly 108 bytes; long
# experiment names can otherwise make DataLoader workers fail before step 1.
TEMP_DIR="${TEMP_DIR:-$RUN_ROOT/tmp}"

MODEL_CONFIG="${MODEL_CONFIG:?the Spatial-Chat launcher must set MODEL_CONFIG}"
DATASET_CONFIG="${DATASET_CONFIG:?the Spatial-Chat launcher must set DATASET_CONFIG}"
VAL_DATASET_CONFIG="${VAL_DATASET_CONFIG:-}"
DISABLE_VALIDATION="${DISABLE_VALIDATION:-0}"
PRETRANSFORM_CKPT="${PRETRANSFORM_CKPT:-/mnt/sdc/ckpts/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt}"
LOAD_PRETRANSFORM="${LOAD_PRETRANSFORM:-0}"

# These defaults match the canonical family-based Spatial-CoT route. Performance
# tuning is applied explicitly after the eight-GPU matrix gate.
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_GPUS="${NUM_GPUS:-8}"
LOGGER="${LOGGER:-wandb}"
TRAINING_STRATEGY="${TRAINING_STRATEGY:-ddp_static}"
DDP_BUCKET_CAP_MB="${DDP_BUCKET_CAP_MB:-50}"
DDP_COMM_HOOK="${DDP_COMM_HOOK:-none}"
ACCUM_BATCHES="${ACCUM_BATCHES:-1}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-1.0}"
VAL_EVERY="${VAL_EVERY:--1}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:--1}"
OVERFIT_BATCHES="${OVERFIT_BATCHES:-0}"
BIND_TO_GPU_NUMA="${BIND_TO_GPU_NUMA:-1}"
RANK_AWARE_TRAINING_SEED="${RANK_AWARE_TRAINING_SEED:-1}"
TRAINING_SEED="${TRAINING_SEED:-42}"
MAX_STEPS="${MAX_STEPS:-125000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
SAVE_TOP_K="${SAVE_TOP_K:--1}"
BENCHMARK="${BENCHMARK:-0}"
BENCHMARK_WARMUP_BATCHES="${BENCHMARK_WARMUP_BATCHES:-1}"
TRAINING_GATE="${TRAINING_GATE:-0}"
TRAINING_GATE_WINDOW="${TRAINING_GATE_WINDOW:-20}"
TRAINING_GATE_MAX_LOSS_RATIO="${TRAINING_GATE_MAX_LOSS_RATIO:--1.0}"
TRAINING_GATE_GRADIENT_EVERY="${TRAINING_GATE_GRADIENT_EVERY:-1}"
ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-0}"
MIN_FREE_DISK_GIB="${MIN_FREE_DISK_GIB:-25}"
RESUME_CKPT="${RESUME_CKPT:-}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
PRETRAINED_ROUTE_WEIGHTS="${PRETRAINED_ROUTE_WEIGHTS:-ema}"
PRETRAINED_MODALITY_CKPT="${PRETRAINED_MODALITY_CKPT:-}"
PRETRAINED_MODALITY_IDS="${PRETRAINED_MODALITY_IDS:-}"
PRETRAINED_MODALITY_ROUTE_WEIGHTS="${PRETRAINED_MODALITY_ROUTE_WEIGHTS:-ema}"
REQUIRE_NO_PRETRAINED="${REQUIRE_NO_PRETRAINED:-0}"

require_positive_int() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "[$RUN_LABEL] $name must be a positive integer, got '$value'" >&2
        exit 2
    fi
}

require_nonnegative_int() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "[$RUN_LABEL] $name must be a non-negative integer, got '$value'" >&2
        exit 2
    fi
}

require_positive_int BATCH_SIZE "$BATCH_SIZE"
require_nonnegative_int NUM_WORKERS "$NUM_WORKERS"
require_positive_int NUM_GPUS "$NUM_GPUS"
require_positive_int ACCUM_BATCHES "$ACCUM_BATCHES"
require_positive_int DDP_BUCKET_CAP_MB "$DDP_BUCKET_CAP_MB"
require_positive_int MAX_STEPS "$MAX_STEPS"
require_positive_int CHECKPOINT_EVERY "$CHECKPOINT_EVERY"
require_nonnegative_int BENCHMARK_WARMUP_BATCHES "$BENCHMARK_WARMUP_BATCHES"
require_nonnegative_int OVERFIT_BATCHES "$OVERFIT_BATCHES"
require_nonnegative_int TRAINING_SEED "$TRAINING_SEED"
require_positive_int TRAINING_GATE_WINDOW "$TRAINING_GATE_WINDOW"
require_positive_int TRAINING_GATE_GRADIENT_EVERY "$TRAINING_GATE_GRADIENT_EVERY"
if [[ ! "$VAL_EVERY" =~ ^(-1|[0-9]+)$ ]]; then
    echo "[$RUN_LABEL] VAL_EVERY must be -1 or a non-negative integer, got '$VAL_EVERY'" >&2
    exit 2
fi
if [[ ! "$LIMIT_VAL_BATCHES" =~ ^(-1|[0-9]+)$ ]]; then
    echo "[$RUN_LABEL] LIMIT_VAL_BATCHES must be -1 or a non-negative integer, got '$LIMIT_VAL_BATCHES'" >&2
    exit 2
fi
if ! awk -v value="$GRADIENT_CLIP_VAL" 'BEGIN {exit !(value + 0 == value && value >= 0)}'; then
    echo "[$RUN_LABEL] GRADIENT_CLIP_VAL must be a non-negative number, got '$GRADIENT_CLIP_VAL'" >&2
    exit 2
fi
if [[ ! "$SAVE_TOP_K" =~ ^(-1|[0-9]+)$ ]]; then
    echo "[$RUN_LABEL] SAVE_TOP_K must be -1 or a non-negative integer, got '$SAVE_TOP_K'" >&2
    exit 2
fi
if [[ "$BENCHMARK" != "0" && "$BENCHMARK" != "1" ]]; then
    echo "[$RUN_LABEL] BENCHMARK must be 0 or 1" >&2
    exit 2
fi
if [[ "$TRAINING_GATE" != "0" && "$TRAINING_GATE" != "1" ]]; then
    echo "[$RUN_LABEL] TRAINING_GATE must be 0 or 1" >&2
    exit 2
fi
if [[ "$ENABLE_TORCH_COMPILE" != "0" && "$ENABLE_TORCH_COMPILE" != "1" ]]; then
    echo "[$RUN_LABEL] ENABLE_TORCH_COMPILE must be 0 or 1" >&2
    exit 2
fi
if ! awk -v value="$TRAINING_GATE_MAX_LOSS_RATIO" \
    'BEGIN {exit !(value + 0 == value && value >= -1)}'; then
    echo "[$RUN_LABEL] TRAINING_GATE_MAX_LOSS_RATIO must be >= -1" >&2
    exit 2
fi
if [[ "$BENCHMARK" == "1" ]] && (( MAX_STEPS <= BENCHMARK_WARMUP_BATCHES )); then
    echo "[$RUN_LABEL] MAX_STEPS must exceed BENCHMARK_WARMUP_BATCHES" >&2
    exit 2
fi
if [[ "$BIND_TO_GPU_NUMA" != "0" && "$BIND_TO_GPU_NUMA" != "1" ]]; then
    echo "[$RUN_LABEL] BIND_TO_GPU_NUMA must be 0 or 1" >&2
    exit 2
fi
if [[ "$RANK_AWARE_TRAINING_SEED" != "0" && "$RANK_AWARE_TRAINING_SEED" != "1" ]]; then
    echo "[$RUN_LABEL] RANK_AWARE_TRAINING_SEED must be 0 or 1" >&2
    exit 2
fi
if [[ "$REQUIRE_NO_PRETRAINED" != "0" && "$REQUIRE_NO_PRETRAINED" != "1" ]]; then
    echo "[$RUN_LABEL] REQUIRE_NO_PRETRAINED must be 0 or 1" >&2
    exit 2
fi
if [[ "$REQUIRE_NO_PRETRAINED" == "1" ]] \
    && [[ -n "$PRETRAINED_CKPT" || -n "$PRETRAINED_MODALITY_CKPT" ]]; then
    echo "[$RUN_LABEL] scratch contract forbids all pretrained checkpoints" >&2
    exit 2
fi
if [[ "$LOAD_PRETRANSFORM" != "0" && "$LOAD_PRETRANSFORM" != "1" ]]; then
    echo "[$RUN_LABEL] LOAD_PRETRANSFORM must be 0 or 1" >&2
    exit 2
fi
if [[ "$DISABLE_VALIDATION" != "0" && "$DISABLE_VALIDATION" != "1" ]]; then
    echo "[$RUN_LABEL] DISABLE_VALIDATION must be 0 or 1" >&2
    exit 2
fi
if [[ "$DISABLE_VALIDATION" == "1" ]]; then
    VAL_DATASET_CONFIG=""
    # A throughput benchmark intentionally has no validation loader.  Clear
    # the cadence knobs together with the dataset path so the generic
    # fail-closed check below does not reject this explicit no-validation
    # mode.  Full and resume runs keep DISABLE_VALIDATION=0 and therefore
    # still require their frozen validation contract.
    VAL_EVERY=-1
    LIMIT_VAL_BATCHES=-1
fi
case "$PRETRAINED_ROUTE_WEIGHTS" in
    ema|online) ;;
    *)
        echo "[$RUN_LABEL] PRETRAINED_ROUTE_WEIGHTS must be ema or online" >&2
        exit 2
        ;;
esac
case "${DDP_COMM_HOOK,,}" in
    none|off|false|bf16|bf16_compress|fp16|fp16_compress) ;;
    *)
        echo "[$RUN_LABEL] unsupported DDP_COMM_HOOK='$DDP_COMM_HOOK'" >&2
        exit 2
        ;;
esac
case "$TRAINING_STRATEGY" in
    auto|ddp_static|ddp|ddp_find_unused_parameters_true) ;;
    *)
        echo "[$RUN_LABEL] unsupported TRAINING_STRATEGY='$TRAINING_STRATEGY'" >&2
        exit 2
        ;;
esac
if [[ "${DDP_COMM_HOOK,,}" != "none" \
    && "${DDP_COMM_HOOK,,}" != "off" \
    && "${DDP_COMM_HOOK,,}" != "false" \
    && "$TRAINING_STRATEGY" == "auto" ]]; then
    echo "[$RUN_LABEL] a DDP communication hook requires a DDP strategy" >&2
    exit 2
fi

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR" "$WANDB_DIR" "$TEMP_DIR"
cd "$REPO"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export ENABLE_TORCH_COMPILE
export WANDB_DIR
export WANDB_NAME="$RUN_NAME"
export WANDB_CACHE_DIR="$RUN_ROOT/wandb-cache"
export WANDB_ARTIFACT_DIR="$RUN_ROOT/wandb-artifacts"
export TMPDIR="$TEMP_DIR"
export TMP="$TEMP_DIR"
export TEMP="$TEMP_DIR"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$RUN_ROOT/torchinductor-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$RUN_ROOT/triton-cache}"

IFS=',' read -r -a visible_gpu_entries <<<"$CUDA_VISIBLE_DEVICES"
visible_gpu_count=0
for entry in "${visible_gpu_entries[@]}"; do
    if [[ -n "${entry//[[:space:]]/}" ]]; then
        ((visible_gpu_count += 1))
    fi
done
if (( visible_gpu_count < NUM_GPUS )); then
    echo "[$RUN_LABEL] NUM_GPUS=$NUM_GPUS but CUDA_VISIBLE_DEVICES exposes only $visible_gpu_count device(s)" >&2
    exit 2
fi

mkdir -p \
    "$WANDB_CACHE_DIR" \
    "$WANDB_ARTIFACT_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" \
    "$TRITON_CACHE_DIR"
hard_nofile="$(ulimit -Hn)"
target_nofile=65536
if [[ "$hard_nofile" != "unlimited" ]] && (( hard_nofile < target_nofile )); then
    target_nofile="$hard_nofile"
fi
ulimit -Sn "$target_nofile"

if [[ ! -r "$MODEL_CONFIG" || ! -r "$DATASET_CONFIG" ]]; then
    echo "[$RUN_LABEL] required model or dataset config is unreadable" >&2
    exit 1
fi
validation_args=()
if [[ -n "$VAL_DATASET_CONFIG" ]]; then
    if [[ ! -r "$VAL_DATASET_CONFIG" ]]; then
        echo "[$RUN_LABEL] validation dataset config is unreadable: $VAL_DATASET_CONFIG" >&2
        exit 1
    fi
    validation_args=(--val-dataset-config "$VAL_DATASET_CONFIG")
    if (( VAL_EVERY > 0 )); then
        validation_args+=(--val-every "$VAL_EVERY")
    fi
    if (( LIMIT_VAL_BATCHES >= 0 )); then
        validation_args+=(--limit-val-batches "$LIMIT_VAL_BATCHES")
    fi
elif (( VAL_EVERY > 0 || LIMIT_VAL_BATCHES > 0 )); then
    echo "[$RUN_LABEL] validation cadence/limit requires VAL_DATASET_CONFIG" >&2
    exit 2
fi
pretransform_args=()
if [[ "$LOAD_PRETRANSFORM" == "1" ]]; then
    if [[ ! -r "$PRETRANSFORM_CKPT" ]]; then
        echo "[$RUN_LABEL] VAE checkpoint is unreadable: $PRETRANSFORM_CKPT" >&2
        exit 1
    fi
    pretransform_args=(--pretransform-ckpt-path "$PRETRANSFORM_CKPT")
fi

free_kib="$(df -Pk "$RUN_ROOT" | awk 'NR==2 {print $4}')"
required_kib="$(awk -v gib="$MIN_FREE_DISK_GIB" 'BEGIN {printf "%.0f", gib * 1024 * 1024}')"
if (( free_kib < required_kib )); then
    echo "[$RUN_LABEL] only $((free_kib / 1024 / 1024)) GiB free at $RUN_ROOT; need ${MIN_FREE_DISK_GIB} GiB" >&2
    exit 1
fi

pick_latest_checkpoint() {
    find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name '*step=*.ckpt' -print \
        | sed -n 's/.*step=\([0-9][0-9]*\)\.ckpt$/\1 &/p' \
        | sort -n \
        | tail -1 \
        | cut -d' ' -f2-
}

resume_args=()
pretrained_args=()
pretrained_modality_args=()
latest_checkpoint=""
if [[ -n "$RESUME_CKPT" ]]; then
    if [[ ! -r "$RESUME_CKPT" ]]; then
        echo "[$RUN_LABEL] RESUME_CKPT is unreadable: $RESUME_CKPT" >&2
        exit 1
    fi
    latest_checkpoint="$RESUME_CKPT"
elif [[ -r "$CHECKPOINT_DIR/last.ckpt" ]]; then
    # Lightning refreshes last.ckpt at epoch boundaries even when permanent
    # step checkpoints are less frequent. Prefer it so an interrupted run does
    # not silently restart merely because no step-named checkpoint exists yet.
    latest_checkpoint="$CHECKPOINT_DIR/last.ckpt"
else
    latest_checkpoint="$(pick_latest_checkpoint)"
fi
if [[ -n "$latest_checkpoint" ]]; then
    latest_step="$(sed -n 's/.*step=\([0-9][0-9]*\)\.ckpt$/\1/p' <<<"$latest_checkpoint")"
    if [[ -n "$latest_step" ]] && (( latest_step >= MAX_STEPS )); then
        echo "[$RUN_LABEL] run already complete at step=$latest_step (max_steps=$MAX_STEPS)"
        exit 0
    fi
    echo "[$RUN_LABEL] resuming from $latest_checkpoint"
    resume_args=(--ckpt-path "$latest_checkpoint")
elif [[ -n "$PRETRAINED_CKPT" ]]; then
    if [[ ! -r "$PRETRAINED_CKPT" ]]; then
        echo "[$RUN_LABEL] PRETRAINED_CKPT is unreadable: $PRETRAINED_CKPT" >&2
        exit 1
    fi
    echo "[$RUN_LABEL] warm-starting model weights from $PRETRAINED_CKPT"
    pretrained_args=(
        --pretrained-ckpt-path "$PRETRAINED_CKPT"
        --pretrained-route-weights "$PRETRAINED_ROUTE_WEIGHTS"
    )
fi
if [[ -z "$latest_checkpoint" && -n "$PRETRAINED_MODALITY_CKPT" ]]; then
    if [[ ! -r "$PRETRAINED_MODALITY_CKPT" ]]; then
        echo "[$RUN_LABEL] PRETRAINED_MODALITY_CKPT is unreadable: $PRETRAINED_MODALITY_CKPT" >&2
        exit 1
    fi
    if [[ -z "$PRETRAINED_MODALITY_IDS" ]]; then
        echo "[$RUN_LABEL] PRETRAINED_MODALITY_IDS is required with PRETRAINED_MODALITY_CKPT" >&2
        exit 2
    fi
    echo "[$RUN_LABEL] warm-starting modality interfaces $PRETRAINED_MODALITY_IDS from $PRETRAINED_MODALITY_CKPT"
    pretrained_modality_args=(
        --pretrained-modality-ckpt-path "$PRETRAINED_MODALITY_CKPT"
        --pretrained-modality-ids "$PRETRAINED_MODALITY_IDS"
        --pretrained-modality-route-weights "$PRETRAINED_MODALITY_ROUTE_WEIGHTS"
    )
fi

numa_args=()
if [[ "$BIND_TO_GPU_NUMA" == "1" ]]; then
    numa_args=(--bind-to-gpu-numa)
fi

MIN_ROOT_FREE_GIB="${MIN_ROOT_FREE_GIB:-5}"
root_free_kib="$(df -Pk / | awk 'NR==2 {print $4}')"
root_required_kib="$(
    awk -v gib="$MIN_ROOT_FREE_GIB" 'BEGIN {printf "%.0f", gib * 1024 * 1024}'
)"
if (( root_free_kib < root_required_kib )); then
    echo "[$RUN_LABEL] root filesystem has only $((root_free_kib / 1024 / 1024)) GiB free; need ${MIN_ROOT_FREE_GIB} GiB" >&2
    echo "[$RUN_LABEL] clean the NVRM kern.log/syslog storm before launching GPU work" >&2
    exit 1
fi

# Do not touch CUDA while the host is exhibiting the known bad-register
# failure. Merely probing the driver has filled /var/log with tens of
# gigabytes of duplicate NVRM messages on this machine.
if [[ "${SAT_IGNORE_NVRM_PREFLIGHT:-0}" != "1" ]] \
    && [[ -r /var/log/kern.log ]] \
    && grep -Eq \
        'gpuHandleSanityCheckRegReadError|Possible bad register read' \
        < <(tail -n 4000 /var/log/kern.log); then
    echo "[$RUN_LABEL] refusing CUDA launch: NVIDIA bad-register errors are at the end of /var/log/kern.log" >&2
    echo "[$RUN_LABEL] recover/reboot the driver and rotate the logs first; SAT_IGNORE_NVRM_PREFLIGHT=1 overrides this guard" >&2
    exit 1
fi

seed_args=(--rank-aware-training-seed "$RANK_AWARE_TRAINING_SEED")
benchmark_args=()
if [[ "$BENCHMARK" == "1" ]]; then
    benchmark_args=(--benchmark --benchmark-warmup-batches "$BENCHMARK_WARMUP_BATCHES")
fi
training_gate_args=()
if [[ "$TRAINING_GATE" == "1" ]]; then
    training_gate_args=(
        --training-gate
        --training-gate-window "$TRAINING_GATE_WINDOW"
        --training-gate-max-loss-ratio "$TRAINING_GATE_MAX_LOSS_RATIO"
        --training-gate-gradient-every "$TRAINING_GATE_GRADIENT_EVERY"
    )
fi

echo "[$RUN_LABEL] run=$RUN_NAME batch/GPU=$BATCH_SIZE global_batch=$((BATCH_SIZE * NUM_GPUS)) workers/rank=$NUM_WORKERS"
echo "[$RUN_LABEL] max_steps=$MAX_STEPS checkpoint_every=$CHECKPOINT_EVERY"
echo "[$RUN_LABEL] load_pretransform=$LOAD_PRETRANSFORM"
echo "[$RUN_LABEL] pretrained_ckpt=${PRETRAINED_CKPT:-none} resume_ckpt=${latest_checkpoint:-none}"
if [[ -n "$PRETRAINED_CKPT" ]]; then
    echo "[$RUN_LABEL] pretrained_route_weights=$PRETRAINED_ROUTE_WEIGHTS"
else
    echo "[$RUN_LABEL] pretrained_route_weights=n/a"
fi
echo "[$RUN_LABEL] pretrained_modality_ckpt=${PRETRAINED_MODALITY_CKPT:-none} ids=${PRETRAINED_MODALITY_IDS:-none} weights=$PRETRAINED_MODALITY_ROUTE_WEIGHTS"
echo "[$RUN_LABEL] require_no_pretrained=$REQUIRE_NO_PRETRAINED"
echo "[$RUN_LABEL] rank_aware_seed=$RANK_AWARE_TRAINING_SEED numa=$BIND_TO_GPU_NUMA strategy=$TRAINING_STRATEGY ddp_bucket_mb=$DDP_BUCKET_CAP_MB ddp_comm_hook=$DDP_COMM_HOOK"
echo "[$RUN_LABEL] training_seed=$TRAINING_SEED"
echo "[$RUN_LABEL] benchmark=$BENCHMARK benchmark_warmup=$BENCHMARK_WARMUP_BATCHES save_top_k=$SAVE_TOP_K"
echo "[$RUN_LABEL] training_gate=$TRAINING_GATE gate_window=$TRAINING_GATE_WINDOW gate_max_loss_ratio=$TRAINING_GATE_MAX_LOSS_RATIO gate_gradient_every=$TRAINING_GATE_GRADIENT_EVERY"
echo "[$RUN_LABEL] torch_compile=$ENABLE_TORCH_COMPILE"
echo "[$RUN_LABEL] accum=$ACCUM_BATCHES grad_clip=$GRADIENT_CLIP_VAL val_every=$VAL_EVERY limit_val_batches=$LIMIT_VAL_BATCHES overfit_batches=$OVERFIT_BATCHES"
echo "[$RUN_LABEL] validation_disabled=$DISABLE_VALIDATION"

"$PY" -u train.py \
    --model-config "$MODEL_CONFIG" \
    --dataset-config "$DATASET_CONFIG" \
    "${validation_args[@]}" \
    "${pretrained_args[@]}" \
    "${pretrained_modality_args[@]}" \
    "${pretransform_args[@]}" \
    --name "$RUN_NAME" \
    --num-gpus "$NUM_GPUS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --precision bf16-mixed \
    --strategy "$TRAINING_STRATEGY" \
    --ddp-bucket-cap-mb "$DDP_BUCKET_CAP_MB" \
    --ddp-comm-hook "$DDP_COMM_HOOK" \
    --ddp-timeout-min 10 \
    --accum-batches "$ACCUM_BATCHES" \
    --gradient-clip-val "$GRADIENT_CLIP_VAL" \
    --overfit-batches "$OVERFIT_BATCHES" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --save-dir "$RUN_ROOT" \
    --wandb-dir "$WANDB_DIR" \
    --temp-dir "$TEMP_DIR" \
    --min-free-disk-gib "$MIN_FREE_DISK_GIB" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --save-top-k "$SAVE_TOP_K" \
    --max-steps "$MAX_STEPS" \
    --logger "$LOGGER" \
    --seed "$TRAINING_SEED" \
    "${benchmark_args[@]}" \
    "${training_gate_args[@]}" \
    "${seed_args[@]}" \
    "${numa_args[@]}" \
    "${resume_args[@]}" 2>&1 | tee -a "$LOG_DIR/train.log"
