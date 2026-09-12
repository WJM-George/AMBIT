#!/usr/bin/env bash
set -euo pipefail

# Active P11 launcher.  The audio-aware G/U/E contract is the only mainline
# candidate; the old v4 arms remain explicit experiment baselines and are
# never selected by default.  Every profile keeps batch/GPU=8 and seed=42.
REPO="${P11_REPO:-${AMBIT_CKPT_ROOT}/stable-audio-tools-workspace}"
PY="${P11_PYTHON:-$REPO/.venv/bin/python}"
PROFILE="${1:-preflight}"
RUN_NAME="${RUN_NAME:?set a unique RUN_NAME}"
P11_ARM="${P11_ARM:-audio_aware}"
case "$P11_ARM" in
    audio_aware)
        P11_GRAPH_FAMILY="audio_aware"
        P11_MODEL_FILE="qwen35_0p8b_sceneplan_p11_audio_aware_v1.json"
        P11_EXPECTED_ARM="audio_aware_flow_r1_v1"
        P11_PILOT_FILE="sceneplan_p11_audio_aware_v1_pilot90.json"
        # Full-corpus audio-aware manifests are intentionally not aliased to
        # v4 data.  Screening/scale stay locked until those manifests exist.
        P11_SCREENING_FILE=""
        ;;
    canonical)
        P11_GRAPH_FAMILY="v4"
        P11_MODEL_FILE="qwen35_0p8b_sceneplan_p11_reliable_asr_assembler_v2.json"
        P11_EXPECTED_ARM="sketch_first_transfusion_cot_v4"
        P11_PILOT_FILE="sceneplan_p11_pilot90_curriculum_pair_aware_transfusion_cot_v4_reliable_asr_v1.json"
        P11_SCREENING_FILE="sceneplan_p11_trial30k_matched_screening_v1_transfusion_cot_v4_reliable_asr.json"
        ;;
    direct_mse)
        P11_GRAPH_FAMILY="v4"
        P11_MODEL_FILE="qwen35_0p8b_sceneplan_p11_baseline_direct_mse.json"
        P11_EXPECTED_ARM="sketch_first_direct_mse_v4"
        P11_PILOT_FILE="sceneplan_p11_pilot90_curriculum_pair_aware_transfusion_cot_v4_reliable_asr_v1.json"
        P11_SCREENING_FILE="sceneplan_p11_trial30k_matched_screening_v1_transfusion_cot_v4_reliable_asr.json"
        ;;
    discrete_d0)
        P11_GRAPH_FAMILY="d0"
        P11_MODEL_FILE="qwen35_0p8b_sceneplan_p11_baseline_discrete_d0.json"
        P11_EXPECTED_ARM="discrete_v0"
        P11_PILOT_FILE="sceneplan_p11_pilot90_curriculum_pair_aware_discrete_d0.json"
        P11_SCREENING_FILE="sceneplan_p11_trial30k_matched_screening_v1_discrete_d0.json"
        ;;
    *)
        echo "[p11] unsupported P11_ARM=$P11_ARM" >&2
        exit 2
        ;;
esac
MODEL_CONFIG="${MODEL_CONFIG:-$REPO/stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/$P11_MODEL_FILE}"
PILOT_DATASET="$REPO/stable_audio_tools/configs/dataset_configs/$P11_PILOT_FILE"
SCREENING_DATASET="$REPO/stable_audio_tools/configs/dataset_configs/$P11_SCREENING_FILE"
SCALE_V4_DATASET="$REPO/stable_audio_tools/configs/dataset_configs/sceneplan_p11_train269568_pair_aware_transfusion_cot_v4_reliable_asr_seed42.json"
SCALE_D0_DATASET="$REPO/stable_audio_tools/configs/dataset_configs/sceneplan_p11_train269568_pair_aware_discrete_d0_seed42.json"
SCALE_CURRICULUM="${AMBIT_DATA_ROOT}/sceneplan_v2_1p124m/p11_single_turn_15s_v2/p11_v4_curriculum/p11_train_269568_ddp8_rank_balanced_batch8_seed42_v2.sqlite"

case "$PROFILE" in
    overfit)
        DATASET_DEFAULT="$PILOT_DATASET"
        GPU_IDS_DEFAULT="0"
        MAX_STEPS_DEFAULT=300
        CHECKPOINT_EVERY_DEFAULT=60
        SAVE_TOP_K_DEFAULT=1
        GATE_WINDOW_DEFAULT=10
        GATE_RATIO_DEFAULT=0.95
        VALIDATOR_BASE_SCENES=30
        ;;
    preflight)
        DATASET_DEFAULT="$PILOT_DATASET"
        GPU_IDS_DEFAULT="0,1"
        MAX_STEPS_DEFAULT=8
        CHECKPOINT_EVERY_DEFAULT=1000000
        SAVE_TOP_K_DEFAULT=0
        GATE_WINDOW_DEFAULT=2
        GATE_RATIO_DEFAULT=-1
        VALIDATOR_BASE_SCENES=30
        ;;
    pilot)
        DATASET_DEFAULT="$PILOT_DATASET"
        GPU_IDS_DEFAULT="0"
        MAX_STEPS_DEFAULT=1000
        CHECKPOINT_EVERY_DEFAULT=250
        SAVE_TOP_K_DEFAULT=1
        GATE_WINDOW_DEFAULT=100
        GATE_RATIO_DEFAULT=0.95
        VALIDATOR_BASE_SCENES=30
        ;;
    screening)
        DATASET_DEFAULT="$SCREENING_DATASET"
        GPU_IDS_DEFAULT="0"
        MAX_STEPS_DEFAULT=10000
        CHECKPOINT_EVERY_DEFAULT=10000
        SAVE_TOP_K_DEFAULT=1
        GATE_WINDOW_DEFAULT=1000
        GATE_RATIO_DEFAULT=0.99
        VALIDATOR_BASE_SCENES=30
        ;;
    scale_preflight)
        # Systems preflight must exercise the exact pair-aware data/order used
        # by the following scale trial. The old matched-triplet screening order
        # cannot guarantee a +/- control-direction pair inside one batch.
        DATASET_DEFAULT="$SCALE_V4_DATASET"
        GPU_IDS_DEFAULT="0,1"
        # Cover enough heterogeneous G/U/E batches that one-time kernels,
        # allocator growth, and loader prefetch transients cannot dominate the
        # sustained two-GPU measurement.
        MAX_STEPS_DEFAULT=128
        CHECKPOINT_EVERY_DEFAULT=1000000
        SAVE_TOP_K_DEFAULT=0
        GATE_WINDOW_DEFAULT=2
        GATE_RATIO_DEFAULT=-1
        VALIDATOR_BASE_SCENES=30
        ;;
    scale_trial)
        DATASET_DEFAULT="$SCALE_V4_DATASET"
        GPU_IDS_DEFAULT="0,1"
        MAX_STEPS_DEFAULT=10000
        CHECKPOINT_EVERY_DEFAULT=10000
        SAVE_TOP_K_DEFAULT=1
        GATE_WINDOW_DEFAULT=1000
        GATE_RATIO_DEFAULT=0.99
        VALIDATOR_BASE_SCENES=30
        ;;
    resume_smoke)
        # This profile is invoked twice by the O(1)-resume orchestrator: first
        # to step 8, then from that full-state checkpoint to step 16.
        DATASET_DEFAULT="$SCALE_V4_DATASET"
        GPU_IDS_DEFAULT="0,1"
        MAX_STEPS_DEFAULT=8
        CHECKPOINT_EVERY_DEFAULT=8
        SAVE_TOP_K_DEFAULT=1
        GATE_WINDOW_DEFAULT=2
        GATE_RATIO_DEFAULT=-1
        VALIDATOR_BASE_SCENES=30
        ;;
    trial)
        echo "[p11-v4] legacy trial alias is retired; use the gated screening profile" >&2
        exit 2
        ;;
    full)
        echo "[p11-v4] full training is locked until matched pilot, held-out, and intervention gates pass" >&2
        exit 2
        ;;
    *)
        echo "usage: RUN_NAME=<name> $0 {overfit|preflight|pilot|screening|scale_preflight|scale_trial|resume_smoke|full}" >&2
        exit 2
        ;;
esac

if [[ "$P11_GRAPH_FAMILY" == "audio_aware" ]]; then
    case "$PROFILE" in
        overfit|preflight|pilot) ;;
        *)
            echo "[p11] audio-aware screening/scale is locked until its manifests and two-GPU ordering contract are frozen" >&2
            exit 2
            ;;
    esac
fi

DATASET_CONFIG="${DATASET_CONFIG:-$DATASET_DEFAULT}"
GPU_IDS="${GPU_IDS:-$GPU_IDS_DEFAULT}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-6}"
BENCHMARK_WARMUP_BATCHES_DEFAULT=3
if [[ "$PROFILE" == "scale_preflight" ]]; then
    # The 16-step/8-warmup probe still encountered a new-shape startup event
    # in its first measured batch.  Thirty-two warmup steps cover four full
    # eight-batch curriculum superblocks before the 96-step steady window.
    BENCHMARK_WARMUP_BATCHES_DEFAULT=32
elif [[ "$PROFILE" == "scale_trial" ]]; then
    # The first pass through all G/U/E batch shapes can trigger one-time kernel
    # compilation after step three. Exclude that startup event from the
    # sustained two-GPU throughput measurement.
    BENCHMARK_WARMUP_BATCHES_DEFAULT=8
elif [[ "$PROFILE" == "resume_smoke" ]]; then
    BENCHMARK_WARMUP_BATCHES_DEFAULT=2
fi
BENCHMARK_WARMUP_BATCHES="${BENCHMARK_WARMUP_BATCHES:-$BENCHMARK_WARMUP_BATCHES_DEFAULT}"
DDP_BUCKET_CAP_MB_DEFAULT=150
DDP_COMM_HOOK_DEFAULT=none
BIND_TO_GPU_NUMA_DEFAULT=0
if [[ "$PROFILE" == "scale_preflight" || "$PROFILE" == "scale_trial" || "$PROFILE" == "resume_smoke" ]]; then
    # This host has one RTX 4090 per NUMA node and every GPU pair is SYS.  Use
    # the already validated P10 locality/communication recipe so 154 MB of
    # trainable P11 gradients do not become a cross-socket FP32 bottleneck.
    DDP_BUCKET_CAP_MB_DEFAULT=50
    DDP_COMM_HOOK_DEFAULT=bf16
    BIND_TO_GPU_NUMA_DEFAULT=1
fi
DDP_BUCKET_CAP_MB="${DDP_BUCKET_CAP_MB:-$DDP_BUCKET_CAP_MB_DEFAULT}"
DDP_COMM_HOOK="${DDP_COMM_HOOK:-$DDP_COMM_HOOK_DEFAULT}"
BIND_TO_GPU_NUMA="${BIND_TO_GPU_NUMA:-$BIND_TO_GPU_NUMA_DEFAULT}"
MAX_STEPS="${MAX_STEPS:-$MAX_STEPS_DEFAULT}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-$CHECKPOINT_EVERY_DEFAULT}"
SAVE_TOP_K="${SAVE_TOP_K:-$SAVE_TOP_K_DEFAULT}"
GATE_WINDOW="${GATE_WINDOW:-$GATE_WINDOW_DEFAULT}"
GATE_RATIO="${GATE_RATIO:-$GATE_RATIO_DEFAULT}"
MIN_FREE_DISK_GIB="${MIN_FREE_DISK_GIB:-100}"
DDP_TIMEOUT_MIN="${DDP_TIMEOUT_MIN:-5}"
P11_AUTO_EVAL="${P11_AUTO_EVAL:-0}"
OVERFIT_BATCHES_DEFAULT=2
if [[ "$PROFILE" == "overfit" && "$P11_GRAPH_FAMILY" == "audio_aware" ]]; then
    # The 90-row pilot contains interleaved G/U/E supervision.  Eleven full
    # local batches cover 88 rows and every edit operation; a two-batch legacy
    # overfit silently omits most Editing supervision.
    OVERFIT_BATCHES_DEFAULT=11
fi
OVERFIT_BATCHES="${OVERFIT_BATCHES:-$OVERFIT_BATCHES_DEFAULT}"
TRAINING_SEED="${P11_SEED:-42}"
RUN_ROOT="${RUN_ROOT:-${AMBIT_CKPT_ROOT}/sceneplan_p11/$RUN_NAME}"
CKPT_PATH="${CKPT_PATH:-}"
PRETRAINED_CKPT_PATH="${PRETRAINED_CKPT_PATH:-}"
PRETRAINED_ROUTE_WEIGHTS="${PRETRAINED_ROUTE_WEIGHTS:-ema}"

positive_int() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "[p11] $name must be a positive integer, got '$value'" >&2
        exit 2
    fi
}

nonnegative_int() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "[p11] $name must be a non-negative integer, got '$value'" >&2
        exit 2
    fi
}

if [[ "$BATCH_SIZE" != "8" ]]; then
    echo "[p11] canonical per-GPU batch size is fixed at 8" >&2
    exit 2
fi
positive_int NUM_WORKERS "$NUM_WORKERS"
positive_int BENCHMARK_WARMUP_BATCHES "$BENCHMARK_WARMUP_BATCHES"
positive_int MAX_STEPS "$MAX_STEPS"
positive_int CHECKPOINT_EVERY "$CHECKPOINT_EVERY"
positive_int MIN_FREE_DISK_GIB "$MIN_FREE_DISK_GIB"
positive_int DDP_TIMEOUT_MIN "$DDP_TIMEOUT_MIN"
positive_int DDP_BUCKET_CAP_MB "$DDP_BUCKET_CAP_MB"
nonnegative_int SAVE_TOP_K "$SAVE_TOP_K"
positive_int OVERFIT_BATCHES "$OVERFIT_BATCHES"
nonnegative_int P11_SEED "$TRAINING_SEED"
if [[ "$P11_AUTO_EVAL" != "0" && "$P11_AUTO_EVAL" != "1" ]]; then
    echo "[p11] P11_AUTO_EVAL must be 0 or 1" >&2
    exit 2
fi
if [[ "$BIND_TO_GPU_NUMA" != "0" && "$BIND_TO_GPU_NUMA" != "1" ]]; then
    echo "[p11] BIND_TO_GPU_NUMA must be 0 or 1" >&2
    exit 2
fi
if [[ "$DDP_COMM_HOOK" != "none" && "$DDP_COMM_HOOK" != "bf16" ]]; then
    echo "[p11] DDP_COMM_HOOK must be none or bf16" >&2
    exit 2
fi
if [[ "$SAVE_TOP_K" -gt 1 ]]; then
    echo "[p11] SAVE_TOP_K may only be 0 or 1" >&2
    exit 2
fi
if [[ "$PROFILE" == "pilot" && "$P11_AUTO_EVAL" == "1" && "$SAVE_TOP_K" != "1" ]]; then
    echo "[p11] automatic pilot evaluation requires SAVE_TOP_K=1" >&2
    exit 2
fi
if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "[p11] GPU_IDS must be a comma-separated physical-GPU list" >&2
    exit 2
fi
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"
declare -A SEEN_GPUS=()
for gpu_id in "${GPU_ARRAY[@]}"; do
    if [[ "$gpu_id" != "0" && "$gpu_id" != "1" ]]; then
        echo "[p11] physical GPUs 2-7 are reserved; P11 may use only GPU0-1" >&2
        exit 2
    fi
    if [[ -n "${SEEN_GPUS[$gpu_id]:-}" ]]; then
        echo "[p11] GPU_IDS contains duplicate physical GPU $gpu_id" >&2
        exit 2
    fi
    SEEN_GPUS[$gpu_id]=1
done
if [[ "$PROFILE" == "preflight" && "$NUM_GPUS" != "2" ]]; then
    echo "[p11] canonical GPU preflight requires exactly 2 GPUs" >&2
    exit 2
fi
if [[ "$PROFILE" == "overfit" && "$NUM_GPUS" != "1" ]]; then
    echo "[p11] canonical overfit gate requires exactly one GPU" >&2
    exit 2
fi
if [[ "$PROFILE" == "screening" && "$NUM_GPUS" != "1" ]]; then
    echo "[p11] idea screening is locked to exactly one GPU" >&2
    exit 2
fi
if [[ "$PROFILE" == "scale_preflight" || "$PROFILE" == "scale_trial" || "$PROFILE" == "resume_smoke" ]]; then
    if [[ "$NUM_GPUS" != "2" || "$GPU_IDS" != "0,1" ]]; then
        echo "[p11] canonical scale gates require exactly physical GPUs 0,1" >&2
        exit 2
    fi
    if [[ "$P11_ARM" != "canonical" && "$P11_ARM" != "direct_mse" ]]; then
        echo "[p11] scale gates allow only canonical Flow-R1 or matched Direct-MSE" >&2
        exit 2
    fi
    if [[ "$BIND_TO_GPU_NUMA" != "1" || "$DDP_COMM_HOOK" != "bf16" || "$DDP_BUCKET_CAP_MB" != "50" ]]; then
        echo "[p11] scale profiles require NUMA binding, BF16 DDP communication, and 50 MB buckets" >&2
        exit 2
    fi
    if nvidia-smi -i "$GPU_IDS" --query-compute-apps=pid --format=csv,noheader,nounits \
        2>/dev/null | awk \
        'BEGIN { found=0 } /^[[:space:]]*[0-9]+[[:space:]]*$/ { found=1 } END { exit found ? 0 : 1 }'; then
        echo "[p11] refusing to share GPU0-1 during a canonical two-GPU run" >&2
        exit 2
    fi
fi
if (( MAX_STEPS <= BENCHMARK_WARMUP_BATCHES )); then
    echo "[p11] MAX_STEPS must exceed BENCHMARK_WARMUP_BATCHES" >&2
    exit 2
fi
case "$RUN_ROOT" in
    ${AMBIT_CKPT_ROOT}/sceneplan_p11/*) ;;
    *)
        echo "[p11] RUN_ROOT must stay under ${AMBIT_CKPT_ROOT}/sceneplan_p11" >&2
        exit 2
        ;;
esac
if [[ ! -x "$PY" || ! -r "$MODEL_CONFIG" || ! -r "$DATASET_CONFIG" ]]; then
    echo "[p11] Python, model config, or dataset config is unreadable" >&2
    exit 1
fi
if [[ "$PROFILE" == "screening" || "$PROFILE" == "scale_preflight" || "$PROFILE" == "scale_trial" || "$PROFILE" == "resume_smoke" ]]; then
    if [[ "$(readlink -f -- "$MODEL_CONFIG")" != "$(readlink -f -- "$REPO/stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/$P11_MODEL_FILE")" ]]; then
        echo "[p11] screening forbids a model-config override" >&2
        exit 2
    fi
    EXPECTED_DATASET="$SCREENING_DATASET"
    if [[ "$PROFILE" == "scale_preflight" || "$PROFILE" == "scale_trial" || "$PROFILE" == "resume_smoke" ]]; then
        EXPECTED_DATASET="$SCALE_V4_DATASET"
    fi
    if [[ "$(readlink -f -- "$DATASET_CONFIG")" != "$(readlink -f -- "$EXPECTED_DATASET")" ]]; then
        echo "[p11] $PROFILE forbids a dataset-config override" >&2
        exit 2
    fi
    if [[ "$TRAINING_SEED" != "42" ]]; then
        echo "[p11] canonical screening/scale training seed is fixed at 42" >&2
        exit 2
    fi
fi
if [[ "$PROFILE" == "screening" ]]; then
    if [[ "$MAX_STEPS" != "10000" || "$CHECKPOINT_EVERY" != "10000" || "$SAVE_TOP_K" != "1" ]]; then
        echo "[p11] screening is pinned to 10k steps and one terminal checkpoint" >&2
        exit 2
    fi
fi
if [[ "$PROFILE" == "scale_preflight" ]]; then
    if [[ "$MAX_STEPS" != "128" || "$BENCHMARK_WARMUP_BATCHES" != "32" || "$SAVE_TOP_K" != "0" ]]; then
        echo "[p11] two-GPU scale preflight is pinned to 128 steps, 32 warmup steps, and no checkpoint" >&2
        exit 2
    fi
fi
if [[ "$PROFILE" == "scale_trial" ]]; then
    if [[ "$MAX_STEPS" != "10000" || "$CHECKPOINT_EVERY" != "10000" || "$SAVE_TOP_K" != "1" ]]; then
        echo "[p11] two-GPU scale trial is pinned to 10k steps and one terminal checkpoint" >&2
        exit 2
    fi
fi
if [[ "$PROFILE" == "resume_smoke" ]]; then
    if [[ "$CHECKPOINT_EVERY" != "8" || "$SAVE_TOP_K" != "1" || "$BENCHMARK_WARMUP_BATCHES" != "2" ]]; then
        echo "[p11] resume smoke is pinned to checkpoint-every=8, save-top-k=1, and two warmup steps" >&2
        exit 2
    fi
    if [[ -z "$CKPT_PATH" && "$MAX_STEPS" != "8" ]]; then
        echo "[p11] fresh resume-smoke stage is pinned to target step 8" >&2
        exit 2
    fi
    if [[ -n "$CKPT_PATH" && "$MAX_STEPS" != "16" ]]; then
        echo "[p11] restored resume-smoke stage is pinned to target step 16" >&2
        exit 2
    fi
fi
if [[ -n "$CKPT_PATH" && -n "$PRETRAINED_CKPT_PATH" ]]; then
    echo "[p11] CKPT_PATH and PRETRAINED_CKPT_PATH are mutually exclusive" >&2
    exit 2
fi
if [[ -n "$PRETRAINED_CKPT_PATH" ]]; then
    echo "[p11-v4] legacy route warm-start is forbidden; matched pilots start from the same pinned Qwen initialization" >&2
    exit 2
fi
if [[ -n "$CKPT_PATH" ]]; then
    if [[ ! -r "$CKPT_PATH" ]]; then
        echo "[p11] CKPT_PATH is unreadable: $CKPT_PATH" >&2
        exit 1
    fi
elif [[ -d "$RUN_ROOT" ]] && [[ -n "$(find "$RUN_ROOT" -mindepth 1 -print -quit)" ]]; then
    echo "[p11] refusing a non-empty fresh-run directory: $RUN_ROOT" >&2
    exit 1
fi

INITIAL_GLOBAL_STEP=0
if [[ -n "$CKPT_PATH" ]]; then
    INITIAL_GLOBAL_STEP="$($PY - "$CKPT_PATH" <<'PY'
import sys
import torch

checkpoint = torch.load(
    sys.argv[1], map_location="cpu", weights_only=False, mmap=True
)
print(int(checkpoint["global_step"]))
PY
)"
fi
if (( MAX_STEPS <= INITIAL_GLOBAL_STEP + BENCHMARK_WARMUP_BATCHES )); then
    echo "[p11] target step leaves no post-resume measured window" >&2
    exit 2
fi

CHECKPOINT_DIR="$RUN_ROOT/checkpoints"
KERNEL_CACHE="${P11_KERNEL_CACHE:-${AMBIT_CKPT_ROOT}/sceneplan_p11/kernel_cache}"
TEMP_PARENT="${P11_TEMP_PARENT:-${AMBIT_CKPT_ROOT}/p11_tmp}"
mkdir -p "$CHECKPOINT_DIR" "$TEMP_PARENT" "$KERNEL_CACHE/torchinductor" "$KERNEL_CACHE/triton"
if [[ -n "${TEMP_DIR:-}" ]]; then
    mkdir -p "$TEMP_DIR"
    P11_OWNS_TEMP_DIR=0
else
    TEMP_DIR="$(mktemp -d "$TEMP_PARENT/p11.XXXXXX")"
    P11_OWNS_TEMP_DIR=1
fi
cleanup_p11_temp() {
    if [[ "$P11_OWNS_TEMP_DIR" == "1" ]]; then
        case "$TEMP_DIR" in
            "$TEMP_PARENT"/p11.*) rm -rf -- "$TEMP_DIR" ;;
            *) echo "[p11] refusing unsafe temp cleanup target: $TEMP_DIR" >&2 ;;
        esac
    fi
}
trap cleanup_p11_temp EXIT
free_kib="$(df -Pk "$RUN_ROOT" | awk 'NR==2 {print $4}')"
required_kib="$((MIN_FREE_DISK_GIB * 1024 * 1024))"
if (( free_kib < required_kib )); then
    echo "[p11] only $((free_kib / 1024 / 1024)) GiB free; need ${MIN_FREE_DISK_GIB} GiB" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export TMPDIR="$TEMP_DIR"
export TMP="$TEMP_DIR"
export TEMP="$TEMP_DIR"
export TORCHINDUCTOR_CACHE_DIR="$KERNEL_CACHE/torchinductor"
export TRITON_CACHE_DIR="$KERNEL_CACHE/triton"

cd "$REPO"
RUN_CONTRACT="$RUN_ROOT/training_launch_contract.json"
run_contract_resume_args=()
if [[ -n "$CKPT_PATH" ]]; then
    RUN_CONTRACT="$RUN_ROOT/training_resume_contract_to_step_${MAX_STEPS}.json"
    run_contract_resume_args+=(--ckpt-path "$CKPT_PATH")
fi
"$PY" scripts/t2a/train/write_sceneplan_p11_run_contract.py \
    --output "$RUN_CONTRACT" \
    --run-name "$RUN_NAME" \
    --profile "$PROFILE" \
    --arm "$P11_ARM" \
    --model-config "$MODEL_CONFIG" \
    --dataset-config "$DATASET_CONFIG" \
    --gpu-ids "$GPU_IDS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --benchmark-warmup-batches "$BENCHMARK_WARMUP_BATCHES" \
    --ddp-bucket-cap-mb "$DDP_BUCKET_CAP_MB" \
    --ddp-comm-hook "$DDP_COMM_HOOK" \
    --bind-to-gpu-numa "$BIND_TO_GPU_NUMA" \
    --max-steps "$MAX_STEPS" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --save-top-k "$SAVE_TOP_K" \
    --seed "$TRAINING_SEED" \
    "${run_contract_resume_args[@]}"
if [[ "$PROFILE" == "screening" || "$PROFILE" == "scale_preflight" || "$PROFILE" == "scale_trial" || "$PROFILE" == "resume_smoke" ]]; then
    "$PY" scripts/t2a/test/audit_sceneplan_p11_lexical_cache.py \
        --threshold 0.85 \
        --output "$RUN_ROOT/lexical_cache_gate.json"
fi
if [[ "$PROFILE" == "screening" || "$PROFILE" == "scale_preflight" ]]; then
    "$PY" scripts/t2a/test/validate_sceneplan_p11_v4_screening.py \
        --v4-dataset-config "$REPO/stable_audio_tools/configs/dataset_configs/sceneplan_p11_trial30k_matched_screening_v1_transfusion_cot_v4_reliable_asr.json" \
        --d0-dataset-config "$REPO/stable_audio_tools/configs/dataset_configs/sceneplan_p11_trial30k_matched_screening_v1_discrete_d0.json" \
        --model-config "$REPO/stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/qwen35_0p8b_sceneplan_p11_reliable_asr_assembler_v2.json" \
        --runtime-scenes 2 \
        --output "$RUN_ROOT/screening_data_gate.json"
fi
if [[ "$PROFILE" == "scale_trial" ]]; then
    "$PY" scripts/t2a/test/validate_sceneplan_p11_v4_curriculum.py \
        --curriculum "$SCALE_CURRICULUM" \
        --v4-dataset-config "$SCALE_V4_DATASET" \
        --d0-dataset-config "$SCALE_D0_DATASET" \
        --model-config "$MODEL_CONFIG" \
        --runtime-scenes 2 \
        --output "$RUN_ROOT/scale_trial_data_gate.json"
fi
if [[ "$P11_GRAPH_FAMILY" == "audio_aware" ]]; then
    "$PY" -m pytest -q tests/test_sceneplan_p11_audio_aware_contract.py
    "$PY" scripts/t2a/test/validate_sceneplan_p11_v4_contract.py \
        --base-scenes "$VALIDATOR_BASE_SCENES" \
        --output "$RUN_ROOT/audio_aware_contract_gate.json"
    "$PY" scripts/t2a/test/validate_sceneplan_p11_v4_sequence_budget.py \
        --model-config "$MODEL_CONFIG" \
        --dataset-config "$DATASET_CONFIG" \
        --num-workers 0 \
        --output "$RUN_ROOT/sequence_budget_gate.json"
elif [[ "$P11_GRAPH_FAMILY" == "v4" ]]; then
    if [[ "$PROFILE" == "scale_preflight" || "$PROFILE" == "scale_trial" || "$PROFILE" == "resume_smoke" ]]; then
        "$PY" scripts/t2a/test/validate_sceneplan_p11_v4_matched_arms.py \
            --output "$RUN_ROOT/matched_arm_config_gate.json"
    fi
    SEQUENCE_BUDGET_WORKERS=0
    if [[ "$PROFILE" == "scale_preflight" || "$PROFILE" == "scale_trial" || "$PROFILE" == "resume_smoke" ]]; then
        # This remains an exact full-row scan. Workers only parallelize the
        # immutable runtime projection/token counting so GPUs do not wait on a
        # single CPU core before a scale run.
        SEQUENCE_BUDGET_WORKERS=8
        "$PY" scripts/t2a/test/audit_sceneplan_p11_v4_ddp_sampler.py \
            --dataset-config "$DATASET_CONFIG" \
            --world-size 8 \
            --local-batch-size "$BATCH_SIZE" \
            --expect-ordering-contract \
                p11_v4_ddp8_rank_balanced_pair_aware_batch8_v7 \
            --output "$RUN_ROOT/ddp_sampler_gate.json"
    fi
    "$PY" scripts/t2a/test/validate_sceneplan_p11_v4_contract.py \
        --base-scenes "$VALIDATOR_BASE_SCENES" \
        --output "$RUN_ROOT/v4_contract_gate.json"
    "$PY" scripts/t2a/test/validate_sceneplan_p11_v4_sequence_budget.py \
        --model-config "$MODEL_CONFIG" \
        --dataset-config "$DATASET_CONFIG" \
        --num-workers "$SEQUENCE_BUDGET_WORKERS" \
        --output "$RUN_ROOT/sequence_budget_gate.json"
    "$PY" scripts/t2a/test/validate_sceneplan_p11_v4_control_direction.py \
        --model-config "$MODEL_CONFIG" \
        --dataset-config "$DATASET_CONFIG" \
        --batch-size "$BATCH_SIZE" \
        --output "$RUN_ROOT/control_direction_cpu_gate.json"
    "$PY" scripts/t2a/test/validate_sceneplan_p11_v4_delta_owner.py \
        --model-config "$MODEL_CONFIG" \
        --dataset-config "$DATASET_CONFIG" \
        --output "$RUN_ROOT/delta_owner_cpu_gate.json"
else
    "$PY" scripts/t2a/test/validate_sceneplan_p11_manifest.py \
        --model-config "$MODEL_CONFIG" \
        --dataset-config "$DATASET_CONFIG" \
        --base-scenes "$VALIDATOR_BASE_SCENES" \
        --grammar-scenes "$VALIDATOR_BASE_SCENES" \
        --scope full
fi
if [[ "$PROFILE" == "preflight" || "$PROFILE" == "screening" || "$PROFILE" == "scale_preflight" || "$PROFILE" == "scale_trial" || "$PROFILE" == "resume_smoke" ]]; then
    if [[ "$P11_GRAPH_FAMILY" == "audio_aware" ]]; then
        CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" \
            "$PY" scripts/t2a/test/smoke_sceneplan_p11_v4_graph.py \
            --model-config "$MODEL_CONFIG" \
            --dataset-config "$DATASET_CONFIG" \
            --device cuda:0 \
            --batch-size 8 \
            --output "$RUN_ROOT/audio_aware_real_graph_smoke.json"
    elif [[ "$P11_GRAPH_FAMILY" == "v4" ]]; then
        graph_smoke_args=()
        graph_smoke_output="$RUN_ROOT/screening_real_graph_smoke.json"
        if [[ "$PROFILE" == "preflight" || "$PROFILE" == "scale_preflight" || "$PROFILE" == "scale_trial" || "$PROFILE" == "resume_smoke" ]]; then
            graph_smoke_args+=(--require-control-direction-pair)
            graph_smoke_output="$RUN_ROOT/control_direction_real_graph_smoke.json"
        fi
        CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" \
            "$PY" scripts/t2a/test/smoke_sceneplan_p11_v4_graph.py \
            --model-config "$MODEL_CONFIG" \
            --dataset-config "$DATASET_CONFIG" \
            --device cuda:0 \
            --batch-size 8 \
            --output "$graph_smoke_output" \
            "${graph_smoke_args[@]}"
    else
        d0_smoke_output="$RUN_ROOT/screening_cpu_graph_smoke.json"
        if [[ "$PROFILE" == "preflight" ]]; then
            d0_smoke_output="$RUN_ROOT/preflight_cpu_graph_smoke.json"
        fi
        "$PY" scripts/t2a/test/smoke_sceneplan_p11.py \
            --model-config "$MODEL_CONFIG" \
            --dataset-config "$DATASET_CONFIG" \
            --batch-size 8 \
            --output "$d0_smoke_output"
    fi
fi

echo "[p11] profile=$PROFILE arm=$P11_ARM GPUs=$GPU_IDS batch/GPU=8 global_batch=$((8 * NUM_GPUS))"
echo "[p11] max_steps=$MAX_STEPS seed=$TRAINING_SEED run_root=$RUN_ROOT save_top_k=$SAVE_TOP_K"
echo "[p11] NUMA=$BIND_TO_GPU_NUMA strategy=ddp_static bucket_mb=$DDP_BUCKET_CAP_MB comm_hook=$DDP_COMM_HOOK warmup=$BENCHMARK_WARMUP_BATCHES"
if [[ -n "$PRETRAINED_CKPT_PATH" ]]; then
    echo "[p11] weights-only warm-start=$PRETRAINED_CKPT_PATH source_weights=$PRETRAINED_ROUTE_WEIGHTS trainer_state=reset"
fi

extra_args=()
if [[ "$PROFILE" == "overfit" ]]; then
    extra_args+=(--overfit-batches "$OVERFIT_BATCHES")
fi
if [[ -n "$CKPT_PATH" ]]; then
    extra_args+=(--ckpt-path "$CKPT_PATH")
fi
if [[ -n "$PRETRAINED_CKPT_PATH" ]]; then
    extra_args+=(
        --pretrained-ckpt-path "$PRETRAINED_CKPT_PATH"
        --pretrained-route-weights "$PRETRAINED_ROUTE_WEIGHTS"
    )
fi
strategy="auto"
if (( NUM_GPUS > 1 )); then
    strategy="ddp_static"
fi
numa_args=()
if [[ "$BIND_TO_GPU_NUMA" == "1" ]]; then
    numa_args+=(--bind-to-gpu-numa)
fi

set +e
"$PY" -u train.py \
    --name "$RUN_NAME" \
    --model-config "$MODEL_CONFIG" \
    --dataset-config "$DATASET_CONFIG" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --num-gpus "$NUM_GPUS" \
    --strategy "$strategy" \
    --ddp-bucket-cap-mb "$DDP_BUCKET_CAP_MB" \
    --ddp-comm-hook "$DDP_COMM_HOOK" \
    --ddp-timeout-min "$DDP_TIMEOUT_MIN" \
    --precision bf16-mixed \
    --gradient-clip-val 1.0 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --save-dir "$RUN_ROOT" \
    --logger none \
    --max-steps "$MAX_STEPS" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --save-top-k "$SAVE_TOP_K" \
    --min-free-disk-gib "$MIN_FREE_DISK_GIB" \
    --benchmark \
    --benchmark-warmup-batches "$BENCHMARK_WARMUP_BATCHES" \
    --training-gate \
    --training-gate-window "$GATE_WINDOW" \
    --training-gate-max-loss-ratio "$GATE_RATIO" \
    --training-gate-gradient-every 1 \
    --pre-encoded \
    --seed "$TRAINING_SEED" \
    --temp-dir "$TEMP_DIR" \
    "${numa_args[@]}" \
    "${extra_args[@]}" \
    2>&1 | tee -a "$RUN_ROOT/train.log"
train_status="${PIPESTATUS[0]}"
set -e
if (( train_status != 0 )); then
    echo "[p11] training exited with status $train_status" >&2
    exit "$train_status"
fi

if [[ "$PROFILE" == "preflight" || "$PROFILE" == "scale_preflight" || "$PROFILE" == "scale_trial" || "$PROFILE" == "resume_smoke" ]]; then
    scale_utilization_args=()
    if [[ "$PROFILE" == "scale_preflight" || "$PROFILE" == "scale_trial" ]]; then
        # The matched single-GPU Flow screen sustained 9.88 samples/s.  Two
        # PCIe/SYS-connected GPUs must deliver useful scaling,
        # keep every rank above 45% mean SM utilization, and materialize the
        # full batch-8 training graph rather than a tiny accidental route.
        scale_utilization_args+=(
            --min-global-loader-samples-per-second 14
            --min-rank-mean-gpu-utilization-percent 45
            --min-peak-allocated-gib 10
        )
    fi
    PREFLIGHT_REPORT="$RUN_ROOT/${PROFILE}_report.json"
    if [[ "$PROFILE" == "resume_smoke" ]]; then
        PREFLIGHT_REPORT="$RUN_ROOT/resume_smoke_${INITIAL_GLOBAL_STEP}_to_${MAX_STEPS}_report.json"
    fi
    "$PY" scripts/t2a/test/validate_sceneplan_p11_gpu_preflight.py \
        --log "$RUN_ROOT/train.log" \
        --expected-world-size "$NUM_GPUS" \
        --expected-batch-size "$BATCH_SIZE" \
        --expected-arm "$P11_EXPECTED_ARM" \
        --expected-optimizer-steps "$MAX_STEPS" \
        --expected-initial-global-step "$INITIAL_GLOBAL_STEP" \
        --expected-measured-steps "$((MAX_STEPS - INITIAL_GLOBAL_STEP - BENCHMARK_WARMUP_BATCHES))" \
        "${scale_utilization_args[@]}" \
        --output "$PREFLIGHT_REPORT"
fi

# Lightning may write the latest numbered checkpoint and last.ckpt separately.
# Collapse them only when their full serialized bytes are identical; resumed
# runs can carry different callback metadata even when model weights match.
shopt -s nullglob
numbered_checkpoints=("$CHECKPOINT_DIR"/epoch=*-step=*.ckpt)
last_checkpoint="$CHECKPOINT_DIR/last.ckpt"
if (( ${#numbered_checkpoints[@]} == 1 )) \
    && [[ -f "$last_checkpoint" ]] \
    && cmp -s -- "${numbered_checkpoints[0]}" "$last_checkpoint"; then
    dedup_link="$CHECKPOINT_DIR/.last.ckpt.dedup.$$"
    if ln -- "${numbered_checkpoints[0]}" "$dedup_link" \
        && mv -f -- "$dedup_link" "$last_checkpoint"; then
        echo "[p11] deduplicated numbered checkpoint and last.ckpt via hard link"
    else
        rm -f -- "$dedup_link"
        echo "[p11] warning: checkpoint hard-link deduplication failed" >&2
    fi
fi

if [[ "$P11_AUTO_EVAL" == "1" ]]; then
    echo "[p11-v4] learned-output evaluation is intentionally separate from training and is not yet promotable" >&2
    exit 2
fi
