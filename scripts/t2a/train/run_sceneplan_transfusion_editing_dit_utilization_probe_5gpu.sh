#!/usr/bin/env bash
set -euo pipefail

REPO="${P10_REPO:-${AMBIT_CKPT_ROOT}/stable-audio-tools-workspace}"
ROOT="${EDITING_DATA_ROOT:-${AMBIT_DATA_ROOT}/sceneplan_transfusion_editing_v1}"
MODEL_CONFIG="${MODEL_CONFIG:-$REPO/stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/qwen35_0p8b_300m_sceneplan_transfusion_editing_dit_full_v1.json}"
CHECKPOINT="${PRETRAINED_CKPT:-${AMBIT_CKPT_ROOT}/dit/sceneplan_dit_v11_semantic_v2_protected_resume_150k/checkpoints/epoch=48-step=150000.ckpt}"
SOURCE_SPLIT="${SOURCE_SPLIT:-validation}"
BATCH_SIZE="${BATCH_SIZE:-72}"
LONG_BATCH_SIZE="${LONG_BATCH_SIZE:-48}"
NUM_WORKERS="${NUM_WORKERS:-12}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[ambit] set CUDA_VISIBLE_DEVICES to the GPUs for this job" >&2
    exit 2
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID

"$REPO/.venv/bin/python" \
    "$REPO/scripts/t2a/train/prepare_sceneplan_transfusion_editing_utilization_probe.py" \
    --root "$ROOT" \
    --model-config "$MODEL_CONFIG" \
    --source-split "$SOURCE_SPLIT" \
    --short-batch-size "$BATCH_SIZE" \
    --long-batch-size "$LONG_BATCH_SIZE"

STEM="${SOURCE_SPLIT}_as_train_b${BATCH_SIZE}_l${LONG_BATCH_SIZE}"
DATASET_CONFIG="$ROOT/contracts/utilization_probe/$STEM.json"
AUDIT="$ROOT/contracts/utilization_probe/$STEM.audit.json"
"$REPO/.venv/bin/python" - "$AUDIT" "$BATCH_SIZE" "$LONG_BATCH_SIZE" <<'PY'
import json,sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).resolve(strict=True).read_text(encoding="utf-8"))
if not (
    value.get("status")=="PASS"
    and value.get("per_rank_short_batch_size")==int(sys.argv[2])
    and value.get("per_rank_long_batch_size")==int(sys.argv[3])
    and value.get("old_sceneplan_input") is False
    and value.get("editing_dit_frame_channels")==384
):
    raise SystemExit("Editing DiT utilization probe contract is not PASS")
PY

export RUN_NAME="${RUN_NAME:-sceneplan_transfusion_editing_dit_b${BATCH_SIZE}_l${LONG_BATCH_SIZE}_5gpu_probe}"
export RUN_LABEL="sceneplan-transfusion-editing-dit-utilization"
export RUN_CATEGORY=benchmarks
export RUN_ROOT="${RUN_ROOT:-${AMBIT_CKPT_ROOT}/transfusion_editing/benchmarks/$RUN_NAME}"
export MODEL_CONFIG DATASET_CONFIG
export VAL_DATASET_CONFIG=""
export DISABLE_VALIDATION=1
export LOAD_PRETRANSFORM=0
export PRETRAINED_CKPT="$CHECKPOINT"
export PRETRAINED_ROUTE_WEIGHTS=ema
export SAT_PRETRAINED_ROUTE_EXPECTATION_JSON='{"loaded":612,"target_total":612,"shape_mismatches":4,"shape_mismatch_targets":2,"partial_expansions":2,"semantic_role_expansions":0,"missing":0,"shape_mismatch_target_names":["model.model.preprocess_conv.weight","model.model.transformer.project_in.weight"],"partial_expansion_targets":["model.model.preprocess_conv.weight","model.model.transformer.project_in.weight"],"semantic_role_expansion_targets":[],"missing_names":[]}'
export PRETRAINED_MODALITY_CKPT=""
export REQUIRE_NO_PRETRAINED=0

export NUM_GPUS=5
export BATCH_SIZE NUM_WORKERS
export ACCUM_BATCHES=1
export TRAINING_STRATEGY=ddp_static
export DDP_BUCKET_CAP_MB="${DDP_BUCKET_CAP_MB:-50}"
export DDP_COMM_HOOK=none
export BIND_TO_GPU_NUMA=0
export RANK_AWARE_TRAINING_SEED=1
export GRADIENT_CLIP_VAL=1.0
export ENABLE_TORCH_COMPILE=0

export BENCHMARK=1
export BENCHMARK_WARMUP_BATCHES="${BENCHMARK_WARMUP_BATCHES:-2}"
export MAX_STEPS="${MAX_STEPS:-22}"
export CHECKPOINT_EVERY=10000
export SAVE_TOP_K=0
export OVERFIT_BATCHES=0
export VAL_EVERY=-1
export LIMIT_VAL_BATCHES=-1
export LOGGER="${LOGGER:-none}"
export TRAINING_GATE=1
export TRAINING_GATE_WINDOW=20
export TRAINING_GATE_MAX_LOSS_RATIO=-1
export TRAINING_GATE_GRADIENT_EVERY=5
export MIN_FREE_DISK_GIB=30
export MIN_ROOT_FREE_GIB=10
export TEMP_DIR="${TEMP_DIR:-/dev/shm/spedit_dit_utilization_b${BATCH_SIZE}_l${LONG_BATCH_SIZE}}"

exec "$REPO/scripts/t2a/train/run_t2a_common_8gpu.sh"
