#!/usr/bin/env bash
set -euo pipefail

REPO="${P10_REPO:-/mnt/sdc/stable-audio-tools-workspace}"
ROOT="${EDITING_DATA_ROOT:-/mnt/sdb/audio_dataset/sceneplan_transfusion_editing_v1}"
MODEL_CONFIG="${MODEL_CONFIG:-$REPO/stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/qwen35_0p8b_300m_sceneplan_transfusion_editing_dit_full_v1.json}"
DATASET_CONFIG="${DATASET_CONFIG:-$ROOT/contracts/full_training/train_dataset.json}"
VAL_DATASET_CONFIG="${VAL_DATASET_CONFIG:-$ROOT/contracts/full_training/validation_dataset.json}"
CHECKPOINT="${PRETRAINED_CKPT:-/mnt/sdc/ckpts/dit/sceneplan_dit_v11_semantic_v2_protected_resume_150k/checkpoints/epoch=48-step=150000.ckpt}"
PREFLIGHT="$ROOT/contracts/full_training/PREFLIGHT.json"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES// /}" != "3,4,5,6,7" ]]; then
    echo "[editing-dit-full] only physical GPUs 3,4,5,6,7 are allowed" >&2
    exit 2
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3,4,5,6,7

# Prevent two direct/full-chain invocations from both deciding the run is fresh
# before either has published its first resumable checkpoint.  The descriptor
# intentionally survives the later exec into Joint selection/training.
LOCK_ROOT="$ROOT/materialized/locks"
mkdir -p "$LOCK_ROOT"
exec 7>"$LOCK_ROOT/training-chain.lock"
if ! flock -n 7; then
    echo "[editing-dit-full] another formal Editing training chain is active" >&2
    exit 1
fi
export EDITING_TRAIN_CHAIN_LOCK_FD=7
export EDITING_TRAIN_CHAIN_LOCK_PATH="$LOCK_ROOT/training-chain.lock"

"$REPO/.venv/bin/python" "$REPO/scripts/t2a/train/prepare_sceneplan_transfusion_editing_full.py" \
    --root "$ROOT" --model-config "$MODEL_CONFIG" --p10-checkpoint "$CHECKPOINT"

"$REPO/.venv/bin/python" - "$PREFLIGHT" <<'PY'
import json,sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not (
    value.get("schema")=="sceneplan_transfusion_editing_full_training_preflight"
    and int(value.get("schema_version",-1))==2
    and value.get("status")=="PASS"
    and value.get("latest_route",{}).get("editing_ar_inputs")
        ==["source_foa_latent","raw_edit_request"]
    and value.get("latest_route",{}).get("old_sceneplan_input") is False
    and value.get("latest_route",{}).get("editing_dit_frame_channels")==384
    and value.get("latest_route",{}).get(
        "editing_dit_clean_source_always_present"
    ) is True
    and value.get("latest_route",{}).get("editing_ar_required_max_plan_tokens")==1024
    and value.get("latest_route",{}).get("editing_evaluation_max_plan_tokens")==512
    and 0 < value.get("latest_route",{}).get("max_plan_tokens_observed",0) <= 512
    and value.get("train_plan_token_audit",{}).get(
        "evaluation_budget_covers_all_rows"
    ) is True
    and value.get("train_plan_token_audit",{}).get(
        "rows_over_evaluation_budget"
    ) == 0
    and value.get("split_disjointness",{}).get("status")=="PASS"
    and all(value.get("split_disjointness",{}).get("checks",{}).values())
    and value.get("editing_dit_utilization_defaults")=={
        "short_batch_size_per_gpu":72,
        "long_batch_size_per_gpu":48,
        "num_workers_per_rank":12,
        "world_size":5,
    }
    and value.get("editing_dit_utilization_benchmark",{}).get("status")=="PASS"
    and value.get("editing_dit_utilization_benchmark",{}).get(
        "editing_dit_trainable_parameters"
    )==319314304
    and value.get("editing_dit_utilization_benchmark",{}).get(
        "selected",{}
    ).get("short_batch_size_per_gpu")==72
    and value.get("editing_dit_utilization_benchmark",{}).get(
        "selected",{}
    ).get("long_batch_size_per_gpu")==48
    and value.get("editing_dit_utilization_benchmark",{}).get(
        "selected",{}
    ).get("num_workers_per_rank")==12
    and all(
        value.get("indices",{}).get(split,{}).get(
            f"{role}_latent_shards_exhaustively_verified"
        ) is True
        and value.get("indices",{}).get(split,{}).get(f"{role}_pair_rows")==rows
        and value.get("indices",{}).get(split,{}).get(
            f"{role}_latent_shards",0
        )>0
        and len(value.get("indices",{}).get(split,{}).get(
            f"{role}_latent_shard_inventory_sha256", ""
        ))==64
        for role in ("source","target")
        for split,rows in (("train",1000000),("validation",20000))
    )
):
    raise SystemExit("full Editing preflight is not latest-route PASS")
PY

export RUN_NAME="${RUN_NAME:-sceneplan_transfusion_editing_dit_full_seed42_v1}"
export RUN_LABEL="sceneplan-transfusion-editing-dit-full"
export RUN_CATEGORY=mainline
export RUN_ROOT="${RUN_ROOT:-/mnt/sdb/model_archives/transfusion_editing/mainline/$RUN_NAME}"
export MODEL_CONFIG DATASET_CONFIG VAL_DATASET_CONFIG
export LOAD_PRETRANSFORM=0
export PRETRAINED_CKPT="$CHECKPOINT"
export PRETRAINED_ROUTE_WEIGHTS=ema
export SAT_PRETRAINED_ROUTE_EXPECTATION_JSON='{"loaded":612,"target_total":612,"shape_mismatches":4,"shape_mismatch_targets":2,"partial_expansions":2,"semantic_role_expansions":0,"missing":0,"shape_mismatch_target_names":["model.model.preprocess_conv.weight","model.model.transformer.project_in.weight"],"partial_expansion_targets":["model.model.preprocess_conv.weight","model.model.transformer.project_in.weight"],"semantic_role_expansion_targets":[],"missing_names":[]}'
export PRETRAINED_MODALITY_CKPT=""
export REQUIRE_NO_PRETRAINED=0

export NUM_GPUS=5
# Measured on physical GPUs 3-7 with the real 432/648-frame Editing dataset.
# 72/48 with 12 workers/rank sustains 74.1k latent positions/s and reserves at
# most 39.56 GiB/GPU.  80/53 reserves 42.67 GiB but is about 32% slower, while
# 16 workers/rank is about 1% slower than 12 at 72/48.
export BATCH_SIZE="${BATCH_SIZE:-72}"
export NUM_WORKERS="${NUM_WORKERS:-12}"
if [[ "$BATCH_SIZE" != "72" || "$NUM_WORKERS" != "12" ]]; then
    echo "[editing-dit-full] formal utilization contract requires BATCH_SIZE=72 and NUM_WORKERS=12" >&2
    exit 2
fi
export ACCUM_BATCHES=1
export TRAINING_STRATEGY=ddp_static
export DDP_COMM_HOOK=none
export BIND_TO_GPU_NUMA=0
export RANK_AWARE_TRAINING_SEED=1
export GRADIENT_CLIP_VAL=1.0
export OVERFIT_BATCHES=0

export MAX_STEPS="${MAX_STEPS:-30000}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
if [[ "$MAX_STEPS" != "30000" || "$CHECKPOINT_EVERY" != "5000" ]]; then
    echo "[editing-dit-full] formal run requires 30,000 steps and 5,000-step checkpoints" >&2
    exit 2
fi
# Lightning 2.5 rejects monitor=None with save_top_k=2.  More importantly,
# joint Editing must not inherit an unvalidated last.ckpt: retain every formal
# 5,000-step EMA candidate so the frozen-20K selector can choose among all six.
if [[ -n "${SAVE_TOP_K:-}" && "$SAVE_TOP_K" != "-1" ]]; then
    echo "[editing-dit-full] formal checkpoint selection requires SAVE_TOP_K=-1" >&2
    exit 2
fi
export SAVE_TOP_K=-1
export VAL_EVERY="${VAL_EVERY:-1000}"
export LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-64}"
export LOGGER="${LOGGER:-none}"
if [[ "$VAL_EVERY" != "1000" || "$LIMIT_VAL_BATCHES" != "64" || "$LOGGER" != "none" ]]; then
    echo "[editing-dit-full] formal monitoring requires VAL_EVERY=1000 LIMIT_VAL_BATCHES=64 LOGGER=none" >&2
    exit 2
fi
export SAT_EDITING_VALIDATION_JSON_LOG=1
export TRAINING_GATE=1
export TRAINING_GATE_WINDOW=20
export TRAINING_GATE_MAX_LOSS_RATIO=-1
export TRAINING_GATE_GRADIENT_EVERY=100
export MIN_FREE_DISK_GIB=100
export MIN_ROOT_FREE_GIB=10
export TEMP_DIR="${TEMP_DIR:-/dev/shm/spedit_dit_full}"
export TRAINING_SEED="${TRAINING_SEED:-42}"
if [[ "$TRAINING_SEED" != "42" ]]; then
    echo "[editing-dit-full] formal lineage requires TRAINING_SEED=42" >&2
    exit 2
fi

# Freeze data, P10, schedule, and implementation identities before the first
# optimizer step.  The generic trainer embeds this only when the formal
# Editing launcher explicitly exports the contract path.
TRAIN_RUN_CONTRACT="$RUN_ROOT/TRAIN_RUN_CONTRACT.json"
"$REPO/.venv/bin/python" \
    "$REPO/scripts/t2a/train/sceneplan_transfusion_editing_dit_run_contract.py" \
    prepare \
    --run-dir "$RUN_ROOT" \
    --preflight "$PREFLIGHT" \
    --model-config "$MODEL_CONFIG" \
    --train-dataset-config "$DATASET_CONFIG" \
    --validation-dataset-config "$VAL_DATASET_CONFIG" \
    --p10-checkpoint "$CHECKPOINT" \
    --max-steps "$MAX_STEPS" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --batch-size "$BATCH_SIZE" \
    --long-batch-size 48 \
    --num-workers "$NUM_WORKERS" \
    --training-seed "$TRAINING_SEED" >/dev/null
export SAT_EDITING_RUN_CONTRACT_PATH="$TRAIN_RUN_CONTRACT"

# A resume is accepted only when its checkpoint embeds the exact frozen
# contract.  This closes the shape-compatible foreign-last.ckpt escape hatch.
mapfile -t resume_resolution < <(
    "$REPO/.venv/bin/python" \
        "$REPO/scripts/t2a/train/sceneplan_transfusion_editing_dit_run_contract.py" \
        resolve-resume --contract "$TRAIN_RUN_CONTRACT"
)
if [[ "${#resume_resolution[@]}" -ne 2 ]]; then
    echo "[editing-dit-full] invalid lineage resume resolution" >&2
    exit 1
fi
case "${resume_resolution[0]}" in
    COMPLETE)
        echo "[editing-dit-full] verified six-candidate inventory; skipping zero-length fit"
        ;;
    FRESH)
        "$REPO/scripts/t2a/train/run_t2a_common_8gpu.sh"
        ;;
    RESUME)
        if [[ -z "${resume_resolution[1]}" ]]; then
            echo "[editing-dit-full] lineage resolver returned an empty resume path" >&2
            exit 1
        fi
        RESUME_CKPT="${resume_resolution[1]}" \
            "$REPO/scripts/t2a/train/run_t2a_common_8gpu.sh"
        ;;
    *)
        echo "[editing-dit-full] unknown lineage action: ${resume_resolution[0]}" >&2
        exit 1
        ;;
esac

# Lightning writes the interval candidate and ``last.ckpt`` in two independent
# ``torch.save`` calls.  The payloads can represent the same optimizer step but
# still have different bytes (including different ModelCheckpoint callback
# state and zip member roots).  Publish ``last.ckpt`` as an atomic byte-for-byte
# copy of the canonical 30K candidate so it is an unambiguous resume pointer,
# rather than weakening the selector's hash binding.
terminal_matches=("$RUN_ROOT"/checkpoints/epoch=*-step=30000.ckpt)
if [[ "${#terminal_matches[@]}" -ne 1 || ! -f "${terminal_matches[0]}" ]]; then
    echo "[editing-dit-full] training did not publish one canonical 30K candidate" >&2
    exit 1
fi
if [[ -e "$RUN_ROOT/checkpoints/last.ckpt" ]] \
    && ! cmp -s -- "${terminal_matches[0]}" "$RUN_ROOT/checkpoints/last.ckpt"; then
    quarantine_dir="$RUN_ROOT/checkpoints/recovery-quarantine-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    mkdir -p "$quarantine_dir"
    mv -- "$RUN_ROOT/checkpoints/last.ckpt" "$quarantine_dir/last.ckpt"
    printf '%s\n' \
        "reason=canonicalize_completed_last_pointer" \
        "preserved_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$quarantine_dir/RECOVERY.txt"
fi
canonical_last_tmp="$RUN_ROOT/checkpoints/.last.ckpt.canonicalize.$$"
cp --reflink=auto --preserve=mode,timestamps -- \
    "${terminal_matches[0]}" "$canonical_last_tmp"
mv -f -- "$canonical_last_tmp" "$RUN_ROOT/checkpoints/last.ckpt"
unset SAT_EDITING_RUN_CONTRACT_PATH

# Promotion is part of the formal DiT stage.  The selector evaluates the EMA
# route and conditioner on every frozen validation row using all five Editing
# GPUs; a failed source-dependence/P10-improvement gate stops before joint AR.
"$REPO/scripts/t2a/eval/run_sceneplan_transfusion_editing_dit_checkpoint_selection_5gpu.sh"
DIT_SELECTION_OUTPUT="${SELECTION_OUTPUT:-$RUN_ROOT/evaluation/validation_20k_checkpoint_selection/SELECTED.json}"

# RF fit is necessary but cannot demonstrate sampled audio editing quality.
# Require actual GT-plan/source-FOA sampling, decoding and reference ablations
# before spending on semantic caches or the shared AR/RF continuation.
GT_AUDIO_OUTPUT="${GT_AUDIO_OUTPUT:-$RUN_ROOT/evaluation/validation_1k_gt_audio}"
env RUN_ROOT="$RUN_ROOT" DIT_CHECKPOINT_SELECTION="$DIT_SELECTION_OUTPUT" \
    GT_AUDIO_OUTPUT="$GT_AUDIO_OUTPUT" \
    "$REPO/scripts/t2a/eval/run_sceneplan_transfusion_editing_gt_audio_5gpu.sh"

# Editing AR's optional semantic branch is trained from frozen, source-only
# M2D-CLAP vectors.  Build those after DiT promotion so the same five GPUs are
# never oversubscribed.  Existing complete caches are immutable and reused;
# durable partials resume only after row/hash/asset validation, while an
# incomplete final publication still fails closed.
M2D_CACHE_ROOT="${EDITING_DATA_ROOT:-/mnt/sdb/audio_dataset/sceneplan_transfusion_editing_v1}/semantic_cache/m2d_clap_v2"
M2D_VALIDATION_CACHE="$M2D_CACHE_ROOT/validation_full/m2d-clap-validation-20000.sqlite"
M2D_TRAIN_CACHE="$M2D_CACHE_ROOT/train_full/m2d-clap-train-1000000.sqlite"
if [[ ! -r "$M2D_VALIDATION_CACHE" || ! -r "$M2D_VALIDATION_CACHE.frozen.json" ]]; then
    "$REPO/scripts/t2a/data/run_sceneplan_transfusion_editing_m2d_clap_cache_5gpu.sh" validation
fi
if [[ ! -r "$M2D_TRAIN_CACHE" || ! -r "$M2D_TRAIN_CACHE.frozen.json" ]]; then
    "$REPO/scripts/t2a/data/run_sceneplan_transfusion_editing_m2d_clap_cache_5gpu.sh" train
fi

# The shared AR/RF continuation can start only after SELECTED.json exists and
# revalidates the exact EMA candidate.  Its own launcher has no last.ckpt
# fallback and will perform a second independent joint promotion gate.
exec env \
    BASE_DIT_RUN="$RUN_ROOT" \
    DIT_CHECKPOINT_SELECTION="$DIT_SELECTION_OUTPUT" \
    DIT_GT_AUDIO_GATE="$GT_AUDIO_OUTPUT/GATE.json" \
    "$REPO/scripts/t2a/train/run_sceneplan_transfusion_editing_ar_joint_full_5gpu.sh"
