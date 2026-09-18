#!/usr/bin/env bash
set -euo pipefail

REPO="${P10_REPO:-.}"
PY="$REPO/.venv/bin/python"
MODEL_CONFIG="${MODEL_CONFIG:-$REPO/stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/qwen35_0p8b_300m_sceneplan_transfusion_editing_dit_v1.json}"
DATASET_CONFIG="${DATASET_CONFIG:-$REPO/stable_audio_tools/configs/dataset_configs/sceneplan_transfusion_editing_v1_overfit10.json}"
CHECKPOINT="${PRETRAINED_CKPT:-${AMBIT_CKPT_ROOT}/dit/sceneplan_dit_v11_semantic_v2_protected_resume_150k/checkpoints/epoch=48-step=150000.ckpt}"
AUDIT="${AMBIT_DATA_ROOT}/sceneplan_transfusion_editing_v1_pilot/audits/train_pilot_v1.json"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "3" ]]; then
    echo "[editing-dit-overfit] CUDA_VISIBLE_DEVICES must be unset or physical GPU 3" >&2
    exit 2
fi

"$PY" - "$MODEL_CONFIG" "$DATASET_CONFIG" "$CHECKPOINT" "$AUDIT" <<'PY'
import hashlib,json,sys
from pathlib import Path
from stable_audio_tools.configuration import load_config,validate_training_configs

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()

model_path,data_path,checkpoint_path,audit_path=map(lambda value:Path(value).resolve(strict=True),sys.argv[1:])
model=load_config(model_path)
data=load_config(data_path)
validate_training_configs(model,data)
audit=json.loads(audit_path.read_text(encoding="utf-8"))
diffusion=model["model"]["diffusion"]
if not (
    sha256(checkpoint_path)=="be8c90cd1434bd71f73951531175c3674ff0f3173d5db591e2e1c476152ff59e"
    and model.get("_source_checkpoint_sha256")==sha256(checkpoint_path)
    and diffusion["input_concat_ids"]==["sceneplan_44","source_foa_latent"]
    and diffusion["config"]["input_concat_dim"]==320
    and model["model"]["conditioning"]["pre_encoded_keys"]==["source_foa_latent"]
    and model["training"]["pre_encoded"] is True
    and model["training"]["sceneplan_speech_active_loss_weight"]==1.0
    and model["training"]["sceneplan_sound_temporal_difference_loss"]["enabled"] is False
    and data["dataset_type"]=="sceneplan_transfusion_editing_preencoded"
    and data["expected_num_samples"]==10
    and data["verify_tensor_hashes_on_access"] is True
    and audit.get("status")=="pass"
    and audit.get("rows")==10
    and audit.get("source_parity_verified_rows")==10
    and audit.get("source_target_exact_equal_rows")==0
    and audit.get("unchanged_stem_checks",{}).get("unavailable")==0
):
    raise SystemExit("Editing-DiT overfit contract failed")
PY

export RUN_NAME="${RUN_NAME:-sceneplan_transfusion_editing_dit_overfit10_seed42_v1}"
export RUN_LABEL="sceneplan-transfusion-editing-dit-overfit10"
export RUN_CATEGORY=pilots
export RUN_ROOT="${RUN_ROOT:-${AMBIT_CKPT_ROOT}/transfusion_editing/pilots/$RUN_NAME}"
export MODEL_CONFIG DATASET_CONFIG
export VAL_DATASET_CONFIG=""
export DISABLE_VALIDATION=1
export LOAD_PRETRANSFORM=0
export PRETRAINED_CKPT="$CHECKPOINT"
export PRETRAINED_ROUTE_WEIGHTS=ema
export SAT_PRETRAINED_ROUTE_EXPECTATION_JSON='{"loaded":612,"target_total":612,"shape_mismatches":4,"shape_mismatch_targets":2,"partial_expansions":2,"semantic_role_expansions":0,"missing":0,"shape_mismatch_target_names":["model.model.preprocess_conv.weight","model.model.transformer.project_in.weight"],"partial_expansion_targets":["model.model.preprocess_conv.weight","model.model.transformer.project_in.weight"],"semantic_role_expansion_targets":[],"missing_names":[]}'
export PRETRAINED_MODALITY_CKPT=""
export RESUME_CKPT="${RESUME_CKPT:-}"
export REQUIRE_NO_PRETRAINED=0

# The common launcher sees one logical CUDA device, backed exclusively by
# physical GPU 3. GPUs 0--2 are owned by Generation and are never exposed.
export CUDA_VISIBLE_DEVICES=3
export NUM_GPUS=1
export BATCH_SIZE="${BATCH_SIZE:-5}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export ACCUM_BATCHES=1
export TRAINING_STRATEGY=auto
export DDP_COMM_HOOK=none
export BIND_TO_GPU_NUMA=0
export RANK_AWARE_TRAINING_SEED=1
export GRADIENT_CLIP_VAL=1.0
export OVERFIT_BATCHES=2

export MAX_STEPS="${MAX_STEPS:-1000}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-250}"
export SAVE_TOP_K="${SAVE_TOP_K:-1}"
export VAL_EVERY=-1
export LIMIT_VAL_BATCHES=-1
export LOGGER="${LOGGER:-none}"
export TRAINING_GATE=1
export TRAINING_GATE_WINDOW=20
export TRAINING_GATE_MAX_LOSS_RATIO=-1
export TRAINING_GATE_GRADIENT_EVERY=20
export MIN_FREE_DISK_GIB=25
export MIN_ROOT_FREE_GIB=10
export TEMP_DIR="${TEMP_DIR:-${AMBIT_CACHE_ROOT:-cache}/spedit_dit_overfit10}"

exec "$REPO/scripts/t2a/train/run_t2a_common_8gpu.sh"
