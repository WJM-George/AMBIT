#!/usr/bin/env bash
set -euo pipefail

# Staged P10-v12 DiT upcycling on physical GPUs 2--7.
#
# Single-module arms are weights-only EMA warm starts from the frozen P10-v11
# 150k checkpoint. The combined arm is deliberately locked until an external
# promotion gate records independent MoE and attention wins, and then starts
# from the promoted MoE-only checkpoint.

REPO="${P10_V12_REPO:-.}"
PY="$REPO/.venv/bin/python"
ARM="${ARM:?set ARM to dense, moe, attention, or combined}"
PROFILE="${PROFILE:-preflight}"
GPU_IDS="${GPU_IDS:-2,3,4,5,6,7}"
CANONICAL_CKPT="${AMBIT_CKPT_ROOT}/dit/sceneplan_dit_v11_semantic_v2_protected_resume_150k/checkpoints/epoch=48-step=150000.ckpt"
CANONICAL_SHA256="be8c90cd1434bd71f73951531175c3674ff0f3173d5db591e2e1c476152ff59e"
CONFIG_ROOT="$REPO/stable_audio_tools/configs/model_configs/txt2audio/t2a/dit"
CANONICAL_DATASET_CONFIG="$REPO/stable_audio_tools/configs/dataset_configs/sceneplan_v2_speech_expansion_noalign_15s_v1_train_semantic_v2.json"
P10_V12_DATASET_CONFIG="$REPO/stable_audio_tools/configs/dataset_configs/sceneplan_v2_speech_expansion_noalign_15s_v1_train_semantic_v2_p10_v12_b64.json"
VAL_DATASET_CONFIG="${VAL_DATASET_CONFIG:-$REPO/stable_audio_tools/configs/dataset_configs/sceneplan_v2_speech_expansion_noalign_15s_v1_validation_semantic_v2.json}"
EVAL_ROOT="${EVAL_ROOT:-${AMBIT_DATA_ROOT}/sceneplan_v2_1p124m/revisions/speech_expansion_noalign_15s_v1/evaluation/p10_v11_balanced_1200_ckpt110k_150k_semantic_v2}"
EVAL_PANEL="$EVAL_ROOT/balanced_test_1200.jsonl"
EVAL_PANEL_SHA256="9327379bf74101c5efc5c300a311956ee7a784ec91d99792af62bf38340353fa"

case "$ARM" in
    dense)
        MODEL_CONFIG="$CONFIG_ROOT/qwen35_0p8b_300m_model_sceneplan_44_upgrade_v12_base.json"
        expected_moe=0
        expected_attention=0
        route_expectation='{"loaded":612,"target_total":612,"shape_mismatches":0,"partial_expansions":0,"semantic_role_expansions":0,"missing":0}'
        ;;
    moe)
        MODEL_CONFIG="$CONFIG_ROOT/qwen35_0p8b_300m_model_sceneplan_44_chunkmoe_routerv4_v12.json"
        expected_moe=1
        expected_attention=0
        route_expectation='{"loaded":612,"target_total":744,"shape_mismatches":0,"partial_expansions":0,"semantic_role_expansions":0,"missing":132}'
        ;;
    attention)
        MODEL_CONFIG="$CONFIG_ROOT/qwen35_0p8b_300m_model_sceneplan_44_softblock_v12.json"
        expected_moe=0
        expected_attention=1
        route_expectation='{"loaded":612,"target_total":642,"shape_mismatches":0,"partial_expansions":0,"semantic_role_expansions":0,"missing":30}'
        ;;
    combined)
        MODEL_CONFIG="$CONFIG_ROOT/qwen35_0p8b_300m_model_sceneplan_44_chunkmoe_routerv4_softblock_v12.json"
        expected_moe=1
        expected_attention=1
        route_expectation='{"loaded":744,"target_total":774,"shape_mismatches":0,"partial_expansions":0,"semantic_role_expansions":0,"missing":30}'
        ;;
    *)
        echo "[p10-v12] unsupported ARM=$ARM" >&2
        exit 2
        ;;
esac

case "$PROFILE" in
    contract)
        default_run_suffix="contract"
        ;;
    preflight)
        default_run_suffix="preflight_b72"
        ;;
    train)
        default_run_suffix="trial"
        ;;
    *)
        echo "[p10-v12] unsupported PROFILE=$PROFILE" >&2
        exit 2
        ;;
esac

if [[ -z "${DATASET_CONFIG+x}" ]]; then
    if [[ "$PROFILE" == "preflight" ]]; then
        # Deliberately stress the exact historical 72x432 / 48x648 ceiling.
        DATASET_CONFIG="$CANONICAL_DATASET_CONFIG"
    else
        DATASET_CONFIG="$P10_V12_DATASET_CONFIG"
    fi
fi

if [[ "$ARM" == "combined" ]]; then
    SOURCE_CKPT="${SOURCE_CKPT:?combined requires the promoted MoE-only SOURCE_CKPT}"
    SOURCE_CKPT_SHA256="${SOURCE_CKPT_SHA256:?combined requires SOURCE_CKPT_SHA256}"
    COMBINED_PROMOTION_GATE="${COMBINED_PROMOTION_GATE:?combined requires a dual-PASS promotion gate}"
else
    SOURCE_CKPT="${SOURCE_CKPT:-$CANONICAL_CKPT}"
    SOURCE_CKPT_SHA256="${SOURCE_CKPT_SHA256:-$CANONICAL_SHA256}"
    if [[ "$SOURCE_CKPT_SHA256" != "$CANONICAL_SHA256" ]]; then
        echo "[p10-v12] $ARM must warm-start from canonical P10-v11 150k EMA" >&2
        exit 2
    fi
    COMBINED_PROMOTION_GATE=""
fi

IFS=',' read -r -a gpu_array <<<"$GPU_IDS"
NUM_GPUS=0
for gpu in "${gpu_array[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
        echo "[p10-v12] invalid GPU id in GPU_IDS=$GPU_IDS" >&2
        exit 2
    fi
    ((NUM_GPUS += 1))
done
if (( NUM_GPUS != 6 )); then
    echo "[p10-v12] this protocol requires exactly six GPUs; got $GPU_IDS" >&2
    exit 2
fi

RUN_NAME="${RUN_NAME:-sceneplan_dit_p10_v12_${ARM}_from_v11_150k_s42_${default_run_suffix}}"
RUN_ROOT="${RUN_ROOT:-${AMBIT_CKPT_ROOT}/dit/$RUN_NAME}"
CONTRACT_PATH="$RUN_ROOT/launch_contract.json"
mkdir -p "$RUN_ROOT"

observed_source_sha="$(sha256sum "$SOURCE_CKPT" | awk '{print $1}')"
if [[ "$observed_source_sha" != "$SOURCE_CKPT_SHA256" ]]; then
    echo "[p10-v12] source checkpoint SHA256 changed: $observed_source_sha != $SOURCE_CKPT_SHA256" >&2
    exit 1
fi

"$PY" - "$ARM" "$MODEL_CONFIG" "$DATASET_CONFIG" "$VAL_DATASET_CONFIG" \
    "$SOURCE_CKPT" "$observed_source_sha" "$CANONICAL_CKPT" \
    "$CANONICAL_SHA256" "$COMBINED_PROMOTION_GATE" "$EVAL_PANEL" \
    "$EVAL_PANEL_SHA256" "$GPU_IDS" "$CONTRACT_PATH" \
    "$expected_moe" "$expected_attention" "$PROFILE" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from stable_audio_tools.configuration import load_config, validate_training_configs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


(
    arm,
    model_path,
    train_path,
    validation_path,
    source_path,
    source_sha,
    canonical_path,
    canonical_sha,
    promotion_path,
    panel_path,
    panel_sha,
    gpu_ids,
    output_path,
    expected_moe,
    expected_attention,
    profile,
) = sys.argv[1:]

model_path = Path(model_path).resolve(strict=True)
train_path = Path(train_path).resolve(strict=True)
validation_path = Path(validation_path).resolve(strict=True)
source_path = Path(source_path).resolve(strict=True)
canonical_path = Path(canonical_path).resolve(strict=True)
panel_path = Path(panel_path).resolve(strict=True)
output_path = Path(output_path).resolve()

model = load_config(model_path)
train = load_config(train_path)
validation = load_config(validation_path)
validate_training_configs(model, train)
validate_training_configs(model, validation)

dit = model["model"]["diffusion"]["config"]
moe = bool((dit.get("sceneplan_chunk_moe") or {}).get("enabled", False))
attention = bool(
    (dit.get("sceneplan_soft_block_attention") or {}).get("enabled", False)
)
if moe != bool(int(expected_moe)) or attention != bool(int(expected_attention)):
    raise SystemExit(
        f"arm/config mismatch: arm={arm} moe={moe} attention={attention}"
    )
if bool((dit.get("sceneplan_frame_text_alignment") or {}).get("enabled", False)):
    raise SystemExit("legacy monotonic alignment is forbidden in P10-v12")
if model.get("_source_checkpoint_sha256") != canonical_sha:
    raise SystemExit("model config no longer names the frozen P10-v11 source")

checkpoint = torch.load(
    source_path,
    map_location="cpu",
    weights_only=True,
    mmap=True,
)
source_step = int(checkpoint.get("global_step", -1))
source_model = checkpoint.get("model_config")
if not isinstance(source_model, dict):
    raise SystemExit("source checkpoint is missing embedded model_config")
source_dit = source_model["model"]["diffusion"]["config"]
source_moe = bool(
    (source_dit.get("sceneplan_chunk_moe") or {}).get("enabled", False)
)
source_attention = bool(
    (source_dit.get("sceneplan_soft_block_attention") or {}).get(
        "enabled", False
    )
)

promotion = None
if arm == "combined":
    if source_path == canonical_path:
        raise SystemExit("combined must start from a promoted MoE checkpoint")
    if not source_moe or source_attention:
        raise SystemExit(
            "combined source must be MoE-only (MoE enabled, attention disabled)"
        )
    source_moe_config = source_dit.get("sceneplan_chunk_moe") or {}
    destination_moe_config = dit.get("sceneplan_chunk_moe") or {}
    if source_moe_config != destination_moe_config:
        raise SystemExit(
            "combined source MoE implementation/config does not exactly match "
            "the destination MoE config"
        )
    promotion_file = Path(promotion_path).resolve(strict=True)
    promotion = json.loads(promotion_file.read_text(encoding="utf-8"))
    if not (
        promotion.get("schema")
        == "stable_audio_tools.p10_v12_dual_promotion_gate"
        and promotion.get("status") == "PASS"
        and promotion.get("moe") == "PASS"
        and promotion.get("attention") == "PASS"
        and promotion.get("moe_checkpoint_sha256") == source_sha
    ):
        raise SystemExit("combined promotion gate is not dual-PASS for this source")
else:
    if source_sha != canonical_sha or source_step != 150_000:
        raise SystemExit(
            f"{arm} source is not canonical P10-v11 step 150000"
        )
    if source_moe or source_attention:
        raise SystemExit("single-arm source checkpoint is not Dense P10-v11")

if sha256(panel_path) != panel_sha:
    raise SystemExit("frozen balanced-1200 evaluation panel changed")
domains = Counter()
panel_ids = set()
with panel_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        domains[str(row["domain"])] += 1
        panel_ids.add(str(row["panel_id"]))
if domains != {"music": 400, "sound": 400, "speech": 400}:
    raise SystemExit(f"balanced-1200 domain counts changed: {dict(domains)}")
if len(panel_ids) != 1200:
    raise SystemExit("balanced-1200 panel ids are not unique")

contract = {
    "schema": "stable_audio_tools.p10_v12_upgrade_launch",
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PASS",
    "arm": arm,
    "profile": profile,
    "model_config": str(model_path),
    "dataset_config": str(train_path),
    "validation_dataset_config": str(validation_path),
    "source_checkpoint": str(source_path),
    "source_checkpoint_sha256": source_sha,
    "source_global_step": source_step,
    "source_moe_enabled": source_moe,
    "source_attention_enabled": source_attention,
    "destination_moe_enabled": moe,
    "destination_attention_enabled": attention,
    "warm_start": "model_only_ema",
    "optimizer_scheduler_ema": "fresh",
    "gpu_ids": [int(value) for value in gpu_ids.split(",")],
    "selection_panel": str(panel_path),
    "selection_panel_sha256": panel_sha,
    "selection_panel_domains": dict(domains),
    "promotion_gate": promotion,
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(
    json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print("SAT_P10_V12_LAUNCH_CONTRACT=" + json.dumps(contract, sort_keys=True))
PY

if [[ "$PROFILE" == "contract" || "${CONTRACT_ONLY:-0}" == "1" ]]; then
    echo "[p10-v12] contract PASS: $CONTRACT_PATH"
    exit 0
fi

lock_root="${AMBIT_CKPT_ROOT}/dit/.locks"
mkdir -p "$lock_root"
lock_key="${GPU_IDS//,/__}"
exec 9>"$lock_root/p10_v12_gpu_${lock_key}.lock"
if ! flock -n 9; then
    echo "[p10-v12] another P10-v12 launcher holds GPUs $GPU_IDS" >&2
    exit 1
fi

if [[ "${SKIP_GPU_FREE_CHECK:-0}" != "1" ]]; then
    for gpu in "${gpu_array[@]}"; do
        active="$({ nvidia-smi -i "$gpu" \
            --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader,nounits || true; } | sed '/^[[:space:]]*$/d')"
        if [[ -n "$active" ]]; then
            echo "[p10-v12] GPU $gpu is already occupied: $active" >&2
            exit 1
        fi
    done
fi

export RUN_NAME RUN_ROOT MODEL_CONFIG DATASET_CONFIG VAL_DATASET_CONFIG
export RUN_LABEL="sceneplan-p10-v12-$ARM-$PROFILE"
export RUN_CATEGORY="${RUN_CATEGORY:-pilots}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export NUM_GPUS
export LOAD_PRETRANSFORM=0
export PRETRAINED_CKPT="$SOURCE_CKPT"
export PRETRAINED_ROUTE_WEIGHTS=ema
export SAT_PRETRAINED_ROUTE_EXPECTATION_JSON="$route_expectation"
export PRETRAINED_MODALITY_CKPT=""
export REQUIRE_NO_PRETRAINED=0
export RESUME_CKPT="${RESUME_CKPT:-}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export ACCUM_BATCHES="${ACCUM_BATCHES:-1}"
export TRAINING_STRATEGY=ddp_static
export DDP_BUCKET_CAP_MB="${DDP_BUCKET_CAP_MB:-50}"
export DDP_COMM_HOOK="${DDP_COMM_HOOK:-none}"
export BIND_TO_GPU_NUMA="${BIND_TO_GPU_NUMA:-1}"
export RANK_AWARE_TRAINING_SEED=1
export TRAINING_SEED="${TRAINING_SEED:-42}"
export GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-1.0}"
export ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export MIN_FREE_DISK_GIB="${MIN_FREE_DISK_GIB:-50}"
export MIN_ROOT_FREE_GIB="${MIN_ROOT_FREE_GIB:-20}"
export TEMP_DIR="${TEMP_DIR:-${AMBIT_CACHE_ROOT:-cache}/p10_v12_${ARM}_${PROFILE}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${AMBIT_CKPT_ROOT}/dit/triton-cache/sceneplan_qwen35_fla052_cc170}"

if [[ "$PROFILE" == "preflight" ]]; then
    export BATCH_SIZE="${BATCH_SIZE:-72}"
    export MAX_STEPS="${MAX_STEPS:-4}"
    export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-999999}"
    export SAVE_TOP_K="${SAVE_TOP_K:-0}"
    export DISABLE_VALIDATION=1
    export LOGGER="${LOGGER:-none}"
    export BENCHMARK=1
    export BENCHMARK_WARMUP_BATCHES="${BENCHMARK_WARMUP_BATCHES:-1}"
    export TRAINING_GATE=1
    export TRAINING_GATE_WINDOW="${TRAINING_GATE_WINDOW:-2}"
    export TRAINING_GATE_MAX_LOSS_RATIO="${TRAINING_GATE_MAX_LOSS_RATIO:--1}"
    export TRAINING_GATE_GRADIENT_EVERY=1
else
    export BATCH_SIZE="${BATCH_SIZE:-64}"
    export MAX_STEPS="${MAX_STEPS:-10000}"
    export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2500}"
    export SAVE_TOP_K="${SAVE_TOP_K:--1}"
    export DISABLE_VALIDATION="${DISABLE_VALIDATION:-0}"
    export VAL_EVERY="${VAL_EVERY:-2500}"
    export LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-64}"
    export LOGGER="${LOGGER:-wandb}"
    export BENCHMARK="${BENCHMARK:-0}"
    export BENCHMARK_WARMUP_BATCHES="${BENCHMARK_WARMUP_BATCHES:-1}"
    export TRAINING_GATE=1
    export TRAINING_GATE_WINDOW="${TRAINING_GATE_WINDOW:-100}"
    export TRAINING_GATE_MAX_LOSS_RATIO="${TRAINING_GATE_MAX_LOSS_RATIO:--1}"
    export TRAINING_GATE_GRADIENT_EVERY="${TRAINING_GATE_GRADIENT_EVERY:-50}"
fi

echo "[p10-v12] arm=$ARM profile=$PROFILE GPUs=$GPU_IDS run=$RUN_NAME"
echo "[p10-v12] model-only EMA source=$SOURCE_CKPT source_sha256=$observed_source_sha"
echo "[p10-v12] balanced selection panel=$EVAL_PANEL (400 music + 400 sound + 400 speech)"

exec "$REPO/scripts/t2a/train/run_t2a_common_8gpu.sh"
