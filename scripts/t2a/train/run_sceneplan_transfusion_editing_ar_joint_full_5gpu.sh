#!/usr/bin/env bash
set -euo pipefail

# Preserve historical arguments below for reproducibility, but never let the
# old automatic chain start a new M2D run after the user's route change.
echo '[editing-ar-joint-full] The M2D route is retired. CLAP44 validation and AR integration must precede the new joint run.' >&2
exit 2

REPO="${P10_REPO:-${AMBIT_CKPT_ROOT}/stable-audio-tools-workspace}"
ROOT="${EDITING_DATA_ROOT:-${AMBIT_DATA_ROOT}/sceneplan_transfusion_editing_v1}"
PREFLIGHT="$ROOT/contracts/full_training/PREFLIGHT.json"
TRAIN_INDEX="$ROOT/training_index/train.sqlite"
VALIDATION_INDEX="$ROOT/training_index/validation.sqlite"
MODEL_CONFIG="${MODEL_CONFIG:-$REPO/stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/qwen35_0p8b_300m_sceneplan_transfusion_editing_dit_full_v1.json}"
BASE_DIT_RUN="${BASE_DIT_RUN:-${AMBIT_CKPT_ROOT}/transfusion_editing/mainline/sceneplan_transfusion_editing_dit_full_seed42_v1}"
DIT_CHECKPOINT_SELECTION="${DIT_CHECKPOINT_SELECTION:-$BASE_DIT_RUN/evaluation/validation_20k_checkpoint_selection/SELECTED.json}"
DIT_GT_AUDIO_GATE="${DIT_GT_AUDIO_GATE:-$BASE_DIT_RUN/evaluation/validation_1k_gt_audio/GATE.json}"
REQUESTED_BASE_DIT_CHECKPOINT="${BASE_DIT_CHECKPOINT:-}"
RUN_DIR="${RUN_DIR:-${AMBIT_CKPT_ROOT}/transfusion_editing/mainline/sceneplan_transfusion_editing_ar_joint_m2d_full_seed42_v3}"
SOURCE_SEMANTIC_MODE="${SOURCE_SEMANTIC_MODE:-m2d_audio_caption_aux}"
TRAIN_M2D_CACHE="${TRAIN_M2D_CACHE:-$ROOT/semantic_cache/m2d_clap_v2/train_full/m2d-clap-train-1000000.sqlite}"
VALIDATION_M2D_CACHE="${VALIDATION_M2D_CACHE:-$ROOT/semantic_cache/m2d_clap_v2/validation_full/m2d-clap-validation-20000.sqlite}"
LAMBDA_SOURCE_CAPTION="${LAMBDA_SOURCE_CAPTION:-0.05}"
SOURCE_CAPTION_TEMPERATURE="${SOURCE_CAPTION_TEMPERATURE:-0.07}"
SOURCE_SEMANTIC_DROPOUT="${SOURCE_SEMANTIC_DROPOUT:-0.10}"
LOG_DIR="$RUN_DIR/logs"
MAX_STEPS="${MAX_STEPS:-25000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
VALIDATE_EVERY="${VALIDATE_EVERY:-500}"
VALIDATION_BATCHES="${VALIDATION_BATCHES:-32}"
VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-4}"
LOG_EVERY="${LOG_EVERY:-10}"

if [[ "$MAX_STEPS" != "25000" || "$SAVE_EVERY" != "5000" \
   || "$VALIDATE_EVERY" != "500" || "$VALIDATION_BATCHES" != "32" \
   || "$VALIDATION_BATCH_SIZE" != "4" || "$LOG_EVERY" != "10" ]]; then
    echo "[editing-ar-joint-full] formal schedule/gate settings are fixed" >&2
    exit 2
fi
if [[ "$SOURCE_SEMANTIC_MODE" != "m2d_audio_caption_aux" \
   || "$LAMBDA_SOURCE_CAPTION" != "0.05" \
   || "$SOURCE_CAPTION_TEMPERATURE" != "0.07" \
   || "$SOURCE_SEMANTIC_DROPOUT" != "0.10" ]]; then
    echo "[editing-ar-joint-full] formal M2D semantic settings are fixed" >&2
    exit 2
fi
if [[ "${M2D_NONCOMMERCIAL_EVALUATION_ACK:-}" != "1" ]]; then
    echo "[editing-ar-joint-full] M2D features are evaluation-only; authorization acknowledgement is required" >&2
    exit 2
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[ambit] set CUDA_VISIBLE_DEVICES to the GPUs for this job" >&2
    exit 2
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# A direct Joint recovery must mutually exclude the wider DiT-to-Joint chain.
# When invoked by the DiT launcher, verify and reuse its inherited descriptor;
# otherwise acquire the same global lock before touching the run.
LOCK_ROOT="$ROOT/materialized/locks"
mkdir -p "$LOCK_ROOT"
GLOBAL_CHAIN_LOCK="$LOCK_ROOT/training-chain.lock"
if [[ "${EDITING_TRAIN_CHAIN_LOCK_FD:-}" == "7" ]]; then
    inherited_lock="$(readlink "/proc/$$/fd/7" 2>/dev/null || true)"
    if [[ "$inherited_lock" != "$GLOBAL_CHAIN_LOCK" ]] || ! flock -n 7; then
        echo "[editing-ar-joint-full] inherited Editing chain lock is invalid" >&2
        exit 1
    fi
else
    exec 7>"$GLOBAL_CHAIN_LOCK"
    if ! flock -n 7; then
        echo "[editing-ar-joint-full] another formal Editing training chain is active" >&2
        exit 1
    fi
    export EDITING_TRAIN_CHAIN_LOCK_FD=7
    export EDITING_TRAIN_CHAIN_LOCK_PATH="$GLOBAL_CHAIN_LOCK"
fi

# The DiT launcher retains its broader chain lock across this handoff.  This
# second lock also protects deliberate direct/recovery Joint invocations.
mkdir -p "$RUN_DIR"
exec 6>"$RUN_DIR/.editing-joint-full.lock"
if ! flock -n 6; then
    echo "[editing-ar-joint-full] another formal Joint run is active" >&2
    exit 1
fi
if [[ ! -r "$PREFLIGHT" || ! -r "$DIT_CHECKPOINT_SELECTION" || ! -r "$DIT_GT_AUDIO_GATE" \
   || ! -r "$TRAIN_M2D_CACHE" || ! -r "$TRAIN_M2D_CACHE.frozen.json" \
   || ! -r "$VALIDATION_M2D_CACHE" || ! -r "$VALIDATION_M2D_CACHE.frozen.json" ]]; then
    echo "[editing-ar-joint-full] preflight, DiT RF/audio gates, or frozen M2D cache is missing" >&2
    exit 1
fi

readarray -t pinned < <(
    "$REPO/.venv/bin/python" - \
        "$PREFLIGHT" "$TRAIN_INDEX" "$VALIDATION_INDEX" \
        "$DIT_CHECKPOINT_SELECTION" "$BASE_DIT_RUN" "$MODEL_CONFIG" \
        "$REQUESTED_BASE_DIT_CHECKPOINT" <<'PY'
import json,sys
from pathlib import Path
from stable_audio_tools.data.sceneplan_transfusion_editing_index import sha256_file
preflight=Path(sys.argv[1]).resolve(strict=True)
train=Path(sys.argv[2]).resolve(strict=True)
validation=Path(sys.argv[3]).resolve(strict=True)
selection_path=Path(sys.argv[4]).resolve(strict=True)
base_run=Path(sys.argv[5]).resolve(strict=True)
model_config=Path(sys.argv[6]).resolve(strict=True)
requested=sys.argv[7]
value=json.loads(preflight.read_text(encoding="utf-8"))
if not (
    value.get("schema")=="sceneplan_transfusion_editing_full_training_preflight"
    and int(value.get("schema_version",-1))==2
    and value.get("status")=="PASS"
    and value.get("latest_route",{}).get("editing_ar_inputs")
        ==["source_foa_latent","raw_edit_request"]
    and value.get("latest_route",{}).get("old_sceneplan_input") is False
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
    and value.get("split_disjointness",{}).get("checks")=={
        "train_vs_test":True,
        "train_vs_validation":True,
        "validation_vs_test":True,
    }
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
indices=value["indices"]
if Path(indices["train"]["path"]).resolve()!=train:
    raise SystemExit("train index differs from full preflight")
if Path(indices["validation"]["path"]).resolve()!=validation:
    raise SystemExit("validation index differs from full preflight")
selection=json.loads(selection_path.read_text(encoding="utf-8"))
selected=Path(selection.get("selected_checkpoint","")).resolve(strict=True)
expected_steps=list(range(5000,30001,5000))
if not (
    selection.get("schema")=="sceneplan_transfusion_editing_dit_checkpoint_selection"
    and int(selection.get("schema_version",-1))==1
    and selection.get("status")=="PASS"
    and selection.get("selection_contract")
        =="full_20k_10k_select_10k_holdout_ema_rf_paired_source_and_p10_gate_5k_v4"
    and selection.get("physical_gpus")==[3,4,5,6,7]
    and int(selection.get("world_size",-1))==5
    and selection.get("cuda_visible_devices")=="3,4,5,6,7"
    and selection.get("latest_route",{}).get("editing_ar_inputs")
        ==["source_foa_latent","raw_edit_request"]
    and selection.get("latest_route",{}).get("old_sceneplan_input") is False
    and selection.get("latest_route",{}).get("editing_dit_frame_channels")==384
    and Path(selection.get("training_run",{}).get("run_dir","")).resolve()
        ==base_run
    and selection.get("training_run",{}).get("training_gate",{}).get("status")
        =="PASS"
    and int(selection.get("training_run",{}).get("training_gate",{}).get(
        "global_step",-1))==30000
    and Path(selection.get("preflight",{}).get("path","")).resolve()==preflight
    and selection.get("preflight",{}).get("sha256")==sha256_file(preflight)
    and Path(selection.get("model_config",{}).get("path","")).resolve()
        ==model_config
    and selection.get("model_config",{}).get("sha256")
        ==sha256_file(model_config)
    and Path(selection.get("validation_index",{}).get("path","")).resolve()
        ==validation
    and selection.get("validation_index",{}).get("sha256")
        ==indices["validation"]["sha256"]
    and int(selection.get("validation_index",{}).get("rows",-1))==20000
    and selection.get("candidate_steps")==expected_steps
    and [int(row.get("step",-1)) for row in selection.get("candidates",[])]
        ==expected_steps
    and selection.get("p10_warmstart_baseline",{}).get(
        "source_suffix_zero_invariance_pass") is True
    and selection.get("selected_promotion_gate",{}).get("pass") is True
    and int(selection.get("selected_checkpoint_step",-1)) in expected_steps
    and selected.parent==base_run / "checkpoints"
    and selection.get("selected_checkpoint_sha256")==sha256_file(selected)
):
    raise SystemExit("Editing-DiT checkpoint selection is stale or did not pass")
if requested and Path(requested).expanduser().resolve(strict=True)!=selected:
    raise SystemExit("BASE_DIT_CHECKPOINT cannot bypass the validated selection")
print(indices["train"]["sha256"])
print(indices["validation"]["sha256"])
print(selected)
print(sha256_file(selection_path))
PY
)
if [[ "${#pinned[@]}" -ne 4 ]]; then
    echo "[editing-ar-joint-full] could not read pinned indices/DiT selection" >&2
    exit 1
fi
TRAIN_SHA256="${pinned[0]}"
VALIDATION_SHA256="${pinned[1]}"
BASE_DIT_CHECKPOINT="${pinned[2]}"
DIT_CHECKPOINT_SELECTION_SHA256="${pinned[3]}"
readarray -t m2d_shas < <(
    "$REPO/.venv/bin/python" - \
        "$TRAIN_M2D_CACHE" "$VALIDATION_M2D_CACHE" <<'PY'
import json,sys
from pathlib import Path
from stable_audio_tools.data.sceneplan_transfusion_editing_index import sha256_file
for raw,split,rows in (
    (sys.argv[1],"train",1_000_000),
    (sys.argv[2],"validation",20_000),
):
    path=Path(raw).resolve(strict=True)
    marker_path=path.with_suffix(path.suffix+".frozen.json")
    parity_path=Path(str(path)+".online_parity.json").resolve(strict=True)
    marker=json.loads(marker_path.read_text(encoding="utf-8"))
    observed=sha256_file(path)
    if not (
        marker.get("schema")=="sceneplan_transfusion_editing_m2d_clap_cache"
        and int(marker.get("schema_version",-1))==2
        and marker.get("state")=="complete_frozen"
        and marker.get("split")==split
        and int(marker.get("rows",-1))==rows
        and Path(marker.get("cache_path","")).resolve()==path
        and marker.get("cache_sha256")==observed
        and parity_path.is_file()
    ):
        raise SystemExit(f"frozen Editing M2D {split} cache changed")
    print(observed)
PY
)
if [[ "${#m2d_shas[@]}" -ne 2 ]]; then
    echo "[editing-ar-joint-full] could not pin M2D caches" >&2
    exit 1
fi
TRAIN_M2D_CACHE_SHA256="${m2d_shas[0]}"
VALIDATION_M2D_CACHE_SHA256="${m2d_shas[1]}"

resolver_args=(
    "$REPO/.venv/bin/python"
    "$REPO/scripts/t2a/train/sceneplan_transfusion_editing_joint_run_contract.py"
    resolve-resume
    --run-dir "$RUN_DIR"
    --max-steps "$MAX_STEPS"
    --save-every "$SAVE_EVERY"
)
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    resolver_args+=(--requested-checkpoint "$RESUME_CHECKPOINT")
fi
mapfile -t resume_resolution < <("${resolver_args[@]}")
if [[ "${#resume_resolution[@]}" -ne 2 ]]; then
    echo "[editing-ar-joint-full] invalid lineage resume resolution" >&2
    exit 1
fi
resume_args=()
training_required=1
case "${resume_resolution[0]}" in
    COMPLETE)
        training_required=0
        echo "[editing-ar-joint-full] verified complete five-candidate inventory"
        ;;
    FRESH)
        ;;
    RESUME)
        if [[ -z "${resume_resolution[1]}" ]]; then
            echo "[editing-ar-joint-full] resolver returned an empty resume path" >&2
            exit 1
        fi
        resume_args=(--resume "${resume_resolution[1]}")
        ;;
    *)
        echo "[editing-ar-joint-full] unknown lineage action: ${resume_resolution[0]}" >&2
        exit 1
        ;;
esac

mkdir -p "$LOG_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

if [[ "$training_required" == "1" ]]; then
    set +e
    "$REPO/.venv/bin/torchrun" \
        --standalone \
        --nproc_per_node=5 \
        "$REPO/scripts/t2a/train/train_sceneplan_transfusion_editing_ar_joint_full.py" \
        --run-dir "$RUN_DIR" \
        --checkpoint "$BASE_DIT_CHECKPOINT" \
        --checkpoint-selection "$DIT_CHECKPOINT_SELECTION" \
        --checkpoint-selection-sha256 "$DIT_CHECKPOINT_SELECTION_SHA256" \
        --dit-gt-audio-gate "$DIT_GT_AUDIO_GATE" \
        --model-config "$MODEL_CONFIG" \
        --train-index "$TRAIN_INDEX" \
        --validation-index "$VALIDATION_INDEX" \
        --train-index-sha256 "$TRAIN_SHA256" \
        --validation-index-sha256 "$VALIDATION_SHA256" \
        --source-semantic-mode "$SOURCE_SEMANTIC_MODE" \
        --train-m2d-cache "$TRAIN_M2D_CACHE" \
        --validation-m2d-cache "$VALIDATION_M2D_CACHE" \
        --train-m2d-cache-sha256 "$TRAIN_M2D_CACHE_SHA256" \
        --validation-m2d-cache-sha256 "$VALIDATION_M2D_CACHE_SHA256" \
        --lambda-source-caption "$LAMBDA_SOURCE_CAPTION" \
        --source-caption-temperature "$SOURCE_CAPTION_TEMPERATURE" \
        --source-semantic-dropout "$SOURCE_SEMANTIC_DROPOUT" \
        --expected-train-rows 1000000 \
        --expected-validation-rows 20000 \
        --short-batch-size "${SHORT_BATCH_SIZE:-8}" \
        --long-batch-size "${LONG_BATCH_SIZE:-5}" \
        --gradient-accumulation "${GRADIENT_ACCUMULATION:-4}" \
        --max-steps "$MAX_STEPS" \
        --num-workers "${NUM_WORKERS:-4}" \
        --log-every "$LOG_EVERY" \
        --validate-every "$VALIDATE_EVERY" \
        --save-every "$SAVE_EVERY" \
        --validation-batches "$VALIDATION_BATCHES" \
        --validation-batch-size "$VALIDATION_BATCH_SIZE" \
        "${resume_args[@]}" 2>&1 | tee -a "$LOG_DIR/train.log"
    training_status="${PIPESTATUS[0]}"
    set -e
    if [[ "$training_status" != "0" ]]; then
        exit "$training_status"
    fi
fi

# Only an independently promoted joint checkpoint can enter the audio E2E
# stage.  LATEST remains a resume pointer and is never a quality decision.
exec env \
    JOINT_RUN="$RUN_DIR" \
    BASE_DIT_RUN="$BASE_DIT_RUN" \
    DIT_CHECKPOINT_SELECTION="$DIT_CHECKPOINT_SELECTION" \
    MODEL_CONFIG="$MODEL_CONFIG" \
    "$REPO/scripts/t2a/eval/run_sceneplan_transfusion_editing_joint_checkpoint_selection_5gpu.sh"
