#!/usr/bin/env bash
set -euo pipefail

REPO="${P10_REPO:-.}"
ROOT="${EDITING_DATA_ROOT:-${AMBIT_DATA_ROOT}/sceneplan_transfusion_editing_v1}"
RUN_DIR="${JOINT_RUN:-${AMBIT_CKPT_ROOT}/transfusion_editing/mainline/sceneplan_transfusion_editing_ar_joint_m2d_full_seed42_v3}"
DIT_RUN="${BASE_DIT_RUN:-${AMBIT_CKPT_ROOT}/transfusion_editing/mainline/sceneplan_transfusion_editing_dit_full_seed42_v1}"
PREFLIGHT="$ROOT/contracts/full_training/PREFLIGHT.json"
VALIDATION_INDEX="$ROOT/training_index/validation.sqlite"
DIT_SELECTION="${DIT_CHECKPOINT_SELECTION:-$DIT_RUN/evaluation/validation_20k_checkpoint_selection/SELECTED.json}"
MODEL_CONFIG="${MODEL_CONFIG:-$REPO/stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/qwen35_0p8b_300m_sceneplan_transfusion_editing_dit_full_v1.json}"
CODEC="${CODEC:-${AMBIT_DATA_ROOT}/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4}"
OUTPUT="${JOINT_SELECTION_OUTPUT:-$RUN_DIR/evaluation/validation_20k_joint_checkpoint_selection/SELECTED.json}"

if [[ "${M2D_NONCOMMERCIAL_EVALUATION_ACK:-}" != "1" ]]; then
    echo "[editing-joint-select] M2D features are evaluation-only; authorization acknowledgement is required" >&2
    exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[ambit] set CUDA_VISIBLE_DEVICES to the GPUs for this job" >&2
    exit 2
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
for required in "$PREFLIGHT" "$VALIDATION_INDEX" "$DIT_SELECTION" "$MODEL_CONFIG" "$CODEC/codec.json" "$RUN_DIR/RUN_CONTRACT.json" "$RUN_DIR/FINAL.json"; do
    if [[ ! -r "$required" ]]; then
        echo "[editing-joint-select] required artifact is missing: $required" >&2
        exit 1
    fi
done

VALIDATION_SHA256="$($REPO/.venv/bin/python - "$PREFLIGHT" "$VALIDATION_INDEX" <<'PY'
import json,sys
from pathlib import Path
from stable_audio_tools.data.sceneplan_transfusion_editing_index import sha256_file
preflight=Path(sys.argv[1]).resolve(strict=True)
validation=Path(sys.argv[2]).resolve(strict=True)
value=json.loads(preflight.read_text(encoding="utf-8"))
record=value.get("indices",{}).get("validation",{})
latest=value.get("latest_route",{})
if not (
    value.get("status")=="PASS"
    and latest.get("editing_ar_inputs")==["source_foa_latent","raw_edit_request"]
    and latest.get("editing_ar_target")=="complete_new_sceneplan"
    and latest.get("old_sceneplan_input") is False
    and latest.get("editing_dit_frame_channels")==384
    and latest.get("editing_dit_clean_source_always_present") is True
    and Path(record.get("path","")).resolve()==validation
    and int(record.get("rows",-1))==20000
    and int(record.get("short_rows",-1))==15000
    and int(record.get("long_rows",-1))==5000
    and record.get("sha256")==sha256_file(validation)
    and all(
        record.get(f"{role}_latent_shards_exhaustively_verified") is True
        and record.get(f"{role}_pair_rows")==20000
        and record.get(f"{role}_latent_shards",0)>0
        and len(record.get(f"{role}_latent_shard_inventory_sha256", ""))==64
        for role in ("source","target")
    )
):
    raise SystemExit("full Editing preflight/validation contract changed")
print(record["sha256"])
PY
)"

mkdir -p "$(dirname "$OUTPUT")/logs"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

set +e
"$REPO/.venv/bin/torchrun" \
    --standalone \
    --nproc_per_node=5 \
    "$REPO/scripts/t2a/eval/select_sceneplan_transfusion_editing_joint_checkpoint.py" \
    --run-dir "$RUN_DIR" \
    --data-root "$ROOT" \
    --preflight "$PREFLIGHT" \
    --model-config "$MODEL_CONFIG" \
    --codec "$CODEC" \
    --validation-index "$VALIDATION_INDEX" \
    --validation-index-sha256 "$VALIDATION_SHA256" \
    --base-dit-selection "$DIT_SELECTION" \
    --short-ar-batch-size "${JOINT_SELECTION_SHORT_AR_BATCH_SIZE:-8}" \
    --long-ar-batch-size "${JOINT_SELECTION_LONG_AR_BATCH_SIZE:-5}" \
    --short-rf-batch-size "${JOINT_SELECTION_SHORT_RF_BATCH_SIZE:-72}" \
    --long-rf-batch-size "${JOINT_SELECTION_LONG_RF_BATCH_SIZE:-48}" \
    --free-batch-size "${JOINT_SELECTION_FREE_BATCH_SIZE:-2}" \
    --num-workers "${JOINT_SELECTION_NUM_WORKERS:-4}" \
    --seed 42 \
    --output "$OUTPUT" \
    2>&1 | tee -a "$(dirname "$OUTPUT")/logs/select.log"
selection_status="${PIPESTATUS[0]}"
set -e
if [[ "$selection_status" != "0" ]]; then
    exit "$selection_status"
fi

# The formal route ends only after a selected (never merely latest) joint
# checkpoint passes real FOA -> VAE -> AR -> DiT -> VAE -> FOA evaluation.
exec env \
    JOINT_RUN="$RUN_DIR" \
    JOINT_CHECKPOINT_SELECTION="$OUTPUT" \
    MODEL_CONFIG="$MODEL_CONFIG" \
    CODEC="$CODEC" \
    "$REPO/scripts/t2a/eval/run_sceneplan_transfusion_editing_end_to_end_5gpu.sh"
