#!/usr/bin/env bash
set -euo pipefail

REPO="${P10_REPO:-/mnt/sdc/stable-audio-tools-workspace}"
ROOT="${EDITING_DATA_ROOT:-/mnt/sdb/audio_dataset/sceneplan_transfusion_editing_v1}"
RUN_DIR="${RUN_ROOT:-/mnt/sdb/model_archives/transfusion_editing/mainline/sceneplan_transfusion_editing_dit_full_seed42_v1}"
PREFLIGHT="$ROOT/contracts/full_training/PREFLIGHT.json"
VALIDATION_INDEX="$ROOT/training_index/validation.sqlite"
MODEL_CONFIG="${MODEL_CONFIG:-$REPO/stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/qwen35_0p8b_300m_sceneplan_transfusion_editing_dit_full_v1.json}"
OUTPUT="${SELECTION_OUTPUT:-$RUN_DIR/evaluation/validation_20k_checkpoint_selection/SELECTED.json}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES// /}" != "3,4,5,6,7" ]]; then
    echo "[editing-dit-select] only physical GPUs 3,4,5,6,7 are allowed" >&2
    exit 2
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3,4,5,6,7
if [[ ! -r "$PREFLIGHT" || ! -r "$VALIDATION_INDEX" || ! -r "$MODEL_CONFIG" ]]; then
    echo "[editing-dit-select] preflight, validation index, or model config is missing" >&2
    exit 1
fi

VALIDATION_SHA256="$($REPO/.venv/bin/python - "$PREFLIGHT" "$VALIDATION_INDEX" "$MODEL_CONFIG" <<'PY'
import json,sys
from pathlib import Path
from stable_audio_tools.data.sceneplan_transfusion_editing_index import sha256_file

preflight=Path(sys.argv[1]).resolve(strict=True)
validation=Path(sys.argv[2]).resolve(strict=True)
model=Path(sys.argv[3]).resolve(strict=True)
value=json.loads(preflight.read_text(encoding="utf-8"))
record=value.get("indices",{}).get("validation",{})
latest=value.get("latest_route",{})
if not (
    value.get("status")=="PASS"
    and latest.get("editing_ar_inputs")==[
        "source_foa_latent","raw_edit_request"
    ]
    and latest.get("editing_ar_target")=="complete_new_sceneplan"
    and latest.get("old_sceneplan_input") is False
    and latest.get("editing_dit_frame_channels")==384
    and latest.get("editing_dit_clean_source_always_present") is True
    and Path(record.get("path","")).resolve()==validation
    and int(record.get("rows",-1))==20000
    and int(record.get("short_rows",-1))==15000
    and int(record.get("long_rows",-1))==5000
    and record.get("sha256")==sha256_file(validation)
    and Path(value.get("model_config","")).resolve()==model
    and value.get("model_config_sha256")==sha256_file(model)
):
    raise SystemExit("full Editing preflight/validation selection contract changed")
print(record["sha256"])
PY
)"

mkdir -p "$(dirname "$OUTPUT")/logs"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

exec "$REPO/.venv/bin/torchrun" \
    --standalone \
    --nproc_per_node=5 \
    "$REPO/scripts/t2a/eval/select_sceneplan_transfusion_editing_dit_checkpoint.py" \
    --run-dir "$RUN_DIR" \
    --data-root "$ROOT" \
    --preflight "$PREFLIGHT" \
    --model-config "$MODEL_CONFIG" \
    --validation-index "$VALIDATION_INDEX" \
    --validation-index-sha256 "$VALIDATION_SHA256" \
    --expected-max-step 30000 \
    --checkpoint-every 5000 \
    --short-batch-size "${SELECTION_SHORT_BATCH_SIZE:-72}" \
    --long-batch-size "${SELECTION_LONG_BATCH_SIZE:-48}" \
    --num-workers "${SELECTION_NUM_WORKERS:-8}" \
    --seed 42 \
    --timesteps 0.1 0.3 0.5 0.7 0.9 1.0 \
    --output "$OUTPUT" \
    2>&1 | tee -a "$(dirname "$OUTPUT")/logs/select.log"
