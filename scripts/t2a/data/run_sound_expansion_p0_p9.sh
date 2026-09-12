#!/usr/bin/env bash
set -euo pipefail

REPO="${SOUND_EXPANSION_REPO:-/home/tanhe/dataset_storage/stable-audio-tools}"
PY="$REPO/.venv/bin/python"
QWEN_PY="/home/tanhe/dataset_storage/.venv-qwen/bin/python"
DATASET_ROOT="/mnt/sdb/audio_dataset/sceneplan_v2_1p124m"
REVISION_ROOT="$DATASET_ROOT/revisions/sound_expansion_v1"
VGG_ROOT="$DATASET_ROOT/supplements/vggsound_sound_delta_v1"
VGG_SELECTION="$VGG_ROOT/source_selection"
VGG_EXTRACTION="$VGG_ROOT/extraction_pilot_20k"
VGG_SNAPSHOT="/mnt/sdc/audio_dataset/datasets/vggsound/snapshot"
FSDK_ROOT="$REVISION_ROOT/sources/fsdkaggle2019"
UNIVERSE_ROOT="$REVISION_ROOT/sources/universe"
ANNOTATION_ROOT="$REVISION_ROOT/source_annotations/annotations"
REGISTRY_ROOT="$REVISION_ROOT/source_annotations/registry"
SCENEPLAN_ROOT="$REVISION_ROOT/sceneplans_model_v1"
MATERIALIZED_ROOT="$REVISION_ROOT/materialized"
SPEECH_TIMING_ROOT="$DATASET_ROOT/source_annotations/speech_forced_alignment_v1/registry"
SPEECH_TIMING="$SPEECH_TIMING_ROOT/speech_timing_train.sqlite"
LOG_ROOT="$REVISION_ROOT/logs/p0_p9_orchestrator"
STATE_ROOT="$REVISION_ROOT/stage_state"
QWEN_CODE="$REPO/dataset/captioning/sceneplan_a2t_v2"

mkdir -p "$LOG_ROOT" "$STATE_ROOT" "$ANNOTATION_ROOT" "$REGISTRY_ROOT"
cd "$REPO"

stage() {
    local name="$1"
    shift
    local log="$LOG_ROOT/${name}.log"
    if [[ -f "$STATE_ROOT/${name}.pass" ]]; then
        printf '%s\n' "$(date -Is) SKIP $name (already passed)"
        return 0
    fi
    printf '%s\n' "$(date -Is) START $name" | tee "$STATE_ROOT/${name}.running"
    if "$@" >"$log" 2>&1; then
        rm -f "$STATE_ROOT/${name}.running" "$STATE_ROOT/${name}.failed"
        printf '%s\n' "$(date -Is) PASS $name" | tee "$STATE_ROOT/${name}.pass"
    else
        local code=$?
        rm -f "$STATE_ROOT/${name}.running"
        printf '%s\n' "$(date -Is) FAIL $name exit=$code log=$log" \
            | tee "$STATE_ROOT/${name}.failed" >&2
        tail -n 80 "$log" >&2 || true
        return "$code"
    fi
}

wait_for_prerequisites() {
    while tmux has-session -t sound_expansion_timing 2>/dev/null; do
        echo "waiting for immutable 500k speech timing index: $(date -Is)"
        sleep 20
    done
    while tmux has-session -t sound_expansion_hf 2>/dev/null; do
        echo "waiting for VGGSound shard repair: $(date -Is)"
        tail -n 3 "$VGG_ROOT/hf_repair_download.log" 2>/dev/null || true
        sleep 20
    done
    "$PY" - "$SPEECH_TIMING" "$FSDK_ROOT/SUMMARY.json" <<'PY'
import json,sys
from pathlib import Path

timing=Path(sys.argv[1]).resolve(strict=True)
receipt=Path(str(timing)+".receipt.json").resolve(strict=True)
doc=json.loads(receipt.read_text())
if doc.get("status")!="PASS" or int(doc.get("rows",-1))!=500000:
    raise SystemExit(f"invalid speech timing receipt: {doc}")
fsd=json.loads(Path(sys.argv[2]).read_text())
if fsd.get("status")!="PASS" or int(fsd.get("counts",{}).get("qc_passed_unique",0))<=0:
    raise SystemExit(f"invalid FSDKaggle extraction summary: {fsd}")
PY
    for shard in 00 03 08; do
        local path="$VGG_SNAPSHOT/vggsound_${shard}.tar.gz"
        test -s "$path"
        gzip -t "$path"
        echo "verified $path"
    done
}

verify_base_and_code() {
    "$PY" - "$DATASET_ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
marker=json.loads((root/"FROZEN_P9.json").read_text())
if not (
    marker.get("state")=="P9_complete_frozen_waiting_for_user_acceptance"
    and marker.get("p10_training_started") is False
    and marker.get("p11_training_started") is False
):
    raise SystemExit("base P9 marker is not frozen and untouched")
PY
    "$PY" -m py_compile \
        dataset/indexing/build_vggsound_sound_delta_manifest.py \
        dataset/indexing/extract_vggsound_sound_delta.py \
        dataset/indexing/extract_fsdkaggle2019_sound_expansion.py \
        dataset/indexing/build_sound_expansion_universe.py \
        dataset/captioning/sceneplan_a2t_v2/repartition_annotation_splits.py \
        scripts/t2a/data/build_model_sceneplan_sound_replacements_v1.py \
        scripts/t2a/data/run_model_sceneplan_sound_replacements_p8.py \
        scripts/t2a/data/freeze_sound_expansion_revision_p9.py \
        scripts/t2a/data/finalize_sound_expansion_p10_readiness.py \
        scripts/t2a/train/prepare_sceneplan_dit_p10_44.py
    bash -n scripts/t2a/train/run_sceneplan_dit_p10_44_8gpu.sh
    "$PY" scripts/t2a/train/test_sceneplan_dit_p10_44.py
    "$PY" -m unittest discover -s dataset/captioning/sceneplan_a2t_v2 -p 'test_*.py'
}

classify_spoken_language() {
    local inputs=()
    local shard
    for shard in 0 1 2 3; do
        inputs+=(
            --input-jsonl
            "$ANNOTATION_ROOT/source_descriptions_instruct.shard$(printf '%03d' "$shard")-of-004.jsonl"
        )
    done
    "$QWEN_PY" "$QWEN_CODE/classify_spoken_language.py" \
        "${inputs[@]}" \
        --out "$REGISTRY_ROOT/spoken_language_background.jsonl" \
        --overwrite
}

run_conditioning_audits() {
    local audit_root="$REVISION_ROOT/audit/conditioning_v3"
    mkdir -p "$audit_root"
    "$PY" scripts/t2a/data/audit_sceneplan_44_contract.py \
        --index "$REVISION_ROOT/training_index/train.sqlite" \
        --samples 20000 \
        --output "$audit_root/sceneplan_44_train_audit.json"
    "$PY" scripts/t2a/data/audit_sceneplan_44_contract.py \
        --index "$REVISION_ROOT/training_index/validation.sqlite" \
        --samples 0 \
        --output "$audit_root/sceneplan_44_validation_audit.json"
    "$PY" scripts/t2a/data/audit_sceneplan_44_contract.py \
        --index "$REVISION_ROOT/training_index/test.sqlite" \
        --samples 0 \
        --output "$audit_root/sceneplan_44_test_audit.json"
}

stage 00_wait_prerequisites wait_for_prerequisites
stage 01_base_and_code_gates verify_base_and_code

stage 02_vggsound_selection \
    "$PY" dataset/indexing/build_vggsound_sound_delta_manifest.py \
    --output-root "$VGG_SELECTION" \
    --pilot-selection-rows 59190

stage 03_vggsound_extract_qc \
    "$PY" dataset/indexing/extract_vggsound_sound_delta.py \
    --selection "$VGG_SELECTION/pilot_extraction_selection.jsonl" \
    --snapshot "$VGG_SNAPSHOT" \
    --output-root "$VGG_EXTRACTION" \
    --target-rows 0 \
    --jobs 32 \
    --qc-jobs 48

stage 04_freeze_source_universe \
    "$PY" dataset/indexing/build_sound_expansion_universe.py \
    --vgg-jsonl "$VGG_EXTRACTION/pilot_frozen.jsonl" \
    --fsdkaggle-jsonl "$FSDK_ROOT/qc_passed.jsonl" \
    --output-root "$UNIVERSE_ROOT"

stage 05_qwen3_omni_instruct_a2t \
    "$QWEN_PY" "$QWEN_CODE/launch_transformers_scaleout.py" \
    --gpu-groups '0,1;2,3;4,5;6,7' \
    --log-dir "$LOG_ROOT/a2t_workers" \
    -- \
    --input-jsonl "$UNIVERSE_ROOT/instruct_input.jsonl" \
    --out "$ANNOTATION_ROOT/source_descriptions_instruct.jsonl" \
    --device-map balanced \
    --attn-implementation sdpa \
    --batch-size 256 \
    --fsync-every-batches 4 \
    --safety-max-generation-tokens 256

stage 05b_repartition_annotation_splits \
    "$QWEN_PY" "$QWEN_CODE/repartition_annotation_splits.py" \
    --input-jsonl "$UNIVERSE_ROOT/instruct_input.jsonl" \
    --annotations-glob "$ANNOTATION_ROOT/source_descriptions_instruct.shard*-of-004.jsonl" \
    --audit-json "$REGISTRY_ROOT/annotation_split_repartition.json" \
    --num-shards 4

stage 06_spoken_language_gate classify_spoken_language

stage 07_finalize_description_registry \
    "$QWEN_PY" "$QWEN_CODE/finalize_source_registry.py" \
    --universe-parquet "$UNIVERSE_ROOT/source_universe.parquet" \
    --input-jsonl "$UNIVERSE_ROOT/instruct_input.jsonl" \
    --annotations-glob "$ANNOTATION_ROOT/source_descriptions_instruct.shard*-of-004.jsonl" \
    --spoken-labels-jsonl "$REGISTRY_ROOT/spoken_language_background.jsonl" \
    --output-root "$REGISTRY_ROOT" \
    --num-shards 4 \
    --finalize

stage 08_build_replacement_sceneplans \
    "$PY" scripts/t2a/data/build_model_sceneplan_sound_replacements_v1.py \
    --universe-parquet "$UNIVERSE_ROOT/source_universe.parquet" \
    --registry-parquet "$REGISTRY_ROOT/source_description_registry.parquet" \
    --candidate-jsonl "$UNIVERSE_ROOT/candidates_frozen.jsonl" \
    --output-root "$SCENEPLAN_ROOT"

stage 09_p8_render_and_vae \
    "$PY" scripts/t2a/data/run_model_sceneplan_sound_replacements_p8.py \
    --sceneplan-root "$SCENEPLAN_ROOT" \
    --output-root "$MATERIALIZED_ROOT" \
    --gpus 0,1,2,3,4,5,6,7 \
    --jobs-per-worker 12 \
    --batch-size 8 \
    --train-render-root /dev/shm/sceneplan_sound_expansion_v1

stage 10_p9_freeze_and_merge \
    "$PY" scripts/t2a/data/freeze_sound_expansion_revision_p9.py \
    --revision-root "$REVISION_ROOT" \
    --sceneplan-root "$SCENEPLAN_ROOT" \
    --materialized-root "$MATERIALIZED_ROOT" \
    --source-universe-summary "$UNIVERSE_ROOT/SUMMARY.json" \
    --registry-audit "$REGISTRY_ROOT/finalizer_audit.json" \
    --speech-timing-index "$SPEECH_TIMING"

stage 11_complete_44_conditioning_audits run_conditioning_audits

stage 12_p10_data_readiness \
    "$PY" scripts/t2a/data/finalize_sound_expansion_p10_readiness.py \
    --revision-root "$REVISION_ROOT"

# The replacement shards have now passed gzip integrity and were consumed by
# the full extraction.  Remove only the exact, previously quarantined corrupt
# copy; it is recoverable by re-downloading the public shard.
CORRUPT_08="$VGG_SNAPSHOT/vggsound_08.tar.gz.corrupt_20260825"
if [[ -f "$CORRUPT_08" ]]; then
    rm -- "$CORRUPT_08"
    printf '%s\n' "$CORRUPT_08" > "$REVISION_ROOT/REMOVED_CORRUPT_INPUT.txt"
fi
rmdir /dev/shm/sceneplan_sound_expansion_v1 2>/dev/null || true
printf '%s\n' "$(date -Is) COMPLETE P0-P9; P10 NOT STARTED" \
    | tee "$STATE_ROOT/ALL_P0_P9_COMPLETE.pass"
