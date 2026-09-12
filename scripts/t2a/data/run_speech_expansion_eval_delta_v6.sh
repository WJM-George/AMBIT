#!/usr/bin/env bash
set -euo pipefail

REPO="${P10_REPO:-/home/tanhe/dataset_storage/stable-audio-tools}"
PY="$REPO/.venv/bin/python"
REVISION_ROOT="/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/speech_expansion_noalign_15s_v1"
EVAL_SOURCE_ROOT="$REVISION_ROOT/eval_sources"
LOG_ROOT="$REVISION_ROOT/logs/eval_expansion_orchestrator"
STATE_ROOT="$REVISION_ROOT/eval_stage_state"
TRAIN_P8="$REVISION_ROOT/materialized_delta/P8_SUMMARY.json"

mkdir -p "$LOG_ROOT" "$STATE_ROOT"
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
        tail -n 100 "$log" >&2 || true
        return "$code"
    fi
}

verify_code() {
    "$PY" -m py_compile \
        scripts/t2a/data/prepare_speech_expansion_eval_delta_v6.py \
        scripts/t2a/data/run_speech_expansion_eval_p8_v6.py
    bash -n scripts/t2a/data/run_speech_expansion_eval_delta_v6.sh
}

wait_for_train_p8() {
    while [[ ! -f "$TRAIN_P8" ]]; do
        printf '%s\n' "$(date -Is) waiting for train P8 before eval GPU work"
        sleep 30
    done
    "$PY" - "$TRAIN_P8" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
if value.get("status") != "complete" or int(value.get("rows",-1)) != 500_000:
    raise SystemExit("train P8 summary is not complete")
PY
    while pgrep -f 'materialize_model_sceneplan_v1_worker.py.*sceneplans_model_v2_delta' >/dev/null; do
        sleep 10
    done
}

run_long_qc() {
    "$PY" scripts/t2a/data/audit_speech_expansion_sources_noalign_15s.py \
        --candidates "$EVAL_SOURCE_ROOT/candidates/eval_long_pending_qc.parquet" \
        --output-root "$EVAL_SOURCE_ROOT/qc/eval_long" \
        --gpus 0,1,2,3,4,5,6,7 \
        --work-shards 64
}

stage 00_verify_code verify_code
stage 01_plan_long_candidates \
    "$PY" scripts/t2a/data/prepare_speech_expansion_eval_delta_v6.py plan-candidates
stage 02_wait_train_p8 wait_for_train_p8
stage 03_strong_qc_eval_long run_long_qc
stage 04_build_eval_sceneplans \
    "$PY" scripts/t2a/data/prepare_speech_expansion_eval_delta_v6.py build-scenes
stage 05_audit_eval_sceneplans \
    "$PY" scripts/t2a/data/prepare_speech_expansion_eval_delta_v6.py audit-scenes
stage 06_render_and_vae_eval_16k \
    "$PY" scripts/t2a/data/run_speech_expansion_eval_p8_v6.py \
    --gpus 0,1,2,3,4,5,6,7 --jobs-per-worker 16 --batch-size 8
stage 07_audit_materialized_eval \
    "$PY" scripts/t2a/data/prepare_speech_expansion_eval_delta_v6.py audit-materialized

printf '%s\n' "$(date -Is) COMPLETE: validation/test expansion ready for merged P9" \
    | tee "$STATE_ROOT/ALL_EVAL_EXPANSION.pass"
