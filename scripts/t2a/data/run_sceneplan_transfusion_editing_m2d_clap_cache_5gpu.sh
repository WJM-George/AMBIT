#!/usr/bin/env bash
set -euo pipefail

# Superseded by the user's 2026-09-05 decision. This entry is downstream of
# the frozen DiT training/audio gates; no running job is interrupted here.
echo '[editing-m2d-cache] M2D retired for Editing AR. Continue with native CLAP44 pretraining and validation; do not rebuild this cache.' >&2
exit 2

REPO="${P10_REPO:-${AMBIT_CKPT_ROOT}/stable-audio-tools-workspace}"
ROOT="${EDITING_DATA_ROOT:-${AMBIT_DATA_ROOT}/sceneplan_transfusion_editing_v1}"
SPLIT="${1:?usage: $0 train|validation|test}"

case "$SPLIT" in
    train)
        INDEX="$ROOT/training_index/train.sqlite"
        EXPECTED_ROWS=1000000
        ;;
    validation)
        INDEX="$ROOT/training_index/validation.sqlite"
        EXPECTED_ROWS=20000
        ;;
    test)
        INDEX="$ROOT/training_index/test.sqlite"
        EXPECTED_ROWS=5000
        ;;
    *)
        echo "[editing-m2d-cache] split must be train, validation, or test" >&2
        exit 2
        ;;
esac

if [[ "${M2D_NONCOMMERCIAL_EVALUATION_ACK:-}" != "1" ]]; then
    echo "[editing-m2d-cache] M2D's bundled license is evaluation-only; set M2D_NONCOMMERCIAL_EVALUATION_ACK=1 only for an authorized internal non-commercial evaluation" >&2
    exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[ambit] set CUDA_VISIBLE_DEVICES to the GPUs for this job" >&2
    exit 2
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
MARKER="$INDEX.frozen.json"
if [[ ! -r "$INDEX" || ! -r "$MARKER" ]]; then
    echo "[editing-m2d-cache] frozen $SPLIT Editing index is missing" >&2
    exit 1
fi

# Before spending five GPUs on the cache, prove with the pinned runtime that a
# real decoded-W source view reaches more projector tokens than its first 10 s
# and that early/middle/late 5 s probes all affect the 768-d embedding.
TEMPORAL_PILOT="${M2D_TEMPORAL_PILOT:-$ROOT/semantic_cache/m2d_clap_v2/temporal_policy_pilot/PASS.json}"
TEMPORAL_PILOT_SCRIPT="$REPO/scripts/t2a/test/validate_sceneplan_transfusion_editing_m2d_temporal_policy.py"
M2D_TEMPORAL_PILOT_ROWS="${M2D_TEMPORAL_PILOT_ROWS:-4}"
if [[ "$M2D_TEMPORAL_PILOT_ROWS" != "4" ]]; then
    echo "[editing-m2d-cache] formal temporal pilot requires exactly four long rows" >&2
    exit 2
fi
if [[ ! -r "$TEMPORAL_PILOT" ]]; then
    env CUDA_VISIBLE_DEVICES=3 \
        "$REPO/.venv/bin/python" "$TEMPORAL_PILOT_SCRIPT" \
        --index "$ROOT/training_index/validation.sqlite" \
        --output "$TEMPORAL_PILOT" \
        --physical-gpu 3 \
        --rows "$M2D_TEMPORAL_PILOT_ROWS"
fi
TEMPORAL_PILOT_SHA256="$($REPO/.venv/bin/python - "$TEMPORAL_PILOT" <<'PY'
import sys,torch
from stable_audio_tools.data.sceneplan_transfusion_editing_m2d_clap import validate_editing_m2d_temporal_pilot
torch.set_float32_matmul_precision("high")
print(validate_editing_m2d_temporal_pilot(sys.argv[1])["sha256"])
PY
)"

INDEX_SHA256="$($REPO/.venv/bin/python - "$INDEX" "$MARKER" "$SPLIT" "$EXPECTED_ROWS" <<'PY'
import json,sys
from pathlib import Path
from stable_audio_tools.data.sceneplan_transfusion_editing_index import sha256_file
index=Path(sys.argv[1]).resolve(strict=True)
marker=Path(sys.argv[2]).resolve(strict=True)
split=sys.argv[3]
expected=int(sys.argv[4])
value=json.loads(marker.read_text(encoding="utf-8"))
actual=sha256_file(index)
if not (
    value.get("schema")=="sceneplan_transfusion_editing_training_index"
    and int(value.get("schema_version",-1))==1
    and value.get("state")=="materialized_complete_frozen"
    and value.get("split")==split
    and int(value.get("rows",-1))==expected
    and Path(value.get("index_path","")).resolve()==index
    and value.get("index_sha256")==actual
):
    raise SystemExit("Editing M2D cache index marker changed")
print(actual)
PY
)"

CACHE_ROWS="${M2D_CACHE_ROWS:-$EXPECTED_ROWS}"
if ! [[ "$CACHE_ROWS" =~ ^[0-9]+$ ]] || (( CACHE_ROWS < 2 || CACHE_ROWS > EXPECTED_ROWS )); then
    echo "[editing-m2d-cache] M2D_CACHE_ROWS must be in [2,$EXPECTED_ROWS]" >&2
    exit 2
fi
if [[ "$CACHE_ROWS" == "$EXPECTED_ROWS" ]]; then
    CACHE_LABEL="full"
else
    CACHE_LABEL="pilot_${CACHE_ROWS}"
fi
OUTPUT_ROOT="${M2D_CACHE_OUTPUT_ROOT:-$ROOT/semantic_cache/m2d_clap_v2/${SPLIT}_${CACHE_LABEL}}"
FINAL="$OUTPUT_ROOT/m2d-clap-${SPLIT}-${CACHE_ROWS}.sqlite"
PARITY_SCRIPT="$REPO/scripts/t2a/test/validate_sceneplan_transfusion_editing_m2d_cache_online_parity.py"
PARITY_REPORT="$FINAL.online_parity.json"
mkdir -p "$OUTPUT_ROOT/logs"

run_online_parity_gate() {
    if [[ "$CACHE_ROWS" != "$EXPECTED_ROWS" ]]; then
        echo "[editing-m2d-cache] pilot cache skips formal online replay gate"
        return 0
    fi
    env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
        "$REPO/.venv/bin/python" "$PARITY_SCRIPT" \
        --cache "$FINAL" \
        --cache-sha256 "$INDEXED_CACHE_SHA256" \
        --source-index "$INDEX" \
        --source-index-sha256 "$INDEX_SHA256" \
        --expected-cache-rows "$CACHE_ROWS" \
        --split "$SPLIT" \
        --output "$PARITY_REPORT" \
        --physical-gpu 3
}

if [[ -e "$FINAL" || -e "$FINAL.frozen.json" ]]; then
    if [[ ! -r "$FINAL" || ! -r "$FINAL.frozen.json" ]]; then
        echo "[editing-m2d-cache] incomplete final cache publication: $FINAL" >&2
        exit 1
    fi
    env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
        "$REPO/.venv/bin/python" - "$FINAL" "$INDEX" "$INDEX_SHA256" "$CACHE_ROWS" "$SPLIT" <<'PY'
import json,sys,torch
from pathlib import Path
from stable_audio_tools.data.sceneplan_transfusion_editing_m2d_clap import ScenePlanTransfusionEditingM2DCLAPCache
torch.set_float32_matmul_precision("high")
cache_path=Path(sys.argv[1]).resolve(strict=True)
marker=json.loads(Path(str(cache_path)+".frozen.json").read_text(encoding="utf-8"))
cache=ScenePlanTransfusionEditingM2DCLAPCache(
    cache_path,
    source_index=Path(sys.argv[2]).resolve(strict=True),
    source_index_sha256=sys.argv[3],
    expected_rows=int(sys.argv[4]),
    expected_split=sys.argv[5],
    expected_cache_sha256=marker.get("cache_sha256"),
    verify_cache_file_hash=True,
)
if len(cache) != int(sys.argv[4]):
    raise SystemExit("Editing M2D frozen cache row count changed")
PY
    INDEXED_CACHE_SHA256="$($REPO/.venv/bin/python - "$FINAL" <<'PY'
import sys
from stable_audio_tools.data.sceneplan_transfusion_editing_index import sha256_file
print(sha256_file(sys.argv[1]))
PY
)"
    run_online_parity_gate
    echo "[editing-m2d-cache] verified existing frozen cache: $FINAL"
    exit 0
fi

pids=()
for shard in 0 1 2 3 4; do
    gpu=$((3 + shard))
    output="$OUTPUT_ROOT/shard-$(printf '%05d' "$shard").sqlite"
    if [[ -e "$output" || -e "$output.shard.json" ]]; then
        if [[ -r "$output" && -r "$output.shard.json" ]]; then
            echo "[editing-m2d-cache] reusing published shard; merge will verify it: $output"
            continue
        fi
        echo "[editing-m2d-cache] incomplete shard publication: $output" >&2
        exit 1
    fi
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        "$REPO/.venv/bin/python" \
            "$REPO/scripts/t2a/data/build_sceneplan_transfusion_editing_m2d_clap_cache.py" \
            --index "$INDEX" \
            --index-sha256 "$INDEX_SHA256" \
            --expected-index-rows "$EXPECTED_ROWS" \
            --split "$SPLIT" \
            --output "$output" \
            --physical-gpu "$gpu" \
            --shard-index "$shard" \
            --num-shards 5 \
            --max-rows "$CACHE_ROWS" \
            --temporal-pilot "$TEMPORAL_PILOT" \
            --temporal-pilot-sha256 "$TEMPORAL_PILOT_SHA256" \
            --vae-batch-size "${M2D_VAE_BATCH_SIZE:-8}" \
            --audio-batch-size "${M2D_AUDIO_BATCH_SIZE:-8}" \
            --text-batch-size "${M2D_TEXT_BATCH_SIZE:-32}" \
            2>&1 | tee "$OUTPUT_ROOT/logs/shard-$(printf '%05d' "$shard").log"
    ) &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
if [[ "$failed" != "0" ]]; then
    echo "[editing-m2d-cache] at least one GPU shard failed" >&2
    exit 1
fi

inputs=()
for shard in 0 1 2 3 4; do
    inputs+=("$OUTPUT_ROOT/shard-$(printf '%05d' "$shard").sqlite")
done
"$REPO/.venv/bin/python" \
    "$REPO/scripts/t2a/data/merge_sceneplan_transfusion_editing_m2d_clap_cache.py" \
    --inputs "${inputs[@]}" \
    --output "$FINAL" \
    --source-index "$INDEX" \
    --source-index-sha256 "$INDEX_SHA256" \
    --expected-index-rows "$EXPECTED_ROWS" \
    --expected-cache-rows "$CACHE_ROWS" \
    --split "$SPLIT"

INDEXED_CACHE_SHA256="$($REPO/.venv/bin/python - "$FINAL" <<'PY'
import sys
from stable_audio_tools.data.sceneplan_transfusion_editing_index import sha256_file
print(sha256_file(sys.argv[1]))
PY
)"
run_online_parity_gate

echo "[editing-m2d-cache] frozen cache: $FINAL"
