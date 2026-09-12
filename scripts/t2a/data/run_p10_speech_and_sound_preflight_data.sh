#!/usr/bin/env bash
set -euo pipefail

# Finish the two expensive, immutable P10 data preflights in dependency order.
# This script never starts a DiT experiment.  Every transition is guarded by a
# machine-readable PASS receipt and all long GPU work remains resumable.

REPO="/home/tanhe/dataset_storage/stable-audio-tools"
PY_STABLE="$REPO/.venv/bin/python"
PY_QWEN="/home/tanhe/dataset_storage/.venv-qwen/bin/python"
DATA_ROOT="/mnt/sdb/audio_dataset/sceneplan_v2_1p124m"
ALIGN_ROOT="$DATA_ROOT/source_annotations/speech_forced_alignment_v1"
DELTA_ROOT="$DATA_ROOT/supplements/vggsound_sound_delta_v1"
A2T_ROOT="$DELTA_ROOT/a2t_pilot_20k"
SCENEPLAN_ROOT="$DELTA_ROOT/sceneplans_pilot_20k"
MATERIALIZED_ROOT="$DELTA_ROOT/materialized_pilot_20k"
FROZEN_ROOT="$DELTA_ROOT/frozen_pilot_20k"
LOG_ROOT="$DELTA_ROOT/orchestrator_logs"

mkdir -p "$LOG_ROOT"
cd "$REPO"

# Do not contend with the already-running eight-GPU forced-aligner job.  The
# launch shell may spend a short time consolidating its shards after workers
# exit, so wait for both process classes.
while pgrep -f "run_sceneplan_speech_alignment_worker.py --root $ALIGN_ROOT" >/dev/null \
   || pgrep -f "run_sceneplan_speech_alignment_pilot_8gpu.sh" >/dev/null; do
  sleep 20
done

# Resume once under the current frozen timing-grid contract.  Passing shards
# skip model load; any old checkpoint metadata is deterministically rechecked
# from stored raw timestamps, without changing the aligner output.
SAT_ALIGNMENT_ROOT="$ALIGN_ROOT" \
SAT_ALIGNMENT_PREPARED=1 \
SAT_ALIGNMENT_BATCH_SIZE=16 \
  bash scripts/t2a/eval/diagnostics/run_sceneplan_speech_alignment_pilot_8gpu.sh \
  >"$LOG_ROOT/speech_alignment_contract_resume.log" 2>&1

"$PY_STABLE" - "$ALIGN_ROOT/ALIGNMENT_SUMMARY.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("status") != "PASS" or int(summary.get("metrics", {}).get("rows", -1)) != 500_000:
    raise SystemExit(f"full speech alignment gate failed: {summary.get('status')}")
PY

"$PY_STABLE" scripts/t2a/eval/diagnostics/audit_sceneplan_speech_alignment_caption_mapping.py \
  --root "$ALIGN_ROOT" \
  --index "$DATA_ROOT/training_index/train.sqlite" \
  >"$LOG_ROOT/speech_caption_mapping.log" 2>&1

"$PY_STABLE" scripts/t2a/data/build_sceneplan_speech_timing_index.py \
  --caption-timing-jsonl "$ALIGN_ROOT/registry/full_caption_token_timing.jsonl" \
  --output "$ALIGN_ROOT/registry/speech_timing_train.sqlite" \
  --expected-rows 500000 \
  >"$LOG_ROOT/speech_timing_index.log" 2>&1

mkdir -p "$A2T_ROOT/annotations/logs"
"$PY_QWEN" dataset/captioning/sceneplan_a2t_v2/launch_transformers_scaleout.py \
  --log-dir "$A2T_ROOT/annotations/logs" \
  -- \
  --input-jsonl "$A2T_ROOT/instruct_input.jsonl" \
  --out "$A2T_ROOT/annotations/source_descriptions_instruct.jsonl" \
  --batch-size 256 \
  --device-map balanced \
  --attn-implementation sdpa \
  --safety-max-generation-tokens 256 \
  >"$LOG_ROOT/sound_a2t_scaleout.log" 2>&1

annotation_inputs=()
for shard in "$A2T_ROOT"/annotations/source_descriptions_instruct.shard*-of-*.jsonl; do
  annotation_inputs+=(--input-jsonl "$shard")
done
if [[ "${#annotation_inputs[@]}" -ne 8 ]]; then
  echo "expected four Sound A2T annotation shards" >&2
  exit 1
fi

"$PY_STABLE" dataset/captioning/sceneplan_a2t_v2/classify_spoken_language.py \
  "${annotation_inputs[@]}" \
  --out "$A2T_ROOT/annotations/spoken_language_background_v1.jsonl" \
  >"$LOG_ROOT/sound_spoken_language.log" 2>&1

"$PY_STABLE" dataset/captioning/sceneplan_a2t_v2/finalize_source_registry.py \
  --universe-parquet "$A2T_ROOT/source_universe.parquet" \
  --input-jsonl "$A2T_ROOT/instruct_input.jsonl" \
  --annotations-glob "$A2T_ROOT/annotations/source_descriptions_instruct.shard*-of-*.jsonl" \
  --spoken-labels-jsonl "$A2T_ROOT/annotations/spoken_language_background_v1.jsonl" \
  --output-root "$A2T_ROOT/registry" \
  --num-shards 4 \
  --finalize \
  >"$LOG_ROOT/sound_registry_finalizer.log" 2>&1

if [[ ! -s "$SCENEPLAN_ROOT/READY" ]]; then
  "$PY_STABLE" scripts/t2a/data/build_model_sceneplan_sound_delta_v1.py \
    --universe-parquet "$A2T_ROOT/source_universe.parquet" \
    --registry-parquet "$A2T_ROOT/registry/source_description_registry.parquet" \
    --output-root "$SCENEPLAN_ROOT" \
    --expected-rows 20000 \
    >"$LOG_ROOT/sound_sceneplan_p75.log" 2>&1
fi

if [[ ! -s "$MATERIALIZED_ROOT/P8_SUMMARY.json" ]]; then
  "$PY_STABLE" scripts/t2a/data/run_model_sceneplan_sound_delta_p8.py \
    --sceneplan-root "$SCENEPLAN_ROOT" \
    --output-root "$MATERIALIZED_ROOT" \
    --expected-rows 20000 \
    >"$LOG_ROOT/sound_p8.log" 2>&1
fi

mkdir -p "$FROZEN_ROOT"
if [[ ! -s "$FROZEN_ROOT/FROZEN_P9.json" ]]; then
  "$PY_STABLE" scripts/t2a/data/run_model_sceneplan_sound_delta_p9.py \
    --supplement-root "$FROZEN_ROOT" \
    --sceneplan-root "$SCENEPLAN_ROOT" \
    --materialized-root "$MATERIALIZED_ROOT" \
    --expected-rows 20000 \
    >"$LOG_ROOT/sound_p9.log" 2>&1
fi

"$PY_STABLE" - \
  "$ALIGN_ROOT/registry/speech_timing_train.sqlite.receipt.json" \
  "$A2T_ROOT/registry/finalizer_audit.json" \
  "$MATERIALIZED_ROOT/P8_SUMMARY.json" \
  "$FROZEN_ROOT/audit/P9_AUDIT.json" \
  "$FROZEN_ROOT/FROZEN_P9.json" \
  "$DELTA_ROOT/PREFLIGHT_DATA_COMPLETE.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

inputs = [Path(value).resolve(strict=True) for value in sys.argv[1:6]]
output = Path(sys.argv[6]).resolve(strict=False)
documents = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
if any(document.get("status", "PASS") != "PASS" for document in documents[:4]):
    raise SystemExit("one or more final P10 data receipts are not PASS")
receipt = {
    "schema": "stable_audio_tools.p10_preflight_data_complete",
    "schema_version": 1,
    "status": "PASS",
    "speech_timing_rows": 500_000,
    "sound_delta_rows": 20_000,
    "p10_training_started": False,
    "artifacts": [
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in inputs
    ],
}
temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(output)
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
