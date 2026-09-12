#!/usr/bin/env bash
set -euo pipefail

# Durable Editing-only entry point:
#   paired target materialization -> frozen train index -> DiT/AR/E2E chain.
# Every stage remains independently fail-closed; this wrapper only makes the
# already-published boundaries safely re-entrant after an external stop.

REPO="${P10_REPO:-${AMBIT_CKPT_ROOT}/stable-audio-tools-workspace}"
ROOT="${EDITING_DATA_ROOT:-${AMBIT_DATA_ROOT}/sceneplan_transfusion_editing_v1}"
PAIR_INDEX="$ROOT/pair_index/train.sqlite"
TRAIN_INDEX="$ROOT/training_index/train.sqlite"
TRAIN_MARKER="$TRAIN_INDEX.frozen.json"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[editing-full-chain] set CUDA_VISIBLE_DEVICES to the GPUs for this job" >&2
    exit 2
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "$REPO"

for required in \
  "$PAIR_INDEX" \
  "$ROOT/training_index/validation.sqlite" \
  "$ROOT/training_index/validation.sqlite.frozen.json" \
  "$ROOT/training_index/test.sqlite" \
  "$ROOT/training_index/test.sqlite.frozen.json"; do
  if [[ ! -r "$required" ]]; then
    echo "[editing-full-chain] required Editing artifact is missing: $required" >&2
    exit 1
  fi
done

LOCK_ROOT="$ROOT/materialized/locks"
mkdir -p "$LOCK_ROOT"
exec 8>"$LOCK_ROOT/full-chain.lock"
if ! flock -n 8; then
  echo "[editing-full-chain] another full Editing chain is already active" >&2
  exit 1
fi

if [[ ! -e "$TRAIN_INDEX" && ! -e "$TRAIN_MARKER" ]]; then
  "$REPO/scripts/t2a/data/run_sceneplan_transfusion_editing_materialization_5gpu.sh" \
    train cleanup
  "$REPO/.venv/bin/python" \
    "$REPO/scripts/t2a/data/finalize_sceneplan_transfusion_editing_index.py" \
    --planned-pair-index "$PAIR_INDEX"
elif [[ -r "$TRAIN_INDEX" && -r "$TRAIN_MARKER" ]]; then
  # This is only the atomic-publication check needed to decide whether the
  # destructive finalizer must remain skipped.  The invoked DiT preflight
  # immediately performs the full index, source-shard, and target-shard audit.
  "$REPO/.venv/bin/python" - "$TRAIN_INDEX" "$TRAIN_MARKER" <<'PY'
import json
import sqlite3
import sys
from pathlib import Path

from stable_audio_tools.data.sceneplan_transfusion_editing_index import sha256_file

index = Path(sys.argv[1]).resolve(strict=True)
marker_path = Path(sys.argv[2]).resolve(strict=True)
marker = json.loads(marker_path.read_text(encoding="utf-8"))
connection = sqlite3.connect(f"file:{index}?mode=ro&immutable=1", uri=True)
try:
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    rows = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])
finally:
    connection.close()
observed_sha = sha256_file(index)
if not (
    metadata.get("schema") == "sceneplan_transfusion_editing_training_index"
    and metadata.get("schema_version") == "1"
    and metadata.get("state") == "materialized_complete_frozen"
    and metadata.get("split") == "train"
    and int(metadata.get("rows", -1)) == rows == 1_000_000
    and marker.get("schema") == "sceneplan_transfusion_editing_training_index"
    and int(marker.get("schema_version", -1)) == 1
    and marker.get("state") == "materialized_complete_frozen"
    and marker.get("split") == "train"
    and int(marker.get("rows", -1)) == rows
    and Path(marker.get("index_path", "")).resolve() == index
    and marker.get("index_sha256") == observed_sha
):
    raise SystemExit("existing frozen train-index publication is invalid")
PY
else
  echo "[editing-full-chain] train index/marker publication is incomplete" >&2
  exit 1
fi

exec "$REPO/scripts/t2a/train/run_sceneplan_transfusion_editing_dit_full_5gpu.sh"
