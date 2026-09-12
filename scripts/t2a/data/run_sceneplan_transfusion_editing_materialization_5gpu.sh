#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 {train|validation|test} [keep_foa|cleanup]" >&2
  exit 2
fi

SPLIT="$1"
RETENTION="${2:-cleanup}"
case "$SPLIT" in
  train) ROWS=1000000 ;;
  validation) ROWS=20000 ;;
  test) ROWS=5000 ;;
  *) echo "invalid split: $SPLIT" >&2; exit 2 ;;
esac
case "$RETENTION" in
  keep_foa|cleanup) ;;
  *) echo "invalid retention policy: $RETENTION" >&2; exit 2 ;;
esac

ROOT=/mnt/sdb/audio_dataset/sceneplan_transfusion_editing_v1
PAIR_INDEX="$ROOT/pair_index/$SPLIT.sqlite"
LOG_ROOT="$ROOT/materialized/logs/$SPLIT"
PYTHON=/mnt/sdc/stable-audio-tools-venv/bin/python
WORKER=/mnt/sdc/stable-audio-tools-workspace/scripts/t2a/data/materialize_sceneplan_transfusion_editing_worker.py
TOTAL_SHARDS=$(( (ROWS + 1023) / 1024 ))

[[ -f "$PAIR_INDEX" ]] || { echo "missing pair index: $PAIR_INDEX" >&2; exit 1; }
[[ -f "$WORKER" ]] || { echo "missing worker: $WORKER" >&2; exit 1; }
mkdir -p "$LOG_ROOT"

# A duplicate launcher would race on the same per-row render paths, shard
# publications, and cleanup directories.  Detect an already-running worker
# (including one started by an older launcher without this lock), then hold a
# split-scoped advisory lock across all five workers for future race-free
# resumes.
if pgrep -f "$WORKER.*--pair-index $PAIR_INDEX" >/dev/null 2>&1; then
  echo "materialization workers are already active for split=$SPLIT" >&2
  exit 1
fi
LOCK_ROOT="$ROOT/materialized/locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/$SPLIT.lock"
if ! flock -n 9; then
  echo "another materialization launcher holds the split=$SPLIT lock" >&2
  exit 1
fi
ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"

# The worker addresses physical CUDA indices directly. Never remap them.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

EXTRA_ARGS=()
PREFETCH_RENDER="${EDITING_PREFETCH_RENDER:-0}"
if [[ "$PREFETCH_RENDER" != 0 && "$PREFETCH_RENDER" != 1 ]]; then
  echo "EDITING_PREFETCH_RENDER must be 0 or 1" >&2
  exit 2
fi
if [[ "$PREFETCH_RENDER" == 1 ]]; then
  EXTRA_ARGS+=(--prefetch-render)
fi
if [[ -n "${EDITING_PREFETCH_CONTROL:-}" ]]; then
  [[ -r "$EDITING_PREFETCH_CONTROL" ]] || { echo "missing prefetch control" >&2; exit 2; }
  EXTRA_ARGS+=(--prefetch-control "$EDITING_PREFETCH_CONTROL")
fi
if [[ "$RETENTION" == cleanup ]]; then
  EXTRA_ARGS+=(--cleanup-foa --cleanup-render-work)
fi

ASSIGNMENTS=()
if [[ -n "${EDITING_MATERIALIZATION_ASSIGNMENTS:-}" ]]; then
  # Validate the entire static partition before starting any GPU process.
  # Existing workers must already be absent and this launcher owns the lock.
  ASSIGNMENT_TSV="$("$PYTHON" - "$EDITING_MATERIALIZATION_ASSIGNMENTS" \
    "$PAIR_INDEX" "$SPLIT" "$TOTAL_SHARDS" "$LOG_ROOT/assignments-$ATTEMPT_ID.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

plan_path, index_path, split, total, snapshot_path = sys.argv[1:]
index_path = Path(index_path).resolve(strict=True)
total = int(total)
plan = json.loads(Path(plan_path).read_text())
digest = hashlib.sha256()
with index_path.open('rb') as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        digest.update(block)
if (plan.get('schema') != 'editing_materialization_assignments_v1'
        or plan.get('split') != split or plan.get('total_shards') != total
        or Path(plan.get('pair_index', '')).resolve() != index_path
        or plan.get('pair_index_sha256') != digest.hexdigest()):
    raise SystemExit('materialization assignment plan does not bind this split/index')
assignments = plan.get('assignments')
if not isinstance(assignments, list) or len(assignments) != 5:
    raise SystemExit('assignment plan must contain five physical GPU assignments')
coverage = [0] * total
gpus = []
lines = []
for item in assignments:
    gpu, start, stop, stride = (item.get(key) for key in ('gpu', 'start', 'stop', 'stride'))
    if (any(type(value) is not int for value in (gpu, start, stop, stride))
            or gpu not in range(3, 8) or not 0 <= start < stop <= total or stride <= 0):
        raise SystemExit('invalid physical GPU or shard bounds in assignment plan')
    gpus.append(gpu)
    for shard in range(start, stop, stride):
        coverage[shard] += 1
    lines.append(f'{gpu} {start} {stop} {stride}')
if sorted(gpus) != list(range(3, 8)) or any(count != 1 for count in coverage):
    raise SystemExit('assignment plan must cover every shard exactly once on GPUs 3-7')
Path(snapshot_path).write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
print('\n'.join(lines))
PY
)"
  mapfile -t ASSIGNMENTS <<<"$ASSIGNMENT_TSV"
else
  for SLOT in 0 1 2 3 4; do
    ASSIGNMENTS+=("$((3 + SLOT)) $SLOT $TOTAL_SHARDS 5")
  done
fi

pids=()
for ASSIGNMENT in "${ASSIGNMENTS[@]}"; do
  read -r GPU SHARD_START SHARD_STOP SHARD_STRIDE <<<"$ASSIGNMENT"
  LOG="$LOG_ROOT/gpu-${GPU}.log"
  (
    printf '{"event":"materialization_attempt_start","attempt_id":"%s","split":"%s","physical_gpu":%d}\n' \
      "$ATTEMPT_ID" "$SPLIT" "$GPU"
    exec "$PYTHON" -u "$WORKER" \
      --pair-index "$PAIR_INDEX" \
      --gpu "$GPU" \
      --shard-start "$SHARD_START" \
      --shard-stop "$SHARD_STOP" \
      --shard-stride "$SHARD_STRIDE" \
      --jobs 20 \
      --batch-size 8 \
      "${EXTRA_ARGS[@]}"
  ) >>"$LOG" 2>&1 &
  pids+=("$!")
  echo "started split=$SPLIT physical_gpu=$GPU pid=$! log=$LOG"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "one or more $SPLIT materialization workers failed" >&2
  exit "$status"
fi
echo "completed split=$SPLIT shards=$TOTAL_SHARDS rows=$ROWS retention=$RETENTION"
