#!/usr/bin/env bash
set -euo pipefail

# Multiple independent batch-1 replicas avoid the variable-length padding and
# low device occupancy observed with a single larger inference batch.
REPO_ROOT="${REPO_ROOT:-.}"
EVAL_ROOT="${EVAL_ROOT:-${AMBIT_DATA_ROOT}/sceneplan_v2_1p124m/revisions/speech_expansion_noalign_15s_v1/evaluation/p10_v11_150k_stratified_test_3000_semantic_v2}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-150000}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7}"
REPLICAS_PER_GPU="${REPLICAS_PER_GPU:-4}"
LOG_ROOT="${EVAL_ROOT}/logs/inference_maxthroughput"

IFS=',' read -r -a GPU_ID_LIST <<<"${GPU_IDS_CSV}"
NUM_GPUS="${#GPU_ID_LIST[@]}"
NUM_SHARDS="${NUM_SHARDS:-$((NUM_GPUS * REPLICAS_PER_GPU))}"
if [[ "${NUM_GPUS}" -lt 1 || "${NUM_SHARDS}" -lt 1 ]]; then
  echo "GPU_IDS and NUM_SHARDS must be non-empty and positive" >&2
  exit 2
fi
if [[ "${REPLICAS_PER_GPU}" -lt 1 ]]; then
  echo "REPLICAS_PER_GPU must be positive" >&2
  exit 2
fi
if [[ "${ALLOW_CONCURRENT_TRAINING:-0}" != "1" ]] \
  && pgrep -af '(^|/)train.py .*--num-gpus' >/dev/null; then
  echo "refusing to evaluate while a training process is active" >&2
  pgrep -af '(^|/)train.py .*--num-gpus' >&2 || true
  exit 2
fi
test -f "${EVAL_ROOT}/EVAL_CONTRACT.json"
mkdir -p "${LOG_ROOT}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

pids=()
names=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu_slot="$((shard % NUM_GPUS))"
    gpu="${GPU_ID_LIST[${gpu_slot}]}"
    replica="$((shard / NUM_GPUS))"
    name="gpu_${gpu}_replica_${replica}_shard_${shard}"
    log="${LOG_ROOT}/${name}.log"
    (
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
        "${REPO_ROOT}/scripts/t2a/eval/generate_sceneplan_dit_p10_panel.py" \
        --eval-root "${EVAL_ROOT}" \
        --checkpoint-step "${CHECKPOINT_STEP}" \
        --device cuda:0 \
        --shard-index "${shard}" \
        --num-shards "${NUM_SHARDS}" \
        --batch-size 1
    ) >"${log}" 2>&1 &
    pids+=("$!")
    names+=("${name}")
    sleep 0.25
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[${index}]}"; then
    printf '%s\tPASS\n' "${names[${index}]}" >>"${LOG_ROOT}/worker_status.tsv"
  else
    printf '%s\tFAIL\n' "${names[${index}]}" >>"${LOG_ROOT}/worker_status.tsv"
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more max-throughput workers failed" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${EVAL_ROOT}" "${CHECKPOINT_STEP}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
step = int(sys.argv[2])
contract = json.loads((root / "EVAL_CONTRACT.json").read_text(encoding="utf-8"))
expected = int(contract["test_set"]["evaluation_rows"])
outputs = list((root / "outputs" / f"step_{step:06d}").rglob("metadata.json"))
if len(outputs) != expected:
    raise RuntimeError(f"inference output count changed: {len(outputs)} != {expected}")
for path in outputs:
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("status") != "PASS" or int(row["checkpoint_step"]) != step:
        raise RuntimeError(f"invalid inference output: {path}")
print(json.dumps({"status": "PASS", "step": step, "outputs": len(outputs)}))
PY

date --iso-8601=seconds >"${EVAL_ROOT}/INFERENCE_COMPLETE"
echo "P10_STRATIFIED_3000_INFERENCE_COMPLETE=${EVAL_ROOT}"
