#!/usr/bin/env bash
set -euo pipefail

# Eight persistent workers evaluate one disjoint eighth of any frozen P10
# panel at every checkpoint listed by its contract.  Work stays balanced across
# GPUs and every sample retains the same frozen noise at all checkpoints.
REPO_ROOT="${REPO_ROOT:-.}"
EVAL_ROOT="${EVAL_ROOT:-${AMBIT_DATA_ROOT}/sceneplan_v2_1p124m/revisions/speech_expansion_noalign_15s_v1/evaluation/p10_v9_balanced_1200_ckpt20k_100k_v1}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
NUM_SHARDS="${NUM_SHARDS:-8}"

if [[ "${NUM_SHARDS}" -ne 8 ]]; then
  echo "the frozen P10 launcher requires exactly eight shards" >&2
  exit 2
fi
if pgrep -af '/train.py .*--num-gpus' >/dev/null; then
  echo "refusing to evaluate while distributed training is active" >&2
  pgrep -af '/train.py .*--num-gpus' >&2 || true
  exit 2
fi
test -f "${EVAL_ROOT}/EVAL_CONTRACT.json"
mapfile -t CHECKPOINT_STEPS < <(
  "${PYTHON_BIN}" -c \
    'import json,sys; print("\n".join(str(x["step"]) for x in json.load(open(sys.argv[1]))["checkpoints"]))' \
    "${EVAL_ROOT}/EVAL_CONTRACT.json"
)
if [[ "${#CHECKPOINT_STEPS[@]}" -eq 0 ]]; then
  echo "evaluation contract has no checkpoints" >&2
  exit 2
fi
mkdir -p "${EVAL_ROOT}/logs/inference"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

pids=()
for gpu in $(seq 0 7); do
  (
    for step in "${CHECKPOINT_STEPS[@]}"; do
      log="${EVAL_ROOT}/logs/inference/gpu_${gpu}_step_${step}.log"
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
        "${REPO_ROOT}/scripts/t2a/eval/generate_sceneplan_dit_p10_panel.py" \
        --eval-root "${EVAL_ROOT}" \
        --checkpoint-step "${step}" \
        --device cuda:0 \
        --shard-index "${gpu}" \
        --num-shards "${NUM_SHARDS}" \
        --batch-size 1 >"${log}" 2>&1
    done
    printf 'PASS\n' >"${EVAL_ROOT}/logs/inference/gpu_${gpu}.status"
  ) &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[${index}]}"; then
    printf 'FAIL\n' >"${EVAL_ROOT}/logs/inference/gpu_${index}.status"
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more frozen-panel inference workers failed" >&2
  exit 1
fi

date --iso-8601=seconds >"${EVAL_ROOT}/INFERENCE_COMPLETE"
echo "P10_FROZEN_PANEL_INFERENCE_COMPLETE=${EVAL_ROOT}"
