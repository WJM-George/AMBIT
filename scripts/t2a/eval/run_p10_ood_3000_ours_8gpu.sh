#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tanhe/dataset_storage/stable-audio-tools}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/mnt/sdb/audio_dataset/evaluation_benchmark/p10_ood_3000_v1/cross_system_benchmark}"
LOG_ROOT="${BENCHMARK_ROOT}/logs/ours_p10_150k_8gpu"
GPU_IDS_CSV="${GPU_IDS:-1,2,3,4,5,6,7}"
mkdir -p "${LOG_ROOT}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

IFS=',' read -r -a GPU_IDS_ARRAY <<<"${GPU_IDS_CSV}"
NUM_GPUS="${#GPU_IDS_ARRAY[@]}"
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "at least one OOD inference GPU is required" >&2
  exit 2
fi

pids=()
for shard in $(seq 0 $((NUM_GPUS - 1))); do
  gpu="${GPU_IDS_ARRAY[${shard}]}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${REPO_ROOT}/.venv/bin/python" \
      "${REPO_ROOT}/scripts/t2a/eval/generate_p10_ood_3000.py" \
      --benchmark-root "${BENCHMARK_ROOT}" \
      --device cuda:0 \
      --shard-index "${shard}" \
      --num-shards "${NUM_GPUS}" \
      --batch-size 4
  ) >"${LOG_ROOT}/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[${index}]}"; then
    printf 'shard_%s\tPASS\n' "${index}" | tee -a "${LOG_ROOT}/status.tsv"
  else
    printf 'shard_%s\tFAIL\n' "${index}" | tee -a "${LOG_ROOT}/status.tsv"
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

"${REPO_ROOT}/.venv/bin/python" \
  "${REPO_ROOT}/scripts/t2a/eval/audit_p10_ood_3000_generation.py" \
  --benchmark-root "${BENCHMARK_ROOT}" \
  >"${LOG_ROOT}/audit.log" 2>&1
date --iso-8601=seconds >"${BENCHMARK_ROOT}/P10_GENERATION_COMPLETE"
echo "P10_OOD_GENERATION_COMPLETE=${BENCHMARK_ROOT}"
