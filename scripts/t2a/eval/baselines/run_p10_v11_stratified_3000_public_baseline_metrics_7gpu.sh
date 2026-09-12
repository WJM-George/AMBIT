#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tanhe/dataset_storage/stable-audio-tools}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/speech_expansion_noalign_15s_v1/evaluation/p10_v11_150k_stratified_test_3000_semantic_v2/cross_system_baselines}"
SCORER="${REPO_ROOT}/scripts/t2a/eval/baselines/score_p10_v11_stratified_3000_public_baselines.py"
LOG_ROOT="${BENCHMARK_ROOT}/logs/public_baseline_metrics"
SPEECH_SHARDS=4

mkdir -p "${LOG_ROOT}"
: >"${LOG_ROOT}/status.tsv"
cd "${REPO_ROOT}"
test -f "${BENCHMARK_ROOT}/BASELINE_GENERATION_COMPLETE"

export HF_HOME="${HF_HOME:-/home/tanhe/dataset_storage/codex-home/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

declare -a names=()
declare -a pids=()

start_job() {
  local name="$1"
  local gpu="$2"
  shift 2
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "$@"
  ) >"${LOG_ROOT}/${name}.log" 2>&1 &
  names+=("${name}")
  pids+=("$!")
}

# GPU 0 remains available to P11.  The three audio feature families and four
# Speech shards occupy physical GPUs 1--7 independently.
start_job clap 1 "${PYTHON_BIN}" "${SCORER}" \
  --benchmark-root "${BENCHMARK_ROOT}" --arm clap --device-index 0
start_job vggish 2 "${PYTHON_BIN}" "${SCORER}" \
  --benchmark-root "${BENCHMARK_ROOT}" --arm vggish --device-index 0
start_job panns 3 "${PYTHON_BIN}" "${SCORER}" \
  --benchmark-root "${BENCHMARK_ROOT}" --arm panns --device-index 0
for shard in $(seq 0 $((SPEECH_SHARDS - 1))); do
  start_job "speech_shard_${shard}" "$((shard + 4))" "${PYTHON_BIN}" "${SCORER}" \
    --benchmark-root "${BENCHMARK_ROOT}" --arm speech --device-index 0 \
    --shard-index "${shard}" --num-shards "${SPEECH_SHARDS}"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[${index}]}"; then
    printf '%s\tPASS\n' "${names[${index}]}" | tee -a "${LOG_ROOT}/status.tsv"
  else
    printf '%s\tFAIL\n' "${names[${index}]}" | tee -a "${LOG_ROOT}/status.tsv"
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

"${PYTHON_BIN}" "${SCORER}" \
  --benchmark-root "${BENCHMARK_ROOT}" --arm merge \
  --num-shards "${SPEECH_SHARDS}" \
  >"${LOG_ROOT}/merge.log" 2>&1

date --iso-8601=seconds >"${BENCHMARK_ROOT}/PUBLIC_BASELINE_METRICS_COMPLETE"
echo "P10_V11_PUBLIC_BASELINE_METRICS_COMPLETE=${BENCHMARK_ROOT}"
