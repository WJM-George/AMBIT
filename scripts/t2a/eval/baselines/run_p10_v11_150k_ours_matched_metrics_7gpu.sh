#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-.}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${AMBIT_DATA_ROOT}/sceneplan_v2_1p124m/revisions/speech_expansion_noalign_15s_v1/evaluation/p10_v11_150k_stratified_test_3000_semantic_v2/cross_system_baselines}"
SOURCE_EVAL="${SOURCE_EVAL:-${BENCHMARK_ROOT%/cross_system_baselines}}"
SCORER="${REPO_ROOT}/scripts/t2a/eval/baselines/score_p10_v11_150k_ours_matched.py"
LOG_ROOT="${BENCHMARK_ROOT}/logs/ours_150k_matched_metrics"
SPEECH_SHARDS=4

mkdir -p "${LOG_ROOT}"
: >"${LOG_ROOT}/status.tsv"
cd "${REPO_ROOT}"
test -f "${BENCHMARK_ROOT}/PUBLIC_BASELINE_METRICS_COMPLETE"
test -f "${SOURCE_EVAL}/INFERENCE_COMPLETE"

export HF_HOME="${HF_HOME:-./codex-home/.cache/huggingface}"
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

common=(--benchmark-root "${BENCHMARK_ROOT}" --source-eval "${SOURCE_EVAL}")
start_job clap 1 "${PYTHON_BIN}" "${SCORER}" "${common[@]}" --arm clap --device-index 0
start_job vggish 2 "${PYTHON_BIN}" "${SCORER}" "${common[@]}" --arm vggish --device-index 0
start_job panns 3 "${PYTHON_BIN}" "${SCORER}" "${common[@]}" --arm panns --device-index 0
for shard in $(seq 0 $((SPEECH_SHARDS - 1))); do
  start_job "speech_shard_${shard}" "$((shard + 4))" \
    "${PYTHON_BIN}" "${SCORER}" "${common[@]}" --arm speech --device-index 0 \
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

"${PYTHON_BIN}" "${SCORER}" "${common[@]}" --arm merge \
  --num-shards "${SPEECH_SHARDS}" >"${LOG_ROOT}/merge.log" 2>&1
echo "P10_V11_OURS_AND_BASELINES_COMPLETE=${BENCHMARK_ROOT}"
