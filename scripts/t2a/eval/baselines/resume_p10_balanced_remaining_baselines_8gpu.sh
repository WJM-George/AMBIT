#!/usr/bin/env bash
set -euo pipefail

# Resume the two expensive baseline lanes after the fast systems have finished.
# Six GPUs take disjoint Stable Audio Open shards and two GPUs take disjoint
# Qwen3-TTS shards. Existing, fully audited outputs are skipped.
REPO_ROOT="${REPO_ROOT:-.}"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-./evaluation_benchmark/repos}"
ASSET_ROOT="${ASSET_ROOT:-${AMBIT_CKPT_ROOT}/baselines/p10_60k_15row_v1}"
SOURCE_EVAL_ROOT="${SOURCE_EVAL_ROOT:-${AMBIT_DATA_ROOT}/sceneplan_v2_1p124m/revisions/speech_expansion_noalign_15s_v1/evaluation/p10_v9_balanced_1200_ckpt20k_100k_v1}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SOURCE_EVAL_ROOT}/cross_system_baselines}"
MANIFEST="${MANIFEST:-${BENCHMARK_ROOT}/generation_requests.jsonl}"
ADAPTER_ROOT="${REPO_ROOT}/scripts/t2a/eval/baselines"
LOG_ROOT="${BENCHMARK_ROOT}/logs/resume_generation"
PIPELINE_LOG_ROOT="${SOURCE_EVAL_ROOT}/logs/full_benchmark"

mkdir -p "${LOG_ROOT}" "${PIPELINE_LOG_ROOT}"
test -f "${MANIFEST}"
test -f "${BENCHMARK_ROOT}/BENCHMARK_CONTRACT.json"
test -f "${SOURCE_EVAL_ROOT}/metrics/CORE_SUMMARY.json"

export HF_HOME="${ASSET_ROOT}/hf-cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export UV_CACHE_DIR="${ASSET_ROOT}/uv-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

pids=()
names=()
for shard in $(seq 0 5); do
  (
    cd "${UPSTREAM_ROOT}/stable-audio-tools"
    CUDA_VISIBLE_DEVICES="${shard}" \
    PYTHONPATH="${UPSTREAM_ROOT}/stable-audio-tools:${ADAPTER_ROOT}" \
    "${REPO_ROOT}/.venv/bin/python" \
      "${ADAPTER_ROOT}/generate_stable_audio_open.py" \
      --manifest "${MANIFEST}" --device cuda:0 \
      --shard-index "${shard}" --num-shards 6
  ) >"${LOG_ROOT}/stable_audio_open_1_0_shard_${shard}.log" 2>&1 &
  pids+=("$!")
  names+=("stable_audio_open_1_0_shard_${shard}")
done

for shard in 0 1; do
  gpu="$((shard + 6))"
  (
    cd "${UPSTREAM_ROOT}/Qwen3-TTS"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${UPSTREAM_ROOT}/Qwen3-TTS:${ADAPTER_ROOT}" \
    "${ASSET_ROOT}/envs/qwen3_tts/bin/python" \
      "${ADAPTER_ROOT}/generate_qwen3_tts.py" \
      --manifest "${MANIFEST}" --device cuda:0 \
      --shard-index "${shard}" --num-shards 2
  ) >"${LOG_ROOT}/qwen3_tts_1p7b_voice_design_shard_${shard}.log" 2>&1 &
  pids+=("$!")
  names+=("qwen3_tts_1p7b_voice_design_shard_${shard}")
done

: >"${LOG_ROOT}/generation_status.tsv"
failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[${index}]}"; then
    printf '%s\tPASS\n' "${names[${index}]}" | tee -a "${LOG_ROOT}/generation_status.tsv"
  else
    printf '%s\tFAIL\n' "${names[${index}]}" | tee -a "${LOG_ROOT}/generation_status.tsv"
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

PYTHONPATH="${ADAPTER_ROOT}" "${REPO_ROOT}/.venv/bin/python" \
  "${ADAPTER_ROOT}/audit_p10_matched_baseline_generation.py" \
  --benchmark-root "${BENCHMARK_ROOT}" --manifest "${MANIFEST}" \
  >"${PIPELINE_LOG_ROOT}/baseline_audit.log" 2>&1
touch "${BENCHMARK_ROOT}/BASELINE_GENERATION_COMPLETE"

CLAP_HF_HOME="${CLAP_HF_HOME:-./codex-home/.cache/huggingface}"
CUDA_VISIBLE_DEVICES=0 \
  HF_HOME="${CLAP_HF_HOME}" \
  HUGGINGFACE_HUB_CACHE="${CLAP_HF_HOME}/hub" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "${REPO_ROOT}/.venv/bin/python" \
  "${ADAPTER_ROOT}/score_p10_60k_cross_system.py" \
  --benchmark-root "${BENCHMARK_ROOT}" \
  --source-eval "${SOURCE_EVAL_ROOT}" --device-index 0 \
  >"${PIPELINE_LOG_ROOT}/score_cross_system.log" 2>&1

date --iso-8601=seconds >"${SOURCE_EVAL_ROOT}/FULL_BENCHMARK_COMPLETE"
echo "P10_BALANCED_FULL_BENCHMARK_COMPLETE=${SOURCE_EVAL_ROOT}"
