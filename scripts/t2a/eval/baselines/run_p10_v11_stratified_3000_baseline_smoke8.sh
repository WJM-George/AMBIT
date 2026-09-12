#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tanhe/dataset_storage/stable-audio-tools}"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-/home/tanhe/dataset_storage/evaluation_benchmark/repos}"
ASSET_ROOT="${ASSET_ROOT:-/mnt/sdc/ckpts/baselines/p10_60k_15row_v1}"
EVAL_ROOT="${EVAL_ROOT:-/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/speech_expansion_noalign_15s_v1/evaluation/p10_v11_150k_stratified_test_3000_semantic_v2}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${EVAL_ROOT}/cross_system_baselines}"
MANIFEST="${MANIFEST:-${BENCHMARK_ROOT}/smoke_requests.jsonl}"
ADAPTER_ROOT="${REPO_ROOT}/scripts/t2a/eval/baselines"
LOG_ROOT="${BENCHMARK_ROOT}/logs/smoke8"

mkdir -p "${LOG_ROOT}"
test -f "${MANIFEST}"
test -f "${BENCHMARK_ROOT}/BENCHMARK_CONTRACT.json"

export HF_HOME="${ASSET_ROOT}/hf-cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export UV_CACHE_DIR="${ASSET_ROOT}/uv-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

names=()
pids=()

launch() {
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

launch stable_audio_open_1_0 0 env \
  PYTHONPATH="${UPSTREAM_ROOT}/stable-audio-tools:${ADAPTER_ROOT}" \
  "${REPO_ROOT}/.venv/bin/python" \
  "${ADAPTER_ROOT}/generate_stable_audio_open.py" \
  --manifest "${MANIFEST}" --device cuda:0

launch tangoflux 1 env \
  PYTHONPATH="${UPSTREAM_ROOT}/TangoFlux:${ADAPTER_ROOT}" \
  "${ASSET_ROOT}/envs/tangoflux/bin/python" \
  "${ADAPTER_ROOT}/generate_tangoflux.py" \
  --manifest "${MANIFEST}" --device cuda:0

launch audiox_turbo 2 env \
  PYTHONPATH="${UPSTREAM_ROOT}/AudioX-Turbo:${ADAPTER_ROOT}" \
  "${ASSET_ROOT}/envs/audiox_turbo/bin/python" \
  "${ADAPTER_ROOT}/generate_audiox_turbo.py" \
  --manifest "${MANIFEST}" --device cuda:0

launch audiox_maf 3 env \
  PYTHONPATH="${UPSTREAM_ROOT}/AudioX:${ADAPTER_ROOT}" \
  "${ASSET_ROOT}/envs/audiox/bin/python" \
  "${ADAPTER_ROOT}/generate_audiox_family.py" \
  --manifest "${MANIFEST}" --baseline-id audiox_maf --device cuda:0

launch audiox_maf_mmdit 4 env \
  PYTHONPATH="${UPSTREAM_ROOT}/AudioX:${ADAPTER_ROOT}" \
  "${ASSET_ROOT}/envs/audiox/bin/python" \
  "${ADAPTER_ROOT}/generate_audiox_family.py" \
  --manifest "${MANIFEST}" --baseline-id audiox_maf_mmdit --device cuda:0

launch mmaudio_large_44k_v2_text_only 5 env \
  PYTHONPATH="${UPSTREAM_ROOT}/MMAudio:${ADAPTER_ROOT}" \
  "${ASSET_ROOT}/envs/mmaudio/bin/python" \
  "${ADAPTER_ROOT}/generate_mmaudio.py" \
  --manifest "${MANIFEST}" --device cuda:0

launch woosh_flow 6 env \
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  PYTHONPATH="${UPSTREAM_ROOT}/Woosh:${ADAPTER_ROOT}" \
  "${ASSET_ROOT}/envs/woosh/bin/python" \
  "${ADAPTER_ROOT}/generate_woosh.py" \
  --manifest "${MANIFEST}" --device cuda:0

launch qwen3_tts_1p7b_voice_design 7 env \
  PYTHONPATH="${UPSTREAM_ROOT}/Qwen3-TTS:${ADAPTER_ROOT}" \
  "${ASSET_ROOT}/envs/qwen3_tts/bin/python" \
  "${ADAPTER_ROOT}/generate_qwen3_tts.py" \
  --manifest "${MANIFEST}" --device cuda:0

: >"${LOG_ROOT}/status.tsv"
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
date --iso-8601=seconds >"${BENCHMARK_ROOT}/BASELINE_SMOKE8_COMPLETE"
echo "P10_V11_BASELINE_SMOKE8_COMPLETE=${BENCHMARK_ROOT}"
