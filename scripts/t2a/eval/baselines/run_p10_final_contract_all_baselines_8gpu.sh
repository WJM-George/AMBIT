#!/usr/bin/env bash
set -euo pipefail

# Run one baseline at a time with one persistent model replica on each selected
# GPU.  GPU0 is reserved for the approved P11 workload by default; GPU_IDS can
# override the seven-GPU evaluation set.  Resumption is per output waveform.
REPO_ROOT="${REPO_ROOT:-/home/tanhe/dataset_storage/stable-audio-tools}"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-/home/tanhe/dataset_storage/evaluation_benchmark/repos}"
ASSET_ROOT="${ASSET_ROOT:-/mnt/sdc/ckpts/baselines/p10_60k_15row_v1}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:?BENCHMARK_ROOT is required}"
MANIFEST="${MANIFEST:-${BENCHMARK_ROOT}/generation_requests.jsonl}"
ADAPTER_ROOT="${REPO_ROOT}/scripts/t2a/eval/baselines"
LOG_ROOT="${BENCHMARK_ROOT}/logs/final_generation_8gpu"
GPU_IDS_CSV="${GPU_IDS:-1,2,3,4,5,6,7}"

IFS=',' read -r -a GPU_IDS <<<"${GPU_IDS_CSV}"
NUM_GPUS="${#GPU_IDS[@]}"
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "at least one benchmark GPU is required" >&2
  exit 2
fi
test -f "${MANIFEST}"
test -f "${BENCHMARK_ROOT}/BENCHMARK_CONTRACT.json"
mkdir -p "${LOG_ROOT}"

export HF_HOME="${ASSET_ROOT}/hf-cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export UV_CACHE_DIR="${ASSET_ROOT}/uv-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

run_sharded() {
  local baseline_id="$1"
  local python_bin="$2"
  local python_path="$3"
  local adapter="$4"
  shift 4
  local marker="${BENCHMARK_ROOT}/GENERATION_${baseline_id}_COMPLETE"
  local -a pids=()
  local -a names=()
  local failed=0

  echo "START ${baseline_id} $(date --iso-8601=seconds)" | tee -a "${LOG_ROOT}/status.tsv"
  for shard in $(seq 0 $((NUM_GPUS - 1))); do
    local gpu="${GPU_IDS[${shard}]}"
    local name="${baseline_id}_gpu_${gpu}_shard_${shard}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export PYTHONPATH="${python_path}:${ADAPTER_ROOT}"
      "${python_bin}" "${ADAPTER_ROOT}/${adapter}" \
        --manifest "${MANIFEST}" \
        --device cuda:0 \
        --shard-index "${shard}" \
        --num-shards "${NUM_GPUS}" \
        "$@"
    ) >"${LOG_ROOT}/${name}.log" 2>&1 &
    pids+=("$!")
    names+=("${name}")
  done
  for index in "${!pids[@]}"; do
    if wait "${pids[${index}]}"; then
      printf '%s\t%s\tPASS\n' "${baseline_id}" "${names[${index}]}" \
        | tee -a "${LOG_ROOT}/status.tsv"
    else
      printf '%s\t%s\tFAIL\n' "${baseline_id}" "${names[${index}]}" \
        | tee -a "${LOG_ROOT}/status.tsv"
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "one or more ${baseline_id} workers failed" >&2
    return 1
  fi
  date --iso-8601=seconds >"${marker}"
  echo "COMPLETE ${baseline_id} $(date --iso-8601=seconds)" | tee -a "${LOG_ROOT}/status.tsv"
}

: >"${LOG_ROOT}/status.tsv"

run_sharded audiox_maf_mmdit \
  "${ASSET_ROOT}/envs/audiox/bin/python" \
  "${UPSTREAM_ROOT}/AudioX" \
  generate_audiox_family.py \
  --baseline-id audiox_maf_mmdit

run_sharded audiox_maf \
  "${ASSET_ROOT}/envs/audiox/bin/python" \
  "${UPSTREAM_ROOT}/AudioX" \
  generate_audiox_family.py \
  --baseline-id audiox_maf

run_sharded stable_audio_open_1_0 \
  "${REPO_ROOT}/.venv/bin/python" \
  "${UPSTREAM_ROOT}/stable-audio-tools" \
  generate_stable_audio_open.py

run_sharded tangoflux \
  "${ASSET_ROOT}/envs/tangoflux/bin/python" \
  "${UPSTREAM_ROOT}/TangoFlux" \
  generate_tangoflux.py

run_sharded audiox_turbo \
  "${ASSET_ROOT}/envs/audiox_turbo/bin/python" \
  "${UPSTREAM_ROOT}/AudioX-Turbo" \
  generate_audiox_turbo.py

run_sharded mmaudio_large_44k_v2_text_only \
  "${ASSET_ROOT}/envs/mmaudio/bin/python" \
  "${UPSTREAM_ROOT}/MMAudio" \
  generate_mmaudio.py

HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 run_sharded woosh_flow \
  "${ASSET_ROOT}/envs/woosh/bin/python" \
  "${UPSTREAM_ROOT}/Woosh" \
  generate_woosh.py

run_sharded qwen3_tts_1p7b_voice_design \
  "${ASSET_ROOT}/envs/qwen3_tts/bin/python" \
  "${UPSTREAM_ROOT}/Qwen3-TTS" \
  generate_qwen3_tts.py

PYTHONPATH="${ADAPTER_ROOT}" "${REPO_ROOT}/.venv/bin/python" \
  "${ADAPTER_ROOT}/audit_p10_matched_baseline_generation.py" \
  --benchmark-root "${BENCHMARK_ROOT}" \
  --manifest "${MANIFEST}" \
  >"${LOG_ROOT}/audit.log" 2>&1

date --iso-8601=seconds >"${BENCHMARK_ROOT}/BASELINE_GENERATION_COMPLETE"
echo "P10_FINAL_BASELINE_GENERATION_COMPLETE=${BENCHMARK_ROOT}"
