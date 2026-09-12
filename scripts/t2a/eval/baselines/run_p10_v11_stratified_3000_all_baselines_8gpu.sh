#!/usr/bin/env bash
set -euo pipefail

# Two balanced phases keep all eight GPUs busy while every runnable public
# baseline is regenerated on the frozen P10 v11 stratified 3k test contract.
REPO_ROOT="${REPO_ROOT:-/home/tanhe/dataset_storage/stable-audio-tools}"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-/home/tanhe/dataset_storage/evaluation_benchmark/repos}"
ASSET_ROOT="${ASSET_ROOT:-/mnt/sdc/ckpts/baselines/p10_60k_15row_v1}"
EVAL_ROOT="${EVAL_ROOT:-/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/speech_expansion_noalign_15s_v1/evaluation/p10_v11_150k_stratified_test_3000_semantic_v2}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${EVAL_ROOT}/cross_system_baselines}"
MANIFEST="${MANIFEST:-${BENCHMARK_ROOT}/generation_requests.jsonl}"
ADAPTER_ROOT="${REPO_ROOT}/scripts/t2a/eval/baselines"
LOG_ROOT="${BENCHMARK_ROOT}/logs/full_generation"
SHARE_GPU0_WITH_P11="${SHARE_GPU0_WITH_P11:-0}"

mkdir -p "${LOG_ROOT}"
test -f "${MANIFEST}"
test -f "${BENCHMARK_ROOT}/BENCHMARK_CONTRACT.json"
test -f "${BENCHMARK_ROOT}/BASELINE_SMOKE8_COMPLETE"

if [[ "${SHARE_GPU0_WITH_P11}" != "1" ]]; then
  while pgrep -af '[t]rain\.py .*--num-gpus' >/dev/null; do
    echo "WAIT_FOR_ACTIVE_TRAINING=$(date --iso-8601=seconds)"
    pgrep -af '[t]rain\.py .*--num-gpus' || true
    sleep 30
  done
else
  echo "SHARED_GPU_MODE=gpu0_reserved_for_p11,gpu1-7_public_baselines"
fi

export HF_HOME="${ASSET_ROOT}/hf-cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export UV_CACHE_DIR="${ASSET_ROOT}/uv-cache"
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

wait_jobs() {
  local phase="$1"
  local failed=0
  for index in "${!pids[@]}"; do
    if wait "${pids[${index}]}"; then
      printf '%s\t%s\tPASS\n' "${phase}" "${names[${index}]}" \
        | tee -a "${LOG_ROOT}/status.tsv"
    else
      printf '%s\t%s\tFAIL\n' "${phase}" "${names[${index}]}" \
        | tee -a "${LOG_ROOT}/status.tsv"
      failed=1
    fi
  done
  names=()
  pids=()
  if [[ "${failed}" -ne 0 ]]; then
    return 1
  fi
}

: >"${LOG_ROOT}/status.tsv"

# Phase 1: the three most expensive diffusion systems. In shared mode GPU 0 is
# reserved for P11 and the 2/3/2-way split is balanced over physical GPUs 1--7.
stable_gpus=(0 1)
maf_gpus=(2 3 4)
mmdit_gpus=(5 6 7)
mmdit_shards=3
if [[ "${SHARE_GPU0_WITH_P11}" == "1" ]]; then
  stable_gpus=(1 2)
  maf_gpus=(3 4 5)
  mmdit_gpus=(6 7)
  mmdit_shards=2
fi

for shard in 0 1; do
  start_job "stable_audio_open_1_0_shard_${shard}" "${stable_gpus[${shard}]}" env \
    PYTHONPATH="${UPSTREAM_ROOT}/stable-audio-tools:${ADAPTER_ROOT}" \
    "${REPO_ROOT}/.venv/bin/python" \
    "${ADAPTER_ROOT}/generate_stable_audio_open.py" \
    --manifest "${MANIFEST}" --device cuda:0 \
    --shard-index "${shard}" --num-shards 2
done

for shard in 0 1 2; do
  start_job "audiox_maf_shard_${shard}" "${maf_gpus[${shard}]}" env \
    PYTHONPATH="${UPSTREAM_ROOT}/AudioX:${ADAPTER_ROOT}" \
    "${ASSET_ROOT}/envs/audiox/bin/python" \
    "${ADAPTER_ROOT}/generate_audiox_family.py" \
    --manifest "${MANIFEST}" --baseline-id audiox_maf --device cuda:0 \
    --shard-index "${shard}" --num-shards 3
done

for shard in $(seq 0 $((mmdit_shards - 1))); do
  start_job "audiox_maf_mmdit_shard_${shard}" "${mmdit_gpus[${shard}]}" env \
    PYTHONPATH="${UPSTREAM_ROOT}/AudioX:${ADAPTER_ROOT}" \
    "${ASSET_ROOT}/envs/audiox/bin/python" \
    "${ADAPTER_ROOT}/generate_audiox_family.py" \
    --manifest "${MANIFEST}" --baseline-id audiox_maf_mmdit --device cuda:0 \
    --shard-index "${shard}" --num-shards "${mmdit_shards}"
done

wait_jobs phase1
date --iso-8601=seconds >"${BENCHMARK_ROOT}/BASELINE_PHASE1_COMPLETE"

# Phase 2: fast generalists plus the task-specific Sound and Speech anchors.
phase2_gpus=(0 1 2 3 4 5 6 7)
woosh_shards=3
if [[ "${SHARE_GPU0_WITH_P11}" == "1" ]]; then
  phase2_gpus=(1 2 3 4 5 6 7)
  woosh_shards=2
fi

start_job tangoflux "${phase2_gpus[0]}" env \
  PYTHONPATH="${UPSTREAM_ROOT}/TangoFlux:${ADAPTER_ROOT}" \
  "${ASSET_ROOT}/envs/tangoflux/bin/python" \
  "${ADAPTER_ROOT}/generate_tangoflux.py" \
  --manifest "${MANIFEST}" --device cuda:0

start_job audiox_turbo "${phase2_gpus[1]}" env \
  PYTHONPATH="${UPSTREAM_ROOT}/AudioX-Turbo:${ADAPTER_ROOT}" \
  "${ASSET_ROOT}/envs/audiox_turbo/bin/python" \
  "${ADAPTER_ROOT}/generate_audiox_turbo.py" \
  --manifest "${MANIFEST}" --device cuda:0

start_job mmaudio_large_44k_v2_text_only "${phase2_gpus[2]}" env \
  PYTHONPATH="${UPSTREAM_ROOT}/MMAudio:${ADAPTER_ROOT}" \
  "${ASSET_ROOT}/envs/mmaudio/bin/python" \
  "${ADAPTER_ROOT}/generate_mmaudio.py" \
  --manifest "${MANIFEST}" --device cuda:0

for shard in $(seq 0 $((woosh_shards - 1))); do
  start_job "woosh_flow_shard_${shard}" "${phase2_gpus[$((shard + 3))]}" env \
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    PYTHONPATH="${UPSTREAM_ROOT}/Woosh:${ADAPTER_ROOT}" \
    "${ASSET_ROOT}/envs/woosh/bin/python" \
    "${ADAPTER_ROOT}/generate_woosh.py" \
    --manifest "${MANIFEST}" --device cuda:0 \
    --shard-index "${shard}" --num-shards "${woosh_shards}"
done

for shard in 0 1; do
  qwen_gpu_offset=$((shard + 3 + woosh_shards))
  start_job "qwen3_tts_1p7b_voice_design_shard_${shard}" "${phase2_gpus[${qwen_gpu_offset}]}" env \
    PYTHONPATH="${UPSTREAM_ROOT}/Qwen3-TTS:${ADAPTER_ROOT}" \
    "${ASSET_ROOT}/envs/qwen3_tts/bin/python" \
    "${ADAPTER_ROOT}/generate_qwen3_tts.py" \
    --manifest "${MANIFEST}" --device cuda:0 \
    --shard-index "${shard}" --num-shards 2
done

wait_jobs phase2
date --iso-8601=seconds >"${BENCHMARK_ROOT}/BASELINE_PHASE2_COMPLETE"

PYTHONPATH="${ADAPTER_ROOT}" "${REPO_ROOT}/.venv/bin/python" \
  "${ADAPTER_ROOT}/audit_p10_matched_baseline_generation.py" \
  --benchmark-root "${BENCHMARK_ROOT}" --manifest "${MANIFEST}" \
  >"${LOG_ROOT}/audit.log" 2>&1

date --iso-8601=seconds >"${BENCHMARK_ROOT}/BASELINE_GENERATION_COMPLETE"
echo "P10_V11_ALL_BASELINES_COMPLETE=${BENCHMARK_ROOT}"
