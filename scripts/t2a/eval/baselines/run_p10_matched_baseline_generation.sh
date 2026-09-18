#!/usr/bin/env bash
set -euo pipefail

# Generate every locally reproducible public baseline on one frozen P10 panel.
# Each incompatible upstream model keeps its own pinned environment.
REPO_ROOT="${REPO_ROOT:-.}"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-./evaluation_benchmark/repos}"
ASSET_ROOT="${ASSET_ROOT:-${AMBIT_CKPT_ROOT}/baselines/p10_60k_15row_v1}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${AMBIT_CKPT_ROOT}/baselines/p10_v7_40k_140k_50x3_v1}"
MANIFEST="${MANIFEST:-${BENCHMARK_ROOT}/generation_requests.jsonl}"
ADAPTER_ROOT="${REPO_ROOT}/scripts/t2a/eval/baselines"
LOG_ROOT="${BENCHMARK_ROOT}/logs"

mkdir -p "${LOG_ROOT}"
: >"${LOG_ROOT}/generation_status.tsv"

if pgrep -af '/train.py .*--num-gpus' >/dev/null; then
  echo "refusing to run baselines while distributed training is active" >&2
  pgrep -af '/train.py .*--num-gpus' >&2 || true
  exit 2
fi

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

names=(
  stable_audio_open_1_0
  tangoflux
  audiox_turbo
  mmaudio_large_44k_v2_text_only
  woosh_flow
  qwen3_tts_1p7b_voice_design
)

(
  cd "${UPSTREAM_ROOT}/stable-audio-tools"
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="${UPSTREAM_ROOT}/stable-audio-tools:${ADAPTER_ROOT}" \
  "${REPO_ROOT}/.venv/bin/python" \
    "${ADAPTER_ROOT}/generate_stable_audio_open.py" \
    --manifest "${MANIFEST}" --device cuda:0
) >"${LOG_ROOT}/generate_stable_audio_open_1_0.log" 2>&1 &
pids=("$!")

(
  cd "${UPSTREAM_ROOT}/TangoFlux"
  CUDA_VISIBLE_DEVICES=1 \
  PYTHONPATH="${UPSTREAM_ROOT}/TangoFlux:${ADAPTER_ROOT}" \
  "${ASSET_ROOT}/envs/tangoflux/bin/python" \
    "${ADAPTER_ROOT}/generate_tangoflux.py" \
    --manifest "${MANIFEST}" --device cuda:0
) >"${LOG_ROOT}/generate_tangoflux.log" 2>&1 &
pids+=("$!")

(
  cd "${UPSTREAM_ROOT}/AudioX-Turbo"
  CUDA_VISIBLE_DEVICES=2 \
  PYTHONPATH="${UPSTREAM_ROOT}/AudioX-Turbo:${ADAPTER_ROOT}" \
  "${ASSET_ROOT}/envs/audiox_turbo/bin/python" \
    "${ADAPTER_ROOT}/generate_audiox_turbo.py" \
    --manifest "${MANIFEST}" --device cuda:0
) >"${LOG_ROOT}/generate_audiox_turbo.log" 2>&1 &
pids+=("$!")

(
  cd "${UPSTREAM_ROOT}/MMAudio"
  CUDA_VISIBLE_DEVICES=3 \
  PYTHONPATH="${UPSTREAM_ROOT}/MMAudio:${ADAPTER_ROOT}" \
  "${ASSET_ROOT}/envs/mmaudio/bin/python" \
    "${ADAPTER_ROOT}/generate_mmaudio.py" \
    --manifest "${MANIFEST}" --device cuda:0
) >"${LOG_ROOT}/generate_mmaudio_large_44k_v2_text_only.log" 2>&1 &
pids+=("$!")

(
  cd "${UPSTREAM_ROOT}/Woosh"
  CUDA_VISIBLE_DEVICES=4 \
  HF_HUB_OFFLINE=0 \
  TRANSFORMERS_OFFLINE=0 \
  PYTHONPATH="${UPSTREAM_ROOT}/Woosh:${ADAPTER_ROOT}" \
  "${ASSET_ROOT}/envs/woosh/bin/python" \
    "${ADAPTER_ROOT}/generate_woosh.py" \
    --manifest "${MANIFEST}" --device cuda:0
) >"${LOG_ROOT}/generate_woosh_flow.log" 2>&1 &
pids+=("$!")

(
  cd "${UPSTREAM_ROOT}/Qwen3-TTS"
  CUDA_VISIBLE_DEVICES=5 \
  PYTHONPATH="${UPSTREAM_ROOT}/Qwen3-TTS:${ADAPTER_ROOT}" \
  "${ASSET_ROOT}/envs/qwen3_tts/bin/python" \
    "${ADAPTER_ROOT}/generate_qwen3_tts.py" \
    --manifest "${MANIFEST}" --device cuda:0
) >"${LOG_ROOT}/generate_qwen3_tts_1p7b_voice_design.log" 2>&1 &
pids+=("$!")

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
  echo "one or more baseline workers failed; inspect ${LOG_ROOT}" >&2
  exit 1
fi

PYTHONPATH="${ADAPTER_ROOT}" "${REPO_ROOT}/.venv/bin/python" \
  "${ADAPTER_ROOT}/audit_p10_matched_baseline_generation.py" \
  --benchmark-root "${BENCHMARK_ROOT}" \
  --manifest "${MANIFEST}"

echo "P10_BASELINE_GENERATION_COMPLETE=${BENCHMARK_ROOT}"
