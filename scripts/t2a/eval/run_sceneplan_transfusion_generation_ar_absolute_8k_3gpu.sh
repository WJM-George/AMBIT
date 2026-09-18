#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-.}"
PYTHON_BIN="${PYTHON_BIN:-${AMBIT_CKPT_ROOT}/stable-audio-tools-venv/bin/python}"
PLAN_EVALUATION_DIR="${PLAN_EVALUATION_DIR:?PLAN_EVALUATION_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
BASELINE_ROOT="${BASELINE_ROOT:-${AMBIT_DATA_ROOT}/sceneplan_v2_1p124m/revisions/speech_expansion_noalign_15s_v1/evaluation/p10_v11_150k_full_test_8000_semantic_v2}"
GPU_LIST="${GPU_LIST:-0,1,2}"
IFS=',' read -r -a GPUS <<<"${GPU_LIST}"
if [[ "${#GPUS[@]}" -ne 3 ]]; then
  echo "GPU_LIST must contain exactly three GPU indices" >&2
  exit 2
fi

RENDERER="${REPO_ROOT}/scripts/t2a/eval/render_sceneplan_transfusion_generation_ar_absolute_8k.py"
SCORER="${REPO_ROOT}/scripts/t2a/eval/score_sceneplan_transfusion_generation_ar_absolute_8k.py"
LOG_ROOT="${OUTPUT_DIR}/logs/absolute_gt_pipeline"
mkdir -p "${LOG_ROOT}"
exec 9>"${OUTPUT_DIR}/.absolute_gt_pipeline.lock"
if ! flock -n 9; then
  echo "another absolute-GT evaluation owns ${OUTPUT_DIR}" >&2
  exit 3
fi
: >"${LOG_ROOT}/status.tsv"

export HF_HOME="${HF_HOME:-./codex-home/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"

render_common=(
  --plan-evaluation-dir "${PLAN_EVALUATION_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --baseline-root "${BASELINE_ROOT}"
  --num-shards 3
)
score_common=(
  --evaluation-root "${OUTPUT_DIR}"
  --baseline-root "${BASELINE_ROOT}"
  --num-speech-shards 3
)

cd "${REPO_ROOT}"
"${PYTHON_BIN}" "${RENDERER}" "${render_common[@]}" --prepare-only \
  >"${LOG_ROOT}/prepare.log" 2>&1

pids=()
names=()
for shard in 0 1 2; do
  gpu="${GPUS[${shard}]}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON_BIN}" "${RENDERER}" "${render_common[@]}" \
      --device cuda:0 --shard-index "${shard}"
  ) >"${LOG_ROOT}/render_${shard}.log" 2>&1 &
  pids+=("$!")
  names+=("render_${shard}")
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[${index}]}"; then
    printf '%s\tPASS\n' "${names[${index}]}" >>"${LOG_ROOT}/status.tsv"
  else
    printf '%s\tFAIL\n' "${names[${index}]}" >>"${LOG_ROOT}/status.tsv"
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

"${PYTHON_BIN}" "${RENDERER}" "${render_common[@]}" --finalize-only \
  >"${LOG_ROOT}/finalize_render.log" 2>&1
"${PYTHON_BIN}" "${RENDERER}" "${render_common[@]}" --verify-only \
  >"${LOG_ROOT}/verify_render.log" 2>&1

pids=()
names=()
for arm_index in 0 1 2; do
  case "${arm_index}" in
    0) arm=clap ;;
    1) arm=vggish ;;
    2) arm=panns ;;
  esac
  gpu="${GPUS[${arm_index}]}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON_BIN}" "${SCORER}" "${score_common[@]}" \
      --arm "${arm}" --device-index 0
  ) >"${LOG_ROOT}/${arm}.log" 2>&1 &
  pids+=("$!")
  names+=("${arm}")
done
(
  export CUDA_VISIBLE_DEVICES=""
  "${PYTHON_BIN}" "${SCORER}" "${score_common[@]}" --arm spatial
) >"${LOG_ROOT}/spatial.log" 2>&1 &
pids+=("$!")
names+=("spatial")

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[${index}]}"; then
    printf '%s\tPASS\n' "${names[${index}]}" >>"${LOG_ROOT}/status.tsv"
  else
    printf '%s\tFAIL\n' "${names[${index}]}" >>"${LOG_ROOT}/status.tsv"
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

pids=()
names=()
for shard in 0 1 2; do
  gpu="${GPUS[${shard}]}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON_BIN}" "${SCORER}" "${score_common[@]}" \
      --arm speech --device-index 0 --shard-index "${shard}"
  ) >"${LOG_ROOT}/speech_${shard}.log" 2>&1 &
  pids+=("$!")
  names+=("speech_${shard}")
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[${index}]}"; then
    printf '%s\tPASS\n' "${names[${index}]}" >>"${LOG_ROOT}/status.tsv"
  else
    printf '%s\tFAIL\n' "${names[${index}]}" >>"${LOG_ROOT}/status.tsv"
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

"${PYTHON_BIN}" "${SCORER}" "${score_common[@]}" --arm merge \
  >"${LOG_ROOT}/merge.log" 2>&1
"${PYTHON_BIN}" "${SCORER}" "${score_common[@]}" --arm compare \
  >"${LOG_ROOT}/compare.log" 2>&1

echo "ABSOLUTE_GT_EVALUATION_COMPLETE=${OUTPUT_DIR}"
