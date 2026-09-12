#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-./stable-audio-tools}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
EVAL_ROOT="${EVAL_ROOT:-${AMBIT_DATA_ROOT}/sceneplan_v2_1p124m/revisions/speech_expansion_noalign_15s_v1/evaluation/p10_v10_music_sound_protection_fixed50}"
LOG_ROOT="${EVAL_ROOT}/logs"

cd "${REPO_ROOT}"
if pgrep -af '[t]rain.py .*--num-gpus' >/dev/null; then
  echo "refusing to evaluate while distributed training is active" >&2
  exit 2
fi

"${PYTHON_BIN}" scripts/t2a/eval/build_p10_music_sound_protection_contract.py \
  --output-root "${EVAL_ROOT}"
mkdir -p "${LOG_ROOT}/inference"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

steps=(100000 105000)
pids=()
worker=0
for step in "${steps[@]}"; do
  for shard in 0 1 2 3; do
    gpu="${worker}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
      scripts/t2a/eval/generate_sceneplan_dit_p10_panel.py \
      --eval-root "${EVAL_ROOT}" \
      --checkpoint-step "${step}" \
      --device cuda:0 \
      --shard-index "${shard}" \
      --num-shards 4 \
      --batch-size 1 \
      >"${LOG_ROOT}/inference/step_${step}_shard_${shard}.log" 2>&1 &
    pids+=("$!")
    worker=$((worker + 1))
  done
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "Music/Sound protection inference failed" >&2
  exit 1
fi
date --iso-8601=seconds >"${EVAL_ROOT}/INFERENCE_COMPLETE"

metric_pids=()
"${PYTHON_BIN}" scripts/t2a/eval/score_sceneplan_dit_p10_core.py \
  --eval-root "${EVAL_ROOT}" >"${LOG_ROOT}/score_core.log" 2>&1 &
metric_pids+=("$!")
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" scripts/t2a/eval/score_sceneplan_dit_p10_clap.py \
  --eval-root "${EVAL_ROOT}" --device cuda:0 \
  >"${LOG_ROOT}/score_clap.log" 2>&1 &
metric_pids+=("$!")
CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" scripts/t2a/eval/score_sceneplan_dit_p10_distributional.py \
  --eval-root "${EVAL_ROOT}" --device cuda:0 \
  >"${LOG_ROOT}/score_distributional.log" 2>&1 &
metric_pids+=("$!")

failed=0
for pid in "${metric_pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "Music/Sound protection scoring failed" >&2
  exit 1
fi

set +e
"${PYTHON_BIN}" scripts/t2a/eval/summarize_p10_music_sound_protection.py \
  --eval-root "${EVAL_ROOT}" >"${LOG_ROOT}/summarize.log" 2>&1
summary_status=$?
set -e
date --iso-8601=seconds >"${EVAL_ROOT}/EVALUATION_COMPLETE"
if [[ "${summary_status}" -eq 0 ]]; then
  echo "P10_MUSIC_SOUND_PROTECTION_PASS=${EVAL_ROOT}"
else
  echo "P10_MUSIC_SOUND_PROTECTION_HOLD=${EVAL_ROOT}"
fi
exit "${summary_status}"
