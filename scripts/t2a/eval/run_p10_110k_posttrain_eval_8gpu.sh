#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tanhe/dataset_storage/stable-audio-tools}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
REVISION_EVAL="/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/speech_expansion_noalign_15s_v1/evaluation"
RUN_ROOT="/mnt/sdb/model_archives/p10_pre_v11_20260831/sceneplan_dit_v10_semantic_v2_protected_resume_110k"
CKPT_110K="${RUN_ROOT}/checkpoints/epoch=35-step=110000.ckpt"
SPEECH_105_ROOT="${REVISION_EVAL}/p10_speech_fixed50_105k_v2"
SPEECH_110_ROOT="${REVISION_EVAL}/p10_speech_fixed50_110k_v2"
SPEECH_SUMMARY_ROOT="${RUN_ROOT}/evaluation/speech_105k_110k_v2_fixed50"
MS_ROOT="${REVISION_EVAL}/p10_v10_music_sound_protection_105k_110k_fixed50"
LOG_ROOT="${RUN_ROOT}/evaluation/posttrain_110k_logs"

cd "${REPO_ROOT}"
if pgrep -af '[t]rain.py .*--num-gpus' >/dev/null; then
  echo "refusing to evaluate while distributed training is active" >&2
  exit 2
fi
mkdir -p "${LOG_ROOT}" "${SPEECH_110_ROOT}/logs/inference" "${SPEECH_SUMMARY_ROOT}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

"${PYTHON_BIN}" scripts/t2a/eval/build_p10_speech_fixed50_semantic_grid_contract.py \
  --output-root "${SPEECH_110_ROOT}" \
  --checkpoint-path "${CKPT_110K}" \
  --checkpoint-step 110000 \
  --semantic-caption-version 2 \
  >"${LOG_ROOT}/build_speech_110k.log" 2>&1

pids=()
for shard in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="${shard}" "${PYTHON_BIN}" \
    scripts/t2a/eval/generate_sceneplan_dit_p10_panel.py \
    --eval-root "${SPEECH_110_ROOT}" \
    --checkpoint-step 110000 \
    --device cuda:0 \
    --shard-index "${shard}" \
    --num-shards 8 \
    --batch-size 1 \
    >"${SPEECH_110_ROOT}/logs/inference/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "110k fixed50 Speech inference failed" >&2
  exit 1
fi
date --iso-8601=seconds >"${SPEECH_110_ROOT}/INFERENCE_COMPLETE"

set +e
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" \
  scripts/t2a/eval/score_p10_speech_v2_checkpoint_pair.py \
  --baseline-root "${SPEECH_105_ROOT}" \
  --candidate-root "${SPEECH_110_ROOT}" \
  --output-root "${SPEECH_SUMMARY_ROOT}" \
  --device-index 0 \
  >"${LOG_ROOT}/score_speech_pair.log" 2>&1
speech_status=$?
set -e

"${PYTHON_BIN}" scripts/t2a/eval/build_p10_music_sound_checkpoint_pair_contract.py \
  --output-root "${MS_ROOT}" >"${LOG_ROOT}/build_music_sound_pair.log" 2>&1
mkdir -p "${MS_ROOT}/logs/inference"

pids=()
worker=0
for step in 105000 110000; do
  for shard in 0 1 2 3; do
    gpu="${worker}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
      scripts/t2a/eval/generate_sceneplan_dit_p10_panel.py \
      --eval-root "${MS_ROOT}" \
      --checkpoint-step "${step}" \
      --device cuda:0 \
      --shard-index "${shard}" \
      --num-shards 4 \
      --batch-size 1 \
      >"${MS_ROOT}/logs/inference/step_${step}_shard_${shard}.log" 2>&1 &
    pids+=("$!")
    worker=$((worker + 1))
  done
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "105k/110k Music/Sound inference failed" >&2
  exit 1
fi
date --iso-8601=seconds >"${MS_ROOT}/INFERENCE_COMPLETE"

metric_pids=()
"${PYTHON_BIN}" scripts/t2a/eval/score_sceneplan_dit_p10_core.py \
  --eval-root "${MS_ROOT}" >"${MS_ROOT}/logs/score_core.log" 2>&1 &
metric_pids+=("$!")
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" scripts/t2a/eval/score_sceneplan_dit_p10_clap.py \
  --eval-root "${MS_ROOT}" --device cuda:0 >"${MS_ROOT}/logs/score_clap.log" 2>&1 &
metric_pids+=("$!")
CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" scripts/t2a/eval/score_sceneplan_dit_p10_distributional.py \
  --eval-root "${MS_ROOT}" --device cuda:0 >"${MS_ROOT}/logs/score_distributional.log" 2>&1 &
metric_pids+=("$!")
failed=0
for pid in "${metric_pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "105k/110k Music/Sound scoring failed" >&2
  exit 1
fi

set +e
"${PYTHON_BIN}" scripts/t2a/eval/summarize_p10_music_sound_protection.py \
  --eval-root "${MS_ROOT}" >"${MS_ROOT}/logs/summarize.log" 2>&1
ms_status=$?
set -e
date --iso-8601=seconds >"${MS_ROOT}/EVALUATION_COMPLETE"

"${PYTHON_BIN}" - "${SPEECH_SUMMARY_ROOT}" "${MS_ROOT}" "${speech_status}" "${ms_status}" <<'PY'
import json
import sys
from pathlib import Path

speech_root, ms_root = map(Path, sys.argv[1:3])
speech_status, ms_status = map(int, sys.argv[3:5])
value = {
    "schema": "stable_audio_tools.p10_110k_posttrain_gate",
    "schema_version": 1,
    "status": "PASS" if speech_status == 0 and ms_status == 0 else "HOLD",
    "speech_summary": str(speech_root / "SUMMARY.json"),
    "music_sound_summary": str(ms_root / "PROTECTION_SUMMARY.json"),
    "speech_exit_status": speech_status,
    "music_sound_exit_status": ms_status,
}
path = speech_root.parent / "P10_110K_POSTTRAIN_GATE.json"
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
print(json.dumps(value, indent=2, sort_keys=True))
PY
date --iso-8601=seconds >"${RUN_ROOT}/evaluation/POSTTRAIN_110K_COMPLETE"
if [[ "${speech_status}" -eq 0 && "${ms_status}" -eq 0 ]]; then
  echo "P10_110K_POSTTRAIN_PASS=${RUN_ROOT}/evaluation/P10_110K_POSTTRAIN_GATE.json"
  exit 0
fi
echo "P10_110K_POSTTRAIN_HOLD=${RUN_ROOT}/evaluation/P10_110K_POSTTRAIN_GATE.json"
exit 2
