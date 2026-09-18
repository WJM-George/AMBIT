#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-.}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
EVAL_ROOT="${EVAL_ROOT:-${AMBIT_DATA_ROOT}/sceneplan_v2_1p124m/revisions/speech_expansion_noalign_15s_v1/evaluation/p10_v11_balanced_1200_ckpt110k_150k_semantic_v2}"
LOG_ROOT="${EVAL_ROOT}/logs"
STEPS=(110000 120000 130000 140000 150000)

cd "${REPO_ROOT}"
if pgrep -af '[t]rain.py .*--num-gpus' >/dev/null; then
  echo "refusing to evaluate while distributed P10 training is active" >&2
  exit 2
fi
mkdir -p "${LOG_ROOT}/inference" "${LOG_ROOT}/metrics"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

"${PYTHON_BIN}" scripts/t2a/eval/build_p10_v11_balanced_checkpoint_grid_contract.py \
  --output-root "${EVAL_ROOT}" \
  >"${LOG_ROOT}/build_contract.log" 2>&1

for step in "${STEPS[@]}"; do
  echo "$(date --iso-8601=seconds) inference step=${step} start"
  pids=()
  for shard in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="${shard}" "${PYTHON_BIN}" \
      scripts/t2a/eval/generate_sceneplan_dit_p10_panel.py \
      --eval-root "${EVAL_ROOT}" \
      --checkpoint-step "${step}" \
      --device cuda:0 \
      --shard-index "${shard}" \
      --num-shards 8 \
      --batch-size 1 \
      >"${LOG_ROOT}/inference/step_${step}_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then failed=1; fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "inference failed for step=${step}; inspect ${LOG_ROOT}/inference" >&2
    exit 1
  fi
  "${PYTHON_BIN}" - "${EVAL_ROOT}" "${step}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
step = int(sys.argv[2])
contract = json.loads((root / "EVAL_CONTRACT.json").read_text())
panel = [
    json.loads(line)
    for line in (root / contract["test_set"]["panel_filename"]).read_text().splitlines()
    if line
]
seen = []
for row in panel:
    path = root / "outputs" / f"step_{step:06d}" / row["domain"] / row["panel_id"] / "metadata.json"
    if not path.is_file():
        raise RuntimeError(f"missing metadata: {path}")
    value = json.loads(path.read_text())
    if value.get("status") != "PASS":
        raise RuntimeError(f"non-PASS output: {path}")
    if int(value.get("semantic_caption_compiler_version", -1)) != 2:
        raise RuntimeError(f"wrong semantic compiler: {path}")
    if int(value["checkpoint_step"]) != step:
        raise RuntimeError(f"wrong checkpoint step: {path}")
    seen.append(value["panel_id"])
if len(seen) != 1200 or len(set(seen)) != 1200:
    raise RuntimeError(f"step {step}: invalid output count {len(seen)}")
print(json.dumps({"status": "PASS", "step": step, "outputs": len(seen)}))
PY
  date --iso-8601=seconds >"${EVAL_ROOT}/STEP_${step}_INFERENCE_COMPLETE"
  echo "$(date --iso-8601=seconds) inference step=${step} complete"
done
date --iso-8601=seconds >"${EVAL_ROOT}/INFERENCE_COMPLETE"

echo "$(date --iso-8601=seconds) metric stages start"
metric_pids=()
"${PYTHON_BIN}" scripts/t2a/eval/score_sceneplan_dit_p10_core.py \
  --eval-root "${EVAL_ROOT}" \
  >"${LOG_ROOT}/metrics/core.log" 2>&1 &
metric_pids+=("$!")
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" scripts/t2a/eval/score_sceneplan_dit_p10_clap.py \
  --eval-root "${EVAL_ROOT}" --device cuda:0 \
  >"${LOG_ROOT}/metrics/clap.log" 2>&1 &
metric_pids+=("$!")
CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" scripts/t2a/eval/score_sceneplan_dit_p10_distributional.py \
  --eval-root "${EVAL_ROOT}" --device cuda:0 \
  >"${LOG_ROOT}/metrics/distributional.log" 2>&1 &
metric_pids+=("$!")
CUDA_VISIBLE_DEVICES=2 "${PYTHON_BIN}" scripts/t2a/eval/score_sceneplan_dit_p10_speech.py \
  --eval-root "${EVAL_ROOT}" --device-index 0 \
  >"${LOG_ROOT}/metrics/speech.log" 2>&1 &
metric_pids+=("$!")

failed=0
for pid in "${metric_pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more metric stages failed; inspect ${LOG_ROOT}/metrics" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/t2a/eval/summarize_sceneplan_dit_p10_panel.py \
  --eval-root "${EVAL_ROOT}" \
  >"${LOG_ROOT}/metrics/summarize_native.log" 2>&1
"${PYTHON_BIN}" scripts/t2a/eval/merge_p10_v11_checkpoint_grid_with_frozen_baselines.py \
  --eval-root "${EVAL_ROOT}" \
  >"${LOG_ROOT}/metrics/merge_baselines.log" 2>&1

date --iso-8601=seconds >"${EVAL_ROOT}/EVALUATION_COMPLETE"
echo "P10_V11_BALANCED_110K_150K_EVALUATION_COMPLETE=${EVAL_ROOT}"
