#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-./stable-audio-tools}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
EVAL_ROOT="${EVAL_ROOT:-${AMBIT_DATA_ROOT}/sceneplan_v2_1p124m/revisions/speech_expansion_noalign_15s_v1/evaluation/p10_v11_balanced_1200_ckpt110k_150k_semantic_v2}"
MAIN_SESSION="${MAIN_SESSION:-p10_v11_grid_eval}"

cd "${REPO_ROOT}"
while [[ ! -f "${EVAL_ROOT}/EVALUATION_COMPLETE" ]]; do
  if ! tmux has-session -t "${MAIN_SESSION}" 2>/dev/null \
      && ! pgrep -af 'run_p10_v11_balanced_110k_150k_eval_8gpu.sh' >/dev/null; then
    echo "main evaluation stopped without EVALUATION_COMPLETE" >&2
    exit 1
  fi
  sleep 30
done

readarray -t selected < <("${PYTHON_BIN}" - "${EVAL_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
best = json.loads((root / "cross_system_baselines/metrics/BEST_CHECKPOINT.json").read_text())
contract = json.loads((root / "EVAL_CONTRACT.json").read_text())
step = int(best["recommended_checkpoint_step"])
matches = [row for row in contract["checkpoints"] if int(row["step"]) == step]
if len(matches) != 1:
    raise RuntimeError(f"best checkpoint not unique: {step}")
print(step)
print(matches[0]["path"])
PY
)
STEP="${selected[0]}"
CHECKPOINT="${selected[1]}"
SHOWCASE_ROOT="${EVAL_ROOT}/listening/spatial_showcase_step_${STEP}"
LOG_ROOT="${SHOWCASE_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" scripts/t2a/eval/build_p10_v11_spatial_listening_showcase_contract.py \
  --output-root "${SHOWCASE_ROOT}" \
  --checkpoint-path "${CHECKPOINT}" \
  --checkpoint-step "${STEP}" \
  >"${LOG_ROOT}/build.log" 2>&1

pids=()
for shard in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES="${shard}" "${PYTHON_BIN}" \
    scripts/t2a/eval/generate_sceneplan_dit_p10_panel.py \
    --eval-root "${SHOWCASE_ROOT}" \
    --checkpoint-step "${STEP}" \
    --device cuda:0 \
    --shard-index "${shard}" \
    --num-shards 6 \
    --batch-size 1 \
    >"${LOG_ROOT}/inference_shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "spatial listening inference failed" >&2
  exit 1
fi

count=$(find "${SHOWCASE_ROOT}/outputs/step_$(printf '%06d' "${STEP}")" -name metadata.json | wc -l)
if [[ "${count}" -ne 6 ]]; then
  echo "expected 6 spatial outputs, got ${count}" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/t2a/eval/build_p10_v11_final_listening_report.py \
  --eval-root "${EVAL_ROOT}" \
  --spatial-showcase-root "${SHOWCASE_ROOT}" \
  >"${LOG_ROOT}/build_final_report.log" 2>&1
date --iso-8601=seconds >"${EVAL_ROOT}/FINAL_LISTENING_COMPLETE"
echo "P10_V11_FINAL_LISTENING_COMPLETE=${EVAL_ROOT}/listening"
