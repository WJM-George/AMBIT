#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tanhe/dataset_storage/stable-audio-tools}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
EVAL_ROOT="${EVAL_ROOT:-/mnt/sdb/model_archives/p10_pre_v11_20260831/sceneplan_dit_fail/sceneplan_dit_v4_r8_300m/evaluation/p10_ckpt_5k_10k_15k_sceneplan44_v1}"
LOG_ROOT="${EVAL_ROOT}/logs"
mkdir -p "${LOG_ROOT}"
cd "${REPO_ROOT}"

worker_pids=()
gpu=0
mapfile -t checkpoint_steps < <(
  "${PYTHON_BIN}" - "${EVAL_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

contract = json.loads((Path(sys.argv[1]) / "EVAL_CONTRACT.json").read_text())
for checkpoint in contract["checkpoints"]:
    print(int(checkpoint["step"]))
PY
)
if [[ "${#checkpoint_steps[@]}" -eq 0 ]]; then
  echo "evaluation contract contains no checkpoints" >&2
  exit 1
fi

if [[ "${EXPORT_VAE_RECONSTRUCTION:-1}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${VAE_GPU:-7}" "${PYTHON_BIN}" \
    scripts/t2a/eval/export_sceneplan_dit_p10_vae_reconstruction.py \
    --eval-root "${EVAL_ROOT}" \
    --device cuda:0 \
    >"${LOG_ROOT}/export_vae_reconstruction.log" 2>&1
fi

for step in "${checkpoint_steps[@]}"; do
  for shard in 0 1; do
    log="${LOG_ROOT}/generate_step_$(printf '%06d' "${step}")_shard_$(printf '%02d' "${shard}").log"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
      scripts/t2a/eval/generate_sceneplan_dit_p10_panel.py \
      --eval-root "${EVAL_ROOT}" \
      --checkpoint-step "${step}" \
      --device cuda:0 \
      --shard-index "${shard}" \
      --num-shards 2 \
      >"${log}" 2>&1 &
    worker_pids+=("$!")
    gpu=$(((gpu + 1) % 8))
  done
done

generation_failed=0
for pid in "${worker_pids[@]}"; do
  if ! wait "${pid}"; then
    generation_failed=1
  fi
done
if [[ "${generation_failed}" -ne 0 ]]; then
  echo "generation phase failed; inspect ${LOG_ROOT}/generate_*.log" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${EVAL_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
contract = json.loads((root / "EVAL_CONTRACT.json").read_text())
panel = [json.loads(line) for line in (root / "listening_panel_15.jsonl").read_text().splitlines() if line]
steps = [int(row["step"]) for row in contract["checkpoints"]]
missing = []
for step in steps:
    for row in panel:
        path = root / "outputs" / f"step_{step:06d}" / row["domain"] / row["panel_id"] / "metadata.json"
        if not path.is_file():
            missing.append(str(path))
if missing:
    raise RuntimeError(f"generation output is incomplete: missing={missing[:8]}")
print(json.dumps({"event": "generation_complete", "outputs": len(panel) * len(steps)}), flush=True)
PY

metric_pids=()
"${PYTHON_BIN}" scripts/t2a/eval/score_sceneplan_dit_p10_core.py \
  --eval-root "${EVAL_ROOT}" >"${LOG_ROOT}/score_core.log" 2>&1 &
metric_pids+=("$!")
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" scripts/t2a/eval/score_sceneplan_dit_p10_clap.py \
  --eval-root "${EVAL_ROOT}" --device cuda:0 >"${LOG_ROOT}/score_clap.log" 2>&1 &
metric_pids+=("$!")
CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" scripts/t2a/eval/score_sceneplan_dit_p10_speech.py \
  --eval-root "${EVAL_ROOT}" --device-index 0 >"${LOG_ROOT}/score_speech.log" 2>&1 &
metric_pids+=("$!")
CUDA_VISIBLE_DEVICES=2 "${PYTHON_BIN}" scripts/t2a/eval/score_sceneplan_dit_p10_distributional.py \
  --eval-root "${EVAL_ROOT}" --device cuda:0 >"${LOG_ROOT}/score_distributional.log" 2>&1 &
metric_pids+=("$!")

metrics_failed=0
for pid in "${metric_pids[@]}"; do
  if ! wait "${pid}"; then
    metrics_failed=1
  fi
done
if [[ "${metrics_failed}" -ne 0 ]]; then
  echo "metric phase failed; inspect ${LOG_ROOT}/score_*.log" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/t2a/eval/summarize_sceneplan_dit_p10_panel.py \
  --eval-root "${EVAL_ROOT}" >"${LOG_ROOT}/summarize.log" 2>&1
"${PYTHON_BIN}" scripts/t2a/eval/build_sceneplan_dit_p10_listening_montages.py \
  --eval-root "${EVAL_ROOT}" >"${LOG_ROOT}/build_listening_montages.log" 2>&1
echo "P10_EVALUATION_COMPLETE=${EVAL_ROOT}"
