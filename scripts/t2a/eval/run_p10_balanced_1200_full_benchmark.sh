#!/usr/bin/env bash
set -euo pipefail

# Persistent end-to-end benchmark continuation for the frozen balanced panel.
# The checkpoint sweep is launched separately; this script waits for its
# completion marker and then performs every remaining stage in contract order.
REPO_ROOT="${REPO_ROOT:-/home/tanhe/dataset_storage/stable-audio-tools}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
EVAL_ROOT="${EVAL_ROOT:-/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/speech_expansion_noalign_15s_v1/evaluation/p10_v9_balanced_1200_ckpt20k_100k_v1}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${EVAL_ROOT}/cross_system_baselines}"
LOG_ROOT="${EVAL_ROOT}/logs/full_benchmark"
INFERENCE_SESSION="${INFERENCE_SESSION:-p10_v9_balanced1200_eval}"

mkdir -p "${LOG_ROOT}"
cd "${REPO_ROOT}"

while [[ ! -f "${EVAL_ROOT}/INFERENCE_COMPLETE" ]]; do
  if ! tmux has-session -t "${INFERENCE_SESSION}" 2>/dev/null \
      && ! pgrep -af "generate_sceneplan_dit_p10_panel.py .*${EVAL_ROOT}" >/dev/null; then
    echo "checkpoint inference stopped without INFERENCE_COMPLETE" >&2
    exit 1
  fi
  sleep 30
done

"${PYTHON_BIN}" - "${EVAL_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
contract = json.loads((root / "EVAL_CONTRACT.json").read_text())
panel_path = root / contract["test_set"]["panel_filename"]
panel = [json.loads(line) for line in panel_path.read_text().splitlines() if line]
steps = [int(row["step"]) for row in contract["checkpoints"]]
missing = []
for step in steps:
    for row in panel:
        path = root / "outputs" / f"step_{step:06d}" / row["domain"] / row["panel_id"] / "metadata.json"
        if not path.is_file():
            missing.append(str(path))
if missing:
    raise RuntimeError(f"checkpoint inference is incomplete: {missing[:8]}")
print(json.dumps({"status": "PASS", "checkpoint_outputs": len(panel) * len(steps)}))
PY

vae_pids=()
for gpu in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    scripts/t2a/eval/export_sceneplan_dit_p10_vae_reconstruction.py \
    --eval-root "${EVAL_ROOT}" --device cuda:0 \
    --shard-index "${gpu}" --num-shards 8 \
    >"${LOG_ROOT}/vae_shard_${gpu}.log" 2>&1 &
  vae_pids+=("$!")
done
vae_failed=0
for pid in "${vae_pids[@]}"; do
  if ! wait "${pid}"; then
    vae_failed=1
  fi
done
if [[ "${vae_failed}" -ne 0 ]]; then
  echo "VAE ceiling export failed; inspect ${LOG_ROOT}/vae_shard_*.log" >&2
  exit 1
fi
"${PYTHON_BIN}" scripts/t2a/eval/export_sceneplan_dit_p10_vae_reconstruction.py \
  --eval-root "${EVAL_ROOT}" --summarize-only \
  >"${LOG_ROOT}/vae_summary.log" 2>&1

# This scorer is CPU-side and can overlap baseline generation safely.
"${PYTHON_BIN}" scripts/t2a/eval/score_sceneplan_dit_p10_core.py \
  --eval-root "${EVAL_ROOT}" >"${LOG_ROOT}/score_core.log" 2>&1 &
core_pid="$!"

SOURCE_EVAL_ROOT="${EVAL_ROOT}" BENCHMARK_ROOT="${BENCHMARK_ROOT}" \
  scripts/t2a/eval/baselines/run_p10_balanced_baseline_generation_8gpu.sh \
  >"${LOG_ROOT}/baseline_generation.log" 2>&1

if ! wait "${core_pid}"; then
  echo "native FOA core scoring failed; inspect ${LOG_ROOT}/score_core.log" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" \
  scripts/t2a/eval/baselines/score_p10_60k_cross_system.py \
  --benchmark-root "${BENCHMARK_ROOT}" \
  --source-eval "${EVAL_ROOT}" --device-index 0 \
  >"${LOG_ROOT}/score_cross_system.log" 2>&1

date --iso-8601=seconds >"${EVAL_ROOT}/FULL_BENCHMARK_COMPLETE"
echo "P10_BALANCED_FULL_BENCHMARK_COMPLETE=${EVAL_ROOT}"
