#!/usr/bin/env bash
set -euo pipefail

# High-throughput checkpoint sweep: one checkpoint/shard replica per GPU.
# This deliberately refuses to share GPUs with a live training job.
REPO_ROOT="${REPO_ROOT:-.}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
EVAL_ROOT="${EVAL_ROOT:?set EVAL_ROOT to a frozen evaluation contract directory}"
LOG_ROOT="${EVAL_ROOT}/logs"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
EXPORT_VAE_RECONSTRUCTION="${EXPORT_VAE_RECONSTRUCTION:-1}"
BUILD_LISTENING_MONTAGES="${BUILD_LISTENING_MONTAGES:-1}"
MONTAGE_CASES_PER_DOMAIN="${MONTAGE_CASES_PER_DOMAIN:-5}"
# Keep one item per sampler call: batched variable-length diffusion changed one
# fixed-seed regression sample materially, so checkpoint selection stays on the
# established deterministic inference path. Parallelism comes from one
# checkpoint replica per GPU instead.
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-1}"
INFERENCE_SHARDS_PER_CHECKPOINT="${INFERENCE_SHARDS_PER_CHECKPOINT:-1}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

mkdir -p "${LOG_ROOT}"
cd "${REPO_ROOT}"

if pgrep -af '/train.py .*--num-gpus' >/dev/null; then
  echo "refusing to evaluate while distributed training is active" >&2
  pgrep -af '/train.py .*--num-gpus' >&2 || true
  exit 2
fi

IFS=',' read -r -a gpus <<<"${GPU_LIST}"
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
if ! [[ "${INFERENCE_SHARDS_PER_CHECKPOINT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "INFERENCE_SHARDS_PER_CHECKPOINT must be a positive integer" >&2
  exit 1
fi
required_replicas=$((${#checkpoint_steps[@]} * INFERENCE_SHARDS_PER_CHECKPOINT))
if [[ "${required_replicas}" -gt "${#gpus[@]}" ]]; then
  echo "checkpoint/shard sweep needs ${required_replicas} GPUs, got ${#gpus[@]}" >&2
  exit 1
fi

if [[ "${EXPORT_VAE_RECONSTRUCTION}" == "1" ]]; then
  vae_gpu="${gpus[$((${#gpus[@]} - 1))]}"
  CUDA_VISIBLE_DEVICES="${vae_gpu}" "${PYTHON_BIN}" \
    scripts/t2a/eval/export_sceneplan_dit_p10_vae_reconstruction.py \
    --eval-root "${EVAL_ROOT}" --device cuda:0 \
    >"${LOG_ROOT}/export_vae_reconstruction.log" 2>&1
fi

worker_pids=()
worker_index=0
for index in "${!checkpoint_steps[@]}"; do
  step="${checkpoint_steps[${index}]}"
  for ((shard_index = 0; shard_index < INFERENCE_SHARDS_PER_CHECKPOINT; shard_index++)); do
    gpu="${gpus[${worker_index}]}"
    if [[ "${INFERENCE_SHARDS_PER_CHECKPOINT}" -eq 1 ]]; then
      log="${LOG_ROOT}/generate_step_$(printf '%06d' "${step}").log"
    else
      log="${LOG_ROOT}/generate_step_$(printf '%06d' "${step}")_shard_$(printf '%02d' "${shard_index}").log"
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
      scripts/t2a/eval/generate_sceneplan_dit_p10_panel.py \
      --eval-root "${EVAL_ROOT}" \
      --checkpoint-step "${step}" \
      --device cuda:0 \
      --batch-size "${INFERENCE_BATCH_SIZE}" \
      --shard-index "${shard_index}" \
      --num-shards "${INFERENCE_SHARDS_PER_CHECKPOINT}" \
      >"${log}" 2>&1 &
    worker_pids+=("$!")
    worker_index=$((worker_index + 1))
  done
done

generation_failed=0
for pid in "${worker_pids[@]}"; do
  if ! wait "${pid}"; then
    generation_failed=1
  fi
done
if [[ "${generation_failed}" -ne 0 ]]; then
  echo "generation phase failed; inspect ${LOG_ROOT}/generate_step_*.log" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${EVAL_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

repo = Path.cwd()
sys.path.insert(0, str(repo))
from scripts.t2a.eval.sceneplan_dit_p10_panel_common import checkpoint_steps, load_panel

root = Path(sys.argv[1]).resolve(strict=True)
panel = load_panel(root)
missing = []
for step in checkpoint_steps(root):
    for row in panel:
        path = root / "outputs" / f"step_{step:06d}" / row["domain"] / row["panel_id"] / "metadata.json"
        if not path.is_file():
            missing.append(str(path))
if missing:
    raise RuntimeError(f"generation output is incomplete: missing={missing[:8]}")
print(json.dumps({"event": "generation_complete", "outputs": len(panel) * len(checkpoint_steps(root))}))
PY

metric_pids=()
"${PYTHON_BIN}" scripts/t2a/eval/score_sceneplan_dit_p10_core.py \
  --eval-root "${EVAL_ROOT}" >"${LOG_ROOT}/score_core.log" 2>&1 &
metric_pids+=("$!")
CUDA_VISIBLE_DEVICES="${gpus[0]}" "${PYTHON_BIN}" scripts/t2a/eval/score_sceneplan_dit_p10_clap.py \
  --eval-root "${EVAL_ROOT}" --device cuda:0 >"${LOG_ROOT}/score_clap.log" 2>&1 &
metric_pids+=("$!")
CUDA_VISIBLE_DEVICES="${gpus[1]}" "${PYTHON_BIN}" scripts/t2a/eval/score_sceneplan_dit_p10_speech.py \
  --eval-root "${EVAL_ROOT}" --device-index 0 >"${LOG_ROOT}/score_speech.log" 2>&1 &
metric_pids+=("$!")
CUDA_VISIBLE_DEVICES="${gpus[2]}" "${PYTHON_BIN}" scripts/t2a/eval/score_sceneplan_dit_p10_distributional.py \
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
if [[ "${BUILD_LISTENING_MONTAGES}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/t2a/eval/build_sceneplan_dit_p10_listening_montages.py \
    --eval-root "${EVAL_ROOT}" \
    --max-cases-per-domain "${MONTAGE_CASES_PER_DOMAIN}" \
    >"${LOG_ROOT}/build_listening_montages.log" 2>&1
fi

echo "P10_EVALUATION_COMPLETE=${EVAL_ROOT}"
