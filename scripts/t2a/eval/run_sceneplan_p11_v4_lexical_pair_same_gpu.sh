#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-.}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
CHECKPOINT="${1:-}"
OUTPUT_ROOT="${2:-}"
PHYSICAL_GPU="${3:-}"
if [[ -z "${CHECKPOINT}" || -z "${OUTPUT_ROOT}" || ! "${PHYSICAL_GPU}" =~ ^[0-9]+$ ]]; then
    echo "usage: $0 CHECKPOINT OUTPUT_ROOT PHYSICAL_GPU" >&2
    exit 2
fi

CHECKPOINT="$(readlink -f -- "${CHECKPOINT}")"
OUTPUT_ROOT="$(readlink -m -- "${OUTPUT_ROOT}")"
MODEL_CONFIG="${REPO_ROOT}/stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/qwen35_0p8b_sceneplan_p11_reliable_asr_assembler_v2.json"
DATASET_CONFIG="${REPO_ROOT}/stable_audio_tools/configs/dataset_configs/sceneplan_p11_heldout900_transfusion_cot_v4_reliable_asr_v1.json"
CHALLENGE="${REPO_ROOT}/artifacts/sceneplan_p11/challenges/p11_v4_heldout_challenge_v1_20260901.sqlite"
EVALUATOR="${REPO_ROOT}/scripts/t2a/eval/evaluate_sceneplan_p11_v4_challenge.py"
SUMMARIZER="${REPO_ROOT}/scripts/t2a/eval/summarize_sceneplan_p11_v4_lexical_ab.py"
ASR_ON="${OUTPUT_ROOT}/understanding_asr_on_k1.json"
ASR_DROP="${OUTPUT_ROOT}/understanding_asr_drop_k1.json"
PAIRING="${OUTPUT_ROOT}/understanding_asr_same_gpu_execution.json"
SUMMARY="${OUTPUT_ROOT}/understanding_asr_causal_ab.json"

for required in "${PYTHON_BIN}" "${CHECKPOINT}" "${MODEL_CONFIG}" \
    "${DATASET_CONFIG}" "${CHALLENGE}" "${EVALUATOR}" "${SUMMARIZER}"; do
    if [[ ! -r "${required}" ]]; then
        echo "same-GPU lexical A/B prerequisite is unreadable: ${required}" >&2
        exit 1
    fi
done
if ! nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=uuid \
    --format=csv,noheader,nounits >/dev/null 2>&1; then
    echo "unavailable physical GPU: ${PHYSICAL_GPU}" >&2
    exit 2
fi
for output in "${ASR_ON}" "${ASR_DROP}" "${PAIRING}" "${SUMMARY}"; do
    if [[ -e "${output}" ]]; then
        echo "refusing existing lexical A/B output: ${output}" >&2
        exit 2
    fi
done
mkdir -p "${OUTPUT_ROOT}/logs"
GPU_UUID="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')"
GPU_NAME="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=name --format=csv,noheader,nounits | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

common=(
    --arm flow
    --checkpoint "${CHECKPOINT}"
    --model-config "${MODEL_CONFIG}"
    --dataset-config "${DATASET_CONFIG}"
    --challenge "${CHALLENGE}"
    --device cuda:0
    --weights ema
    --seed 42
    --discrete-decode-mode prefix_recompute
    --qwen-kernel-mode torch_reference
    --draws 1
    --k-values 1
    --rows-per-view 30
    --families understanding_degraded_evidence
)

# Both arms run sequentially on one physical GPU.  Running them concurrently
# on different cards is not a valid autoregressive causal A/B: tiny device-side
# rounding differences can alter an early token and then amplify downstream.
CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" "${PYTHON_BIN}" "${EVALUATOR}" \
    "${common[@]}" --lexical-authority-intervention normal \
    --output "${ASR_ON}" >"${OUTPUT_ROOT}/logs/understanding_asr_on_k1.log" 2>&1
CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" "${PYTHON_BIN}" "${EVALUATOR}" \
    "${common[@]}" --lexical-authority-intervention drop_reliable_asr \
    --output "${ASR_DROP}" >"${OUTPUT_ROOT}/logs/understanding_asr_drop_k1.log" 2>&1

"${PYTHON_BIN}" - "${ASR_ON}" "${ASR_DROP}" "${PAIRING}" \
    "${PHYSICAL_GPU}" "${GPU_UUID}" "${GPU_NAME}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

on, drop, output = (Path(value).resolve(strict=True) for value in sys.argv[1:4])
physical_id, uuid, name = sys.argv[4:7]

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

report = {
    "schema": "stable_audio_tools.p11_v4_lexical_pair_execution",
    "schema_version": 1,
    "status": "PASS",
    "sequential": True,
    "same_physical_gpu": True,
    "physical_gpu": {"id": int(physical_id), "uuid": uuid, "name": name},
    "ordering": ["asr_on", "asr_drop"],
    "outputs": {
        "asr_on": {"path": str(on), "sha256": sha256_file(on)},
        "asr_drop": {"path": str(drop), "sha256": sha256_file(drop)},
    },
}
payload = json.dumps(
    report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
).encode("utf-8")
report["report_sha256_without_self"] = hashlib.sha256(payload).hexdigest()
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

"${PYTHON_BIN}" "${SUMMARIZER}" \
    --asr-on "${ASR_ON}" --asr-drop "${ASR_DROP}" \
    --pairing-contract "${PAIRING}" --output "${SUMMARY}"
echo "P11_LEXICAL_SAME_GPU_AB_COMPLETE=${SUMMARY}"
