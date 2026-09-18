#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-.}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/sceneplan_p11/evals/p11_v4_reference_baselines_20260903}"
EVAL="${REPO_ROOT}/scripts/t2a/eval/evaluate_sceneplan_p11_v4_challenge.py"
MERGE="${REPO_ROOT}/scripts/t2a/eval/merge_sceneplan_p11_v4_challenge_shards.py"
DATASET="${REPO_ROOT}/stable_audio_tools/configs/dataset_configs/sceneplan_p11_heldout900_transfusion_cot_v4_reliable_asr_v1.json"
CHALLENGE="${REPO_ROOT}/artifacts/sceneplan_p11/challenges/p11_v4_heldout_challenge_v1_20260901.sqlite"
D0_CKPT="${AMBIT_CKPT_ROOT}/sceneplan_p11/p11_v4_d0_screen10k_seed42_20260902/checkpoints/epoch=2-step=10000.ckpt"
DIRECT_CKPT="${AMBIT_CKPT_ROOT}/sceneplan_p11/p11_v4_direct_screen10k_seed42_20260902/checkpoints/epoch=2-step=10000.ckpt"
FLOW_CKPT="${AMBIT_CKPT_ROOT}/sceneplan_p11/p11_v4_flow_r1_screen10k_seed42_20260902/checkpoints/epoch=2-step=10000.ckpt"
D0_OUTPUT="${REPO_ROOT}/artifacts/sceneplan_p11/evals/p11_v4_d0_screen10k_s42_ema_heldout300_k1_refv10_20260903.json"
DIRECT_OUTPUT="${REPO_ROOT}/artifacts/sceneplan_p11/evals/p11_v4_direct_screen10k_s42_ema_heldout300_k1_refv10_20260903.json"
FLOW_OUTPUT="${REPO_ROOT}/artifacts/sceneplan_p11/evals/p11_v4_flow_r1_screen10k_s42_ema_heldout300_k148_refv10_20260903.json"

for required in "${PYTHON_BIN}" "${EVAL}" "${MERGE}" "${DATASET}" \
    "${CHALLENGE}" "${D0_CKPT}" "${DIRECT_CKPT}" "${FLOW_CKPT}"; do
    if [[ ! -r "${required}" ]]; then
        echo "required input is unreadable: ${required}" >&2
        exit 1
    fi
done
if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)" != "8" ]]; then
    echo "reference baseline panel requires exactly eight visible host GPUs" >&2
    exit 2
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | awk \
    'BEGIN { found=0 } /^[[:space:]]*[0-9]+[[:space:]]*$/ { found=1 } END { exit found ? 0 : 1 }'; then
    echo "refusing to share GPUs with an existing compute process" >&2
    exit 2
fi
for target in "${OUTPUT_ROOT}" "${D0_OUTPUT}" "${DIRECT_OUTPUT}" "${FLOW_OUTPUT}"; do
    if [[ -e "${target}" ]]; then
        echo "refusing existing output: ${target}" >&2
        exit 2
    fi
done

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${REPO_ROOT}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

families=(
    generation_numeric_posterior
    understanding_degraded_evidence
    exact_compatibility
    editing_counterfactual_causality
)

pids=()
names=()
launch() {
    local gpu="$1"
    local name="$2"
    shift 2
    CUDA_VISIBLE_DEVICES="${gpu}" "$@" >"${OUTPUT_ROOT}/logs/${name}.log" 2>&1 &
    pids+=("$!")
    names+=("${name}")
    echo "[p11-ref-baseline] launched gpu=${gpu} job=${name} pid=${pids[-1]}"
}

wait_phase() {
    local failed=0
    for index in "${!pids[@]}"; do
        if ! wait "${pids[${index}]}"; then
            echo "[p11-ref-baseline] failed job=${names[${index}]}" >&2
            failed=1
        else
            echo "[p11-ref-baseline] complete job=${names[${index}]}"
        fi
    done
    pids=()
    names=()
    if [[ "${failed}" != "0" ]]; then
        exit 1
    fi
}

common=(
    --dataset-config "${DATASET}"
    --challenge "${CHALLENGE}"
    --device cuda:0
    --weights ema
    --seed 42
    --discrete-decode-mode prefix_recompute
    --qwen-kernel-mode torch_reference
    --rows-per-view 30
)

# Phase 1: matched K=1 D0 and Direct baselines, one family per physical GPU.
for index in "${!families[@]}"; do
    family="${families[${index}]}"
    launch "${index}" "d0_${family}" "${PYTHON_BIN}" "${EVAL}" \
        --arm d0 --checkpoint "${D0_CKPT}" "${common[@]}" \
        --draws 1 --k-values 1 --d0-temperature 0.8 \
        --families "${family}" --output "${OUTPUT_ROOT}/d0_${family}.json"
    launch "$((index + 4))" "direct_${family}" "${PYTHON_BIN}" "${EVAL}" \
        --arm direct --checkpoint "${DIRECT_CKPT}" "${common[@]}" \
        --draws 1 --k-values 1 --families "${family}" \
        --output "${OUTPUT_ROOT}/direct_${family}.json"
done
wait_phase

"${PYTHON_BIN}" "${MERGE}" \
    --shard "${OUTPUT_ROOT}/d0_generation_numeric_posterior.json" \
    --shard "${OUTPUT_ROOT}/d0_understanding_degraded_evidence.json" \
    --shard "${OUTPUT_ROOT}/d0_exact_compatibility.json" \
    --shard "${OUTPUT_ROOT}/d0_editing_counterfactual_causality.json" \
    --output "${D0_OUTPUT}"
"${PYTHON_BIN}" "${MERGE}" \
    --shard "${OUTPUT_ROOT}/direct_generation_numeric_posterior.json" \
    --shard "${OUTPUT_ROOT}/direct_understanding_degraded_evidence.json" \
    --shard "${OUTPUT_ROOT}/direct_exact_compatibility.json" \
    --shard "${OUTPUT_ROOT}/direct_editing_counterfactual_causality.json" \
    --output "${DIRECT_OUTPUT}"

# Phase 2: Flow screening checkpoint with its real K=1/4/8 posterior.
for index in "${!families[@]}"; do
    family="${families[${index}]}"
    launch "${index}" "flow_${family}" "${PYTHON_BIN}" "${EVAL}" \
        --arm flow --checkpoint "${FLOW_CKPT}" "${common[@]}" \
        --draws 8 --k-values 1,4,8 --families "${family}" \
        --output "${OUTPUT_ROOT}/flow_${family}.json"
done
wait_phase

"${PYTHON_BIN}" "${MERGE}" \
    --shard "${OUTPUT_ROOT}/flow_generation_numeric_posterior.json" \
    --shard "${OUTPUT_ROOT}/flow_understanding_degraded_evidence.json" \
    --shard "${OUTPUT_ROOT}/flow_exact_compatibility.json" \
    --shard "${OUTPUT_ROOT}/flow_editing_counterfactual_causality.json" \
    --output "${FLOW_OUTPUT}"

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" "${D0_OUTPUT}" "${DIRECT_OUTPUT}" "${FLOW_OUTPUT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
paths = [Path(value).resolve(strict=True) for value in sys.argv[2:]]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
summary = {
    "schema": "stable_audio_tools.p11_v4_reference_baseline_panel",
    "schema_version": 1,
    "status": "PASS" if all(report.get("status") == "PASS" for report in reports) else "FAIL",
    "seed": 42,
    "weights": "ema",
    "qwen_kernel_mode": "torch_reference",
    "reports": {
        report["arm"]: {
            "path": str(path),
            "sha256": sha256(path),
            "report_sha256_without_self": report["report_sha256_without_self"],
            "rows": report["rows"],
            "draws": report["draws"],
            "k_values": report["k_values"],
        }
        for path, report in zip(paths, reports)
    },
}
payload = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
summary["report_sha256_without_self"] = hashlib.sha256(payload).hexdigest()
output = root / "REFERENCE_BASELINE_SUMMARY.json"
output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
if summary["status"] != "PASS":
    raise SystemExit(1)
PY

echo "P11_REFERENCE_BASELINES_COMPLETE=${OUTPUT_ROOT}/REFERENCE_BASELINE_SUMMARY.json"
