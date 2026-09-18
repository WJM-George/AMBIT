#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-.}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
CHECKPOINT="${1:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/sceneplan_p11/evals/p11_v4_flow_r1_scale10k_seed42_refv10}"
MODEL_CONFIG="${REPO_ROOT}/stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/qwen35_0p8b_sceneplan_p11_reliable_asr_assembler_v2.json"
DATASET_CONFIG="${REPO_ROOT}/stable_audio_tools/configs/dataset_configs/sceneplan_p11_heldout900_transfusion_cot_v4_reliable_asr_v1.json"
CHALLENGE="${REPO_ROOT}/artifacts/sceneplan_p11/challenges/p11_v4_heldout_challenge_v1_20260901.sqlite"
CHALLENGE_EVAL="${REPO_ROOT}/scripts/t2a/eval/evaluate_sceneplan_p11_v4_challenge.py"
CAUSAL_EVAL="${REPO_ROOT}/scripts/t2a/eval/evaluate_sceneplan_p11_v4.py"
EXPOSURE_EVAL="${REPO_ROOT}/scripts/t2a/eval/evaluate_sceneplan_p11_v4_sketch_exposure.py"
GPU_TELEMETRY_MONITOR="${REPO_ROOT}/scripts/t2a/eval/monitor_sceneplan_p11_v4_gpu_usage.py"
EVAL_PROTOCOL_MANAGER="${REPO_ROOT}/scripts/t2a/eval/manage_sceneplan_p11_v4_scale_eval_protocol.py"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-${REPO_ROOT}/artifacts/sceneplan_p11/protocols/p11_v4_uinventoryaux_scale10k_posttrain_protocol_20260903.json}"
K1_PROJECTOR="${REPO_ROOT}/scripts/t2a/eval/project_sceneplan_p11_v4_k1_quality.py"
U_INVENTORY_SUMMARIZER="${REPO_ROOT}/scripts/t2a/eval/summarize_sceneplan_p11_v4_u_inventory.py"
DECISION_SUMMARIZER="${REPO_ROOT}/scripts/t2a/eval/summarize_sceneplan_p11_v4_scale_decision.py"
LEXICAL_PAIR_RUNNER="${REPO_ROOT}/scripts/t2a/eval/run_sceneplan_p11_v4_lexical_pair_same_gpu.sh"
D0_BASELINE="${D0_BASELINE:-${REPO_ROOT}/artifacts/sceneplan_p11/evals/p11_v4_d0_screen10k_s42_ema_heldout300_k1_refv10_20260903.json}"
DIRECT_BASELINE="${DIRECT_BASELINE:-${REPO_ROOT}/artifacts/sceneplan_p11/evals/p11_v4_direct_screen10k_s42_ema_heldout300_k1_refv10_20260903.json}"
SCREEN_FLOW_BASELINE="${SCREEN_FLOW_BASELINE:-${REPO_ROOT}/artifacts/sceneplan_p11/evals/p11_v4_flow_r1_screen10k_s42_ema_heldout300_k148_refv10_20260903.json}"

if [[ -z "${CHECKPOINT}" ]]; then
    echo "usage: $0 /absolute/path/to/epoch=*-step=10000.ckpt" >&2
    exit 2
fi
CHECKPOINT="$(readlink -f -- "${CHECKPOINT}")"
RUN_ROOT="$(dirname "$(dirname "${CHECKPOINT}")")"
TRAINING_REPORT="${RUN_ROOT}/scale_trial_report.json"
LAUNCH_CONTRACT="${RUN_ROOT}/training_launch_contract.json"
PRETRAIN_REPORTS=(
    "${RUN_ROOT}/lexical_cache_gate.json"
    "${RUN_ROOT}/ddp_sampler_gate.json"
    "${RUN_ROOT}/scale_trial_data_gate.json"
    "${RUN_ROOT}/v4_contract_gate.json"
    "${RUN_ROOT}/sequence_budget_gate.json"
    "${RUN_ROOT}/control_direction_cpu_gate.json"
    "${RUN_ROOT}/delta_owner_cpu_gate.json"
    "${RUN_ROOT}/control_direction_real_graph_smoke.json"
)
if [[ ! -r "${CHECKPOINT}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "checkpoint or Python is unreadable" >&2
    exit 1
fi
for required in \
    "${MODEL_CONFIG}" "${DATASET_CONFIG}" "${CHALLENGE}" \
    "${CHALLENGE_EVAL}" "${CAUSAL_EVAL}" "${EXPOSURE_EVAL}" \
    "${GPU_TELEMETRY_MONITOR}" "${EVAL_PROTOCOL_MANAGER}" "${EVAL_PROTOCOL}" \
    "${K1_PROJECTOR}" "${U_INVENTORY_SUMMARIZER}" \
    "${DECISION_SUMMARIZER}" "${LEXICAL_PAIR_RUNNER}" \
    "${D0_BASELINE}" "${DIRECT_BASELINE}" "${SCREEN_FLOW_BASELINE}" \
    "${TRAINING_REPORT}" "${LAUNCH_CONTRACT}" \
    "${PRETRAIN_REPORTS[@]}"; do
    if [[ ! -r "${required}" ]]; then
        echo "required input is unreadable: ${required}" >&2
        exit 1
    fi
done
if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)" != "8" ]]; then
    echo "canonical medium evaluation requires exactly eight visible host GPUs" >&2
    exit 2
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | awk \
    'BEGIN { found=0 } /^[[:space:]]*[0-9]+[[:space:]]*$/ { found=1 } END { exit found ? 0 : 1 }'; then
    echo "refusing to share GPUs with an existing compute process" >&2
    exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "refusing an existing output directory: ${OUTPUT_ROOT}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}/logs"
PROTOCOL_VERIFICATION_REPORT="${OUTPUT_ROOT}/EVAL_PROTOCOL_VERIFICATION.json"
"${PYTHON_BIN}" "${EVAL_PROTOCOL_MANAGER}" verify \
    --protocol "${EVAL_PROTOCOL}" --checkpoint "${CHECKPOINT}" \
    --output "${PROTOCOL_VERIFICATION_REPORT}" \
    >"${OUTPUT_ROOT}/logs/eval_protocol_verification.log" 2>&1
cd "${REPO_ROOT}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=1

common_challenge_args=(
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
    echo "[p11-eval8] launched gpu=${gpu} job=${name} pid=${pids[-1]}"
}

G_REPORT="${OUTPUT_ROOT}/generation_k148.json"
U_REPORT="${OUTPUT_ROOT}/understanding_degraded_k148.json"
X_REPORT="${OUTPUT_ROOT}/exact_compatibility_k148.json"
E_REPORT="${OUTPUT_ROOT}/editing_counterfactual_k148.json"
CAUSAL_REPORT="${OUTPUT_ROOT}/flow_causal_interventions.json"
ASR_ON_REPORT="${OUTPUT_ROOT}/understanding_asr_on_k1.json"
ASR_DROP_REPORT="${OUTPUT_ROOT}/understanding_asr_drop_k1.json"
ASR_PAIRING_REPORT="${OUTPUT_ROOT}/understanding_asr_same_gpu_execution.json"
ASR_SUMMARY="${OUTPUT_ROOT}/understanding_asr_causal_ab.json"
REPRO_A_REPORT="${OUTPUT_ROOT}/cross_process_repro_a_k148.json"
REPRO_B_REPORT="${OUTPUT_ROOT}/cross_process_repro_b_k148.json"
U_EXPOSURE_REPORT="${OUTPUT_ROOT}/understanding_sketch_exposure_u40.json"
U_INVENTORY_REPORT="${OUTPUT_ROOT}/understanding_inventory_exact30_matched.json"
GPU_USAGE_REPORT="${OUTPUT_ROOT}/initial_parallel_gpu_telemetry.json"

gpu_monitor_pid=""
stop_gpu_monitor() {
    if [[ -z "${gpu_monitor_pid}" ]]; then
        return 0
    fi
    if kill -0 "${gpu_monitor_pid}" 2>/dev/null; then
        kill -TERM "${gpu_monitor_pid}"
    fi
    wait "${gpu_monitor_pid}"
    local monitor_status=$?
    gpu_monitor_pid=""
    return "${monitor_status}"
}

"${PYTHON_BIN}" "${GPU_TELEMETRY_MONITOR}" \
    --output "${GPU_USAGE_REPORT}" --interval-sec 5 --expected-gpus 8 \
    --assignment 0=generation_k148 \
    --assignment 1=understanding_degraded_k148 \
    --assignment 2=exact_compatibility_k148 \
    --assignment 3=editing_counterfactual_k148 \
    --assignment 4=thought_causal_interventions \
    --assignment 5=same_gpu_reliable_asr_on_drop_pair \
    --assignment 6=cross_process_reference_replay \
    --assignment 7=understanding_scene_sketch_exposure_u40 \
    >"${OUTPUT_ROOT}/logs/initial_parallel_gpu_telemetry.log" 2>&1 &
gpu_monitor_pid="$!"
trap 'stop_gpu_monitor >/dev/null 2>&1 || true' EXIT

launch 0 generation_k148 "${PYTHON_BIN}" "${CHALLENGE_EVAL}" \
    "${common_challenge_args[@]}" --draws 8 --k-values 1,4,8 \
    --rows-per-view 30 --families generation_numeric_posterior \
    --output "${G_REPORT}"
launch 1 understanding_degraded_k148 "${PYTHON_BIN}" "${CHALLENGE_EVAL}" \
    "${common_challenge_args[@]}" --draws 8 --k-values 1,4,8 \
    --rows-per-view 30 --families understanding_degraded_evidence \
    --output "${U_REPORT}"
launch 2 exact_compatibility_k148 "${PYTHON_BIN}" "${CHALLENGE_EVAL}" \
    "${common_challenge_args[@]}" --draws 8 --k-values 1,4,8 \
    --rows-per-view 30 --families exact_compatibility \
    --output "${X_REPORT}"
launch 3 editing_counterfactual_k148 "${PYTHON_BIN}" "${CHALLENGE_EVAL}" \
    "${common_challenge_args[@]}" --draws 8 --k-values 1,4,8 \
    --rows-per-view 30 --families editing_counterfactual_causality \
    --output "${E_REPORT}"
launch 4 flow_causal_interventions "${PYTHON_BIN}" "${CAUSAL_EVAL}" \
    --arm flow --checkpoint "${CHECKPOINT}" --model-config "${MODEL_CONFIG}" \
    --dataset-config "${DATASET_CONFIG}" --challenge "${CHALLENGE}" \
    --device cuda:0 --weights ema --rows-per-view 3 \
    --discrete-decode-mode prefix_recompute --qwen-kernel-mode torch_reference \
    --seed 42 \
    --output "${CAUSAL_REPORT}"
# Reliable-ASR on/drop is one sequential same-physical-GPU job.  Separate GPUs
# are not causally comparable for an autoregressive decoder because a tiny BF16
# difference can change one token and amplify through the rest of the plan.
launch 5 understanding_asr_pair "${LEXICAL_PAIR_RUNNER}" \
    "${CHECKPOINT}" "${OUTPUT_ROOT}" 5
launch 6 cross_process_repro_a_k148 "${PYTHON_BIN}" "${CHALLENGE_EVAL}" \
    "${common_challenge_args[@]}" --draws 8 --k-values 1,4,8 \
    --rows-per-view 1 --output "${REPRO_A_REPORT}"
# The repaired objective targets U's finite SceneSketch inventory.  GPU 7 runs
# the paired deployment/teacher-context exposure audit over all four U evidence
# views (10 rows each).  This is a planner-only root-cause diagnostic and does
# not alter the predeclared promotion thresholds.
launch 7 understanding_sketch_exposure_u40 "${PYTHON_BIN}" "${EXPOSURE_EVAL}" \
    --checkpoint "${CHECKPOINT}" --model-config "${MODEL_CONFIG}" \
    --dataset-config "${DATASET_CONFIG}" --challenge "${CHALLENGE}" \
    --device cuda:0 --weights ema --rows-per-view 10 --tasks understanding \
    --discrete-decode-mode prefix_recompute --qwen-kernel-mode torch_reference \
    --seed 42 --output "${U_EXPOSURE_REPORT}"

failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[${index}]}"; then
        echo "[p11-eval8] failed job=${names[${index}]}" >&2
        failed=1
    else
        echo "[p11-eval8] complete job=${names[${index}]}"
    fi
done
set +e
stop_gpu_monitor
telemetry_status=$?
set -e
trap - EXIT
if [[ "${telemetry_status}" != "0" ]]; then
    echo "[p11-eval8] GPU telemetry did not prove all eight assignments" >&2
    failed=1
fi
if [[ "${failed}" != "0" ]]; then
    missing=0
    for report in "${G_REPORT}" "${U_REPORT}" "${X_REPORT}" "${E_REPORT}" \
        "${CAUSAL_REPORT}" "${ASR_ON_REPORT}" "${ASR_DROP_REPORT}" \
        "${ASR_PAIRING_REPORT}" "${ASR_SUMMARY}" "${REPRO_A_REPORT}" \
        "${U_EXPOSURE_REPORT}" "${GPU_USAGE_REPORT}"; do
        if [[ ! -s "${report}" ]]; then
            echo "[p11-eval8] missing report after worker failure: ${report}" >&2
            missing=1
        fi
    done
    if [[ "${missing}" != "0" ]]; then
        exit 1
    fi
    echo "[p11-eval8] workers returned a scientific FAIL report; continuing the complete evaluation panel" >&2
fi

MERGED_REPORT="${OUTPUT_ROOT}/heldout300_k148_merged.json"
K1_REPORT="${OUTPUT_ROOT}/heldout300_k1_first_draw_projection.json"
REPRO_SUMMARY="${OUTPUT_ROOT}/cross_process_repro_summary.json"
P10_CLOSURE="${OUTPUT_ROOT}/frozen_p10_closure12_100step.json"
DECISION_REPORT="${OUTPUT_ROOT}/MEDIUM_SCALE_DECISION.json"

set +e
"${PYTHON_BIN}" scripts/t2a/eval/merge_sceneplan_p11_v4_challenge_shards.py \
    --shard "${G_REPORT}" --shard "${U_REPORT}" \
    --shard "${X_REPORT}" --shard "${E_REPORT}" \
    --output "${MERGED_REPORT}"
merge_status=$?
projection_status=1
inventory_status=1
if [[ "${merge_status}" == "0" && -s "${MERGED_REPORT}" ]]; then
    "${PYTHON_BIN}" "${K1_PROJECTOR}" \
        --input "${MERGED_REPORT}" --output "${K1_REPORT}"
    projection_status=$?
    if [[ "${projection_status}" == "0" && -s "${K1_REPORT}" ]]; then
        "${PYTHON_BIN}" "${U_INVENTORY_SUMMARIZER}" \
            --report "candidate_inventory_repair=${K1_REPORT}" \
            --report "screen_flow_pre_repair=${SCREEN_FLOW_BASELINE}" \
            --report "d0=${D0_BASELINE}" \
            --report "direct_pre_repair=${DIRECT_BASELINE}" \
            --challenge "${CHALLENGE}" --view audio_evidence_exact_v1 \
            --seed 42 --output "${U_INVENTORY_REPORT}"
        inventory_status=$?
    fi
fi
if "${PYTHON_BIN}" - "${ASR_SUMMARY}" "${ASR_PAIRING_REPORT}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).resolve(strict=True).read_text())
pairing = json.loads(Path(sys.argv[2]).resolve(strict=True).read_text())
if summary.get("status") != "PASS" or pairing.get("status") != "PASS":
    raise SystemExit(1)
PY
then
    asr_status=0
else
    asr_status=1
fi
set -e

# A fresh second process on the same physical GPU proves exact reference replay.
set +e
CUDA_VISIBLE_DEVICES=6 "${PYTHON_BIN}" "${CHALLENGE_EVAL}" \
    "${common_challenge_args[@]}" --draws 8 --k-values 1,4,8 \
    --rows-per-view 1 --output "${REPRO_B_REPORT}" \
    >"${OUTPUT_ROOT}/logs/cross_process_repro_b_k148.log" 2>&1
repro_b_eval_status=$?
set -e
if [[ "${repro_b_eval_status}" != "0" ]]; then
    echo "[p11-eval8] second reproducibility evaluator returned a scientific failure" >&2
fi
set +e
"${PYTHON_BIN}" scripts/t2a/eval/summarize_sceneplan_p11_v4_cross_process_repro.py \
    --first "${REPRO_A_REPORT}" --second "${REPRO_B_REPORT}" \
    --output "${REPRO_SUMMARY}"
repro_status=$?
set -e

# Downstream closure is deliberately started only after the quality report is
# complete.  It consumes the fail-closed first-draw projection of the K=1/4/8
# report, re-decodes those exact K=1 predictions, and binds them to the frozen
# P10-v11 150k executor without another stochastic evaluator pass.
closure_status=1
if [[ "${projection_status}" == "0" && -s "${K1_REPORT}" ]]; then
    set +e
    CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" \
        scripts/t2a/eval/evaluate_sceneplan_p11_v4_p10_closure.py \
        --arm flow --checkpoint "${CHECKPOINT}" --quality-report "${K1_REPORT}" \
        --device cuda:0 --weights ema --rows-per-task 12 --seed 42 --p10-steps 100 \
        --output "${P10_CLOSURE}" \
        >"${OUTPUT_ROOT}/logs/frozen_p10_closure12_100step.log" 2>&1
    closure_status=$?
    set -e
else
    echo "[p11-eval8] P10 closure skipped because the strict K=1 projection is unavailable" >&2
fi

if [[ "${merge_status}" != "0" || "${projection_status}" != "0" \
    || "${inventory_status}" != "0" || "${asr_status}" != "0" \
    || "${repro_status}" != "0" \
    || "${closure_status}" != "0" ]]; then
    echo "[p11-eval8] one or more diagnostics failed; preserving every completed report" >&2
fi

# A scientific REVISE/STOP is a valid completed outcome.  The summarizer exits
# nonzero only when the evidence bundle itself is missing or malformed.
decision_status=1
if [[ -s "${MERGED_REPORT}" && -s "${CAUSAL_REPORT}" \
    && -s "${ASR_SUMMARY}" && -s "${REPRO_SUMMARY}" \
    && -s "${P10_CLOSURE}" ]]; then
    set +e
    "${PYTHON_BIN}" "${DECISION_SUMMARIZER}" \
        --candidate-quality "${MERGED_REPORT}" \
        --candidate-causal "${CAUSAL_REPORT}" \
        --candidate-asr "${ASR_SUMMARY}" \
        --candidate-repro "${REPRO_SUMMARY}" \
        --candidate-p10-closure "${P10_CLOSURE}" \
        --d0-quality "${D0_BASELINE}" \
        --direct-quality "${DIRECT_BASELINE}" \
        --screen-flow-quality "${SCREEN_FLOW_BASELINE}" \
        --training-report "${TRAINING_REPORT}" \
        --launch-contract "${LAUNCH_CONTRACT}" \
        --pretrain-report "${PRETRAIN_REPORTS[0]}" \
        --pretrain-report "${PRETRAIN_REPORTS[1]}" \
        --pretrain-report "${PRETRAIN_REPORTS[2]}" \
        --pretrain-report "${PRETRAIN_REPORTS[3]}" \
        --pretrain-report "${PRETRAIN_REPORTS[4]}" \
        --pretrain-report "${PRETRAIN_REPORTS[5]}" \
        --pretrain-report "${PRETRAIN_REPORTS[6]}" \
        --pretrain-report "${PRETRAIN_REPORTS[7]}" \
        --output "${DECISION_REPORT}" \
        >"${OUTPUT_ROOT}/logs/medium_scale_decision.log" 2>&1
    decision_status=$?
    set -e
fi
if [[ "${decision_status}" != "0" || ! -s "${DECISION_REPORT}" ]]; then
    echo "[p11-eval8] medium-scale decision artifact could not be constructed" >&2
    exit 1
fi

set +e
"${PYTHON_BIN}" - "${OUTPUT_ROOT}" "${CHECKPOINT}" "${EVAL_PROTOCOL}" \
    "${PROTOCOL_VERIFICATION_REPORT}" \
    "${MERGED_REPORT}" "${K1_REPORT}" "${CAUSAL_REPORT}" "${ASR_SUMMARY}" \
    "${REPRO_SUMMARY}" "${U_EXPOSURE_REPORT}" "${U_INVENTORY_REPORT}" \
    "${GPU_USAGE_REPORT}" \
    "${P10_CLOSURE}" "${DECISION_REPORT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
checkpoint = Path(sys.argv[2]).resolve()
protocol_path = Path(sys.argv[3]).resolve(strict=True)
report_paths = [Path(value).resolve() for value in sys.argv[4:]]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
statuses = {path.name: report.get("status") for path, report in zip(report_paths, reports)}
decision = reports[-1].get("decision")
summary = {
    "schema": "stable_audio_tools.p11_v4_scale10k_posttrain_eval",
    "schema_version": 5,
    "status": "PASS",
    "panel_complete": True,
    "all_component_statuses_pass": all(value == "PASS" for value in statuses.values()),
    "scientific_decision": decision,
    "frozen_threshold_decision": reports[-1].get("frozen_threshold_decision"),
    "canonical_promotion_authorized": reports[-1].get(
        "canonical_promotion_authorized", False
    ),
    "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
    "evaluation_protocol": {
        "path": str(protocol_path),
        "sha256": sha256(protocol_path),
        "status": json.loads(protocol_path.read_text(encoding="utf-8")).get("status"),
        "frozen_before_candidate_checkpoint": True,
    },
    "seed": 42,
    "gpu_count": 8,
    "gpu_assignment_contract": {
        "contract": "p11_posttrain_eight_independent_diagnostics_v1",
        "all_physical_gpus_receive_scientific_work_required": True,
        "gpu_0": "generation_k148",
        "gpu_1": "understanding_degraded_k148",
        "gpu_2": "exact_compatibility_k148",
        "gpu_3": "editing_counterfactual_k148",
        "gpu_4": "thought_causal_interventions",
        "gpu_5": "same_gpu_reliable_asr_on_drop_pair",
        "gpu_6": "cross_process_reference_replay",
        "gpu_7": "understanding_scene_sketch_exposure_u40",
    },
    "qwen_scientific_kernel": "torch_reference",
    "reports": {
        path.name: {"path": str(path), "sha256": sha256(path), "status": status}
        for path, status in zip(report_paths, statuses.values())
    },
}
payload = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
summary["report_sha256_without_self"] = hashlib.sha256(payload).hexdigest()
output = root / "POSTTRAIN_EVAL_SUMMARY.json"
output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
PY
summary_status=$?
set -e

echo "P11_SCALE10K_EVAL_COMPLETE=${OUTPUT_ROOT}/POSTTRAIN_EVAL_SUMMARY.json"
exit "${summary_status}"
