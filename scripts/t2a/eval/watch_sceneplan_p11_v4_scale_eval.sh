#!/usr/bin/env bash
set -euo pipefail

# Wait for one terminal 10k checkpoint and its PASS training report, then hand
# the now-idle eight GPUs directly to the canonical post-training evaluator.
# This watcher never launches training and never accepts a non-terminal ckpt.

REPO_ROOT="${REPO_ROOT:-/mnt/sdc/stable-audio-tools-workspace}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
EVAL_RUNNER="${REPO_ROOT}/scripts/t2a/eval/run_sceneplan_p11_v4_scale_eval_8gpu.sh"
RUN_ROOT="${1:-}"
OUTPUT_ROOT="${2:-}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-21600}"

if [[ -z "${RUN_ROOT}" || -z "${OUTPUT_ROOT}" ]]; then
    echo "usage: $0 RUN_ROOT OUTPUT_ROOT" >&2
    exit 2
fi
if [[ ! "${POLL_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ \
    || ! "${MAX_WAIT_SECONDS}" =~ ^[0-9]+$ \
    || "${MAX_WAIT_SECONDS}" == "0" ]]; then
    echo "invalid POLL_SECONDS or MAX_WAIT_SECONDS" >&2
    exit 2
fi
RUN_ROOT="$(readlink -f -- "${RUN_ROOT}")"
OUTPUT_ROOT="$(readlink -m -- "${OUTPUT_ROOT}")"
TRAINING_REPORT="${RUN_ROOT}/scale_trial_report.json"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
LOCK_FILE="${RUN_ROOT}/.posttrain_eval_watcher.lock"

for required in "${PYTHON_BIN}" "${EVAL_RUNNER}" "${RUN_ROOT}"; do
    if [[ ! -e "${required}" ]]; then
        echo "watcher prerequisite is missing: ${required}" >&2
        exit 1
    fi
done
if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "refusing existing evaluation output: ${OUTPUT_ROOT}" >&2
    exit 2
fi

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "another post-training evaluator watcher already owns ${RUN_ROOT}" >&2
    exit 2
fi

sleep_pid=""
cleanup_waiter() {
    if [[ -n "${sleep_pid}" ]] && kill -0 "${sleep_pid}" 2>/dev/null; then
        kill -TERM "${sleep_pid}" 2>/dev/null || true
    fi
}
trap cleanup_waiter EXIT
trap 'exit 143' INT TERM
wait_interval() {
    sleep "${POLL_SECONDS}" &
    sleep_pid="$!"
    wait "${sleep_pid}"
    sleep_pid=""
}

started_epoch="$(date +%s)"
checkpoint=""
echo "[p11-eval-watch] waiting for terminal 10k evidence under ${RUN_ROOT}"
while [[ -z "${checkpoint}" ]]; do
    now_epoch="$(date +%s)"
    if (( now_epoch - started_epoch >= MAX_WAIT_SECONDS )); then
        echo "timed out waiting for the terminal 10k checkpoint" >&2
        exit 1
    fi
    if [[ -s "${TRAINING_REPORT}" && -d "${CHECKPOINT_DIR}" ]]; then
        report_status="$("${PYTHON_BIN}" - "${TRAINING_REPORT}" <<'PY'
import json
import sys
from pathlib import Path

try:
    report = json.loads(Path(sys.argv[1]).resolve(strict=True).read_text())
except (OSError, json.JSONDecodeError):
    print("INCOMPLETE")
else:
    print(str(report.get("status", "MISSING")))
PY
)"
        if [[ "${report_status}" == "INCOMPLETE" ]]; then
            wait_interval
            continue
        elif [[ "${report_status}" != "PASS" ]]; then
            echo "terminal training report is not PASS: ${report_status}" >&2
            exit 1
        fi
        mapfile -t checkpoints < <(
            find "${CHECKPOINT_DIR}" -maxdepth 1 -type f \
                -name '*step=10000.ckpt' -print | sort
        )
        if [[ "${#checkpoints[@]}" == "1" ]]; then
            checkpoint="$(readlink -f -- "${checkpoints[0]}")"
        elif [[ "${#checkpoints[@]}" -gt "1" ]]; then
            echo "ambiguous terminal checkpoint inventory: ${#checkpoints[@]}" >&2
            exit 1
        fi
    fi
    if [[ -z "${checkpoint}" ]]; then
        wait_interval
    fi
done

echo "[p11-eval-watch] terminal evidence found: ${checkpoint}"
echo "[p11-eval-watch] waiting for all training CUDA processes to exit"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | awk \
    'BEGIN { found=0 } /^[[:space:]]*[0-9]+[[:space:]]*$/ { found=1 } END { exit found ? 0 : 1 }'; do
    now_epoch="$(date +%s)"
    if (( now_epoch - started_epoch >= MAX_WAIT_SECONDS )); then
        echo "timed out waiting for the eight GPUs to become idle" >&2
        exit 1
    fi
    wait_interval
done

echo "[p11-eval-watch] GPUs idle; starting canonical eight-way diagnostics"
exec env OUTPUT_ROOT="${OUTPUT_ROOT}" "${EVAL_RUNNER}" "${checkpoint}"
