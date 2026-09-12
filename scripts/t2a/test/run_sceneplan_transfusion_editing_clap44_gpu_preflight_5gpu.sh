#!/usr/bin/env bash
set -euo pipefail
REPO="${P10_REPO:-${AMBIT_CKPT_ROOT}/stable-audio-tools-workspace}"
PROBE_PARENT="${CLAP44_PROBE_ROOT:-${AMBIT_CKPT_ROOT}/transfusion_editing/diagnostics/clap44_gpu}"
PROBE_SECONDS="${CLAP44_PROBE_MAX_WALL_SECONDS:-900}"
PROBE_CONFIG="${CLAP44_PROBE_CONFIG:-$REPO/stable_audio_tools/configs/model_configs/txt2audio/t2a/editing_clap44_v1.json}"
GPUS="${EDITING_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ $# -ne 0 || ! "$PROBE_SECONDS" =~ ^[0-9]+$ ]] || (( PROBE_SECONDS < 60 || PROBE_SECONDS > 900 )); then
    echo '[clap44 probe] no positional overrides; wall budget must be 60-900 seconds' >&2
    exit 2
fi
if [[ -z "$GPUS" ]]; then
    echo '[clap44 probe] set EDITING_GPUS to the physical GPUs assigned to this probe' >&2
    exit 2
fi
if [[ ! -r "$PROBE_CONFIG" ]]; then
    echo '[clap44 probe] experiment configuration is not readable' >&2
    exit 2
fi
PROBE_CONFIG="$(realpath -e -- "$PROBE_CONFIG")"
if [[ -z "${EDITING_GPU_LEASES:-}" ]]; then
    exec "$REPO/.venv/bin/python" "$REPO/scripts/t2a/train/editing_gpu_runtime.py" \
        --gpus "$GPUS" -- bash "$0" "$@"
fi
IFS=',' read -r -a GPU_IDS <<< "$EDITING_GPUS"
NPROC="${#GPU_IDS[@]}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
ulimit -Sn 65536
mkdir -p "$PROBE_PARENT"
PROBE_OUTPUT="$(mktemp -d "$PROBE_PARENT/probe-$(date +%Y%m%dT%H%M%S)-XXXXXX")"
printf '[clap44 probe] artifacts: %s\n' "$PROBE_OUTPUT"
cd "$REPO"
set +e
timeout --signal=TERM --kill-after=20s "${PROBE_SECONDS}s" \
    "$REPO/.venv/bin/torchrun" --standalone --nproc_per_node="$NPROC" \
    "$REPO/scripts/t2a/test/profile_sceneplan_transfusion_editing_clap44_gpu.py" \
    --config "$PROBE_CONFIG" --output "$PROBE_OUTPUT" --max-wall-seconds "$PROBE_SECONDS" >"$PROBE_OUTPUT/torchrun.log" 2>&1
PROBE_EXIT=$?
set -e
"$REPO/.venv/bin/python" - "$PROBE_OUTPUT" "$PROBE_EXIT" "$PROBE_SECONDS" <<'PY'
import datetime,json,pathlib,sys
output=pathlib.Path(sys.argv[1])
value={'status':'LAUNCH_EXITED','exit_code':int(sys.argv[2]),'max_wall_seconds':int(sys.argv[3]),
       'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'quality_gate_passed':False}
(output/'LAUNCH_EXIT.json').write_text(json.dumps(value,indent=2)+'\n')
PY
exit "$PROBE_EXIT"
