#!/usr/bin/env bash
set -euo pipefail

# Evaluate one P10-v12 arm on the immutable 400 Music + 400 Sound + 400 Speech
# panel. CHECKPOINT_SPECS is a comma-separated LOCAL_STEP=/path list.

REPO_ROOT="${REPO_ROOT:-/home/tanhe/dataset_storage/stable-audio-tools}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
ARM="${ARM:?set ARM to dense, moe, attention, or combined}"
MODEL_CONFIG="${MODEL_CONFIG:?set MODEL_CONFIG to the matching P10-v12 config}"
CHECKPOINT_SPECS="${CHECKPOINT_SPECS:?set CHECKPOINT_SPECS, e.g. 2500=/path/model.ckpt}"
EVAL_ROOT="${EVAL_ROOT:?set EVAL_ROOT to a new or matching evaluation directory}"
TRAINING_LAUNCH_CONTRACT="${TRAINING_LAUNCH_CONTRACT:-}"
GPU_IDS="${GPU_IDS:-2,3,4,5,6,7}"
MIN_FREE_GIB="${MIN_FREE_GIB:-40}"
SOURCE_EVAL_ROOT="${SOURCE_EVAL_ROOT:-/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/speech_expansion_noalign_15s_v1/evaluation/p10_v11_balanced_1200_ckpt110k_150k_semantic_v2}"

cd "$REPO_ROOT"

IFS=',' read -r -a gpu_array <<<"$GPU_IDS"
if (( ${#gpu_array[@]} != 6 )); then
    echo "[p10-v12-eval] exactly six GPUs are required; got $GPU_IDS" >&2
    exit 2
fi
declare -A seen_gpus=()
for index in "${!gpu_array[@]}"; do
    gpu="${gpu_array[$index]//[[:space:]]/}"
    if [[ ! "$gpu" =~ ^[0-9]+$ ]] || [[ -n "${seen_gpus[$gpu]:-}" ]]; then
        echo "[p10-v12-eval] invalid or duplicate GPU in GPU_IDS=$GPU_IDS" >&2
        exit 2
    fi
    seen_gpus[$gpu]=1
    gpu_array[$index]="$gpu"
done

lock_root="/mnt/sdc/ckpts/dit/.locks"
mkdir -p "$lock_root"
lock_key="${GPU_IDS//,/__}"
exec 9>"$lock_root/p10_v12_gpu_${lock_key}.lock"
if ! flock -n 9; then
    echo "[p10-v12-eval] a P10-v12 job already owns GPUs $GPU_IDS" >&2
    exit 1
fi

for gpu in "${gpu_array[@]}"; do
    active="$({ nvidia-smi -i "$gpu" \
        --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits || true; } | sed '/^[[:space:]]*$/d')"
    if [[ -n "$active" ]]; then
        echo "[p10-v12-eval] GPU $gpu is occupied: $active" >&2
        exit 1
    fi
done

if [[ -r /var/log/kern.log ]] && grep -Eq \
    'gpuHandleSanityCheckRegReadError|Possible bad register read' \
    < <(tail -n 4000 /var/log/kern.log); then
    echo "[p10-v12-eval] refusing CUDA launch due to recent NVIDIA bad-register errors" >&2
    exit 1
fi

mkdir -p "$EVAL_ROOT/logs/inference" "$EVAL_ROOT/logs/metrics"
free_kib="$(df -Pk "$EVAL_ROOT" | awk 'NR==2 {print $4}')"
if (( free_kib < MIN_FREE_GIB * 1024 * 1024 )); then
    echo "[p10-v12-eval] insufficient free space at $EVAL_ROOT" >&2
    exit 1
fi

IFS=',' read -r -a checkpoint_specs <<<"$CHECKPOINT_SPECS"
if (( ${#checkpoint_specs[@]} == 0 )); then
    echo "[p10-v12-eval] CHECKPOINT_SPECS is empty" >&2
    exit 2
fi
builder_args=(
    --source-root "$SOURCE_EVAL_ROOT"
    --output-root "$EVAL_ROOT"
    --arm "$ARM"
    --model-config "$MODEL_CONFIG"
)
steps=()
for spec in "${checkpoint_specs[@]}"; do
    spec="${spec#${spec%%[![:space:]]*}}"
    spec="${spec%${spec##*[![:space:]]}}"
    step="${spec%%=*}"
    path="${spec#*=}"
    if [[ "$spec" != *=* || ! "$step" =~ ^[1-9][0-9]*$ || -z "$path" ]]; then
        echo "[p10-v12-eval] invalid checkpoint spec: $spec" >&2
        exit 2
    fi
    steps+=("$step")
    builder_args+=(--checkpoint "$step=$path")
done
if [[ -n "$TRAINING_LAUNCH_CONTRACT" ]]; then
    builder_args+=(--training-launch-contract "$TRAINING_LAUNCH_CONTRACT")
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

"$PYTHON_BIN" scripts/t2a/eval/build_p10_v12_upgrade_eval_contract.py \
    "${builder_args[@]}" >"$EVAL_ROOT/logs/build_contract.log" 2>&1

for step in "${steps[@]}"; do
    echo "$(date --iso-8601=seconds) p10-v12 inference arm=$ARM step=$step start"
    pids=()
    for shard in 0 1 2 3 4 5; do
        gpu="${gpu_array[$shard]}"
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" \
            scripts/t2a/eval/generate_sceneplan_dit_p10_panel.py \
            --eval-root "$EVAL_ROOT" \
            --checkpoint-step "$step" \
            --device cuda:0 \
            --shard-index "$shard" \
            --num-shards 6 \
            --batch-size 1 \
            >"$EVAL_ROOT/logs/inference/step_${step}_shard_${shard}.log" 2>&1 &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    if (( failed != 0 )); then
        echo "[p10-v12-eval] inference failed for step=$step" >&2
        exit 1
    fi

    "$PYTHON_BIN" - "$EVAL_ROOT" "$step" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
step = int(sys.argv[2])
contract = json.loads((root / "EVAL_CONTRACT.json").read_text(encoding="utf-8"))
panel = [
    json.loads(line)
    for line in (root / contract["test_set"]["panel_filename"])
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]
seen = []
for row in panel:
    path = (
        root
        / "outputs"
        / f"step_{step:06d}"
        / row["domain"]
        / row["panel_id"]
        / "metadata.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not (
        value.get("status") == "PASS"
        and int(value["checkpoint_step"]) == step
        and int(value.get("semantic_caption_compiler_version", -1)) == 2
        and value["panel_id"] == row["panel_id"]
        and value["sample_id"] == row["sample_id"]
    ):
        raise RuntimeError(f"invalid generated metadata: {path}")
    seen.append(value["panel_id"])
if len(seen) != 1200 or len(set(seen)) != 1200:
    raise RuntimeError(f"expected 1200 unique outputs, got {len(seen)}")
print(json.dumps({"status": "PASS", "step": step, "outputs": len(seen)}))
PY
    date --iso-8601=seconds >"$EVAL_ROOT/STEP_${step}_INFERENCE_COMPLETE"
    echo "$(date --iso-8601=seconds) p10-v12 inference arm=$ARM step=$step complete"
done
date --iso-8601=seconds >"$EVAL_ROOT/INFERENCE_COMPLETE"

metric_pids=()
"$PYTHON_BIN" scripts/t2a/eval/score_sceneplan_dit_p10_core.py \
    --eval-root "$EVAL_ROOT" >"$EVAL_ROOT/logs/metrics/core.log" 2>&1 &
metric_pids+=("$!")
CUDA_VISIBLE_DEVICES="${gpu_array[0]}" "$PYTHON_BIN" \
    scripts/t2a/eval/score_sceneplan_dit_p10_clap.py \
    --eval-root "$EVAL_ROOT" --device cuda:0 \
    >"$EVAL_ROOT/logs/metrics/clap.log" 2>&1 &
metric_pids+=("$!")
CUDA_VISIBLE_DEVICES="${gpu_array[1]}" "$PYTHON_BIN" \
    scripts/t2a/eval/score_sceneplan_dit_p10_distributional.py \
    --eval-root "$EVAL_ROOT" --device cuda:0 \
    >"$EVAL_ROOT/logs/metrics/distributional.log" 2>&1 &
metric_pids+=("$!")
CUDA_VISIBLE_DEVICES="${gpu_array[2]}" "$PYTHON_BIN" \
    scripts/t2a/eval/score_sceneplan_dit_p10_speech.py \
    --eval-root "$EVAL_ROOT" --device-index 0 \
    >"$EVAL_ROOT/logs/metrics/speech.log" 2>&1 &
metric_pids+=("$!")

failed=0
for pid in "${metric_pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
if (( failed != 0 )); then
    echo "[p10-v12-eval] one or more metric stages failed" >&2
    exit 1
fi

"$PYTHON_BIN" scripts/t2a/eval/summarize_sceneplan_dit_p10_panel.py \
    --eval-root "$EVAL_ROOT" >"$EVAL_ROOT/logs/metrics/summarize.log" 2>&1
date --iso-8601=seconds >"$EVAL_ROOT/EVALUATION_COMPLETE"
echo "P10_V12_BALANCED_1200_EVALUATION_COMPLETE=$EVAL_ROOT"
