#!/usr/bin/env python3
"""Prepare the 350k handoff run with a cosine HF overshoot schedule."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _repo import repo_root

REPO = repo_root()
SOURCE_ROOT = Path("/mnt/sdc/ckpts/vae_ds1024_z64_hf_overshoot_full_1019k_8gpu")
OUTPUT_ROOT = Path("/mnt/sdc/ckpts/vae_ds1024_z64_hf_overshoot_decay_350k_8gpu")
SOURCE_MODEL = SOURCE_ROOT / "configs/model_hf_overshoot_full.json"
SOURCE_DATASET = SOURCE_ROOT / "configs/dataset_frozen_1018957.json"
RUN_NAME = "vae_ds1024_z64_hf_overshoot_decay_350k_8gpu"
SOURCE_SESSION = "vae_hf_overshoot_full_8gpu"
HANDOFF_SESSION = "vae_hf_overshoot_handoff_350k"
CHECKPOINT_STEP = 350_000
MAX_STEPS = 2_000_000
CHECKPOINT_EVERY = 50_000

SCHEDULE = {
    "type": "cosine",
    "start_step": CHECKPOINT_STEP,
    "end_step": 650_000,
    "start_weight": 0.20,
    "end_weight": 0.05,
}


def write_text(path: Path, value: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    if executable:
        path.chmod(0o775)


def main() -> None:
    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(SOURCE_MODEL)
    if not SOURCE_DATASET.is_file():
        raise FileNotFoundError(SOURCE_DATASET)

    model = json.loads(SOURCE_MODEL.read_text(encoding="utf-8"))
    dataset_text = SOURCE_DATASET.read_text(encoding="utf-8")
    hf_loss = model["training"]["loss_configs"]["high_frequency_overshoot"]
    if hf_loss["weights"]["overshoot"] != SCHEDULE["start_weight"]:
        raise ValueError("The 350k handoff must start from overshoot weight 0.2")
    hf_loss["schedule"] = SCHEDULE
    model["_experiment"] = {
        "name": RUN_NAME,
        "purpose": "resume 350k full VAE run with cosine HF overshoot decay",
        "source_run": str(SOURCE_ROOT),
        "resume_step": CHECKPOINT_STEP,
        "max_steps": MAX_STEPS,
        "checkpoint_every": CHECKPOINT_EVERY,
        "num_gpus": 8,
        "batch_size_per_gpu": 2,
        "global_batch_size": 16,
        "schedule": SCHEDULE,
    }

    config_dir = OUTPUT_ROOT / "configs"
    script_dir = OUTPUT_ROOT / "scripts"
    model_path = config_dir / "model_hf_overshoot_decay_350k.json"
    dataset_path = config_dir / "dataset_frozen_1018957.json"
    write_text(model_path, json.dumps(model, indent=2) + "\n")
    write_text(dataset_path, dataset_text)

    run_script = f"""#!/usr/bin/env bash
set -euo pipefail

ROOT={OUTPUT_ROOT}
REPO={REPO}
SOURCE_ROOT={SOURCE_ROOT}
CHECKPOINT_STEP={CHECKPOINT_STEP}

find_resume_checkpoint() {{
  find "$SOURCE_ROOT/checkpoints" -type f -name "*step=${{CHECKPOINT_STEP}}.ckpt" -size +1G -print | sort
}}

mapfile -t matches < <(find_resume_checkpoint)
if (( ${{#matches[@]}} != 1 )); then
  echo "expected exactly one complete checkpoint at step=${{CHECKPOINT_STEP}}, found ${{#matches[@]}}" >&2
  printf '%s\\n' "${{matches[@]}}" >&2
  exit 1
fi
RESUME_CKPT="${{matches[0]}}"

cd "$REPO"
mkdir -p "$ROOT/logs" "$ROOT/checkpoints" "$ROOT/wandb"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${{WANDB_MODE:-offline}}
export WANDB_DIR="$ROOT/wandb"
export WANDB_NAME={RUN_NAME}

echo "[hf-decay] resuming from $RESUME_CKPT"
uv run --frozen python train_4ch.py \\
  --model-config {model_path} \\
  --dataset-config {dataset_path} \\
  --ckpt-path "$RESUME_CKPT" \\
  --name {RUN_NAME} \\
  --batch-size 2 \\
  --num-workers 8 \\
  --num-gpus 8 \\
  --precision bf16-mixed \\
  --strategy ddp_find_unused_parameters_true \\
  --save-dir "$ROOT/checkpoints" \\
  --checkpoint-every {CHECKPOINT_EVERY} \\
  --max-steps {MAX_STEPS} \\
  --logger wandb \\
  --seed 42 2>&1 | tee -a "$ROOT/logs/train.log"
"""
    run_path = script_dir / "run_decay_8gpu.sh"
    write_text(run_path, run_script, executable=True)

    handoff_script = f"""#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT={SOURCE_ROOT}
SOURCE_SESSION={SOURCE_SESSION}
CHECKPOINT_STEP={CHECKPOINT_STEP}
RUN_SCRIPT={run_path}

find_resume_checkpoint() {{
  find "$SOURCE_ROOT/checkpoints" -type f -name "*step=${{CHECKPOINT_STEP}}.ckpt" -size +1G -print | sort
}}

echo "[handoff] waiting for a complete step=${{CHECKPOINT_STEP}} checkpoint"
while true; do
  mapfile -t matches < <(find_resume_checkpoint)
  if (( ${{#matches[@]}} == 1 )); then
    candidate="${{matches[0]}}"
    size_before=$(stat -c '%s' "$candidate")
    sleep 10
    size_after=$(stat -c '%s' "$candidate")
    if [[ "$size_before" == "$size_after" ]]; then
      echo "[handoff] checkpoint ready: $candidate"
      break
    fi
  fi
  sleep 60
done

if tmux has-session -t "{SOURCE_SESSION}" 2>/dev/null; then
  echo "[handoff] stopping source session {SOURCE_SESSION}"
  tmux send-keys -t "{SOURCE_SESSION}" C-c
  while tmux has-session -t "{SOURCE_SESSION}" 2>/dev/null; do
    sleep 10
  done
else
  echo "[handoff] source session {SOURCE_SESSION} is not running" >&2
  exit 1
fi

echo "[handoff] starting decay run"
exec "$RUN_SCRIPT"
"""
    handoff_path = script_dir / "handoff_after_350k.sh"
    write_text(handoff_path, handoff_script, executable=True)

    launch_script = f"""#!/usr/bin/env bash
set -euo pipefail
if tmux has-session -t {HANDOFF_SESSION} 2>/dev/null; then
  echo "tmux session already exists: {HANDOFF_SESSION}" >&2
  exit 1
fi
tmux new-session -d -s {HANDOFF_SESSION} {handoff_path}
echo "launched: {HANDOFF_SESSION}"
"""
    write_text(script_dir / "launch_handoff.sh", launch_script, executable=True)

    status_script = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT={OUTPUT_ROOT}
tmux list-sessions 2>/dev/null | rg '^(vae_hf_overshoot_handoff_350k|vae_hf_overshoot_decay_350k_8gpu):' || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
find "$ROOT/checkpoints" -type f -name '*.ckpt' -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\\n' 2>/dev/null | sort | tail -5
tail -3 "$ROOT/logs/train.log" 2>/dev/null || true
"""
    write_text(script_dir / "status.sh", status_script, executable=True)

    print(f"prepared {OUTPUT_ROOT}")
    print(f"handoff launcher: {script_dir / 'launch_handoff.sh'}")


if __name__ == "__main__":
    main()
