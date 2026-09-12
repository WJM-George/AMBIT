#!/usr/bin/env python3
"""Build the selected 8-GPU full VAE training run with HF overshoot loss."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _repo import repo_root

REPO = repo_root()
SOURCE_ROOT = Path("/mnt/sdc/ckpts/vae_ds1024_z64_hf_ab_4gpu")
OUTPUT_ROOT = Path("/mnt/sdc/ckpts/vae_ds1024_z64_hf_overshoot_full_1019k_8gpu")
SOURCE_MODEL = SOURCE_ROOT / "configs/model_hf_loss_only.json"
SOURCE_DATASET = SOURCE_ROOT / "configs/dataset_frozen_1018957.json"
RESUME_CKPT = (
    SOURCE_ROOT
    / "loss_only/checkpoints/vae_ds1024_z64_hf_loss_only_4gpu/lj1h2w20/"
    "checkpoints/epoch=0-step=40000.ckpt"
)
RUN_NAME = "vae_ds1024_z64_hf_overshoot_full_1019k_8gpu"
TMUX_NAME = "vae_hf_overshoot_full_8gpu"
MAX_STEPS = 2_000_000
CHECKPOINT_EVERY = 50_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, value: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    if executable:
        path.chmod(0o775)


def main() -> None:
    if not RESUME_CKPT.is_file():
        raise FileNotFoundError(RESUME_CKPT)

    model = json.loads(SOURCE_MODEL.read_text(encoding="utf-8"))
    dataset_text = SOURCE_DATASET.read_text(encoding="utf-8")
    decoder = model["model"]["decoder"]["config"]
    if decoder.get("antialias_activation", False):
        raise ValueError("Selected full run must not enable decoder anti-alias activation")
    if "high_frequency_overshoot" not in model["training"]["loss_configs"]:
        raise ValueError("Selected full run is missing high_frequency_overshoot")

    model["_experiment"] = {
        "name": RUN_NAME,
        "purpose": "full VAE training with selected asymmetric HF overshoot loss",
        "resume_from": str(RESUME_CKPT),
        "resume_step": 40_000,
        "max_steps": MAX_STEPS,
        "checkpoint_every": CHECKPOINT_EVERY,
        "num_gpus": 8,
        "batch_size_per_gpu": 2,
        "global_batch_size": 16,
        "num_workers_per_rank": 8,
        "seed": 42,
        "selection": {
            "step20k_sls_hf_excess_median_db": 1.687,
            "step20k_w_si_sdr_median_db": 6.099,
            "decoder_antialias_activation": False,
        },
    }

    config_dir = OUTPUT_ROOT / "configs"
    script_dir = OUTPUT_ROOT / "scripts"
    model_path = config_dir / "model_hf_overshoot_full.json"
    dataset_path = config_dir / "dataset_frozen_1018957.json"
    write_text(model_path, json.dumps(model, indent=2) + "\n")
    write_text(dataset_path, dataset_text)

    run_script = f"""#!/usr/bin/env bash
set -euo pipefail

ROOT={OUTPUT_ROOT}
REPO={REPO}
RESUME_CKPT={RESUME_CKPT}

cd "$REPO"
mkdir -p "$ROOT/logs" "$ROOT/checkpoints" "$ROOT/wandb"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${{WANDB_MODE:-offline}}
export WANDB_DIR="$ROOT/wandb"
export WANDB_NAME={RUN_NAME}

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
    run_path = script_dir / "run_full_8gpu.sh"
    write_text(run_path, run_script, executable=True)

    launch_script = f"""#!/usr/bin/env bash
set -euo pipefail
tmux has-session -t {TMUX_NAME} 2>/dev/null && {{
  echo "tmux session already exists: {TMUX_NAME}" >&2
  exit 1
}}
tmux new-session -d -s {TMUX_NAME} {run_path}
echo "launched: {TMUX_NAME}"
"""
    write_text(script_dir / "launch.sh", launch_script, executable=True)

    status_script = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT={OUTPUT_ROOT}
tmux list-sessions 2>/dev/null | rg '^{TMUX_NAME}:' || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
find "$ROOT/checkpoints" -type f -name '*.ckpt' -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' 2>/dev/null | sort | tail -5
tail -3 "$ROOT/logs/train.log" 2>/dev/null || true
"""
    write_text(script_dir / "status.sh", status_script, executable=True)

    manifest = {
        "run_name": RUN_NAME,
        "tmux_session": TMUX_NAME,
        "selected_variant": "hf_loss_only",
        "resume_checkpoint": str(RESUME_CKPT),
        "resume_checkpoint_size": RESUME_CKPT.stat().st_size,
        "max_steps": MAX_STEPS,
        "checkpoint_every": CHECKPOINT_EVERY,
        "num_gpus": 8,
        "batch_size_per_gpu": 2,
        "global_batch_size": 16,
        "files": {
            str(model_path.relative_to(OUTPUT_ROOT)): sha256(model_path),
            str(dataset_path.relative_to(OUTPUT_ROOT)): sha256(dataset_path),
            str(run_path.relative_to(OUTPUT_ROOT)): sha256(run_path),
        },
    }
    write_text(OUTPUT_ROOT / "RUN.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
