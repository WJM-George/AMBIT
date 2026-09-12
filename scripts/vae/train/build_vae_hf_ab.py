#!/usr/bin/env python3
"""Build a frozen two-run VAE high-frequency artifact A/B experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


import sys
from pathlib import Path as _PathForRepo
_SCRIPTS_DIR = _PathForRepo(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _repo import repo_root
SAT_ROOT = repo_root()
DEFAULT_BASE_MODEL = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/vae_ds1024_z64_baseline_full_1019k/configs/"
    "model_baseline_ds1024_z64.json"
)
DEFAULT_DATASET = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/vae_ds1024_z64_baseline_full_1019k/configs/"
    "dataset_train_vae_4ch_v2_frozen.json"
)
DEFAULT_OUT = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/vae_ds1024_z64_hf_ab_4gpu")
STEREO_CKPT = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/stable-audio-open-1.0/model.safetensors")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_script(
    *,
    out_root: Path,
    config: Path,
    dataset: Path,
    gpu_ids: str,
    run_name: str,
    variant_dir: str,
) -> str:
    log_path = out_root / "logs" / f"{variant_dir}.log"
    checkpoint_root = out_root / variant_dir / "checkpoints"
    wandb_root = out_root / "wandb"
    return f"""#!/usr/bin/env bash
set -euo pipefail

cd {SAT_ROOT}
mkdir -p {log_path.parent} {checkpoint_root} {wandb_root}

export CUDA_VISIBLE_DEVICES={gpu_ids}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${{WANDB_MODE:-offline}}
export WANDB_DIR={wandb_root}

uv run --frozen python train_4ch.py \\
  --model-config {config} \\
  --dataset-config {dataset} \\
  --pretrained-ckpt-2ch {STEREO_CKPT} \\
  --name {run_name} \\
  --batch-size 2 \\
  --num-workers 8 \\
  --num-gpus 4 \\
  --precision bf16-mixed \\
  --strategy ddp_find_unused_parameters_true \\
  --save-dir {checkpoint_root} \\
  --checkpoint-every 10000 \\
  --max-steps 50000 \\
  --logger wandb \\
  --seed 42 2>&1 | tee {log_path}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    base_model = json.loads(args.base_model.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset_config.read_text(encoding="utf-8"))
    out_root = args.out_root.resolve()
    configs_dir = out_root / "configs"
    scripts_dir = out_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    overshoot_config = {
        "type": "asymmetric_stft",
        "config": {
            "fmin": 8000,
            "fmax": 22050,
            "n_fft": 2048,
            "hop_length": 512,
            "win_length": 2048,
            "margin_db": 1.0,
            "floor_db": -80.0,
            "log_excess_scale_db": 20.0,
            "linear_weight": 0.1,
        },
        "weights": {"overshoot": 0.2},
    }

    common = json.loads(json.dumps(base_model))
    common.pop("_trial", None)
    common["training"]["loss_configs"].pop("band_mel", None)
    common["training"]["loss_configs"]["high_frequency_overshoot"] = overshoot_config
    common["_experiment"] = {
        "name": "vae_ds1024_z64_hf_ab_4gpu",
        "purpose": "suppress spurious 8-22.05 kHz reconstruction energy",
        "initialization": str(STEREO_CKPT),
        "training_steps": 50000,
        "checkpoint_every": 10000,
        "gpus_per_run": 4,
        "batch_size_per_gpu": 2,
        "seed": 42,
        "acceptance": {
            "sls_hf_excess_median_db_max": 3.0,
            "common_metric_regression_max_percent": 3.0,
            "full_band_hf_attenuation_regression_max_db": 1.0,
        },
    }

    loss_only = json.loads(json.dumps(common))
    loss_only["_experiment"]["variant"] = "hf_loss_only"
    loss_only["_experiment"]["only_difference_vs_other"] = (
        "decoder antialias_activation is disabled"
    )

    antialias = json.loads(json.dumps(common))
    antialias["model"]["decoder"]["config"]["antialias_activation"] = True
    antialias["_experiment"]["variant"] = "hf_loss_decoder_antialias"
    antialias["_experiment"]["only_difference_vs_other"] = (
        "decoder antialias_activation is enabled"
    )

    loss_config_path = configs_dir / "model_hf_loss_only.json"
    antialias_config_path = configs_dir / "model_hf_loss_decoder_antialias.json"
    dataset_path = configs_dir / "dataset_frozen_1018957.json"
    _write_json(loss_config_path, loss_only)
    _write_json(antialias_config_path, antialias)
    _write_json(dataset_path, dataset)

    scripts = {
        "run_loss_only_gpu0-3.sh": _run_script(
            out_root=out_root,
            config=loss_config_path,
            dataset=dataset_path,
            gpu_ids="0,1,2,3",
            run_name="vae_ds1024_z64_hf_loss_only_4gpu",
            variant_dir="loss_only",
        ),
        "run_antialias_gpu4-7.sh": _run_script(
            out_root=out_root,
            config=antialias_config_path,
            dataset=dataset_path,
            gpu_ids="4,5,6,7",
            run_name="vae_ds1024_z64_hf_antialias_4gpu",
            variant_dir="antialias",
        ),
    }
    for name, contents in scripts.items():
        path = scripts_dir / name
        path.write_text(contents, encoding="utf-8")
        os.chmod(path, 0o755)

    launch_both = scripts_dir / "launch_both.sh"
    launch_both.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

tmux has-session -t vae_hf_loss_0_3 2>/dev/null && {{ echo 'session vae_hf_loss_0_3 exists'; exit 1; }}
tmux has-session -t vae_hf_antialias_4_7 2>/dev/null && {{ echo 'session vae_hf_antialias_4_7 exists'; exit 1; }}

tmux new-session -d -s vae_hf_loss_0_3 {scripts_dir / 'run_loss_only_gpu0-3.sh'}
tmux new-session -d -s vae_hf_antialias_4_7 {scripts_dir / 'run_antialias_gpu4-7.sh'}
tmux list-sessions | grep -E 'vae_hf_(loss_0_3|antialias_4_7)'
""",
        encoding="utf-8",
    )
    os.chmod(launch_both, 0o755)

    status = scripts_dir / "status.sh"
    status.write_text(
        f"""#!/usr/bin/env bash
set -u
tmux list-sessions 2>/dev/null | grep -E 'vae_hf_(loss_0_3|antialias_4_7)' || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
for log in {out_root}/logs/loss_only.log {out_root}/logs/antialias.log; do
  echo "$log"
  test -f "$log" && tail -c 5000 "$log" | tr '\\r' '\\n' | tail -3 || true
done
""",
        encoding="utf-8",
    )
    os.chmod(status, 0o755)

    files = [loss_config_path, antialias_config_path, dataset_path, *scripts_dir.iterdir()]
    manifest = {
        "experiment": "vae_ds1024_z64_hf_ab_4gpu",
        "base_model_config": str(args.base_model.resolve()),
        "source_dataset_config": str(args.dataset_config.resolve()),
        "gpu_assignment": {
            "0,1,2,3": "hf_loss_only",
            "4,5,6,7": "hf_loss_decoder_antialias",
        },
        "files": {
            str(path.relative_to(out_root)): _sha256(path)
            for path in sorted(files)
            if path.is_file()
        },
    }
    _write_json(out_root / "EXPERIMENT.json", manifest)
    print(f"Built VAE HF A/B experiment -> {out_root}")


if __name__ == "__main__":
    main()
