#!/usr/bin/env python3
"""Build a fixed eval subset for 4ch VAE comparison.

Modes:
  heldout  — N held-out SLS FOA files (NOT in VAE training; leakage-free). Default.
  mixed    — SLS + AudioCaps + MRSDrama symlinks (legacy architecture comparison).

Run:
  uv run python scripts/vae/eval/build_vae_eval_subset.py
  uv run python scripts/vae/eval/build_vae_eval_subset.py --mode mixed --n-sls 400
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[build_subset] %(message)s")
log = logging.getLogger(__name__)

import sys
from pathlib import Path as _PathForRepo
_SCRIPTS_DIR = _PathForRepo(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _repo import repo_root
SAT = repo_root()
CFG_DIR = SAT / "stable_audio_tools/configs/dataset_configs"
SLS_DIR = Path("/mnt/sdb/audio_dataset/datasets/spatial_librispeech/ambisonics")
AUDIOCAPS_DIR = Path("/mnt/sdc/audio_dataset_tmp/audiocaps_foa/train")
MRSDRAMA_DIR = Path("/mnt/sdd/audio_dataset/datasets/mrsdrama/snapshot")
TRAIN_ID = "spatial_librispeech"
TRAIN_MAX_FILES = 60255


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(os.path.realpath(src), dst)


def clear_dir(d: Path) -> None:
    if d.is_dir():
        for f in d.iterdir():
            if f.is_symlink() or f.is_file():
                f.unlink()
    d.mkdir(parents=True, exist_ok=True)


def sample_sorted(files: list[Path], n: int, seed: int) -> list[Path]:
    files = sorted(files)
    if n >= len(files):
        log.warning("requested %d but only %d available; using all", n, len(files))
        return files
    return sorted(random.Random(seed).sample(files, n))


def held_out_sls_files() -> list[str]:
    from stable_audio_tools.data.dataset import get_audio_filenames

    files = get_audio_filenames(str(SLS_DIR))
    train = set(sorted(random.Random(TRAIN_ID).sample(files, TRAIN_MAX_FILES)))
    return sorted(f for f in files if f not in train)


def build_heldout(out: Path, num: int, seed: int) -> Path:
    held = held_out_sls_files()
    log.info("held-out pool: %d files", len(held))
    rng = random.Random(seed)
    picks = held[:]
    rng.shuffle(picks)
    picks = picks[:num]

    out_dir = out / "sls_heldout"
    clear_dir(out_dir)
    for src in picks:
        link(Path(src), out_dir / Path(src).name)

    config_path = CFG_DIR / "local_4ch_vae_heldout_eval_100.json"
    config = {
        "_comment": f"{len(picks)} held-out SLS FOA files (NOT in VAE training). seed={seed}.",
        "dataset_type": "audio_dir_multichannel",
        "random_crop": False,
        "normalize": "joint_peak",
        "peak": 0.9,
        "drop_last": False,
        "datasets": [{"id": "sls_heldout", "path": str(out_dir), "format": "foa"}],
    }
    config_path.write_text(json.dumps(config, indent=4))
    log.info("heldout: linked %d -> %s", len(picks), out_dir)
    log.info("config -> %s", config_path)
    return config_path


def build_mixed(out: Path, n_sls: int, n_ac: int, n_mr: int, seed: int) -> Path:
    sls_out, ac_out, mr_out = out / "sls_foa", out / "audiocaps_foa", out / "mrsdrama_bin"
    for d in (sls_out, ac_out, mr_out):
        clear_dir(d)

    sls_pick = sample_sorted(list(SLS_DIR.glob("*.flac")), n_sls, seed)
    for f in sls_pick:
        link(f, sls_out / f.name)
    log.info("SLS: %d/%d", len(sls_pick), len(list(SLS_DIR.glob("*.flac"))))

    ac_pick = sample_sorted(list(AUDIOCAPS_DIR.glob("*.flac")), n_ac, seed)
    for f in ac_pick:
        link(f, ac_out / f.name)
    log.info("AudioCaps: %d", len(ac_pick))

    mr_pick = sample_sorted(list(MRSDRAMA_DIR.glob("*/wav/*.wav")), n_mr, seed)
    for i, f in enumerate(mr_pick):
        link(f, mr_out / f"bin_{i:04d}_{_safe(f.parent.parent.name)}__{f.name}")
    log.info("MRSDrama: %d", len(mr_pick))

    total = len(sls_pick) + len(ac_pick) + len(mr_pick)
    config_path = CFG_DIR / "local_4ch_vae_eval_subset.json"
    config = {
        "_comment": f"Fixed {total}-file mixed eval subset (seed={seed}).",
        "dataset_type": "audio_dir_multichannel",
        "random_crop": False,
        "normalize": "joint_peak",
        "peak": 0.9,
        "drop_last": False,
        "datasets": [
            {"id": "sls_foa", "path": str(sls_out), "format": "foa"},
            {"id": "audiocaps_foa", "path": str(ac_out), "format": "foa"},
            {"id": "mrsdrama_bin", "path": str(mr_out), "format": "binaural"},
        ],
    }
    config_path.write_text(json.dumps(config, indent=4))
    log.info("mixed: total %d -> %s", total, config_path)
    return config_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("heldout", "mixed"), default="heldout")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--num", type=int, default=100, help="heldout: number of SLS files")
    ap.add_argument("--n-sls", type=int, default=400)
    ap.add_argument("--n-audiocaps", type=int, default=400)
    ap.add_argument("--n-mrsdrama", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.mode == "heldout":
        out = args.out or Path("/mnt/sdc/audio_latents/vae_eval/heldout_100")
        build_heldout(out, args.num, args.seed)
    else:
        out = args.out or Path("/mnt/sdc/audio_latents/vae_eval/subset")
        build_mixed(out, args.n_sls, args.n_audiocaps, args.n_mrsdrama, args.seed)


if __name__ == "__main__":
    main()
