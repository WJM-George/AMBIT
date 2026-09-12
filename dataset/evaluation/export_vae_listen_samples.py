#!/usr/bin/env python3
"""Export listenable FLAC for multi-checkpoint 4ch VAE comparison (held-out SLS).

Methods compared (same held-out clips, encode->decode fresh):
  - baseline: 2x Stable-Audio-Open stereo VAE (wy_zx pairing)
  - native 4ch VAE at training steps (--steps)

Output layout (under --out):
  foa_baseline/, foa_660k/, ...       quad 4ch FOA [W,Y,Z,X] only
  stereo_baseline/, stereo_660k/, ...  W channel duplicated for headphones
  manifest.json

Run:
  cd ./stable-audio-tools
  CUDA_VISIBLE_DEVICES=1 uv run python dataset/evaluation/export_vae_listen_samples.py \\
    --steps 660000 700000 720000 740000 --with-baseline --num 10 --seed 1234 \\
    --out ${AMBIT_CKPT_ROOT}/eval_metric/result_compare
"""
from __future__ import annotations
import os

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch

_HERE = Path(__file__).resolve().parent
_SAT_ROOT = _HERE.parents[1]
for p in (str(_HERE), str(_SAT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from compare_vae_4ch_vs_2stereo import (  # noqa: E402
    PAIRINGS,
    baseline_2stereo,
    load_2ch_vae,
    load_4ch_vae,
    _enc_dec_pad,
)
from compare_vae_heldout import held_out_sls_files  # noqa: E402
from eval_vae_recon import _read_4ch, _resample  # noqa: E402

DEFAULT_CKPT_ROOT = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/vae_ds1024_z64_construct")
DEFAULT_VAE4_CONFIG = (
    "stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae_ds1024.json"
)
DEFAULT_VAE2_CONFIG = (
    "stable_audio_tools/configs/model_configs/autoencoders/"
    "stable_audio_open_1_0_oobleck_2ch.json"
)
DEFAULT_VAE2_CKPT = os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/stable-audio-open-1.0/model.safetensors"


def step_tag(step: int) -> str:
    return f"{step // 1000}k"


def method_dirs(out: Path, tag: str) -> tuple[Path, Path]:
    return out / f"foa_{tag}", out / f"stereo_{tag}"


def save_quad_flac(audio: torch.Tensor, path: Path, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.numpy().T, sr, format="FLAC", subtype="PCM_24")


def save_stereo_w(audio: torch.Tensor, path: Path, sr: int) -> None:
    """FOA W channel duplicated to L/R for headphones."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stereo = audio[0:1].repeat(2, 1)
    sf.write(str(path), stereo.numpy().T, sr, format="FLAC", subtype="PCM_24")


def clear_out_dir(out: Path) -> None:
    if out.exists():
        for child in out.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    out.mkdir(parents=True, exist_ok=True)


def find_raw_ckpt(ckpt_root: Path, step: int) -> Path:
    matches = sorted(ckpt_root.glob(f"epoch=*-step={step}.ckpt"))
    if not matches:
        raise FileNotFoundError(f"No raw checkpoint for step={step} under {ckpt_root}")
    return matches[0]


def ensure_unwrapped(step: int, ckpt_root: Path, model_cfg: str) -> Path:
    unwrapped = ckpt_root / f"unwrapped_ds1024_z64_step={step}.ckpt"
    if unwrapped.exists():
        return unwrapped
    raw = find_raw_ckpt(ckpt_root, step)
    print(f"[unwrap] step={step} <- {raw.name}")
    subprocess.run(
        [
            sys.executable,
            str(_SAT_ROOT / "unwrap_model.py"),
            "--model-config",
            model_cfg,
            "--ckpt-path",
            str(raw),
            "--name",
            str(ckpt_root / f"unwrapped_ds1024_z64_step={step}"),
        ],
        check=True,
        cwd=str(_SAT_ROOT),
    )
    if not unwrapped.exists():
        raise RuntimeError(f"Unwrap failed: {unwrapped} not created")
    return unwrapped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/eval_metric/result_compare"))
    ap.add_argument("--num", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, nargs="+", default=[660000, 700000, 720000, 740000])
    ap.add_argument("--with-baseline", action="store_true")
    ap.add_argument("--pairing", choices=list(PAIRINGS), default="wy_zx")
    ap.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    ap.add_argument("--vae4-config", default=DEFAULT_VAE4_CONFIG)
    ap.add_argument("--vae2-base-config", default=DEFAULT_VAE2_CONFIG)
    ap.add_argument("--vae2-ckpt", default=DEFAULT_VAE2_CKPT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--keep-existing", action="store_true", help="do not wipe --out first")
    args = ap.parse_args()

    if not args.with_baseline and not args.steps:
        ap.error("Provide --with-baseline and/or --steps")

    device = torch.device(args.device)
    pairing = PAIRINGS[args.pairing]
    out = args.out
    if not args.keep_existing:
        print(f"[clean] wiping {out}")
        clear_out_dir(out)
    else:
        out.mkdir(parents=True, exist_ok=True)

    held = held_out_sls_files()
    rng = random.Random(args.seed)
    picks = held[:]
    rng.shuffle(picks)

    clip_data: list[dict] = []
    for fpath in picks:
        if len(clip_data) >= args.num:
            break
        stem = Path(fpath).stem
        try:
            raw, s_sr = _read_4ch(fpath)
            if raw.shape[0] < 4:
                continue
            src4 = torch.from_numpy(_resample(raw[:4], s_sr, 44100)).float()
            clip_data.append({"stem": stem, "source": fpath, "audio": src4})
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {Path(fpath).name}: {e!r}")

    if not clip_data:
        raise RuntimeError("No clips exported.")

    sr = 44100
    methods: list[dict] = []
    if args.with_baseline:
        methods.append({"tag": "baseline", "kind": "baseline", "ckpt": args.vae2_ckpt})
    for step in args.steps:
        ckpt = ensure_unwrapped(step, args.ckpt_root, args.vae4_config)
        methods.append({"tag": step_tag(step), "kind": "4ch", "step": step, "ckpt": str(ckpt)})

    for m in methods:
        foa_dir, stereo_dir = method_dirs(out, m["tag"])
        foa_dir.mkdir(parents=True, exist_ok=True)
        stereo_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] held-out pool={len(held)}, exporting {len(clip_data)} clips (seed={args.seed})")
    print(f"[setup] methods: {[m['tag'] for m in methods]}")

    manifest_clips: list[dict] = []

    if args.with_baseline:
        print(f"[load] baseline 2x stereo: {args.vae2_ckpt}  pairing={args.pairing}")
        model2, _ = load_2ch_vae(args.vae2_base_config, args.vae2_ckpt, device)
        foa_dir, stereo_dir = method_dirs(out, "baseline")
        for i, clip in enumerate(clip_data):
            stem = clip["stem"]
            recon = baseline_2stereo(model2, clip["audio"], pairing, device)
            n = min(clip["audio"].shape[-1], recon.shape[-1])
            audio = recon[:, :n].clamp(-1, 1)
            foa_path = foa_dir / f"{stem}_baseline_4ch.flac"
            stereo_path = stereo_dir / f"{stem}_baseline_stereo.flac"
            save_quad_flac(audio, foa_path, sr)
            save_stereo_w(audio, stereo_path, sr)
            print(f"[baseline {i + 1}/{len(clip_data)}] {stem}")
        del model2
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for m in methods:
        if m["kind"] != "4ch":
            continue
        tag = m["tag"]
        print(f"[load] 4ch VAE step={m['step']} ({tag}): {m['ckpt']}")
        model4, model_sr = load_4ch_vae(args.vae4_config, m["ckpt"], device)
        if model_sr != sr:
            raise RuntimeError(f"Expected sample_rate={sr}, model has {model_sr}")
        foa_dir, stereo_dir = method_dirs(out, tag)
        for i, clip in enumerate(clip_data):
            stem = clip["stem"]
            recon = _enc_dec_pad(model4, clip["audio"], device).clamp(-1, 1)
            n = min(clip["audio"].shape[-1], recon.shape[-1])
            audio = recon[:, :n]
            foa_path = foa_dir / f"{stem}_{tag}_4ch.flac"
            stereo_path = stereo_dir / f"{stem}_{tag}_stereo.flac"
            save_quad_flac(audio, foa_path, sr)
            save_stereo_w(audio, stereo_path, sr)
            print(f"[{tag} {i + 1}/{len(clip_data)}] {stem}")
        del model4
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for i, clip in enumerate(clip_data):
        stem = clip["stem"]
        entry: dict = {
            "idx": i,
            "file": stem,
            "source": clip["source"],
            "sample_rate": sr,
            "methods": {},
        }
        for m in methods:
            tag = m["tag"]
            foa_rel = f"foa_{tag}/{stem}_{tag}_4ch.flac"
            stereo_rel = f"stereo_{tag}/{stem}_{tag}_stereo.flac"
            entry["methods"][tag] = {
                "foa_4ch": foa_rel,
                "stereo_w": stereo_rel,
                "ckpt": m["ckpt"],
            }
            if m["kind"] == "4ch":
                entry["methods"][tag]["step"] = m["step"]
        manifest_clips.append(entry)

    meta = {
        "n_clips": len(manifest_clips),
        "test_set": "held-out Spatial-LibriSpeech (NOT in VAE training)",
        "seed": args.seed,
        "pairing": args.pairing,
        "sample_rate": sr,
        "methods": [
            {
                "tag": m["tag"],
                "kind": m["kind"],
                **({"step": m["step"]} if m["kind"] == "4ch" else {}),
                "ckpt": m["ckpt"],
                "foa_dir": f"foa_{m['tag']}",
                "stereo_dir": f"stereo_{m['tag']}",
            }
            for m in methods
        ],
        "clips": manifest_clips,
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))

    method_list = ", ".join(m["tag"] for m in methods)
    (out / "LISTEN.txt").write_text(
        f"""VAE reconstruction listening comparison ({sr} Hz, held-out SLS)

Methods: {method_list}
Pairing (baseline only): {args.pairing}
Clips: {len(manifest_clips)} (seed={args.seed})

Folder layout
  foa_<method>/       4ch FOA quad [W,Y,Z,X] — for spatial playback
  stereo_<method>/    W channel -> L/R stereo — for headphone A/B

Suggested A/B (headphones)
  1. Pick a clip stem, e.g. {manifest_clips[0]['file']}
  2. Compare stereo_baseline/{manifest_clips[0]['file']}_baseline_stereo.flac
     vs stereo_660k/ ... stereo_740k/ (same stem, different method suffix)
  3. For spatial check, open matching files in foa_* folders in a 4ch player

File naming: <stem>_<method>_4ch.flac / <stem>_<method>_stereo.flac
See manifest.json for full mapping.
""",
        encoding="utf-8",
    )
    print(f"\nDone -> {out}  ({len(manifest_clips)} clips, {len(methods)} methods)")


if __name__ == "__main__":
    main()
