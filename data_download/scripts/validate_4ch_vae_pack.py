#!/usr/bin/env python3
"""
Build a rigorous 4ch VAE source/reconstruction validation pack.

The metrics are computed in the VAE input space:
  source file -> canonical 4ch layout -> resample to model_sr -> joint_peak normalize
  latent      -> VAE decode at model_sr

This avoids judging quality after an extra downsample. For listening convenience the
script also writes source/recon pairs in each source file's native sample rate and a
stereo W-channel folder that plays in Reaper/headphones.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from shutil import copy2

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

_ROOT = Path(__file__).resolve().parents[1]
_SAT_ROOT = _ROOT / "stable-audio-tools"
if _SAT_ROOT.is_dir() and str(_SAT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAT_ROOT))

from stable_audio_tools.data.dataset_4ch import FOA, BINAURAL, layout_to_4ch, joint_normalize
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict


def save_flac(audio: torch.Tensor, path: Path, sr: int, *, subtype: str = "PCM_24") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio.detach().cpu().clamp(-1.0, 1.0)
    sf.write(str(path), audio.numpy().T, sr, format="FLAC", subtype=subtype)


def stereo_w(audio: torch.Tensor) -> torch.Tensor:
    return audio[0:1].repeat(2, 1)


def foa_preview_lr(audio: torch.Tensor) -> torch.Tensor:
    # Not binaural/HRTF; just a stable audible FOA horizontal preview.
    left = audio[0] + 0.6 * audio[1]
    right = audio[0] - 0.6 * audio[1]
    return torch.stack([left, right], dim=0).clamp(-1.0, 1.0)


def load_model(details_path: Path):
    details = json.loads(details_path.read_text())
    model_config_path = Path(details["model_config"])
    if not model_config_path.is_absolute():
        model_config_path = _SAT_ROOT / model_config_path
    with open(model_config_path) as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)
    copy_state_dict(model, load_ckpt_state_dict(details["args"]["ckpt_path"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().requires_grad_(False).to(device)
    return model, model_config, details


def source_to_4ch(path: str, fmt: str) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    out, _ = layout_to_4ch(wav, fmt)
    return out, sr


def preprocess_source_for_vae(source_4ch: torch.Tensor, source_sr: int, model_sr: int) -> torch.Tensor:
    audio = source_4ch
    if source_sr != model_sr:
        audio = torchaudio.functional.resample(audio, source_sr, model_sr)
    return joint_normalize(audio, mode="joint_peak", peak=0.9).clamp(-1.0, 1.0)


def decode_latent(model, npy_path: Path) -> torch.Tensor:
    device = next(model.parameters()).device
    latent = torch.from_numpy(np.load(npy_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.decode(latent).squeeze(0).cpu().clamp(-1.0, 1.0)


def align_pair(source: torch.Tensor, recon: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(source.shape[-1], recon.shape[-1])
    return source[:, :n], recon[:, :n]


def safe_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if denom.item() < 1e-12:
        return float("nan")
    return float((x * y).sum().item() / denom.item())


def si_sdr_db(ref: torch.Tensor, est: torch.Tensor) -> float:
    ref_zm = ref - ref.mean()
    est_zm = est - est.mean()
    ref_energy = (ref_zm**2).sum()
    if ref_energy.item() < 1e-12:
        return float("nan")
    scale = (est_zm * ref_zm).sum() / ref_energy
    target = scale * ref_zm
    noise = est_zm - target
    ratio = (target**2).sum() / ((noise**2).sum() + 1e-12)
    return float(10.0 * torch.log10(ratio + 1e-12).item())


def channel_metrics(source: torch.Tensor, recon: torch.Tensor) -> dict[str, float]:
    names = ["W", "Y", "Z", "X"]
    out: dict[str, float] = {}
    for i, name in enumerate(names):
        s = source[i]
        r = recon[i]
        mse = torch.mean((s - r) ** 2)
        mae = torch.mean(torch.abs(s - r))
        source_rms = torch.sqrt(torch.mean(s**2) + 1e-12)
        recon_rms = torch.sqrt(torch.mean(r**2) + 1e-12)
        out[f"{name}_mse"] = float(mse.item())
        out[f"{name}_mae"] = float(mae.item())
        out[f"{name}_corr"] = safe_corr(s, r)
        out[f"{name}_si_sdr_db"] = si_sdr_db(s, r)
        out[f"{name}_energy_ratio_db"] = float(20.0 * torch.log10((recon_rms + 1e-12) / (source_rms + 1e-12)).item())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-root", type=Path, default=Path("/mnt/sdc/audio_latents/stage1_vae_4ch_trial"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/sdc/vae_4ch_validation_pack_step50000"))
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--glob-ranks", default="*")
    args = parser.parse_args()

    details_path = args.latent_root / "details.json"
    model, model_config, details = load_model(details_path)
    model_sr = int(model_config["sample_rate"])

    all_latents = sorted(args.latent_root.glob(f"{args.glob_ranks}/*.npy"))
    if not all_latents:
        raise FileNotFoundError(f"No latents under {args.latent_root}/{args.glob_ranks}/")

    random.seed(args.seed)
    picked = random.sample(all_latents, min(args.num_samples, len(all_latents)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sub in [
        "source_original_4ch",
        "source_vae_input_4ch",
        "recon_vae_output_4ch",
        "native_ab_4ch",
        "listen_w_stereo",
        "preview_lr_stereo",
    ]:
        (args.output_dir / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    manifest = {
        "latent_root": str(args.latent_root),
        "model_config": details["model_config"],
        "ckpt_path": details["args"]["ckpt_path"],
        "model_sample_rate": model_sr,
        "num_samples": len(picked),
        "seed": args.seed,
        "notes": [
            "Metrics are computed at model_sample_rate after VAE training preprocessing.",
            "source_original_4ch contains the original source file copied without processing.",
            "native_ab_4ch contains source/recon in the source file sample rate for quick DAW A/B.",
            "preview_lr_stereo is not true binaural; it is W +/- 0.6Y for audible spot checks.",
        ],
        "samples": [],
    }

    for idx, latent_path in enumerate(picked):
        md = json.loads(latent_path.with_suffix(".json").read_text())
        fmt = md.get("spatial_format", FOA)
        if fmt not in (FOA, BINAURAL):
            raise ValueError(f"Unexpected spatial_format={fmt!r} for {latent_path}")

        source_path = md["path"]
        source_stem = Path(source_path).stem
        tag = "WYZX" if fmt == FOA else "LR00"
        base = f"{idx:02d}_{latent_path.parent.name}_{latent_path.stem}_{source_stem}_{tag}"

        source_4ch, source_sr = source_to_4ch(source_path, fmt)
        source_vae = preprocess_source_for_vae(source_4ch, source_sr, model_sr)
        recon_vae = decode_latent(model, latent_path)
        source_cmp, recon_cmp = align_pair(source_vae, recon_vae)

        # Files for exact provenance and A/B.
        copy2(source_path, args.output_dir / "source_original_4ch" / f"{base}_SOURCE_ORIGINAL{Path(source_path).suffix}")
        save_flac(source_vae, args.output_dir / "source_vae_input_4ch" / f"{base}_source_modelsr_4ch.flac", model_sr)
        save_flac(recon_vae, args.output_dir / "recon_vae_output_4ch" / f"{base}_recon_modelsr_4ch.flac", model_sr)

        source_native = source_vae
        recon_native = recon_vae
        if source_sr != model_sr:
            source_native = torchaudio.functional.resample(source_native, model_sr, source_sr)
            recon_native = torchaudio.functional.resample(recon_native, model_sr, source_sr)
        save_flac(source_native, args.output_dir / "native_ab_4ch" / f"{base}_A_source_native_sr_4ch.flac", source_sr)
        save_flac(recon_native, args.output_dir / "native_ab_4ch" / f"{base}_B_recon_native_sr_4ch.flac", source_sr)

        save_flac(stereo_w(source_vae), args.output_dir / "listen_w_stereo" / f"{base}_A_source_W_stereo.flac", model_sr)
        save_flac(stereo_w(recon_vae), args.output_dir / "listen_w_stereo" / f"{base}_B_recon_W_stereo.flac", model_sr)
        if fmt == FOA:
            save_flac(foa_preview_lr(source_vae), args.output_dir / "preview_lr_stereo" / f"{base}_A_source_FOA_preview_lr.flac", model_sr)
            save_flac(foa_preview_lr(recon_vae), args.output_dir / "preview_lr_stereo" / f"{base}_B_recon_FOA_preview_lr.flac", model_sr)

        row = {
            "idx": idx,
            "latent": str(latent_path.relative_to(args.latent_root)),
            "source": source_path,
            "spatial_format": fmt,
            "source_sr": source_sr,
            "model_sr": model_sr,
            "compare_samples": source_cmp.shape[-1],
            "latent_T": int(np.load(latent_path, mmap_mode="r").shape[1]),
        }
        row.update(channel_metrics(source_cmp, recon_cmp))
        rows.append(row)
        manifest["samples"].append(row)
        print(f"[{idx + 1}/{len(picked)}] {base}  W_corr={row['W_corr']:.3f} W_SI-SDR={row['W_si_sdr_db']:.2f}dB")

    csv_path = args.output_dir / "metrics_per_sample.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    metric_keys = [k for k in rows[0] if any(k.endswith(s) for s in ("_mse", "_mae", "_corr", "_si_sdr_db", "_energy_ratio_db"))]
    for key in metric_keys:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        summary[key] = {
            "mean": float(vals.mean()) if vals.size else None,
            "median": float(np.median(vals)) if vals.size else None,
            "min": float(vals.min()) if vals.size else None,
            "max": float(vals.max()) if vals.size else None,
        }

    (args.output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output_dir / "README.txt").write_text(
        f"""4ch VAE validation pack

Model checkpoint:
  {details['args']['ckpt_path']}

How to inspect:
  1. listen_w_stereo/*_A_source_W_stereo.flac vs *_B_recon_W_stereo.flac
     - Normal stereo headphone/Reaper check for speech clarity.
  2. preview_lr_stereo/*_FOA_preview_lr.flac
     - Not true binaural; W +/- 0.6Y quick FOA horizontal preview.
  3. native_ab_4ch/*_A_source_native_sr_4ch.flac vs *_B_recon_native_sr_4ch.flac
     - Same source sample rate, 4ch FLAC, for Ambisonics/DAW A/B.
  4. source_vae_input_4ch and recon_vae_output_4ch
     - Fair metric space at model_sr={model_sr}; metrics are computed here.

Metrics:
  metrics_per_sample.csv
  summary_metrics.json

Notes:
  - source_original_4ch files are copied untouched for provenance.
  - True binaural/HRTF decode is not included; preview_lr_stereo is only a monitor mix.
""",
        encoding="utf-8",
    )

    print(f"\nWrote validation pack: {args.output_dir}")
    print(f"Metrics: {csv_path}")


if __name__ == "__main__":
    main()
