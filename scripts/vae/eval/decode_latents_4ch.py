"""
Decode pre-encoded 4ch VAE latents to canonical 4-channel FLAC (same container as SLS/MRSDrama).

Channel layout (matches dataset_4ch / pre_encode_4ch):
  * FOA:      [W, Y, Z, X]  -> *_WYZX_4ch.flac
  * binaural: [L, R, 0, 0]  -> *_LR00_4ch.flac

Writes quad FLAC with soundfile (24-bit, ffmpeg layout "quad") to match SLS on disk.
Also writes reaper_listen/*_stereo_W.flac — W (or L/R) on stereo for headphones + WaveOut.

Run:
  stable-audio-tools/.venv/bin/python scripts/vae/eval/decode_latents_4ch.py \\
    --latent-root /mnt/sdc/audio_latents/stage1_vae_4ch_trial \\
    --output-dir /mnt/sdc/vae_4ch_train_result_audio
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_SAT_ROOT = Path(__file__).resolve().parents[3]
if _SAT_ROOT.is_dir() and str(_SAT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAT_ROOT))

import numpy as np
import soundfile as sf
import torch
import torchaudio

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, copy_state_dict

SUFFIX = {"foa": "WYZX", "binaural": "LR00"}


def load_model(details_path: Path):
    details = json.loads(details_path.read_text())
    with open(details["model_config"]) as f:
        model_config = json.load(f)
    model = create_model_from_config(model_config)
    copy_state_dict(model, load_ckpt_state_dict(details["args"]["ckpt_path"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().requires_grad_(False).to(device)
    return model, model_config


def format_suffix(spatial_format: str) -> str:
    key = (spatial_format or "foa").lower()
    if key not in SUFFIX:
        raise ValueError(f"Unknown spatial_format {spatial_format!r}; expected foa or binaural")
    return SUFFIX[key]


def source_sample_rate(path: str) -> int:
    return int(torchaudio.info(path).sample_rate)


def save_quad_flac(audio: torch.Tensor, path: Path, sr: int) -> None:
    """Match SLS ambisonics: 16 kHz (caller sets sr), 4ch, 24-bit, quad."""
    sf.write(str(path), audio.numpy().T, sr, format="FLAC", subtype="PCM_24")


def make_listen_stereo(audio: torch.Tensor, spatial_format: str) -> torch.Tensor:
    """Stereo preview tensor for normal headphones."""
    fmt = (spatial_format or "foa").lower()
    if fmt == "binaural":
        return audio[:2]
    # FOA: monitor omnidirectional W on both ears (intelligible speech check).
    return audio[0:1].repeat(2, 1)


def save_listen_stereo(audio: torch.Tensor, path: Path, sr: int, spatial_format: str) -> None:
    """Stereo FLAC at native/source sr for Reaper A/B."""
    stereo = make_listen_stereo(audio, spatial_format)
    sf.write(str(path), stereo.numpy().T, sr, format="FLAC", subtype="PCM_24")


def save_listen_wav_48k(audio: torch.Tensor, path: Path, sr: int, spatial_format: str) -> None:
    """Compatibility preview: 48 kHz 16-bit WAV, RMS-normalized for easy listening."""
    stereo = make_listen_stereo(audio, spatial_format)
    if sr != 48000:
        stereo = torchaudio.functional.resample(stereo, sr, 48000)
    rms = stereo.square().mean().sqrt()
    if rms > 1e-8:
        target = 10 ** (-18.0 / 20.0)
        stereo = stereo * (target / rms)
    peak = stereo.abs().max()
    if peak > 0.98:
        stereo = stereo * (0.98 / peak)
    sf.write(str(path), stereo.numpy().T, 48000, format="WAV", subtype="PCM_16")


def decode_latent(
    model,
    npy_path: Path,
    out_quad: Path,
    out_listen: Path | None,
    model_sr: int,
    source_path: str | None,
    resample_to_source: bool,
    spatial_format: str,
):
    lat = torch.from_numpy(np.load(npy_path)).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        audio = model.decode(lat).squeeze(0).cpu().clamp(-1.0, 1.0)

    if audio.shape[0] != 4:
        raise ValueError(f"Expected 4 channels after decode, got {audio.shape[0]}")

    out_sr = model_sr
    if resample_to_source and source_path and Path(source_path).exists():
        src_sr = source_sample_rate(source_path)
        if src_sr != model_sr:
            audio = torchaudio.functional.resample(audio, model_sr, src_sr)
        out_sr = src_sr

    save_quad_flac(audio, out_quad, out_sr)
    if out_listen is not None:
        save_listen_stereo(audio, out_listen, out_sr, spatial_format)


def quad_to_listen_from_file(quad_path: Path, listen_path: Path, spatial_format: str) -> None:
    audio, sr = torchaudio.load(quad_path)
    save_listen_stereo(audio, listen_path, sr, spatial_format)


def main(*, default_listen_wav: bool = False):
    parser = argparse.ArgumentParser(description="Decode 4ch latents to quad + Reaper stereo listen FLAC")
    parser.add_argument("--latent-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--decode-all", action="store_true",
                        help="Decode every latent; overrides --num-samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--glob-ranks", default="*")
    parser.add_argument("--no-resample-to-source", action="store_true")
    parser.add_argument(
        "--from-existing-quad", action="store_true",
        help="Only build reaper_listen stereo from existing *_4ch.flac (no GPU decode)",
    )
    parser.add_argument("--no-listen-stereo", action="store_true",
                        help="Skip reaper_listen/ stereo exports")
    wav_options = parser.add_mutually_exclusive_group()
    wav_options.add_argument("--listen-wav", dest="no_listen_wav", action="store_false",
                             help="Also write normalized 48 kHz WAV previews")
    wav_options.add_argument("--no-listen-wav", action="store_true",
                             help="Skip normalized WAV previews")
    parser.set_defaults(no_listen_wav=not default_listen_wav)
    args = parser.parse_args()

    listen_dir = args.output_dir / "reaper_listen"
    listen_wav_dir = args.output_dir / "listen_wav_48k"
    if not args.no_listen_stereo:
        listen_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_listen_stereo and not args.no_listen_wav:
        listen_wav_dir.mkdir(parents=True, exist_ok=True)

    if args.from_existing_quad:
        manifest_path = args.output_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest:
            quad_name = entry.get("output_quad") or entry.get("output")
            quad = args.output_dir / quad_name
            if not quad.exists():
                continue
            listen = listen_dir / quad.name.replace("_4ch.flac", "_stereo_W.flac")
            quad_to_listen_from_file(quad, listen, entry.get("spatial_format", "foa"))
            if not args.no_listen_wav:
                audio, sr = torchaudio.load(quad)
                wav = listen_wav_dir / listen.name.replace(".flac", "_48k_norm.wav")
                save_listen_wav_48k(audio, wav, sr, entry.get("spatial_format", "foa"))
            print(f"listen <- {quad.name}")
        _write_reaper_readme(listen_dir)
        print(f"Wrote stereo listen FLAC -> {listen_dir}")
        return

    latent_root = args.latent_root
    details_path = latent_root / "details.json"
    if not details_path.exists():
        raise FileNotFoundError(f"Missing {details_path}")

    model, model_config = load_model(details_path)
    model_sr = int(model_config["sample_rate"])
    resample_to_source = not args.no_resample_to_source

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_npy = sorted(latent_root.glob(f"{args.glob_ranks}/*.npy"))
    if not all_npy:
        raise FileNotFoundError(f"No .npy under {latent_root}/{args.glob_ranks}/")

    if args.decode_all:
        picked = all_npy
    else:
        random.seed(args.seed)
        picked = random.sample(all_npy, min(args.num_samples, len(all_npy)))

    manifest = []
    for i, npy_path in enumerate(picked):
        md = json.loads(npy_path.with_suffix(".json").read_text())
        fmt = md.get("spatial_format", "foa")
        tag = format_suffix(fmt)
        src = md.get("path", "")
        src_stem = Path(src).stem
        rank = npy_path.parent.name
        base = f"{i:02d}_{rank}_{npy_path.stem}_{src_stem}_{tag}"
        out_quad = args.output_dir / f"{base}_4ch.flac"
        out_listen = None if args.no_listen_stereo else listen_dir / f"{base}_stereo_W.flac"

        decode_latent(
            model, npy_path, out_quad, out_listen, model_sr, src,
            resample_to_source, fmt,
        )
        if out_listen is not None and not args.no_listen_wav:
            audio, sr = torchaudio.load(out_quad)
            out_wav = listen_wav_dir / f"{base}_stereo_W_48k_norm.wav"
            save_listen_wav_48k(audio, out_wav, sr, fmt)
        info = torchaudio.info(str(out_quad))
        entry = {
            "idx": i,
            "output_quad": out_quad.name,
            "output_listen_stereo": out_listen.name if out_listen else None,
            "layout": "[W, Y, Z, X]" if tag == "WYZX" else "[L, R, 0, 0]",
            "spatial_format": fmt,
            "sample_rate": info.sample_rate,
            "channels": info.num_channels,
            "latent": str(npy_path.relative_to(latent_root)),
            "source": src,
            "latent_shape": list(np.load(npy_path).shape),
        }
        manifest.append(entry)
        print(f"[{i + 1}/{len(picked)}] {out_quad.name}  listen={out_listen.name if out_listen else '-'}")

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if not args.no_listen_stereo:
        _write_reaper_readme(listen_dir)
        # reference AB
        ref_src = picked[1].with_suffix(".json") if len(picked) > 1 else picked[0].with_suffix(".json")
        ref_path = json.loads(ref_src.read_text()).get("path")
        if ref_path and Path(ref_path).exists():
            w, sr = torchaudio.load(ref_path)
            ref_listen = listen_dir / f"_REF_{Path(ref_path).stem}_stereo_W.flac"
            save_listen_stereo(w[:4] if w.shape[0] >= 4 else w, ref_listen, sr, "foa")
            if not args.no_listen_wav:
                ref_wav = listen_wav_dir / f"_REF_{Path(ref_path).stem}_stereo_W_48k_norm.wav"
                save_listen_wav_48k(w[:4] if w.shape[0] >= 4 else w, ref_wav, sr, "foa")
            print(f"reference listen: {ref_listen.name}")
    print(f"Done -> {args.output_dir}")


def _write_reaper_readme(listen_dir: Path) -> None:
    (listen_dir / "REAPER_README.txt").write_text(
        """Reaper / WaveOut / stereo headphones — use files in THIS folder only

WHY quad *_4ch.flac may be silent in Reaper:
  - WaveOut + 4 output channels often does not reach stereo headphones (only out 1–2 work).
  - Raw FOA [W,Y,Z,X] is not stereo; DAW must decode or you must solo channel 1.

FILES HERE:
  *_stereo_W.flac  = FOA channel W (omnidirectional / speech) -> L & R (guaranteed audible)
  *_4ch.flac stay in parent folder for pipeline / ambisonic tools.

If your player/DAW still does not play 16 kHz / 24-bit FLAC reliably, use:
  ../listen_wav_48k/*_48k_norm.wav  = 48 kHz / 16-bit / normalized stereo previews.

REAPER SETTINGS (recommended):
  1. Preferences -> Audio -> Device: Output channels = 2 (stereo), Sample rate = 16000
  2. File -> Project settings -> Sample rate = 16000
  3. Import *_stereo_W.flac onto a STEREO track (default), press Play

To audition true 4ch quad: use ASIO 4-out, or IEM/Ambix decoder on parent *_4ch.flac.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
