"""Calibrate weights for the new phase_ifgd / foa_scm losses.

Runs the frozen 800k VAE on a few real FOA clips (CPU), measures the raw value
of each loss on (decoded, reals), and prints weights that make each new term
contribute a target fraction of the current dominant reconstruction term
(mrstft, weight 1.0).

Run:
  uv run python scripts/vae/loss/calibrate_new_loss_weights.py \
      --model-config /mnt/sdc/ckpts/vae_ds1024_z64_hf_overshoot_decay_350k_8gpu/configs/model_hf_overshoot_decay_350k.json \
      --ckpt /mnt/sdc/ckpts/vae_ds1024_z64_hf_overshoot_decay_350k_8gpu/checkpoints/vae_ds1024_z64_hf_overshoot_decay_350k_8gpu/cx9iuuoa/checkpoints/epoch=12-step=800000.ckpt \
      --audio-dir /mnt/sdd/audio_dataset/spatial_foa/audio --num-files 6
"""

import argparse
import glob
import json
import os

import soundfile as sf
import torch
import torchaudio

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.training.losses import auraloss as auraloss
from stable_audio_tools.training.losses.semantic import (
    FOASpatialConsistencyLoss,
    FOASpatialCovarianceLoss,
    FrequencyGatedIFGDPhaseLoss,
)


def load_autoencoder(model_config_path, ckpt_path):
    with open(model_config_path) as f:
        model_config = json.load(f)
    model = create_model_from_config(model_config)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    prefixes = ["autoencoder_ema.ema_model.", "autoencoder."]
    for prefix in prefixes:
        sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if sub:
            missing, unexpected = model.load_state_dict(sub, strict=False)
            print(f"loaded prefix '{prefix}': {len(sub)} tensors, "
                  f"{len(missing)} missing, {len(unexpected)} unexpected")
            break
    else:
        raise RuntimeError(f"No known prefix found; sample keys: {list(state)[:5]}")
    model.eval()
    return model, model_config


def load_batch(audio_dir, num_files, sample_rate, sample_size, seed=0):
    files = sorted(glob.glob(os.path.join(audio_dir, "*.flac")))
    if not files:
        raise RuntimeError(f"No flac files in {audio_dir}")
    g = torch.Generator().manual_seed(seed)
    picked, clips = [], []
    for path in files:
        if len(clips) >= num_files:
            break
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = torch.from_numpy(audio.T)  # [C, T]
        if audio.shape[0] != 4:
            continue
        if sr != sample_rate:
            audio = torchaudio.functional.resample(audio, sr, sample_rate)
        if audio.shape[-1] < sample_size:
            continue
        start = torch.randint(0, audio.shape[-1] - sample_size + 1, (1,), generator=g).item()
        clip = audio[:, start:start + sample_size]
        if clip.abs().max() < 1e-3:  # skip near-silent crops
            continue
        clips.append(clip)
        picked.append(os.path.basename(path))
    if len(clips) < num_files:
        raise RuntimeError(f"Only found {len(clips)} usable clips")
    print("clips:", ", ".join(picked))
    return torch.stack(clips)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--num-files", type=int, default=6)
    parser.add_argument("--target-fraction", type=float, default=0.08,
                        help="desired contribution of each new term relative to the mrstft term")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    model, model_config = load_autoencoder(args.model_config, args.ckpt)
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    stft_cfg = model_config["training"]["loss_configs"]["spectral"]["config"]
    mrstft_weight = model_config["training"]["loss_configs"]["spectral"]["weights"]["mrstft"]

    reals = load_batch(args.audio_dir, args.num_files, sample_rate, sample_size)
    print(f"batch: {tuple(reals.shape)} @ {sample_rate} Hz")

    latents, _ = model.encode(reals, return_info=True)
    decoded = model.decode(latents)
    common = min(decoded.shape[-1], reals.shape[-1])
    decoded, reals = decoded[..., :common], reals[..., :common]
    print(f"decoded: {tuple(decoded.shape)}, latents: {tuple(latents.shape)}")

    mrstft = auraloss.MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_cfg)
    phase_ifgd = FrequencyGatedIFGDPhaseLoss(sample_rate=sample_rate)
    foa_scm = FOASpatialCovarianceLoss(sample_rate=sample_rate)
    foa_spatial = FOASpatialConsistencyLoss(sample_rate=sample_rate)

    values = {
        # AuralossLoss wrapper calls mrstft(target, input); mirror that order.
        "mrstft": mrstft(reals, decoded).item(),
        "phase_ifgd": phase_ifgd(decoded, reals).item(),
        "foa_scm": foa_scm(decoded, reals).item(),
        "foa_spatial": foa_spatial(decoded, reals).item(),
    }
    sub = {
        "phase_ifgd": (phase_ifgd.last_if_loss.item(), phase_ifgd.last_gd_loss.item(),
                       phase_ifgd.last_cd_loss.item()),
        "foa_scm": (foa_scm.last_level_loss.item(), foa_scm.last_coherence_loss.item(),
                    foa_scm.last_ipd_loss.item()),
        "foa_spatial": (foa_spatial.last_direction_loss.item(), foa_spatial.last_ratio_loss.item()),
    }

    print("\nraw loss values on (decoded, reals):")
    for name, value in values.items():
        extra = f"   sub-terms: {tuple(round(v, 4) for v in sub[name])}" if name in sub else ""
        print(f"  {name:12s} = {value:.4f}{extra}")

    anchor = mrstft_weight * values["mrstft"]
    print(f"\nanchor (mrstft term, weight {mrstft_weight}) = {anchor:.4f}")
    print(f"recommended weights for ~{args.target_fraction:.0%} contribution each:")
    for name in ("phase_ifgd", "foa_scm", "foa_spatial"):
        rec = args.target_fraction * anchor / max(values[name], 1e-8)
        print(f"  {name:12s}: {rec:.3f}")


if __name__ == "__main__":
    main()
