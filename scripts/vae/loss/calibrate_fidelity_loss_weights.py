"""Calibrate weights for the new fidelity terms: cepstral (w_cep) + SI-SDR.

Runs the frozen EMA VAE (e.g. 900k phase_scm) on a few real FOA clips (CPU),
measures each new term's raw value on (decoded, reals), and prints weights that
make each contribute a target fraction of the mrstft reconstruction anchor
(weight 1.0). Companion to calibrate_new_loss_weights.py (phase/scm).

  * cepstral rides inside MultiResolutionSTFTLoss via w_cep (outer mrstft weight
    is 1.0), so its contribution is  w_cep * cep_mean  and we solve
    w_cep = frac * mrstft_base / cep_mean.
  * SI-SDR is a standalone term; SISDRLoss returns -SI-SDR (dB), so we size the
    weight against its magnitude.

Run:
  uv run python scripts/vae/loss/calibrate_fidelity_loss_weights.py \
      --model-config stable_audio_tools/configs/model_configs/autoencoders/ablation_arms/stable_audio_4ch_vae_ds1024_z64_phase_scm.json \
      --ckpt ${AMBIT_CKPT_ROOT}/vae_abl_phase_scm/checkpoints/vae_abl_phase_scm/hkfa5pts/checkpoints/epoch=13-step=900000.ckpt \
      --audio-dir ${AMBIT_DATA_ROOT}/datasets/spatial_librispeech/ambisonics --num-files 6
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


def load_autoencoder(model_config_path, ckpt_path):
    with open(model_config_path) as f:
        model_config = json.load(f)
    model = create_model_from_config(model_config)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    for prefix in ("autoencoder_ema.ema_model.", "autoencoder."):
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
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception:  # noqa: BLE001
            continue
        audio = torch.from_numpy(audio.T)  # [C, T]
        if audio.shape[0] < 4:
            continue
        audio = audio[:4]
        if sr != sample_rate:
            audio = torchaudio.functional.resample(audio, sr, sample_rate)
        if audio.shape[-1] < sample_size:
            continue
        start = torch.randint(0, audio.shape[-1] - sample_size + 1, (1,), generator=g).item()
        clip = audio[:, start:start + sample_size]
        if clip.abs().max() < 1e-3:
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
    parser.add_argument("--target-fraction", type=float, default=0.05,
                        help="desired contribution of each new term relative to the mrstft anchor")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    model, model_config = load_autoencoder(args.model_config, args.ckpt)
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    stft_cfg = dict(model_config["training"]["loss_configs"]["spectral"]["config"])
    mrstft_weight = model_config["training"]["loss_configs"]["spectral"]["weights"]["mrstft"]

    reals = load_batch(args.audio_dir, args.num_files, sample_rate, sample_size)
    print(f"batch: {tuple(reals.shape)} @ {sample_rate} Hz")

    latents, _ = model.encode(reals, return_info=True)
    decoded = model.decode(latents)
    common = min(decoded.shape[-1], reals.shape[-1])
    decoded, reals = decoded[..., :common], reals[..., :common]
    print(f"decoded: {tuple(decoded.shape)}, latents: {tuple(latents.shape)}")

    # Base mrstft anchor (as configured, w_cep = 0). AuralossLoss calls mrstft(target, input).
    mrstft = auraloss.MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_cfg)
    mrstft_val = mrstft(reals, decoded).item()

    # Isolated cepstral term: same STFT grid, only w_cep active.
    cep_only = auraloss.MultiResolutionSTFTLoss(
        sample_rate=sample_rate, **stft_cfg,
        w_sc=0.0, w_log_mag=0.0, w_lin_mag=0.0, w_phs=0.0, w_cep=1.0,
    )
    cep_val = cep_only(reals, decoded).item()

    # SI-SDR term (standalone). SISDRLoss returns -SI-SDR (dB).
    sisdr = auraloss.SISDRLoss()
    sisdr_val = sisdr(decoded, reals).item()

    anchor = mrstft_weight * mrstft_val
    print("\nraw loss values on (decoded, reals):")
    print(f"  mrstft (anchor, w={mrstft_weight}) = {mrstft_val:.4f}")
    print(f"  cepstral (w_cep=1, mean over res)  = {cep_val:.4f}")
    print(f"  sisdr  (-SI-SDR dB)                = {sisdr_val:.4f}  (|.|={abs(sisdr_val):.4f})")

    frac = args.target_fraction
    w_cep_rec = frac * anchor / max(cep_val, 1e-8)
    sisdr_rec = frac * anchor / max(abs(sisdr_val), 1e-8)
    print(f"\nanchor = {anchor:.4f};  recommended for ~{frac:.0%} contribution each:")
    print(f"  w_cep (inside mrstft config) : {w_cep_rec:.4f}")
    print(f"  sisdr weight                 : {sisdr_rec:.5f}")
    print("\nnote: ramp both 900k->915k from 0; sisdr is scheduled, w_cep is static in mrstft.")


if __name__ == "__main__":
    main()
