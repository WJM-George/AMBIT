"""
=============================================================================
SANITY CHECK ONLY -- this is NOT the real Stage-1 trainer.
For real 4ch VAE training use  train_4ch.py  (repo root), which goes through the
stock AutoencoderTrainingWrapper (GAN discriminator + feature matching + EMA + demo).
=============================================================================

Overfit a tiny fixed batch with the 4-channel VAE to validate, in isolation from the
GAN training complexity:
  (a) the 2ch->4ch weight surgery is numerically sane,
  (b) the format-aware [4, T] layout + joint (spatial-preserving) normalization works,
  (c) gradients flow end-to-end (encode -> bottleneck -> decode),
  (d) reconstruction error trends to ~0 and the spatial image is preserved
      (low inter-channel correlation drift).

What this DOES NOT validate: final perceptual audio quality. There is no adversarial
discriminator here, and the Oobleck decoder's "crispness" comes largely from that GAN
loss. A clean overfit here is necessary-but-not-sufficient; real quality is judged in
train_4ch.py.

Run (from repo root, inside the env):

  python -m stable_audio_tools.training.overfit_sanity_4ch \
      --config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae.json \
      --pretrained-ckpt /path/to/stable_audio_open_model.safetensors \
      --sls-root /mnt/sdb/audio_dataset/datasets/spatial_librispeech \
      --mrsdrama-root /mnt/sdd/audio_dataset/datasets/mrsdrama/snapshot \
      --steps 1000 --batch-size 4 --out-dir /mnt/sdc/vae_4ch_sanity_out

  Checkpoints are written under out-dir/step_XXXXXX/ (e.g. step_000250/, step_001000/).
  Each folder holds {clip}_in.wav and {clip}_rec.wav for that step.
"""

import os
import argparse

import torch
import torchaudio

from ..models.autoencoders_4ch import build_4ch_vae_with_warm_start
from ..data.dataset_4ch import (
    FourChannelSampleDataset, collation_fn,
    list_spatial_librispeech, list_mrsdrama,
)


def _select_valid_channels(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[B, C, T], mask [B, C] -> [N, 1, T] stacking only the valid channels."""
    B, C, T = x.shape
    sel = [x[b, c] for b in range(B) for c in range(C) if mask[b, c] > 0]
    if not sel:
        return x.reshape(B * C, 1, T)
    return torch.stack(sel, dim=0).unsqueeze(1)


def _interchannel_corr(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean pairwise correlation over valid channels for one [C, T] example."""
    valid = [c for c in range(x.shape[0]) if mask[c] > 0]
    if len(valid) < 2:
        return torch.tensor(0.0, device=x.device)
    corrs = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            a = x[valid[i]] - x[valid[i]].mean()
            b = x[valid[j]] - x[valid[j]].mean()
            denom = (a.norm() * b.norm()).clamp_min(1e-8)
            corrs.append((a * b).sum() / denom)
    return torch.stack(corrs).mean()


def run_overfit_sanity(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    model = build_4ch_vae_with_warm_start(
        args.config, pretrained_ckpt_path=args.pretrained_ckpt,
        replicate_output=not args.zero_init_output, verbose=True,
    ).to(device).train()
    sample_rate = model.sample_rate

    items = []
    if args.sls_root:
        items += list_spatial_librispeech(args.sls_root, limit=args.sls_count)
    if args.mrsdrama_root:
        items += list_mrsdrama(args.mrsdrama_root, limit=args.mrsdrama_count)
    assert items, "No data found. Pass --sls-root and/or --mrsdrama-root."
    items = items[:args.batch_size]
    print(f"[sanity] overfitting on {len(items)} fixed files:")
    for p, fmt in items:
        print(f"    [{fmt}] {p}")

    ds = FourChannelSampleDataset(
        items, sample_size=args.sample_size, sample_rate=sample_rate,
        normalize=args.normalize, random_crop=False,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), collate_fn=collation_fn)
    reals_cpu, infos = next(iter(loader))  # framework-native batch: (reals[B,4,T], info_list)
    reals = reals_cpu.to(device)
    mask = torch.stack([info["channel_mask"] for info in infos]).to(device)  # [B, 4]
    paths = [info["path"] for info in infos]

    import auraloss
    mrstft = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[2048, 1024, 512, 256, 128],
        hop_sizes=[512, 256, 128, 64, 32],
        win_lengths=[2048, 1024, 512, 256, 128],
        sample_rate=sample_rate, perceptual_weighting=True,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.8, 0.99))
    mask_bt = mask.unsqueeze(-1)  # [B, 4, 1]

    for step in range(1, args.steps + 1):
        opt.zero_grad()
        latents, info = model.encode(reals, return_info=True)
        decoded = model.decode(latents)

        reals_sel = _select_valid_channels(reals, mask)
        dec_sel = _select_valid_channels(decoded, mask)

        stft_loss = mrstft(dec_sel, reals_sel)
        l1 = ((decoded - reals).abs() * mask_bt).sum() / mask_bt.sum().clamp_min(1.0) / decoded.shape[-1]
        kl = info.get("kl", torch.tensor(0.0, device=device))
        loss = stft_loss + args.l1_weight * l1 + args.kl_weight * kl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                corr_drift = torch.stack([
                    (_interchannel_corr(reals[b], mask[b]) - _interchannel_corr(decoded[b], mask[b])).abs()
                    for b in range(reals.shape[0])
                ]).mean()
            print(f"step {step:5d} | loss {loss.item():.4f} | mrstft {stft_loss.item():.4f} "
                  f"| l1 {l1.item():.5f} | kl {float(kl):.4f} | corr_drift {corr_drift.item():.4f}")

        if step % args.save_every == 0 or step == args.steps:
            save_dir = _dump_audio(reals, decoded, paths, sample_rate, args.out_dir, step)
            print(f"[sanity] saved audio -> {save_dir}")

    print(f"[sanity] done. Reconstructions under {args.out_dir}/step_XXXXXX/")


def _step_subdir(out_dir: str, step: int) -> str:
    """e.g. /mnt/sdc/vae_4ch_sanity_out/step_000250"""
    return os.path.join(out_dir, f"step_{step:06d}")


def _dump_audio(reals, decoded, paths, sr, out_dir, step) -> str:
    save_dir = _step_subdir(out_dir, step)
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for b in range(reals.shape[0]):
            name = os.path.splitext(os.path.basename(paths[b]))[0]
            torchaudio.save(os.path.join(save_dir, f"{name}_in.wav"),
                            reals[b].detach().cpu().clamp(-1, 1), sr)
            torchaudio.save(os.path.join(save_dir, f"{name}_rec.wav"),
                            decoded[b].detach().cpu().clamp(-1, 1), sr)
    return save_dir


def build_argparser():
    p = argparse.ArgumentParser(description="4ch VAE overfit SANITY check (not the real trainer)")
    p.add_argument("--config", required=True, help="Path to the 4ch VAE model config json")
    p.add_argument("--pretrained-ckpt", default=None,
                   help="Pretrained 2ch VAE checkpoint (.safetensors/.ckpt) for warm start")
    p.add_argument("--zero-init-output", action="store_true",
                   help="Zero-init extra decoder output channels instead of replicate-init")
    p.add_argument("--sls-root", default=None, help="spatial_librispeech root (FOA)")
    p.add_argument("--mrsdrama-root", default=None, help="mrsdrama snapshot root (binaural)")
    p.add_argument("--sls-count", type=int, default=2)
    p.add_argument("--mrsdrama-count", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--sample-size", type=int, default=65536)
    p.add_argument("--normalize", default="joint_peak", choices=["joint_peak", "w_rms", "none"])
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--l1-weight", type=float, default=1.0)
    p.add_argument("--kl-weight", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--out-dir", default="./vae_4ch_sanity_out")
    p.add_argument("--device", default="cuda")
    return p


if __name__ == "__main__":
    run_overfit_sanity(build_argparser().parse_args())
