"""
Stage-2: pre-encode the 4-channel (FOA / binaural) audio dataset to VAE latents.

Writes latents in the EXACT on-disk format the stock PreEncodedDataset expects, so the
Stage-3 DiT training can consume them directly:

  <output>/
    details.json                      # model_config / dataset_config / args
    silence.npy                       # encoded silence latent [1, C, N] (for padding)
    <rank>/<latent_id>.npy            # latent [C, N]
    <rank>/<latent_id>.json           # metadata (padding_mask downsampled to N, fmt, ...)

It mirrors the framework's pre_encode.py (PyTorch Lightning, multi-GPU via trainer.validate)
but swaps in create_4ch_dataloader_from_config (no channel-collapse, no phase flip) and
records the spatial extras (fmt id + channel_mask) into each metadata json.

Run (single node, all visible GPUs):

  python pre_encode_4ch.py \
    --model-config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae.json \
    --ckpt-path /mnt/sdc/ckpts/vae_4ch/unwrapped_4ch_vae.ckpt \
    --dataset-config stable_audio_tools/configs/dataset_configs/local_4ch_example.json \
    --output-path /mnt/sdc/audio_latents/stage1_vae_4ch \
    --no-pad --batch-size 1 --num-workers 8

Notes:
  * --ckpt-path should be an UNWRAPPED 4ch VAE checkpoint (after Stage-1 training,
    via unwrap_model.py). For a dry run before Stage-1 finishes you may instead pass
    --pretrained-ckpt-2ch to warm-start from the stereo VAE (latents won't be final).
  * --no-pad (variable length) requires --batch-size 1; it stores each file's true
    latent length and relies on silence.npy for later padding. Recommended for caching.
  * --limit-batches N caps validation to N batches per GPU (trial / format check).
    Omit for the full dataset. Use a separate --output-path for trials.
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import json
import argparse
from pathlib import Path

from scripts.t2a.train.gpu_preflight import assert_gpu_driver_healthy

assert_gpu_driver_healthy()

import numpy as np
import torch
from torch.nn import functional as F
import pytorch_lightning as pl

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, copy_state_dict
from stable_audio_tools.models.autoencoders_4ch import (
    load_raw_state_dict, extract_autoencoder_state_dict, warm_start_2ch_to_4ch,
)
from stable_audio_tools.data.dataset_4ch import create_4ch_dataloader_from_config


def load_4ch_model(model_config, ckpt_path=None, pretrained_ckpt_2ch=None,
                   replicate_output=True, model_half=False):
    model = create_model_from_config(model_config)
    if ckpt_path is not None:
        print(f"Loading unwrapped 4ch VAE checkpoint from {ckpt_path}")
        copy_state_dict(model, load_ckpt_state_dict(ckpt_path))
    elif pretrained_ckpt_2ch is not None:
        print(f"[dry-run] warm-starting 4ch model from stereo ckpt {pretrained_ckpt_2ch}")
        ae_sd = extract_autoencoder_state_dict(load_raw_state_dict(pretrained_ckpt_2ch))
        warm_start_2ch_to_4ch(model, ae_sd, replicate_output=replicate_output, verbose=True)
    else:
        raise ValueError("Provide --ckpt-path (trained 4ch VAE) or --pretrained-ckpt-2ch (dry run)")

    model.eval().requires_grad_(False)
    if model_half:
        model.to(torch.float16)
    print("Done loading 4ch model")
    return model


class PreEncode4chWrapper(pl.LightningModule):
    def __init__(self, model, output_path, model_half=False,
                 model_config=None, dataset_config=None, sample_size=1320960,
                 no_pad=False, args_dict=None):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.output_path = Path(output_path)

    def prepare_data(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        details_path = self.output_path / "details.json"
        if not details_path.exists():
            details = {
                "model_config": self.hparams.model_config,
                "dataset_config": self.hparams.dataset_config,
                "sample_size": self.hparams.sample_size,
                "variable_length": self.hparams.args_dict.get("no_pad", False),
                "channels": "4ch [W,Y,Z,X] (FOA) / [L,R,0,0] (binaural)",
                "args": self.hparams.args_dict,
            }
            details_path.write_text(json.dumps(details))

    def setup(self, stage=None):
        (self.output_path / str(self.global_rank)).mkdir(parents=True, exist_ok=True)

    def _maybe_save_silence(self, device):
        if self.global_rank != 0:
            return
        silence_path = self.output_path / "silence.npy"
        if silence_path.exists():
            return
        print("Saving silence latent")
        silence_audio = torch.zeros(1, self.model.io_channels, self.hparams.sample_size, device=device)
        if self.hparams.model_half:
            silence_audio = silence_audio.to(torch.float16)
        with torch.no_grad():
            silence_latent = self.model.encode(silence_audio).cpu().numpy()
        with open(silence_path, "wb") as f:
            np.save(f, silence_latent)

    def validation_step(self, batch, batch_idx):
        audio, metadata = batch
        if audio.ndim == 4 and audio.shape[0] == 1:
            audio = audio[0]

        self._maybe_save_silence(audio.device)

        # Pad to a multiple of the model's minimum (downsampling) length.
        min_len = self.model.min_length
        audio_len = audio.shape[-1]
        if audio_len % min_len != 0:
            audio = F.pad(audio, (0, ((audio_len // min_len) + 1) * min_len - audio_len))

        if self.hparams.model_half:
            audio = audio.to(torch.float16)

        with torch.no_grad():
            latents = self.model.encode(audio).cpu().numpy()

        for i, latent in enumerate(latents):
            latent_id = f"{self.global_rank:03d}{batch_idx:06d}{i:04d}"
            md = metadata[i]

            padding_mask = F.interpolate(
                md["padding_mask"][0].unsqueeze(0).unsqueeze(1).float(),
                size=latent.shape[1], mode="nearest",
            ).squeeze(0).squeeze(0).int()

            if self.hparams.no_pad:
                padding_np = padding_mask.cpu().numpy()
                valid = np.where(padding_np == 1)[0]
                if len(valid) > 0:
                    valid_length = valid[-1] + 1
                    latent = latent[:, :valid_length]
                    padding_mask = padding_mask[:valid_length]

            with open(self.output_path / str(self.global_rank) / f"{latent_id}.npy", "wb") as f:
                np.save(f, latent)

            md["padding_mask"] = padding_mask.cpu().numpy().tolist()
            for k, v in md.items():
                if isinstance(v, torch.Tensor):
                    md[k] = v.cpu().numpy().tolist()

            with open(self.output_path / str(self.global_rank) / f"{latent_id}.json", "w") as f:
                json.dump(md, f)

    def configure_optimizers(self):
        return None


def main():
    parser = argparse.ArgumentParser(description="Encode 4ch audio dataset to VAE latents")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--ckpt-path", default=None, help="Unwrapped trained 4ch VAE checkpoint")
    parser.add_argument("--pretrained-ckpt-2ch", default=None,
                        help="Stereo VAE ckpt for warm-start dry run (latents not final)")
    parser.add_argument("--zero-init-output", action="store_true")
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--output-path", default="/mnt/sdc/audio_latents/stage1_vae_4ch")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=1320960,
                        help="Pad/crop length in samples when not using --no-pad")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--strategy", default="auto")
    parser.add_argument(
        "--limit-batches", type=int, default=None,
        help="Max validation batches per GPU rank (trial only). "
             "Omit to encode the full dataloader (~268k clips with local_4ch_preencode.json).",
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--model-half", action="store_true")
    parser.add_argument("--no-pad", action="store_true",
                        help="Variable-length mode (requires --batch-size 1)")
    args = parser.parse_args()

    if args.no_pad and args.batch_size > 1:
        parser.error("--no-pad requires --batch-size 1 (variable-length samples cannot be batched)")

    with open(args.model_config) as f:
        model_config = json.load(f)
    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    model = load_4ch_model(
        model_config, ckpt_path=args.ckpt_path,
        pretrained_ckpt_2ch=args.pretrained_ckpt_2ch,
        replicate_output=not args.zero_init_output, model_half=args.model_half,
    )

    data_loader = create_4ch_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=args.sample_size,
        shuffle=args.shuffle,
        pad=not args.no_pad,
    )

    pl_module = PreEncode4chWrapper(
        model=model, output_path=args.output_path, model_half=args.model_half,
        model_config=args.model_config, dataset_config=args.dataset_config,
        sample_size=args.sample_size, no_pad=args.no_pad, args_dict=vars(args),
    )

    trainer_kwargs = dict(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        precision="16-true" if args.model_half else "32",
        logger=False,
        enable_checkpointing=False,
    )
    if args.limit_batches is not None:
        trainer_kwargs["limit_val_batches"] = args.limit_batches
        print(
            f"[pre_encode_4ch] trial mode: limit_val_batches={args.limit_batches} "
            f"(per GPU rank; use a dedicated --output-path)"
        )

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.validate(pl_module, data_loader)


if __name__ == "__main__":
    main()
