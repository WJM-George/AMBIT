#!/usr/bin/env python
"""Stage-1 4-channel spatial-audio VAE trainer.

This is a thin front end around stable_audio_tools:
  * model            : create_model_from_config
  * warm start       : stereo-to-FOA channel surgery
  * training wrapper : create_training_wrapper_from_config
  * objectives       : adversarial reconstruction, feature matching, MR-STFT, KL, and EMA
  * checkpointing    : resumable PyTorch Lightning checkpoints
  * optional spatial : FOA intensity-vector consistency loss
  * data             : create_4ch_dataloader_from_config (format-aware [4,T], no phaseflip)
  * demo/ckpt/trainer: stock PyTorch Lightning pieces

It deliberately uses a small argparse CLI instead of prefigure/defaults.ini so it is
self-contained, while delegating ALL model/loss/training logic to the framework.

Example:

  python train_4ch.py \
    --model-config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae.json \
    --dataset-config stable_audio_tools/configs/dataset_configs/local_4ch_example.json \
    --pretrained-ckpt-2ch ${AMBIT_CKPT_ROOT}/stable-audio-open-1.0/model.safetensors \
    --name vae_4ch_stage1 --batch-size 8 --num-gpus 1 --precision bf16-mixed \
    --save-dir ${AMBIT_CKPT_ROOT}/vae_4ch --checkpoint-every 5000
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import json
import argparse
from datetime import timedelta
from pathlib import Path

from scripts.t2a.train.gpu_preflight import assert_gpu_driver_healthy

assert_gpu_driver_healthy()

import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.autoencoders_4ch import (
    load_raw_state_dict, extract_autoencoder_state_dict, warm_start_2ch_to_4ch,
)
from stable_audio_tools.data.dataset_4ch import create_4ch_dataloader_from_config
from stable_audio_tools.training import (
    create_training_wrapper_from_config, create_demo_callback_from_config,
)


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f"{type(err).__name__}: {err}")


class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config


def build_argparser():
    p = argparse.ArgumentParser(description="Stage-1 4ch VAE trainer")
    p.add_argument("--model-config", required=True)
    p.add_argument("--dataset-config", required=True)
    p.add_argument("--pretrained-ckpt-2ch", default=None,
                   help="Stereo VAE checkpoint to warm-start the 4ch model (channel surgery)")
    p.add_argument("--zero-init-output", action="store_true",
                   help="Zero-init extra decoder output channels instead of replicate-init")
    p.add_argument("--ckpt-path", default=None,
                   help="Resume a wrapped Lightning checkpoint. When set, the 2ch warm-start is skipped.")
    p.add_argument("--name", default="vae_4ch_stage1")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--num-nodes", type=int, default=1)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--accum-batches", type=int, default=1)
    p.add_argument("--strategy", default=None)
    p.add_argument("--save-dir", default="./ckpts_vae_4ch")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Fixed directory for checkpoints. Overrides the default "
                        "(wandb-run-nested) path so an auto-restarting launcher can always "
                        "resume the highest-step ckpt from ONE stable directory.")
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--ddp-timeout-min", type=int, default=30,
                   help="torch.distributed process-group timeout (minutes) for multi-GPU runs. "
                        "A shorter value makes a hung collective (NCCL stall) abort faster so a "
                        "resilient launcher can restart sooner instead of waiting the 30min default.")
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Stop after this many optimizer steps (-1 = unlimited).")
    p.add_argument("--logger", default="none", choices=["none", "wandb"])
    p.add_argument("--wandb-watch", action="store_true",
                   help="Enable wandb.watch (logs grads/params). Off by default: on rank0 it adds "
                        "overhead that can stall collectives on long multi-GPU runs.")
    p.add_argument("--demo", action="store_true", help="Enable audio demo callback (needs --logger wandb)")
    p.add_argument("--seed", type=int, default=42)
    # --- FOA spatial consistency loss (DirAC intensity-vector loss, arXiv:2510.22241 style) ---
    p.add_argument("--foa-spatial-loss", action="store_true",
                   help="Enable the FOA spatial consistency loss (intensity-vector direction + "
                        "directional-energy-ratio matching). Injects training.loss_configs.foa_spatial "
                        "into the model config; fine-grained knobs (n_ffts, thresholds, ratio_weight, "
                        "channel_order) can instead be set directly in the model config JSON.")
    p.add_argument("--foa-spatial-weight", type=float, default=0.5,
                   help="Final weight of the FOA spatial loss in the generator objective (default 0.5; "
                        "raw loss magnitude is ~0.7 on current recon quality, i.e. comparable to mrstft, "
                        "so keep this <= 1.0 unless the GAN balance is monitored)")
    p.add_argument("--foa-spatial-ramp-steps", type=int, default=10000,
                   help="Linearly ramp the FOA spatial loss weight 0 -> weight over this many steps "
                        "starting at --foa-spatial-start-step, to avoid shocking the GAN balance when "
                        "resuming. 0 = constant weight from the first step.")
    p.add_argument("--foa-spatial-start-step", default="auto",
                   help="Global step at which the ramp starts. Integer, or 'auto' to read global_step "
                        "from --ckpt-path (falls back to 0 for fresh runs).")
    return p


def _resolve_foa_spatial_start_step(args) -> int:
    raw = str(args.foa_spatial_start_step).strip().lower()
    if raw != "auto":
        return int(raw)
    if not args.ckpt_path:
        return 0
    print(f"[train_4ch] --foa-spatial-start-step auto: reading global_step from {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    step = int(ckpt.get("global_step", 0))
    del ckpt
    print(f"[train_4ch] resume checkpoint global_step = {step}")
    return step


def inject_foa_spatial_loss(model_config: dict, args) -> None:
    """Enable training.loss_configs.foa_spatial from CLI flags (idempotent).

    A block already present in the model config JSON is respected; the CLI only
    fills in weight/schedule when they are absent, so a hand-tuned JSON config
    wins over the generic switch.
    """
    training_cfg = model_config.get("training")
    if not isinstance(training_cfg, dict) or "loss_configs" not in training_cfg:
        raise SystemExit(
            "--foa-spatial-loss requires the model config to define training.loss_configs "
            "(the stock spectral/discriminator losses); refusing to fabricate a full loss config."
        )

    loss_cfgs = training_cfg["loss_configs"]
    block = loss_cfgs.setdefault("foa_spatial", {})
    block.setdefault("config", {})
    weights = block.setdefault("weights", {})
    weights.setdefault("spatial", args.foa_spatial_weight)

    if "schedule" not in block and args.foa_spatial_ramp_steps > 0:
        start_step = _resolve_foa_spatial_start_step(args)
        block["schedule"] = {
            "type": "linear",
            "start_step": start_step,
            "end_step": start_step + args.foa_spatial_ramp_steps,
            "start_weight": 0.0,
            "end_weight": weights["spatial"],
        }

    print(f"[train_4ch] FOA spatial loss ENABLED: {json.dumps(block, indent=2)}")


def main():
    args = build_argparser().parse_args()
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed, workers=True)

    resume_ckpt_path = None
    if args.ckpt_path:
        resume_ckpt = Path(args.ckpt_path).expanduser()
        if not resume_ckpt.is_file():
            raise FileNotFoundError(f"--ckpt-path does not exist or is not a file: {resume_ckpt}")
        resume_ckpt_path = str(resume_ckpt)
        print(f"[train_4ch] RESUMING Lightning checkpoint: {resume_ckpt_path}")
        if args.pretrained_ckpt_2ch:
            print("[train_4ch] --ckpt-path was provided; skipping --pretrained-ckpt-2ch warm-start")

    with open(args.model_config) as f:
        model_config = json.load(f)
    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    if args.foa_spatial_loss:
        inject_foa_spatial_loss(model_config, args)

    # --- model + 2ch->4ch warm start ---
    model = create_model_from_config(model_config)
    if resume_ckpt_path is None and args.pretrained_ckpt_2ch:
        print(f"[train_4ch] warm-starting from {args.pretrained_ckpt_2ch}")
        raw = load_raw_state_dict(args.pretrained_ckpt_2ch)
        ae_sd = extract_autoencoder_state_dict(raw)
        warm_start_2ch_to_4ch(model, ae_sd, replicate_output=not args.zero_init_output, verbose=True)
    elif resume_ckpt_path is None:
        print("[train_4ch] starting from randomly initialized 4ch model")

    # --- dataloader (4ch, format-aware, no phase flip) ---
    train_dl = create_4ch_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
    )

    # --- stock training wrapper (GAN + EMA + losses from config) ---
    training_wrapper = create_training_wrapper_from_config(model_config, model)

    # --- logger ---
    if args.logger == "wandb":
        logger = pl.loggers.WandbLogger(project=args.name)
        if args.wandb_watch:
            logger.watch(training_wrapper)
        checkpoint_dir = os.path.join(args.save_dir, args.name,
                                      str(getattr(logger.experiment, "id", "run")), "checkpoints")
    else:
        logger = None
        checkpoint_dir = args.save_dir

    # A fixed --checkpoint-dir decouples checkpoints from the (per-restart) wandb
    # run id, so an auto-restarting launcher can always find and resume the
    # highest-step ckpt from ONE stable directory instead of a fresh run_id dir.
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"[train_4ch] checkpoint_dir = {checkpoint_dir}")

    # save_last=True so Ctrl-C / crash still leaves a recoverable ckpt.
    # enable_version_counter=False avoids epoch=N-v1 filename churn on resume.
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            every_n_train_steps=args.checkpoint_every,
            dirpath=checkpoint_dir,
            save_top_k=-1,
            save_last=True,
            enable_version_counter=False,
            filename="epoch={epoch}-step={step}",
            auto_insert_metric_name=False,
        ),
        ExceptionCallback(),
        ModelConfigEmbedderCallback(model_config),
        pl.callbacks.ModelSummary(max_depth=2),
    ]
    if args.demo:
        if logger is None:
            print("[train_4ch] --demo requested but logger is 'none'; skipping demo callback")
        else:
            callbacks.append(create_demo_callback_from_config(model_config, demo_dl=train_dl))

    # Refuse to silently fall back to CPU. An empty or stale CUDA_VISIBLE_DEVICES in the
    # launching shell makes torch.cuda.is_available() false, and Lightning will then train
    # a full run on CPU: no GPU memory, all ranks pegged, host RAM climbing, and tqdm too
    # slow to print anything. That looks like a hang and wastes hours before anyone notices.
    if args.num_gpus > 0 and not torch.cuda.is_available():
        raise RuntimeError(
            f"--num-gpus {args.num_gpus} was requested but torch.cuda.is_available() is False. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}. "
            "Unset it (or set it to the GPUs to use) and relaunch; refusing to train on CPU."
        )

    # Multi-GPU: always use DDPStrategy with an explicit process-group timeout so a
    # hung NCCL collective aborts after --ddp-timeout-min instead of the 30min default,
    # letting the resilient launcher restart sooner. find_unused_parameters defaults to
    # True (the generator/discriminator alternation + masked W-downmix decode leave some
    # params grad-less on a given step); pass a *_false strategy string to opt out.
    if args.num_gpus > 1:
        find_unused = not (args.strategy and "find_unused_parameters_false" in args.strategy)
        strategy = DDPStrategy(
            find_unused_parameters=find_unused,
            timeout=timedelta(minutes=args.ddp_timeout_min),
        )
    elif args.strategy:
        strategy = args.strategy
    else:
        strategy = "auto"

    trainer = pl.Trainer(
        devices=args.num_gpus if args.num_gpus > 0 else "auto",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        max_steps=args.max_steps,
        default_root_dir=args.save_dir,
        num_sanity_val_steps=0,
    )

    trainer.fit(training_wrapper, train_dl, ckpt_path=resume_ckpt_path)


if __name__ == "__main__":
    main()
