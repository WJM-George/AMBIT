#!/usr/bin/env python3
"""Train only the discrete adapter/head on frozen P10-v11 shared blocks."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import functools
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.model_sceneplan_codec_v4 import (  # noqa: E402
    ModelScenePlanCodecV4,
)
from stable_audio_tools.data.sceneplan_transfusion_generation_ar_contract import (  # noqa: E402
    canonical_sha256 as _canonical_sha256,
    sha256_file as _sha256_file,
    stage_checkpoint_steps as _stage_checkpoint_steps,
)
from stable_audio_tools.data.sceneplan_transfusion_generation_ar_dataset import (  # noqa: E402
    GenerationARSQLiteDataset,
    LengthBucketDistributedSampler,
    collate_generation_ar,
    load_target_token_lengths,
    manifest_summary,
    select_tiny_overfit_ordinals,
)
from stable_audio_tools.models.sceneplan_transfusion_generation_ar import (  # noqa: E402
    GENERATION_AR_CONTRACT,
    load_p10v11_generation_ar,
)


DATA_ROOT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/transfusion_shared_v1/generation_ar"
)
CODEC_PATH = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)
EXPECTED_QWEN_TEXT_BACKBONE_PARAMETER_COUNT = 752_393_024


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("tiny", "full"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--train-manifest", type=Path, default=DATA_ROOT / "train.sqlite"
    )
    parser.add_argument(
        "--validation-manifest", type=Path, default=DATA_ROOT / "validation.sqlite"
    )
    parser.add_argument("--tiny-rows", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--checkpoint-policy",
        choices=("interval", "half_epoch_and_end"),
        default="interval",
        help=(
            "Use half_epoch_and_end for the full run so only the midpoint and "
            "the explicit final checkpoint are retained."
        ),
    )
    parser.add_argument("--validation-batches", type=int, default=64)
    parser.add_argument("--length-bucket-batches", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume",
        type=Path,
        help="Resume an interrupted checkpoint from this same run directory.",
    )
    resume.add_argument(
        "--extend-from",
        type=Path,
        help=(
            "Start a new, auditable continuation run from the selected terminal "
            "checkpoint of a completed earlier run."
        ),
    )
    parser.add_argument(
        "--parent-selection-manifest",
        type=Path,
        help="Required CHECKPOINT_SELECTION.json proof for --extend-from.",
    )
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    return parser.parse_args()


def _cosine_floor_multiplier(
    local_step: int,
    *,
    schedule_steps: int,
    warmup_steps: int,
    floor_ratio: float = 0.1,
) -> float:
    """Return the LR multiplier for one self-contained training stage."""

    step = int(local_step)
    total = int(schedule_steps)
    warmup = int(warmup_steps)
    floor = float(floor_ratio)
    if step < 0 or total <= 0 or not 0 <= warmup < total:
        raise ValueError("invalid stage-local learning-rate schedule")
    if not 0.0 < floor <= 1.0:
        raise ValueError("learning-rate floor ratio must be in (0, 1]")
    if warmup and step < warmup:
        return float(step + 1) / float(warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _normalise_resume_position(
    *, epoch: int, batch_in_epoch: int, batches_per_epoch: int
) -> tuple[int, int]:
    """Represent an end-of-epoch checkpoint as the next epoch at batch zero."""

    current_epoch = int(epoch)
    batch = int(batch_in_epoch)
    batches = int(batches_per_epoch)
    if current_epoch < 0 or batch < 0 or batches <= 0 or batch > batches:
        raise ValueError("invalid checkpoint epoch/batch position")
    if batch == batches:
        return current_epoch + 1, 0
    return current_epoch, batch


def _validate_extension_parent(
    *,
    checkpoint: Path,
    state: dict[str, Any],
    selection_manifest: Path,
    target_epochs: int,
    steps_per_epoch: int,
    batches_per_epoch: int,
    train_manifest_sha256: str,
    validation_manifest_sha256: str,
    codec_fingerprint: str,
    p10_load: dict[str, Any],
    world_size: int,
    batch_size: int,
    gradient_accumulation: int,
) -> dict[str, Any]:
    """Fail closed unless a new run extends the selected completed parent."""

    parent_checkpoint = checkpoint.expanduser().resolve(strict=True)
    parent_run_dir = parent_checkpoint.parent.parent.resolve(strict=True)
    parent_contract_path = (parent_run_dir / "RUN_CONTRACT.json").resolve(strict=True)
    parent_final_path = (parent_run_dir / "FINAL.json").resolve(strict=True)
    selected_path = selection_manifest.expanduser().resolve(strict=True)
    parent_contract = json.loads(parent_contract_path.read_text(encoding="utf-8"))
    parent_final = json.loads(parent_final_path.read_text(encoding="utf-8"))
    selection = json.loads(selected_path.read_text(encoding="utf-8"))
    parent_step = int(state.get("global_step", -1))
    parent_epochs = int(parent_contract.get("epochs_requested", -1))
    parent_requested_steps = int(parent_contract.get("requested_steps", -1))
    parent_checkpoint_sha256 = _sha256_file(parent_checkpoint)
    coverage = dict(parent_contract.get("training_row_coverage") or {})
    if (
        state.get("contract") != GENERATION_AR_CONTRACT
        or state.get("run_contract") != parent_contract
        or parent_contract.get("mode") != "full"
        or int(parent_contract.get("seed", -1)) != 42
        or int(parent_contract.get("world_size", -1)) != int(world_size)
        or int(parent_contract.get("batch_size_per_rank", -1)) != int(batch_size)
        or int(parent_contract.get("gradient_accumulation", -1))
        != int(gradient_accumulation)
        or int(parent_contract.get("steps_per_epoch", -1)) != int(steps_per_epoch)
        or not 5 <= parent_epochs < int(target_epochs) <= 10
        or parent_requested_steps != parent_epochs * int(steps_per_epoch)
        or parent_step != parent_requested_steps
        or int(state.get("epoch", -1)) != parent_epochs - 1
        or int(state.get("batch_in_epoch", -1)) != int(batches_per_epoch)
        or parent_contract.get("train_manifest_sha256") != str(train_manifest_sha256)
        or parent_contract.get("validation_manifest_sha256")
        != str(validation_manifest_sha256)
        or parent_contract.get("codec_fingerprint") != str(codec_fingerprint)
        or parent_contract.get("p10_load") != p10_load
        or int(coverage.get("unique_rows_per_epoch", -1)) != 1_600_000
        or int(coverage.get("dropped_rows_per_epoch", -1)) != 0
        or int(coverage.get("duplicated_rows_per_epoch", -1)) != 0
        or bool(coverage.get("drop_last", True))
        or parent_final.get("event") != "complete"
        or parent_final.get("mode") != "full"
        or int(parent_final.get("step", -1)) != parent_step
    ):
        raise RuntimeError("Generation AR extension parent run contract mismatch")
    if (
        selection.get("schema")
        != "stable_audio_tools.sceneplan_transfusion_generation_ar_checkpoint_selection"
        or selection.get("status") != "COMPLETE"
        or Path(str(selection.get("training_run_dir", ""))).resolve() != parent_run_dir
        or Path(str(selection.get("selected_checkpoint", ""))).resolve()
        != parent_checkpoint
        or selection.get("selected_checkpoint_sha256") != parent_checkpoint_sha256
        or int(selection.get("selected_checkpoint_step", -1)) != parent_step
        or selection.get("training_run_contract_sha256")
        != _sha256_file(parent_contract_path)
        or selection.get("training_final_sha256") != _sha256_file(parent_final_path)
    ):
        raise RuntimeError("Generation AR extension parent selection mismatch")
    optimizer_groups = list(
        dict(state.get("optimizer") or {}).get("param_groups") or ()
    )
    parent_lrs = sorted({float(group["lr"]) for group in optimizer_groups})
    if len(parent_lrs) != 1 or not math.isfinite(parent_lrs[0]) or parent_lrs[0] <= 0:
        raise RuntimeError("Generation AR parent optimizer LR is invalid")
    return {
        "contract": "selected_completed_generation_ar_parent_v1",
        "run_dir": str(parent_run_dir),
        "run_contract": str(parent_contract_path),
        "run_contract_sha256": _sha256_file(parent_contract_path),
        "run_contract_canonical_sha256": _canonical_sha256(parent_contract),
        "final": str(parent_final_path),
        "final_sha256": _sha256_file(parent_final_path),
        "selection_manifest": str(selected_path),
        "selection_manifest_sha256": _sha256_file(selected_path),
        "checkpoint": str(parent_checkpoint),
        "checkpoint_sha256": parent_checkpoint_sha256,
        "checkpoint_step": parent_step,
        "completed_epochs": parent_epochs,
        "terminal_learning_rate": parent_lrs[0],
        "legacy_rng_scope": "rank0_only; no stochastic train-mode modules",
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Generation AR training requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl", device_id=device)
    return rank, local_rank, world_size, device


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _seed_everything(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    for key in (
        "plan_input_ids",
        "plan_labels",
        "plan_loss_group_ids",
        "plan_attention_mask",
        "ordinals",
        "source_counts",
        "target_lengths",
    ):
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def _parameter_contract(model) -> dict[str, Any]:
    """Prove that only the discrete adapter/head can receive optimizer updates."""

    expected_names = tuple(
        f"ar_adapter.{name}" for name, _ in model.ar_adapter.named_parameters()
    )
    trainable_named = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    qwen_backbone = getattr(model.prompt_conditioner, "model", None)
    if not isinstance(qwen_backbone, torch.nn.Module):
        raise RuntimeError("Generation AR Qwen text backbone is missing")
    qwen_backbone_parameters = tuple(qwen_backbone.parameters())
    qwen_backbone_parameter_count = sum(
        int(parameter.numel()) for parameter in qwen_backbone_parameters
    )
    registered_model_parameter_ids = {
        id(parameter) for _, parameter in model.named_parameters()
    }
    actual_names = tuple(name for name, _ in trainable_named)
    if (
        not expected_names
        or actual_names != expected_names
        or any(parameter.requires_grad for parameter in model.p10_dit.parameters())
        or any(
            parameter.requires_grad
            for parameter in model.prompt_conditioner.parameters()
        )
        or qwen_backbone_parameter_count
        != EXPECTED_QWEN_TEXT_BACKBONE_PARAMETER_COUNT
        or any(parameter.requires_grad for parameter in qwen_backbone_parameters)
        or any(
            id(parameter) in registered_model_parameter_ids
            for parameter in qwen_backbone_parameters
        )
    ):
        raise RuntimeError(
            "Generation AR parameter-freeze contract changed: "
            f"expected={expected_names}, actual={actual_names}, "
            f"qwen_backbone_parameters={qwen_backbone_parameter_count}"
        )
    return {
        "parameter_freeze_contract": "only_discrete_ar_adapter_trainable_v1",
        "trainable_parameter_names": list(actual_names),
        "trainable_parameter_count": sum(
            int(parameter.numel()) for _, parameter in trainable_named
        ),
        "frozen_p10_parameter_count": sum(
            int(parameter.numel()) for parameter in model.p10_dit.parameters()
        ),
        "frozen_qwen_parameter_count": qwen_backbone_parameter_count,
        "frozen_qwen_registered_auxiliary_parameter_count": sum(
            int(parameter.numel())
            for parameter in model.prompt_conditioner.parameters()
        ),
        "qwen_backbone_storage_contract": (
            "external_frozen_text_backbone_excluded_from_generation_checkpoint_v1"
        ),
    }


def _teacher_forced_metrics(
    model,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None,
    cached_context: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict[str, float]:
    model.eval()
    token_correct = 0
    token_total = 0
    sequence_correct = 0
    sequence_total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = _move_batch(raw_batch, device)
            if cached_context is None:
                context, context_mask = model.encode_requests(
                    batch["raw_user_requests"], device=device
                )
            else:
                context, context_mask = _cached_context_batch(
                    batch["ordinals"], cached_context, device=device
                )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(
                    batch["plan_input_ids"],
                    batch["plan_attention_mask"],
                    context,
                    context_mask,
                )
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    batch["plan_labels"].reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
            predictions = logits.argmax(dim=-1)
            valid = batch["plan_labels"] != -100
            correct = predictions.eq(batch["plan_labels"]) & valid
            token_correct += int(correct.sum().item())
            valid_count = int(valid.sum().item())
            token_total += valid_count
            sequence_correct += int((correct | ~valid).all(dim=1).sum().item())
            sequence_total += int(valid.shape[0])
            loss_sum += float(loss.float().item())
    model.train()
    return {
        "loss": loss_sum / max(1, token_total),
        "token_accuracy": token_correct / max(1, token_total),
        "teacher_forced_sequence_exact": sequence_correct / max(1, sequence_total),
        "tokens": float(token_total),
        "sequences": float(sequence_total),
    }


def _cached_context_batch(
    ordinals: torch.Tensor,
    cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = [cache[int(value)] for value in ordinals.detach().cpu().tolist()]
    maximum = max(int(context.shape[0]) for context, _ in selected)
    hidden_dim = int(selected[0][0].shape[-1])
    dtype = selected[0][0].dtype
    context_batch = torch.zeros(
        len(selected), maximum, hidden_dim, device=device, dtype=dtype
    )
    mask_batch = torch.zeros(len(selected), maximum, device=device, dtype=torch.bool)
    for index, (context, mask) in enumerate(selected):
        length = int(mask.sum().item())
        context_batch[index, :length] = context[:length].to(device)
        mask_batch[index, :length] = True
    return context_batch, mask_batch


def _build_tiny_context_cache(
    model,
    dataset: GenerationARSQLiteDataset,
    *,
    device: torch.device,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    rows = [dataset[index] for index in range(len(dataset))]
    context, mask = model.encode_requests(
        [row["raw_user_request"] for row in rows], device=device
    )
    result = {}
    for index, row in enumerate(rows):
        length = int(mask[index].sum().item())
        result[int(row["ordinal"])] = (
            context[index, :length].detach(),
            mask[index, :length].detach(),
        )
    return result


def _autoregressive_exact_metrics(
    model,
    dataset: GenerationARSQLiteDataset,
    codec: ModelScenePlanCodecV4,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    exact = 0
    parseable = 0
    total = len(dataset)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, total, int(batch_size)):
            rows = [
                dataset[index]
                for index in range(start, min(total, start + int(batch_size)))
            ]
            predicted = model.generate_constrained(
                [row["raw_user_request"] for row in rows],
                codec,
                device=device,
                max_plan_tokens=1024,
            )
            for row, token_ids in zip(rows, predicted):
                target = row["target_token_ids"].tolist()
                exact += int(token_ids == target)
                try:
                    codec.decode(token_ids, sample_id="tiny_autoregressive_gate")
                    parseable += 1
                except Exception:
                    pass
    model.train()
    return {
        "autoregressive_sequence_exact": exact / max(1, total),
        "autoregressive_parse_rate": parseable / max(1, total),
        "autoregressive_sequences": float(total),
    }


def _save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    global_step: int,
    epoch: int,
    batch_in_epoch: int,
    run_contract: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "schema": "stable_audio_tools.sceneplan_transfusion_generation_ar_checkpoint",
            "schema_version": 1,
            "contract": GENERATION_AR_CONTRACT,
            "global_step": int(global_step),
            "epoch": int(epoch),
            "batch_in_epoch": int(batch_in_epoch),
            "ar_adapter": model.trainable_state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "run_contract": run_contract,
        },
        temporary,
    )
    os.replace(temporary, path)
    _atomic_json(
        path.parent / "LATEST.json",
        {
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": _sha256_file(path),
            "global_step": int(global_step),
        },
    )


def main() -> int:
    args = _parse_args()
    rank, local_rank, world_size, device = _distributed()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    expected_visible = "0" if args.mode == "tiny" else "0,1,2"
    if visible.replace(" ", "") != expected_visible:
        raise RuntimeError(
            f"Generation {args.mode} must use CUDA_VISIBLE_DEVICES={expected_visible}, "
            f"got {visible!r}"
        )
    if args.mode == "tiny" and world_size != 1:
        raise RuntimeError("tiny overfit is a single-GPU correctness gate")
    if args.mode == "full" and world_size != 3:
        raise RuntimeError("full Generation AR requires exactly GPUs 0-2")
    if int(args.seed) != 42:
        raise ValueError("Generation AR v1 uses frozen seed 42")
    if min(args.batch_size, args.grad_accum, args.epochs) <= 0:
        raise ValueError("batch size, accumulation, and epochs must be positive")
    if (args.extend_from is None) != (args.parent_selection_manifest is None):
        raise ValueError(
            "--extend-from and --parent-selection-manifest must be provided together"
        )
    if args.extend_from is not None and (
        args.mode != "full" or args.max_steps is not None or int(args.warmup_steps) != 0
    ):
        raise ValueError(
            "Generation AR continuation requires full mode, epoch-based length, "
            "and --warmup-steps 0"
        )

    _seed_everything(args.seed, rank)
    run_dir = args.run_dir.expanduser().resolve(strict=False)
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
    _barrier(world_size)
    codec = ModelScenePlanCodecV4(CODEC_PATH)
    train_manifest = args.train_manifest.expanduser().resolve(strict=True)
    validation_manifest = args.validation_manifest.expanduser().resolve(strict=True)

    tiny_ordinals = None
    if args.mode == "tiny":
        tiny_ordinals = select_tiny_overfit_ordinals(
            train_manifest, rows=int(args.tiny_rows)
        )
    train_dataset = GenerationARSQLiteDataset(
        train_manifest, split="train", row_ordinals=tiny_ordinals
    )
    validation_dataset = (
        train_dataset
        if args.mode == "tiny"
        else GenerationARSQLiteDataset(validation_manifest, split="validation")
    )
    train_sampler = None
    if world_size > 1:
        train_sampler = LengthBucketDistributedSampler(
            load_target_token_lengths(train_manifest),
            num_replicas=world_size,
            rank=rank,
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            bucket_batches=int(args.length_bucket_batches),
        )
    collate = functools.partial(collate_generation_ar, pad_id=codec.pad_id)
    loader_generator = torch.Generator().manual_seed(int(args.seed) + rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        sampler=train_sampler,
        shuffle=train_sampler is None,
        generator=loader_generator,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=int(args.num_workers) > 0,
        drop_last=False,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate,
    )

    model, p10_report = load_p10v11_generation_ar(
        pad_id=codec.pad_id,
        verify_sha256=rank == 0,
        activation_checkpointing=not args.no_activation_checkpointing,
    )
    model.p10_dit.to(device=device, dtype=torch.bfloat16)
    model.prompt_conditioner.to(device=device, dtype=torch.bfloat16)
    model.ar_adapter.to(device=device, dtype=torch.float32)
    model.train()
    parameter_contract = _parameter_contract(model)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(args.learning_rate),
        betas=(0.9, 0.95),
        weight_decay=float(args.weight_decay),
        fused=True,
    )
    steps_per_epoch = len(train_loader) // int(args.grad_accum)
    step_limited = args.max_steps is not None or args.mode == "tiny"
    requested_steps = (
        int(args.max_steps)
        if args.max_steps is not None
        else (1000 if args.mode == "tiny" else int(args.epochs) * steps_per_epoch)
    )
    if requested_steps <= 0:
        raise ValueError("training configuration produces no optimizer steps")
    if int(args.save_every) <= 0:
        raise ValueError("--save-every must be positive")
    if len(train_loader) % int(args.grad_accum):
        raise ValueError(
            "train batches per epoch must be divisible by gradient accumulation"
        )
    global_step = 0
    start_epoch = 0
    resume_batch = 0
    resume_path: Path | None = None
    resume_state: dict[str, Any] | None = None
    parent_path: Path | None = None
    parent_state: dict[str, Any] | None = None
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve(strict=True)
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state.get("contract") != GENERATION_AR_CONTRACT:
            raise RuntimeError("resume checkpoint contract mismatch")
        model.load_trainable_state_dict(state["ar_adapter"])
        optimizer.load_state_dict(state["optimizer"])
        global_step = int(state["global_step"])
        start_epoch = int(state["epoch"])
        resume_batch = int(state["batch_in_epoch"])
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        random.setstate(state["python_rng_state"])
        resume_state = state
    elif args.extend_from is not None:
        parent_path = args.extend_from.expanduser().resolve(strict=True)
        state = torch.load(parent_path, map_location="cpu", weights_only=False)
        if state.get("contract") != GENERATION_AR_CONTRACT:
            raise RuntimeError("extension checkpoint contract mismatch")
        model.load_trainable_state_dict(state["ar_adapter"])
        optimizer.load_state_dict(state["optimizer"])
        global_step = int(state["global_step"])
        start_epoch = int(state["epoch"])
        resume_batch = int(state["batch_in_epoch"])
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        random.setstate(state["python_rng_state"])
        parent_state = state

    manifest_sha256: list[str | None] = [None, None]
    if rank == 0:
        manifest_sha256[:] = [
            _sha256_file(train_manifest),
            _sha256_file(validation_manifest),
        ]
    if world_size > 1:
        dist.broadcast_object_list(manifest_sha256, src=0, device=device)

    parent_lineage_values: list[dict[str, Any] | None] = [None]
    if parent_state is not None:
        if rank == 0:
            assert parent_path is not None
            assert args.parent_selection_manifest is not None
            parent_lineage_values[0] = _validate_extension_parent(
                checkpoint=parent_path,
                state=parent_state,
                selection_manifest=args.parent_selection_manifest,
                target_epochs=int(args.epochs),
                steps_per_epoch=steps_per_epoch,
                batches_per_epoch=len(train_loader),
                train_manifest_sha256=str(manifest_sha256[0]),
                validation_manifest_sha256=str(manifest_sha256[1]),
                codec_fingerprint=codec.fingerprint,
                p10_load=p10_report.as_dict(),
                world_size=world_size,
                batch_size=int(args.batch_size),
                gradient_accumulation=int(args.grad_accum),
            )
        if world_size > 1:
            dist.broadcast_object_list(parent_lineage_values, src=0, device=device)
    parent_lineage = parent_lineage_values[0]

    if resume_state is not None:
        resume_schedule = dict(
            dict(resume_state.get("run_contract") or {}).get("learning_rate_schedule")
            or {}
        )
        stage_start_step = int(resume_schedule.get("stage_start_step", 0))
    elif parent_lineage is not None:
        stage_start_step = int(parent_lineage["checkpoint_step"])
        parent_lr = float(parent_lineage["terminal_learning_rate"])
        if not math.isclose(
            float(args.learning_rate), parent_lr, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(
                "continuation --learning-rate must equal the parent terminal LR "
                f"({parent_lr:.17g})"
            )
    else:
        stage_start_step = 0
    if not 0 <= stage_start_step <= global_step < requested_steps:
        raise RuntimeError("Generation AR stage/global step contract is invalid")
    start_epoch, resume_batch = _normalise_resume_position(
        epoch=start_epoch,
        batch_in_epoch=resume_batch,
        batches_per_epoch=len(train_loader),
    )
    warmup = min(int(args.warmup_steps), max(0, requested_steps - stage_start_step - 1))
    schedule_steps = requested_steps - stage_start_step
    schedule_base_lr = float(args.learning_rate)
    for group in optimizer.param_groups:
        group["initial_lr"] = schedule_base_lr
        if parent_lineage is not None:
            group["lr"] = schedule_base_lr

    def lr_lambda(local_step: int) -> float:
        return _cosine_floor_multiplier(
            local_step,
            schedule_steps=schedule_steps,
            warmup_steps=warmup,
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if resume_state is not None:
        scheduler.load_state_dict(resume_state["scheduler"])
        expected_local_step = global_step - stage_start_step
        if int(scheduler.last_epoch) != expected_local_step:
            raise RuntimeError("resume scheduler is not aligned to the staged step")
        for group, learning_rate in zip(
            optimizer.param_groups, scheduler.get_last_lr(), strict=True
        ):
            group["lr"] = float(learning_rate)

    checkpoint_steps = _stage_checkpoint_steps(
        stage_start_step=stage_start_step,
        requested_steps=requested_steps,
        save_every_steps=int(args.save_every),
        policy=str(args.checkpoint_policy),
    )
    if args.checkpoint_policy == "half_epoch_and_end" and args.mode != "full":
        raise ValueError("half_epoch_and_end is only valid for full training")
    periodic_checkpoint_steps = set(checkpoint_steps) - {requested_steps}

    source_files = [
        REPO_ROOT / "stable_audio_tools/models/sceneplan_transfusion_generation_ar.py",
        REPO_ROOT / "stable_audio_tools/models/transformer.py",
        REPO_ROOT / "stable_audio_tools/data/sceneplan_transfusion_generation.py",
        REPO_ROOT
        / "stable_audio_tools/data/sceneplan_transfusion_generation_ar_dataset.py",
        REPO_ROOT
        / "stable_audio_tools/data/sceneplan_transfusion_generation_ar_contract.py",
        Path(__file__).resolve(),
    ]
    snapshot_manifest_path = REPO_ROOT / "SOURCE_SNAPSHOT_MANIFEST.json"
    source_snapshot = None
    if snapshot_manifest_path.is_file():
        snapshot_manifest = json.loads(
            snapshot_manifest_path.read_text(encoding="utf-8")
        )
        if (
            snapshot_manifest.get("status") != "IMMUTABLE_FOR_GENERATION_AR_RUN"
            or Path(str(snapshot_manifest.get("snapshot_root", ""))).resolve()
            != REPO_ROOT
        ):
            raise RuntimeError("Generation AR source snapshot manifest mismatch")
        source_snapshot = {
            "path": str(snapshot_manifest_path.resolve()),
            "sha256": _sha256_file(snapshot_manifest_path),
            "tree_sha256_excluding_manifest": snapshot_manifest.get(
                "tree_sha256_excluding_this_manifest"
            ),
        }
    run_contract = {
        "schema": "stable_audio_tools.sceneplan_transfusion_generation_ar_run",
        "schema_version": 3,
        "contract": GENERATION_AR_CONTRACT,
        "mode": args.mode,
        "seed": int(args.seed),
        "world_size": world_size,
        "cuda_visible_devices": visible,
        "batch_size_per_rank": int(args.batch_size),
        "gradient_accumulation": int(args.grad_accum),
        "effective_batch_size": int(args.batch_size)
        * world_size
        * int(args.grad_accum),
        "dataloader_num_workers_per_rank": int(args.num_workers),
        "dataloader_pin_memory": True,
        "online_validation_every_steps": int(args.validate_every),
        "online_validation_batches": int(args.validation_batches),
        "gpu_memory_total_gib_per_rank": round(
            torch.cuda.get_device_properties(device).total_memory / (1024**3), 6
        ),
        "epochs_requested": int(args.epochs),
        "requested_steps": requested_steps,
        "steps_per_epoch": steps_per_epoch,
        "stage_start_step": stage_start_step,
        "stage_start_epoch": stage_start_step // max(1, steps_per_epoch),
        "stage_epochs": (
            (requested_steps - stage_start_step) // max(1, steps_per_epoch)
            if (requested_steps - stage_start_step) % max(1, steps_per_epoch) == 0
            else None
        ),
        "checkpoint_policy": str(args.checkpoint_policy),
        "save_every_steps": int(args.save_every),
        "checkpoint_steps": list(checkpoint_steps),
        "selection_candidate_steps": (
            [stage_start_step, *checkpoint_steps]
            if parent_lineage is not None
            else list(checkpoint_steps)
        ),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "warmup_steps": warmup,
        "learning_rate_schedule": {
            "type": "stage_local_cosine_floor_v1",
            "stage_start_step": stage_start_step,
            "stage_end_step": requested_steps,
            "schedule_steps": schedule_steps,
            "base_learning_rate": schedule_base_lr,
            "floor_ratio": 0.1,
            "warmup_steps": warmup,
        },
        "parent_lineage": parent_lineage,
        "activation_checkpointing": not args.no_activation_checkpointing,
        "tiny_ordinals": list(tiny_ordinals or ()),
        "train_manifest": manifest_summary(train_manifest),
        "train_manifest_sha256": manifest_sha256[0],
        "validation_manifest": manifest_summary(validation_manifest),
        "validation_manifest_sha256": manifest_sha256[1],
        "training_row_coverage": {
            "manifest_rows": len(train_dataset),
            "unique_rows_per_epoch": len(train_dataset),
            "duplicated_rows_per_epoch": 0,
            "dropped_rows_per_epoch": 0,
            "last_global_batch_rows": (
                len(train_dataset) % (int(args.batch_size) * world_size)
            ),
            "rank_sampler_rows": (
                [
                    (
                        len(train_dataset) // world_size
                        + int(rank_index < len(train_dataset) % world_size)
                    )
                    for rank_index in range(world_size)
                ]
                if train_sampler is not None
                else [len(train_dataset)]
            ),
            "drop_last": False,
            "stage_epochs": (
                (requested_steps - stage_start_step) // max(1, steps_per_epoch)
                if (requested_steps - stage_start_step) % max(1, steps_per_epoch) == 0
                else None
            ),
            "cumulative_epochs": (
                requested_steps // max(1, steps_per_epoch)
                if requested_steps % max(1, steps_per_epoch) == 0
                else None
            ),
        },
        "p10_load": p10_report.as_dict(),
        "codec_path": str(CODEC_PATH),
        "codec_fingerprint": codec.fingerprint,
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): _sha256_file(path)
            for path in source_files
        },
        "source_snapshot": source_snapshot,
        **parameter_contract,
        "active_dropout_modules": [
            name
            for name, module in model.named_modules()
            if module.training
            and isinstance(
                module,
                (
                    torch.nn.Dropout,
                    torch.nn.Dropout1d,
                    torch.nn.Dropout2d,
                    torch.nn.Dropout3d,
                    torch.nn.AlphaDropout,
                    torch.nn.FeatureAlphaDropout,
                ),
            )
        ],
    }
    if args.mode == "full":
        if (
            len(train_dataset) != 1_600_000
            or len(validation_dataset) != 32_000
            or run_contract["training_row_coverage"]["dropped_rows_per_epoch"] != 0
            or run_contract["training_row_coverage"]["duplicated_rows_per_epoch"] != 0
            or requested_steps % steps_per_epoch != 0
            or stage_start_step % steps_per_epoch != 0
            or run_contract["active_dropout_modules"]
        ):
            raise RuntimeError(
                "full Generation AR training requires exact 1.6M/32K coverage"
            )
    if resume_state is not None:
        expected_checkpoint_dir = (run_dir / "checkpoints").resolve(strict=True)
        if (
            resume_path is None
            or resume_path.parent != expected_checkpoint_dir
            or resume_state.get("run_contract") != run_contract
        ):
            raise RuntimeError("resume checkpoint run contract mismatch")
    if rank == 0:
        contract_path = run_dir / "RUN_CONTRACT.json"
        if contract_path.exists():
            existing = json.loads(contract_path.read_text(encoding="utf-8"))
            if existing != run_contract:
                raise RuntimeError("run directory already has a different contract")
        else:
            _atomic_json(contract_path, run_contract)
    _barrier(world_size)

    tiny_context = None
    if args.mode == "tiny":
        tiny_context = _build_tiny_context_cache(model, train_dataset, device=device)
        train_dataset.close()
    wrapped = (
        DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        if world_size > 1
        else model
    )
    module = wrapped.module if isinstance(wrapped, DistributedDataParallel) else wrapped
    metrics_path = run_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    last_log_started = training_started
    last_log_step = global_step
    accumulated_loss = 0.0
    accumulated_tokens = 0
    exact_streak = 0
    stop = False

    for epoch in range(start_epoch, 10**9 if step_limited else int(args.epochs)):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch_index, raw_batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < resume_batch:
                continue
            batch = _move_batch(raw_batch, device)
            microstep = batch_index % int(args.grad_accum)
            should_step = microstep == int(args.grad_accum) - 1
            sync_context = (
                wrapped.no_sync()
                if isinstance(wrapped, DistributedDataParallel) and not should_step
                else nullcontext()
            )
            if tiny_context is None:
                with torch.no_grad():
                    context, context_mask = module.encode_requests(
                        batch["raw_user_requests"], device=device
                    )
            else:
                context, context_mask = _cached_context_batch(
                    batch["ordinals"], tiny_context, device=device
                )
            with sync_context, torch.autocast("cuda", dtype=torch.bfloat16):
                logits = wrapped(
                    batch["plan_input_ids"],
                    batch["plan_attention_mask"],
                    context,
                    context_mask,
                )
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    batch["plan_labels"].reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
            valid_tokens = int((batch["plan_labels"] != -100).sum().item())
            global_tokens = torch.tensor(
                [valid_tokens], device=device, dtype=torch.float64
            )
            if world_size > 1:
                dist.all_reduce(global_tokens, op=dist.ReduceOp.SUM)
            # DDP averages gradients across ranks.  Multiplying each local
            # token-loss sum by world_size/global_tokens yields the exact
            # global token mean, including the uneven final 22/21/21 batch.
            loss = loss * (float(world_size) / float(global_tokens.item()))
            scaled_loss = loss / int(args.grad_accum)
            scaled_loss.backward()
            accumulated_loss += (
                float(loss.detach().float().item())
                * float(global_tokens.item())
                / float(world_size)
            )
            accumulated_tokens += valid_tokens
            if not should_step:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not bool(torch.isfinite(grad_norm)):
                raise RuntimeError(
                    f"non-finite Generation AR gradient at step {global_step}"
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % int(args.log_every) == 0:
                now = time.perf_counter()
                global_loss_tokens = torch.tensor(
                    [accumulated_loss, accumulated_tokens],
                    device=device,
                    dtype=torch.float64,
                )
                memory_bytes = torch.tensor(
                    [
                        torch.cuda.memory_allocated(device),
                        torch.cuda.memory_reserved(device),
                        torch.cuda.max_memory_allocated(device),
                        torch.cuda.max_memory_reserved(device),
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                interval_seconds = torch.tensor(
                    [now - last_log_started], device=device, dtype=torch.float64
                )
                if world_size > 1:
                    dist.all_reduce(global_loss_tokens, op=dist.ReduceOp.SUM)
                    dist.all_reduce(memory_bytes, op=dist.ReduceOp.MAX)
                    dist.all_reduce(interval_seconds, op=dist.ReduceOp.MAX)
                if rank == 0:
                    interval = max(float(interval_seconds.item()), 1e-9)
                    gib = 1024**3
                    event = {
                        "event": "train",
                        "step": global_step,
                        "epoch": epoch,
                        "batch_in_epoch": batch_index + 1,
                        "loss": float(global_loss_tokens[0].item())
                        / max(1.0, float(global_loss_tokens[1].item())),
                        "tokens": int(global_loss_tokens[1].item()),
                        "grad_norm": float(grad_norm.detach().float().item()),
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                        "elapsed_sec": round(now - training_started, 3),
                        "interval_sec": round(interval, 6),
                        "steps_per_sec": (global_step - last_log_step) / interval,
                        "global_tokens_per_sec": float(global_loss_tokens[1].item())
                        / interval,
                        "cuda_memory_allocated_gib_max_rank": float(
                            memory_bytes[0].item()
                        )
                        / gib,
                        "cuda_memory_reserved_gib_max_rank": float(
                            memory_bytes[1].item()
                        )
                        / gib,
                        "cuda_peak_memory_allocated_gib_max_rank": float(
                            memory_bytes[2].item()
                        )
                        / gib,
                        "cuda_peak_memory_reserved_gib_max_rank": float(
                            memory_bytes[3].item()
                        )
                        / gib,
                    }
                    _append_jsonl(metrics_path, event)
                    print(json.dumps(event, sort_keys=True), flush=True)
                accumulated_loss = 0.0
                accumulated_tokens = 0
                last_log_started = now
                last_log_step = global_step

            if global_step % int(args.validate_every) == 0:
                _barrier(world_size)
                if rank == 0:
                    metrics = _teacher_forced_metrics(
                        module,
                        validation_loader,
                        device=device,
                        max_batches=(
                            None
                            if args.mode == "tiny"
                            else int(args.validation_batches)
                        ),
                        cached_context=tiny_context,
                    )
                    event = {"event": "validation", "step": global_step, **metrics}
                    _append_jsonl(metrics_path, event)
                    print(json.dumps(event, sort_keys=True), flush=True)
                    if (
                        args.mode == "tiny"
                        and metrics["teacher_forced_sequence_exact"] == 1.0
                    ):
                        exact_streak += 1
                    else:
                        exact_streak = 0
                    if args.mode == "tiny" and exact_streak >= 2:
                        autoregressive = _autoregressive_exact_metrics(
                            module,
                            train_dataset,
                            codec,
                            device=device,
                            batch_size=int(args.batch_size),
                        )
                        ar_event = {
                            "event": "autoregressive_validation",
                            "step": global_step,
                            **autoregressive,
                        }
                        _append_jsonl(metrics_path, ar_event)
                        print(json.dumps(ar_event, sort_keys=True), flush=True)
                        if autoregressive["autoregressive_sequence_exact"] == 1.0:
                            stop = True
                        else:
                            exact_streak = 0
                if world_size > 1:
                    stop_tensor = torch.tensor(
                        [int(stop)], device=device, dtype=torch.int32
                    )
                    dist.broadcast(stop_tensor, src=0)
                    stop = bool(stop_tensor.item())
                _barrier(world_size)

            if global_step in periodic_checkpoint_steps and rank == 0:
                _save_checkpoint(
                    run_dir / "checkpoints" / f"step_{global_step:08d}.pt",
                    model=module,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    epoch=epoch,
                    batch_in_epoch=batch_index + 1,
                    run_contract=run_contract,
                )
            if stop or global_step >= requested_steps:
                stop = True
                break
        resume_batch = 0
        if stop:
            break

    _barrier(world_size)
    if rank == 0:
        final_metrics = _teacher_forced_metrics(
            module,
            validation_loader,
            device=device,
            max_batches=None if args.mode == "tiny" else int(args.validation_batches),
            cached_context=tiny_context,
        )
        if args.mode == "tiny":
            final_metrics.update(
                _autoregressive_exact_metrics(
                    module,
                    train_dataset,
                    codec,
                    device=device,
                    batch_size=int(args.batch_size),
                )
            )
        final_event = {
            "event": "complete",
            "mode": args.mode,
            "step": global_step,
            **final_metrics,
        }
        _append_jsonl(metrics_path, final_event)
        _save_checkpoint(
            run_dir / "checkpoints" / f"step_{global_step:08d}.pt",
            model=module,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            epoch=epoch,
            batch_in_epoch=batch_index + 1,
            run_contract=run_contract,
        )
        _atomic_json(run_dir / "FINAL.json", final_event)
        print(json.dumps(final_event, sort_keys=True), flush=True)
        if args.mode == "tiny" and (
            final_metrics["teacher_forced_sequence_exact"] != 1.0
            or final_metrics["autoregressive_sequence_exact"] != 1.0
            or final_metrics["autoregressive_parse_rate"] != 1.0
        ):
            raise RuntimeError("tiny Generation AR did not exactly overfit its targets")
    _barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
