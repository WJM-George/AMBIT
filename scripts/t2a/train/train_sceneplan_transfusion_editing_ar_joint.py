#!/usr/bin/env python3
"""Tiny/full Editing AR training with mandatory Editing-DiT RF replay.

Latest route only:

    clean source FOA latent + raw edit instruction -> complete new ScenePlan

The paired database may store an old ScenePlan for offline audit, but this
program uses the fail-closed Editing-AR reader that never selects that column.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.data.model_sceneplan_codec_v4 import (  # noqa: E402
    ModelScenePlanCodecV4,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_ar_dataset import (  # noqa: E402
    collate_editing_ar,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import (  # noqa: E402
    ScenePlanTransfusionEditingDataset,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_joint_dataset import (  # noqa: E402
    ScenePlanTransfusionEditingJointDataset,
)
from stable_audio_tools.data.sceneplan_bucket_sampler import (  # noqa: E402
    sceneplan_bucket_collation,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_index import (  # noqa: E402
    sha256_file,
)
from stable_audio_tools.models.factory import create_model_from_config  # noqa: E402
from stable_audio_tools.models.sceneplan_transfusion_editing_ar import (  # noqa: E402
    EDITING_AR_CONTRACT,
    ScenePlanTransfusionEditingAR,
)
from stable_audio_tools.models.utils import load_ckpt_state_dict  # noqa: E402
from stable_audio_tools.training.factory import (  # noqa: E402
    create_training_wrapper_from_config,
)


DEFAULT_MODEL_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m_sceneplan_transfusion_editing_dit_v1.json"
)
DEFAULT_DATASET_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/dataset_configs/"
    "sceneplan_transfusion_editing_v1_overfit10.json"
)
DEFAULT_INDEX = Path(
    "/mnt/sdb/audio_dataset/sceneplan_transfusion_editing_v1_pilot/"
    "training_index/train.sqlite"
)
DEFAULT_INDEX_SHA256 = (
    "6535b4a04b38a07ff641a876361e0dd16d4d4d4737ad27c11304d1d99c785600"
)
DEFAULT_CODEC = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)
DEFAULT_CHECKPOINT = Path(
    "/mnt/sdb/model_archives/transfusion_editing/pilots/"
    "sceneplan_transfusion_editing_dit_overfit10_seed42_v1/checkpoints/"
    "epoch=499-step=1000.ckpt"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tiny", "full"), default="tiny")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--index-sha256", default=DEFAULT_INDEX_SHA256)
    parser.add_argument("--codec", type=Path, default=DEFAULT_CODEC)
    parser.add_argument("--expected-rows", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--shared-learning-rate", type=float, default=2e-6)
    parser.add_argument("--dit-learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--lambda-ar", type=float, default=0.1)
    parser.add_argument("--lambda-rf", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--evaluate-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physical-gpu", type=int, default=3)
    return parser.parse_args()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _editing_device(physical_gpu: int) -> torch.device:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    expected = str(int(physical_gpu))
    if int(physical_gpu) not in range(3, 8) or visible != expected:
        raise RuntimeError(
            "Editing training requires exactly one remapped physical GPU 3--7; "
            f"expected CUDA_VISIBLE_DEVICES={expected}, got {visible!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Editing training requires exactly one visible CUDA device")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def _unique_trainable_parameters(modules: Iterable[torch.nn.Module]):
    seen: set[int] = set()
    output = []
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                output.append(parameter)
    if not output:
        raise RuntimeError("joint Editing training found no trainable parameters")
    return output


def _copy_ema_to_online(wrapper) -> None:
    if wrapper.diffusion_ema is None:
        raise RuntimeError("Editing warm-up checkpoint must contain DiT EMA")
    incompatible = wrapper.diffusion.model.load_state_dict(
        wrapper.diffusion_ema.ema_model.state_dict(), strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("could not promote Editing-DiT EMA to online route")
    if wrapper.conditioner_ema is not None:
        wrapper.conditioner_ema.copy_to(wrapper.diffusion.conditioner)


def _prepare_dit_batch(
    rows: list[tuple[torch.Tensor, dict[str, Any]]], device: torch.device
) -> tuple[torch.Tensor, list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    target_cpu, metadata = sceneplan_bucket_collation(rows)
    target = target_cpu.to(device=device, dtype=torch.float32)
    mask = torch.stack([row["padding_mask"][0] for row in metadata]).to(device)
    source = torch.stack([row["source_foa_latent"] for row in metadata]).to(
        device=device, dtype=torch.float32
    )
    return target, metadata, mask, source


def _conditioning_with_exact_source(diffusion, metadata, source, device):
    conditioning = diffusion.conditioner(metadata, device)
    conditioning["source_foa_latent"] = [source, None]
    return diffusion.get_conditioning_inputs(conditioning)


@torch.no_grad()
def _teacher_forced_ar_metrics(
    ar: ScenePlanTransfusionEditingAR,
    rows: list[dict[str, Any]],
    *,
    codec: ModelScenePlanCodecV4,
    device: torch.device,
) -> dict[str, float]:
    ar.eval()
    clean_loss = 0.0
    zero_loss = 0.0
    shuffled_loss = 0.0
    clean_zero_l1 = 0.0
    clean_shuffle_l1 = 0.0
    correct = 0
    tokens = 0
    exact = 0
    for index, row in enumerate(rows):
        batch = collate_editing_ar([row], pad_id=codec.pad_id)
        donor = rows[(index + 1) % len(rows)]
        source = batch["source_foa_latent"].to(device=device, dtype=torch.float32)
        source_mask = batch["source_attention_mask"].to(device)
        source_frames = int(source.shape[-1])
        donor_source = donor["source_foa_latent"][None, :, :source_frames].to(
            device=device, dtype=torch.float32
        )
        donor_mask = donor["source_attention_mask"][None, :source_frames].to(device)
        ids = batch["plan_input_ids"].to(device)
        labels = batch["plan_labels"].to(device)
        plan_mask = batch["plan_attention_mask"].to(device)
        context, context_mask = ar.encode_edit_instructions(
            batch["raw_edit_requests"], device=device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = ar(source, source_mask, ids, plan_mask, context, context_mask)
            zero_logits = ar(
                torch.zeros_like(source),
                source_mask,
                ids,
                plan_mask,
                context,
                context_mask,
            )
            shuffle_logits = ar(
                donor_source,
                donor_mask,
                ids,
                plan_mask,
                context,
                context_mask,
            )
        clean_loss += float(
            F.cross_entropy(logits.float().flatten(0, 1), labels.flatten()).item()
        ) * labels.numel()
        zero_loss += float(
            F.cross_entropy(zero_logits.float().flatten(0, 1), labels.flatten()).item()
        ) * labels.numel()
        shuffled_loss += float(
            F.cross_entropy(
                shuffle_logits.float().flatten(0, 1), labels.flatten()
            ).item()
        ) * labels.numel()
        clean_zero_l1 += float((logits.float() - zero_logits.float()).abs().mean())
        clean_shuffle_l1 += float(
            (logits.float() - shuffle_logits.float()).abs().mean()
        )
        predicted = logits.argmax(dim=-1)
        row_correct = predicted.eq(labels)
        correct += int(row_correct.sum().item())
        tokens += int(labels.numel())
        exact += int(bool(row_correct.all()))
    ar.train()
    count = len(rows)
    return {
        "clean_ce": clean_loss / max(1, tokens),
        "zero_reference_ce": zero_loss / max(1, tokens),
        "shuffled_reference_ce": shuffled_loss / max(1, tokens),
        "clean_zero_logits_l1": clean_zero_l1 / count,
        "clean_shuffle_logits_l1": clean_shuffle_l1 / count,
        "token_accuracy": correct / max(1, tokens),
        "teacher_forced_sequence_exact": exact / count,
        "tokens": float(tokens),
        "rows": float(count),
    }


@torch.no_grad()
def _fixed_rf_metric(
    diffusion,
    route_wrapper,
    rows: list[tuple[torch.Tensor, dict[str, Any]]],
    *,
    device: torch.device,
) -> float:
    route_wrapper.eval()
    diffusion.conditioner.eval()
    numerator = 0.0
    denominator = 0
    for index, row in enumerate(rows):
        target, metadata, mask, source = _prepare_dit_batch([row], device)
        generator = torch.Generator(device=device).manual_seed(420042 + index)
        noise = torch.randn(
            target.shape, generator=generator, device=device, dtype=target.dtype
        )
        t = torch.full((1,), 0.5, device=device)
        noised = 0.5 * target + 0.5 * noise
        inputs = _conditioning_with_exact_source(
            diffusion, metadata, source, device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = route_wrapper(
                noised,
                t,
                **inputs,
                cfg_dropout_prob=0.0,
                padding_mask=mask,
            )
        squared = (prediction.float() - (noise - target)) ** 2
        numerator += float((squared * mask[:, None]).sum().item())
        denominator += int(mask.sum().item()) * int(target.shape[1])
    route_wrapper.train()
    diffusion.conditioner.train()
    if "prompt" in diffusion.conditioner.conditioners:
        diffusion.conditioner.conditioners["prompt"].eval()
    return numerator / max(1, denominator)


def _save_bundle(
    path: Path,
    *,
    diffusion,
    ar: ScenePlanTransfusionEditingAR,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ar_specific = {
        key: value.detach().cpu()
        for key, value in ar.state_dict().items()
        if not key.startswith("editing_dit.")
        and not key.startswith("instruction_conditioner.")
    }
    payload = {
        "schema": "sceneplan_transfusion_editing_joint_bundle",
        "schema_version": 1,
        "metadata": metadata,
        "diffusion_state_dict": {
            key: value.detach().cpu() for key, value in diffusion.state_dict().items()
        },
        "editing_ar_specific_state_dict": ar_specific,
    }
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> int:
    args = _parse_args()
    if args.mode != "tiny":
        raise NotImplementedError(
            "full mode requires the frozen 1M index and distributed launcher"
        )
    if (
        args.expected_rows <= 0
        or args.batch_size != 1
        or args.max_steps <= 0
        or args.lambda_ar <= 0
        or args.lambda_rf <= 0
        or args.learning_rate <= 0
        or args.shared_learning_rate <= 0
        or args.dit_learning_rate <= 0
        or args.gradient_clip <= 0
    ):
        raise ValueError(
            "tiny joint Editing requires positive hyperparameters and batch_size=1 "
            "so every AR/RF replay uses one canonical 432/648 source envelope"
        )
    device = _editing_device(args.physical_gpu)
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")

    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    index = args.index.expanduser().resolve(strict=True)
    if sha256_file(index) != args.index_sha256:
        raise RuntimeError("joint Editing index SHA256 mismatch")

    model_config_path = args.model_config.expanduser().resolve(strict=True)
    dataset_config = load_config(args.dataset_config.expanduser().resolve(strict=True))
    model_config = load_config(model_config_path)
    diffusion = create_model_from_config(model_config)
    wrapper = create_training_wrapper_from_config(model_config, diffusion)
    state, _ = load_ckpt_state_dict(str(checkpoint), return_metadata=True)
    incompatible = wrapper.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Editing-DiT warm-up checkpoint mismatch: "
            f"missing={incompatible.missing_keys[:8]}, "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    del state
    _copy_ema_to_online(wrapper)
    diffusion = wrapper.diffusion
    wrapper.diffusion_ema = None
    wrapper.conditioner_ema = None
    diffusion.pretransform = None
    route_wrapper = diffusion.model
    route_core = route_wrapper.model
    prompt_conditioner = diffusion.conditioner.conditioners["prompt"]
    codec = ModelScenePlanCodecV4(args.codec.expanduser().resolve(strict=True))
    ar = ScenePlanTransfusionEditingAR(
        editing_dit=route_core,
        instruction_conditioner=prompt_conditioner,
        pad_id=codec.pad_id,
        activation_checkpointing=True,
    )
    if ar.shared_transformer is not route_core.transformer:
        raise RuntimeError("Editing AR and DiT do not share one block object")

    dit_dataset = ScenePlanTransfusionEditingDataset(
        index,
        tokenizer_spec=(prompt_conditioner.tokenizer, 512),
        expected_num_samples=int(args.expected_rows),
        index_num_samples=int(args.expected_rows),
        expected_index_sha256=args.index_sha256,
        latent_crop_length=648,
        require_frozen=True,
        verify_tensor_hashes_on_access=True,
    )
    joint_dataset = ScenePlanTransfusionEditingJointDataset(dit_dataset, codec=codec)
    joint_rows = [joint_dataset[index] for index in range(len(joint_dataset))]
    ar_rows = [ar_row for _, _, ar_row in joint_rows]
    dit_rows = [(target, metadata) for target, metadata, _ in joint_rows]
    if [row["pair_id"] for row in ar_rows] != [
        row[1]["pair_id"] for row in dit_rows
    ]:
        raise RuntimeError("AR and RF replay rows are not ordinal-aligned")
    if any(
        "old_sceneplan" in key.lower() for row in ar_rows for key in row
    ):
        raise RuntimeError("old ScenePlan leaked into Editing AR rows")
    for ar_row, (_, dit_metadata) in zip(ar_rows, dit_rows):
        ar_row["source_prefix_frames"] = int(
            dit_metadata["latent_bucket_frames"]
        )

    route_wrapper.to(device)
    diffusion.conditioner.to(device)
    ar.to(device)
    route_wrapper.train()
    diffusion.conditioner.train()
    ar.train()
    ar_specific = _unique_trainable_parameters(
        (ar.source_audio_adapter, ar.plan_adapter)
    ) + [ar.source_audio_type_embedding, ar.plan_type_embedding]
    ar_specific_ids = {id(parameter) for parameter in ar_specific}
    shared = _unique_trainable_parameters((route_core.transformer.layers,))
    shared_ids = {id(parameter) for parameter in shared}
    remaining = [
        parameter
        for parameter in _unique_trainable_parameters(
            (route_wrapper, diffusion.conditioner)
        )
        if id(parameter) not in ar_specific_ids and id(parameter) not in shared_ids
    ]
    parameters = [*ar_specific, *shared, *remaining]
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise RuntimeError("joint Editing optimizer contains duplicate parameters")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": ar_specific,
                "lr": float(args.learning_rate),
                "group_name": "editing_ar_adapters",
                "base_lr": float(args.learning_rate),
            },
            {
                "params": shared,
                "lr": float(args.shared_learning_rate),
                "group_name": "shared_transformer_blocks",
                "base_lr": float(args.shared_learning_rate),
            },
            {
                "params": remaining,
                "lr": float(args.dit_learning_rate),
                "group_name": "editing_dit_and_conditioners",
                "base_lr": float(args.dit_learning_rate),
            },
        ],
        betas=(0.9, 0.999),
        weight_decay=float(args.weight_decay),
        fused=True,
    )

    initial_ar = _teacher_forced_ar_metrics(
        ar, ar_rows, codec=codec, device=device
    )
    initial_rf = _fixed_rf_metric(
        diffusion, route_wrapper, dit_rows, device=device
    )
    initial_event = {
        "event": "initial",
        "step": 0,
        "ar": initial_ar,
        "fixed_rf_mse": initial_rf,
    }
    _append_jsonl(metrics_path, initial_event)
    print(json.dumps(initial_event, sort_keys=True), flush=True)

    order: list[int] = []
    cursor = 0
    accumulated_ar = 0.0
    accumulated_rf = 0.0
    accumulated_tokens = 0
    started = time.perf_counter()
    exact_streak = 0
    final_ar = initial_ar
    final_rf = initial_rf
    completed_step = 0
    for step in range(1, int(args.max_steps) + 1):
        if cursor >= len(order):
            generator = np.random.default_rng(seed + step // max(1, len(ar_rows)))
            order = generator.permutation(len(ar_rows)).tolist()
            cursor = 0
        indices = order[cursor : cursor + int(args.batch_size)]
        cursor += len(indices)
        if not indices:
            continue
        ar_batch = collate_editing_ar(
            [ar_rows[index] for index in indices], pad_id=codec.pad_id
        )
        source = ar_batch["source_foa_latent"].to(
            device=device, dtype=torch.float32
        )
        source_mask = ar_batch["source_attention_mask"].to(device)
        plan_ids = ar_batch["plan_input_ids"].to(device)
        plan_labels = ar_batch["plan_labels"].to(device)
        plan_mask = ar_batch["plan_attention_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        context, context_mask = ar.encode_edit_instructions(
            ar_batch["raw_edit_requests"], device=device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = ar(
                source,
                source_mask,
                plan_ids,
                plan_mask,
                context,
                context_mask,
            )
            ar_loss = F.cross_entropy(
                logits.flatten(0, 1), plan_labels.flatten(), ignore_index=-100
            )
            weighted_ar = float(args.lambda_ar) * ar_loss
        weighted_ar.backward()
        del logits, context, context_mask, weighted_ar

        target, metadata, rf_mask, rf_source = _prepare_dit_batch(
            [dit_rows[index] for index in indices], device
        )
        if tuple(rf_source.shape) != tuple(source.shape) or not torch.equal(
            rf_source, source
        ):
            raise RuntimeError("tiny joint AR/RF source tensors diverged")
        rf_source = source
        noise = torch.randn_like(target)
        t = torch.rand(target.shape[0], device=device)
        noised = (1.0 - t[:, None, None]) * target + t[:, None, None] * noise
        inputs = _conditioning_with_exact_source(
            diffusion, metadata, rf_source, device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = route_wrapper(
                noised,
                t,
                **inputs,
                cfg_dropout_prob=0.0,
                padding_mask=rf_mask,
            )
            squared = (prediction.float() - (noise - target)) ** 2
            rf_loss = (squared * rf_mask[:, None]).sum() / (
                rf_mask.sum().clamp_min(1) * target.shape[1]
            )
            weighted_rf = float(args.lambda_rf) * rf_loss
        weighted_rf.backward()
        ar_grad_norm = torch.nn.utils.clip_grad_norm_(
            ar_specific, float(args.gradient_clip)
        )
        shared_grad_norm = torch.nn.utils.clip_grad_norm_(
            shared, float(args.gradient_clip)
        )
        dit_grad_norm = torch.nn.utils.clip_grad_norm_(
            remaining, float(args.gradient_clip)
        )
        grad_norms = (ar_grad_norm, shared_grad_norm, dit_grad_norm)
        if not all(bool(torch.isfinite(value)) for value in grad_norms):
            raise RuntimeError(f"non-finite joint gradient at step {step}")
        warmup_scale = min(1.0, step / max(1, int(args.warmup_steps)))
        for group in optimizer.param_groups:
            group["lr"] = float(group["base_lr"]) * warmup_scale
        optimizer.step()
        completed_step = step

        valid_tokens = int((plan_labels != -100).sum().item())
        accumulated_ar += float(ar_loss.detach()) * valid_tokens
        accumulated_rf += float(rf_loss.detach()) * len(indices)
        accumulated_tokens += valid_tokens
        if step % int(args.log_every) == 0:
            event = {
                "event": "train",
                "step": step,
                "ar_ce": accumulated_ar / max(1, accumulated_tokens),
                "rf_mse": accumulated_rf / int(args.log_every),
                "tokens": accumulated_tokens,
                "ar_adapter_gradient_norm": float(ar_grad_norm.detach()),
                "shared_block_gradient_norm": float(shared_grad_norm.detach()),
                "dit_gradient_norm": float(dit_grad_norm.detach()),
                "learning_rates": {
                    str(group["group_name"]): float(group["lr"])
                    for group in optimizer.param_groups
                },
                "elapsed_sec": round(time.perf_counter() - started, 3),
            }
            _append_jsonl(metrics_path, event)
            print(json.dumps(event, sort_keys=True), flush=True)
            accumulated_ar = 0.0
            accumulated_rf = 0.0
            accumulated_tokens = 0

        if step % int(args.evaluate_every) == 0:
            final_ar = _teacher_forced_ar_metrics(
                ar, ar_rows, codec=codec, device=device
            )
            final_rf = _fixed_rf_metric(
                diffusion, route_wrapper, dit_rows, device=device
            )
            event = {
                "event": "evaluation",
                "step": step,
                "ar": final_ar,
                "fixed_rf_mse": final_rf,
                "fixed_rf_ratio_to_initial": final_rf / max(initial_rf, 1e-12),
            }
            _append_jsonl(metrics_path, event)
            print(json.dumps(event, sort_keys=True), flush=True)
            if (
                final_ar["teacher_forced_sequence_exact"] == 1.0
                and final_ar["zero_reference_ce"] > final_ar["clean_ce"]
                and final_ar["shuffled_reference_ce"] > final_ar["clean_ce"]
                and final_rf <= initial_rf * 1.25
            ):
                exact_streak += 1
            else:
                exact_streak = 0
            if exact_streak >= 2:
                break

    passed = bool(
        final_ar["teacher_forced_sequence_exact"] == 1.0
        and final_ar["zero_reference_ce"] > final_ar["clean_ce"]
        and final_ar["shuffled_reference_ce"] > final_ar["clean_ce"]
        and final_rf <= initial_rf * 1.25
    )
    summary = {
        "schema": "sceneplan_transfusion_editing_ar_joint_tiny_training",
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "contract": EDITING_AR_CONTRACT,
        "latest_route": {
            "editing_ar_inputs": ["source_foa_latent", "raw_edit_request"],
            "editing_ar_target": "complete_new_sceneplan",
            "old_sceneplan_input": False,
            "editing_dit_frame_input": [
                "noisy_target_64",
                "new_sceneplan_256",
                "source_foa_latent_64",
            ],
        },
        "physical_gpu": int(args.physical_gpu),
        "seed": seed,
        "steps": completed_step,
        "base_checkpoint": str(checkpoint),
        "base_checkpoint_sha256": sha256_file(checkpoint),
        "model_config": str(model_config_path),
        "model_config_sha256": sha256_file(model_config_path),
        "index": str(index),
        "index_sha256": sha256_file(index),
        "rows": len(ar_rows),
        "shared_transformer_same_object": ar.shared_transformer
        is route_core.transformer,
        "old_sceneplan_exposed": False,
        "canonical_source_prefix": "latent_bucket_432_or_648",
        "initial_ar": initial_ar,
        "final_ar": final_ar,
        "initial_fixed_rf_mse": initial_rf,
        "final_fixed_rf_mse": final_rf,
        "fixed_rf_ratio_to_initial": final_rf / max(initial_rf, 1e-12),
    }
    _atomic_json(run_dir / "summary.json", summary)
    _save_bundle(
        run_dir / "checkpoints" / f"step-{completed_step}.pt",
        diffusion=diffusion,
        ar=ar,
        metadata=summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise RuntimeError("joint Editing AR tiny gate did not pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
