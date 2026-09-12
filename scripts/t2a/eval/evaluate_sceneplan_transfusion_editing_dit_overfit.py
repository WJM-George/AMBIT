#!/usr/bin/env python3
"""Measure tiny-pair RF fit and clean-source dependence for Editing DiT."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import (  # noqa: E402
    load_config,
    validate_training_configs,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import (  # noqa: E402
    ScenePlanTransfusionEditingDataset,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_index import (  # noqa: E402
    sha256_file,
)
from stable_audio_tools.models.factory import create_model_from_config  # noqa: E402
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
CANONICAL_P10 = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/dit/"
    "sceneplan_dit_v11_semantic_v2_protected_resume_150k/"
    "checkpoints/epoch=48-step=150000.ckpt"
)
CANONICAL_P10_SHA256 = (
    "be8c90cd1434bd71f73951531175c3674ff0f3173d5db591e2e1c476152ff59e"
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _physical_gpu(index: int) -> torch.device:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise RuntimeError("evaluation requires unremapped physical CUDA indices")
    if index not in range(3, 8):
        raise ValueError("Editing evaluation is restricted to physical GPUs 3--7")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= index:
        raise RuntimeError(f"cuda:{index} is unavailable")
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def _masked_per_sample(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numerator = (value * mask[:, None]).sum(dim=(1, 2))
    denominator = mask.sum(dim=1).clamp_min(1) * value.shape[1]
    return numerator / denominator


def _variant_conditioning(
    positive: dict[str, Any], source: torch.Tensor
) -> dict[str, Any]:
    output = dict(positive)
    output["source_foa_latent"] = [source, None]
    return output


def _evaluate_variant(
    route,
    diffusion,
    conditioning,
    target: torch.Tensor,
    mask: torch.Tensor,
    noise: torch.Tensor,
    timesteps: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    losses = []
    predictions = []
    model_inputs = diffusion.get_conditioning_inputs(conditioning)
    for timestep in timesteps:
        t = torch.full(
            (target.shape[0],), float(timestep), device=target.device
        )
        noised = (1.0 - t[:, None, None]) * target + t[:, None, None] * noise
        velocity_target = noise - target
        prediction = route(
            noised,
            t,
            **model_inputs,
            cfg_dropout_prob=0.0,
            padding_mask=mask,
        )
        losses.append(
            _masked_per_sample((prediction.float() - velocity_target) ** 2, mask)
        )
        predictions.append(prediction.float())
    return torch.stack(losses), torch.stack(predictions)


def _ode_sample(
    route,
    diffusion,
    conditioning,
    initial_noise: torch.Tensor,
    mask: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    value = initial_noise.clone()
    model_inputs = diffusion.get_conditioning_inputs(conditioning)
    delta = 1.0 / int(steps)
    for index in range(int(steps)):
        timestep = 1.0 - index * delta
        t = torch.full(
            (value.shape[0],), timestep, device=value.device
        )
        velocity = route(
            value,
            t,
            **model_inputs,
            cfg_dropout_prob=0.0,
            padding_mask=mask,
        )
        value = value - delta * velocity.float()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-kind",
        choices=("editing_training", "canonical_p10_route"),
        default="editing_training",
    )
    parser.add_argument("--weights", choices=("online", "ema"), default="online")
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--ode-steps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.ode_steps <= 0:
        raise ValueError("ode-steps must be positive")
    device = _physical_gpu(args.gpu)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)

    model_config = load_config(args.model_config.expanduser().resolve(strict=True))
    dataset_config = load_config(
        args.dataset_config.expanduser().resolve(strict=True)
    )
    validate_training_configs(model_config, dataset_config)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    checkpoint_sha = sha256_file(checkpoint)
    model = create_model_from_config(model_config)
    checkpoint_state, checkpoint_metadata = load_ckpt_state_dict(
        str(checkpoint), return_metadata=True
    )
    wrapper = None
    ema_context = nullcontext()
    if args.checkpoint_kind == "canonical_p10_route":
        if checkpoint != CANONICAL_P10.resolve() or checkpoint_sha != CANONICAL_P10_SHA256:
            raise RuntimeError("canonical route evaluation requires pinned P10-v11")
        if args.weights != "online":
            raise ValueError("canonical route is copied from P10 EMA into online weights")
        report = model.load_pretrained_route_state_dict(
            checkpoint_state,
            prefer_ema=True,
            source_model_config=checkpoint_metadata.get("model_config"),
            source_conditioner_ema_names=checkpoint_metadata.get(
                "conditioner_ema_parameter_names"
            ),
        )
        if report["missing"] or len(report.get("partial_expansions", [])) != 2:
            raise RuntimeError("canonical P10 Editing warm-start changed")
        diffusion = model
        route = model.model
    else:
        wrapper = create_training_wrapper_from_config(model_config, model)
        incompatible = wrapper.load_state_dict(checkpoint_state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Editing checkpoint/model mismatch: "
                f"missing={incompatible.missing_keys[:12]}, "
                f"unexpected={incompatible.unexpected_keys[:12]}"
            )
        diffusion = wrapper.diffusion
        if args.weights == "ema":
            if wrapper.diffusion_ema is None:
                raise RuntimeError("Editing checkpoint has no DiT EMA")
            route = wrapper.diffusion_ema.ema_model
            ema_context = wrapper.ema_conditioner_context()
        else:
            route = diffusion.model
    del checkpoint_state, checkpoint_metadata

    tokenizer = diffusion.conditioner.conditioners["prompt"].tokenizer
    dataset_entry = dataset_config["datasets"][0]
    dataset = ScenePlanTransfusionEditingDataset(
        dataset_entry["path"],
        tokenizer_spec=(tokenizer, 512),
        expected_num_samples=int(dataset_config["expected_num_samples"]),
        index_num_samples=int(dataset_config["index_num_samples"]),
        expected_index_sha256=dataset_config.get("index_sha256"),
        latent_crop_length=648,
        require_frozen=True,
        verify_tensor_hashes_on_access=True,
    )
    loaded = [dataset[index] for index in range(len(dataset))]
    targets = torch.stack([value[0] for value in loaded]).to(
        device=device, dtype=torch.float32
    )
    metadata = [value[1] for value in loaded]
    masks = torch.stack([row["padding_mask"][0] for row in metadata]).to(device)
    sources = torch.stack([row["source_foa_latent"] for row in metadata]).to(device)
    permutation = torch.tensor(
        [1, 0, 3, 2, 5, 4, 7, 6, 9, 8], device=device
    )
    if len(dataset) != len(permutation):
        raise RuntimeError("overfit source intervention expects exactly ten rows")
    zero_sources = torch.zeros_like(sources)
    shuffled_sources = sources[permutation]
    generator = torch.Generator(device=device).manual_seed(420042)
    noise = torch.randn(
        targets.shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    timesteps = (0.1, 0.3, 0.5, 0.7, 0.9)

    diffusion.pretransform = None
    route = route.eval().requires_grad_(False).to(device)
    diffusion.conditioner = diffusion.conditioner.eval().requires_grad_(False).to(device)
    with ema_context, torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        positive = diffusion.conditioner(metadata, device)
        clean_conditioning = _variant_conditioning(positive, sources)
        zero_conditioning = _variant_conditioning(positive, zero_sources)
        shuffled_conditioning = _variant_conditioning(positive, shuffled_sources)
        clean_losses, clean_predictions = _evaluate_variant(
            route,
            diffusion,
            clean_conditioning,
            targets,
            masks,
            noise,
            timesteps,
        )
        zero_losses, zero_predictions = _evaluate_variant(
            route,
            diffusion,
            zero_conditioning,
            targets,
            masks,
            noise,
            timesteps,
        )
        shuffle_losses, shuffle_predictions = _evaluate_variant(
            route,
            diffusion,
            shuffled_conditioning,
            targets,
            masks,
            noise,
            timesteps,
        )
        clean_sample = _ode_sample(
            route, diffusion, clean_conditioning, noise, masks, args.ode_steps
        )
        zero_sample = _ode_sample(
            route, diffusion, zero_conditioning, noise, masks, args.ode_steps
        )
        shuffle_sample = _ode_sample(
            route, diffusion, shuffled_conditioning, noise, masks, args.ode_steps
        )

    clean_mse = float(clean_losses.mean().item())
    zero_mse = float(zero_losses.mean().item())
    shuffle_mse = float(shuffle_losses.mean().item())
    clean_ode = _masked_per_sample((clean_sample - targets) ** 2, masks)
    zero_ode = _masked_per_sample((zero_sample - targets) ** 2, masks)
    shuffle_ode = _masked_per_sample((shuffle_sample - targets) ** 2, masks)
    zero_sensitivity = _masked_per_sample(
        (clean_predictions - zero_predictions).abs().mean(dim=0), masks
    )
    shuffle_sensitivity = _masked_per_sample(
        (clean_predictions - shuffle_predictions).abs().mean(dim=0), masks
    )
    per_operation: dict[str, dict[str, float]] = {}
    operation_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        operation_indices[str(row["operation"])].append(index)
    for operation, indices in sorted(operation_indices.items()):
        selected = torch.tensor(indices, device=device)
        per_operation[operation] = {
            "clean_rf_mse": float(clean_losses[:, selected].mean().item()),
            "zero_source_rf_mse": float(zero_losses[:, selected].mean().item()),
            "shuffled_source_rf_mse": float(
                shuffle_losses[:, selected].mean().item()
            ),
        }
    match = re.search(r"step=(\d+)", checkpoint.name)
    result = {
        "schema": "sceneplan_transfusion_editing_dit_overfit_evaluation",
        "schema_version": 1,
        "status": "PASS",
        "checkpoint_kind": args.checkpoint_kind,
        "weights": args.weights,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": int(match.group(1)) if match else None,
        "physical_gpu": args.gpu,
        "rows": len(dataset),
        "timesteps": list(timesteps),
        "ode_steps": int(args.ode_steps),
        "metrics": {
            "clean_rf_mse": clean_mse,
            "zero_source_rf_mse": zero_mse,
            "shuffled_source_rf_mse": shuffle_mse,
            "zero_source_rf_mse_ratio": zero_mse / max(clean_mse, 1e-12),
            "shuffled_source_rf_mse_ratio": shuffle_mse
            / max(clean_mse, 1e-12),
            "clean_zero_prediction_l1": float(zero_sensitivity.mean().item()),
            "clean_shuffle_prediction_l1": float(
                shuffle_sensitivity.mean().item()
            ),
            "clean_ode_latent_mse": float(clean_ode.mean().item()),
            "zero_source_ode_latent_mse": float(zero_ode.mean().item()),
            "shuffled_source_ode_latent_mse": float(shuffle_ode.mean().item()),
            "zero_source_ode_mse_ratio": float(
                zero_ode.mean().item() / max(clean_ode.mean().item(), 1e-12)
            ),
            "shuffled_source_ode_mse_ratio": float(
                shuffle_ode.mean().item()
                / max(clean_ode.mean().item(), 1e-12)
            ),
        },
        "per_operation": per_operation,
        "pair_ids": [row["pair_id"] for row in metadata],
        "source_permutation": permutation.cpu().tolist(),
    }
    output = args.output.expanduser().resolve()
    _atomic_json(output, result)
    print(json.dumps({**result, "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
