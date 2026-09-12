#!/usr/bin/env python3
"""Prove exact P10-v11 inheritance and one real Editing-DiT forward pass."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import (  # noqa: E402
    load_config,
    validate_training_configs,
)
from stable_audio_tools.data.sceneplan_bucket_sampler import (  # noqa: E402
    sceneplan_bucket_collation,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import (  # noqa: E402
    ScenePlanTransfusionEditingDataset,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_index import (  # noqa: E402
    sha256_file,
)
from stable_audio_tools.models.factory import create_model_from_config  # noqa: E402
from stable_audio_tools.models.utils import load_ckpt_state_dict  # noqa: E402


DEFAULT_MODEL_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m_sceneplan_transfusion_editing_dit_v1.json"
)
DEFAULT_DATASET_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/dataset_configs/"
    "sceneplan_transfusion_editing_v1_pilot_validation.json"
)
EXPECTED_CHECKPOINT_SHA256 = (
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


def _require_physical_gpu(index: int) -> torch.device:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise RuntimeError(
            "physical-GPU audit forbids CUDA_VISIBLE_DEVICES remapping"
        )
    if int(index) not in range(3, 8):
        raise ValueError("Editing work is restricted to physical GPUs 3--7")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= int(index):
        raise RuntimeError(f"physical cuda:{index} is unavailable")
    torch.cuda.set_device(int(index))
    return torch.device(f"cuda:{index}")


def _assert_equal(left: torch.Tensor, right: torch.Tensor, label: str) -> None:
    if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(
        left, right
    ):
        raise RuntimeError(f"P10 inheritance differs at {label}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/mnt/sdc/ckpts/dit/"
            "sceneplan_dit_v11_semantic_v2_protected_resume_150k/"
            "checkpoints/epoch=48-step=150000.ckpt"
        ),
    )
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/mnt/sdb/audio_dataset/sceneplan_transfusion_editing_v1_pilot/"
            "audits/p10_v11_warmstart_forward_gate.json"
        ),
    )
    args = parser.parse_args()
    device = _require_physical_gpu(args.gpu)
    torch.set_float32_matmul_precision("high")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("canonical P10-v11 checkpoint SHA256 changed")
    model_config = load_config(args.model_config.expanduser().resolve(strict=True))
    dataset_config = load_config(args.dataset_config.expanduser().resolve(strict=True))
    validate_training_configs(model_config, dataset_config)
    if (
        model_config.get("_source_checkpoint_sha256")
        != EXPECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError("Editing model config does not pin canonical P10-v11")

    model = create_model_from_config(model_config)
    source_state, source_metadata = load_ckpt_state_dict(
        str(checkpoint), return_metadata=True
    )
    source_config = source_metadata.get("model_config")
    if not isinstance(source_config, dict):
        raise RuntimeError("P10 checkpoint has no embedded model config")
    warmstart = model.load_pretrained_route_state_dict(
        source_state,
        prefer_ema=True,
        source_model_config=source_config,
        source_conditioner_ema_names=source_metadata.get(
            "conditioner_ema_parameter_names"
        ),
    )
    if warmstart["modality_mapping"] != (
        "exact_name_and_shape_plus_trained_prefix_input_expansion"
    ):
        raise RuntimeError("P10 warm-start did not use trained-prefix expansion")
    if warmstart["missing"]:
        raise RuntimeError(
            f"Editing warm-start left parameters missing: {warmstart['missing'][:12]}"
        )

    target_state = model.state_dict()
    layer_prefix = "model.model.transformer.layers."
    block_keys = sorted(key for key in target_state if key.startswith(layer_prefix))
    layer_ids = sorted({int(key[len(layer_prefix) :].split(".", 1)[0]) for key in block_keys})
    if layer_ids != list(range(15)) or len(block_keys) != 225:
        raise RuntimeError(
            f"Editing block inventory changed: layers={layer_ids}, tensors={len(block_keys)}"
        )
    for target_key in block_keys:
        source_key = "diffusion_ema.ema_model." + target_key[len("model.") :]
        source_value = source_state.get(source_key)
        if not torch.is_tensor(source_value):
            raise RuntimeError(f"P10 EMA block tensor absent: {source_key}")
        _assert_equal(target_state[target_key], source_value, target_key)

    project_key = "model.model.transformer.project_in.weight"
    source_project_key = (
        "diffusion_ema.ema_model.model.transformer.project_in.weight"
    )
    project = target_state[project_key]
    source_project = source_state[source_project_key]
    if tuple(project.shape) != (1024, 384) or tuple(source_project.shape) != (
        1024,
        320,
    ):
        raise RuntimeError("P10/Editing project_in geometry changed")
    _assert_equal(project[:, :320], source_project, "project_in trained prefix")
    if torch.count_nonzero(project[:, 320:]).item() != 0:
        raise RuntimeError("new 64-column source project_in suffix is not zero")

    preprocess_key = "model.model.preprocess_conv.weight"
    source_preprocess_key = "diffusion_ema.ema_model.model.preprocess_conv.weight"
    preprocess = target_state[preprocess_key]
    source_preprocess = source_state[source_preprocess_key]
    if tuple(preprocess.shape) != (384, 384, 1) or tuple(
        source_preprocess.shape
    ) != (320, 320, 1):
        raise RuntimeError("P10/Editing preprocess geometry changed")
    _assert_equal(
        preprocess[:320, :320], source_preprocess, "preprocess trained block"
    )
    outside = preprocess.clone()
    outside[:320, :320].zero_()
    if torch.count_nonzero(outside).item() != 0:
        raise RuntimeError("new preprocess source rows/columns are not zero")

    dataset_entry = dataset_config["datasets"][0]
    tokenizer = model.conditioner.conditioners["prompt"].tokenizer
    dataset = ScenePlanTransfusionEditingDataset(
        dataset_entry["path"],
        tokenizer_spec=(tokenizer, 512),
        expected_num_samples=1,
        index_num_samples=int(dataset_config["index_num_samples"]),
        expected_index_sha256=dataset_config.get("index_sha256"),
        sample_ordinals=[2],
        latent_crop_length=648,
        require_frozen=True,
        verify_tensor_hashes_on_access=True,
    )
    target, metadata = dataset[0]
    target_batch, metadata_batch = sceneplan_bucket_collation([(target, metadata)])
    if tuple(target_batch.shape) != (1, 64, 432):
        raise RuntimeError("real Editing pilot batch did not trim to 432 frames")
    if tuple(metadata_batch[0]["source_foa_latent"].shape) != (64, 432):
        raise RuntimeError("source reference did not follow target bucket trimming")

    # The frozen VAE is not used for pre-encoded Editing training and would only
    # consume device memory in this forward graph gate.
    model.pretransform = None
    del target_state
    model = model.eval().requires_grad_(False).to(device)
    target_device = target_batch.to(device=device, dtype=torch.float32)
    padding_mask = torch.stack(
        [row["padding_mask"][0] for row in metadata_batch], dim=0
    ).to(device)
    captured: dict[str, tuple[int, ...]] = {}

    def capture_project_input(_module, values):
        captured["project_in"] = tuple(int(value) for value in values[0].shape)

    hook = model.model.model.transformer.project_in.register_forward_pre_hook(
        capture_project_input
    )
    try:
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            conditioning = model.conditioner(metadata_batch, device)
            ordered = model.get_conditioning_inputs(conditioning)[
                "input_concat_cond"
            ]
            if tuple(ordered.shape) != (1, 320, 432):
                raise RuntimeError("Editing condition is not [plan256,source64]")
            if not torch.equal(
                ordered[:, 256:].to(torch.float16),
                metadata_batch[0]["source_foa_latent"].unsqueeze(0).to(device),
            ):
                raise RuntimeError("clean source latent is not the 64-channel suffix")
            time = torch.full((1,), 0.5, device=device)
            noise = torch.randn_like(target_device)
            noised_target = 0.5 * target_device + 0.5 * noise
            output = model(
                noised_target,
                time,
                cond=conditioning,
                cfg_dropout_prob=0.0,
                padding_mask=padding_mask,
            )
    finally:
        hook.remove()
    if tuple(output.shape) != (1, 64, 432) or not torch.isfinite(output).all():
        raise RuntimeError("real Editing-DiT forward output is invalid")
    if captured.get("project_in") != (1, 432, 384):
        raise RuntimeError(
            f"transformer did not receive [B,T,384]: {captured.get('project_in')}"
        )

    result = {
        "schema": "sceneplan_transfusion_editing_p10_warmstart_gate",
        "schema_version": 1,
        "status": "PASS",
        "physical_gpu": int(args.gpu),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "model_config": str(args.model_config.expanduser().resolve()),
        "dataset_config": str(args.dataset_config.expanduser().resolve()),
        "warmstart": warmstart,
        "proof": {
            "shared_transformer_layers": layer_ids,
            "shared_transformer_tensor_count": len(block_keys),
            "shared_transformer_tensors_bitwise_equal": True,
            "p10_frame_input": [1, 432, 320],
            "editing_frame_input": list(captured["project_in"]),
            "project_in_prefix_columns_bitwise_equal": 320,
            "project_in_zero_source_suffix_columns": 64,
            "preprocess_prefix_block_bitwise_equal": [320, 320],
            "preprocess_new_rows_and_columns_zero": True,
            "condition_order": ["new_sceneplan_control_256", "source_foa_latent_64"],
            "rf_state": "z_t=(1-t)*target+t*noise",
            "rf_velocity_target": "noise-target",
            "real_forward_output_shape": list(output.shape),
            "real_forward_finite": True,
        },
    }
    output_path = args.output.expanduser().resolve()
    _atomic_json(output_path, result)
    print(json.dumps({**result, "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
