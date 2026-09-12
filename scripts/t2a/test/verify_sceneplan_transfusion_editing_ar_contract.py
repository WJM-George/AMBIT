#!/usr/bin/env python3
"""Verify the latest audio-reference Editing-AR contract on a real pair."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.data.model_sceneplan_codec_v4 import (  # noqa: E402
    ModelScenePlanCodecV4,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_ar_dataset import (  # noqa: E402
    EDITING_AR_SELECT_COLUMNS,
    ScenePlanTransfusionEditingARDataset,
    collate_editing_ar,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_index import (  # noqa: E402
    sha256_file,
)
from stable_audio_tools.models.factory import create_model_from_config  # noqa: E402
from stable_audio_tools.models.sceneplan_transfusion_editing_ar import (  # noqa: E402
    EDITING_AR_CONTRACT,
    ScenePlanTransfusionEditingAR,
    editing_ar_prefix_attention_allowed,
)
from stable_audio_tools.models.utils import load_ckpt_state_dict  # noqa: E402
from stable_audio_tools.training.factory import (  # noqa: E402
    create_training_wrapper_from_config,
)


DEFAULT_MODEL_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m_sceneplan_transfusion_editing_dit_v1.json"
)
DEFAULT_INDEX = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_transfusion_editing_v1_pilot/"
    "training_index/train.sqlite"
)
DEFAULT_INDEX_SHA256 = (
    "6535b4a04b38a07ff641a876361e0dd16d4d4d4737ad27c11304d1d99c785600"
)
DEFAULT_CODEC = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)
DEFAULT_CHECKPOINT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives//transfusion_editing/pilots/"
    "sceneplan_transfusion_editing_dit_overfit10_seed42_v1/checkpoints/"
    "epoch=499-step=1000.ckpt"
)
DEFAULT_OUTPUT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_transfusion_editing_v1_pilot/audits/"
    "editing_ar_audio_reference_contract_gate.json"
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
        raise RuntimeError("gate requires unremapped physical CUDA indices")
    if index not in range(3, 8):
        raise ValueError("Editing gates are restricted to physical GPUs 3--7")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= index:
        raise RuntimeError(f"cuda:{index} is unavailable")
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--index-sha256", default=DEFAULT_INDEX_SHA256)
    parser.add_argument("--expected-num-samples", type=int, default=10)
    parser.add_argument("--codec", type=Path, default=DEFAULT_CODEC)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if int(args.expected_num_samples) <= 0:
        raise ValueError("--expected-num-samples must be positive")

    device = _physical_gpu(args.gpu)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    model_config = load_config(args.model_config.expanduser().resolve(strict=True))
    diffusion = create_model_from_config(model_config)
    wrapper = create_training_wrapper_from_config(model_config, diffusion)
    state, _ = load_ckpt_state_dict(str(checkpoint), return_metadata=True)
    incompatible = wrapper.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Editing checkpoint/model mismatch: "
            f"missing={incompatible.missing_keys[:8]}, "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    del state
    diffusion = wrapper.diffusion
    diffusion.pretransform = None
    route_wrapper = diffusion.model
    route = route_wrapper.model
    conditioner = diffusion.conditioner.conditioners["prompt"]
    codec = ModelScenePlanCodecV4(args.codec.expanduser().resolve(strict=True))
    dataset = ScenePlanTransfusionEditingARDataset(
        args.index,
        codec=codec,
        expected_num_samples=int(args.expected_num_samples),
        expected_index_sha256=args.index_sha256,
        verify_tensor_hashes_on_access=True,
    )
    row = dataset[0]
    batch = collate_editing_ar([row], pad_id=codec.pad_id)
    if any("old_sceneplan" in key.lower() for key in row):
        raise RuntimeError("Editing AR row exposed an old ScenePlan")
    forward_parameters = inspect.signature(
        ScenePlanTransfusionEditingAR.forward
    ).parameters
    if "old_sceneplan" in forward_parameters or "source_sceneplan" in forward_parameters:
        raise RuntimeError("Editing AR forward exposes an old/source ScenePlan")

    ar = ScenePlanTransfusionEditingAR(
        editing_dit=route,
        instruction_conditioner=conditioner,
        pad_id=codec.pad_id,
        activation_checkpointing=False,
    )
    if ar.shared_transformer is not route.transformer:
        raise RuntimeError("Editing AR copied instead of sharing Transformer blocks")
    adapter_initialization_equal = torch.equal(
        ar.source_audio_adapter.weight.detach().cpu(),
        route.transformer.project_in.weight[:, :64].detach().cpu(),
    )
    if not adapter_initialization_equal:
        raise RuntimeError("Editing AR source adapter lost P10 audio initialization")

    ar = ar.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)
    source = batch["source_foa_latent"].to(device=device, dtype=torch.bfloat16)
    source_mask = batch["source_attention_mask"].to(device)
    source_prefix_frames = int(batch["source_prefix_frames"][0].item())
    if (
        source_prefix_frames not in (432, 648)
        or int(source.shape[-1]) != source_prefix_frames
    ):
        raise RuntimeError("Editing AR did not retain its canonical bucket prefix")
    plan_ids = batch["plan_input_ids"].to(device)
    plan_mask = batch["plan_attention_mask"].to(device)
    context, context_mask = ar.encode_edit_instructions(
        batch["raw_edit_requests"], device=device
    )
    changed_plan = plan_ids.clone()
    changed_plan[:, -1] = (changed_plan[:, -1] + 1) % ar.vocab_size
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = ar(source, source_mask, plan_ids, plan_mask, context, context_mask)
        changed_logits = ar(
            source,
            source_mask,
            changed_plan,
            plan_mask,
            context,
            context_mask,
        )
        zero_logits = ar(
            torch.zeros_like(source),
            source_mask,
            plan_ids,
            plan_mask,
            context,
            context_mask,
        )
    if tuple(logits.shape) != (1, int(plan_ids.shape[1]), 4096):
        raise RuntimeError(f"unexpected Editing AR logits shape: {tuple(logits.shape)}")
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Editing AR produced non-finite logits")
    future_prefix_max_abs = float(
        (logits[:, :-1].float() - changed_logits[:, :-1].float())
        .abs()
        .max()
        .item()
    )
    changed_position_l1 = float(
        (logits[:, -1].float() - changed_logits[:, -1].float()).abs().mean().item()
    )
    source_zero_l1 = float(
        (logits.float() - zero_logits.float()).abs().mean().item()
    )
    if future_prefix_max_abs != 0.0 or changed_position_l1 <= 0.0:
        raise RuntimeError("Editing AR causal teacher-token isolation failed")
    if source_zero_l1 <= 0.0:
        raise RuntimeError("Editing AR logits ignore the reference latent")

    source_frames = int(source.shape[-1])
    plan_tokens = int(plan_ids.shape[-1])
    allowed = editing_ar_prefix_attention_allowed(source_frames, plan_tokens)
    result = {
        "schema": "sceneplan_transfusion_editing_ar_audio_reference_gate",
        "schema_version": 1,
        "status": "PASS",
        "contract": EDITING_AR_CONTRACT,
        "verifier": str(Path(__file__).resolve()),
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "physical_gpu": int(args.gpu),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "index": str(args.index.expanduser().resolve(strict=True)),
        "index_sha256": sha256_file(args.index.expanduser().resolve(strict=True)),
        "expected_index_rows": int(args.expected_num_samples),
        "observed_index_rows": len(dataset),
        "pair_id": row["pair_id"],
        "ar_row_keys": sorted(row),
        "ar_sql_select_columns": list(EDITING_AR_SELECT_COLUMNS),
        "old_sceneplan_exposed": False,
        "forward_parameters": list(forward_parameters),
        "shared_transformer_same_object": True,
        "shared_layer_count": len(ar.shared_transformer.layers),
        "source_adapter_copied_from_audio_columns": adapter_initialization_equal,
        "source_shape": list(source.shape),
        "canonical_source_prefix_frames": source_prefix_frames,
        "plan_input_shape": list(plan_ids.shape),
        "logits_shape": list(logits.shape),
        "mask": {
            "source_to_plan_allowed_count": int(
                allowed[:source_frames, source_frames:].sum().item()
            ),
            "plan_to_source_allowed_count": int(
                allowed[source_frames:, :source_frames].sum().item()
            ),
            "future_plan_allowed_count": int(
                allowed[source_frames:, source_frames:]
                .triu(diagonal=1)
                .sum()
                .item()
            ),
        },
        "interventions": {
            "changed_future_token_prefix_logits_max_abs": future_prefix_max_abs,
            "changed_token_position_logits_l1": changed_position_l1,
            "clean_vs_zero_reference_logits_l1": source_zero_l1,
        },
    }
    if (
        result["mask"]["source_to_plan_allowed_count"] != 0
        or result["mask"]["future_plan_allowed_count"] != 0
        or result["mask"]["plan_to_source_allowed_count"]
        != source_frames * plan_tokens
    ):
        raise RuntimeError("Editing AR attention mask truth table changed")
    output = args.output.expanduser().resolve()
    _atomic_json(output, result)
    print(json.dumps({**result, "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
