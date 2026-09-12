#!/usr/bin/env python3
"""Two-rank real-pair forward/backward gate for joint Editing AR + RF."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.train.train_sceneplan_transfusion_editing_ar_joint_full import (  # noqa: E402
    JointEditingModule,
    _copy_ema_to_online,
    _losses,
    _move_joint_batch,
)
from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4  # noqa: E402
from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import (  # noqa: E402
    ScenePlanTransfusionEditingDataset,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_index import sha256_file  # noqa: E402
from stable_audio_tools.data.sceneplan_transfusion_editing_joint_dataset import (  # noqa: E402
    ScenePlanTransfusionEditingJointDataset,
    collate_editing_joint,
)
from stable_audio_tools.models.factory import create_model_from_config  # noqa: E402
from stable_audio_tools.models.sceneplan_transfusion_editing_ar import (  # noqa: E402
    EDITING_AR_CONTRACT,
    ScenePlanTransfusionEditingAR,
)
from stable_audio_tools.models.utils import load_ckpt_state_dict  # noqa: E402
from stable_audio_tools.training.factory import create_training_wrapper_from_config  # noqa: E402


MODEL_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m_sceneplan_transfusion_editing_dit_v1.json"
)
INDEX = Path(
    "/mnt/sdb/audio_dataset/sceneplan_transfusion_editing_v1_pilot/"
    "training_index/train.sqlite"
)
INDEX_SHA256 = "6535b4a04b38a07ff641a876361e0dd16d4d4d4737ad27c11304d1d99c785600"
CHECKPOINT = Path(
    "/mnt/sdb/model_archives/transfusion_editing/pilots/"
    "sceneplan_transfusion_editing_dit_overfit10_seed42_v1/checkpoints/"
    "epoch=499-step=1000.ckpt"
)
CODEC = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)


def _gradient_summary(parameters: list[torch.nn.Parameter]) -> dict[str, float]:
    values = [
        parameter.grad.detach().float().norm()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not values:
        return {"tensors": 0.0, "norm_sum": 0.0, "finite": 0.0}
    return {
        "tensors": float(len(values)),
        "norm_sum": float(torch.stack(values).sum()),
        "finite": float(all(bool(torch.isfinite(value)) for value in values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").replace(" ", "") != "3,4":
        raise RuntimeError("joint DDP gate is restricted to physical GPUs 3,4")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError("joint DDP gate requires exactly two ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_float32_matmul_precision("high")

    config = load_config(MODEL_CONFIG)
    diffusion = create_model_from_config(config)
    wrapper = create_training_wrapper_from_config(config, diffusion)
    state, _ = load_ckpt_state_dict(str(CHECKPOINT), return_metadata=True)
    incompatible = wrapper.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("joint DDP gate checkpoint mismatch")
    del state
    _copy_ema_to_online(wrapper)
    diffusion = wrapper.diffusion
    wrapper.diffusion_ema = None
    wrapper.conditioner_ema = None
    diffusion.pretransform = None
    route_core = diffusion.model.model
    prompt = diffusion.conditioner.conditioners["prompt"]
    codec = ModelScenePlanCodecV4(CODEC)
    ar = ScenePlanTransfusionEditingAR(
        editing_dit=route_core,
        instruction_conditioner=prompt,
        pad_id=codec.pad_id,
        activation_checkpointing=True,
    )
    module = JointEditingModule(diffusion=diffusion, ar=ar).to(device).train()

    base = ScenePlanTransfusionEditingDataset(
        INDEX,
        tokenizer_spec=(prompt.tokenizer, 512),
        expected_num_samples=10,
        expected_index_sha256=INDEX_SHA256,
        latent_crop_length=648,
        require_frozen=True,
        verify_tensor_hashes_on_access=True,
    )
    dataset = ScenePlanTransfusionEditingJointDataset(base, codec=codec)
    candidates = next(
        list(indices)
        for _, indices in sorted(dataset.length_bucket_indices().items())
        if len(indices) >= world_size
    )
    sample = dataset[candidates[rank]]
    raw_batch = collate_editing_joint([sample], pad_id=codec.pad_id)
    ar_batch, target, metadata, rf_mask = _move_joint_batch(raw_batch, device)
    if any("old_sceneplan" in key.lower() for key in metadata[0]):
        raise RuntimeError("joint DDP gate exposed old ScenePlan")

    ar_specific = list(ar.source_audio_adapter.parameters()) + list(
        ar.plan_adapter.parameters()
    ) + [ar.source_audio_type_embedding, ar.plan_type_embedding]
    ar_ids = {id(parameter) for parameter in ar_specific}
    shared = list(route_core.transformer.layers.parameters())
    shared_ids = {id(parameter) for parameter in shared}
    remaining = [
        parameter
        for parameter in diffusion.parameters()
        if parameter.requires_grad
        and id(parameter) not in ar_ids
        and id(parameter) not in shared_ids
    ]
    parameters = [*ar_specific, *shared, *remaining]
    optimizer = torch.optim.AdamW(parameters, lr=1e-6, fused=True)
    wrapped = DistributedDataParallel(
        module,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    generator = torch.Generator(device=device).manual_seed(42_000 + rank)
    noise = torch.randn(target.shape, generator=generator, device=device)
    timesteps = torch.full((target.shape[0],), 0.5, device=device)
    noised = 0.5 * target + 0.5 * noise
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, prediction = wrapped(
            source_foa_latent=ar_batch["source_foa_latent"],
            source_attention_mask=ar_batch["source_attention_mask"],
            plan_input_ids=ar_batch["plan_input_ids"],
            plan_attention_mask=ar_batch["plan_attention_mask"],
            raw_edit_requests=ar_batch["raw_edit_requests"],
            metadata=metadata,
            noised_target=noised,
            timesteps=timesteps,
            rf_padding_mask=rf_mask,
        )
        ar_loss, rf_loss, _, _ = _losses(
            logits,
            ar_batch["plan_labels"],
            prediction,
            noise - target,
            rf_mask,
        )
        loss = 0.1 * ar_loss + rf_loss
    loss.backward()
    summaries = {
        "editing_ar_adapters": _gradient_summary(ar_specific),
        "shared_transformer_blocks": _gradient_summary(shared),
        "editing_dit_and_conditioners": _gradient_summary(remaining),
    }
    if any(
        value["tensors"] <= 0 or value["norm_sum"] <= 0 or value["finite"] != 1.0
        for value in summaries.values()
    ):
        raise RuntimeError(f"joint DDP gradient gate failed: {summaries}")
    optimizer.step()
    checksum = torch.stack(
        [parameter.detach().float().sum() for parameter in parameters[:8]]
    ).sum()
    checksums = [torch.zeros_like(checksum) for _ in range(world_size)]
    dist.all_gather(checksums, checksum)
    synchronized = all(
        bool(torch.equal(checksums[0], value)) for value in checksums[1:]
    )
    payload = {
        "rank": rank,
        "pair_id": metadata[0]["pair_id"],
        "ar_ce": float(ar_loss.detach()),
        "rf_mse": float(rf_loss.detach()),
        "gradients": summaries,
        "parameter_checksum": float(checksum),
    }
    gathered: list[dict | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, payload)
    if rank == 0:
        result = {
            "schema": "sceneplan_transfusion_editing_joint_ddp_gate",
            "schema_version": 1,
            "status": "PASS" if synchronized else "FAIL",
            "contract": EDITING_AR_CONTRACT,
            "latest_route": {
                "editing_ar_inputs": ["source_foa_latent", "raw_edit_request"],
                "editing_ar_target": "complete_new_sceneplan",
                "old_sceneplan_input": False,
                "editing_dit_frame_input": [
                    "noisy_target_64",
                    "new_sceneplan_256",
                    "clean_source_foa_latent_64",
                ],
            },
            "physical_gpus": [3, 4],
            "world_size": world_size,
            "shared_transformer_same_object": (
                module.ar.shared_transformer
                is module.diffusion.model.model.transformer
            ),
            "ddp_parameters_synchronized": synchronized,
            "base_checkpoint": str(CHECKPOINT.resolve()),
            "base_checkpoint_sha256": sha256_file(CHECKPOINT),
            "ranks": gathered,
        }
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        print(json.dumps(result, sort_keys=True), flush=True)
        if result["status"] != "PASS":
            raise RuntimeError("joint DDP parameters diverged")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
