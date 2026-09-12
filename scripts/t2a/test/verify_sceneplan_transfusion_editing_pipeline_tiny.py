#!/usr/bin/env python3
"""Real-pair AR -> new ScenePlan -> Editing-DiT inference gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    ScenePlanTransfusionEditingAR,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_pipeline import (  # noqa: E402
    ScenePlanTransfusionEditingPipeline,
    _load_ar_specific,
)


MODEL_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m_sceneplan_transfusion_editing_dit_v1.json"
)
INDEX = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_transfusion_editing_v1_pilot/"
    "training_index/train.sqlite"
)
INDEX_SHA256 = "6535b4a04b38a07ff641a876361e0dd16d4d4d4737ad27c11304d1d99c785600"
BUNDLE = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives//transfusion_editing/pilots/"
    "sceneplan_transfusion_editing_ar_joint_overfit10_seed42_v3/"
    "checkpoints/step-500.pt"
)
BUNDLE_SHA256 = "3043652a24820fc61f75f34caaee251a672fc5a301233c7d7853b2d1058bcac0"
CODEC = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)


def _stable_row_seed(pair_id: str, namespace: str) -> int:
    """Derive a batch-order-independent CPU RNG seed for one Editing pair."""

    digest = hashlib.blake2b(
        f"{namespace}\0{pair_id}".encode("utf-8"),
        digest_size=8,
        person=b"edit-tiny-v1",
    ).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _write_json_atomic(path: Path, payload: dict) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--ode-steps", type=int, default=4)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--bundle-sha256", default=BUNDLE_SHA256)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise RuntimeError("tiny pipeline gate requires unremapped physical GPUs")
    if args.gpu not in range(3, 8) or min(args.rows, args.ode_steps) <= 0:
        raise ValueError("invalid Editing tiny pipeline gate settings")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    bundle = args.bundle.expanduser().resolve()
    if sha256_file(bundle) != args.bundle_sha256:
        raise RuntimeError("tiny joint Editing bundle SHA256 changed")
    payload = torch.load(bundle, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != "sceneplan_transfusion_editing_joint_bundle"
        or payload.get("metadata", {}).get("old_sceneplan_exposed") is not False
    ):
        raise RuntimeError("tiny bundle is not latest-route Editing")
    config = load_config(MODEL_CONFIG)
    diffusion = create_model_from_config(config)
    diffusion.pretransform = None
    diffusion.load_state_dict(payload["diffusion_state_dict"], strict=True)
    prompt = diffusion.conditioner.conditioners["prompt"]
    codec = ModelScenePlanCodecV4(CODEC)
    ar = ScenePlanTransfusionEditingAR(
        editing_dit=diffusion.model.model,
        instruction_conditioner=prompt,
        pad_id=codec.pad_id,
        activation_checkpointing=False,
    )
    _load_ar_specific(ar, payload["editing_ar_specific_state_dict"])
    pipeline = ScenePlanTransfusionEditingPipeline(
        diffusion=diffusion, editing_ar=ar, codec=codec
    ).eval().requires_grad_(False).to(device)
    base = ScenePlanTransfusionEditingDataset(
        INDEX,
        tokenizer_spec=(prompt.tokenizer, 512),
        expected_num_samples=10,
        expected_index_sha256=INDEX_SHA256,
        latent_crop_length=648,
        require_frozen=True,
        verify_tensor_hashes_on_access=True,
    )
    joint = ScenePlanTransfusionEditingJointDataset(base, codec=codec)
    selected_buckets = []
    remaining = int(args.rows)
    for bucket_frames, values in sorted(joint.length_bucket_indices().items()):
        take = min(remaining, len(values))
        if take:
            selected_buckets.append((int(bucket_frames), list(values)[:take]))
            remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise ValueError(
            f"requested {args.rows} rows but tiny index only contains "
            f"{args.rows - remaining} rows"
        )

    prepared_batches = []
    diagnostics = []
    pair_ids = []
    for bucket_frames, indices in selected_buckets:
        samples = [joint[index] for index in indices]
        batch = collate_editing_joint(samples, pad_id=codec.pad_id)
        metadata = list(batch["metadata"])
        source = torch.stack([row["source_foa_latent"] for row in metadata]).to(
            device=device, dtype=torch.float32
        )
        mask = torch.stack([row["padding_mask"][0] for row in metadata]).to(device)
        instructions = [str(row["raw_edit_request"]) for row in metadata]
        durations = [float(row["seconds_total"]) for row in metadata]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            plans, tokens = pipeline.generate_new_sceneplans(
                source,
                mask,
                instructions,
                duration_sec=durations,
                max_plan_tokens=512,
            )
        target_tokens = [
            sample[2]["target_token_ids"].tolist() for sample in samples
        ]
        generated_tokens = [value.tolist() for value in tokens]
        for pair_id, generated, target in zip(
            [row["pair_id"] for row in metadata], generated_tokens, target_tokens
        ):
            common = min(len(generated), len(target))
            first_mismatch = next(
                (index for index in range(common) if generated[index] != target[index]),
                common if len(generated) != len(target) else None,
            )
            diagnostics.append(
                {
                    "pair_id": pair_id,
                    "bucket_frames": bucket_frames,
                    "generated_length": len(generated),
                    "target_length": len(target),
                    "first_mismatch": first_mismatch,
                    "generated_token": (
                        generated[first_mismatch]
                        if first_mismatch is not None
                        and first_mismatch < len(generated)
                        else None
                    ),
                    "target_token": (
                        target[first_mismatch]
                        if first_mismatch is not None and first_mismatch < len(target)
                        else None
                    ),
                }
            )
        pair_ids.extend(str(row["pair_id"]) for row in metadata)
        prepared_batches.append((bucket_frames, source, mask, plans, metadata))

    mismatches = [row for row in diagnostics if row["first_mismatch"] is not None]
    if mismatches:
        failure = {
            "schema": "sceneplan_transfusion_editing_pipeline_tiny_gate",
            "schema_version": 1,
            "status": "FAIL",
            "reason": "free_ar_sequence_not_exact",
            "physical_gpu": int(args.gpu),
            "rows": len(pair_ids),
            "sequence_diagnostics": diagnostics,
            "bundle": str(bundle),
            "bundle_sha256": args.bundle_sha256,
            "index": str(INDEX.resolve()),
            "index_sha256": INDEX_SHA256,
            "verifier_sha256": sha256_file(Path(__file__).resolve()),
        }
        _write_json_atomic(args.output, failure)
        print(json.dumps(failure, sort_keys=True), flush=True)
        raise RuntimeError("tiny pipeline free AR decode is not exactly learned")

    edited_batches = []
    for bucket_frames, source, mask, plans, metadata in prepared_batches:
        edited = pipeline.sample_edited_latents(
            source,
            mask,
            plans,
            model_num_samples=[int(row["model_num_samples"]) for row in metadata],
            steps=int(args.ode_steps),
            cfg_scale=1.0,
            seed=[
                _stable_row_seed(str(row["pair_id"]), "dit-noise")
                for row in metadata
            ],
        )
        if tuple(edited.shape) != tuple(source.shape) or not torch.isfinite(edited).all():
            raise RuntimeError("tiny pipeline Editing-DiT output is invalid")
        edited_batches.append(
            {
                "bucket_frames": bucket_frames,
                "rows": int(edited.shape[0]),
                "edited_latent_shape": list(edited.shape),
            }
        )
    result = {
        "schema": "sceneplan_transfusion_editing_pipeline_tiny_gate",
        "schema_version": 1,
        "status": "PASS",
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
        "physical_gpu": int(args.gpu),
        "rows": len(pair_ids),
        "pair_ids": pair_ids,
        "free_ar_sequence_exact": 1.0,
        "ode_steps": int(args.ode_steps),
        "edited_batches": edited_batches,
        "shared_transformer_same_object": (
            pipeline.editing_ar.shared_transformer
            is pipeline.diffusion.model.model.transformer
        ),
        "bundle": str(bundle),
        "bundle_sha256": args.bundle_sha256,
        "index": str(INDEX.resolve()),
        "index_sha256": INDEX_SHA256,
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
    }
    _write_json_atomic(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
