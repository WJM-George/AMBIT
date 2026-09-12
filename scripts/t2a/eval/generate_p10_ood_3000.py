#!/usr/bin/env python3
"""Run the frozen P10 150k renderer on the deduplicated OOD 3k panel.

Each natural single-source example is adapted to a neutral dry ScenePlan.  The
model therefore uses its native ScenePlan interface, while scoring uses only
the generated W channel against the natural mono reference.  No OOD spatial
metric is valid because the references contain no ground-truth FOA geometry.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.sceneplan_44_eval_common import atomic_wav, audio_qc
from scripts.t2a.eval.sceneplan_dit_p10_panel_common import atomic_json, read_jsonl
from stable_audio_tools.configuration import load_config
from stable_audio_tools.data.model_sceneplan import (
    compile_model_44_controls,
    compile_model_semantic_caption_v2,
    make_sceneplan_cfg_unknown_metadata,
    tokenize_model_semantic_caption,
    validate_model_sceneplan,
)
from stable_audio_tools.data.text_conditioning import collect_conditioner_tokenizers
from stable_audio_tools.inference.sampling import sample_diffusion
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.training.factory import create_training_wrapper_from_config


DEFAULT_ROOT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/evaluation_benchmark/p10_ood_3000_v1/"
    "cross_system_benchmark"
)
SAMPLE_RATE = 44_100
HOP = 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _model_sceneplan(row: dict[str, Any]) -> dict[str, Any]:
    plan = copy.deepcopy(row["scene_plan"])
    plan["sample_id"] = str(row["sample_id"])
    samples = int(round(float(row["requested_duration_sec"]) * SAMPLE_RATE))
    plan["duration_sec"] = samples / SAMPLE_RATE
    for slot, source in enumerate(plan["sources"]):
        source["source_id"] = f"source_{slot}"
    validate_model_sceneplan(plan)
    return plan


def _metadata(
    row: dict[str, Any],
    *,
    tokenizer: Any,
) -> tuple[dict[str, Any], int, int, str, dict[str, Any]]:
    plan = _model_sceneplan(row)
    samples = int(round(float(row["requested_duration_sec"]) * SAMPLE_RATE))
    frames = int(math.ceil(samples / HOP))
    caption = compile_model_semantic_caption_v2(plan)
    tokenized = tokenize_model_semantic_caption(caption, tokenizer, max_length=512)
    prompt = {
        "input_ids": torch.as_tensor(tokenized["input_ids"], dtype=torch.long),
        "attention_mask": torch.as_tensor(
            tokenized["attention_mask"], dtype=torch.bool
        ),
        "event_source_ids": torch.as_tensor(
            tokenized["event_source_ids"], dtype=torch.int8
        ),
        "speech_source_ids": torch.as_tensor(
            tokenized["speech_source_ids"], dtype=torch.int8
        ),
        "speech_lexical_mask": torch.as_tensor(
            tokenized["speech_lexical_mask"], dtype=torch.bool
        ),
    }
    structured = compile_model_44_controls(
        plan,
        model_num_samples=samples,
        latent_frames_valid=frames,
    )
    valid = torch.ones(frames, dtype=torch.bool)
    controls = {
        "source_event_frame_ids": torch.as_tensor(
            structured["source_event_frame_ids"], dtype=torch.int8
        ),
        "source_trajectory_features": torch.as_tensor(
            structured["source_trajectory_features"], dtype=torch.float32
        ),
        "frame_valid_mask": valid.clone(),
        "speech_active_frame_mask": torch.as_tensor(
            structured["speech_active_frame_mask"], dtype=torch.bool
        ),
    }
    metadata = {
        "sample_id": str(row["sample_id"]),
        "model_sceneplan": plan,
        "model_num_samples": samples,
        "prompt": prompt,
        "prompt_text": caption["text"],
        "semantic_caption_compiler_version": 2,
        "semantic_caption_epoch": None,
        "sceneplan_44": controls,
        "padding_mask": [valid],
        "seconds_start": 0.0,
        "seconds_total": samples / SAMPLE_RATE,
        "latent_stored_length": frames,
        "latent_crop_length": frames,
        "latent_bucket_frames": 432,
        "latent_crop_start": 0,
    }
    return metadata, frames, samples, caption["text"], plan


def _paths(root: Path, panel_id: str) -> tuple[Path, Path, Path]:
    sample_root = root / "outputs" / "ours_p10_150k" / panel_id
    return (
        sample_root / "native_foa.wav",
        sample_root / "quality_w.wav",
        sample_root / "generation.json",
    )


def _complete(
    root: Path,
    row: dict[str, Any],
    *,
    checkpoint_sha256: str,
) -> bool:
    native, quality, metadata_path = _paths(root, str(row["panel_id"]))
    if not (native.is_file() and quality.is_file() and metadata_path.is_file()):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return bool(
            metadata.get("status") == "PASS"
            and metadata.get("panel_id") == row["panel_id"]
            and metadata.get("sample_id") == row["sample_id"]
            and metadata.get("checkpoint_sha256") == checkpoint_sha256
            and metadata.get("reference_audio_sha256")
            == row["reference_audio_sha256"]
            and _sha256(native) == metadata["native_foa_sha256"]
            and _sha256(quality) == metadata["quality_w_sha256"]
        )
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    root = args.benchmark_root.expanduser().resolve(strict=True)
    contract_path = root / "BENCHMARK_CONTRACT.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    panel_path = Path(contract["source_panel_path"]).resolve(strict=True)
    if _sha256(panel_path) != contract["source_panel_sha256"]:
        raise RuntimeError("OOD panel SHA256 changed")
    panel = read_jsonl(panel_path)
    if len(panel) != int(contract["source_panel_rows"]):
        raise RuntimeError("OOD panel row count changed")
    if not 0 <= int(args.shard_index) < int(args.num_shards):
        raise ValueError("invalid shard")
    selected = [
        row
        for index, row in enumerate(panel)
        if index % int(args.num_shards) == int(args.shard_index)
    ]
    if not selected:
        raise RuntimeError("empty OOD shard")

    p10 = contract["p10"]
    checkpoint = Path(p10["checkpoint_path"]).resolve(strict=True)
    if _sha256(checkpoint) != p10["checkpoint_sha256"]:
        raise RuntimeError("P10 checkpoint SHA256 changed")
    source_contract_path = Path(p10["source_eval_contract"]).resolve(strict=True)
    if _sha256(source_contract_path) != p10["source_eval_contract_sha256"]:
        raise RuntimeError("P10 source evaluation contract SHA256 changed")
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    sampling = source_contract["sampling"]
    if int(sampling["semantic_caption_compiler_version"]) != 2:
        raise RuntimeError("OOD inference requires semantic caption compiler v2")
    model_config = Path(sampling["model_config"]).resolve(strict=True)
    vae_checkpoint = Path(sampling["vae_checkpoint"]).resolve(strict=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("P10 OOD generation requires CUDA")

    pending = [
        row
        for row in selected
        if not _complete(root, row, checkpoint_sha256=p10["checkpoint_sha256"])
    ]
    print(
        json.dumps(
            {
                "event": "p10_ood_load_start",
                "device": str(device),
                "shard": int(args.shard_index),
                "selected": len(selected),
                "pending": len(pending),
            }
        ),
        flush=True,
    )
    if not pending:
        return 0

    config = load_config(model_config)
    model = create_model_from_config(config)
    wrapper = create_training_wrapper_from_config(config, model)
    state = load_ckpt_state_dict(str(checkpoint))
    incompatible = wrapper.load_state_dict(state, strict=False)
    del state
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint/model mismatch: "
            f"missing={incompatible.missing_keys[:12]}, "
            f"unexpected={incompatible.unexpected_keys[:12]}"
        )
    vae_state = load_ckpt_state_dict(str(vae_checkpoint))
    if hasattr(model, "load_pretransform_state_dict"):
        model.load_pretransform_state_dict(vae_state, strict=False)
    elif model.pretransform is not None:
        model.pretransform.load_state_dict(vae_state, strict=False)
    else:
        raise RuntimeError("P10 has no VAE pretransform")
    del vae_state

    diffusion = wrapper.diffusion
    if wrapper.diffusion_ema is None or wrapper.conditioner_ema is None:
        raise RuntimeError("P10 checkpoint lacks EMA modules")
    sample_model = wrapper.diffusion_ema.ema_model.to(device).eval().requires_grad_(False)
    diffusion.conditioner.to(device).eval().requires_grad_(False)
    diffusion.pretransform.to(device).eval().requires_grad_(False)
    tokenizers = collect_conditioner_tokenizers(model, config)
    tokenizer = tokenizers["prompt"][0]
    dtype = next(sample_model.parameters()).dtype
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)

    completed = 0
    for batch_start in range(0, len(pending), int(args.batch_size)):
        batch_started = time.monotonic()
        batch_rows = pending[batch_start : batch_start + int(args.batch_size)]
        positives: list[dict[str, Any]] = []
        negatives: list[dict[str, Any]] = []
        frames_list: list[int] = []
        samples_list: list[int] = []
        prompt_texts: list[str] = []
        plans: list[dict[str, Any]] = []
        noises: list[torch.Tensor] = []
        for row in batch_rows:
            positive, frames, samples, prompt_text, plan = _metadata(
                row, tokenizer=tokenizer
            )
            positives.append(positive)
            negatives.append(make_sceneplan_cfg_unknown_metadata(positive))
            frames_list.append(frames)
            samples_list.append(samples)
            prompt_texts.append(prompt_text)
            plans.append(plan)
            generator = torch.Generator(device="cpu").manual_seed(int(row["noise_seed"]))
            noises.append(
                torch.randn(
                    (int(diffusion.io_channels), frames),
                    generator=generator,
                    dtype=torch.float32,
                )
            )
        max_frames = max(frames_list)
        noise = torch.stack(
            [F.pad(value, (0, max_frames - value.shape[-1])) for value in noises]
        ).to(device=device, dtype=dtype)
        padding_mask = torch.stack(
            [
                F.pad(
                    torch.ones(frames, dtype=torch.bool),
                    (0, max_frames - frames),
                    value=False,
                )
                for frames in frames_list
            ]
        ).to(device)
        _seed_everything(int(batch_rows[0]["noise_seed"]))
        with torch.inference_mode(), autocast:
            with wrapper.ema_conditioner_context():
                positive_tensors = diffusion.conditioner(positives, device)
                negative_tensors = diffusion.conditioner(negatives, device)
            cond_inputs = diffusion.get_conditioning_inputs(positive_tensors)
            cond_inputs.update(
                diffusion.get_conditioning_inputs(negative_tensors, negative=True)
            )
            cond_inputs = {
                key: value.to(dtype) if isinstance(value, torch.Tensor) else value
                for key, value in cond_inputs.items()
            }
            generated_batch = sample_diffusion(
                model=sample_model,
                noise=noise,
                cond_inputs=cond_inputs,
                diffusion_objective=diffusion.diffusion_objective,
                steps=int(sampling["steps"]),
                cfg_scale=float(sampling["cfg_scale"]),
                conditioning=positives,
                sample_rate=int(config["sample_rate"]),
                pretransform=diffusion.pretransform,
                mask_padding_attention=True,
                use_effective_length_for_schedule=False,
                padding_mask=padding_mask,
                dist_shift=diffusion.sampling_dist_shift,
                sampler_type="euler",
                batch_cfg=True,
                rescale_cfg=bool(sampling["rescale_cfg"]),
                cfg_rescale_phi=float(sampling.get("cfg_rescale_phi", 0.4)),
                apg_scale=float(sampling["apg_scale"]),
                decode=True,
                disable_tqdm=True,
            ).float().cpu()
        batch_seconds = time.monotonic() - batch_started

        for index, row in enumerate(batch_rows):
            generated = generated_batch[index, :, : samples_list[index]].contiguous()
            qc = audio_qc(generated)
            if not qc.get("finite") or qc.get("channels") != 4:
                raise RuntimeError(f"invalid OOD FOA: {row['panel_id']} {qc}")
            native, quality, metadata_path = _paths(root, str(row["panel_id"]))
            atomic_wav(native, generated, int(config["sample_rate"]), subtype="FLOAT")
            atomic_wav(
                quality,
                generated[0:1],
                int(config["sample_rate"]),
                subtype="FLOAT",
            )
            result = {
                "schema": "sceneplan_foa.p10_ood_generation_output",
                "schema_version": 1,
                "status": "PASS",
                "system_id": "ours_p10_150k",
                "checkpoint_step": 150000,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": p10["checkpoint_sha256"],
                "panel_id": row["panel_id"],
                "sample_id": row["sample_id"],
                "domain": row["domain"],
                "noise_seed": int(row["noise_seed"]),
                "model_sceneplan": plans[index],
                "model_prompt_text": prompt_texts[index],
                "semantic_text": row["semantic_text"],
                "model_num_samples": samples_list[index],
                "latent_frames": frames_list[index],
                "sample_rate_hz": int(config["sample_rate"]),
                "reference_audio_path": row["reference_audio_path"],
                "reference_audio_sha256": row["reference_audio_sha256"],
                "native_foa_path": str(native.resolve()),
                "native_foa_sha256": _sha256(native),
                "quality_w_path": str(quality.resolve()),
                "quality_w_sha256": _sha256(quality),
                "quality_view": "native generated FOA W channel",
                "spatial_metrics": "N/A; natural OOD reference is mono",
                "sampling_seconds": round(batch_seconds, 3),
                "amortized_sampling_seconds": round(
                    batch_seconds / len(batch_rows), 3
                ),
                "sampling": {
                    "steps": int(sampling["steps"]),
                    "cfg_scale": float(sampling["cfg_scale"]),
                    "rescale_cfg": bool(sampling["rescale_cfg"]),
                    "cfg_rescale_phi": float(sampling.get("cfg_rescale_phi", 0.4)),
                    "apg_scale": float(sampling["apg_scale"]),
                },
                "qc": qc,
            }
            atomic_json(metadata_path, result)
            completed += 1
            print(
                json.dumps(
                    {
                        "event": "p10_ood_sample_complete",
                        "completed": completed,
                        "pending": len(pending),
                        "panel_id": row["panel_id"],
                        "amortized_seconds": result["amortized_sampling_seconds"],
                    }
                ),
                flush=True,
            )
        del generated_batch, noise, padding_mask

    worker = {
        "schema": "sceneplan_foa.p10_ood_worker",
        "schema_version": 1,
        "status": "PASS",
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "selected": len(selected),
        "generated": len(pending),
    }
    atomic_json(root / "logs" / f"ours_shard_{int(args.shard_index):02d}.json", worker)
    del sample_model, wrapper, model, diffusion
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(worker), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
