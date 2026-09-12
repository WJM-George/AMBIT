#!/usr/bin/env python3
"""Generate a deterministic checkpoint comparison panel for ScenePlan 4+4.

Each worker loads one checkpoint, consumes a disjoint shard of the frozen
15-row test panel, and writes raw four-channel FOA plus a fixed stereo preview.
The per-sample noise seed belongs to the panel contract, so a sample receives
identical initial noise at every checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
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

from scripts.t2a.eval.sceneplan_44_eval_common import (
    atomic_wav,
    audio_qc,
    virtual_stereo,
)
from scripts.t2a.eval.sceneplan_dit_p10_panel_common import (
    atomic_json,
    load_panel,
    read_jsonl,
    sha256_file,
)
from stable_audio_tools.configuration import load_config
from stable_audio_tools.data.model_sceneplan import (
    compile_model_semantic_caption,
    compile_model_semantic_caption_v2,
    make_sceneplan_cfg_unknown_metadata,
    tokenize_model_semantic_caption,
)
from stable_audio_tools.data.sceneplan_v2_dataset import ScenePlanV2Dataset
from stable_audio_tools.data.text_conditioning import collect_conditioner_tokenizers
from stable_audio_tools.inference.sampling import sample_diffusion
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.training.factory import create_training_wrapper_from_config


DEFAULT_EVAL_ROOT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives//p10_pre_v11_20260831/sceneplan_dit_fail/"
    "sceneplan_dit_v4_r8_300m/evaluation/"
    "p10_ckpt_5k_10k_15k_sceneplan44_v1"
)

# Backward-compatible helper names used by the VAE and diagnostic scripts.
_atomic_json = atomic_json
_atomic_wav = atomic_wav
_load_jsonl = read_jsonl
_qc = audio_qc
_sha256_file = sha256_file
_virtual_stereo = virtual_stereo


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _crop_metadata(metadata: dict[str, Any], frames: int) -> dict[str, Any]:
    """Remove batch padding while preserving exact framewise 4+4 controls."""

    controls = dict(metadata["sceneplan_44"])
    controls["source_event_frame_ids"] = controls[
        "source_event_frame_ids"
    ][:, :frames].clone()
    controls["source_trajectory_features"] = controls[
        "source_trajectory_features"
    ][:, :frames, :].clone()
    controls["speech_active_frame_mask"] = controls[
        "speech_active_frame_mask"
    ][:frames].clone()
    controls["frame_valid_mask"] = torch.ones(frames, dtype=torch.bool)
    output = dict(metadata)
    output["sceneplan_44"] = controls
    output["padding_mask"] = [torch.ones(frames, dtype=torch.bool)]
    output["latent_crop_length"] = frames
    output.pop("audio", None)
    return output


def _apply_semantic_caption_version(
    metadata: dict[str, Any],
    *,
    dataset: ScenePlanV2Dataset,
    compiler_version: int,
) -> dict[str, Any]:
    """Retokenize the contract's explicit semantic compiler version.

    The production dataset now defaults to canonical v2.  Historical v1
    comparisons must therefore compile v1 explicitly rather than inheriting a
    mutable dataset default.  Both versions preserve the same ScenePlan 4+4
    controls and exact spoken text.
    """

    if int(compiler_version) not in {1, 2}:
        raise ValueError("semantic caption compiler version must be 1 or 2")
    has_timing_teacher = any(
        str(key).startswith("speech_duration_") for key in metadata["prompt"]
    )
    if int(compiler_version) == 2 and has_timing_teacher:
        raise RuntimeError(
            "v2 prompt compatibility evaluation forbids a v1 timing sidecar"
        )
    if int(compiler_version) == 1 and has_timing_teacher:
        return metadata
    compiler = (
        compile_model_semantic_caption
        if int(compiler_version) == 1
        else compile_model_semantic_caption_v2
    )
    caption = compiler(metadata["model_sceneplan"])
    tokenized = tokenize_model_semantic_caption(
        caption,
        dataset.tokenizer,
        max_length=dataset.caption_max_tokens,
    )
    output = dict(metadata)
    output["prompt"] = {
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
    output["prompt_text"] = caption["text"]
    return output


def _checkpoint(contract: dict[str, Any], step: int) -> tuple[Path, str]:
    matches = [row for row in contract["checkpoints"] if int(row["step"]) == step]
    if len(matches) != 1:
        raise RuntimeError(f"contract does not contain exactly one step={step} checkpoint")
    item = matches[0]
    path = Path(item["path"]).expanduser().resolve(strict=True)
    if path.stat().st_size != int(item["bytes"]):
        raise RuntimeError(f"checkpoint size changed: {path}")
    digest = str(item.get("sha256") or sha256_file(path))
    if item.get("sha256") and sha256_file(path) != digest:
        raise RuntimeError(f"checkpoint SHA256 changed: {path}")
    return path, digest


def _already_complete(
    path: Path,
    *,
    step: int,
    checkpoint_sha256: str,
    semantic_caption_compiler_version: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        generated = Path(row["generated_foa_path"]).resolve(strict=True)
        preview = Path(row["generated_stereo_path"]).resolve(strict=True)
        return bool(
            row.get("status") == "PASS"
            and int(row["checkpoint_step"]) == int(step)
            and row["checkpoint_sha256"] == checkpoint_sha256
            and int(row.get("semantic_caption_compiler_version", -1))
            == int(semantic_caption_compiler_version)
            and sha256_file(generated) == row["generated_foa_sha256"]
            and sha256_file(preview) == row["generated_stereo_sha256"]
        )
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch per checkpoint replica; variable latent lengths are padded and masked.",
    )
    args = parser.parse_args()

    root = args.eval_root.expanduser().resolve(strict=True)
    contract = json.loads((root / "EVAL_CONTRACT.json").read_text(encoding="utf-8"))
    sampling = contract["sampling"]
    if "semantic_caption_compiler_version" not in sampling:
        raise RuntimeError(
            "evaluation contract must explicitly freeze the semantic caption compiler"
        )
    semantic_caption_compiler_version = int(
        sampling["semantic_caption_compiler_version"]
    )
    if semantic_caption_compiler_version not in {1, 2}:
        raise RuntimeError("unsupported semantic caption compiler version")
    if sampling.get("architecture") != "semantic_cross_attention_plus_direct_sceneplan_4+4":
        raise RuntimeError("evaluation contract is not the ScenePlan 4+4 architecture")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0,num-shards)")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ScenePlan P10 generation requires CUDA")

    full_panel = load_panel(root)
    selected = [
        row for index, row in enumerate(full_panel)
        if index % args.num_shards == args.shard_index
    ]
    if not selected:
        raise RuntimeError("worker shard is empty")
    checkpoint, checkpoint_sha256 = _checkpoint(contract, args.checkpoint_step)
    model_config = Path(sampling["model_config"]).resolve(strict=True)
    vae_checkpoint = Path(sampling["vae_checkpoint"]).resolve(strict=True)
    index_path = Path(contract["test_set"]["index"]).resolve(strict=True)
    if sha256_file(index_path) != contract["test_set"]["index_sha256"]:
        raise RuntimeError("frozen P9 test index SHA256 changed")

    pending = []
    for row in selected:
        metadata_path = (
            root / "outputs" / f"step_{args.checkpoint_step:06d}"
            / row["domain"] / row["panel_id"] / "metadata.json"
        )
        if _already_complete(
            metadata_path,
            step=args.checkpoint_step,
            checkpoint_sha256=checkpoint_sha256,
            semantic_caption_compiler_version=semantic_caption_compiler_version,
        ):
            print(json.dumps({"event": "resume_skip", "panel_id": row["panel_id"]}), flush=True)
        else:
            pending.append(row)

    print(
        json.dumps(
            {
                "event": "load_start",
                "step": args.checkpoint_step,
                "device": str(device),
                "rows": len(pending),
                "checkpoint": str(checkpoint),
                "semantic_caption_compiler_version": semantic_caption_compiler_version,
            }
        ),
        flush=True,
    )
    if not pending:
        return 0

    started_load = time.monotonic()
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
        incompatible_vae = model.pretransform.load_state_dict(vae_state, strict=False)
        if incompatible_vae is not None and incompatible_vae.unexpected_keys:
            raise RuntimeError(
                f"VAE checkpoint has unexpected keys: {incompatible_vae.unexpected_keys[:12]}"
            )
    else:
        raise RuntimeError("ScenePlan DiT has no VAE pretransform")
    del vae_state

    tokenizers = collect_conditioner_tokenizers(model, config)
    dataset = ScenePlanV2Dataset(
        index_path,
        tokenizer_spec=tokenizers["prompt"],
        expected_num_samples=len(pending),
        index_num_samples=int(contract["test_set"]["all_rows"]),
        sample_ordinals=[int(row["ordinal"]) for row in pending],
        latent_crop_length=int(
            contract["test_set"].get(
                "max_latent_frames",
                sampling.get("max_latent_frames", 432),
            )
        ),
    )
    diffusion = wrapper.diffusion
    if wrapper.diffusion_ema is None or wrapper.conditioner_ema is None:
        raise RuntimeError("checkpoint must contain EMA DiT and EMA conditioner")
    sample_model = wrapper.diffusion_ema.ema_model.to(device).eval().requires_grad_(False)
    diffusion.conditioner.to(device).eval().requires_grad_(False)
    diffusion.pretransform.to(device).eval().requires_grad_(False)
    dtype = next(sample_model.parameters()).dtype
    print(
        json.dumps(
            {
                "event": "load_complete",
                "step": args.checkpoint_step,
                "seconds": round(time.monotonic() - started_load, 3),
                "model_dtype": str(dtype),
            }
        ),
        flush=True,
    )

    outputs: list[str] = []
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    completed_rows = 0
    for batch_start in range(0, len(pending), args.batch_size):
        batch_started = time.monotonic()
        batch_rows = pending[batch_start : batch_start + args.batch_size]
        positives = []
        negatives = []
        frame_lengths = []
        sample_lengths = []
        noises = []
        model_prompt_texts = []
        for local_index, panel_row in enumerate(batch_rows):
            _latent, metadata = dataset[batch_start + local_index]
            if metadata["sample_id"] != panel_row["sample_id"]:
                raise RuntimeError("dataset row order disagrees with frozen panel")
            frames = int(panel_row["latent_frames_valid"])
            frame_lengths.append(frames)
            sample_lengths.append(int(panel_row["model_num_samples"]))
            positive = _crop_metadata(metadata, frames)
            positive = _apply_semantic_caption_version(
                positive,
                dataset=dataset,
                compiler_version=semantic_caption_compiler_version,
            )
            positives.append(positive)
            negatives.append(make_sceneplan_cfg_unknown_metadata(positive))
            model_prompt_texts.append(str(positive["prompt_text"]))
            seed = int(panel_row["noise_seed"])
            generator = torch.Generator(device="cpu").manual_seed(seed)
            noises.append(
                torch.randn(
                    (int(diffusion.io_channels), frames),
                    generator=generator,
                    dtype=torch.float32,
                )
            )
        max_frames = max(frame_lengths)
        _seed_everything(int(batch_rows[0]["noise_seed"]))
        noise = torch.stack(
            [F.pad(value, (0, max_frames - int(value.shape[-1]))) for value in noises]
        ).to(device=device, dtype=dtype)
        padding_mask = torch.stack(
            [
                F.pad(
                    torch.ones(frames, dtype=torch.bool),
                    (0, max_frames - frames),
                    value=False,
                )
                for frames in frame_lengths
            ]
        ).to(device)

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

        for local_index, panel_row in enumerate(batch_rows):
            frames = frame_lengths[local_index]
            samples = sample_lengths[local_index]
            seed = int(panel_row["noise_seed"])
            generated = generated_batch[local_index, :, :samples].contiguous()
            qc = audio_qc(generated)
            if not qc.get("finite") or qc.get("channels") != 4 or qc.get("samples") != samples:
                raise RuntimeError(f"invalid generated FOA for {panel_row['panel_id']}: {qc}")
            preview, preview_info = virtual_stereo(generated)
            sample_root = (
                root / "outputs" / f"step_{args.checkpoint_step:06d}"
                / panel_row["domain"] / panel_row["panel_id"]
            )
            raw_path = sample_root / "generated_foa_float32.wav"
            preview_path = sample_root / "generated_stereo.wav"
            atomic_wav(raw_path, generated, int(config["sample_rate"]), subtype="FLOAT")
            atomic_wav(preview_path, preview, int(config["sample_rate"]), subtype="PCM_16")
            result = {
                "schema": "stable_audio_tools.sceneplan_dit_p10_panel_output",
                "schema_version": 3,
                "status": "PASS",
                "architecture": "semantic_cross_attention_plus_direct_sceneplan_4+4",
                "checkpoint_step": int(args.checkpoint_step),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "panel_id": panel_row["panel_id"],
                "domain": panel_row["domain"],
                "sample_id": panel_row["sample_id"],
                "ordinal": int(panel_row["ordinal"]),
                "noise_seed": seed,
                "latent_frames": frames,
                "model_num_samples": samples,
                "sample_rate": int(config["sample_rate"]),
                "renderer_caption": panel_row["renderer_caption"],
                "semantic_text": panel_row["semantic_text"],
                "model_prompt_text": model_prompt_texts[local_index],
                "semantic_caption_compiler_version": (
                    semantic_caption_compiler_version
                ),
                "scene_plan": panel_row["scene_plan"],
                "source_kinds": panel_row.get("source_kinds"),
                "source_kind_counts": panel_row.get("source_kind_counts"),
                "source_semantic_texts": panel_row.get("source_semantic_texts"),
                "scene_composition": panel_row.get("scene_composition"),
                "length_bucket": panel_row.get("length_bucket"),
                "speech_seen_speaker": panel_row.get("speech_seen_speaker"),
                "speech_speaker_key": panel_row.get("speech_speaker_key"),
                "reference_foa_path": panel_row["reference_foa_path"],
                "reference_foa_sha256": panel_row["reference_foa_sha256"],
                "generated_foa_path": str(raw_path.resolve()),
                "generated_foa_sha256": sha256_file(raw_path),
                "generated_stereo_path": str(preview_path.resolve()),
                "generated_stereo_sha256": sha256_file(preview_path),
                "sampling_seconds": round(batch_seconds, 3),
                "amortized_sampling_seconds": round(
                    batch_seconds / len(batch_rows), 3
                ),
                "inference_batch_size": len(batch_rows),
                "inference_batch_max_frames": max_frames,
                "sampling": {
                    "steps": int(sampling["steps"]),
                    "cfg_scale": float(sampling["cfg_scale"]),
                    "rescale_cfg": bool(sampling["rescale_cfg"]),
                    "cfg_rescale_phi": float(sampling.get("cfg_rescale_phi", 0.4)),
                    "apg_scale": float(sampling["apg_scale"]),
                },
                "qc": qc,
                **preview_info,
            }
            atomic_json(sample_root / "metadata.json", result)
            outputs.append(str(raw_path.resolve()))
            completed_rows += 1
            print(
                json.dumps(
                    {
                        "event": "sample_complete",
                        "step": args.checkpoint_step,
                        "index": completed_rows,
                        "count": len(pending),
                        "panel_id": panel_row["panel_id"],
                        "batch_size": len(batch_rows),
                        "batch_seconds": result["sampling_seconds"],
                        "amortized_seconds": result["amortized_sampling_seconds"],
                        "peak": qc["peak"],
                        "rms": qc["rms"],
                    }
                ),
                flush=True,
            )
        del generated_batch, noise, padding_mask

    summary = {
        "schema": "stable_audio_tools.sceneplan_dit_p10_panel_worker",
        "schema_version": 2,
        "status": "PASS",
        "checkpoint_step": int(args.checkpoint_step),
        "device": str(device),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "batch_size": int(args.batch_size),
        "rows": len(pending),
        "panel_ids": [row["panel_id"] for row in pending],
        "outputs": outputs,
    }
    worker_path = (
        root / "logs"
        / f"worker_step_{args.checkpoint_step:06d}_shard_{args.shard_index:02d}.json"
    )
    atomic_json(worker_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    del sample_model, wrapper, model, dataset, diffusion
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
