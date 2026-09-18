#!/usr/bin/env python3
"""Generate and gate the revision-6 mixed 0--15 second overfit panel."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.sceneplan_44_eval_common import (  # noqa: E402
    atomic_wav,
    audio_qc,
    virtual_stereo,
)
from scripts.t2a.eval.score_sceneplan_dit_p10_clap import (  # noqa: E402
    _audio_embeddings,
    _text_embeddings,
)
from scripts.t2a.eval.score_sceneplan_dit_p10_core import (  # noqa: E402
    _activity_metrics,
    _doa_metrics,
)
from scripts.t2a.eval.score_sceneplan_dit_p10_speech import (  # noqa: E402
    DEFAULT_WHISPER,
    _error_rates,
    _mono_16k,
    _transcribe,
)
from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.data.model_sceneplan import (  # noqa: E402
    compile_model_44_controls,
    make_sceneplan_cfg_unknown_metadata,
)
from stable_audio_tools.data.sceneplan_v2_dataset import (  # noqa: E402
    ScenePlanV2Dataset,
)
from stable_audio_tools.data.text_conditioning import (  # noqa: E402
    collect_conditioner_tokenizers,
)
from stable_audio_tools.inference.sampling import sample_diffusion  # noqa: E402
from stable_audio_tools.models import create_model_from_config  # noqa: E402
from stable_audio_tools.models.utils import (  # noqa: E402
    load_ckpt_state_dict,
)
from stable_audio_tools.training.factory import (  # noqa: E402
    create_training_wrapper_from_config,
)
from stable_audio_tools.training.metrics.fad_metrics import (  # noqa: E402
    load_clap_model,
)


DEFAULT_RUN = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives/pilots/"
    "overfit"
)
MODEL_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m_model_sceneplan_44_soundexp_noalign_15s.json"
)
DATASET_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/dataset_configs/"
    "sceneplan_44_speechexp_noalign_15s_overfit10.json"
)
VAE_CHECKPOINT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt"
)
CLAP_CHECKPOINT = REPO_ROOT / "load/clap_score/630k-audioset-fusion-best.pt"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def latest_checkpoint(root: Path) -> Path:
    choices = []
    for path in (root / "checkpoints").glob("*step=*.ckpt"):
        match = re.search(r"step=(\d+)\.ckpt$", path.name)
        if match:
            choices.append((int(match.group(1)), path))
    if not choices:
        raise FileNotFoundError(f"no permanent checkpoint below {root}")
    return max(choices)[1].resolve(strict=True)


def crop_metadata(metadata: dict[str, Any], frames: int) -> dict[str, Any]:
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


def mean(values) -> float:
    kept = [float(value) for value in values if value is not None]
    return float(sum(kept) / len(kept)) if kept else float("nan")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_receipts(run_root: Path) -> dict[str, Any]:
    raw = (run_root / "logs/train.log").read_text(
        encoding="utf-8", errors="replace"
    )
    cfg_matches = re.findall(
        r"SAT_SCENEPLAN_CFG_DROPOUT_RESULT=(\{[^\r\n]+\})", raw
    )
    health_matches = re.findall(
        r"SAT_TRAINING_GATE_RESULT=(\{[^\r\n]+\})", raw
    )
    if len(cfg_matches) != 1 or len(health_matches) != 1:
        raise RuntimeError("overfit run omitted unique CFG/health receipts")
    cfg = json.loads(cfg_matches[0])
    health = json.loads(health_matches[0])
    observed = cfg.get("observed_fractions") or {}
    caption = float(observed.get("caption_unknown_fraction", float("nan")))
    structured = float(
        observed.get("structured_unknown_fraction", float("nan"))
    )
    joint = float(observed.get("joint_unknown_fraction", float("nan")))
    full = float(observed.get("full_condition_fraction", float("nan")))
    cfg_ok = bool(
        cfg.get("mode") == "independent"
        and cfg.get("configured")
        == {"caption_unknown_prob": 0.15, "structured_unknown_prob": 0.15}
        and int(cfg.get("samples", 0)) >= 9_000
        and 0.135 <= caption <= 0.165
        and 0.135 <= structured <= 0.165
        and 0.015 <= joint <= 0.03
        and 0.70 <= full <= 0.745
        and abs(joint - caption * structured) <= 0.0075
    )
    windows = health.get("metric_windows") or {}
    loss_window = windows.get("train/loss") or {}
    loss_first = float(loss_window.get("first_mean", float("nan")))
    loss_last = float(loss_window.get("last_mean", float("nan")))
    health_ok = bool(
        health.get("status") == "PASS"
        and int(health.get("global_step", 0)) == 2_000
        and math.isfinite(loss_first)
        and math.isfinite(loss_last)
        and loss_last < loss_first * 0.70
        and health.get("optimizer_state_step_max", 0) >= 2_000
        and health.get("ema_final", {}).get("diffusion_ema", 0) >= 2_000
        and health.get("ema_final", {}).get("conditioner_ema", 0) >= 2_000
    )
    return {
        "cfg": cfg,
        "cfg_ok": cfg_ok,
        "health": health,
        "health_ok": health_ok,
        "loss_first_mean": loss_first,
        "loss_last_mean": loss_last,
    }


def spatially_identifiable(sceneplan: dict[str, Any], samples: int, frames: int) -> bool:
    controls = compile_model_44_controls(
        sceneplan,
        model_num_samples=samples,
        latent_frames_valid=frames,
    )
    simultaneous = (controls["source_event_frame_ids"] > 0).sum(axis=0)
    return bool((simultaneous <= 1).all())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve(strict=True)
    checkpoint = (
        args.checkpoint.expanduser().resolve(strict=True)
        if args.checkpoint
        else latest_checkpoint(root)
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("revision-6 overfit evaluation requires CUDA")
    receipts = training_receipts(root)
    config = load_config(MODEL_CONFIG.resolve(strict=True))
    dataset_config = load_config(DATASET_CONFIG.resolve(strict=True))
    if not (
        dataset_config.get("expected_num_samples") == 10
        and dataset_config.get("index_num_samples") == 1_600_000
        and dataset_config.get("speech_timing_sidecar") is None
        and dataset_config.get("word_level_timestamp_teacher") is False
        and len(dataset_config.get("sample_ordinals") or ()) == 10
    ):
        raise RuntimeError("revision-6 overfit panel contract changed")

    print(json.dumps({"event": "load", "checkpoint": str(checkpoint)}), flush=True)
    model = create_model_from_config(config)
    wrapper = create_training_wrapper_from_config(config, model)
    incompatible = wrapper.load_state_dict(
        load_ckpt_state_dict(str(checkpoint)), strict=False
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint/model mismatch: "
            f"missing={incompatible.missing_keys[:12]} "
            f"unexpected={incompatible.unexpected_keys[:12]}"
        )
    vae_state = load_ckpt_state_dict(str(VAE_CHECKPOINT.resolve(strict=True)))
    if hasattr(model, "load_pretransform_state_dict"):
        model.load_pretransform_state_dict(vae_state, strict=False)
    elif model.pretransform is not None:
        model.pretransform.load_state_dict(vae_state, strict=False)
    else:
        raise RuntimeError("ScenePlan P10 model has no frozen VAE decoder")
    del vae_state

    tokenizers = collect_conditioner_tokenizers(model, config)
    dataset = ScenePlanV2Dataset(
        dataset_config["datasets"][0]["path"],
        tokenizer_spec=tokenizers["prompt"],
        expected_num_samples=10,
        index_num_samples=1_600_000,
        sample_ordinals=dataset_config["sample_ordinals"],
        latent_crop_length=648,
        caption_max_tokens=512,
        random_crop=False,
        require_frozen=True,
    )
    diffusion = wrapper.diffusion
    if wrapper.diffusion_ema is None or wrapper.conditioner_ema is None:
        raise RuntimeError("overfit checkpoint has no EMA DiT/conditioner")
    sample_model = wrapper.diffusion_ema.ema_model.to(device).eval().requires_grad_(False)
    diffusion.conditioner.to(device).eval().requires_grad_(False)
    diffusion.pretransform.to(device).eval().requires_grad_(False)
    dtype = next(sample_model.parameters()).dtype
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    output = root / "evaluation/revision6_overfit10"
    rows: list[dict[str, Any]] = []

    for panel_index in range(len(dataset)):
        started = time.monotonic()
        latent, metadata = dataset[panel_index]
        sceneplan = metadata["model_sceneplan"]
        frames = int(metadata["latent_stored_length"])
        samples = int(metadata["model_num_samples"])
        positive = crop_metadata(metadata, frames)
        negative = make_sceneplan_cfg_unknown_metadata(positive)
        seed = 202_608_280 + panel_index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        noise = torch.randn(
            (1, int(diffusion.io_channels), frames),
            generator=torch.Generator(device="cpu").manual_seed(seed),
            dtype=torch.float32,
        ).to(device=device, dtype=dtype)
        with torch.inference_mode(), autocast:
            with wrapper.ema_conditioner_context():
                positive_tensors = diffusion.conditioner([positive], device)
                negative_tensors = diffusion.conditioner([negative], device)
            cond_inputs = diffusion.get_conditioning_inputs(positive_tensors)
            cond_inputs.update(
                diffusion.get_conditioning_inputs(negative_tensors, negative=True)
            )
            cond_inputs = {
                key: value.to(dtype) if isinstance(value, torch.Tensor) else value
                for key, value in cond_inputs.items()
            }
            generated = sample_diffusion(
                model=sample_model,
                noise=noise,
                cond_inputs=cond_inputs,
                diffusion_objective=diffusion.diffusion_objective,
                steps=int(args.steps),
                cfg_scale=float(args.cfg_scale),
                conditioning=[positive],
                sample_rate=int(config["sample_rate"]),
                pretransform=diffusion.pretransform,
                mask_padding_attention=True,
                use_effective_length_for_schedule=False,
                padding_mask=torch.ones(
                    1, frames, dtype=torch.bool, device=device
                ),
                dist_shift=diffusion.sampling_dist_shift,
                sampler_type="euler",
                batch_cfg=True,
                rescale_cfg=True,
                cfg_rescale_phi=0.4,
                apg_scale=0.0,
                decode=True,
                disable_tqdm=True,
            )[0, :, :samples]
            reference = diffusion.pretransform.decode(
                latent[:, :frames].unsqueeze(0).to(device=device, dtype=dtype)
            )[0, :, :samples]
        generated = generated.float().cpu().contiguous()
        reference = reference.float().cpu().contiguous()
        sample_root = output / f"{panel_index:02d}_{metadata['sample_id']}"
        generated_path = sample_root / "generated_foa.wav"
        reference_path = sample_root / "reference_vae_foa.wav"
        generated_preview, _ = virtual_stereo(generated)
        reference_preview, _ = virtual_stereo(reference)
        atomic_wav(generated_path, generated, 44_100, subtype="FLOAT")
        atomic_wav(reference_path, reference, 44_100, subtype="FLOAT")
        atomic_wav(
            sample_root / "generated_stereo.wav",
            generated_preview,
            44_100,
            subtype="PCM_16",
        )
        atomic_wav(
            sample_root / "reference_vae_stereo.wav",
            reference_preview,
            44_100,
            subtype="PCM_16",
        )
        speech = [source for source in sceneplan["sources"] if source["kind"] == "speech"]
        spatial_eligible = spatially_identifiable(sceneplan, samples, frames)
        row = {
            "panel_index": panel_index,
            "sample_id": metadata["sample_id"],
            "ordinal": int(dataset_config["sample_ordinals"][panel_index]),
            "duration_sec": samples / 44_100.0,
            "latent_frames_valid": frames,
            "source_kinds": [source["kind"] for source in sceneplan["sources"]],
            "sceneplan": sceneplan,
            "semantic_text": metadata["prompt_text"],
            "exact_transcript": speech[0]["transcript"] if speech else None,
            "generated_foa_path": str(generated_path.resolve()),
            "reference_foa_path": str(reference_path.resolve()),
            "generated_stereo_path": str((sample_root / "generated_stereo.wav").resolve()),
            "reference_stereo_path": str((sample_root / "reference_vae_stereo.wav").resolve()),
            "generated_qc": audio_qc(generated),
            "reference_qc": audio_qc(reference),
            "generated_activity": _activity_metrics(
                generated,
                sceneplan,
                model_num_samples=samples,
                latent_frames=frames,
            ),
            "reference_activity": _activity_metrics(
                reference,
                sceneplan,
                model_num_samples=samples,
                latent_frames=frames,
            ),
            "spatial_metric_eligible": spatial_eligible,
            "generated_doa": (
                _doa_metrics(
                    generated,
                    sceneplan,
                    model_num_samples=samples,
                    latent_frames=frames,
                )
                if spatial_eligible
                else None
            ),
            "reference_doa": (
                _doa_metrics(
                    reference,
                    sceneplan,
                    model_num_samples=samples,
                    latent_frames=frames,
                )
                if spatial_eligible
                else None
            ),
            "sampling_seconds": round(time.monotonic() - started, 3),
        }
        rows.append(row)
        atomic_json(sample_root / "metadata.json", row)
        print(
            json.dumps(
                {
                    "event": "generated",
                    "row": panel_index + 1,
                    "sample_id": metadata["sample_id"],
                    "frames": frames,
                    "seconds": row["sampling_seconds"],
                }
            ),
            flush=True,
        )

    del sample_model, wrapper, model, dataset, diffusion
    gc.collect()
    torch.cuda.empty_cache()

    clap = load_clap_model(str(CLAP_CHECKPOINT.resolve(strict=True)), device=str(device))
    audio_items = []
    captions = {}
    for row in rows:
        key = row["sample_id"]
        audio_items.extend(
            [
                (f"generated:{key}", row["generated_foa_path"]),
                (f"reference:{key}", row["reference_foa_path"]),
            ]
        )
        captions[key] = row["semantic_text"]
    audio_embeddings = _audio_embeddings(clap, audio_items, device)
    text_embeddings = _text_embeddings(clap, captions)
    for row in rows:
        key = row["sample_id"]
        generated_embedding = audio_embeddings[f"generated:{key}"]
        reference_embedding = audio_embeddings[f"reference:{key}"]
        text_embedding = text_embeddings[key]
        row["clap"] = {
            "generated_text_cosine": float(generated_embedding @ text_embedding),
            "reference_text_cosine": float(reference_embedding @ text_embedding),
            "generated_reference_audio_cosine": float(
                generated_embedding @ reference_embedding
            ),
        }
    del clap
    gc.collect()
    torch.cuda.empty_cache()

    from faster_whisper import WhisperModel

    whisper = WhisperModel(
        str(DEFAULT_WHISPER.resolve(strict=True)),
        device="cuda",
        device_index=device.index or 0,
        compute_type="float16",
    )
    speech_rows = [row for row in rows if row["exact_transcript"]]
    for row in speech_rows:
        generated_asr = _transcribe(whisper, _mono_16k(row["generated_foa_path"]))
        reference_asr = _transcribe(whisper, _mono_16k(row["reference_foa_path"]))
        row["speech_metrics"] = {
            "generated_asr": generated_asr,
            "generated_errors": _error_rates(
                generated_asr["text"], row["exact_transcript"]
            ),
            "reference_asr": reference_asr,
            "reference_errors": _error_rates(
                reference_asr["text"], row["exact_transcript"]
            ),
        }
    del whisper
    gc.collect()
    torch.cuda.empty_cache()

    all_audio_valid = all(
        row["generated_qc"]["finite"]
        and row["generated_qc"]["rms"] > 1.0e-5
        and row["generated_qc"]["fraction_abs_ge_1"] < 0.1
        for row in rows
    )
    paired_clap = mean(
        row["clap"]["generated_reference_audio_cosine"] for row in rows
    )
    clap_deficit = mean(
        row["clap"]["reference_text_cosine"]
        - row["clap"]["generated_text_cosine"]
        for row in rows
    )
    generated_wer = mean(
        row["speech_metrics"]["generated_errors"]["wer"] for row in speech_rows
    )
    reference_wer = mean(
        row["speech_metrics"]["reference_errors"]["wer"] for row in speech_rows
    )
    spatial_rows = [row for row in rows if row["spatial_metric_eligible"]]
    generated_doa = mean(
        row["generated_doa"]["spherical_error_mean_deg"] for row in spatial_rows
    )
    reference_doa = mean(
        row["reference_doa"]["spherical_error_mean_deg"] for row in spatial_rows
    )
    direction_fraction = mean(
        row["generated_doa"]["valid_direction_fraction"] for row in spatial_rows
    )
    activity_iou = mean(
        row["generated_activity"]["temporal_iou"] for row in rows
    )
    audio_gate = bool(all_audio_valid and paired_clap >= 0.65)
    semantic_gate = bool(
        clap_deficit <= 0.10
        and generated_wer <= 0.50
        and generated_wer <= reference_wer + 0.25
    )
    spatial_gate = bool(
        generated_doa <= max(35.0, reference_doa + 15.0)
        and direction_fraction >= 0.65
        and activity_iou >= 0.70
    )
    report = {
        "schema": "stable_audio_tools.sceneplan_44_revision6_overfit_gate",
        "schema_version": 1,
        "status": (
            "PASS"
            if all(
                (
                    audio_gate,
                    semantic_gate,
                    spatial_gate,
                    receipts["cfg_ok"],
                    receipts["health_ok"],
                )
            )
            else "FAIL"
        ),
        "architecture": "semantic_cross_attention_plus_4_event_plus_4_trajectory",
        "dataset_contract_revision": 6,
        "initialization": "from_scratch",
        "forced_aligner": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "rows": len(rows),
        "short_rows": sum(row["latent_frames_valid"] <= 432 for row in rows),
        "long_rows": sum(row["latent_frames_valid"] > 432 for row in rows),
        "sequential_rows": sum(
            row["source_kinds"] in (["speech", "sound"], ["sound", "speech"], ["speech", "music"], ["music", "speech"])
            and row["spatial_metric_eligible"]
            for row in rows
        ),
        "audio_gate": audio_gate,
        "semantic_gate": semantic_gate,
        "spatial_temporal_gate": spatial_gate,
        "independent_cfg_gate": receipts["cfg_ok"],
        "training_health_gate": receipts["health_ok"],
        "aggregates": {
            "paired_generated_reference_clap_cosine": paired_clap,
            "clap_text_deficit_vs_vae_reference": clap_deficit,
            "speech_generated_wer": generated_wer,
            "speech_vae_reference_wer": reference_wer,
            "spatial_eligible_rows": len(spatial_rows),
            "generated_spherical_error_mean_deg": generated_doa,
            "vae_reference_spherical_error_mean_deg": reference_doa,
            "generated_valid_direction_fraction": direction_fraction,
            "activity_temporal_iou": activity_iou,
            "loss_first_mean": receipts["loss_first_mean"],
            "loss_last_mean": receipts["loss_last_mean"],
        },
        "thresholds": {
            "paired_clap_audio_cosine_min": 0.65,
            "clap_text_deficit_max": 0.10,
            "speech_wer_max": 0.50,
            "speech_wer_excess_over_vae_max": 0.25,
            "spherical_error_max_deg": max(35.0, reference_doa + 15.0),
            "valid_direction_fraction_min": 0.65,
            "activity_iou_min": 0.70,
            "loss_last_over_first_max": 0.70,
        },
        "training_receipts": receipts,
        "listening_manifest": str((output / "LISTENING_MANIFEST.jsonl").resolve()),
    }
    atomic_jsonl(output / "LISTENING_MANIFEST.jsonl", rows)
    atomic_json(output / "EVALUATION.json", report)
    atomic_json(root / "OVERFIT_GATE.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
