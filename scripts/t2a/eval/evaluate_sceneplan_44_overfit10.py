#!/usr/bin/env python3
"""Generate and score the fixed ten-sample ScenePlan 4+4 overfit gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import sys
import time
import zlib
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# Direct script execution puts ``scripts/t2a/eval`` on sys.path, not the
# repository root.  Make the checked-out implementation importable without
# relying on a caller-specific PYTHONPATH.
REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.t2a.eval.sceneplan_44_eval_common import (
    atomic_wav,
    audio_qc,
    virtual_stereo,
)
from scripts.t2a.eval.score_sceneplan_dit_p10_clap import (
    _audio_embeddings,
    _text_embeddings,
)
from scripts.t2a.eval.score_sceneplan_dit_p10_core import (
    _activity_metrics,
    _doa_metrics,
)
from scripts.t2a.eval.score_sceneplan_dit_p10_speech import (
    DEFAULT_WHISPER,
    _error_rates,
    _mono_16k,
    _transcribe,
)
from stable_audio_tools.configuration import load_config
from stable_audio_tools.data.model_sceneplan import (
    make_sceneplan_cfg_unknown_metadata,
)
from stable_audio_tools.data.sceneplan_v2_dataset import ScenePlanV2Dataset
from stable_audio_tools.data.text_conditioning import collect_conditioner_tokenizers
from stable_audio_tools.inference.sampling import sample_diffusion
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.training.factory import create_training_wrapper_from_config
from stable_audio_tools.training.metrics.fad_metrics import load_clap_model


DEFAULT_RUN = Path(
    "/mnt/sdb/model_archives/p10_pre_v11_20260831/pilots/"
    "sceneplan_dit_v4_r10_44_scratch_indcfg15_overfit10"
)
DATASET_CONFIG = REPO / (
    "stable_audio_tools/configs/dataset_configs/sceneplan_44_overfit10.json"
)
VAE_CHECKPOINT = Path(
    "/mnt/sdc/ckpts/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt"
)
CLAP_CHECKPOINT = REPO / "load/clap_score/630k-audioset-fusion-best.pt"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _crop_metadata(metadata: dict[str, Any], frames: int) -> dict[str, Any]:
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


def _latest_checkpoint(root: Path) -> Path:
    candidates = []
    for path in (root / "checkpoints").glob("*step=*.ckpt"):
        try:
            step = int(path.stem.rsplit("step=", 1)[1])
        except ValueError:
            continue
        candidates.append((step, path))
    if not candidates:
        last = root / "checkpoints/last.ckpt"
        if last.is_file():
            return last.resolve(strict=True)
        raise FileNotFoundError(f"no checkpoint below {root / 'checkpoints'}")
    return max(candidates)[1].resolve(strict=True)


def _sceneplan_rows(index: Path, ordinals: list[int]) -> list[dict[str, Any]]:
    connection = sqlite3.connect(
        f"file:{index}?mode=ro&immutable=1", uri=True, check_same_thread=False
    )
    output = []
    for ordinal in ordinals:
        row = connection.execute(
            """
            SELECT sample_id, model_num_samples, latent_frames_valid,
                   scene_plan_zlib FROM samples WHERE ordinal=?
            """,
            (int(ordinal),),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"missing overfit ordinal {ordinal}")
        sample_id, samples, frames, compressed = row
        sceneplan = json.loads(zlib.decompress(compressed))
        output.append(
            {
                "ordinal": int(ordinal),
                "sample_id": sample_id,
                "model_num_samples": int(samples),
                "latent_frames": int(frames),
                "scene_plan": sceneplan,
            }
        )
    connection.close()
    return output


def _si_sdr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    reference = reference.double().flatten()
    estimate = estimate.double().flatten()
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    projection = (
        torch.dot(estimate, reference)
        / reference.square().sum().clamp_min(1.0e-12)
    ) * reference
    residual = estimate - projection
    return float(
        10.0
        * torch.log10(
            projection.square().sum().clamp_min(1.0e-12)
            / residual.square().sum().clamp_min(1.0e-12)
        )
    )


def _log_stft_l1(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    values = []
    for n_fft in (512, 1024, 2048):
        hop = n_fft // 4
        window = torch.hann_window(n_fft)
        left = torch.stft(
            reference[0].float(), n_fft, hop, window=window, return_complex=True
        ).abs()
        right = torch.stft(
            estimate[0].float(), n_fft, hop, window=window, return_complex=True
        ).abs()
        values.append((torch.log1p(left) - torch.log1p(right)).abs().mean())
    return float(torch.stack(values).mean())


def _domain(sceneplan: dict[str, Any]) -> str:
    if len(sceneplan["sources"]) != 1:
        raise RuntimeError("overfit evaluation requires single-source ScenePlans")
    return str(sceneplan["sources"][0]["kind"])


def _semantic_text(sceneplan: dict[str, Any]) -> str:
    source = sceneplan["sources"][0]
    return str(
        source["transcript"]
        if source["kind"] == "speech"
        else source["description"]
    )


def _mean(values) -> float:
    values = [float(value) for value in values if value is not None]
    return float(sum(values) / len(values)) if values else float("nan")


def _independent_cfg_training_receipt(run_root: Path) -> tuple[dict[str, Any], bool]:
    """Load and validate the aggregate emitted by the finished overfit run."""

    train_log = run_root / "logs/train.log"
    raw = train_log.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"SAT_SCENEPLAN_CFG_DROPOUT_RESULT=(\{[^\r\n]+\})", raw
    )
    if len(matches) != 1:
        raise RuntimeError(
            "overfit training must emit exactly one independent-CFG receipt; "
            f"found {len(matches)}"
        )
    receipt = json.loads(matches[0])
    configured = receipt.get("configured", {})
    observed = receipt.get("observed_fractions", {})
    caption = float(observed.get("caption_unknown_fraction", float("nan")))
    structured = float(
        observed.get("structured_unknown_fraction", float("nan"))
    )
    joint = float(observed.get("joint_unknown_fraction", float("nan")))
    full = float(observed.get("full_condition_fraction", float("nan")))
    probability_mass = full + caption + structured - joint
    ok = bool(
        receipt.get("mode") == "independent"
        and int(receipt.get("samples", 0)) >= 20_000
        and configured
        == {
            "caption_unknown_prob": 0.15,
            "structured_unknown_prob": 0.15,
        }
        and 0.14 <= caption <= 0.16
        and 0.14 <= structured <= 0.16
        and 0.0175 <= joint <= 0.0275
        and 0.71 <= full <= 0.735
        and math.isclose(probability_mass, 1.0, abs_tol=1.0e-9)
        and abs(joint - caption * structured) <= 0.005
    )
    return receipt, ok


def _candidate_auxiliary_training_receipt(
    run_root: Path,
) -> tuple[dict[str, Any], bool, bool]:
    """Validate that the new alignment and Sound objectives actually trained."""

    raw = (run_root / "logs/train.log").read_text(
        encoding="utf-8", errors="replace"
    )
    matches = re.findall(
        r"SAT_TRAINING_GATE_RESULT=(\{[^\r\n]+\})", raw
    )
    if len(matches) != 1:
        raise RuntimeError(
            "overfit training must emit exactly one health-gate receipt; "
            f"found {len(matches)}"
        )
    receipt = json.loads(matches[0])
    objectives = dict(receipt.get("objectives") or {})
    windows = dict(receipt.get("metric_windows") or {})
    required = {
        "train/speech_duration_kl",
        "train/speech_duration_uniform_kl",
        "train/speech_duration_teacher_rows",
        "train/sceneplan_alignment_nonzero_fraction",
        "train/sound_temporal_difference_aux",
        "train/sound_temporal_difference_eligible_fraction",
    }
    missing = required - set(objectives)
    if missing:
        raise RuntimeError(
            "candidate health gate omitted Qwen-alignment/Sound metrics: "
            f"{sorted(missing)}"
        )

    def finite(name: str) -> float:
        value = float(objectives[name])
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite candidate objective {name}={value}")
        return value

    def window_mean(name: str, which: str) -> float:
        value = float(windows[name][which])
        if not math.isfinite(value):
            raise RuntimeError(
                f"non-finite candidate objective window {name}.{which}={value}"
            )
        return value

    duration_first = window_mean("train/speech_duration_kl", "first_mean")
    duration_last = window_mean("train/speech_duration_kl", "last_mean")
    uniform_last = window_mean(
        "train/speech_duration_uniform_kl", "last_mean"
    )
    teacher_last = window_mean(
        "train/speech_duration_teacher_rows", "last_mean"
    )
    alignment_nonzero_last = window_mean(
        "train/sceneplan_alignment_nonzero_fraction", "last_mean"
    )
    sound_first = window_mean(
        "train/sound_temporal_difference_aux", "first_mean"
    )
    sound_last = window_mean(
        "train/sound_temporal_difference_aux", "last_mean"
    )
    sound_eligible_last = window_mean(
        "train/sound_temporal_difference_eligible_fraction", "last_mean"
    )
    alignment_gate = bool(
        receipt.get("status") == "PASS"
        and int(receipt.get("global_step", 0)) == 2000
        and finite("train/speech_duration_teacher_rows") >= 0.0
        and teacher_last > 0.0
        and alignment_nonzero_last > 0.0
        and duration_first > 0.0
        and duration_last < duration_first * 0.80
        and duration_last < uniform_last * 0.80
    )
    sound_temporal_gate = bool(
        receipt.get("status") == "PASS"
        and finite("train/sound_temporal_difference_eligible_fraction") > 0.0
        and sound_eligible_last > 0.0
        and sound_first > 0.0
        and sound_last < sound_first * 0.80
    )
    return receipt, alignment_gate, sound_temporal_gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--skip-clap", action="store_true")
    parser.add_argument("--skip-asr", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve(strict=True)
    contract = json.loads(
        (run_root / "SCENEPLAN_44_RUN_CONTRACT.json").read_text(encoding="utf-8")
    )
    if not (
        contract.get("schema_version", 0) >= 5
        and contract.get("mode") == "overfit10"
        and contract.get("initialization")
        == "random_seeded_sceneplan_44_from_scratch"
        and contract.get("warm_start") is False
        and contract.get("pretrained_checkpoint") is None
        and contract.get("pretrained_route_weights") is None
        and contract.get("pretrained_route_expectation") is None
    ):
        raise RuntimeError("overfit evaluation refuses a non-scratch run")
    model_config = Path(contract["model_config"]).expanduser().resolve(strict=True)
    if _sha256(model_config) != contract.get("model_config_sha256"):
        raise RuntimeError("model config changed after the overfit contract was frozen")
    cfg_dropout_receipt, cfg_dropout_gate = (
        _independent_cfg_training_receipt(run_root)
    )
    (
        candidate_training_receipt,
        alignment_gate,
        sound_temporal_gate,
    ) = _candidate_auxiliary_training_receipt(run_root)
    checkpoint = (
        args.checkpoint.expanduser().resolve(strict=True)
        if args.checkpoint is not None
        else _latest_checkpoint(run_root)
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the overfit gate requires CUDA")
    config = load_config(model_config)
    dataset_config_path = Path(contract["dataset_config"]).expanduser().resolve(
        strict=True
    )
    if _sha256(dataset_config_path) != contract.get("dataset_config_sha256"):
        raise RuntimeError("overfit dataset config changed after contract freeze")
    dataset_config = load_config(dataset_config_path)
    if not (
        dataset_config.get("require_speech_timing") is True
        and dataset_config.get("expected_speech_timing_rows") == 500_000
        and dataset_config.get("sample_contract")
        == {
            "music": 3,
            "new_sound_replacements": 3,
            "speech": 4,
            "single_source_only": True,
        }
    ):
        raise RuntimeError("evaluation is not using the integrated Sound/Qwen panel")
    ordinals = [int(value) for value in dataset_config["sample_ordinals"]]
    index = Path(dataset_config["datasets"][0]["path"]).resolve(strict=True)
    panel = _sceneplan_rows(index, ordinals)
    output_root = run_root / "evaluation/overfit10"

    print(json.dumps({"event": "load", "checkpoint": str(checkpoint)}), flush=True)
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
    vae_state = load_ckpt_state_dict(str(VAE_CHECKPOINT.resolve(strict=True)))
    if hasattr(model, "load_pretransform_state_dict"):
        model.load_pretransform_state_dict(vae_state, strict=False)
    elif model.pretransform is not None:
        incompatible_vae = model.pretransform.load_state_dict(vae_state, strict=False)
        if (
            incompatible_vae is not None
            and incompatible_vae.unexpected_keys
        ):
            raise RuntimeError(
                "VAE checkpoint has unexpected keys: "
                f"{incompatible_vae.unexpected_keys[:12]}"
            )
    else:
        raise RuntimeError("ScenePlan DiT has no VAE pretransform to decode FOA")
    del vae_state

    tokenizers = collect_conditioner_tokenizers(model, config)
    dataset = ScenePlanV2Dataset(
        index,
        tokenizer_spec=tokenizers["prompt"],
        expected_num_samples=10,
        index_num_samples=1_100_000,
        sample_ordinals=ordinals,
    )
    diffusion = wrapper.diffusion
    if wrapper.diffusion_ema is None or wrapper.conditioner_ema is None:
        raise RuntimeError("overfit checkpoint must contain DiT and conditioner EMA")
    sample_model = wrapper.diffusion_ema.ema_model.to(device).eval().requires_grad_(False)
    diffusion.conditioner.to(device).eval().requires_grad_(False)
    diffusion.pretransform.to(device).eval().requires_grad_(False)
    dtype = next(sample_model.parameters()).dtype
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)

    rows: list[dict[str, Any]] = []
    for index_in_panel, panel_row in enumerate(panel):
        started = time.monotonic()
        latent, metadata = dataset[index_in_panel]
        if metadata["sample_id"] != panel_row["sample_id"]:
            raise RuntimeError("overfit dataset order changed")
        frames = panel_row["latent_frames"]
        samples = panel_row["model_num_samples"]
        positive = _crop_metadata(metadata, frames)
        negative = make_sceneplan_cfg_unknown_metadata(positive)
        seed = 44_000 + index_in_panel
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(
            (1, int(diffusion.io_channels), frames),
            generator=generator,
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
                padding_mask=torch.ones(1, frames, dtype=torch.bool, device=device),
                dist_shift=diffusion.sampling_dist_shift,
                sampler_type="euler",
                batch_cfg=True,
                rescale_cfg=True,
                apg_scale=0.0,
                decode=True,
                disable_tqdm=True,
            )[0, :, :samples]
            reference = diffusion.pretransform.decode(
                latent[:, :frames].unsqueeze(0).to(device=device, dtype=dtype)
            )[0, :, :samples]
        generated = generated.float().cpu().contiguous()
        reference = reference.float().cpu().contiguous()
        generated_qc = audio_qc(generated)
        reference_qc = audio_qc(reference)
        if not generated_qc.get("finite") or not reference_qc.get("finite"):
            raise RuntimeError(f"non-finite overfit output: {panel_row['sample_id']}")

        sample_root = output_root / f"{index_in_panel:02d}_{panel_row['sample_id']}"
        generated_path = sample_root / "generated_foa.wav"
        reference_path = sample_root / "reference_vae_foa.wav"
        generated_preview_path = sample_root / "generated_stereo.wav"
        reference_preview_path = sample_root / "reference_vae_stereo.wav"
        generated_preview, _ = virtual_stereo(generated)
        reference_preview, _ = virtual_stereo(reference)
        atomic_wav(generated_path, generated, 44_100, subtype="FLOAT")
        atomic_wav(reference_path, reference, 44_100, subtype="FLOAT")
        atomic_wav(
            generated_preview_path, generated_preview, 44_100, subtype="PCM_16"
        )
        atomic_wav(
            reference_preview_path, reference_preview, 44_100, subtype="PCM_16"
        )
        sceneplan = panel_row["scene_plan"]
        row = {
            **panel_row,
            "panel_index": index_in_panel,
            "domain": _domain(sceneplan),
            "semantic_text": _semantic_text(sceneplan),
            "semantic_caption": metadata["prompt_text"],
            "generated_foa_path": str(generated_path.resolve()),
            "reference_foa_path": str(reference_path.resolve()),
            "generated_stereo_path": str(generated_preview_path.resolve()),
            "reference_stereo_path": str(reference_preview_path.resolve()),
            "generated_qc": generated_qc,
            "reference_qc": reference_qc,
            "w_si_sdr_db": _si_sdr(reference[0], generated[0]),
            "w_log_stft_l1": _log_stft_l1(reference, generated),
            "generated_doa": _doa_metrics(
                generated,
                sceneplan,
                model_num_samples=samples,
                latent_frames=frames,
            ),
            "reference_doa": _doa_metrics(
                reference,
                sceneplan,
                model_num_samples=samples,
                latent_frames=frames,
            ),
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
            "sampling_seconds": round(time.monotonic() - started, 3),
        }
        rows.append(row)
        _atomic_json(sample_root / "metadata.json", row)
        print(
            json.dumps(
                {
                    "event": "generated",
                    "index": index_in_panel + 1,
                    "sample_id": panel_row["sample_id"],
                    "domain": row["domain"],
                    "doa_error": row["generated_doa"][
                        "spherical_error_mean_deg"
                    ],
                    "seconds": row["sampling_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    del sample_model, wrapper, model, dataset, diffusion
    gc.collect()
    torch.cuda.empty_cache()

    nonspeech = [row for row in rows if row["domain"] in {"music", "sound"}]
    if not args.skip_clap:
        clap = load_clap_model(str(CLAP_CHECKPOINT.resolve(strict=True)), device=str(device))
        items = []
        captions = {}
        for row in nonspeech:
            key = row["sample_id"]
            items.extend(
                [
                    (f"generated:{key}", row["generated_foa_path"]),
                    (f"reference:{key}", row["reference_foa_path"]),
                ]
            )
            captions[key] = row["semantic_text"]
        audio_embeddings = _audio_embeddings(clap, items, device)
        text_embeddings = _text_embeddings(clap, captions)
        for row in nonspeech:
            key = row["sample_id"]
            generated_embedding = audio_embeddings[f"generated:{key}"]
            reference_embedding = audio_embeddings[f"reference:{key}"]
            text_embedding = text_embeddings[key]
            candidates = [
                other for other in nonspeech if other["domain"] == row["domain"]
            ]
            scores = torch.stack(
                [text_embeddings[other["sample_id"]] for other in candidates]
            ) @ generated_embedding
            row["clap"] = {
                "generated_text_cosine": float(generated_embedding @ text_embedding),
                "reference_text_cosine": float(reference_embedding @ text_embedding),
                "generated_reference_audio_cosine": float(
                    generated_embedding @ reference_embedding
                ),
                "within_domain_retrieval_correct": (
                    candidates[int(scores.argmax())]["sample_id"] == key
                ),
            }
        del clap
        gc.collect()
        torch.cuda.empty_cache()

    speech = [row for row in rows if row["domain"] == "speech"]
    if not args.skip_asr:
        from faster_whisper import WhisperModel

        whisper = WhisperModel(
            str(DEFAULT_WHISPER.resolve(strict=True)),
            device="cuda",
            device_index=device.index or 0,
            compute_type="float16",
        )
        for row in speech:
            generated_asr = _transcribe(
                whisper, _mono_16k(row["generated_foa_path"])
            )
            reference_asr = _transcribe(
                whisper, _mono_16k(row["reference_foa_path"])
            )
            row["speech_metrics"] = {
                "exact_transcript": row["semantic_text"],
                "generated_asr": generated_asr,
                "generated_errors": _error_rates(
                    generated_asr["text"], row["semantic_text"]
                ),
                "reference_asr": reference_asr,
                "reference_errors": _error_rates(
                    reference_asr["text"], row["semantic_text"]
                ),
            }
        del whisper
        gc.collect()
        torch.cuda.empty_cache()

    generated_doa = _mean(
        row["generated_doa"]["spherical_error_mean_deg"] for row in rows
    )
    reference_doa = _mean(
        row["reference_doa"]["spherical_error_mean_deg"] for row in rows
    )
    generated_direction_fraction = _mean(
        row["generated_doa"]["valid_direction_fraction"] for row in rows
    )
    speech_activity_iou = _mean(
        row["generated_activity"]["temporal_iou"] for row in speech
    )
    all_audio_valid = all(
        row["generated_qc"]["finite"]
        and row["generated_qc"]["rms"] > 1.0e-5
        and row["generated_qc"]["fraction_abs_ge_1"] < 0.1
        for row in rows
    )

    clap_audio_cosine = _mean(
        row.get("clap", {}).get("generated_reference_audio_cosine")
        for row in nonspeech
    )
    clap_deficit = _mean(
        row.get("clap", {}).get("reference_text_cosine", 0.0)
        - row.get("clap", {}).get("generated_text_cosine", 0.0)
        for row in nonspeech
        if "clap" in row
    )
    retrieval = _mean(
        row.get("clap", {}).get("within_domain_retrieval_correct")
        for row in nonspeech
        if "clap" in row
    )
    generated_wer = _mean(
        row.get("speech_metrics", {}).get("generated_errors", {}).get("wer")
        for row in speech
        if "speech_metrics" in row
    )
    reference_wer = _mean(
        row.get("speech_metrics", {}).get("reference_errors", {}).get("wer")
        for row in speech
        if "speech_metrics" in row
    )

    mean_si_sdr = _mean(row["w_si_sdr_db"] for row in rows)
    mean_log_stft = _mean(row["w_log_stft_l1"] for row in rows)
    audio_gate = bool(
        all_audio_valid
        and math.isfinite(clap_audio_cosine)
        and clap_audio_cosine >= 0.80
        and math.isfinite(mean_si_sdr)
        and mean_si_sdr >= 10.0
        and math.isfinite(mean_log_stft)
        and mean_log_stft <= 0.05
    )
    semantic_gate = bool(
        math.isfinite(clap_deficit)
        and clap_deficit <= 0.10
        and math.isfinite(retrieval)
        and retrieval >= 0.50
        and math.isfinite(generated_wer)
        and generated_wer <= 0.50
        and generated_wer <= reference_wer + 0.25
    )
    spatial_gate = bool(
        math.isfinite(generated_doa)
        and generated_doa <= max(35.0, reference_doa + 15.0)
        and generated_direction_fraction >= 0.70
        and speech_activity_iou >= 0.70
    )
    report = {
        "schema": "stable_audio_tools.sceneplan_44_overfit_gate",
        "schema_version": 5,
        "architecture": "semantic_cross_attention_plus_4+4",
        "initialization": "random_seeded_sceneplan_44_from_scratch",
        "warm_start": False,
        "ok": bool(
            audio_gate
            and semantic_gate
            and spatial_gate
            and cfg_dropout_gate
            and alignment_gate
            and sound_temporal_gate
        ),
        "checkpoint": str(checkpoint),
        "model_config": str(model_config),
        "model_config_sha256": _sha256(model_config),
        "rows": len(rows),
        "domain_counts": {
            domain: sum(row["domain"] == domain for row in rows)
            for domain in ("music", "sound", "speech")
        },
        "sampling": {"steps": args.steps, "cfg_scale": args.cfg_scale},
        "audio_gate": audio_gate,
        "semantic_gate": semantic_gate,
        "spatial_gate": spatial_gate,
        "cfg_dropout_gate": cfg_dropout_gate,
        "alignment_gate": alignment_gate,
        "sound_temporal_gate": sound_temporal_gate,
        "cfg_dropout": {
            "mode": "independent",
            "caption_unknown_prob": 0.15,
            "structured_unknown_prob": 0.15,
            "expected_joint_unknown_prob": 0.0225,
            "expected_full_condition_prob": 0.7225,
        },
        "cfg_dropout_training_receipt": cfg_dropout_receipt,
        "candidate_training_receipt": candidate_training_receipt,
        "aggregates": {
            "mean_w_si_sdr_db": mean_si_sdr,
            "mean_w_log_stft_l1": mean_log_stft,
            "nonspeech_paired_clap_audio_cosine": clap_audio_cosine,
            "nonspeech_clap_text_deficit_vs_vae_reference": clap_deficit,
            "nonspeech_within_domain_retrieval_top1": retrieval,
            "speech_generated_wer": generated_wer,
            "speech_vae_reference_wer": reference_wer,
            "generated_plan_spherical_error_mean_deg": generated_doa,
            "vae_reference_plan_spherical_error_mean_deg": reference_doa,
            "generated_valid_direction_fraction": generated_direction_fraction,
            "speech_activity_temporal_iou": speech_activity_iou,
        },
        "thresholds": {
            "paired_clap_audio_cosine_min": 0.80,
            "mean_w_si_sdr_db_min": 10.0,
            "mean_w_log_stft_l1_max": 0.05,
            "clap_text_deficit_max": 0.10,
            "retrieval_top1_min": 0.50,
            "speech_wer_max": 0.50,
            "speech_wer_excess_over_vae_reference_max": 0.25,
            "plan_spherical_error_max_deg": max(35.0, reference_doa + 15.0),
            "valid_direction_fraction_min": 0.70,
            "speech_activity_iou_min": 0.70,
            "duration_last_over_first_max": 0.80,
            "duration_last_over_uniform_max": 0.80,
            "sound_temporal_last_over_first_max": 0.80,
        },
        "listening_manifest": str(
            (output_root / "LISTENING_MANIFEST.jsonl").resolve()
        ),
    }
    _atomic_jsonl(output_root / "LISTENING_MANIFEST.jsonl", rows)
    _atomic_json(output_root / "EVALUATION.json", report)
    _atomic_json(run_root / "OVERFIT_GATE.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
