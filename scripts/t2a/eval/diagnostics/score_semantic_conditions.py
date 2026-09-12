#!/usr/bin/env python3
"""Score semantic-condition intervention WAVs with the evaluator's CLAP model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torchaudio

from stable_audio_tools.training.metrics.fad_metrics import load_clap_model

from scripts.t2a.eval.diagnostics.diagnose_spatial_conditions import _atomic_json
from scripts.t2a.eval.evaluate_spatial_cot_checkpoint import (
    _silence_alignment_metrics,
)


CORRECT_CONDITION = "plan_semantics_correct__caption_correct"


def _load_foa(path: Path) -> tuple[torch.Tensor, int]:
    audio, sample_rate = torchaudio.load(str(path))
    if audio.ndim != 2 or audio.shape[0] != 4:
        raise ValueError(f"expected four-channel FOA at {path}, got {tuple(audio.shape)}")
    if not bool(torch.isfinite(audio).all()):
        raise ValueError(f"non-finite FOA at {path}")
    return audio.float(), int(sample_rate)


def _load_w_channel(path: Path, *, device: torch.device) -> torch.Tensor:
    audio, sample_rate = _load_foa(path)
    mono = audio[:1].float().to(device)
    peak = mono.abs().amax().clamp_min(1.0e-8)
    mono = mono / peak * (10.0 ** (-1.0 / 20.0))
    if sample_rate != 48_000:
        mono = torchaudio.functional.resample(mono, sample_rate, 48_000)
    return mono.clamp(-1.0, 1.0)


def _cosine_matrix(audio_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
    audio_embeddings = torch.nn.functional.normalize(audio_embeddings.float(), dim=-1)
    text_embeddings = torch.nn.functional.normalize(text_embeddings.float(), dim=-1)
    return audio_embeddings @ text_embeddings.transpose(0, 1)


def _target_fidelity_summary(
    report: dict[str, Any],
    *,
    generated_audio: torch.Tensor,
    target_audio: torch.Tensor,
) -> dict[str, Any]:
    """Expose target-relative acoustic quality beside semantic CLAP scores.

    CLAP can reward the right event even when the waveform is noisy, spectrally
    damaged, or fills a target-silent tail.  These values are diagnostic rather
    than a perceptual scalar, but make that failure visible in the same result.
    """

    condition = report.get("conditions", {}).get(CORRECT_CONDITION)
    if not isinstance(condition, dict) or not isinstance(
        condition.get("vs_target"), dict
    ):
        raise ValueError(
            "semantic diagnostic must contain target-relative metrics for the "
            f"correct condition {CORRECT_CONDITION!r}"
        )
    target_metrics = condition["vs_target"]
    latent = target_metrics.get("latent")
    audio = target_metrics.get("audio")
    field = target_metrics.get("field")
    if not all(isinstance(value, dict) for value in (latent, audio, field)):
        raise ValueError("target-relative latent/audio/field metrics are incomplete")
    correlations = audio.get("channel_correlations_wyzx")
    if not isinstance(correlations, list) or len(correlations) != 4:
        raise ValueError("target-relative audio metrics require four correlations")
    hop = int(report.get("settings", {}).get("downsampling_ratio", 0))
    if hop <= 0:
        raise ValueError("semantic diagnostic is missing a positive codec hop")
    silence = _silence_alignment_metrics(
        generated_audio,
        target_audio,
        frame_samples=hop,
    )
    return {
        "condition": CORRECT_CONDITION,
        "latent_relative_rmse": float(latent["relative_rmse"]),
        "audio_relative_rmse": float(audio["relative_rmse"]),
        "audio_cosine": float(audio["cosine"]),
        "audio_multiresolution_log_spectral_mae": float(
            audio["multiresolution_log_spectral_mae"]
        ),
        "audio_channel_correlation_mean": float(sum(correlations) / 4.0),
        "audio_channel_correlation_min": float(min(correlations)),
        "field_angular_error_mean_deg": field.get("angular_error_mean_deg"),
        "field_diffuseness_mae": float(field["diffuseness_mae"]),
        "field_valid_direction_fraction": float(
            field["valid_direction_fraction"]
        ),
        "silence_alignment": silence,
    }


@torch.inference_mode()
def _score_report(clap_model, result_path: Path) -> dict[str, Any]:
    report = json.loads(result_path.read_text(encoding="utf-8"))
    if report.get("schema") != "stable_audio_tools.semantic_condition_diagnostic":
        raise ValueError(f"not a semantic diagnostic result: {result_path}")
    device = next(clap_model.model.parameters()).device
    condition_names = list(report["conditions"])
    audio_names = ["target", *condition_names]
    audio_paths = [Path(report["target_audio_path"])] + [
        Path(report["conditions"][name]["audio_path"]) for name in condition_names
    ]
    audio_values = [_load_w_channel(path, device=device) for path in audio_paths]
    sample_counts = {int(value.shape[-1]) for value in audio_values}
    if len(sample_counts) != 1:
        raise ValueError(f"CLAP inputs have inconsistent lengths: {sorted(sample_counts)}")
    captions = [report["correct_caption"], report["donor_caption"]]
    audio_embeddings = clap_model.get_audio_embedding_from_data(
        x=torch.cat(audio_values, dim=0), use_tensor=True
    )
    text_embeddings = clap_model.get_text_embedding(captions, use_tensor=True)
    text_cosines = _cosine_matrix(audio_embeddings, text_embeddings).cpu()
    normalized_audio = torch.nn.functional.normalize(audio_embeddings.float(), dim=-1)
    target_cosines = (normalized_audio @ normalized_audio[0:1].transpose(0, 1))[
        :, 0
    ].cpu()

    rows: dict[str, Any] = {}
    for index, name in enumerate(audio_names):
        correct_score = float(text_cosines[index, 0])
        donor_score = float(text_cosines[index, 1])
        rows[name] = {
            "correct_caption_cosine": correct_score,
            "donor_caption_cosine": donor_score,
            "donor_minus_correct_caption_margin": donor_score - correct_score,
            "target_audio_cosine": float(target_cosines[index]),
            "audio_path": str(audio_paths[index]),
        }

    cc = CORRECT_CONDITION
    cd = "plan_semantics_correct__caption_donor"
    dc = "plan_semantics_donor__caption_correct"
    dd = "plan_semantics_donor__caption_donor"
    margin = lambda name: rows[name]["donor_minus_correct_caption_margin"]
    target_audio, target_rate = _load_foa(audio_paths[0])
    generated_audio, generated_rate = _load_foa(audio_paths[1 + condition_names.index(cc)])
    if target_rate != generated_rate:
        raise ValueError(
            f"target/generated sample rates differ: {target_rate} != {generated_rate}"
        )
    return {
        "schema": "stable_audio_tools.semantic_condition_clap_scores",
        "schema_version": 1,
        "source_result": str(result_path.resolve()),
        "family_rank": report["family_rank"],
        "family_id": report["family_id"],
        "donor_family_rank": report["donor_family_rank"],
        "donor_family_id": report["donor_family_id"],
        "captions": {"correct": captions[0], "donor": captions[1]},
        "scores": rows,
        "causal_margin_shifts": {
            "caption_swap_with_correct_plan_semantics": margin(cd) - margin(cc),
            "caption_swap_with_donor_plan_semantics": margin(dd) - margin(dc),
            "plan_semantic_swap_with_correct_caption": margin(dc) - margin(cc),
            "plan_semantic_swap_with_donor_caption": margin(dd) - margin(cd),
        },
        "target_fidelity": _target_fidelity_summary(
            report,
            generated_audio=generated_audio,
            target_audio=target_audio,
        ),
        "interpretation": (
            "Positive shifts mean the intervention moved generated audio toward the "
            "donor caption relative to the correct caption in CLAP space."
        ),
        "channel": "W",
        "sample_rate": 48_000,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clap-model", default="630k-audioset-fusion-best.pt")
    args = parser.parse_args()
    for path in args.result_json:
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA scoring requested but unavailable")
    clap_model = load_clap_model(args.clap_model, device=str(device))
    summaries = []
    for result_path in args.result_json:
        score = _score_report(clap_model, result_path)
        output_path = result_path.parent / "CONTENT_METRICS.json"
        score["clap_model"] = args.clap_model
        _atomic_json(output_path, score)
        summaries.append(
            {
                "family_id": score["family_id"],
                "output": str(output_path),
                "causal_margin_shifts": score["causal_margin_shifts"],
            }
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
