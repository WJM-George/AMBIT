#!/usr/bin/env python3
"""Transcribe semantic-condition WAVs to test causal speech control."""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torchaudio
from faster_whisper import WhisperModel

from scripts.t2a.eval.diagnostics.diagnose_spatial_conditions import _atomic_json


def _normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", value.lower()))


def _text_similarity(value: str, reference: str) -> float | None:
    if not reference.strip():
        return None
    return difflib.SequenceMatcher(
        None, _normalize_text(value), _normalize_text(reference)
    ).ratio()


def _speech_reference(semantics: list[dict[str, Any]]) -> str:
    return " ".join(
        str(source.get("transcript") or "").strip()
        for source in semantics
        if str(source.get("transcript") or "").strip()
    ).strip()


def _load_mono(path: Path, *, expected_channels: int) -> Any:
    audio, sample_rate = torchaudio.load(str(path))
    if audio.ndim != 2 or audio.shape[0] != expected_channels:
        raise ValueError(
            f"expected {expected_channels}-channel audio at {path}, "
            f"got {tuple(audio.shape)}"
        )
    mono = audio[:1].float()
    if sample_rate != 16_000:
        mono = torchaudio.functional.resample(mono, sample_rate, 16_000)
    return mono[0].numpy()


def _transcribe(
    model: WhisperModel,
    path: Path,
    *,
    expected_channels: int,
    vad_filter: bool,
) -> dict[str, Any]:
    segments_iterator, info = model.transcribe(
        _load_mono(path, expected_channels=expected_channels),
        language="en",
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=vad_filter,
    )
    segments = list(segments_iterator)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return {
        "transcript": text,
        "language": info.language,
        "language_probability": float(info.language_probability),
        "segment_count": len(segments),
        "mean_no_speech_probability": (
            None
            if not segments
            else sum(float(segment.no_speech_prob) for segment in segments)
            / len(segments)
        ),
        "mean_average_log_probability": (
            None
            if not segments
            else sum(float(segment.avg_logprob) for segment in segments) / len(segments)
        ),
    }


def _source_location_speech_scores(
    model: WhisperModel,
    *,
    report: dict[str, Any],
    source_location_path: Path,
) -> dict[str, Any]:
    location = json.loads(source_location_path.read_text(encoding="utf-8"))
    if location.get("schema") != "stable_audio_tools.source_location_semantic_scores":
        raise ValueError(f"not a source-location result: {source_location_path}")
    if location.get("family_id") != report.get("family_id"):
        raise ValueError("semantic and source-location family IDs disagree")
    semantics = {
        str(source["source_id"]): source
        for source in report["correct_plan_semantics"]
    }
    source_ids = list(location["source_ids"])
    if set(source_ids) != set(semantics):
        raise ValueError("semantic and source-location source IDs disagree")

    source_scores: dict[str, Any] = {}
    for source_id in source_ids:
        source_reference = str(semantics[source_id].get("transcript") or "").strip()
        paths = location["stem_paths"][source_id]
        values = {}
        for kind in ("target", "generated"):
            path = Path(paths[kind])
            value = _transcribe(
                model,
                path,
                expected_channels=1,
                vad_filter=True,
            )
            value["audio_path"] = str(path)
            value["similarity_to_source_speech"] = _text_similarity(
                value["transcript"], source_reference
            )
            value["similarity_to_all_correct_speech"] = _text_similarity(
                value["transcript"],
                _speech_reference(report["correct_plan_semantics"]),
            )
            values[kind] = value
        source_scores[source_id] = {
            "speech_reference": source_reference,
            **values,
        }

    assignments = []
    for reference_source_id in source_ids:
        reference = source_scores[reference_source_id]["speech_reference"]
        if not reference:
            continue
        assignment: dict[str, Any] = {
            "reference_source_id": reference_source_id,
            "speech_reference": reference,
        }
        for kind in ("target", "generated"):
            similarities = {
                source_id: _text_similarity(
                    source_scores[source_id][kind]["transcript"], reference
                )
                for source_id in source_ids
            }
            # The reference is non-empty, so every similarity is a float.
            best_source_id = max(
                source_ids,
                key=lambda source_id: float(similarities[source_id]),
            )
            assignment[kind] = {
                "best_source_id": best_source_id,
                "correct": best_source_id == reference_source_id,
                "best_similarity": float(similarities[best_source_id]),
                "reference_source_similarity": float(
                    similarities[reference_source_id]
                ),
                "similarities": similarities,
            }
        target_similarity = assignment["target"]["reference_source_similarity"]
        generated_similarity = assignment["generated"][
            "reference_source_similarity"
        ]
        assignment["generated_minus_target_similarity"] = (
            generated_similarity - target_similarity
        )
        assignment["generated_to_target_similarity_ratio"] = (
            None
            if target_similarity <= 1.0e-8
            else generated_similarity / target_similarity
        )
        assignments.append(assignment)
    return {
        "source_metrics": source_scores,
        "speech_assignments": assignments,
        "source_location_metrics": str(source_location_path.resolve()),
        "channel": "plan-guided mono directional stems",
        "sample_rate": 16_000,
        "interpretation": (
            "Speech assignment is target-calibrated evidence about which planned "
            "source trajectory contains a transcript; reverberant crosstalk and "
            "ASR uncertainty remain visible in the per-source target scores."
        ),
    }


def _score_report(
    model: WhisperModel,
    result_path: Path,
    *,
    require_source_location: bool = False,
) -> dict[str, Any]:
    report = json.loads(result_path.read_text(encoding="utf-8"))
    if report.get("schema") != "stable_audio_tools.semantic_condition_diagnostic":
        raise ValueError(f"not a semantic diagnostic result: {result_path}")
    references = {
        "correct": _speech_reference(report["correct_plan_semantics"]),
        "donor": _speech_reference(report["donor_plan_semantics"]),
    }
    audio_paths = {"target": Path(report["target_audio_path"])}
    audio_paths.update(
        {
            name: Path(value["audio_path"])
            for name, value in report["conditions"].items()
        }
    )
    scores = {}
    for name, audio_path in audio_paths.items():
        value = _transcribe(
            model,
            audio_path,
            expected_channels=4,
            vad_filter=False,
        )
        value["audio_path"] = str(audio_path)
        value["similarity_to_correct_speech"] = _text_similarity(
            value["transcript"], references["correct"]
        )
        value["similarity_to_donor_speech"] = _text_similarity(
            value["transcript"], references["donor"]
        )
        scores[name] = value
    source_location_path = result_path.parent / "SOURCE_LOCATION_METRICS.json"
    if require_source_location and not source_location_path.is_file():
        raise FileNotFoundError(source_location_path)
    source_location = (
        _source_location_speech_scores(
            model,
            report=report,
            source_location_path=source_location_path,
        )
        if source_location_path.is_file()
        else None
    )
    return {
        "schema": "stable_audio_tools.semantic_condition_speech_scores",
        "schema_version": 2,
        "source_result": str(result_path.resolve()),
        "family_rank": report["family_rank"],
        "family_id": report["family_id"],
        "donor_family_rank": report["donor_family_rank"],
        "donor_family_id": report["donor_family_id"],
        "speech_references": references,
        "scores": scores,
        "source_location": source_location,
        "channel": "W",
        "sample_rate": 16_000,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", nargs="+", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--require-source-location", action="store_true")
    args = parser.parse_args()
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    for path in args.result_json:
        if not path.is_file():
            raise FileNotFoundError(path)
    model = WhisperModel(
        str(args.model),
        device=args.device,
        device_index=args.device_index,
        compute_type=args.compute_type,
    )
    summaries = []
    for result_path in args.result_json:
        score = _score_report(
            model,
            result_path,
            require_source_location=args.require_source_location,
        )
        score["whisper_model"] = str(args.model.resolve())
        output_path = result_path.parent / "SPEECH_METRICS.json"
        _atomic_json(output_path, score)
        summaries.append(
            {
                "family_id": score["family_id"],
                "output": str(output_path),
                "transcripts": {
                    name: value["transcript"]
                    for name, value in score["scores"].items()
                },
                "source_speech_assignments": (
                    None
                    if score["source_location"] is None
                    else score["source_location"]["speech_assignments"]
                ),
            }
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
