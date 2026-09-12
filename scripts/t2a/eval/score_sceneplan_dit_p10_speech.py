#!/usr/bin/env python3
"""Score ASR and TTS diagnostics on a frozen P10 speech panel."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torchaudio
from faster_whisper import WhisperModel
from pesq import pesq
from pystoi import stoi
from scipy.signal import correlate, correlation_lags

from scripts.t2a.eval.sceneplan_dit_p10_panel_common import (
    DEFAULT_EVAL_ROOT,
    atomic_json,
    checkpoint_steps,
    load_foa,
    load_output_rows,
    load_panel,
    summarize,
)


DEFAULT_WHISPER = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/models/faster-distil-whisper-large-v3"
)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _normalize_words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", value.lower())


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _error_rates(hypothesis: str, reference: str) -> dict[str, Any]:
    reference_words = _normalize_words(reference)
    hypothesis_words = _normalize_words(hypothesis)
    reference_chars = list("".join(reference_words))
    hypothesis_chars = list("".join(hypothesis_words))
    word_edits = _edit_distance(reference_words, hypothesis_words)
    char_edits = _edit_distance(reference_chars, hypothesis_chars)
    return {
        "wer": word_edits / max(len(reference_words), 1),
        "cer": char_edits / max(len(reference_chars), 1),
        "word_edits": word_edits,
        "reference_words": len(reference_words),
        "hypothesis_words": len(hypothesis_words),
        "char_edits": char_edits,
        "reference_chars": len(reference_chars),
    }


def _formal_speech_transcript(scene_plan: dict[str, Any]) -> str:
    """Return the sole structured Speech transcript, never the full caption.

    Some evaluation panels persist ``semantic_text`` as the complete model
    caption (speaker description + protocol marker + transcript), while older
    speech-only panels used the bare transcript in that field.  WER/CER must
    not depend on that presentation-layer convention.
    """

    sources = [
        source
        for source in scene_plan.get("sources", [])
        if source.get("kind") == "speech"
    ]
    if len(sources) != 1:
        raise RuntimeError(
            "Speech scoring requires exactly one structured formal Speech source"
        )
    transcript = str(sources[0].get("transcript", "")).strip()
    if not transcript:
        raise RuntimeError("structured formal Speech transcript is empty")
    return transcript


def _mono_16k(path: str | Path) -> np.ndarray:
    audio, sample_rate = load_foa(path)
    mono = audio[0:1]
    if sample_rate != 16_000:
        mono = torchaudio.functional.resample(mono, sample_rate, 16_000)
    value = mono[0].numpy().astype(np.float32, copy=False)
    peak = float(np.max(np.abs(value))) if value.size else 0.0
    if peak > 1.0e-8:
        value = value / peak * (10.0 ** (-1.0 / 20.0))
    return value


def _active_mono_16k(path: str | Path, scene_plan: dict[str, Any]) -> np.ndarray:
    audio, sample_rate = load_foa(path)
    source = scene_plan["sources"][0]
    onset = max(0.0, float(source["activity"]["onset_sec"]) - 0.05)
    offset = min(float(scene_plan["duration_sec"]), float(source["activity"]["offset_sec"]) + 0.05)
    start = max(0, int(round(onset * sample_rate)))
    stop = min(int(audio.shape[-1]), int(round(offset * sample_rate)))
    mono = audio[0:1, start:stop]
    if sample_rate != 16_000:
        mono = torchaudio.functional.resample(mono, sample_rate, 16_000)
    value = mono[0].numpy().astype(np.float64, copy=False)
    value = value - value.mean() if value.size else value
    peak = float(np.max(np.abs(value))) if value.size else 0.0
    if peak > 1.0e-8:
        value = value / peak * (10.0 ** (-1.0 / 20.0))
    return value


def _align_pair(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    sample_rate: int = 16_000,
    max_lag_sec: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, int]:
    length = min(reference.size, generated.size)
    reference = reference[:length]
    generated = generated[:length]
    if length < int(0.25 * sample_rate):
        raise ValueError("paired speech segment is too short")
    corr = correlate(generated, reference, mode="full", method="fft")
    lags = correlation_lags(generated.size, reference.size, mode="full")
    limit = int(round(max_lag_sec * sample_rate))
    allowed = np.abs(lags) <= limit
    lag = int(lags[allowed][int(np.argmax(np.abs(corr[allowed])))])
    if lag > 0:
        generated = generated[lag:]
        reference = reference[: generated.size]
    elif lag < 0:
        reference = reference[-lag:]
        generated = generated[: reference.size]
    length = min(reference.size, generated.size)
    return reference[:length], generated[:length], lag


def _si_sdr(reference: np.ndarray, generated: np.ndarray) -> float:
    reference_tensor = torch.from_numpy(reference).to(torch.float64)
    generated_tensor = torch.from_numpy(generated).to(torch.float64)
    scale = torch.dot(generated_tensor, reference_tensor) / reference_tensor.square().sum().clamp_min(1e-12)
    target = scale * reference_tensor
    residual = generated_tensor - target
    return float(10.0 * torch.log10(target.square().sum().clamp_min(1e-12) / residual.square().sum().clamp_min(1e-12)))


def _paired_metrics(
    generated_path: str,
    reference_path: str,
    scene_plan: dict[str, Any],
) -> dict[str, Any]:
    reference = _active_mono_16k(reference_path, scene_plan)
    generated = _active_mono_16k(generated_path, scene_plan)
    reference, generated, lag = _align_pair(reference, generated)
    result: dict[str, Any] = {
        "sample_rate": 16_000,
        "aligned_samples": int(reference.size),
        "alignment_lag_samples": lag,
        "alignment_lag_sec": lag / 16_000.0,
        "si_sdr_db": _si_sdr(reference, generated),
        "stoi": None,
        "pesq_wb": None,
        "warning": "Paired waveform scores are diagnostic for stochastic TTS and are not hard gates.",
    }
    try:
        result["stoi"] = float(stoi(reference, generated, 16_000, extended=False))
    except Exception as error:
        result["stoi_error"] = f"{type(error).__name__}: {error}"
    try:
        result["pesq_wb"] = float(pesq(16_000, reference, generated, "wb"))
    except Exception as error:
        result["pesq_error"] = f"{type(error).__name__}: {error}"
    return result


def _transcribe(model: WhisperModel, waveform: np.ndarray) -> dict[str, Any]:
    segments_iterator, info = model.transcribe(
        waveform,
        language="en",
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    segments = list(segments_iterator)
    return {
        "text": " ".join(segment.text.strip() for segment in segments).strip(),
        "language": info.language,
        "language_probability": float(info.language_probability),
        "segment_count": len(segments),
        "mean_no_speech_probability": (
            None
            if not segments
            else sum(float(segment.no_speech_prob) for segment in segments) / len(segments)
        ),
        "mean_average_log_probability": (
            None
            if not segments
            else sum(float(segment.avg_logprob) for segment in segments) / len(segments)
        ),
    }


def _load_utmos(device: torch.device):
    try:
        model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True,
            verbose=False,
        )
        return model.to(device).eval(), None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


@torch.inference_mode()
def _utmos_score(model, waveform: np.ndarray, device: torch.device) -> float:
    # SpeechMOS' UTMOS wrapper accepts [batch,samples] and inserts its own
    # channel axis.  Supplying [B,1,T] here would become [B,1,1,T] inside the
    # wrapper and fail at its first Conv1d.
    value = torch.from_numpy(waveform).float().to(device).view(1, -1)
    score = model(value, 16_000)
    return float(torch.as_tensor(score).float().mean().cpu())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--whisper-model", type=Path, default=DEFAULT_WHISPER)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--skip-utmos", action="store_true")
    args = parser.parse_args()
    root = args.eval_root.expanduser().resolve(strict=True)
    steps = checkpoint_steps(root)
    whisper_path = args.whisper_model.expanduser().resolve(strict=True)
    panel = [row for row in load_panel(root) if row["domain"] == "speech"]
    outputs = [row for row in load_output_rows(root) if row["domain"] == "speech"]
    expected_outputs = len(panel) * len(steps)
    if not panel or len(outputs) != expected_outputs:
        raise RuntimeError(
            f"speech phase requires {len(panel)} references and {expected_outputs} outputs"
        )
    model = WhisperModel(
        str(whisper_path),
        device="cuda",
        device_index=args.device_index,
        compute_type=args.compute_type,
    )
    device = torch.device(f"cuda:{args.device_index}")
    utmos_model, utmos_error = (None, "disabled") if args.skip_utmos else _load_utmos(device)

    reference_cache: dict[str, dict[str, Any]] = {}
    vae_cache: dict[str, dict[str, Any]] = {}
    for row in panel:
        exact_transcript = _formal_speech_transcript(row["scene_plan"])
        waveform = _mono_16k(row["reference_foa_path"])
        transcription = _transcribe(model, waveform)
        reference_cache[row["panel_id"]] = {
            "transcription": transcription,
            "errors": _error_rates(transcription["text"], exact_transcript),
            "utmos": (
                _utmos_score(utmos_model, waveform, device) if utmos_model is not None else None
            ),
        }
        vae_metadata_path = root / "vae_reconstruction" / row["panel_id"] / "metadata.json"
        if vae_metadata_path.is_file():
            vae_metadata = json.loads(vae_metadata_path.read_text(encoding="utf-8"))
            vae_waveform = _mono_16k(vae_metadata["reconstruction_foa_path"])
            vae_transcription = _transcribe(model, vae_waveform)
            vae_cache[row["panel_id"]] = {
                "transcription": vae_transcription,
                "errors": _error_rates(vae_transcription["text"], exact_transcript),
                "utmos": (
                    _utmos_score(utmos_model, vae_waveform, device)
                    if utmos_model is not None
                    else None
                ),
                "foa_path": vae_metadata["reconstruction_foa_path"],
            }
        print(json.dumps({"event": "speech_reference", "panel_id": row["panel_id"]}), flush=True)

    scored: list[dict[str, Any]] = []
    for index, row in enumerate(outputs, start=1):
        exact_transcript = _formal_speech_transcript(row["scene_plan"])
        speech_source = next(
            source
            for source in row["scene_plan"]["sources"]
            if source["kind"] == "speech"
        )
        waveform = _mono_16k(row["generated_foa_path"])
        transcription = _transcribe(model, waveform)
        errors = _error_rates(transcription["text"], exact_transcript)
        value = {
            "checkpoint_step": int(row["checkpoint_step"]),
            "panel_id": row["panel_id"],
            "sample_id": row["sample_id"],
            "speaker_description": speech_source["speaker_description"],
            "exact_transcript": exact_transcript,
            "generated_asr": transcription,
            "generated_errors": errors,
            "reference_asr": reference_cache[row["panel_id"]]["transcription"],
            "reference_errors": reference_cache[row["panel_id"]]["errors"],
            "generated_utmos": (
                _utmos_score(utmos_model, waveform, device) if utmos_model is not None else None
            ),
            "reference_utmos": reference_cache[row["panel_id"]]["utmos"],
            "vae_codec_ceiling": vae_cache.get(row["panel_id"]),
            "paired_reference_diagnostics": _paired_metrics(
                row["generated_foa_path"], row["reference_foa_path"], row["scene_plan"]
            ),
            "generated_foa_path": row["generated_foa_path"],
            "generated_stereo_path": row["generated_stereo_path"],
            "reference_foa_path": row["reference_foa_path"],
        }
        scored.append(value)
        print(
            json.dumps(
                {
                    "event": "speech_scored",
                    "index": index,
                    "count": len(outputs),
                    "step": row["checkpoint_step"],
                    "panel_id": row["panel_id"],
                    "wer": errors["wer"],
                    "asr": transcription["text"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    aggregates: dict[str, Any] = {}
    for step in steps:
        chosen = [row for row in scored if row["checkpoint_step"] == step]
        total_word_edits = sum(row["generated_errors"]["word_edits"] for row in chosen)
        total_reference_words = sum(
            row["generated_errors"]["reference_words"] for row in chosen
        )
        total_char_edits = sum(row["generated_errors"]["char_edits"] for row in chosen)
        total_reference_chars = sum(
            row["generated_errors"]["reference_chars"] for row in chosen
        )
        aggregates[str(step)] = {
            "rows": len(chosen),
            "wer": summarize(row["generated_errors"]["wer"] for row in chosen),
            "cer": summarize(row["generated_errors"]["cer"] for row in chosen),
            "corpus_wer": total_word_edits / max(total_reference_words, 1),
            "corpus_cer": total_char_edits / max(total_reference_chars, 1),
            "utmos": summarize(row["generated_utmos"] for row in chosen),
            "pesq_wb_diagnostic": summarize(
                row["paired_reference_diagnostics"]["pesq_wb"] for row in chosen
            ),
            "stoi_diagnostic": summarize(
                row["paired_reference_diagnostics"]["stoi"] for row in chosen
            ),
            "si_sdr_db_diagnostic": summarize(
                row["paired_reference_diagnostics"]["si_sdr_db"] for row in chosen
            ),
        }
    reference_values = list(reference_cache.values())
    vae_values = list(vae_cache.values())
    report = {
        "schema": "stable_audio_tools.sceneplan_dit_p10_speech_metrics",
        "schema_version": 1,
        "status": "PASS",
        "evaluation_rows": len(panel),
        "checkpoint_outputs": expected_outputs,
        "whisper_model": str(whisper_path),
        "asr_channel": "W",
        "asr_sample_rate": 16_000,
        "utmos_available": utmos_model is not None,
        "utmos_error": utmos_error,
        "reference_ceiling": {
            "wer": summarize(row["errors"]["wer"] for row in reference_values),
            "cer": summarize(row["errors"]["cer"] for row in reference_values),
            "utmos": summarize(row["utmos"] for row in reference_values),
        },
        "vae_codec_ceiling": {
            "available": len(vae_values) == len(panel),
            "rows": len(vae_values),
            "wer": summarize(row["errors"]["wer"] for row in vae_values),
            "cer": summarize(row["errors"]["cer"] for row in vae_values),
            "utmos": summarize(row["utmos"] for row in vae_values),
        },
        "paired_metric_warning": "PESQ-WB, STOI and SI-SDR compare a stochastic generated utterance with the deterministic rendered donor after bounded-lag alignment; they are diagnostic, not hard gates.",
        "aggregates": aggregates,
    }
    metrics_root = root / "metrics"
    _atomic_jsonl(metrics_root / "speech_per_output.jsonl", scored)
    atomic_json(metrics_root / "SPEECH_SUMMARY.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
