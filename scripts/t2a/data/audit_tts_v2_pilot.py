#!/usr/bin/env python3
"""Fail-closed speech and FOA audit for the 2k TTS v2 pilot.

The audit distinguishes *duration* from *completeness*.  A row passes only when
the complete Parquet utterance is consumed, the reference transcript is heard
through its final word in both the dry source and rendered FOA W channel, and
the endpoint has no evidence of an active-speech hard cut.  A short complete
row is valid; a longer incomplete row is quarantined.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer


MODEL_SAMPLE_RATE = 44_100
MAX_MODEL_SAMPLES = 442_368
ASR_SAMPLE_RATE = 16_000
DEFAULT_MODEL = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/models/"
    "faster-distil-whisper-large-v3"
)
DEFAULT_WORK_ROOT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/pilots/tts_2k/qc/asr"
)
DEFAULT_MANIFESTS = (
    Path(
        os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/pilots/tts_2k/"
        "pilot_manifest.jsonl"
    ),
)

# These are intentionally conservative pilot thresholds.  A failed row is not
# repaired by weakening a threshold; it is quarantined and replaced from the
# same duration/dataset/motion stratum.
MAX_REFERENCE_WER = 0.25
MIN_ORDERED_WORD_COVERAGE = 0.80
MIN_SUFFIX_WORD_COVERAGE = 2.0 / 3.0
MAX_FOA_WER_DEGRADATION = 0.05
MAX_REFERENCE_CER = 0.20
MIN_HYPOTHESIS_CHAR_RATIO = 0.70
MIN_SUFFIX_CHAR_SIMILARITY = 0.45
MAX_FOA_CER_DEGRADATION = 0.15
MIN_NATURAL_TAIL_MARGIN_SEC = 0.030
TAIL_ACTIVE_RELATIVE_DB = -24.0
ASR_TEXT_NORMALIZER = EnglishTextNormalizer({})


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"{path} contains a non-object row")
                yield value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def words(value: Any) -> list[str]:
    # Use Whisper's own English normalizer on reference and hypothesis alike.
    # This canonicalizes spoken-vs-written numbers, curly apostrophes,
    # contractions, and punctuation before WER. The former raw regex falsely
    # quarantined exact audio for pairs such as "thirty-six"/"36" and
    # "d'Artagnan"/"D’Artagnan".
    normalized = ASR_TEXT_NORMALIZER(str(value or ""))
    return re.findall(r"[a-z0-9]+", normalized)


def resample_mono(audio: np.ndarray, native_rate: int) -> np.ndarray:
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1, dtype=np.float32)
    if mono.ndim != 1 or not len(mono):
        raise ValueError("empty or invalid mono source")
    if native_rate != ASR_SAMPLE_RATE:
        divisor = math.gcd(int(native_rate), ASR_SAMPLE_RATE)
        mono = resample_poly(
            mono,
            ASR_SAMPLE_RATE // divisor,
            int(native_rate) // divisor,
        ).astype(np.float32, copy=False)
    # Dry and FOA-W differ by intentional spatial distance gain.  Normalize only
    # the ASR probe copy so the dry-vs-W comparison measures content loss rather
    # than Whisper's level sensitivity; the stored waveform is never changed.
    mono = mono - float(np.mean(mono, dtype=np.float64))
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    if not math.isfinite(rms) or rms < 1e-8:
        raise ValueError("silent or non-finite ASR probe")
    mono = (mono * (0.05 / rms)).astype(np.float32, copy=False)
    return mono


def read_parquet_source(row: dict[str, Any]) -> tuple[np.ndarray, int]:
    lineage = row["source_audio"]
    parquet_file = pq.ParquetFile(lineage["parquet_path"])
    table = parquet_file.read_row_group(int(lineage["row_group"]), columns=["audio"])
    records = table.column("audio").to_pylist()
    record = records[int(lineage["row_in_group"])] or {}
    blob = record.get("bytes")
    if not blob:
        raise ValueError("Parquet source has no embedded audio bytes")
    if hashlib.sha256(bytes(blob)).hexdigest() != lineage["source_audio_sha256"]:
        raise ValueError("Parquet source SHA256 changed after render")
    audio, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError(f"canonical speech source is not mono: {audio.shape}")
    if len(audio) != int(lineage["native_num_samples"]):
        raise ValueError("Parquet native sample count changed after render")
    return audio[:, 0], int(rate)


def read_foa_w(row: dict[str, Any]) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(row["foa_path"], dtype="float32", always_2d=True)
    if audio.shape != (int(row["audio"]["num_samples"]), 4):
        raise ValueError(f"unexpected FOA shape: {audio.shape}")
    if int(rate) != MODEL_SAMPLE_RATE:
        raise ValueError(f"unexpected FOA sample rate: {rate}")
    return audio[:, 0], int(rate)


def asr_tokens(segments: list[dict[str, Any]]) -> tuple[list[str], list[float]]:
    tokens: list[str] = []
    ends: list[float] = []
    for segment in segments:
        for word in segment["words"]:
            normalized = words(word["word"])
            for token in normalized:
                tokens.append(token)
                ends.append(float(word["end"]))
    return tokens, ends


def alignment(reference: list[str], hypothesis: list[str]) -> dict[str, Any]:
    """Levenshtein alignment with exact reference-to-hypothesis match indices."""
    n, m = len(reference), len(hypothesis)
    distance = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        distance[i][0] = i
    for j in range(m + 1):
        distance[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitution = distance[i - 1][j - 1] + (
                0 if reference[i - 1] == hypothesis[j - 1] else 1
            )
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                substitution,
            )
    i, j = n, m
    exact_pairs: list[tuple[int, int]] = []
    while i or j:
        if (
            i
            and j
            and reference[i - 1] == hypothesis[j - 1]
            and distance[i][j] == distance[i - 1][j - 1]
        ):
            exact_pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i and j and distance[i][j] == distance[i - 1][j - 1] + 1:
            i -= 1
            j -= 1
        elif i and distance[i][j] == distance[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    exact_pairs.reverse()
    matched_reference = {ref_index: hyp_index for ref_index, hyp_index in exact_pairs}
    suffix_size = min(3, n)
    suffix_matches = sum(
        index in matched_reference for index in range(n - suffix_size, n)
    )
    return {
        "edit_distance": distance[n][m],
        "wer": distance[n][m] / max(1, n),
        "ordered_word_coverage": len(exact_pairs) / max(1, n),
        "suffix_word_count": suffix_size,
        "suffix_word_coverage": suffix_matches / max(1, suffix_size),
        "final_reference_word_matched": bool(n and n - 1 in matched_reference),
        "final_reference_hypothesis_index": matched_reference.get(n - 1),
        "exact_pairs": exact_pairs,
    }


def _edit_distance_1d(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row_index, left in enumerate(reference, start=1):
        current = [row_index]
        for column_index, right in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + int(left != right),
                )
            )
        previous = current
    return int(previous[-1])


def completion_text_evidence(reference: str, hypothesis: str) -> dict[str, Any]:
    """Robust English completion evidence after Whisper normalization.

    Word-exact WER remains in the report, but is not allowed to quarantine a
    complete proper name or number-format pair by itself. Character evidence
    is calibrated against explicit hard-cut controls.
    """

    reference_chars = "".join(words(reference))
    hypothesis_chars = "".join(words(hypothesis))
    distance = _edit_distance_1d(reference_chars, hypothesis_chars)
    character_error_rate = distance / max(1, len(reference_chars))
    length_ratio = len(hypothesis_chars) / max(1, len(reference_chars))
    suffix_length = min(24, len(reference_chars))
    reference_suffix = reference_chars[-suffix_length:]
    hypothesis_suffix = (
        hypothesis_chars[-suffix_length:]
        if len(hypothesis_chars) >= suffix_length
        else hypothesis_chars
    )
    suffix_distance = _edit_distance_1d(reference_suffix, hypothesis_suffix)
    suffix_similarity = 1.0 - suffix_distance / max(
        1, len(reference_suffix), len(hypothesis_suffix)
    )
    passed = bool(
        character_error_rate <= MAX_REFERENCE_CER
        and length_ratio >= MIN_HYPOTHESIS_CHAR_RATIO
        and suffix_similarity >= MIN_SUFFIX_CHAR_SIMILARITY
    )
    return {
        "normalized_reference_chars": reference_chars,
        "normalized_hypothesis_chars": hypothesis_chars,
        "edit_distance": distance,
        "character_error_rate": character_error_rate,
        "hypothesis_to_reference_char_ratio": length_ratio,
        "suffix_char_count": suffix_length,
        "suffix_char_similarity": suffix_similarity,
        "pass": passed,
    }


def transcribe(model: Any, audio: np.ndarray, rate: int) -> dict[str, Any]:
    source = resample_mono(audio, rate)
    raw_segments, info = model.transcribe(
        source,
        language="en",
        beam_size=5,
        best_of=5,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 200, "speech_pad_ms": 80},
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    segments = []
    for segment in raw_segments:
        segment_words = [
            {
                "start": float(word.start),
                "end": float(word.end),
                "word": str(word.word),
                "probability": float(word.probability),
            }
            for word in (segment.words or [])
            if word.start is not None and word.end is not None
        ]
        segments.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": str(segment.text).strip(),
                "avg_logprob": float(segment.avg_logprob),
                "no_speech_prob": float(segment.no_speech_prob),
                "words": segment_words,
            }
        )
    tokens, ends = asr_tokens(segments)
    return {
        "recognized_text": " ".join(segment["text"] for segment in segments).strip(),
        "tokens": tokens,
        "token_end_sec": ends,
        "segments": segments,
        "language": str(info.language),
        "language_probability": float(info.language_probability),
        "duration_sec": len(source) / ASR_SAMPLE_RATE,
    }


def acoustic_endpoint(audio: np.ndarray, rate: int) -> dict[str, float]:
    signal = np.asarray(audio, dtype=np.float64)
    signal = signal - float(np.mean(signal))
    overall_rms = float(np.sqrt(np.mean(np.square(signal)))) + 1e-12
    tail_count = max(1, min(len(signal), int(round(0.010 * rate))))
    tail_rms = float(np.sqrt(np.mean(np.square(signal[-tail_count:]))))
    tail_relative_db = 20.0 * math.log10(max(tail_rms, 1e-12) / overall_rms)
    edge_relative = abs(float(signal[-1])) / overall_rms
    return {
        "overall_rms": overall_rms,
        "tail_10ms_rms": tail_rms,
        "tail_10ms_relative_db": tail_relative_db,
        "last_sample_abs_over_rms": edge_relative,
    }


def structural_checks(row: dict[str, Any]) -> list[str]:
    failures = []
    audio = row["audio"]
    source = row["source_audio"]
    count = int(audio["num_samples"])
    source_count = int(audio["source_num_samples"])
    render_tail = int(audio["render_tail_samples"])
    if not 0 < count <= MAX_MODEL_SAMPLES:
        failures.append("model_sample_count_outside_contract")
    if int(source["model_num_samples"]) != source_count:
        failures.append("source_resample_count_mismatch")
    if count != source_count + render_tail:
        failures.append("source_plus_tail_scene_count_mismatch")
    if render_tail < 40:
        failures.append("tail_does_not_preserve_pyroom_delay")
    if int(audio["latent_frames_valid"]) != math.ceil(count / 1024):
        failures.append("latent_frame_count_mismatch")
    if int(source["consumed_native_start_sample"]) != 0:
        failures.append("source_does_not_start_at_zero")
    if int(source["consumed_native_end_sample"]) != int(source["native_num_samples"]):
        failures.append("source_does_not_end_at_native_end")
    if float(source["coverage_fraction"]) != 1.0:
        failures.append("source_coverage_not_one")
    if bool(source["random_crop"]):
        failures.append("random_crop_true")
    if words(row["transcript"]) != words(row["parquet_transcript"]):
        failures.append("manifest_parquet_transcript_word_mismatch")
    caption = row.get("renderer_caption") or {}
    transcript_regions = caption.get("transcript_regions") or []
    if len(transcript_regions) != 1:
        failures.append("caption_transcript_region_count")
    else:
        region = transcript_regions[0]
        start, end = int(region["start"]), int(region["end"])
        text = str(caption.get("text") or "")
        if not (0 < start < end < len(text)):
            failures.append("caption_transcript_region_bounds")
        else:
            if text[start - 1] != '"' or text[end] != '"':
                failures.append("caption_transcript_not_strictly_quoted")
            if text[start:end] != row["transcript"]:
                failures.append("caption_transcript_region_exact_mismatch")
        for speaker_region in caption.get("speaker_info_regions") or []:
            if int(speaker_region["end"]) > start:
                failures.append("speaker_info_mask_intersects_quoted_speech")
    renderer_qc = ((row.get("spatial") or {}).get("renderer_qc") or {})
    diagnostics = renderer_qc.get("keyframe_rir_diagnostics") or []
    if not diagnostics:
        failures.append("missing_rir_direction_diagnostics")
    for diagnostic in diagnostics:
        if not 39 <= int(diagnostic["residual_direct_peak_sample"]) <= 41:
            failures.append("pyroom_residual_delay_mismatch")
        if float(diagnostic["max_abs_direction_ratio_error"]) > 0.005:
            failures.append("foa_direction_ratio_error")
    ceiling = 10.0 ** (-1.0 / 20.0)
    if float((row.get("signal") or {}).get("stored_true_peak", 2.0)) > ceiling + 2e-5:
        failures.append("stored_true_peak_above_minus_1_dbfs")
    info = sf.info(row["foa_path"])
    if (info.frames, info.samplerate, info.channels) != (count, MODEL_SAMPLE_RATE, 4):
        failures.append("foa_file_format_or_length_mismatch")
    return failures


def score_channel(
    model: Any,
    audio: np.ndarray,
    rate: int,
    reference_tokens: list[str],
) -> dict[str, Any]:
    result = transcribe(model, audio, rate)
    aligned = alignment(reference_tokens, result["tokens"])
    completion = completion_text_evidence(" ".join(reference_tokens), result["recognized_text"])
    final_hypothesis_index = aligned["final_reference_hypothesis_index"]
    final_word_end = (
        None
        if final_hypothesis_index is None
        or final_hypothesis_index >= len(result["token_end_sec"])
        else float(result["token_end_sec"][final_hypothesis_index])
    )
    tail_margin = (
        None
        if final_word_end is None
        else result["duration_sec"] - final_word_end
    )
    last_asr_word_end = max(result["token_end_sec"], default=None)
    if last_asr_word_end is None:
        last_asr_word_end = max(
            (float(segment["end"]) for segment in result["segments"]),
            default=None,
        )
    last_asr_word_tail_margin = (
        None
        if last_asr_word_end is None
        else result["duration_sec"] - last_asr_word_end
    )
    endpoint = acoustic_endpoint(audio, rate)
    active_at_boundary = endpoint["tail_10ms_relative_db"] > TAIL_ACTIVE_RELATIVE_DB
    endpoint_natural = bool(
        completion["pass"]
        and last_asr_word_end is not None
        and (
            last_asr_word_tail_margin >= MIN_NATURAL_TAIL_MARGIN_SEC
            or not active_at_boundary
        )
    )
    word_exact_complete = bool(
        aligned["wer"] <= MAX_REFERENCE_WER
        and aligned["ordered_word_coverage"] >= MIN_ORDERED_WORD_COVERAGE
        and aligned["suffix_word_coverage"] >= MIN_SUFFIX_WORD_COVERAGE
        and aligned["final_reference_word_matched"]
    )
    semantic_complete = bool(completion["pass"])
    result.update(
        {
            "alignment": aligned,
            "word_exact_complete": word_exact_complete,
            "completion_text_evidence": completion,
            "final_reference_word_end_sec": final_word_end,
            "tail_margin_after_final_reference_word_sec": tail_margin,
            "last_asr_word_end_sec": last_asr_word_end,
            "tail_margin_after_last_asr_word_sec": last_asr_word_tail_margin,
            "acoustic_endpoint": endpoint,
            "active_speech_energy_at_boundary": active_at_boundary,
            "semantic_complete": semantic_complete,
            "endpoint_natural": endpoint_natural,
            "pass": semantic_complete and endpoint_natural,
        }
    )
    return result


def audit_one(model: Any, row: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    failures = structural_checks(row)
    reference_tokens = words(row["transcript"])
    if not reference_tokens:
        failures.append("empty_reference_transcript")
    dry, dry_rate = read_parquet_source(row)
    foa_w, foa_rate = read_foa_w(row)
    dry_score = score_channel(model, dry, dry_rate, reference_tokens)
    foa_score = score_channel(model, foa_w, foa_rate, reference_tokens)
    wer_degradation = float(
        foa_score["alignment"]["wer"] - dry_score["alignment"]["wer"]
    )
    cer_degradation = float(
        foa_score["completion_text_evidence"]["character_error_rate"]
        - dry_score["completion_text_evidence"]["character_error_rate"]
    )
    if not dry_score["semantic_complete"]:
        failures.append("dry_reference_not_complete")
    if not dry_score["endpoint_natural"]:
        failures.append("dry_endpoint_not_natural")
    if not foa_score["semantic_complete"]:
        failures.append("foa_w_reference_not_complete")
    if not foa_score["endpoint_natural"]:
        failures.append("foa_w_endpoint_not_natural")
    if cer_degradation > MAX_FOA_CER_DEGRADATION:
        failures.append("foa_w_asr_cer_degradation")
    return {
        "schema": "stable_audio_tools.tts_v2_pilot_speech_qc",
        "schema_version": 1,
        "sample_id": row["sample_id"],
        "partition": row["partition"],
        "source_dataset": row["source_dataset"],
        "source_id": row["source_id"],
        "foa_path": row["foa_path"],
        "reference_transcript": row["transcript"],
        "reference_tokens": reference_tokens,
        "num_samples": int(row["audio"]["num_samples"]),
        "duration_sec": float(row["audio"]["duration_sec"]),
        "strata": row["strata"],
        "status": "pass" if not failures else "quarantine",
        "failure_reasons": failures,
        "dry": dry_score,
        "foa_w": foa_score,
        "foa_w_minus_dry_wer": wer_degradation,
        "foa_w_minus_dry_cer": cer_degradation,
        "elapsed_sec": round(time.time() - started, 4),
    }


def worker(
    gpu_index: int,
    rows: list[dict[str, Any]],
    model_path: str,
    output_path: str,
) -> None:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        model_path,
        device="cuda",
        device_index=gpu_index,
        compute_type="float16",
    )
    results = []
    for position, row in enumerate(rows, start=1):
        try:
            result = audit_one(model, row)
        except Exception as exc:  # noqa: BLE001
            result = {
                "schema": "stable_audio_tools.tts_v2_pilot_speech_qc",
                "schema_version": 1,
                "sample_id": row.get("sample_id"),
                "partition": row.get("partition"),
                "source_dataset": row.get("source_dataset"),
                "source_id": row.get("source_id"),
                "status": "error",
                "failure_reasons": [f"audit_exception:{type(exc).__name__}"],
                "error": repr(exc),
            }
        results.append(result)
        if position % 25 == 0:
            print(
                json.dumps(
                    {
                        "gpu": gpu_index,
                        "done": position,
                        "total": len(rows),
                        "pass": sum(item["status"] == "pass" for item in results),
                    }
                ),
                flush=True,
            )
    atomic_write_jsonl(Path(output_path), results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifests = tuple(args.manifest or DEFAULT_MANIFESTS)
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    rows = []
    for manifest in manifests:
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        source_rows = list(iter_jsonl(manifest))
        if any(row.get("status") != "ok" for row in source_rows):
            raise RuntimeError(f"render manifest is not all-ok: {manifest}")
        rows.extend(source_rows)
    rows.sort(
        key=lambda row: (
            row["source_audio"]["parquet_path"],
            int(row["source_audio"]["row_group"]),
            int(row["source_audio"]["row_in_group"]),
        )
    )
    if args.limit is not None:
        rows = rows[: args.limit]
    gpu_indices = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpu_indices:
        raise ValueError("--gpus must name at least one GPU")
    shards = [rows[index:: len(gpu_indices)] for index in range(len(gpu_indices))]
    args.work_root.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    processes = []
    shard_paths = []
    started = time.time()
    for worker_index, (gpu_index, shard) in enumerate(zip(gpu_indices, shards)):
        if not shard:
            continue
        shard_path = args.work_root / f"speech_qc_shard_{worker_index:02d}.jsonl"
        shard_paths.append(shard_path)
        process = context.Process(
            target=worker,
            args=(gpu_index, shard, str(args.model), str(shard_path)),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"speech QC worker exited {process.exitcode}")
    results = []
    for shard_path in shard_paths:
        results.extend(iter_jsonl(shard_path))
    results.sort(key=lambda row: (str(row.get("partition")), str(row.get("sample_id"))))
    combined_path = args.work_root / "speech_qc_manifest.jsonl"
    atomic_write_jsonl(combined_path, results)
    statuses = Counter(str(row["status"]) for row in results)
    reasons = Counter(
        reason
        for row in results
        for reason in row.get("failure_reasons", [])
    )
    strata = Counter(
        (
            str(row.get("partition")),
            str(row.get("source_dataset")),
            int((row.get("strata") or {}).get("duration_bin", -1)),
            str((row.get("strata") or {}).get("motion", "unknown")),
            str(row["status"]),
        )
        for row in results
    )
    summary = {
        "schema": "stable_audio_tools.tts_v2_pilot_speech_qc_summary",
        "schema_version": 1,
        "ok": statuses["pass"] == len(rows),
        "policy": {
            "shorter_complete_preferred": True,
            "incomplete_or_uncertain_rows": "quarantine_and_replace_same_stratum",
            "max_reference_wer": MAX_REFERENCE_WER,
            "min_ordered_word_coverage": MIN_ORDERED_WORD_COVERAGE,
            "min_suffix_word_coverage": MIN_SUFFIX_WORD_COVERAGE,
            "require_final_reference_word_matched": True,
            "max_foa_w_minus_dry_wer": MAX_FOA_WER_DEGRADATION,
            "max_reference_cer": MAX_REFERENCE_CER,
            "min_hypothesis_to_reference_char_ratio": MIN_HYPOTHESIS_CHAR_RATIO,
            "min_suffix_char_similarity": MIN_SUFFIX_CHAR_SIMILARITY,
            "max_foa_w_minus_dry_cer": MAX_FOA_CER_DEGRADATION,
            "word_exact_metrics_are_diagnostic_not_sole_quarantine_authority": True,
            "min_natural_tail_margin_sec_unless_tail_inactive": MIN_NATURAL_TAIL_MARGIN_SEC,
            "tail_active_relative_db": TAIL_ACTIVE_RELATIVE_DB,
        },
        "rows": len(rows),
        "status_counts": dict(statuses),
        "failure_reason_counts": dict(reasons),
        "strata_counts": {"|".join(map(str, key)): value for key, value in sorted(strata.items())},
        "manifest": str(combined_path),
        "elapsed_sec": round(time.time() - started, 3),
    }
    atomic_write_json(args.work_root / "speech_qc_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
