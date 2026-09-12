#!/usr/bin/env python3
"""Audit forced-alignment timing against the actual P10 caption token stream.

The forced-aligner pilot first maps timestamps to an isolated transcript.  P10
does not consume that string in isolation: it consumes the complete compiled
semantic caption, including the room cue and speaker description.  This audit
therefore recompiles each frozen ScenePlan, tokenizes the full caption through
the production Qwen tokenizer, maps word timing onto tokens carrying a positive
``speech_source_ids`` role, and quantizes those intervals to the frozen VAE
frame grid.  It is deliberately an offline sidecar audit; no timing teacher is
added to the user-facing ScenePlan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import statistics
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROOT = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/"
    "audit/p10_speech_alignment_pilot_100_v1"
)
DEFAULT_TEST_INDEX = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/training_index/test.sqlite"
)
DEFAULT_TOKENIZER = Path("/mnt/sdc/ckpts/pretrained/Qwen/Qwen3.5-0.8B")
SAMPLE_RATE = 44_100
VAE_HOP_SAMPLES = 1024
MAX_CAPTION_TOKENS = 512


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _normalized_chars_with_positions(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    positions: list[int] = []
    for index, character in enumerate(text.casefold()):
        if character.isalnum():
            chars.append(character)
            positions.append(index)
    return "".join(chars), positions


def _alignment_item_spans(
    transcript: str, items: list[dict[str, Any]]
) -> list[tuple[int, int, float, float, int]]:
    normalized, positions = _normalized_chars_with_positions(transcript)
    cursor = 0
    spans: list[tuple[int, int, float, float, int]] = []
    for item_index, item in enumerate(items):
        item_normalized, _ = _normalized_chars_with_positions(str(item["text"]))
        if not item_normalized:
            continue
        found = normalized.find(item_normalized, cursor)
        if found < 0:
            raise RuntimeError(
                f"cannot place aligned item {item['text']!r} in transcript"
            )
        stop = found + len(item_normalized)
        spans.append(
            (
                positions[found],
                positions[stop - 1] + 1,
                float(item["start_sec"]),
                float(item["end_sec"]),
                int(item_index),
            )
        )
        cursor = stop
    return spans


def _scene_rows(index_path: Path, sample_ids: list[str]) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        output: dict[str, dict[str, Any]] = {}
        # SQLite's default variable limit can be lower than arbitrary pilot
        # sizes, so keep the query chunked even though this pilot has 100 rows.
        for start in range(0, len(sample_ids), 500):
            chunk = sample_ids[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"""
                SELECT sample_id, model_num_samples, latent_frames_valid,
                       scene_plan_zlib
                FROM samples
                WHERE sample_id IN ({placeholders})
                """,
                chunk,
            ).fetchall()
            for sample_id, model_num_samples, latent_frames_valid, blob in rows:
                output[str(sample_id)] = {
                    "model_num_samples": int(model_num_samples),
                    "latent_frames_valid": int(latent_frames_valid),
                    "scene_plan": json.loads(zlib.decompress(blob)),
                }
    finally:
        connection.close()
    if set(output) != set(sample_ids):
        missing = sorted(set(sample_ids) - set(output))
        raise RuntimeError(f"test index missed {len(missing)} pilot rows: {missing[:3]}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--index",
        "--test-index",
        dest="index",
        type=Path,
        default=DEFAULT_TEST_INDEX,
        help="Frozen P9 split index containing every manifest sample.",
    )
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    args = parser.parse_args()

    root = args.root.expanduser().resolve(strict=True)
    index_path = args.index.expanduser().resolve(strict=True)
    registry_path = root / "registry/source_alignment_registry.jsonl"
    registry = _read_jsonl(registry_path)
    prepare_summary = json.loads(
        (root / "PREPARE_SUMMARY.json").read_text(encoding="utf-8")
    )
    expected_rows = int(prepare_summary["rows"])
    if len(registry) != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} aligned donors, got {len(registry)}"
        )
    sample_ids = [str(row["sample_id"]) for row in registry]
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("alignment registry contains duplicate sample ids")

    from stable_audio_tools.data.model_sceneplan import (
        compile_model_44_controls,
        compile_model_semantic_caption,
        tokenize_model_semantic_caption,
    )
    from stable_audio_tools.data.speech_alignment import (
        load_aligner_timing_contract,
    )

    timing_contract = load_aligner_timing_contract(
        prepare_summary["aligner_model_root"]
    )
    nominal_endpoint_grid_overshoot_sec = float(
        timing_contract["nominal_endpoint_grid_overshoot_sec"]
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer.expanduser().resolve(strict=True), local_files_only=True
    )
    scenes = _scene_rows(index_path, sample_ids)
    audited: list[dict[str, Any]] = []
    for source_row in registry:
        sample_id = str(source_row["sample_id"])
        indexed = scenes[sample_id]
        plan = indexed["scene_plan"]
        speech_sources = [
            source for source in plan["sources"] if source.get("kind") == "speech"
        ]
        if len(speech_sources) != 1:
            raise RuntimeError(f"{sample_id}: expected exactly one formal speech source")
        speech_source = speech_sources[0]
        transcript = str(speech_source["transcript"])
        if transcript != str(source_row["transcript"]):
            raise RuntimeError(f"{sample_id}: registry/ScenePlan transcript changed")

        caption = compile_model_semantic_caption(plan)
        speech_regions = list(caption.get("speech_regions") or ())
        if len(speech_regions) != 1:
            raise RuntimeError(f"{sample_id}: semantic caption lacks one speech region")
        region = speech_regions[0]
        region_start = int(region["start"])
        region_end = int(region["end"])
        source_label = int(region["source_label"])
        if caption["text"][region_start:region_end] != transcript:
            raise RuntimeError(f"{sample_id}: speech region is not the exact transcript")

        tokenized = tokenize_model_semantic_caption(
            caption, tokenizer, max_length=MAX_CAPTION_TOKENS
        )
        input_ids = np.asarray(tokenized["input_ids"], dtype=np.int64)
        attention = np.asarray(tokenized["attention_mask"], dtype=np.bool_)
        offsets = np.asarray(tokenized["offset_mapping"], dtype=np.int64)
        speech_roles = np.asarray(tokenized["speech_source_ids"], dtype=np.int64)
        event_roles = np.asarray(tokenized["event_source_ids"], dtype=np.int64)
        if np.any((speech_roles > 0) & (event_roles > 0)):
            raise RuntimeError(f"{sample_id}: event/speech token roles overlap")
        positive_roles = speech_roles[speech_roles > 0]
        if positive_roles.size == 0 or np.any(positive_roles != source_label):
            raise RuntimeError(f"{sample_id}: speech tokens carry a wrong source label")

        item_spans = _alignment_item_spans(
            transcript, list(source_row["alignment_items"])
        )
        raw_zero_duration_items = sum(
            abs(item[3] - item[2]) <= 1.0e-9 for item in item_spans
        )
        activity = speech_source["activity"]
        activity_onset = float(activity["onset_sec"])
        activity_offset = float(activity["offset_sec"])
        activity_duration = activity_offset - activity_onset
        controls = compile_model_44_controls(
            plan,
            model_num_samples=indexed["model_num_samples"],
            latent_frames_valid=indexed["latent_frames_valid"],
        )
        active_mask = np.asarray(controls["speech_active_frame_mask"], dtype=np.bool_)
        full_token_timing: list[dict[str, Any]] = []
        mapped_semantic = 0
        eligible_semantic = 0
        mapped_items: set[int] = set()
        previous_start = -math.inf
        max_endpoint_clip_sec = 0.0
        for token_index, (token_id, token_offset, valid, role) in enumerate(
            zip(input_ids, offsets, attention, speech_roles)
        ):
            if not valid or int(role) <= 0:
                continue
            token_start, token_end = map(int, token_offset)
            local_start = max(token_start, region_start) - region_start
            local_end = min(token_end, region_end) - region_start
            local_start = max(local_start, 0)
            local_end = min(local_end, len(transcript))
            token_text = transcript[local_start:local_end]
            semantic = bool(re.search(r"\w", token_text, flags=re.UNICODE))
            overlaps = [
                item
                for item in item_spans
                if local_start < item[1] and local_end > item[0]
            ]
            if semantic:
                eligible_semantic += 1
                if overlaps:
                    mapped_semantic += 1
            if overlaps:
                mapped_items.update(item[4] for item in overlaps)
                source_start = min(item[2] for item in overlaps)
                aligner_source_end = max(item[3] for item in overlaps)
                # The frozen Qwen aligner emits an 80-ms timestamp grid.  The
                # waveform and ScenePlan interval remain authoritative: retain
                # the raw interval, then clamp a bounded terminal suffix.  A
                # zero-width final word may otherwise start just beyond the
                # waveform and quantize to an empty latent interval.
                endpoint_clip_sec = max(0.0, aligner_source_end - activity_duration)
                max_endpoint_clip_sec = max(max_endpoint_clip_sec, endpoint_clip_sec)
                source_start = min(source_start, activity_duration)
                source_end = min(aligner_source_end, activity_duration)
                absolute_start = activity_onset + source_start
                absolute_end = activity_onset + source_end
                start_frame = max(
                    0,
                    int(math.floor(absolute_start * SAMPLE_RATE / VAE_HOP_SAMPLES)),
                )
                end_frame = min(
                    indexed["latent_frames_valid"],
                    int(math.ceil(absolute_end * SAMPLE_RATE / VAE_HOP_SAMPLES)),
                )
                if end_frame <= start_frame:
                    # Give a boundary-clipped/zero-width lexical token the
                    # final active frame.  The immutable sidecar subsequently
                    # refines all lexical centers into a strictly monotonic
                    # positive-duration partition.
                    end_frame = min(
                        indexed["latent_frames_valid"], max(end_frame, 1)
                    )
                    start_frame = end_frame - 1
                if absolute_start + 1.0e-6 < previous_start:
                    raise RuntimeError(f"{sample_id}: token timing is not monotonic")
                previous_start = absolute_start
                if not bool(active_mask[start_frame:end_frame].all()):
                    raise RuntimeError(
                        f"{sample_id}: aligned token falls outside speech-active frames"
                    )
            else:
                source_start = source_end = None
                aligner_source_end = None
                endpoint_clip_sec = 0.0
                absolute_start = absolute_end = None
                start_frame = end_frame = None
            full_token_timing.append(
                {
                    "token_index": int(token_index),
                    "token_id": int(token_id),
                    "source_label": int(role),
                    "caption_char_start": int(token_start),
                    "caption_char_end": int(token_end),
                    "transcript_char_start": int(local_start),
                    "transcript_char_end": int(local_end),
                    "text": token_text,
                    "semantic": semantic,
                    "source_start_sec": source_start,
                    "source_end_sec": source_end,
                    "aligner_source_end_sec": aligner_source_end,
                    "endpoint_clip_sec": endpoint_clip_sec,
                    "absolute_start_sec": absolute_start,
                    "absolute_end_sec": absolute_end,
                    "start_latent_frame": start_frame,
                    "end_latent_frame_exclusive": end_frame,
                }
            )

        semantic_mapping_rate = (
            mapped_semantic / eligible_semantic if eligible_semantic else 0.0
        )
        item_mapping_rate = len(mapped_items) / len(item_spans) if item_spans else 0.0
        quantized_token_interval_frames = [
            int(item["end_latent_frame_exclusive"])
            - int(item["start_latent_frame"])
            for item in full_token_timing
            if item["start_latent_frame"] is not None
        ]
        if not quantized_token_interval_frames:
            raise RuntimeError(f"{sample_id}: no quantized speech-token intervals")
        audited.append(
            {
                "pilot_index": int(source_row["pilot_index"]),
                "sample_id": sample_id,
                "source_audio_sha256": source_row["source_audio_sha256"],
                "normalized_transcript_sha256": source_row[
                    "normalized_transcript_sha256"
                ],
                "source_dataset": source_row["source_dataset"],
                "source_label": source_label,
                "caption_text": caption["text"],
                "caption_valid_tokens": int(attention.sum()),
                "speech_role_tokens": int((speech_roles > 0).sum()),
                "semantic_speech_tokens": int(eligible_semantic),
                "mapped_semantic_speech_tokens": int(mapped_semantic),
                "semantic_token_mapping_rate": float(semantic_mapping_rate),
                "aligned_items": len(item_spans),
                "raw_zero_duration_aligned_items": raw_zero_duration_items,
                "mapped_aligned_items": len(mapped_items),
                "aligned_item_mapping_rate": float(item_mapping_rate),
                "quantized_token_interval_frames_min": min(
                    quantized_token_interval_frames
                ),
                "activity_onset_sec": activity_onset,
                "activity_offset_sec": activity_offset,
                "max_endpoint_clip_sec": float(max_endpoint_clip_sec),
                "latent_frames_valid": indexed["latent_frames_valid"],
                "full_caption_token_timing": full_token_timing,
            }
        )

    token_rates = [row["semantic_token_mapping_rate"] for row in audited]
    item_rates = [row["aligned_item_mapping_rate"] for row in audited]
    valid_lengths = [row["caption_valid_tokens"] for row in audited]
    endpoint_clips = [row["max_endpoint_clip_sec"] for row in audited]
    raw_zero_item_counts = [
        row["raw_zero_duration_aligned_items"] for row in audited
    ]
    quantized_min_frames = [
        row["quantized_token_interval_frames_min"] for row in audited
    ]
    endpoint_clip_p999 = _quantile(endpoint_clips, 0.999)
    gate_checks = {
        "rows": len(audited) == expected_rows,
        "semantic_token_mapping": min(token_rates) >= 0.99,
        "aligned_item_mapping": min(item_rates) >= 0.99,
        "caption_not_truncated": max(valid_lengths) <= MAX_CAPTION_TOKENS,
        "aligner_grid_overshoot_p999_within_nominal_bound": endpoint_clip_p999
        <= nominal_endpoint_grid_overshoot_sec + 1.0e-6,
        "aligner_terminal_clip_absolute_safety": max(endpoint_clips) <= 0.500,
        "quantized_training_targets_nonempty": min(quantized_min_frames) >= 1,
        # The per-row loop fails closed on source-label mismatch, non-monotonic
        # intervals, out-of-activity timing, and frame-grid disagreement.
        "role_timing_frame_contract": True,
    }
    status = "PASS" if all(gate_checks.values()) else "FAIL"
    sidecar = root / "registry/full_caption_token_timing.jsonl"
    _atomic_text(
        sidecar,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in audited
        ),
    )
    summary = {
        "schema": "stable_audio_tools.p10_full_caption_alignment_mapping_audit",
        "schema_version": 2,
        "status": status,
        "gate_checks": gate_checks,
        "metrics": {
            "rows": len(audited),
            "semantic_token_mapping_rate_min": min(token_rates),
            "semantic_token_mapping_rate_mean": statistics.fmean(token_rates),
            "aligned_item_mapping_rate_min": min(item_rates),
            "aligned_item_mapping_rate_mean": statistics.fmean(item_rates),
            "caption_valid_tokens_min": min(valid_lengths),
            "caption_valid_tokens_median": statistics.median(valid_lengths),
            "caption_valid_tokens_max": max(valid_lengths),
            "rows_with_endpoint_grid_clip": sum(
                value > 1.0e-6 for value in endpoint_clips
            ),
            "endpoint_grid_clip_max_sec": max(endpoint_clips),
            "endpoint_grid_clip_p999_sec": endpoint_clip_p999,
            "endpoint_grid_clip_nominal_bound_sec": (
                nominal_endpoint_grid_overshoot_sec
            ),
            "raw_zero_duration_aligned_items": sum(raw_zero_item_counts),
            "rows_with_raw_zero_duration_items": sum(
                value > 0 for value in raw_zero_item_counts
            ),
            "quantized_token_interval_frames_min": min(quantized_min_frames),
        },
        "inputs": {
            "source_alignment_registry": str(registry_path.resolve(strict=True)),
            "source_alignment_registry_sha256": _sha256_file(registry_path),
            "index": str(index_path),
            "tokenizer": str(args.tokenizer.expanduser().resolve(strict=True)),
            "caption_max_tokens": MAX_CAPTION_TOKENS,
            "sample_rate_hz": SAMPLE_RATE,
            "vae_hop_samples": VAE_HOP_SAMPLES,
            "aligner_timing_contract": timing_contract,
        },
        "sidecar": str(sidecar.resolve(strict=True)),
        "sidecar_sha256": _sha256_file(sidecar),
        "interpretation_boundary": (
            "This validates offline teacher-to-production-caption token/frame "
            "mapping. Raw Qwen alignment remains immutable and includes audited "
            "zero-duration short-word spans; the training sidecar quantizes every "
            "mapped speech token to at least one 1024-sample VAE frame. It does "
            "not establish that a duration-conditioned P10 model improves "
            "generated speech until a model ablation is run."
        ),
    }
    summary_path = root / "FULL_CAPTION_MAPPING_SUMMARY.json"
    _atomic_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
