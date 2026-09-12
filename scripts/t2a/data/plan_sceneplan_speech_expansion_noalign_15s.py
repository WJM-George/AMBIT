#!/usr/bin/env python3
"""Plan the unique-source inventory for the no-align/15-second expansion.

This stage is read-only with respect to the frozen 1.124M base.  It freezes
candidate manifests only; no external audio is downloaded and no ScenePlan is
rendered here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.data.sceneplan_v2_common import (  # noqa: E402
    atomic_write_json,
    deterministic_digest,
    normalized_transcript,
)


DATASET_ROOT = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m")
DEFAULT_CONTRACT = REPO_ROOT / (
    "docs/sceneplan_v2/sceneplan_speech_expansion_noalign_15s_v1.json"
)
DEFAULT_OUTPUT = DATASET_ROOT / "revisions/speech_expansion_noalign_15s_v1"
DEFAULT_LEDGER = DATASET_ROOT / (
    "split_ledgers/speech_v2/speech_split_ledger.parquet"
)
DEFAULT_CATALOG = DATASET_ROOT / "source_catalog/speech/catalog.sqlite"
DEFAULT_HIFITTS2 = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/speech_dataset/nvidia__hifitts-2/44khz/manifest_44khz.json"
)
DEFAULT_HIFITTS2_CHAPTERS = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/speech_dataset/nvidia__hifitts-2/44khz/chapters_44khz.json"
)
SHORT_MAX_SEC = 442_368 / 44_100
LONG_DRY_MAX_SEC = 14.5
EXTERNAL_QC_SURVIVAL = 0.85
MAX_EXTERNAL_PER_SPEAKER_BUCKET = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty candidate table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    os.replace(temporary, path)


def read_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema")
        != "stable_audio_tools.sceneplan_speech_expansion_contract"
        or int(value.get("schema_version", -1)) != 1
        or value.get("revision_id") != "speech_expansion_noalign_15s_v1"
    ):
        raise RuntimeError("unexpected speech-expansion contract")
    if int(value["audio"]["max_latent_frames"]) != 648:
        raise RuntimeError("speech-expansion contract is not the frozen 648-frame plan")
    if value["conditioning"].get("speech_timing_sidecar") is not None:
        raise RuntimeError("no-align contract unexpectedly names a timing sidecar")
    return value


def is_passed_ledger_row(row: dict[str, Any]) -> bool:
    return (
        str(row.get("signal_qc")) == "pass"
        and str(row.get("endpoint_qc")) == "pass"
        and str(row.get("asr_qc") or "").startswith("pass")
    )


def canonical_candidate(
    row: dict[str, Any],
    *,
    source_family: str,
    length_bucket: int,
    qc_state: str,
    locator: dict[str, Any],
) -> dict[str, Any]:
    transcript = normalized_transcript(
        row.get("normalized_transcript")
        or row.get("normalized_text")
        or row.get("renderer_text")
        or row.get("source_text")
        or row.get("text")
    )
    if not transcript:
        raise RuntimeError("candidate has an empty normalized transcript")
    source_id = str(row.get("source_id") or row.get("audio_filepath") or "")
    speaker_key = str(row.get("speaker_key") or row.get("speaker") or "")
    if not source_id or not speaker_key:
        raise RuntimeError("candidate has an empty source or speaker identity")
    duration = float(row.get("duration_sec") or row.get("duration") or 0.0)
    return {
        "candidate_id": f"{source_family}:{source_id}",
        "source_family": source_family,
        "source_dataset": str(row.get("source_dataset") or source_family),
        "source_id": source_id,
        "speaker_key": speaker_key,
        "duration_sec": duration,
        "length_bucket_frames": int(length_bucket),
        "source_text": str(
            row.get("source_text")
            or row.get("text")
            or row.get("normalized_text")
            or ""
        ),
        "renderer_text": str(
            row.get("renderer_text")
            or row.get("normalized_text")
            or row.get("source_text")
            or row.get("text")
            or ""
        ),
        "normalized_transcript": transcript,
        "normalized_transcript_sha256": text_sha256(transcript),
        "source_audio_sha256": str(row.get("source_audio_sha256") or ""),
        "qc_state": qc_state,
        "selection_rank": deterministic_digest(
            "speech_expansion_noalign_15s_v1",
            source_family,
            speaker_key,
            source_id,
        ),
        "locator_json": json.dumps(locator, sort_keys=True),
    }


def ledger_candidates(
    ledger_path: Path,
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    table = pq.read_table(ledger_path)
    rows = table.to_pylist()
    occupied_audio = {
        str(row["source_audio_sha256"])
        for row in rows
        if row.get("source_audio_sha256")
    }
    occupied_text = {
        str(row["normalized_transcript_sha256"])
        for row in rows
        if row.get("normalized_transcript_sha256")
    }
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not (
            row.get("pool") == "reserve"
            and row.get("replacement_split") == "train"
            and is_passed_ledger_row(row)
        ):
            continue
        selected.append(
            canonical_candidate(
                row,
                source_family="existing_train_reserve",
                length_bucket=432,
                qc_state="strong_qc_pass_v2",
                locator={
                    "type": "parquet_row",
                    "parquet_path": row["parquet_path"],
                    "row_group": int(row["row_group"]),
                    "row_in_group": int(row["row_in_group"]),
                },
            )
        )
    selected.sort(key=lambda row: row["selection_rank"])
    return selected, occupied_audio, occupied_text


def catalog_text_hashes(catalog_path: Path) -> set[str]:
    connection = sqlite3.connect(
        f"file:{catalog_path}?mode=ro", uri=True, check_same_thread=False
    )
    try:
        return {
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT normalized_transcript_sha256 FROM assets"
            )
            if row[0]
        }
    finally:
        connection.close()


def current_long_candidates(
    catalog_path: Path,
    *,
    occupied_audio: set[str],
    occupied_text: set[str],
) -> list[dict[str, Any]]:
    connection = sqlite3.connect(
        f"file:{catalog_path}?mode=ro", uri=True, check_same_thread=False
    )
    connection.row_factory = sqlite3.Row
    query = """
        SELECT * FROM assets
        WHERE source_split LIKE 'train%'
          AND duration_sec > ?
          AND duration_sec <= ?
        ORDER BY source_dataset, source_id
    """
    output: list[dict[str, Any]] = []
    seen_audio: set[str] = set()
    seen_text: set[str] = set()
    try:
        for sqlite_row in connection.execute(
            query, (SHORT_MAX_SEC, LONG_DRY_MAX_SEC)
        ):
            row = dict(sqlite_row)
            audio_hash = str(row["source_audio_sha256"])
            transcript_hash = str(row["normalized_transcript_sha256"])
            if (
                audio_hash in occupied_audio
                or transcript_hash in occupied_text
                or audio_hash in seen_audio
                or transcript_hash in seen_text
            ):
                continue
            candidate = canonical_candidate(
                row,
                source_family="existing_train_long",
                length_bucket=648,
                qc_state="pending_strong_qc_15s",
                locator={
                    "type": "parquet_row",
                    "parquet_path": row["parquet_path"],
                    "row_group": int(row["row_group"]),
                    "row_in_group": int(row["row_in_group"]),
                },
            )
            output.append(candidate)
            seen_audio.add(audio_hash)
            seen_text.add(transcript_hash)
    finally:
        connection.close()
    output.sort(key=lambda row: row["selection_rank"])
    return output


def external_row(
    row: dict[str, Any], existing_text_hashes: set[str]
) -> tuple[str, str, str, str] | None:
    if str(row.get("set")) != "train" or int(row.get("speaker_count") or 0) != 1:
        return None
    duration = float(row.get("duration") or 0.0)
    if not 0.5 <= duration <= LONG_DRY_MAX_SEC:
        return None
    if float(row.get("wer") if row.get("wer") is not None else 999.0) > 0.1:
        return None
    if float(row.get("cer") if row.get("cer") is not None else 999.0) > 0.03:
        return None
    transcript = normalized_transcript(row.get("normalized_text"))
    if not transcript:
        return None
    transcript_hash = text_sha256(transcript)
    if transcript_hash in existing_text_hashes:
        return None
    audio_path = str(row.get("audio_filepath") or "")
    speaker = str(row.get("speaker") or "")
    if not audio_path or not speaker or not audio_path.endswith(".flac"):
        return None
    chapter_path = audio_path.rsplit("_", 1)[0] + ".flac"
    bucket = "short" if duration <= SHORT_MAX_SEC else "long"
    return speaker, chapter_path, bucket, transcript_hash


def scan_external_chapter_counts(
    manifest_path: Path, existing_text_hashes: set[str]
) -> tuple[dict[tuple[str, str], list[int]], int]:
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    scanned = 0
    started = time.monotonic()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            scanned += 1
            row = json.loads(line)
            accepted = external_row(row, existing_text_hashes)
            if accepted is not None:
                speaker, chapter, bucket, _ = accepted
                counts[(speaker, chapter)][0 if bucket == "short" else 1] += 1
            if scanned % 2_000_000 == 0:
                print(
                    json.dumps(
                        {
                            "event": "hifitts2_manifest_count_progress",
                            "rows": scanned,
                            "chapters": len(counts),
                            "elapsed_sec": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
    return dict(counts), scanned


def choose_dense_chapters(
    counts: dict[tuple[str, str], list[int]],
    *,
    short_quota: int,
    long_quota: int,
) -> set[str]:
    by_speaker: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for (speaker, chapter), (short_count, long_count) in counts.items():
        by_speaker[speaker].append((chapter, short_count, long_count))
    for speaker, chapters in by_speaker.items():
        chapters.sort(
            key=lambda value: (
                -(value[1] + value[2]),
                deterministic_digest("hifitts2_chapter", speaker, value[0]),
            )
        )
    speakers = sorted(
        by_speaker,
        key=lambda value: deterministic_digest("hifitts2_speaker", value),
    )
    selected: set[str] = set()
    capacity = {speaker: [0, 0] for speaker in speakers}
    totals = [0, 0]
    target = [math.ceil(short_quota * 1.25), math.ceil(long_quota * 1.25)]
    round_index = 0
    while totals[0] < target[0] or totals[1] < target[1]:
        added = False
        for speaker in speakers:
            chapters = by_speaker[speaker]
            if round_index >= len(chapters):
                continue
            chapter, short_count, long_count = chapters[round_index]
            contributions = [
                min(
                    short_count,
                    MAX_EXTERNAL_PER_SPEAKER_BUCKET - capacity[speaker][0],
                ),
                min(
                    long_count,
                    MAX_EXTERNAL_PER_SPEAKER_BUCKET - capacity[speaker][1],
                ),
            ]
            if not any(contributions):
                continue
            if not any(
                totals[index] < target[index] and contributions[index] > 0
                for index in (0, 1)
            ):
                continue
            selected.add(chapter)
            for index in (0, 1):
                capacity[speaker][index] += contributions[index]
                totals[index] += contributions[index]
            added = True
            if totals[0] >= target[0] and totals[1] >= target[1]:
                break
        if not added:
            raise RuntimeError(
                "HiFiTTS-2 dense-chapter selection cannot meet external quotas"
            )
        round_index += 1
    return selected


def collect_external_candidates(
    manifest_path: Path,
    *,
    selected_chapters: set[str],
    existing_text_hashes: set[str],
    short_quota: int,
    long_quota: int,
) -> list[dict[str, Any]]:
    by_bucket_speaker: dict[str, dict[str, list[dict[str, Any]]]] = {
        "short": defaultdict(list),
        "long": defaultdict(list),
    }
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            accepted = external_row(row, existing_text_hashes)
            if accepted is None:
                continue
            speaker, chapter, bucket, _ = accepted
            if chapter not in selected_chapters:
                continue
            row = dict(row)
            row["source_dataset"] = "nvidia_hifitts2"
            row["source_id"] = str(row["audio_filepath"])
            row["speaker_key"] = f"nvidia_hifitts2:{speaker}"
            row["source_text"] = str(row.get("text") or row.get("normalized_text"))
            candidate = canonical_candidate(
                row,
                source_family="nvidia_hifitts2_44khz",
                length_bucket=432 if bucket == "short" else 648,
                qc_state="pending_download_and_strong_qc",
                locator={
                    "type": "hifitts2_chapter_segment",
                    "audio_filepath": row["audio_filepath"],
                    "chapter_filepath": chapter,
                },
            )
            candidate["manifest_wer"] = float(row["wer"])
            candidate["manifest_cer"] = float(row["cer"])
            candidate["text_source"] = str(row.get("text_source") or "")
            by_bucket_speaker[bucket][speaker].append(candidate)

    selected: list[dict[str, Any]] = []
    used_text = set(existing_text_hashes)
    for bucket, quota in (("short", short_quota), ("long", long_quota)):
        pools = by_bucket_speaker[bucket]
        for rows in pools.values():
            rows.sort(key=lambda value: value["selection_rank"])
        speakers = sorted(
            pools,
            key=lambda value: deterministic_digest(
                "hifitts2_round_robin", bucket, value
            ),
        )
        bucket_rows: list[dict[str, Any]] = []
        depth = 0
        while len(bucket_rows) < quota:
            added = False
            for speaker in speakers:
                rows = pools[speaker]
                if depth >= len(rows) or depth >= MAX_EXTERNAL_PER_SPEAKER_BUCKET:
                    continue
                row = rows[depth]
                transcript_hash = row["normalized_transcript_sha256"]
                if transcript_hash in used_text:
                    continue
                used_text.add(transcript_hash)
                bucket_rows.append(row)
                added = True
                if len(bucket_rows) == quota:
                    break
            if not added:
                raise RuntimeError(
                    f"HiFiTTS-2 selected chapters provide only {len(bucket_rows)} "
                    f"globally unique {bucket} rows, need {quota}"
                )
            depth += 1
        selected.extend(bucket_rows)
    return selected


def selected_chapter_metadata(
    chapters_path: Path, selected_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Freeze chapter URLs plus exact utterance extraction coordinates.

    The public utterance manifest intentionally contains only an utterance
    pathname and duration.  Exact offsets live in the chapter manifest.  A
    downloadable candidate plan is not self-contained unless those offsets
    are joined and frozen here; reconstructing them heuristically later would
    make the audio lineage non-reproducible.
    """

    selected_by_audio: dict[str, dict[str, Any]] = {}
    selected_paths: set[str] = set()
    for row in selected_rows:
        locator = json.loads(row["locator_json"])
        audio_path = str(locator["audio_filepath"])
        chapter_path = str(locator["chapter_filepath"])
        if audio_path in selected_by_audio:
            raise RuntimeError(f"duplicate selected HiFiTTS-2 utterance: {audio_path}")
        selected_by_audio[audio_path] = row
        selected_paths.add(chapter_path)
    output = []
    found = set()
    found_audio: set[str] = set()
    with chapters_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            chapter = str(row.get("chapter_filepath") or "")
            if chapter not in selected_paths:
                continue
            found.add(chapter)
            url = str(row["url"]).replace("http://", "https://", 1)
            selected_utterances = []
            for utterance in row.get("utterances") or ():
                audio_path = str(utterance.get("audio_filepath") or "")
                candidate = selected_by_audio.get(audio_path)
                if candidate is None:
                    continue
                if audio_path in found_audio:
                    raise RuntimeError(
                        f"HiFiTTS-2 chapter manifest duplicates utterance: {audio_path}"
                    )
                offset = float(utterance["offset"])
                duration = float(utterance["duration"])
                if offset < 0.0 or duration <= 0.0:
                    raise RuntimeError(f"invalid extraction coordinates: {audio_path}")
                if not math.isclose(
                    duration,
                    float(candidate["duration_sec"]),
                    rel_tol=0.0,
                    abs_tol=0.011,
                ):
                    raise RuntimeError(
                        f"HiFiTTS-2 duration mismatch for {audio_path}: "
                        f"{duration} != {candidate['duration_sec']}"
                    )
                candidate["locator_json"] = json.dumps(
                    {
                        "type": "hifitts2_chapter_segment",
                        "audio_filepath": audio_path,
                        "chapter_filepath": chapter,
                        "chapter_url": url,
                        "offset_sec": offset,
                        "duration_sec": duration,
                        "sample_rate_hz": 44_100,
                    },
                    sort_keys=True,
                )
                selected_utterances.append(
                    {
                        "audio_filepath": audio_path,
                        "offset_sec": offset,
                        "duration_sec": duration,
                    }
                )
                found_audio.add(audio_path)
            if not selected_utterances:
                raise RuntimeError(
                    f"selected HiFiTTS-2 chapter has no selected utterances: {chapter}"
                )
            selected_utterances.sort(key=lambda value: value["audio_filepath"])
            output.append(
                {
                    "chapter_filepath": chapter,
                    "url": url,
                    "duration_sec": float(row["duration"]),
                    "bandwidth_hz": int(row["bandwidth"]),
                    "selected_utterances": selected_utterances,
                }
            )
    missing = selected_paths - found
    if missing:
        raise RuntimeError(f"HiFiTTS-2 chapter metadata is incomplete: {len(missing)}")
    missing_audio = set(selected_by_audio) - found_audio
    if missing_audio:
        raise RuntimeError(
            "HiFiTTS-2 utterance extraction metadata is incomplete: "
            f"{len(missing_audio)}"
        )
    output.sort(key=lambda row: row["chapter_filepath"])
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--hifitts2-manifest", type=Path, default=DEFAULT_HIFITTS2)
    parser.add_argument(
        "--hifitts2-chapters", type=Path, default=DEFAULT_HIFITTS2_CHAPTERS
    )
    args = parser.parse_args()

    contract_path = args.contract.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve(strict=False)
    revisions_root = (DATASET_ROOT / "revisions").resolve(strict=True)
    if revisions_root not in output_root.parents:
        raise ValueError("output-root must remain below the frozen revisions root")
    contract = read_contract(contract_path)
    length_targets = contract["new_unique_speech_inventory"]["length_targets"]
    target_short = int(length_targets["432"])
    target_long = int(length_targets["648"])

    reserve, occupied_audio, occupied_text = ledger_candidates(
        args.ledger.expanduser().resolve(strict=True)
    )
    if len(reserve) != 42_726:
        raise RuntimeError(f"train reserve drifted: {len(reserve)} != 42726")
    long_rows = current_long_candidates(
        args.catalog.expanduser().resolve(strict=True),
        occupied_audio=occupied_audio,
        occupied_text=occupied_text,
    )
    if len(long_rows) > 39_642 or not long_rows:
        raise RuntimeError(f"unexpected current long candidate count: {len(long_rows)}")

    short_needed = target_short - len(reserve)
    long_needed = target_long - len(long_rows)
    if short_needed <= 0 or long_needed <= 0:
        raise RuntimeError("external allocation unexpectedly became non-positive")
    external_short_quota = math.ceil(short_needed / EXTERNAL_QC_SURVIVAL)
    external_long_quota = math.ceil(long_needed / EXTERNAL_QC_SURVIVAL)
    existing_catalog_text = catalog_text_hashes(
        args.catalog.expanduser().resolve(strict=True)
    )

    chapter_counts, manifest_rows = scan_external_chapter_counts(
        args.hifitts2_manifest.expanduser().resolve(strict=True),
        existing_catalog_text,
    )
    dense_chapters = choose_dense_chapters(
        chapter_counts,
        short_quota=external_short_quota,
        long_quota=external_long_quota,
    )
    external = collect_external_candidates(
        args.hifitts2_manifest.expanduser().resolve(strict=True),
        selected_chapters=dense_chapters,
        existing_text_hashes=existing_catalog_text,
        short_quota=external_short_quota,
        long_quota=external_long_quota,
    )
    chapter_rows = selected_chapter_metadata(
        args.hifitts2_chapters.expanduser().resolve(strict=True), external
    )

    candidates_root = output_root / "sources/candidates"
    reserve_path = candidates_root / "existing_train_reserve.parquet"
    long_path = candidates_root / "existing_train_long_pending_qc.parquet"
    external_path = candidates_root / "hifitts2_download_pending_qc.parquet"
    chapters_output = candidates_root / "hifitts2_selected_chapters.jsonl"
    atomic_write_parquet(reserve_path, reserve)
    atomic_write_parquet(long_path, long_rows)
    atomic_write_parquet(external_path, external)
    chapters_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = chapters_output.with_name(
        chapters_output.name + f".tmp.{os.getpid()}"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        for row in chapter_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, chapters_output)

    external_counts = Counter(
        int(row["length_bucket_frames"]) for row in external
    )
    summary = {
        "schema": "stable_audio_tools.sceneplan_speech_expansion_candidate_plan",
        "schema_version": 1,
        "revision_id": contract["revision_id"],
        "status": "PASS_CANDIDATES_ONLY",
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "final_scene_delta_target": int(contract["splits"]["train"]["delta_rows"]),
        "final_unique_speech_source_target": int(
            contract["new_unique_speech_inventory"]["target_rows"]
        ),
        "final_length_targets": {"432": target_short, "648": target_long},
        "existing_train_reserve_strong_qc": len(reserve),
        "existing_train_long_pending_qc": len(long_rows),
        "external_final_need_before_qc": {
            "432": short_needed,
            "648": long_needed,
        },
        "external_download_candidate_rows": {
            "432": external_counts[432],
            "648": external_counts[648],
            "total": len(external),
        },
        "hifitts2_manifest_rows_scanned": manifest_rows,
        "hifitts2_candidate_chapters_observed": len(chapter_counts),
        "hifitts2_selected_chapters": len(chapter_rows),
        "hifitts2_selected_chapter_hours": round(
            sum(row["duration_sec"] for row in chapter_rows) / 3600.0, 3
        ),
        "hifitts2_selected_utterance_hours": round(
            sum(float(row["duration_sec"]) for row in external) / 3600.0, 3
        ),
        "hifitts2_selected_speakers": len(
            {row["speaker_key"] for row in external}
        ),
        "artifacts": {
            "reserve": str(reserve_path),
            "reserve_sha256": sha256_file(reserve_path),
            "long_pending_qc": str(long_path),
            "long_pending_qc_sha256": sha256_file(long_path),
            "external_pending_qc": str(external_path),
            "external_pending_qc_sha256": sha256_file(external_path),
            "external_chapters": str(chapters_output),
            "external_chapters_sha256": sha256_file(chapters_output),
        },
        "next_gate": "download selected chapters and run strong source QC",
    }
    atomic_write_json(output_root / "CANDIDATE_PLAN.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
