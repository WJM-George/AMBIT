#!/usr/bin/env python3
"""Download and materialize the frozen HiFiTTS-2 expansion candidates.

Only chapter MP3 files named by the frozen candidate plan are downloaded.
Each chapter is decoded once, its selected utterances are cut at the official
HiFiTTS-2 offsets, and the temporary chapter audio is removed immediately.
Per-chapter completion records make the operation safe to resume.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf


DATASET_ROOT = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m")
REVISION_ROOT = DATASET_ROOT / "revisions/speech_expansion_noalign_15s_v1"
DEFAULT_CANDIDATES = (
    REVISION_ROOT / "sources/candidates/hifitts2_download_pending_qc.parquet"
)
DEFAULT_CHAPTERS = (
    REVISION_ROOT / "sources/candidates/hifitts2_selected_chapters.jsonl"
)
DEFAULT_OUTPUT = REVISION_ROOT / "sources/materialized/nvidia_hifitts2_44khz"
SAMPLE_RATE = 44_100
VAE_HOP_SAMPLES = 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_checked(command: list[str], *, label: str) -> None:
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode:
        tail = result.stderr[-4000:].replace("\n", " ")
        raise RuntimeError(f"{label} failed ({result.returncode}): {tail}")


def safe_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_chapters(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len({str(row["chapter_filepath"]) for row in rows}) != len(rows):
        raise RuntimeError("frozen chapter manifest is empty or non-unique")
    for row in rows:
        utterances = row.get("selected_utterances")
        if not isinstance(utterances, list) or not utterances:
            raise RuntimeError(
                "chapter manifest lacks frozen selected_utterances; rerun the candidate planner"
            )
    return rows


def marker_is_complete(
    marker: Path, chapter: dict[str, Any], audio_root: Path
) -> dict[str, Any] | None:
    if not marker.is_file():
        return None
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if (
        value.get("schema") != "stable_audio_tools.hifitts2_materialized_chapter"
        or value.get("state") != "complete"
        or value.get("chapter_filepath") != chapter["chapter_filepath"]
        or len(value.get("utterances") or ())
        != len(chapter["selected_utterances"])
    ):
        return None
    expected = {
        str(row["audio_filepath"]) for row in chapter["selected_utterances"]
    }
    observed = {str(row["audio_filepath"]) for row in value["utterances"]}
    if expected != observed:
        return None
    for row in value["utterances"]:
        path = audio_root / str(row["audio_filepath"])
        if not path.is_file() or path.stat().st_size <= 64:
            return None
    return value


def materialize_chapter(
    chapter: dict[str, Any],
    *,
    audio_root: Path,
    marker_root: Path,
    work_root: Path,
) -> dict[str, Any]:
    chapter_path = str(chapter["chapter_filepath"])
    key = safe_key(chapter_path)
    marker = marker_root / key[:2] / f"{key}.json"
    existing = marker_is_complete(marker, chapter, audio_root)
    if existing is not None:
        return {"state": "skip_verified", "marker": str(marker), **existing}

    started = time.monotonic()
    work = work_root / key
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=False)
    mp3 = work / "chapter.mp3"
    decoded = work / "chapter.flac"
    try:
        run_checked(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "12",
                "--retry-delay",
                "2",
                "--retry-all-errors",
                "--connect-timeout",
                "30",
                "--max-time",
                "3600",
                "--output",
                str(mp3),
                str(chapter["url"]),
            ],
            label=f"download {chapter_path}",
        )
        if mp3.stat().st_size <= 1024:
            raise RuntimeError(f"downloaded chapter is implausibly small: {chapter_path}")
        run_checked(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(mp3),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-sample_fmt",
                "s16",
                "-f",
                "flac",
                str(decoded),
            ],
            label=f"decode {chapter_path}",
        )
        results: list[dict[str, Any]] = []
        with sf.SoundFile(decoded, mode="r") as source:
            if source.channels != 1 or source.samplerate != SAMPLE_RATE:
                raise RuntimeError(f"decoded chapter geometry changed: {chapter_path}")
            for item in chapter["selected_utterances"]:
                audio_filepath = str(item["audio_filepath"])
                offset = float(item["offset_sec"])
                duration = float(item["duration_sec"])
                first = int(round(offset * SAMPLE_RATE))
                frames = int(round(duration * SAMPLE_RATE))
                if first < 0 or frames <= 0 or first + frames > len(source) + 2:
                    raise RuntimeError(
                        f"utterance extraction lies outside chapter: {audio_filepath}"
                    )
                source.seek(first)
                samples = source.read(frames, dtype="float32", always_2d=False)
                if samples.ndim != 1 or len(samples) != frames:
                    raise RuntimeError(f"short utterance extraction: {audio_filepath}")
                if not np.isfinite(samples).all():
                    raise RuntimeError(f"non-finite utterance extraction: {audio_filepath}")
                centered = samples.astype(np.float64) - float(
                    np.mean(samples, dtype=np.float64)
                )
                rms = float(np.sqrt(np.mean(np.square(centered))))
                peak = float(np.max(np.abs(centered)))
                if rms < 1.0e-5 or peak < 1.0e-4:
                    raise RuntimeError(f"silent utterance extraction: {audio_filepath}")
                destination = audio_root / audio_filepath
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(
                    destination.name
                    + f".tmp.{os.getpid()}.{threading.get_ident()}.flac"
                )
                sf.write(
                    temporary,
                    samples,
                    SAMPLE_RATE,
                    format="FLAC",
                    subtype="PCM_16",
                )
                reopened = sf.info(temporary)
                if (
                    reopened.channels != 1
                    or reopened.samplerate != SAMPLE_RATE
                    or reopened.frames != frames
                ):
                    raise RuntimeError(f"utterance reopen mismatch: {audio_filepath}")
                os.replace(temporary, destination)
                results.append(
                    {
                        "audio_filepath": audio_filepath,
                        "source_audio_path": str(destination),
                        "source_audio_sha256": sha256_file(destination),
                        "native_sample_rate_hz": SAMPLE_RATE,
                        "native_num_samples": frames,
                        "model_num_samples": frames,
                        "latent_frames_valid": math.ceil(frames / VAE_HOP_SAMPLES),
                        "signal_rms": rms,
                        "signal_peak": peak,
                        "offset_sec": offset,
                        "duration_sec": duration,
                    }
                )
        value = {
            "schema": "stable_audio_tools.hifitts2_materialized_chapter",
            "schema_version": 1,
            "state": "complete",
            "chapter_filepath": chapter_path,
            "chapter_url": str(chapter["url"]),
            "utterances": results,
            "elapsed_sec": round(time.monotonic() - started, 3),
        }
        atomic_json(marker, value)
        return {"state": "materialized", "marker": str(marker), **value}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def atomic_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    pq.write_table(table, temporary, compression="zstd", row_group_size=8192)
    if pq.read_metadata(temporary).num_rows != table.num_rows:
        raise RuntimeError("materialized candidate Parquet reopen count mismatch")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--chapters", type=Path, default=DEFAULT_CHAPTERS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit-chapters", type=int)
    args = parser.parse_args()
    candidates_path = args.candidates.expanduser().resolve(strict=True)
    chapters_path = args.chapters.expanduser().resolve(strict=True)
    output = args.output_root.expanduser().resolve(strict=False)
    if not str(output).startswith(os.environ.get("AMBIT_DATA_ROOT", "data")):
        raise ValueError("HiFiTTS-2 materialized audio must remain on SDB")
    workers = int(args.workers)
    if not 1 <= workers <= 64:
        raise ValueError("workers must be within [1,64]")
    chapters = read_chapters(chapters_path)
    if args.limit_chapters is not None:
        chapters = chapters[: int(args.limit_chapters)]
    audio_root = output / "audio"
    marker_root = output / "done"
    work_root = output / ".work"
    audio_root.mkdir(parents=True, exist_ok=True)
    marker_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_chapter = {
            executor.submit(
                materialize_chapter,
                chapter,
                audio_root=audio_root,
                marker_root=marker_root,
                work_root=work_root,
            ): chapter
            for chapter in chapters
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(future_to_chapter), start=1
        ):
            chapter = future_to_chapter[future]
            try:
                value = future.result()
                results.append(value)
            except Exception as error:  # noqa: BLE001
                failures.append(
                    {
                        "chapter_filepath": str(chapter["chapter_filepath"]),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            if completed % 25 == 0 or completed == len(chapters):
                print(
                    json.dumps(
                        {
                            "event": "hifitts2_materialize_progress",
                            "chapters": f"{completed}/{len(chapters)}",
                            "materialized": len(results),
                            "failed": len(failures),
                            "utterances": sum(
                                len(row.get("utterances") or ()) for row in results
                            ),
                            "elapsed_sec": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )

    if failures:
        failure_path = output / "failures.json"
        atomic_json(failure_path, failures)
        raise RuntimeError(
            f"{len(failures)} HiFiTTS-2 chapters failed; resume after inspecting {failure_path}"
        )

    materialized_by_audio: dict[str, dict[str, Any]] = {}
    for chapter in results:
        for row in chapter["utterances"]:
            key = str(row["audio_filepath"])
            if key in materialized_by_audio:
                raise RuntimeError(f"duplicate materialized utterance: {key}")
            materialized_by_audio[key] = row
    candidates = pq.read_table(candidates_path).to_pylist()
    if args.limit_chapters is not None:
        allowed = materialized_by_audio.keys()
        candidates = [
            row
            for row in candidates
            if json.loads(row["locator_json"])["audio_filepath"] in allowed
        ]
    output_rows = []
    for candidate in candidates:
        locator = json.loads(candidate["locator_json"])
        audio_filepath = str(locator["audio_filepath"])
        materialized = materialized_by_audio.get(audio_filepath)
        if materialized is None:
            raise RuntimeError(f"missing materialized utterance: {audio_filepath}")
        row = dict(candidate)
        row.update(materialized)
        row["qc_state"] = "materialized_pending_strong_qc"
        output_rows.append(row)
    if len(output_rows) != len(candidates):
        raise RuntimeError("materialized candidate row count changed")
    table = pa.Table.from_pylist(output_rows)
    manifest = output / "materialized_pending_strong_qc.parquet"
    atomic_parquet(manifest, table)
    summary = {
        "schema": "stable_audio_tools.hifitts2_materialization_summary",
        "schema_version": 1,
        "state": "complete_pending_strong_qc",
        "chapters": len(chapters),
        "utterances": len(output_rows),
        "length_bucket_counts": dict(
            Counter(str(row["length_bucket_frames"]) for row in output_rows)
        ),
        "speakers": len({str(row["speaker_key"]) for row in output_rows}),
        "source_audio_bytes": sum(
            Path(row["source_audio_path"]).stat().st_size for row in output_rows
        ),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "elapsed_sec": round(time.monotonic() - started, 3),
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
