#!/usr/bin/env python3
"""Fingerprint-deduplicate and freeze the balanced P10 OOD 3k benchmark.

The upstream candidate builders already enforce official-split, lineage,
train exact-file hash, and (for Speech) normalized-transcript gates.  This
finalizer adds two missing benchmark-level guarantees:

* no repeated audio payload inside the external benchmark;
* no repeated Chromaprint fingerprint after decoding/resampling.

Music and Sound are selected from the shared 2,403-row public candidate pool.
Speech is selected from the 1,160-row VCTK/FLEURS pool while preserving the
frozen 500/500 dataset and 500/500 gender balance whenever possible.  Repeated
transcripts spoken by different speakers are allowed: they are distinct audio
observations and are useful for speaker robustness, while every transcript is
already disjoint from P10 train.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sqlite3
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = "sceneplan_foa.p10_ood_3000"
SCHEMA_VERSION = 1
SAMPLE_RATE_HZ = 44_100
REQUESTED_DURATION_SEC = 10.0
CHROMAPRINT_SAMPLE_RATE_HZ = 11_025
CHROMAPRINT_ALGORITHM = 1
NEAR_DUPLICATE_HAMMING_RATE = 0.12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path(
            "/mnt/sdb/audio_dataset/evaluation_benchmark/"
            "p10_evaluation_benchmark_v1/manifests"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/sdb/audio_dataset/evaluation_benchmark/p10_ood_3000_v1"
        ),
    )
    parser.add_argument("--fingerprint-workers", type=int, default=4)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    keys = sorted({key for row in rows for key in row})
    normalized = [{key: row.get(key) for key in keys} for row in rows]
    pq.write_table(
        pa.Table.from_pylist(normalized),
        temporary,
        compression="zstd",
        compression_level=9,
    )
    os.replace(temporary, path)


def read_parquet(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def init_fingerprint_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS fingerprints (
            candidate_id TEXT PRIMARY KEY,
            audio_path TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            file_mtime_ns INTEGER NOT NULL,
            audio_sha256 TEXT NOT NULL,
            fingerprint_sha256 TEXT NOT NULL,
            fingerprint_raw_b64 TEXT NOT NULL,
            fingerprint_words INTEGER NOT NULL,
            status TEXT NOT NULL,
            error TEXT
        )
        """
    )
    connection.commit()
    return connection


def fingerprint_audio(
    ffmpeg: str, candidate_id: str, audio_path: str, expected_sha256: str
) -> dict[str, Any]:
    path = Path(audio_path).resolve(strict=True)
    stat = path.stat()
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"audio SHA256 changed for {candidate_id}: "
            f"{observed_sha256} != {expected_sha256}"
        )
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-threads",
        "1",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(CHROMAPRINT_SAMPLE_RATE_HZ),
        "-algorithm",
        str(CHROMAPRINT_ALGORITHM),
        "-fp_format",
        "raw",
        "-f",
        "chromaprint",
        "-",
    ]
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode != 0:
        return {
            "candidate_id": candidate_id,
            "audio_path": str(path),
            "file_size": int(stat.st_size),
            "file_mtime_ns": int(stat.st_mtime_ns),
            "audio_sha256": observed_sha256,
            "fingerprint_sha256": "",
            "fingerprint_raw_b64": "",
            "fingerprint_words": 0,
            "status": "FAIL",
            "error": (
                "ffmpeg_chromaprint_failed:"
                + result.stderr.decode("utf-8", errors="replace")[-2000:]
            ),
        }
    raw = result.stdout
    if len(raw) < 16 or len(raw) % 4:
        return {
            "candidate_id": candidate_id,
            "audio_path": str(path),
            "file_size": int(stat.st_size),
            "file_mtime_ns": int(stat.st_mtime_ns),
            "audio_sha256": observed_sha256,
            "fingerprint_sha256": "",
            "fingerprint_raw_b64": "",
            "fingerprint_words": 0,
            "status": "FAIL",
            "error": f"invalid_or_empty_chromaprint_payload:{len(raw)}_bytes",
        }
    return {
        "candidate_id": candidate_id,
        "audio_path": str(path),
        "file_size": int(stat.st_size),
        "file_mtime_ns": int(stat.st_mtime_ns),
        "audio_sha256": observed_sha256,
        "fingerprint_sha256": hashlib.sha256(raw).hexdigest(),
        "fingerprint_raw_b64": base64.b64encode(raw).decode("ascii"),
        "fingerprint_words": len(raw) // 4,
        "status": "PASS",
        "error": None,
    }


def load_or_compute_fingerprints(
    rows: list[dict[str, Any]], db_path: Path, workers: int, ffmpeg: str
) -> dict[str, dict[str, Any]]:
    connection = init_fingerprint_db(db_path)
    cached: dict[str, dict[str, Any]] = {}
    for row in connection.execute(
        "SELECT candidate_id,audio_path,file_size,file_mtime_ns,audio_sha256,"
        "fingerprint_sha256,fingerprint_raw_b64,fingerprint_words,status,error "
        "FROM fingerprints"
    ):
        cached[str(row[0])] = {
            "candidate_id": row[0],
            "audio_path": row[1],
            "file_size": row[2],
            "file_mtime_ns": row[3],
            "audio_sha256": row[4],
            "fingerprint_sha256": row[5],
            "fingerprint_raw_b64": row[6],
            "fingerprint_words": row[7],
            "status": row[8],
            "error": row[9],
        }

    pending: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = str(row["candidate_id"])
        path = Path(str(row["audio_path"])).resolve(strict=True)
        stat = path.stat()
        item = cached.get(candidate_id)
        if (
            item
            and item["status"] in {"PASS", "FAIL"}
            and item["audio_path"] == str(path)
            and int(item["file_size"]) == int(stat.st_size)
            and int(item["file_mtime_ns"]) == int(stat.st_mtime_ns)
            and item["audio_sha256"] == row["audio_sha256"]
        ):
            results[candidate_id] = item
        else:
            pending.append(row)

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(
                    fingerprint_audio,
                    ffmpeg,
                    str(row["candidate_id"]),
                    str(row["audio_path"]),
                    str(row["audio_sha256"]),
                ): row
                for row in pending
            }
            completed = 0
            for future in as_completed(futures):
                item = future.result()
                connection.execute(
                    "INSERT OR REPLACE INTO fingerprints VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        item["candidate_id"],
                        item["audio_path"],
                        item["file_size"],
                        item["file_mtime_ns"],
                        item["audio_sha256"],
                        item["fingerprint_sha256"],
                        item["fingerprint_raw_b64"],
                        item["fingerprint_words"],
                        item["status"],
                        item["error"],
                    ),
                )
                connection.commit()
                results[str(item["candidate_id"])] = item
                completed += 1
                if completed % 100 == 0 or completed == len(pending):
                    print(
                        canonical_json(
                            {
                                "event": "fingerprint_progress",
                                "newly_completed": completed,
                                "newly_total": len(pending),
                                "cached": len(rows) - len(pending),
                            }
                        ),
                        flush=True,
                    )
    connection.close()
    if len(results) != len(rows):
        raise RuntimeError(f"fingerprint count mismatch: {len(results)} != {len(rows)}")
    return results


def exact_duplicate_exclusions(
    rows: list[dict[str, Any]], key: str, reason: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for value in sorted(groups):
        group = sorted(
            groups[value], key=lambda row: (str(row["selection_rank"]), str(row["candidate_id"]))
        )
        kept.append(group[0])
        for row in group[1:]:
            item = dict(row)
            item["ood_exclusion_reason"] = reason
            item["duplicate_of_candidate_id"] = group[0]["candidate_id"]
            excluded.append(item)
    return kept, excluded


def aligned_hamming_rate(left: bytes, right: bytes) -> float:
    if len(left) != len(right) or len(left) % 4:
        return 1.0
    a = np.frombuffer(left, dtype="<u4")
    b = np.frombuffer(right, dtype="<u4")
    # NumPy 1.x has no vectorized bit_count.  The candidate set is small after
    # exact fingerprint grouping, and this conservative path runs only within
    # equal-length buckets.
    errors = sum(int(value).bit_count() for value in np.bitwise_xor(a, b))
    return float(errors) / float(32 * len(a))


def near_duplicate_exclusions(
    rows: list[dict[str, Any]], fingerprints: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove only very close, aligned Chromaprint matches.

    Exact source-lineage checks already handle shifted YouTube excerpts.  The
    aligned gate is intentionally conservative so acoustically similar events
    (for example two different applause clips) are not collapsed.
    """

    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        words = int(fingerprints[str(row["candidate_id"])]["fingerprint_words"])
        buckets[words].append(row)
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for words in sorted(buckets):
        accepted: list[tuple[dict[str, Any], bytes]] = []
        ordered = sorted(
            buckets[words],
            key=lambda row: (str(row["selection_rank"]), str(row["candidate_id"])),
        )
        for row in ordered:
            raw = base64.b64decode(
                fingerprints[str(row["candidate_id"])]["fingerprint_raw_b64"]
            )
            duplicate_of = None
            duplicate_rate = None
            for prior, prior_raw in accepted:
                rate = aligned_hamming_rate(raw, prior_raw)
                if rate <= NEAR_DUPLICATE_HAMMING_RATE:
                    duplicate_of = prior
                    duplicate_rate = rate
                    break
            if duplicate_of is None:
                accepted.append((row, raw))
                kept.append(row)
            else:
                item = dict(row)
                item["ood_exclusion_reason"] = "near_duplicate_chromaprint"
                item["duplicate_of_candidate_id"] = duplicate_of["candidate_id"]
                item["chromaprint_hamming_rate"] = duplicate_rate
                excluded.append(item)
    return kept, excluded


def attach_fingerprint(
    rows: list[dict[str, Any]], fingerprints: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        fingerprint = fingerprints[str(row["candidate_id"])]
        item = dict(row)
        item["chromaprint_algorithm"] = CHROMAPRINT_ALGORITHM
        item["chromaprint_sample_rate_hz"] = CHROMAPRINT_SAMPLE_RATE_HZ
        item["chromaprint_sha256"] = fingerprint["fingerprint_sha256"]
        item["chromaprint_words"] = int(fingerprint["fingerprint_words"])
        item["acoustic_fingerprint_gate_status"] = "pass"
        output.append(item)
    return output


def select_music_sound(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for modality in ("music", "sound"):
        candidates = sorted(
            (row for row in rows if row["modality"] == modality),
            key=lambda row: (str(row["selection_rank"]), str(row["candidate_id"])),
        )
        if len(candidates) < 1000:
            raise RuntimeError(f"insufficient {modality} after dedup: {len(candidates)}")
        selected.extend(candidates[:1000])
    return selected


def select_speech(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quotas = {
        ("google_fleurs", "female"): 349,
        ("google_fleurs", "male"): 151,
        ("cstr_vctk_0p92", "female"): 151,
        ("cstr_vctk_0p92", "male"): 349,
    }
    selected: list[dict[str, Any]] = []
    for key, quota in quotas.items():
        dataset, gender = key
        candidates = sorted(
            (
                row
                for row in rows
                if row["source_dataset"] == dataset and row["gender"] == gender
            ),
            key=lambda row: (str(row["selection_rank"]), str(row["candidate_id"])),
        )
        if len(candidates) < quota:
            raise RuntimeError(f"insufficient Speech quota {key}: {len(candidates)} < {quota}")
        selected.extend(candidates[:quota])
    return selected


def semantic_prompt(row: dict[str, Any], domain: str) -> str:
    if domain == "speech":
        return f"{str(row['speaker_description']).strip()} says: {str(row['exact_transcript']).strip()}"
    return str(row["prompt"]).strip()


def scene_plan(row: dict[str, Any], domain: str) -> dict[str, Any]:
    natural_duration = min(float(row["duration_sec"]), REQUESTED_DURATION_SEC)
    source: dict[str, Any] = {
        "kind": domain,
        "activity": {"onset_sec": 0.0, "offset_sec": natural_duration},
        "trajectory": {
            "type": "static",
            "position": {
                "azimuth_deg": 0.0,
                "elevation_deg": 0.0,
                "distance_m": 1.5,
            },
        },
        "gain_db": 0.0,
    }
    if domain == "speech":
        source["speaker_description"] = str(row["speaker_description"]).strip()
        source["transcript"] = str(row["exact_transcript"]).strip()
    else:
        source["description"] = str(row["prompt"]).strip()
    return {
        "duration_sec": REQUESTED_DURATION_SEC,
        "room": {"type": "dry"},
        "sources": [source],
    }


def build_panel(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_domain = {
        "music": sorted(
            (row for row in selected if row.get("modality") == "music"),
            key=lambda row: (str(row["selection_rank"]), str(row["candidate_id"])),
        ),
        "sound": sorted(
            (row for row in selected if row.get("modality") == "sound"),
            key=lambda row: (str(row["selection_rank"]), str(row["candidate_id"])),
        ),
        "speech": sorted(
            (
                row
                for row in selected
                if row.get("source_dataset") in {"google_fleurs", "cstr_vctk_0p92"}
            ),
            key=lambda row: (str(row["selection_rank"]), str(row["candidate_id"])),
        ),
    }
    panel: list[dict[str, Any]] = []
    ordinal = 0
    for domain in ("music", "sound", "speech"):
        if len(by_domain[domain]) != 1000:
            raise RuntimeError(f"frozen {domain} count changed: {len(by_domain[domain])}")
        for domain_ordinal, row in enumerate(by_domain[domain]):
            sample_id = f"ood_{domain}_{domain_ordinal:04d}"
            prompt = semantic_prompt(row, domain)
            plan = scene_plan(row, domain)
            panel.append(
                {
                    "schema": SCHEMA,
                    "schema_version": SCHEMA_VERSION,
                    "ordinal": ordinal,
                    "panel_id": sample_id,
                    "sample_id": sample_id,
                    "domain": domain,
                    "source_count": 1,
                    "source_kinds": [domain],
                    "semantic_text": prompt,
                    "source_semantic_texts": [prompt],
                    "requested_duration_sec": REQUESTED_DURATION_SEC,
                    "reference_duration_sec": float(row["duration_sec"]),
                    "reference_audio_path": str(Path(row["audio_path"]).resolve(strict=True)),
                    "reference_audio_sha256": str(row["audio_sha256"]),
                    "scene_plan": plan,
                    "noise_seed": int(
                        hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16
                    ),
                    "candidate_id": row["candidate_id"],
                    "source_dataset": row["source_dataset"],
                    "selection_rank": row["selection_rank"],
                    "chromaprint_sha256": row["chromaprint_sha256"],
                    "chromaprint_words": row["chromaprint_words"],
                    "exact_transcript": row.get("exact_transcript"),
                    "speaker_description": row.get("speaker_description"),
                    "gender": row.get("gender"),
                }
            )
            ordinal += 1
    return panel


def assert_unique(rows: list[dict[str, Any]], key: str) -> None:
    values = [row.get(key) for row in rows]
    if any(value is None for value in values) or len(values) != len(set(values)):
        raise RuntimeError(f"selected OOD set is not unique by {key}")


def main() -> int:
    args = parse_args()
    candidate_root = args.candidate_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    music_sound_path = candidate_root / "candidate_pool.parquet"
    speech_path = candidate_root / "external_speech_1k/candidate_pool.parquet"
    music_sound = read_parquet(music_sound_path)
    speech = read_parquet(speech_path)
    if len(music_sound) != 2403 or len(speech) != 1160:
        raise RuntimeError(
            f"candidate pools changed: Music/Sound={len(music_sound)}, Speech={len(speech)}"
        )
    candidates = music_sound + speech
    if len({str(row["candidate_id"]) for row in candidates}) != len(candidates):
        raise RuntimeError("candidate_id collision across OOD pools")

    fingerprints = load_or_compute_fingerprints(
        candidates,
        output_root / "audit/chromaprint_cache.sqlite",
        args.fingerprint_workers,
        args.ffmpeg,
    )

    fingerprint_failed: list[dict[str, Any]] = []
    fingerprint_candidates: list[dict[str, Any]] = []
    for row in candidates:
        fingerprint = fingerprints[str(row["candidate_id"])]
        if fingerprint["status"] == "PASS":
            fingerprint_candidates.append(row)
        else:
            item = dict(row)
            item["ood_exclusion_reason"] = "chromaprint_qc_failed"
            item["chromaprint_error"] = fingerprint["error"]
            fingerprint_failed.append(item)

    exact_clean, exact_excluded = exact_duplicate_exclusions(
        fingerprint_candidates,
        "audio_sha256",
        "duplicate_audio_sha256_within_ood_candidates",
    )
    fingerprint_rows = attach_fingerprint(exact_clean, fingerprints)
    fingerprint_clean, fingerprint_exact_excluded = exact_duplicate_exclusions(
        fingerprint_rows,
        "chromaprint_sha256",
        "duplicate_exact_chromaprint_within_ood_candidates",
    )
    near_clean, near_excluded = near_duplicate_exclusions(
        fingerprint_clean, fingerprints
    )

    music_sound_clean = [
        row for row in near_clean if row.get("modality") in {"music", "sound"}
    ]
    speech_clean = [
        row
        for row in near_clean
        if row.get("source_dataset") in {"google_fleurs", "cstr_vctk_0p92"}
    ]
    selected_music_sound = select_music_sound(music_sound_clean)
    selected_speech = select_speech(speech_clean)
    selected = selected_music_sound + selected_speech
    assert_unique(selected, "candidate_id")
    assert_unique(selected, "audio_sha256")
    assert_unique(selected, "chromaprint_sha256")

    panel = build_panel(selected)
    assert_unique(panel, "panel_id")
    assert_unique(panel, "reference_audio_sha256")
    assert_unique(panel, "chromaprint_sha256")

    manifests = output_root / "manifests"
    selected_path = manifests / "ood_3000_selected.jsonl"
    panel_path = manifests / "ood_3000_panel.jsonl"
    exclusions = (
        fingerprint_failed
        + exact_excluded
        + fingerprint_exact_excluded
        + near_excluded
    )
    exclusions_path = manifests / "ood_candidate_dedup_exclusions.jsonl"
    atomic_jsonl(selected_path, selected)
    atomic_parquet(manifests / "ood_3000_selected.parquet", selected)
    atomic_jsonl(panel_path, panel)
    atomic_parquet(manifests / "ood_3000_panel.parquet", panel)
    atomic_jsonl(exclusions_path, exclusions)
    atomic_parquet(manifests / "ood_candidate_dedup_exclusions.parquet", exclusions)

    selected_domain_counts = Counter(row["domain"] for row in panel)
    selected_dataset_counts = Counter(row["source_dataset"] for row in panel)
    selected_gender_counts = Counter(
        row["gender"] for row in panel if row["domain"] == "speech"
    )
    selected_transcript_hashes = {
        row.get("normalized_transcript_sha256")
        for row in selected_speech
        if row.get("normalized_transcript_sha256")
    }
    summary = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_FROZEN",
        "rows": len(panel),
        "domain_counts": dict(selected_domain_counts),
        "source_dataset_counts": dict(selected_dataset_counts),
        "speech_gender_counts": dict(selected_gender_counts),
        "speech_unique_normalized_transcripts": len(selected_transcript_hashes),
        "speech_repeated_transcript_policy": (
            "allowed only across distinct audio observations/speakers; every normalized "
            "transcript remains disjoint from P10 train"
        ),
        "candidate_counts": {
            "music_sound": len(music_sound),
            "speech": len(speech),
            "total": len(candidates),
        },
        "dedup": {
            "chromaprint_qc_failed": len(fingerprint_failed),
            "audio_sha256_excluded": len(exact_excluded),
            "exact_chromaprint_excluded": len(fingerprint_exact_excluded),
            "near_chromaprint_excluded": len(near_excluded),
            "near_duplicate_hamming_rate_threshold": NEAR_DUPLICATE_HAMMING_RATE,
            "selected_audio_sha256_unique": True,
            "selected_chromaprint_unique": True,
            "train_overlap_gates_inherited": [
                "source_lineage",
                "exact_file_sha256",
                "Speech_normalized_transcript_sha256",
            ],
            "scope_note": (
                "Chromaprint is exhaustive within the external candidate pool. "
                "Train separation is enforced by frozen lineage/exact-hash gates; "
                "the benchmark does not claim an all-836k train Chromaprint index."
            ),
        },
        "adapter": {
            "baseline_input": "raw semantic text only",
            "ours_input": "deterministic neutral single-source ScenePlan",
            "requested_duration_sec": REQUESTED_DURATION_SEC,
            "room": "dry",
            "trajectory": "static front, azimuth 0 deg, elevation 0 deg, distance 1.5 m",
            "spatial_metrics": "N/A because natural OOD references are mono",
            "common_content_reference": "reference natural mono audio",
        },
        "artifacts": {
            "selected_jsonl": str(selected_path),
            "selected_jsonl_sha256": sha256_file(selected_path),
            "panel_jsonl": str(panel_path),
            "panel_jsonl_sha256": sha256_file(panel_path),
            "dedup_exclusions_jsonl": str(exclusions_path),
            "dedup_exclusions_jsonl_sha256": sha256_file(exclusions_path),
            "music_sound_candidate_parquet": str(music_sound_path),
            "music_sound_candidate_parquet_sha256": sha256_file(music_sound_path),
            "speech_candidate_parquet": str(speech_path),
            "speech_candidate_parquet_sha256": sha256_file(speech_path),
        },
    }
    atomic_json(output_root / "OOD_3000_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
