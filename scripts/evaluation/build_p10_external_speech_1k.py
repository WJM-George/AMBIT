#!/usr/bin/env python3
"""Build the balanced 1,000-row external P10 speech benchmark candidate set.

The provisional core contains 500 FLEURS en-US rows and 500 VCTK rows.  VCTK
supplies speaker age, gender, accent, and region metadata that FLEURS does not,
while FLEURS supplies a distinct multilingual-benchmark recording domain.  The
final core is deliberately balanced to 500 female and 500 male metadata labels.

This script does not declare the benchmark frozen.  It performs metadata,
duration, normalized-transcript, and exact-file-hash gates.  Acoustic
fingerprint comparison against P10 train remains a mandatory separate gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import requests
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.build_p10_external_speech_candidates import (
    load_p10_train_identity,
)
from scripts.t2a.data.sceneplan_v2_common import normalized_transcript


SCHEMA = "sceneplan_foa.p10_external_speech_1k_candidate_pool"
SCHEMA_VERSION = 1
VCTK_DATASET_ID = "sanchit-gandhi/vctk"
VCTK_REVISION = "73ef4ee7d49a6fed4ea1efd65f82b4c95faeb9de"
VCTK_LICENSE = "cc-by-4.0"
VCTK_UPSTREAM_ROWS = 88_156
VIEWER_BASE = "https://datasets-server.huggingface.co"
HUB_API_BASE = "https://huggingface.co/api/datasets"
MIN_DURATION_SEC = 3.0
MAX_DURATION_SEC = 10.0

# FLEURS has only 151 eligible male-labelled rows.  Complementary VCTK quotas
# make the complete 1k core exactly balanced without discarding useful FLEURS
# speaker diversity.
FLEURS_CORE_QUOTA = {"male": 151, "female": 349}
VCTK_CORE_QUOTA = {"male": 349, "female": 151}
VCTK_RESERVE_QUOTA = {"male": 36, "female": 19}
ACCENT_DISPLAY_NAMES = {
    "NorthernIrish": "Northern Irish",
    "NewZealand": "New Zealand",
    "SouthAfrican": "South African",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sceneplan-root",
        type=Path,
        default=Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get("AMBIT_DATA_ROOT", "data") + "/evaluation_benchmark/"
            "p10_evaluation_benchmark_v1"
        ),
    )
    parser.add_argument("--metadata-blocks", type=int, default=96)
    parser.add_argument("--metadata-block-length", type=int, default=50)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--request-timeout-sec", type=float, default=60.0)
    parser.add_argument("--refresh-metadata", action="store_true")
    return parser.parse_args()


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_rank(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    keys = sorted({key for record in records for key in record})
    normalized = [{key: record.get(key) for key in keys} for record in records]
    pq.write_table(
        pa.Table.from_pylist(normalized),
        temporary,
        compression="zstd",
        compression_level=9,
    )
    os.replace(temporary, path)


def request_json(url: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(8):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "ScenePlan-P10-evaluation/1.0"},
            )
            if response.status_code == 429:
                last_error = requests.HTTPError(
                    f"429 rate limited: {response.url}", response=response
                )
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 0.0
                except ValueError:
                    delay = 0.0
                if delay <= 0:
                    delay = min(60.0, 5.0 * (2**attempt))
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as error:  # network retry path
            last_error = error
            if attempt == 7:
                break
            time.sleep(min(30.0, 1.5 * (2**attempt)))
    assert last_error is not None
    raise last_error


def verify_vctk_revision(timeout: float) -> None:
    payload = request_json(
        f"{HUB_API_BASE}/{VCTK_DATASET_ID}", {}, timeout=timeout
    )
    observed = str(payload.get("sha") or "")
    if observed != VCTK_REVISION:
        raise RuntimeError(
            f"VCTK revision changed: expected {VCTK_REVISION}, observed {observed}"
        )


def viewer_rows(offset: int, length: int, timeout: float) -> list[dict[str, Any]]:
    payload = request_json(
        f"{VIEWER_BASE}/rows",
        {
            "dataset": VCTK_DATASET_ID,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        },
        timeout=timeout,
    )
    return list(payload.get("rows") or [])


def metadata_offsets(blocks: int, block_length: int) -> list[int]:
    if blocks < 2:
        raise ValueError("metadata-blocks must be >= 2")
    high = VCTK_UPSTREAM_ROWS - block_length
    return sorted({round(index * high / (blocks - 1)) for index in range(blocks)})


def fetch_metadata_sample(args: argparse.Namespace, cache_path: Path) -> list[dict[str, Any]]:
    if cache_path.is_file() and not args.refresh_metadata:
        return read_jsonl(cache_path)

    sampled: dict[int, dict[str, Any]] = {}
    offsets = metadata_offsets(args.metadata_blocks, args.metadata_block_length)
    # Dataset Viewer enforces a much lower request-rate limit than the signed
    # static audio assets.  Six metadata workers retain most of the latency
    # gain without repeatedly tripping HTTP 429.
    workers = min(max(1, args.download_workers), 6)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                viewer_rows,
                offset,
                args.metadata_block_length,
                args.request_timeout_sec,
            ): offset
            for offset in offsets
        }
        for page_index, future in enumerate(as_completed(futures), start=1):
            rows = future.result()
            for wrapped in rows:
                row_index = int(wrapped["row_idx"])
                row = dict(wrapped["row"])
                audio_cell = row.pop("audio", None) or []
                audio_url = audio_cell[0].get("src") if audio_cell else None
                row["source_dataset_row"] = row_index
                row["audio_url"] = audio_url
                sampled[row_index] = row
            if page_index % 16 == 0 or page_index == len(offsets):
                print(
                    f"metadata pages {page_index}/{len(offsets)}; "
                    f"sampled rows={len(sampled)}",
                    flush=True,
                )
    rows = [sampled[key] for key in sorted(sampled)]
    write_jsonl(cache_path, rows)
    return rows


def age_group(raw_age: str) -> str:
    try:
        age = int(raw_age)
    except (TypeError, ValueError):
        return "adult"
    if age < 30:
        return "young adult"
    if age < 50:
        return "adult"
    return "mature adult"


def gender_name(raw_gender: str) -> str | None:
    value = str(raw_gender or "").strip().upper()
    if value == "M":
        return "male"
    if value == "F":
        return "female"
    return None


def display_accent(raw_accent: str) -> str:
    value = str(raw_accent or "").strip()
    return ACCENT_DISPLAY_NAMES.get(value, value)


def display_region(raw_region: str, raw_accent: str) -> str:
    region = str(raw_region or "").strip()
    accent = str(raw_accent or "").strip()
    # These two VCTK metadata values describe the speaker's English-language
    # background rather than a geographic region.
    if region == "English":
        return ""
    if accent == "Australian" and region == "English Sydney":
        return "Sydney"
    return region


def accent_article(accent: str) -> str:
    return "an" if accent[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def speaker_description(row: dict[str, Any]) -> str:
    group = age_group(str(row.get("age") or ""))
    gender = gender_name(str(row.get("gender") or "")) or "adult"
    raw_accent = str(row.get("accent") or "").strip()
    accent = display_accent(raw_accent)
    region = display_region(str(row.get("region") or ""), raw_accent)
    pieces = [f"a {group} {gender} speaker"]
    if accent:
        pieces.append(f"with {accent_article(accent)} {accent} accent")
    if region:
        pieces.append(f"from {region}")
    return " ".join(pieces) + ", speaking clearly and naturally"


def make_prompt(description: str, transcript: str) -> str:
    return f'{description[0].upper() + description[1:]} says "{transcript}".'


def make_vctk_candidates(
    metadata: list[dict[str, Any]], train_transcript_hashes: set[str]
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_source_keys: set[tuple[str, str]] = set()
    for row in metadata:
        source_file = str(row.get("file") or "")
        if not source_file.endswith("_mic1.flac"):
            continue
        gender = gender_name(str(row.get("gender") or ""))
        if gender is None:
            continue
        transcript = str(row.get("text") or "").strip()
        normalized = normalized_transcript(transcript)
        normalized_hash = sha256_text(normalized)
        if not normalized or normalized_hash in train_transcript_hashes:
            continue
        speaker_id = str(row.get("speaker_id") or "").strip()
        text_id = str(row.get("text_id") or "").strip()
        source_key = (speaker_id, text_id)
        if not speaker_id or not text_id or source_key in seen_source_keys:
            continue
        seen_source_keys.add(source_key)
        row_index = int(row["source_dataset_row"])
        description = speaker_description(row)
        candidate_id = f"vctk:{speaker_id}:{text_id}:mic1:row_{row_index:05d}"
        candidates.append(
            {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "source_dataset": "cstr_vctk_0p92",
                "dataset_mirror": VCTK_DATASET_ID,
                "dataset_revision": VCTK_REVISION,
                "dataset_license": VCTK_LICENSE,
                "upstream_split": "train_unsplit_corpus",
                "source_dataset_row": row_index,
                "speaker_id": speaker_id,
                "text_id": text_id,
                "age": str(row.get("age") or ""),
                "age_group": age_group(str(row.get("age") or "")),
                "gender": gender,
                "accent": display_accent(str(row.get("accent") or "")),
                "region": display_region(
                    str(row.get("region") or ""),
                    str(row.get("accent") or ""),
                ),
                "speaker_description": description,
                "speaker_description_provenance": "VCTK_age_gender_accent_region_metadata_template",
                "exact_transcript": transcript,
                "normalized_transcript": normalized,
                "normalized_transcript_sha256": normalized_hash,
                "prompt": make_prompt(description, transcript),
                "source_file": source_file,
                "audio_url": row.get("audio_url"),
                "audio_path": None,
                "audio_sha256": None,
                "duration_sec": None,
                "sample_rate_hz": None,
                "num_channels": None,
                "lineage_gate_status": "pass",
                "exact_hash_gate_status": "pending",
                "acoustic_fingerprint_gate_status": "pending",
                "exclusion_reasons": [],
                "selection_rank": stable_rank(candidate_id),
            }
        )
    return candidates


def round_robin_by_speaker(
    rows: list[dict[str, Any]], gender: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["gender"] == gender:
            grouped[row["speaker_id"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: row["selection_rank"])
    speakers = sorted(
        grouped,
        key=lambda speaker: stable_rank(f"vctk-speaker:{gender}:{speaker}"),
    )
    ordered: list[dict[str, Any]] = []
    depth = 0
    while True:
        added = 0
        for speaker in speakers:
            values = grouped[speaker]
            if depth < len(values):
                ordered.append(values[depth])
                added += 1
        if added == 0:
            break
        depth += 1
    return ordered


def refresh_audio_url(row_index: int, timeout: float) -> str:
    rows = viewer_rows(row_index, 1, timeout)
    if len(rows) != 1 or int(rows[0]["row_idx"]) != row_index:
        raise RuntimeError(f"failed to refresh VCTK row {row_index}")
    audio_cell = rows[0]["row"].get("audio") or []
    if not audio_cell or not audio_cell[0].get("src"):
        raise RuntimeError(f"VCTK row {row_index} has no audio asset")
    return str(audio_cell[0]["src"])


def download_candidate(
    record: dict[str, Any], audio_root: Path, train_audio_hashes: set[str], timeout: float
) -> dict[str, Any]:
    record = dict(record)
    target = audio_root / (
        f"row_{int(record['source_dataset_row']):05d}_"
        f"{record['speaker_id']}_{record['text_id']}_mic1.wav"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        url = str(record.get("audio_url") or "")
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                if not url or attempt > 0:
                    url = refresh_audio_url(
                        int(record["source_dataset_row"]), timeout
                    )
                response = requests.get(url, timeout=timeout, stream=True)
                response.raise_for_status()
                temporary = target.with_suffix(
                    f".wav.tmp.{os.getpid()}.{threading.get_ident()}"
                )
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            handle.write(chunk)
                os.replace(temporary, target)
                break
            except Exception as error:  # network retry path
                last_error = error
                time.sleep(1.5 * (attempt + 1))
        else:
            assert last_error is not None
            record["lineage_gate_status"] = "fail"
            record["exact_hash_gate_status"] = "not_run"
            record["exclusion_reasons"] = [f"audio_download_failed:{type(last_error).__name__}"]
            return record

    try:
        info = sf.info(str(target))
        duration = float(info.frames) / float(info.samplerate)
        channels = int(info.channels)
    except Exception as error:
        record["lineage_gate_status"] = "fail"
        record["exact_hash_gate_status"] = "not_run"
        record["exclusion_reasons"] = [f"invalid_audio:{type(error).__name__}"]
        return record

    digest = sha256_file(target)
    record["audio_path"] = str(target)
    record["audio_sha256"] = digest
    record["duration_sec"] = duration
    record["sample_rate_hz"] = int(info.samplerate)
    record["num_channels"] = channels
    reasons: list[str] = []
    if channels != 1:
        reasons.append("not_mono")
    if not (MIN_DURATION_SEC <= duration <= MAX_DURATION_SEC):
        reasons.append("duration_outside_P10_3_to_10_sec")
    if digest in train_audio_hashes:
        reasons.append("exact_file_sha256_used_by_P10_train")
    record["lineage_gate_status"] = "pass" if not reasons else "fail"
    record["exact_hash_gate_status"] = "pass" if not reasons else "fail"
    record["exclusion_reasons"] = reasons
    return record


def collect_vctk_pool(
    candidates: list[dict[str, Any]],
    audio_root: Path,
    train_audio_hashes: set[str],
    workers: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pass_targets = {
        gender: VCTK_CORE_QUOTA[gender] + VCTK_RESERVE_QUOTA[gender]
        for gender in VCTK_CORE_QUOTA
    }
    orders = {
        gender: round_robin_by_speaker(candidates, gender)
        for gender in VCTK_CORE_QUOTA
    }
    cursors = {gender: 0 for gender in VCTK_CORE_QUOTA}
    passed: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    pass_counts: Counter[str] = Counter()

    while any(pass_counts[g] < pass_targets[g] for g in pass_targets):
        batch: list[dict[str, Any]] = []
        for gender in sorted(pass_targets):
            if pass_counts[gender] >= pass_targets[gender]:
                continue
            need = pass_targets[gender] - pass_counts[gender]
            take = min(max(need * 2, 32), 128)
            start = cursors[gender]
            stop = min(start + take, len(orders[gender]))
            batch.extend(orders[gender][start:stop])
            cursors[gender] = stop
        if not batch:
            raise RuntimeError(
                f"VCTK metadata sample exhausted before quotas: "
                f"passed={dict(pass_counts)}, targets={pass_targets}"
            )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    download_candidate,
                    record,
                    audio_root,
                    train_audio_hashes,
                    timeout,
                ): record["candidate_id"]
                for record in batch
            }
            for future in as_completed(futures):
                record = future.result()
                if not record["exclusion_reasons"]:
                    passed.append(record)
                    pass_counts[record["gender"]] += 1
                else:
                    excluded.append(record)
        print(
            f"VCTK downloaded={len(passed) + len(excluded)} "
            f"passed={dict(pass_counts)} excluded={len(excluded)}",
            flush=True,
        )
    passed.sort(key=lambda row: row["selection_rank"])
    excluded.sort(key=lambda row: row["selection_rank"])
    return passed, excluded


def select_quota(
    rows: list[dict[str, Any]], quotas: dict[str, int], *, speaker_round_robin: bool
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for gender, quota in quotas.items():
        if speaker_round_robin:
            ordered = round_robin_by_speaker(rows, gender)
        else:
            ordered = sorted(
                (row for row in rows if row["gender"] == gender),
                key=lambda row: row["selection_rank"],
            )
        if len(ordered) < quota:
            raise RuntimeError(
                f"not enough {gender} candidates: need={quota}, have={len(ordered)}"
            )
        selected.extend(ordered[:quota])
    selected.sort(key=lambda row: stable_rank(f"speech1k:{row['candidate_id']}"))
    return selected


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    output = dict(record)
    output.pop("audio_url", None)
    return output


def main() -> None:
    args = parse_args()
    verify_vctk_revision(args.request_timeout_sec)
    train_transcript_hashes, train_audio_hashes = load_p10_train_identity(
        args.sceneplan_root
    )

    manifest_root = args.output_root / "manifests/external_speech_1k"
    cache_path = manifest_root / "vctk_metadata_sample.jsonl"
    metadata = fetch_metadata_sample(args, cache_path)
    vctk_pre_download = make_vctk_candidates(metadata, train_transcript_hashes)
    print(
        f"VCTK metadata rows={len(metadata)} candidates={len(vctk_pre_download)} "
        f"speakers={len({row['speaker_id'] for row in vctk_pre_download})}",
        flush=True,
    )
    vctk_passed, vctk_excluded = collect_vctk_pool(
        vctk_pre_download,
        args.output_root / "staging/vctk_external_speech_audio",
        train_audio_hashes,
        args.download_workers,
        args.request_timeout_sec,
    )

    fleurs_path = (
        args.output_root
        / "manifests/external_speech/candidate_pool.jsonl"
    )
    if not fleurs_path.is_file():
        raise FileNotFoundError(fleurs_path)
    fleurs_candidates = read_jsonl(fleurs_path)

    fleurs_core = select_quota(
        fleurs_candidates, FLEURS_CORE_QUOTA, speaker_round_robin=False
    )
    vctk_core = select_quota(
        vctk_passed, VCTK_CORE_QUOTA, speaker_round_robin=True
    )
    provisional_core = [public_record(row) for row in fleurs_core + vctk_core]
    provisional_core.sort(key=lambda row: stable_rank(f"speech1k:{row['candidate_id']}"))
    all_candidates = [
        public_record(row) for row in fleurs_candidates + vctk_passed
    ]
    all_candidates.sort(key=lambda row: stable_rank(f"speech1k:{row['candidate_id']}"))
    vctk_excluded_public = [public_record(row) for row in vctk_excluded]

    write_jsonl(manifest_root / "candidate_pool.jsonl", all_candidates)
    write_parquet(manifest_root / "candidate_pool.parquet", all_candidates)
    write_jsonl(
        manifest_root / "vctk_candidate_exclusions.jsonl", vctk_excluded_public
    )
    write_parquet(
        manifest_root / "vctk_candidate_exclusions.parquet", vctk_excluded_public
    )
    write_jsonl(
        manifest_root / "provisional_core_1000.jsonl", provisional_core
    )
    write_parquet(
        manifest_root / "provisional_core_1000.parquet", provisional_core
    )

    summary = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "provisional_core_acoustic_fingerprint_gate_pending",
        "target_core_rows": 1_000,
        "provisional_core_rows": len(provisional_core),
        "provisional_dataset_counts": dict(
            sorted(Counter(row["source_dataset"] for row in provisional_core).items())
        ),
        "provisional_gender_counts": dict(
            sorted(Counter(row["gender"] for row in provisional_core).items())
        ),
        "provisional_vctk_speaker_count": len(
            {
                row["speaker_id"]
                for row in provisional_core
                if row["source_dataset"] == "cstr_vctk_0p92"
            }
        ),
        "candidate_rows": len(all_candidates),
        "candidate_dataset_counts": dict(
            sorted(Counter(row["source_dataset"] for row in all_candidates).items())
        ),
        "candidate_gender_counts": dict(
            sorted(Counter(row["gender"] for row in all_candidates).items())
        ),
        "vctk_metadata_sample_rows": len(metadata),
        "vctk_metadata_candidate_rows": len(vctk_pre_download),
        "vctk_passed_rows": len(vctk_passed),
        "vctk_excluded_rows": len(vctk_excluded),
        "vctk_exclusion_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in vctk_excluded
                    for reason in row["exclusion_reasons"]
                ).items()
            )
        ),
        "source_contract": {
            "fleurs": {
                "rows": 500,
                "revision": "70bb2e84b976b7e960aa89f1c648e09c59f894dd",
                "license": "cc-by-4.0",
            },
            "vctk": {
                "rows": 500,
                "mirror": VCTK_DATASET_ID,
                "revision": VCTK_REVISION,
                "license": VCTK_LICENSE,
                "official_doi": "10.7488/ds/2645",
            },
        },
        "gate_contract": {
            "duration_3_to_10_sec": "complete",
            "exact_transcript_present": "complete",
            "normalized_transcript_vs_P10_train": "complete",
            "exact_file_sha256_vs_P10_train": "complete",
            "acoustic_fingerprint_vs_P10_train": "pending",
        },
    }
    (args.output_root / "EXTERNAL_SPEECH_1K_CANDIDATE_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
