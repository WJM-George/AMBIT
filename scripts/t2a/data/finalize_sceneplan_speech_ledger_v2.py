#!/usr/bin/env python3
"""Replace failed formal speech donors with strong-QC-passed compatible reserves."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sceneplan_v2_common import (  # noqa: E402
    DATASET_ROOT,
    atomic_write_json,
    require_dataset_not_frozen,
)


LEDGER = DATASET_ROOT / "split_ledgers/speech_v2/speech_split_ledger.parquet"
QC_ROOT = DATASET_ROOT / "source_catalog/speech/strong_qc"
AUDIT_ROOT = DATASET_ROOT / "audit/speech_ledger_strong_qc"
FORMAL = {"train", "validation", "test"}
EXPECTED = {
    ("libritts", "train"): 250_000,
    ("libritts", "validation"): 5_000,
    ("libritts", "test"): 1_000,
    ("hifi_tts", "train"): 250_000,
    ("hifi_tts", "validation"): 5_000,
    ("hifi_tts", "test"): 1_000,
}
DURATION_BINS = ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.031020408163266))


def duration_bin(seconds: float) -> int:
    for index, (start, stop) in enumerate(DURATION_BINS):
        if start <= seconds < stop or (index == len(DURATION_BINS) - 1 and seconds <= stop):
            return index
    raise ValueError(f"speech duration outside frozen bins: {seconds}")


def atomic_parquet(path: Path, table: pa.Table) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    pq.write_table(table, temporary, compression="zstd")
    if pq.read_metadata(temporary).num_rows != table.num_rows:
        raise RuntimeError(f"atomic Parquet reopen row count mismatch: {path}")
    os.replace(temporary, path)


def main() -> int:
    require_dataset_not_frozen()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--qc-root", type=Path, default=QC_ROOT)
    parser.add_argument("--audit-root", type=Path, default=AUDIT_ROOT)
    args = parser.parse_args()
    ledger_path = args.ledger.expanduser().resolve(strict=True)
    qc_root = args.qc_root.expanduser().resolve(strict=True)
    audit_root = args.audit_root.expanduser().resolve(strict=False)
    try:
        ledger_path.relative_to(os.environ.get("AMBIT_DATA_ROOT", "data"))
        audit_root.relative_to(os.environ.get("AMBIT_DATA_ROOT", "data"))
    except ValueError as error:
        raise ValueError("speech ledger and audit outputs must remain on SDB") from error
    qc_summary = json.loads((qc_root / "summary.json").read_text(encoding="utf-8"))
    if not qc_summary.get("audit_complete"):
        raise RuntimeError("strong speech QC is not complete")
    ledger_table = pq.read_table(ledger_path)
    rows = ledger_table.to_pylist()
    qc_table = pq.read_table(sorted(qc_root.glob("part-*.parquet")))
    if qc_table.num_rows != len(rows):
        raise RuntimeError("strong-QC/ledger row counts differ")
    qc_status_counts = Counter(map(str, qc_table["status"].to_pylist()))
    if qc_status_counts.get("error", 0):
        raise RuntimeError(
            "strong speech QC contains audit errors; repair and rerun those "
            f"rows before ledger finalization: {qc_status_counts}"
        )
    if set(qc_status_counts) - {"pass", "quarantine"}:
        raise RuntimeError(f"unknown strong speech QC status: {qc_status_counts}")
    qc_rows = {str(row["asset_id"]): row for row in qc_table.to_pylist()}
    if len(qc_rows) != len(rows):
        raise RuntimeError("strong-QC asset ids are not unique")
    if set(qc_rows) != {str(row["asset_id"]) for row in rows}:
        raise RuntimeError("strong-QC asset-id set differs from the ledger")

    failed_formal: list[dict[str, Any]] = []
    reserves: defaultdict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    fallback_reserves: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_id = {str(row["asset_id"]): row for row in rows}
    for row in rows:
        qc = qc_rows[str(row["asset_id"])]
        pool = str(row["pool"])
        passed = qc["status"] == "pass"
        row["lineage_qc"] = "strong_qc_pass_v2" if passed else "strong_qc_quarantine_v2"
        row["signal_qc"] = "pass" if passed else "quarantine"
        row["endpoint_qc"] = "pass" if qc["endpoint_natural"] else "quarantine"
        row["asr_qc"] = "pass_distil_large_v3" if qc["semantic_complete"] else "quarantine"
        if pool in FORMAL and not passed:
            failed_formal.append(row)
        elif pool == "reserve" and passed:
            key = (str(row["source_dataset"]), str(row["replacement_split"]))
            bin_index = duration_bin(float(row["duration_sec"]))
            reserves[(*key, bin_index)].append(row)
            fallback_reserves[key].append(row)
        elif pool == "reserve" and not passed:
            row["pool"] = "quarantine_reserve"

    for values in reserves.values():
        values.sort(key=lambda row: str(row["selection_rank"]))
    for values in fallback_reserves.values():
        values.sort(key=lambda row: str(row["selection_rank"]))
    used_replacements: set[str] = set()
    replacements = []
    failed_formal.sort(
        key=lambda row: (
            str(row["pool"]),
            str(row["source_dataset"]),
            duration_bin(float(row["duration_sec"])),
            str(row["selection_rank"]),
        )
    )
    for failed in failed_formal:
        split = str(failed["pool"])
        dataset = str(failed["source_dataset"])
        bin_index = duration_bin(float(failed["duration_sec"]))
        candidates = reserves[(dataset, split, bin_index)]
        candidate = None
        while candidates:
            value = candidates.pop(0)
            if str(value["asset_id"]) not in used_replacements and value["pool"] == "reserve":
                candidate = value
                break
        if candidate is None:
            for value in fallback_reserves[(dataset, split)]:
                if str(value["asset_id"]) not in used_replacements and value["pool"] == "reserve":
                    candidate = value
                    break
        if candidate is None:
            raise RuntimeError(
                f"no passed reserve for failed {dataset}/{split}/{failed['asset_id']}"
            )
        candidate_id = str(candidate["asset_id"])
        used_replacements.add(candidate_id)
        failed["pool"] = f"quarantine_{split}"
        candidate["pool"] = split
        replacements.append(
            {
                "split": split,
                "source_dataset": dataset,
                "failed_asset_id": str(failed["asset_id"]),
                "failed_duration_sec": float(failed["duration_sec"]),
                "failed_duration_bin": bin_index,
                "failure_reasons": qc_rows[str(failed["asset_id"])]["failure_reasons"],
                "replacement_asset_id": candidate_id,
                "replacement_duration_sec": float(candidate["duration_sec"]),
                "replacement_duration_bin": duration_bin(float(candidate["duration_sec"])),
                "same_duration_bin": duration_bin(float(candidate["duration_sec"])) == bin_index,
                "replacement_speaker_key": str(candidate["speaker_key"]),
            }
        )

    updated = pa.Table.from_pylist(rows, schema=ledger_table.schema)
    formal = updated.filter(pc.is_in(updated["pool"], value_set=pa.array(sorted(FORMAL))))
    if formal.num_rows != 512_000:
        raise RuntimeError(f"final formal speech count {formal.num_rows} != 512000")
    counts = Counter(
        zip(
            map(str, formal["source_dataset"].to_pylist()),
            map(str, formal["pool"].to_pylist()),
        )
    )
    if counts != Counter(EXPECTED):
        raise RuntimeError(f"final formal dataset/split quotas drift: {counts}")
    formal_ids = formal["asset_id"].to_pylist()
    if len(set(formal_ids)) != formal.num_rows:
        raise RuntimeError("final formal speech asset ids are not unique")
    for column in ("source_audio_sha256", "normalized_transcript"):
        if pc.count_distinct(formal[column]).as_py() != formal.num_rows:
            raise RuntimeError(f"final formal speech {column} values are not unique")
    if set(formal["signal_qc"].to_pylist()) != {"pass"}:
        raise RuntimeError("final formal speech signal QC is not all pass")
    if set(formal["endpoint_qc"].to_pylist()) != {"pass"}:
        raise RuntimeError("final formal speech endpoint QC is not all pass")
    if set(formal["asr_qc"].to_pylist()) != {"pass_distil_large_v3"}:
        raise RuntimeError("final formal speech ASR QC is not all pass")
    speaker_splits: dict[str, str] = {}
    for speaker, split in zip(
        formal["speaker_key"].to_pylist(), formal["pool"].to_pylist()
    ):
        previous = speaker_splits.setdefault(str(speaker), str(split))
        if previous != split:
            raise RuntimeError(f"speaker leaks across final splits: {speaker}")
    reserve = updated.filter(pc.equal(updated["pool"], "reserve"))
    if reserve.num_rows < 50_000:
        raise RuntimeError(f"final passed reserve {reserve.num_rows} < 50000")
    if set(reserve["asr_qc"].to_pylist()) != {"pass_distil_large_v3"}:
        raise RuntimeError("remaining reserve contains non-passing ASR rows")

    audit_root.mkdir(parents=True, exist_ok=True)
    backup = audit_root / "speech_split_ledger_before_strong_qc.parquet"
    if not backup.exists():
        shutil.copy2(ledger_path, backup)
    replacement_schema = pa.schema(
        [
            ("split", pa.string()),
            ("source_dataset", pa.string()),
            ("failed_asset_id", pa.string()),
            ("failed_duration_sec", pa.float64()),
            ("failed_duration_bin", pa.int8()),
            ("failure_reasons", pa.list_(pa.string())),
            ("replacement_asset_id", pa.string()),
            ("replacement_duration_sec", pa.float64()),
            ("replacement_duration_bin", pa.int8()),
            ("same_duration_bin", pa.bool_()),
            ("replacement_speaker_key", pa.string()),
        ]
    )
    atomic_parquet(
        audit_root / "replacement_map.parquet",
        pa.Table.from_pylist(replacements, schema=replacement_schema),
    )
    atomic_parquet(ledger_path, updated)
    reopened = pq.read_table(ledger_path)
    if reopened.num_rows != len(rows):
        raise RuntimeError("final speech ledger row count changed after atomic reopen")
    replacement_counts = Counter(
        (row["source_dataset"], row["split"]) for row in replacements
    )
    summary = {
        "schema": "stable_audio_tools.sceneplan_speech_split_ledger_strong_qc",
        "schema_version": 2,
        "ok": True,
        "ledger": str(ledger_path),
        "rows": len(rows),
        "formal_rows": formal.num_rows,
        "remaining_passed_reserve_rows": reserve.num_rows,
        "replacement_rows": len(replacements),
        "strong_qc_status_counts": dict(qc_status_counts),
        "replacement_same_duration_bin": sum(row["same_duration_bin"] for row in replacements),
        "replacement_counts": {
            "|".join(key): value for key, value in sorted(replacement_counts.items())
        },
        "formal_counts": {"|".join(key): value for key, value in sorted(counts.items())},
        "formal_signal_qc": "all_pass",
        "formal_endpoint_qc": "all_pass",
        "formal_asr_qc": "all_pass_distil_large_v3",
        "formal_unique_assets_audio_hashes_transcripts": True,
        "speaker_split_leakage": 0,
        "backup": str(backup),
        "replacement_map": str(audit_root / "replacement_map.parquet"),
    }
    atomic_write_json(audit_root / "summary.json", summary)
    atomic_write_json(
        ledger_path.with_name("strong_qc_summary.json"), summary
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
