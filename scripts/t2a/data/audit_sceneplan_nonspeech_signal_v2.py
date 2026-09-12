#!/usr/bin/env python3
"""Decode and checksum every eligible dry sound/music asset before P7."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf


SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sceneplan_v2_common import DATASET_ROOT, atomic_write_json  # noqa: E402


DEFAULT_INPUT = DATASET_ROOT / "source_catalog/nonspeech/nonspeech_catalog.parquet"
DEFAULT_OUTPUT = DATASET_ROOT / "source_catalog/nonspeech/nonspeech_signal_catalog.parquet"
MIN_RMS = 1e-5
MIN_PEAK = 1e-4
MIN_ACTIVE_100MS_FRACTION = 0.01


OUTPUT_SCHEMA = pa.schema(
    [
        ("asset_id", pa.string()),
        ("source_id", pa.string()),
        ("source_dataset", pa.string()),
        ("kind", pa.string()),
        ("description", pa.string()),
        ("dry_audio_path", pa.string()),
        ("identity_hash", pa.string()),
        ("source_audio_sha256", pa.string()),
        ("native_sample_rate_hz", pa.int32()),
        ("native_num_samples", pa.int64()),
        ("native_channels", pa.int16()),
        ("model_sample_rate_hz", pa.int32()),
        ("model_num_samples", pa.int64()),
        ("duration_sec", pa.float64()),
        ("file_num_bytes", pa.int64()),
        ("file_mtime_ns", pa.int64()),
        ("eligible", pa.bool_()),
        ("rejection_reason", pa.string()),
        ("selection_rank", pa.string()),
        ("lineage_policy", pa.string()),
        ("signal_rms", pa.float64()),
        ("signal_peak", pa.float64()),
        ("dc_abs", pa.float64()),
        ("active_100ms_fraction", pa.float64()),
        ("decoded_finite", pa.bool_()),
    ]
)


def inspect(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    reason = None
    digest = None
    rms = peak = dc = active = 0.0
    finite = False
    try:
        path = Path(row["dry_audio_path"])
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        audio, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
        if audio.shape != (int(row["native_num_samples"]), 1):
            reason = "decoded_geometry_changed"
        elif int(rate) != int(row["native_sample_rate_hz"]):
            reason = "decoded_sample_rate_changed"
        else:
            mono = audio[:, 0]
            finite = bool(np.isfinite(mono).all())
            if not finite:
                reason = "non_finite_signal"
            else:
                rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
                peak = float(np.max(np.abs(mono)))
                dc = abs(float(np.mean(mono, dtype=np.float64)))
                block = max(1, int(rate) // 10)
                active = float(
                    np.mean(
                        [
                            np.sqrt(
                                np.mean(
                                    np.square(mono[start : start + block], dtype=np.float64)
                                )
                            )
                            >= MIN_RMS
                            for start in range(0, len(mono), block)
                        ]
                    )
                )
                if rms < MIN_RMS:
                    reason = "signal_rms_too_low"
                elif peak < MIN_PEAK:
                    reason = "signal_peak_too_low"
                elif active < MIN_ACTIVE_100MS_FRACTION:
                    reason = "signal_active_fraction_too_low"
    except Exception as exc:  # noqa: BLE001
        reason = f"decode_or_checksum_error:{type(exc).__name__}"
    result.update(
        {
            "identity_hash": digest or str(row["identity_hash"]),
            "source_audio_sha256": digest,
            "eligible": reason is None,
            "rejection_reason": reason,
            "lineage_policy": "complete_file_sha256_plus_full_decode_signal_qc_v2",
            "signal_rms": rms,
            "signal_peak": peak,
            "dc_abs": dc,
            "active_100ms_fraction": active,
            "decoded_finite": finite,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=min(96, os.cpu_count() or 1))
    parser.add_argument("--batch-rows", type=int, default=10_000)
    args = parser.parse_args()
    source = args.input.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    try:
        output.relative_to("/mnt/sdb")
    except ValueError as error:
        raise ValueError(f"signal catalog must be on SDB: {output}") from error
    table = pq.read_table(source, filters=[("eligible", "=", True)])
    input_rows = table.to_pylist()
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(temporary, OUTPUT_SCHEMA, compression="zstd")
    pending = []
    counts: Counter[str] = Counter()
    by_kind: defaultdict[str, Counter[str]] = defaultdict(Counter)
    started = time.time()
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for index, result in enumerate(pool.map(inspect, input_rows, chunksize=8), start=1):
                status = "eligible" if result["eligible"] else str(result["rejection_reason"])
                counts[status] += 1
                by_kind[str(result["kind"])][status] += 1
                pending.append(result)
                if len(pending) >= args.batch_rows:
                    writer.write_table(pa.Table.from_pylist(pending, schema=OUTPUT_SCHEMA))
                    pending.clear()
                if index % 10_000 == 0:
                    print(
                        json.dumps(
                            {
                                "decoded": index,
                                "total": len(input_rows),
                                "eligible": counts["eligible"],
                                "rejected": index - counts["eligible"],
                                "elapsed_sec": round(time.time() - started, 1),
                            }
                        ),
                        flush=True,
                    )
            if pending:
                writer.write_table(pa.Table.from_pylist(pending, schema=OUTPUT_SCHEMA))
    finally:
        writer.close()
    os.replace(temporary, output)
    reopened = pq.read_table(output, columns=["asset_id", "source_audio_sha256", "eligible"])
    if reopened.num_rows != len(input_rows):
        raise RuntimeError("signal catalog row count changed after reopen")
    valid_hashes = [
        value
        for value, eligible in zip(
            reopened["source_audio_sha256"].to_pylist(), reopened["eligible"].to_pylist()
        )
        if eligible
    ]
    summary = {
        "schema": "stable_audio_tools.sceneplan_nonspeech_signal_catalog_summary",
        "schema_version": 2,
        "input": str(source),
        "output": str(output),
        "rows": len(input_rows),
        "eligible": counts["eligible"],
        "rejected": len(input_rows) - counts["eligible"],
        "status_counts": dict(counts),
        "by_kind": {kind: dict(values) for kind, values in sorted(by_kind.items())},
        "unique_eligible_content_sha256": len(set(valid_hashes)),
        "duplicate_eligible_content_sha256": len(valid_hashes) - len(set(valid_hashes)),
        "thresholds": {
            "min_rms": MIN_RMS,
            "min_peak": MIN_PEAK,
            "min_active_100ms_fraction": MIN_ACTIVE_100MS_FRACTION,
        },
        "elapsed_sec": round(time.time() - started, 3),
    }
    atomic_write_json(output.with_name("signal_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if counts["eligible"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
