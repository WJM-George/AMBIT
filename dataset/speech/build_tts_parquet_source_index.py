#!/usr/bin/env python3
"""Build a lightweight source_id -> parquet file index for TTS parquet audio.

The renderer uses this index to avoid scanning every parquet shard for every
micro-batch. The index stores only source id, dataset, parquet path, and row
number within the parquet file; it does not duplicate audio bytes.
"""
from __future__ import annotations
import os

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


DATASET_ROOTS = {
    "libritts": Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/speech_dataset/mythicinfinity__libritts"),
    "hifi_tts": Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/speech_dataset/MikhailT__hifi-tts"),
}


def plan_needed(plan_paths: list[Path]) -> dict[str, set[str]]:
    needed = {"libritts": set(), "hifi_tts": set()}
    for path in plan_paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                ds = row.get("source_dataset")
                sid = row.get("source_id")
                if ds in needed and sid:
                    needed[ds].add(sid)
    return needed


def hifi_id_from_row(row: dict) -> str | None:
    path = None
    audio = row.get("audio")
    if isinstance(audio, dict):
        path = audio.get("path")
    path = path or row.get("path") or row.get("file")
    if not path:
        return None
    return "hifi_tts_" + Path(path).stem


def build_index(plan_paths: list[Path], out_path: Path) -> dict:
    needed = plan_needed(plan_paths)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "needed": {k: len(v) for k, v in needed.items()},
        "datasets": {},
        "out_path": str(out_path),
    }
    written = Counter()
    with out_path.open("w", encoding="utf-8") as out:
        for dataset, ids in needed.items():
            root = DATASET_ROOTS[dataset]
            remaining = set(ids)
            parquet_files = sorted(root.rglob("*.parquet"))
            scanned_files = scanned_rows = matched = 0
            for parquet_path in parquet_files:
                if not remaining:
                    break
                scanned_files += 1
                if dataset == "libritts":
                    cols = ["id", "audio.path", "path"]
                else:
                    cols = ["file", "audio.path", "speaker"]
                pf = pq.ParquetFile(parquet_path)
                global_row_idx = 0
                for row_group in range(pf.num_row_groups):
                    table = pf.read_row_group(row_group, columns=cols)
                    rows = table.to_pylist()
                    scanned_rows += len(rows)
                    for row_in_group, row in enumerate(rows):
                        if dataset == "libritts":
                            source_id = row.get("id")
                        else:
                            source_id = hifi_id_from_row(row)
                        if source_id not in remaining:
                            continue
                        audio_path = row.get("path") or row.get("file")
                        rec = {
                            "source_dataset": dataset,
                            "source_id": source_id,
                            "parquet_path": str(parquet_path),
                            "row_index": global_row_idx + row_in_group,
                            "row_group": row_group,
                            "row_in_group": row_in_group,
                            "audio_path_in_parquet": audio_path,
                        }
                        out.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                        remaining.remove(source_id)
                        matched += 1
                        written[dataset] += 1
                    global_row_idx += len(rows)
                    del table, rows
            summary["datasets"][dataset] = {
                "needed": len(ids),
                "matched": matched,
                "missing": len(remaining),
                "missing_sample": sorted(remaining)[:20],
                "scanned_files": scanned_files,
                "scanned_rows": scanned_rows,
            }
    summary["written"] = dict(written)
    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    args = ap.parse_args()
    summary = build_index(args.plan, args.out)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
