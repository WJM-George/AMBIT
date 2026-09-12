#!/usr/bin/env python3
"""Calibrate speech-completion QC against deterministic hard-cut controls."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_tts_v2_pilot import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_WORK_ROOT,
    acoustic_endpoint,
    atomic_write_json,
    atomic_write_jsonl,
    iter_jsonl,
    read_parquet_source,
    transcribe,
)


DEFAULT_MANIFEST = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/pilots/tts_2k/pilot_manifest.jsonl"
)
DEFAULT_FULL_QC = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/pilots/tts_2k/qc/"
    "asr_distil_large_v3/speech_qc_manifest.jsonl"
)
CUT_SECONDS = (0.15, 0.35, 0.75)
CUT_FRACTIONS = (0.20, 0.40)


def select_rows(manifest: Path, full_qc: Path, per_cell: int) -> list[dict[str, Any]]:
    by_id = {row["sample_id"]: row for row in iter_jsonl(manifest)}
    cells: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for result in iter_jsonl(full_qc):
        row = by_id[str(result["sample_id"])]
        # Calibrate on examples whose complete dry audio the strong model
        # already transcribes exactly; otherwise a baseline recognition error
        # would be mislabeled as an effect of cutting.
        if float(result["dry"]["alignment"]["wer"]) != 0.0:
            continue
        cell = (str(row["source_dataset"]), int(row["strata"]["duration_bin"]))
        cells[cell].append(row)
    selected = []
    for dataset in ("libritts", "hifi_tts"):
        for bucket in range(5):
            rows = sorted(cells[(dataset, bucket)], key=lambda row: str(row["sample_id"]))
            if len(rows) < per_cell:
                raise RuntimeError(f"not enough exact baseline rows for {dataset}/bin{bucket}")
            selected.extend(rows[:per_cell])
    return selected


def worker(gpu: int, rows: list[dict[str, Any]], model_path: str, output: str) -> None:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_path, device="cuda", device_index=gpu, compute_type="float16")
    results = []
    for index, row in enumerate(rows, start=1):
        audio, rate = read_parquet_source(row)
        variants = []
        cut_specs = [(f"fixed_{value:.2f}s", value) for value in CUT_SECONDS]
        cut_specs.extend(
            (f"fraction_{fraction:.2f}", len(audio) / rate * fraction)
            for fraction in CUT_FRACTIONS
        )
        for cut_label, cut_sec in cut_specs:
            cut_samples = int(round(cut_sec * rate))
            if len(audio) <= cut_samples + int(0.20 * rate):
                continue
            value = audio[:-cut_samples]
            asr = transcribe(model, value, rate)
            variants.append(
                {
                    "cut_label": cut_label,
                    "cut_sec": cut_sec,
                    "cut_native_samples": cut_samples,
                    "remaining_native_samples": len(value),
                    "recognized_text": asr["recognized_text"],
                    "segments": asr["segments"],
                    "duration_sec": asr["duration_sec"],
                    "acoustic_endpoint": acoustic_endpoint(value, rate),
                }
            )
        results.append(
            {
                "sample_id": row["sample_id"],
                "source_dataset": row["source_dataset"],
                "duration_bin": int(row["strata"]["duration_bin"]),
                "reference_transcript": row["transcript"],
                "variants": variants,
            }
        )
        if index % 10 == 0:
            print(json.dumps({"gpu": gpu, "done": index, "total": len(rows)}), flush=True)
    atomic_write_jsonl(Path(output), results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--full-qc", type=Path, default=DEFAULT_FULL_QC)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_WORK_ROOT.parent / "cutoff_calibration_distil_large_v3",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--per-cell", type=int, default=16)
    args = parser.parse_args()
    output = args.output_root.expanduser().resolve(strict=False)
    if not str(output).startswith(os.environ.get("AMBIT_DATA_ROOT", "data")):
        raise ValueError("cutoff calibration output must be on SDB")
    output.mkdir(parents=True, exist_ok=True)
    rows = select_rows(args.manifest, args.full_qc, args.per_cell)
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    shards = [rows[index:: len(gpus)] for index in range(len(gpus))]
    context = mp.get_context("spawn")
    processes = []
    paths = []
    started = time.time()
    for index, (gpu, shard) in enumerate(zip(gpus, shards)):
        if not shard:
            continue
        path = output / f"cutoff_qc_shard_{index:02d}.jsonl"
        process = context.Process(target=worker, args=(gpu, shard, str(args.model), str(path)))
        process.start()
        processes.append(process)
        paths.append(path)
    for process in processes:
        process.join()
        if process.exitcode:
            raise RuntimeError(f"cutoff calibration worker exited {process.exitcode}")
    results = [row for path in paths for row in iter_jsonl(path)]
    results.sort(key=lambda row: str(row["sample_id"]))
    manifest = output / "cutoff_controls.jsonl"
    atomic_write_jsonl(manifest, results)
    summary = {
        "schema": "stable_audio_tools.tts_v2_cutoff_calibration",
        "schema_version": 1,
        "model": str(args.model),
        "complete_baseline_rows": len(results),
        "cut_variants": sum(len(row["variants"]) for row in results),
        "cut_seconds": list(CUT_SECONDS),
        "cut_fractions": list(CUT_FRACTIONS),
        "manifest": str(manifest),
        "elapsed_sec": round(time.time() - started, 3),
    }
    atomic_write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
