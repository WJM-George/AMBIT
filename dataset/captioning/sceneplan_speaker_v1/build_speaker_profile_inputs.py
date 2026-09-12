#!/usr/bin/env python3
"""Build deterministic Qwen fallback speaker-profile inputs and composites."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly


DATASET_ROOT = Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m")
DEFAULT_LEDGER = DATASET_ROOT / "split_ledgers/speech_v2/speech_split_ledger.parquet"
DEFAULT_OUTPUT = DATASET_ROOT / "source_annotations/speech_speaker_instruct_v1"
LIBRITTS_P = DEFAULT_OUTPUT / "external/LibriTTS-P"
FORMAL_POOLS = {"train", "validation", "test"}
EXCLUDED = {"2074", "4455", "6032", "3546", "2262", "8097", "1734", "3793", "8295"}
HIFI_GENDER = {
    "92": "F", "6097": "M", "9017": "M", "6670": "M", "6671": "M",
    "8051": "F", "9136": "F", "11614": "F", "11697": "F", "12787": "F",
}


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temporary, path)


def load_prompt_speakers(root: Path) -> set[str]:
    result: set[str] = set()
    for name in ("df1_en.csv", "df2_en.csv", "df3_en.csv"):
        with (root / "data" / name).open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    result.add(line.split("|", 1)[0].strip())
    return result


def load_style(root: Path) -> dict[str, dict[str, str]]:
    path = root / "data/metadata_w_style_prompt_tags_v230922.csv"
    with path.open(encoding="utf-8", newline="") as source:
        return {str(row["item_name"]): row for row in csv.DictReader(source)}


def profile_key(row: dict[str, Any], prompts: set[str], style: dict[str, dict[str, str]]) -> str | None:
    if row["source_dataset"] == "hifi_tts":
        return f"hifi_tts:{row['speaker_id']}:{HIFI_GENDER[str(row['speaker_id'])]}"
    speaker = str(row["speaker_id"])
    if speaker not in prompts:
        gender = style.get(str(row["source_id"]), {}).get("gender") or "U"
        return f"libritts:{speaker}:{gender}"
    if speaker in EXCLUDED:
        gender = style.get(str(row["source_id"]), {}).get("gender") or "U"
        return f"libritts:{speaker}:{gender}"
    return None


def read_audio(row: dict[str, Any], target_sr: int = 24000) -> np.ndarray:
    parquet = pq.ParquetFile(str(row["parquet_path"]))
    table = parquet.read_row_group(int(row["row_group"]), columns=["audio"])
    audio = table.slice(int(row["row_in_group"]), 1).column("audio")[0].as_py()
    samples, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if int(sample_rate) != target_sr:
        divisor = math.gcd(int(sample_rate), target_sr)
        samples = resample_poly(samples, target_sr // divisor, int(sample_rate) // divisor).astype(np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 0.98:
        samples = samples * (0.98 / peak)
    return np.asarray(samples, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--libritts-p", type=Path, default=LIBRITTS_P)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    ledger = args.ledger.expanduser().resolve(strict=True)
    libri = args.libritts_p.expanduser().resolve(strict=True)
    output = args.output_root.expanduser().resolve(strict=False)
    rows = [
        row for row in pq.read_table(ledger).to_pylist()
        if str(row["pool"]) in FORMAL_POOLS
    ]
    prompts = load_prompt_speakers(libri)
    style = load_style(libri)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = profile_key(row, prompts, style)
        if key is not None:
            grouped[key].append(row)
    composites = output / "qwen_inputs/audio"
    composites.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    silence = np.zeros(round(0.35 * 24000), dtype=np.float32)
    for key in sorted(grouped):
        candidates = sorted(
            grouped[key],
            key=lambda row: (
                not (2.5 <= float(row["duration_sec"]) <= 8.0),
                abs(float(row["duration_sec"]) - 4.5),
                str(row["selection_rank"]),
            ),
        )
        chosen: list[dict[str, Any]] = []
        chapters: set[str] = set()
        for row in candidates:
            if str(row["chapter_id"]) in chapters and len(chosen) < 2:
                continue
            chosen.append(row)
            chapters.add(str(row["chapter_id"]))
            if len(chosen) == min(3, len(candidates)):
                break
        if len(chosen) < min(3, len(candidates)):
            chosen = candidates[: min(3, len(candidates))]
        chunks: list[np.ndarray] = []
        for index, row in enumerate(chosen):
            if index:
                chunks.append(silence)
            chunks.append(read_audio(row))
        composite = np.concatenate(chunks)
        safe = key.replace(":", "__")
        audio_path = composites / f"{safe}.wav"
        sf.write(audio_path, composite, 24000, subtype="PCM_16")
        gender = key.rsplit(":", 1)[-1]
        manifest.append(
            {
                "id": key,
                "audio_path": str(audio_path),
                "source_dataset": key.split(":", 1)[0],
                "speaker_id": key.split(":")[1],
                "official_gender": gender if gender in {"F", "M"} else None,
                "representative_asset_ids": [str(row["asset_id"]) for row in chosen],
                "representative_durations_sec": [float(row["duration_sec"]) for row in chosen],
            }
        )
    manifest_path = output / "qwen_inputs/speaker_profile_inputs.jsonl"
    atomic_jsonl(manifest_path, manifest)
    summary = {
        "schema": "stable_audio_tools.speaker_profile_qwen_input_summary",
        "schema_version": 1,
        "formal_ledger_rows": len(rows),
        "formal_speakers": len({str(row["speaker_key"]) for row in rows}),
        "qwen_profile_groups": len(manifest),
        "groups_by_dataset": dict(Counter(row["source_dataset"] for row in manifest)),
        "input_jsonl": str(manifest_path),
        "audio_root": str(composites),
    }
    summary_path = output / "qwen_inputs/summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
