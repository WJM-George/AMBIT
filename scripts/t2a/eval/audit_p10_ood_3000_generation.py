#!/usr/bin/env python3
"""Fail-closed audit for all 3,000 P10 OOD outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.benchmark_root.expanduser().resolve(strict=True)
    contract_path = (root / "BENCHMARK_CONTRACT.json").resolve(strict=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    panel_path = Path(contract["source_panel_path"]).resolve(strict=True)
    if _sha256(panel_path) != contract["source_panel_sha256"]:
        raise RuntimeError("OOD panel SHA256 changed")
    panel = _read_jsonl(panel_path)
    if len(panel) != 3000:
        raise RuntimeError("OOD panel row count changed")

    peaks: list[float] = []
    rms_values: list[float] = []
    bytes_total = 0
    for index, row in enumerate(panel, start=1):
        sample_root = root / "outputs" / "ours_p10_150k" / row["panel_id"]
        metadata_path = sample_root / "generation.json"
        native = sample_root / "native_foa.wav"
        quality = sample_root / "quality_w.wav"
        if not (metadata_path.is_file() and native.is_file() and quality.is_file()):
            raise RuntimeError(f"missing P10 OOD output: {row['panel_id']}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not (
            metadata.get("status") == "PASS"
            and metadata.get("system_id") == "ours_p10_150k"
            and metadata.get("panel_id") == row["panel_id"]
            and metadata.get("sample_id") == row["sample_id"]
            and metadata.get("domain") == row["domain"]
            and metadata.get("reference_audio_sha256")
            == row["reference_audio_sha256"]
            and metadata.get("checkpoint_sha256")
            == contract["p10"]["checkpoint_sha256"]
        ):
            raise RuntimeError(f"metadata mismatch: {metadata_path}")
        if _sha256(native) != metadata["native_foa_sha256"]:
            raise RuntimeError(f"native SHA256 mismatch: {native}")
        if _sha256(quality) != metadata["quality_w_sha256"]:
            raise RuntimeError(f"quality SHA256 mismatch: {quality}")
        native_info = sf.info(native)
        quality_info = sf.info(quality)
        expected_frames = int(round(float(row["requested_duration_sec"]) * 44_100))
        if not (
            native_info.channels == 4
            and quality_info.channels == 1
            and native_info.samplerate == quality_info.samplerate == 44_100
            and native_info.frames == quality_info.frames == expected_frames
        ):
            raise RuntimeError(f"waveform shape changed: {row['panel_id']}")
        audio, _ = sf.read(native, dtype="float32", always_2d=True)
        if not np.isfinite(audio).all():
            raise RuntimeError(f"non-finite P10 OOD output: {native}")
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        if rms <= 1.0e-7:
            raise RuntimeError(f"near-silent P10 OOD output: {native}")
        peaks.append(peak)
        rms_values.append(rms)
        bytes_total += native.stat().st_size + quality.stat().st_size
        if index % 100 == 0 or index == len(panel):
            print(json.dumps({"event": "p10_ood_audit", "completed": index}), flush=True)

    summary = {
        "schema": "sceneplan_foa.p10_ood_generation_summary",
        "schema_version": 1,
        "status": "PASS",
        "outputs": len(panel),
        "by_domain": dict(sorted(Counter(row["domain"] for row in panel).items())),
        "all_finite": True,
        "minimum_rms": min(rms_values),
        "maximum_peak": max(peaks),
        "bytes": bytes_total,
        "panel": str(panel_path),
        "panel_sha256": _sha256(panel_path),
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
    }
    _atomic_json(root / "P10_GENERATION_SUMMARY.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
