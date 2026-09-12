#!/usr/bin/env python3
"""Audit every waveform produced for a matched P10 public-baseline contract."""

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

from baseline_common import output_is_valid


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    root = args.benchmark_root.expanduser().resolve(strict=True)
    manifest = (
        args.manifest.expanduser().resolve(strict=True)
        if args.manifest is not None
        else (root / "generation_requests.jsonl").resolve(strict=True)
    )
    contract_path = (root / "BENCHMARK_CONTRACT.json").resolve(strict=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(manifest)
    expected = int(contract["generation_request_count"])
    if len(rows) != expected:
        raise RuntimeError(f"manifest row count changed: {len(rows)} != {expected}")

    pairs = [(row["baseline_id"], row["panel_id"]) for row in rows]
    if len(pairs) != len(set(pairs)):
        raise RuntimeError("duplicate baseline/panel request")
    native_paths = [str(Path(row["native_output_path"]).resolve()) for row in rows]
    quality_paths = [str(Path(row["quality_w_path"]).resolve()) for row in rows]
    if len(native_paths) != len(set(native_paths)):
        raise RuntimeError("native output paths are not unique")
    if len(quality_paths) != len(set(quality_paths)):
        raise RuntimeError("quality output paths are not unique")

    contract_channels = {
        row["id"]: int(row["native_channels"]) for row in contract["baselines"]
    }
    peaks: list[float] = []
    rms_values: list[float] = []
    native_bytes = 0
    quality_bytes = 0
    for index, row in enumerate(rows, start=1):
        if not output_is_valid(row):
            raise RuntimeError(
                f"invalid waveform output: {row['baseline_id']}/{row['panel_id']}"
            )
        native_path = Path(row["native_output_path"]).resolve(strict=True)
        quality_path = Path(row["quality_w_path"]).resolve(strict=True)
        metadata_path = native_path.with_name("generation.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("baseline_id", "panel_id", "domain", "seed"):
            if metadata[key] != row[key]:
                raise RuntimeError(
                    f"metadata mismatch for {key}: {metadata_path}"
                )
        if metadata["status"] != "PASS":
            raise RuntimeError(f"non-PASS metadata: {metadata_path}")
        info = sf.info(native_path)
        if info.channels != contract_channels[row["baseline_id"]]:
            raise RuntimeError(
                f"native channel count changed at {native_path}: "
                f"{info.channels} != {contract_channels[row['baseline_id']]}"
            )
        audio, _ = sf.read(native_path, dtype="float32", always_2d=True)
        quality, _ = sf.read(quality_path, dtype="float32", always_2d=True)
        if not np.isfinite(audio).all() or not np.isfinite(quality).all():
            raise RuntimeError(f"non-finite waveform: {native_path}")
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        if rms <= 1.0e-7:
            raise RuntimeError(f"near-silent waveform: {native_path} rms={rms}")
        peaks.append(peak)
        rms_values.append(rms)
        native_bytes += native_path.stat().st_size
        quality_bytes += quality_path.stat().st_size
        if index % 50 == 0 or index == len(rows):
            print(
                json.dumps(
                    {"event": "baseline_audit", "completed": index, "total": len(rows)}
                ),
                flush=True,
            )

    summary = {
        "schema": "sceneplan_foa.p10_matched_baseline_generation_summary",
        "schema_version": 2,
        "status": "PASS",
        "outputs": len(rows),
        "by_baseline": dict(
            sorted(Counter(row["baseline_id"] for row in rows).items())
        ),
        "by_domain": dict(sorted(Counter(row["domain"] for row in rows).items())),
        "unique_baseline_panel_pairs": len(set(pairs)),
        "all_native_finite": True,
        "all_quality_finite": True,
        "minimum_native_rms": min(rms_values),
        "maximum_native_peak": max(peaks),
        "native_bytes": native_bytes,
        "quality_bytes": quality_bytes,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
    }
    output_path = root / "GENERATION_SUMMARY.json"
    _atomic_json(output_path, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
