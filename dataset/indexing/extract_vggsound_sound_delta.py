#!/usr/bin/env python3
"""Stream selected VGGSound tar members into a QC'd mono Sound delta.

Only selected MP4 members are materialized in a per-tar temporary directory.
They are converted in parallel, then the temporary video directory is removed.
The immutable source archives and the frozen P0--P9 dataset are never changed.
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
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


SCHEMA = "stable_audio_tools.vggsound_sound_delta_extraction"
SCHEMA_VERSION = 1
MODEL_SAMPLE_RATE = 44_100
MAX_MODEL_SAMPLES = 442_368
REQUIRED_TAIL_SAMPLES = 40
MIN_RMS = 1.0e-5
MIN_PEAK = 1.0e-4
MIN_ACTIVE_100MS_FRACTION = 0.01


def canonical_json(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audio_geometry_ok(path: Path, sample_rate: int) -> bool:
    try:
        info = sf.info(str(path))
    except Exception:
        return False
    return (
        int(info.channels) == 1
        and int(info.samplerate) == int(sample_rate)
        and int(info.frames) > 0
    )


def convert_one(task: tuple[str, str, int]) -> dict[str, Any]:
    source_text, target_text, sample_rate = task
    source, target = Path(source_text), Path(target_text)
    temporary = target.with_name(f".{target.stem}.{os.getpid()}.tmp.wav")
    temporary.unlink(missing_ok=True)
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(temporary),
    ]
    started = time.monotonic()
    result = subprocess.run(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    ok = result.returncode == 0 and audio_geometry_ok(temporary, sample_rate)
    if ok:
        os.replace(temporary, target)
    else:
        temporary.unlink(missing_ok=True)
    return {
        "stem": source.stem,
        "status": "PASS" if ok else "FAIL",
        "returncode": int(result.returncode),
        "error": None if ok else result.stderr[-1000:],
        "elapsed_sec": time.monotonic() - started,
    }


def qc_one(task: tuple[dict[str, Any], str]) -> dict[str, Any]:
    source_row, path_text = task
    path = Path(path_text)
    output = dict(source_row)
    reasons: list[str] = []
    try:
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        frames, channels = audio.shape
        model_frames = math.ceil(frames * MODEL_SAMPLE_RATE / int(rate))
        finite = bool(np.isfinite(audio).all())
        mono = audio[:, 0] if channels == 1 else np.zeros(1, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
        peak = float(np.max(np.abs(mono)))
        dc_abs = abs(float(np.mean(mono, dtype=np.float64)))
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
        if channels != 1:
            reasons.append("not_mono")
        if int(rate) != 48_000:
            reasons.append("wrong_sample_rate")
        if not finite:
            reasons.append("non_finite")
        if rms < MIN_RMS:
            reasons.append("signal_rms_too_low")
        if peak < MIN_PEAK:
            reasons.append("signal_peak_too_low")
        if active < MIN_ACTIVE_100MS_FRACTION:
            reasons.append("signal_active_fraction_too_low")
        if model_frames + REQUIRED_TAIL_SAMPLES > MAX_MODEL_SAMPLES:
            reasons.append("complete_source_plus_pyroom_delay_over_limit")
        output.update(
            {
                "audio_path": str(path),
                "source_audio_sha256": digest,
                "native_sample_rate_hz": int(rate),
                "native_num_samples": int(frames),
                "native_channels": int(channels),
                "model_num_samples": int(model_frames),
                "duration_sec": float(frames / rate),
                "file_num_bytes": int(len(blob)),
                "decoded_finite": finite,
                "signal_rms": rms,
                "signal_peak": peak,
                "dc_abs": dc_abs,
                "active_100ms_fraction": active,
            }
        )
    except Exception as error:  # noqa: BLE001
        reasons.append(f"decode_or_checksum_error:{type(error).__name__}")
    output["qc_status"] = "PASS" if not reasons else "FAIL"
    output["qc_exclusion_reasons"] = sorted(set(reasons))
    return output


def load_hashes(path: Path) -> set[str]:
    result: set[str] = set()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        columns=["source_audio_sha256", "eligible"], batch_size=65_536
    ):
        for digest, eligible in zip(
            batch.column(0).to_pylist(), batch.column(1).to_pylist()
        ):
            if eligible and digest:
                result.add(str(digest))
    return result


def load_eval_hashes(root: Path) -> set[str]:
    result: set[str] = set()
    for name in ("candidate_pool.jsonl", "candidate_exclusions.jsonl"):
        with (root / name).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                digest = json.loads(line).get("audio_sha256")
                if digest:
                    result.add(str(digest))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path(
            os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/supplements/"
            "vggsound_sound_delta_v1/source_selection/"
            "pilot_extraction_selection.jsonl"
        ),
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/audio_dataset/datasets/vggsound/snapshot"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/supplements/"
            "vggsound_sound_delta_v1/extraction_pilot_20k"
        ),
    )
    parser.add_argument(
        "--base-signal-catalog",
        type=Path,
        default=Path(
            os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/source_catalog/"
            "nonspeech/nonspeech_signal_catalog.parquet"
        ),
    )
    parser.add_argument(
        "--external-manifest-root",
        type=Path,
        default=Path(
            os.environ.get("AMBIT_DATA_ROOT", "data") + "/evaluation_benchmark/"
            "p10_evaluation_benchmark_v1/manifests"
        ),
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=20_000,
        help=(
            "Freeze exactly this many passing rows. Use 0 to retain every "
            "locally available, QC-passing row (the production expansion mode)."
        ),
    )
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--qc-jobs", type=int, default=min(48, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selection_path = args.selection.expanduser().resolve(strict=True)
    snapshot = args.snapshot.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    base_catalog = args.base_signal_catalog.expanduser().resolve(strict=True)
    external_root = args.external_manifest_root.expanduser().resolve(strict=True)
    if args.target_rows < 0 or args.jobs <= 0 or args.qc_jobs <= 0:
        raise ValueError("target rows must be non-negative and worker counts positive")
    try:
        output_root.relative_to(Path(os.environ.get("AMBIT_DATA_ROOT", "data")))
    except ValueError as error:
        raise ValueError("delta extraction must live on ${AMBIT_DATA_ROOT}") from error

    selection = read_jsonl(selection_path)
    if len({row["stem"] for row in selection}) != len(selection):
        raise RuntimeError("selection contains duplicate stems")
    selection.sort(key=lambda row: (row["selection_rank"], row["stem"]))
    wanted = {str(row["stem"]): row for row in selection}
    selection_sha = sha256_file(selection_path)
    audio_root = output_root / "audio"
    receipt_root = output_root / "tar_receipts"
    temp_root = output_root / "temporary"
    for path in (audio_root, receipt_root, temp_root):
        path.mkdir(parents=True, exist_ok=True)

    completed = {
        stem
        for stem in wanted
        if audio_geometry_ok(audio_root / f"{stem}.wav", args.sample_rate)
    }
    tarballs = sorted(snapshot.glob("vggsound_*.tar.gz"))
    if not tarballs:
        raise FileNotFoundError(f"no VGGSound tarballs under {snapshot}")
    started = time.monotonic()
    for tar_index, tar_path in enumerate(tarballs, start=1):
        pending = set(wanted) - completed
        if not pending:
            break
        tar_stat = tar_path.stat()
        receipt_path = receipt_root / f"{tar_path.name}.json"
        if receipt_path.is_file():
            prior = json.loads(receipt_path.read_text(encoding="utf-8"))
            prior_stems = list(prior.get("converted_stems") or ())
            receipt_matches = (
                prior.get("selection_sha256") == selection_sha
                and int(prior.get("tar_num_bytes", -1)) == int(tar_stat.st_size)
                and int(prior.get("tar_mtime_ns", -1)) == int(tar_stat.st_mtime_ns)
            )
            converted_outputs_exist = all(
                audio_geometry_ok(audio_root / f"{stem}.wav", args.sample_rate)
                for stem in prior_stems
            )
            legacy_empty_receipt = (
                int(prior.get("converted", -1)) == 0
                and int(prior.get("selected_members_found", -1)) == 0
            )
            if (
                receipt_matches
                and prior.get("status") == "PASS"
                and converted_outputs_exist
                and (prior_stems or legacy_empty_receipt)
            ):
                completed.update(prior_stems)
                print(
                    json.dumps(
                        {
                            "event": "tar_resume_skip",
                            "tar": tar_path.name,
                            "converted": len(prior_stems),
                            "completed_total": len(completed),
                        }
                    ),
                    flush=True,
                )
                continue
        with tempfile.TemporaryDirectory(
            prefix=f"{tar_path.stem}.", dir=temp_root
        ) as temporary_text:
            temporary = Path(temporary_text)
            extracted: list[Path] = []
            archive_error = None
            try:
                with tarfile.open(tar_path, mode="r|gz") as archive:
                    for member in archive:
                        if not member.isfile() or not member.name.endswith(".mp4"):
                            continue
                        basename = Path(member.name).name
                        stem = Path(basename).stem
                        if stem not in pending:
                            continue
                        source = archive.extractfile(member)
                        if source is None:
                            continue
                        target = temporary / basename
                        with target.open("wb") as handle:
                            shutil.copyfileobj(source, handle, length=8 << 20)
                        extracted.append(target)
            except (tarfile.TarError, OSError, EOFError) as error:
                archive_error = f"{type(error).__name__}: {error}"

            # A corrupt compressed stream cannot prove that previously emitted
            # members are complete.  Discard that tar's temporary members and
            # keep the archive failure explicit rather than accepting a prefix.
            tasks = [] if archive_error else [
                (str(path), str(audio_root / f"{path.stem}.wav"), args.sample_rate)
                for path in extracted
            ]
            conversions: list[dict[str, Any]] = []
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.jobs
            ) as pool:
                for result in pool.map(convert_one, tasks, chunksize=1):
                    conversions.append(result)
                    if result["status"] == "PASS":
                        completed.add(str(result["stem"]))
            receipt = {
                "schema": "stable_audio_tools.vggsound_sound_delta_tar_receipt",
                "schema_version": 1,
                "status": "PASS" if archive_error is None else "FAIL",
                "selection_sha256": selection_sha,
                "tar_path": str(tar_path),
                "tar_num_bytes": int(tar_stat.st_size),
                "tar_mtime_ns": int(tar_stat.st_mtime_ns),
                "selected_members_found": len(extracted),
                "converted": sum(row["status"] == "PASS" for row in conversions),
                "failed": sum(row["status"] != "PASS" for row in conversions),
                "archive_error": archive_error,
                "converted_stems": sorted(
                    row["stem"] for row in conversions if row["status"] == "PASS"
                ),
                "failures": [
                    row for row in conversions if row["status"] != "PASS"
                ],
                "completed_total": len(completed),
            }
            atomic_json(receipt_path, receipt)
        print(
            json.dumps(
                {
                    "event": "tar_complete",
                    "tar": tar_path.name,
                    "tar_index": tar_index,
                    "tar_total": len(tarballs),
                    "selected_members_found": receipt["selected_members_found"],
                    "converted": receipt["converted"],
                    "failed": receipt["failed"],
                    "completed_total": len(completed),
                    "elapsed_sec": round(time.monotonic() - started, 1),
                }
            ),
            flush=True,
        )

    available = [
        row for row in selection if (audio_root / f"{row['stem']}.wav").is_file()
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.qc_jobs) as pool:
        audited = list(
            pool.map(
                qc_one,
                [
                    (row, str(audio_root / f"{row['stem']}.wav"))
                    for row in available
                ],
                chunksize=8,
            )
        )
    base_hashes = load_hashes(base_catalog)
    eval_hashes = load_eval_hashes(external_root)
    seen_new_hashes: set[str] = set()
    passed: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row in sorted(audited, key=lambda item: (item["selection_rank"], item["stem"])):
        reasons = list(row["qc_exclusion_reasons"])
        digest = row.get("source_audio_sha256")
        if digest in base_hashes:
            reasons.append("exact_file_sha256_in_base_catalog")
        if digest in eval_hashes:
            reasons.append("exact_file_sha256_in_external_benchmark")
        if digest in seen_new_hashes:
            reasons.append("duplicate_exact_file_sha256_in_delta")
        if not reasons and digest:
            seen_new_hashes.add(str(digest))
            row["exact_hash_gate_status"] = "pass"
            row["qc_status"] = "PASS"
            passed.append(row)
        else:
            row["exact_hash_gate_status"] = "fail"
            row["qc_status"] = "FAIL"
            row["qc_exclusion_reasons"] = sorted(set(reasons))
            exclusions.append(row)

    frozen = passed if args.target_rows == 0 else passed[: args.target_rows]
    status = (
        "PASS"
        if (args.target_rows == 0 and bool(frozen))
        or len(frozen) == args.target_rows
        else "FAIL"
    )
    atomic_jsonl(output_root / "qc_passed.jsonl", passed)
    atomic_jsonl(output_root / "qc_exclusions.jsonl", exclusions)
    atomic_jsonl(output_root / "pilot_frozen.jsonl", frozen)
    summary = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "inputs": {
            "selection": str(selection_path),
            "selection_sha256": selection_sha,
            "snapshot": str(snapshot),
            "base_signal_catalog": str(base_catalog),
            "external_manifest_root": str(external_root),
        },
        "counts": {
            "selected": len(selection),
            "local_tarballs": len(tarballs),
            "extracted_available": len(available),
            "qc_passed_unique": len(passed),
            "qc_excluded": len(exclusions),
            "pilot_frozen": len(frozen),
            "requested_target_rows": int(args.target_rows),
            "unavailable_in_local_tarballs": len(selection) - len(available),
        },
        "contracts": {
            "lossless_pcm16_mono_48khz": True,
            "full_decode_signal_qc": True,
            "complete_source_plus_pyroom_delay_fits": True,
            "base_exact_hash_disjoint": True,
            "external_benchmark_exact_hash_disjoint": True,
            "delta_exact_hash_unique": True,
        },
        "elapsed_sec": time.monotonic() - started,
        "artifacts": {
            "audio_root": str(audio_root),
            "qc_passed": str(output_root / "qc_passed.jsonl"),
            "qc_exclusions": str(output_root / "qc_exclusions.jsonl"),
            "pilot_frozen": str(output_root / "pilot_frozen.jsonl"),
        },
    }
    atomic_json(output_root / "EXTRACTION_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
