#!/usr/bin/env python3
"""Measure five-band energy distributions for the 4-channel VAE dataset.

The preprocessing mirrors the Stage-1 training path:
  * load native 4-channel FOA audio
  * resample the complete clip to 44.1 kHz
  * select one deterministic uniform random 4-second crop per file
  * jointly peak-normalize all four channels to 0.9
  * compute a 2048-point STFT with hop length 512

No decoded audio or spectrogram cache is written. The only persistent outputs are
the aggregate JSON, CSV, and Markdown summaries requested by the user.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


SAMPLE_RATE = 44_100
SAMPLE_SIZE = 176_400
PEAK = 0.9
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
DEFAULT_SEED = 20260710
ACTIVE_POWER_THRESHOLD = 1e-12

CONFIG_DIR = Path(
    "." + "/stable-audio-tools/"
    "stable_audio_tools/configs/dataset_configs/vae_v2_dataset"
)


@dataclass(frozen=True)
class Band:
    key: str
    label: str
    fmin: float
    fmax: float


@dataclass(frozen=True)
class Source:
    key: str
    label: str
    path: Path
    count: int
    filelist: Path | None = None


BANDS = (
    Band("0_250", "0-250 Hz", 0.0, 250.0),
    Band("250_2000", "250 Hz-2 kHz", 250.0, 2_000.0),
    Band("2000_8000", "2-8 kHz", 2_000.0, 8_000.0),
    Band("8000_14000", "8-14 kHz", 8_000.0, 14_000.0),
    Band("14000_22050", "14-22.05 kHz", 14_000.0, 22_050.0),
)

CHANNEL_GROUPS = {
    "W": (0,),
    "XYZ": (1, 2, 3),
    "all": (0, 1, 2, 3),
}

SOURCES = (
    Source(
        "existing_non_speech",
        "Existing non-speech FOA",
        Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/spatial_foa/audio"),
        139_745,
    ),
    Source(
        "expansion_non_speech",
        "Expansion non-speech FOA",
        Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/spatial_foa_v2/audio/train"),
        460_255,
    ),
    Source(
        "spatial_librispeech",
        "Spatial LibriSpeech",
        Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/datasets/spatial_librispeech/ambisonics"),
        218_957,
        Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/spatial_foa/caption_jsonl/sls_train_218957.filelist.txt"),
    ),
    Source(
        "tts_sdb",
        "TTS SDB QC-clean subset",
        Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/datasets/spatial_speech_foa_tts_v1_part_sdb/audio"),
        125_000,
        Path(
            os.environ.get("AMBIT_DATA_ROOT", "data") + "/datasets/spatial_speech_foa_tts_v1_part_sdb/"
            "manifests/vae_4ch_v2_tts_sdb_qc_clean_125000.filelist.txt"
        ),
    ),
    Source(
        "tts_sdc",
        "TTS SDC QC-clean subset",
        Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/speech_dataset/spatial_speech_foa_tts_v1_part_sdc/audio"),
        75_000,
        Path(
            os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/speech_dataset/spatial_speech_foa_tts_v1_part_sdc/"
            "manifests/vae_4ch_v2_tts_sdc_qc_clean_75000.filelist.txt"
        ),
    ),
)

AGGREGATES = {
    "mixture_all": {
        "label": "Training mixture",
        "sources": {source.key: source.count for source in SOURCES},
    },
    "non_speech": {
        "label": "All non-speech",
        "sources": {
            "existing_non_speech": 139_745,
            "expansion_non_speech": 460_255,
        },
    },
    "speech_all": {
        "label": "All speech",
        "sources": {
            "spatial_librispeech": 218_957,
            "tts_sdb": 125_000,
            "tts_sdc": 75_000,
        },
    },
    "tts_combined": {
        "label": "Combined TTS",
        "sources": {
            "tts_sdb": 125_000,
            "tts_sdc": 75_000,
        },
    },
}

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".opus"}
_RESAMPLERS: dict[tuple[int, int], torchaudio.transforms.Resample] = {}


def list_audio_files(source: Source) -> list[str]:
    if source.filelist is not None:
        rows = [
            line.strip()
            for line in source.filelist.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return sorted(
            str(Path(row) if Path(row).is_absolute() else source.path / row)
            for row in rows
        )

    files = []
    for base, _, names in os.walk(source.path):
        for name in names:
            if name.startswith(".") or Path(name).suffix.lower() not in _AUDIO_EXTENSIONS:
                continue
            files.append(str(Path(base) / name))
    return sorted(files)


def deterministic_sample(files: list[str], count: int, seed: int, key: str) -> list[str]:
    if count > len(files):
        raise ValueError(f"{key}: requested {count} files from a pool of {len(files)}")
    picks = random.Random(f"{seed}:{key}:file-selection").sample(files, count)
    return sorted(picks)


def path_seed(path: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{path}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def selection_sha256(paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def resample_audio(audio: torch.Tensor, source_rate: int) -> torch.Tensor:
    if source_rate == SAMPLE_RATE:
        return audio
    key = (int(source_rate), SAMPLE_RATE)
    resampler = _RESAMPLERS.get(key)
    if resampler is None:
        resampler = torchaudio.transforms.Resample(source_rate, SAMPLE_RATE)
        _RESAMPLERS[key] = resampler
    return resampler(audio)


def preprocess_audio(path: str, seed: int) -> torch.Tensor:
    audio, source_rate = torchaudio.load(path)
    if audio.ndim != 2 or audio.shape[0] < 4:
        raise ValueError(f"expected at least 4 channels, got {tuple(audio.shape)}")
    audio = audio[:4].float()
    audio = resample_audio(audio, int(source_rate))

    n_samples = audio.shape[-1]
    if n_samples > SAMPLE_SIZE:
        offset = random.Random(path_seed(path, seed)).randint(0, n_samples - SAMPLE_SIZE)
        audio = audio[:, offset : offset + SAMPLE_SIZE]
    elif n_samples < SAMPLE_SIZE:
        padded = audio.new_zeros((4, SAMPLE_SIZE))
        padded[:, :n_samples] = audio
        audio = padded

    peak = audio.abs().amax()
    if peak > 0:
        audio = audio * (PEAK / peak)
    return audio.clamp_(-1.0, 1.0)


class AudioCropDataset(Dataset):
    def __init__(self, paths: list[str], seed: int):
        self.paths = paths
        self.seed = seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        try:
            return {"audio": preprocess_audio(path, self.seed), "path": path, "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"audio": None, "path": path, "error": repr(exc)}


def collate_audio(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["audio"] is not None]
    return {
        "audio": torch.stack([row["audio"] for row in valid]) if valid else None,
        "paths": [row["path"] for row in valid],
        "errors": [(row["path"], row["error"]) for row in rows if row["error"] is not None],
    }


class BandEnergyComputer:
    def __init__(self, device: torch.device):
        self.device = device
        self.window = torch.hann_window(WIN_LENGTH, device=device)
        frequencies = torch.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE).to(device)

        one_sided_weights = torch.full_like(frequencies, 2.0)
        one_sided_weights[0] = 1.0
        one_sided_weights[-1] = 1.0

        masks = []
        for index, band in enumerate(BANDS):
            if index == len(BANDS) - 1:
                mask = (frequencies >= band.fmin) & (frequencies <= band.fmax)
            else:
                mask = (frequencies >= band.fmin) & (frequencies < band.fmax)
            masks.append(mask.to(torch.float32) * one_sided_weights)
        self.band_weights = torch.stack(masks)
        self.parseval_denom = float(N_FFT) * float(self.window.square().sum().item())

    @torch.inference_mode()
    def __call__(self, audio: torch.Tensor) -> dict[str, np.ndarray]:
        audio = audio.to(self.device, non_blocking=True)
        batch_size = audio.shape[0]
        flat_audio = audio.reshape(batch_size * 4, SAMPLE_SIZE)
        stft = torch.stft(
            flat_audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self.window,
            center=False,
            return_complex=True,
        )
        power = stft.abs().square().reshape(batch_size, 4, stft.shape[-2], stft.shape[-1])

        # [B, C, K, frames] -> frame-mean band mean-square power [B, C, K].
        band_power_channels = torch.einsum(
            "bcft,kf->bckt", power, self.band_weights
        ).mean(dim=-1) / self.parseval_denom

        group_power = []
        for channel_indices in CHANNEL_GROUPS.values():
            group_power.append(band_power_channels[:, channel_indices].mean(dim=1))
        linear_power = torch.stack(group_power, dim=1)

        total_power = linear_power.sum(dim=-1, keepdim=True)
        active_mask = total_power.squeeze(-1) > ACTIVE_POWER_THRESHOLD
        power_db = 10.0 * torch.log10(linear_power.clamp_min(1e-12))
        share_percent = torch.where(
            active_mask.unsqueeze(-1),
            100.0 * linear_power / total_power.clamp_min(ACTIVE_POWER_THRESHOLD),
            torch.full_like(linear_power, torch.nan),
        )

        bandwidths = torch.tensor(
            [band.fmax - band.fmin for band in BANDS],
            device=self.device,
            dtype=linear_power.dtype,
        )
        density_db = 10.0 * torch.log10(
            (linear_power / bandwidths.view(1, 1, -1)).clamp_min(1e-16)
        )

        result = {
            "linear_power": linear_power.cpu().numpy(),
            "power_db": power_db.cpu().numpy(),
            "share_percent": share_percent.cpu().numpy(),
            "density_db": density_db.cpu().numpy(),
            "active_mask": active_mask.cpu().numpy(),
        }
        del audio, flat_audio, stft, power, band_power_channels, linear_power
        return result


def summarize_unweighted(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
    }


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    centers = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    centers /= sorted_weights.sum()
    return float(np.interp(quantile, centers, sorted_values))


def summarize_weighted(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0:
        return {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    weights = weights / weights.sum()
    return {
        "mean": float(np.sum(values * weights)),
        "p10": weighted_quantile(values, weights, 0.10),
        "p50": weighted_quantile(values, weights, 0.50),
        "p90": weighted_quantile(values, weights, 0.90),
    }


def format_stats(
    arrays: dict[str, np.ndarray],
    weights: np.ndarray | None = None,
) -> dict:
    summary: dict[str, dict] = {}
    for group_index, group_name in enumerate(CHANNEL_GROUPS):
        group_summary = {}
        for band_index, band in enumerate(BANDS):
            metrics = {}
            for metric_name in ("power_db", "share_percent", "density_db"):
                values = arrays[metric_name][:, group_index, band_index]
                metrics[metric_name] = (
                    summarize_weighted(values, weights)
                    if weights is not None
                    else summarize_unweighted(values)
                )

            linear_values = arrays["linear_power"][:, group_index, band_index]
            metrics["linear_power_mean"] = (
                float(np.average(linear_values, weights=weights))
                if weights is not None
                else float(np.mean(linear_values))
            )
            group_summary[band.key] = {
                "label": band.label,
                "fmin_hz": band.fmin,
                "fmax_hz": band.fmax,
                **metrics,
            }
        summary[group_name] = group_summary
    return summary


def format_activity(
    active_mask: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, dict[str, float | int]]:
    activity = {}
    for group_index, group_name in enumerate(CHANNEL_GROUPS):
        values = active_mask[:, group_index].astype(np.float64)
        if weights is None:
            active_fraction = float(values.mean())
        else:
            active_fraction = float(np.average(values, weights=weights))
        activity[group_name] = {
            "active_fraction": active_fraction,
            "inactive_fraction": 1.0 - active_fraction,
            "active_count": int(values.sum()),
            "sample_count": int(values.shape[0]),
        }
    return activity


def combine_sources(
    source_arrays: dict[str, dict[str, np.ndarray]],
    source_counts: dict[str, int],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {
        "linear_power": [],
        "power_db": [],
        "share_percent": [],
        "density_db": [],
        "active_mask": [],
    }
    weights = []
    total_count = float(sum(source_counts.values()))
    for source_key, count in source_counts.items():
        sample_count = source_arrays[source_key]["power_db"].shape[0]
        source_weight = count / total_count
        for metric_name in arrays:
            arrays[metric_name].append(source_arrays[source_key][metric_name])
        weights.append(np.full(sample_count, source_weight / sample_count, dtype=np.float64))
    return (
        {metric_name: np.concatenate(parts, axis=0) for metric_name, parts in arrays.items()},
        np.concatenate(weights, axis=0),
    )


def write_csv(path: Path, report: dict) -> None:
    fields = [
        "scope",
        "scope_type",
        "sample_count",
        "channel_group",
        "band",
        "fmin_hz",
        "fmax_hz",
        "power_db_mean",
        "power_db_p10",
        "power_db_p50",
        "power_db_p90",
        "energy_share_mean_percent",
        "energy_share_p10_percent",
        "energy_share_p50_percent",
        "energy_share_p90_percent",
        "density_db_per_hz_mean",
        "density_db_per_hz_p10",
        "density_db_per_hz_p50",
        "density_db_per_hz_p90",
        "linear_power_mean",
        "active_fraction",
        "inactive_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for scope_type in ("sources", "aggregates"):
            for scope, payload in report[scope_type].items():
                for channel_group, bands in payload["stats"].items():
                    for band_key, stats in bands.items():
                        writer.writerow(
                            {
                                "scope": scope,
                                "scope_type": scope_type[:-1],
                                "sample_count": payload["sample_count"],
                                "channel_group": channel_group,
                                "band": band_key,
                                "fmin_hz": stats["fmin_hz"],
                                "fmax_hz": stats["fmax_hz"],
                                "power_db_mean": stats["power_db"]["mean"],
                                "power_db_p10": stats["power_db"]["p10"],
                                "power_db_p50": stats["power_db"]["p50"],
                                "power_db_p90": stats["power_db"]["p90"],
                                "energy_share_mean_percent": stats["share_percent"]["mean"],
                                "energy_share_p10_percent": stats["share_percent"]["p10"],
                                "energy_share_p50_percent": stats["share_percent"]["p50"],
                                "energy_share_p90_percent": stats["share_percent"]["p90"],
                                "density_db_per_hz_mean": stats["density_db"]["mean"],
                                "density_db_per_hz_p10": stats["density_db"]["p10"],
                                "density_db_per_hz_p50": stats["density_db"]["p50"],
                                "density_db_per_hz_p90": stats["density_db"]["p90"],
                                "linear_power_mean": stats["linear_power_mean"],
                                "active_fraction": payload["activity"][channel_group]["active_fraction"],
                                "inactive_fraction": payload["activity"][channel_group]["inactive_fraction"],
                            }
                        )


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# VAE 4ch Five-Band Energy Statistics",
        "",
        f"- Samples: {report['processed_samples']:,}",
        f"- Seed: `{report['seed']}`",
        "- Preprocessing: full-clip resample to 44.1 kHz, deterministic 4 s crop, joint peak 0.9",
        "- STFT: n_fft=2048, hop=512, Hann, center=False",
        "- No persistent decoded-audio or STFT cache was created.",
        "",
    ]

    display_scopes = [
        ("mixture_all", report["aggregates"]["mixture_all"]),
        ("non_speech", report["aggregates"]["non_speech"]),
        ("speech_all", report["aggregates"]["speech_all"]),
        ("tts_combined", report["aggregates"]["tts_combined"]),
    ]
    for scope_key, payload in display_scopes:
        lines.extend(
            [
                f"## {payload['label']} (`{scope_key}`)",
                "",
                "All-channel mean; energy share is conditioned on active crops.",
                (
                    "Active crop fraction (W / XYZ / all): "
                    f"{payload['activity']['W']['active_fraction'] * 100:.3f}% / "
                    f"{payload['activity']['XYZ']['active_fraction'] * 100:.3f}% / "
                    f"{payload['activity']['all']['active_fraction'] * 100:.3f}%"
                ),
                "",
                "| Band | Power mean dB | Power p50 dB | Mean share | Share p10-p90 | Density mean dB/Hz |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for band in BANDS:
            stats = payload["stats"]["all"][band.key]
            lines.append(
                "| {label} | {pmean:.3f} | {p50:.3f} | {smean:.3f}% | "
                "{s10:.3f}-{s90:.3f}% | {dmean:.3f} |".format(
                    label=band.label,
                    pmean=stats["power_db"]["mean"],
                    p50=stats["power_db"]["p50"],
                    smean=stats["share_percent"]["mean"],
                    s10=stats["share_percent"]["p10"],
                    s90=stats["share_percent"]["p90"],
                    dmean=stats["density_db"]["mean"],
                )
            )
        lines.append("")

    lines.extend(["## Per-Source Median Energy Share (All Channels)", ""])
    lines.append("| Source | Active all | 0-250 | 250-2k | 2-8k | 8-14k | 14-22.05k |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for source in SOURCES:
        payload = report["sources"][source.key]
        shares = [
            payload["stats"]["all"][band.key]["share_percent"]["p50"] for band in BANDS
        ]
        lines.append(
            f"| {source.label} | "
            f"{payload['activity']['all']['active_fraction'] * 100:.3f}% | "
            + " | ".join(f"{value:.3f}%" for value in shares)
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-source", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=CONFIG_DIR / "vae_4ch_v2_band_energy_stats_50k",
    )
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    computer = BandEnergyComputer(device)
    source_arrays: dict[str, dict[str, np.ndarray]] = {}
    source_reports = {}
    total_processed = 0
    total_failed = 0
    start_time = time.time()

    print(
        f"[setup] device={device} samples_per_source={args.samples_per_source} "
        f"batch_size={args.batch_size} workers={args.num_workers}",
        flush=True,
    )

    for source in SOURCES:
        files = list_audio_files(source)
        if len(files) != source.count:
            raise ValueError(
                f"{source.key}: config count={source.count}, discovered files={len(files)}"
            )
        selected = deterministic_sample(
            files, args.samples_per_source, args.seed, source.key
        )
        print(
            f"[source] {source.key}: selected={len(selected):,}/{len(files):,} "
            f"sha256={selection_sha256(selected)}",
            flush=True,
        )

        dataset = AudioCropDataset(selected, args.seed)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            pin_memory=device.type == "cuda",
            prefetch_factor=2 if args.num_workers > 0 else None,
            collate_fn=collate_audio,
        )

        chunks: dict[str, list[np.ndarray]] = {
            "linear_power": [],
            "power_db": [],
            "share_percent": [],
            "density_db": [],
            "active_mask": [],
        }
        failures = []
        processed = 0
        source_start = time.time()
        for batch in loader:
            failures.extend(batch["errors"])
            if batch["audio"] is None:
                continue
            metrics = computer(batch["audio"])
            for metric_name, values in metrics.items():
                chunks[metric_name].append(values)
            processed += batch["audio"].shape[0]
            if processed % args.progress_every < batch["audio"].shape[0]:
                elapsed = time.time() - source_start
                print(
                    f"[progress] {source.key}: {processed:,}/{len(selected):,} "
                    f"({processed / max(elapsed, 1e-6):.1f} clips/s)",
                    flush=True,
                )

        arrays = {
            metric_name: np.concatenate(parts, axis=0)
            for metric_name, parts in chunks.items()
        }
        source_arrays[source.key] = arrays
        source_reports[source.key] = {
            "label": source.label,
            "configured_count": source.count,
            "sample_count": processed,
            "failed_count": len(failures),
            "failure_examples": failures[:10],
            "selection_sha256": selection_sha256(selected),
            "stats": format_stats(arrays),
            "activity": format_activity(arrays["active_mask"]),
        }
        total_processed += processed
        total_failed += len(failures)
        print(
            f"[done] {source.key}: processed={processed:,} failed={len(failures)} "
            f"elapsed={time.time() - source_start:.1f}s",
            flush=True,
        )

    aggregate_reports = {}
    for aggregate_key, aggregate in AGGREGATES.items():
        arrays, weights = combine_sources(source_arrays, aggregate["sources"])
        aggregate_reports[aggregate_key] = {
            "label": aggregate["label"],
            "source_counts": aggregate["sources"],
            "sample_count": int(arrays["power_db"].shape[0]),
            "stats": format_stats(arrays, weights=weights),
            "activity": format_activity(arrays["active_mask"], weights=weights),
        }

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    markdown_path = output_prefix.with_suffix(".md")

    report = {
        "schema_version": 2,
        "created_unix_time": time.time(),
        "seed": args.seed,
        "processed_samples": total_processed,
        "failed_samples": total_failed,
        "elapsed_seconds": time.time() - start_time,
        "preprocessing": {
            "sample_rate": SAMPLE_RATE,
            "sample_size": SAMPLE_SIZE,
            "crop_seconds": SAMPLE_SIZE / SAMPLE_RATE,
            "crop": "deterministic uniform random offset after full-clip resampling",
            "normalize": "joint_peak",
            "peak": PEAK,
            "channel_layout": "WYZX",
            "channel_groups": {key: list(value) for key, value in CHANNEL_GROUPS.items()},
        },
        "stft": {
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "win_length": WIN_LENGTH,
            "window": "hann",
            "center": False,
            "onesided_parseval_weighting": True,
            "active_power_threshold": ACTIVE_POWER_THRESHOLD,
            "share_statistics": "conditioned on channel-group total power above active threshold",
        },
        "cache_policy": {
            "persistent_audio_cache": False,
            "persistent_spectrogram_cache": False,
            "persistent_intermediate_cache": False,
            "process_local_resample_kernel_cache": True,
            "os_page_cache_possible": True,
        },
        "bands": [
            {
                "key": band.key,
                "label": band.label,
                "fmin_hz": band.fmin,
                "fmax_hz": band.fmax,
            }
            for band in BANDS
        ],
        "sources": source_reports,
        "aggregates": aggregate_reports,
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(csv_path, report)
    write_markdown(markdown_path, report)
    print(f"[output] {json_path}", flush=True)
    print(f"[output] {csv_path}", flush=True)
    print(f"[output] {markdown_path}", flush=True)
    print(
        f"[complete] processed={total_processed:,} failed={total_failed} "
        f"elapsed={time.time() - start_time:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
