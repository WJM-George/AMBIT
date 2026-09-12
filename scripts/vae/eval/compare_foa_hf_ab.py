#!/usr/bin/env python3
"""Compare high-frequency reconstruction artifacts for two FOA VAE exports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


BANDS = {
    "low_0_8k": (0.0, 8_000.0),
    "high_8_22k": (8_000.0, 22_050.0),
    "high_8_12k": (8_000.0, 12_000.0),
    "high_12_16k": (12_000.0, 16_000.0),
    "high_16_22k": (16_000.0, 22_050.0),
}
COMMON_METRICS = (
    "w_si_sdr_db",
    "lsd_db",
    "doa_az_err_deg",
    "doa_el_err_deg",
    "dir_energy_ratio_err",
    "ic_corr_err",
    "ild_err_db",
    "itd_err_us",
)


def read_audio(path: Path) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf.read(
        str(path), dtype="float32", always_2d=True
    )
    if audio.shape[1] != 4:
        raise ValueError(f"Expected 4-channel FOA in {path}, got {audio.shape[1]}")
    return torch.from_numpy(audio.T.copy()), int(sample_rate)


def stft_power(audio: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, torch.Tensor]:
    n_fft = 2048
    window = torch.hann_window(n_fft)
    spectrum = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=512,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    frequencies = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return spectrum.abs().square(), frequencies


def band_power(power: torch.Tensor, frequencies: torch.Tensor, lo: float, hi: float) -> float:
    mask = (frequencies >= lo) & (frequencies < hi)
    return float(power[:, mask, :].mean().item())


def db_ratio(numerator: float, denominator: float) -> float:
    return 10.0 * math.log10((numerator + 1e-20) / (denominator + 1e-20))


def load_metrics(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = {}
            for key in COMMON_METRICS:
                try:
                    parsed[key] = float(row[key])
                except (KeyError, TypeError, ValueError):
                    parsed[key] = float("nan")
            result[row["file"]] = parsed
    return result


def aggregate(values: list[float]) -> dict[str, float | int]:
    finite = np.asarray([value for value in values if math.isfinite(value)])
    return {
        "n": int(finite.size),
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
    }


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a-dir", default="loss_only")
    parser.add_argument("--b-dir", default="antialias")
    parser.add_argument("--a-label", default="loss_only")
    parser.add_argument("--b-label", default="antialias")
    args = parser.parse_args()

    root = args.root.resolve()
    variants = {
        args.a_label: root / args.a_dir,
        args.b_label: root / args.b_dir,
    }
    manifests = {
        label: json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for label, directory in variants.items()
    }
    metrics = {
        label: load_metrics(directory / "recon_per_file.csv")
        for label, directory in variants.items()
    }
    by_file = {
        label: {item["file"]: item for item in manifest}
        for label, manifest in manifests.items()
    }
    file_ids = [item["file"] for item in manifests[args.a_label]]
    if set(file_ids) != set(by_file[args.b_label]):
        raise ValueError("A/B manifests do not contain the same sample IDs")

    rows: list[dict[str, object]] = []
    listen_dir = root / "listen_foa"
    for index, file_id in enumerate(file_ids):
        a_item = by_file[args.a_label][file_id]
        source_path = variants[args.a_label] / a_item["source_foa"]
        source, sample_rate = read_audio(source_path)
        source_power, frequencies = stft_power(source, sample_rate)
        source_bands = {
            key: band_power(source_power, frequencies, *limits)
            for key, limits in BANDS.items()
        }

        row: dict[str, object] = {"idx": index, "file": file_id}
        link_or_copy(
            source_path,
            listen_dir / f"{index:02d}_{file_id}_A_source_foa_4ch.flac",
        )
        for letter, (label, directory) in zip("BC", variants.items()):
            item = by_file[label][file_id]
            recon_path = directory / item["recon_foa"]
            recon, recon_rate = read_audio(recon_path)
            if recon_rate != sample_rate:
                raise ValueError(f"Sample-rate mismatch for {file_id}: {label}")
            recon_power, recon_frequencies = stft_power(recon, recon_rate)
            recon_bands = {
                key: band_power(recon_power, recon_frequencies, *limits)
                for key, limits in BANDS.items()
            }
            for band_name in BANDS:
                row[f"{label}_{band_name}_excess_db"] = db_ratio(
                    recon_bands[band_name], source_bands[band_name]
                )
            row[f"{label}_hf_share_delta_db"] = (
                float(row[f"{label}_high_8_22k_excess_db"])
                - float(row[f"{label}_low_0_8k_excess_db"])
            )
            for metric_name, value in metrics[label][file_id].items():
                row[f"{label}_{metric_name}"] = value
            link_or_copy(
                recon_path,
                listen_dir / f"{index:02d}_{file_id}_{letter}_{label}_recon_foa_4ch.flac",
            )
        rows.append(row)

    numeric_keys = [
        key for key in rows[0] if key not in {"idx", "file"}
    ]
    summary = {
        "n_clips": len(rows),
        "layout": "[W, Y, Z, X]",
        "bands_hz": BANDS,
        "metric_definition": (
            "band excess = 10*log10(mean reconstruction STFT power / "
            "mean source STFT power), all four FOA channels"
        ),
        "variants": {
            label: {key: aggregate([float(row[key]) for row in rows]) for key in numeric_keys if key.startswith(f"{label}_")}
            for label in variants
        },
    }
    (root / "hf_comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    with (root / "hf_comparison_per_file.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# FOA VAE high-frequency A/B ({root.name})",
        "",
        f"- clips: {len(rows)} held-out Spatial-LibriSpeech FOA files",
        "- layout: `[W, Y, Z, X]`",
        "- band excess: `10*log10(reconstruction power / source power)`",
        f"- listening files: `{listen_dir}`",
        "",
        "| metric (median) | loss-only | anti-alias |",
        "|---|---:|---:|",
    ]
    report_keys = (
        "high_8_22k_excess_db",
        "high_8_12k_excess_db",
        "high_12_16k_excess_db",
        "high_16_22k_excess_db",
        "hf_share_delta_db",
        "w_si_sdr_db",
        "lsd_db",
        "doa_az_err_deg",
        "doa_el_err_deg",
        "ic_corr_err",
    )
    for key in report_keys:
        a_value = summary["variants"][args.a_label][f"{args.a_label}_{key}"]["median"]
        b_value = summary["variants"][args.b_label][f"{args.b_label}_{key}"]["median"]
        lines.append(f"| {key} | {a_value:.3f} | {b_value:.3f} |")
    lines.extend(["", "## Screenshot sample 199154", ""])
    sample = next(row for row in rows if row["file"] == "199154")
    for key in report_keys[:5]:
        lines.append(
            f"- `{key}`: loss-only {float(sample[f'{args.a_label}_{key}']):.3f}; "
            f"anti-alias {float(sample[f'{args.b_label}_{key}']):.3f}"
        )
    (root / "HF_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
