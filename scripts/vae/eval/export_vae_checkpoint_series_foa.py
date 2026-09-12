#!/usr/bin/env python3
"""Export and compare several VAE checkpoints on one fixed FOA manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


import sys
from pathlib import Path as _PathForRepo
_SCRIPTS_DIR = _PathForRepo(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _repo import repo_root
REPO = repo_root()
EVAL_DIR = REPO / "dataset/evaluation"
for path in (str(REPO), str(EVAL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from compare_vae_4ch_vs_2stereo import _enc_dec_pad, load_4ch_vae  # noqa: E402
from eval_vae_recon import evaluate_pair  # noqa: E402


SAMPLE_RATE = 44_100
SAMPLE_SIZE = 176_400
JOINT_PEAK = 0.9
PREPROCESS_SEED = 20260715
METRICS = (
    "si_sdr_db",
    "w_si_sdr_db",
    "lsd_db",
    "stft_mag_l1",
    "doa_az_err_deg",
    "doa_el_err_deg",
    "dir_energy_ratio_err",
    "ic_corr_err",
    "hf_power_delta_db",
    "hf_share_delta_db",
)


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Checkpoint must be LABEL=/path/to/model.ckpt")
    label, path = value.split("=", 1)
    if not label or not Path(path).is_file():
        raise argparse.ArgumentTypeError(f"Invalid checkpoint specification: {value}")
    return label, Path(path)


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def item_seed(item_id: str) -> int:
    digest = hashlib.sha256(f"{PREPROCESS_SEED}:{item_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def preprocess(row: dict) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(row["path"])
    if audio.ndim != 2 or audio.shape[0] < 4:
        raise ValueError(f"Expected at least four channels in {row['path']}")
    audio = audio[:4].float()
    if sample_rate != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sample_rate, SAMPLE_RATE)
    if audio.shape[-1] < SAMPLE_SIZE:
        audio = torch.nn.functional.pad(audio, (0, SAMPLE_SIZE - audio.shape[-1]))
    elif audio.shape[-1] > SAMPLE_SIZE:
        generator = np.random.default_rng(item_seed(row["id"]))
        start = int(generator.integers(0, audio.shape[-1] - SAMPLE_SIZE + 1))
        audio = audio[:, start : start + SAMPLE_SIZE]
    peak = audio.abs().amax()
    if peak > 0:
        audio = audio * (JOINT_PEAK / peak)
    return audio.clamp(-1, 1)


def save_foa(path: Path, audio: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        audio.detach().cpu().numpy().T,
        SAMPLE_RATE,
        format="FLAC",
        subtype="PCM_24",
    )


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def spectral_power(audio: torch.Tensor) -> tuple[float, float]:
    n_fft = 2048
    spectrum = torch.stft(
        audio.float(),
        n_fft=n_fft,
        hop_length=512,
        win_length=n_fft,
        window=torch.hann_window(n_fft),
        center=False,
        return_complex=True,
    ).abs().square()
    frequencies = torch.fft.rfftfreq(n_fft, d=1.0 / SAMPLE_RATE)
    low = float(spectrum[:, frequencies < 8_000].mean().item())
    high = float(spectrum[:, frequencies >= 8_000].mean().item())
    return low, high


def db_ratio(numerator: float, denominator: float) -> float:
    return 10.0 * math.log10((numerator + 1e-20) / (denominator + 1e-20))


def aggregate(rows: list[dict], metric: str) -> dict:
    values = [
        float(row[metric])
        for row in rows
        if metric in row and math.isfinite(float(row[metric]))
    ]
    if not values:
        return {}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    device = torch.device(args.device)
    items = load_manifest(args.manifest)
    if not items:
        raise RuntimeError("Evaluation manifest is empty")
    labels = [label for label, _ in args.checkpoint]
    if len(labels) != len(set(labels)):
        raise ValueError("Checkpoint labels must be unique")

    args.out.mkdir(parents=True, exist_ok=True)
    source_dir = args.out / "source_foa"
    recon_root = args.out / "recon_foa"
    listen_dir = args.out / "listen_foa"
    sources: dict[str, torch.Tensor] = {}
    source_power: dict[str, tuple[float, float]] = {}
    output_manifest = []

    for index, item in enumerate(items):
        source = preprocess(item)
        sources[item["id"]] = source
        source_power[item["id"]] = spectral_power(source)
        safe_id = item["sample_id"].replace(":", "_")
        filename = f"{index:02d}_{item['source_group']}_{safe_id}_A_source_foa_4ch.flac"
        source_path = source_dir / filename
        save_foa(source_path, source)
        link_or_copy(source_path, listen_dir / filename)
        output_manifest.append(
            {
                "idx": index,
                **item,
                "source_foa": str(source_path.relative_to(args.out)),
                "recon_foa": {},
                "layout": "[W, Y, Z, X]",
                "sample_rate": SAMPLE_RATE,
                "duration_seconds": SAMPLE_SIZE / SAMPLE_RATE,
            }
        )

    rows = []
    for variant_index, (label, checkpoint) in enumerate(args.checkpoint):
        print(f"[load] {label}: {checkpoint}", flush=True)
        model, model_rate = load_4ch_vae(args.model_config, str(checkpoint), device)
        if model_rate != SAMPLE_RATE:
            raise ValueError(f"Unexpected model sample rate: {model_rate}")
        for item_index, item in enumerate(items):
            source = sources[item["id"]]
            seed = item_seed(f"latent:{item['id']}")
            torch.manual_seed(seed)
            with torch.inference_mode():
                reconstruction = _enc_dec_pad(model, source, device).clamp(-1, 1)
            common = evaluate_pair(reconstruction.numpy(), source.numpy())
            source_low, source_high = source_power[item["id"]]
            recon_low, recon_high = spectral_power(reconstruction)
            common["hf_power_delta_db"] = db_ratio(recon_high, source_high)
            common["hf_share_delta_db"] = db_ratio(
                recon_high / (recon_low + 1e-20),
                source_high / (source_low + 1e-20),
            )
            row = {
                "variant": label,
                "idx": item_index,
                "id": item["id"],
                "sample_id": item["sample_id"],
                "source_group": item["source_group"],
                **common,
            }
            rows.append(row)

            safe_id = item["sample_id"].replace(":", "_")
            filename = (
                f"{item_index:02d}_{item['source_group']}_{safe_id}_{label}_recon_foa_4ch.flac"
            )
            recon_path = recon_root / label / filename
            save_foa(recon_path, reconstruction)
            output_manifest[item_index]["recon_foa"][label] = str(
                recon_path.relative_to(args.out)
            )
            letter = chr(ord("B") + variant_index)
            link_or_copy(
                recon_path,
                listen_dir
                / f"{item_index:02d}_{item['source_group']}_{safe_id}_{letter}_{label}_recon_foa_4ch.flac",
            )
            print(f"[{label} {item_index + 1}/{len(items)}] {item['id']}", flush=True)
        del model

    groups = ["ALL", "sls", "music", "sound"]
    summary = {}
    for label in labels:
        summary[label] = {}
        variant_rows = [row for row in rows if row["variant"] == label]
        for group in groups:
            group_rows = (
                variant_rows
                if group == "ALL"
                else [row for row in variant_rows if row["source_group"] == group]
            )
            summary[label][group] = {
                metric: aggregate(group_rows, metric) for metric in METRICS
            }

    with (args.out / "per_file_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["variant", "idx", "id", "sample_id", "source_group", *METRICS]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in columns} for row in rows)

    report = {
        "manifest": str(args.manifest),
        "model_config": args.model_config,
        "checkpoints": {label: str(path) for label, path in args.checkpoint},
        "preprocessing": {
            "sample_rate": SAMPLE_RATE,
            "sample_size": SAMPLE_SIZE,
            "duration_seconds": SAMPLE_SIZE / SAMPLE_RATE,
            "joint_peak": JOINT_PEAK,
            "seed": PREPROCESS_SEED,
        },
        "items": len(items),
        "summary": summary,
    }
    (args.out / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# New-dataset VAE checkpoint comparison",
        "",
        "- Fixed set: 5 held-out SLS + 5 held-out music + 5 held-out sound",
        "- Audio: 4 seconds, 44.1 kHz, joint-peak 0.9, native 4-channel FOA `[W,Y,Z,X]`",
        f"- Listening directory: `{listen_dir}`",
        "",
    ]
    for group in groups:
        lines.extend(
            [
                f"## {group}",
                "",
                "| metric (median) | " + " | ".join(labels) + " |",
                "|---|" + "---:|" * len(labels),
            ]
        )
        for metric in METRICS:
            values = [summary[label][group][metric].get("median") for label in labels]
            if any(value is None for value in values):
                continue
            lines.append(
                f"| {metric} | " + " | ".join(f"{value:.4f}" for value in values) + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## FOA files",
            "",
            "| # | group | clip | source | " + " | ".join(labels) + " |",
            "|---:|---|---|---|" + "---|" * len(labels),
        ]
    )
    for item in output_manifest:
        variant_links = " | ".join(
            f"[{label}]({item['recon_foa'][label]})" for label in labels
        )
        lines.append(
            f"| {item['idx']} | {item['source_group']} | `{item['sample_id']}` | "
            f"[source]({item['source_foa']}) | {variant_links} |"
        )
    (args.out / "comparison_table.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
