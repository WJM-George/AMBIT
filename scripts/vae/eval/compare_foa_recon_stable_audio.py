#!/usr/bin/env python3
"""Compare an exported native-FOA VAE reconstruction with Stable Audio Open 1.0.

The baseline splits FOA [W,Y,Z,X] into two stereo pairs, reconstructs each pair
with the same Stable Audio Open stereo VAE, and joins the pairs back into FOA.
The input export must have been produced by export_vae_heldout_recon_foa.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

import soundfile as sf
import torch

import sys
from pathlib import Path as _PathForRepo
_SCRIPTS_DIR = _PathForRepo(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _repo import repo_root
_SAT_ROOT = repo_root()
_EVAL_DIR = _SAT_ROOT / "dataset" / "evaluation"
for path in (str(_SAT_ROOT), str(_EVAL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from compare_vae_4ch_vs_2stereo import (  # noqa: E402
    PAIRINGS,
    baseline_2stereo,
    load_2ch_vae,
)
import compare_vae_heldout as heldout_eval  # noqa: E402
from eval_vae_recon import _agg, evaluate_pair  # noqa: E402


DEFAULT_VAE2_CONFIG = (
    _SAT_ROOT
    / "stable_audio_tools/configs/model_configs/autoencoders/"
    "stable_audio_open_1_0_oobleck_2ch.json"
)
DEFAULT_VAE2_CKPT = Path("/mnt/sdc/ckpts/stable-audio-open-1.0/model.safetensors")
DEFAULT_CACHE_DIR = Path("/mnt/sdc/eval_metric/foa_recon/result_compare/foa_baseline")
BASELINE_TAG = "stable_audio_open_1_0_2x_stereo"
BASELINE_LABEL = "2x Stable Audio Open 1.0 stereo VAE"
SLS_PARQUET_CANDIDATES = (
    Path("/mnt/sdb/audio_dataset/datasets/spatial_librispeech/metadata/metadata.parquet"),
    Path("/mnt/sdd/audio_dataset/datasets/spatial_librispeech/metadata/metadata.parquet"),
)

METRIC_KEYS = [
    "si_sdr_db",
    "w_si_sdr_db",
    "lsd_db",
    "stft_mag_l1",
    "doa_az_err_deg",
    "doa_el_err_deg",
    "dir_energy_ratio_err",
    "ic_corr_err",
    "gt_doa_az_err_deg",
    "gt_doa_el_err_deg",
    "ild_err_db",
    "itd_err_us",
    "pesq_w",
]
HIGHER_BETTER = {"si_sdr_db", "w_si_sdr_db", "pesq_w"}
METRIC_LABELS = {
    "si_sdr_db": "SI-SDR (dB)",
    "w_si_sdr_db": "W SI-SDR (dB)",
    "lsd_db": "LSD (dB)",
    "stft_mag_l1": "STFT magnitude L1",
    "doa_az_err_deg": "DoA azimuth error (deg)",
    "doa_el_err_deg": "DoA elevation error (deg)",
    "dir_energy_ratio_err": "Directional energy ratio error",
    "ic_corr_err": "Inter-channel correlation error",
    "gt_doa_az_err_deg": "GT DoA azimuth error (deg)",
    "gt_doa_el_err_deg": "GT DoA elevation error (deg)",
    "ild_err_db": "ILD error (dB)",
    "itd_err_us": "ITD error (us)",
    "pesq_w": "PESQ on W",
}


def _read_foa(path: Path) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[1] != 4:
        raise ValueError(f"Expected 4 channels in {path}, found {audio.shape[1]}")
    return torch.from_numpy(audio.T.copy()), int(sample_rate)


def _save_foa(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        audio.detach().cpu().numpy().T,
        sample_rate,
        format="FLAC",
        subtype="PCM_24",
    )


def _valid_cached_foa(path: Path, sample_rate: int, frames: int) -> bool:
    if not path.exists():
        return False
    try:
        info = sf.info(str(path))
    except RuntimeError:
        return False
    return info.channels == 4 and info.samplerate == sample_rate and info.frames == frames


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _load_existing_metrics(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = dict(row)
            for key in METRIC_KEYS:
                value = row.get(key, "")
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = float("nan")
            rows[row["file"]] = parsed
    return rows


def _first_existing(paths: tuple[Path, ...]) -> Path:
    return next((path for path in paths if path.exists()), paths[0])


def _available_agg(rows: list[dict], key: str) -> dict:
    block = _agg(rows, key)
    return block if block.get("n", 0) else {}


def _winner(ours: float | None, baseline: float | None, key: str) -> str:
    if ours is None or baseline is None:
        return "n/a"
    if math.isclose(ours, baseline, rel_tol=1e-9, abs_tol=1e-12):
        return "tie"
    if key in HIGHER_BETTER:
        return "new_dataset_vae" if ours > baseline else "stable_audio"
    return "new_dataset_vae" if ours < baseline else "stable_audio"


def _format_metric(key: str, value: float) -> str:
    return f"{value:.6f}" if key == "stft_mag_l1" else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--vae2-base-config", type=Path, default=DEFAULT_VAE2_CONFIG)
    parser.add_argument("--vae2-ckpt", type=Path, default=DEFAULT_VAE2_CKPT)
    parser.add_argument("--baseline-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--pairing", choices=list(PAIRINGS), default="wy_zx")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    device = torch.device(args.device)
    eval_dir = args.eval_dir.resolve()
    manifest_path = eval_dir / "manifest.json"
    metrics_path = eval_dir / "recon_per_file.csv"
    if not manifest_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            f"Expected manifest.json and recon_per_file.csv under {eval_dir}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ours_by_file = _load_existing_metrics(metrics_path)
    if not manifest:
        raise RuntimeError("The reconstruction manifest is empty")
    reconstruction_report_path = eval_dir / "recon_summary.json"
    reconstruction_report = (
        json.loads(reconstruction_report_path.read_text(encoding="utf-8"))
        if reconstruction_report_path.exists()
        else {}
    )

    baseline_dir = eval_dir / f"{BASELINE_TAG}_foa"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    pairing = PAIRINGS[args.pairing]

    pending: list[dict] = []
    for item in manifest:
        source_path = eval_dir / item["source_foa"]
        info = sf.info(str(source_path))
        output_path = baseline_dir / (
            f"{int(item['idx']):02d}_{item['file']}_{BASELINE_TAG}_recon_foa_4ch.flac"
        )
        if args.force or not _valid_cached_foa(output_path, info.samplerate, info.frames):
            pending.append({**item, "output_path": output_path, "source_info": info})

    reused = 0
    if not args.force and args.baseline_cache_dir:
        still_pending: list[dict] = []
        for item in pending:
            cached = args.baseline_cache_dir / f"{item['file']}_baseline_4ch.flac"
            info = item["source_info"]
            if _valid_cached_foa(cached, info.samplerate, info.frames):
                _link_or_copy(cached, item["output_path"])
                reused += 1
                print(f"[cache {reused}] {item['file']} <- {cached}")
            else:
                still_pending.append(item)
        pending = still_pending

    if pending:
        print(
            f"[load] {BASELINE_LABEL}: {args.vae2_ckpt} "
            f"(pairing={args.pairing}, device={device}, threads={torch.get_num_threads()})"
        )
        model2, model_sr = load_2ch_vae(
            str(args.vae2_base_config), str(args.vae2_ckpt), device
        )
        for number, item in enumerate(pending, start=1):
            source, sample_rate = _read_foa(eval_dir / item["source_foa"])
            if sample_rate != model_sr:
                raise ValueError(
                    f"Sample-rate mismatch for {item['file']}: {sample_rate} vs {model_sr}"
                )
            torch.manual_seed(args.seed + int(item["idx"]))
            reconstruction = baseline_2stereo(model2, source, pairing, device)
            reconstruction = reconstruction[:, : source.shape[-1]].clamp(-1, 1)
            _save_foa(item["output_path"], reconstruction, sample_rate)
            print(f"[baseline {number}/{len(pending)}] {item['file']}")
        del model2

    sample_ids = {int(item["file"]) for item in manifest}
    heldout_eval.SLS_PARQUET = str(_first_existing(SLS_PARQUET_CANDIDATES))
    try:
        gt_map = heldout_eval.load_gt_doa_map(sample_ids)
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        print(f"[metrics] GT DoA unavailable ({exc}); GT metrics will be omitted")
        gt_map = {}

    ours_rows: list[dict] = []
    baseline_rows: list[dict] = []
    comparison_manifest: list[dict] = []
    for number, item in enumerate(manifest, start=1):
        stem = item["file"]
        source, sample_rate = _read_foa(eval_dir / item["source_foa"])
        baseline_path = baseline_dir / (
            f"{int(item['idx']):02d}_{stem}_{BASELINE_TAG}_recon_foa_4ch.flac"
        )
        baseline, baseline_sr = _read_foa(baseline_path)
        if baseline_sr != sample_rate:
            raise ValueError(f"Sample-rate mismatch in {baseline_path}")
        n = min(source.shape[-1], baseline.shape[-1])
        source_np = source[:, :n].numpy()
        baseline_np = baseline[:, :n].numpy()
        baseline_metrics = evaluate_pair(baseline_np, source_np)
        baseline_metrics.update(
            heldout_eval.extra_metrics(
                baseline_np,
                source_np,
                sample_rate,
                gt_map.get(int(stem)),
            )
        )
        baseline_row = {
            "idx": int(item["idx"]),
            "file": stem,
            "source": item["source_path"],
            **baseline_metrics,
        }
        baseline_rows.append(baseline_row)

        ours = ours_by_file.get(stem)
        if ours is None:
            raise KeyError(f"No native-FOA metrics found for {stem}")
        ours_rows.append(ours)
        comparison_manifest.append(
            {
                "idx": int(item["idx"]),
                "file": stem,
                "sample_rate": sample_rate,
                "channels": 4,
                "layout": "[W, Y, Z, X]",
                "source_foa": item["source_foa"],
                "new_dataset_vae_foa": item["recon_foa"],
                "stable_audio_2x_stereo_foa": str(baseline_path.relative_to(eval_dir)),
            }
        )
        print(f"[metrics {number}/{len(manifest)}] {stem}")

    ours_tag = manifest[0].get("tag", "new_dataset_vae")
    aggregate = {
        ours_tag: {key: _available_agg(ours_rows, key) for key in METRIC_KEYS},
        BASELINE_TAG: {key: _available_agg(baseline_rows, key) for key in METRIC_KEYS},
    }
    winners: dict[str, str] = {}
    for key in METRIC_KEYS:
        ours_median = aggregate[ours_tag][key].get("median")
        baseline_median = aggregate[BASELINE_TAG][key].get("median")
        winners[key] = _winner(ours_median, baseline_median, key)

    cache_backed = 0
    if args.baseline_cache_dir:
        for item in manifest:
            cached = args.baseline_cache_dir / f"{item['file']}_baseline_4ch.flac"
            output = baseline_dir / (
                f"{int(item['idx']):02d}_{item['file']}_{BASELINE_TAG}_recon_foa_4ch.flac"
            )
            try:
                cache_backed += int(os.path.samefile(cached, output))
            except (FileNotFoundError, OSError):
                pass

    report = {
        "n_clips": len(manifest),
        "test_set": "held-out Spatial-LibriSpeech (NOT in VAE training)",
        "foa_layout": "[W, Y, Z, X]",
        "pairing": args.pairing,
        "pair_indices": pairing,
        "new_dataset_vae": {
            "tag": ours_tag,
            "checkpoint": reconstruction_report.get("vae4_ckpt"),
            "config": reconstruction_report.get("vae4_config"),
            "note": manifest[0].get("note"),
        },
        "stable_audio_baseline": {
            "tag": BASELINE_TAG,
            "label": BASELINE_LABEL,
            "checkpoint": str(args.vae2_ckpt),
            "method": "split FOA into [W,Y] and [Z,X], reconstruct both, then reassemble",
        },
        "cache_backed_baseline_clips": cache_backed,
        "generated_baseline_clips_this_run": len(pending),
        "aggregate": aggregate,
        "winner_by_median": winners,
    }
    (eval_dir / "comparison_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (eval_dir / "comparison_manifest.json").write_text(
        json.dumps(comparison_manifest, indent=2), encoding="utf-8"
    )

    with (eval_dir / "comparison_per_file.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        columns = ["method", "idx", "file"] + METRIC_KEYS + ["foa_path", "source"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for method, rows in ((ours_tag, ours_rows), (BASELINE_TAG, baseline_rows)):
            for row in rows:
                stem = row["file"]
                manifest_row = next(m for m in comparison_manifest if m["file"] == stem)
                path_key = (
                    "new_dataset_vae_foa"
                    if method == ours_tag
                    else "stable_audio_2x_stereo_foa"
                )
                writer.writerow(
                    {
                        "method": method,
                        **{key: row.get(key, "") for key in ["idx", "file"] + METRIC_KEYS},
                        "foa_path": manifest_row[path_key],
                        "source": row.get("source", ""),
                    }
                )

    ours_note = reconstruction_report.get("note") or manifest[0].get(
        "note", "VAE trained on the new full dataset"
    )
    lines = [
        "# FOA VAE reconstruction comparison",
        "",
        f"- Evaluation: {len(manifest)} identical held-out Spatial-LibriSpeech clips",
        f"- New VAE: `{ours_tag}` ({ours_note})",
        f"- Baseline: `{BASELINE_TAG}` using `{args.vae2_ckpt}`",
        "- Baseline path: `[W,Y]` and `[Z,X]` -> the same stereo VAE twice -> `[W,Y,Z,X]`",
        "- Both methods have equal latent budget: `64*T/1024`",
        "- All listening files below are native 4-channel FOA; no stereo preview was generated",
        "",
        "## Aggregate metrics",
        "",
        "| metric | direction | new-dataset VAE median | 2x Stable Audio median | winner |",
        "|---|---|---:|---:|---|",
    ]
    for key in METRIC_KEYS:
        ours_block = aggregate[ours_tag][key]
        baseline_block = aggregate[BASELINE_TAG][key]
        if not ours_block or not baseline_block:
            continue
        direction = "higher" if key in HIGHER_BETTER else "lower"
        winner = winners[key].replace("new_dataset_vae", "new VAE").replace(
            "stable_audio", "Stable Audio"
        )
        lines.append(
            f"| {METRIC_LABELS[key]} | {direction} | "
            f"{_format_metric(key, ours_block['median'])} | "
            f"{_format_metric(key, baseline_block['median'])} | **{winner}** |"
        )

    ours_wins = sum(value == "new_dataset_vae" for value in winners.values())
    baseline_wins = sum(value == "stable_audio" for value in winners.values())
    lines.extend(
        [
            "",
            f"Median wins: **new-dataset VAE {ours_wins}**, "
            f"**2x Stable Audio {baseline_wins}** (PESQ unavailable).",
            "",
            "## FOA files by clip",
            "",
            "| # | clip | source FOA | new-dataset VAE | 2x Stable Audio Open 1.0 |",
            "|---:|---|---|---|---|",
        ]
    )
    for item in comparison_manifest:
        lines.append(
            f"| {item['idx']} | `{item['file']}` | "
            f"[source]({item['source_foa']}) | "
            f"[new VAE]({item['new_dataset_vae_foa']}) | "
            f"[Stable Audio]({item['stable_audio_2x_stereo_foa']}) |"
        )
    lines.extend(
        [
            "",
            "Detailed per-file metrics: `comparison_per_file.csv`",
            "Machine-readable summary: `comparison_summary.json`",
            "FOA mapping: `comparison_manifest.json`",
        ]
    )
    (eval_dir / "comparison_table.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"\nWrote comparison -> {eval_dir}")


if __name__ == "__main__":
    main()
