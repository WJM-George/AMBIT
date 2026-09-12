#!/usr/bin/env python3
"""Export held-out FOA reconstructions for a 4ch VAE checkpoint.

This is a single-model companion to dataset/evaluation/compare_vae_heldout.py.
It uses the same leakage-free held-out Spatial-LibriSpeech sampling logic, then
saves source and reconstructed FOA [W,Y,Z,X] 4-channel FLAC files plus per-file
and aggregate reconstruction metrics.
"""

from __future__ import annotations
import os

import argparse
import csv
import json
import random
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
for p in (str(_SAT_ROOT), str(_EVAL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from compare_vae_4ch_vs_2stereo import _enc_dec_pad, load_4ch_vae  # noqa: E402
import compare_vae_heldout as heldout_eval  # noqa: E402
from eval_vae_recon import _agg, _read_4ch, _resample, evaluate_pair  # noqa: E402


BASE_KEYS = heldout_eval.BASE_KEYS
EXTRA_KEYS = heldout_eval.EXTRA_KEYS
HIGHER_BETTER = heldout_eval.HIGHER_BETTER
METRIC_KEYS = BASE_KEYS + EXTRA_KEYS


def _first_existing(candidates: list[str]) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return candidates[0]


def _save_foa(path: Path, audio_ct: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio_ct.detach().cpu().numpy().T, sample_rate, format="FLAC", subtype="PCM_24")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae4-config", required=True)
    parser.add_argument("--vae4-ckpt", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tag", default="new_dataset_vae_ds1024_z64_step250k")
    parser.add_argument("--note", default="VAE trained on the new full dataset")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sls-dir", default=None)
    parser.add_argument("--sls-parquet", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    source_dir = args.out / "source_foa"
    recon_dir = args.out / "recon_foa"

    heldout_eval.SLS_DIR = args.sls_dir or _first_existing(
        [
            os.environ.get("AMBIT_DATA_ROOT", "data") + "/datasets/spatial_librispeech/ambisonics",
            os.environ.get("AMBIT_DATA_ROOT", "data") + "/datasets/spatial_librispeech/ambisonics",
        ]
    )
    heldout_eval.SLS_PARQUET = args.sls_parquet or _first_existing(
        [
            os.environ.get("AMBIT_DATA_ROOT", "data") + "/datasets/spatial_librispeech/metadata/metadata.parquet",
            os.environ.get("AMBIT_DATA_ROOT", "data") + "/datasets/spatial_librispeech/metadata/metadata.parquet",
        ]
    )

    print("[setup] building held-out SLS list ...")
    held = heldout_eval.held_out_sls_files()
    rng = random.Random(args.seed)
    rng.shuffle(held)
    picks = held[: args.num * 2]

    sample_ids = {int(Path(f).stem) for f in picks}
    print("[setup] loading GT DoA from parquet ...")
    try:
        gt_map = heldout_eval.load_gt_doa_map(sample_ids)
    except ModuleNotFoundError as exc:
        if exc.name != "pyarrow":
            raise
        print("[setup] pyarrow is not installed; continuing without GT-DoA metrics")
        gt_map = {}

    print(f"[load] 4ch VAE: {args.vae4_ckpt}")
    model4, sr4 = load_4ch_vae(args.vae4_config, args.vae4_ckpt, device)

    rows = []
    manifest = []
    used = 0
    for fpath in picks:
        if used >= args.num:
            break
        stem = Path(fpath).stem
        try:
            raw, src_sr = _read_4ch(fpath)
            if raw.shape[0] < 4:
                continue
            src4 = torch.from_numpy(_resample(raw[:4], src_sr, sr4)).float()
            recon4 = _enc_dec_pad(model4, src4, device).clamp(-1, 1)
            n = min(src4.shape[-1], recon4.shape[-1])
            src4 = src4[:, :n]
            recon4 = recon4[:, :n]

            metrics = evaluate_pair(recon4.numpy(), src4.numpy())
            metrics.update(heldout_eval.extra_metrics(recon4.numpy(), src4.numpy(), sr4, gt_map.get(int(stem))))
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {Path(fpath).name}: {exc!r}")
            continue

        source_name = f"{used:02d}_{stem}_source_foa_4ch.flac"
        recon_name = f"{used:02d}_{stem}_{args.tag}_recon_foa_4ch.flac"
        _save_foa(source_dir / source_name, src4, sr4)
        _save_foa(recon_dir / recon_name, recon4, sr4)

        row = {"idx": used, "file": stem, "source": fpath, **metrics}
        rows.append(row)
        manifest.append(
            {
                "idx": used,
                "file": stem,
                "source_path": fpath,
                "source_foa": str((source_dir / source_name).relative_to(args.out)),
                "recon_foa": str((recon_dir / recon_name).relative_to(args.out)),
                "layout": "[W, Y, Z, X]",
                "sample_rate": sr4,
                "channels": 4,
                "tag": args.tag,
                "note": args.note,
            }
        )
        used += 1
        print(
            f"[{used}/{args.num}] {stem} "
            f"LSD={metrics['lsd_db']:.2f} W_SI-SDR={metrics['w_si_sdr_db']:.2f} "
            f"DoAaz={metrics.get('doa_az_err_deg', float('nan')):.1f} PESQ={metrics.get('pesq_w', float('nan')):.2f}"
        )

    if not rows:
        raise RuntimeError("No clips exported.")

    aggregate = {k: _agg(rows, k) for k in METRIC_KEYS}
    report = {
        "tag": args.tag,
        "note": args.note,
        "n_clips": len(rows),
        "test_set": "held-out Spatial-LibriSpeech (NOT in VAE training)",
        "held_out_pool": len(held),
        "vae4_ckpt": args.vae4_ckpt,
        "vae4_config": args.vae4_config,
        "sample_rate": sr4,
        "outputs": {
            "source_foa": str(source_dir),
            "recon_foa": str(recon_dir),
        },
        "aggregate": {args.tag: aggregate},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.out / "recon_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    with (args.out / "recon_per_file.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["idx", "file", "source"] + METRIC_KEYS
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in cols})

    lines = [
        f"# FOA reconstruction export ({args.tag})",
        "",
        f"- note: {args.note}",
        f"- clips: {len(rows)} held-out Spatial-LibriSpeech samples",
        f"- checkpoint: `{args.vae4_ckpt}`",
        f"- FOA layout: `[W, Y, Z, X]`; stereo/listen files intentionally not generated",
        "",
        "| metric | median | mean | direction |",
        "|---|---:|---:|---|",
    ]
    for key in METRIC_KEYS:
        block = aggregate.get(key, {})
        median = block.get("median")
        mean = block.get("mean")
        if median is None or mean is None:
            continue
        direction = "higher better" if key in HIGHER_BETTER else "lower better"
        lines.append(f"| {key} | {median:.3f} | {mean:.3f} | {direction} |")
    lines.extend(
        [
            "",
            "Files:",
            f"- source FOA: `{source_dir}`",
            f"- reconstructed FOA: `{recon_dir}`",
        ]
    )
    (args.out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote FOA recon export -> {args.out}")


if __name__ == "__main__":
    main()
