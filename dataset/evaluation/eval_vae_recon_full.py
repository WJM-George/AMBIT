#!/usr/bin/env python3
"""Full VAE reconstruction eval on a decode_latents_4ch manifest (all metrics).

Same metric set as compare_vae_heldout.py (base + GT-DoA + ILD/ITD + PESQ + optional SELD),
but reads reconstructions from manifest.json instead of live encode->decode.

Run:
  uv run python dataset/evaluation/eval_vae_recon_full.py \
      --recon-dir /mnt/sdc/eval_metric/ckpt_sweep/step_400000/recon \
      --out /mnt/sdc/eval_metric/ckpt_sweep/step_400000 \
      --with-seld
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SAT_ROOT = _HERE.parents[1]
for p in (str(_HERE), str(_SAT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_vae_recon import _read_4ch, _resample, evaluate_pair, _agg  # noqa: E402
from compare_vae_heldout import (  # noqa: E402
    BASE_KEYS, EXTRA_KEYS, SELD_KEYS, HIGHER_BETTER,
    load_gt_doa_map, extra_metrics,
)
from eval_vae_recon import _ang_err  # noqa: E402

SLS_PARQUET = "/mnt/sdb/audio_dataset/datasets/spatial_librispeech/metadata/metadata.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon-dir", type=Path, required=True, help="folder with manifest.json + *_4ch.flac")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="ours_preencoded")
    ap.add_argument("--with-seld", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    manifest_path = args.recon_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())

    sample_ids = {int(Path(e["source"]).stem) for e in manifest if e.get("source")}
    gt_map = load_gt_doa_map(sample_ids)

    device = torch.device(args.device)
    seld = None
    metric_keys = BASE_KEYS + EXTRA_KEYS
    if args.with_seld:
        try:
            from seld_pseldnets import PSELDNetsScorer  # noqa: E402
            seld = PSELDNetsScorer(device=device)
            metric_keys = metric_keys + SELD_KEYS
        except Exception as e:  # noqa: BLE001
            print(f"[seld] disabled ({e!r})")

    rows = []
    for entry in manifest:
        src = entry.get("source", "")
        quad_name = entry.get("output_quad") or entry.get("output")
        recon_path = args.recon_dir / quad_name
        if not src or not recon_path.exists():
            continue
        src4, sr = _read_4ch(src)
        recon4, sr_r = _read_4ch(str(recon_path))
        if sr_r != sr:
            recon4 = _resample(recon4, sr_r, sr)
        m = evaluate_pair(recon4, src4)
        sid = int(Path(src).stem)
        m.update(extra_metrics(recon4, src4, sr, gt_map.get(sid)))
        if seld is not None:
            gt = gt_map.get(sid)
            sr_pred = seld.doa(recon4, sr)
            ss_pred = seld.doa(src4, sr)
            m["seld_doa_err_deg"] = _ang_err(sr_pred[0], ss_pred[0])
            if gt is not None:
                m["seld_gt_doa_err_deg"] = _ang_err(sr_pred[0], gt[0])
                m["src_seld_gt_doa_err_deg"] = _ang_err(ss_pred[0], gt[0])
        m["file"] = src
        m["idx"] = entry.get("idx", len(rows))
        rows.append(m)

    agg = {k: _agg(rows, k) for k in metric_keys}
    report = {
        "n_clips": len(rows),
        "test_set": "held-out SLS preencode->decode",
        "tag": args.tag,
        "recon_dir": str(args.recon_dir),
        "with_seld": seld is not None,
        "aggregate": {args.tag: agg},
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "preencode_summary.json").write_text(json.dumps(report, indent=2))

    with (args.out / "preencode_per_file.csv").open("w", newline="") as f:
        cols = ["idx", "file"] + metric_keys
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for m in rows:
            w.writerow({k: m.get(k, "") for k in cols})

    lines = [
        f"# Pre-encode VAE recon eval ({args.tag})",
        f"clips: {len(rows)}  |  recon: `{args.recon_dir}`",
        "",
        "| metric | median | mean |",
        "|---|---|---|",
    ]
    for k in metric_keys:
        block = agg.get(k, {})
        med = block.get("median")
        mean = block.get("mean")
        if med is None:
            continue
        d = "up" if k in HIGHER_BETTER else "down"
        lines.append(f"| {k} ({d}) | {med:.3f} | {mean:.3f} |")
    (args.out / "preencode_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote -> {args.out}/preencode_summary.{{md,json}}")


if __name__ == "__main__":
    main()
