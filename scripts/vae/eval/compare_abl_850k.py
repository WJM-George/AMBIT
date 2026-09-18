#!/usr/bin/env python3
"""Matched-step ablation compare: base_cont vs phase+SCM.

(phase+INT abandoned after 850k eval — dir_energy / elevation collapsed.)
Runs the same held-out Spatial-LibriSpeech clips through both checkpoints
and reports aggregate metrics side-by-side. Designed to run on CPU while
training keeps the GPUs busy.

Example:
  uv run python scripts/vae/eval/compare_abl_850k.py --step 900 --num 40 --device cpu
"""

from __future__ import annotations
import os

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch

_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _repo import repo_root

_SAT = repo_root()
_EVAL = _SAT / "dataset" / "evaluation"
for p in (str(_SAT), str(_EVAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

import compare_vae_heldout as heldout_eval  # noqa: E402
from compare_vae_4ch_vs_2stereo import _enc_dec_pad, load_4ch_vae  # noqa: E402
from eval_vae_recon import _agg, _read_4ch, _resample, evaluate_pair  # noqa: E402


BASE_KEYS = heldout_eval.BASE_KEYS
EXTRA_KEYS = heldout_eval.EXTRA_KEYS
HIGHER_BETTER = heldout_eval.HIGHER_BETTER
METRIC_KEYS = BASE_KEYS + EXTRA_KEYS

ARMS = {
    "base_cont_850k": {
        "config": os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/vae_ds1024_z64_hf_overshoot_decay_350k_8gpu/configs/model_hf_overshoot_decay_350k.json",
        "ckpt": os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/vae_ds1024_z64_hf_overshoot_decay_350k_8gpu/checkpoints/vae_ds1024_z64_hf_overshoot_decay_350k_8gpu/cr8wgqb6/checkpoints/epoch=13-step=850000.ckpt",
        "note": "base continue: no phase / no spatial loss (hf overshoot only)",
    },
    "phase_scm_850k": {
        "config": os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/vae_abl_phase_scm/configs/model_phase_scm.json",
        "ckpt": os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/vae_abl_phase_scm/checkpoints/vae_abl_phase_scm/l67fhyka/checkpoints/epoch=12-step=850000.ckpt",
        "note": "ablation: frequency-gated IFGD phase + FOA SCM spatial",
    },
}

ARMS_900K = {
    "base_cont_900k": {
        "config": os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/vae_ds1024_z64_hf_overshoot_decay_350k_8gpu/configs/model_hf_overshoot_decay_350k.json",
        "ckpt": os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/vae_ds1024_z64_hf_overshoot_decay_350k_8gpu/checkpoints/vae_abl_base_cont/kmmm2uwb/checkpoints/epoch=13-step=900000.ckpt",
        "note": "base continue: no phase / no spatial loss (hf overshoot only)",
    },
    "phase_scm_900k": {
        "config": os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/vae_abl_phase_scm/configs/model_phase_scm.json",
        "ckpt": os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/vae_abl_phase_scm/checkpoints/vae_abl_phase_scm/hkfa5pts/checkpoints/epoch=13-step=900000.ckpt",
        "note": "frequency-gated IFGD phase + FOA SCM spatial",
    },
}


def _first_existing(candidates: list[str]) -> str:
    for c in candidates:
        if Path(c).exists():
            return c
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, choices=(850, 900), default=850)
    parser.add_argument("--num", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    arms = ARMS if args.step == 850 else ARMS_900K
    if args.out is None:
        args.out = Path(fos.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/vae_abl_compare_{args.step}k")

    for arm, meta in arms.items():
        if not Path(meta["ckpt"]).exists():
            raise FileNotFoundError(f"{arm}: missing ckpt {meta['ckpt']}")
        if not Path(meta["config"]).exists():
            raise FileNotFoundError(f"{arm}: missing config {meta['config']}")

    heldout_eval.SLS_DIR = _first_existing(
        [
            os.environ.get("AMBIT_DATA_ROOT", "data") + "/datasets/spatial_librispeech/ambisonics",
            os.environ.get("AMBIT_DATA_ROOT", "data") + "/datasets/spatial_librispeech/ambisonics",
        ]
    )
    heldout_eval.SLS_PARQUET = _first_existing(
        [
            os.environ.get("AMBIT_DATA_ROOT", "data") + "/datasets/spatial_librispeech/metadata/metadata.parquet",
            os.environ.get("AMBIT_DATA_ROOT", "data") + "/datasets/spatial_librispeech/metadata/metadata.parquet",
        ]
    )

    print("[setup] building held-out SLS list ...")
    held = heldout_eval.held_out_sls_files()
    rng = random.Random(args.seed)
    picks = held[:]
    rng.shuffle(picks)
    # oversample a bit so skips don't shrink the set
    candidates = picks[: args.num * 3]

    sample_ids = {int(Path(f).stem) for f in candidates if Path(f).stem.isdigit()}
    print("[setup] loading GT DoA from parquet ...")
    try:
        gt_map = heldout_eval.load_gt_doa_map(sample_ids)
    except ModuleNotFoundError as exc:
        if exc.name != "pyarrow":
            raise
        print("[setup] pyarrow missing; skipping GT-DoA metrics")
        gt_map = {}

    # Materialize the fixed clip list once (first arm defines which files succeed).
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    first_tag = next(iter(arms))
    first_meta = arms[first_tag]
    print(f"[load] {first_tag}: {first_meta['ckpt']}")
    model, sr = load_4ch_vae(first_meta["config"], first_meta["ckpt"], device)

    fixed_clips: list[tuple[str, torch.Tensor]] = []
    for fpath in candidates:
        if len(fixed_clips) >= args.num:
            break
        stem = Path(fpath).stem
        try:
            raw, src_sr = _read_4ch(fpath)
            if raw.shape[0] < 4:
                continue
            src4 = torch.from_numpy(_resample(raw[:4], src_sr, sr)).float()
            if src4.shape[-1] < sr:  # <1s
                continue
            # Cap length to 4s to keep CPU eval tractable and matched across arms.
            max_len = sr * 4
            if src4.shape[-1] > max_len:
                start = (src4.shape[-1] - max_len) // 2
                src4 = src4[:, start : start + max_len]
            fixed_clips.append((fpath, src4))
        except Exception as exc:  # noqa: BLE001
            print(f"[skip-pick] {stem}: {exc!r}")
            continue

    if len(fixed_clips) < args.num:
        print(f"[warn] only got {len(fixed_clips)} usable clips (wanted {args.num})")
    print(f"[setup] evaluating {len(fixed_clips)} fixed clips @ {sr} Hz, device={device}")

    per_arm_rows: dict[str, list[dict]] = {}
    aggregates: dict[str, dict] = {}

    # Free the clip-picking model; each arm loads its own EMA weights below.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for tag, meta in arms.items():
        print(f"[load] {tag}: {meta['ckpt']}")
        model_arm, sr_arm = load_4ch_vae(meta["config"], meta["ckpt"], device)
        assert sr_arm == sr

        rows: list[dict] = []
        for idx, (fpath, src4) in enumerate(fixed_clips):
            stem = Path(fpath).stem
            with torch.no_grad():
                recon4 = _enc_dec_pad(model_arm, src4, device).clamp(-1, 1)
            n = min(src4.shape[-1], recon4.shape[-1])
            src_np = src4[:, :n].numpy()
            recon_np = recon4[:, :n].cpu().numpy()
            metrics = evaluate_pair(recon_np, src_np)
            metrics.update(
                heldout_eval.extra_metrics(
                    recon_np, src_np, sr, gt_map.get(int(stem)) if stem.isdigit() else None
                )
            )
            rows.append({"idx": idx, "file": stem, "source": fpath, **metrics})
            print(
                f"[{tag} {idx + 1}/{len(fixed_clips)}] {stem} "
                f"LSD={metrics['lsd_db']:.2f} DoAaz={metrics.get('doa_az_err_deg', float('nan')):.1f} "
                f"IC={metrics.get('ic_corr_err', float('nan')):.4f}"
            )

        per_arm_rows[tag] = rows
        aggregates[tag] = {k: _agg(rows, k) for k in METRIC_KEYS}

        with (args.out / f"{tag}_per_file.csv").open("w", newline="", encoding="utf-8") as f:
            cols = ["idx", "file", "source"] + METRIC_KEYS
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in cols})

        del model_arm
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Side-by-side summary
    report = {
        "n_clips": len(fixed_clips),
        "seed": args.seed,
        "device": args.device,
        "sample_rate": sr,
        "test_set": "held-out Spatial-LibriSpeech (NOT in VAE training)",
        "held_out_pool": len(held),
        "arms": {k: {"ckpt": v["ckpt"], "config": v["config"], "note": v["note"]} for k, v in arms.items()},
        "aggregate": aggregates,
    }
    summary_stem = f"compare_{args.step}k_summary"
    (args.out / f"{summary_stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Markdown table: median for each metric across arms
    lines = [
        f"# Ablation compare @ {args.step}k: base_cont vs phase+SCM",
        "",
        f"- clips: {len(fixed_clips)} held-out SLS (seed={args.seed}, 4s center crop)",
        f"- device: `{args.device}`",
        "- note: phase+INT abandoned (850k eval showed dir_energy/elevation collapse)",
        "",
        "| metric | base_cont | phase+SCM | direction |",
        "|---|---:|---:|---|",
    ]
    tags = list(arms.keys())
    for key in METRIC_KEYS:
        cells = []
        for tag in tags:
            block = aggregates[tag].get(key, {})
            med = block.get("median")
            cells.append(f"{med:.3f}" if med is not None else "nan")
        direction = "↑" if key in HIGHER_BETTER else "↓"
        lines.append(f"| {key} | {cells[0]} | {cells[1]} | {direction} |")

    # Quick deltas vs base
    lines.extend(["", "## Deltas vs base_cont (median)", ""])
    lines.append("| metric | Δ SCM | direction |")
    lines.append("|---|---:|---|")
    for key in METRIC_KEYS:
        base = aggregates[tags[0]].get(key, {}).get("median")
        scm = aggregates[tags[1]].get(key, {}).get("median")
        if base is None or scm is None:
            continue
        direction = "↑ better" if key in HIGHER_BETTER else "↓ better"
        lines.append(f"| {key} | {scm - base:+.3f} | {direction} |")

    md = "\n".join(lines) + "\n"
    (args.out / f"{summary_stem}.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Wrote -> {args.out}")


if __name__ == "__main__":
    main()
