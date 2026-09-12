#!/usr/bin/env python3
"""Compare 4ch VAE experiments by parsing their TensorBoard scalars.

For each experiment under EXP_ROOT we pick the event file with the most
logged steps (handles aborted version_0 dirs), then report reconstruction
quality (train/mrstft_loss) and supporting metrics, both as the final value
and smoothed over the last window, evaluated at a common matched step so the
four configs are compared fairly.
"""
from __future__ import annotations

import glob
import os
from collections import defaultdict

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

EXP_ROOT = "/mnt/sdc/ckpts/vae_exp"
KEY = "train/mrstft_loss"          # primary reconstruction-quality metric
EXTRA = [
    "train/loss",
    "train/kl_loss",
    "train/loss_adv",
    "train/feature_matching_loss",
    "train/latent_std",
    "train/latent_mean",
    "train/data_std",
]
SMOOTH_STEPS = 4000   # window (in steps) for the trailing average


def load_best_event(exp_dir: str):
    """Return EventAccumulator for the event file with the most steps."""
    files = glob.glob(os.path.join(exp_dir, "**", "events.out.tfevents*"), recursive=True)
    best_ea, best_n = None, -1
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags()["scalars"]
        n = len(ea.Scalars(KEY)) if KEY in tags else (
            len(ea.Scalars("train/loss")) if "train/loss" in tags else 0)
        if n > best_n:
            best_ea, best_n, best_f = ea, n, f
    return best_ea, best_f


def series(ea, tag):
    if tag not in ea.Tags()["scalars"]:
        return np.array([]), np.array([])
    ev = ea.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def trailing_mean(steps, vals, upto_step, window):
    if len(steps) == 0:
        return float("nan")
    mask = (steps <= upto_step) & (steps > upto_step - window)
    if not mask.any():
        mask = steps <= upto_step
    return float(np.mean(vals[mask]))


def value_at(steps, vals, upto_step):
    if len(steps) == 0:
        return float("nan")
    mask = steps <= upto_step
    if not mask.any():
        return float("nan")
    return float(vals[mask][-1])


def main():
    exps = sorted(d for d in glob.glob(os.path.join(EXP_ROOT, "*")) if os.path.isdir(d))
    data = {}
    max_steps = {}
    for exp in exps:
        name = os.path.basename(exp)
        ea, fpath = load_best_event(exp)
        s, v = series(ea, KEY)
        data[name] = ea
        max_steps[name] = int(s.max()) if len(s) else 0
        print(f"[load] {name:14s} steps={max_steps[name]:>7d}  ({os.path.relpath(fpath, exp)})")

    common = min(max_steps.values()) if max_steps else 0
    print(f"\nCommon matched step for comparison: {common}\n")

    # Build comparison table on mrstft_loss
    rows = []
    for name, ea in data.items():
        s, v = series(ea, KEY)
        rows.append((
            name,
            value_at(s, v, common),
            trailing_mean(s, v, common, SMOOTH_STEPS),
            value_at(s, v, max_steps[name]),
            trailing_mean(s, v, max_steps[name], SMOOTH_STEPS),
        ))

    print("=" * 92)
    print(f"RECONSTRUCTION QUALITY  ({KEY}, LOWER = BETTER)")
    print("=" * 92)
    print(f"{'config':14s} | {'@'+str(common)+' last':>14s} | {'@common smooth':>14s} | "
          f"{'final last':>12s} | {'final smooth':>13s}")
    print("-" * 92)
    rows_sorted = sorted(rows, key=lambda r: (np.nan_to_num(r[2], nan=1e9)))
    for name, c_last, c_sm, f_last, f_sm in rows_sorted:
        print(f"{name:14s} | {c_last:14.4f} | {c_sm:14.4f} | {f_last:12.4f} | {f_sm:13.4f}")
    best = rows_sorted[0][0]
    print("-" * 92)
    print(f"BEST by mrstft (smoothed @ common step): {best}\n")

    # Supporting metrics at common step (smoothed)
    print("=" * 92)
    print(f"SUPPORTING METRICS (smoothed over last {SMOOTH_STEPS} steps @ step {common})")
    print("=" * 92)
    header = f"{'config':14s} | " + " | ".join(f"{t.split('/')[-1]:>16s}" for t in EXTRA)
    print(header)
    print("-" * len(header))
    for name, ea in data.items():
        cells = []
        for t in EXTRA:
            s, v = series(ea, t)
            cells.append(f"{trailing_mean(s, v, common, SMOOTH_STEPS):16.4f}")
        print(f"{name:14s} | " + " | ".join(cells))

    # Save mrstft curves to CSV for plotting if desired
    out_csv = os.path.join(os.path.dirname(__file__), "vae_exp_mrstft.csv")
    all_steps = sorted({st for ea in data.values() for st in series(ea, KEY)[0].tolist()})
    series_map = {name: dict(zip(*[arr.tolist() for arr in series(ea, KEY)])) for name, ea in data.items()}
    with open(out_csv, "w") as fh:
        fh.write("step," + ",".join(data.keys()) + "\n")
        for st in all_steps:
            fh.write(str(st) + "," + ",".join(
                ("" if st not in series_map[n] else f"{series_map[n][st]:.6f}") for n in data.keys()) + "\n")
    print(f"\n[csv] wrote mrstft curves -> {out_csv}")


if __name__ == "__main__":
    main()
