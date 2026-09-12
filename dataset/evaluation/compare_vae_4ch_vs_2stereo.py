#!/usr/bin/env python3
"""Compare our native 4ch FOA VAE vs a "2x stereo VAE" baseline on reconstruction.

Question this answers
---------------------
Do we actually need a purpose-built 4-channel (FOA) VAE, or could we just reuse an
off-the-shelf stereo VAE twice (split the 4 ambisonic channels into two stereo pairs,
encode/decode each, concatenate back)? We measure reconstruction fidelity -- ordinary
audio AND spatial (FOA) fidelity -- for both on the SAME source clips.

Two reconstruction paths (per source FOA clip [W, Y, Z, X])
-----------------------------------------------------------
A) OURS  (4ch native)   : decode the pre-encoded ds1024_z64 latent  -> recon_4ch
                          (exactly the latents that will feed the Stage-2 DiT)
B) BASE  (2x stereo)    : split [W,Y,Z,X] into ([W,Y],[Z,X]); run each stereo pair
                          through the pretrained Stable-Audio-Open stereo VAE
                          (encode->decode); concatenate -> recon_4ch

Fairness: latent budgets are EQUAL.
  ours  = 64 * T/1024
  base  = 2 pairs * 64 * T/2048 = 64 * T/1024
so this isolates "joint 4ch modeling" vs "two independent stereo encodes" at the
same compression rate.

Metrics (same for both, vs the same source): si_sdr_db (4ch mean), w_si_sdr_db,
lsd_db, stft_mag_l1, doa_az_err_deg, doa_el_err_deg, dir_energy_ratio_err,
ic_corr_err  (see eval_vae_recon.py for definitions).

Outputs (--out, default ${AMBIT_CKPT_ROOT}/eval_metric):
  per_file.csv     one row per (clip, method)
  summary.json     aggregate medians/means for both methods + deltas
  summary.md       human-readable side-by-side table with a winner per metric

Run:
  cd ./stable-audio-tools
  CUDA_VISIBLE_DEVICES=0 uv run python dataset/evaluation/compare_vae_4ch_vs_2stereo.py \
      --num 100 --out ${AMBIT_CKPT_ROOT}/eval_metric
"""
from __future__ import annotations
import os

import argparse
import copy
import csv
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SAT_ROOT = _HERE.parents[1]            # stable-audio-tools/
for p in (str(_HERE), str(_SAT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# reuse the exact metric + io helpers from the single-model evaluator
from eval_vae_recon import _read_4ch, _resample, _align, evaluate_pair, _agg  # noqa: E402

from stable_audio_tools.models.factory import create_model_from_config  # noqa: E402
from stable_audio_tools.models.utils import load_ckpt_state_dict, copy_state_dict  # noqa: E402
from stable_audio_tools.models.autoencoders_4ch import (  # noqa: E402
    load_raw_state_dict, extract_autoencoder_state_dict,
)

METRIC_KEYS = ["si_sdr_db", "w_si_sdr_db", "lsd_db", "stft_mag_l1",
               "doa_az_err_deg", "doa_el_err_deg", "dir_energy_ratio_err", "ic_corr_err"]
HIGHER_BETTER = {"si_sdr_db", "w_si_sdr_db"}

# channel pairings of FOA [W(0), Y(1), Z(2), X(3)] into two stereo pairs
PAIRINGS = {
    "wy_zx": ((0, 1), (2, 3)),
    "wx_yz": ((0, 3), (1, 2)),
    "wz_yx": ((0, 2), (1, 3)),
}


# --------------------------------------------------------------------------- models

def load_4ch_vae(config_path: str, ckpt_path: str, device):
    with open(config_path) as f:
        cfg = json.load(f)
    model = create_model_from_config(cfg)
    copy_state_dict(model, load_ckpt_state_dict(ckpt_path))
    model.eval().requires_grad_(False).to(device)
    return model, int(cfg["sample_rate"])


def load_2ch_vae(base_config_path: str, ckpt_path: str, device):
    """Build a native 2-channel Oobleck VAE from the ds2048 config (channels->2) and
    load the pretrained Stable-Audio-Open autoencoder weights."""
    with open(base_config_path) as f:
        cfg = copy.deepcopy(json.load(f))
    cfg["audio_channels"] = 2
    cfg["model"]["encoder"]["config"]["in_channels"] = 2
    cfg["model"]["decoder"]["config"]["out_channels"] = 2
    cfg["model"]["io_channels"] = 2
    model = create_model_from_config(cfg)
    ae_sd = extract_autoencoder_state_dict(load_raw_state_dict(ckpt_path))
    missing, unexpected = model.load_state_dict(ae_sd, strict=False)
    real_missing = [k for k in missing if "bottleneck" not in k]
    print(f"[2ch VAE] loaded pretrained AE: missing(non-bottleneck)={len(real_missing)} "
          f"unexpected={len(unexpected)}")
    if real_missing:
        print(f"[2ch VAE] sample missing: {real_missing[:5]}")
    model.eval().requires_grad_(False).to(device)
    return model, int(cfg["sample_rate"])


# ----------------------------------------------------------------------- recon paths

def decode_our_latent(model4, npy_path: Path, device) -> torch.Tensor:
    lat = torch.from_numpy(np.load(npy_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        audio = model4.decode(lat).squeeze(0).float().cpu().clamp(-1, 1)
    return audio  # [4, T]


def _enc_dec_pad(model, audio_bc_t: torch.Tensor, device) -> torch.Tensor:
    """Encode->decode a [C,T] tensor, padding T to a multiple of model.min_length,
    then trimming back to the original length."""
    c, t = audio_bc_t.shape
    min_len = model.min_length
    pad = (-t) % min_len
    x = audio_bc_t
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        lat = model.encode(x)
        rec = model.decode(lat).squeeze(0).float().cpu()
    return rec[:, :t]


def baseline_2stereo(model2, src4: torch.Tensor, pairing, device) -> torch.Tensor:
    """src4 [4,T] -> two stereo pairs through the 2ch VAE -> reassembled [4,T]."""
    (a0, a1), (b0, b1) = pairing
    pair_a = torch.stack([src4[a0], src4[a1]], dim=0)   # [2,T]
    pair_b = torch.stack([src4[b0], src4[b1]], dim=0)
    rec_a = _enc_dec_pad(model2, pair_a, device)        # [2,T]
    rec_b = _enc_dec_pad(model2, pair_b, device)
    out = torch.zeros_like(src4)
    n = min(out.shape[-1], rec_a.shape[-1], rec_b.shape[-1])
    out[a0, :n], out[a1, :n] = rec_a[0, :n], rec_a[1, :n]
    out[b0, :n], out[b1, :n] = rec_b[0, :n], rec_b[1, :n]
    return out.clamp(-1, 1)


# ------------------------------------------------------------------------- sampling

def collect_latents(latent_root: Path):
    items = []
    for rank_dir in sorted(p for p in latent_root.iterdir() if p.is_dir()):
        for jp in rank_dir.glob("*.json"):
            npy = jp.with_suffix(".npy")
            if npy.exists():
                items.append((npy, jp))
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latent-root", type=Path, default=Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/audio_latents/construct_4ch"))
    ap.add_argument("--vae4-config", default="stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae_ds1024.json")
    ap.add_argument("--vae4-ckpt", default=os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/vae_ds1024_z64_construct/unwrapped_ds1024_z64.ckpt")
    ap.add_argument("--vae2-base-config", default="stable_audio_tools/configs/model_configs/autoencoders/stable_audio_open_1_0_oobleck_2ch.json",
                    help="ds2048 config used as the channel-2 template for the stereo baseline VAE")
    ap.add_argument("--vae2-ckpt", default=os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/stable-audio-open-1.0/model.safetensors")
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--pairing", choices=list(PAIRINGS), default="wy_zx")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/eval_metric"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    pairing = PAIRINGS[args.pairing]

    print(f"[load] 4ch VAE: {args.vae4_ckpt}")
    model4, sr4 = load_4ch_vae(args.vae4_config, args.vae4_ckpt, device)
    print(f"[load] 2ch baseline VAE: {args.vae2_ckpt}")
    model2, sr2 = load_2ch_vae(args.vae2_base_config, args.vae2_ckpt, device)
    if sr2 != sr4:
        print(f"[warn] sample-rate mismatch sr4={sr4} sr2={sr2}; using sr4 for both")

    all_items = collect_latents(args.latent_root)
    print(f"[data] {len(all_items)} pre-encoded latents found under {args.latent_root}")

    rng = random.Random(args.seed)
    rng.shuffle(all_items)

    rows_ours, rows_base = [], []
    used = 0
    for npy_path, jp in all_items:
        if used >= args.num:
            break
        meta = json.loads(jp.read_text())
        fmt = meta.get("spatial_format", "foa")
        src_path = meta.get("path", "")
        if fmt != "foa" or not src_path or not Path(src_path).exists():
            continue
        try:
            src_raw, s_sr = _read_4ch(src_path)            # [C,T] @ native sr
            if src_raw.shape[0] < 4:
                continue
            src4_np = _resample(src_raw[:4], s_sr, sr4)     # reference @ 44.1k
            src4 = torch.from_numpy(src4_np).float()

            ours4 = decode_our_latent(model4, npy_path, device)            # [4,T] @ 44.1k
            base4 = baseline_2stereo(model2, src4, pairing, device)        # [4,T] @ 44.1k

            # align all three to common length
            n = min(src4.shape[-1], ours4.shape[-1], base4.shape[-1])
            src_n = src4[:, :n].numpy()
            m_ours = evaluate_pair(ours4[:, :n].numpy(), src_n)
            m_base = evaluate_pair(base4[:, :n].numpy(), src_n)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {Path(src_path).name}: {e!r}")
            continue

        stem = Path(src_path).stem
        for m, rows in ((m_ours, rows_ours), (m_base, rows_base)):
            m["idx"] = used
            m["source"] = src_path
            m["file"] = stem
        rows_ours.append(m_ours)
        rows_base.append(m_base)
        used += 1
        if used % 10 == 0 or used == args.num:
            print(f"[{used}/{args.num}] {stem}  "
                  f"ours LSD={m_ours['lsd_db']:.2f} DoAaz={m_ours.get('doa_az_err_deg', float('nan')):.1f}  |  "
                  f"base LSD={m_base['lsd_db']:.2f} DoAaz={m_base.get('doa_az_err_deg', float('nan')):.1f}")

    if not rows_ours:
        raise RuntimeError("No clips evaluated. Check latent-root / source paths.")

    agg = {
        "ours_4ch_native": {k: _agg(rows_ours, k) for k in METRIC_KEYS},
        "baseline_2x_stereo": {k: _agg(rows_base, k) for k in METRIC_KEYS},
    }

    # winner per metric (by median)
    winners = {}
    for k in METRIC_KEYS:
        o = agg["ours_4ch_native"][k].get("median")
        b = agg["baseline_2x_stereo"][k].get("median")
        if o is None or b is None:
            winners[k] = "n/a"
        elif k in HIGHER_BETTER:
            winners[k] = "ours" if o > b else ("base" if b > o else "tie")
        else:
            winners[k] = "ours" if o < b else ("base" if b < o else "tie")

    report = {
        "n_clips": len(rows_ours),
        "pairing": args.pairing,
        "vae4_ckpt": args.vae4_ckpt,
        "vae2_ckpt": args.vae2_ckpt,
        "sample_rate": sr4,
        "aggregate": agg,
        "winner_by_median": winners,
        "metric_help": {
            "si_sdr_db": "higher better (4ch mean)", "w_si_sdr_db": "higher better (W only)",
            "lsd_db": "lower better", "stft_mag_l1": "lower better",
            "doa_az_err_deg": "lower better (FOA azimuth)", "doa_el_err_deg": "lower better (FOA elevation)",
            "dir_energy_ratio_err": "lower better", "ic_corr_err": "lower better",
        },
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2))

    # per-file csv
    with (args.out / "per_file.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "idx", "file"] + METRIC_KEYS + ["source"])
        w.writeheader()
        for m in rows_ours:
            w.writerow({"method": "ours_4ch_native", **{k: m.get(k, "") for k in ["idx", "file"] + METRIC_KEYS + ["source"]}})
        for m in rows_base:
            w.writerow({"method": "baseline_2x_stereo", **{k: m.get(k, "") for k in ["idx", "file"] + METRIC_KEYS + ["source"]}})

    # markdown side-by-side
    lines = [
        "# VAE reconstruction: 4ch native vs 2x stereo baseline", "",
        f"clips: {len(rows_ours)}  |  pairing: `{args.pairing}` "
        f"(FOA split into {pairing[0]} + {pairing[1]})  |  sr: {sr4}", "",
        f"- ours = decode pre-encoded **ds1024_z64** latents (`{Path(args.vae4_ckpt).name}`)",
        f"- baseline = 2x **Stable-Audio-Open stereo VAE** (`{Path(args.vae2_ckpt).name}`), encode/decode each pair",
        "- latent budget is EQUAL for both (64*T/1024).", "",
        "| metric | ours (median) | baseline (median) | winner | ours mean | base mean |",
        "|---|---|---|---|---|---|",
    ]
    arrow = {True: "↑", False: "↓"}
    for k in METRIC_KEYS:
        o = agg["ours_4ch_native"][k]
        b = agg["baseline_2x_stereo"][k]
        if not o or not b:
            continue
        lines.append(
            f"| {k} {arrow[k in HIGHER_BETTER]} | {o['median']:.3f} | {b['median']:.3f} | "
            f"**{winners[k]}** | {o['mean']:.3f} | {b['mean']:.3f} |"
        )
    n_ours_win = sum(1 for v in winners.values() if v == "ours")
    lines += ["", f"**ours wins {n_ours_win}/{len(METRIC_KEYS)} metrics (by median).**", ""]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWrote -> {args.out}/  (summary.md, summary.json, per_file.csv)")


if __name__ == "__main__":
    main()
