#!/usr/bin/env python3
"""LEAKAGE-FREE VAE reconstruction comparison: our 4ch native VAE vs 2x stereo baseline.

Fixes the train/test leakage of compare_vae_4ch_vs_2stereo.py (which sampled the
pre-encoded latents = the VAE's own training files). Here the test set is HELD-OUT
Spatial-LibriSpeech: the complement of the deterministic 60,255-file subset the VAE
was trained on (reproduced with the exact same seeded sampling as dataset_4ch). These
clips were NEVER seen by our VAE, and never by the off-the-shelf stereo VAE either.

Two reconstruction paths per held-out FOA clip [W,Y,Z,X] (encode-fresh, since these
were never pre-encoded):
  A) OURS : our 4ch ds1024_z64 VAE  encode -> decode                      -> recon_4ch
  B) BASE : split [W,Y,Z,X] -> ([W,Y],[Z,X]); each through Stable-Audio-Open stereo
            VAE encode/decode; concat                                      -> recon_4ch
(latent budgets equal: ours 64*T/1024 ; base 2*64*T/2048 = 64*T/1024.)

Metrics (per clip, vs the same source unless noted):
  base set (eval_vae_recon): si_sdr_db, w_si_sdr_db, lsd_db, stft_mag_l1,
      doa_az_err_deg, doa_el_err_deg, dir_energy_ratio_err, ic_corr_err  (recon vs source)
  + GT-anchored localisation (SLS metadata.parquet has ground-truth az/el):
      gt_doa_az_err_deg, gt_doa_el_err_deg          (recon intensity-DoA vs GROUND TRUTH)
      (also reported for the clean source as the achievable floor: src_gt_*_err_deg)
  + binaural cues (FOA virtual-stereo L=W+Y, R=W-Y):
      ild_err_db, itd_err_us                        (|recon-source| of ILD / ITD)
  + perceptual quality on the audible W channel:
      pesq_w                                        (wideband PESQ, ref=source W; higher better)
  + (optional) SELD localisation via PSELDNets, if --with-seld and the wrapper imports:
      seld_doa_err_deg (recon vs source pred), seld_gt_doa_err_deg (recon vs GT)

Outputs (--out, default /mnt/sdc/eval_metric):
  heldout_per_file.csv, heldout_summary.json, heldout_summary.md

Run:
  cd /home/tanhe/dataset_storage/stable-audio-tools
  CUDA_VISIBLE_DEVICES=1 uv run python dataset/evaluation/compare_vae_heldout.py \
      --num 100 --out /mnt/sdc/eval_metric
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SAT_ROOT = _HERE.parents[1]
for p in (str(_HERE), str(_SAT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_vae_recon import _read_4ch, _resample, evaluate_pair, _agg, foa_intensity_doa, _ang_err  # noqa: E402
from compare_vae_4ch_vs_2stereo import load_4ch_vae, load_2ch_vae, baseline_2stereo, _enc_dec_pad, PAIRINGS  # noqa: E402

from stable_audio_tools.data.dataset import get_audio_filenames  # noqa: E402

try:
    from pesq import pesq as _pesq  # python-pesq
    _HAVE_PESQ = True
except Exception:  # noqa: BLE001
    _HAVE_PESQ = False

try:
    from scipy.signal import resample_poly  # noqa: E402
    _HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    _HAVE_SCIPY = False

EPS = 1e-9
SLS_DIR = "/mnt/sdd/audio_dataset/datasets/spatial_librispeech/ambisonics"
SLS_PARQUET = "/mnt/sdd/audio_dataset/datasets/spatial_librispeech/metadata/metadata.parquet"
TRAIN_ID = "spatial_librispeech"   # dataset id used as the RNG seed in dataset_4ch
TRAIN_MAX_FILES = 60255

BASE_KEYS = ["si_sdr_db", "w_si_sdr_db", "lsd_db", "stft_mag_l1",
             "doa_az_err_deg", "doa_el_err_deg", "dir_energy_ratio_err", "ic_corr_err"]
EXTRA_KEYS = ["gt_doa_az_err_deg", "gt_doa_el_err_deg",
              "src_gt_doa_az_err_deg", "src_gt_doa_el_err_deg",
              "ild_err_db", "itd_err_us", "pesq_w"]
SELD_KEYS = ["seld_doa_err_deg", "seld_gt_doa_err_deg", "src_seld_gt_doa_err_deg"]
HIGHER_BETTER = {"si_sdr_db", "w_si_sdr_db", "pesq_w"}


# ----------------------------------------------------------------- held-out set

def held_out_sls_files():
    files = get_audio_filenames(SLS_DIR)
    if len(files) <= TRAIN_MAX_FILES:
        print(
            f"[heldout] pool={len(files)} <= train cap {TRAIN_MAX_FILES}; "
            f"using all available files as eval pool ({SLS_DIR})"
        )
        return sorted(files)
    train = set(sorted(random.Random(TRAIN_ID).sample(files, TRAIN_MAX_FILES)))
    held = [f for f in files if f not in train]
    return sorted(held)


def load_gt_doa_map(sample_ids: set[int]) -> dict:
    """sample_id -> (az_deg, el_deg) from SLS parquet. SLS: az>0=left, 0=front; el>0=up."""
    import pyarrow.parquet as pq
    cols = ["sample_id", "speech/azimuth", "speech/elevation"]
    out = {}
    pf = pq.ParquetFile(SLS_PARQUET)
    for batch in pf.iter_batches(batch_size=8192, columns=cols):
        for r in batch.to_pylist():
            sid = int(r["sample_id"])
            if sid in sample_ids:
                out[sid] = (math.degrees(r["speech/azimuth"]), math.degrees(r["speech/elevation"]))
    return out


# ----------------------------------------------------------------- extra metrics

def _resample_1d(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    if _HAVE_SCIPY:
        g = math.gcd(int(sr_in), int(sr_out))
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    n = int(round(len(x) * sr_out / sr_in))
    xo = np.linspace(0, 1, len(x), endpoint=False)
    xn = np.linspace(0, 1, n, endpoint=False)
    return np.interp(xn, xo, x).astype(np.float32)


def gt_doa_err(foa4: np.ndarray, az_gt_deg: float, el_gt_deg: float):
    az, el = foa_intensity_doa(foa4)
    return _ang_err(az, az_gt_deg), _ang_err(el, el_gt_deg)


def _virtual_stereo(foa4: np.ndarray):
    """FOA [W,Y,Z,X] -> crude L/R via first-order cardioids (L=W+Y, R=W-Y)."""
    w, y = foa4[0], foa4[1]
    return w + y, w - y


def ild_db(foa4: np.ndarray) -> float:
    L, R = _virtual_stereo(foa4)
    return 10.0 * math.log10((float(np.sum(L**2)) + EPS) / (float(np.sum(R**2)) + EPS))


def itd_samples(foa4: np.ndarray, max_lag: int = 64) -> float:
    """GCC-PHAT lag (in samples) between virtual L and R; + = source toward left."""
    L, R = _virtual_stereo(foa4)
    n = 1
    while n < len(L) + len(R):
        n <<= 1
    FL = np.fft.rfft(L, n); FR = np.fft.rfft(R, n)
    cc = FL * np.conj(FR)
    cc = cc / (np.abs(cc) + EPS)
    r = np.fft.irfft(cc, n)
    r = np.concatenate((r[-max_lag:], r[:max_lag + 1]))
    lag = int(np.argmax(r)) - max_lag
    return float(lag)


def pesq_w(recon_w: np.ndarray, src_w: np.ndarray, sr: int) -> float:
    if not _HAVE_PESQ:
        return float("nan")
    ref = _resample_1d(src_w, sr, 16000)
    deg = _resample_1d(recon_w, sr, 16000)
    n = min(len(ref), len(deg))
    if n < 16000 // 2:
        return float("nan")
    try:
        return float(_pesq(16000, ref[:n], deg[:n], "wb"))
    except Exception:  # noqa: BLE001
        return float("nan")


def extra_metrics(recon4: np.ndarray, src4: np.ndarray, sr: int, gt):
    m = {}
    if gt is not None:
        az_gt, el_gt = gt
        m["gt_doa_az_err_deg"], m["gt_doa_el_err_deg"] = gt_doa_err(recon4, az_gt, el_gt)
        m["src_gt_doa_az_err_deg"], m["src_gt_doa_el_err_deg"] = gt_doa_err(src4, az_gt, el_gt)
    m["ild_err_db"] = abs(ild_db(recon4) - ild_db(src4))
    m["itd_err_us"] = abs(itd_samples(recon4) - itd_samples(src4)) / sr * 1e6
    m["pesq_w"] = pesq_w(recon4[0], src4[0], sr)
    return m


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vae4-config", default="stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae_ds1024.json")
    ap.add_argument("--vae4-ckpt", default="/mnt/sdc/ckpts/vae_ds1024_z64_construct/unwrapped_ds1024_z64.ckpt")
    ap.add_argument("--vae2-base-config", default="stable_audio_tools/configs/model_configs/autoencoders/stable_audio_open_1_0_oobleck_2ch.json")
    ap.add_argument("--vae2-ckpt", default="/mnt/sdc/ckpts/stable-audio-open-1.0/model.safetensors")
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--pairing", choices=list(PAIRINGS), default="wy_zx")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, default=Path("/mnt/sdc/eval_metric"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--with-seld", action="store_true", help="add PSELDNets SELD-DoA columns (needs seld_pseldnets.py)")
    args = ap.parse_args()

    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    pairing = PAIRINGS[args.pairing]

    seld = None
    metric_keys = BASE_KEYS + EXTRA_KEYS
    if args.with_seld:
        try:
            from seld_pseldnets import PSELDNetsScorer  # noqa: E402
            seld = PSELDNetsScorer(device=device)
            metric_keys = metric_keys + SELD_KEYS
            print("[seld] PSELDNets scorer ready")
        except Exception as e:  # noqa: BLE001
            print(f"[seld] disabled ({e!r})")

    print("[setup] building held-out SLS list ...")
    held = held_out_sls_files()
    print(f"[setup] held-out SLS available: {len(held)}")
    rng = random.Random(args.seed)
    rng.shuffle(held)
    picks = held[: args.num * 2]   # oversample; some may fail to read

    sample_ids = {int(Path(f).stem) for f in picks}
    print("[setup] loading GT DoA from parquet ...")
    gt_map = load_gt_doa_map(sample_ids)
    print(f"[setup] GT DoA entries: {len(gt_map)}")

    print(f"[load] 4ch VAE: {args.vae4_ckpt}")
    model4, sr4 = load_4ch_vae(args.vae4_config, args.vae4_ckpt, device)
    print(f"[load] 2ch baseline VAE: {args.vae2_ckpt}")
    model2, _ = load_2ch_vae(args.vae2_base_config, args.vae2_ckpt, device)

    rows_ours, rows_base = [], []
    used = 0
    for fpath in picks:
        if used >= args.num:
            break
        sid = int(Path(fpath).stem)
        gt = gt_map.get(sid)
        try:
            raw, s_sr = _read_4ch(fpath)
            if raw.shape[0] < 4:
                continue
            src4 = torch.from_numpy(_resample(raw[:4], s_sr, sr4)).float()
            ours4 = _enc_dec_pad(model4, src4, device).clamp(-1, 1)
            base4 = baseline_2stereo(model2, src4, pairing, device)
            n = min(src4.shape[-1], ours4.shape[-1], base4.shape[-1])
            src_np = src4[:, :n].numpy()
            ours_np = ours4[:, :n].numpy()
            base_np = base4[:, :n].numpy()

            m_ours = evaluate_pair(ours_np, src_np)
            m_base = evaluate_pair(base_np, src_np)
            m_ours.update(extra_metrics(ours_np, src_np, sr4, gt))
            m_base.update(extra_metrics(base_np, src_np, sr4, gt))
            if seld is not None:
                so = seld.doa(ours_np, sr4); sb = seld.doa(base_np, sr4); ss = seld.doa(src_np, sr4)
                m_ours["seld_doa_err_deg"] = _ang_err(so[0], ss[0])
                m_base["seld_doa_err_deg"] = _ang_err(sb[0], ss[0])
                if gt is not None:
                    m_ours["seld_gt_doa_err_deg"] = _ang_err(so[0], gt[0])
                    m_base["seld_gt_doa_err_deg"] = _ang_err(sb[0], gt[0])
                    m_ours["src_seld_gt_doa_err_deg"] = _ang_err(ss[0], gt[0])
                    m_base["src_seld_gt_doa_err_deg"] = _ang_err(ss[0], gt[0])
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {Path(fpath).name}: {e!r}")
            continue

        for m in (m_ours, m_base):
            m["idx"] = used; m["file"] = Path(fpath).stem; m["source"] = fpath
        rows_ours.append(m_ours); rows_base.append(m_base)
        used += 1
        if used % 10 == 0 or used == args.num:
            print(f"[{used}/{args.num}] {Path(fpath).stem}  "
                  f"ours: LSD={m_ours['lsd_db']:.2f} DoAaz={m_ours.get('doa_az_err_deg', float('nan')):.1f} "
                  f"PESQ={m_ours['pesq_w']:.2f}  |  base: LSD={m_base['lsd_db']:.2f} "
                  f"DoAaz={m_base.get('doa_az_err_deg', float('nan')):.1f} PESQ={m_base['pesq_w']:.2f}")

    if not rows_ours:
        raise RuntimeError("No clips evaluated.")

    agg = {
        "ours_4ch_native": {k: _agg(rows_ours, k) for k in metric_keys},
        "baseline_2x_stereo": {k: _agg(rows_base, k) for k in metric_keys},
    }
    winners = {}
    for k in metric_keys:
        o = agg["ours_4ch_native"][k].get("median")
        b = agg["baseline_2x_stereo"][k].get("median")
        if o is None or b is None:
            winners[k] = "n/a"
        elif k in HIGHER_BETTER:
            winners[k] = "ours" if o > b else ("base" if b > o else "tie")
        else:
            winners[k] = "ours" if o < b else ("base" if b < o else "tie")

    report = {
        "n_clips": len(rows_ours), "test_set": "held-out Spatial-LibriSpeech (NOT in VAE training)",
        "held_out_pool": len(held), "pairing": args.pairing,
        "vae4_ckpt": args.vae4_ckpt, "vae2_ckpt": args.vae2_ckpt, "sample_rate": sr4,
        "have_pesq": _HAVE_PESQ, "with_seld": seld is not None,
        "aggregate": agg, "winner_by_median": winners,
    }
    (args.out / "heldout_summary.json").write_text(json.dumps(report, indent=2))

    with (args.out / "heldout_per_file.csv").open("w", newline="") as f:
        cols = ["method", "idx", "file"] + metric_keys + ["source"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for tag, rows in (("ours_4ch_native", rows_ours), ("baseline_2x_stereo", rows_base)):
            for m in rows:
                w.writerow({"method": tag, **{k: m.get(k, "") for k in cols if k != "method"}})

    arrow = {True: "up", False: "down"}
    lines = [
        "# VAE reconstruction (LEAKAGE-FREE): 4ch native vs 2x stereo baseline", "",
        f"test set: **held-out Spatial-LibriSpeech** ({len(rows_ours)} clips, from a pool of "
        f"{len(held)} files NOT in VAE training)  |  pairing `{args.pairing}`  |  sr {sr4}", "",
        f"- ours = our **ds1024_z64** 4ch VAE encode->decode (`{Path(args.vae4_ckpt).name}`)",
        f"- baseline = 2x **Stable-Audio-Open** stereo VAE (`{Path(args.vae2_ckpt).name}`)",
        "- equal latent budget (64*T/1024). GT az/el from SLS metadata.", "",
        "| metric | ours (median) | baseline (median) | winner | ours mean | base mean |",
        "|---|---|---|---|---|---|",
    ]
    for k in metric_keys:
        o = agg["ours_4ch_native"][k]; b = agg["baseline_2x_stereo"][k]
        if not o or not b:
            continue
        d = "up" if k in HIGHER_BETTER else "down"
        lines.append(f"| {k} ({d}) | {o['median']:.3f} | {b['median']:.3f} | **{winners[k]}** | "
                     f"{o['mean']:.3f} | {b['mean']:.3f} |")
    n_ours = sum(1 for k in metric_keys if winners[k] == "ours")
    lines += ["", f"**ours wins {n_ours}/{len(metric_keys)} metrics (by median).**",
              "", "_Note: `src_gt_*` rows are the clean source's own DoA error vs GT = the achievable floor._", ""]
    (args.out / "heldout_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote -> {args.out}/ (heldout_summary.md/json, heldout_per_file.csv)")


if __name__ == "__main__":
    main()
