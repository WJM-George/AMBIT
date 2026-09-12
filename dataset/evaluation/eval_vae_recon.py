#!/usr/bin/env python3
"""Evaluate 4ch VAE reconstruction quality vs the original source audio.

Reads a reconstruction folder produced by ``scripts/vae/eval/decode_latents_4ch.py`` (a
``manifest.json`` + ``*_4ch.flac`` quads, each entry carrying the original
``source`` path), aligns each reconstruction with its source, and reports
objective metrics -- both ordinary audio fidelity AND spatial (FOA) fidelity,
which is what actually matters for this project.

Metrics per clip (and aggregated mean/median):
  * si_sdr_db        : scale-invariant SDR, mean over the 4 channels (higher=better)
  * lsd_db           : log-spectral distance, mean over channels (lower=better)
  * stft_mag_l1      : magnitude-STFT L1 after optimal scaling (lower=better)
  * w_si_sdr_db      : SI-SDR of the W (omni) channel only -- "content" fidelity
  * doa_az_err_deg   : azimuth error of the FOA active-intensity vector (spatial!)
  * doa_el_err_deg   : elevation error of the FOA active-intensity vector
  * dir_energy_ratio_err : |recon - src| of directional/omni energy ratio (image width)
  * ic_corr_err      : mean |Δ| of inter-channel correlations (W-Y, W-X, W-Z, Y-X)

Why these: SI-SDR / LSD measure waveform+timbre reconstruction; the FOA intensity
DoA + directional-energy + inter-channel-correlation metrics measure whether the
*spatial image* (where sounds are, how wide/diffuse) survives the VAE. A VAE can
score well on SI-SDR yet smear direction -- these spatial metrics catch that.

Output: writes into  <output-base>/<tag>/  (default base = recon dir's parent):
    eval_report.json    aggregate + config
    eval_per_file.csv   one row per clip
    summary.md          human-readable table

Run:
    cd /home/tanhe/dataset_storage/stable-audio-tools
    uv run python dataset/evaluation/eval_vae_recon.py \
        --recon-dir /mnt/sdc/vae_4ch_train_result_audio \
        --tag oobleck_ds2048_z64 \
        --output-base /mnt/sdc/vae_4ch_train_result_audio

The --tag is the SHORT name of the VAE setting; results land in
/mnt/sdc/vae_4ch_train_result_audio/<tag>/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

try:
    from scipy.signal import resample_poly, stft as _scipy_stft  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    _HAVE_SCIPY = False

EPS = 1e-9


# --------------------------------------------------------------------------- io

def _read_4ch(path: str) -> tuple[np.ndarray, int]:
    """Read an audio file as [C, T] float32 (up to 4 channels kept)."""
    data, sr = sf.read(path, always_2d=True, dtype="float32")  # [T, C]
    audio = data.T  # [C, T]
    if audio.shape[0] > 4:
        audio = audio[:4]
    return audio, int(sr)


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    if _HAVE_SCIPY:
        g = math.gcd(sr_in, sr_out)
        return resample_poly(audio, sr_out // g, sr_in // g, axis=-1).astype(np.float32)
    # Fallback: linear interpolation (lower quality, dependency-free)
    n_out = int(round(audio.shape[-1] * sr_out / sr_in))
    x_old = np.linspace(0.0, 1.0, audio.shape[-1], endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.stack([np.interp(x_new, x_old, ch) for ch in audio]).astype(np.float32)


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


# ---------------------------------------------------------------------- metrics

def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    """Scale-invariant SDR in dB for one channel (1D arrays)."""
    ref = ref - ref.mean()
    est = est - est.mean()
    ref_energy = float(np.sum(ref**2)) + EPS
    proj = float(np.sum(est * ref)) / ref_energy * ref
    noise = est - proj
    return 10.0 * math.log10((float(np.sum(proj**2)) + EPS) / (float(np.sum(noise**2)) + EPS))


def _stft_mag(x: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    if _HAVE_SCIPY:
        _, _, z = _scipy_stft(x, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
        return np.abs(z).astype(np.float32)
    # numpy framing fallback
    win = np.hanning(n_fft).astype(np.float32)
    frames = 1 + max(0, (len(x) - n_fft) // hop)
    out = np.empty((n_fft // 2 + 1, frames), dtype=np.float32)
    for i in range(frames):
        seg = x[i * hop:i * hop + n_fft] * win
        out[:, i] = np.abs(np.fft.rfft(seg))
    return out


def lsd_db(est: np.ndarray, ref: np.ndarray) -> float:
    """Log-spectral distance (dB) for one channel."""
    E = _stft_mag(est) ** 2 + EPS
    R = _stft_mag(ref) ** 2 + EPS
    m = min(E.shape[1], R.shape[1])
    diff = 10.0 * np.log10(R[:, :m]) - 10.0 * np.log10(E[:, :m])
    return float(np.sqrt(np.mean(diff**2)))


def stft_mag_l1(est: np.ndarray, ref: np.ndarray) -> float:
    """L1 of magnitude STFT after optimal scalar gain on est (one channel)."""
    E = _stft_mag(est)
    R = _stft_mag(ref)
    m = min(E.shape[1], R.shape[1])
    E, R = E[:, :m], R[:, :m]
    g = float(np.sum(E * R)) / (float(np.sum(E * E)) + EPS)
    return float(np.mean(np.abs(g * E - R)))


def foa_intensity_doa(foa: np.ndarray) -> tuple[float, float]:
    """Broadband active-intensity DoA (deg) from FOA [W,Y,Z,X].

    az: + = left (matches SLS/synthesis convention), el: + = up.
    """
    w, y, z, x = foa[0], foa[1], foa[2], foa[3]
    ix = float(np.mean(w * x))
    iy = float(np.mean(w * y))
    iz = float(np.mean(w * z))
    az = math.degrees(math.atan2(iy, ix))
    el = math.degrees(math.atan2(iz, math.hypot(ix, iy) + EPS))
    return az, el


def _ang_err(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def directional_energy_ratio(foa: np.ndarray) -> float:
    w_e = float(np.mean(foa[0] ** 2)) + EPS
    dir_e = float(np.mean(foa[1] ** 2 + foa[2] ** 2 + foa[3] ** 2))
    return dir_e / w_e


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(np.sum(a**2)) * float(np.sum(b**2))) + EPS
    return float(np.sum(a * b)) / denom


def inter_channel_corr_err(est: np.ndarray, ref: np.ndarray) -> float:
    pairs = [(0, 1), (0, 3), (0, 2), (1, 3)]  # W-Y, W-X, W-Z, Y-X
    errs = [abs(_corr(est[i], est[j]) - _corr(ref[i], ref[j])) for i, j in pairs]
    return float(np.mean(errs))


def evaluate_pair(recon: np.ndarray, source: np.ndarray) -> dict:
    recon, source = _align(recon, source)
    n_ch = min(recon.shape[0], source.shape[0], 4)
    si = [si_sdr(recon[c], source[c]) for c in range(n_ch)]
    lsd = [lsd_db(recon[c], source[c]) for c in range(n_ch)]
    ml1 = [stft_mag_l1(recon[c], source[c]) for c in range(n_ch)]
    out = {
        "si_sdr_db": float(np.mean(si)),
        "w_si_sdr_db": float(si[0]),
        "lsd_db": float(np.mean(lsd)),
        "stft_mag_l1": float(np.mean(ml1)),
    }
    if n_ch >= 4:
        raz, rel = foa_intensity_doa(recon)
        saz, sel = foa_intensity_doa(source)
        out["doa_az_err_deg"] = _ang_err(raz, saz)
        out["doa_el_err_deg"] = _ang_err(rel, sel)
        out["dir_energy_ratio_err"] = abs(
            directional_energy_ratio(recon) - directional_energy_ratio(source)
        )
        out["ic_corr_err"] = inter_channel_corr_err(recon, source)
    return out


# ------------------------------------------------------------------------- main

def _agg(rows: list[dict], key: str) -> dict:
    vals = [r[key] for r in rows if key in r and r[key] == r[key]]  # drop NaN
    if not vals:
        return {}
    return {
        "mean": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "n": len(vals),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon-dir", type=Path, required=True,
                    help="Folder with manifest.json + *_4ch.flac (from decode_latents_4ch.py).")
    ap.add_argument("--tag", required=True,
                    help="Short VAE-setting name; results go to <output-base>/<tag>/.")
    ap.add_argument("--output-base", type=Path, default=None,
                    help="Base dir for the <tag> result folder. Default: recon-dir parent.")
    ap.add_argument("--manifest", type=Path, default=None, help="Override manifest path.")
    args = ap.parse_args()

    recon_dir = args.recon_dir.resolve()
    manifest_path = args.manifest or (recon_dir / "manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    out_base = (args.output_base or recon_dir.parent).resolve()
    out_dir = out_base / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for entry in manifest:
        quad_name = entry.get("output_quad") or entry.get("output")
        recon_path = recon_dir / quad_name if quad_name else None
        src_path = entry.get("source")
        if not recon_path or not recon_path.exists() or not src_path or not Path(src_path).exists():
            print(f"[skip] missing recon/source for entry {entry.get('idx')}")
            continue
        recon, r_sr = _read_4ch(str(recon_path))
        source, s_sr = _read_4ch(src_path)
        if s_sr != r_sr:
            source = _resample(source, s_sr, r_sr)
        m = evaluate_pair(recon, source)
        m["idx"] = entry.get("idx")
        m["file"] = quad_name
        m["source"] = src_path
        m["sample_rate"] = r_sr
        rows.append(m)
        print(f"[{m.get('idx')}] SI-SDR={m['si_sdr_db']:.2f}dB LSD={m['lsd_db']:.2f} "
              f"DoA_az_err={m.get('doa_az_err_deg', float('nan')):.1f}deg "
              f"DoA_el_err={m.get('doa_el_err_deg', float('nan')):.1f}deg")

    if not rows:
        raise RuntimeError("No (recon, source) pairs evaluated. Check manifest source paths.")

    metric_keys = ["si_sdr_db", "w_si_sdr_db", "lsd_db", "stft_mag_l1",
                   "doa_az_err_deg", "doa_el_err_deg", "dir_energy_ratio_err", "ic_corr_err"]
    aggregate = {k: _agg(rows, k) for k in metric_keys}

    report = {
        "tag": args.tag,
        "recon_dir": str(recon_dir),
        "n_clips": len(rows),
        "scipy": _HAVE_SCIPY,
        "aggregate": aggregate,
        "metric_help": {
            "si_sdr_db": "higher better (scale-invariant SDR, all 4ch)",
            "w_si_sdr_db": "higher better (omni W only)",
            "lsd_db": "lower better (log-spectral distance)",
            "stft_mag_l1": "lower better (scaled magnitude-STFT L1)",
            "doa_az_err_deg": "lower better (FOA intensity azimuth error)",
            "doa_el_err_deg": "lower better (FOA intensity elevation error)",
            "dir_energy_ratio_err": "lower better (directional/omni energy ratio error)",
            "ic_corr_err": "lower better (inter-channel correlation error)",
        },
    }
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2))

    with (out_dir / "eval_per_file.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "file"] + metric_keys + ["source"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})

    lines = [f"# VAE reconstruction eval - `{args.tag}`", "",
             f"clips: {len(rows)}  |  recon: `{recon_dir}`", "",
             "| metric | mean | median | min | max |", "|---|---|---|---|---|"]
    for k in metric_keys:
        a = aggregate.get(k) or {}
        if a:
            lines.append(f"| {k} | {a['mean']:.3f} | {a['median']:.3f} | {a['min']:.3f} | {a['max']:.3f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWrote report -> {out_dir}")


if __name__ == "__main__":
    main()
