#!/usr/bin/env python
"""Paired objective evaluation for generated V2A demos.

This compares generated 4ch FOA WAVs against the corresponding dynamic10 GT
audio referenced by generation_results.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import torch
import torchaudio

EPS = 1e-9


def read_audio(path: str) -> tuple[torch.Tensor, int]:
    audio, sr = torchaudio.load(path)  # [C, T]
    audio = audio.float()
    if audio.shape[0] > 4:
        audio = audio[:4]
    if audio.shape[0] < 4:
        audio = torch.nn.functional.pad(audio, (0, 0, 0, 4 - audio.shape[0]))
    return audio, int(sr)


def resample(audio: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    if sr_in == sr_out:
        return audio
    return torchaudio.functional.resample(audio, sr_in, sr_out)


def align(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


def rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x**2, dim=-1).clamp_min(EPS))


def rms_db(x: torch.Tensor) -> float:
    return float(20.0 * torch.log10(torch.sqrt(torch.mean(x**2)).clamp_min(EPS)))


def rms_match(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    gain = torch.sqrt(torch.mean(ref**2).clamp_min(EPS) / torch.mean(est**2).clamp_min(EPS))
    return est * gain


def si_sdr_1d(est: torch.Tensor, ref: torch.Tensor) -> float:
    est = est - est.mean()
    ref = ref - ref.mean()
    ref_energy = torch.sum(ref**2).clamp_min(EPS)
    proj = torch.sum(est * ref) / ref_energy * ref
    noise = est - proj
    val = 10.0 * torch.log10(torch.sum(proj**2).clamp_min(EPS) / torch.sum(noise**2).clamp_min(EPS))
    return float(val)


def si_sdr(est: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    vals = [si_sdr_1d(est[c], ref[c]) for c in range(min(est.shape[0], ref.shape[0], 4))]
    return float(statistics.mean(vals)), vals[0]


def same_channel_corr(est: torch.Tensor, ref: torch.Tensor) -> float:
    vals = []
    for c in range(min(est.shape[0], ref.shape[0], 4)):
        a = est[c] - est[c].mean()
        b = ref[c] - ref[c].mean()
        vals.append(float(torch.sum(a * b) / torch.sqrt(torch.sum(a**2) * torch.sum(b**2)).clamp_min(EPS)))
    return float(statistics.mean(vals))


def stft_mag(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    win = torch.hann_window(n_fft, device=x.device)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
    return spec.abs().clamp_min(EPS)


def lsd_db(est: torch.Tensor, ref: torch.Tensor, n_fft: int = 2048, hop: int = 512) -> float:
    est = rms_match(est, ref)
    vals = []
    for c in range(min(est.shape[0], ref.shape[0], 4)):
        e = stft_mag(est[c], n_fft, hop)
        r = stft_mag(ref[c], n_fft, hop)
        m = min(e.shape[-1], r.shape[-1])
        diff = 20.0 * torch.log10(r[..., :m]) - 20.0 * torch.log10(e[..., :m])
        vals.append(float(torch.sqrt(torch.mean(diff**2))))
    return float(statistics.mean(vals))


def mrstft_logmag_l1(est: torch.Tensor, ref: torch.Tensor) -> float:
    est = rms_match(est, ref)
    vals = []
    for n_fft in (512, 1024, 2048, 4096):
        hop = n_fft // 4
        ch_vals = []
        for c in range(min(est.shape[0], ref.shape[0], 4)):
            e = stft_mag(est[c], n_fft, hop)
            r = stft_mag(ref[c], n_fft, hop)
            m = min(e.shape[-1], r.shape[-1])
            ch_vals.append(float(torch.mean(torch.abs(torch.log(e[..., :m]) - torch.log(r[..., :m])))))
        vals.append(float(statistics.mean(ch_vals)))
    return float(statistics.mean(vals))


def mel_metrics(est: torch.Tensor, ref: torch.Tensor, sr: int) -> tuple[float, float]:
    est_w = rms_match(est[0], ref[0])
    ref_w = ref[0]
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        f_min=20,
        f_max=min(20000, sr // 2),
        power=2.0,
        norm="slaney",
        mel_scale="slaney",
    )
    e = torch.log10(mel(est_w).clamp_min(1e-8))
    r = torch.log10(mel(ref_w).clamp_min(1e-8))
    m = min(e.shape[-1], r.shape[-1])
    e = e[..., :m]
    r = r[..., :m]
    l1_db = float(torch.mean(torch.abs(10.0 * (e - r))))
    e_flat = (e - e.mean()).flatten()
    r_flat = (r - r.mean()).flatten()
    cos = float(torch.sum(e_flat * r_flat) / torch.sqrt(torch.sum(e_flat**2) * torch.sum(r_flat**2)).clamp_min(EPS))
    return l1_db, cos


def envelope_corr(est: torch.Tensor, ref: torch.Tensor, frame: int = 2048, hop: int = 512) -> float:
    def env(x: torch.Tensor) -> torch.Tensor:
        frames = x.unfold(-1, frame, hop)
        return torch.sqrt(torch.mean(frames**2, dim=-1).clamp_min(EPS))

    e = env(est[0])
    r = env(ref[0])
    m = min(e.numel(), r.numel())
    e = e[:m] - e[:m].mean()
    r = r[:m] - r[:m].mean()
    return float(torch.sum(e * r) / torch.sqrt(torch.sum(e**2) * torch.sum(r**2)).clamp_min(EPS))


def foa_intensity_doa(foa: torch.Tensor) -> tuple[float, float]:
    w, y, z, x = foa[0], foa[1], foa[2], foa[3]
    ix = float(torch.mean(w * x))
    iy = float(torch.mean(w * y))
    iz = float(torch.mean(w * z))
    az = math.degrees(math.atan2(iy, ix))
    el = math.degrees(math.atan2(iz, math.hypot(ix, iy) + EPS))
    return az, el


def ang_err(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def directional_energy_ratio(foa: torch.Tensor) -> float:
    w_e = float(torch.mean(foa[0] ** 2)) + EPS
    dir_e = float(torch.mean(foa[1] ** 2 + foa[2] ** 2 + foa[3] ** 2))
    return dir_e / w_e


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float(torch.sum(a * b) / torch.sqrt(torch.sum(a**2) * torch.sum(b**2)).clamp_min(EPS))


def inter_channel_corr_err(est: torch.Tensor, ref: torch.Tensor) -> float:
    pairs = [(0, 1), (0, 3), (0, 2), (1, 3)]
    errs = [abs(corr(est[i], est[j]) - corr(ref[i], ref[j])) for i, j in pairs]
    return float(statistics.mean(errs))


def gt_path_from_entry(entry: dict) -> str:
    latent_json = Path(entry["latent_json"])
    with latent_json.open("r", encoding="utf-8") as f:
        md = json.load(f)
    return md.get("foa_path") or md.get("audio_path") or md.get("path")


def evaluate_entry(entry: dict) -> dict:
    gen_path = entry["generated_wav"]
    gt_path = gt_path_from_entry(entry)
    gen, gen_sr = read_audio(gen_path)
    gt, gt_sr = read_audio(gt_path)
    gt = resample(gt, gt_sr, gen_sr)
    gen, gt = align(gen, gt)

    si, w_si = si_sdr(gen, gt)
    mel_l1, mel_cos = mel_metrics(gen, gt, gen_sr)
    gen_doa = foa_intensity_doa(gen)
    gt_doa = foa_intensity_doa(gt)
    return {
        "id": entry["id"],
        "generated_wav": gen_path,
        "gt_wav": gt_path,
        "sample_rate": gen_sr,
        "duration_sec": gen.shape[-1] / gen_sr,
        "si_sdr_db": si,
        "w_si_sdr_db": w_si,
        "same_channel_corr": same_channel_corr(gen, gt),
        "rms_gen_db": rms_db(gen),
        "rms_gt_db": rms_db(gt),
        "rms_delta_db": rms_db(gen) - rms_db(gt),
        "lsd_db": lsd_db(gen, gt),
        "mrstft_logmag_l1": mrstft_logmag_l1(gen, gt),
        "mel_l1_db": mel_l1,
        "mel_cosine": mel_cos,
        "w_envelope_corr": envelope_corr(gen, gt),
        "doa_az_err_deg": ang_err(gen_doa[0], gt_doa[0]),
        "doa_el_err_deg": ang_err(gen_doa[1], gt_doa[1]),
        "dir_energy_ratio_gen": directional_energy_ratio(gen),
        "dir_energy_ratio_gt": directional_energy_ratio(gt),
        "dir_energy_ratio_err": abs(directional_energy_ratio(gen) - directional_energy_ratio(gt)),
        "ic_corr_err": inter_channel_corr_err(gen, gt),
    }


def agg(rows: list[dict], key: str) -> dict:
    vals = [float(r[key]) for r in rows if key in r and math.isfinite(float(r[key]))]
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
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="/mnt/sdc/video_demos/generation_results.jsonl")
    p.add_argument("--out-dir", default="/mnt/sdc/video_demos/eval")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = [json.loads(line) for line in Path(args.results).open("r", encoding="utf-8") if line.strip()]
    rows = []
    for i, entry in enumerate(entries, 1):
        row = evaluate_entry(entry)
        rows.append(row)
        print(
            f"[{i:02d}/{len(entries)}] {row['id']} "
            f"SI-SDR={row['si_sdr_db']:.2f}dB W={row['w_si_sdr_db']:.2f}dB "
            f"mel_cos={row['mel_cosine']:.3f} env={row['w_envelope_corr']:.3f} "
            f"DoAaz={row['doa_az_err_deg']:.1f}deg"
        )

    metric_keys = [
        "si_sdr_db",
        "w_si_sdr_db",
        "same_channel_corr",
        "rms_delta_db",
        "lsd_db",
        "mrstft_logmag_l1",
        "mel_l1_db",
        "mel_cosine",
        "w_envelope_corr",
        "doa_az_err_deg",
        "doa_el_err_deg",
        "dir_energy_ratio_err",
        "ic_corr_err",
    ]
    aggregate = {key: agg(rows, key) for key in metric_keys}
    report = {
        "n": len(rows),
        "results": str(Path(args.results).resolve()),
        "aggregate": aggregate,
        "metric_help": {
            "si_sdr_db": "higher better; strict waveform match, usually harsh for generative V2A",
            "w_si_sdr_db": "higher better; strict waveform match on FOA W/omni channel",
            "same_channel_corr": "higher better; waveform correlation by matching channels",
            "rms_delta_db": "closer to 0 better; generated RMS minus GT RMS",
            "lsd_db": "lower better; RMS-matched log spectral distance",
            "mrstft_logmag_l1": "lower better; RMS-matched multi-resolution log-STFT L1",
            "mel_l1_db": "lower better; RMS-matched W-channel log-mel L1 in dB",
            "mel_cosine": "higher better; W-channel log-mel pattern similarity",
            "w_envelope_corr": "higher better; W-channel RMS envelope correlation",
            "doa_az_err_deg": "lower better; broadband FOA active-intensity azimuth error",
            "doa_el_err_deg": "lower better; broadband FOA active-intensity elevation error",
            "dir_energy_ratio_err": "lower better; FOA directional/omni energy ratio difference",
            "ic_corr_err": "lower better; inter-channel correlation image error",
        },
    }
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out_dir / "eval_per_file.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "generated_wav", "gt_wav", "sample_rate", "duration_sec"] + metric_keys
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    lines = [
        "# V2A Demo Paired Evaluation",
        "",
        f"clips: {len(rows)}",
        "",
        "| metric | mean | median | min | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in metric_keys:
        a = aggregate[key]
        lines.append(f"| {key} | {a['mean']:.4f} | {a['median']:.4f} | {a['min']:.4f} | {a['max']:.4f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
