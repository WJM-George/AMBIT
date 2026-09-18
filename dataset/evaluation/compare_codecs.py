#!/usr/bin/env python3
"""Mono codec comparison: our 4ch FOA VAE (omni W) vs DAC / EnCodec / WavTokenizer.

Fair single-channel comparison on the FOA omni (W) content:
  * DAC / EnCodec / WavTokenizer are mono/stereo full-band codecs that cannot
    represent 4ch FOA, so they encode/decode the W channel only.
  * Our VAE encodes/decodes the full 4ch FOA; we take the W channel of its
    reconstruction for the mono table, and additionally report the FOA spatial
    metrics (where the codecs do not apply).

Test set (categories evaluated with the metric set each deserves):
  * speech (default 10)  : held-out Spatial-LibriSpeech FOA -> PESQ/STOI/MCD/DNSMOS
  * music  (default 5)   : AudioCaps FOA (music-captioned)  -> traditional recon
  * sound  (default 5)   : AudioCaps FOA (non-music)        -> traditional recon
DNSMOS (no-reference) is reported for every category.

Outputs (--out, default ${AMBIT_CKPT_ROOT}/eval_metric/codec_compare):
  compare_codecs_per_file.csv, compare_codecs_summary.json, compare_codecs_summary.md

Run (CPU-safe; all training GPUs may be busy):
  uv run python dataset/evaluation/compare_codecs.py --device cpu
"""
from __future__ import annotations
import os

import argparse
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SAT_ROOT = _HERE.parents[1]
_WT_REPO = _SAT_ROOT / "other_repo_compare" / "WavTokenizer"
for p in (str(_HERE), str(_SAT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_vae_recon import (  # noqa: E402
    _read_4ch, _resample, evaluate_pair, si_sdr, lsd_db, stft_mag_l1, _agg,
)
import speech_metrics as sm  # noqa: E402

REF_SR = 44100  # comparison reference SR = our VAE sample rate

# ---- default data locations (overridable) ---------------------------------
SLS_DIR = os.environ.get("AMBIT_DATA_ROOT", "data") + "/datasets/spatial_librispeech/ambisonics"
SLS_TRAIN_SEED = "spatial_librispeech"
SLS_TRAIN_MAX = 60255
AUDIOCAPS_JSONL = os.environ.get("AMBIT_CACHE_ROOT", "cache/tmp") + "/audiocaps_train.jsonl"
AUDIOCAPS_FOA = os.environ.get("AMBIT_CACHE_ROOT", "cache/tmp") + "/audiocaps_foa/train"
MUSIC_KW = ("music", "song", "guitar", "piano", "violin", "drum", "melody",
            "singing", "orchestra", "trumpet", "flute", "instrument", "choir",
            "harmonica", "accordion", "cello", "saxophone", "banjo")

# Default "ours" VAE arms (name, model-config, ckpt). Overridable via repeated
# --vae-arm NAME CONFIG CKPT. Each arm reconstructs full 4ch and we take W.
# Ckpts must be EMA-unwrapped (clean encoder.*/decoder.* keys) — raw Lightning
# training ckpts do not load via load_4ch_vae (prefix mismatch → random init).
_ABL_ROOT = os.environ.get("AMBIT_CKPT_ROOT", "checkpoints")
DEFAULT_VAE_ARMS = [
    [
        "base_overshoot_900k",
        f"{_ABL_ROOT}/vae_ds1024_z64_hf_overshoot_decay_350k_8gpu/configs/model_hf_overshoot_decay_350k.json",
        f"{_ABL_ROOT}/compareVAE_ckpt/unwrapped_base_overshoot_900k.ckpt",
    ],
    [
        "phase_scm_900k",
        f"{_ABL_ROOT}/vae_abl_phase_scm/configs/model_phase_scm.json",
        f"{_ABL_ROOT}/compareVAE_ckpt/unwrapped_phase_scm_900k.ckpt",
    ],
]

TRAD_KEYS = ["si_sdr_db", "w_si_sdr_db", "lsd_db", "stft_mag_l1"]
SPATIAL_KEYS = ["doa_az_err_deg", "doa_el_err_deg", "dir_energy_ratio_err", "ic_corr_err"]
SPEECH_KEYS = sm.SPEECH_KEYS
HIGHER_BETTER = set(sm.HIGHER_BETTER) | {"si_sdr_db", "w_si_sdr_db"}


# =========================================================== codec wrappers

class OursVAECodec:
    """Our 4ch FOA VAE. `mono_mode` turns it into a capacity diagnostic:
      * None    : encode the real FOA [W,Y,Z,X] (normal operation).
      * 'wwww'  : replicate the omni W into all 4 channels before encoding.
                  The decoder then only has to store ONE unique signal, so its
                  W reconstruction shows the best-case W fidelity our architecture
                  can reach when it is NOT forced to also carry Y/Z/X.
      * 'w000'  : feed [W,0,0,0] (omni + silent directional channels), a valid
                  purely-diffuse FOA field with almost no spatial info to encode.
    In both diagnostic modes the reconstructed W is still scored against the TRUE
    source W, so the numbers are directly comparable to the normal arm and to DAC.
    """

    def __init__(self, config_path, ckpt_path, device, name="ours_vae_W", mono_mode=None):
        from compare_vae_4ch_vs_2stereo import load_4ch_vae, _enc_dec_pad
        self.name = name
        self.mono_mode = mono_mode
        self._enc_dec_pad = _enc_dec_pad
        self.model, self.sr = load_4ch_vae(config_path, ckpt_path, device)
        self.device = device

    def recon4(self, src4: np.ndarray) -> np.ndarray:
        """src4 [4,T] at REF_SR -> recon [4,T] at REF_SR (numpy)."""
        src4 = np.ascontiguousarray(src4)
        if self.mono_mode == "wwww":
            w = src4[0:1]
            src4 = np.ascontiguousarray(np.repeat(w, 4, axis=0))
        elif self.mono_mode == "w000":
            fed = np.zeros_like(src4)
            fed[0] = src4[0]
            src4 = np.ascontiguousarray(fed)
        x = torch.from_numpy(src4).float()
        rec = self._enc_dec_pad(self.model, x, self.device).clamp(-1, 1)
        return rec.numpy()


class DACCodec:
    def __init__(self, ckpt_path, device, tag="dac_44k"):
        import dac
        self.name = tag
        self.model = dac.DAC.load(ckpt_path).eval().to(device)
        self.codec_sr = int(self.model.sample_rate)
        self.device = device

    @torch.no_grad()
    def recon_mono(self, w_ref: np.ndarray, sr: int) -> np.ndarray:
        w = sm.resample_1d(w_ref, sr, self.codec_sr)
        x = torch.from_numpy(w).float().view(1, 1, -1).to(self.device)
        x = self.model.preprocess(x, self.codec_sr)
        z, *_ = self.model.encode(x)
        y = self.model.decode(z).squeeze().float().cpu().numpy()
        y = np.asarray(y).reshape(-1)[: len(w)]
        return sm.resample_1d(y, self.codec_sr, sr)


class EncodecCodec:
    def __init__(self, device, bandwidth=6.0, which="24k"):
        from encodec import EncodecModel
        self.name = f"encodec_{which}_{bandwidth}kbps"
        if which == "24k":
            m = EncodecModel.encodec_model_24khz()
        else:
            m = EncodecModel.encodec_model_48khz()
        m.set_target_bandwidth(bandwidth)
        self.model = m.eval().to(device)
        self.codec_sr = int(m.sample_rate)
        self.channels = m.channels
        self.device = device

    @torch.no_grad()
    def recon_mono(self, w_ref: np.ndarray, sr: int) -> np.ndarray:
        w = sm.resample_1d(w_ref, sr, self.codec_sr)
        x = torch.from_numpy(w).float().view(1, 1, -1)
        if self.channels == 2:
            x = x.repeat(1, 2, 1)
        x = x.to(self.device)
        frames = self.model.encode(x)
        y = self.model.decode(frames)
        y = y.squeeze().float().cpu().numpy()
        if y.ndim > 1:
            y = y.mean(axis=0)
        y = np.asarray(y).reshape(-1)[: len(w)]
        return sm.resample_1d(y, self.codec_sr, sr)


class WavTokenizerCodec:
    name = "wavtokenizer_40tok"

    def __init__(self, config_path, ckpt_path, device):
        if str(_WT_REPO) not in sys.path:
            sys.path.insert(0, str(_WT_REPO))
        from decoder.pretrained import WavTokenizer
        self.model = WavTokenizer.from_pretrained0802(config_path, ckpt_path).eval().to(device)
        self.codec_sr = 24000
        self.device = device
        self.bw = torch.tensor([0], device=device)

    @torch.no_grad()
    def recon_mono(self, w_ref: np.ndarray, sr: int) -> np.ndarray:
        w = sm.resample_1d(w_ref, sr, self.codec_sr)
        x = torch.from_numpy(w).float().view(1, -1).to(self.device)
        feats, _ = self.model.encode_infer(x, bandwidth_id=self.bw)
        y = self.model.decode(feats, bandwidth_id=self.bw)
        y = y.squeeze().float().cpu().numpy().reshape(-1)[: len(w)]
        return sm.resample_1d(y, self.codec_sr, sr)


# =========================================================== test-set building

def _load_ref4(path: str, max_seconds: float) -> Optional[np.ndarray]:
    try:
        raw, s_sr = _read_4ch(path)
    except Exception:  # noqa: BLE001
        return None
    if raw.shape[0] < 4:
        return None
    ref4 = _resample(raw[:4], s_sr, REF_SR)
    if max_seconds and max_seconds > 0:
        ref4 = ref4[:, : int(max_seconds * REF_SR)]
    if ref4.shape[-1] < REF_SR // 2:
        return None
    return np.ascontiguousarray(ref4.astype(np.float32))


def held_out_sls(n: int, seed: int, sls_dir: str = SLS_DIR) -> list[str]:
    from stable_audio_tools.data.dataset import get_audio_filenames
    files = get_audio_filenames(sls_dir)
    if len(files) > SLS_TRAIN_MAX:
        train = set(random.Random(SLS_TRAIN_SEED).sample(files, SLS_TRAIN_MAX))
        pool = sorted(f for f in files if f not in train)
    else:
        pool = sorted(files)
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[: n * 4]  # oversample; some may fail to read


def audiocaps_split(jsonl: str, foa_dir: str, seed: int):
    music, sound = [], []
    fdir = Path(foa_dir)
    with open(jsonl) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            cid = d.get("id") or f"audiocaps_{d.get('audiocap_id')}"
            cap = (d.get("caption") or "").lower()
            foa = fdir / f"{cid}_WYZX_4ch.flac"
            (music if any(k in cap for k in MUSIC_KW) else sound).append((str(foa), cap))
    rng = random.Random(seed)
    rng.shuffle(music)
    rng.shuffle(sound)
    return music, sound


# =========================================================== metric evaluation

def eval_ours(codec: OursVAECodec, ref4: np.ndarray) -> tuple[dict, dict]:
    """Returns (speech/trad metrics on W, spatial metrics on 4ch)."""
    rec4 = codec.recon4(ref4)
    n = min(rec4.shape[-1], ref4.shape[-1])
    rec4, ref4c = rec4[:, :n], ref4[:, :n]
    spatial = {k: v for k, v in evaluate_pair(rec4, ref4c).items() if k in SPATIAL_KEYS}
    return rec4[0], spatial


def align_to_ref(ref: np.ndarray, deg: np.ndarray, max_lag: int = 2048) -> np.ndarray:
    """Compensate a constant coding delay: shift `deg` by the GCC-PHAT best lag
    so waveform metrics (SI-SDR/STOI) are not dominated by codec latency.
    Any residual mismatch (true generative distortion) is preserved."""
    n = min(len(ref), len(deg))
    if n < 64:
        return deg[:n]
    r = ref[:n].astype(np.float32)
    d = deg[:n].astype(np.float32)
    nfft = 1
    while nfft < 2 * n:
        nfft <<= 1
    R = np.fft.rfft(r, nfft)
    D = np.fft.rfft(d, nfft)
    cc = R * np.conj(D)
    cc = cc / (np.abs(cc) + 1e-9)
    corr = np.fft.irfft(cc, nfft)
    corr = np.concatenate((corr[-max_lag:], corr[: max_lag + 1]))
    lag = int(np.argmax(corr)) - max_lag
    if lag > 0:
        d = np.concatenate([np.zeros(lag, np.float32), d])[:n]
    elif lag < 0:
        d = d[-lag:]
        d = np.concatenate([d, np.zeros(n - len(d), np.float32)])
    return d


def mono_fidelity(rec_w: np.ndarray, ref_w: np.ndarray) -> dict:
    n = min(len(rec_w), len(ref_w))
    r, s = rec_w[:n], ref_w[:n]
    return {
        "si_sdr_db": si_sdr(r, s),
        "w_si_sdr_db": si_sdr(r, s),
        "lsd_db": lsd_db(r, s),
        "stft_mag_l1": stft_mag_l1(r, s),
    }


def build_ours_arms(args, device):
    specs = args.vae_arm if args.vae_arm else list(DEFAULT_VAE_ARMS)
    if args.include_800k:
        specs = specs + [["ours_800k_W", args.vae_config, args.vae_ckpt]]
    arms = []
    mono_names = set()
    for name, cfg, ckpt in specs:
        print(f"[ours] loading arm '{name}': {ckpt}")
        arms.append(OursVAECodec(cfg, ckpt, device, name=name))
    for name, cfg, ckpt, mode in (args.vae_mono_arm or []):
        if mode not in ("wwww", "w000"):
            raise SystemExit(f"--vae-mono-arm MODE must be wwww|w000, got {mode!r}")
        print(f"[ours] loading MONO-DIAGNOSTIC arm '{name}' (mode={mode}): {ckpt}")
        arms.append(OursVAECodec(cfg, ckpt, device, name=name, mono_mode=mode))
        mono_names.add(name)
    return arms, mono_names


def build_ext_codecs(args, device):
    ext = {}
    if args.dac_ckpt:
        ext["dac"] = DACCodec(args.dac_ckpt, device, tag="dac_44k")
    try:
        ext["encodec"] = EncodecCodec(device, bandwidth=args.encodec_bw, which="24k")
    except Exception as e:  # noqa: BLE001
        print(f"[encodec] disabled ({e!r})")
    try:
        ext["wavtok"] = WavTokenizerCodec(args.wt_config, args.wt_ckpt, device)
    except Exception as e:  # noqa: BLE001
        print(f"[wavtokenizer] disabled ({e!r})")
    return ext


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vae-config", default="stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae_ds1024_z64.json",
                    help="config for the optional --include-800k reference arm")
    ap.add_argument("--vae-ckpt", default=os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/compareVAE_ckpt/unwrapped_ds1024_z64_800k.ckpt",
                    help="ckpt for the optional --include-800k reference arm")
    ap.add_argument("--vae-arm", nargs=3, action="append", metavar=("NAME", "CONFIG", "CKPT"),
                    help="Add an 'ours' VAE arm. Repeatable. Defaults to the two 900k arms.")
    ap.add_argument("--vae-mono-arm", nargs=4, action="append",
                    metavar=("NAME", "CONFIG", "CKPT", "MODE"),
                    help="Add a capacity-diagnostic 'ours' arm that feeds a degenerate "
                         "input before encoding. MODE=wwww replicates W to all 4 channels; "
                         "MODE=w000 feeds [W,0,0,0]. W recon is still scored vs the true W. "
                         "Repeatable. Excluded from the spatial table.")
    ap.add_argument("--include-800k", action="store_true",
                    help="Also include the unwrapped 800k VAE (--vae-config/--vae-ckpt) as a reference arm.")
    ap.add_argument("--dac-ckpt", default=os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/compareVAE_ckpt/dac_44khz.pth")
    ap.add_argument("--encodec-bw", type=float, default=6.0)
    ap.add_argument("--wt-config", default=str(_WT_REPO / "configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"))
    ap.add_argument("--wt-ckpt", default=os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/compareVAE_ckpt/wavtokenizer_large_unify_600_24k.ckpt")
    ap.add_argument("--dnsmos-dir", default=sm.DNSMOS_DIR_DEFAULT)
    ap.add_argument("--speech-num", type=int, default=10)
    ap.add_argument("--music-num", type=int, default=5)
    ap.add_argument("--sound-num", type=int, default=5)
    ap.add_argument("--audiocaps-jsonl", default=AUDIOCAPS_JSONL)
    ap.add_argument("--audiocaps-foa", default=AUDIOCAPS_FOA)
    ap.add_argument("--sls-dir", default=SLS_DIR)
    ap.add_argument("--max-seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/eval_metric/codec_compare"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stats", choices=["median", "meanstd"], default="median",
                    help="table statistic; 'meanstd' for the expanded (hundreds-of-clips) report")
    args = ap.parse_args()

    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    print("[metrics] backends:", sm.backend_status())
    dnsmos = None
    try:
        dnsmos = sm.DNSMOS(args.dnsmos_dir)
        print("[dnsmos] loaded")
    except Exception as e:  # noqa: BLE001
        print(f"[dnsmos] disabled ({e!r})")

    print("[codecs] loading ...")
    ours_arms, mono_arm_names = build_ours_arms(args, device)
    ext = build_ext_codecs(args, device)
    ours_names = [c.name for c in ours_arms]
    print(f"[codecs] ours arms: {ours_names}  |  ext: {[c.name for c in ext.values()]}")

    # ---- assemble test set -------------------------------------------------
    tasks: list[tuple[str, str]] = []  # (category, path)
    for f in held_out_sls(args.speech_num, args.seed, args.sls_dir):
        tasks.append(("speech", f))
    music, sound = audiocaps_split(args.audiocaps_jsonl, args.audiocaps_foa, args.seed)
    tasks += [("music", p) for p, _ in music]
    tasks += [("sound", p) for p, _ in sound]

    need = {"speech": args.speech_num, "music": args.music_num, "sound": args.sound_num}
    got = {"speech": 0, "music": 0, "sound": 0}

    rows: list[dict] = []

    for cat, path in tasks:
        if got[cat] >= need[cat]:
            continue
        ref4 = _load_ref4(path, args.max_seconds)
        if ref4 is None:
            continue
        ref_w = ref4[0]
        clip = Path(path).stem

        # ours arms (each: 4ch -> W + its own spatial metrics)
        recons: dict[str, np.ndarray] = {}
        spatial_by_arm: dict[str, dict] = {}
        failed_arm = False
        for arm in ours_arms:
            try:
                arm_w, spatial = eval_ours(arm, ref4)
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {clip} arm {arm.name} failed: {e!r}")
                failed_arm = True
                break
            recons[arm.name] = arm_w
            spatial_by_arm[arm.name] = spatial
        if failed_arm:
            continue

        for _, c in ext.items():
            try:
                recons[c.name] = c.recon_mono(ref_w, REF_SR)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {clip} {c.name} failed: {e!r}")

        for cname, rec_w in recons.items():
            rec_w = align_to_ref(ref_w, rec_w)
            m = {"category": cat, "clip": clip, "codec": cname, "source": path}
            m.update(mono_fidelity(rec_w, ref_w))
            m.update(sm.compute_speech_metrics(ref_w, rec_w, REF_SR, dnsmos=dnsmos,
                                               with_mcd=(cat == "speech")))
            if cname in spatial_by_arm:
                m.update(spatial_by_arm[cname])
            rows.append(m)

        got[cat] += 1
        done = sum(got.values())
        print(f"[{done}] {cat}/{clip} done  (speech {got['speech']}/{need['speech']}, "
              f"music {got['music']}/{need['music']}, sound {got['sound']}/{need['sound']})")
        if all(got[c] >= need[c] for c in need):
            break

    if not rows:
        raise RuntimeError("No clips evaluated. Check data paths.")

    ext_names = [c.name for c in ext.values()]
    write_outputs(args, rows, got, ours_names, ext_names, dnsmos is not None,
                  mono_arm_names=mono_arm_names)


# =========================================================== output writing

def _agg_cat(rows, category, codec, key):
    vals = [r[key] for r in rows
            if r["category"] == category and r["codec"] == codec
            and key in r and isinstance(r[key], (int, float)) and r[key] == r[key]]
    if not vals:
        return None
    return {"mean": float(statistics.mean(vals)), "median": float(statistics.median(vals)),
            "std": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
            "n": len(vals)}


def _fmt(x):
    """Median (kept for the canvas and backward-compatible tables)."""
    if x is None:
        return "n/a"
    v = x["median"]
    return f"{v:.3f}" if abs(v) >= 0.1 else f"{v:.4g}"


def _fmt_ms(x):
    """mean +/- std for the expanded (n>=~hundreds) report."""
    if x is None:
        return "n/a"
    m, s = x["mean"], x.get("std", 0.0)
    if abs(m) >= 0.1:
        return f"{m:.3f}\u00b1{s:.3f}"
    return f"{m:.4g}\u00b1{s:.2g}"


def write_outputs(args, rows, got, ours_names, ext_names, have_dnsmos, mono_arm_names=None):
    mono_arm_names = set(mono_arm_names or ())
    all_names = list(ours_names) + list(ext_names)
    codec_names = all_names
    stats_mode = getattr(args, "stats", "median")
    fmt = _fmt_ms if stats_mode == "meanstd" else _fmt
    stat_label = "mean+/-std" if stats_mode == "meanstd" else "median"

    # ---- JSON aggregate ----
    agg = {}
    for cat in ("speech", "music", "sound"):
        agg[cat] = {}
        for cname in all_names:
            agg[cat][cname] = {k: _agg_cat(rows, cat, cname, k)
                               for k in (SPEECH_KEYS + TRAD_KEYS + SPATIAL_KEYS)}
    report = {
        "reference_sr": REF_SR,
        "counts": got,
        "codecs": codec_names,
        "have_dnsmos": have_dnsmos,
        "max_seconds": args.max_seconds,
        "note": "codecs run on FOA omni (W) only; ours reports 4ch spatial too",
        "aggregate": agg,
    }
    (args.out / "compare_codecs_summary.json").write_text(json.dumps(report, indent=2))

    # ---- per-file CSV ----
    cols = (["category", "clip", "codec", "source"] + TRAD_KEYS + SPEECH_KEYS + SPATIAL_KEYS)
    with (args.out / "compare_codecs_per_file.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    # ---- markdown ----
    L = ["# Mono codec comparison — our FOA VAE (omni W) vs DAC / EnCodec / WavTokenizer",
         "",
         f"reference SR **{REF_SR} Hz**  |  clips: speech {got['speech']}, music {got['music']}, "
         f"sound {got['sound']}  |  max {args.max_seconds:.0f}s/clip  |  values = **{stat_label}** (full mean/median/std/n in JSON)",
         "",
         "Codecs reconstruct the FOA **omni (W)** channel only (they are not FOA-capable); "
         "our VAE reconstructs full 4ch and we take W. DNSMOS is no-reference.",
         ""]

    # speech table
    L += ["## Speech — PESQ / STOI / MCD / DNSMOS", "",
          "| codec | PESQ_wb (up) | STOI (up) | ESTOI (up) | MCD dB (down) | DNSMOS SIG (up) | DNSMOS BAK (up) | DNSMOS OVRL (up) | P808 (up) | SI-SDR dB (up) | LSD dB (down) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cname in all_names:
        g = lambda k: fmt(_agg_cat(rows, "speech", cname, k))  # noqa: E731
        L.append(f"| {cname} | {g('pesq_wb')} | {g('stoi')} | {g('estoi')} | {g('mcd')} | "
                 f"{g('dnsmos_sig')} | {g('dnsmos_bak')} | {g('dnsmos_ovrl')} | {g('dnsmos_p808')} | "
                 f"{g('si_sdr_db')} | {g('lsd_db')} |")

    # music/sound tables (traditional recon + DNSMOS no-ref)
    for cat in ("music", "sound"):
        L += ["", f"## {cat.capitalize()} — traditional recon fidelity (+ DNSMOS no-ref)", "",
              "| codec | SI-SDR dB (up) | LSD dB (down) | STFT-mag L1 (down) | DNSMOS OVRL (up) | P808 (up) |",
              "|---|---|---|---|---|---|"]
        for cname in all_names:
            g = lambda k: fmt(_agg_cat(rows, cat, cname, k))  # noqa: E731
            L.append(f"| {cname} | {g('si_sdr_db')} | {g('lsd_db')} | {g('stft_mag_l1')} | "
                     f"{g('dnsmos_ovrl')} | {g('dnsmos_p808')} |")

    # spatial (ours arms only — codecs cannot represent FOA)
    L += ["", "## Spatial (our VAE arms only — codecs cannot represent FOA)", "",
          "| model | category | DoA az err deg (down) | DoA el err deg (down) | dir-energy-ratio err (down) | IC-corr err (down) |",
          "|---|---|---|---|---|---|"]
    for arm in ours_names:
        if arm in mono_arm_names:
            continue  # degenerate input -> spatial metrics meaningless
        for cat in ("speech", "music", "sound"):
            g = lambda k: fmt(_agg_cat(rows, cat, arm, k))  # noqa: E731
            L.append(f"| {arm} | {cat} | {g('doa_az_err_deg')} | {g('doa_el_err_deg')} | "
                     f"{g('dir_energy_ratio_err')} | {g('ic_corr_err')} |")

    L += ["", "_Caveat: EnCodec/WavTokenizer operate at 24 kHz (band-limited to 12 kHz); "
          "DAC and our VAE are full-band at 44.1 kHz. Speech metrics on music/sound are "
          "omitted as less meaningful; traditional recon fidelity is shown instead._", ""]

    (args.out / "compare_codecs_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nWrote -> {args.out}/compare_codecs_summary.{{md,json}} + per_file.csv")


if __name__ == "__main__":
    main()
