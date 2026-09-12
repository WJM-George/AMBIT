#!/usr/bin/env python3
"""Speech reconstruction metrics: PESQ / STOI / MCD / DNSMOS.

Mono, single-channel speech quality metrics used to compare our VAE (omni W
channel) against off-the-shelf codecs (DAC / EnCodec / WavTokenizer). Every
metric is guarded: if its backend is missing or the call fails, the metric is
returned as NaN instead of raising, matching the ``_HAVE_PESQ`` pattern already
used in ``compare_vae_heldout.py``.

Metric directions (for reporting):
  PESQ  (wideband, [-0.5, 4.5])   higher better
  STOI  ([0, 1])                  higher better
  ESTOI ([0, 1])                  higher better
  MCD   (dB, mel-cepstral dist)   lower  better
  DNSMOS SIG / BAK / OVRL / P808  higher better  (no-reference, P.835/P.808)

DNSMOS uses the Microsoft DNS-Challenge ONNX models (P.835 sig_bak_ovr.onnx +
P.808 model_v8.onnx). Point ``--dnsmos-dir`` / ``DNSMOS_DIR`` at the folder that
holds them (default: ${AMBIT_CKPT_ROOT}/compareVAE_ckpt/dnsmos).
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

EPS = 1e-9
DNSMOS_DIR_DEFAULT = os.environ.get(
    "DNSMOS_DIR", os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/compareVAE_ckpt/dnsmos"
)

# ---------------------------------------------------------------- optional deps
try:
    from pesq import pesq as _pesq
    _HAVE_PESQ = True
except Exception:  # noqa: BLE001
    _HAVE_PESQ = False

try:
    from pystoi import stoi as _stoi
    _HAVE_STOI = True
except Exception:  # noqa: BLE001
    _HAVE_STOI = False

try:
    from scipy.signal import resample_poly as _resample_poly
    _HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    _HAVE_SCIPY = False

try:
    import librosa as _librosa
    _HAVE_LIBROSA = True
except Exception:  # noqa: BLE001
    _HAVE_LIBROSA = False


# --------------------------------------------------------------------- resample

def resample_1d(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if sr_in == sr_out:
        return x
    if _HAVE_SCIPY:
        g = math.gcd(int(sr_in), int(sr_out))
        return _resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    n = int(round(len(x) * sr_out / sr_in))
    xo = np.linspace(0.0, 1.0, len(x), endpoint=False)
    xn = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(xn, xo, x).astype(np.float32)


def _match_len(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


# ------------------------------------------------------------------------- PESQ

def pesq_wb(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """Wideband PESQ (ITU-T P.862.2) at 16 kHz."""
    if not _HAVE_PESQ:
        return float("nan")
    r = resample_1d(ref, sr, 16000)
    d = resample_1d(deg, sr, 16000)
    r, d = _match_len(r, d)
    if len(r) < 16000 // 2:
        return float("nan")
    try:
        return float(_pesq(16000, r, d, "wb"))
    except Exception:  # noqa: BLE001
        return float("nan")


def pesq_nb(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """Narrowband PESQ (ITU-T P.862) at 8 kHz."""
    if not _HAVE_PESQ:
        return float("nan")
    r = resample_1d(ref, sr, 8000)
    d = resample_1d(deg, sr, 8000)
    r, d = _match_len(r, d)
    if len(r) < 8000 // 2:
        return float("nan")
    try:
        return float(_pesq(8000, r, d, "nb"))
    except Exception:  # noqa: BLE001
        return float("nan")


# ------------------------------------------------------------------------- STOI

def stoi(ref: np.ndarray, deg: np.ndarray, sr: int, extended: bool = False) -> float:
    if not _HAVE_STOI:
        return float("nan")
    r, d = _match_len(np.asarray(ref, np.float32).reshape(-1),
                      np.asarray(deg, np.float32).reshape(-1))
    if len(r) < sr // 2:
        return float("nan")
    try:
        return float(_stoi(r, d, sr, extended=extended))
    except Exception:  # noqa: BLE001
        return float("nan")


def estoi(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    return stoi(ref, deg, sr, extended=True)


# -------------------------------------------------------------------------- MCD

_MCD_BACKEND = None  # cache the pymcd instance


def mcd_dtw(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """Mel-cepstral distortion (dB) with DTW alignment via pymcd (WORLD/pysptk).

    pymcd operates on file paths, so ref/deg are written to temp 16 kHz wavs.
    Lower is better; typical clean-vs-codec range ~2-8 dB.
    """
    global _MCD_BACKEND
    try:
        import soundfile as sf
        if _MCD_BACKEND is None:
            from pymcd.mcd import Calculate_MCD
            _MCD_BACKEND = Calculate_MCD(MCD_mode="dtw")
    except Exception:  # noqa: BLE001
        return float("nan")

    r = resample_1d(ref, sr, 16000)
    d = resample_1d(deg, sr, 16000)
    if len(r) < 16000 // 4 or len(d) < 16000 // 4:
        return float("nan")
    tmp_r = tmp_d = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fr:
            tmp_r = fr.name
            sf.write(tmp_r, r, 16000)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fd:
            tmp_d = fd.name
            sf.write(tmp_d, d, 16000)
        return float(_MCD_BACKEND.calculate_mcd(tmp_r, tmp_d))
    except Exception:  # noqa: BLE001
        return float("nan")
    finally:
        for p in (tmp_r, tmp_d):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


# ----------------------------------------------------------------------- DNSMOS

class DNSMOS:
    """No-reference DNSMOS (P.835 SIG/BAK/OVRL + P.808) via ONNX Runtime.

    Faithful port of the Microsoft DNS-Challenge ``dnsmos_local.py`` inference:
    9.01 s segments at 16 kHz, per-segment SIG/BAK/OVR from sig_bak_ovr.onnx and
    P.808 MOS from model_v8.onnx (log-mel input), averaged over hops, then mapped
    through the published non-personalized polynomial fit.
    """

    SR = 16000
    INPUT_LENGTH = 9.01

    def __init__(self, model_dir: str = DNSMOS_DIR_DEFAULT):
        import onnxruntime as ort
        p835 = Path(model_dir) / "sig_bak_ovr.onnx"
        p808 = Path(model_dir) / "model_v8.onnx"
        if not p835.exists() or not p808.exists():
            raise FileNotFoundError(f"DNSMOS onnx not found in {model_dir}")
        so = ort.SessionOptions()
        self.sess = ort.InferenceSession(str(p835), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.p808 = ort.InferenceSession(str(p808), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.p808_in = self.p808.get_inputs()[0].name

    @staticmethod
    def _melspec(audio: np.ndarray, sr: int = 16000, n_mels: int = 120,
                 frame_size: int = 320, hop: int = 160) -> np.ndarray:
        mel = _librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop, n_mels=n_mels)
        mel = (_librosa.power_to_db(mel, ref=np.max) + 40.0) / 40.0
        return mel.T.astype(np.float32)

    @staticmethod
    def _polyfit(sig: float, bak: float, ovr: float) -> tuple[float, float, float]:
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        return float(p_sig(sig)), float(p_bak(bak)), float(p_ovr(ovr))

    def __call__(self, deg: np.ndarray, sr: int) -> dict:
        if not _HAVE_LIBROSA:
            return {"dnsmos_sig": float("nan"), "dnsmos_bak": float("nan"),
                    "dnsmos_ovrl": float("nan"), "dnsmos_p808": float("nan")}
        audio = resample_1d(deg, sr, self.SR)
        n_need = int(self.INPUT_LENGTH * self.SR)
        while len(audio) < n_need:
            audio = np.concatenate([audio, audio])
        hop_samps = self.SR
        num_hops = int(np.floor(len(audio) / self.SR) - self.INPUT_LENGTH) + 1
        num_hops = max(num_hops, 1)
        sigs, baks, ovrs, p808s = [], [], [], []
        for idx in range(num_hops):
            start = int(idx * hop_samps)
            seg = audio[start:start + n_need]
            if len(seg) < n_need:
                continue
            feats = seg.astype(np.float32)[np.newaxis, :]
            p808_feat = self._melspec(seg[:-160])[np.newaxis, :, :]
            try:
                sig_raw, bak_raw, ovr_raw = self.sess.run(
                    None, {self.in_name: feats})[0][0]
                p808_mos = float(self.p808.run(
                    None, {self.p808_in: p808_feat})[0][0][0])
            except Exception:  # noqa: BLE001
                continue
            s, b, o = self._polyfit(float(sig_raw), float(bak_raw), float(ovr_raw))
            sigs.append(s); baks.append(b); ovrs.append(o); p808s.append(p808_mos)
        if not ovrs:
            return {"dnsmos_sig": float("nan"), "dnsmos_bak": float("nan"),
                    "dnsmos_ovrl": float("nan"), "dnsmos_p808": float("nan")}
        return {
            "dnsmos_sig": float(np.mean(sigs)),
            "dnsmos_bak": float(np.mean(baks)),
            "dnsmos_ovrl": float(np.mean(ovrs)),
            "dnsmos_p808": float(np.mean(p808s)),
        }


# ------------------------------------------------------------------ aggregate API

SPEECH_KEYS = ["pesq_wb", "stoi", "estoi", "mcd",
               "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"]
HIGHER_BETTER = {"pesq_wb", "stoi", "estoi",
                 "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"}


def compute_speech_metrics(ref: np.ndarray, deg: np.ndarray, sr: int,
                           dnsmos: Optional[DNSMOS] = None,
                           with_mcd: bool = True) -> dict:
    """Full-reference (PESQ/STOI/MCD) + no-reference (DNSMOS on deg) metrics."""
    ref = np.asarray(ref, np.float32).reshape(-1)
    deg = np.asarray(deg, np.float32).reshape(-1)
    out = {
        "pesq_wb": pesq_wb(ref, deg, sr),
        "stoi": stoi(ref, deg, sr, extended=False),
        "estoi": stoi(ref, deg, sr, extended=True),
        "mcd": mcd_dtw(ref, deg, sr) if with_mcd else float("nan"),
    }
    if dnsmos is not None:
        out.update(dnsmos(deg, sr))
    else:
        out.update({k: float("nan") for k in
                    ["dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"]})
    return out


def backend_status() -> dict:
    return {"pesq": _HAVE_PESQ, "pystoi": _HAVE_STOI,
            "scipy": _HAVE_SCIPY, "librosa": _HAVE_LIBROSA}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Self-test speech metrics on random / file input")
    ap.add_argument("--dnsmos-dir", default=DNSMOS_DIR_DEFAULT)
    args = ap.parse_args()
    print("backends:", backend_status())
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(32000).astype(np.float32) * 0.1
    deg = ref + 0.02 * rng.standard_normal(32000).astype(np.float32)
    dn = None
    try:
        dn = DNSMOS(args.dnsmos_dir)
    except Exception as e:  # noqa: BLE001
        print("DNSMOS unavailable:", repr(e))
    print(compute_speech_metrics(ref, deg, 16000, dnsmos=dn))
