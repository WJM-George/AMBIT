#!/usr/bin/env python3
"""Standalone PSELDNets (ACCDOA-HTSAT) inference wrapper for a DoA metric.

Loads the pre-trained ACCDOA-HTSAT checkpoint from the PSELDNets repo and exposes
``PSELDNetsScorer.doa(foa[4,T], sr) -> (az_deg, el_deg)`` for a single dominant
source. Bypasses hydra / lightning / h5: builds the HTSAT model directly, loads
the state dict, runs the repo's LogmelIV feature extractor, and decodes ACCDOA.

Channel order: VALIDATED against SLS ground truth (see _selftest) -- PSELDNets was
trained on ACN/SN3D [W, Y, Z, X], so we feed our FOA as-is, channel_order=(0,1,2,3)
(median az error vs GT = 6.8 deg; the [0,3,1,2] permutation gives 86.5 deg = wrong).

ACCDOA convention (matches SLS GT): az>0 = left, 0 = front; el>0 = up.
"""
from __future__ import annotations
import os

import math
import sys
from pathlib import Path

import numpy as np
import torch

PSELD_ROOT = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/seld_tools/PSELDNets")
PSELD_SRC = PSELD_ROOT / "src"
DEFAULT_CKPT = PSELD_ROOT / "ckpts_dl" / "model" / "ACCDOA-HTSAT-0.566.ckpt"
NUM_CLASSES = 170
IN_CHANNELS = 7

_DATA_CFG = dict(audio_type="foa", audio_feature="logmelIV", sample_rate=24000,
                 nfft=1024, n_mels=64, hoplen=240, window="hann")
_HTSAT_KWARGS = dict(spec_size=256, patch_size=4, patch_stride=[4, 4], embed_dim=96,
                     depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32], window_size=8,
                     mlp_ratio=4, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
                     drop_path_rate=0.1, ape=False, patch_norm=True, norm_before_mlp="ln")


class DotDict(dict):
    """dict that also supports attribute access (and nested wrapping)."""
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return DotDict(v) if isinstance(v, dict) else v


def _build_cfg():
    return DotDict(data=DotDict(_DATA_CFG), adapt=DotDict(method="none"))


def _install_lightning_stubs():
    """PSELDNets' models/__init__ pulls utils.utilities -> lightning at import time.
    We only need the HTSAT model + feature extractor, never the Trainer/loggers, so
    inject minimal stub modules instead of installing the heavy `lightning` package
    (which could clash with our pytorch_lightning in the training env)."""
    import types
    if "lightning" in sys.modules:
        return

    def passthrough(f=None, *a, **k):
        return f if callable(f) else (lambda g: g)

    specs = {
        "lightning": {"Callback": type("Callback", (), {}), "Trainer": type("Trainer", (), {}),
                      "LightningModule": type("LightningModule", (), {}),
                      "LightningDataModule": type("LightningDataModule", (), {})},
        "lightning.pytorch": {},
        "lightning.pytorch.loggers": {"Logger": type("Logger", (), {})},
        "lightning.pytorch.utilities": {"rank_zero_only": passthrough},
        "lightning.pytorch.callbacks": {"Callback": type("Callback", (), {})},
    }
    for name, attrs in specs.items():
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m


class PSELDNetsScorer:
    def __init__(self, ckpt_path: Path = DEFAULT_CKPT, device=None, channel_order=(0, 1, 2, 3),
                 sed_threshold: float = 0.5):
        if str(PSELD_SRC) not in sys.path:
            sys.path.insert(0, str(PSELD_SRC))
        _install_lightning_stubs()              # avoid heavy lightning dep at import time
        from models.accdoa import HTSAT          # noqa: E402  (needs PSELD_SRC on path)
        from utils.feature import LogmelIV_Extractor  # noqa: E402

        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = _build_cfg()
        self.sr = self.cfg.data.sample_rate
        self.clip_len = int(10 * self.sr)         # 10-second input
        self.channel_order = list(channel_order)
        self.sed_threshold = sed_threshold

        self.extractor = LogmelIV_Extractor(self.cfg).to(self.device).eval()
        model = HTSAT(self.cfg, num_classes=NUM_CLASSES, in_channels=IN_CHANNELS,
                      audioset_pretrain=False, pretrained_path=None, **_HTSAT_KWARGS)
        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)["state_dict"]
        sd = {k.replace("net._orig_mod.", "").replace("net.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        miss_real = [k for k in missing if "num_batches_tracked" not in k]
        print(f"[PSELDNets] loaded ckpt: missing={len(miss_real)} unexpected={len(unexpected)}")
        if miss_real[:5]:
            print(f"[PSELDNets] sample missing: {miss_real[:5]}")
        if list(unexpected)[:5]:
            print(f"[PSELDNets] sample unexpected: {list(unexpected)[:5]}")
        self.model = model.to(self.device).eval().requires_grad_(False)

    def _prep(self, foa4: np.ndarray, sr: int) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(foa4[:4])).float()
        if sr != self.sr:
            import torchaudio
            x = torchaudio.functional.resample(x, sr, self.sr)
        x = x[self.channel_order]                 # ACN [W,Y,Z,X] -> [W,X,Y,Z]
        T = x.shape[-1]
        if T < self.clip_len:
            x = torch.nn.functional.pad(x, (0, self.clip_len - T))
        else:
            x = x[:, : self.clip_len]
        return x.unsqueeze(0).to(self.device)     # [1,4,clip_len]

    @torch.no_grad()
    def doa(self, foa4: np.ndarray, sr: int):
        x = self._prep(foa4, sr)
        feat = self.extractor(x)                   # [1,7,Tf,mel]
        out = self.model(feat)["accdoa"][0]        # [frames, 3*C]
        C = NUM_CLASSES
        xx, yy, zz = out[:, :C], out[:, C:2 * C], out[:, 2 * C:]
        mag = torch.sqrt(xx ** 2 + yy ** 2 + zz ** 2)   # [frames, C]
        cls = torch.argmax(mag, dim=-1)                  # dominant class per frame
        fr = torch.arange(out.shape[0])
        vx, vy, vz = xx[fr, cls], yy[fr, cls], zz[fr, cls]
        m = mag[fr, cls]
        active = m > self.sed_threshold
        if active.sum() < 1:
            active = m > (m.max() * 0.5)             # fallback: relative
        X, Y, Z = float(vx[active].sum()), float(vy[active].sum()), float(vz[active].sum())
        az = math.degrees(math.atan2(Y, X))
        el = math.degrees(math.atan2(Z, math.hypot(X, Y) + 1e-9))
        return az, el


# --------------------------------------------------------------- calibration test

def _selftest():
    """Validate channel order against SLS ground truth on a few held-out clips."""
    import math as _m
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval_vae_recon import _read_4ch
    import pyarrow.parquet as pq

    sls = os.environ.get("AMBIT_DATA_ROOT", "data") + "/datasets/spatial_librispeech/ambisonics"
    parquet = os.environ.get("AMBIT_DATA_ROOT", "data") + "/datasets/spatial_librispeech/metadata/metadata.parquet"
    test_ids = [134, 50, 79, 90, 110, 222, 333, 444]
    gt = {}
    pf = pq.ParquetFile(parquet)
    for batch in pf.iter_batches(batch_size=8192, columns=["sample_id", "speech/azimuth", "speech/elevation"]):
        for r in batch.to_pylist():
            if int(r["sample_id"]) in test_ids:
                gt[int(r["sample_id"])] = (_m.degrees(r["speech/azimuth"]), _m.degrees(r["speech/elevation"]))
        if len(gt) >= len(test_ids):
            break

    for order in [(0, 3, 1, 2), (0, 1, 2, 3)]:
        scorer = PSELDNetsScorer(channel_order=order)
        errs = []
        print(f"\n=== channel_order={order} ===")
        for sid in test_ids:
            f = f"{sls}/{sid:06d}.flac"
            if not Path(f).exists() or sid not in gt:
                continue
            foa, sr = _read_4ch(f)
            az, el = scorer.doa(foa, sr)
            gaz, gel = gt[sid]
            d = abs((az - gaz + 180) % 360 - 180)
            errs.append(d)
            print(f"  {sid:06d}: pred az={az:7.1f} el={el:6.1f} | GT az={gaz:7.1f} el={gel:6.1f} | az_err={d:5.1f}")
        if errs:
            print(f"  MEDIAN az_err = {sorted(errs)[len(errs)//2]:.1f} deg  (lower => correct order)")


if __name__ == "__main__":
    _selftest()
