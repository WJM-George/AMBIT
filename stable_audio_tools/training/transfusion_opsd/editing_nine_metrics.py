"""The existing nine Editing metrics on a fixed development panel.

The independent paper CLAP is evaluation-only; OPSD training retains the native
FOA-latent CLAP. No score here is returned to request-side teacher collection.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch

from stable_audio_tools.paths import editing_bench

BENCH = editing_bench()
sys.path.insert(0, str(BENCH))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


class EditingNineMetrics:
    def __init__(self, device, cache):
        from benchmark_audio_v1 import KemarFoaDecoder
        from benchmark_feature_models_v1 import ClapFeatures, PannsFeatures, VggishFeatures
        import stereocrw_metrics_v2 as spatial
        self.cache = Path(cache); self.cache.mkdir(parents=True, exist_ok=True)
        self.decoder = KemarFoaDecoder()
        self.clap = ClapFeatures(str(device))
        self.panns = PannsFeatures(str(device))
        self.vgg = VggishFeatures(str(device))
        self.crw = spatial.StereoCRW()
        self.crw.load_verified(BENCH / 'assets/StereoCRW/FreeMusic-StereoCRW-1024.pth.tar')
        self.crw.to(device).eval().requires_grad_(False)
        self.spatial = spatial

    def extract(self, stereo, samples):
        features = self.panns.extract(stereo, 44100)
        features.update(self.vgg.extract(stereo, 44100))
        features['clap_audio'] = self.clap.audio(stereo, 44100, seed=42)
        a, _ = self.spatial.prepare_audio(stereo, 44100, samples)
        result = self.crw.extract(a, patch_chunk=256, window_batch=2)
        features['crw_itd_ms'] = result['crw_itd_ms']
        features['fsad_embedding_seconds'] = result['fsad_embedding_seconds']
        if not all(np.isfinite(value).all() for value in features.values()):
            raise RuntimeError('Nonfinite evaluation feature.')
        return features

    def score(self, wave, target_path, pair_id, samples, output):
        from benchmark_metrics_v2 import basic_values, target_basic
        from paired_clap_v1 import paired_clap
        from summarize_dev100_metrics_v1 import kl
        raw, sr = sf.read(target_path, dtype='float64', always_2d=True)
        if sr != 44100 or raw.shape != (samples, 4):
            raise RuntimeError('Incorrect target waveform geometry.')
        truth = self.decoder.decode(raw.T)
        pred = self.decoder.decode(wave[0, :, :samples].float().cpu().numpy().astype(np.float64))
        target_file = self.cache / (pair_id + '.npz')
        target_binding = self.cache / (pair_id + '.json')
        target_hash = hashlib.sha256(Path(target_path).read_bytes()).hexdigest()
        if target_file.exists():
            binding = json.loads(target_binding.read_text())
            if binding != {'pair_id': pair_id, 'target_path': str(target_path), 'sha256': target_hash}:
                raise RuntimeError('Evaluation target cache identity changed.')
            with np.load(target_file) as f:
                target_features = dict(f)
        else:
            target_features = self.extract(truth, samples)
            temp = target_file.with_suffix('.tmp.npz')
            np.savez_compressed(temp, **target_features); temp.replace(target_file)
            write(target_binding, {'pair_id': pair_id, 'target_path': str(target_path), 'sha256': target_hash})
        features = self.extract(pred, samples)
        basic = basic_values(pred, sr, samples, target_basic(truth, sr, samples))
        scalar = {'Paired CLAP': paired_clap(features['clap_audio'], target_features['clap_audio']),
            'KL': float(kl(features['panns_logits'], target_features['panns_logits'])),
            'LSD': basic['LSD_log10power'], 'GCC': basic['gcc_mean_itd_mse_ms2_x1000'],
            'CRW': self.spatial.paired_crw(features['crw_itd_ms'], target_features['crw_itd_ms'])['CRW']}
        if any(value is None or not np.isfinite(value) for value in scalar.values()):
            raise RuntimeError('The fixed validation panel has an undefined metric; do not silently drop it.')
        output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **features)
        return dict(scalar=scalar, features=str(output), target_features=str(target_file))


def aggregate_records(records):
    from summarize_dev100_metrics_v1 import frechet
    from stereocrw_metrics_v2 import fsad
    def load(key):
        results = []
        for record in records:
            with np.load(record[key]) as f:
                results.append(dict(f))
        return results
    generated, target = load('features'), load('target_features')
    metrics = {key: float(np.mean([r['scalar'][key] for r in records])) for key in records[0]['scalar']}
    def norm(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)
    for name, key in [('FD-CLAP', 'clap_audio'), ('FD-PANN', 'panns_embedding'), ('FAD', 'vggish_frames')]:
        combine = np.concatenate if name == 'FAD' else np.stack
        a, b = combine([x[key] for x in generated]), combine([x[key] for x in target])
        if name == 'FD-CLAP':
            a, b = norm(a.astype(np.float64)), norm(b.astype(np.float64))
        metrics[name] = frechet(a, b)
    metrics['FSAD'] = fsad(np.stack([x['fsad_embedding_seconds'] for x in generated]),
                           np.stack([x['fsad_embedding_seconds'] for x in target]))['value']
    return metrics
