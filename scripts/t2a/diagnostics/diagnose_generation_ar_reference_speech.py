#!/usr/bin/env python3
"""Measure frozen ASR on validation reference FOA latent reconstructions."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import zlib


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main(root):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    cfg = json.loads((root / 'PROTOCOL.json').read_text())
    assert sha(__file__) == cfg['script_sha256']
    for path, expected in cfg['input_files_sha256'].items():
        assert sha(path) == expected, path
    sys.path.insert(0, cfg['snapshot'])
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from safetensors import safe_open
    from scripts.t2a.data.preencode_spatial_cot_family_shard import _load_vae
    from scripts.t2a.eval.score_sceneplan_dit_p10_speech import _transcribe, _error_rates
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.25)
    started = time.monotonic()
    rows = []
    db = sqlite3.connect('file:' + cfg['validation_index'] + '?mode=ro&immutable=1', uri=True)
    db.row_factory = sqlite3.Row
    baseline = {r['id']: r for r in json.loads(Path(cfg['baseline_report']).read_text())['rows']}
    save(root / 'STATUS.json', {'status': 'LOADING_FROZEN_REFERENCE_VAE', 'pid': os.getpid()})
    vae, _ = _load_vae(Path(cfg['vae_config']), Path(cfg['vae_checkpoint']), torch.device('cuda:0'))
    for sid in cfg['case_ids']:
        assert sid in baseline
        if time.monotonic() - started > cfg['wall_cap_s']:
            raise TimeoutError('Reference diagnostic budget exceeded')
        source = dict(db.execute('SELECT s.*,l.path AS latent_path FROM samples s JOIN latent_shards l ON l.id=s.latent_shard_id WHERE s.sample_id=?', (sid,)).fetchone())
        plan = json.loads(zlib.decompress(source['scene_plan_zlib']))
        speech = [s for s in plan['sources'] if s['kind'] == 'speech']
        assert len(speech) == 1
        reference_text = speech[0]['transcript']
        # No reference text is ever supplied to the recognizer. It is used only
        # after recognition, and never enters any Generation AR worker.
        assert reference_text.split() == baseline[sid]['ar_transcript'].split()
        with safe_open(source['latent_path'], framework='pt', device='cpu') as handle:
            latent = handle.get_tensor(source['latent_key'])
        assert hashlib.sha256(latent.contiguous().numpy().tobytes()).hexdigest() == source['latent_tensor_sha256']
        assert latent.ndim == 2
        with torch.inference_mode():
            wave = vae.decode(latent.float().cuda()[None])[0].float().cpu()
        wave = wave[:, :int(source['model_num_samples'])]
        assert wave.shape == (4, int(source['model_num_samples'])) and bool(torch.isfinite(wave).all())
        path = root / 'reference_reconstructions' / f'{sid}.foa.wav'
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, wave.numpy().T, 44100, subtype='FLOAT')
        row = {'id': sid, 'reference_plan': plan, 'transcript': reference_text,
               'latent_path': source['latent_path'], 'latent_key': source['latent_key'],
               'latent_tensor_sha256': source['latent_tensor_sha256'],
               'foa': str(path), 'foa_sha256': sha(path), 'source_count': len(plan['sources']),
               'speech_source_id': speech[0]['source_id'], 'model_num_samples': wave.shape[1]}
        rows.append(row)
        save(root / 'STATUS.json', {'status': 'REFERENCE_VAE_DECODE', 'rows_done': len(rows), 'rows': len(cfg['case_ids'])})
    del vae, latent, wave
    db.close()
    gc.collect()
    torch.cuda.empty_cache()
    from faster_whisper import WhisperModel
    asr_contract = json.loads(Path(cfg['asr_contract']).read_text())
    for name, digest in asr_contract['model_assets_sha256'].items():
        assert sha(Path(cfg['asr_model']) / name) == digest
    model = WhisperModel(cfg['asr_model'], device='cuda', device_index=0, compute_type='float16', local_files_only=True)
    scored = []
    for row in rows:
        wave, rate = sf.read(row['foa'], dtype='float32', always_2d=True)
        speech = next(s for s in row['reference_plan']['sources'] if s['kind'] == 'speech')
        start = max(0., speech['activity']['onset_sec'] - .05)
        end = min(len(wave) / rate, speech['activity']['offset_sec'] + .05)
        crop = torch.from_numpy(wave[round(start * rate):round(end * rate), 0].copy()).reshape(1, -1)
        crop = torchaudio.functional.resample(crop, rate, 16000)[0].numpy()
        crop = crop / max(1e-8, float(np.abs(crop).max())) * (10 ** (-1 / 20))
        recognized = _transcribe(model, crop.astype(np.float32))
        errors = _error_rates(recognized['text'], row['transcript'])
        scored.append({**row, 'asr_crop': recognized, 'crop_vs_reference': errors,
                       'p10_generated_crop_wer': baseline[row['id']]['crop_vs_ar']['wer']})
    save(root / 'REPORT.json', {'status': 'REFERENCE_DIAGNOSTIC_COMPLETE', 'rows': scored,
         'elapsed_s': time.monotonic() - started, 'test_used': False, 'goal_complete': False,
         'limits': 'Reference mixtures reconstructed through the frozen VAE, not original dry recordings. Reference plans can have different lawful times/positions/slot IDs from generated plans. This diagnoses recognition/codec/content difficulty and is not raw-request accuracy or listening acceptance.'})
    save(root / 'STATUS.json', {'status': 'COMPLETE', 'report': str(root / 'REPORT.json')})


if __name__ == '__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.root.resolve())
    except BaseException as exc:
        save(args.root / 'STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION', 'error': f'{type(exc).__name__}: {exc}'})
        raise
