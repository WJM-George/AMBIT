#!/usr/bin/env python3
"""ASR diagnostic separating raw-request, AR-text and saved-audio discrepancies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def atomic(path, value):
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def main(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    sys.path.insert(0, str(args.snapshot))
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from faster_whisper import WhisperModel
    from scripts.t2a.eval.score_sceneplan_dit_p10_speech import _error_rates, _transcribe
    torch.set_num_threads(4)
    summary = json.loads(args.summary.read_text()); assert summary['status'] == 'COMPLETE'
    witnesses = json.loads(args.witnesses.read_text())
    witnesses = {r['id']: r for r in witnesses['pairs']}
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/models/faster-distil-whisper-large-v3")
    identity = {'schema': 'generation_ar_p10_speech_audio_diagnostic_v1', 'summary_sha256': sha(args.summary),
        'witnesses_sha256': sha(args.witnesses), 'script_sha256': sha(Path(__file__)),
        'asr_helper_sha256': sha(args.snapshot / 'scripts/t2a/eval/score_sceneplan_dit_p10_speech.py'),
        'model_assets_sha256': {p.name: sha(p) for p in sorted(model_path.iterdir()) if p.is_file() and p.suffix in ('.json', '.bin')},
        'asr': 'Existing faster-distil-whisper-large-v3, float16 GPU0, English, beam5, no prompt, no VAD, no previous-text conditioning.',
        'crop_policy': 'Whole W channel and AR-chosen speech interval plus 0.05 seconds. References never enter ASR.',
        'test_used': False, 'gpu_scope': [0], 'acceptance_established': False}
    atomic(args.output / 'CONTRACT.json', identity)
    atomic(args.output / 'STATUS.json', {'status': 'LOADING', 'pid': os.getpid()})
    model = WhisperModel(str(model_path), device='cuda', device_index=0, compute_type='float16', local_files_only=True)
    rows = []; started = time.monotonic()
    for row in summary['rows']:
        refs = [(s['key'], c['value']) for s in witnesses[row['id']]['requirements']['sources'] for c in s['constraints'] if c['op'] == 'transcript']
        if not refs: continue
        assert len(refs) == 1, 'This diagnostic panel must have exactly one requested speech source per speech scene'
        assert sha(row['foa']) == row['foa_sha256']
        plan = json.loads(Path(row['sceneplan']).read_text())['p10_plan']
        sources = [s for s in plan['sources'] if s['kind'] == 'speech']
        assert len(sources) <= 1, 'Multiple predicted speech sources need a coupled transcript assignment'
        audio, rate = sf.read(row['foa'], dtype='float32', always_2d=True); assert rate == 44100 and audio.shape[1] == 4
        def recognize(value):
            mono = torchaudio.functional.resample(torch.from_numpy(value.copy()).reshape(1, -1), rate, 16000)[0].numpy()
            peak = float(np.abs(mono).max())
            if peak > 1e-8: mono = mono / peak * (10 ** (-1 / 20))
            return _transcribe(model, mono.astype(np.float32))
        whole = recognize(audio[:, 0]); cropped = None
        if sources:
            source = sources[0]; onset = max(0., source['activity']['onset_sec'] - .05); offset = min(plan['duration_sec'], source['activity']['offset_sec'] + .05)
            cropped = recognize(audio[round(onset * rate):round(offset * rate), 0])
        requested = refs[0][1]; ar_text = sources[0]['transcript'] if sources else ''
        rows.append({'id': row['id'], 'requested_transcript': requested, 'ar_transcript': ar_text,
            'ar_vs_request': _error_rates(ar_text, requested), 'asr_whole': whole, 'asr_crop': cropped,
            'whole_vs_request': _error_rates(whole['text'], requested), 'whole_vs_ar': _error_rates(whole['text'], ar_text),
            'crop_vs_request': _error_rates(cropped['text'], requested) if cropped else None,
            'crop_vs_ar': _error_rates(cropped['text'], ar_text) if cropped else None})
        atomic(args.output / 'PARTIAL.json', {'rows': rows}); atomic(args.output / 'STATUS.json', {'status': 'RUNNING', 'speech_scenes_done': len(rows)})
    result = {'status': 'COMPLETE_DIAGNOSTIC_ONLY', 'rows': rows, 'elapsed_s': time.monotonic() - started,
        'test_used': False, 'acceptance_established': False,
        'limitations': 'Automatic ASR can err on speech, names, fragments and overlapping music/sounds. Crops use generated timing and do not isolate sources. WER is diagnostic, not an independent human listening verdict.'}
    atomic(args.output / 'REPORT.json', result); atomic(args.output / 'STATUS.json', {'status': 'COMPLETE', 'speech_scenes': len(rows), 'report': str(args.output / 'REPORT.json')})
    print(json.dumps({'status': 'COMPLETE', 'speech_scenes': len(rows), 'elapsed_s': result['elapsed_s']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'summary', 'witnesses', 'output'): p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    try: main(args)
    except BaseException as exc:
        args.output.mkdir(parents=True, exist_ok=True); atomic(args.output / 'STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION', 'error': f'{type(exc).__name__}: {exc}'}); raise
