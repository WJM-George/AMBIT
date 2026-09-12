#!/usr/bin/env python3
"""CPU diagnostics of saved raw-AR FOA against its own generated P10 controls.

No hidden GT waveform or completion numbers are used. These measurements are
renderer diagnostics, not proof of per-source audible semantics in a mixture.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def main(args):
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    sys.path.insert(0, str(args.snapshot))
    import numpy as np
    import soundfile as sf
    import torch
    from scripts.t2a.eval.score_sceneplan_dit_p10_core import _activity_metrics, _doa_metrics
    torch.set_num_threads(4)
    summary = json.loads(args.summary.read_text())
    assert summary['status'] in ('COMPLETE', 'PARTIAL')
    rows = []
    for row in summary['rows']:
        if row['status'] != 'FOA_WRITTEN':
            rows.append({'id': row['id'], 'status': row['status']}); continue
        assert sha(row['foa']) == row['foa_sha256']
        normalized = json.loads(Path(row['sceneplan']).read_text())
        assert normalized['non_text_fields_unchanged']
        plan = normalized['p10_plan']
        raw, rate = sf.read(row['foa'], dtype='float32', always_2d=True)
        assert rate == 44100 and raw.shape == (row['model_num_samples'], 4) and np.isfinite(raw).all()
        assert row['channel_order'] == 'WYZX' and row['ambisonic_convention'] == 'ACN/SN3D'
        audio = torch.from_numpy(raw.T.copy())
        settings = {'model_num_samples': row['model_num_samples'], 'latent_frames': row['latent_frames']}
        rows.append({'id': row['id'], 'status': 'MEASURED', 'source_count': len(plan['sources']),
            'foa_sha256': row['foa_sha256'], 'p10_plan_sha256': normalized['p10_plan_sha256'],
            'peak': float(np.abs(raw).max()), 'rms': float(np.sqrt(np.mean(raw.astype(np.float64) ** 2))),
            'samples_outside_unit_range_fraction': float((np.abs(raw) > 1).mean()),
            'p10_control_doa': _doa_metrics(audio, plan, **settings),
            'p10_scene_activity': _activity_metrics(audio, plan, **settings)})
    result = {'schema': 'generation_ar_p10_control_audio_diagnostics_v1', 'status': 'COMPLETE',
        'summary_sha256': sha(args.summary), 'script_sha256': sha(Path(__file__)),
        'metric_source_sha256': sha(args.snapshot / 'scripts/t2a/eval/score_sceneplan_dit_p10_core.py'),
        'intensity_source_sha256': sha(args.snapshot / 'stable_audio_tools/data/foa_intensity.py'),
        'rows': rows, 'test_used': False, 'gpu_used': False, 'acceptance_established': False,
        'interpretation': 'DoA compares exactly-one-active-source, coherent, energetic frames with the AR-chosen P10 plan. Activity measures scene-level energy only. Overlapping-source semantics, isolated event timing, transcript/speaker fidelity, and direct request satisfaction are not established by these proxies. No GT completion values or waveforms were consulted.'}
    text = json.dumps(result, indent=2) + '\n'
    if args.output.exists() and args.output.read_text() != text:
        raise ValueError('Refusing to replace diagnostics of different saved artifacts')
    args.output.parent.mkdir(parents=True, exist_ok=True); temp = args.output.with_suffix('.tmp'); temp.write_text(text); temp.replace(args.output)
    print(json.dumps({'status': 'COMPLETE', 'measured': sum(r['status'] == 'MEASURED' for r in rows), 'output': str(args.output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True); parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); main(parser.parse_args())
