#!/usr/bin/env python3
"""Measure saved P10 FOA timing/direction against its generated plan on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES', '') == ''
    sys.path.insert(0, str(args.snapshot))
    import numpy as np
    import soundfile as sf
    import torch
    from stable_audio_tools.data.model_sceneplan import compile_model_44_controls
    from scripts.t2a.eval.score_sceneplan_dit_p10_core import _doa_metrics, _activity_metrics
    torch.set_num_threads(4)
    summary = json.loads(args.summary.read_text())
    assert summary['status'] == 'COMPLETE'
    args.output.mkdir(parents=True, exist_ok=True)
    contract = {'schema': 'generation_ar_p10_spatial_audio_diagnostic_v1',
                'script_sha256': sha(Path(__file__)), 'summary_sha256': sha(args.summary),
                'frozen_dependencies_sha256': {name: sha(args.snapshot / name) for name in (
                    'scripts/t2a/eval/score_sceneplan_dit_p10_core.py',
                    'stable_audio_tools/data/model_sceneplan.py',
                    'stable_audio_tools/data/foa_intensity.py')},
                'device': 'CPU', 'hop_samples': 1024, 'audio_sample_rate': 44100,
                'direction_policy': 'Existing energy-weighted intensity metric, exactly one planned active source, coherence >= 0.1. Overlapping source identity is not inferred.',
                'activity_policy': 'Existing scene-energy threshold against union of planned intervals. Reverb tails and naturally intermittent content can affect the estimate.',
                'target_policy': 'Generated P10 plan, not hidden GT numbers. Request-to-plan accuracy is scored separately.',
                'acceptance_established': False, 'test_used': False}
    save(args.output / 'CONTRACT.json', contract)
    rows, calibration = [], []
    started = time.monotonic()
    for row in summary['rows']:
        assert sha(row['foa']) == row['foa_sha256']
        plan = json.loads(Path(row['sceneplan']).read_text())['p10_plan']
        waveform, rate = sf.read(row['foa'], dtype='float32', always_2d=True)
        assert rate == 44100 and waveform.shape == (row['model_num_samples'], 4)
        audio = torch.from_numpy(waveform.T.copy())
        kwargs = {'model_num_samples': row['model_num_samples'], 'latent_frames': row['latent_frames']}
        doa = _doa_metrics(audio, plan, **kwargs)
        activity = _activity_metrics(audio, plan, **kwargs)
        # An analytic plane wave checks the convention and the metric's input
        # plumbing. It is a calibration control, never a generated-model output.
        if len(plan['sources']) == 1:
            controls = compile_model_44_controls(plan, model_num_samples=kwargs['model_num_samples'],
                                                  latent_frames_valid=kwargs['latent_frames'])
            features = controls['source_trajectory_features'][0]
            active = controls['source_event_frame_ids'][0] > 0
            direction = np.stack((features[:, 3] * features[:, 1],
                                  features[:, 3] * features[:, 0], features[:, 2]), axis=-1)
            frames = kwargs['latent_frames']
            signal = np.sin(np.arange(frames * 1024) * (2 * np.pi * 220 / rate)).reshape(frames, 1024)
            signal = signal * active[:, None] * .1
            control = np.stack((signal, signal * direction[:, 1, None],
                                signal * direction[:, 2, None], signal * direction[:, 0, None]))
            control = torch.from_numpy(control.reshape(4, -1).astype('float32'))
            calibrated = _doa_metrics(control, plan, **kwargs)
            calibrated_activity = _activity_metrics(control, plan, **kwargs)
            assert calibrated['spherical_error_mean_deg'] < .1
            assert calibrated_activity['temporal_iou'] == 1.
            calibration.append({'id': row['id'], 'analytic_plane_wave_doa': calibrated,
                                'analytic_activity': calibrated_activity})
        rows.append({'id': row['id'], 'source_count': len(plan['sources']),
                     'kinds': [source['kind'] for source in plan['sources']],
                     'generated_plan_sha256': sha(row['sceneplan']), 'foa_sha256': row['foa_sha256'],
                     'generated_doa': doa, 'generated_activity': activity})
        save(args.output / 'STATUS.json', {'status': 'RUNNING', 'rows_done': len(rows)})
    result = {'status': 'COMPLETE_DIAGNOSTIC_ONLY', 'rows': rows, 'calibration': calibration,
              'elapsed_s': time.monotonic() - started, 'test_used': False, 'acceptance_established': False,
              'limitations': 'Aggregate intensity cannot identify overlapping sources. These metrics do not establish audible event content or source-specific timing. Silence and diffuse frames remain explicit in coverage denominators.'}
    save(args.output / 'REPORT.json', result)
    save(args.output / 'STATUS.json', {'status': 'COMPLETE', 'rows': len(rows),
                                      'report': str(args.output / 'REPORT.json')})
    print(json.dumps({'status': 'COMPLETE', 'rows': len(rows), 'calibration_controls': len(calibration)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'summary', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        save(args.output / 'STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION',
              'error': f'{type(exc).__name__}: {exc}'})
        raise
