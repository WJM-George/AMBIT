#!/usr/bin/env python3
"""Bounded frozen-P10 sampling and isolated-speech controls on generated plans."""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main(root, protocol_path=None):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    cfg = json.loads((protocol_path or root / 'PROTOCOL.json').read_text())
    assert cfg['script_sha256'] == sha(Path(__file__))
    assert cfg['summary_sha256'] == sha(cfg['summary'])
    assert cfg['p10_identity_sha256'] == sha(cfg['p10_identity'])
    assert cfg['asr_contract_sha256'] == sha(cfg['asr_contract'])
    snapshot = Path(cfg['snapshot'])
    sys.path.insert(0, str(snapshot))
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from transformers import AutoTokenizer
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    from stable_audio_tools.data.sceneplan_p11_single_turn import finalize_sceneplan_for_p10, P11Task
    from scripts.t2a.test.evaluate_sceneplan_transfusion_generation_ar_p10_audio_8k import _executor_from_identity, _stable_seed, CODEC_PATH, QWEN_PATH
    from scripts.t2a.eval.score_sceneplan_dit_p10_core import _doa_metrics, _activity_metrics
    from scripts.t2a.eval.score_sceneplan_dit_p10_speech import _error_rates, _transcribe
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_per_process_memory_fraction(cfg['budget']['gpu0_memory_fraction'])
    summary = json.loads(Path(cfg['summary']).read_text())
    assert summary['status'] == 'COMPLETE'
    saved = {row['id']: row for row in summary['rows']}
    codec = ModelScenePlanCodecV4(CODEC_PATH)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, local_files_only=True)
    rows = []
    plans = {}
    bundles = {}
    for sid in cfg['case_ids']:
        row = saved[sid]
        assert sha(row['foa']) == row['foa_sha256']
        plan = json.loads(Path(row['sceneplan']).read_text())['p10_plan']
        plans[sid] = plan
        bundle = finalize_sceneplan_for_p10(codec, codec.encode(plan), tokenizer=tokenizer,
                    task=P11Task.GENERATION, sample_id=sid)
        bundle.assert_external_p10_boundary()
        assert bundle.sceneplan == plan
        bundles[sid] = bundle
        rows.append({'arm': 'canonical', 'id': sid, 'plan': plan, 'foa': row['foa'],
                     'foa_sha256': row['foa_sha256'], 'source_count': len(plan['sources']),
                     'model_num_samples': bundle.model_num_samples, 'latent_frames': bundle.latent_frames_valid,
                     'reused_existing_audio': True})
    identity = json.loads(Path(cfg['p10_identity']).read_text())
    for asset in identity['files'].values():
        assert sha(asset['path']) == asset['sha256']
    atomic(root / 'STATUS.json', {'status': 'LOADING_FROZEN_P10', 'pid': os.getpid()})
    executor = _executor_from_identity(identity, device=torch.device('cuda:0'))

    def parameter_state():
        return {name: {'object_id': id(value), 'pointer': value.data_ptr(), 'version': value._version,
                      'sha256': hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
                for name, value in executor.wrapper.named_parameters()}

    before = parameter_state()
    started = time.monotonic()
    gate_id = cfg['case_ids'][0]
    reproduced = executor.render(bundles[gate_id], seed=_stable_seed(42, gate_id))
    existing, rate = sf.read(saved[gate_id]['foa'], dtype='float32', always_2d=True)
    assert rate == 44100 and np.array_equal(reproduced.numpy().T, existing)
    atomic(root / 'RENDER_GATE.json', {'status': 'PASS', 'id': gate_id,
           'canonical_saved_audio_bit_exact': True, 'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
           'same_gpu_raw_generation_untouched': True})
    del reproduced, existing
    tasks = []
    for arm, setting in cfg['sampling_arms'].items():
        tasks.extend((arm, sid, setting, bundles[sid]) for sid in cfg['case_ids'])
    for sid, plan in plans.items():
        speech = [source for source in plan['sources'] if source['kind'] == 'speech']
        if speech and len(plan['sources']) > 1:
            isolated = deepcopy(plan)
            isolated['sources'] = [deepcopy(speech[0])]
            isolated['sources'][0]['source_id'] = 'source_0'
            bundle = finalize_sceneplan_for_p10(codec, codec.encode(isolated), tokenizer=tokenizer,
                        task=P11Task.GENERATION, sample_id=sid)
            bundle.assert_external_p10_boundary()
            tasks.append(('isolated_speech_control', sid, cfg['canonical_sampling'], bundle))
    reused_controls = 0
    for arm, sid, setting, bundle in tasks:
        if time.monotonic() - started > cfg['budget']['max_render_wall_s']:
            raise TimeoutError('P10 control rendering budget exceeded')
        folder = root / 'outputs' / arm
        folder.mkdir(parents=True, exist_ok=True)
        stem = hashlib.sha256(sid.encode()).hexdigest()[:16]
        receipt = folder / (stem + '.json')
        if receipt.exists():
            row = json.loads(receipt.read_text())
            assert row['plan'] == bundle.sceneplan and row['sampling'] == setting
            assert row['foa_sha256'] == sha(row['foa'])
            rows.append(row)
            reused_controls += 1
            continue
        executor.steps = setting['steps']
        executor.cfg_scale = setting['cfg_scale']
        executor.cfg_rescale_phi = setting['cfg_rescale_phi']
        executor.rescale_cfg = setting.get('rescale_cfg', True)
        atomic(root / 'STATUS.json', {'status': 'RENDERING_VALID_PLAN_CONTROLS', 'arm': arm, 'id': sid,
               'new_audio_done': len(rows) - len(plans), 'new_audio_total': len(tasks)})
        audio = executor.render(bundle, seed=_stable_seed(42, sid))
        assert tuple(audio.shape) == (4, bundle.model_num_samples) and bool(torch.isfinite(audio).all())
        path = folder / (stem + '.foa.wav')
        temp = path.with_suffix('.tmp.wav')
        sf.write(temp, audio.numpy().T, 44100, subtype='FLOAT')
        temp.replace(path)
        row = {'arm': arm, 'id': sid, 'plan': bundle.sceneplan, 'sampling': setting,
               'foa': str(path), 'foa_sha256': sha(path), 'source_count': len(bundle.sceneplan['sources']),
               'model_num_samples': bundle.model_num_samples, 'latent_frames': bundle.latent_frames_valid,
               'reused_existing_audio': False, 'seed': _stable_seed(42, sid),
               'role': 'Explicit valid-plan diagnostic control; not a new AR prediction or a deployed sampler.'}
        atomic(receipt, row)
        rows.append(row)
        del audio
    after = parameter_state()
    assert before.keys() == after.keys()
    assert all(before[name][field] == after[name][field] for name in before
               for field in ('object_id', 'pointer', 'sha256'))
    version_changes = {name: after[name]['version'] - before[name]['version'] for name in before
                       if after[name]['version'] != before[name]['version']}
    assert all(name.startswith('diffusion.conditioner.') for name in version_changes)
    atomic(root / 'PARAMETER_GUARD.json', {'status': 'PASS', 'parameter_count': len(before),
           'all_parameter_values_objects_and_pointers_unchanged': True,
           'version_counter_changes': version_changes,
           'explanation': 'Frozen P10 temporarily copies EMA conditioner values and restores online values using copy_. Version counters change; exact tensor bytes must be restored.',
           'cached_render_receipts_reused': reused_controls})
    atomic(root / 'RENDER_COMPLETE.json', {'status': 'COMPLETE', 'total_control_audio': len(tasks),
           'new_control_audio_this_invocation': len(tasks) - reused_controls,
           'reused_control_audio': reused_controls, 'canonical_replay_this_invocation': 1,
           'render_elapsed_s': time.monotonic() - started, 'all_p10_parameters_unchanged': True,
           'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    del executor, bundle, bundles, tokenizer, codec
    gc.collect()
    torch.cuda.empty_cache()
    from faster_whisper import WhisperModel
    atomic(root / 'STATUS.json', {'status': 'LOADING_DIAGNOSTIC_ASR'})
    asr_contract = json.loads(Path(cfg['asr_contract']).read_text())
    for name, expected in asr_contract['model_assets_sha256'].items():
        assert sha(Path(cfg['asr_model']) / name) == expected
    model = WhisperModel(cfg['asr_model'], device='cuda', device_index=0,
                         compute_type='float16', local_files_only=True)

    def recognize(value, rate):
        mono = torchaudio.functional.resample(torch.from_numpy(value.copy()).reshape(1, -1), rate, 16000)[0].numpy()
        peak = float(np.abs(mono).max())
        if peak > 1e-8:
            mono = mono / peak * (10 ** (-1 / 20))
        return _transcribe(model, mono.astype(np.float32))

    scored = []
    for row in rows:
        wave, rate = sf.read(row['foa'], dtype='float32', always_2d=True)
        plan = row['plan']
        audio = torch.from_numpy(wave.T.copy())
        dimensions = {'model_num_samples': row['model_num_samples'], 'latent_frames': row['latent_frames']}
        score = {**row, 'generated_doa': _doa_metrics(audio, plan, **dimensions),
                 'generated_activity': _activity_metrics(audio, plan, **dimensions)}
        speech = [source for source in plan['sources'] if source['kind'] == 'speech']
        if speech:
            assert len(speech) == 1
            source = speech[0]
            onset = max(0., source['activity']['onset_sec'] - .05)
            offset = min(plan['duration_sec'], source['activity']['offset_sec'] + .05)
            # Neither the transcript nor the request is passed to the ASR.
            whole = recognize(wave[:, 0], rate)
            crop = recognize(wave[round(onset * rate):round(offset * rate), 0], rate)
            score.update(asr_whole=whole, asr_crop=crop,
                         whole_vs_plan=_error_rates(whole['text'], source['transcript']),
                         crop_vs_plan=_error_rates(crop['text'], source['transcript']))
        scored.append(score)
        atomic(root / 'PARTIAL_SCORES.json', {'rows': scored})
        atomic(root / 'STATUS.json', {'status': 'SCORING_AUDIO', 'done': len(scored), 'rows': len(rows)})
    by_arm = {}
    speech_ids = cfg['long_speech_case_ids']
    music_id = cfg['moving_music_case_id']
    canonical = {row['id']: row for row in scored if row['arm'] == 'canonical'}
    for arm in cfg['sampling_arms']:
        selected = {row['id']: row for row in scored if row['arm'] == arm}
        differences = {sid: selected[sid]['crop_vs_plan']['wer'] - canonical[sid]['crop_vs_plan']['wer'] for sid in speech_ids}
        mean_delta = sum(differences.values()) / len(differences)
        music_delta = selected[music_id]['generated_doa']['spherical_error_mean_deg'] - canonical[music_id]['generated_doa']['spherical_error_mean_deg']
        promising = (mean_delta <= -.1 and max(differences.values()) <= .05 and music_delta <= 2.)
        by_arm[arm] = {'long_speech_crop_wer_delta': differences, 'long_speech_macro_wer_delta': mean_delta,
                       'music_doa_error_delta_deg': music_delta,
                       'status': 'PROMISING_FOR_LARGER_VALIDATION' if promising else 'FAIL_TARGETED_SAMPLING_RULE'}
    report = {'status': 'COMPLETE_DIAGNOSTIC_ONLY', 'rows': scored, 'sampling_comparison': by_arm,
              'test_used': False, 'model_promoted': False, 'goal_complete': False,
              'limitations': 'Targeted validation failures, not a representative acceptance panel. ASR can err on unusual words/abbreviations and overlapping sources. Isolation changes the conditioning scene. No final audio-accuracy claim or P10 release replacement.'}
    atomic(root / 'REPORT.json', report)
    atomic(root / 'STATUS.json', {'status': 'COMPLETE', 'report': str(root / 'REPORT.json')})


if __name__ == '__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    args = parser.parse_args()
    try:
        main(args.root.resolve(), args.protocol)
    except BaseException as exc:
        atomic(args.root / 'STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION',
               'error': f'{type(exc).__name__}: {exc}', 'updated_unix': time.time()})
        raise
