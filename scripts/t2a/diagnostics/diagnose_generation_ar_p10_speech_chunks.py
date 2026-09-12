#!/usr/bin/env python3
"""Test short transcript chunks with the same P10 on isolated speech controls."""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
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


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def split_source(plan, maximum_words):
    """Preserve words and the integer frame budget; no external text generator."""
    speech = [source for source in plan['sources'] if source['kind'] == 'speech']
    assert len(speech) == 1
    source = speech[0]
    words = source['transcript'].split()
    count = math.ceil(len(words) / maximum_words)
    boundaries = [round(index * len(words) / count) for index in range(count + 1)]
    texts = [' '.join(words[a:b]) for a, b in zip(boundaries, boundaries[1:])]
    assert ' '.join(texts).split() == words
    onset = round(source['activity']['onset_sec'] * 44100 / 1024)
    offset = round(source['activity']['offset_sec'] * 44100 / 1024)
    frames = offset - onset
    timing = [round(boundary / len(words) * frames) for boundary in boundaries]
    trajectory = source['trajectory']

    def position(fraction):
        if trajectory['type'] == 'static':
            return deepcopy(trajectory['position'])
        first, last = trajectory['start'], trajectory['end']
        angle = (last['azimuth_deg'] - first['azimuth_deg'] + 180) % 360 - 180
        return {'azimuth_deg': (first['azimuth_deg'] + fraction * angle + 180) % 360 - 180,
                'elevation_deg': first['elevation_deg'] + fraction * (last['elevation_deg'] - first['elevation_deg']),
                'distance_m': first['distance_m'] + fraction * (last['distance_m'] - first['distance_m'])}

    chunks = []
    for index, text in enumerate(texts):
        start, end = timing[index:index + 2]
        length = end - start
        assert length >= 32
        local = deepcopy(plan)
        local['duration_sec'] = length * 1024 / 44100
        local['sources'] = [deepcopy(source)]
        local_source = local['sources'][0]
        local_source.update(source_id='source_0', transcript=text,
                            activity={'onset_sec': 0., 'offset_sec': local['duration_sec']})
        if trajectory['type'] == 'linear':
            local_source['trajectory'] = {'type': 'linear', 'start': position(start / max(1, frames - 1)),
                                          'end': position((end - 1) / max(1, frames - 1))}
        chunks.append({'index': index, 'text': text, 'start_frame': onset + start,
                       'end_frame': onset + end, 'frames': length, 'plan': local})
    assert chunks[0]['start_frame'] == onset and chunks[-1]['end_frame'] == offset
    assert all(a['end_frame'] == b['start_frame'] for a, b in zip(chunks, chunks[1:]))
    isolated = deepcopy(plan)
    isolated['sources'] = [deepcopy(source)]
    isolated['sources'][0]['source_id'] = 'source_0'
    return isolated, chunks


def main(root):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    cfg = json.loads((root / 'PROTOCOL.json').read_text())
    assert sha(Path(__file__)) == cfg['script_sha256']
    assert sha(cfg['baseline_report']) == cfg['baseline_report_sha256']
    assert sha(cfg['p10_identity']) == cfg['p10_identity_sha256']
    assert sha(cfg['asr_contract']) == cfg['asr_contract_sha256']
    sys.path.insert(0, cfg['snapshot'])
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
    torch.cuda.set_per_process_memory_fraction(.35)
    source_rows = json.loads(Path(cfg['baseline_report']).read_text())['rows']
    baseline = {row['id']: row for row in source_rows if row['arm'] == 'isolated_speech_control'}
    baseline.update({row['id']: row for row in source_rows if row['arm'] == 'canonical' and row['source_count'] == 1})
    codec = ModelScenePlanCodecV4(CODEC_PATH)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, local_files_only=True)
    prepared = []
    for sid in cfg['case_ids']:
        row = baseline[sid]
        assert sha(row['foa']) == row['foa_sha256']
        plan, chunks = split_source(row['plan'], cfg['maximum_words'])
        for chunk in chunks:
            bundle = finalize_sceneplan_for_p10(codec, codec.encode(chunk['plan']), tokenizer=tokenizer,
                        task=P11Task.GENERATION, sample_id=sid)
            bundle.assert_external_p10_boundary()
            assert bundle.latent_frames_valid == chunk['frames']
            assert bundle.sceneplan['sources'][0]['transcript'].split() == chunk['text'].split()
            chunk['quantized_p10_plan'] = bundle.sceneplan
            chunk['bundle'] = bundle
        prepared.append((sid, plan, chunks))
    assert sum(len(chunks) for _, _, chunks in prepared) <= cfg['budget']['maximum_chunk_renders']
    save(root / 'INPUT_GATE.json', {'status': 'PASS', 'source_cases': len(prepared),
         'chunks': sum(len(chunks) for _, _, chunks in prepared), 'all_original_words_preserved': True,
         'original_activity_frame_budget_preserved': True, 'all_local_p10_bundles_valid': True,
         'final_ar_plans_modified': False})
    identity = json.loads(Path(cfg['p10_identity']).read_text())
    for asset in identity['files'].values():
        assert sha(asset['path']) == asset['sha256']
    save(root / 'STATUS.json', {'status': 'LOADING_FROZEN_P10', 'pid': os.getpid()})
    executor = _executor_from_identity(identity, device=torch.device('cuda:0'))

    def parameter_hashes():
        return {name: hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
                for name, value in executor.wrapper.named_parameters()}

    before = parameter_hashes()
    outputs = []
    started = time.monotonic()
    for sid, plan, chunks in prepared:
        samples = baseline[sid]['model_num_samples']
        combined = torch.zeros(4, samples)
        folder = root / 'outputs' / sid
        folder.mkdir(parents=True, exist_ok=True)
        traces = []
        for chunk in chunks:
            if time.monotonic() - started > cfg['budget']['maximum_render_wall_s']:
                raise TimeoutError('Chunk-control budget exceeded')
            index = chunk['index']
            save(root / 'STATUS.json', {'status': 'RENDERING_SPEECH_CHUNKS', 'id': sid, 'chunk': index})
            seed = _stable_seed(42, f'{sid}/speech_chunk/{index}')
            audio = executor.render(chunk['bundle'], seed=seed)
            assert audio.shape == (4, chunk['frames'] * 1024) and bool(torch.isfinite(audio).all())
            start, end = chunk['start_frame'] * 1024, chunk['end_frame'] * 1024
            assert end - start == audio.shape[1]
            combined[:, start:end] = audio
            path = folder / f'chunk_{index}.wav'
            sf.write(path, audio.numpy().T, 44100, subtype='FLOAT')
            traces.append({key: value for key, value in chunk.items() if key != 'bundle'})
            traces[-1].update(foa=str(path), foa_sha256=sha(path), seed=seed)
        path = folder / 'concatenated.foa.wav'
        sf.write(path, combined.numpy().T, 44100, subtype='FLOAT')
        record = {'id': sid, 'plan': plan, 'chunks': traces, 'foa': str(path), 'foa_sha256': sha(path),
                  'model_num_samples': samples, 'latent_frames': baseline[sid]['latent_frames'],
                  'assembly': 'Native P10 FOA chunks placed contiguously in the original speech activity window; silence outside. No spatial reprojection, normalization or other-source mixing.'}
        save(folder / 'RECEIPT.json', record)
        outputs.append(record)
    assert before == parameter_hashes()
    save(root / 'RENDER_COMPLETE.json', {'status': 'COMPLETE', 'render_elapsed_s': time.monotonic() - started,
         'p10_parameter_bytes_unchanged': True, 'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    del executor, bundle
    gc.collect()
    torch.cuda.empty_cache()
    from faster_whisper import WhisperModel
    asr_contract = json.loads(Path(cfg['asr_contract']).read_text())
    for name, expected in asr_contract['model_assets_sha256'].items():
        assert sha(Path(cfg['asr_model']) / name) == expected
    model = WhisperModel(cfg['asr_model'], device='cuda', device_index=0, compute_type='float16', local_files_only=True)
    scores = []
    for row in outputs:
        wave, rate = sf.read(row['foa'], dtype='float32', always_2d=True)
        source = row['plan']['sources'][0]
        start = max(0., source['activity']['onset_sec'] - .05)
        end = min(row['plan']['duration_sec'], source['activity']['offset_sec'] + .05)
        crop = torch.from_numpy(wave[round(start * rate):round(end * rate), 0].copy()).reshape(1, -1)
        crop = torchaudio.functional.resample(crop, rate, 16000)[0].numpy()
        crop = crop / max(1e-8, float(np.abs(crop).max())) * (10 ** (-1 / 20))
        recognized = _transcribe(model, crop.astype(np.float32))
        errors = _error_rates(recognized['text'], source['transcript'])
        dimensions = {'model_num_samples': row['model_num_samples'], 'latent_frames': row['latent_frames']}
        scores.append({**row, 'asr_crop': recognized, 'crop_vs_plan': errors,
                       'baseline_crop_wer': baseline[row['id']]['crop_vs_plan']['wer'],
                       'delta_wer': errors['wer'] - baseline[row['id']]['crop_vs_plan']['wer'],
                       'generated_doa': _doa_metrics(torch.from_numpy(wave.T.copy()), row['plan'], **dimensions),
                       'generated_activity': _activity_metrics(torch.from_numpy(wave.T.copy()), row['plan'], **dimensions)})
    mean_delta = sum(row['delta_wer'] for row in scores) / len(scores)
    promising = mean_delta <= -.1 and max(row['delta_wer'] for row in scores) <= .05
    save(root / 'REPORT.json', {'status': 'PROMISING_FOR_LARGER_VALIDATION' if promising else 'FAIL_CHUNK_CONTROL_RULE',
         'rows': scores, 'macro_crop_wer_delta_vs_isolated_baseline': mean_delta, 'test_used': False,
         'goal_complete': False, 'model_promoted': False,
         'limitations': 'Selected long-speech controls. Chunks use different per-chunk noise seeds, lose cross-chunk linguistic context and may vary in voice/intonation. No mixed-scene or final listening acceptance.'})
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
