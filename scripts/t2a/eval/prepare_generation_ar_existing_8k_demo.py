#!/usr/bin/env python3
"""Build a deterministic local listening panel from existing benchmark audio."""
import argparse
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def prepare(root, freeze):
    folder = root / 'demo'; folder.mkdir(exist_ok=True)
    assert not (folder / 'SELECTION.json').exists(), 'Preserve the existing fixed selection'
    mapping = read(root / 'BENCHMARK_MAPPING.json')['rows']
    inputs = {}
    for gpu in range(3):
        for index, row in enumerate(read(root / f'inputs/raw_gpu{gpu}.json')['requests']):
            sid = row['id']; stem = f'{index:04d}_' + hashlib.sha256(sid.encode()).hexdigest()[:12]
            inputs[sid] = {**row, 'new_audio_receipt': str(root / f'gpu{gpu}/output/{stem}.audio.json')}
    chosen = []
    for count in range(1, 5):
        for template in range(5):
            speech = (count + template) % 2 == 1
            eligible = [row for row in mapping if row['source_count'] == count and
                row['template_id'] == f'test_{template:03d}' and ('speech' in row['source_kinds']) == speech]
            assert eligible, (count, template, speech)
            row = min(eligible, key=lambda row: hashlib.sha256(('demo20-v1|' + row['id']).encode()).hexdigest())
            chosen.append({**row, **inputs[row['id']], 'baselines': []})
    by_id = {row['id']: row for row in chosen}
    manifest = Path(freeze['existing_public_benchmark']) / 'generation_requests.jsonl'
    assert sha(manifest) == freeze['files_sha256'][str(manifest)]
    with manifest.open() as handle:
        for line in handle:
            item = json.loads(line)
            if item['sample_id'] not in by_id:
                continue
            metadata_path = Path(item['native_output_path']).parent / 'generation.json'
            metadata = read(metadata_path)
            assert metadata['status'] == 'PASS' and metadata['panel_id'] == item['panel_id']
            by_id[item['sample_id']]['baselines'].append({
                'system': item['baseline_id'], 'display_name': item['baseline_display_name'],
                'native_audio': item['native_output_path'], 'sha256': sha(item['native_output_path']),
                'prompt': item['semantic_prompt'], 'generation_receipt': str(metadata_path),
                'generation_receipt_sha256': sha(metadata_path)})
    preview = Path(freeze['model_snapshot']) / 'scripts/t2a/eval/sceneplan_44_eval_common.py'
    selection = {'schema': 'generation_ar_existing_8k_listening_panel_v1', 'created_unix': time.time(),
        'builder_sha256': sha(__file__), 'freeze_sha256': sha(root / 'FREEZE.json'),
        'selection_rule': '20 scenes: one per source-count(1–4) × test-template(0–4), alternating speech presence; smallest SHA256(demo20-v1|ID) within each cell. IDs depend only on pre-existing metadata, not generated audio or scores. Selection declared after test rendering started.',
        'preview_helper': str(preview), 'preview_helper_sha256': sha(preview),
        'preview_scope': 'Native FOA is preserved. Listening copies use the existing WYZX ±30-degree virtual stereo decoder and separate preview gain. This is not HRTF binaural or full 3D listening. Baselines use existing native mono/stereo at their saved levels.',
        'conditioning_disclosure': freeze['input_disclosure'], 'test_used': True,
        'audio_quality_selected': False, 'human_listening_review_complete': False,
        'goal_complete': False, 'rows': chosen}
    save(folder / 'SELECTION.json', selection)
    save(folder / 'STATUS.json', {'status': 'SELECTION_FIXED', 'rows': 20, 'test_used': True, 'goal_complete': False})


def link_audio(folder, name, path, digest):
    path = Path(path).resolve(strict=True)
    assert sha(path) == digest, path
    target = folder / name
    if target.is_symlink():
        assert target.resolve(strict=True) == path
    else:
        assert not target.exists()
        target.symlink_to(path)
    return target


def build(root, freeze, args):
    folder = root / 'demo'; selection = read(folder / 'SELECTION.json')
    assert selection['builder_sha256'] == sha(__file__)
    assert selection['freeze_sha256'] == sha(root / 'FREEZE.json')
    assert sha(selection['preview_helper']) == selection['preview_helper_sha256']
    lock = (folder / 'LOCK').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0, freeze['model_snapshot'])
    import soundfile as sf
    import torch
    from scripts.t2a.eval.sceneplan_44_eval_common import virtual_stereo, atomic_wav
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    assets = folder / 'assets'; assets.mkdir(exist_ok=True)
    cases = folder / 'cases'; cases.mkdir(exist_ok=True)
    done = {}
    def preview(name, path, digest):
        assert sha(path) == digest
        audio, rate = sf.read(path, dtype='float32', always_2d=True)
        assert rate == 44100 and audio.shape[1] == 4
        stereo, meta = virtual_stereo(torch.from_numpy(audio.T.copy()))
        output = assets / name
        atomic_wav(output, stereo, rate, subtype='PCM_16')
        return {'preview': str(output), 'sha256': sha(output), 'native_audio': str(path),
                'native_sha256': digest, 'preview_gain': meta}
    for row in selection['rows']:
        stem = row['panel_id']
        reference = preview(stem + '_gt.wav', row['reference_foa'], row['reference_foa_sha256'])
        old = read(row['existing_gt_plan_p10_metadata'])
        assert sha(row['existing_gt_plan_p10_metadata']) == row['existing_gt_plan_p10_metadata_sha256']
        p10 = link_audio(assets, stem + '_gt_plan_p10.wav', old['generated_stereo_path'], old['generated_stereo_sha256'])
        row['existing_media'] = {'original_gt': reference, 'gt_plan_p10': {'preview': str(p10),
            'sha256': old['generated_stereo_sha256'], 'native_audio': row['existing_gt_plan_p10_foa'],
            'native_sha256': row['existing_gt_plan_p10_foa_sha256']}}
        for baseline in row['baselines']:
            baseline['preview'] = str(link_audio(assets, stem + '_' + baseline['system'] + '.wav',
                baseline['native_audio'], baseline['sha256']))
    deadline = time.monotonic() + args.wall_cap_seconds
    while len(done) < len(selection['rows']):
        assert time.monotonic() < deadline, 'Demo wait budget exhausted'
        for row in selection['rows']:
            if row['id'] in done:
                continue
            receipt_path = Path(row['new_audio_receipt'])
            if not receipt_path.exists():
                continue
            receipt = read(receipt_path)
            if receipt.get('status') != 'FOA_WRITTEN':
                continue
            assert receipt['id'] == row['id'] and receipt['sample_rate'] == 44100
            row['new_ar_media'] = preview(row['panel_id'] + '_ar_p10.wav', receipt['foa'], receipt['foa_sha256'])
            plan = read(receipt['sceneplan']); assert plan['sample_id'] == row['id']
            row['sceneplan'] = receipt['sceneplan']; row['sceneplan_sha256'] = sha(receipt['sceneplan'])
            row['audio_receipt_sha256'] = sha(receipt_path); done[row['id']] = row
        write_pages(folder, selection, done)
        save(folder / 'STATUS.json', {'status': 'COMPLETE' if len(done) == 20 else 'WAITING_FOR_SELECTED_AR_FOA',
            'rows_done': len(done), 'rows': 20, 'pid': os.getpid(), 'updated_unix': time.time(),
            'test_used': True, 'goal_complete': False})
        if len(done) < 20:
            time.sleep(60)
    save(folder / 'COMPLETE.json', {'status': 'COMPLETE', 'rows': 20, 'index': str(folder / 'index.html'),
        'manifest': str(folder / 'MANIFEST.json'), 'human_listening_review_complete': False,
        'all_audio_reused_or_previewed_without_synthesis': True, 'goal_complete': False})


def write_pages(folder, selection, done):
    escape = html.escape
    header = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Generation AR: matched audio examples</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:36px auto;padding:0 20px;color:#172333;background:#f7f9fc}h1{font-size:28px}details{padding:18px;margin:14px 0;background:white;border:1px solid #dce3eb;border-radius:8px}summary{cursor:pointer;font-weight:600}.request{white-space:pre-wrap;line-height:1.6}.audio-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}audio{width:100%}.note{line-height:1.5;color:#4a5b70}a{color:#165ea6}</style>',
        '<h1>Raw English → Generation AR → P10</h1>',
        f'<p>{len(done)}/20 fixed examples ready. Select a case to compare audio.</p>',
        '<p class="note">' + escape(selection['preview_scope']) + '</p>',
        '<p class="note">' + escape(selection['conditioning_disclosure']) + '</p>',
        '<p class="note">' + escape(selection['selection_rule']) + '</p>']
    index = ['# Generation AR matched listening examples', '', selection['selection_rule'], '',
        selection['preview_scope'], '', selection['conditioning_disclosure'], '',
        f'Open [the local listening page]({folder / "index.html"}), or open a case below. This panel has not yet received a human listening review.', '']
    for selected in selection['rows']:
        sid = selected['id']; row = done.get(sid)
        label = f"{selected['panel_id']} · {selected['source_count']} sources · {selected['template_id']} · {', '.join(selected['source_kinds'])}"
        if row is None:
            header.append('<details><summary>' + escape(label) + '</summary><p>AR audio is still rendering.</p></details>')
            continue
        media = [('Generation AR → P10', row['new_ar_media']['preview']),
                 ('Existing GT-plan → P10', row['existing_media']['gt_plan_p10']['preview']),
                 ('Original GT FOA', row['existing_media']['original_gt']['preview'])]
        media += [(baseline['display_name'], baseline['preview']) for baseline in row['baselines']]
        header += ['<details><summary>' + escape(label) + '</summary>', '<p class="request">' + escape(row['request']) + '</p>', '<div class="audio-grid">']
        case = [f'# {label}', '', row['request'], '',
                f'[Generated ScenePlan]({row["sceneplan"]})', '', selection['preview_scope'], '']
        for name, path in media:
            relative = Path(path).relative_to(folder).as_posix()
            header.append('<div><p>' + escape(name) + '</p><audio controls preload="none" src="' + escape(relative, quote=True) + '"></audio></div>')
            case += [f'**{name}**', '', f'![{name}]({path})', '']
        header += ['</div><p class="note">A missing baseline was not generated for this scene in the existing benchmark. No replacement audio is synthesized.</p></details>']
        case_path = folder / 'cases' / (row['panel_id'] + '.md'); case_path.write_text('\n'.join(case))
        index.append(f'- [{label}]({case_path})')
    header.append('</html>')
    for name, content in [('index.html', '\n'.join(header)), ('README.md', '\n'.join(index) + '\n')]:
        temporary = folder / (name + '.tmp'); temporary.write_text(content); temporary.replace(folder / name)
    save(folder / 'MANIFEST.json', {'status': 'COMPLETE' if len(done) == 20 else 'PARTIAL',
        'selection_sha256': sha(folder / 'SELECTION.json'), 'rows': list(done.values()),
        'human_listening_review_complete': False, 'test_used': True, 'goal_complete': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=('prepare', 'build'), required=True)
    parser.add_argument('--wall-cap-seconds', type=int, default=28800)
    args = parser.parse_args(); root = args.root.resolve()
    try:
        freeze = read(root / 'FREEZE.json')
        prepare(root, freeze) if args.mode == 'prepare' else build(root, freeze, args)
    except BlockingIOError:
        raise
    except BaseException as error:
        save(root / 'demo/STATUS.json', {'status': 'FAILED', 'pid': os.getpid(),
             'updated_unix': time.time(), 'error': repr(error), 'goal_complete': False})
        raise
