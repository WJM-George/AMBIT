#!/usr/bin/env python3
"""Versioned English text repair for 19 known original Generation GT rows.

Original files are read-only. Counts, speech transcripts/speaker text, activity,
room and trajectories are preserved. Only the audited mixed-script sound/music
descriptions and the corresponding old request wording/tokenization change.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import zlib
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
import numpy as np
from transformers import AutoTokenizer
from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4

ROOT = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/generation_ar/template_run")
SOURCE = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/transfusion_shared_v1/generation_ar")
REPAIRS = {
    'wooden打击 instrument': 'wooden percussion instrument',
    'Да, сегодняшний матч.': "Yes, today's match.",
    'Вперёд, держись!': 'Forward, hang in there!',
    'Хорошо, молодец': 'Good, well done',
    'Вот': 'Here',
    'придурковатый': 'foolish',
    'Ты хороший, ты хороший,': "You are good, you are good,",
    'Это не совсем так': "It is not quite so",
    'Привет!': 'Hello!',
    'Твою мать': 'Damn it',
    'Блин': 'Darn',
}
NON_ENGLISH_SCRIPT = re.compile(r'[\u3400-\u9fff\u0400-\u052f\u0600-\u06ff]')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def canonical(v): return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
def digest(v): return hashlib.sha256(v).hexdigest()
def english(text):
    for before, after in REPAIRS.items(): text = text.replace(before, after)
    assert not NON_ENGLISH_SCRIPT.search(text)
    return text


def main():
    out = ROOT / 'source_english_v1'; out.mkdir(exist_ok=True)
    report_path = out / 'REPAIR_REPORT.json'
    if report_path.exists():
        report = json.loads(report_path.read_text()); assert sha(out / 'train.sqlite') == report['repaired_sha256']; print('Already complete'); return
    assert not (out / 'train.sqlite').exists()
    source_sha = sha(SOURCE / 'train.sqlite')
    building = out / 'train.building'; shutil.copyfile(SOURCE / 'train.sqlite', building)
    db = sqlite3.connect(building); db.execute('PRAGMA journal_mode=OFF'); db.execute('PRAGMA synchronous=OFF')
    audit = json.loads((ROOT / 'ORIGINAL_LANGUAGE_AUDIT.json').read_text())
    assert audit['count'] == 19 and all(r['split'] == 'train' for r in audit['bad_rows'])
    codec = ModelScenePlanCodecV4(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4")
    tokenizer = AutoTokenizer.from_pretrained(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/Qwen/Qwen3.5-0.8B", local_files_only=True)
    records = []
    for bad in audit['bad_rows']:
        ordinal = bad['ordinal']
        blob, old_request = db.execute('SELECT target_sceneplan_zlib,raw_user_request FROM rows WHERE ordinal=?', (ordinal,)).fetchone()
        old = json.loads(zlib.decompress(blob)); new = copy.deepcopy(old)
        for field in bad['fields']:
            assert field['field'] == 'description'
            source = next(s for s in new['sources'] if s['source_id'] == field['source_id'])
            assert source['kind'] in ('sound', 'music') and source['description'] == field['text']
            source['description'] = english(source['description'])
            # The Russian phrase is reported in English while retaining the
            # fact that the underlying sound asset includes Russian speech.
            if 'in Russian' in source['description']:
                source['description'] = source['description'].replace('saying, ', 'with the English meaning, ').replace('saying ', 'with the English meaning ')
                source['description'] = source['description'].replace('which translates to ', 'meaning ')
        target = codec.project_plan(new)
        stripped = copy.deepcopy(target)
        for original_source, changed_source in zip(old['sources'], stripped['sources']):
            if 'description' in changed_source: changed_source['description'] = original_source['description']
        assert stripped == old, 'Repair changed a non-description field or source order'
        encoded = codec.encode(target, max_tokens=1024)
        assert codec.decode(encoded['input_ids'].tolist(), sample_id=old['sample_id']) == target
        tokens = encoded['input_ids'].numpy().astype('<u2').tobytes()
        groups = encoded['loss_group_ids'].numpy().astype('<i2').tobytes()
        request = english(old_request); length = len(tokenizer(request, add_special_tokens=True)['input_ids']); assert length <= 512
        plan_bytes = canonical(target)
        db.execute('UPDATE rows SET raw_user_request=?,raw_user_request_sha256=?,raw_request_qwen_tokens=?,target_sceneplan_zlib=?,target_sceneplan_sha256=?,target_token_ids_u16le=?,target_loss_group_ids_i16le=?,target_token_count=?,target_tokens_sha256=? WHERE ordinal=?',
            (request, digest(request.encode()), length, zlib.compress(plan_bytes, 1), digest(plan_bytes), tokens, groups, len(encoded['input_ids']), digest(tokens), ordinal))
        records.append({'ordinal': ordinal, 'before': old, 'after': target, 'before_plan_sha256': digest(zlib.decompress(blob)), 'after_plan_sha256': digest(plan_bytes), 'roundtrip': True})
    db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('english_description_repair_rows', '19'))
    db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('original_generation_manifest_sha256', source_sha))
    db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('english_repair_report', str(report_path)))
    db.commit(); assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'; db.close()
    building.replace(out / 'train.sqlite')
    for split in ('validation', 'test'): (out / (split + '.sqlite')).symlink_to(SOURCE / (split + '.sqlite'))
    report = {'status': 'PASS', 'original': str(SOURCE / 'train.sqlite'), 'original_sha256': source_sha,
        'repaired': str(out / 'train.sqlite'), 'repaired_sha256': sha(out / 'train.sqlite'), 'repaired_rows': 19,
        'original_files_modified': False, 'speech_transcripts_and_speaker_text_changed': False,
        'non_text_fields_and_source_order_unchanged': True, 'records': records,
        'review': 'Explicit bilingual text corrections authored by Codex; no model-generated label acceptance.'}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n'); (out / 'train.sqlite').chmod(0o444)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}), flush=True)


if __name__ == '__main__': main()
