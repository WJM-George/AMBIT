"""Validated reuse of FOA artifacts; contains no model or GPU loading code."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sqlite3


class InvalidFOAArtifact(ValueError):
    """An existing receipt disagrees with its inputs or audio; retain evidence."""


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); temp.replace(path)


def read_raw_results(raw, *, requests, requests_sha256, checkpoint_sha256,
                     snapshot_manifest_sha256, entry_sha256):
    """Reuse only a completed raw-only job with matching immutable inputs."""
    status_path = raw / 'STATUS.json'
    if not status_path.exists() or json.loads(status_path.read_text()).get('status') != 'COMPLETE':
        return None
    contract = json.loads((raw / 'CONTRACT.json').read_text())
    expected = {'input_sha256': requests_sha256, 'checkpoint_sha256': checkpoint_sha256,
        'snapshot_manifest_sha256': snapshot_manifest_sha256, 'script_sha256': entry_sha256,
        'declared_count_constraint': False, 'target_or_annotation_inputs': False, 'binding_strength': 0}
    if any(contract.get(k) != v for k, v in expected.items()) or 'learned_count_expert' in contract:
        raise InvalidFOAArtifact('AR cache belongs to a different input, checkpoint or inference policy')
    seal = raw / 'RAW_CACHE_SEAL.json'
    fingerprints = {name: sha(raw / name) for name in ('CONTRACT.json', 'predictions.sqlite', 'SUMMARY.json')}
    if seal.exists() and json.loads(seal.read_text()) != fingerprints:
        raise InvalidFOAArtifact('Completed AR cache changed after it was sealed')
    db = sqlite3.connect('file:' + str(raw / 'predictions.sqlite') + '?mode=ro&immutable=1', uri=True)
    results = {sid: json.loads(payload) for sid, payload in db.execute('SELECT id,payload FROM results')}; db.close()
    if len(results) != len(requests) or set(results) != {r['id'] for r in requests}:
        raise InvalidFOAArtifact('AR cache does not cover exactly the requested batch')
    for item in requests:
        row = results[item['id']]
        if row['request'] != item['request'] or row['model_input_sha256'] != hashlib.sha256(item['request'].encode()).hexdigest():
            raise InvalidFOAArtifact('AR cache request text changed')
    if not seal.exists(): atomic_json(seal, fingerprints)
    return results


def read_verified_foa(receipt_path, *, expected):
    """Check receipt, exact WAV hash, format, size and finiteness before reuse.

    A WAV without a committed receipt is incomplete and may be regenerated.
    A committed artifact with mismatched metadata/data fails closed, preserving
    the existing files so the caller can audit and recover the specific row.
    """
    if not receipt_path.exists(): return None
    import numpy as np
    import soundfile as sf
    try:
        row = json.loads(receipt_path.read_text())
        if row.get('status') != 'FOA_WRITTEN' or row.get('render_identity') != expected:
            raise InvalidFOAArtifact('FOA receipt input identity mismatch')
        wav = Path(row['foa'])
        if wav.parent.resolve() != receipt_path.parent.resolve():
            raise InvalidFOAArtifact('FOA receipt points outside this output directory')
        if not wav.is_file() or sha(wav) != row['foa_sha256']:
            raise InvalidFOAArtifact('FOA waveform missing or its hash changed')
        info = sf.info(wav)
        if info.samplerate != 44100 or info.channels != 4 or info.frames != expected['model_num_samples'] or info.subtype != 'FLOAT':
            raise InvalidFOAArtifact('FOA sample rate, channel count, length or subtype changed')
        waveform, rate = sf.read(wav, dtype='float32', always_2d=True)
        if not np.isfinite(waveform).all(): raise InvalidFOAArtifact('FOA contains nonfinite values')
        if row.get('channel_order') != 'WYZX' or row.get('ambisonic_convention') != 'ACN/SN3D':
            raise InvalidFOAArtifact('FOA channel convention receipt changed')
        return row
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise InvalidFOAArtifact('Unreadable FOA receipt or waveform') from exc
