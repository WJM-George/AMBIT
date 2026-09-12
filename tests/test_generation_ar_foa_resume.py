"""Resume must reject changed inputs/artifacts and require real FOA format."""
import copy
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from stable_audio_tools.inference.sceneplan_generation_ar_foa_resume import (
    InvalidFOAArtifact, atomic_json, read_raw_results, read_verified_foa, sha)


def raw_cache(tmp_path):
    request = {'id': 'bell', 'request': 'A bell rings on my left.'}
    expected = dict(requests=[request], requests_sha256='request-sha', checkpoint_sha256='checkpoint-sha',
                    snapshot_manifest_sha256='snapshot-sha', entry_sha256='entry-sha')
    atomic_json(tmp_path / 'CONTRACT.json', {'input_sha256': 'request-sha', 'checkpoint_sha256': 'checkpoint-sha',
        'snapshot_manifest_sha256': 'snapshot-sha', 'script_sha256': 'entry-sha',
        'declared_count_constraint': False, 'target_or_annotation_inputs': False, 'binding_strength': 0})
    atomic_json(tmp_path / 'STATUS.json', {'status': 'COMPLETE'})
    atomic_json(tmp_path / 'SUMMARY.json', {'status': 'COMPLETE', 'rows': 1})
    db = sqlite3.connect(tmp_path / 'predictions.sqlite')
    db.execute('CREATE TABLE results(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    value = {**request, 'model_input_sha256': hashlib.sha256(request['request'].encode()).hexdigest(), 'prediction': None}
    db.execute('INSERT INTO results VALUES (?,?)', ('bell', json.dumps(value))); db.commit(); db.close()
    return expected


def test_raw_cache_seals_then_reuses_without_model_loading_and_detects_tampering(tmp_path):
    expected = raw_cache(tmp_path)
    first = read_raw_results(tmp_path, **expected)
    assert read_raw_results(tmp_path, **expected) == first
    db = sqlite3.connect(tmp_path / 'predictions.sqlite'); db.execute('DELETE FROM results'); db.commit(); db.close()
    with pytest.raises(InvalidFOAArtifact, match='changed after'):
        read_raw_results(tmp_path, **expected)


def test_raw_cache_rejects_a_different_checkpoint_and_count_forcing(tmp_path):
    expected = raw_cache(tmp_path)
    with pytest.raises(InvalidFOAArtifact): read_raw_results(tmp_path, **{**expected, 'checkpoint_sha256': 'new'})
    p = tmp_path / 'CONTRACT.json'; contract = json.loads(p.read_text()); contract['declared_count_constraint'] = True; atomic_json(p, contract)
    with pytest.raises(InvalidFOAArtifact): read_raw_results(tmp_path, **expected)


def waveform_receipt(tmp_path, *, channels=4):
    wav = tmp_path / 'sample.foa.wav'; audio = np.zeros((1024, channels), dtype=np.float32)
    audio[100:200, 0] = .1; sf.write(wav, audio, 44100, subtype='FLOAT')
    identity = {'run_contract_sha256': 'run', 'id': 'bell', 'p10_plan_sha256': 'plan', 'seed': 42, 'model_num_samples': 1024}
    row = {'status': 'FOA_WRITTEN', 'render_identity': identity, 'foa': str(wav), 'foa_sha256': sha(wav),
           'channel_order': 'WYZX', 'ambisonic_convention': 'ACN/SN3D', 'repeat_bit_exact': True}
    receipt = tmp_path / 'sample.audio.json'; atomic_json(receipt, row)
    return receipt, row, identity


def test_foa_reuses_only_identical_inputs_and_uncorrupted_audio(tmp_path):
    receipt, row, expected = waveform_receipt(tmp_path)
    assert read_verified_foa(receipt, expected=expected) == row
    with pytest.raises(InvalidFOAArtifact, match='identity'):
        read_verified_foa(receipt, expected={**expected, 'p10_plan_sha256': 'different-source-position'})
    with Path(row['foa']).open('ab') as f: f.write(b'changed')
    with pytest.raises(InvalidFOAArtifact, match='hash'):
        read_verified_foa(receipt, expected=expected)


def test_matching_hash_does_not_make_stereo_a_valid_foa_artifact(tmp_path):
    receipt, _, expected = waveform_receipt(tmp_path, channels=2)
    with pytest.raises(InvalidFOAArtifact, match='channel count'):
        read_verified_foa(receipt, expected=expected)


def test_wav_without_committed_receipt_is_incomplete(tmp_path):
    receipt, row, expected = waveform_receipt(tmp_path); receipt.unlink()
    assert read_verified_foa(receipt, expected=expected) is None
