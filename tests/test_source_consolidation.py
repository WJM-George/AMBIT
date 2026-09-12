"""Regression checks for shared readers, rendering helpers and legacy CLIs."""
import json
import hashlib
import io
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import (
    ScenePlanTransfusionEditingDataset as OriginalDataset,
    EDITING_GAIN_POLICY,
)
from stable_audio_tools.data.sceneplan_compound_editing_dataset import (
    ScenePlanTransfusionEditingDataset as CompoundDataset,
)
from stable_audio_tools.data.editing_memory_io import memoized_sources, memory_audio_reader
from stable_audio_tools.data.editing_rir_pool import FrozenRIRPool


def index_file(tmp_path, cls):
    path = tmp_path / 'pairs.sqlite'
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE metadata (key TEXT, value TEXT)')
    con.executemany('INSERT INTO metadata VALUES (?,?)', {
        'schema': 'sceneplan_transfusion_editing_training_index',
        'schema_version': '1', 'state': 'materialized_complete_frozen',
        'editing_pair_contract': cls.pair_contract,
        'editing_instruction_contract': cls.instruction_contract,
        'pair_gain_policy': EDITING_GAIN_POLICY,
        'target_latents_exhaustively_reopened': 'true',
        'target_tensor_hashes_exhaustively_verified': 'true',
        'split': 'train', 'rows': '1',
    }.items())
    con.execute('''CREATE TABLE pairs (
        pair_ordinal INTEGER, split TEXT, materialization_status TEXT,
        target_latent_tensor_sha256 TEXT, target_latent_shard_sha256 TEXT,
        source_latent_tensor_sha256 TEXT, source_latent_shard_sha256 TEXT,
        materialized_record_sha256 TEXT)''')
    con.execute('INSERT INTO pairs VALUES (0,?,?,?,?,?,?,?)',
                ('train', 'encoded', *['a' * 64] * 5))
    con.commit()
    con.close()
    return path


@pytest.mark.parametrize('cls,other', [(OriginalDataset, CompoundDataset), (CompoundDataset, OriginalDataset)])
def test_reader_accepts_only_its_own_contract(tmp_path, cls, other):
    path = index_file(tmp_path, cls)
    args = dict(tokenizer_spec=(None, 512), expected_num_samples=1, require_frozen=False)
    assert len(cls(path, **args)) == 1
    with pytest.raises(RuntimeError, match='editing_pair_contract changed'):
        other(path, **args)
    con = sqlite3.connect(path)
    con.execute("UPDATE pairs SET materialization_status='pending'")
    con.commit()
    con.close()
    with pytest.raises(RuntimeError, match='incomplete rows'):
        cls(path, **args)


def test_memory_cache_returns_independent_copies_and_restores_on_error(monkeypatch):
    from scripts.t2a import data
    calls = []

    def load(source, **kwargs):
        calls.append(source)
        return np.array([1., 2.]), {'nested': [3]}

    fake = SimpleNamespace(load_complete_source=load, render_complete_mono_source=load)
    monkeypatch.setattr(data, 'materialize_sceneplan_transfusion_editing_targets', fake, raising=False)
    with pytest.raises(ValueError, match='deliberate'):
        with memoized_sources() as stats:
            first, metadata = fake.load_complete_source({'id': 1})
            first[0] = 99
            metadata['nested'][0] = 99
            second, metadata = fake.load_complete_source({'id': 1})
            assert second.tolist() == [1., 2.]
            assert metadata == {'nested': [3]}
            assert len(calls) == 1 and stats['load_hits'] == 1
            raise ValueError('deliberate')
    assert fake.load_complete_source is load
    assert fake.render_complete_mono_source is load


def test_rir_restores_functions_and_preserves_logical_threads(monkeypatch, tmp_path):
    names = ('rir_builder', 'delay_sum', 'fractional_delay')
    originals = {name: object() for name in names}
    replacement = SimpleNamespace(**{name: object() for name in names})
    pra = SimpleNamespace(constants={'num_threads': 128}, libroom=SimpleNamespace(**originals))
    monkeypatch.setitem(sys.modules, 'pyroomacoustics', pra)
    pool = FrozenRIRPool(tmp_path)
    monkeypatch.setattr(pool, 'native_module', lambda: replacement)
    with pytest.raises(ValueError):
        with pool.bounded_rir_pool():
            assert pra.constants['num_threads'] == 128
            assert pra.libroom.rir_builder is replacement.rir_builder
            raise ValueError()
    assert vars(pra.libroom) == originals
    pra.constants['num_threads'] = 4
    with pytest.raises(RuntimeError, match='128 logical'):
        with pool.bounded_rir_pool():
            pass
    with pool.bounded_rir_pool(enabled=False):
        assert vars(pra.libroom) == originals


def test_memory_audio_replay_matches_disk_decoder_and_restores(monkeypatch):
    import soundfile as sf
    from scripts.t2a import data
    stream = io.BytesIO()
    sf.write(stream, np.linspace(-0.5, 0.5, 40).reshape(10, 4), 44100,
             format='FLAC', subtype='PCM_24')
    blob = stream.getvalue()
    real_read = sf.read
    fake = SimpleNamespace(sf=SimpleNamespace(read=real_read))
    monkeypatch.setattr(data, 'materialize_sceneplan_transfusion_editing_targets', fake, raising=False)
    results = [dict(_memory_foa_flac=blob, target_foa_sha256=hashlib.sha256(blob).hexdigest(),
                    target_foa_path=f'memory-{i}.flac') for i in range(20)]
    expected, sample_rate = real_read(io.BytesIO(blob), dtype='float32', always_2d=True)
    with memory_audio_reader(results):
        assert all('_memory_foa_flac' not in row for row in results)
        for row in results:
            actual, rate = fake.sf.read(row['target_foa_path'], dtype='float32', always_2d=True)
            np.testing.assert_array_equal(actual, expected)
            assert rate == sample_rate
    assert fake.sf.read is real_read


def test_rir_rejects_changed_extension(tmp_path):
    native = tmp_path / 'native/rir_pool1'
    native.mkdir(parents=True)
    extension = native / 'test.so'
    extension.write_bytes(b'not-an-extension')
    (native / 'BUILD.json').write_text(json.dumps({
        'extension': str(extension), 'extension_sha256': '0' * 64,
    }))
    with pytest.raises(RuntimeError, match='checksum changed'):
        FrozenRIRPool(tmp_path).native_module()


@pytest.mark.parametrize('legacy,flags,wav', [
    (False, [], False), (True, [], True),
    (False, ['--listen-wav'], True), (True, ['--no-listen-wav'], False),
])
def test_decode_cli_keeps_each_entry_default(tmp_path, monkeypatch, legacy, flags, wav):
    from scripts.vae.eval import decode_latents_4ch as canonical
    from data_download.scripts import decode_latents_4ch as compatible
    (tmp_path / 'manifest.json').write_text('[]')
    monkeypatch.setattr(sys, 'argv', ['decode', '--output-dir', str(tmp_path),
                                    '--from-existing-quad', '--decode-all', *flags])
    (compatible if legacy else canonical).main()
    assert (tmp_path / 'listen_wav_48k').is_dir() == wav
