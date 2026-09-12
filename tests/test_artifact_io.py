"""Keep frozen artifact bytes and report hashes stable after consolidation."""
import hashlib
import json
import sqlite3
import zlib

import pytest

from stable_audio_tools.data import artifact_io as io
from scripts.t2a.eval import merge_sceneplan_p11_v4_challenge_shards as merger


@pytest.mark.parametrize('value', [
    {}, {'中文': ['音频', 0, -0.0, None, True]},
    {'b': 1.23456789012345, 'a': {'key': '\n"\\'}},
])
def test_frozen_serialization_and_compression(value):
    expected = json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'), allow_nan=False).encode()
    assert io.canonical(value).encode() == expected
    assert io.digest(value) == hashlib.sha256(expected).hexdigest()
    assert io.pack(value) == zlib.compress(expected, 6)
    assert io.unpack(io.pack(value)) == value


def test_report_nan_policy_does_not_leak_into_frozen_artifacts():
    value = {'diagnostic': float('nan'), 'infinite': float('inf')}
    expected = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    assert merger._json_sha256(value) == hashlib.sha256(expected).hexdigest()
    with pytest.raises(ValueError):
        io.digest(value)
    with pytest.raises(ValueError):
        io.pack(value)


def test_atomic_replacement_preserves_existing_file_on_serialization_failure(tmp_path):
    path = tmp_path / 'nested/state.json'
    io.write(path, {'value': 1})
    assert path.read_bytes() == b'{"value":1}\n'
    assert io.read(path) == {'value': 1}
    assert io.sha(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        io.write(path, {'value': float('nan')})
    assert io.read(path) == {'value': 1}
    io.write(path, {'value': 2})
    assert io.read(path) == {'value': 2}


def test_immutable_database_is_read_only_and_does_not_create_missing_files(tmp_path):
    path = tmp_path / 'index.sqlite'
    connection = sqlite3.connect(path)
    connection.execute('CREATE TABLE metadata (value INTEGER)')
    connection.execute('INSERT INTO metadata VALUES (7)')
    connection.commit()
    connection.close()
    connection = io.db(path)
    try:
        assert connection.execute('SELECT value FROM metadata').fetchone()['value'] == 7
        with pytest.raises(sqlite3.OperationalError):
            connection.execute('DELETE FROM metadata')
    finally:
        connection.close()
    missing = tmp_path / 'missing.sqlite'
    with pytest.raises(sqlite3.OperationalError):
        io.db(missing)
    assert not missing.exists()
