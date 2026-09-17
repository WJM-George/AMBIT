"""Commit resumable metric records without retaining generated waveforms."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.',
                                         suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class EvaluationCaseStore:
    """A result is reusable only after its complete feature files are committed.

    The identity binds model, panel, seeds, inference settings and scorer source.
    Raw audio is unnecessary for the existing feature-based aggregate metrics.
    """
    feature_keys = {'clap_audio', 'panns_logits', 'panns_embedding', 'vggish_frames',
                    'crw_itd_ms', 'fsad_embedding_seconds'}
    scalar_keys = {'Paired CLAP', 'KL', 'LSD', 'GCC', 'CRW'}

    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(identity, sort_keys=True, allow_nan=False).encode()
        self.identity_sha256 = hashlib.sha256(encoded).hexdigest()
        binding = dict(schema='editing_metric_case_store_v1',
                       identity_sha256=self.identity_sha256, identity=identity)
        path = self.directory / 'IDENTITY.json'
        if path.exists():
            if json.loads(path.read_text()) != binding:
                raise ValueError('Evaluation model, panel or scoring identity changed.')
        else:
            atomic_json(path, binding)

    def path(self, ordinal, seed):
        return self.directory / f'{int(ordinal):07d}_{int(seed)}.json'

    def _validate_features(self, record):
        for key in ('features', 'target_features'):
            with np.load(record[key], allow_pickle=False) as values:
                if not self.feature_keys <= set(values.files):
                    raise ValueError('Incomplete metric feature file: ' + record[key])
                if any(not values[k].size or not np.isfinite(values[k]).all() for k in values.files):
                    raise ValueError('Empty or nonfinite evaluation features.')
        if (set(record['scalar']) != self.scalar_keys or
                not all(np.isfinite(v) for v in record['scalar'].values())):
            raise ValueError('Incomplete or nonfinite per-output metrics.')

    def get(self, ordinal, seed):
        path = self.path(ordinal, seed)
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        if value['identity_sha256'] != self.identity_sha256:
            raise ValueError('Cached evaluation belongs to a different model or panel.')
        record = value['record']
        if (record['ordinal'], record['seed']) != (ordinal, seed):
            raise ValueError('Cached evaluation sample identity changed.')
        if value['phase'] != 'COMPLETE':
            return None
        for key in ('features', 'target_features'):
            if not Path(record[key]).is_file():
                return None
            if file_sha256(record[key]) != value[key + '_sha256']:
                if key == 'target_features':
                    raise ValueError('Scoring target features changed; rebuild the target cache before resuming.')
                return None
        return record

    def put(self, record):
        self._validate_features(record)
        hashes = {}
        for key in ('features', 'target_features'):
            with Path(record[key]).open('rb') as handle:
                os.fsync(handle.fileno())
            hashes[key + '_sha256'] = file_sha256(record[key])
        atomic_json(self.path(record['ordinal'], record['seed']), dict(
            phase='COMPLETE', identity_sha256=self.identity_sha256, record=record, **hashes))
