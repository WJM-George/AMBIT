#!/usr/bin/env python3
"""Prepare replaceable, hash-verified RAM caches of immutable AR manifests."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import numpy as np

DATA = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/transfusion_shared_v1/generation_ar")
CACHE = Path(os.environ.get("AMBIT_CACHE_ROOT", "cache")) / "generation_ar_manifests"
EXPECTED = {'train': 'e1921b167a09135ae3c3bdf28ecac509f7175a387cb9c0f54b5d7a3608a1799c',
            'validation': '697113f9c38f190cb7e54bf8863c3e3b78dfa77de76daeb9b1ea7c035fa6dd4c'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    CACHE.mkdir(exist_ok=True)
    record = {'purpose': 'rebuildable_runtime_cache', 'manifests': {}, 'started_unix': time.time()}
    for split, expected in EXPECTED.items():
        target = CACHE / f'{split}.sqlite'
        if not target.exists():
            temporary = target.with_suffix('.copying-' + str(os.getpid()))
            shutil.copyfile(DATA / f'{split}.sqlite', temporary)
            if digest(temporary) != expected:
                raise RuntimeError('cache SHA mismatch: ' + split)
            temporary.chmod(0o444)
            os.replace(temporary, target)
        else:
            assert digest(target) == expected
        record['manifests'][split] = {'source': str(DATA / f'{split}.sqlite'), 'cache': str(target), 'sha256': expected}
        print(json.dumps({'event': 'cached', 'split': split, 'path': str(target)}), flush=True)
    db = sqlite3.connect(f"file:{CACHE / 'train.sqlite'}?mode=ro&immutable=1", uri=True)
    values = np.asarray(db.execute('SELECT ordinal,target_token_count,source_count FROM rows ORDER BY ordinal').fetchall(), dtype=np.int32)
    assert values.shape == (1600000,3) and np.array_equal(values[:,0], np.arange(1600000))
    db.close()
    np.savez(CACHE / 'train_columns.npz', lengths=values[:,1]-1, source_counts=values[:,2])
    record['columns_sha256'] = digest(CACHE / 'train_columns.npz')
    record['finished_unix'] = time.time()
    (CACHE / 'CACHE.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'event': 'complete', 'cache_manifest': str(CACHE / 'CACHE.json')}), flush=True)


if __name__ == '__main__':
    main()
