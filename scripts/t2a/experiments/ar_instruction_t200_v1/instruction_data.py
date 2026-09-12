"""Read a frozen text overlay while retaining native audio/target data readers.

The renderer and its old-plan labels are offline tools. This reader selects
only request text and identity hashes; it never reads the stored evidence blob
or old/new ScenePlan contents. It is opt-in and changes no native global class.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

CONTRACT = 'editing_instruction_overlay_native_row_adapter_v1'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def readonly(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro&immutable=1',
                         uri=True, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    return db


class InstructionOverlay:
    def __init__(self, path, *, expected_sha256, native_index_path, native_index_sha256,
                 expected_rows, split, allow_final_test=False):
        self.path = Path(path).resolve(strict=True)
        self.native_path = Path(native_index_path).resolve(strict=True)
        assert split in ('train', 'validation', 'test')
        assert split != 'test' or allow_final_test, 'Test is unavailable to development readers'
        self.split = split
        marker = json.loads(self.path.with_suffix('.sqlite.frozen.json').read_text())
        assert marker['schema'] == 'editing_instruction_overlay_v1' and marker['state'] == 'frozen'
        assert marker['sha256'] == expected_sha256 == sha(self.path)
        assert marker['native_index_sha256'] == native_index_sha256 == sha(self.native_path)
        assert Path(marker['native_index_path']).resolve() == self.native_path
        assert marker['split'] == split and marker['rows'] == expected_rows
        self.sha256 = expected_sha256; self.native_sha256 = native_index_sha256
        self._overlay = None; self._native = None
        db = readonly(self.path)
        try:
            metadata = dict(db.execute('SELECT key,value FROM metadata'))
            assert metadata['state'] == 'frozen' and metadata['split'] == split
            assert metadata['native_index_sha256'] == native_index_sha256
            assert db.execute('SELECT COUNT(*) FROM instructions').fetchone()[0] == expected_rows
        finally:
            db.close()

    def __getstate__(self):
        state = dict(self.__dict__); state['_overlay'] = None; state['_native'] = None
        return state

    def __del__(self):
        for name in ('_overlay', '_native'):
            db = getattr(self, name, None)
            if db is not None:
                db.close()

    def request(self, *, pair_ordinal, pair_id, operation, native_request):
        if self._overlay is None:
            self._overlay = readonly(self.path)
            self._native = readonly(self.native_path)
        row = self._overlay.execute(
            'SELECT pair_id,operation,template_id,raw_edit_request,instruction_sha256,'
            'native_instruction_sha256,new_sceneplan_sha256 FROM instructions WHERE pair_ordinal=?',
            (int(pair_ordinal),)).fetchone()
        original = self._native.execute(
            'SELECT pair_id,operation,instruction_sha256,new_sceneplan_sha256 FROM pairs WHERE pair_ordinal=?',
            (int(pair_ordinal),)).fetchone()
        assert row is not None and original is not None
        assert row['pair_id'] == original['pair_id'] == pair_id
        assert row['operation'] == original['operation'] == operation
        assert row['native_instruction_sha256'] == original['instruction_sha256'] == hashlib.sha256(native_request.encode()).hexdigest()
        assert row['new_sceneplan_sha256'] == original['new_sceneplan_sha256']
        assert row['instruction_sha256'] == hashlib.sha256(row['raw_edit_request'].encode()).hexdigest()
        return row['raw_edit_request'], row['template_id']

    def ar_row(self, native_row):
        text, _ = self.request(pair_ordinal=native_row['pair_ordinal'], pair_id=native_row['pair_id'],
                               operation=native_row['operation'], native_request=native_row['raw_edit_request'])
        return {**native_row, 'raw_edit_request': text}

    def joint_sample(self, native_sample):
        target, metadata, ar = native_sample
        assert metadata['raw_edit_request'] == ar['raw_edit_request']
        text, tid = self.request(pair_ordinal=ar['pair_ordinal'], pair_id=ar['pair_id'],
                                 operation=ar['operation'], native_request=ar['raw_edit_request'])
        return (target, {**metadata, 'raw_edit_request': text, 'instruction_template_id': tid},
                {**ar, 'raw_edit_request': text})


class DatasetWithInstructions:
    """Transparent map-style wrapper; native samplers and collators stay usable."""
    def __init__(self, dataset, overlay, *, joint):
        self.dataset = dataset; self.overlay = overlay; self.joint = bool(joint)
        self.semantic_caption_requires_epoch_key = False

    def __len__(self):
        return len(self.dataset)

    def length_bucket_indices(self):
        return self.dataset.length_bucket_indices()

    def __getitem__(self, index):
        item = self.dataset[index]
        return self.overlay.joint_sample(item) if self.joint else self.overlay.ar_row(item)
