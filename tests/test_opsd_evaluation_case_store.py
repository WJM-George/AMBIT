import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from stable_audio_tools.training.transfusion_opsd.evaluation_case_store import EvaluationCaseStore


class EvaluationCaseStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.identity = dict(model_sha256='model-A', ordinals=[3, 4], seeds=[11, 12])
        self.store = EvaluationCaseStore(self.root / 'cases', self.identity)
        features = {key: np.arange(1, 5, dtype=np.float32) for key in self.store.feature_keys}
        self.record = dict(ordinal=3, seed=11, audio=None,
                           scalar={key: 1. for key in self.store.scalar_keys})
        for key in ('features', 'target_features'):
            path = self.root / (key + '.npz')
            np.savez_compressed(path, **features)
            self.record[key] = str(path)

    def test_committed_features_resume_without_audio(self):
        self.store.put(self.record)
        reopened = EvaluationCaseStore(self.root / 'cases', self.identity)
        self.assertEqual(reopened.get(3, 11), self.record)
        self.assertEqual(list(self.root.rglob('*.wav')), [])

    def test_features_without_commit_are_recomputed(self):
        self.assertIsNone(self.store.get(3, 11))

    def test_missing_or_corrupt_generated_features_are_not_reused(self):
        self.store.put(self.record)
        path = Path(self.record['features'])
        path.write_bytes(b'interrupted write')
        self.assertIsNone(self.store.get(3, 11))
        path.unlink()
        self.assertIsNone(self.store.get(3, 11))

    def test_replaced_target_features_are_not_reused(self):
        self.store.put(self.record)
        Path(self.record['target_features']).write_bytes(b'different target')
        with self.assertRaises(ValueError):
            self.store.get(3, 11)

    def test_model_or_panel_change_rejected(self):
        for change in [dict(model_sha256='model-B'), dict(seeds=[13, 14]), dict(ordinals=[4, 5])]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                EvaluationCaseStore(self.root / 'cases', dict(self.identity, **change))

    def test_incomplete_features_cannot_be_committed(self):
        np.savez_compressed(self.record['features'], clap_audio=np.ones(3))
        with self.assertRaises(ValueError):
            self.store.put(self.record)
        self.assertFalse(self.store.path(3, 11).exists())

    def test_nonfinite_metrics_cannot_be_committed(self):
        self.record['scalar']['GCC'] = float('nan')
        with self.assertRaises(ValueError):
            self.store.put(self.record)
        self.assertFalse(self.store.path(3, 11).exists())

    def test_wrong_sample_in_record_rejected(self):
        self.store.put(self.record)
        path = self.store.path(3, 11)
        value = json.loads(path.read_text())
        value['record']['ordinal'] = 4
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            self.store.get(3, 11)


if __name__ == '__main__':
    unittest.main()
