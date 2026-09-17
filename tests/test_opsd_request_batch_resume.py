import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.t2a.rl.train_editing_opsd_stream import sha, validate_resume_configuration
from stable_audio_tools.training.transfusion_opsd.editing_stream import OrdinalStream


class RequestBatchResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent_path = Path(self.temp.name) / 'parent.json'
        self.current_path = Path(self.temp.name) / 'current.json'
        self.parent = dict(physical_gpus=list(range(8)), request_rows_per_rank=1,
            global_request_batch=8, paired_rows_per_rank=64, global_paired_batch=512,
            maximum_updates=2000, save_every=500, recovery_every_seconds=600,
            learning_rates={'shared': 2e-6}, request_fraction=.1, connected_credit=False)
        self.parent_path.write_text(json.dumps(self.parent))
        self.current = copy.deepcopy(self.parent)
        self.current.update(request_rows_per_rank=2, global_request_batch=16,
            resume_config_parent=str(self.parent_path), request_batch_transition=dict(
                after_update=569, previous_global_request_batch=8, global_request_batch=16))

    def validate(self, config=None, *, step=569, digest=None):
        self.current_path.write_text(json.dumps(self.current if config is None else config))
        validate_resume_configuration(self.current_path, digest or sha(self.parent_path), checkpoint_step=step)

    def test_resume_at_declared_boundary_and_later_checkpoint(self):
        self.validate()
        validate_resume_configuration(self.current_path, sha(self.current_path), checkpoint_step=570)

    def test_wrong_or_missing_boundary_rejected(self):
        for step in (568, 570, None):
            with self.subTest(step=step), self.assertRaises(ValueError):
                self.validate(step=step)
        self.current.pop('request_batch_transition')
        with self.assertRaises(ValueError):
            self.validate()

    def test_unverified_parent_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(digest='0' * 64)

    def test_batch_change_cannot_smuggle_other_recipe_changes(self):
        for key, value in [('learning_rates', {'shared': 1e-5}), ('request_fraction', .2),
                           ('global_paired_batch', 1024), ('connected_credit', True),
                           ('physical_gpus', [0, 1, 2, 3]), ('maximum_updates', 4000)]:
            changed = copy.deepcopy(self.current)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(changed)

    def test_inconsistent_or_nonpositive_request_batch_rejected(self):
        for per_rank in (1, 0, 2.0):
            changed = copy.deepcopy(self.current)
            changed['request_rows_per_rank'] = per_rank
            with self.subTest(per_rank=per_rank), self.assertRaises(ValueError):
                self.validate(changed)

    def test_schedule_only_resume_still_supported(self):
        changed = copy.deepcopy(self.parent)
        changed.update(maximum_updates=2500, resume_config_parent=str(self.parent_path))
        self.validate(changed)

    def test_sampler_continues_without_replaying_or_skipping_rows(self):
        streams = [OrdinalStream(range(512), seed=37, rank=r, world=8) for r in range(8)]
        used = [row for stream in streams for row in stream.take(5)]
        resumed = [OrdinalStream(range(512), seed=37, rank=r, world=8, state=s.state_dict())
                   for r, s in enumerate(streams)]
        next_rows = [row for stream in resumed for row in stream.take(2)]
        expected = [row for stream in streams for row in stream.take(2)]
        self.assertEqual(next_rows, expected)
        tail = [row for stream in resumed for row in stream.take(len(stream.order) - stream.cursor)]
        self.assertEqual(sorted(used + next_rows + tail), list(range(512)))


if __name__ == '__main__':
    unittest.main()
