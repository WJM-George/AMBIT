import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from triton.runtime.autotuner import Autotuner
from scripts.t2a.rl.evaluate_editing_opsd_branch_cross import finish_runtime


class CachedEvaluationRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        self.learner = SimpleNamespace(
            q=dict(evaluation_save_audio=False, validation_ordinals=[3, 4], evaluation_seeds=[11, 12]),
            out=self.path, runtime_dir=self.path, rank=0, world=1, step=500,
            finish_numerics=Mock(), restore_flash=Mock())
        self.original = Autotuner.run

    def result(self, resumed):
        (self.path / 'eval_step000500_rank0.json').write_text(json.dumps(
            dict(rows=[dict(ordinal=i, seed=s) for i in (3, 4) for s in (11, 12)],
                 resumed_outputs=resumed)))

    def test_fully_cached_evaluation_restores_hook_without_fake_native_calls(self):
        self.result(4)
        with patch.object(Autotuner, 'run', Mock()):
            finish_runtime(self.learner, self.original, evaluation_completed=True)
            self.assertIs(Autotuner.run, self.original)
        self.learner.finish_numerics.assert_not_called()
        self.learner.restore_flash.assert_called_once()
        self.assertTrue((self.path / 'CACHED_EVALUATION_ONLY.json').exists())

    def test_any_new_output_requires_normal_native_audit(self):
        self.result(3)
        finish_runtime(self.learner, self.original, evaluation_completed=True)
        self.learner.finish_numerics.assert_called_once()
        self.learner.restore_flash.assert_called_once()
        self.assertFalse((self.path / 'CACHED_EVALUATION_ONLY.json').exists())

    def test_failed_generation_does_not_bypass_native_audit_and_still_restores_flash(self):
        self.learner.finish_numerics.side_effect = AssertionError('native audit failed')
        with self.assertRaisesRegex(AssertionError, 'native audit failed'):
            finish_runtime(self.learner, self.original, evaluation_completed=False)
        self.learner.finish_numerics.assert_called_once()
        self.learner.restore_flash.assert_called_once()


if __name__ == '__main__':
    unittest.main()
