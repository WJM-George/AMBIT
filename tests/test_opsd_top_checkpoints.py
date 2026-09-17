import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import torch

from stable_audio_tools.training.transfusion_opsd.top_checkpoints import (
    METRIC_DIRECTIONS, RANKING, _sha, relative_metric_score, retain_top_checkpoints,
)


class TopCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.config = self.out / 'config.json'
        self.config.write_text(json.dumps(dict(validation_ordinals=[1, 2], evaluation_seeds=[7, 8])))
        self.baseline = {name: .8 if name == 'Paired CLAP' else float(i + 1)
                         for i, name in enumerate(METRIC_DIRECTIONS)}
        self.policy = dict(keep=5, ranking=RANKING)
        self.eval_path(0).write_text(json.dumps(self.evaluation(0, self.baseline)))

    def eval_path(self, step):
        return self.out / f'EVALUATION_step{step:06d}.json'

    def evaluation(self, step, metrics):
        return dict(step=step, requests=2, outputs=4, metrics=metrics, target_audio_in_inference=False)

    def save(self, step, gain):
        metrics = {name: value * (1 + METRIC_DIRECTIONS[name] * gain)
                   for name, value in self.baseline.items()}
        self.eval_path(step).write_text(json.dumps(self.evaluation(step, metrics)))
        state = dict(step=step, model_sha256=f'model-{step}', config_sha256=_sha(self.config),
                     model={'test': torch.tensor([step])}, optimizer={'step': step})
        temp = self.out / 'new.pt'
        torch.save(state, temp)
        temp.replace(self.out / 'resume_latest.pt')
        (self.out / 'RESUME.json').write_text(json.dumps(dict(step=step, model_sha256=f'model-{step}')))

    def retain(self, step):
        return retain_top_checkpoints(self.out, self.eval_path(step), self.config, self.policy)

    def test_relative_directions_and_units(self):
        metrics = {name: value * (1 + METRIC_DIRECTIONS[name] * .01)
                   for name, value in self.baseline.items()}
        score = relative_metric_score(metrics, self.baseline)
        self.assertAlmostEqual(score['score'], 1.)
        self.assertEqual(score['improved_metrics'], 9)
        self.assertTrue(score['all_nine_improved'])
        self.assertTrue(score['priority_seven_all_improved'])
        self.assertAlmostEqual(relative_metric_score(self.baseline, self.baseline)['score'], 0.)

    def test_high_average_does_not_hide_a_priority_metric_regression(self):
        metrics = {name: value * (1 + METRIC_DIRECTIONS[name] * .01)
                   for name, value in self.baseline.items()}
        metrics['CRW'] = self.baseline['CRW'] * .5
        metrics['LSD'] = self.baseline['LSD'] * 1.001
        score = relative_metric_score(metrics, self.baseline)
        self.assertGreater(score['score'], 0.)
        self.assertEqual(score['improved_metrics'], 8)
        self.assertEqual(score['priority_seven_improved_metrics'], 6)
        self.assertFalse(score['all_nine_improved'])
        self.assertFalse(score['priority_seven_all_improved'])
        self.assertLess(score['worst_relative_improvement_percent'], 0.)

    def test_best_five_survive_rotation_and_eviction_preserves_milestone(self):
        self.retain(0)
        for step, gain in enumerate([.05, .01, .04, .02, .03, .10, .08], 1):
            self.save(step, gain)
            if step == 2:
                (self.out / 'checkpoints').mkdir()
                os.link(self.out / 'resume_latest.pt', self.out / 'checkpoints/step-00000002.pt')
            ledger = self.retain(step)
        self.assertEqual(ledger['ranked_steps'], [6, 7, 1, 3, 5])
        self.assertEqual(len(list((self.out / 'top_checkpoints').glob('*.pt'))), 5)
        self.assertFalse((self.out / 'top_checkpoints/step-00000002.pt').exists())
        self.assertTrue((self.out / 'checkpoints/step-00000002.pt').exists())
        retained = torch.load(self.out / 'top_checkpoints/step-00000001.pt', weights_only=False)
        self.assertEqual(retained['model']['test'].item(), 1)
        self.assertEqual(retained['optimizer']['step'], 1)
        self.assertEqual(len(ledger['history']), 7)
        self.assertEqual(self.retain(7)['ranked_steps'], ledger['ranked_steps'])

    def test_tie_prefers_earlier_update_and_original_is_not_a_trained_slot(self):
        self.assertEqual(self.retain(0)['ranked_steps'], [])
        self.policy['keep'] = 1
        # Start a separate fixed policy before any candidate has been selected.
        (self.out / 'top_checkpoints/INDEX.json').unlink()
        for step in [1, 2]:
            self.save(step, .01)
            ledger = self.retain(step)
        self.assertEqual(ledger['ranked_steps'], [1])

    def test_checkpoint_must_match_evaluation(self):
        self.save(1, .01)
        self.eval_path(2).write_text(json.dumps(self.evaluation(2, self.baseline)))
        with self.assertRaises(ValueError):
            self.retain(2)

    def test_incomplete_or_changed_panel_and_changed_baseline_rejected(self):
        self.save(1, .01)
        self.retain(1)
        bad = self.evaluation(2, self.baseline)
        bad['outputs'] = 3
        self.eval_path(2).write_text(json.dumps(bad))
        with self.assertRaises(ValueError):
            self.retain(2)
        changed = dict(self.baseline, CRW=self.baseline['CRW'] * 1.1)
        self.eval_path(0).write_text(json.dumps(self.evaluation(0, changed)))
        with self.assertRaises(ValueError):
            self.retain(1)

    def test_missing_or_nonfinite_metrics_do_not_select_a_checkpoint(self):
        for value in (math.nan, math.inf):
            with self.assertRaises(ValueError):
                relative_metric_score(dict(self.baseline, FAD=value), self.baseline)
        missing = dict(self.baseline)
        del missing['CRW']
        with self.assertRaises(KeyError):
            relative_metric_score(missing, self.baseline)

    def test_unmanaged_files_are_never_pruned(self):
        self.retain(0)
        other = self.out / 'top_checkpoints/not_owned.pt'
        other.write_bytes(b'other task')
        for step in range(1, 8):
            self.save(step, step * .01)
            self.retain(step)
        self.assertEqual(other.read_bytes(), b'other task')


if __name__ == '__main__':
    unittest.main()
