"""Recheck the archived0915 objective with a separately declared learning rate.

The archived proposal, retention, self-training and decoded-FOA objectives
are loaded by hash. The common runner retains current checkpoint and sampler
fixes. Only evaluation/storage methods come from the current SpatialLearner;
the CompleteLearner additions are never instantiated during training.
"""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.t2a.rl import train_editing_opsd_spatial as current

base = current.base
RUNNER_BINDINGS = ('Learner', 'request_facts', 'propose_current_decision', 'request_spatial_measure')


def verified_source(identity):
    path = Path(identity['path'])
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != identity['sha256']:
        raise ValueError('Archived0915 recipe source changed: ' + str(path))
    return path, content.decode('utf-8')


def load_recipe(recipe):
    """Build the old learner without leaving changes in the common runner."""
    retention_path, _ = verified_source(recipe['retention'])
    trainer_path, source = verified_source(recipe['trainer'])
    name = 'stable_audio_tools.training.transfusion_opsd._archived0915_' + recipe['retention']['sha256'][:16]
    spec = importlib.util.spec_from_file_location(name, retention_path)
    retention = importlib.util.module_from_spec(spec)
    sys.modules[name] = retention
    spec.loader.exec_module(retention)
    original_import = 'from stable_audio_tools.training.transfusion_opsd.editing_spatial_retention import ('
    if source.count(original_import) != 1:
        raise ValueError('Unexpected archived training module import.')
    source = source.replace(original_import, 'from ' + name + ' import (')
    archived = types.ModuleType('_archived0915_trainer')
    archived.__file__ = str(trainer_path)
    saved = {key: getattr(base, key) for key in RUNNER_BINDINGS}
    search_path = sys.path[:]
    try:
        # Other entrypoints replace base.Learner at import time. The archived
        # class must inherit the original stream runner, never another recipe.
        base.Learner = current.SpatialLearner.__bases__[0]
        exec(compile(source, str(trainer_path), 'exec'), archived.__dict__)
    finally:
        sys.path[:] = search_path
        for key, value in saved.items():
            setattr(base, key, value)

    class Learner0915(archived.SpatialLearner):
        evaluate = current.SpatialLearner.evaluate
        record_evaluation = current.SpatialLearner.record_evaluation

        def __init__(self, q, rank, world, out):
            if any(key in q for key in ('complete_recipe', 'selective_recipe', 'initialization_checkpoint')):
                raise ValueError('The0915 recipe must start at original40k without newer objectives.')
            super().__init__(q, rank, world, out)
            base.write(out / f'ARCHIVED_RECIPE_rank{rank}.json', dict(
                recipe=recipe, learning_rates=q['learning_rates'], optimizer='new_AdamW',
                frozen_reference=q['base_checkpoint'], initial_step=self.step,
                training_class=str(archived.SpatialLearner),
                native_parameter_scope='AR_and_DiT_and_shared_Transformer',
                request_target_access=False, connected_credit=False,
                reproduction_scope='Archived0915 objective and teacher rules; declared native-token alignment fix; current common runtime and evaluator.'))

    return Learner0915, retention


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=Path, required=True)
    args, _ = parser.parse_known_args()
    q = base.read(args.config)
    learner, retention = load_recipe(q['legacy_recipe'])
    base.Learner = learner
    for key in RUNNER_BINDINGS[1:]:
        setattr(base, key, getattr(retention, key))
    base.main()


if __name__ == '__main__':
    main()
