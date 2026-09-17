import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.t2a.rl.launch_editing_opsd_0915_lr import validate_configuration
from scripts.t2a.rl.train_editing_opsd_0915_recipe import base, current, load_recipe, RUNNER_BINDINGS
from stable_audio_tools.paths import ckpt_path
from test_opsd_native_token_alignment import codec, native_plan, alternative_segmentation

RUN = ckpt_path('transfusion_opsd/editing_opsd_0915_lr_alignment_20260916_v1')


@pytest.fixture
def experiment():
    if not (RUN / 'PROTOCOL.json').exists():
        pytest.skip('Archived0915 diagnostic artifacts are not installed.')
    read = lambda path: json.loads(Path(path).read_text())
    p = read(RUN / 'PROTOCOL.json')
    return p, read(RUN / 'legacy_current_lr.json'), read(p['control_config']), read(p['archived_config'])


def test_archived_objective_has_one_runner_and_does_not_mutate_other_recipes(experiment):
    p, _, _, _ = experiment
    before = {key: getattr(base, key) for key in RUNNER_BINDINGS}
    search_path = sys.path[:]
    learner, _ = load_recipe(p['legacy_recipe'])
    assert sys.path == search_path
    assert before == {key: getattr(base, key) for key in RUNNER_BINDINGS}
    assert learner.__bases__[0].__bases__ == current.SpatialLearner.__bases__
    assert learner.collect.__code__.co_filename == p['legacy_recipe']['trainer']['path']
    assert learner.backward_extra.__code__.co_filename == p['legacy_recipe']['trainer']['path']


def test_recipe_source_mismatch_fails_before_model_loading(experiment):
    p, _, _, _ = experiment
    recipe = copy.deepcopy(p['legacy_recipe'])
    recipe['trainer']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='source changed'):
        load_recipe(recipe)


@pytest.mark.parametrize('new_field', ['complete_recipe', 'selective_recipe', 'initialization_checkpoint'])
def test_new_objectives_or_initialization_cannot_silently_enter_old_recipe(experiment, new_field, tmp_path):
    p, q, _, _ = experiment
    learner, _ = load_recipe(p['legacy_recipe'])
    q[new_field] = {}
    with pytest.raises(ValueError, match='without newer objectives'):
        learner(q, 0, 2, tmp_path)


def test_full0915_keeps_unverified_proposed_fields_anchored(experiment, monkeypatch):
    p, _, _, _ = experiment
    learner, _ = load_recipe(p['legacy_recipe'])
    original = current.SpatialLearner.__bases__[0]
    decision = {'field': 'start/azimuth_deg'}
    item = dict(obs=object(), tokens=torch.tensor([1, 2]), plan={}, decision=decision, metrics=[])
    monkeypatch.setattr(original, 'collect', lambda self, ordinal: item)
    observed = []
    def reference_targets(*args, **kwargs):
        observed.append(kwargs['decision'])
        return [], []
    monkeypatch.setitem(learner.collect.__globals__, 'reference_field_targets', reference_targets)
    instance = object.__new__(learner)
    instance.progress = lambda *args, **kwargs: None
    instance.reference = SimpleNamespace(student_logits=lambda *args: torch.zeros(1, 1, 3))
    instance.adapter = SimpleNamespace(codec=None)
    instance.q = {'spatial_recipe': {'reference_cone_deg': 12.}, 'base_checkpoint': {'sha256': 'reference'}}
    instance.collect(0)
    assert observed == [None]
    assert item['reference_exempt_field'] is None


def test_matched_configuration_accepts_only_declared_recipe_and_lr_changes(experiment):
    _, q, control, old = experiment
    validate_configuration(q, control, old, 'legacy_current_lr')


@pytest.mark.parametrize('field,value', [
    ('learning_rates', {'shared_Transformer': 2e-6}),
    ('seed', 42), ('request_rows_per_rank', 4), ('paired_rows_per_rank', 128),
    ('paired_microbatch', 32), ('request_fraction', .2), ('connected_credit', True),
    ('physical_gpus', [0, 1]),
])
def test_unmatched_lr_data_or_gpu_configuration_is_rejected(experiment, field, value):
    _, q, control, old = experiment
    q[field] = value
    with pytest.raises(ValueError):
        validate_configuration(q, control, old, 'legacy_current_lr')


@pytest.mark.parametrize('segmented', [False, True])
def test_archived_proposal_keeps_the_actual_native_prefix(experiment, codec, segmented):
    p, _, _, _ = experiment
    _, retention = load_recipe(p['legacy_recipe'])
    plan, ids = native_plan(codec)
    if segmented:
        ids = alternative_segmentation(codec, ids)
    adapter = SimpleNamespace(codec=codec, allowed_next_ids=lambda obs, prefix: codec.allowed_next_ids(prefix))
    result = retention.propose_current_decision(adapter, SimpleNamespace(model_num_samples=3 * 44100),
        plan, ids, dict(kind='music', azimuths=[10.], elevations=[], distances=[], activity=None), step=0)
    assert result['prefix'] == ids[:result['position']]
    assert result['position'] == ids.index(codec._tid('<azimuth_bin>')) + 1
    changed = ids.copy()
    changed[result['position']] = result['choice_ids'][1]
    assert sum(a != b for a, b in zip(ids, changed)) == 1
    from stable_audio_tools.training.transfusion_opsd.native_token_alignment import validate_native_plan_tokens
    validate_native_plan_tokens(codec, changed, result['plans'][1])
    assert result['plans'][1]['sources'][0]['description'] == plan['sources'][0]['description']
