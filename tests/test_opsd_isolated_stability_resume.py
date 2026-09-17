import copy

import pytest

from scripts.t2a.rl.train_editing_opsd_stability import validate_isolated_resize


def configs():
    parent = dict(output='/tmp/prior_opsd', physical_gpus=[6, 7],
                  request_rows_per_rank=8, global_request_batch=16,
                  paired_rows_per_rank=256, global_paired_batch=512,
                  learning_rates={'shared_Transformer': 3.75e-7},
                  selective_recipe={'reference_native_prefix': False, 'select_same_plan_improvements': True},
                  base_checkpoint={'sha256': 'original'}, request_fraction=.1, maximum_updates=2000)
    current = copy.deepcopy(parent)
    current.update(output='/tmp/continued_opsd', physical_gpus=[4, 5, 6, 7],
                   request_rows_per_rank=4, paired_rows_per_rank=128,
                   resize_parent_config='/tmp/prior_opsd.json',
                   top_checkpoint_policy=dict(keep=1, ranking='mean_signed_relative_improvement_percent'),
                   checkpoint_retention=dict(rolling_recoveries=1))
    return current, parent


def test_isolated_storage_preserves_recipe_and_global_batches():
    current, parent = configs()
    validate_isolated_resize(current, parent)


@pytest.mark.parametrize('field,value', [
    ('learning_rates', {'shared_Transformer': 2e-6}),
    ('selective_recipe', {'reference_native_prefix': True, 'select_same_plan_improvements': True}),
    ('base_checkpoint', {'sha256': 'different'}),
    ('request_fraction', .2),
    ('maximum_updates', 4000),
    ('paired_rows_per_rank', 256),
    ('request_rows_per_rank', 8),
    ('output', '/tmp/prior_opsd'),
    ('physical_gpus', [0, 1, 2, 3]),
])
def test_migration_rejects_model_recipe_data_batch_or_placement_changes(field, value):
    current, parent = configs()
    current[field] = value
    with pytest.raises(ValueError):
        validate_isolated_resize(current, parent)
