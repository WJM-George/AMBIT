import copy
import json

import pytest

from scripts.t2a.rl.launch_editing_opsd_continuation import validate_checkpoint, audit_first_update


def matching_state():
    q = dict(base_checkpoint={'sha256': 'base'}, initial_overlay=None, physical_gpus=[4, 5, 6, 7],
        connected_credit=False, global_request_batch=16, request_rows_per_rank=4,
        global_paired_batch=512, paired_rows_per_rank=128, paired_microbatch=48,
        native_plan_audit_every=25, learning_rates={'AR': 5e-6, 'DiT': 3.75e-7}, seed=12)
    execution = {k: q[k] for k in ('request_rows_per_rank', 'global_request_batch',
        'paired_rows_per_rank', 'paired_microbatch', 'global_paired_batch', 'native_plan_audit_every')}
    state = dict(step=100, world_size=4, config_sha256='config', original_checkpoint=q['base_checkpoint'],
        initial_overlay=None, execution=execution, execution_schedule=None,
        optimizer=dict(param_groups=[{'group_name': k, 'lr': v} for k, v in q['learning_rates'].items()],
                       state={0: dict(step=100, exp_avg=0, exp_avg_sq=0)}), rank_states=[])
    for rank in range(4):
        state['rank_states'].append(dict(
            request=dict(rank=rank, world=4, seed=13), paired=dict(rank=rank, world=4, seed=14),
            random=(), numpy=(), cpu_rng=(), cuda_rng=()))
    return q, state


def test_same_recipe_retains_four_ranks_and_adam():
    q, state = matching_state()
    validate_checkpoint(q, state, 'config')


@pytest.mark.parametrize('field,value', [('step', 0), ('world_size', 2),
    ('config_sha256', 'different'), ('original_checkpoint', {'sha256': 'different'})])
def test_other_checkpoint_or_topology_is_rejected(field, value):
    q, state = matching_state()
    state[field] = value
    with pytest.raises(ValueError):
        validate_checkpoint(q, state, 'config')


@pytest.mark.parametrize('mutation', ['lr', 'moments', 'adam_step', 'rng', 'sampler', 'microbatch'])
def test_resume_cannot_reset_or_change_training_state(mutation):
    q, state = matching_state()
    if mutation == 'lr':
        state['optimizer']['param_groups'][0]['lr'] *= 2
    elif mutation == 'moments':
        state['optimizer']['state'][0].pop('exp_avg')
    elif mutation == 'adam_step':
        state['optimizer']['state'][0]['step'] = 0
    elif mutation == 'rng':
        state['rank_states'][2].pop('cuda_rng')
    elif mutation == 'sampler':
        state['rank_states'][2]['request']['rank'] = 0
    elif mutation == 'microbatch':
        state['execution']['paired_microbatch'] = 32
    with pytest.raises(ValueError):
        validate_checkpoint(q, state, 'config')


def sampler_audit_fixture(tmp_path):
    output = tmp_path / 'training'
    output.mkdir()
    expected = dict(ranks={})
    p = dict(training_output=str(output), expected_first_update=str(tmp_path / 'expected.json'),
             update_log_offsets={}, checkpoint={'step': 100})
    for rank in range(4):
        previous = 'old updates are not parsed\n'
        expected['ranks'][str(rank)] = dict(step=101, request_ordinals=[rank], paired_ordinals=[rank + 4])
        record = dict(step=101, request_updates=[dict(ordinal=rank)], paired_ordinals=[rank + 4], clip_norms={})
        (output / f'UPDATES_rank{rank}.jsonl').write_text(previous + json.dumps(record) + '\n')
        p['update_log_offsets'][str(rank)] = len(previous.encode())
    (tmp_path / 'expected.json').write_text(json.dumps(expected))
    return p


def test_first_resumed_update_uses_saved_cursors_after_existing_logs(tmp_path):
    p = sampler_audit_fixture(tmp_path)
    assert audit_first_update(tmp_path, p)
    assert (tmp_path / 'FIRST_RESUMED_UPDATE_AUDIT.json').exists()


def test_replayed_or_skipped_first_sample_is_rejected(tmp_path):
    p = sampler_audit_fixture(tmp_path)
    file = tmp_path / 'training/UPDATES_rank2.jsonl'
    file.write_text(file.read_text().replace('"ordinal": 2', '"ordinal": 99'))
    with pytest.raises(ValueError, match='repeated, skipped or changed'):
        audit_first_update(tmp_path, p)


def test_pending_first_update_is_not_reported_as_verified(tmp_path):
    p = sampler_audit_fixture(tmp_path)
    file = tmp_path / 'training/UPDATES_rank2.jsonl'
    file.write_text(file.read_text()[:p['update_log_offsets']['2']] + '{"step": 101')
    assert not audit_first_update(tmp_path, p)
    with pytest.raises(ValueError, match='Missing first resumed'):
        audit_first_update(tmp_path, p, required=True)
