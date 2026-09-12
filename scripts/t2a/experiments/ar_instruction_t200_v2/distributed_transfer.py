"""Declared single-rank to three-rank AR continuation; no exact-resume claim.

Model, AdamW and scheduler state survive the transfer. A new sampler epoch
and two new RNG streams are explicit changes. The original GPU7 stream moves
from logical rank0 to rank2, keeping its physical GPU and every RNG byte.
"""
import copy
import math
import random

import numpy as np
import torch

from scripts.t2a.train.train_sceneplan_transfusion_editing_ar_joint_full import (
    JOINT_RNG_SCHEMA, JOINT_RNG_SCHEMA_VERSION,
)

SCHEMA = 'editing_ar_T200_three_rank_transfer_v1'
GPUS = [5, 6, 7]


def validate_policy(cfg):
    policy = cfg['distributed_transfer']
    assert policy['schema'] == SCHEMA
    assert policy['physical_gpus'] == GPUS and policy['world_size'] == 3
    assert policy['parent_physical_gpus'] == [7] and policy['parent_world_size'] == 1
    assert policy['inherited_rng_rank'] == 2
    assert policy['data_start'] == {'epoch': 0, 'next_batch': 0}
    assert policy['data_policy'] == 'new_native_three_rank_epoch; parent_cursor_not_reinterpreted'
    assert policy['scheduler_policy'] == 'preserve_state; continuous_LR_rewarm_then_cosine'
    assert 0 < policy['start_lr_factor'] <= 1
    assert 0 < policy['rewarm_steps'] < cfg['schedule']['max_steps'] - policy['parent_step']
    assert cfg['schedule']['short_batch_size'] == 8 and cfg['schedule']['long_batch_size'] == 5
    assert cfg['schedule']['gradient_accumulation'] == 4
    assert cfg['instruction_data']['mode'] == 't200'
    assert cfg['lambda_ar'] == 1 and cfg['lambda_rf'] == 0
    return policy


def lr_multiplier(step, cfg):
    """Keep the loaded LR continuous without resetting AdamW or its step count."""
    policy = cfg['distributed_transfer']
    offset = max(0, int(step) - policy['parent_step'])
    rewarm = policy['rewarm_steps']
    if offset <= rewarm:
        return policy['start_lr_factor'] + (1. - policy['start_lr_factor']) * offset / rewarm
    span = cfg['schedule']['max_steps'] - policy['parent_step'] - rewarm
    progress = min(1., (offset - rewarm) / span)
    return .1 + .9 * .5 * (1. + math.cos(math.pi * progress))


def rng_for_rank(parent_states, *, rank, cfg, device):
    policy = validate_policy(cfg)
    assert len(parent_states) == 1 and parent_states[0]['rank'] == 0
    assert rank in range(3) and device.type == 'cuda'
    if rank == policy['inherited_rng_rank']:
        return {**copy.deepcopy(parent_states[0]), 'rank': rank}
    seed = policy['new_rank_seed'] + 10003 * rank
    numpy = np.random.RandomState(seed % (2**32)).get_state()
    return {
        'schema': JOINT_RNG_SCHEMA, 'schema_version': JOINT_RNG_SCHEMA_VERSION,
        'rank': rank, 'python_random_state': random.Random(seed).getstate(),
        'numpy_random_state': {
            'bit_generator': numpy[0],
            'keys': torch.from_numpy(numpy[1].astype(np.int64)),
            'position': int(numpy[2]), 'has_gauss': int(numpy[3]),
            'cached_gaussian': float(numpy[4]),
        },
        'torch_cpu_rng_state': torch.Generator().manual_seed(seed).get_state(),
        'torch_cuda_rng_state': torch.Generator(device=device).manual_seed(seed).get_state().cpu(),
    }


def transferred_state(parent, rank_states, cfg):
    policy = validate_policy(cfg)
    assert parent['global_step'] == parent['scheduler']['last_epoch'] == policy['parent_step']
    assert [x['rank'] for x in rank_states] == [0, 1, 2]
    assert parent['run_contract']['world_size'] == 1 and parent['run_contract']['physical_gpus'] == [7]
    for group, base_lr, last_lr in zip(parent['optimizer']['param_groups'],
            parent['scheduler']['base_lrs'], parent['scheduler']['_last_lr'], strict=True):
        assert group['lr'] == last_lr and group['initial_lr'] == base_lr
        assert math.isclose(last_lr / base_lr, policy['start_lr_factor'], rel_tol=0., abs_tol=1e-15)
    # Large immutable tensors are reused, never copied into a second start file.
    return {**parent, 'rng_states_by_rank': rank_states, **policy['data_start']}
