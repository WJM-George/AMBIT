"""Isolated spatial/multi-source expansion. Never mutate frozen predecessors."""
import hashlib
import json
import os
import sqlite3
import sys
import zlib
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parents[1]
PREVIOUS = BASE / 'revisions/addition250k_20260910'
REPO = Path(__file__).resolve().parents[3]
PYTHON = '/mnt/sdc/stable-audio-tools-venv/bin/python'
OLD_RUN = BASE / 'materialized/logs/takeover-20260905T095122+0800'
LAUNCHER = OLD_RUN / 'editing_allocated_gpu_runtime_v1.py'
REPORT = Path('/home/tanhe/dataset_storage/reports/editing_spatial_multi500k_20260912')
sys.path.insert(0, str(REPO))

from stable_audio_tools.data.artifact_io import now, canonical, digest, sha, read, write, unpack, db, pack
GPUS = [2, 3, 4]
COUNTS = {'train': 500000, 'validation': 10000, 'test': 2500}
SHARD_SIZE = 128
DIAGONALS = {'front_left':45., 'front_right':-45., 'rear_left':135., 'rear_right':-135.}
COMPASS = {'front':0., 'front_left':45., 'left':90., 'rear_left':135., 'rear':-180., 'rear_right':-135., 'right':-90., 'front_right':-45.}
TRAIN_TASKS = {'diagonal_relocation':266667, 'diagonal_start_motion':133333,
               'dual_relocation':35000, 'position_swap':30000,
               'dual_start_motion':10000, 'dual_stop_motion':10000,
               'mixed_motion_toggle':10000, 'three_position_cycle':5000}
OPERATIONS = {'diagonal_relocation':'stationary_spatial_relocation',
              'diagonal_start_motion':'static_to_linear',
              'dual_relocation':'multi_source_relocation',
              'position_swap':'source_position_swap',
              'dual_start_motion':'multi_source_static_to_linear',
              'dual_stop_motion':'multi_source_linear_to_static',
              'mixed_motion_toggle':'multi_source_motion_toggle',
              'three_position_cycle':'multi_source_position_cycle'}

def seed(*parts):
    return int(digest([ROOT.name, *parts])[:16], 16)

def allocation(weights, total, caps=None):
    """Largest remainder, with optional integer capacities."""
    caps = {k:total for k in weights} if caps is None else dict(caps)
    result = {k:0 for k in weights}
    left = total
    while left:
        active = {k:w for k,w in weights.items() if w > 0 and result[k] < caps.get(k, 0)}
        if not active: raise RuntimeError(f'Insufficient allocation capacity: missing {left}')
        denom = sum(active.values())
        amounts = {k:min(caps[k]-result[k], left*w//denom) for k,w in active.items()}
        granted = sum(amounts.values())
        for k,n in amounts.items(): result[k] += n
        left -= granted
        if not left: break
        order = sorted(active, key=lambda k:(-((left+granted)*active[k] % denom), str(k)))
        for k in order:
            if result[k] < caps[k]: result[k] += 1; left -= 1
            if not left: break
    assert sum(result.values()) == total
    return result

def tasks_for(split):
    return allocation(TRAIN_TASKS, COUNTS[split])

def state(stage, **extra):
    value = dict(stage=stage, pid=os.getpid(), observed_at=now(), **extra)
    write(ROOT/'BUILD_STATE.json', value)
    print(canonical(value), flush=True)

def cpu_env():
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    return env
