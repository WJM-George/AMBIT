"""Isolated 1.75M-pair continuation; component datasets stay immutable."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import os
import sqlite3
import sys
import zlib

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]
PREVIOUS=BASE/'revisions/addition250k_20260910'
SPATIAL=BASE/'revisions/spatial_multi500k_20260912'
REPO=Path(__file__).resolve().parents[3]
OLD_RUN=BASE/'materialized/logs/takeover-20260905T095122+0800'
MAIN=OLD_RUN/'MAINLINE'
PYTHON='/mnt/sdc/stable-audio-tools-venv/bin/python'
PIPE=Path('/home/tanhe/dataset_storage/reports/editing_pipeline_20260912')
REPORT=Path('/home/tanhe/dataset_storage/reports/editing_dit_mixed1750k_30k_20260912')
sys.path.insert(1,str(REPO))

from stable_audio_tools.data.artifact_io import now, canonical, digest, sha, read, write, unpack, db
GPUS=[2,3,4]
STEPS=[50000,55000,60000,65000,70000,75000,80000]
COUNTS={'train':1750000,'validation':35000,'test':8750}
COMPONENT_COUNTS={'original':{'train':1000000,'validation':20000,'test':5000},
    'addition250k':{'train':250000,'validation':5000,'test':1250},
    'spatial_multi500k':{'train':500000,'validation':10000,'test':2500}}
SAMPLER={'steps':20,'cfg_scale':1.,'cfg_rescale_phi':0.,'grid':'editing_uniform'}
SCOPES=['validation','test_original','test_addition250k','test_spatial_multi500k']

def component_path(name,split):
    return (BASE/'training_index' if name=='original' else PREVIOUS/'addition_training_index'
        if name=='addition250k' else SPATIAL/'training_index')/f'{split}.sqlite'
def components(split,verify=True):
    result=[];offset=0
    for name,counts in COMPONENT_COUNTS.items():
        path=component_path(name,split);marker=read(str(path)+'.frozen.json')
        assert marker['rows']==counts[split] and marker['split']==split
        assert marker['state']=='materialized_complete_frozen' and Path(marker['index_path'])==path
        if verify:assert sha(path)==marker['index_sha256']
        result.append(dict(name=name,offset=offset,**marker));offset+=counts[split]
    assert offset==COUNTS[split]
    return dict(schema='three_immutable_editing_components_v1',split=split,rows=offset,components=result)
def verify_binding(binding):
    assert sum(c['rows'] for c in binding['components'])==binding['rows']
    for c in binding['components']:
        marker=read(c['index_path']+'.frozen.json')
        assert all(marker[k]==c[k] for k in marker)
        assert sha(c['index_path'])==c['index_sha256']
def env_cpu():
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    return env
