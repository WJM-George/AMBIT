import copy
import json

import pytest

from scripts.t2a.rl.train_editing_opsd_eight_gpu import validate_contract,migrated_rank_states
from scripts.t2a.rl.opsd_canonical_validation import canonical_batches
from stable_audio_tools.paths import opsd_config_path
from stable_audio_tools.training.transfusion_opsd.editing_stream import OrdinalStream


def pair():
    path=opsd_config_path()
    if not path.exists():
        pytest.skip('Pinned eight-GPU parent config is not installed.')
    parent=json.loads(path.read_text());current=copy.deepcopy(parent)
    current.update(physical_gpus=list(range(8)),request_rows_per_rank=2,paired_rows_per_rank=64,
                   output='/isolated/eight/training',resize_parent_config=str(path))
    current['complete_recipe']['decoded_rows_per_rank']=1
    return current,parent


def test_global_auxiliary_budget_is_preserved_and_learning_rate_changes_rejected():
    current,parent=pair();validate_contract(current,parent)
    current['learning_rates']['AR_adapters']*=2
    with pytest.raises(ValueError):validate_contract(current,parent)


@pytest.mark.parametrize('field,value', [('global_request_batch',32),('global_paired_batch',1024),
                                      ('request_rows_per_rank',4),('paired_rows_per_rank',128)])
def test_expansion_cannot_silently_double_global_training(field,value):
    current,parent=pair();current[field]=value
    with pytest.raises(ValueError):validate_contract(current,parent)


def test_decoded_auxiliary_examples_do_not_double_with_world_size():
    current,parent=pair();current['complete_recipe']['decoded_rows_per_rank']=2
    with pytest.raises(ValueError):validate_contract(current,parent)


def test_next_request_and_paired_global_batches_match_old_consumed_frontier():
    streams={key:[OrdinalStream(range(16384),seed=seed,rank=r,world=4) for r in range(4)]
             for key,seed in [('request',101),('paired',102)]}
    rows=[]
    for rank in range(4):
        for key,count in [('request',28),('paired',896)]:streams[key][rank].take(count)
        rows.append(dict(request=streams['request'][rank].state_dict(),paired=streams['paired'][rank].state_dict(),
                         costs={'total':rank+1,**({'rare':8} if rank==0 else {})}))
    migrated=migrated_rank_states(rows)
    for key,old_count,new_count,seed in [('request',4,2,101),('paired',128,64,102)]:
        old_next=[x for stream in streams[key] for x in stream.take(old_count)]
        new_streams=[OrdinalStream(range(16384),seed=seed,rank=r,world=8,state=migrated[r][key]) for r in range(8)]
        new_next=[x for stream in new_streams for x in stream.take(new_count)]
        assert sorted(new_next)==sorted(old_next)
        assert len(new_next)==len(set(new_next))
    assert sum(r['costs']['total'] for r in migrated)==10
    assert sum(r['costs']['rare'] for r in migrated)==8


def test_eight_worker_native_validation_preserves_every_four_rank_batch_and_noise():
    entries=[(i,432 if i<15000 else 648,'speech_only') for i in range(20000)]
    old=[batch for rank in range(4) for batch in canonical_batches(entries,rank,4)]
    new=[batch for rank in range(8) for batch in canonical_batches(entries,rank,8)]
    assert sorted(new)==sorted(old)
    assert sorted(i for _,_,ids in new for i in ids)==list(range(20000))
    # Noise is keyed by logical rank and original batch index, independent
    # of physical worker count. Residual short/long batches also survive.
    assert {(r,500000+i,tuple(ids)) for r,i,ids in new}=={(r,500000+i,tuple(ids)) for r,i,ids in old}
    assert min(len(ids) for _,_,ids in new)<32
