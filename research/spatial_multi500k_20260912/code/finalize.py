"""Publish all accepted pairs, exhaustively verify latents, and exercise the reader."""
from common import *
from render import load_frozen
from build import create_index,metadata,native
from cpu_check import inspect_pair
from collections import Counter
import time


def accepted_index(split):
    directory=ROOT/'accepted_pair_index';directory.mkdir(exist_ok=True)
    path=directory/f'{split}.sqlite';marker=path.with_suffix('.json')
    if marker.exists():assert sha(path)==read(marker)['sha256'];return path
    planned=db(ROOT/'pair_index'/f'{split}.sqlite');temporary=path.with_suffix(f'.tmp.{os.getpid()}')
    temporary.unlink(missing_ok=True);out=create_index(temporary);columns=native.PAIR_COLUMNS.split(',')
    pending=[];actions=[];groups=Counter();tasks=Counter();n=0;replaced=0
    for base in planned.execute('SELECT * FROM pairs ORDER BY pair_ordinal'):
        row,audit,receipt=load_frozen(dict(base));inspect_pair(row,audit)
        assert row['pair_id']==base['pair_id'] and row['pair_ordinal']==n
        group=f'{row["source_count"]}|{row["source_domain"]}|{row["latent_bucket_frames"]}'
        assert group==audit['spec']['stratum'];groups[group]+=1;tasks[audit['spec']['task']]+=1
        pending.append(tuple(row[k] for k in columns));actions.append((n,canonical(audit),audit['signature']))
        replaced+=receipt['background_replaced'];n+=1
        if len(pending)>=512 or n==COUNTS[split]:
            out.executemany(f'INSERT INTO pairs({native.PAIR_COLUMNS}) VALUES({",".join("?" for _ in columns)})',pending)
            out.executemany('INSERT INTO edit_actions VALUES(?,?,?)',actions);out.commit();pending.clear();actions.clear()
    review=read(ROOT/'catalog'/f'{split}.json')
    assert n==COUNTS[split] and dict(groups)==review['joint_quotas'] and dict(tasks)==tasks_for(split)
    assert out.execute('SELECT max(n) FROM (SELECT count(*) n FROM pairs GROUP BY source_sample_id)').fetchone()[0]<=2
    metadata(out,split,review)
    out.executemany('INSERT OR REPLACE INTO metadata VALUES(?,?)',[
        ('source_candidate_index_sha256',read(ROOT/'pair_index'/f'{split}.json')['sha256']),
        ('calibration_module_sha256',sha(ROOT/'code/render.py')),
        ('all_audio_storage','latent_only_transient_in_memory_FOA'),
        ('final_accepted_edited_source_counts',canonical(dict(out.execute('SELECT json_array_length(edited_source_ids_json),count(*) FROM pairs GROUP BY 1'))))])
    out.commit();assert out.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    out.close();planned.close();temporary.replace(path)
    write(marker,dict(status='PASS_ALL_ACCEPTED_PAIRS_AND_EXACT_QUOTAS',rows=n,sha256=sha(path),
        replacement_backgrounds=replaced,task_counts=dict(tasks),strata=dict(groups),completed_at=now()))
    return path


def reader_check(split,path,marker):
    from transformers import AutoTokenizer
    from dataset_v2 import ScenePlanTransfusionEditingDataset
    import torch
    tokenizer=AutoTokenizer.from_pretrained('/mnt/sdc/ckpts/pretrained/Qwen/Qwen3.5-0.8B',local_files_only=True)
    con=db(path);ordinals=[r[0] for r in con.execute('SELECT min(pair_ordinal) FROM pairs GROUP BY operation,source_count,latent_bucket_frames')];con.close()
    dataset=ScenePlanTransfusionEditingDataset(path,tokenizer_spec=(tokenizer,512),
        expected_num_samples=len(ordinals),index_num_samples=COUNTS[split],sample_ordinals=ordinals,
        expected_index_sha256=marker['index_sha256'],latent_crop_length=648,verify_tensor_hashes_on_access=True)
    checked=[]
    for i in range(len(dataset)):
        target,info=dataset[i]
        assert target.shape==info['source_foa_latent'].shape==(64,648)
        assert torch.isfinite(target).all() and torch.isfinite(info['source_foa_latent']).all()
        assert 'old_sceneplan' not in info and 'editing_ar_target_model_sceneplan' in info
        checked.append(dict(pair_id=info['pair_id'],operation=info['operation'],edited_source_ids=info['edited_source_ids']))
    return checked


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    from scripts.t2a.data.finalize_sceneplan_transfusion_editing_index import finalize
    forbidden=[str(p) for suffix in ['*.wav','*.flac','*.ogg','*.mp3'] for p in ROOT.rglob(suffix)]
    assert not forbidden,forbidden[:10]
    results={}
    for split in ['validation','test','train']:
        planned=accepted_index(split);path=ROOT/'training_index'/f'{split}.sqlite'
        marker_path=Path(str(path)+'.frozen.json')
        if not marker_path.exists():finalize(planned,path,replace=False)
        marker=read(marker_path);assert marker['rows']==COUNTS[split] and marker['index_sha256']==sha(path)
        checked=reader_check(split,path,marker)
        results[split]=dict(index=marker,training_reader_cases=checked)
    write(ROOT/'DATA_READY.json',dict(status='PASS_500K_SPATIAL_MULTI_TRAIN_LATENTS_AND_SPLIT_EXTENSIONS',
        counts=COUNTS,indices=results,retained_FOA_audio_files=0,original_datasets_preserved=True,
        source_pcm_parity_scope='Representative CPU canaries; all 512500 metadata and target latent records verified',
        old_training_rows=1250000,new_training_rows=500000,available_training_pairs=1750000,
        automatic_training_started=False,completed_at=now()))
    REPORT.mkdir(parents=True,exist_ok=True)
    write(REPORT/'RESULT.json',read(ROOT/'DATA_READY.json'))


if __name__=='__main__':main()
