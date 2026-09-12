"""Actual frozen VAE + native finalizer + v2 training reader, CPU-only fixture."""
from common import *
from render import render_one,encode
from build import create_index,native
from mutations import DIGEST_FIELDS,sha256_json
from collections import Counter
import time


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    from scripts.t2a.data.materialize_sceneplan_transfusion_editing_targets import load_vae
    from scripts.t2a.data.finalize_sceneplan_transfusion_editing_index import finalize
    from transformers import AutoTokenizer
    from dataset_v2 import ScenePlanTransfusionEditingDataset
    fixture=ROOT/'reviews'/f'cpu_roundtrip_{time.time_ns()}';fixture.mkdir()
    con=db(ROOT/'pair_index/train.sqlite');rows=[];results=[];audits=[]
    replay=read(ROOT/'reviews/CPU_RENDER_REVIEW.json')['replays']
    tasks=['diagonal_relocation','dual_start_motion','position_swap','three_position_cycle']
    for task in tasks:
        choice=next(r for r in replay if r['task']==task and r['bucket']==432)
        base=dict(con.execute('SELECT * FROM pairs WHERE pair_id=?',(choice['pair_id'],)).fetchone())
        audit=json.loads(con.execute('SELECT audit_json FROM edit_actions WHERE pair_ordinal=?',(base['pair_ordinal'],)).fetchone()[0])
        row,accepted,result=render_one((base,audit,False));i=len(rows)
        latent=fixture/'materialized/latents/train/latents-train-00000.safetensors'
        row.update(pair_ordinal=i,work_shard=0,row_in_shard=i,target_latent_path=str(latent),
            target_latent_ref=f'{latent}#{row["target_sample_id"]}')
        row['pair_record_sha256']=sha256_json({k:row[k] for k in DIGEST_FIELDS})
        result.update(work_shard=0,row_in_shard=i,pair_record_sha256=row['pair_record_sha256'])
        rows.append(row);results.append(result);audits.append(accepted)
    assert len(rows)==4
    metadata=dict(con.execute('SELECT key,value FROM metadata'));con.close()
    planned=fixture/'planned.sqlite';out=create_index(planned);columns=native.PAIR_COLUMNS.split(',')
    out.executemany(f'INSERT INTO pairs({native.PAIR_COLUMNS}) VALUES({",".join("?" for _ in columns)})',[tuple(r[k] for k in columns) for r in rows])
    out.executemany('INSERT INTO edit_actions VALUES(?,?,?)',[(i,canonical(a),a['signature']) for i,a in enumerate(audits)])
    metadata.update(rows='4',target_root=str(fixture),operation_counts_json=canonical(dict(Counter(r['operation'] for r in rows))),work_shard_size='4')
    out.executemany('INSERT INTO metadata VALUES(?,?)',metadata.items());out.commit();out.close()
    tick=time.monotonic();model=load_vae(torch.device('cpu'))
    manifest=encode(rows,results,fixture,torch.device('cpu'),model);del model
    target=fixture/'training.sqlite';marker=finalize(planned,target,replace=False)
    marker=read(str(target)+'.frozen.json')
    tokenizer=AutoTokenizer.from_pretrained('/mnt/sdc/ckpts/pretrained/Qwen/Qwen3.5-0.8B',local_files_only=True)
    dataset=ScenePlanTransfusionEditingDataset(target,tokenizer_spec=(tokenizer,512),expected_num_samples=4,
        index_num_samples=4,expected_index_sha256=marker['index_sha256'],latent_crop_length=648,verify_tensor_hashes_on_access=True)
    examples=[]
    for i in range(4):
        latent,info=dataset[i]
        assert tuple(latent.shape)==tuple(info['source_foa_latent'].shape)==(64,648)
        assert torch.isfinite(latent).all() and torch.isfinite(info['source_foa_latent']).all()
        assert 'old_sceneplan' not in info
        examples.append(dict(pair_id=info['pair_id'],operation=info['operation'],edited_source_ids=info['edited_source_ids']))
    assert not list(fixture.rglob('*.wav')) and not list(fixture.rglob('*.flac'))
    write(ROOT/'reviews/CPU_VAE_ROUNDTRIP.json',dict(status='PASS_REAL_FROZEN_CPU_VAE_FINALIZER_AND_COMPOUND_DIT_READER',
        fixture=str(fixture),cases=examples,elapsed_seconds=time.monotonic()-tick,device='cpu',retained_FOA_audio_files=0,
        numeric_CPU_GPU_equivalence_claimed=False,completed_at=now()))
    print('PASS_REAL_CPU_VAE_ROUNDTRIP',str(fixture),flush=True)


if __name__=='__main__':main()
