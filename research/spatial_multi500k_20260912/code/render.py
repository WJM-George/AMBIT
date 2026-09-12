"""Render accepted spatial edits into memory; persist metadata and latents only."""
from common import *
from mutations import mutate,validate,Ineligible
from rir_pool import bounded_rir_pool
from memory_io import memoized_sources,memory_audio_reader
from contextlib import contextmanager
import base64
import copy
import io
import time
import numpy as np
import reservation_db


def encoded_row(row):
    return {k:{'zlib_base64':base64.b64encode(v).decode()} if isinstance(v,bytes) else v for k,v in row.items()}


def decoded_row(row):
    return {k:base64.b64decode(v['zlib_base64']) if isinstance(v,dict) and set(v)=={'zlib_base64'} else v for k,v in row.items()}


def frozen_path(row):
    return ROOT/'calibrated_rows'/row['split']/f'work-{row["work_shard"]:05d}'/f'{row["pair_ordinal"]:07d}.json.zlib'


def load_frozen(candidate):
    record=unpack(frozen_path(candidate).read_bytes())
    assert record['candidate_pair_record_sha256']==candidate['pair_record_sha256']
    assert digest(record['row'])==record['row_sha256']
    row=decoded_row(record['row']);validate(row,record['audit'])
    return row,record['audit'],record


def reservations():
    path=ROOT/'RESERVATIONS.sqlite'
    if (ROOT/'RESERVATIONS_READY.json').exists():return
    con=sqlite3.connect(path,timeout=60)
    con.executescript('''PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS assignments(split TEXT,ordinal INTEGER,source_pair_ordinal INTEGER,
        source_sample_id TEXT,signature TEXT,PRIMARY KEY(split,ordinal));
        CREATE INDEX IF NOT EXISTS assignment_source ON assignments(split,source_pair_ordinal);
        CREATE UNIQUE INDEX IF NOT EXISTS assignment_signature ON assignments(split,signature);''')
    for split in COUNTS:
        source=db(ROOT/'pair_index'/f'{split}.sqlite');pending=[]
        for r in source.execute('SELECT p.pair_ordinal,p.source_sample_id,a.audit_json,a.signature FROM pairs p JOIN edit_actions a USING(pair_ordinal)'):
            audit=json.loads(r['audit_json'])
            pending.append((split,r['pair_ordinal'],audit['source_pair_ordinal'],r['source_sample_id'],r['signature']))
            if len(pending)==2048:con.executemany('INSERT OR IGNORE INTO assignments VALUES(?,?,?,?,?)',pending);con.commit();pending.clear()
        if pending:con.executemany('INSERT OR IGNORE INTO assignments VALUES(?,?,?,?,?)',pending);con.commit()
        assert con.execute('SELECT count(*) FROM assignments WHERE split=?',(split,)).fetchone()[0]==COUNTS[split]
        source.close()
    assert con.execute('SELECT max(n) FROM (SELECT count(*) n FROM assignments GROUP BY split,source_pair_ordinal)').fetchone()[0]<=2
    con.close();write(ROOT/'RESERVATIONS_READY.json',dict(status='PASS_INITIAL_SOURCE_AND_SIGNATURE_RESERVATIONS',counts=COUNTS,created_at=now()))


def reserved_base(candidate,audit):
    o=reservation_db.source_ordinal(ROOT,candidate['split'],candidate['pair_ordinal'])
    old=db(BASE/'training_index'/f'{candidate["split"]}.sqlite')
    base=dict(old.execute('SELECT * FROM pairs WHERE pair_ordinal=?',(o,)).fetchone());old.close()
    return base


def replacement(candidate,audit,attempt):
    spec=audit['spec'];split=candidate['split'];cat=db(ROOT/'catalog'/f'{split}.sqlite')
    for trial in range(1024):
        rank=seed(candidate['pair_id'],'replacement',attempt,trial)%(2**63-1)
        choices=cat.execute('''SELECT e.ordinal,s.sample_id FROM eligibility e JOIN sources s USING(ordinal)
            WHERE e.task=? AND e.stratum=? AND e.rank>=? ORDER BY e.rank LIMIT 16''',(spec['task'],spec['stratum'],rank)).fetchall()
        for choice in choices:
            if not reservation_db.reserve_source(ROOT,split,candidate['pair_ordinal'],choice['ordinal'],choice['sample_id']):continue
            cat.close()
            old=db(BASE/'training_index'/f'{split}.sqlite')
            base=dict(old.execute('SELECT * FROM pairs WHERE pair_ordinal=?',(choice['ordinal'],)).fetchone());old.close()
            return base
    cat.close();raise RuntimeError(f'No spare background in the exact stratum for {candidate["pair_id"]}')


def reserve_signature(row,audit):
    return reservation_db.reserve_signature(ROOT,row['split'],row['pair_ordinal'],
        audit['source_pair_ordinal'],audit['signature'])


def transient_flac(audio):
    from scripts.t2a.data import materialize_sceneplan_transfusion_editing_targets as target
    buffer=io.BytesIO();target.sf.write(buffer,audio.T,44100,format='FLAC',subtype='PCM_24')
    blob=buffer.getvalue();stored,rate=target.sf.read(io.BytesIO(blob),dtype='float32',always_2d=True)
    assert rate==44100 and stored.shape==(audio.shape[-1],4) and np.isfinite(stored).all()
    assert target.true_peak(stored.T)<=target.TRUE_PEAK_CEILING+2e-5
    return blob


def render_mix(row,source_parity=False):
    from scripts.t2a.data import materialize_sceneplan_transfusion_editing_targets as target
    old=unpack(row['old_sceneplan_zlib']);new=unpack(row['new_sceneplan_zlib'])
    old_recipe=unpack(row['source_render_recipe_zlib']);recipe=unpack(row['target_render_recipe_zlib'])
    previous=unpack(row['source_render_result_zlib']);parity_sha=None
    if source_parity:
        old_mix,_,_,_=target._render_fixed_pair_gain(old,old_recipe,previous,
            unchanged_source_ids={s['source_id'] for s in old['sources']},allow_nonboosting_peak_clamp=False)
        parity_sha=hashlib.sha256(transient_flac(old_mix)).hexdigest()
        assert parity_sha==row['source_foa_sha256'],'Frozen source PCM24 parity failed'
    mix,_,qc,gain=target._render_fixed_pair_gain(new,recipe,previous,
        unchanged_source_ids=set(json.loads(row['unchanged_source_ids_json'])),allow_nonboosting_peak_clamp=True)
    return mix,qc,gain,parity_sha


def render_one(job):
    candidate,audit,parity=job
    from scripts.t2a.data import materialize_sceneplan_transfusion_editing_targets as target
    started=time.monotonic();path=frozen_path(candidate);seen=set();failures=[]
    with bounded_rir_pool(),memoized_sources() as memo:
        if path.exists():
            row,accepted,receipt=load_frozen(candidate)
            mix,qc,gain,parity_sha=render_mix(row,parity)
        else:
            base=reserved_base(candidate,audit)
            found=False
            for attempt in range(64):
                for variant in range(32):
                    try:row,accepted=mutate(base,audit['spec'],variant)
                    except Ineligible:continue
                    if accepted['signature'] in seen:continue
                    seen.add(accepted['signature'])
                    mix,qc,gain,parity_sha=render_mix(row,False)
                    if gain['target_master_delta_db'] < -1.0001:
                        failures.append(dict(source=base['pair_ordinal'],variant=variant,master_delta_db=gain['target_master_delta_db']));continue
                    if not reserve_signature(row,accepted):continue
                    found=True;break
                if found:break
                base=replacement(candidate,audit,attempt)
            if not found:raise RuntimeError(f'No peak-safe target in requested stratum: {candidate["pair_id"]}')
            if parity:
                mix,qc,gain,parity_sha=render_mix(row,True)
            encoded=encoded_row(row)
            receipt=dict(version='spatial_multi_v2',candidate_pair_record_sha256=candidate['pair_record_sha256'],
                row=encoded,row_sha256=digest(encoded),audit=accepted,
                peak_rejected_candidates=failures,background_replaced=row['source_sample_id']!=candidate['source_sample_id'],
                frozen_at=now())
            path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(f'.tmp.{os.getpid()}')
            with tmp.open('wb') as f:f.write(pack(receipt));f.flush();os.fsync(f.fileno())
            tmp.replace(path)
        assert gain['target_master_delta_db']>=-1.0001
        blob=transient_flac(mix);blob_sha=hashlib.sha256(blob).hexdigest()
        # A virtual path is only an in-memory lookup key; no audio file is created.
        virtual=ROOT/'transient_audio_never_written'/row['split']/(row['pair_id']+'.flac')
        result={k:row[k] for k in ['pair_id','pair_record_sha256','split','work_shard','row_in_shard',
            'source_sample_id','target_sample_id','operation_family','operation','model_num_samples','latent_frames_valid',
            'old_sceneplan_sha256','new_sceneplan_sha256','source_render_recipe_sha256','target_render_recipe_sha256','source_foa_sha256']}
        result.update(schema=target.MATERIALIZATION_CONTRACT,schema_version=1,status='ok',
            source_parity_requested=parity,source_parity_sha256=parity_sha,
            source_parity_verified=bool(parity and parity_sha==row['source_foa_sha256']),source_parity_stem_refs=[],
            target_foa_path=str(virtual),target_foa_sha256=blob_sha,pair_gain_qc=gain,source_qc=qc,stem_refs=[],
            target_audio_storage='transient_in_memory_PCM24_FLAC',row_memoization=dict(memo),
            calibration_record_sha256=sha(path),elapsed_sec=time.monotonic()-started,_memory_foa_flac=blob)
        return row,accepted,result


def encode(rows,results,output_root,device,model):
    from scripts.t2a.data import materialize_sceneplan_transfusion_editing_targets as target
    with memory_audio_reader(results):
        return target._encode(rows,results,output_root=Path(output_root),device=device,batch_size=4,cleanup_foa=True,model=model)


def init_cpu():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
