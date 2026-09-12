"""Restore latent-only validation/test waveform truth from frozen render recipes."""
from common import *
from evaluation_common import cases
from legacy_truth import ExpandedTruthResolver
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
import argparse
import io
import multiprocessing as mp


def cache_directory(component,ordinal):
    assert component['split'] in ['validation','test'],'Training audio must never be retained'
    return ROOT/'evaluation_truth'/component['split']/component['name']/f'{ordinal:07d}'


def flac_blob(wave):
    import soundfile as sf
    buffer=io.BytesIO();sf.write(buffer,wave.T,44100,format='FLAC',subtype='PCM_24')
    return buffer.getvalue()


def restore_one(job):
    component,ordinal=job
    assert component['name']=='spatial_multi500k' and component['split']!='train'
    directory=cache_directory(component,ordinal);marker=directory/'TRUTH.json'
    con=db(component['index_path']);row=dict(con.execute('SELECT * FROM pairs WHERE pair_ordinal=?',(ordinal,)).fetchone());con.close()
    if marker.exists():
        result=read(marker);assert result['pair_record_sha256']==row['pair_record_sha256']
        assert result['source_foa_sha256']==row['source_foa_sha256'] and result['target_foa_sha256']==row['target_foa_sha256']
        for ref in result['files'].values():assert sha(ref['path'])==ref['sha256']
        return dict(pair_id=row['pair_id'],marker=str(marker),sha256=sha(marker))
    from scripts.t2a.data import materialize_sceneplan_transfusion_editing_targets as render
    from rir_pool import bounded_rir_pool
    from memory_io import memoized_sources
    old=unpack(row['old_sceneplan_zlib']);new=unpack(row['new_sceneplan_zlib'])
    assert digest(old)==row['old_sceneplan_sha256'] and digest(new)==row['new_sceneplan_sha256']
    previous=unpack(row['source_render_result_zlib']);unchanged=set(json.loads(row['unchanged_source_ids_json']))
    with bounded_rir_pool(),memoized_sources():
        source,stems,_,_=render._render_fixed_pair_gain(old,unpack(row['source_render_recipe_zlib']),previous,
            unchanged_source_ids={s['source_id'] for s in old['sources']},allow_nonboosting_peak_clamp=False)
        target,_,_,gain=render._render_fixed_pair_gain(new,unpack(row['target_render_recipe_zlib']),previous,
            unchanged_source_ids=unchanged,allow_nonboosting_peak_clamp=True)
    blobs={'source':flac_blob(source),'target':flac_blob(target)}
    assert hashlib.sha256(blobs['source']).hexdigest()==row['source_foa_sha256']
    assert hashlib.sha256(blobs['target']).hexdigest()==row['target_foa_sha256']
    assert gain['target_master_delta_db']>=-1.0001
    for s,stem in zip(old['sources'],stems):
        if s['source_id'] in unchanged:blobs['stem_'+s['source_id']]=flac_blob(stem)
    directory.mkdir(parents=True,exist_ok=True);files={}
    for key,blob in blobs.items():
        path=directory/(key+'.flac');temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
        with temp.open('wb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
        temp.replace(path);files[key]=dict(path=str(path),sha256=hashlib.sha256(blob).hexdigest())
    write(marker,dict(status='PASS_BYTE_EXACT_SOURCE_AND_TARGET_RESTORED_FOR_OFFLINE_EVALUATION',
        pair_id=row['pair_id'],pair_record_sha256=row['pair_record_sha256'],native_pair_ordinal=ordinal,
        source_foa_sha256=row['source_foa_sha256'],target_foa_sha256=row['target_foa_sha256'],
        files=files,old_sceneplan_sha256=row['old_sceneplan_sha256'],new_sceneplan_sha256=row['new_sceneplan_sha256'],
        unchanged_source_ids=sorted(unchanged),target_gain_qc=gain,model_input=False,restored_at=now()))
    return dict(pair_id=row['pair_id'],marker=str(marker),sha256=sha(marker))


class TruthResolver:
    def __init__(self,binding):
        self.binding=binding;self.starts=[c['offset'] for c in binding['components']]
        self.resolvers=[ExpandedTruthResolver(Path(c['index_path'])) for c in binding['components']]

    def row(self,ordinal):
        i=bisect_right(self.starts,ordinal)-1;c=self.binding['components'][i];local=ordinal-c['offset'];resolver=self.resolvers[i]
        if c['name']!='spatial_multi500k':return dict(resolver.row(local),pair_ordinal=ordinal)
        record=resolver.connection.execute('SELECT * FROM pairs WHERE pair_ordinal=?',(local,)).fetchone();record=dict(record)
        receipt=read(cache_directory(c,local)/'TRUTH.json')
        assert receipt['pair_record_sha256']==record['pair_record_sha256']
        names=['pair_id','source_sample_id','target_sample_id','operation','raw_edit_request','source_count','target_count',
            'latent_bucket_frames','model_num_samples','latent_frames_valid','source_domain','target_domain',
            'source_manifest_path','source_manifest_sha256','source_foa_sha256','target_foa_sha256']
        value={k:record[k] for k in names};value['pair_ordinal']=ordinal
        value.update(offline_old_sceneplan=resolver._unpack(record['old_sceneplan_zlib'],record['old_sceneplan_sha256'],'old'),
            offline_new_sceneplan=resolver._unpack(record['new_sceneplan_zlib'],record['new_sceneplan_sha256'],'new'),
            edited_source_ids=tuple(json.loads(record['edited_source_ids_json'])),
            unchanged_source_ids=tuple(json.loads(record['unchanged_source_ids_json'])))
        for role in ['source','target']:
            ref=receipt['files'][role];assert ref['sha256']==record[role+'_foa_sha256']
            path=resolver._verify_file(ref['path'],ref['sha256']);value[role+'_foa_path']=str(path)
            value[role+'_foa']=resolver._read_foa(path,record['model_num_samples'])
        value['source_stem_foa']={};value['source_stem_refs']={}
        for sid in value['unchanged_source_ids']:
            ref=receipt['files']['stem_'+sid];path=resolver._verify_file(ref['path'],ref['sha256'])
            value['source_stem_foa'][sid]=resolver._read_foa(path,record['model_num_samples'])
            value['source_stem_refs'][sid]=dict(path=str(path),sha256=ref['sha256'])
        return value

    def close(self):
        for resolver in self.resolvers:resolver.close()


def init_cpu():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)


def prepare(scope):
    index,specs=cases(scope)
    if scope.startswith('test_'):assert read(ROOT/'SELECTION.json')['selection_split']=='validation'
    jobs=[]
    for row in specs:
        if row['cohort']=='spatial_multi500k':
            component=next(c for c in index['components'] if c['name']==row['cohort']);jobs.append((component,row['native_pair_ordinal']))
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),initializer=init_cpu) as pool:
        receipts=list(pool.map(restore_one,jobs,chunksize=1))
    write(ROOT/'truth_reviews'/f'{scope}.json',dict(status='PASS_SCOPE_TRUTH_READY',scope=scope,
        exact_restored_rows=len(receipts),receipts=receipts,original_audio_reused_for_other_components=True,completed_at=now()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=SCOPES,required=True);args=parser.parse_args();prepare(args.scope)
