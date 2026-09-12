"""Choose validation by metadata and seed only, before new model scores exist."""
from common import *
from evaluation_common import all_cases
from collections import Counter,defaultdict


def choose(rows,n,label):
    assert len(rows)>=n,(label,len(rows),n)
    groups=defaultdict(list)
    for row in rows:groups[(row['latent_bucket_frames'],row['source_count'],row['target_domain'])].append(row)
    quotas={k:n*len(v)//len(rows) for k,v in groups.items()}
    order=sorted(groups,key=lambda k:(-(n*len(groups[k])%len(rows)),str(k)))
    for k in order[:n-sum(quotas.values())]:quotas[k]+=1
    return [r for k in sorted(groups) for r in sorted(groups[k],key=lambda x:digest([202609121750,label,x['pair_id']]))[:quotas[k]]]


def balanced_new_subset(binding,rows):
    old=[r for r in rows if r['cohort']=='original'];newadd=[r for r in rows if r['cohort']=='addition250k']
    spatial=[r for r in rows if r['cohort']=='spatial_multi500k'];result=[]
    for op,n in [('event_addition',160),('event_removal',320),('linear_to_static',320),('static_to_linear',160),('stationary_spatial_relocation',160)]:
        result+=choose([r for r in old if r['operation']==op],n,'old/'+op)
    result+=choose(newadd,160,'addition250k')
    con=db(SPATIAL/'training_index/validation.sqlite')
    directions={}
    for row in con.execute("SELECT pair_ordinal,work_shard FROM pairs WHERE operation IN ('stationary_spatial_relocation','static_to_linear')"):
        frozen=SPATIAL/'calibrated_rows/validation'/f'work-{row["work_shard"]:05d}'/f'{row["pair_ordinal"]:07d}.json.zlib'
        directions[row['pair_ordinal']]=unpack(frozen.read_bytes())['audit']['spec']['direction']
    con.close()
    for op in ['stationary_spatial_relocation','static_to_linear']:
        for direction in ['front_left','front_right','rear_left','rear_right']:
            result+=choose([r for r in spatial if r['operation']==op and directions[r['native_pair_ordinal']]==direction],40,op+'/'+direction)
    for op,n in [('multi_source_relocation',140),('source_position_swap',120),('multi_source_static_to_linear',40),
        ('multi_source_linear_to_static',40),('multi_source_motion_toggle',40),('multi_source_position_cycle',20)]:
        result+=choose([r for r in spatial if r['operation']==op],n,'compound/'+op)
    assert len(result)==2000 and len({r['pair_id'] for r in result})==2000
    return sorted(result,key=lambda r:r['pair_ordinal'])


def main():
    if (ROOT/'EVALUATION_PREPARED.json').exists():
        ref=read(ROOT/'EVALUATION_PREPARED.json')
        assert ref['casesets_sha256']==sha(ROOT/'CASESETS.json') and ref['protocol_sha256']==sha(ROOT/'SELECTION_PROTOCOL.json');return
    binding=components('validation');rows=all_cases(binding);policy=read(ROOT/'POLICY.json')
    if policy['validation_subset_policy']=='reuse_previous_2000':
        old=read(PREVIOUS/'VALIDATION_SUBSET.json');wanted={r['pair_id'] for r in old['cases']}
        selected=[r for r in rows if r['pair_id'] in wanted];assert len(selected)==2000
    else:
        assert policy['validation_subset_policy']=='balanced_three_components_2000'
        selected=balanced_new_subset(binding,rows)
    sets={'validation':dict(index=binding,cases=selected,case_manifest_sha256=digest(selected))}
    for c in components('test')['components']:
        single=dict(schema='three_immutable_editing_components_v1',split='test',rows=c['rows'],components=[dict(c,offset=0)])
        values=all_cases(single);sets['test_'+c['name']]=dict(index=single,cases=values,case_manifest_sha256=digest(values))
    write(ROOT/'CASESETS.json',sets)
    protocol=read(PREVIOUS/'SELECTION_PROTOCOL.json');operations=sorted({r['operation'] for r in selected})
    protocol.update(schema='editing_mixed1750k_validation_selection_v1',steps=STEPS,rows=2000,
        frozen_before_new_training_and_inference=True,validation_subset=str(ROOT/'CASESETS.json'),
        validation_subset_sha256=sha(ROOT/'CASESETS.json'),validation_policy=policy['validation_subset_policy'],
        cohort_counts=dict(Counter(r['cohort'] for r in selected)),operation_counts=dict(Counter(r['operation'] for r in selected)),
        eligibility='All 2000 identical validation cases and independent review required for every candidate including 50k; original-operation guards retained.',
        scope_change_reason='User authorized original50k -> +30k on 1.75M pairs, fixed validation2k selection then three separate held-out test sets',
        scope_updated_at=now())
    for dimension in ['audio','preservation','timing']:protocol['dimensions'][dimension]['operations']=operations
    protocol['dimensions']['spatial']['operations']=[op for op in operations if op not in ['event_addition','event_removal']]
    write(ROOT/'SELECTION_PROTOCOL.json',protocol)
    write(ROOT/'EVALUATION_PREPARED.json',dict(status='PASS_FIXED_VALIDATION2K_AND_THREE_TEST_COHORTS',
        casesets_sha256=sha(ROOT/'CASESETS.json'),protocol_sha256=sha(ROOT/'SELECTION_PROTOCOL.json'),
        selection_uses_test_results=False,cohort_counts=protocol['cohort_counts'],operation_counts=protocol['operation_counts'],prepared_at=now()))


if __name__=='__main__':main()
