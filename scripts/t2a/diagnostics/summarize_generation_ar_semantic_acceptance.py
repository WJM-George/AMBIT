#!/usr/bin/env python3
"""Join semantic judgments to exact source pairs and score per-count fidelity."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from stable_audio_tools.data.sceneplan_generation_ar_acceptance import score_scene,summarize_scenes


def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=args.root
    assert json.loads((root/'scoring/STATUS.json').read_text())['status']=='COMPLETE'
    binding=json.loads((root/'bindings.json').read_text());by_scene=defaultdict(dict)
    for b in binding['sources']:by_scene[b['candidate'],b['ordinal']][b['source_id']]=b
    db=sqlite3.connect(f'file:{root / "scoring/results.sqlite"}?mode=ro&immutable=1',uri=True)
    judged={i:json.loads(p) for i,p in db.execute('SELECT id,payload FROM results')};db.close()
    groups=defaultdict(list);transcripts=defaultdict(list);count_confusion=defaultdict(lambda:defaultdict(int))
    paths={s['candidate']:s['prediction_db'] for s in binding['scenes']}
    for name,path in paths.items():
        db=sqlite3.connect(f'file:{path}?mode=ro&immutable=1',uri=True)
        for (payload,) in db.execute('SELECT payload FROM results ORDER BY panel_index'):
            v=json.loads(payload);pred={s['source_id']:s for s in v['prediction']['sources']} if v['prediction'] else {}
            labels={};bound=by_scene[name,v['ordinal']]
            assert set(bound)=={s['source_id'] for s in v['target']['sources']}
            for ref in v['target']['sources']:
                b=bound[ref['source_id']];hyp=pred.get(ref['source_id'])
                if 'pair_id' in b:
                    field='speaker_description' if ref['kind']=='speech' else 'description'
                    core={'kind':ref['kind'],'reference':ref[field],'candidate':hyp[field]}
                    assert digest(core)==b['pair_id'];judge=judged[b['pair_id']]
                    assert {k:judge[k] for k in core}==core
                    labels[ref['source_id']]=judge['prediction']=='PASS'
                elif b['reason']=='exact_text_up_to_whitespace':
                    field='speaker_description' if ref['kind']=='speech' else 'description'
                    assert hyp['kind']==ref['kind'] and ' '.join(ref[field].split())==' '.join(hyp[field].split())
                    labels[ref['source_id']]=True
                else:
                    assert hyp is None or hyp['kind']!=ref['kind'];labels[ref['source_id']]=False
                if ref['kind']=='speech':transcripts[name,v['source_count']].append(b['transcript_word_sequence_equal_auxiliary'])
            scored=score_scene(v['target'],v['prediction'],labels);groups[name,v['source_count']].append(scored)
            count_confusion[name,str(v['source_count'])][str(scored['predicted'])]+=1
        db.close()
    report={'schema':'generation_ar_core_description_acceptance_diagnostic_v1','test_used':False,'goal_complete':False,
            'scope':'Core description includes speaker_description for speech; quoted transcript word equality is separate and must also be resolved before full request-execution delivery.',
            'semantic_method':'frozen local Qwen3.5-27B binary judge; 96 authored calibration pairs and 23 unambiguous blind Codex audit pairs passed; no independent human labels',
            'by_candidate':{},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for name in paths:
        report['by_candidate'][name]={}
        for count in range(1,5):
            result=summarize_scenes(groups[name,count]);values=transcripts[name,count]
            result['speech_transcript_word_equality_auxiliary']={'correct':sum(values),'requested_speech_sources':len(values),'rate':sum(values)/len(values) if values else None}
            result['count_confusion']=dict(count_confusion[name,str(count)])
            report['by_candidate'][name][str(count)]=result
    out=root/'ACCEPTANCE_DIAGNOSTIC.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(out)
    for name,values in report['by_candidate'].items():
        print(json.dumps({'candidate':name,'groups':{k:{m:v['metrics'][m]['rate'] for m in ['count','semantic_precision','semantic_recall','motion','onset','offset','start','end','joint']} for k,v in values.items()}}))


if __name__=='__main__':main()
