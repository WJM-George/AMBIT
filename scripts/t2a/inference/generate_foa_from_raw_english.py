#!/usr/bin/env python3
"""English raw requests -> Generation AR -> preserved ScenePlan -> frozen P10 FOA.

Request JSON contains only {requests: [{id, request}]}. No target plan or
request-derived hints are passed to AR. Output is native 44.1 kHz float32
WYZX / ACN / SN3D FOA. A runnable pipeline is not a model-quality certificate.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8<<20),b''):h.update(part)
    return h.hexdigest()


def atomic(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def main(args):
    args.output.mkdir(parents=True,exist_ok=True);lock=(args.output/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    resume_module=Path(__file__).resolve().parents[3]/'stable_audio_tools/inference/sceneplan_generation_ar_foa_resume.py'
    # Loading this utility must not pre-import the live model package before
    # the caller's immutable model snapshot is selected below.
    resume_spec=importlib.util.spec_from_file_location('raw_foa_resume_utils',resume_module)
    resume_utils=importlib.util.module_from_spec(resume_spec);resume_spec.loader.exec_module(resume_utils)
    read_raw_results=resume_utils.read_raw_results;read_verified_foa=resume_utils.read_verified_foa
    requests=json.loads(args.requests.read_text())['requests']
    if (not requests or any(not isinstance(r.get('id'),str) or not r['id'].strip()
            or not isinstance(r.get('request'),str) or not r['request'].strip() for r in requests)
            or len({r['id'] for r in requests})!=len(requests)):
        raise ValueError('requests must contain raw text and unique nonempty identifiers')
    render_seeds = None
    if args.render_seeds:
        seed_rows = json.loads(args.render_seeds.read_text())['rows']
        if any(set(row) != {'id', 'seed'} or not isinstance(row['seed'], int)
               or isinstance(row['seed'], bool) or not 0 <= row['seed'] < 2**63
               for row in seed_rows):
            raise ValueError('render seeds must contain only id and nonnegative int64 seed')
        render_seeds = {row['id']: row['seed'] for row in seed_rows}
        if len(render_seeds) != len(seed_rows) or set(render_seeds) != {row['id'] for row in requests}:
            raise ValueError('render-seed identifiers must match the raw request shard exactly')
    identity={'schema':'generation_ar_raw_english_to_p10_foa_v2','requests_sha256':sha(args.requests),'checkpoint_sha256':sha(args.checkpoint),
              'model_snapshot':str(args.snapshot),'model_snapshot_manifest_sha256':sha(args.snapshot/'SOURCE_SNAPSHOT_MANIFEST.json'),
              'entry_sha256':sha(Path(__file__)),'raw_entry_sha256':sha(args.raw_entry),'normalizer_sha256':sha(args.normalizer),
              'resume_module_sha256':sha(resume_module),
              'seed':args.seed,'p10_steps':100,
              'resume_policy':'Reuse only completed raw jobs and hash-verified FOA with identical input identities. Preview and repeat verification do not change audio inputs.',
              'raw_policy':'Raw English + fixed schema syntax grammar; no target/count hints/planning teacher.','p10_policy':'Only whitespace canonicalization allowed; original tokens and plans retained.',
              'audio_format':'44,100 Hz float32 WAV, native WYZX / ACN / SN3D FOA; no channel reorder or gain normalization.',
              'acceptance':'Execution/format checks only. Request satisfaction and audible fidelity require a separate validation report.'}
    # Keep the historical default identity unchanged for existing demo receipts.
    # New bulk controls are pinned whenever supplied and cannot change on resume.
    if args.render_seeds or args.raw_decoder or args.raw_gate_requests or args.ar_batch_size != 16 or args.ar_max_wall_seconds != 1800:
        identity['bulk_inference'] = {
            'render_seeds_sha256':sha(args.render_seeds) if args.render_seeds else None,
            'render_seeds_scope':'P10 noise only; never passed to AR',
            'raw_decoder_sha256':sha(args.raw_decoder) if args.raw_decoder else None,
            'raw_gate_requests_sha256':sha(args.raw_gate_requests) if args.raw_gate_requests else None,
            'ar_batch_size':args.ar_batch_size,'ar_max_wall_seconds':args.ar_max_wall_seconds}
    contract=args.output/'CONTRACT.json'
    if contract.exists():assert json.loads(contract.read_text())==identity
    else:atomic(contract,identity)
    raw=args.output/'ar'
    raw_identity=dict(requests=requests,requests_sha256=identity['requests_sha256'],checkpoint_sha256=identity['checkpoint_sha256'],
                      snapshot_manifest_sha256=identity['model_snapshot_manifest_sha256'],entry_sha256=identity['raw_entry_sha256'])
    results=read_raw_results(raw,**raw_identity)
    if results is None:
        atomic(args.output/'STATUS.json',{'status':'GENERATING_AR','pid':os.getpid()})
        command=[sys.executable,str(args.raw_entry),'--requests',str(args.requests),'--checkpoint',str(args.checkpoint),
                 '--snapshot',str(args.snapshot),'--output',str(raw),'--batch-size',str(args.ar_batch_size),
                 '--max-wall-seconds',str(args.ar_max_wall_seconds)]
        if args.raw_decoder:
            command += ['--decoder', str(args.raw_decoder)]
        if args.raw_gate_requests:
            command += ['--gate-requests', str(args.raw_gate_requests)]
        with (args.output/'ar_process.log').open('a') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=args.ar_max_wall_seconds+300)
        results=read_raw_results(raw,**raw_identity)
        if results is None:raise RuntimeError('AR process exited without a completed result')
    # AR child has exited and released its GPU allocation before loading P10.
    sys.path.insert(0,str(args.snapshot))
    import numpy as np
    import torch
    import soundfile as sf
    from transformers import AutoTokenizer
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    from stable_audio_tools.data.sceneplan_p11_single_turn import finalize_sceneplan_for_p10,P11Task
    from scripts.t2a.test.evaluate_sceneplan_transfusion_generation_ar_p10_audio_8k import _build_p10_identity,_executor_from_identity,_stable_seed,CODEC_PATH,QWEN_PATH
    spec=importlib.util.spec_from_file_location('raw_foa_whitespace_normalizer',args.normalizer);normalizer=importlib.util.module_from_spec(spec);spec.loader.exec_module(normalizer)
    torch.set_num_threads(4);torch.manual_seed(args.seed);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device('cuda:0')
    codec=ModelScenePlanCodecV4(CODEC_PATH);tokenizer=AutoTokenizer.from_pretrained(QWEN_PATH,local_files_only=True)
    prepared=[];rows=[];run_sha=sha(contract)
    for index,item in enumerate(requests):
        sid=item['id'];result=results[sid];stem=f'{index:04d}_'+hashlib.sha256(sid.encode()).hexdigest()[:12]
        if result['prediction'] is None:
            rows.append({'id':sid,'status':'AR_GENERATION_FAILED','error':result['error']});continue
        normalized=normalizer.normalize_generated_sceneplan(codec,result['tokens'],sample_id=sid,max_tokens=512)
        assert normalized['raw_plan']==result['prediction']
        bundle=finalize_sceneplan_for_p10(codec,normalized['p10_token_ids'],tokenizer=tokenizer,task=P11Task.GENERATION,sample_id=sid)
        assert bundle.sceneplan==normalized['p10_plan'];bundle.assert_external_p10_boundary()
        atomic(args.output/(stem+'.sceneplan.json'),normalized)
        render_seed = render_seeds[sid] if render_seeds is not None else _stable_seed(args.seed,sid)
        render_identity={'run_contract_sha256':run_sha,'id':sid,'p10_plan_sha256':normalized['p10_plan_sha256'],
                         'seed':render_seed,'model_num_samples':bundle.model_num_samples}
        receipt=args.output/(stem+'.audio.json')
        cached=read_verified_foa(receipt,expected=render_identity)
        row=cached or {'id':sid,'status':'P10_INPUT_READY','sceneplan':str(args.output/(stem+'.sceneplan.json')),
                       'source_count':len(bundle.sceneplan['sources']),'model_num_samples':bundle.model_num_samples,
                       'latent_frames':bundle.latent_frames_valid,'render_identity':render_identity}
        rows.append(row)
        if cached is None or (args.verify_repeat and not cached.get('repeat_bit_exact')):
            prepared.append((item,stem,bundle,cached))
    if args.prepare_only:
        summary={'status':'PREPARED' if all(r['status']!='AR_GENERATION_FAILED' for r in rows) else 'PARTIAL','rows':rows,'audio_rendered':False,'goal_complete':False}
        atomic(args.output/'SUMMARY.json',summary);atomic(args.output/'STATUS.json',summary);return 0 if summary['status']=='PREPARED' else 2
    byid={r['id']:r for r in rows};reused=sum(r['status']=='FOA_WRITTEN' for r in rows)
    if prepared:
        torch.cuda.set_device(device)
        atomic(args.output/'STATUS.json',{'status':'LOADING_FROZEN_P10','pid':os.getpid()})
        p10_identity=_build_p10_identity(args.snapshot/'artifacts/releases/P10_SCENEPLAN_DIT_V11_150K_RELEASE.json',CODEC_PATH)
        atomic(args.output/'P10_IDENTITY.json',p10_identity);executor=_executor_from_identity(p10_identity,device=device)
    render_started=time.monotonic()
    completed=reused
    for item,stem,bundle,cached in prepared:
        if args.max_render_wall_seconds and time.monotonic()-render_started > args.max_render_wall_seconds:
            raise TimeoutError('P10 wall budget exceeded; verified completed receipts can resume')
        started=time.monotonic();sid=item['id'];seed=render_seeds[sid] if render_seeds is not None else _stable_seed(args.seed,sid)
        atomic(args.output/'STATUS.json',{'status':'RENDERING_P10','pid':os.getpid(),'id':sid,
               'rows_done':completed,'rows':len(requests)})
        audio=executor.render(bundle,seed=seed)
        assert tuple(audio.shape)==(4,bundle.model_num_samples) and bool(torch.isfinite(audio).all())
        if cached is not None:
            previous,_=sf.read(cached['foa'],dtype='float32',always_2d=True)
            assert np.array_equal(previous,audio.numpy().T),'Repeat differs from committed FOA'
            byid[sid].update(repeat_bit_exact=True,repeat_verification_elapsed_s=time.monotonic()-started)
        else:
            repeat_exact=None
            if args.verify_repeat:repeat_exact=torch.equal(audio,executor.render(bundle,seed=seed));assert repeat_exact
            waveform=audio.numpy().T;wav=args.output/(stem+'.foa.wav');temporary=wav.with_suffix('.tmp.wav')
            sf.write(temporary,waveform,44100,subtype='FLOAT');restored,sample_rate=sf.read(temporary,dtype='float32',always_2d=True)
            assert sample_rate==44100 and np.array_equal(restored,waveform);temporary.replace(wav)
            byid[sid].update(status='FOA_WRITTEN',foa=str(wav),foa_sha256=sha(wav),sample_rate=sample_rate,
                channel_order='WYZX',ambisonic_convention='ACN/SN3D',peak=float(audio.abs().max()),
                rms=float(audio.square().mean().sqrt()),seed=seed,repeat_bit_exact=repeat_exact,render_elapsed_s=time.monotonic()-started)
        atomic(args.output/(stem+'.audio.json'),byid[sid])
        completed += int(cached is None)
        if args.compact_progress:
            atomic(args.output/'PROGRESS.json',{'rows_done':completed,'rows':len(requests),
                   'last_id':sid,'last_render_elapsed_s':time.monotonic()-started})
        else:
            atomic(args.output/'PROGRESS.json',{'rows':rows})
    summary={'status':'COMPLETE' if all(r['status']=='FOA_WRITTEN' for r in rows) else 'PARTIAL','rows':rows,'audio_rendered':True,'reused_completed_rows':reused,'request_and_audio_quality_acceptance':'NOT_ESTABLISHED_BY_EXECUTION_CHECK','goal_complete':False}
    atomic(args.output/'SUMMARY.json',summary)
    status=({'status':summary['status'],'rows_done':sum(r['status']=='FOA_WRITTEN' for r in rows),
             'rows':len(rows),'summary':str(args.output/'SUMMARY.json')} if args.compact_progress else summary)
    atomic(args.output/'STATUS.json',status);print(json.dumps(status),flush=True)
    return 0 if summary['status']=='COMPLETE' else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--requests',type=Path,help='Batch JSON containing identifiers and English request strings only')
    inputs.add_argument('--request',type=str,help='One raw English request, without a JSON wrapper')
    for name in ('checkpoint','snapshot','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--raw-entry',type=Path,default=Path(__file__).with_name('generate_sceneplan_from_raw_english.py'))
    parser.add_argument('--normalizer',type=Path,default=Path(__file__).resolve().parents[3]/'stable_audio_tools/inference/sceneplan_generation_ar_normalization.py')
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--verify-repeat',action='store_true');parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--render-seeds',type=Path,help='P10-only id/seed mapping for an existing benchmark; never passed to AR')
    parser.add_argument('--raw-decoder',type=Path,help='Explicit frozen learned-head decoder implementation')
    parser.add_argument('--raw-gate-requests',type=Path,help='Pure validation requests for the technical raw-generation gate')
    parser.add_argument('--ar-batch-size',type=int,default=16)
    parser.add_argument('--ar-max-wall-seconds',type=int,default=1800)
    parser.add_argument('--max-render-wall-seconds',type=int,default=0)
    parser.add_argument('--compact-progress',action='store_true')
    args=parser.parse_args()
    if args.request is not None:
        if not args.request.strip():parser.error('--request must be nonempty')
        # The exact user string is retained; this file is only an IO envelope.
        payload={'requests':[{'id':'raw_request','request':args.request}]}
        args.output.mkdir(parents=True,exist_ok=True)
        request_hash=hashlib.sha256(args.request.encode()).hexdigest()[:16]
        args.requests=args.output/('request-'+request_hash+'.json')
        if args.requests.exists():assert json.loads(args.requests.read_text())==payload
        else:atomic(args.requests,payload)
    try:raise SystemExit(main(args))
    except Exception as exc:
        args.output.mkdir(parents=True,exist_ok=True);atomic(args.output/'STATUS.json',{'status':'FAILED','error':f'{type(exc).__name__}: {exc}'})
        raise
