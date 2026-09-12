#!/usr/bin/env python3
"""Local request-first teacher planning with preserved raw generations and QA.

The teacher sees only the already frozen English request and a general output
contract. Internal expected counts and future target plans are not model input.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

PROMPT = '''You turn a normal English request into one reasonable spatial-audio plan and an auditable list of what that request actually requires. Return only a JSON object. All text must be English. Do not include markdown or reasoning.
The plan is one possible implementation, not a unique hidden answer. Preserve the requested sounds, voice properties, quoted words, motions, spatial relations and timing relations. Do not add independent sounds. Do not invent a requirement for an unspecified number. Unspecified valid coordinates, durations, transcripts, room acoustics and timing may be chosen reasonably.
Task limits: one to four independently positioned sources, at most one speech source, total duration at most 15 seconds, one active time interval per source, static or linear motion. Intermittent sounds belong in the source description. Use 0 dB gains (added by the serializer). Sound effects use kind sound; instrumental or sung music uses music; intelligible requested spoken words use speech. Make free distances 0.5 to 10 meters. Positive azimuth is left: front 0, left +90, right -90, behind +/-180. Positive elevation is above. Linear azimuth follows the shortest angular path; right-to-left passes in front should use endpoints such as -60 to +60. For 'after someone finishes', leave at least 0.1 seconds between the speech end and the next onset so frame quantization preserves the relation.
Output this compact JSON:
{"duration":10,"room":"outdoor","sources":[{"key":"dog","kind":"sound","core":"A dog barking intermittently","evidence":"a dog barking","description":"A dog barking intermittently.","activity":[1,8],"motion":"static","start":[90,0,2],"constraints":[]}],"scene_constraints":[],"relations":[],"english_only":true}
For linear motion also provide end:[azimuth,elevation,distance]. Speech has speaker_description and transcript instead of description; core covers only requested voice properties, and a transcript constraint preserves requested words. For each source, evidence must be a nonempty exact substring of the request that identifies that sound. Source keys are arbitrary unique English labels and only for internal bookkeeping.
Each constraint must have evidence copied verbatim from the request and origin equal to explicit, entailed, or relative. Capture ALL requested properties and relations. Never annotate a number you freely chose as a user requirement.
Critical annotation rules: core must identify the source and every requested salient voice attribute; for 'a woman calmly says', core is 'A woman speaking calmly', not just 'calmly'. A location stated without a movement qualifier ('on my right', 'behind me') applies to both endpoints; do not check only start. 'Directly ahead' means azimuth 0 at both endpoints, encoded as numeric constraints with origin entailed; unqualified 'ahead' may use the front sector. Add an explicit linear motion constraint for crossing or approaching. 'Under twelve seconds' is an upper bound, NOT an exact duration of twelve: use duration_range, choose a duration at least 0.05 seconds below the bound, and do not add a numeric equality. Likewise distinguish at-least and at-most from exact numbers. If speech timing is unspecified, use a natural speaking interval around two to three words per second with padding; do not stretch a five-word utterance across twelve seconds.
Keep requirement bounds exact: for 'under twelve seconds', record max 12, not 11.95. The 0.05-second planning margin is a free choice in the target plan and must never tighten the user's acceptance bound. When you encode 'directly ahead', include BOTH start.azimuth_deg and end.azimuth_deg constraints with value 0.
Allowed source constraints:
{"op":"numeric","field":"onset_sec","value":2,"evidence":"at two seconds","origin":"explicit"}; numeric fields: onset_sec, offset_sec, start.azimuth_deg, start.elevation_deg, start.distance_m, end.azimuth_deg, end.elevation_deg, end.distance_m. Do not supply tolerances.
{"op":"motion","value":"static","evidence":"stays still","origin":"explicit"}; value static or linear.
{"op":"sector","point":"both","value":"left","evidence":"on my left","origin":"relative"}; point start/end/both/path; value left/right/front/behind/above/below. For a crossing use linear plus separate start and end sectors; path can constrain all intermediate points.
{"op":"distance_change","value":"approaching","evidence":"gets closer","origin":"relative"}; approaching or receding.
{"op":"distance_range","point":"both","min":0.5,"max":2.5,"evidence":"nearby","origin":"relative"}; use a broad reasonable interpretation only when closeness is requested.
{"op":"full_scene","evidence":"throughout the clip","origin":"relative"}.
{"op":"transcript","value":"Hello there.","evidence":"say Hello there.","origin":"explicit"}; value must itself occur within the evidence.
Allowed scene_constraints: numeric field duration_sec; room with value dry/moderate/reverberant/outdoor; duration_range with min/max. All have verbatim evidence and origin. If no duration or room is requested, scene_constraints can be empty.
Allowed relations are {"op":"starts_after","a":"dog","b":"rain","evidence":"the dog comes in later","origin":"relative"}. a and b reference source keys. starts_after compares onsets; starts_after_end requires a's onset after b's offset; ends_before compares offsets; overlaps requires a nonempty temporal overlap; during contains a's interval within b; starts_with/ends_with allow 0.25 seconds. nearer_than/left_of/right_of/higher_than compare a to b at point start/end/both (default start). An optional nonnegative min_gap encodes an explicitly requested numeric difference, not an invented one. Do not use unsupported operation names.
Every requested relation must be reflected both in annotations and in the chosen plan. Add no spatial or temporal constraint for information the request leaves open. Check your JSON and all intervals before returning it.'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); temporary.replace(path)


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def convert(seed, value, codec, constraints_module):
    if value.get('english_only') is not True: raise ValueError('teacher did not attest English text')
    req = {'schema': constraints_module.SCHEMA, 'output_order': None, 'sources': [],
           'scene': value['scene_constraints'], 'relations': value['relations']}
    plan = {'sample_id': seed['id'], 'duration_sec': value['duration'], 'room': {'type': value['room']}, 'sources': []}
    bindings = {}; provenance = []
    for i, row in enumerate(value['sources']):
        sid = f'source_{i}'; bindings[row['key']] = sid
        position = lambda v: dict(zip(('azimuth_deg', 'elevation_deg', 'distance_m'), v)) if len(v) == 3 else (_ for _ in ()).throw(ValueError('position must have three numbers'))
        trajectory = {'type': row['motion']}
        if row['motion'] == 'static': trajectory['position'] = position(row['start'])
        elif row['motion'] == 'linear': trajectory.update(start=position(row['start']), end=position(row['end']))
        else: raise ValueError('unsupported teacher trajectory')
        source = {'source_id': sid, 'kind': row['kind'], 'gain_db': 0.,
                  'activity': dict(zip(('onset_sec', 'offset_sec'), row['activity'])), 'trajectory': trajectory}
        if len(row['activity']) != 2: raise ValueError('activity must have two endpoints')
        fields = ('speaker_description', 'transcript') if row['kind'] == 'speech' else ('description',)
        for field in fields: source[field] = row[field]
        plan['sources'].append(source)
        req['sources'].append({k: row[k] for k in ('key', 'kind', 'core', 'evidence', 'constraints')})
        for c in row['constraints']:
            if c.get('origin') not in ('explicit', 'entailed', 'relative'): raise ValueError('missing constraint origin')
        explicit_fields = {c['field'] for c in row['constraints'] if c['op'] == 'numeric'}
        for field in ('onset_sec', 'offset_sec', 'start.azimuth_deg', 'start.elevation_deg', 'start.distance_m', 'end.azimuth_deg', 'end.elevation_deg', 'end.distance_m'):
            relevant = [c for c in row['constraints'] if c.get('field') == field]
            relative = [c for c in row['constraints'] if c['op'] in ('sector', 'distance_change', 'distance_range', 'full_scene')]
            relations = [c for c in req['relations'] if row['key'] in (c['a'], c['b'])]
            # Conservative dependency marking: retain all relative clauses for
            # review rather than invent exact values from the witness plan.
            provenance.append({'source': row['key'], 'field': field,
                               'origin': 'explicit_or_entailed' if field in explicit_fields else 'completion_subject_to_relations' if relative or relations else 'free_completion',
                               'numeric_evidence': relevant, 'relative_context': relative + relations})
    constraints_module.validate_requirements(seed['request'], req)
    if len(plan['sources']) != seed['expected_count_for_quality_only']: raise ValueError('teacher count disagrees with frozen request annotation')
    labels = {(r['key'], s['source_id']): bindings[r['key']] == s['source_id'] for r in req['sources'] for s in plan['sources']}
    before = constraints_module.evaluate_natural_request(seed['request'], req, plan, labels)
    if not before['request_constraints_joint']: raise ValueError('teacher witness violates its own request constraints before codec projection')
    projected = codec.project_plan(plan)
    after = constraints_module.evaluate_natural_request(seed['request'], req, projected, labels)
    if not after['request_constraints_joint']: raise ValueError('codec projection violates request constraints')
    encoded = codec.encode(projected); tokens = encoded['input_ids'].tolist()
    decoded = codec.decode(tokens, sample_id=seed['id'])
    if decoded != projected: raise ValueError('codec text/numeric round trip changed teacher plan')
    all_text = [seed['request']] + [s[field] for s in plan['sources'] for field in (('speaker_description', 'transcript') if s['kind'] == 'speech' else ('description',))]
    if any(any('\u3400' <= c <= '\u9fff' or '\u0400' <= c <= '\u052f' or '\u0600' <= c <= '\u06ff' for c in text) for text in all_text):
        raise ValueError('non-English-script screening failure; independent English QA still required')
    return {**seed, 'schema': 'generation_ar_natural_pair_v2', 'requirements': req, 'target_sceneplan': projected,
            'teacher_unprojected_plan': plan, 'source_bindings': bindings, 'completion_provenance': provenance,
            'target_token_ids': tokens, 'deterministic_quality': after,
            'semantic_pairing_review': 'PENDING_INDEPENDENT_AUDIT', 'english_review': 'TEACHER_ATTESTATION_AND_SCRIPT_SCREEN_ONLY'}


def run(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1,2'
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'LOCK').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.wait_for:
        atomic(args.output / 'STATUS.json', {'status': 'WAITING_FOR_SEMANTIC_JOB', 'pid': os.getpid()})
        while True:
            status = json.loads(args.wait_for.read_text()) if args.wait_for.exists() else {}
            if status.get('status') == 'COMPLETE': break
            if status.get('status') == 'FAILED': raise RuntimeError('prerequisite semantic job failed')
            time.sleep(60)
        time.sleep(5)
    import subprocess
    busy = subprocess.check_output(['nvidia-smi', '-i', '1,2', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if busy: raise RuntimeError('GPUs 1,2 still occupied; no processes were terminated: ' + busy)
    started = time.monotonic()
    parent = json.loads((args.judge / 'CONTRACT.json').read_text())
    seeds = json.loads((args.root / 'request_first_seeds.json').read_text())['seeds']
    if args.gate:
        # One example per count, including requested speech and relative motion.
        selected = {f'natural_request_first_v2/train/{n}/{i}' for n, i in [(1,4),(2,3),(3,0),(4,0)]}
        seeds = [s for s in seeds if s['id'] in selected]
    assert len(seeds) == (4 if args.gate else 48)
    identity = {'schema': 'generation_ar_request_first_teacher_v1', 'seeds_sha256': sha(args.root / 'request_first_seeds.json'),
                'script_sha256': sha(Path(__file__)), 'constraint_code_sha256': sha(args.constraints),
                'teacher_asset_contract_sha256': sha(args.judge / 'CONTRACT.json'), 'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
                'input_policy': 'Only system output contract and raw English request; no expected_count or target input',
                'ids': [s['id'] for s in seeds], 'batch_size': args.batch_size, 'max_new_tokens': 2048,
                'seed': 42, 'sampling': 'greedy, thinking disabled', 'wall_cap_s': args.max_wall_seconds, 'gpu_scope': [1,2], 'test_used': False,
                'reused_reviewed_records_sha256':sha(args.reviewed_records) if args.reviewed_records else None}
    if (args.output / 'CONTRACT.json').exists(): assert json.loads((args.output / 'CONTRACT.json').read_text()) == identity
    else: atomic(args.output / 'CONTRACT.json', identity)
    (args.output / 'prompt.txt').write_text(PROMPT + '\n')
    sys.path.insert(0, str(args.snapshot)); sys.path.insert(0, parent['dependencies'])
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    codec = ModelScenePlanCodecV4(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4")
    module = module_at('frozen_natural_constraints', args.constraints)
    torch.set_num_threads(8); torch.manual_seed(42)
    atomic(args.output / 'STATUS.json', {'status': 'LOADING', 'pid': os.getpid()})
    tokenizer = AutoTokenizer.from_pretrained(parent['model'], local_files_only=True); tokenizer.padding_side = 'left'
    model = Qwen3_5ForConditionalGeneration.from_pretrained(parent['model'], local_files_only=True, dtype=torch.bfloat16,
        device_map='auto', max_memory={0:'42GiB',1:'42GiB'}, attn_implementation='sdpa')
    assert set(model.hf_device_map.values()).issubset({0,1,'cuda:0','cuda:1'})
    model.eval(); device = model.get_input_embeddings().weight.device
    db = sqlite3.connect(args.output / 'results.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,status TEXT NOT NULL,payload TEXT NOT NULL)')
    if args.reviewed_records:
        prior = json.loads(args.reviewed_records.read_text()); by_id = {s['id']:s for s in seeds}
        for record in prior['records']:
            sid = record['id']; seed = by_id[sid]; pair = record['pair']
            assert record['request'] == seed['request'] and pair['request'] == seed['request']
            assert pair['semantic_pairing_review'] == 'CODEX_AUDITED_WITH_RECORDED_ANNOTATION_REPAIRS'
            labels = {(r['key'],s['source_id']):pair['source_bindings'][r['key']]==s['source_id'] for r in pair['requirements']['sources'] for s in pair['target_sceneplan']['sources']}
            assert module.evaluate_natural_request(seed['request'],pair['requirements'],pair['target_sceneplan'],labels)['request_constraints_joint']
            old = db.execute('SELECT payload FROM results WHERE id=?',(sid,)).fetchone(); payload=json.dumps(record,ensure_ascii=False)
            if old: assert old[0] == payload
            else: db.execute('INSERT INTO results VALUES (?,?,?)',(sid,'REVIEWED_SEED_REUSED',payload))
        db.commit()
    done = {r[0] for r in db.execute('SELECT id FROM results')}; pending = [s for s in seeds if s['id'] not in done]
    generated_tokens = 0
    for offset in range(0, len(pending), args.batch_size):
        if time.monotonic() - started > args.max_wall_seconds: raise TimeoutError('N0 teacher generation budget exceeded')
        batch = pending[offset:offset+args.batch_size]
        texts = [tokenizer.apply_chat_template([{'role':'system','content':PROMPT},{'role':'user','content':s['request']}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for s in batch]
        inputs = tokenizer(texts, padding=True, return_tensors='pt').to(device)
        atomic(args.output / 'STATUS.json', {'status':'GENERATING','pid':os.getpid(),'rows_done':len(done)+offset,
               'rows':len(seeds),'batch_rows':len(batch),'elapsed_s':time.monotonic()-started})
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=identity['max_new_tokens'], do_sample=False, use_cache=True,
                                 max_time=max(1.,args.max_wall_seconds-(time.monotonic()-started)),pad_token_id=tokenizer.pad_token_id)
        output_ids = ids[:, inputs['input_ids'].shape[1]:]; outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        generated_tokens += sum(int((row != tokenizer.pad_token_id).sum()) for row in output_ids)
        for seed, raw in zip(batch, outputs):
            record = {'id': seed['id'], 'request': seed['request'], 'raw_teacher_output': raw,
                      'teacher_model_input_sha256': hashlib.sha256(seed['request'].encode()).hexdigest(),
                      'generation_provenance_contract_sha256':sha(args.output/'CONTRACT.json')}
            try:
                first, last = raw.find('{'), raw.rfind('}'); value = json.loads(raw[first:last+1])
                record['pair'] = convert(seed, value, codec, module); status = 'DETERMINISTIC_PASS_SEMANTIC_REVIEW_PENDING'
            except Exception as exc: status = 'REJECTED'; record['error'] = f'{type(exc).__name__}: {exc}'
            db.execute('INSERT INTO results VALUES (?,?,?)', (seed['id'], status, json.dumps(record, ensure_ascii=False)))
        db.commit(); atomic(args.output / 'STATUS.json', {'status':'RUNNING','rows_done':len(done)+offset+len(batch),'rows':len(seeds),
                'elapsed_s':time.monotonic()-started,'generated_tokens':generated_tokens})
    counts = dict(db.execute('SELECT status,COUNT(*) FROM results GROUP BY status')); db.close()
    summary = {'status':'COMPLETE_GENERATIONS_REVIEW_PENDING','rows':len(seeds),'status_counts':counts,
               'elapsed_generation_s':time.monotonic()-started,'generated_tokens':generated_tokens,
               'new_generations':len(pending),'reused_reviewed_records':len(done),
               'generated_tokens_per_s':generated_tokens/max(time.monotonic()-started,1e-6), 'gate':args.gate, 'trainable_data_accepted':False}
    atomic(args.output / 'SUMMARY.json', summary); atomic(args.output / 'STATUS.json', summary); print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--judge',type=Path,required=True); parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--constraints',type=Path,required=True); parser.add_argument('--gate',action='store_true')
    parser.add_argument('--reviewed-records',type=Path)
    parser.add_argument('--wait-for',type=Path); parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--max-wall-seconds',type=int,default=3600); args=parser.parse_args()
    try: run(args)
    except BaseException as exc:
        args.output.mkdir(parents=True,exist_ok=True); atomic(args.output/'STATUS.json',{'status':'FAILED','error':f'{type(exc).__name__}: {exc}'})
        raise
