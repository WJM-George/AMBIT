#!/usr/bin/env python3
"""Generate English request paraphrases for QA, retaining train-family lineage.

The teacher sees an existing raw request and a style instruction, never a
ScenePlan, expected count or annotation. Generated text is not accepted training
data until semantic constraints, bindings and witness compatibility are reviewed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import sys
import time


class ResourceYield(Exception):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp'); temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); temp.replace(path)


def run(args):
    c = json.loads(args.contract.read_text())
    task = c.get('task', 'paraphrase')
    assert task in ('paraphrase', 'review_paraphrase')
    assert c.get('gpu',2) in (0,1,2) and os.environ.get('CUDA_VISIBLE_DEVICES') == str(c.get('gpu',2))
    assert all(os.environ.get(k)==v for k,v in c.get('required_environment',{}).items())
    assert sha(Path(__file__)) == c['script_sha256'] and sha(c['input']) == c['input_sha256']
    assert sha(c['prompt']) == c['prompt_sha256']
    args.output.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(ResourceYield('GPU returned to higher-priority AR evaluation')))
    for dependency in reversed(c['dependency_paths']):
        sys.path.insert(0, dependency)
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
    torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    inputs_data = json.loads(Path(c['input']).read_text())['rows']
    assert len({r['id'] for r in inputs_data}) == len(inputs_data)
    status = lambda stage, **kw: atomic(args.output / 'STATUS.json', {'status': stage, 'pid': os.getpid(), 'elapsed_s': time.monotonic() - started, **kw})
    status('LOADING', teacher=c['teacher'])
    if c.get('asset_proof'):
        proof_path=Path(c['asset_proof']);assert sha(proof_path)==c['asset_proof_sha256']
        proof=json.loads(proof_path.read_text())
        for path,digest in proof['model_asset_sha256'].items():
            stat=Path(path).stat()
            if max(stat.st_mtime,stat.st_ctime)>=c['asset_verified_since']:
                assert sha(path)==digest,path
    tokenizer = AutoTokenizer.from_pretrained(c['model'], local_files_only=True); tokenizer.padding_side = 'left'
    kwargs = dict(local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa')
    if c['quantization'] == 'nf4_double_bf16':
        kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                      bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16), device_map={'': 0})
    model = Qwen3_5ForConditionalGeneration.from_pretrained(c['model'], **kwargs)
    if c['quantization'] is None:
        model.to(device)
    model.eval()
    assert all(p.device.type == 'cuda' and p.device.index == 0 for p in model.parameters())
    load_seconds = time.monotonic() - started
    status('MODEL_READY', load_seconds=load_seconds, memory_allocated_bytes=torch.cuda.memory_allocated())
    prompt = Path(c['prompt']).read_text().strip()
    db = sqlite3.connect(args.output / 'results.sqlite')
    db.execute('CREATE TABLE results(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    emitted_tokens = 0; generation_seconds = 0.; rows_done = 0

    class StopAtBudget(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return time.monotonic() - started >= c['wall_cap_seconds']

    try:
        for offset in range(0, len(inputs_data), c['batch_size']):
            if time.monotonic() - started >= c['wall_cap_seconds']:
                raise ResourceYield('Teacher pilot wall budget reached')
            batch = inputs_data[offset:offset + c['batch_size']]
            formatted = [tokenizer.apply_chat_template([
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': (
                    json.dumps({'original_request': row['original_request'], 'candidate_request': row['request']}, ensure_ascii=False)
                    if task == 'review_paraphrase' else
                    'Style: ' + row['style'] + '\nOriginal request: ' + row['original_request'])},
            ], tokenize=False, add_generation_prompt=True, enable_thinking=False) for row in batch]
            encoded = tokenizer(formatted, padding=True, return_tensors='pt').to(device)
            assert encoded['input_ids'].shape[1] <= c['max_input_tokens']
            before = time.monotonic(); status('GENERATING', rows_done=rows_done, rows=len(inputs_data), batch_rows=len(batch))
            with torch.inference_mode():
                generated = model.generate(**encoded, max_new_tokens=c['max_new_tokens'], do_sample=False,
                    use_cache=True, pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=StoppingCriteriaList([StopAtBudget()]))
            duration = time.monotonic() - before; generation_seconds += duration
            generated = generated[:, encoded['input_ids'].shape[1]:]
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for row, ids, text in zip(batch, generated.cpu().tolist(), decoded):
                text = text.strip(); nonpad = [x for x in ids if x != tokenizer.pad_token_id]
                emitted_tokens += len(nonpad); flags = []
                if not text or re.search(r'[\u3400-\u9fff]', text):
                    flags.append('empty_or_non_English_output')
                if len(nonpad) >= c['max_new_tokens']:
                    flags.append('generation_limit_reached_review_truncation')
                if re.search(r'\b(ScenePlan|source_\d|azimuth_bin|source_count)\b', text):
                    flags.append('unwanted_schema_language')
                original_quotes = re.findall(r'"([^"]+)"', row['original_request'])
                words = lambda value: re.findall(r"[a-z0-9]+(?:['’][a-z0-9]+)*", value.lower().replace('’', "'"))
                for phrase in original_quotes:
                    if ' '.join(words(phrase)) not in ' '.join(words(text)):
                        flags.append('quoted_words_missing')
                result = {**row, 'request': text, 'teacher': c['teacher'], 'automatic_flags': flags,
                          'semantic_pairing_status': 'PENDING_REVIEW', 'training_accepted': False,
                          'generated_token_count': len(nonpad), 'batch_seconds': duration,
                          'input_message_sha256': hashlib.sha256(formatted[batch.index(row)].encode()).hexdigest()}
                if task == 'review_paraphrase':
                    # A second teacher pass is evidence for QA, never an
                    # automatic training acceptance or a calibrated AR judge.
                    result['request'] = row['request']
                    result['review_text'] = text
                    result['automatic_flags'] = []
                    try:
                        review = json.loads(text)
                        assert review['verdict'] in ('PASS', 'FAIL', 'REVIEW')
                        assert isinstance(review['reason'], str) and review['reason'].strip()
                        assert len(nonpad) < c['max_new_tokens']
                        result['teacher_review'] = review
                    except (ValueError, KeyError, TypeError, AssertionError):
                        result['teacher_review'] = {'verdict': 'REVIEW', 'reason': 'Malformed or truncated teacher review.'}
                        result['automatic_flags'] = ['invalid_review_output']
                db.execute('INSERT INTO results VALUES (?,?)', (row['id'], json.dumps(result, ensure_ascii=False)))
            db.commit(); rows_done += len(batch)
            status('GENERATING', rows_done=rows_done, rows=len(inputs_data), generated_tokens=emitted_tokens)
        db.close()
        result = {'status': 'GENERATION_COMPLETE_QA_PENDING', 'rows': rows_done, 'load_seconds': load_seconds,
                  'generation_seconds': generation_seconds, 'generated_tokens': emitted_tokens,
                  'aggregate_generated_tokens_per_second': emitted_tokens / generation_seconds,
                  'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'elapsed_s': time.monotonic() - started,
                  'contract_sha256': sha(args.contract), 'training_accepted': False, 'test_used': False}
        atomic(args.output / 'RESULT.json', result); status(result['status'], **{k: result[k] for k in ('rows', 'generation_seconds', 'peak_gpu_bytes')})
        print(json.dumps(result), flush=True)
    finally:
        db.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--contract', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        run(args)
    except ResourceYield as exc:
        atomic(args.output / 'YIELD.json', {'status': 'YIELDED_FOR_RESOURCE_OR_BUDGET', 'reason': str(exc), 'results_preserved': True})
        atomic(args.output / 'STATUS.json', {'status': 'YIELDED_FOR_RESOURCE_OR_BUDGET', 'reason': str(exc)})
    except BaseException as exc:
        if args.output.exists():
            atomic(args.output / 'STATUS.json', {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'})
        raise
