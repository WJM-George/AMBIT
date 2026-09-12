"""Auditable whitespace-only normalization between AR decoding and P10.

For new inference/evaluation contracts only: retain raw output and both hashes.
Never use this to rewrite an already frozen evaluation or conceal field changes.
"""
from __future__ import annotations
import hashlib
import json
import struct

_TEXT_FIELDS={'description','speaker_description','transcript'}


def _whitespace_changes(raw,canonical,path=()):
    if isinstance(raw,dict) and isinstance(canonical,dict):
        if raw.keys()!=canonical.keys():raise ValueError(f'canonicalization changed fields at {path}')
        return [change for key in raw for change in _whitespace_changes(raw[key],canonical[key],(*path,key))]
    if isinstance(raw,list) and isinstance(canonical,list):
        if len(raw)!=len(canonical):raise ValueError(f'canonicalization changed list length at {path}')
        return [change for i,(left,right) in enumerate(zip(raw,canonical)) for change in _whitespace_changes(left,right,(*path,i))]
    if raw==canonical:return []
    if (path and path[-1] in _TEXT_FIELDS and isinstance(raw,str) and isinstance(canonical,str)
            and ' '.join(raw.split())==' '.join(canonical.split())):
        return [{'path':list(path),'raw':raw,'canonical':canonical}]
    raise ValueError(f'canonicalization changed a non-whitespace value at {path}: {raw!r} -> {canonical!r}')


def _json_sha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _token_sha(values):
    return hashlib.sha256(struct.pack('<'+'H'*len(values),*values)).hexdigest()


def normalize_generated_sceneplan(codec,token_ids,*,sample_id,max_tokens=None):
    """Return raw/canonical plans and proof that only text whitespace changed."""
    raw_tokens=[int(x) for x in token_ids]
    raw=codec.decode(raw_tokens,sample_id=sample_id)
    encoded=codec.canonicalize(raw_tokens,max_tokens=max_tokens)['input_ids']
    canonical_tokens=[int(x) for x in encoded]
    canonical=codec.decode(canonical_tokens,sample_id=sample_id)
    changes=_whitespace_changes(raw,canonical)
    return {'schema':'generation_ar_whitespace_only_p10_input_v1','sample_id':sample_id,
            'raw_plan':raw,'p10_plan':canonical,'raw_token_ids':raw_tokens,'p10_token_ids':canonical_tokens,
            'raw_plan_sha256':_json_sha(raw),'p10_plan_sha256':_json_sha(canonical),
            'raw_tokens_u16le_sha256':_token_sha(raw_tokens),'p10_tokens_u16le_sha256':_token_sha(canonical_tokens),
            'whitespace_changes':changes,'non_text_fields_unchanged':True}
