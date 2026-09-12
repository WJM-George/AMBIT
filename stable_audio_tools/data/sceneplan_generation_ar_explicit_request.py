"""Decode facts explicitly stated in the existing Generation request protocol.

This is a request/data audit and a possible source of inference constraints,
not a learned AR prediction. Only the existing closed request contract is
accepted. Unknown syntax or contradictory extra instructions are rejected.
No target plan, sample metadata or renderer lineage is accepted as input.
"""
from __future__ import annotations
import hashlib
import math
import re
from stable_audio_tools.data.model_sceneplan import MAX_DURATION_SEC, validate_model_sceneplan
from stable_audio_tools.data.model_sceneplan_codec_v3 import DISTANCE_MIN_M, DISTANCE_MAX_M
from stable_audio_tools.data.sceneplan_generation_ar_request_sources import request_source_spans

NUMBER=r'[+-]?\d+(?:\.\d+)?'
GUIDE='Position triples mean (azimuth degrees, elevation degrees, distance millimeters).'
FOOTERS={
    'Use this order, fixed 0 dB gains, and return the complete ScenePlan.',
    'Return one complete ScenePlan in this order with every gain at 0 dB.',
    'Output the full ScenePlan; keep this order and set every gain to 0 dB.',
    'Preserve the listed order and canonical 0 dB gains in one complete ScenePlan.',
}
OPENERS=[rf'Create (?P<duration>{NUMBER})s of FOA audio',
         rf'Generate a (?P<duration>{NUMBER})s first-order Ambisonic scene',
         rf'I need (?P<duration>{NUMBER})s of spatial FOA audio',
         rf'Render a (?P<duration>{NUMBER})s FOA scene']


def _quoted_field(body,name):
    match=re.match(re.escape(name)+r'\s*=\s*“',body,re.IGNORECASE)
    if match is None:raise ValueError('expected explicitly quoted '+name)
    start=match.end();depth=1;stop=None
    for index in range(start,len(body)):
        if body[index]=='“':depth+=1
        elif body[index]=='”':
            depth-=1
            if depth==0:stop=index;break
    if stop is None:raise ValueError('unclosed quote in '+name)
    value=body[start:stop]
    if not value.strip():raise ValueError('empty '+name)
    tail=body[stop+1:].lstrip()
    if not tail.startswith(';'):raise ValueError('expected field separator after '+name)
    return value,tail[1:].lstrip()


def _position(value,codec):
    pieces=[float(x.strip()) for x in value.split(',')]
    if len(pieces)!=3 or not all(math.isfinite(x) for x in pieces):raise ValueError('invalid position triple')
    azimuth,elevation,millimeters=pieces;distance=millimeters/1000.
    if not -180<=azimuth<=180 or not -90<=elevation<=90 or not DISTANCE_MIN_M<=distance<=DISTANCE_MAX_M:
        raise ValueError('explicit position is outside the established codec range')
    return {key:codec.snap_numeric_to_grid(kind,raw) for key,kind,raw in [
        ('azimuth_deg','azimuth',azimuth),('elevation_deg','elevation',elevation),('distance_m','distance',distance)]}


def parse_explicit_generation_request(request,codec,*,sample_id='explicit-request-audit'):
    spans=request_source_spans(request)
    if not spans:raise ValueError('request does not have unambiguous contiguous numbered sources')
    header=' '.join(request[:spans[0].start].split());duration=None;room=None
    for opener in OPENERS:
        match=re.fullmatch(opener+r'; room type=(?P<room>dry|moderate|reverberant|outdoor)\. '+re.escape(GUIDE),header)
        if match:
            duration=float(match['duration']);room=match['room'];break
    if duration is None or not 0<duration<=MAX_DURATION_SEC+1e-5:raise ValueError('unsupported or out-of-range explicit request header')
    duration=codec.snap_numeric_to_grid('seconds',duration)
    if duration<=0:raise ValueError('duration rounds to zero frames')
    footer=' '.join(request[spans[-1].end:].split())
    if footer not in FOOTERS:raise ValueError('unsupported footer or additional instructions; cannot assume fixed gains/order')
    sources=[]
    for index,span in enumerate(spans):
        if index and request[spans[index-1].end:span.start].strip():raise ValueError('unparsed instruction between source clauses')
        clause=request[span.start:span.end];body=clause[clause.index(':')+1:].lstrip()
        match=re.match(re.escape(span.kind)+r'\s*;\s*',body,re.IGNORECASE)
        if not match:raise ValueError('source kind does not match explicit clause')
        body=body[match.end():]
        source={'source_id':f'source_{index}','kind':span.kind,'gain_db':0.}
        if span.kind=='speech':
            source['speaker_description'],body=_quoted_field(body,'speaker')
            source['transcript'],body=_quoted_field(body,'transcript')
        else:source['description'],body=_quoted_field(body,'description')
        match=re.match(rf'active from (?P<onset>{NUMBER}) to (?P<offset>{NUMBER})ms;\s*',body)
        if not match:raise ValueError('expected explicit onset/offset in milliseconds')
        start,stop=float(match['onset'])/1000.,float(match['offset'])/1000.
        if start<0 or stop<=start or stop>duration+.001:raise ValueError('invalid explicit activity interval')
        source['activity']={'onset_sec':codec.snap_numeric_to_grid('seconds',start),'offset_sec':codec.snap_numeric_to_grid('seconds',stop)}
        body=body[match.end():]
        triple=rf'({NUMBER}\s*,\s*{NUMBER}\s*,\s*{NUMBER})'
        static=re.fullmatch(rf'static at \({triple}\)\.',body)
        linear=re.fullmatch(rf'linear from \({triple}\) to \({triple}\)\.',body)
        if static:source['trajectory']={'type':'static','position':_position(static[1],codec)}
        elif linear:source['trajectory']={'type':'linear','start':_position(linear[1],codec),'end':_position(linear[2],codec)}
        else:raise ValueError('unsupported explicit trajectory or trailing source instructions')
        sources.append(source)
    plan=validate_model_sceneplan({'sample_id':sample_id,'duration_sec':duration,'room':{'type':room},'sources':sources})
    return {'schema':'generation_ar_explicit_request_facts_v1','origin':'request text only, not an AR prediction',
            'request_sha256':hashlib.sha256(request.encode()).hexdigest(),'plan':plan,
            'source_spans':[span._asdict() for span in spans],
            'numeric_policy':'nearest existing codec frame/bin after explicit seconds/ms/mm conversion; no target values used'}
