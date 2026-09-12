"""Request-sourced natural-English constraints; never an AR model input.

Reference plans are optional witnesses, not unique numerical answers. A single
global source assignment is shared by semantics, every field and all relations.
Semantic labels must be supplied by a separately calibrated judge/annotation.
"""
from __future__ import annotations

import itertools
import math
import re
from typing import Mapping

from stable_audio_tools.data.model_sceneplan import validate_model_sceneplan

SCHEMA = 'generation_ar_natural_requirements_v2'
NUMERIC_TOLERANCES = {'duration_sec': .25, 'event_duration_sec': .25, 'onset_sec': .25, 'offset_sec': .25,
                      'azimuth_deg': 10., 'elevation_deg': 5., 'distance_m': .25}
SOURCE_OPS = {'numeric', 'motion', 'sector', 'direct', 'angular_direction', 'distance_range', 'distance_change', 'full_scene', 'ends_scene', 'starts_scene', 'transcript', 'compass', 'time_phase'}
COMPASS_CENTERS = {'front': 0., 'front_left': 45., 'left': 90., 'rear_left': 135.,
                   'rear': 180., 'rear_right': -135., 'right': -90., 'front_right': -45.}
COMPASS_HALF_WIDTH = 27.5  # 22.5-degree cell plus a fixed 5-degree boundary tolerance.
TIME_PHASE_RANGES = {'beginning': (0., .2), 'early': (.1, .4), 'middle': (.3, .7),
                     'late': (.6, .9), 'ending': (.8, 1.)}
RELATION_OPS = {'starts_after', 'starts_after_end', 'ends_before', 'overlaps', 'during', 'starts_with',
                'ends_with', 'nearer_than', 'left_of', 'right_of', 'higher_than'}
SECTORS = {'left', 'right', 'front', 'behind', 'above', 'below'}
FRAME_SECONDS = 1024 / 44100


def normalized_words(text):
    return re.findall(r"[a-z0-9]+(?:['’][a-z0-9]+)*", text.lower().replace('’', "'"))


def _evidence(request, evidence):
    if not isinstance(evidence, str) or not evidence.strip() or evidence not in request:
        raise ValueError('constraint evidence must be a nonempty verbatim request span')


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('expected a finite number')
    return float(value)


def validate_requirements(request: str, requirements: Mapping):
    """Check annotation shape and traceability, not English semantic truth."""
    if requirements.get('schema') != SCHEMA:
        raise ValueError('unknown natural requirement schema')
    if not isinstance(request, str) or not request.strip():
        raise ValueError('empty request')
    sources = requirements['sources']
    if not 1 <= len(sources) <= 4 or len({s['key'] for s in sources}) != len(sources):
        raise ValueError('expected one to four distinct requested source identities')
    if requirements.get('output_order') is not None and requirements['output_order'] != [s['key'] for s in sources]:
        raise ValueError('ordered annotations must list the requested source order')
    if requirements.get('output_order') is not None:
        _evidence(request, requirements.get('order_evidence'))
    for source in sources:
        if source['kind'] not in ('sound', 'music', 'speech') or not source['core'].strip():
            raise ValueError('source kind/core missing')
        _evidence(request, source['evidence'])
        for c in source.get('constraints', []):
            _evidence(request, c['evidence'])
            if c['op'] not in SOURCE_OPS:
                raise ValueError('unsupported source constraint')
            if c['op'] == 'numeric':
                _validate_numeric(c, scene=False)
            elif c['op'] == 'motion' and c['value'] not in ('static', 'linear'):
                raise ValueError('unsupported motion constraint')
            elif c['op'] == 'distance_change' and c['value'] not in ('approaching', 'receding'):
                raise ValueError('unknown radial movement')
            elif c['op'] == 'compass':
                if c['value'] not in COMPASS_CENTERS or c.get('point') not in ('start', 'end', 'both'):
                    raise ValueError('unknown listener-centered compass endpoint')
            elif c['op'] == 'time_phase':
                if c['value'] not in TIME_PHASE_RANGES or c.get('field') not in ('onset_sec', 'offset_sec'):
                    raise ValueError('unknown relative scene time phase')
            elif c['op'] in ('direct', 'angular_direction'):
                if c['value'] not in SECTORS: raise ValueError('unknown direction')
                if c['op'] == 'direct' and c.get('point') not in ('start', 'end', 'both', 'path'):
                    raise ValueError('unknown direction endpoint')
            elif c['op'] in ('sector', 'distance_range'):
                if c.get('point') not in ('start', 'end', 'both', 'path'):
                    raise ValueError('unknown constraint endpoint/path')
                if c['op'] == 'sector' and c['value'] not in SECTORS:
                    raise ValueError('unknown sector')
                if c['op'] == 'distance_range' and not 0 < _number(c['min']) <= _number(c['max']):
                    raise ValueError('invalid distance range')
            elif c['op'] == 'transcript':
                if source['kind'] != 'speech' or not normalized_words(c['value']):
                    raise ValueError('transcript constraint requires speech and words')
                if c['value'] not in c['evidence']:
                    raise ValueError('requested utterance must occur verbatim in its evidence')
    if sum(s['kind'] == 'speech' for s in sources) > 1:
        raise ValueError('existing P10 schema supports at most one formal speech source')
    keys = {s['key'] for s in sources}
    for c in requirements.get('relations', []):
        _evidence(request, c['evidence'])
        if c['op'] not in RELATION_OPS or c['a'] not in keys or c['b'] not in keys or c['a'] == c['b']:
            raise ValueError('invalid source relation')
        if 'min_gap' in c and _number(c['min_gap']) < 0:
            raise ValueError('relative margin must be nonnegative')
        if c.get('point', 'start') not in ('start', 'end', 'both'):
            raise ValueError('invalid relation endpoints')
    for c in requirements.get('scene', []):
        _evidence(request, c['evidence'])
        if c['op'] == 'numeric':
            _validate_numeric(c, scene=True)
        elif c['op'] == 'room' and c['value'] in ('dry', 'moderate', 'reverberant', 'outdoor'):
            pass
        elif c['op'] == 'duration_range' and 0 <= _number(c['min']) < _number(c['max']):
            pass
        else:
            raise ValueError('unsupported scene constraint')
    return requirements


def _validate_numeric(c, *, scene):
    _number(c['value'])
    allowed = {'duration_sec'} if scene else {
        'onset_sec', 'offset_sec', 'event_duration_sec', 'start.azimuth_deg', 'start.elevation_deg',
        'start.distance_m', 'end.azimuth_deg', 'end.elevation_deg', 'end.distance_m'}
    if c['field'] not in allowed or 'tolerance' in c:
        raise ValueError('unknown numeric field or attempt to override frozen tolerance')


def _position(source, point):
    trajectory = source['trajectory']
    if trajectory['type'] == 'static':
        return trajectory['position']
    if trajectory['type'] != 'linear':
        raise ValueError('natural Generation codec scope is static/linear')
    return trajectory[point]


def _positions(source, point):
    if point in ('start', 'end'):
        return [_position(source, point)]
    start, end = _position(source, 'start'), _position(source, 'end')
    if point == 'both':
        return [start, end]
    # Match P10's shortest azimuth interpolation, including the 180-degree tie.
    delta = (end['azimuth_deg'] - start['azimuth_deg'] + 180) % 360 - 180
    return [{k: (start[k] + t * delta if k == 'azimuth_deg' else start[k] + t * (end[k] - start[k]))
             for k in start} for t in (.25, .5, .75)]


def _sector(position, name):
    az = math.radians(position['azimuth_deg'])
    # Exclude the 10-degree center/side boundaries where rounding is ambiguous.
    margin = math.sin(math.radians(10))
    return {'left': math.sin(az) > margin, 'right': math.sin(az) < -margin,
            'front': math.cos(az) > margin, 'behind': math.cos(az) < -margin,
            'above': position['elevation_deg'] > 5, 'below': position['elevation_deg'] < -5}[name]


def _direction_component(position, direction):
    az, el = math.radians(position['azimuth_deg']), math.radians(position['elevation_deg'])
    return {'front': math.cos(az)*math.cos(el), 'behind': -math.cos(az)*math.cos(el),
            'left': math.sin(az)*math.cos(el), 'right': -math.sin(az)*math.cos(el),
            'above': math.sin(el), 'below': -math.sin(el)}[direction]


def _numeric(observed, requested, field):
    error = abs(float(observed) - float(requested))
    name = field.split('.')[-1]
    if name == 'azimuth_deg':
        error = abs((float(observed) - float(requested) + 180) % 360 - 180)
    return error <= NUMERIC_TOLERANCES[name] + 1e-9


def source_constraint_pass(source, c, *, scene_duration=None):
    if source is None:
        return False
    op = c['op']
    if op == 'numeric':
        field = c['field']
        if field == 'event_duration_sec':
            observed = source['activity']['offset_sec'] - source['activity']['onset_sec']
        elif '.' in field:
            point, name = field.split('.')
            observed = _position(source, point)[name]
        else:
            observed = source['activity'][field]
        return _numeric(observed, c['value'], field)
    if op == 'motion':
        return source['trajectory']['type'] == c['value']
    if op == 'compass':
        return all(abs((p['azimuth_deg'] - COMPASS_CENTERS[c['value']] + 180) % 360 - 180)
                   <= COMPASS_HALF_WIDTH + 1e-9 for p in _positions(source, c['point']))
    if op == 'time_phase':
        if scene_duration is None or scene_duration <= 0:
            raise ValueError('time phase needs the predicted scene duration')
        lower, upper = TIME_PHASE_RANGES[c['value']]
        return lower - 1e-9 <= source['activity'][c['field']] / scene_duration <= upper + 1e-9
    if op == 'sector':
        return all(_sector(p, c['value']) for p in _positions(source, c['point']))
    if op == 'direct':
        return all(_direction_component(p,c['value']) >= math.cos(math.radians(10)) for p in _positions(source,c['point']))
    if op == 'angular_direction':
        return _direction_component(_position(source,'end'),c['value']) > _direction_component(_position(source,'start'),c['value']) + 1e-6
    if op == 'distance_range':
        return all(c['min'] <= p['distance_m'] <= c['max'] for p in _positions(source, c['point']))
    if op == 'distance_change':
        delta = _position(source, 'end')['distance_m'] - _position(source, 'start')['distance_m']
        return delta < -.25 if c['value'] == 'approaching' else delta > .25
    if op == 'full_scene':
        if scene_duration is None: raise ValueError('full_scene needs the predicted duration')
        return source['activity']['onset_sec'] <= .25 and abs(source['activity']['offset_sec'] - scene_duration) <= .25
    if op == 'starts_scene':
        return source['activity']['onset_sec'] <= .25
    if op == 'ends_scene':
        if scene_duration is None: raise ValueError('ends_scene needs the predicted duration')
        return abs(source['activity']['offset_sec'] - scene_duration) <= .25
    if op == 'transcript':
        return normalized_words(source.get('transcript', '')) == normalized_words(c['value'])
    raise ValueError('unknown constraint operation')


def relation_pass(a, b, c):
    if a is None or b is None:
        return False
    aa, ba = a['activity'], b['activity']; op = c['op']; margin = c.get('min_gap', 0.)
    if op == 'starts_after':
        return aa['onset_sec'] - ba['onset_sec'] >= max(margin, FRAME_SECONDS / 2)
    if op == 'starts_after_end':
        return aa['onset_sec'] - ba['offset_sec'] >= margin - 1e-6
    if op == 'ends_before':
        return ba['offset_sec'] - aa['offset_sec'] >= max(margin, FRAME_SECONDS / 2)
    if op == 'overlaps':
        return min(aa['offset_sec'], ba['offset_sec']) - max(aa['onset_sec'], ba['onset_sec']) >= max(margin, FRAME_SECONDS / 2)
    if op == 'during':
        return aa['onset_sec'] >= ba['onset_sec'] - 1e-6 and aa['offset_sec'] <= ba['offset_sec'] + 1e-6
    if op in ('starts_with', 'ends_with'):
        field = 'onset_sec' if op == 'starts_with' else 'offset_sec'
        return abs(aa[field] - ba[field]) <= .25
    points = ('start', 'end') if c.get('point', 'start') == 'both' else (c.get('point', 'start'),)
    for point in points:
        ap, bp = _position(a, point), _position(b, point)
        if op == 'nearer_than':
            gap = bp['distance_m'] - ap['distance_m']
        elif op == 'higher_than':
            gap = ap['elevation_deg'] - bp['elevation_deg']
        else:
            # Listener-centered lateral positions, positive towards the left.
            ay = ap['distance_m'] * math.sin(math.radians(ap['azimuth_deg']))
            by = bp['distance_m'] * math.sin(math.radians(bp['azimuth_deg']))
            gap = ay - by if op == 'left_of' else by - ay
        if gap <= margin + 1e-6:
            return False
    return True


def _assignments(keys, predictions, ordered):
    ids = list(predictions)
    if ordered:
        yield dict(zip(keys, ids + [None] * max(0, len(keys) - len(ids))))
        return
    pool = ids + [None] * max(0, len(keys) - len(ids))
    for values in set(itertools.permutations(pool, len(keys))):
        yield dict(zip(keys, values))


def evaluate_natural_request(request, requirements, prediction, semantic_labels, *, completion_reasonable=None):
    """Labels indexed by (request source key, predicted source_id), bool/None.

    No target ScenePlan argument exists. Unspecified fields are never compared
    with a hidden witness. Completion reasonableness remains a separate gate.
    """
    validate_requirements(request, requirements)
    refs = requirements['sources']; keys = [r['key'] for r in refs]
    try:
        validate_model_sceneplan(prediction)
        if any(s['trajectory']['type'] not in ('static', 'linear') for s in prediction['sources']):
            raise ValueError('outside the established Generation trajectory scope')
        if any(not .1 <= p['distance_m'] <= 50 for s in prediction['sources'] for p in _positions(s, 'both')):
            raise ValueError('outside codec distance range')
        sources = {s['source_id']: s for s in prediction['sources']}; valid = True
    except (KeyError, TypeError, ValueError):
        sources = {}; valid = False
    count_correct = valid and len(sources) == len(refs)
    scene_results = []
    for c in requirements.get('scene', []):
        if not valid: passed = False
        elif c['op'] == 'numeric': passed = _numeric(prediction['duration_sec'], c['value'], c['field'])
        elif c['op'] == 'duration_range':
            duration = prediction['duration_sec']
            passed = c['min'] <= duration and (duration <= c['max'] if c.get('max_inclusive', True) else duration < c['max'])
        else: passed = prediction['room']['type'] == c['value']
        scene_results.append({'constraint': c, 'pass': bool(passed)})
    best = None
    for assignment in _assignments(keys, sources, requirements.get('output_order') is not None):
        rows = []
        for ref in refs:
            sid = assignment[ref['key']]; source = sources.get(sid)
            kind_ok = source is not None and source['kind'] == ref['kind']
            semantic = semantic_labels.get((ref['key'], sid)) if kind_ok else False
            if semantic is not None and type(semantic) is not bool:
                raise ValueError('semantic labels must be bool or None')
            checks = [{'constraint': c, 'pass': bool(kind_ok and source_constraint_pass(source, c, scene_duration=prediction['duration_sec']))}
                      for c in ref.get('constraints', [])]
            rows.append({'key': ref['key'], 'prediction_source_id': sid, 'matched': source is not None,
                         'kind': kind_ok, 'core_semantics': semantic, 'constraints': checks,
                         'all_requested_pass': semantic is True and all(c['pass'] for c in checks)})
        relations = [{'constraint': c, 'pass': relation_pass(sources.get(assignment[c['a']]), sources.get(assignment[c['b']]), c)}
                     for c in requirements.get('relations', [])]
        score = (sum(r['core_semantics'] is True for r in rows),
                 sum(r['all_requested_pass'] for r in rows),
                 sum(c['pass'] for r in rows for c in r['constraints']) + sum(c['pass'] for c in relations))
        tie = tuple(sid or '~missing' for sid in assignment.values())
        if best is None or score > best[0] or (score == best[0] and tie < best[1]):
            best = (score, tie, rows, relations, assignment)
    _, _, rows, relations, assignment = best
    constraint_joint = (count_correct and all(r['all_requested_pass'] for r in rows)
                        and all(c['pass'] for c in relations + scene_results))
    return {'schema': SCHEMA, 'valid': valid, 'requested_count': len(refs), 'predicted_count': len(sources),
            'count_correct': count_correct, 'missing': max(0, len(refs)-len(sources)), 'extra': max(0, len(sources)-len(refs)),
            'assignment': assignment, 'sources': rows, 'relations': relations, 'scene_constraints': scene_results,
            'request_constraints_joint': bool(constraint_joint), 'completion_reasonable': completion_reasonable,
            'acceptance_joint': bool(constraint_joint and completion_reasonable is True),
            'semantic_pending': any(r['core_semantics'] is None for r in rows),
            'hidden_target_fields_compared': False}
