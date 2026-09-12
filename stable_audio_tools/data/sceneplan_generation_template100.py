"""Split-disjoint natural-English recipes for existing Generation ScenePlans.

Ten scene formulations times ten event formulations define the 100 training
recipes. Validation uses two new scene formulations and five new event forms;
test uses a third scene formulation and five further event forms. Model inputs
contain prose and quoted acoustic content, with no schema serialization.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import random
import re

CONTRACT = 'sceneplan_generation_templated_qualitative_v3'
NUMERIC_CONTRACT = 'request_qualitative_compass_time_v3'
SPLIT_COUNTS = {'train': 100, 'validation': 10, 'test': 5}
ROOMS = {'dry': 'dry acoustics', 'moderate': 'moderate reverberation',
         'reverberant': 'strong reverberation', 'outdoor': 'outdoor acoustics'}
ORDINALS = ('first', 'second', 'third', 'fourth')
COUNTS = ('one', 'two', 'three', 'four')

OPENERS = {
    'train': (
        'Create a scene around me with {room} and {count} sound sources.',
        'I would like to hear {count} sound sources around me, with {room}.',
        'Please make a spatial audio scene with {count} sources and {room}.',
        'Use {room} for a scene containing {count} sound sources around me.',
        'Could you arrange {count} sound sources around me in a scene with {room}?',
        'The audio should surround me with {count} sources and {room}.',
        'Make a scene with {count} sound sources around me and {room}.',
        'Build an audio scene using {room} and exactly {count} sound sources.',
        'For this scene around me, use {count} sound sources and {room}.',
        'Give me a spatial scene featuring {count} sound sources with {room}.',
    ),
    'validation': (
        'Let me hear a scene of {count} sound sources around me with {room}.',
        'Set up an audio scene around the listener with {room}, including {count} sound sources.',
    ),
    'test': ('Compose a spatial audio clip around me with {count} sources and {room}.',),
}

# Each event form binds its own content, activity interval and motion. Changing
# their mention order never changes which source owns a field.
EVENT_FORMS = {
    'train': (
        'For the {ordinal} source, use {identity}. {activity} {motion}',
        'The {ordinal} source is {identity}. {motion} {activity}',
        'Include {identity} as the {ordinal} source. {activity} {motion}',
        'For the {ordinal} source, {activity_lower} Use {identity}. {motion}',
        'For the {ordinal} source, {motion_lower} Use {identity}. {activity}',
        'The {ordinal} source should be {identity}. {activity} {motion}',
        'For the {ordinal} sound source: {identity}. {motion} {activity}',
        'Use {identity} for the {ordinal} sound. {activity} {motion}',
        'For the {ordinal} sound, use {identity}; {activity_lower} {motion}',
        'As the {ordinal} source, include {identity}. {motion} {activity}',
    ),
    'validation': (
        'For the {ordinal} source, I want {identity}. {motion} {activity}',
        'The {ordinal} sound should be {identity}. {activity} {motion}',
        'For the {ordinal} sound, {activity_lower} Its content is {identity}. {motion}',
        'Choose {identity} for the {ordinal} source. {motion} {activity}',
        'Let the {ordinal} source be {identity}. {activity} {motion}',
    ),
    'test': (
        'The {ordinal} sound is {identity}. {activity} {motion}',
        'For the {ordinal} sound, {motion_lower} Its content should be {identity}. {activity}',
        'Place {identity} in the scene as the {ordinal} sound. {activity} {motion}',
        'I want the {ordinal} sound to contain {identity}. {motion} {activity}',
        'For the {ordinal} source, {activity_lower} Include {identity}. {motion}',
    ),
}


def quoted(value):
    return '“' + ' '.join(str(value).split()) + '”'


COMPASS_NAMES = ('front', 'front_left', 'left', 'rear_left', 'rear', 'rear_right', 'right', 'front_right')
COMPASS_PHRASES = ('in front of me', 'in front of me on my left', 'on my left',
                   'behind me on my left', 'behind me', 'behind me on my right',
                   'on my right', 'in front of me on my right')
MOVING_PHRASES = dict(zip(COMPASS_NAMES, ('in front of me', 'my front left', 'my left',
                         'my back left', 'behind me', 'my back right', 'my right', 'my front right')))
PHASE_NAMES = ('beginning', 'early', 'middle', 'late', 'ending')
PHASE_PHRASES = ('near the beginning', 'early in the scene', 'around the middle',
                 'late in the scene', 'near the end')


def compass(azimuth):
    index = int(((float(azimuth) + 22.5) % 360) // 45)
    return COMPASS_NAMES[index], COMPASS_PHRASES[index]


def phase(time, duration):
    fraction = float(time) / float(duration)
    index = sum(fraction >= boundary for boundary in (.15, .35, .65, .85))
    return PHASE_NAMES[index], PHASE_PHRASES[index]


def catalog():
    rows = []
    for split, count in SPLIT_COUNTS.items():
        for opener in range(len(OPENERS[split])):
            for form in range(len(EVENT_FORMS[split])):
                rows.append({'id': f'{split}_{opener * len(EVENT_FORMS[split]) + form:03d}',
                             'split': split, 'opener': OPENERS[split][opener],
                             'event_form': EVENT_FORMS[split][form],
                             'activity_style': form % 3, 'motion_style': form % 2})
        assert sum(r['split'] == split for r in rows) == count
    assert len({(r['opener'], r['event_form']) for r in rows}) == 115
    return rows


RECIPES = {r['id']: r for r in catalog()}


@lru_cache(maxsize=512)
def block_permutation(split, block, seed=42):
    ids = [f'{split}_{i:03d}' for i in range(SPLIT_COUNTS[split])]
    value = hashlib.sha256(f'{seed}/{split}/{block}'.encode()).digest()
    random.Random(int.from_bytes(value, 'big')).shuffle(ids)
    return tuple(ids)


def template_for(split, ordinal, seed=42):
    size = SPLIT_COUNTS[split]
    return block_permutation(split, int(ordinal) // size, seed)[int(ordinal) % size]


def render_pair(plan, template_id):
    recipe = RECIPES[template_id]
    sources = plan['sources']
    if not 1 <= len(sources) <= 4:
        raise ValueError('Expected 1–4 sources')
    duration = float(plan['duration_sec'])
    parts = [recipe['opener'].format(room=ROOMS[plan['room']['type']], count=COUNTS[len(sources) - 1])]
    requirements = {'schema': 'generation_ar_natural_requirements_v2', 'sources': [],
                    'relations': [], 'scene': [{'op': 'room', 'value': plan['room']['type'], 'evidence': parts[0]}]}
    for index, source in enumerate(sources):
        kind = source['kind']
        if kind == 'speech':
            identity = 'speech by ' + ' '.join(source['speaker_description'].split()) + ' saying ' + quoted(source['transcript'])
        elif kind in ('sound', 'music'):
            identity = ('a sound effect described as ' if kind == 'sound' else 'music described as ') + quoted(source['description'])
        else:
            raise ValueError('Unsupported source kind')
        on = float(source['activity']['onset_sec']); off = float(source['activity']['offset_sec'])
        if not 0 <= on < off <= duration + 1e-7:
            raise ValueError('Invalid event activity')
        onset_name, onset = phase(on, duration)
        offset_name, offset = phase(off, duration)
        activity = (
            f'Have it begin {onset} and stop {offset}.',
            f'It should start {onset} and continue until {offset}.',
            f'Let it come in {onset} and finish {offset}.',
        )[recipe['activity_style']]
        constraints = [{'op': 'time_phase', 'field': field, 'value': value, 'evidence': activity}
                       for field, value in [('onset_sec', onset_name), ('offset_sec', offset_name)]]
        trajectory = source['trajectory']
        if trajectory['type'] == 'static':
            where_name, where = compass(trajectory['position']['azimuth_deg'])
            motion = (f'Keep it stationary {where}.', f'It should remain still {where}.')[recipe['motion_style']]
            constraints.append({'op': 'compass', 'point': 'both', 'value': where_name, 'evidence': motion})
        elif trajectory['type'] == 'linear':
            start_name, start = compass(trajectory['start']['azimuth_deg'])
            end_name, end = compass(trajectory['end']['azimuth_deg'])
            # The clauses express endpoints only. They do not invent which
            # path a diametrically opposite pair must take around the listener.
            if start_name == end_name:
                motion = f'It should be moving {start}.'
            else:
                motion = f'Have it move from {MOVING_PHRASES[start_name]} to {MOVING_PHRASES[end_name]}.'
            delta = trajectory['end']['distance_m'] - trajectory['start']['distance_m']
            if abs(delta) > .3:
                radial = 'It should move closer to me as it goes.' if delta < 0 else 'It should move farther away from me as it goes.'
                motion += ' ' + radial
                constraints.append({'op': 'distance_change', 'value': 'approaching' if delta < 0 else 'receding', 'evidence': radial})
            constraints.extend([{'op': 'compass', 'point': point, 'value': value, 'evidence': motion}
                                for point, value in [('start', start_name), ('end', end_name)]])
        else:
            raise ValueError('Unsupported trajectory')
        constraints.append({'op': 'motion', 'value': trajectory['type'], 'evidence': motion})
        if kind == 'speech':
            constraints.append({'op': 'transcript', 'value': ' '.join(source['transcript'].split()), 'evidence': identity})
        parts.append(recipe['event_form'].format(ordinal=ORDINALS[index],
                     identity=identity, activity=activity, motion=motion,
                     activity_lower=activity[0].lower() + activity[1:],
                     motion_lower=motion[0].lower() + motion[1:]))
        # Some forms lowercase a sentence after a comma; use the exact emitted
        # case for evidence while keeping all constraint ownership internal.
        for constraint in constraints:
            if constraint['evidence'] not in parts[-1]:
                evidence = constraint['evidence']
                constraint['evidence'] = evidence[0].lower() + evidence[1:]
        requirements['sources'].append({'key': source['source_id'], 'kind': kind,
            'core': source['speaker_description'] if kind == 'speech' else source['description'],
            'evidence': identity, 'constraints': constraints})
    parts.append('Use only these sources.')
    text = ' '.join(parts)
    if re.search(r'[\u3400-\u9fff\u0400-\u052f\u0600-\u06ff]', text):
        raise ValueError('Non-English script found in a text field')
    return text, requirements


def render(plan, template_id):
    return render_pair(plan, template_id)[0]
