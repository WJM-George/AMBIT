"""Multi-template descriptions of actual scenes, separate from AR requests.

Templates are selected by the source scene identity. Both paired audio roles
and their anchor-local negatives share a format; no requested edit is read.
"""
import copy
import hashlib
import json
import zlib

from stable_audio_tools.data.model_sceneplan import validate_model_sceneplan
from stable_audio_tools.data.sceneplan_transfusion_editing import sha256_json
from stable_audio_tools.data.sceneplan_transfusion_editing_clap44 import (
    EditingCLAP44Dataset, _content, _content_text, _scene_key,
)
from scripts.t2a.experiments.clap_scene_supervision_v1.data import (
    _number, _position, _event, ordered_events, structured_targets,
    collate_scene_supervision,
)

SCHEMA = 'clap44_factual_scene_templates200_v1'
TRAIN_HEADERS = (
    'This {duration}-second recording has {count} audible sources in {room} acoustics. ',
    'Across {duration} seconds, {count} sources are audible in a {room} environment. ',
    'The scene lasts {duration} seconds and contains {count} sources with {room} acoustics. ',
    'There are {count} audible sources in this {duration}-second {room} acoustic scene. ',
    'In a {room} acoustic environment, this {duration}-second scene includes {count} sources. ',
    'A recording of {duration} seconds presents {count} sources with {room} acoustics. ',
    'The audible scene contains {count} sources; its duration is {duration} seconds and its acoustics are {room}. ',
    'During this {duration}-second recording, {count} sources sound in a {room} environment. ',
    'This scene has {room} acoustics, a duration of {duration} seconds, and {count} audible sources. ',
    'Within a {duration}-second scene with {room} acoustics, {count} sources can be heard. ',
    'The recording captures {count} sources over {duration} seconds, with {room} acoustics. ',
    'A {room} acoustic setting contains {count} sources in a recording lasting {duration} seconds. ',
    'For a scene lasting {duration} seconds, the audible source count is {count} and the acoustics are {room}. ',
    'Recorded over {duration} seconds, the scene has {count} sources and {room} acoustics. ',
    'The {count} audible sources belong to a {duration}-second scene in a {room} environment. ',
    'A total of {count} sources is audible in this {duration}-second recording with {room} acoustics. ',
    'The scene described here lasts {duration} seconds, has {room} acoustics, and contains {count} sources. ',
    'In this recording, {count} sources occupy a {duration}-second scene with {room} acoustics. ',
    'With {room} acoustics, the recording contains {count} audible sources across {duration} seconds. ',
    'The following {count} sources are audible in a {duration}-second scene with {room} acoustics. ',
)
TRAIN_BODIES = (
    '{identity} is active from {onset} to {offset} seconds, {motion}, with a source gain of {gain} decibels.',
    'From {onset} until {offset} seconds, {identity} is audible, {motion}. Its source gain is {gain} decibels.',
    'One source is {identity}. It sounds between {onset} and {offset} seconds, {motion}, at a source gain of {gain} decibels.',
    'The source described as {identity} has an active interval of {onset} to {offset} seconds. It is {motion}; its source gain is {gain} decibels.',
    'At a source gain of {gain} decibels, {identity} sounds from {onset} to {offset} seconds while {motion}.',
    '{identity} begins at {onset} seconds and ends at {offset} seconds. The source is {motion}, and its gain is {gain} decibels.',
    'An audible event consists of {identity}, active from {onset} to {offset} seconds. Its source is {motion} with a gain of {gain} decibels.',
    'The activity interval for {identity} starts at {onset} seconds and finishes at {offset} seconds. It is {motion}, at a source gain of {gain} decibels.',
    '{identity} is heard at a source gain of {gain} decibels during {onset} to {offset} seconds, {motion}.',
    'A source sounding during {onset} to {offset} seconds is {identity}. With a source gain of {gain} decibels, it is {motion}.',
)
VALIDATION_HEADERS = (
    'This sound scene spans {duration} seconds: {count} sources are present, and the acoustics are {room}. ',
    'Over a recorded duration of {duration} seconds, a {room} acoustic scene has {count} audible sources. ',
    'The recording has a duration of {duration} seconds; it depicts {count} sources with {room} acoustics. ',
    'The {duration}-second audio depicts a {room} acoustic environment containing {count} sources. ',
)
VALIDATION_BODIES = (
    '{identity} has onset {onset} seconds and offset {offset} seconds. Its source gain is {gain} decibels, and it is {motion}.',
    'Between times {onset} and {offset} seconds, {identity} is present. The source is {motion} at a gain of {gain} decibels.',
    'The event {identity} occupies the interval {onset} to {offset} seconds, with its source {motion} and its source gain set to {gain} decibels.',
    'A source is audible as {identity} from time {onset} seconds to time {offset} seconds. At a gain of {gain} decibels, it is {motion}.',
    'At a gain of {gain} decibels, the source {identity} is audible over {onset} to {offset} seconds. It is {motion}.',
)
RECIPES = {f'{split}/scene/{i:03d}': (header, body)
           for split, headers, bodies in (
               ('train', TRAIN_HEADERS, TRAIN_BODIES),
               ('validation', VALIDATION_HEADERS, VALIDATION_BODIES))
           for i, (header, body) in enumerate((h, b) for h in headers for b in bodies)}
CATALOG_SHA256 = sha256_json({'schema': SCHEMA, 'recipes': RECIPES})


def choose_recipe(split, source_scene_key, seed=42):
    if split not in ('train', 'validation') or not source_scene_key:
        raise ValueError('Only full TRAIN and development validation are supported')
    n = 200 if split == 'train' else 20
    digest = hashlib.sha256(f'{SCHEMA}:{seed}:{source_scene_key}'.encode()).digest()
    return f'{split}/scene/{int.from_bytes(digest[:8], "big") % n:03d}'


def describe(plan, recipe):
    header, body = RECIPES[recipe]
    sources = ordered_events(plan)
    text = header.format(count=len(sources), duration=_number(plan['duration_sec']), room=plan['room']['type'])
    spans = []
    for source in sources:
        trajectory = source['trajectory']
        if trajectory['type'] == 'static':
            motion = 'remaining stationary at ' + _position(trajectory['position'])
        elif trajectory['type'] == 'linear':
            motion = 'moving in a straight line from ' + _position(trajectory['start']) + ' to ' + _position(trajectory['end'])
        else:
            motion = 'following position keyframes: ' + '; '.join(
                f"at {_number(k['time_sec'])} seconds, {_position(k['position'])}" for k in trajectory['keyframes'])
        identity = _content_text(source)
        clause = body.format(identity=identity, onset=_number(source['activity']['onset_sec']),
            offset=_number(source['activity']['offset_sec']), motion=motion, gain=_number(source['gain_db']))
        if spans:
            text += ' '
        start = len(text); text += clause
        content_start = start + clause.index(identity)
        spans.append({'source_id': source['source_id'], 'event_key': sha256_json(_event(source)),
            'start': start, 'end': len(text), 'content_start': content_start,
            'content_end': content_start + len(identity)})
    return {'text': text, 'recipe': recipe, 'event_spans': spans, 'scene_key': _scene_key(plan)}


def counterfactuals(plan, recipe, maximum=4):
    """Use the native negative membership/order with exactly the positive format."""
    validate_model_sceneplan(plan)
    seen = {_scene_key(plan)}; result = []
    sources = plan['sources']
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            if _content(sources[i]) == _content(sources[j]):
                continue
            for field in ('trajectory', 'activity'):
                if sources[i][field] == sources[j][field]:
                    continue
                if field == 'activity' and abs(sources[i][field]['onset_sec'] - sources[j][field]['onset_sec']) < .25:
                    continue
                changed = copy.deepcopy(plan)
                changed['sources'][i][field], changed['sources'][j][field] = copy.deepcopy(sources[j][field]), copy.deepcopy(sources[i][field])
                try:
                    validate_model_sceneplan(changed)
                except ValueError:
                    continue
                key = _scene_key(changed)
                if key in seen:
                    continue
                seen.add(key)
                result.append({'kind': 'swapped_' + field, 'scene_key': key,
                    'scene_text': describe(changed, recipe)['text']})
                if len(result) == maximum:
                    return result
    return result


class FactualDataset(EditingCLAP44Dataset):
    def __init__(self, *args, caption_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.caption_seed = int(caption_seed)

    def __getitem__(self, index):
        views = super().__getitem__(index)
        ordinal = index if self.ordinals is None else self.ordinals[index]
        row = self._connection.execute(
            'SELECT old_sceneplan_zlib,new_sceneplan_zlib,old_sceneplan_sha256,new_sceneplan_sha256 '
            'FROM pairs WHERE pair_ordinal=?', (ordinal,)).fetchone()
        recipe = choose_recipe(self.split, views[0]['label']['scene_key'], self.caption_seed)
        for view, role in zip(views, ('old', 'new')):
            plan = json.loads(zlib.decompress(row[f'{role}_sceneplan_zlib']))
            if sha256_json(plan) != row[f'{role}_sceneplan_sha256']:
                raise RuntimeError('Factual label provenance changed')
            facts = describe(plan, recipe)
            view['event_targets'] = structured_targets(plan)
            view['factual_description'] = facts
            view['label'] = {**view['label'], 'scene_text': facts['text']}
            negatives = counterfactuals(plan, recipe)
            if [(x['kind'], x['scene_key']) for x in negatives] != [(x['kind'], x['scene_key']) for x in view['counterfactuals']]:
                raise RuntimeError('Factual wording changed the native negative pool')
            view['counterfactuals'] = negatives
        return views
