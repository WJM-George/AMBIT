"""Training labels for actual FOA scenes; edit requests are never consumed.

This opt-in reader keeps native latent loading, contrastive identities, and
counterfactual selection. Factual text is one independent experiment variable;
event targets are supplied separately for a future supervised set head.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import zlib

import torch

from stable_audio_tools.data.model_sceneplan import (
    KIND_IDS, MAX_SOURCES, _trajectory_keyframes, validate_model_sceneplan,
)
from stable_audio_tools.data.sceneplan_transfusion_editing import sha256_json
from stable_audio_tools.data.sceneplan_transfusion_editing_clap44 import (
    EditingCLAP44Dataset, _content, _content_text, _scene_key, binding_counterfactuals,
    collate_clap44,
)

CONTRACT = 'clap44_actual_scene_facts_and_event_targets_v1'
MAX_KEYFRAMES = 8
MOTION_IDS = {'static': 0, 'linear': 1, 'keyframed': 2}
# These are factual-scene templates, independent of the T200 edit requests.
RECIPES = {
    'train/facts/0': ('There are {count} audible sources in a {room} acoustic environment. ',
        '{identity} is active from {onset} to {offset} seconds, {motion}, with a source gain of {gain} decibels.'),
    'train/facts/1': ('In this {room} acoustic environment, {count} sources can be heard. ',
        'From {onset} until {offset} seconds, the scene contains {identity}, {motion}. Its source gain is {gain} decibels.'),
    'train/facts/2': ('The audible scene has {count} sources and a {room} acoustic environment. ',
        'The source described as {identity} sounds during the interval from {onset} to {offset} seconds. It is {motion}; its source gain is {gain} decibels.'),
    'train/facts/3': ('This recording contains {count} audible sources with {room} acoustics. ',
        'One source is {identity}. It is audible between {onset} and {offset} seconds, {motion}, at a source gain of {gain} decibels.'),
    'validation/facts/0': ('A {room} acoustic environment surrounds {count} audible sources. ',
        '{identity} begins at {onset} seconds and ends at {offset} seconds. It is {motion}, and its source gain is {gain} decibels.'),
    'validation/facts/1': ('The recording presents {count} audible sources in {room} acoustics. ',
        'An audible source is {identity}, with activity starting at {onset} seconds and ending at {offset} seconds. It is {motion}, with a source gain of {gain} decibels.'),
}


def _number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Non-finite scene fact')
    return repr(number)


def _position(value):
    return (f"azimuth {_number(value['azimuth_deg'])} degrees, elevation "
            f"{_number(value['elevation_deg'])} degrees and distance {_number(value['distance_m'])} meters")


def _event(source):
    event = copy.deepcopy(dict(source))
    event.pop('source_id')
    trajectory = event['trajectory']
    expected = {'static': {'type', 'position'}, 'linear': {'type', 'start', 'end'},
                'keyframed': {'type', 'keyframes'}}[trajectory['type']]
    if set(trajectory) != expected:
        raise ValueError('Unexpected trajectory fields would be lost from factual text')
    return event


def ordered_events(plan):
    """Persistent IDs are provenance only; sort complete content/time/space records."""
    validate_model_sceneplan(plan)
    return sorted(plan['sources'], key=lambda s: json.dumps(_event(s), sort_keys=True, ensure_ascii=False))


def choose_recipe(split, source_scene_key, seed=42):
    if split not in ('train', 'validation') or not source_scene_key:
        raise ValueError('Scene supervision only supports TRAIN and development validation')
    choices = sorted(k for k in RECIPES if k.startswith(split + '/'))
    digest = hashlib.sha256(f'{CONTRACT}:{seed}:{source_scene_key}'.encode()).digest()
    return choices[int.from_bytes(digest[:8], 'big') % len(choices)]


def factual_description(plan, recipe):
    """Complete factual clauses with exact numeric literals and event span labels."""
    if recipe not in RECIPES:
        raise ValueError('Unknown factual recipe')
    header, body = RECIPES[recipe]
    sources = ordered_events(plan)
    text = header.format(count=len(sources), room=plan['room']['type'])
    spans = []
    for index, source in enumerate(sources):
        trajectory = source['trajectory']
        if trajectory['type'] == 'static':
            motion = 'remaining stationary at ' + _position(trajectory['position'])
        elif trajectory['type'] == 'linear':
            motion = 'moving in a straight line from ' + _position(trajectory['start']) + ' to ' + _position(trajectory['end'])
        else:
            motion = 'following position keyframes: ' + '; '.join(
                f"at {_number(k['time_sec'])} seconds, {_position(k['position'])}"
                for k in trajectory['keyframes'])
        identity = _content_text(source)
        clause = body.format(identity=identity, onset=_number(source['activity']['onset_sec']),
            offset=_number(source['activity']['offset_sec']), motion=motion, gain=_number(source['gain_db']))
        if index:
            text += ' '
        start = len(text)
        text += clause
        identity_start = start + clause.index(identity)
        spans.append({'source_id': source['source_id'], 'event_key': sha256_json(_event(source)),
            'start': start, 'end': len(text), 'content_start': identity_start,
            'content_end': identity_start + len(identity)})
    return {'text': text, 'recipe': recipe, 'event_spans': spans, 'scene_key': _scene_key(plan)}


def structured_targets(plan):
    """One set element binds content, activity, gain, and every trajectory keyframe.

    No source order or source ID is a model input. Temporal masks label facts;
    they do not assert that overlapping mixture intervals isolate a source.
    Float64 preserves numeric labels before any explicit training normalization.
    """
    sources = ordered_events(plan)
    shape = (MAX_SOURCES, MAX_KEYFRAMES)
    targets = {
        'source_count': torch.tensor(len(sources), dtype=torch.long),
        'presence': torch.zeros(MAX_SOURCES, dtype=torch.bool),
        'kind': torch.zeros(MAX_SOURCES, dtype=torch.long),
        'motion': torch.full((MAX_SOURCES,), -100, dtype=torch.long),
        'activity_sec': torch.zeros(MAX_SOURCES, 2, dtype=torch.float64),
        'gain_db': torch.zeros(MAX_SOURCES, dtype=torch.float64),
        'keyframe_mask': torch.zeros(shape, dtype=torch.bool),
        'keyframe_time_sec': torch.zeros(shape, dtype=torch.float64),
        'position_spherical': torch.zeros(*shape, 3, dtype=torch.float64),
        'direction_xyz': torch.zeros(*shape, 3, dtype=torch.float64),
    }
    for i, source in enumerate(sources):
        targets['presence'][i] = True
        targets['kind'][i] = KIND_IDS[source['kind']]
        targets['motion'][i] = MOTION_IDS[source['trajectory']['type']]
        targets['activity_sec'][i] = torch.tensor([source['activity']['onset_sec'], source['activity']['offset_sec']], dtype=torch.float64)
        targets['gain_db'][i] = source['gain_db']
        for j, frame in enumerate(_trajectory_keyframes(source)):
            p = frame['position']; az, el = math.radians(p['azimuth_deg']), math.radians(p['elevation_deg'])
            targets['keyframe_mask'][i, j] = True
            targets['keyframe_time_sec'][i, j] = frame['time_sec']
            targets['position_spherical'][i, j] = torch.tensor([p['azimuth_deg'], p['elevation_deg'], p['distance_m']], dtype=torch.float64)
            targets['direction_xyz'][i, j] = torch.tensor([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)], dtype=torch.float64)
    return {'tensors': targets, 'source_ids': [s['source_id'] for s in sources],
        'event_keys': [sha256_json(_event(s)) for s in sources],
        'content_keys': [sha256_json(_content(s)) for s in sources],
        'content_texts': [_content_text(s) for s in sources],
        'scene_key': _scene_key(plan), 'duration_sec': plan['duration_sec']}


def factual_counterfactuals(plan, recipe):
    """Keep native negative membership/order; replace only their factual wording."""
    native = binding_counterfactuals(plan)
    wanted = {x['scene_key'] for x in native}
    captions = {}
    for i in range(len(plan['sources'])):
        for j in range(i + 1, len(plan['sources'])):
            for field in ('trajectory', 'activity'):
                changed = copy.deepcopy(plan)
                changed['sources'][i][field], changed['sources'][j][field] = (
                    copy.deepcopy(plan['sources'][j][field]), copy.deepcopy(plan['sources'][i][field]))
                try:
                    validate_model_sceneplan(changed)
                except ValueError:
                    continue
                key = _scene_key(changed)
                if key in wanted:
                    captions[key] = factual_description(changed, recipe)['text']
    if wanted != captions.keys():
        raise RuntimeError('Could not reconstruct exactly the native binding negatives')
    return [{**x, 'scene_text': captions[x['scene_key']]} for x in native]


class CLAP44SceneSupervisionDataset(EditingCLAP44Dataset):
    """An opt-in native reader with separate facts/targets, no new audio copies."""

    def __init__(self, *args, text_view='native', caption_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        if text_view not in ('native', 'factual'):
            raise ValueError('Unknown scene text view')
        self.text_view = text_view
        self.caption_seed = int(caption_seed)

    def __getitem__(self, index):
        views = super().__getitem__(index)
        ordinal = index if self.ordinals is None else self.ordinals[index]
        row = self._connection.execute(
            'SELECT old_sceneplan_zlib,new_sceneplan_zlib,old_sceneplan_sha256,new_sceneplan_sha256 FROM pairs WHERE pair_ordinal=?',
            (ordinal,)).fetchone()
        # Every paired role and its negatives use the same recipe. It depends
        # on the source scene, never the edit instruction or target change.
        recipe = choose_recipe(self.split, views[0]['label']['scene_key'], self.caption_seed)
        for view, role in zip(views, ('old', 'new')):
            plan = json.loads(zlib.decompress(row[f'{role}_sceneplan_zlib']))
            if sha256_json(plan) != row[f'{role}_sceneplan_sha256']:
                raise RuntimeError('Event-label plan provenance changed')
            view['event_targets'] = structured_targets(plan)
            view['factual_description'] = factual_description(plan, recipe)
            if self.text_view == 'factual':
                view['label'] = {**view['label'], 'scene_text': view['factual_description']['text']}
                view['counterfactuals'] = factual_counterfactuals(plan, recipe)
        return views


def collate_scene_supervision(pairs):
    batch = collate_clap44(pairs)
    views = [view for pair in pairs for view in pair]
    targets = [view['event_targets'] for view in views]
    batch['event_targets'] = {key: torch.stack([target['tensors'][key] for target in targets])
                             for key in targets[0]['tensors']}
    batch['event_metadata'] = [{key: value for key, value in target.items() if key != 'tensors'} for target in targets]
    batch['factual_descriptions'] = [view['factual_description'] for view in views]
    return batch
