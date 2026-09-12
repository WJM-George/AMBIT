#!/usr/bin/env python3
"""Prepare five-view request pairs and independent request-first pilot seeds.

Only the training split supplies the five-view parents. Request-first seeds
are authored before a teacher sees them, with no target ScenePlan in this file.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
from pathlib import Path
import sqlite3
import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from stable_audio_tools.data.sceneplan_generation_ar_natural_constraints import SCHEMA, evaluate_natural_request

TRAIN_REQUESTS = {
1: [
    'Make a short outdoor recording of a dog barking off to my left. Leave the exact timing up to you.',
    'I want the soft patter of rain behind me for about ten seconds, with nothing else audible.',
    'Let a car drive past from my right to my left. A brief scene is fine.',
    'Play a gentle solo piano melody straight ahead, staying in the same place.',
    'A woman on my right should calmly say "The garden is quiet now." Keep the scene under twelve seconds.',
    'Give me one distant church bell, high above and in front of me.',
    'Can you make a helicopter approach from behind, getting closer without crossing over me?',
    'I just need a few seconds of a kettle whistling somewhere nearby.',
],
2: [
    'Put steady rain in the background and an occasional dog bark on the left. Let the dog come in after the rain starts.',
    'A slow cello melody should stay ahead of me while a bicycle bell crosses from left to right.',
    'Start with footsteps on my right, then add a softly rattling window on my left while the footsteps continue.',
    'A man says "We should leave soon." from directly ahead. After he finishes, let a door slam behind me.',
    'I want ocean waves and a gull calling above them. Keep the gull farther away than the waves.',
    'A quiet electric fan remains on the left and a small bell rings on the right. Make them overlap for a while.',
    'In a dry room, put a plucked guitar close to me and a ticking clock farther back.',
    'Make a passing train move from right to left, with a stationary wind chime behind me throughout its passage.',
],
3: [
    'Keep rain falling behind me. A dog barks intermittently on the left, and a car passes from right to left a little later.',
    'I would like a steady bass line ahead, a soft shaker to my right and a flute on my left. Let the flute begin last.',
    'First a woman ahead says "Please take your seats." Then applause starts on the left while a bell rings on the right.',
    'A small fountain trickles nearby, birds sing farther ahead, and bicycle tires roll past from left to right.',
    'Make a factory sound with a motor on the left, a hiss of steam on the right, and metal clanks behind me. Let their activity overlap.',
    'A cat meows on the right. Before it stops, add footsteps crossing from behind toward the front and a key jingling on the left.',
    'Put a crackling fire straight ahead, rustling leaves behind, and a distant owl above me. Keep each in a fixed position.',
    'I need ten seconds with thunder far ahead, rain on my left and a creaking gate on my right. Bring the gate in after the thunder begins.',
],
4: [
    'Create a garden scene: a fountain ahead, birds above to the left, wind chimes to the right and a dog barking behind me. Keep the dog quiet until the others have begun.',
    'A piano and a violin stay on opposite sides, piano left. Put soft drums ahead and a shaker behind, all playing together.',
    'A man ahead says "The doors are opening." Then let a door slide on the right, footsteps cross from left to right, and a bell ring behind him.',
    'Give me steady rain behind, a dog barking on the left, a bicycle bell on the right and a car passing right to left. Start the car after the bell starts.',
    'Make an outdoor scene with waves ahead, a gull above, a boat motor moving from left to right, and pebbles crunching close behind me.',
    'I want a ticking clock on the left and a fan on the right throughout a twelve-second clip. Halfway through, add a kettle whistle ahead and a door creak behind.',
    'A quiet guitar is close on the left, a cello is farther away on the right, a flute is above the guitar and soft finger snaps are directly behind. Let them overlap.',
    'Begin with wind in the trees behind me and a distant cow ahead. Add running footsteps crossing from right to left, followed by a dog barking on the right while the footsteps continue.',
]}

VALIDATION_REQUESTS = {
1: [
    'Could you give me an owl hooting somewhere above my left shoulder? Choose a sensible length.',
    'I need a lone motorcycle to sweep across the front of the listener, starting on the left and finishing on the right.',
    'A gentle male voice says "Your tea is ready." from behind me. No other sounds, please.',
    'For eight seconds, keep a solo clarinet playing softly in front of me.',
],
2: [
    'There is a dripping tap nearby on the right. A much more distant dog starts barking to the left a little later; let both be heard together.',
    'Have a tram move right to left as a stationary busker plays an accordion behind the listener.',
    'The woman says "Mind the last step." on my left, and only once she is done should a wooden step creak on my right.',
    'Put a buzzing bee above the listener and a rustling paper bag below. Neither should change position.',
],
3: [
    'An air conditioner hums behind me. Somewhere to the right, keys jingle, and then a suitcase rolls from right to left while the hum continues.',
    'Let me hear a quiet harp up front, a violin farther away on the left, and a tambourine on the right that joins them later.',
    'A calm man on the right says "We have reached the station." Keep a distant engine ahead while he talks, and add a short chime behind him after his words.',
    'I want a forest atmosphere with a stream nearby on the left, wind through branches above, and a woodpecker farther away on the right.',
],
4: [
    'In a workshop, keep a fan running behind me and a radio playing music on my left. Add a drill to the right, then footsteps that approach from ahead while the drill is still active.',
    'A woman in front says "The show begins shortly." Meanwhile a cello plays on the left and a piano on the right. End with a gong behind me after she finishes speaking.',
    'I would like flowing water on the right, a croaking frog close on the left, crickets farther behind and a duck calling ahead. The duck should enter after the frog, with some overlap.',
    'Make a nine-second clip of a skateboard passing left to right, a dog behind me, a fountain ahead and a bicycle bell on the left. Keep the fountain audible for the whole clip.',
]}


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def dump(path, value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    if path.exists() and path.read_text() != text:
        raise ValueError('refusing to overwrite a different frozen pilot artifact: ' + str(path))
    path.write_text(text)


def position(p):
    return f"azimuth {p['azimuth_deg']:g} degrees, elevation {p['elevation_deg']:g} degrees and distance {p['distance_m']:.6g} meters"


def seed_caption_risks(plan):
    """Conservative screening for this tiny seed panel, not a corpus classifier."""
    flags = []
    for source in plan['sources']:
        text = source.get('description', '').lower()
        if source['trajectory']['type'] != 'static' and re.search(r'\b(at rest|stationary|stands still|stays still)\b', text):
            flags.append('caption requires a stationary source but target trajectory moves')
        if re.search(r'\b(past|passing|approach\w*|reced\w*|walk\w*|running|moves along|moving along|into the distance|left to right|right to left)\b', text):
            flags.append('caption carries spatial movement requiring manual reconciliation')
        if re.search(r'\b(nearby|far away|distant|close to|behind|above the listener|below the listener|on the left|on the right)\b', text):
            flags.append('caption supplies spatial facts needing reconciliation with coordinates')
        if source['kind'] == 'sound' and re.search(r'\b(followed by|while|accompanied|backdrop|against|as it|people|in the background|over a|over the|layered with)\b', text):
            flags.append('potential composite event or background source')
    return flags


def clause(source, variant):
    if source['kind'] == 'speech':
        event = f"a voice described as \"{source['speaker_description']}\" saying \"{source['transcript']}\""
    else:
        event = f"{source['kind']} described as \"{source['description']}\""
    a = source['activity']; onset = f"{a['onset_sec']:.6g}"; offset = f"{a['offset_sec']:.6g}"
    t = source['trajectory']
    motion = (f"remain stationary at {position(t['position'])}" if t['type'] == 'static' else
              f"move smoothly on a linear trajectory from {position(t['start'])} to {position(t['end'])}")
    if variant == 0:
        return f"Include {event} from {onset} to {offset} seconds; it should {motion}."
    if variant == 1:
        return f"Between {onset} and {offset} seconds, I want {event}. Have it {motion}."
    if variant == 2:
        return f"For {event}, use {onset} seconds as the start and {offset} seconds as the finish, and let it {motion}."
    if variant == 3:
        return f"Let {event} {motion}, becoming audible at {onset} seconds and stopping at {offset} seconds."
    return f"Could {event} start at {onset} seconds, stop at {offset} seconds, and {motion}?"


def five_view(parent, variant):
    plan = json.loads(json.dumps(parent)); pid = parent['sample_id'] + f'/five_view/{variant}'; plan['sample_id'] = pid
    room = {'dry': 'a dry space without reverberation', 'moderate': 'a moderately reverberant room',
            'reverberant': 'a reverberant room', 'outdoor': 'an outdoor environment'}[plan['room']['type']]
    header = [f"Please make {plan['duration_sec']:.6g} seconds of spatial audio in {room}.",
              f"I am planning a {plan['duration_sec']:.6g}-second sound scene in {room}.",
              f"Here is the mix I need, lasting {plan['duration_sec']:.6g} seconds in {room}.",
              f"Set the scene in {room} and make its total length {plan['duration_sec']:.6g} seconds.",
              f"Can you create a {plan['duration_sec']:.6g}-second spatial recording in {room}?"][variant]
    clauses = [clause(s, variant) for s in plan['sources']]
    indices = list(range(len(clauses)))
    if variant in (2, 4):
        indices.reverse()
    request = ' '.join([header] + [clauses[i] for i in indices] + ['Use these sounds only.'])
    req = {'schema': SCHEMA, 'output_order': None, 'sources': [], 'relations': [],
           'scene': [{'op': 'numeric', 'field': 'duration_sec', 'value': plan['duration_sec'], 'evidence': header, 'origin': 'explicit'},
                     {'op': 'room', 'value': plan['room']['type'], 'evidence': header, 'origin': 'explicit'}]}
    bindings = {}
    for i, s in enumerate(plan['sources']):
        key = f'event_{i}'; bindings[key] = s['source_id']; evidence = clauses[i]
        core = s['speaker_description'] if s['kind'] == 'speech' else s['description']
        constraints = [{'op': 'numeric', 'field': field, 'value': value, 'evidence': evidence, 'origin': 'explicit'}
                       for field, value in s['activity'].items()]
        constraints.append({'op': 'motion', 'value': s['trajectory']['type'], 'evidence': evidence, 'origin': 'explicit'})
        for endpoint in ('start', 'end'):
            p = s['trajectory']['position'] if s['trajectory']['type'] == 'static' else s['trajectory'][endpoint]
            constraints.extend({'op': 'numeric', 'field': endpoint + '.' + field, 'value': value,
                                'evidence': evidence, 'origin': 'explicit'} for field, value in p.items())
        if s['kind'] == 'speech':
            constraints.append({'op': 'transcript', 'value': s['transcript'], 'evidence': evidence, 'origin': 'explicit'})
        req['sources'].append({'key': key, 'kind': s['kind'], 'core': core, 'evidence': evidence, 'constraints': constraints})
    labels = {(r['key'], s['source_id']): bindings[r['key']] == s['source_id'] for r in req['sources'] for s in plan['sources']}
    checks = evaluate_natural_request(request, req, plan, labels, completion_reasonable=True)
    assert checks['request_constraints_joint']
    return {'id': pid, 'schema': 'generation_ar_natural_pair_v2', 'route': 'plan_to_request', 'split': 'train',
            'family_id': 'old_train/' + parent['sample_id'], 'view': variant, 'request': request,
            'requirements': req, 'target_sceneplan': plan, 'source_bindings': bindings,
            'provenance': {'request_generator': 'five auditable precise-English renderers, not free LLM paraphrases',
                           'parent': parent['sample_id'], 'parent_split': 'train', 'script_sha256': sha(Path(__file__).read_text()),
                           'unspecified': ['source gain (canonical 0 dB)'], 'semantic_review': 'PENDING'},
            'deterministic_quality': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--train-db', type=Path, default=Path('/dev/shm/generation_ar_manifests_20260905/train.sqlite'))
    args = parser.parse_args(); args.root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(f'file:{args.train_db}?mode=ro&immutable=1', uri=True)
    # The original manifest starts with speech-heavy blocks. Draw candidates
    # over the entire training ordinal range, then balance speech presence.
    low, high = db.execute('SELECT MIN(ordinal),MAX(ordinal) FROM rows').fetchone()
    indices = random.Random(42).sample(range(low, high + 1), 4096)
    plans = []
    for offset in range(0, len(indices), 256):
        chunk = indices[offset:offset+256]
        rows = db.execute('SELECT target_sceneplan_zlib FROM rows WHERE ordinal IN (' + ','.join('?' for _ in chunk) + ')', chunk).fetchall()
        plans.extend(json.loads(zlib.decompress(b)) for (b,) in rows)
    plans = [p for p in plans if not re_non_english_script(json.dumps(p))]
    parents = []
    for count in range(1, 5):
        chosen = []
        for speech in (False, True):
            candidates = [p for p in plans if len(p['sources']) == count and any(s['kind']=='speech' for s in p['sources']) == speech and not seed_caption_risks(p)]
            # Prefer concise seed content for the first quality gate; audit
            # longer/composite captions separately before full reconstruction.
            candidates.sort(key=lambda p: (sum(len(s.get('description', s.get('speaker_description', '')).split()) for s in p['sources']), sha('natural-pilot-42/' + p['sample_id'])))
            assert candidates; chosen.append(candidates[0])
        assert len(chosen) == 2; parents.extend(chosen)
    db.close()
    pairs = [five_view(p, view) for p in parents for view in range(5)]
    dump(args.root / 'plan_to_request_pairs.json', {'pairs': pairs, 'test_used': False})
    seeds = []
    for split, groups in [('train', TRAIN_REQUESTS), ('validation', VALIDATION_REQUESTS)]:
        for count, texts in groups.items():
            for i, request in enumerate(texts):
                sid = f'natural_request_first_v2/{split}/{count}/{i}'
                seeds.append({'id': sid, 'family_id': sid, 'split': split, 'route': 'request_to_plan',
                              'request': request, 'expected_count_for_quality_only': count,
                              'provenance': 'English request authored by Codex before any target plan; count is internal stratification, never teacher/AR input'})
    assert len({s['request'] for s in seeds}) == len(seeds)
    dump(args.root / 'request_first_seeds.json', {'seeds': seeds, 'test_used': False})
    dump(args.root / 'PREPARATION.json', {'status': 'PREPARED_TEACHER_AND_SEMANTIC_QA_PENDING',
          'plan_first_pairs': len(pairs), 'plan_first_parents': len(parents), 'views_per_parent': 5,
          'request_first_train_seeds': 32, 'independent_natural_validation_seeds': 16,
          'train_pair_target': 72, 'total_pair_target': 88, 'test_used': False,
          'pilot_size_amendment': 'User requested five views per existing GT after v2 draft; 8 train parents x5 plus 32 request-first train seeds, 16 separate validation seeds. No natural-data training has run.',
          'parent_sampling': '4096 seed42 uniform ordinals across the entire training manifest; one speech-containing and one nonspeech parent per count; concise content for startup quality gate',
          'caption_screen': 'Conservative movement/composite-event regex for tiny seed selection only; manual source-count and semantic audit still required',
          'limits': 'Five renderers are an initial precise-language augmentation; neither 1000-pattern coverage nor natural AR acceptance is established.'})
    print(json.dumps({'root': str(args.root), 'plan_first_pairs': len(pairs), 'request_first_seeds': len(seeds)}))


def re_non_english_script(text):
    # Screening only; Latin-script text can still be non-English.
    return any('\u3400' <= c <= '\u9fff' or '\u0400' <= c <= '\u052f' or '\u0600' <= c <= '\u06ff' for c in text)


if __name__ == '__main__':
    main()
