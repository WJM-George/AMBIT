import copy
import json

import numpy as np
import pytest
import soundfile as sf

from stable_audio_tools.training.transfusion_opsd.event_native_reference import (
    native_reference_spec, native_reference_identity, load_native_audio_reference,
)
from stable_audio_tools.training.transfusion_opsd.provenance import sha256_file


def reference(tmp_path):
    release = tmp_path/'release.json'
    release.write_text('{}')
    audio = tmp_path/'original.wav'
    sf.write(audio, np.zeros((128, 4), np.float32), 44100, subtype='FLOAT')
    rows = [{'sample_id': 'speech', 'request': 'Say hello on the left.',
             'requirements': {'transcript': 'hello'}}]
    protocol = {'release_bundle': str(release), 'initial_model_fingerprint': 'original',
        'reference_dit_fingerprint': 'D0', 'audio_protection_receipt': {'model': 'fixed'},
        'collection': {'feedback_query': 50, 'min_coherence': .1}, 'guard_seeds': {'speech': [12, 13]}}
    plan = {'duration_sec': 4.13}
    query = {'seed': 12, 'plan': plan, 'original_plan': copy.deepcopy(plan), 'actions': [0],
        'reference_frames': 16, 'model_num_samples': 128,
        'audio': {'path': str(audio), 'sha256': sha256_file(audio)},
        'score': {'utility': .8, 'costs': {'asr_wer': 0.}}}
    document = {'contract': 'event_original_native_audio_reference_v1',
        'identity': native_reference_identity(protocol), 'optimizer_steps': 0, 'model_unchanged': True,
        'records': [{**rows[0], 'queries': [query, {**copy.deepcopy(query), 'seed': 13}]}]}
    return protocol, rows, document


def persist(tmp_path, document):
    path = tmp_path/'reference.json'
    path.write_text(json.dumps(document))
    return {'path': str(path), 'sha256': sha256_file(path)}


def test_original_native_reference_preserves_free_plan_and_gaussian_geometry(tmp_path):
    protocol, rows, document = reference(tmp_path)
    # Different valid native free completions may own different reference
    # geometry. The loader must not silently select or re-decode one of them.
    second = document['records'][0]['queries'][1]
    second.update(plan={'duration_sec': 5.41}, original_plan={'duration_sec': 5.41}, reference_frames=21)
    pointer = persist(tmp_path, document)
    assert load_native_audio_reference(pointer, protocol=protocol, rows=rows) == document


@pytest.mark.parametrize('change', ['updated', 'different_original', 'refinement', 'transcript',
                                   'noise_missing', 'noise_order', 'audio_tampered', 'geometry'])
def test_invalid_reference_cannot_reset_or_weaken_the_original_guard(tmp_path, change):
    protocol, rows, document = reference(tmp_path)
    query = document['records'][0]['queries'][0]
    if change == 'updated': document['optimizer_steps'] = 1
    if change == 'different_original': document['identity']['original_policy_fingerprint'] = 'candidate'
    if change == 'refinement': query['actions'] = [3]
    if change == 'transcript': document['records'][0]['requirements'] = {'transcript': 'goodbye'}
    if change == 'noise_missing': document['records'][0]['queries'].pop()
    if change == 'noise_order': document['records'][0]['queries'].reverse()
    if change == 'geometry': query['model_num_samples'] += 1
    pointer = persist(tmp_path, document)
    if change == 'audio_tampered':
        with open(query['audio']['path'], 'ab') as stream:
            stream.write(b'changed')
    with pytest.raises(ValueError):
        load_native_audio_reference(pointer, protocol=protocol, rows=rows)


def test_resume_must_keep_checkpoint_reference_and_pin_its_manifest():
    pointer = {'path': '/original/reference.json', 'sha256': 'original-audio'}
    protocol = {'native_audio_reference': {'mode': 'capture_original'},
        'parent_candidates': {'query_feedback': {'path': '/candidate.pt'}}, 'input_files': [pointer]}
    collection = {'original_native_reference': pointer}
    with pytest.raises(ValueError, match='cannot recapture'):
        native_reference_spec(protocol, 'query_feedback', parent_collection=collection)
    protocol['native_audio_reference'] = {'mode': 'reuse_original', 'artifact': pointer}
    assert native_reference_spec(protocol, 'query_feedback', parent_collection=collection)['artifact'] == pointer
    with pytest.raises(ValueError, match='checkpoint original'):
        native_reference_spec(protocol, 'query_feedback', parent_collection={})
    protocol['input_files'] = []
    with pytest.raises(ValueError, match='pin the reused'):
        native_reference_spec(protocol, 'query_feedback', parent_collection=collection)


def test_later_student_query_can_keep_the_original_reference_identity(tmp_path):
    protocol, rows, document = reference(tmp_path)
    pointer = persist(tmp_path, document)
    protocol['collection']['feedback_query'] = 75
    with pytest.raises(ValueError, match='unchanged original'):
        load_native_audio_reference(pointer, protocol=protocol, rows=rows)
    protocol['native_audio_reference'] = {'mode': 'reuse_original', 'artifact': pointer,
                                          'reference_query_index': 50}
    protocol['input_files'] = [pointer]
    native_reference_spec(protocol, 'query_feedback')
    assert load_native_audio_reference(pointer, protocol=protocol, rows=rows) == document
    assert native_reference_identity(protocol) == document['identity']
    assert sha256_file(pointer['path']) == pointer['sha256']


def test_capture_cannot_claim_another_query_index(tmp_path):
    protocol, _, _ = reference(tmp_path)
    protocol['native_audio_reference'] = {'mode': 'capture_original', 'reference_query_index': 75}
    with pytest.raises(ValueError, match='actual current query'):
        native_reference_identity(protocol)
