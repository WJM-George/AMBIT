"""Immutable native audio references, carried forward across joint updates."""
import json
import math
from pathlib import Path

from .provenance import sha256_file
from .event_query_panel import validate_collection_seeds


def native_reference_spec(protocol, arm, *, parent_collection=None):
    """A resumed candidate must keep the original reference it was trained with."""
    spec = protocol.get('native_audio_reference')
    if spec is None:
        return None
    if spec.get('mode') not in ('capture_original', 'reuse_original'):
        raise ValueError('declare capture_original or reuse_original audio references')
    parent = protocol.get('parent_candidates', {}).get(arm)
    if spec['mode'] == 'capture_original':
        if parent:
            raise ValueError('a resumed policy cannot recapture its original reference')
    else:
        pointer = spec['artifact']
        if not any(Path(entry['path']).resolve() == Path(pointer['path']).resolve()
                and entry['sha256'] == pointer['sha256'] for entry in protocol['input_files']):
            raise ValueError('pin the reused audio reference in protocol inputs')
        if parent and (parent_collection or {}).get('original_native_reference') != pointer:
            raise ValueError('resumed training must carry its checkpoint original reference')
    return spec


def native_reference_identity(protocol):
    current_query = protocol['collection']['feedback_query']
    spec = protocol.get('native_audio_reference', {})
    reference_query = spec.get('reference_query_index', current_query)
    if type(reference_query) is not int or not 0 <= reference_query < 100:
        raise ValueError('declare the recorded original query index within the native schedule')
    if reference_query != current_query and spec.get('mode') != 'reuse_original':
        raise ValueError('a new capture must record its actual current query index')
    return {
        'release_bundle_sha256': sha256_file(protocol['release_bundle']),
        'original_policy_fingerprint': protocol['initial_model_fingerprint'],
        'original_dit_fingerprint': protocol['reference_dit_fingerprint'],
        'audio_protection_receipt': protocol['audio_protection_receipt'],
        'min_coherence': protocol['collection']['min_coherence'],
        # A timing experiment can retain the same immutable action-zero audio
        # captured at an earlier query. Its identity must describe that recorded
        # run, not relabel it as a new capture at the student's current query.
        'query_index': reference_query,
        'sample_rate': 44100, 'steps': 100,
        'noise_geometry': 'paired_gaussian_noise_reference_frames_v1',
    }


def load_native_audio_reference(pointer, *, protocol, rows):
    """Verify stored original outputs; never rescore or replace their baseline.

    Seeds alone do not identify common Gaussian coordinates when durations
    differ. Every query therefore owns its original plan and reference frames.
    Audio hashes certify the artifact, not a requirement for future bitwise
    waveform equality. Capability checks use the declared task tolerances.
    """
    import soundfile as sf
    path = Path(pointer['path'])
    if sha256_file(path) != pointer['sha256']:
        raise ValueError('original audio reference manifest changed')
    value = json.loads(path.read_text())
    if (value.get('contract') != 'event_original_native_audio_reference_v1'
            or value['identity'] != native_reference_identity(protocol)
            or value['optimizer_steps'] != 0 or value['model_unchanged'] is not True):
        raise ValueError('audio reference must originate from the unchanged original policy')
    records = value['records']
    if [record['sample_id'] for record in records] != [row['sample_id'] for row in rows]:
        raise ValueError('original audio reference request panel changed')
    for record, row in zip(records, rows):
        if any(record[key] != row[key] for key in ('request', 'requirements')):
            raise ValueError('original audio reference request or requirements changed')
        queries = record['queries']
        seeds = list(validate_collection_seeds(protocol['guard_seeds'][row['sample_id']]))
        if len(seeds) < 2 or [query['seed'] for query in queries] != seeds:
            raise ValueError('original audio reference noise coverage changed')
        for query in queries:
            if (not query['actions'] or any(action != 0 for action in query['actions'])
                    or query['plan'] != query['original_plan']):
                raise ValueError('original audio reference cannot include learned refinements')
            if any(type(query[key]) is not int or query[key] <= 0
                    for key in ('reference_frames', 'model_num_samples')):
                raise ValueError('original audio reference geometry is missing')
            score = query['score']
            if not all(math.isfinite(x) for x in [score['utility'], *score['costs'].values()]):
                raise ValueError('original audio reference scores must be finite')
            audio = query['audio']
            if sha256_file(audio['path']) != audio['sha256']:
                raise ValueError('original reference audio changed')
            info = sf.info(audio['path'])
            if (info.samplerate != 44100 or info.channels != 4
                    or info.frames != query['model_num_samples']):
                raise ValueError('original reference FOA geometry changed')
    return value
