"""Validate fixed request/noise panels independently of model outputs."""


def load_original_native_guard_plans(path, *, release_bundle, sample_ids):
    """Recover recorded original AR plans, never a newly decoded reference.

    Audio-refinement protection is deliberately checked separately. A native
    proposal can be valid even when a learned later refinement is unsafe.
    """
    import json
    from pathlib import Path
    from .provenance import sha256_file
    manifest = json.loads(Path(path).read_text())
    if (manifest.get('contract') != 'recorded_original_native_guard_plans_v1'
            or manifest['release_bundle_sha256'] != sha256_file(release_bundle)
            or manifest['sample_ids'] != list(sample_ids) or len(set(sample_ids)) != len(sample_ids)):
        raise ValueError('native guard reference identity or request panel differs')
    plans = {}
    for entry in manifest['sources']:
        if sha256_file(entry['path']) != entry['sha256']:
            raise ValueError('recorded native reference changed')
        source = json.loads(Path(entry['path']).read_text())
        if (source.get('contract') != 'event_response_native_warmstart_preflight_v1'
                or source['phase'] != 'NATIVE_RESPONSE_PREFLIGHT_COMPLETE'
                or source['optimizer_steps'] != 0 or not source['model_unchanged']
                or source['initial_model_fingerprint'] != manifest['source_model_fingerprint']
                or source['final_model_fingerprint'] != manifest['source_model_fingerprint']
                or Path(source['protocol']['release_bundle']).resolve() != Path(release_bundle).resolve()):
            raise ValueError('guard reference must precede native model updates')
        for row in source['records']:
            key = row['sample_id']
            if key not in sample_ids:
                continue
            queries = row['queries']
            if (key in plans or not queries or
                    any(query['original_plan'] != queries[0]['original_plan'] for query in queries)):
                raise ValueError('native guard proposal is duplicated or varies within its recorded noise panel')
            plans[key] = queries[0]['original_plan']
    if set(plans) != set(sample_ids):
        raise ValueError('native guard references cannot omit a declared request')
    return plans


def validate_collection_seeds(seeds):
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError('declare a nonempty noise panel before collecting teachers')
    if any(type(seed) is not int or not 0 <= seed < 2 ** 63 for seed in seeds):
        raise ValueError('noise seeds must be nonnegative signed 64-bit integers')
    if len(set(seeds)) != len(seeds):
        raise ValueError('duplicate noise seeds cannot count as independent queries')
    return tuple(seeds)


def validate_query_panel(manifest):
    if manifest.get('contract') != 'event_opsd_native_query_panel_v1':
        raise ValueError('declare the native query collection contract')
    rows = manifest.get('rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError('the fixed native request panel is empty')
    identifiers, seeds = set(), set()
    for row in rows:
        key = row['sample_id']
        if not isinstance(key, str) or not key or key in identifiers:
            raise ValueError('native panel request IDs must be nonempty and unique')
        identifiers.add(key)
        if row.get('split') not in ('feedback_train', 'feedback_development'):
            raise ValueError('reserved heldout requests are not a development query panel')
        if any(field in row for field in ('proposal', 'alternatives', 'cached_scores', 'paired_scores', 'teacher')):
            raise ValueError('native panels use freshly generated proposals and scores')
        noise = validate_collection_seeds(row['seeds'])
        if seeds.intersection(noise):
            raise ValueError('training/development requests require disjoint declared noise streams')
        seeds.update(noise)
    storage = manifest['storage']
    limit = storage['max_retained_bytes']
    if type(limit) is not int or limit <= 0:
        raise ValueError('declare the compact artifact byte budget')
    examples = storage['audio_sample_ids']
    if not isinstance(examples, list) or len(set(examples)) != len(examples) or not set(examples) <= identifiers:
        raise ValueError('audio examples must be fixed members of this request shard')
    if storage.get('audio_policy') != 'first_noise_baseline_teacher_and_worst_semantic':
        raise ValueError('declare which full audio evidence remains available')
    return sum(len(row['seeds']) for row in rows)
