import copy
import json

import pytest

from stable_audio_tools.training.transfusion_opsd.event_query_panel import (
    validate_collection_seeds, validate_query_panel, load_original_native_guard_plans,
)
from stable_audio_tools.training.transfusion_opsd.provenance import sha256_file


def panel():
    return {'contract': 'event_opsd_native_query_panel_v1',
        'rows': [{'sample_id': 'train', 'split': 'feedback_train', 'seeds': [10, 11]},
                 {'sample_id': 'dev', 'split': 'feedback_development', 'seeds': [20, 21]}],
        'storage': {'max_retained_bytes': 1024, 'audio_sample_ids': ['dev'],
                    'audio_policy': 'first_noise_baseline_teacher_and_worst_semantic'}}


def test_panel_counts_all_queries_before_any_score_exists():
    value = panel()
    before = copy.deepcopy(value)
    assert validate_query_panel(value) == 4
    assert value == before


@pytest.mark.parametrize('seeds', [[], [7, 7], [True], [-1], [2 ** 63], [1.5]])
def test_invalid_or_duplicate_noise_cannot_inflate_denominator(seeds):
    with pytest.raises(ValueError):
        validate_collection_seeds(seeds)


@pytest.mark.parametrize('field', ['proposal', 'alternatives', 'cached_scores', 'paired_scores', 'teacher'])
def test_fresh_native_panel_does_not_accept_old_outputs(field):
    value = panel()
    value['rows'][0][field] = {}
    with pytest.raises(ValueError, match='freshly generated'):
        validate_query_panel(value)


def test_dev_noise_cannot_overlap_training_noise():
    value = panel()
    value['rows'][1]['seeds'][0] = 10
    with pytest.raises(ValueError, match='disjoint'):
        validate_query_panel(value)


def test_reserved_holdout_is_not_implicitly_allowed():
    value = panel()
    value['rows'][1]['split'] = 'heldout'
    with pytest.raises(ValueError, match='heldout'):
        validate_query_panel(value)


def test_audio_retention_cannot_silently_expand_to_undeclared_requests():
    value = panel()
    value['storage']['audio_sample_ids'].append('later_chosen_request')
    with pytest.raises(ValueError, match='fixed members'):
        validate_query_panel(value)


@pytest.mark.parametrize('change', ['none', 'updated', 'varied_plan', 'tampered'])
def test_native_reference_requires_unchanged_original_plans_and_provenance(tmp_path, change):
    release = tmp_path/'release.json'; release.write_text('{}')
    source_path = tmp_path/'native.json'
    source = {'contract': 'event_response_native_warmstart_preflight_v1',
        'phase': 'NATIVE_RESPONSE_PREFLIGHT_COMPLETE', 'optimizer_steps': 0, 'model_unchanged': True,
        'initial_model_fingerprint': 'original', 'final_model_fingerprint': 'original',
        'protocol': {'release_bundle': str(release)}, 'protected': False,
        'records': [{'sample_id': 'train', 'queries': [
            {'original_plan': {'duration_sec': 5.4}}, {'original_plan': {'duration_sec': 5.4}}]}]}
    if change == 'updated': source['optimizer_steps'] = 1
    if change == 'varied_plan': source['records'][0]['queries'][1]['original_plan']['duration_sec'] = 4.1
    source_path.write_text(json.dumps(source))
    manifest = {'contract': 'recorded_original_native_guard_plans_v1',
        'release_bundle_sha256': sha256_file(release), 'source_model_fingerprint': 'original',
        'sample_ids': ['train'], 'sources': [{'path': str(source_path), 'sha256': sha256_file(source_path)}]}
    path = tmp_path/'reference.json'; path.write_text(json.dumps(manifest))
    if change == 'tampered': source_path.write_text('{}')
    if change == 'none':
        # A valid original proposal remains usable even when a later
        # refinement's audio was unsafe; the initial audio guard is separate.
        assert load_original_native_guard_plans(path, release_bundle=release,
            sample_ids=['train']) == {'train': {'duration_sec': 5.4}}
    else:
        with pytest.raises(ValueError):
            load_original_native_guard_plans(path, release_bundle=release, sample_ids=['train'])
