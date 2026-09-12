"""Check missing-source accounting and physical tolerance boundaries."""
import copy
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'scripts/t2a/diagnostics/generation_ar_fidelity.py'
spec = importlib.util.spec_from_file_location('generation_ar_fidelity', path)
fidelity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fidelity)


def source(identifier='source_0', azimuth=179):
    return {'source_id': identifier, 'kind': 'sound', 'description': 'dog barking',
            'activity': {'onset_sec': 1., 'offset_sec': 2.},
            'trajectory': {'type': 'static', 'position': {'azimuth_deg': azimuth, 'elevation_deg': 0., 'distance_m': 1.}}}


def test_circular_azimuth_and_missing_source_denominator():
    target = {'sources': [source(), source('source_1')]}
    prediction = {'sources': [source(azimuth=-179)]}
    row = fidelity.compare_fields(target, prediction)
    summary = fidelity.summarize_fields([row])
    assert summary['missing_sources'] == 1
    assert summary['source_count_accuracy'] == 0
    assert summary['motion_accuracy_per_requested_source'] == .5
    errors = summary['field_errors']['start_azimuth_deg']
    assert errors['matched_source_mae'] == 2
    assert errors['within_tolerance_per_requested_source']['strict'] == .5
    assert summary['scene_spatiotemporal_pass_rate']['loose'] == 0


def test_extra_source_cannot_pass_scene_conjunction():
    target = {'sources': [source()]}
    prediction = {'sources': [source(), source('source_1')]}
    row = fidelity.compare_fields(target, prediction)
    assert row['extra_sources'] == 1
    assert row['sources'][0]['passes']['strict']
    assert not any(row['scene_spatiotemporal_pass'].values())


def test_time_boundary_and_motion_are_required():
    target = {'sources': [source()]}
    prediction = copy.deepcopy(target)
    prediction['sources'][0]['activity']['offset_sec'] += .25
    row = fidelity.compare_fields(target, prediction)
    assert row['scene_spatiotemporal_pass']['medium']
    assert not row['scene_spatiotemporal_pass']['strict']
    prediction['sources'][0]['activity']['offset_sec'] += .0001
    assert not fidelity.compare_fields(target, prediction)['scene_spatiotemporal_pass']['medium']


def test_generation_failure_is_a_failed_requested_source():
    row = fidelity.compare_fields({'sources': [source()]}, None)
    summary = fidelity.summarize_fields([row])
    assert summary['missing_sources'] == 1
    assert summary['matched_sources'] == 0
    assert summary['source_count_accuracy'] == 0
    assert summary['scene_spatiotemporal_pass_rate']['loose'] == 0
