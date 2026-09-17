import copy

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.coarse_native_plan_retention import (
    NativePlanTolerance, coarse_native_plan_retention,
)
from stable_audio_tools.training.transfusion_opsd.editing_native_update import (
    guarded_editing_joint_update, name_editing_optimizer_groups,
)


TOLERANCE = NativePlanTolerance(seconds=.5, azimuth_deg=10., gain_db=1.)
GROUPS = ('AR_adapters', 'shared_Transformer',
          'Editing_DiT_adapters_and_conditioning', 'structured_heads')


def scene():
    def source(kind, index, onset, azimuth):
        return dict(source_id=f'source_{index}', kind=kind, description='A quiet sound.',
                    activity=dict(onset_sec=onset, offset_sec=8.), gain_db=0.,
                    trajectory=dict(type='static', position=dict(
                        azimuth_deg=azimuth, elevation_deg=0., distance_m=2.)))
    return dict(duration_sec=10., room=dict(type='moderate'),
                sources=[source('speech', 0, 2.3, 120.), source('music', 1, .5, -35.)])


def compare(before, after):
    return coarse_native_plan_retention(before, after, tolerance=TOLERANCE, preserve_room=True)


def test_small_changes_and_canonical_slot_permutation_are_allowed():
    before = scene(); after = copy.deepcopy(before)
    after['sources'].reverse()
    for i, source in enumerate(after['sources']):
        source['source_id'] = f'source_{i}'
        source['description'] = 'A soft, quiet sound.'
        source['activity']['onset_sec'] += .4
        source['trajectory']['position']['azimuth_deg'] += 1.
    result = compare(before, after)
    assert result['passed'] and len(result['text_changes']) == 2


def test_large_source_timing_shift_is_detected_despite_slot_permutation():
    before = scene(); after = copy.deepcopy(before)
    after['sources'][0]['activity']['onset_sec'] = .65
    after['sources'].reverse()
    result = compare(before, after)
    assert not result['passed']
    assert [x['field'] for x in result['failures']] == ['sources.speech.activity.onset_sec']


def test_wraparound_is_shortest_azimuth_and_unrequested_axes_are_diagnostic():
    before = scene(); before['sources'][0]['trajectory']['position']['azimuth_deg'] = 179.
    after = copy.deepcopy(before)
    p = after['sources'][0]['trajectory']['position']
    p.update(azimuth_deg=-179., elevation_deg=30., distance_m=5.)
    result = compare(before, after)
    assert result['passed']
    strict = NativePlanTolerance(seconds=.5, azimuth_deg=10., gain_db=1., elevation_deg=10., distance_ratio=1.25)
    checked = coarse_native_plan_retention(before, after, tolerance=strict, preserve_room=True)
    assert len(checked['failures']) == 2


def test_room_and_motion_direction_changes_are_independent_failures():
    before = scene()
    p = before['sources'][0]['trajectory']['position']
    before['sources'][0]['trajectory'] = dict(type='linear', start=p, end=dict(p, azimuth_deg=-30.))
    after = copy.deepcopy(before)
    trajectory = after['sources'][0]['trajectory']
    trajectory['start'], trajectory['end'] = trajectory['end'], trajectory['start']
    after['room']['type'] = 'dry'
    result = compare(before, after)
    assert not result['passed'] and len(result['failures']) == 3


def test_ambiguous_binding_and_invalid_candidate_cannot_pass():
    before = scene(); ambiguous = copy.deepcopy(before)
    ambiguous['sources'][1]['kind'] = 'speech'
    result = compare(ambiguous, ambiguous)
    assert not result['passed'] and not result['available']
    invalid = copy.deepcopy(before)
    invalid['sources'][0]['activity']['onset_sec'] = float('nan')
    assert not compare(before, invalid)['passed']
    invalid = copy.deepcopy(before); invalid['sources'].pop()
    assert compare(before, invalid)['failures'][0]['field'] == 'source_inventory'


def test_native_group_adapter_keeps_parameters_and_rates():
    values = [torch.nn.Parameter(torch.zeros(())) for _ in GROUPS]
    groups = [dict(params=[p], lr=(i+1)*1e-5, group_name=n) for i,(n,p) in enumerate(zip(GROUPS,values))]
    named = name_editing_optimizer_groups(groups)
    assert named is groups
    assert [g['lr'] for g in named] == [(i+1)*1e-5 for i in range(4)]
    assert all(g['params'][0] is p and g['name'] == n for g,p,n in zip(named,values,GROUPS))
    with pytest.raises(ValueError, match='four complete'):
        name_editing_optimizer_groups(groups[:-1])


class NativePlanModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(())) for _ in GROUPS])

    def native_plan(self, observation):
        result = copy.deepcopy(observation)
        if self.weights[0] + self.weights[1] + self.weights[3] > .001:
            result['sources'][0]['activity']['onset_sec'] = .65
        return result, None


def test_joint_proposal_uses_real_decoded_plan_and_keeps_both_routes_updating():
    model = NativePlanModel().eval()
    groups = [dict(params=[p], group_name=n, lr=.001, weight_decay=0.) for n,p in zip(GROUPS,model.weights)]
    optimizer = torch.optim.AdamW(groups)
    (sum(model.weights)-1).square().backward()
    full = dict.fromkeys(GROUPS, 1.)
    limited = dict.fromkeys(GROUPS, .1)
    limited['Editing_DiT_adapters_and_conditioning'] = 1.
    plan = scene()
    result = guarded_editing_joint_update(model, optimizer,
        anchors=[dict(identifier='development-only', observation=plan, plan=plan)],
        tolerance=TOLERANCE, preserve_room=True, trial_scales=[full, limited])
    assert result.accepted and len(result.trials) == 2 and result.scales == limited
    assert not result.trials[0]['validation']['passed']
    assert all(p > 0 for p in model.weights)
    assert model.weights[2] > 5*model.weights[0]
    assert all(float(state['step']) == 1 for state in optimizer.state.values())
    assert all(g['changed_elements'] == 1 for g in result.actual_displacements.values())
