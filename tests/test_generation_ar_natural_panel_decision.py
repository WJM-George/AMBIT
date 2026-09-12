"""Acceptance must distinguish natural transfer, grammar fit and pending labels."""
from copy import deepcopy
import importlib.util
from pathlib import Path

_path=Path(__file__).resolve().parents[1]/'scripts/t2a/diagnostics/evaluate_generation_ar_natural_candidates.py'
_spec=importlib.util.spec_from_file_location('natural_candidate_scoring',_path)
_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)


def panel_result(joint,upper=None,pending=0):
    metrics={k:{'rate':.9} for k in ('count','core_recall','core_precision','motion','onset','offset','start','end','speech_words','all_requested_fields_joint')}
    metrics.update(joint={'rate':joint},joint_upper_bound={'rate':joint if upper is None else upper})
    return {'metrics':metrics,'pending_semantics':0,'pending_completion_that_can_change_joint':pending}


def candidates():
    return {candidate:{panel:{str(n):panel_result(.2 if candidate=='natural_mix' else .05) for n in range(1,5)} for panel in ('independent_N0_natural','mixed_request_first','precise_retention')} for candidate in ('precise_control','natural_mix','r3_baseline')}


def test_grammar_fit_cannot_mask_failed_independent_natural_transfer():
    data=candidates()
    for n in data['natural_mix']['independent_N0_natural']:
        data['natural_mix']['independent_N0_natural'][n]=panel_result(.05)
        data['natural_mix']['mixed_request_first'][n]=panel_result(1.)
    assert _module.n2_decision(data)['status']=='FAIL_PILOT_RULE'


def test_pending_free_completion_can_neither_pass_nor_fail_a_resolvable_gain():
    data=candidates()
    for panel in ('independent_N0_natural','mixed_request_first'):
        for n in data['natural_mix'][panel]:data['natural_mix'][panel][n]=panel_result(.1,upper=.3,pending=1)
    assert _module.n2_decision(data)['status']=='PENDING_ACTIONABLE_REVIEWS'


def test_completed_pilot_pass_never_promotes_full_corpus():
    result=_module.n2_decision(candidates())
    assert result['status']=='PASS_PILOT_ONLY'
    assert result['full_data_promotion'] is False


def test_precise_regression_against_retained_baseline_blocks_promotion():
    data=candidates()
    for candidate in ('natural_mix','precise_control'):
        data[candidate]['precise_retention']['4']['metrics']['core_recall']['rate']=.85
    result=_module.n2_decision(data)
    assert result['status']=='FAIL_PILOT_RULE'
    assert any(g['comparison']=='r3_baseline' and not g['pass'] for g in result['precise_guardrails'])


def n3_candidates():
    base=candidates()['natural_mix']
    for panel in base.values():
        for cell in panel.values():
            cell['metrics']['count'].update(rate=.5,correct=2,total=4)
            cell['metrics']['joint']['rate']=0.;cell['metrics']['joint_upper_bound']['rate']=0.
    data={c:deepcopy(base) for c in ('n2_natural_mix','r3_baseline','count_aux','count_header_aug')}
    for c in ('count_aux','count_header_aug'):
        for n in ('2','3','4'):data[c]['mixed_request_first'][n]['metrics']['count'].update(rate=.75,correct=3,total=4)
    return data


def test_count_component_success_does_not_hide_zero_request_success():
    result=_module.n3_decision(n3_candidates())
    for arm in result['arms'].values():
        assert arm['count_hypothesis_status']=='PASS_COUNT_COMPONENT_ONLY'
        assert arm['overall_request_pilot_rule']['status']=='FAIL_PILOT_RULE'
        assert not arm['model_promoted'] and not arm['full_data_promotion']


def test_count_component_retention_is_checked_per_group_and_independently():
    data=n3_candidates()
    data['count_aux']['precise_retention']['3']['metrics']['count']['rate']=.47
    data['count_header_aug']['independent_N0_natural']['1']['metrics']['count']['correct']=1
    result=_module.n3_decision(data)
    assert all(a['count_hypothesis_status']=='FAIL_COUNT_HYPOTHESIS' for a in result['arms'].values())
