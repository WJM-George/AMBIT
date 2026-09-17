"""Connect actual Editing plans to the existing shared Adam displacement guard."""
from __future__ import annotations

from .coarse_native_plan_retention import coarse_native_plan_retention
from .guarded_joint_update import guarded_joint_adam_update


def name_editing_optimizer_groups(groups):
    """Keep native groups/rates; provide the common guard's explicit names."""
    expected = {'AR_adapters', 'shared_Transformer',
                'Editing_DiT_adapters_and_conditioning', 'structured_heads'}
    names = [g.get('group_name') for g in groups]
    if len(groups) != len(expected) or set(names) != expected:
        raise ValueError('Expected the four complete native Editing optimizer groups.')
    if any(not g['params'] or (g.get('name') is not None and g['name'] != g['group_name']) for g in groups):
        raise ValueError('Native Editing groups must be nonempty and consistently named.')
    for group in groups:
        group['name'] = group['group_name']
    return groups


def guarded_editing_joint_update(adapter, optimizer, *, anchors, tolerance,
                                 trial_scales, preserve_room, scheduler=None):
    """Validate full greedy plans on fixed source/request observations.

    Call after joint backward and clipping, with the adapter in eval mode.
    Each anchor contains an identifier, observation and the original native
    plan. Targets, scores and teacher-forced prefixes are not validator inputs.
    All validation forwards count toward the method's compute budget.
    """
    if adapter.training:
        raise ValueError('Native greedy validation requires eval mode.')
    anchors = list(anchors)
    if not anchors or len({a['identifier'] for a in anchors}) != len(anchors):
        raise ValueError('Declare a nonempty, distinct native retention panel.')
    name_editing_optimizer_groups(optimizer.param_groups)
    # Reject unsupported anchors before proposing a parameter update.
    for item in anchors:
        check = coarse_native_plan_retention(item['plan'], item['plan'],
                                            tolerance=tolerance, preserve_room=preserve_room)
        if not check['passed']:
            raise ValueError('The native anchor lacks supported source binding.')

    def validate():
        rows = []
        for item in anchors:
            plan, _ = adapter.native_plan(item['observation'])
            result = coarse_native_plan_retention(item['plan'], plan,
                                                   tolerance=tolerance, preserve_room=preserve_room)
            rows.append(dict(identifier=item['identifier'], plan=plan, result=result))
        return dict(passed=all(row['result']['passed'] for row in rows), rows=rows,
                    scope='Actual greedy native plans; coarse behavior only, no audio success claim.')

    return guarded_joint_adam_update(adapter, optimizer, trial_scales=trial_scales,
                                     validate_native=validate, scheduler=scheduler)
