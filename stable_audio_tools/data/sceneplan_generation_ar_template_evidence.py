"""Training-only producer trace for qualitative evidence, never raw inference.

Reconstruct each existing template block from its saved request-owned evidence
and track format-field offsets. This disambiguates repeated words and quoted
control-like content without parsing an inference request into a plan.
"""
from string import Formatter


def template_evidence_spans(request, requirements, recipe):
    rows = []; cursor = 0; ordinals = ('first', 'second', 'third', 'fourth')
    for index, source in enumerate(requirements['sources']):
        activity = next(c['evidence'] for c in source['constraints'] if c['op'] == 'time_phase')
        motion = next(c['evidence'] for c in source['constraints'] if c['op'] == 'motion')
        activity = activity[0].upper() + activity[1:]; motion = motion[0].upper() + motion[1:]
        values = {'ordinal': ordinals[index], 'identity': source['evidence'], 'activity': activity, 'motion': motion,
            'activity_lower': activity[0].lower() + activity[1:], 'motion_lower': motion[0].lower() + motion[1:]}
        block = ''; spans = {}
        for literal, field, specification, conversion in Formatter().parse(recipe['event_form']):
            block += literal
            if field is None: continue
            if specification or conversion: raise ValueError('Unexpected template format conversion')
            start = len(block); block += values[field]
            if field in ('activity', 'activity_lower', 'motion', 'motion_lower', 'identity'):
                spans[field.removesuffix('_lower')] = (start, len(block) - 1)
        start = request.find(block, cursor)
        if start < 0 or (index and request[cursor:start] != ' '):
            raise ValueError('Saved evidence does not reconstruct the source template block')
        cursor = start + len(block)
        spans = {key: (start + low, start + high) for key, (low, high) in spans.items()}
        if set(spans) != {'identity', 'activity', 'motion'}: raise ValueError('Incomplete producer trace')
        for key, evidence in [('identity', source['evidence']), ('activity', activity), ('motion', motion)]:
            low, high = spans[key]
            if request[low:high + 1].lower() != evidence.lower(): raise ValueError('Producer trace text mismatch')
        rows.append({'source_key': source['key'], 'block': (start, cursor - 1), **spans})
    if request[cursor:] != ' Use only these sources.': raise ValueError('Unexpected trailing source/template text')
    return rows


def focused_motion_evidence(request, source, spans):
    """Training-only field ownership within the already traced motion block.

    An explicit radial sentence receives its own attention target. Angular
    endpoints and motion type use the preceding motion sentence. Search stays
    inside the owning producer-traced block, so quoted content cannot hijack it.
    """
    low, high = spans['motion']
    motion = request[low:high + 1]
    constraints = [c for c in source['constraints'] if c['op'] == 'distance_change']
    angular = radial = (low, high)
    if constraints:
        if len(constraints) != 1:
            raise ValueError('Expected one owned radial direction constraint')
        cue = constraints[0]['evidence']
        start = motion.lower().find(cue.lower())
        if start < 2 or motion.lower().find(cue.lower(), start + 1) >= 0:
            raise ValueError('Radial evidence does not uniquely address the traced motion block')
        radial = (low + start, low + start + len(cue) - 1)
        angular = (low, low + start - 2)
        if motion[start - 1] != ' ' or radial[1] != high or not request[angular[0]:angular[1] + 1].endswith('.'):
            raise ValueError('Unexpected producer layout for angular and radial evidence')
    return {'motion': angular, 'start': angular, 'end': angular,
        'onset': spans['activity'], 'offset': spans['activity'], 'radial': radial}
