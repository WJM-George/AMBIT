"""Finite native-plan retention for a block without positive planning evidence.

The anchor is the student's decoded plan, not an unavailable paired target.
This is a conservative update constraint, not a request reward or a test of
text meaning. A subsequent block with verified beneficial planning changes
must supply a different, explicitly authorized constraint.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class NativePlanTolerance:
    seconds: float
    azimuth_deg: float
    gain_db: float
    elevation_deg: float | None = None
    distance_ratio: float | None = None

    def __post_init__(self):
        for key, value in asdict(self).items():
            if value is None and key in ('elevation_deg', 'distance_ratio'):
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError('Native-plan tolerances must be finite and nonnegative.')
        if self.azimuth_deg > 180 or (self.elevation_deg is not None and self.elevation_deg > 180):
            raise ValueError('Angular tolerances cannot exceed 180 degrees.')
        if self.distance_ratio is not None and self.distance_ratio < 1:
            raise ValueError('A symmetric distance ratio must be at least one.')


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('A required plan value is not a finite number.')
    return value


def _validate(plan):
    duration = _number(plan['duration_sec'])
    if duration <= 0 or not isinstance(plan['room']['type'], str):
        raise ValueError('Invalid duration or room type.')
    sources = plan['sources']
    if not isinstance(sources, list) or not sources:
        raise ValueError('Expected a nonempty native source list.')
    for source in sources:
        if source['kind'] not in ('speech', 'music', 'sound'):
            raise ValueError('Unsupported source kind.')
        activity = source['activity']
        onset, offset = (_number(activity[k]) for k in ('onset_sec', 'offset_sec'))
        if not 0 <= onset < offset <= duration + 1e-6:
            raise ValueError('Invalid source activity interval.')
        _number(source.get('gain_db', 0.))
        trajectory = source['trajectory']
        if trajectory['type'] not in ('static', 'linear'):
            raise ValueError('Only native static and linear trajectories are covered.')
        for name in ('position',) if trajectory['type'] == 'static' else ('start', 'end'):
            p = trajectory[name]
            _number(p['azimuth_deg'])
            if not -90 <= _number(p['elevation_deg']) <= 90 or _number(p['distance_m']) <= 0:
                raise ValueError('Invalid source position.')
    return {s['kind']: s for s in sources}


def coarse_native_plan_retention(before, after, *, tolerance: NativePlanTolerance,
                                 preserve_room: bool):
    """Check decoded behavior, allowing slot permutations and small changes.

    Binding is available only when each kind occurs once. Same-kind scenes
    need a separate verified binding; they must not silently pass. Text is
    recorded as changed but is not required to match an often imperfect
    anchor. No content or audio-preservation claim follows from passing.
    """
    if not isinstance(tolerance, NativePlanTolerance) or type(preserve_room) is not bool:
        raise TypeError('Supply explicit native tolerances and a room retention policy.')
    a = _validate(before)  # An invalid anchor is a caller error, not a reward.
    result = dict(passed=False, available=True, failures=[], changes=[], text_changes=[],
                  tolerance=asdict(tolerance), preserve_room=preserve_room,
                  binding='Unique source kind; native slot labels may permute.',
                  scope='Finite coarse plan retention only. No text-meaning, audio quality or full request certification.')
    try:
        b = _validate(after)
    except (KeyError, TypeError, ValueError) as error:
        result['failures'].append(dict(field='native_plan_validity', reason=str(error)))
        return result
    if len(a) != len(before['sources']) or len(b) != len(after['sources']):
        result.update(available=False)
        result['failures'].append(dict(field='source_binding', reason='Repeated source kinds require explicit binding.'))
        return result
    if len(before['sources']) != len(after['sources']) or set(a) != set(b):
        result['failures'].append(dict(field='source_inventory', before=sorted(a), after=sorted(b)))
        return result

    def category(field, old, new, checked=True):
        if old != new:
            item = dict(field=field, before=old, after=new, checked=checked)
            result['changes'].append(item)
            if checked:
                result['failures'].append(item)

    def number(field, old, new, limit, *, circular=False, ratio=False):
        delta = (max(old/new, new/old) if ratio else
                 abs((new-old+180.) % 360.-180.) if circular else abs(new-old))
        if delta > (1. if ratio else 0.):
            item = dict(field=field, before=old, after=new, difference=delta, allowed=limit)
            result['changes'].append(item)
            if limit is not None and delta > limit + 1e-9:
                result['failures'].append(item)

    number('duration_sec', before['duration_sec'], after['duration_sec'], tolerance.seconds)
    category('room.type', before['room']['type'], after['room']['type'], preserve_room)
    for kind, old in a.items():
        new = b[kind]
        prefix = 'sources.'+kind+'.'
        for key in ('transcript', 'speaker_description', 'description'):
            if old.get(key) != new.get(key):
                result['text_changes'].append(prefix+key)
        for key in ('onset_sec', 'offset_sec'):
            number(prefix+'activity.'+key, old['activity'][key], new['activity'][key], tolerance.seconds)
        number(prefix+'gain_db', old.get('gain_db', 0.), new.get('gain_db', 0.), tolerance.gain_db)
        x, y = old['trajectory'], new['trajectory']
        category(prefix+'trajectory.type', x['type'], y['type'])
        if x['type'] != y['type']:
            continue
        for point in ('position',) if x['type'] == 'static' else ('start', 'end'):
            p, q = x[point], y[point]
            path = prefix+'trajectory.'+point+'.'
            number(path+'azimuth_deg', p['azimuth_deg'], q['azimuth_deg'], tolerance.azimuth_deg, circular=True)
            number(path+'elevation_deg', p['elevation_deg'], q['elevation_deg'], tolerance.elevation_deg)
            number(path+'distance_m', p['distance_m'], q['distance_m'], tolerance.distance_ratio, ratio=True)
    result['passed'] = not result['failures']
    return result
