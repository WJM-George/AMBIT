"""Request-grounded proposals and fixed-reference native field retention.

New-request functions accept only the instruction, source observations and
student/reference predictions. Paired labels remain in the paired objective.
"""
from __future__ import annotations

import copy
import json
import math
import re

import torch

from .editing_stream import NUMBER, angular_distance
from .native_coarse_choice_retention import native_azimuth_cone_targets
from .native_token_alignment import validate_native_plan_tokens, single_native_structured_edit


def request_facts(request, operation):
    if operation == 'event_removal':
        return None
    fields, kind = {}, None
    for match in re.finditer(r'(speech saying|voice described as|sound described as|music described as)\s*(?=["“])', request, re.I):
        tail = request[match.end():]
        try:
            value = json.JSONDecoder().raw_decode(tail)[0] if tail.startswith('"') else tail[1:tail.index('”', 1)]
        except (ValueError, json.JSONDecodeError):
            continue
        cue = match[1].lower()
        if cue == 'voice described as':
            fields['speaker_description'] = value
        else:
            incoming = cue.split()[0]
            if kind is not None and incoming != kind:
                return None
            kind = incoming
            fields['transcript' if kind == 'speech' else 'description'] = value
    if kind is None:
        return None
    values = {}
    for key, unit in [('azimuth', 'degrees'), ('elevation', 'degrees'), ('distance', 'meters')]:
        values[key] = [float(x) for x in re.findall(key + r'\s+(' + NUMBER + r')\s+' + unit, request, re.I)]
    # "active from exactly ... through exactly ..." is present in the native
    # data but absent from the earlier exploration's interval parser.
    patterns = [
        r'active\s+(?:from|between)\s+(?:exactly\s+)?(' + NUMBER + r')\s+seconds?\s+(?:through|to|and)\s+(?:exactly\s+)?(' + NUMBER + r')\s+seconds?',
        r'(?:between|from)\s+(?:exactly\s+)?(' + NUMBER + r')\s+(?:and|to)\s+(?:exactly\s+)?(' + NUMBER + r')\s+seconds',
        r'(?:active interval to|active interval of)\s+(' + NUMBER + r')\s+(?:through|to)\s+(' + NUMBER + r')\s+seconds',
    ]
    intervals = {tuple(map(float, m.groups())) for pattern in patterns for m in re.finditer(pattern, request, re.I)}
    activity = list(next(iter(intervals))) if len(intervals) == 1 else None
    return dict(kind=kind, fields=fields, azimuths=values['azimuth'], elevations=values['elevation'],
                distances=values['distance'], activity=activity, operation=operation)


def propose_current_decision(adapter, observation, plan, tokens, facts, *, step):
    if facts is None:
        return None
    from .editing_request_constraints import bind_edit_target
    binding = bind_edit_target(plan, facts)
    if not binding['available']:
        return None
    source = next(s for s in plan['sources'] if s['source_id'] == binding['source_id'])
    if source['kind'] != facts['kind']:
        # Direct request supervision repairs this mismatch. Do not reinforce
        # a wrong-category plan through a spatially qualified terminal teacher.
        return None
    candidates = []

    def propose(field, edit):
        p = copy.deepcopy(plan)
        item = next(s for s in p['sources'] if s['source_id'] == source['source_id'])
        edit(item)
        candidates.append((field, p))

    trajectory = source['trajectory']
    points = ['position'] if trajectory['type'] == 'static' else (['start', 'end'] if trajectory['type'] == 'linear' else [])
    for keys, field in [('azimuths', 'azimuth_deg'), ('elevations', 'elevation_deg'), ('distances', 'distance_m')]:
        requested = facts[keys]
        if len(requested) != len(points):
            continue
        for i, point in enumerate(points):
            old, target = trajectory[point][field], requested[i]
            if field == 'azimuth_deg':
                delta = (target - old + 180) % 360 - 180
                value = old + (max(-10., min(10., delta)) if abs(delta) >= 1 else (5. if step % 2 else -5.))
                value = (round(value) + 180) % 360 - 180
                if angular_distance(value, target) > max(30., angular_distance(old, target)):
                    continue
            elif field == 'elevation_deg':
                value = min(90., max(-90., old + max(-5., min(5., target - old))))
            else:
                value = max(.1, old + max(-.25, min(.25, target - old)))
            propose(point + '/' + field, lambda s, p=point, f=field, v=value: s['trajectory'][p].__setitem__(f, v))
    if facts['activity']:
        for index, field in enumerate(['onset_sec', 'offset_sec']):
            desired = min(plan['duration_sec'], max(0., facts['activity'][index]))
            valid = desired < source['activity']['offset_sec'] if index == 0 else desired > source['activity']['onset_sec']
            if valid:
                propose(field, lambda s, f=field, v=desired: s['activity'].__setitem__(f, v))
    if not candidates:
        return None
    from ...models.sceneplan_transfusion_editing_pipeline import _align_decoded_sceneplan_to_audio_duration
    old = list(map(int, tokens))
    canonical = validate_native_plan_tokens(adapter.codec, old, plan)
    start = step % len(candidates)
    for field, candidate in candidates[start:] + candidates[:start]:
        try:
            ids = adapter.codec.encode(candidate)['input_ids'].tolist()
            candidate = _align_decoded_sceneplan_to_audio_duration(
                adapter.codec.decode(ids, sample_id=plan['sample_id']), observation.model_num_samples / 44100)
            ids = adapter.codec.encode(candidate)['input_ids'].tolist()
        except (ValueError, AssertionError):
            continue
        edit = single_native_structured_edit(adapter.codec, old, canonical, ids)
        if edit is None:
            continue
        position, native_candidate = edit
        allowed = sorted(adapter.allowed_next_ids(observation, old[:position]))
        if native_candidate[position] in allowed:
            candidate = _align_decoded_sceneplan_to_audio_duration(
                adapter.codec.decode(native_candidate, sample_id=plan['sample_id']), observation.model_num_samples / 44100)
            if adapter.codec.encode(candidate)['input_ids'].tolist() != ids:
                raise ValueError('Native candidate changed fields outside its single authorized edit.')
            return dict(field=field, position=position, prefix=old[:position], legal_ids=allowed,
                        choice_ids=[old[position], native_candidate[position]], plans=[plan, candidate], source_id=source['source_id'])
    return None


def execution_authorized_decision(decision, metrics, rewards):
    """Relax a reference field only after a verified alternative improves it.

    Merely proposing a field, or finding a usable original-plan terminal, is
    insufficient. Other fields and unavailable/failed proposals stay anchored.
    """
    if decision is None or len(rewards) != 2 or rewards[1] <= rewards[0]:
        return None
    alternatives = [m for m in metrics if m['plan_index'] == 1]
    if len(alternatives) != 2 or not all(m.get('qualified_terminal', False) for m in alternatives):
        return None
    return decision


def reference_field_targets(codec, tokens, plan, allowed_fn, reference_logits, *, decision=None, radius_deg=12.):
    """Cover every non-text native decision, including all numeric fields.

    The reference logits must be from frozen parameters, recomputed on these
    exact student prefixes. Never use current-student argmax as the anchor.
    """
    ids = list(map(int, tokens))
    begin, end = codec._tid('<text_begin>'), codec._tid('<text_end>')
    inside = False
    holds = []
    for pos in range(1, len(ids)):
        previous = ids[pos - 1]
        if previous == begin:
            inside = True
        elif previous == end:
            inside = False
        if inside or (decision is not None and pos == decision['position']):
            continue
        allowed = sorted(allowed_fn(ids[:pos]))
        if len(allowed) > 1:
            holds.append(dict(position=pos, ids=allowed,
                              p=reference_logits[pos - 1, allowed].float().softmax(-1).detach()))
    cones = native_azimuth_cone_targets(codec, ids, plan, allowed_fn, radius_deg=radius_deg)
    angles = {token: float(i - 180) for i, token in enumerate(codec.azimuth_ids)}
    anchored = []
    for item in cones:
        if decision is not None and item['position'] == decision['position']:
            continue
        allowed = item['allowed_ids']
        token = allowed[int(reference_logits[item['position'] - 1, allowed].argmax())]
        center = angles[token]
        item = dict(item, reference_azimuth_deg=center, anchor='frozen40k_on_current_prefix',
                    acceptable_ids=[i for i in allowed if angular_distance(angles[i], center) <= radius_deg])
        anchored.append(item)
    return holds, anchored


def reference_kl(logits, holds):
    if not holds:
        return logits.sum() * 0
    return torch.stack([(h['p'] * (h['p'].clamp_min(1e-30).log()
        - logits[h['position'] - 1, h['ids']].float().log_softmax(-1))).sum() for h in holds]).mean()


def _cartesian(azimuth, elevation, distance=1.):
    a, e = math.radians(azimuth), math.radians(elevation)
    return torch.tensor([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]) * distance


@torch.no_grad()
def request_spatial_measure(wave, plan, facts):
    """Time-local requested 3D direction on isolated target windows.

    Distance is not identifiable from uncalibrated amplitude; its supervision
    stays in native planning and paired audio. Overlap is not source separation.
    """
    if not facts or len(facts['azimuths']) not in (1, 2):
        return dict(available=False)
    from .editing_request_constraints import bind_edit_target
    binding = bind_edit_target(plan, facts)
    if not binding['available']:
        return dict(available=False)
    target = next(s for s in plan['sources'] if s['source_id'] == binding['source_id'])
    points = len(facts['azimuths'])
    if target['trajectory']['type'] != ('static' if points == 1 else 'linear'):
        return dict(available=False)
    a, b = facts['activity'] or [target['activity']['onset_sec'], target['activity']['offset_sec']]
    ranges = [(max(0., a), min(wave.shape[-1] / 44100, b))]
    for other in plan['sources']:
        if other['source_id'] == target['source_id']:
            continue
        u, v = other['activity']['onset_sec'], other['activity']['offset_sec']
        ranges = [(x, y) for l, r in ranges for x, y in ((l, min(r, u)), (max(l, v), r)) if y - x >= .2]
    ranges = [(l, r) for l, r in ranges if r - l >= .2]
    if not ranges:
        return dict(available=False)
    has_elevation = len(facts.get('elevations', [])) == points
    positions = [_cartesian(az, facts['elevations'][i] if has_elevation else 0.,
        facts['distances'][i] if len(facts.get('distances', [])) == points else 1.)
        for i, az in enumerate(facts['azimuths'])]
    windows = []
    for l, r in ranges[:3]:
        for fraction in (.2, .5, .8):
            center = l + (r - l) * fraction
            i, j = max(0, round((center - .1) * 44100)), min(wave.shape[-1], round((center + .1) * 44100))
            chunk = wave[0, :, i:j].double()
            intensity = (chunk[0, None] * chunk[[3, 1, 2]]).sum(-1)
            norm = (chunk[0].square().sum() * chunk[[3, 1, 2]].square().sum()).sqrt().clamp_min(1e-20)
            valid = j > i and float(chunk[0].square().mean().sqrt()) > 1e-5 and float(intensity.norm() / norm) >= .1
            u = max(0., min(1., (center - a) / max(b - a, 1e-6)))
            expected = positions[0] if points == 1 else positions[0] * (1 - u) + positions[1] * u
            expected_az = math.degrees(math.atan2(float(expected[1]), float(expected[0])))
            expected_el = math.degrees(math.atan2(float(expected[2]), float(expected[:2].norm())))
            az = float(torch.rad2deg(torch.atan2(intensity[1], intensity[0]))) if valid else None
            el = float(torch.rad2deg(torch.atan2(intensity[2], intensity[:2].norm()))) if valid else None
            horizontal = angular_distance(az, expected_az) if valid else 180.
            vertical = abs(el - expected_el) if valid and has_elevation else (180. if has_elevation else 0.)
            error = max(horizontal, vertical)
            windows.append(dict(observable=valid, angle_error_deg=error, horizontal_error_deg=horizontal,
                elevation_error_deg=vertical if has_elevation else None, azimuth_deg=az, elevation_deg=el,
                expected_azimuth_deg=expected_az, interval=[i, j]))
    return dict(available=True, windows=windows,
        mean_capped_angle_deg=sum(min(90., w['angle_error_deg']) for w in windows) / len(windows),
        failures=sum(w['angle_error_deg'] > 30 for w in windows),
        unobservable=sum(not w['observable'] for w in windows),
        trajectory=target['trajectory']['type'], elevation_requested=has_elevation)


def local_covariance(audio, *, sample_rate=44100):
    """Signed four-channel time-local covariance; no single-source assumption."""
    width = max(1, sample_rate // 5)
    length = audio.shape[-1] // width * width
    x = audio[..., :length].float().reshape(4, -1, width)
    covariance = torch.einsum('ctw,dtw->tcd', x, x) / width
    energy = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
    normalized = covariance / energy[:, None, None].clamp_min(1e-10)
    return normalized, energy


@torch.no_grad()
def preserved_window_measure(wave, source_wave, plan, facts):
    """Compare source only where the requested edit is temporally absent.

    Whole-mixture equality is not asserted in overlapping edit intervals.
    Removal/ambiguous binding is reported unavailable, not passed by default.
    """
    if facts is None:
        return dict(available=False, reason='No uniquely bound edited interval')
    from .editing_request_constraints import bind_edit_target
    binding = bind_edit_target(plan, facts)
    if not binding['available']:
        return dict(available=False, reason='Ambiguous target binding')
    target = next(s for s in plan['sources'] if s['source_id'] == binding['source_id'])
    a, b = facts['activity'] or [target['activity']['onset_sec'], target['activity']['offset_sec']]
    generated, ge = local_covariance(wave[0])
    original, se = local_covariance(source_wave[0])
    n = min(len(generated), len(original))
    centers = (torch.arange(n, device=wave.device) + .5) * .2
    mask = ((centers < a - .1) | (centers > b + .1)) & (se[:n] > 1e-9)
    if not bool(mask.any()):
        return dict(available=False, reason='No isolated unchanged source window')
    error = (generated[:n] - original[:n]).square().sum((-1, -2))
    weights = se[:n][mask]
    return dict(available=True, windows=int(mask.sum()),
        covariance_error=float((error[mask] * weights).sum() / weights.sum()),
        missing_energy_fraction=float((ge[:n][mask] < se[:n][mask] * .01).float().mean()))
