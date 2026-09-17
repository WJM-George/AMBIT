"""Streaming data and current-prefix proposals for the E79 continuation.

The request reader never selects a source/target plan or target latent. The
paired reader remains the native joint dataset and is a separate partition.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def readonly(path):
    return sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro&immutable=1', uri=True)


class OrdinalStream:
    """Deterministic epoch permutations, disjoint rank shards, explicit cursor."""
    def __init__(self, ordinals, *, seed, rank=0, world=1, state=None):
        self.ordinals = np.asarray(ordinals, dtype=np.int64)
        if len(self.ordinals) < world or len(np.unique(self.ordinals)) != len(self.ordinals):
            raise ValueError('Use enough distinct rows for each rank.')
        self.seed, self.rank, self.world = seed, rank, world
        self.epoch, self.cursor = (0, 0) if state is None else (state['epoch'], state['cursor'])
        # A topology change can leave a few unconsumed positions behind its
        # new common frontier (e.g. uneven fixed warmup examples). Preserve
        # these exact permutation positions before reading beyond the frontier.
        self.pending_positions = [] if state is None else list(state.get('pending_positions', []))
        if any(p < 0 or p >= len(self.ordinals) for p in self.pending_positions):
            raise ValueError('Pending stream position outside this epoch.')
        self._order()
        if not 0 <= self.cursor <= len(self.order):
            raise ValueError('Stream cursor outside this epoch.')

    def _order(self):
        order = np.random.default_rng(self.seed + self.epoch).permutation(self.ordinals)
        self._global_order = order
        # Equal epoch lengths keep ranks on the same global permutation. A
        # maximum of world-1 tail rows rotate out for this epoch only.
        self.order = order[:len(order) // self.world * self.world][self.rank::self.world]

    def take(self, count):
        result = []
        for _ in range(count):
            if self.pending_positions:
                result.append(int(self._global_order[self.pending_positions.pop(0)]))
                continue
            if self.cursor == len(self.order):
                self.epoch += 1
                self.cursor = 0
                self._order()
            result.append(int(self.order[self.cursor]))
            self.cursor += 1
        return result

    def state_dict(self):
        state = dict(epoch=self.epoch, cursor=self.cursor, seed=self.seed, rank=self.rank, world=self.world)
        if self.pending_positions:
            state['pending_positions'] = list(self.pending_positions)
        return state


def training_partitions(rows, seed, request_fraction=.1, request_only_extra=()):
    order = np.random.default_rng(seed).permutation(rows)
    request = np.unique(np.concatenate((order[:int(rows * request_fraction)], np.asarray(request_only_extra, dtype=np.int64))))
    paired = np.setdiff1d(np.arange(rows), request)
    return request, paired


def homogeneous_microbatches(ordinals, bucket_for, microbatch_size, *, bucket_limits=None):
    """Keep the sampled update intact, respecting the native length buckets."""
    if microbatch_size < 1:
        raise ValueError('Positive microbatch size required.')
    buckets = {}
    for ordinal in ordinals:
        buckets.setdefault(bucket_for(ordinal), []).append(ordinal)
    result = []
    for bucket, ids in sorted(buckets.items()):
        size = min(microbatch_size, (bucket_limits or {}).get(bucket, microbatch_size))
        if size < 1:
            raise ValueError('Bucket limits must be positive.')
        result.extend(ids[j:j+size] for j in range(0,len(ids),size))
    return result


def resize_stream_position(states, *, new_world, rank):
    """Preserve consumed examples, including uneven rank cursors.

    Continue at the next common new-world frontier, but queue every unconsumed
    hole before it. Neither replay a consumed row nor silently skip a warmup
    hole. Epochs and permutation seeds must still agree across saved ranks.
    """
    old_world = len(states)
    if (not states or new_world < 1 or not 0 <= rank < new_world
            or len({(s['epoch'], s['seed']) for s in states}) != 1
            or {s['rank'] for s in states} != set(range(old_world))
            or any(s['world'] != old_world or s['cursor'] < 0 for s in states)):
        raise ValueError('Resize requires one complete shared-epoch rank set.')
    by_rank = {s['rank']: s for s in states}
    frontier = max(s['cursor'] for s in states) * old_world
    frontier = (frontier + new_world - 1) // new_world * new_world
    pending = {p for s in states for p in s.get('pending_positions', [])}
    if sum(len(s.get('pending_positions', [])) for s in states) != len(pending):
        raise ValueError('Duplicated pending stream positions.')
    for position in range(min(s['cursor'] for s in states) * old_world, frontier):
        owner = by_rank[position % old_world]
        if position // old_world >= owner['cursor']:
            pending.add(position)
    result = dict(epoch=states[0]['epoch'], seed=states[0]['seed'],
                  cursor=frontier // new_world, rank=rank, world=new_world)
    holes = sorted(p for p in pending if p % new_world == rank)
    if holes:
        result['pending_positions'] = holes
    return result


class RequestInputs:
    columns = ('pair_ordinal', 'pair_id', 'operation', 'model_num_samples', 'latent_frames_valid',
               'latent_bucket_frames', 'source_latent_path', 'source_latent_key', 'source_latent_tensor_sha256')

    def __init__(self, record):
        self.db = readonly(record['native_index_path'])
        self.instructions = readonly(record['path'])

    def row(self, ordinal):
        values = self.db.execute('SELECT ' + ','.join(self.columns) + ' FROM pairs WHERE pair_ordinal=?', (ordinal,)).fetchone()
        if values is None:
            raise IndexError(ordinal)
        row = dict(zip(self.columns, values))
        instruction = self.instructions.execute('SELECT pair_id,raw_edit_request FROM instructions WHERE pair_ordinal=?', (ordinal,)).fetchone()
        if instruction is None or instruction[0] != row['pair_id']:
            raise ValueError('Source and instruction identity mismatch.')
        row['request'] = instruction[1]
        return row

    def observe(self, adapter, ordinal):
        row = self.row(ordinal)
        with safe_open(row['source_latent_path'], framework='pt', device='cpu') as handle:
            source = handle.get_tensor(row['source_latent_key'])
        if source.shape != (64, row['latent_frames_valid']) or hashlib.sha256(source.contiguous().numpy().tobytes()).hexdigest() != row['source_latent_tensor_sha256']:
            raise ValueError('Source tensor identity changed.')
        z = torch.zeros(1, 64, row['latent_bucket_frames'], device=adapter.device)
        z[0, :, :source.shape[-1]] = source.to(adapter.device)
        mask = torch.arange(z.shape[-1], device=adapter.device)[None] < source.shape[-1]
        return row, adapter.observe_editing(sample_id=row['pair_id'], request=row['request'], source_foa_latent=z,
            source_attention_mask=mask, model_num_samples=row['model_num_samples'])


NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)'
QUOTES = re.compile(r'(?P<cue>speech saying|voice described as|sound described as|music described as)\s*["“](?P<text>[^"”]+)["”]', re.I)


def request_facts(request, operation):
    """Parse only affirmative structured request fields, never removed content."""
    supported = {'event_addition', 'stationary_spatial_relocation', 'static_to_linear', 'linear_to_static'}
    if operation not in supported:
        return None
    fields, kind = {}, None
    for m in QUOTES.finditer(request):
        cue, value = m['cue'].lower(), m['text']
        if cue == 'voice described as':
            fields['speaker_description'] = value
        else:
            k = cue.split()[0]
            if kind is not None and k != kind:
                return None
            kind = k
            fields['transcript' if k == 'speech' else 'description'] = value
    if kind is None:
        return None
    az = [float(x) for x in re.findall(r'azimuth\s+(' + NUMBER + r')\s+degrees', request, re.I)]
    activity = None
    patterns = [r'(?:between|from)\s+(' + NUMBER + r')\s+(?:and|to)\s+(' + NUMBER + r')\s+seconds',
                r'(?:active interval to|active interval of)\s+(' + NUMBER + r')\s+(?:through|to)\s+(' + NUMBER + r')\s+seconds']
    if operation == 'event_addition':
        matches = [m for pattern in patterns for m in re.finditer(pattern, request, re.I)]
        if len(matches) == 1:
            activity = [float(matches[0][1]), float(matches[0][2])]
    return dict(kind=kind, fields=fields, azimuths=az if len(az) in (1, 2) else [], activity=activity, operation=operation)


def angular_distance(a, b):
    return abs((a - b + 180.) % 360. - 180.)


def propose_current_decision(adapter, observation, plan, tokens, facts, *, step):
    """One requested timing/direction alternative with its actual visited prefix.

    Both candidates preserve the same remaining fields. No old-prefix cache is
    accepted. Codec roundtrips and legal next-token support are checked.
    """
    if facts is None:
        return None
    sources = [s for s in plan['sources'] if s['kind'] == facts['kind']]
    if len(sources) != 1:
        return None
    s = sources[0]
    choices = []
    if len(facts['azimuths']) == 1 and s['trajectory']['type'] == 'static':
        current = s['trajectory']['position']['azimuth_deg']
        desired = facts['azimuths'][0]
        delta = (desired - current + 180.) % 360. - 180.
        # A nearby requested-equivalent or request-directed plan, not an
        # unconstrained high-score compensation on the opposite side.
        angle = current + (max(-10., min(10., delta)) if abs(delta) >= 1 else (5. if step % 2 else -5.))
        angle = (round(angle) + 180) % 360 - 180
        if angular_distance(angle, desired) <= max(30., angular_distance(current, desired)):
            p = copy.deepcopy(plan)
            next(x for x in p['sources'] if x['source_id'] == s['source_id'])['trajectory']['position']['azimuth_deg'] = angle
            choices.append(('azimuth', p))
    if facts['activity'] is not None:
        p = copy.deepcopy(plan)
        target = next(x for x in p['sources'] if x['source_id'] == s['source_id'])
        desired = min(facts['activity'][1], plan['duration_sec'])
        if desired > target['activity']['onset_sec']:
            target['activity']['offset_sec'] = desired
            choices.append(('offset', p))
    if not choices:
        return None
    from ...models.sceneplan_transfusion_editing_pipeline import _align_decoded_sceneplan_to_audio_duration
    current_ids = list(map(int, tokens))
    if adapter.codec.encode(plan)['input_ids'].tolist() != current_ids:
        raise ValueError('Proposal must start at the actual native decode.')
    for name, candidate in choices[step % len(choices):] + choices[:step % len(choices)]:
        ids = adapter.codec.encode(candidate)['input_ids'].tolist()
        candidate = _align_decoded_sceneplan_to_audio_duration(adapter.codec.decode(ids, sample_id=plan['sample_id']), observation.model_num_samples / 44100)
        ids = adapter.codec.encode(candidate)['input_ids'].tolist()
        if len(ids) != len(current_ids):
            continue
        changed = [i for i, (a, b) in enumerate(zip(current_ids, ids)) if a != b]
        if len(changed) != 1:
            continue
        pos = changed[0]
        legal = sorted(adapter.allowed_next_ids(observation, current_ids[:pos]))
        if ids[pos] not in legal:
            continue
        return dict(field=name, position=pos, prefix=current_ids[:pos], legal_ids=legal,
                    choice_ids=[current_ids[pos], ids[pos]], plans=[plan, candidate], source_id=s['source_id'])
    return None


@torch.no_grad()
def request_spatial_measure(wave, plan, facts):
    """Coarse horizontal evidence on non-overlapping plan activity windows.

    Window isolation is a planning-based cue, not a source separation claim.
    Unobservable windows incur a cost, instead of disappearing from the mean.
    """
    if not facts or len(facts['azimuths']) != 1:
        return dict(available=False)
    bound = [s for s in plan['sources'] if s['kind'] == facts['kind']]
    if len(bound) != 1 or bound[0]['trajectory']['type'] != 'static':
        return dict(available=False)
    target = bound[0]
    a, b = facts['activity'] or [target['activity']['onset_sec'], target['activity']['offset_sec']]
    ranges = [(a, b)]
    for other in plan['sources']:
        if other['source_id'] == target['source_id']:
            continue
        u, v = other['activity']['onset_sec'], other['activity']['offset_sec']
        ranges = [(x, y) for l, r in ranges for x, y in ((l, min(r, u)), (max(l, v), r)) if y - x >= .2]
    if not ranges:
        return dict(available=False)
    windows = []
    for l, r in ranges[:2]:
        for frac in (.2, .5, .8):
            center = l + (r - l) * frac
            i, j = max(0, round((center - .1) * 44100)), min(wave.shape[-1], round((center + .1) * 44100))
            w = wave[0, :, i:j].double()
            intensity = (w[0, None] * w[[3, 1, 2]]).sum(-1)
            denom = (w[0].square().sum() * w[[3, 1, 2]].square().sum()).sqrt().clamp_min(1e-20)
            coherence = float(intensity[:2].norm() / denom)
            valid = j > i and float(w[0].square().mean().sqrt()) > 1e-5 and coherence >= .1
            az = float(torch.rad2deg(torch.atan2(intensity[1], intensity[0]))) if valid else None
            err = angular_distance(az, facts['azimuths'][0]) if valid else 180.
            windows.append(dict(observable=valid, angle_error_deg=err, azimuth_deg=az, interval=[i, j]))
    return dict(available=True, windows=windows, mean_capped_angle_deg=sum(min(90., w['angle_error_deg']) for w in windows)/len(windows),
                failures=sum(w['angle_error_deg'] > 30. for w in windows), unobservable=sum(not w['observable'] for w in windows))
