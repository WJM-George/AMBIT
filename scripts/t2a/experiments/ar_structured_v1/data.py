"""Keep source/target supervision separate from the audio/request model API."""
import json
import sqlite3
import zlib

import torch

from scripts.t2a.experiments.clap_scene_supervision_v1.data import structured_targets, ordered_events
from scripts.t2a.experiments.clap_scene_supervision_v1.events import DEFAULT_WEIGHTS
from scripts.t2a.experiments.ar_instruction_t200_v1.instruction_data import (
    InstructionOverlay, DatasetWithInstructions,
)
from stable_audio_tools.data.sceneplan_transfusion_editing import sha256_json
from stable_audio_tools.data.sceneplan_transfusion_editing_ar_dataset import (
    ScenePlanTransfusionEditingARDataset, collate_editing_ar,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_joint_dataset import collate_editing_joint
from stable_audio_tools.models.sceneplan_editing_gain_adapter import attach_desired_gain_tracks

OPERATION_ACTION = {'event_removal': 2, 'stationary_spatial_relocation': 3,
    'static_to_linear': 4, 'linear_to_static': 5, 'event_addition': 6}
FIELDS = tuple(DEFAULT_WEIGHTS)
LABEL_COLUMNS = ('pair_id', 'operation', 'old_sceneplan_zlib', 'new_sceneplan_zlib',
    'old_sceneplan_sha256', 'new_sceneplan_sha256', 'edited_source_ids_json', 'unchanged_source_ids_json')


def labels_from_plans(old, new, operation, edited_ids, unchanged_ids):
    source_meta = structured_targets(old)
    target_meta = structured_targets(new)
    source_targets = source_meta['tensors']
    target_targets = target_meta['tensors']
    old_events, new_events = ordered_events(old), ordered_events(new)
    old_ids = [s['source_id'] for s in old_events]
    new_ids = [s['source_id'] for s in new_events]
    edited_ids, unchanged_ids = set(edited_ids), set(unchanged_ids)
    if len(edited_ids) != 1 or operation not in OPERATION_ACTION:
        raise ValueError('This recipe requires one of the five single-object editing operations')
    edited_id = next(iter(edited_ids))
    added = set(new_ids) - set(old_ids)
    removed = set(old_ids) - set(new_ids)
    if operation == 'event_addition':
        if added != edited_ids or removed or unchanged_ids != set(old_ids):
            raise ValueError('Addition source identity contract changed')
        target_label = 4
    else:
        if added or edited_id not in old_ids or unchanged_ids != set(old_ids) - edited_ids:
            raise ValueError('Existing-source edit identity contract changed')
        if removed != (edited_ids if operation == 'event_removal' else set()):
            raise ValueError('Removed-source identity contract changed')
        target_label = old_ids.index(edited_id)
    source_actions = torch.zeros(5, dtype=torch.long)
    source_actions[:len(old_ids)] = 1
    source_actions[target_label] = OPERATION_ACTION[operation]
    label_ids = old_ids + [None] * (4 - len(old_ids)) + [edited_id if added else None]
    post = {}
    for key, value in target_targets.items():
        if key == 'source_count':
            post[key] = value.clone()
        else:
            post[key] = torch.zeros((5, *value.shape[1:]), dtype=value.dtype)
    content_texts = [''] * 5
    preserve = torch.zeros(5, len(FIELDS), dtype=torch.bool)
    for label, source_id in enumerate(label_ids):
        if source_id not in new_ids:
            continue
        j = new_ids.index(source_id)
        for key in post:
            if key != 'source_count':
                post[key][label] = target_targets[key][j]
        content_texts[label] = target_meta['content_texts'][j]
        if source_id not in old_ids:
            continue
        i = old_ids.index(source_id)
        same = {
            'kind': torch.equal(source_targets['kind'][i], target_targets['kind'][j]),
            'content': source_meta['content_keys'][i] == target_meta['content_keys'][j],
            'motion': torch.equal(source_targets['motion'][i], target_targets['motion'][j]),
            'frame_count': torch.equal(source_targets['keyframe_mask'][i], target_targets['keyframe_mask'][j]),
            'activity': torch.equal(source_targets['activity_sec'][i], target_targets['activity_sec'][j]),
            'frame_time': torch.equal(source_targets['keyframe_time_sec'][i], target_targets['keyframe_time_sec'][j]),
            'direction': torch.equal(source_targets['direction_xyz'][i], target_targets['direction_xyz'][j]),
            'log_distance': torch.equal(source_targets['position_spherical'][i, :, 2], target_targets['position_spherical'][j, :, 2]),
            'gain': torch.equal(source_targets['gain_db'][i], target_targets['gain_db'][j]),
        }
        preserve[label] = torch.tensor([same[k] for k in FIELDS])
        if source_id in unchanged_ids and not bool(preserve[label].all()):
            raise ValueError('Unchanged event labels contain an actual field change')
    return {'source_targets': source_targets, 'post_targets': post,
        'source_content_texts': source_meta['content_texts'], 'post_content_texts': content_texts,
        'edit_target': torch.tensor(target_label), 'actions': source_actions,
        'preserve_fields': preserve, 'operation': operation}


class StructuredDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, index_path, *, joint=False):
        self.dataset = dataset
        self.index_path = str(index_path)
        self.joint = bool(joint)
        self._connection = None
        self._buckets = None

    def __len__(self):
        return len(self.dataset)

    def _db(self):
        if self._connection is None:
            self._connection = sqlite3.connect(f'file:{self.index_path}?mode=ro&immutable=1', uri=True)
            self._connection.execute('PRAGMA query_only=ON')
        return self._connection

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_connection'] = None
        return state

    def __del__(self):
        db = getattr(self, '_connection', None)
        if db is not None:
            db.close()

    def length_bucket_indices(self):
        if self._buckets is None:
            buckets = {432: [], 648: []}
            for ordinal, bucket in self._db().execute('SELECT pair_ordinal,latent_bucket_frames FROM pairs ORDER BY pair_ordinal'):
                buckets[int(bucket)].append(int(ordinal))
            if sum(map(len, buckets.values())) != len(self):
                raise ValueError('Structured sampler requires the complete declared split')
            self._buckets = {k: tuple(v) for k, v in buckets.items()}
        return self._buckets

    def __getitem__(self, index):
        value = self.dataset[index]
        ar = value[2] if self.joint else value
        values = self._db().execute('SELECT ' + ','.join(LABEL_COLUMNS)
            + ' FROM pairs WHERE pair_ordinal=?', (ar['pair_ordinal'],)).fetchone()
        row = dict(zip(LABEL_COLUMNS, values))
        if row['pair_id'] != ar['pair_id'] or row['operation'] != ar['operation']:
            raise ValueError('Structured labels belong to a different example')
        old = json.loads(zlib.decompress(row['old_sceneplan_zlib']))
        new = json.loads(zlib.decompress(row['new_sceneplan_zlib']))
        if sha256_json(old) != row['old_sceneplan_sha256'] or sha256_json(new) != row['new_sceneplan_sha256']:
            raise ValueError('Structured supervision hash changed')
        labels = labels_from_plans(old, new, row['operation'],
            json.loads(row['edited_source_ids_json']), json.loads(row['unchanged_source_ids_json']))
        return {'model_row': value, 'supervision': labels}


def make_source_only_dataset(record, codec, split):
    base = ScenePlanTransfusionEditingARDataset(record['native_index_path'], codec=codec,
        expected_num_samples=record['rows'], expected_index_sha256=record['native_index_sha256'],
        verify_tensor_hashes_on_access=True)
    overlay = InstructionOverlay(record['path'], expected_sha256=record['sha256'],
        native_index_path=record['native_index_path'], native_index_sha256=record['native_index_sha256'],
        expected_rows=record['rows'], split=split)
    return StructuredDataset(DatasetWithInstructions(base, overlay, joint=False), record['native_index_path'])


def collate(rows, *, pad_id, joint=False):
    raw = [row['model_row'] for row in rows]
    batch = collate_editing_joint(raw, pad_id=pad_id) if joint else {'ar': collate_editing_ar(raw, pad_id=pad_id)}
    if joint:
        # Native collation trims432-frame batches from648-frame storage.
        # Derive RF gain tracks afterward so they share that exact time axis.
        batch['metadata'] = [attach_desired_gain_tracks(row) for row in batch['metadata']]
    labels = [row['supervision'] for row in rows]
    result = {}
    for key in labels[0]:
        values = [row[key] for row in labels]
        if isinstance(values[0], dict):
            result[key] = {k: torch.stack([v[k] for v in values]) for k in values[0]}
        elif isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    batch['supervision'] = result
    return batch


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value
