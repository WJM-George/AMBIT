"""Source identity, edit supervision and pretrained readout invariants."""
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sqlite3
import sys
import zlib

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from scripts.t2a.experiments.ar_structured_v1 import data, model, losses
from scripts.t2a.experiments.ar_source_grounding_v1.evidence import write
from scripts.t2a.experiments.clap_scene_supervision_v1.events import AudioEventReadout


def stack(labels):
    result = {}
    for key in labels[0]:
        values = [row[key] for row in labels]
        if isinstance(values[0], dict):
            result[key] = {k: torch.stack([v[k] for v in values]) for k in values[0]}
        elif isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result


def content(rows, slots):
    output = torch.zeros(len(rows), slots, 1024)
    for i, row in enumerate(rows):
        for j, text in enumerate(row):
            if text:
                seed = int(hashlib.sha256(text.encode()).hexdigest()[:14], 16)
                output[i, j] = torch.randn(1024, generator=torch.Generator().manual_seed(seed))
    return output


def perfect(targets, text):
    def logits(labels, classes):
        return F.one_hot(labels.long(), classes).float() * 16 - 8
    return {'kind_logits': logits(targets['kind'], 4),
        'motion_logits': logits(targets['motion'].clamp_min(0), 3),
        'frame_count_logits': logits((targets['keyframe_mask'].sum(-1)-1).clamp_min(0), 8),
        'content': text, 'activity_sec': targets['activity_sec'].clone(),
        'gain_db': targets['gain_db'].clone(), 'keyframe_time_sec': targets['keyframe_time_sec'].clone(),
        'direction_xyz': targets['direction_xyz'].clone(),
        'log_distance': targets['position_spherical'][..., 2].clamp_min(1e-12).log()}


def run(args):
    torch.set_num_threads(2); torch.manual_seed(42)
    cfg = json.loads(args.config.read_text())
    db = sqlite3.connect('file:'+cfg['data']['train']['native_index_path']+'?mode=ro&immutable=1', uri=True)
    db.row_factory = sqlite3.Row
    labels = []
    for operation in data.OPERATION_ACTION:
        rows = db.execute('SELECT * FROM pairs WHERE operation=? LIMIT 100', (operation,)).fetchall()
        assert len(rows) == 100
        for row in rows:
            labels.append(data.labels_from_plans(json.loads(zlib.decompress(row['old_sceneplan_zlib'])),
                json.loads(zlib.decompress(row['new_sceneplan_zlib'])), operation,
                json.loads(row['edited_source_ids_json']), json.loads(row['unchanged_source_ids_json'])))
    supervision = stack(labels)
    source_content = content(supervision['source_content_texts'], 4)
    post_content = content(supervision['post_content_texts'], 5)
    prediction = {'source': perfect(supervision['source_targets'], source_content),
        'target': perfect(supervision['post_targets'], post_content),
        'target_logits': F.one_hot(supervision['edit_target'], 5).float()*16-8,
        'operation_logits': F.one_hot(supervision['actions'], 7).float()*16-8}
    base = losses.structured_objective(prediction, supervision, source_content, post_content)
    assert torch.equal(base['edit_object_slots'], supervision['edit_target'])
    permutation = torch.tensor([2, 0, 3, 1]); five = torch.cat((permutation, torch.tensor([4])))
    permuted = {'source': {k: v[:, permutation] for k, v in prediction['source'].items()},
        'target': {k: v[:, five] for k, v in prediction['target'].items()},
        'target_logits': prediction['target_logits'][:, five],
        'operation_logits': prediction['operation_logits'][:, five]}
    rotated = losses.structured_objective(permuted, supervision, source_content, post_content)
    torch.testing.assert_close(base['per_example'], rotated['per_example'], atol=1e-6, rtol=1e-6)
    broken = copy.deepcopy(prediction)
    broken['target'] = {k: v[:, five] for k, v in prediction['target'].items()}
    mismatch = losses.structured_objective(broken, supervision, source_content, post_content)
    assert mismatch['loss_sum'] > base['loss_sum'] + 1
    removal = torch.tensor([x['operation']=='event_removal' for x in labels])
    assert bool((base['post_kind_by_predicted_slot'][removal].gather(1,
        base['edit_object_slots'][removal, None]) == 0).all())
    addition = torch.tensor([x['operation']=='event_addition' for x in labels])
    assert bool((base['edit_object_slots'][addition] == 4).all())
    checkpoint = cfg['base_AR_configuration']['clap_dependency']['checkpoint']['path']
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    readout = AudioEventReadout(**payload['contract']['readout'])
    readout.load_state_dict(payload['readout']['model'], strict=True)
    structure = model.SourceStructure(readout).eval()
    sequence = torch.randn(2, 13, 384); mask = torch.ones(2, 13, dtype=torch.bool)
    mask[1, 9:] = False
    with torch.no_grad():
        expected = readout(sequence, mask)
        actual = structure(sequence, mask, torch.randn(2, 11, 1024), torch.ones(2,11,dtype=torch.bool))
        assert all(torch.equal(expected[k], actual['source'][k]) for k in expected)
        residual = model.SlotResidual()
        hidden = torch.randn(2, 17, 1024)
        assert torch.equal(residual(hidden, actual['memory']), hidden)
    assert not {'metadata', 'old_sceneplan', 'target_sceneplan', 'supervision'} & set(inspect.signature(model.ar_forward).parameters)
    assert set(inspect.signature(model.SourceStructure.forward).parameters) == {
        'self','sequence','sequence_mask','instruction_context','instruction_mask'}
    result = {'label_examples': len(labels), 'operations': list(data.OPERATION_ACTION),
        'consistent_slot_permutation_invariant': True, 'incorrect_post_edit_identity_penalized': True,
        'removal_retains_old_slot_with_EMPTY_target': True, 'addition_uses_independent_NEW_slot': True,
        'paired_CLAP10k_source_readout_exact': True, 'zero_residual_preserves_decoder_exactly': True,
        'AR_and_source_slot_API_excludes_ground_truth': True,
        'perfect_fixture_loss': float(base['loss_sum']), 'broken_identity_fixture_loss': float(mismatch['loss_sum']),
        'fixture_content_vectors_are_synthetic_only_for_CPU_test': True, 'quality_gate_passed': False}
    write(args.output, result); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
