"""Matched validation-only free plans with the existing decoder and scorers."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
OPS = Path('/mnt/sdb/audio_dataset/sceneplan_transfusion_editing_v1/materialized/logs/ar_clap_task_20260906')
sys.path[:0] = [str(ROOT), str(OPS / 'diagnostics'), str(OPS)]
import torch
from torch import distributed as dist
from torch.utils.data import DataLoader
import ar_requested_edit_audit as requested
from scripts.t2a.experiments.ar_factual_clap_v1 import integration
from scripts.t2a.experiments.ar_instruction_t200_v1 import runtime as base
from scripts.t2a.experiments.ar_instruction_t200_v1.fingerprints import fingerprint
from scripts.t2a.experiments.ar_instruction_t200_v1.instruction_data import InstructionOverlay, DatasetWithInstructions
from scripts.t2a.experiments.ar_instruction_t200_v3 import policy
from scripts.t2a.experiments.ar_source_grounding_v1.evidence import sha, state_hash, write
from scripts.t2a.eval.select_sceneplan_transfusion_editing_joint_checkpoint import _JointDonorResolver
from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import ScenePlanTransfusionEditingDataset
from stable_audio_tools.data.sceneplan_transfusion_editing_joint_dataset import ScenePlanTransfusionEditingJointDataset, collate_editing_joint
from stable_audio_tools.training.sceneplan_transfusion_editing_clap44_selection import evaluate_ar, free_ar_pass


class View:
    def __init__(self, dataset, indices, codec):
        self.dataset, self.indices, self.codec = dataset, indices, codec
        original = dataset.length_bucket_indices()
        buckets = {i: bucket for bucket, values in original.items() for i in values}
        self.buckets = {bucket: [j for j, i in enumerate(indices) if buckets[i] == bucket] for bucket in (432, 648)}
    def __len__(self): return len(self.indices)
    def __getitem__(self, i): return self.dataset[self.indices[i]]
    def length_bucket_indices(self): return self.buckets


def serial(value):
    if isinstance(value, torch.Tensor): return value.cpu().tolist()
    if isinstance(value, dict): return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [serial(v) for v in value]
    return value


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = policy.read(args.plan)
    for path, expected in plan['source_sha256'].items():
        assert sha(path) == expected, path
    protocol = policy.read(plan['protocol'])
    assert sha(plan['protocol']) == plan['protocol_sha256']
    rank, _, world, device, topology = base.allocated_runtime.distributed(timeout_seconds=1800)
    assert topology['physical_indices'] == [5, 6, 7] and world == 3
    out = args.plan.parent / f'rank{rank}'
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    finish_numerics = base.numerics.install_autotune_observer({'name': 'autotune', 'mode': 'capture'}, out)
    # Every arm in this process uses the same cached native FLA configurations.
    cfg = policy.read(plan['training_config'])
    flash, restore_flash = base.install_flash(cfg['numerical_execution'])
    parent_ref = cfg['initial_state_transfer']['parent_checkpoint']
    module, codec, parent, _, transfer = integration.build_from_ar_parent(parent_ref,
        cfg['clap_dependency']['checkpoint'], preflight_path=cfg['clap_dependency']['preflight'])
    parent_contract = parent['run_contract']
    del parent
    module.to(device).eval().requires_grad_(False)
    truth = {item['pair_ordinal']: item for item in policy.read(protocol['truth']['path'])['cases']}
    assert sha(protocol['truth']['path']) == protocol['truth']['sha256']
    cases = protocol['population']['cases']
    ordinals = [case['pair_ordinal'] for case in cases]
    assert len(ordinals) == len(set(ordinals)) == 100 and set(truth) == set(ordinals)
    val = parent_contract['indices']['validation']
    dataset = ScenePlanTransfusionEditingDataset(val['path'],
        tokenizer_spec=(module.diffusion.conditioner.conditioners['prompt'].tokenizer, 512),
        expected_num_samples=100, index_num_samples=20000, expected_index_sha256=val['sha256'],
        sample_ordinals=ordinals, latent_crop_length=648, require_frozen=True, verify_tensor_hashes_on_access=True)
    dataset = ScenePlanTransfusionEditingJointDataset(dataset, codec=codec)
    binding = cfg['instruction_data']['overlays']['validation']
    overlay = InstructionOverlay(binding['path'], expected_sha256=binding['sha256'],
        native_index_path=binding['native_index_path'], native_index_sha256=binding['native_index_sha256'],
        expected_rows=20000, split='validation')
    dataset = DatasetWithInstructions(dataset, overlay, joint=True)
    local = View(dataset, list(range(rank, len(dataset), world)), codec)
    batches = [values[i:i + 4] for values in local.length_bucket_indices().values() for i in range(0, len(values), 4)]
    loader = DataLoader(local, batch_sampler=batches, num_workers=0,
        collate_fn=lambda samples: collate_editing_joint(samples, pad_id=codec.pad_id),
        generator=torch.Generator().manual_seed(42))
    donors = _JointDonorResolver(dataset, ordinals, {int(k): v for k, v in protocol['population']['donors'].items()})
    seen_encoder = cfg['clap_dependency']['checkpoint']
    started = time.monotonic()
    results = {}
    try:
        for arm in plan['arms']:
            name, reference, encoder = arm['name'], arm['AR_checkpoint'], arm['encoder']
            if encoder != seen_encoder:
                loaded, _, _ = integration.load_source_encoder(encoder,
                    preflight_path=cfg['clap_dependency']['preflight'], device=device)
                module.ar.source_clap_model = loaded
                seen_encoder = encoder
            payload, identity = integration.load_joint_checkpoint(reference['checkpoint'], verify_sources=True, require_latest=False)
            assert identity == reference
            if reference != parent_ref:
                current = payload['run_contract']
                policy.check_selection(current['config'])
                assert current['clap_checkpoint'] == {k: encoder[k] for k in ('path', 'sha256', 'step')}
                assert reference['step'] == parent_ref['step'] + 250
            integration.restore_ar_weights(module, payload)
            del payload
            module.eval().requires_grad_(False)
            before = {'module': state_hash(module), 'qwen': state_hash(module.ar.instruction_conditioner.model)}
            raw = evaluate_ar(module.ar, loader, device=device,
                variants=('clean', 'zero', 'shuffled', 'clap_zero', 'clap_shuffled'), donor_resolver=donors)
            assert len(raw['ordinals']) == len(local)
            assert all(torch.isfinite(v).all() for v in raw['losses'].values())
            write(out / name / 'TEACHER.json', serial(raw))
            records = []
            # Keep the native free decoder, and write progress after each actual batch.
            for indices in batches:
                batch_view = View(local, indices, codec)
                generated = free_ar_pass(module.ar, codec, batch_view, device=device, batch_size=4)
                for record in generated:
                    ordinal = record['pair_ordinal']
                    assert record['target_token_ids'] == truth[ordinal]['target_token_ids']
                    write(out / name / 'free_plan_records' / f'{ordinal:05d}.json', record)
                    strict = requested.score(record.get('predicted_sceneplan'), record['status'], truth[ordinal])
                    annotation = {'canonical_target_source_roles': {s['source_id']: 'edited' if s['source_id'] == truth[ordinal]['canonical_edited_source_id'] else 'unchanged' for s in truth[ordinal]['target_plan']['sources']},
                        'raw_edit_request': 'T200 request; identity stored in training overlay binding', 'removed_source_ids_not_in_target': []}
                    endpoint = requested.endpoints.describe(record, truth[ordinal]['target_plan'], annotation)
                    write(out / name / 'scored' / f'{ordinal:05d}.json', {'native': record, 'requested': strict, 'source_fields': endpoint})
                    records.append(ordinal)
                print(json.dumps({'rank': rank, 'arm': name, 'free_plans': len(records), 'elapsed_seconds': time.monotonic() - started}), flush=True)
            assert len(records) == len(set(records)) == len(local)
            after = {'module': state_hash(module), 'qwen': state_hash(module.ar.instruction_conditioner.model)}
            assert before == after
            results[name] = {'checkpoint': identity, 'encoder': encoder, 'cases': len(records),
                'model_and_frozen_Qwen_unchanged': True, 'weights_sha256': before,
                'mean_teacher_CE': {k: float(v.mean()) for k, v in raw['losses'].items()}}
        finish_numerics()
        write(out / 'COMPLETE.json', {'at': datetime.now().astimezone().isoformat(), 'arms': results,
            'physical_gpus': [5, 6, 7], 'rank': rank, 'world': world, 'flash_calls': flash,
            'same_native_autotune_configs_for_all_arms_in_process': True,
            'independent_test_used': False, 'AR_quality_gate_passed': False})
        dist.barrier()
    finally:
        restore_flash()
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__': main()
