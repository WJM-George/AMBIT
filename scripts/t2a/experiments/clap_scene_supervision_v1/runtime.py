"""Bounded two-rank CLAP event training, with complete native-parent provenance."""
import argparse
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parents[4]
OPS = Path('/mnt/sdb/audio_dataset/sceneplan_transfusion_editing_v1/materialized/logs/ar_clap_task_20260906')
OLD = OPS.parent / 'takeover-20260905T095122+0800'
sys.path[:0] = [str(REPO), str(OPS), str(OLD)]

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import editing_allocated_gpu_runtime_v1 as allocated
import probe_clap44_fla_autotune_replay_v1 as kernels
from diagnose_clap44_objective_gradients_v1 import state_hash
from scripts.t2a.eval.select_sceneplan_transfusion_editing_dit_checkpoint import _rank0_audit
from scripts.t2a.train.train_sceneplan_transfusion_editing_clap44 import rng_state, restore_rng
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_text import FrozenCLAP44TextFeatures
from stable_audio_tools.training.sceneplan_transfusion_editing_clap44_pretrain import load_training_checkpoint
from scripts.t2a.experiments.clap_scene_supervision_v1.cursor import NativeResumeBatches
from scripts.t2a.experiments.clap_scene_supervision_v1.data import CLAP44SceneSupervisionDataset, collate_scene_supervision
from scripts.t2a.experiments.clap_scene_supervision_v1.training import SceneEventTrainingModel, build_optimizers, training_step
from scripts.t2a.experiments.clap_scene_supervision_v1 import state


def run(plan_path, phase_name):
    plan = state.read(plan_path); phase = plan['phases'][phase_name]
    contract = state.read(phase['contract'])
    if plan['physical_gpus'] != [5, 6] or contract['physical_gpus'] != [5, 6]:
        raise RuntimeError('This stage may only use physical GPU5–6')
    if not 1 <= contract['max_new_updates'] <= 250 or not 0 < phase['stop_after_new_updates'] <= contract['max_new_updates']:
        raise RuntimeError('This entry supports bounded development, not unreviewed full training')
    if contract['method']['text_view'] != 'native' or contract['data']['rows'] != 1000000:
        raise RuntimeError('First event-loss comparison must retain native facts and full train index')
    for source, expected in {**contract['source_sha256'], **plan['source_sha256']}.items():
        if state.sha(source) != expected:
            raise RuntimeError(f'Bound source identity changed: {source}')
    kernels.autotune_tree()
    if torch.cuda.is_initialized():
        raise RuntimeError('GPU visibility must be established by the allocated launcher first')
    rank, local, world, device, topology = allocated.distributed(timeout_seconds=600)
    if world != contract['world_size'] or topology != contract['gpu_topology']:
        raise RuntimeError('Actual GPU rank topology differs from this training contract')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8' or os.environ.get('FLA_CACHE_MODE') != 'disabled':
        raise RuntimeError('Numerical runtime contract is absent')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    seed = contract['data']['seed']
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed + rank)
    parent_path = Path(contract['parent']['path'])
    native_parent_contract = state.read(parent_path.parent / 'TRAIN_CONTRACT.json')
    parent, parent_identity = load_training_checkpoint(parent_path, contract=native_parent_contract)
    if parent_identity['sha256'] != contract['parent']['sha256'] or native_parent_contract['gpu_topology'] != topology:
        raise RuntimeError('Native parent or rank ownership changed')
    if {key: parent[key] for key in ('step', 'epoch', 'next_batch')} != {key: contract['parent'][key] for key in ('step', 'epoch', 'next_batch')}:
        raise RuntimeError('Parent data/step provenance differs')
    if parent['step'] < native_parent_contract['config']['training']['max_steps']:
        raise RuntimeError('This fork only implements the verified constant native tail schedule')
    artifacts = Path(phase['artifacts']); artifacts.mkdir(parents=True, exist_ok=True)
    output = Path(phase['output'])

    def audit_output():
        output.mkdir(parents=True, exist_ok=True)
        path = output / 'TRAIN_CONTRACT.json'
        if path.exists():
            if state.read(path) != contract:
                raise RuntimeError('Output belongs to another experiment')
        else:
            state.write(path, contract)
        latest = state.resolve_resume(output, contract)
        if phase.get('resume'):
            if latest is None or latest['checkpoint'] != str(Path(phase['resume']).resolve(strict=True)):
                raise RuntimeError('Only the verified latest complete state may resume in place')
        elif latest is not None:
            raise RuntimeError('Existing run requires explicit resume; never duplicate training')
        for source, expected in native_parent_contract['text_files'].items():
            if state.sha(source) != expected:
                raise RuntimeError(f'Frozen Qwen asset changed: {source}')
        for source, expected in native_parent_contract['frontend_files'].items():
            if state.sha(source) != expected:
                raise RuntimeError(f'FOA frontend asset changed: {source}')
        if state.sha(contract['data']['index']) != contract['data']['index_sha256']:
            raise RuntimeError('Full train index identity changed')
        return True
    _rank0_audit(audit_output, rank=rank, device=device)
    model = SceneEventTrainingModel(contract['native_config']['model'], contract['readout'], contract['readout_seed'])
    model.encoder.load_state_dict(parent['model'], strict=True)
    model.to(device).train()
    optimizers, schedulers = build_optimizers(model, contract, parent)
    updates = 0; rank_rngs = parent['rng_states']
    if phase.get('resume'):
        payload, _ = state.load_checkpoint(phase['resume'], contract)
        for key in optimizers:
            getattr(model, key).load_state_dict(payload[key]['model'], strict=True)
            optimizers[key].load_state_dict(payload[key]['optimizer'])
            schedulers[key].load_state_dict(payload[key]['scheduler'])
        updates = payload['new_updates']; rank_rngs = payload['rng_states']
        del payload
    if updates >= phase['stop_after_new_updates']:
        raise RuntimeError('Phase has no remaining updates; do not relaunch a completed phase')
    initial = state.make_payload(model, optimizers, schedulers, updates, contract, rank_rngs)
    state.validate_payload(initial, contract)
    state.write(artifacts / f'INITIAL_rank{rank}.json', {
        'new_updates': updates, 'encoder': state.fingerprint(initial['encoder']),
        'readout': state.fingerprint(initial['readout']), 'rank_rng': state.fingerprint(rank_rngs[rank]),
        'epoch': initial['epoch'], 'next_batch': initial['next_batch'],
        'parent_initial_encoder_exact': state.fingerprint(initial['encoder']) == state.fingerprint({k: parent[k] for k in ('model', 'optimizer', 'scheduler')}) if updates == 0 else None,
    })
    del initial, parent
    model = DistributedDataParallel(model, device_ids=[local])
    teacher_cfg = contract['native_config']['text']
    teacher = FrozenCLAP44TextFeatures(teacher_cfg['model_path'], hidden_dim=contract['native_config']['model']['text_dim'],
                                     max_tokens=teacher_cfg['max_tokens'], batch_size=teacher_cfg['batch_size']).eval()
    if state_hash(teacher.conditioner.model) != contract['qwen_state_sha256']:
        raise RuntimeError('Frozen Qwen initial tensor identity differs')
    mode = {'name': f'rank{rank}', 'mode': phase['kernel_mode']}
    if mode['mode'] == 'pin':
        mode['reference_case'] = str(Path(phase['kernel_reference']) / f'rank{rank}')
    finish_catalog = kernels.install_autotune_observer(mode, artifacts)
    dataset = CLAP44SceneSupervisionDataset(contract['data']['index'], expected_rows=1000000,
        text_view='native', verify_tensor_hashes=True)
    if dataset.split != 'train' or dataset.index_sha256 != contract['data']['index_sha256']:
        raise RuntimeError('Dataset split or frozen marker changed')
    resume_rng = rank_rngs[rank]
    began = time.monotonic(); complete_identity = None
    while updates < phase['stop_after_new_updates']:
        epoch, offset = state.expected_cursor(contract, updates)
        batches = NativeResumeBatches(dataset, rank=rank, world=world, seed=seed,
            pairs_per_rank=contract['data']['pairs_per_rank'], epoch=epoch, next_batch=offset,
            limit=phase['stop_after_new_updates'] - updates)
        generator = torch.Generator().manual_seed(seed + 900001 + epoch * world + rank)
        loader = DataLoader(dataset, batch_sampler=batches, num_workers=contract['data']['workers_per_rank'],
            pin_memory=True, collate_fn=collate_scene_supervision, generator=generator,
            multiprocessing_context='spawn' if contract['data']['workers_per_rank'] else None)
        for batch in loader:
            if resume_rng is not None:
                restore_rng(resume_rng); resume_rng = None
            gradients = {}
            def observe(core):
                if plan['record_full_gradients']:
                    gradients.update({key: state.fingerprint({n: p.grad for n, p in getattr(core, key).named_parameters()}) for key in ('encoder', 'readout')})
            values = training_step(model, teacher, optimizers, schedulers, batch,
                contract=contract, device=device, observer=observe)
            updates += 1
            cursor = state.expected_cursor(contract, updates)
            rng = rng_state()
            record = {'new_updates': updates, 'step': contract['parent']['step'] + updates,
                'next_epoch': cursor[0], 'next_batch': cursor[1], 'metrics': values,
                'batch_pair_ids': [row['pair_id'] for row in batch['labels']],
                'batch_roles': [row['role'] for row in batch['labels']],
                'batch_fingerprint': state.fingerprint({k: batch[k] for k in ('latent', 'mask', 'labels', 'negative_scene_texts', 'negative_owners', 'event_targets', 'event_metadata')}),
                'gradients': gradients, 'rank_rng_after': state.fingerprint(rng),
                'encoder_state': state.fingerprint(model.module.encoder.state_dict()),
                'readout_state': state.fingerprint(model.module.readout.state_dict()),
                'quality_gate_passed': False}
            state.write(artifacts / f'update{updates:04d}_rank{rank}.json', record)
            if rank == 0:
                state.write(artifacts / 'PROGRESS.json', {'step': record['step'], 'new_updates': updates,
                    'elapsed_seconds': time.monotonic() - began}, replace=True)
                print(json.dumps({'phase': phase_name, 'new_updates': updates, 'global': values['global']}), flush=True)
            if updates == phase['stop_after_new_updates'] or updates % contract['checkpoint_every_new_updates'] == 0:
                rank_states = [None] * world; dist.all_gather_object(rank_states, rng)
                payload = state.make_payload(model.module, optimizers, schedulers, updates, contract, rank_states)
                state.validate_payload(payload, contract)
                signature = state.fingerprint(payload)
                signatures = [None] * world; dist.all_gather_object(signatures, signature)
                if any(x != signature for x in signatures):
                    raise RuntimeError('DDP ranks disagree on complete model/optimizer/scheduler states')
                complete_identity = _rank0_audit(lambda: state.save_checkpoint(output / f'step-{payload["step"]:06d}.pt', payload, contract), rank=rank, device=device)
                state.write(artifacts / f'SAVED{updates:04d}_rank{rank}.json', {'identity': complete_identity, 'full_state_fingerprint': signature})
                del payload
        del loader
    finish_catalog()
    if state_hash(teacher.conditioner.model) != contract['qwen_state_sha256']:
        raise RuntimeError('Frozen Qwen changed during training')
    state.write(artifacts / f'COMPLETE_rank{rank}.json', {
        'at': datetime.now().astimezone().isoformat(), 'phase': phase_name, 'new_updates': updates,
        'checkpoint': complete_identity, 'qwen_state_unchanged': True,
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'transformers', 'triton')},
        'quality_gate_passed': False, 'independent_test_used': False, 'notifications_enabled': False})
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--phase', required=True)
    args = parser.parse_args(); run(args.plan.resolve(strict=True), args.phase)
