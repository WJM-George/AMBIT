"""Actual GPU5-7 factual CLAP training from weights, with fresh full states."""
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
sys.path[:0] = [str(REPO), str(OPS), str(OPS / 'methods'), str(OLD)]
import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import editing_allocated_gpu_runtime_v1 as allocated
import probe_clap44_fla_autotune_replay_v1 as kernels
from diagnose_clap44_objective_gradients_v1 import state_hash
from clap_event_native_balance import compiled_training_step
from scripts.t2a.eval.select_sceneplan_transfusion_editing_dit_checkpoint import _rank0_audit
from scripts.t2a.train.train_sceneplan_transfusion_editing_clap44 import rng_state, restore_rng
from scripts.t2a.experiments.clap_scene_supervision_v1.training import SceneEventTrainingModel
from scripts.t2a.experiments.clap_scene_supervision_v1.cursor import NativeResumeBatches
from scripts.t2a.experiments.clap_scene_supervision_v1 import kernel_continuation
from scripts.t2a.experiments.clap_factual50k_v1 import checkpoint as state
from scripts.t2a.experiments.clap_factual50k_v1.data import FactualDataset, collate_scene_supervision, CATALOG_SHA256
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_text import FrozenCLAP44TextFeatures
from stable_audio_tools.training.sceneplan_transfusion_editing_clap44_pretrain import load_training_checkpoint


def run(plan_path, phase_name):
    plan = state.read(plan_path); phase = plan['phases'][phase_name]
    contract = state.read(phase['contract'])
    if plan['physical_gpus'] != [5, 6, 7] or contract['schema'] != state.SCHEMA or contract['max_new_updates'] != 50000:
        raise RuntimeError('Wrong authorized allocation or fresh training lineage')
    if contract['data']['rows'] != 1000000 or contract['data']['catalog_sha256'] != CATALOG_SHA256:
        raise RuntimeError('Wrong full factual-scene population')
    if contract['method']['text_view'] != 'factual_templates200' or not 0 < phase['stop_after_new_updates'] <= 50000:
        raise RuntimeError('Factual text view and update budget are mandatory')
    for path, digest in {**contract['source_sha256'], **plan['source_sha256']}.items():
        if state.sha(path) != digest: raise RuntimeError(('Bound source changed', path))
    if phase['stop_after_new_updates'] > 4:
        proof = state.read(plan['prefix_review'])
        if state.sha(plan['prefix_review']) != plan['prefix_review_sha256'] or not proof['continuous_vs_resume_full_states_exact']:
            raise RuntimeError('Long training needs the actual three-rank restart proof')
        if proof['contract_sha256'] != state.sha(phase['contract']):
            raise RuntimeError('Restart proof is for a different training contract')
        qa = state.read(plan['data_qa'])
        if state.sha(plan['data_qa']) != plan['data_qa_sha256'] or qa['status'] != 'FULL1M20K_FACTUAL_POSITIVES_AUDITED':
            raise RuntimeError('Full factual data audit is missing')
    kernels.autotune_tree()
    if torch.cuda.is_initialized(): raise RuntimeError('GPU selection must precede CUDA initialization')
    rank, local, world, device, topology = allocated.distributed(timeout_seconds=600)
    if world != 3 or topology != contract['gpu_topology']:
        raise RuntimeError('Actual three-rank GPU topology differs from the contract')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8' or os.environ.get('FLA_CACHE_MODE') != 'disabled':
        raise RuntimeError('Required deterministic runtime environment is absent')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    seed = contract['data']['seed']
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed + rank)
    output, artifacts = Path(phase['output']), Path(phase['artifacts'])
    artifacts.mkdir(parents=True, exist_ok=True)
    initial_path = Path(contract['initialization']['checkpoint'])
    parent_contract = state.read(initial_path.parent / 'TRAIN_CONTRACT.json')

    def audit():
        output.mkdir(parents=True, exist_ok=True)
        cp = output / 'TRAIN_CONTRACT.json'
        if cp.exists():
            if state.read(cp) != contract: raise RuntimeError('Output belongs to another training lineage')
        else: state.write(cp, contract)
        latest = state.latest(output, contract)
        if phase.get('resume'):
            if latest is None or latest != phase['resume']:
                raise RuntimeError('Only the current verified full state can resume in place')
        elif latest is not None:
            raise RuntimeError('Existing output requires explicit resume; do not duplicate training')
        if state.sha(initial_path) != contract['initialization']['sha256'] or state.sha(initial_path.parent / 'TRAIN_CONTRACT.json') != contract['initialization']['contract_sha256']:
            raise RuntimeError('Warm-start model identity changed')
        for path, digest in {**parent_contract['text_files'], **parent_contract['frontend_files']}.items():
            if state.sha(path) != digest: raise RuntimeError(('Frozen asset changed', path))
        if state.sha(contract['data']['index']) != contract['data']['index_sha256']:
            raise RuntimeError('Full training index changed')
        return True
    _rank0_audit(audit, rank=rank, device=device)
    parent, original_identity = load_training_checkpoint(initial_path, contract=parent_contract)
    if original_identity != contract['initialization'] or parent['step'] != 20000:
        raise RuntimeError('Initialization is not the selected native 20k model')
    model = SceneEventTrainingModel(contract['native_config']['model'], contract['readout'], contract['readout_seed'])
    model.encoder.load_state_dict(parent['model'], strict=True)
    if state.fingerprint(model.encoder.state_dict()) != state.fingerprint(parent['model']):
        raise RuntimeError('Warm-start weights were not loaded exactly')
    model.to(device).train()
    optimizers, schedulers = state.build_optimizers(model, contract)
    updates = 0; rank_rngs = None
    if phase.get('resume'):
        payload, ident = state.load(phase['resume']['checkpoint'], contract)
        if ident != phase['resume']: raise RuntimeError('Resume state identity changed')
        for key in optimizers:
            getattr(model, key).load_state_dict(payload[key]['model'], strict=True)
            optimizers[key].load_state_dict(payload[key]['optimizer'])
            schedulers[key].load_state_dict(payload[key]['scheduler'])
        updates = payload['new_updates']; rank_rngs = payload['rng_states']
        del payload
    if updates >= phase['stop_after_new_updates']:
        raise RuntimeError('Completed phase cannot be relaunched')
    model = DistributedDataParallel(model, device_ids=[local])
    cfg = contract['native_config']['text']
    teacher = FrozenCLAP44TextFeatures(cfg['model_path'], hidden_dim=contract['native_config']['model']['text_dim'],
        max_tokens=cfg['max_tokens'], batch_size=cfg['batch_size']).eval()
    if state_hash(teacher.conditioner.model) != contract['qwen_state_sha256']:
        raise RuntimeError('Frozen Qwen state identity changed')
    if rank_rngs is None:
        rank_rngs = [None] * world; dist.all_gather_object(rank_rngs, rng_state())
    initial = state.make(model.module, optimizers, schedulers, updates, contract, rank_rngs)
    state.validate(initial, contract)
    state.write(artifacts / f'INITIAL_rank{rank}.json', {'new_updates': updates,
        'encoder': state.fingerprint(initial['encoder']), 'readout': state.fingerprint(initial['readout']),
        'rank_rng': state.fingerprint(rank_rngs[rank]), 'epoch': initial['epoch'], 'next_batch': initial['next_batch'],
        'initial_encoder_equals_native20k': state.fingerprint(initial['encoder']['model']) == state.fingerprint(parent['model']) if updates == 0 else None,
        'both_optimizer_states_empty': all(not initial[k]['optimizer']['state'] for k in optimizers) if updates == 0 else None})
    del initial, parent
    mode = {'name': f'rank{rank}', 'mode': phase['kernel_mode'],
        'reference_case': phase['kernel_references'][str(rank)]}
    finish_catalog = kernel_continuation.install(mode, artifacts)
    dataset = FactualDataset(contract['data']['index'], expected_rows=1000000, verify_tensor_hashes=True)
    if dataset.split != 'train' or dataset.index_sha256 != contract['data']['index_sha256']:
        raise RuntimeError('Dataset split or frozen marker changed')
    training_step = compiled_training_step()
    resume_rng = rank_rngs[rank]; began = time.monotonic(); saved_identity = None
    metrics_file = (artifacts / f'records_rank{rank}.jsonl').open('x', buffering=1)
    while updates < phase['stop_after_new_updates']:
        epoch, offset = state.cursor(contract, updates)
        batches = NativeResumeBatches(dataset, rank=rank, world=3, seed=seed,
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
                if updates < 4:
                    gradients.update({k: state.fingerprint({n: p.grad for n, p in getattr(core, k).named_parameters()}) for k in optimizers})
            values = training_step(model, teacher, optimizers, schedulers, batch,
                contract=contract, device=device, observer=observe)
            updates += 1; epoch_next, offset_next = state.cursor(contract, updates)
            rng = rng_state()
            record = {'new_updates': updates, 'next_epoch': epoch_next, 'next_batch': offset_next,
                'metrics': values, 'pair_ids': [r['pair_id'] for r in batch['labels']],
                'batch_roles': [r['role'] for r in batch['labels']],
                'supervision_fingerprint': state.fingerprint({k: batch[k] for k in ('labels', 'negative_scene_texts', 'negative_owners', 'event_targets', 'event_metadata')}),
                'rank_rng_after': state.fingerprint(rng)}
            if updates <= 4:
                record.update({'batch_fingerprint': state.fingerprint({k: batch[k] for k in ('latent', 'mask', 'labels', 'negative_scene_texts', 'negative_owners', 'event_targets', 'event_metadata')}),
                    'gradients': gradients, 'encoder_state': state.fingerprint(model.module.encoder.state_dict()),
                    'readout_state': state.fingerprint(model.module.readout.state_dict())})
                state.write(artifacts / f'update{updates:06d}_rank{rank}.json', record)
            metrics_file.write(json.dumps(record, allow_nan=False) + '\n')
            if rank == 0 and (updates <= 4 or updates % 20 == 0):
                progress = {'at': datetime.now().astimezone().isoformat(), 'new_updates': updates,
                    'initialization_step': 20000, 'epoch': epoch_next, 'next_batch': offset_next,
                    'elapsed_seconds': time.monotonic() - began, 'metrics': values['global'], 'quality_gate_passed': False}
                state.write(artifacts / 'PROGRESS.json', progress, replace=True)
                print(json.dumps(progress, allow_nan=False), flush=True)
            if updates == phase['stop_after_new_updates'] or updates % contract['checkpoint_every_new_updates'] == 0:
                rank_states = [None] * world; dist.all_gather_object(rank_states, rng)
                payload = state.make(model.module, optimizers, schedulers, updates, contract, rank_states)
                state.validate(payload, contract)
                signature = state.fingerprint(payload); signatures = [None] * world
                dist.all_gather_object(signatures, signature)
                if any(x != signature for x in signatures):
                    raise RuntimeError('Three ranks disagree on complete training state')
                saved_identity = _rank0_audit(lambda: state.save(output / f'update-{updates:06d}.pt', payload, contract), rank=rank, device=device)
                state.write(artifacts / f'SAVED{updates:06d}_rank{rank}.json', {'identity': saved_identity, 'full_state_fingerprint': signature})
                del payload
        del loader
    metrics_file.flush(); os.fsync(metrics_file.fileno()); metrics_file.close()
    finish_catalog()
    if state_hash(teacher.conditioner.model) != contract['qwen_state_sha256']:
        raise RuntimeError('Qwen changed during training')
    state.write(artifacts / f'COMPLETE_rank{rank}.json', {'at': datetime.now().astimezone().isoformat(),
        'new_updates': updates, 'checkpoint': saved_identity, 'qwen_state_unchanged': True,
        'VAE_not_instantiated_precomputed_verified_latents': True,
        'versions': {n: importlib.metadata.version(n) for n in ('torch', 'numpy', 'transformers', 'triton')},
        'physical_gpus': [5, 6, 7], 'quality_gate_passed': False, 'independent_test_used': False})
    dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--plan', type=Path, required=True); p.add_argument('--phase', required=True)
    a = p.parse_args(); run(a.plan.resolve(strict=True), a.phase)
