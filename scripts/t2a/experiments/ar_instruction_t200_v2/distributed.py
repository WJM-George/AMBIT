"""Three-GPU T200 AR-only extension of the frozen, verified native runtime.

The native DDP sampler, AR CE, gradient accumulation, clipping, AdamW,
validation and checkpoint writer are retained. Only declared transfer and LR
policy plus rank-local evidence handling differ from the single-GPU runtime.
"""
import ast
import copy
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from torch import distributed as dist

from scripts.t2a.experiments.ar_instruction_t200_v2 import runtime, distributed_transfer as transfer
from scripts.t2a.experiments.ar_instruction_t200_v1.fingerprints import fingerprint, training_state
from scripts.t2a.train.train_sceneplan_transfusion_editing_ar_joint_full import _gather_rank_rng_states
from scripts.t2a.train.sceneplan_transfusion_editing_joint_run_contract import validate_rng_inventory
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_joint_io import ar_specific_state, validate_run_contract

base, native, sha, write = runtime.base, runtime.native, runtime.sha, runtime.write


def source_inventory():
    return {**runtime.source_inventory(), **{str(p.relative_to(ROOT)): sha(p)
            for p in (Path(__file__), Path(transfer.__file__))}}


def amend_contract(contract, cfg):
    runtime.validate_config(cfg, contract['training_mode'])
    transfer.validate_policy(cfg)
    assert contract['physical_gpus'] == transfer.GPUS and contract['world_size'] == 3
    assert contract['variant'] == 'global_and_sequence'
    assert contract['runtime_inputs'] == ['source_foa_latent', 'raw_edit_request']
    contract.update(instruction_data=copy.deepcopy(cfg['instruction_data']),
        initial_state_transfer=copy.deepcopy(cfg['initial_state_transfer']),
        training_objective='AR_CE_ONLY', rf_training=False,
        rf_mode_scope='Read-only P10 generation RF validation; RF forward never participates in training.',
        auxiliary=None, loss_normalization='global_valid_plan_tokens_per_optimizer_step',
        optimizer_scope=copy.deepcopy(cfg['optimizer_scope']),
        distributed_transfer=copy.deepcopy(cfg['distributed_transfer']))


def trainer_tree():
    original, counts = runtime.trainer_tree()
    changed = copy.deepcopy(original); functions = [n for n in ast.walk(changed)
            if isinstance(n, ast.FunctionDef) and n.name == 'lr_multiplier']
    assert len(functions) == 1
    saved = copy.deepcopy(functions[0].body)
    functions[0].body = ast.parse('return _distributed_lr(step, cfg)').body
    undo = copy.deepcopy(changed)
    next(n for n in ast.walk(undo) if isinstance(n, ast.FunctionDef) and n.name == 'lr_multiplier').body = saved
    assert ast.dump(undo, include_attributes=False) == ast.dump(original, include_attributes=False)
    return ast.fix_missing_locations(changed), {**counts, 'declared_continuous_LR_policy_reversible': 1}


def check_allocation(cfg, phase, spec):
    transfer.validate_policy(cfg)
    assert os.environ['EDITING_GPUS'] == '5,6,7' and int(os.environ['WORLD_SIZE']) == 3
    assert int(os.environ['RANK']) == int(os.environ['LOCAL_RANK']) in range(3)
    assert spec['physical_gpus'] == transfer.GPUS
    assert base.allocated_runtime.gpu_topology(transfer.GPUS) == spec['destination_topology']
    if not phase['stop_after_prefix']:
        gate = base.read(phase['production_gate'])
        assert sha(phase['production_gate']) == phase['production_gate_sha256']
        assert gate['full_AR_expansion_allowed'] and gate['three_rank_independent_and_native_resume_exact']


def distributed_namespace(namespace):
    return {**namespace, '_distributed_lr': transfer.lr_multiplier}


def entry_tree():
    original = ast.parse(textwrap.dedent(inspect.getsource(runtime.main)))
    counts = dict.fromkeys(('allocation', 'namespace', 'autotune'), 0)
    class Edit(ast.NodeTransformer):
        def visit_Assert(self, node):
            if ast.unparse(node.test) == "os.environ['EDITING_GPUS'] == '7' and int(os.environ['WORLD_SIZE']) == 1":
                counts['allocation'] += 1
                changed = ast.parse('_check_allocation(cfg, phase, spec)').body[0]
                changed._original = copy.deepcopy(node); return changed
            return self.generic_visit(node)
        def visit_Assign(self, node):
            node = self.generic_visit(node)
            if ast.unparse(node.targets[0]) == 'namespace':
                counts['namespace'] += 1; node._original = copy.deepcopy(node)
                node.value = ast.Call(ast.Name('_distributed_namespace', ast.Load()), [node.value], [])
            return node
        def visit_Dict(self, node):
            node = self.generic_visit(node)
            for k, value in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == 'name' and isinstance(value, ast.Constant) and value.value == 'rank0':
                    counts['autotune'] += 1; node._original = copy.deepcopy(node)
                    value.value = 'autotune'
            return node
    changed = Edit().visit(copy.deepcopy(original))
    assert all(v == 1 for v in counts.values()), counts
    class Undo(ast.NodeTransformer):
        def generic_visit(self, node):
            return node._original if hasattr(node, '_original') else super().generic_visit(node)
    assert ast.dump(Undo().visit(copy.deepcopy(changed)), include_attributes=False) == ast.dump(original, include_attributes=False)
    return ast.fix_missing_locations(changed), counts


class Hooks(runtime.Hooks):
    def __init__(self, spec, phase, cfg):
        super().__init__(spec, phase, cfg)
        self.rank = int(os.environ['RANK'])
        self.out = self.out / f'rank{self.rank}'

    def load(self, path, *, expected_contract=None, verify_sources=True, require_latest=False):
        parent = self.cfg['initial_state_transfer']['parent_checkpoint']
        if Path(path).resolve() != Path(parent['checkpoint']).resolve():
            return super().load(path, expected_contract=expected_contract,
                        verify_sources=verify_sources, require_latest=require_latest)
        assert not self.initial_loaded and verify_sources and require_latest
        run = Path(self.phase['run_dir'])
        assert not list(run.glob('checkpoints/step-*.pt'))
        assert expected_contract == base.read(run / 'RUN_CONTRACT.json')
        validate_run_contract(expected_contract, run, verify_sources=True)
        payload, identity = native.load_joint_checkpoint(path,
                expected_contract=base.read(self.spec['parent_contract']), verify_sources=True, require_latest=True)
        assert identity == parent and payload['run_contract']['training_objective'] == 'AR_CE_ONLY'
        changed = sorted(k for k in set(payload['run_contract']) | set(expected_contract)
                if payload['run_contract'].get(k) != expected_contract.get(k))
        assert changed == self.spec['allowed_changed_contract_fields']
        initial = fingerprint(training_state(payload))
        assert initial == base.read(self.spec['initial_state_fingerprint'])
        rank_state = transfer.rng_for_rank(payload['rng_states_by_rank'], rank=self.rank,
                cfg=self.cfg, device=torch.device('cuda', self.rank))
        states = [None] * 3; dist.all_gather_object(states, rank_state)
        validate_rng_inventory(states, world_size=3)
        inherited = {**states[2], 'rank': 0}
        assert fingerprint(inherited) == fingerprint(payload['rng_states_by_rank'][0])
        result = transfer.transferred_state(payload, states, self.cfg)
        for key in ('diffusion_state_dict', 'editing_ar_specific_state_dict', 'optimizer', 'scheduler'):
            assert result[key] is payload[key]
        write(self.out / 'INITIAL_STATE_LOADED.json', {
            'parent_checkpoint': identity, 'parent_complete_state_sha256': initial['sha256'],
            'parent_data_position': {'epoch': payload['epoch'], 'next_batch': payload['next_batch']},
            'destination_data_position': self.cfg['distributed_transfer']['data_start'],
            'destination_contract_sha256': sha(run / 'RUN_CONTRACT.json'),
            'changed_contract_fields': changed, 'model_optimizer_scheduler_preserved': True,
            'GPU7_RNG_bytes_preserved_at_logical_rank2': True,
            'new_rank0_and_rank1_streams': self.cfg['distributed_transfer']['new_rank_seed'],
            'all_rank_rng_sha256': fingerprint(states)['sha256'],
            'declared_fork_not_exact_single_to_three_rank_resume': True,
            'new_start_checkpoint_files': 0, 'native_parent_loader_all_checks_enabled': True})
        self.initial_loaded = True; self.loaded_step = int(result['global_step'])
        return result, identity

    def update(self, module, optimizer, scheduler, step, epoch, next_batch, rank, world,
               device, denominators, sums, grad_norms):
        assert world == 3 and rank == self.rank and step == self.window_data['step_before'] + 1
        record = {'step': step, 'epoch': epoch, 'next_batch': next_batch, 'rank': rank,
            'native_denominators': denominators.cpu().tolist(), 'native_local_loss_sums': sums.cpu().tolist(),
            'grad_norms_before_clip': grad_norms,
            'learning_rates_after_scheduler': [g['lr'] for g in optimizer.param_groups],
            'microbatches': self.window_data['microbatches']}
        with (self.out / 'training_windows.jsonl').open('a') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')
        self.window_data = None
        stop = self.phase['stop_after_prefix'] and step == self.phase['stop_at_step']
        if step in self.spec['fingerprint_steps'] or stop:
            states = _gather_rank_rng_states(rank=rank, world_size=world, device=device)
            snapshot = {'diffusion_state_dict': module.diffusion.state_dict(),
                'editing_ar_specific_state_dict': ar_specific_state(module.ar),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'rng_states_by_rank': states, 'global_step': step, 'epoch': epoch, 'next_batch': next_batch}
            result = fingerprint(snapshot)
            hashes = [None] * world; dist.all_gather_object(hashes, result['sha256'])
            assert len(set(hashes)) == 1, 'DDP model/optimizer/scheduler/all-rank state differs between ranks'
            write(self.out / f'PREFIX_FINGERPRINT_step{step}.json', result)
            self.snapshots.append(step)
            if stop and self.phase.get('checkpoint_at_stop', False):
                if rank == 0:
                    native.save_joint_checkpoint(Path(self.phase['run_dir']) / f'checkpoints/step-{step:08d}.pt',
                        module=module, optimizer=optimizer, scheduler=scheduler, step=step,
                        epoch=epoch, next_batch=next_batch,
                        contract=base.read(Path(self.phase['run_dir']) / 'RUN_CONTRACT.json'), rng_states=states)
                dist.barrier()
        if stop:
            raise base.PrefixComplete()


def main():
    tree, _ = entry_tree()
    namespace = {**runtime.main.__globals__, '__file__': __file__, 'Hooks': Hooks,
        'trainer_tree': trainer_tree, 'source_inventory': source_inventory,
        'amend_contract': amend_contract, '_check_allocation': check_allocation,
        '_distributed_namespace': distributed_namespace}
    exec(compile(tree, __file__ + '::native_three_rank_entry', 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__': main()
