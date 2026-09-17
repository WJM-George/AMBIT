"""Native Editing AR with a bound request overlay and declared full-state start.

The original native trainer, forward, losses, optimizer and checkpoint writer
are reused. A protected native parent supplies an explicitly declared state
transfer at the first start; this is not an in-place resume under changed data.
After publication, own-run resume uses the unchanged native loader/contract.
"""
import argparse
import ast
import copy
from datetime import datetime
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
from scripts.t2a.train import train_sceneplan_transfusion_editing_ar_clap44 as native
from scripts.t2a.train.train_sceneplan_transfusion_editing_ar_joint_full import _capture_rank_rng_state
from scripts.t2a.experiments.ar_source_grounding_v1 import allocated_runtime, numerics
from scripts.t2a.experiments.ar_source_grounding_v1.evidence import sha, state_hash, write
from scripts.t2a.experiments.ar_instruction_t200_v1.instruction_data import InstructionOverlay, DatasetWithInstructions
from scripts.t2a.experiments.ar_instruction_t200_v1.fingerprints import fingerprint, training_state
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_joint_io import ar_specific_state, validate_run_contract
from stable_audio_tools.models import transformer

CONTRACT = 'editing_ar_instruction_t200_native_training_v1'
RELATIVE_ROOT = 'scripts/t2a/experiments/ar_instruction_t200_v1'


class PrefixComplete(Exception):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def source_inventory():
    paths = [ROOT / RELATIVE_ROOT / name for name in ('runtime.py', 'fingerprints.py', 'instruction_data.py', 'ORIGINS.json')]
    paths += [ROOT / 'scripts/t2a/experiments/ar_source_grounding_v1' / name
              for name in ('allocated_runtime.py', 'numerics.py', 'evidence.py')]
    return {**native.run_source_inventory(), **{str(p.relative_to(ROOT)): sha(p) for p in paths}}


def amend_contract(contract, cfg):
    data = cfg['instruction_data']
    assert data['contract'] == CONTRACT and data['mode'] in ('native', 't200')
    assert set(data['overlays']) == {'train', 'validation'}
    assert contract['training_mode'] == 'ar_pretrain' and contract['variant'] == 'global_and_sequence'
    assert contract['runtime_inputs'] == ['source_foa_latent', 'raw_edit_request']
    assert cfg['lambda_source_distillation'] == cfg['source_feature_dropout'] == 0
    assert 'source_grounding' not in cfg
    assert contract['physical_gpus'] == [7]
    contract['instruction_data'] = copy.deepcopy(data)
    contract['initial_state_transfer'] = copy.deepcopy(cfg['initial_state_transfer'])


def trainer_tree():
    original = ast.parse(textwrap.dedent(inspect.getsource(native.main)))
    counts = dict.fromkeys(('module', 'dataset', 'contract', 'window', 'update'), 0)
    class Insert(ast.NodeTransformer):
        def visit_Assign(self, node):
            node = self.generic_visit(node)
            name = ast.unparse(node.targets[0])
            if name == 'module':
                counts['module'] += 1
                node.value = ast.Call(ast.Name('_text_module', ast.Load()), [node.value], [])
            if name == 'datasets[split]':
                counts['dataset'] += 1
                node.value = ast.Call(ast.Name('_text_dataset', ast.Load()), [node.value, ast.Name('split', ast.Load())], [])
            return node
        def visit_Expr(self, node):
            node = self.generic_visit(node)
            if isinstance(node.value, ast.Call):
                if ast.unparse(node.value.func) == 'contract.update':
                    counts['contract'] += 1
                    return [node, ast.parse('_text_contract(contract, cfg)').body[0]]
                if ast.unparse(node.value) == 'dist.all_reduce(denominators)':
                    counts['window'] += 1
                    return [node, ast.parse('_text_window(window, step, epoch)').body[0]]
            return node
        def visit_AugAssign(self, node):
            node = self.generic_visit(node)
            if ast.unparse(node.target) == 'next_batch':
                counts['update'] += 1
                return [node, ast.parse('_text_update(module, optimizer, scheduler, step, epoch, next_batch, rank, world, device, denominators, sums, grad_norms)').body[0]]
            return node
    tree = Insert().visit(copy.deepcopy(original))
    assert all(v == 1 for v in counts.values()), counts
    class Strip(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in ('_text_contract', '_text_window', '_text_update'):
                return None
            return self.generic_visit(node)
        def visit_Assign(self, node):
            node = self.generic_visit(node)
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in ('_text_module', '_text_dataset'):
                node.value = node.value.args[0]
            return node
    assert ast.dump(Strip().visit(copy.deepcopy(tree)), include_attributes=False) == ast.dump(original, include_attributes=False)
    return ast.fix_missing_locations(tree), counts


class Hooks:
    def __init__(self, spec, phase, cfg):
        self.spec = spec; self.phase = phase; self.cfg = cfg
        self.out = Path(phase['artifacts']); self.module = None; self.window_data = None
        self.snapshots = []; self.initial_loaded = False; self.loaded_step = None

    def wrap_module(self, module):
        self.module = module
        assert state_hash(module.ar.instruction_conditioner.model) == self.spec['qwen_state_sha256']
        assert all(not p.requires_grad for p in module.ar.instruction_conditioner.model.parameters())
        assert all(not p.requires_grad for p in module.ar.source_clap_model.parameters())
        write(self.out / 'FROZEN_STATE_BEFORE.json', {'qwen': state_hash(module.ar.instruction_conditioner.model),
                                                    'clap': state_hash(module.ar.source_clap_model)})
        return module

    def dataset(self, dataset, split):
        item = self.cfg['instruction_data']['overlays'][split]
        # Both scientific arms bind and validate identical data assets; only
        # t200 replaces the text. Native control returns the original object.
        overlay = InstructionOverlay(item['path'], expected_sha256=item['sha256'],
                    native_index_path=item['native_index_path'], native_index_sha256=item['native_index_sha256'],
                    expected_rows=item['rows'], split=split)
        if self.cfg['instruction_data']['mode'] == 'native':
            return dataset
        return DatasetWithInstructions(dataset, overlay, joint=True)

    def load(self, path, *, expected_contract=None, verify_sources=True, require_latest=False):
        parent = self.cfg['initial_state_transfer']['parent_checkpoint']
        if Path(path).resolve() != Path(parent['checkpoint']).resolve():
            payload, identity = native.load_joint_checkpoint(path, expected_contract=expected_contract,
                         verify_sources=verify_sources, require_latest=require_latest)
            self.loaded_step = int(payload['global_step'])
            return payload, identity
        assert not self.initial_loaded and verify_sources and require_latest
        run = Path(self.phase['run_dir'])
        assert not list(run.glob('checkpoints/step-*.pt')), 'Own published state exists; resume its actual LATEST'
        assert expected_contract == read(run / 'RUN_CONTRACT.json')
        validate_run_contract(expected_contract, run, verify_sources=True)
        payload, identity = native.load_joint_checkpoint(path, expected_contract=read(self.spec['parent_contract']),
                                                         verify_sources=True, require_latest=True)
        assert identity == parent
        original = payload['run_contract']
        changed = sorted(k for k in set(original) | set(expected_contract) if original.get(k) != expected_contract.get(k))
        assert changed == self.spec['allowed_changed_contract_fields']
        assert original['physical_gpus'] == expected_contract['physical_gpus'] == [7]
        assert original['gpu_topology'] == expected_contract['gpu_topology'] == allocated_runtime.gpu_topology([7])
        result = fingerprint(training_state(payload))
        assert result == read(self.spec['initial_state_fingerprint'])
        write(self.out / 'INITIAL_STATE_LOADED.json', {'parent_checkpoint': identity,
              'destination_contract_sha256': sha(run / 'RUN_CONTRACT.json'), 'changed_contract_fields': changed,
              'model_optimizer_scheduler_all_rank_rng_and_data_fingerprint': result['sha256'],
              'tensor_elements': result['tensor_elements'], 'state_transfer_without_payload_mutation': True,
              'new_start_checkpoint_files': 0, 'native_parent_loader_all_checks_enabled': True,
              'initialization_is_declared_fork_not_in_place_resume': True})
        self.initial_loaded = True
        self.loaded_step = int(payload['global_step'])
        return payload, identity

    def window(self, window, step, epoch):
        assert self.window_data is None
        self.window_data = {'step_before': step, 'epoch': epoch, 'microbatches': [
            {'pair_ids': list(b['ar']['pair_ids']), 'pair_ordinals': [int(m['pair_ordinal']) for m in b['metadata']],
             'source_sample_ids': [m['source_sample_id'] for m in b['metadata']],
             'request_sha256': [__import__('hashlib').sha256(s.encode()).hexdigest() for s in b['ar']['raw_edit_requests']]}
            for b in window]}

    def update(self, module, optimizer, scheduler, step, epoch, next_batch, rank, world, device, denominators, sums, grad_norms):
        assert world == 1 and rank == 0 and step == self.window_data['step_before'] + 1
        record = {'step': step, 'epoch': epoch, 'next_batch': next_batch,
                  'native_denominators': denominators.cpu().tolist(), 'native_local_loss_sums': sums.cpu().tolist(),
                  'grad_norms_before_clip': grad_norms, 'learning_rates_after_scheduler': [g['lr'] for g in optimizer.param_groups],
                  'microbatches': self.window_data['microbatches']}
        with (self.out / 'training_windows.jsonl').open('a') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')
        self.window_data = None
        if step == self.spec['parent_step'] + self.spec['prefix_updates']:
            snapshot = {'diffusion_state_dict': module.diffusion.state_dict(),
                        'editing_ar_specific_state_dict': ar_specific_state(module.ar),
                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                        'rng_states_by_rank': [_capture_rank_rng_state(rank=rank, device=device)],
                        'global_step': step, 'epoch': epoch, 'next_batch': next_batch}
            result = fingerprint(snapshot)
            write(self.out / f'PREFIX_FINGERPRINT_rank{rank}.json', result)
            write(self.out / f'PREFIX_RECEIPT_rank{rank}.json', {'at': datetime.now().astimezone().isoformat(),
                  'step': step, 'fingerprint_sha256': result['sha256'], 'tensor_elements': result['tensor_elements'],
                  'scope': 'Every model/optimizer tensor, scheduler, rank RNG and data cursor; typed SHA256 evidence, not a recoverable optimizer payload.',
                  'full_state_files_created': 0, 'quality_gate_passed': False})
            self.snapshots.append(step)
            if self.phase['stop_after_prefix']:
                raise PrefixComplete()


def install_flash(policy):
    assert policy['deterministic_backward'] is True
    for path, expected in policy['implementation_sha256'].items():
        assert sha(path) == expected, path
    originals = {name: getattr(transformer, name) for name in ('flash_attn_func', 'flash_attn_varlen_func')}
    calls = {name: 0 for name in originals}
    def wrap(name, fn):
        signature = inspect.signature(fn)
        assert signature.parameters['deterministic'].default is False
        def invoke(*args, **kwargs):
            assert 'deterministic' not in signature.bind_partial(*args, **kwargs).arguments
            kwargs['deterministic'] = True; calls[name] += 1
            return fn(*args, **kwargs)
        return invoke
    for name, fn in originals.items(): setattr(transformer, name, wrap(name, fn))
    def restore():
        for name, fn in originals.items(): setattr(transformer, name, fn)
    return calls, restore


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--experiment-plan', type=Path, required=True); p.add_argument('--phase', required=True)
    args, native_args = p.parse_known_args()
    plan = read(args.experiment_plan); spec = read(plan['experiment_spec']); phase = spec['phases'][args.phase]
    cfg = read(phase['config'])
    for path, expected in plan['source_sha256'].items(): assert sha(path) == expected, path
    assert os.environ['EDITING_GPUS'] == '7' and int(os.environ['WORLD_SIZE']) == 1
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8' and os.environ['FLA_CACHE_MODE'] == 'disabled'
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    hooks = Hooks(spec, phase, cfg); hooks.out.mkdir(exist_ok=True, parents=True)
    tree, counts = trainer_tree()
    write(hooks.out / 'NATIVE_AST_QA.json', {'counts': counts, 'reverse_transform_recovers_native_main_exactly': True,
            'native_source_sha256': sha(native.__file__), 'native_forward_and_loss_arithmetic_unchanged': True})
    finish = numerics.install_autotune_observer({'name': 'rank0', 'mode': 'pin',
            'reference_case': spec['fla_reference_artifacts']}, hooks.out)
    calls, restore = install_flash(cfg['numerical_execution'])
    namespace = dict(native.__dict__, _distributed=allocated_runtime.distributed, run_source_inventory=source_inventory,
            load_joint_checkpoint=hooks.load, _text_module=hooks.wrap_module, _text_dataset=hooks.dataset,
            _text_contract=amend_contract, _text_window=hooks.window, _text_update=hooks.update)
    exec(compile(tree, __file__ + '::native_text_overlay', 'exec'), namespace)
    sys.argv = [native.__file__, *native_args]
    stopped = False
    try:
        try:
            namespace['main']()
        except PrefixComplete:
            assert phase['stop_after_prefix']; stopped = True; dist.barrier()
        finish()
        assert hooks.initial_loaded or not phase['stop_after_prefix']
        before = read(hooks.out / 'FROZEN_STATE_BEFORE.json')
        after = {'qwen': state_hash(hooks.module.ar.instruction_conditioner.model),
                 'clap': state_hash(hooks.module.ar.source_clap_model)}
        assert before == after and hooks.loaded_step is not None
        assert sum(calls.values()) > 0 or hooks.loaded_step >= cfg['schedule']['max_steps']
        prefix_step = spec['parent_step'] + spec['prefix_updates']
        expected_snapshots = [prefix_step] if hooks.loaded_step < prefix_step <= cfg['schedule']['max_steps'] else []
        assert hooks.snapshots == expected_snapshots
        write(hooks.out / 'RUNTIME_COMPLETE.json', {'at': datetime.now().astimezone().isoformat(),
              'prefix_stopped': stopped, 'frozen_states_unchanged': after, 'flash_calls': calls,
              'all_flash_backward_deterministic': True, 'no_auxiliary_loss_added': True,
              'quality_gate_passed': False, 'independent_test_used': False, 'notifications_enabled': False})
    finally:
        restore()
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__':
    main()
