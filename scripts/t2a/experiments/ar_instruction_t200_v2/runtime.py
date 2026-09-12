"""T200 Editing AR: native CE, frozen CLAP20k, no RF training forward."""
import ast
import copy
from datetime import datetime
import os
from pathlib import Path
import sys

import torch
from torch import distributed as dist

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.t2a.experiments.ar_instruction_t200_v1 import runtime as base
from scripts.t2a.experiments.ar_instruction_t200_v1.fingerprints import fingerprint
from scripts.t2a.experiments.ar_instruction_t200_v2 import ar_only
from scripts.t2a.experiments.ar_source_grounding_v1.evidence import sha, state_hash, write
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_joint_io import validate_run_contract

native = base.native
CONTRACT = 'editing_ar_instruction_t200_ar_only_v2'


def source_inventory():
    paths = [Path(__file__), Path(ar_only.__file__)]
    return {**base.source_inventory(), **{str(p.relative_to(ROOT)): sha(p) for p in paths}}


def validate_config(cfg, training_mode):
    assert training_mode == 'ar_pretrain' and cfg['lambda_ar'] == 1
    assert cfg['lambda_rf'] == cfg['lambda_source_distillation'] == cfg['source_feature_dropout'] == 0
    assert cfg['instruction_data']['contract'] == CONTRACT
    # Reuse every native validation except its historical positive RF weight.
    compatibility = copy.deepcopy(cfg); compatibility['lambda_rf'] = 1.
    native.validate_config(compatibility, training_mode)


def amend_contract(contract, cfg):
    assert contract['physical_gpus'] == [7] and contract['variant'] == 'global_and_sequence'
    assert contract['runtime_inputs'] == ['source_foa_latent', 'raw_edit_request']
    validate_config(cfg, contract['training_mode'])
    contract.update(instruction_data=copy.deepcopy(cfg['instruction_data']),
                    initial_state_transfer=copy.deepcopy(cfg['initial_state_transfer']),
                    training_objective='AR_CE_ONLY', rf_training=False,
                    rf_mode_scope='Read-only P10 generation RF validation; RF forward never participates in training.',
                    auxiliary=None, loss_normalization='global_valid_plan_tokens_per_optimizer_step',
                    optimizer_scope=copy.deepcopy(cfg['optimizer_scope']))


def trainer_tree():
    tree, counts = base.trainer_tree()
    # Each change is individually source-bound and invertible. The rest of the
    # native sampler, accumulation, clipping, scheduler and checkpoint loop is kept.
    replacements = {
        'rf_seed': None, 'generator': None, 'noise': None, 'times': None, 'noised': None,
        '(logits, prediction, query, teacher)': 'logits = wrapped(source_foa_latent=ar["source_foa_latent"], source_attention_mask=ar["source_attention_mask"], plan_input_ids=ar["plan_input_ids"], plan_attention_mask=ar["plan_attention_mask"], raw_edit_requests=ar["raw_edit_requests"])',
        '(_, _, ar_sum, rf_sum)': 'ar_sum = _ar_ce_sum(logits, ar["plan_labels"]); rf_sum = ar_sum.new_zeros(())',
        'aux_sum': 'aux_sum = ar_sum.new_zeros(())',
        'loss': 'loss = ar_sum * world / denominators[0].clamp_min(1.0)',
    }
    seen = dict.fromkeys(replacements, 0); originals = {}
    class Edit(ast.NodeTransformer):
        def visit_Assign(self, node):
            name = ast.unparse(node.targets[0])
            if name in replacements:
                seen[name] += 1; originals[name] = copy.deepcopy(node)
                if replacements[name] is None:
                    replacement = ast.Pass(); replacement._original = copy.deepcopy(node)
                    return replacement
                items = ast.parse(replacements[name]).body
                for i, item in enumerate(items): item._original = copy.deepcopy(node) if i == 0 else False
                return items
            if name == 'wrapped':
                counts['ready'] = counts.get('ready', 0) + 1
                return [node, ast.parse('_ar_ready(module, optimizer)').body[0]]
            return self.generic_visit(node)
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'optimizer.step':
                counts['gradients'] = counts.get('gradients', 0) + 1
                return [ast.parse('_ar_gradient_audit(module, step)').body[0], node]
            return self.generic_visit(node)
        def visit_Dict(self, node):
            node = self.generic_visit(node)
            # Training zero placeholders are not measured RF/distillation errors.
            keys = [k.value if isinstance(k, ast.Constant) else None for k in node.keys]
            if 'ar_ce' in keys and 'rf_mse' in keys and 'grad_norms' in keys:
                counts['logging'] = counts.get('logging', 0) + 1
                node._original = copy.deepcopy(node)
                for key in ('rf_mse', 'source_distillation'): node.values[keys.index(key)] = ast.Constant(None)
            return node
    changed = Edit().visit(copy.deepcopy(tree))
    assert all(v == 1 for v in seen.values()), seen
    assert all(counts.get(k) == 1 for k in ('ready', 'gradients', 'logging'))
    class Undo(ast.NodeTransformer):
        def generic_visit(self, node):
            if hasattr(node, '_original'):
                return node._original if node._original is not False else None
            return super().generic_visit(node)
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in ('_ar_ready', '_ar_gradient_audit'):
                return None
            return self.generic_visit(node)
    assert ast.dump(Undo().visit(copy.deepcopy(changed)), include_attributes=False) == ast.dump(tree, include_attributes=False)
    return ast.fix_missing_locations(changed), {**counts, **{'replace_' + k: v for k, v in seen.items()}}


class Hooks(base.Hooks):
    def wrap_module(self, module):
        super().wrap_module(module)
        self.groups, self.audit = ar_only.configure(module, self.cfg)
        assert self.audit == base.read(self.spec['optimizer_scope_audit'])
        self.frozen_before = None; self.rf_training_calls = 0; self.rf_readonly_calls = 0
        def guard(_module, _args):
            if torch.is_grad_enabled():
                self.rf_training_calls += 1
                raise RuntimeError('RF forward forbidden in this AR-only training contract')
            self.rf_readonly_calls += 1
        module.diffusion.model.register_forward_pre_hook(guard)
        return module

    def optimizer_groups(self, module, **kwargs):
        assert module is self.module
        return self.groups, self.audit['trainable_parameters']

    def load(self, path, *, expected_contract=None, verify_sources=True, require_latest=False):
        parent = self.cfg['initial_state_transfer']['parent_checkpoint']
        if Path(path).resolve() != Path(parent['checkpoint']).resolve():
            payload, identity = native.load_joint_checkpoint(path, expected_contract=expected_contract,
                            verify_sources=verify_sources, require_latest=require_latest)
            self.loaded_step = int(payload['global_step']); return payload, identity
        assert not self.initial_loaded and verify_sources and require_latest
        run = Path(self.phase['run_dir'])
        assert not list(run.glob('checkpoints/step-*.pt'))
        assert expected_contract == base.read(run / 'RUN_CONTRACT.json')
        validate_run_contract(expected_contract, run)
        payload, identity = native.load_joint_checkpoint(path, expected_contract=base.read(self.spec['parent_contract']),
                                                        verify_sources=True, require_latest=True)
        assert identity == parent
        assert payload['run_contract']['gpu_topology'] == expected_contract['gpu_topology'] == base.allocated_runtime.gpu_topology([7])
        original_fingerprint = fingerprint(base.training_state(payload))
        assert original_fingerprint == base.read(self.spec['initial_state_fingerprint'])
        projected, receipt = ar_only.project_optimizer(payload['optimizer'], self.audit)
        payload = {**payload, 'optimizer': projected}
        receipt.update(parent_checkpoint=identity, parent_complete_state_sha256=original_fingerprint['sha256'],
                       destination_contract_sha256=sha(run / 'RUN_CONTRACT.json'),
                       model_scheduler_all_rank_rng_and_data_unchanged=True,
                       new_start_checkpoint_files=0, native_parent_loader_all_checks_enabled=True)
        write(self.out / 'INITIAL_STATE_LOADED.json', receipt)
        self.initial_loaded = True; self.loaded_step = int(payload['global_step'])
        return payload, identity

    def frozen_state(self):
        return fingerprint({n: p for n, p in self.module.named_parameters() if not p.requires_grad})

    def ready(self, module, optimizer):
        assert module is self.module
        names = {id(p): n for n, p in module.named_parameters()}
        actual = [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]
        assert actual == self.audit['new_parameter_names']
        self.frozen_before = self.frozen_state()
        write(self.out / 'OPTIMIZER_SCOPE_VERIFIED.json', {**self.audit, 'frozen_parameter_sha256': self.frozen_before['sha256'],
              'objective': 'AR_CE_ONLY', 'RF_training_forward_forbidden': True, 'all_selected_states_restored': True})

    def gradient_audit(self, module, step):
        missing = [n for n, p in module.named_parameters() if p.requires_grad and p.grad is None]
        leaked = [n for n, p in module.named_parameters() if not p.requires_grad and p.grad is not None]
        assert not missing and not leaked, {'missing_trainable_gradients': missing, 'frozen_gradients': leaked}
        if step == self.loaded_step:
            write(self.out / 'GRADIENT_SCOPE_VERIFIED.json', {'every_trainable_parameter_has_gradient': True,
                'every_frozen_parameter_has_no_gradient': True, 'rf_training_calls': self.rf_training_calls})


def main():
    p = base.argparse.ArgumentParser(add_help=False)
    p.add_argument('--experiment-plan', type=Path, required=True); p.add_argument('--phase', required=True)
    args, native_args = p.parse_known_args()
    plan = base.read(args.experiment_plan); spec = base.read(plan['experiment_spec']); phase = spec['phases'][args.phase]
    cfg = base.read(phase['config'])
    for path, expected in plan['source_sha256'].items(): assert sha(path) == expected, path
    assert os.environ['EDITING_GPUS'] == '7' and int(os.environ['WORLD_SIZE']) == 1
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8' and os.environ['FLA_CACHE_MODE'] == 'disabled'
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    hooks = Hooks(spec, phase, cfg); hooks.out.mkdir(exist_ok=True, parents=True)
    tree, counts = trainer_tree()
    write(hooks.out / 'NATIVE_AST_QA.json', {'counts': counts, 'reverse_transform_recovers_native_main_exactly': True,
          'native_source_sha256': sha(native.__file__), 'native_AR_forward_and_CE_kept': True,
          'RF_noise_forward_and_loss_removed_from_training': True})
    finish = base.numerics.install_autotune_observer({'name': 'rank0', 'mode': 'pin',
             'reference_case': spec['fla_reference_artifacts']}, hooks.out)
    calls, restore = base.install_flash(cfg['numerical_execution'])
    namespace = dict(native.__dict__, _distributed=base.allocated_runtime.distributed, run_source_inventory=source_inventory,
          validate_config=validate_config, optimizer_groups=hooks.optimizer_groups, load_joint_checkpoint=hooks.load,
          _text_module=hooks.wrap_module, _text_dataset=hooks.dataset, _text_contract=amend_contract,
          _text_window=hooks.window, _text_update=hooks.update, _ar_ce_sum=ar_only.ce_sum,
          _ar_ready=hooks.ready, _ar_gradient_audit=hooks.gradient_audit)
    exec(compile(tree, __file__ + '::native_ar_only', 'exec'), namespace)
    sys.argv = [native.__file__, *native_args]; stopped = False
    try:
        try: namespace['main']()
        except base.PrefixComplete:
            assert phase['stop_after_prefix']; stopped = True; dist.barrier()
        finish()
        before = base.read(hooks.out / 'FROZEN_STATE_BEFORE.json')
        after = {'qwen': state_hash(hooks.module.ar.instruction_conditioner.model),
                 'clap': state_hash(hooks.module.ar.source_clap_model)}
        assert before == after and hooks.frozen_before == hooks.frozen_state()
        assert hooks.rf_training_calls == 0 and hooks.loaded_step is not None
        write(hooks.out / 'RUNTIME_COMPLETE.json', {'at': datetime.now().astimezone().isoformat(),
              'prefix_stopped': stopped, 'frozen_states_unchanged': after,
              'all_frozen_parameters_exact': True, 'frozen_parameter_elements': hooks.frozen_before['tensor_elements'],
              'flash_calls': calls, 'all_flash_backward_deterministic': True,
              'rf_training_calls': hooks.rf_training_calls, 'rf_readonly_validation_calls': hooks.rf_readonly_calls,
              'training_objective': 'AR_CE_ONLY', 'quality_gate_passed': False,
              'independent_test_used': False, 'notifications_enabled': False})
    finally:
        restore()
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__': main()
