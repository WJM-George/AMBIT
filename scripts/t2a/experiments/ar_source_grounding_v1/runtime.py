"""Bounded unchanged-field source preference added to the native AR/RF loop.

Native core sources remain immutable. The reversible AST additions expose only
contract, window-context, auxiliary-loss and evidence hooks. Native AR/RF
arithmetic, sampling, RF noise, optimizer, schedule and checkpoint IO are reused.
"""
import argparse
import ast
import copy
from datetime import datetime
import inspect
import json
import math
import os
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from torch import distributed as dist
from torch.nn import functional as F
from scripts.t2a.experiments.ar_source_grounding_v1 import allocated_runtime, numerics
from scripts.t2a.experiments.ar_source_grounding_v1.evidence import cpu_copy, sha, state_hash, write
from scripts.t2a.experiments.ar_source_grounding_v1.field_loss import CONTENT_FIELDS, target_field_masks, source_preference_terms
from scripts.t2a.train import train_sceneplan_transfusion_editing_ar_clap44 as native
from scripts.t2a.train.train_sceneplan_transfusion_editing_ar_joint_full import _capture_rank_rng_state
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_joint_io import ar_specific_state

METHOD_CONTRACT = 'editing_ar_unchanged_source_field_preference_training_v1'
RELATIVE_ROOT = 'scripts/t2a/experiments/ar_source_grounding_v1'
EXTENSION_FILES = ('runtime.py', 'field_loss.py', 'numerics.py', 'evidence.py', 'allocated_runtime.py', 'ORIGINS.json')


class PrefixReplayComplete(Exception):
    pass


def source_inventory():
    result = native.run_source_inventory()
    result.update({f'{RELATIVE_ROOT}/{name}': sha(ROOT / RELATIVE_ROOT / name) for name in EXTENSION_FILES})
    return result


def amend_contract(contract, cfg):
    extra = cfg['source_grounding']
    assert contract['training_mode'] == 'ar_pretrain' and contract['variant'] == 'global_and_sequence'
    assert contract['runtime_inputs'] == ['source_foa_latent', 'raw_edit_request']
    assert extra['contract'] == METHOD_CONTRACT
    assert math.isfinite(extra['weight']) and extra['weight'] >= 0
    assert math.isfinite(extra['margin']) and extra['margin'] >= 0
    assert cfg['source_feature_dropout'] == cfg['lambda_source_distillation'] == 0
    assert contract['auxiliary'] == 'frozen_source_audio_dual_head_distillation'
    assert contract['loss_normalization'] == 'global_valid_plan_tokens_RF_values_source_rows_per_optimizer_step'
    contract['auxiliary'] = {'native': contract['auxiliary'], 'source_grounding': METHOD_CONTRACT}
    contract['loss_normalization'] = {'native': contract['loss_normalization'],
        'source_grounding': 'global_eligible_unchanged_row_field_count_per_optimizer_accumulation_window'}


def prepare_microbatch(batch, codec):
    """Use target labels only for training masks; never append them to AR inputs."""
    ar, metadata = batch['ar'], batch['metadata']
    labels = ar['plan_labels']
    assert labels.ndim == 2 and len(metadata) == len(labels)
    source_ids = [row['source_sample_id'] for row in metadata]
    assert all(isinstance(x, str) and x for x in source_ids)
    donor, valid_donor = [], []
    for i in range(len(source_ids)):
        other = next(((i + offset) % len(source_ids) for offset in range(1, len(source_ids))
                      if source_ids[(i + offset) % len(source_ids)] != source_ids[i]), None)
        donor.append(i if other is None else other)
        valid_donor.append(other is not None)
    masks = {f'unchanged/{field}': torch.zeros_like(labels, dtype=torch.bool) for field in CONTENT_FIELDS}
    for i, row in enumerate(metadata):
        assert row['editing_split'] == 'train', 'Auxiliary training masks cannot consume development labels.'
        encoded = codec.encode(row['editing_ar_target_model_sceneplan'])
        local = target_field_masks(codec, encoded, edited_ids=row['edited_source_ids'], unchanged_ids=row['unchanged_source_ids'])
        length = len(encoded['input_ids']) - 1
        assert length <= labels.shape[1]
        assert torch.equal(labels[i, :length].cpu(), encoded['input_ids'][1:].cpu())
        assert bool(labels[i, length:].eq(-100).all())
        if valid_donor[i]:
            for key in masks:
                masks[key][i, :length] = local[key][1:]
    eligible = torch.stack([masks[f'unchanged/{field}'].sum(1) > 0 for field in CONTENT_FIELDS], 1)
    return {'labels': labels, 'masks': masks, 'donor': torch.tensor(donor, dtype=torch.long),
        'eligible': eligible, 'pair_ids': [row['pair_id'] for row in metadata],
        'pair_ordinals': [int(row['pair_ordinal']) for row in metadata], 'source_sample_ids': source_ids,
        'valid_donor': valid_donor}


def trainer_tree():
    original = ast.parse(textwrap.dedent(inspect.getsource(native.main)))
    counts = dict.fromkeys(['contract', 'module', 'window', 'bind', 'unpack', 'loss', 'micro_done', 'update'], 0)
    inserted = {'_grounding_contract', '_grounding_window', '_grounding_bind', '_grounding_micro_done', '_grounding_update'}

    class Insert(ast.NodeTransformer):
        def visit_Assign(self, node):
            node = self.generic_visit(node)
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name == 'module':
                    counts['module'] += 1
                    node.value = ast.Call(func=ast.Name(id='_grounding_wrap', ctx=ast.Load()), args=[node.value, ast.Name(id='codec', ctx=ast.Load()), ast.Name(id='cfg', ctx=ast.Load())], keywords=[])
                if name == 'loss':
                    assert isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'normalized_joint_loss'
                    counts['loss'] += 1
                    node.value = ast.Call(func=ast.Name(id='_grounding_add', ctx=ast.Load()), args=[node.value, ast.Name(id='grounding_sum', ctx=ast.Load()), ast.Name(id='world', ctx=ast.Load())], keywords=[])
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'wrapped':
                assert ast.unparse(node.targets[0]) == '(logits, prediction, query, teacher)'
                counts['unpack'] += 1
                node.targets[0].elts.append(ast.Name(id='grounding_sum', ctx=ast.Store()))
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == '_move_joint_batch':
                counts['bind'] += 1
                return [ast.parse('_grounding_bind(micro)').body[0], node]
            return node

        def visit_Expr(self, node):
            node = self.generic_visit(node)
            if isinstance(node.value, ast.Call):
                if ast.unparse(node.value.func) == 'contract.update':
                    counts['contract'] += 1
                    return [node, ast.parse('_grounding_contract(contract, cfg)').body[0]]
                if ast.unparse(node.value) == 'dist.all_reduce(denominators)':
                    counts['window'] += 1
                    return [node, ast.parse('_grounding_window(window, codec, device, step, epoch)').body[0]]
            return node

        def visit_AugAssign(self, node):
            node = self.generic_visit(node)
            if isinstance(node.target, ast.Name) and node.target.id == 'sums':
                counts['micro_done'] += 1
                return [node, ast.parse('_grounding_micro_done(grounding_sum)').body[0]]
            if isinstance(node.target, ast.Name) and node.target.id == 'next_batch':
                counts['update'] += 1
                return [node, ast.parse('_grounding_update(module, optimizer, scheduler, step, epoch, next_batch, rank, world, device, denominators, sums, grad_norms)').body[0]]
            return node

    tree = Insert().visit(copy.deepcopy(original))
    assert all(v == 1 for v in counts.values()), counts

    class Strip(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in inserted:
                return None
            return self.generic_visit(node)

        def visit_Assign(self, node):
            node = self.generic_visit(node)
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) in {'_grounding_wrap', '_grounding_add'}:
                node.value = node.value.args[0]
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'wrapped':
                assert node.targets[0].elts[-1].id == 'grounding_sum'
                node.targets[0].elts.pop()
            return node

    assert ast.dump(Strip().visit(copy.deepcopy(tree)), include_attributes=False) == ast.dump(original, include_attributes=False)
    return ast.fix_missing_locations(tree), counts


class TrainingHooks:
    def __init__(self, spec, phase_name, config):
        self.spec, self.phase_name, self.cfg = spec, phase_name, config
        self.phase = spec['phases'][phase_name]
        self.out = Path(self.phase['artifacts'])
        self.weight = float(config['source_grounding']['weight'])
        self.margin = float(config['source_grounding']['margin'])
        self.current = None
        self.prepared = []
        self.module = None
        self.snapshot_steps = []
        self.catalog_finish = None

    def wrap(self, module, codec, cfg):
        assert cfg == self.cfg and module.planning_pretrain
        self.module = module
        self.original_forward = module.forward
        assert state_hash(module.ar.instruction_conditioner.model) == self.spec['qwen_state_sha256']
        assert all(not p.requires_grad for p in module.ar.instruction_conditioner.model.parameters())
        assert all(not p.requires_grad for p in module.ar.source_clap_model.parameters())
        write(self.out / 'FROZEN_STATE_BEFORE.json', {'qwen': self.spec['qwen_state_sha256'],
            'clap': state_hash(module.ar.source_clap_model), 'quality_gate_passed': False})
        module.forward = self.forward
        return module

    def window(self, window, codec, device, step, epoch):
        assert self.current is None
        self.prepared = [prepare_microbatch(batch, codec) for batch in window]
        count = sum(int(item['eligible'].sum()) for item in self.prepared)
        self.denominator = torch.tensor(count, device=device, dtype=torch.float64)
        dist.all_reduce(self.denominator)
        self.auxiliary_sum = torch.zeros((), device=device, dtype=torch.float64)
        self.field_terms = torch.zeros(3, device=device, dtype=torch.float64)
        self.field_differences = torch.zeros(3, device=device, dtype=torch.float64)
        self.field_counts = torch.zeros(3, device=device, dtype=torch.float64)
        self.window_start_step = step
        self.window_epoch = epoch
        self.forward_count = 0

    def bind(self, micro):
        assert self.current is None
        self.current = self.prepared[micro]

    def forward(self, **kwargs):
        if self.current is None:
            return self.original_forward(**kwargs)
        self.forward_count += 1
        item = self.current
        assert self.module.training
        if self.weight == 0 or not bool(item['eligible'].any()):
            native_outputs = self.original_forward(**kwargs)
            assert len(native_outputs) == 4
            return (*native_outputs, native_outputs[0].new_zeros(()))
        ar = self.module.ar
        encoder = ar.source_clap_model
        original_encode, original_features = ar.encode_edit_instructions, encoder.source_features
        captured = {}
        def encode(*args, **kw):
            assert 'context' not in captured
            result = original_encode(*args, **kw)
            captured['context'] = result
            return result
        def features(*args, **kw):
            assert 'features' not in captured
            result = original_features(*args, **kw)
            captured['features'] = result
            return result
        ar.encode_edit_instructions, encoder.source_features = encode, features
        try:
            native_outputs = self.original_forward(**kwargs)
        finally:
            ar.encode_edit_instructions, encoder.source_features = original_encode, original_features
        assert set(captured) == {'context', 'features'} and len(native_outputs) == 4
        logits = native_outputs[0]
        donor = item['donor'].to(logits.device)
        masks = {k: v.to(logits.device) for k, v in item['masks'].items()}
        labels = item['labels'].to(logits.device)
        donor_features = {k: v.index_select(0, donor.to(v.device)) if isinstance(v, torch.Tensor) else v
                          for k, v in captured['features'].items()}
        context, context_mask = captured['context']
        negative = ar(kwargs['source_foa_latent'].index_select(0, donor),
            kwargs['source_attention_mask'].index_select(0, donor), kwargs['plan_input_ids'],
            kwargs['plan_attention_mask'], context, context_mask, source_clap_features=donor_features)
        clean_ce = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), ignore_index=-100, reduction='none').reshape_as(labels)
        donor_ce = F.cross_entropy(negative.float().flatten(0, 1), labels.flatten(), ignore_index=-100, reduction='none').reshape_as(labels)
        terms, eligible, differences = source_preference_terms(clean_ce, donor_ce, masks, margin=self.margin)
        assert torch.equal(eligible.cpu(), item['eligible'])
        assert bool(torch.isfinite(terms).all() and torch.isfinite(differences).all())
        self.field_terms += terms.detach().sum(0).double()
        self.field_differences += (differences.detach() * eligible).sum(0).double()
        self.field_counts += eligible.sum(0).double()
        return (*native_outputs, terms.sum())

    def add(self, native_loss, auxiliary_sum, world):
        if self.weight == 0:
            return native_loss
        return native_loss + world * self.weight * auxiliary_sum / self.denominator.clamp_min(1)

    def micro_done(self, auxiliary_sum):
        assert self.current is not None
        self.auxiliary_sum += auxiliary_sum.detach().double()
        self.current = None

    def update(self, module, optimizer, scheduler, step, epoch, next_batch, rank, world, device, denominators, sums, grad_norms):
        assert self.current is None and self.forward_count == len(self.prepared)
        assert step == self.window_start_step + 1
        record = {'step': step, 'epoch': epoch, 'next_batch': next_batch,
            'native_denominators': denominators.cpu().tolist(), 'native_local_loss_sums': sums.cpu().tolist(),
            'source_preference_denominator': float(self.denominator), 'source_preference_local_sum': float(self.auxiliary_sum),
            'field_names': list(CONTENT_FIELDS), 'field_local_terms': self.field_terms.cpu().tolist(),
            'field_local_donor_minus_clean_ce': self.field_differences.cpu().tolist(),
            'field_local_eligible_counts': self.field_counts.cpu().tolist(), 'grad_norms_before_clip': grad_norms,
            'learning_rates_after_scheduler': [g['lr'] for g in optimizer.param_groups],
            'microbatches': [{k: cpu_copy(x[k]).tolist() if isinstance(x[k], torch.Tensor) else x[k]
                             for k in ['pair_ids', 'pair_ordinals', 'source_sample_ids', 'donor', 'valid_donor']} for x in self.prepared]}
        if rank == 0:
            with (self.out / 'grounding_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
        if step == self.spec['parent_step'] + self.spec['replay_updates']:
            snapshot = {'diffusion_state_dict': cpu_copy(module.diffusion.state_dict()),
                'editing_ar_specific_state_dict': cpu_copy(ar_specific_state(module.ar)),
                'optimizer': cpu_copy(optimizer.state_dict()), 'scheduler': cpu_copy(scheduler.state_dict()),
                'rng': cpu_copy(_capture_rank_rng_state(rank=rank, device=device)),
                'global_step': step, 'epoch': epoch, 'next_batch': next_batch, 'window_record': record}
            path = self.out / f'prefix-rank{rank}.pt'
            assert not path.exists()
            temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
            with temporary.open('xb') as stream:
                torch.save(snapshot, stream); stream.flush(); os.fsync(stream.fileno())
            temporary.replace(path)
            write(self.out / f'prefix-rank{rank}.json', {'step': step, 'snapshot': str(path), 'sha256': sha(path), 'quality_gate_passed': False})
            self.snapshot_steps.append(step)
            del snapshot
            if self.phase_name == 'replay':
                raise PrefixReplayComplete()
        self.prepared = []


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-plan', type=Path, required=True)
    p.add_argument('--phase', choices=['control', 'replay', 'candidate'], required=True)
    args, native_args = p.parse_known_args()
    plan = json.loads(args.experiment_plan.read_text())
    spec = json.loads(Path(plan['experiment_spec']).read_text())
    for path, expected in plan['source_sha256'].items():
        assert sha(path) == expected, path
    phase = spec['phases'][args.phase]
    config = json.loads(Path(phase['config']).read_text())
    assert config['source_grounding']['weight'] == (spec['candidate_weight'] if args.phase == 'candidate' else 0.)
    assert os.environ['EDITING_GPUS'] == '7' and int(os.environ['WORLD_SIZE']) == 1
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8' and os.environ['FLA_CACHE_MODE'] == 'disabled'
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rank = int(os.environ['RANK'])
    assert rank == 0
    hooks = TrainingHooks(spec, args.phase, config)
    hooks.out.mkdir(parents=True, exist_ok=True)
    tree, counts = trainer_tree()
    write(hooks.out / 'NATIVE_AST_QA.json', {'counts': counts, 'reverse_transform_recovers_native_main_exactly': True,
        'native_source_sha256': sha(native.__file__), 'extension_sources': {k: v for k, v in source_inventory().items() if k.startswith(RELATIVE_ROOT)}})
    case = {'name': f'rank{rank}', 'mode': 'capture' if args.phase == 'control' else 'pin'}
    if args.phase != 'control':
        case['reference_case'] = str(Path(spec['phases']['control']['artifacts']) / f'rank{rank}')
    finish_catalog = numerics.install_autotune_observer(case, hooks.out)
    namespace = dict(native.__dict__, _distributed=allocated_runtime.distributed, run_source_inventory=source_inventory,
        _grounding_contract=amend_contract, _grounding_wrap=hooks.wrap, _grounding_window=hooks.window,
        _grounding_bind=hooks.bind, _grounding_add=hooks.add, _grounding_micro_done=hooks.micro_done,
        _grounding_update=hooks.update)
    exec(compile(tree, __file__ + '::native_main_with_training_hooks', 'exec'), namespace)
    sys.argv = [native.__file__, *native_args]
    prefix_stopped = False
    try:
        try:
            namespace['main']()
        except PrefixReplayComplete:
            assert args.phase == 'replay'
            prefix_stopped = True
            dist.barrier()
        finish_catalog()
        before = json.loads((hooks.out / 'FROZEN_STATE_BEFORE.json').read_text())
        after = {'qwen': state_hash(hooks.module.ar.instruction_conditioner.model),
            'clap': state_hash(hooks.module.ar.source_clap_model), 'quality_gate_passed': False}
        assert before == after
        assert hooks.snapshot_steps == [spec['parent_step'] + spec['replay_updates']]
        write(hooks.out / f'RUNTIME_COMPLETE_rank{rank}.json', {'at': datetime.now().astimezone().isoformat(),
            'phase': args.phase, 'frozen_states_unchanged': after, 'prefix_replay_stopped_as_declared': prefix_stopped,
            'snapshot_steps': hooks.snapshot_steps, 'autotune_catalog_sha256': sha(hooks.out / f'rank{rank}/AUTOTUNE_CATALOG.json'),
            'quality_gate_passed': False, 'independent_test_used': False, 'notifications_enabled': False})
    finally:
        if hooks.module is not None:
            hooks.module.forward = hooks.original_forward
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
