"""Evaluate new AR candidates after a truthful native-encoder historical replay.

The old evaluator remains immutable. Its scoring, source ablations and decoder
are reused; the historical and candidate frozen encoders have separate roles.
"""
import argparse
import ast
import copy
from datetime import datetime
import inspect
from pathlib import Path
import sys
import textwrap

REPO = Path(__file__).resolve().parents[4]
OPS = Path('/mnt/sdb/audio_dataset/sceneplan_transfusion_editing_v1/materialized/logs/ar_clap_task_20260906')
sys.path[:0] = [str(REPO), str(OPS / 'diagnostics'), str(OPS)]
import ar_t200_development as t200
from scripts.t2a.experiments.ar_factual_clap_v1 import integration, training_binding
from scripts.t2a.experiments.ar_instruction_t200_v1.fingerprints import fingerprint

grounded, native = t200.grounded, t200.native
read, sha, write = grounded.read, native.sha, native.write
SCHEMA = 'editing_ar_factual_encoder_development_v1'
ROLES_SCHEMA = 'editing_ar_historical_and_candidate_encoder_roles_v1'


def stamp(): return datetime.now().astimezone().isoformat()


def checkpoint_fields(reference):
    return {key: reference[key] for key in ('path', 'sha256', 'step')}


def load_encoder_role(reference, preflight, device='cpu'):
    model, identity, dependency = integration.load_source_encoder(reference,
        preflight_path=preflight, device=device)
    assert checkpoint_fields(identity) == checkpoint_fields(reference)
    return model, identity, dependency


def replace_encoder(module, reference, *, preflight, device):
    """Replace only the frozen external audio encoder, preserving AR/Qwen weights.

    This component operation does not claim that the AR was trained with the
    replacement encoder. The evaluation entry checks its real run contract.
    """
    before = fingerprint({'diffusion': module.diffusion.state_dict(),
        'adapters': integration.ar_specific_state(module.ar)})
    instruction = module.ar.instruction_conditioner
    model, identity, dependency = load_encoder_role(reference, preflight, device)
    module.ar.source_clap_model = model
    assert module.ar.instruction_conditioner is instruction
    assert fingerprint({'diffusion': module.diffusion.state_dict(),
        'adapters': integration.ar_specific_state(module.ar)}) == before
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    return identity, dependency


def validate_roles(plan, current, parent):
    if plan['schema'] != SCHEMA or plan['clap_roles']['schema'] != ROLES_SCHEMA:
        raise RuntimeError('Explicit historical/candidate encoder roles are required')
    roles = plan['clap_roles']; historical, candidate = roles['historical'], roles['candidate']
    if (historical['format'] != 'native_clap44' or historical['step'] != 20000
            or historical['sha256'] != training_binding.NATIVE20K_SHA):
        raise RuntimeError('Historical AR replay requires the protected native20k encoder')
    if parent['clap_checkpoint'] != checkpoint_fields(historical):
        raise RuntimeError('Historical AR and historical encoder identities differ')
    if current['clap_checkpoint'] != checkpoint_fields(candidate) or plan['frozen_clap'] != current['clap_checkpoint']:
        raise RuntimeError('Candidate AR contract and declared encoder differ')
    for key in ('model_config', 'model_config_sha256', 'base_selection', 'codec', 'codec_sha256',
                'indices', 'variant', 'training_mode'):
        if current[key] != parent[key]: raise RuntimeError(f'Historical/candidate AR architecture or data differs: {key}')
    for value in (current, parent):
        if (value['training_mode'] != 'ar_pretrain' or value['variant'] != 'global_and_sequence'
                or value['runtime_inputs'] != ['source_foa_latent', 'raw_edit_request']
                or any(value[k] is not False for k in ('old_plan_input', 'source_caption_input',
                    'target_audio_ar_input', 'independent_test_used'))):
            raise RuntimeError('AR development must retain source-audio and raw-request input roles')
    step = plan['checkpoint']['step']
    if plan['candidate_steps'] != [step] or plan['free_generation_step'] != step:
        raise RuntimeError('This matched development stage evaluates one identified AR candidate')
    if candidate['format'] == 'factual50k':
        binding = current['config']['clap_dependency']
        if binding != roles['candidate_effect_binding'] or binding['checkpoint'] != candidate:
            raise RuntimeError('Evaluation selection differs from the factual AR training binding')
        dependency = current['clap_dependency']
        if (dependency['format'] != 'factual50k' or dependency['checkpoint'] != checkpoint_fields(candidate)
                or dependency['training_contract_sha256'] != candidate['contract_sha256']
                or current['clap_effect_selection'] != binding['effect_selection']):
            raise RuntimeError('AR factual encoder provenance is incomplete')
    elif candidate['format'] != 'native_clap44' or candidate != historical:
        raise RuntimeError('The retained native control must use its actual original encoder')


class EncoderRoles:
    def __init__(self, plan):
        self.plan = plan
        self.current = read(plan['run_contract'])
        self.development = read(plan['development_plan'])
        self.parent = read(self.development['run_contract'])
        validate_roles(plan, self.current, self.parent)
        self.roles = plan['clap_roles']
        self.phase = 'uninitialized'
        self.historical_dependency = self.candidate_dependency = None
        self.before = {}

    def historical_encoder(self, contract, plan):
        assert plan == self.plan and contract == self.current and self.phase == 'uninitialized'
        model, identity, dependency = load_encoder_role(self.roles['historical'], self.roles['preflight'])
        self.historical_dependency = dependency; self.phase = 'historical_encoder_loaded'
        return model, identity

    def replay_parent(self, module, dataset, donors, codec, plan, out, device):
        assert self.phase == 'historical_encoder_loaded' and plan == self.plan
        assert fingerprint(module.ar.source_clap_model.state_dict()) == self.historical_dependency['encoder_state_fingerprint']
        function, _ = replay_function()
        function(module, dataset, donors, codec, plan, out, device)
        assert read(out / 'PARENT_FREE_REPLAY_VERIFIED.json')['all100_native_records_exact']
        assert fingerprint(module.ar.source_clap_model.state_dict()) == self.historical_dependency['encoder_state_fingerprint']
        self.phase = 'historical_replay_verified_in_this_process'
        identity, dependency = replace_encoder(module, self.roles['candidate'],
            preflight=self.roles['preflight'], device=device)
        if self.roles['candidate']['format'] == 'factual50k':
            assert dependency == self.current['clap_dependency']
        self.candidate_dependency = dependency; self.phase = 'candidate_encoder_loaded'
        write(out / 'ENCODER_ROLE_SWITCH.json', {'at': stamp(),
            'historical': self.historical_dependency, 'candidate': dependency,
            'native_parent_replay_completed_with_original_encoder_in_this_process': True,
            'candidate_weights_not_relabelled_as_historical_native20k': True,
            'AR_diffusion_adapters_and_Qwen_unchanged_by_encoder_switch': True,
            'selected_encoder_checkpoint': checkpoint_fields(identity), 'quality_gate_passed': False})

    def capture(self, module, identity, plan):
        assert self.phase == 'candidate_encoder_loaded' and identity == plan['checkpoint']
        assert fingerprint(module.ar.source_clap_model.state_dict()) == self.candidate_dependency['encoder_state_fingerprint']
        self.before.update(module=grounded.controls.state_hash(module),
            qwen=grounded.controls.state_hash(module.ar.instruction_conditioner.model))

    def finish(self, module, codec, dataset, donors, plan, out, device):
        assert self.phase == 'candidate_encoder_loaded' and self.before
        function, _ = finish_function(self.before)
        function(module, codec, dataset, donors, plan, out, device)
        assert fingerprint(module.ar.source_clap_model.state_dict()) == self.candidate_dependency['encoder_state_fingerprint']
        self.phase = 'candidate_diagnostics_complete'
        write(out / 'ENCODER_ROLE_VERIFICATION.json', {'at': stamp(),
            'historical_replay_encoder': self.historical_dependency['checkpoint'],
            'candidate_evaluation_encoder': self.candidate_dependency['checkpoint'],
            'candidate_encoder_unchanged': True, 'historical_and_candidate_roles_remained_separate': True,
            'quality_gate_passed': False, 'independent_test_used': False})


def reversible(tree, transformer):
    original = copy.deepcopy(tree)
    changed = transformer.visit(copy.deepcopy(tree))
    class Undo(ast.NodeTransformer):
        def generic_visit(self, node):
            return node._factual_original if hasattr(node, '_factual_original') else super().generic_visit(node)
    assert ast.dump(Undo().visit(copy.deepcopy(changed)), include_attributes=False) == ast.dump(original, include_attributes=False)
    return ast.fix_missing_locations(changed)


def replay_function():
    original = ast.parse(textwrap.dedent(inspect.getsource(grounded.restore_parent)))
    counts = {'candidate_step': 0, 'explicit_encoder_roles_allow_different_checkpoint': 0}
    class Edit(ast.NodeTransformer):
        def visit_Assert(self, node):
            if ast.unparse(node.test) == "plan['candidate_steps'] == [5250] and plan['free_generation_step'] == 5250":
                counts['candidate_step'] += 1
                changed = ast.parse("assert plan['candidate_steps'] == [plan['checkpoint']['step']] and plan['free_generation_step'] == plan['checkpoint']['step']").body[0]
                changed._factual_original = copy.deepcopy(node); return changed
            return self.generic_visit(node)
        def visit_For(self, node):
            if (isinstance(node.iter, ast.List) and ast.unparse(node.target) == 'key'
                    and any(isinstance(v, ast.Constant) and v.value == 'clap_checkpoint' for v in node.iter.elts)):
                counts['explicit_encoder_roles_allow_different_checkpoint'] += 1
                before = copy.deepcopy(node)
                node.iter.elts = [v for v in node.iter.elts if not (isinstance(v, ast.Constant) and v.value == 'clap_checkpoint')]
                node._factual_original = before
            return self.generic_visit(node)
    tree = reversible(original, Edit()); assert all(v == 1 for v in counts.values()), counts
    namespace = dict(grounded.restore_parent.__globals__)
    exec(compile(tree, __file__ + '::native_parent_replay', 'exec'), namespace)
    return namespace['restore_parent'], counts


def finish_function(before):
    original = ast.parse(textwrap.dedent(inspect.getsource(grounded.finish)))
    counts = {'teacher_artifact_step': 0, 'repeat_step': 0}
    class Edit(ast.NodeTransformer):
        def visit_Constant(self, node):
            expression = None
            if node.value == 'teacher-forced-step-5250.json':
                counts['teacher_artifact_step'] += 1
                expression = "f'teacher-forced-step-{plan[\"free_generation_step\"]}.json'"
            elif node.value == 5250:
                counts['repeat_step'] += 1; expression = "plan['free_generation_step']"
            if expression:
                value = ast.parse(expression, mode='eval').body
                value._factual_original = copy.deepcopy(node); return value
            return node
    tree = reversible(original, Edit()); assert all(v == 1 for v in counts.values()), counts
    namespace = dict(grounded.finish.__globals__, BEFORE=before)
    exec(compile(tree, __file__ + '::native_source_controls', 'exec'), namespace)
    return namespace['finish'], counts


def instrument(roles):
    # Start from the already reversible native T200 evaluator, including its
    # historical prefix, T200 text overlay and full clean/shuffled source probes.
    _, old_counts = t200.instrument()
    original = ast.parse(textwrap.dedent(inspect.getsource(native.main)))
    # Rebuild the original T200 instrumentation through its own compiler, then
    # replace only the initial encoder expression in its preserved native tree.
    # The original insertion nodes are emitted by the same source-bound helper.
    counts = {'historical_encoder_initialization': 0}
    class InitialEncoder(ast.NodeTransformer):
        def visit_Assign(self, node):
            if ast.unparse(node.targets[0]) == '(clap, clap_identity)':
                assert ast.unparse(node.value.func) == 'load_clap44_checkpoint'
                counts['historical_encoder_initialization'] += 1
                saved = copy.deepcopy(node)
                node.value = ast.parse('_historical_encoder(contract,plan)', mode='eval').body
                node._factual_original = saved
            return self.generic_visit(node)
    tree = reversible(original, InitialEncoder())
    assert counts['historical_encoder_initialization'] == 1
    # Insert the exact three hooks in the existing T200 instrumenter. Its body
    # performs and asserts the reverse transform; only its native source input
    # is supplied as the above explicitly reviewed tree.
    compiler_tree = ast.parse(textwrap.dedent(inspect.getsource(t200.instrument)))
    replaced = 0
    class Input(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal replaced
            if ast.unparse(node.targets[0]) == 'original':
                replaced += 1
                saved = copy.deepcopy(node)
                node.value = ast.parse('_encoder_tree', mode='eval').body
                node._factual_original = saved
            return self.generic_visit(node)
    compiler_tree = reversible(compiler_tree, Input()); assert replaced == 1
    scope = dict(t200.instrument.__globals__, _encoder_tree=tree)
    exec(compile(compiler_tree, __file__ + '::T200_hook_composition', 'exec'), scope)
    function, repeated_counts = scope['instrument'](); assert repeated_counts == old_counts
    function.__globals__.update(_historical_encoder=roles.historical_encoder,
        _restore_parent=roles.replay_parent, _capture_model=roles.capture, _finish=roles.finish)
    return function, {**old_counts, **counts}


def verify_plan(plan):
    for path in (__file__, integration.__file__, training_binding.__file__,
                 native.__file__, grounded.__file__, t200.__file__):
        path = str(Path(path).resolve(strict=True))
        if plan['source_sha256'].get(path) != sha(path):
            raise RuntimeError(f'Evaluation execution source is not bound: {path}')
    grounded.verify_files(plan)
    current = read(plan['run_contract']); dev = read(plan['development_plan']); parent = read(dev['run_contract'])
    validate_roles(plan, current, parent)
    roles = plan['clap_roles']
    if roles['candidate']['format'] == 'factual50k':
        receipt = training_binding.effect_startup_check(current['config'])
        assert receipt['factual_encoder_effect_accepted']
    else:
        cfg = {'clap_dependency': {'schema': training_binding.SCHEMA, 'checkpoint': roles['candidate'],
            'preflight': roles['preflight'], 'validation_report': roles['native20k_validation_report']}}
        training_binding.effect_startup_check(cfg)
    payload, identity = native.load_joint_checkpoint(plan['checkpoint']['checkpoint'], expected_contract=current)
    assert identity == plan['checkpoint']; del payload
    assert plan['physical_gpus'] == [7] and plan['instruction_view']['mode'] in ('native', 't200')
    assert plan['cases'] == dev['cases'] and plan['donors'] == dev['donors'] and len(plan['cases']) == 100


def main(path):
    plan = read(path); verify_plan(plan)
    roles = EncoderRoles(plan)
    function, _ = instrument(roles)
    sys.argv = [__file__, '--plan', str(path)]
    function()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); main(args.plan.resolve(strict=True))
