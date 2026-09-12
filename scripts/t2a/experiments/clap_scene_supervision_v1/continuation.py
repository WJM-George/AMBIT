"""Explicit bounded loss fork from a complete event250 state; no clock reset."""
import argparse
import ast
import copy
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from scripts.t2a.experiments.clap_scene_supervision_v1 import runtime as native, state, kernel_continuation


def check_contract_transfer(source, destination):
    """Allow the declared kind weight and provenance, never hidden state changes."""
    assert source['max_new_updates'] == 250 and destination['max_new_updates'] == 500
    assert source['method']['encoder_aux_enabled'] and destination['method']['encoder_aux_enabled']
    assert source['method']['event_weights']['kind'] == 1.
    assert destination['method']['event_weights']['kind'] in (1., 4.)
    left, right = copy.deepcopy(source), copy.deepcopy(destination)
    for key in ('experiment_arm', 'source_sha256', 'scope', 'continuation_provenance',
                'max_new_updates', 'readout_schedule'):
        left.pop(key); right.pop(key)
    for value in (left, right):
        value['method'].pop('control_definition')
        value['method']['event_weights'].pop('kind')
    assert left == right, 'Undeclared training, optimizer, schedule, data or topology change'
    assert all(destination['source_sha256'].get(k) == v for k, v in source['source_sha256'].items())


def load_training_origin(phase, contract):
    if phase.get('resume'):
        assert not phase.get('fork_parent')
        return state.load_checkpoint(phase['resume'], contract)
    parent = phase['fork_parent']; path = Path(parent['checkpoint'])
    original = state.read(path.parent / 'TRAIN_CONTRACT.json')
    assert state.sha(path.parent / 'TRAIN_CONTRACT.json') == parent['contract_sha256']
    check_contract_transfer(original, contract)
    payload, identity = state.load_checkpoint(path, original)
    assert identity == parent
    assert payload['step'] == 20250 and payload['new_updates'] == 250
    assert contract['continuation_provenance']['source_checkpoint'] == identity
    # The native runtime restores every numeric block and all rank RNG states,
    # then constructs and validates a new payload under the explicit new contract.
    return payload, identity


def compiled_runtime(install):
    """Three auditable orchestration changes; training arithmetic stays native."""
    original = ast.parse(inspect.getsource(native.run))
    changes = {
        "1 <= contract['max_new_updates'] <= 250": "1 <= contract['max_new_updates'] <= 500",
        "state.load_checkpoint(phase['resume'], contract)": "_load_training_origin(phase, contract)",
    }
    counts = dict.fromkeys(changes, 0); fork_condition_count = 0

    class Forward(ast.NodeTransformer):
        def visit(self, node):
            nonlocal fork_condition_count
            if isinstance(node, (ast.Compare, ast.Call)) and ast.unparse(node) in changes:
                key = ast.unparse(node); counts[key] += 1
                return ast.copy_location(ast.parse(changes[key], mode='eval').body, node)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "phase.get('resume')" and any(
                isinstance(item, ast.Assign) and ast.unparse(item.value) == "state.load_checkpoint(phase['resume'], contract)"
                for item in node.body):
                node.test = ast.parse("phase.get('resume') or phase.get('fork_parent')", mode='eval').body
                fork_condition_count += 1
            return super().visit(node)

    modified = Forward().visit(copy.deepcopy(original))
    assert all(v == 1 for v in counts.values()) and fork_condition_count == 1
    reverse = {v: k for k, v in changes.items()}

    class Reverse(ast.NodeTransformer):
        def visit(self, node):
            if isinstance(node, (ast.Compare, ast.Call)) and ast.unparse(node) in reverse:
                return ast.copy_location(ast.parse(reverse[ast.unparse(node)], mode='eval').body, node)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "phase.get('resume') or phase.get('fork_parent')":
                node.test = ast.parse("phase.get('resume')", mode='eval').body
            return super().visit(node)

    restored = Reverse().visit(copy.deepcopy(modified))
    assert ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False)
    namespace = dict(native.__dict__, _load_training_origin=load_training_origin,
        kernels=SimpleNamespace(autotune_tree=native.kernels.autotune_tree, install_autotune_observer=install))
    exec(compile(ast.fix_missing_locations(modified), __file__ + '::bounded_native_continuation', 'exec'), namespace)
    return namespace['run']


def main(plan_path, phase_name):
    plan = state.read(plan_path); phase = plan['phases'][phase_name]
    assert plan['trial_kind'] == 'kind_weight_repair_from_event250'
    contract = state.read(phase['contract'])
    assert contract['max_new_updates'] == 500
    assert phase['stop_after_new_updates'] in (252, 254, 500)
    assert bool(phase.get('resume')) != bool(phase.get('fork_parent'))
    if phase['stop_after_new_updates'] == 500:
        assert plan.get('prefix_review'), 'A longer repair needs the actual prefix/restart review'
        review = state.read(plan['prefix_review'])
        assert state.sha(plan['prefix_review']) == plan['prefix_review_sha256']
        assert review['both_arms_complete_continuous_vs_resume_states_exact']
        assert review['all_rank_data_RNG_and_training_schedule_matched']
        assert review['trial_contract_sha256'][contract['experiment_arm']] == state.sha(phase['contract'])
    def install(case, stage):
        if case['mode'] == 'inherit_capture':
            case = {**case, 'reference_case': str(Path(phase['kernel_reference']) / case['name'])}
        return kernel_continuation.install(case, stage)
    compiled_runtime(install)(plan_path, phase_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--phase', required=True); args = parser.parse_args()
    main(args.plan.resolve(strict=True), args.phase)
