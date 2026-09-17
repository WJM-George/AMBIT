"""Native AR-only training entry with explicit factual-CLAP dependency binding.

The one-rank adaptation and three-rank AR implementation retain their existing
loss, sampler, optimizer transfer and checkpoint code. This entry only changes
frozen encoder loading/provenance and checks effect acceptance before startup.
"""
import ast
import copy
import inspect
import os
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.t2a.experiments.ar_instruction_t200_v2 import runtime as single, distributed as three
from scripts.t2a.experiments.ar_factual_clap_v1 import integration, training_binding as binding


def source_inventory(world):
    original = single.source_inventory if world == 1 else three.source_inventory
    paths = [Path(__file__), Path(binding.__file__), Path(integration.__file__),
        ROOT / 'scripts/t2a/experiments/clap_factual50k_v1/evaluation.py',
        ROOT / 'scripts/t2a/experiments/clap_factual50k_v1/checkpoint.py',
        ROOT / 'scripts/t2a/experiments/clap_scene_supervision_v1/state.py']
    return {**original(), **{str(p.relative_to(ROOT)): integration.file_sha256(p) for p in paths}}


def trainer_tree(world):
    original, counts = (single.trainer_tree() if world == 1 else three.trainer_tree())
    before = "clap_identity['contract']['preflight_sha256'] != file_sha256(args.preflight)"
    changed_count = 0
    class Replace(ast.NodeTransformer):
        def visit_Compare(self, node):
            nonlocal changed_count
            if ast.unparse(node) == before:
                changed_count += 1
                replacement = ast.parse('not _clap_preflight_matches(clap_identity, args.preflight)', mode='eval').body
                replacement._clap_original = copy.deepcopy(node)
                return replacement
            return self.generic_visit(node)
    changed = Replace().visit(copy.deepcopy(original))
    assert changed_count == 1
    class Undo(ast.NodeTransformer):
        def generic_visit(self, node):
            return node._clap_original if hasattr(node, '_clap_original') else super().generic_visit(node)
    assert ast.dump(Undo().visit(copy.deepcopy(changed)), include_attributes=False) == ast.dump(original, include_attributes=False)
    return ast.fix_missing_locations(changed), {**counts, 'explicit_factual_clap_preflight': changed_count}


def entry_tree(world):
    original = (ast.parse(textwrap.dedent(inspect.getsource(single.main))) if world == 1 else three.entry_tree()[0])
    counts = {'effect_before_GPU_setup': 0, 'frozen_encoder_namespace': 0}
    class Replace(ast.NodeTransformer):
        def visit_Assign(self, node):
            node = self.generic_visit(node)
            target = ast.unparse(node.targets[0])
            if target == 'cfg':
                counts['effect_before_GPU_setup'] += 1
                extra = ast.parse('_effect_startup_check(cfg)').body[0]
                extra._clap_remove = True
                return [node, extra]
            if target == 'namespace':
                counts['frozen_encoder_namespace'] += 1
                previous = copy.deepcopy(node)
                node.value = ast.Call(ast.Name('_clap_training_namespace', ast.Load()),
                    [node.value, ast.Name('cfg', ast.Load()), ast.Name('hooks', ast.Load())], [])
                node._clap_original = previous
            return node
    changed = Replace().visit(copy.deepcopy(original))
    assert counts == {'effect_before_GPU_setup': 1, 'frozen_encoder_namespace': 1}
    class Undo(ast.NodeTransformer):
        def generic_visit(self, node):
            if getattr(node, '_clap_remove', False): return None
            return node._clap_original if hasattr(node, '_clap_original') else super().generic_visit(node)
    assert ast.dump(Undo().visit(copy.deepcopy(changed)), include_attributes=False) == ast.dump(original, include_attributes=False)
    return ast.fix_missing_locations(changed), counts


def main():
    world = int(os.environ.get('WORLD_SIZE', '0'))
    if world not in (1, 3):
        raise RuntimeError('AR adaptation requires world1/GPU7; full AR requires world3/GPU5-7')
    tree, _ = entry_tree(world)
    namespace = {**single.main.__globals__, '__file__': __file__,
        'Hooks': single.Hooks if world == 1 else three.Hooks,
        'trainer_tree': lambda: trainer_tree(world),
        'source_inventory': lambda: source_inventory(world),
        'amend_contract': single.amend_contract if world == 1 else three.amend_contract,
        '_check_allocation': three.check_allocation,
        '_distributed_namespace': three.distributed_namespace,
        '_effect_startup_check': binding.effect_startup_check,
        '_clap_training_namespace': binding.training_namespace}
    exec(compile(tree, __file__ + '::factual_clap_AR_entry', 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
