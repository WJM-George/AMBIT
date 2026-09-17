"""Frozen native FLA observer functions, reused under the AR source contract."""
import ast
import copy
from datetime import datetime
import inspect
import json
from pathlib import Path
import textwrap
import torch
from triton.runtime.autotuner import Autotuner
from scripts.t2a.experiments.ar_source_grounding_v1.evidence import write, sha


def autotune_tree():
    original = ast.parse(textwrap.dedent(inspect.getsource(Autotuner.run)))
    counts = {'_pin_fla_config': 0, '_observe_fla_config': 0}

    class Insert(ast.NodeTransformer):
        def visit_If(self, node):
            node = self.generic_visit(node)
            if ast.dump(node.test, include_attributes=False) == ast.dump(ast.parse('key not in self.cache', mode='eval').body, include_attributes=False):
                counts['_pin_fla_config'] += 1
                return [ast.parse('_pin_fla_config(self, locals())').body[0], node]
            return node

        def visit_Assign(self, node):
            node = self.generic_visit(node)
            if isinstance(node.targets[0], ast.Attribute) and isinstance(node.targets[0].value, ast.Name) and node.targets[0].value.id == 'self' and node.targets[0].attr == 'best_config':
                counts['_observe_fla_config'] += 1
                return [node, ast.parse('_observe_fla_config(self, locals())').body[0]]
            return node

    tree = Insert().visit(copy.deepcopy(original)); assert all(n == 1 for n in counts.values()), counts

    class Strip(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in counts:
                return None
            return self.generic_visit(node)

    assert ast.dump(Strip().visit(copy.deepcopy(tree)), include_attributes=False) == ast.dump(original, include_attributes=False)
    return ast.fix_missing_locations(tree)


def install_autotune_observer(case, stage):
    original_run = Autotuner.run
    tree = autotune_tree(); records = {}; total_calls = 0
    reference = None
    if case['mode'] == 'pin':
        reference_path = stage / case['reference_case'] / 'AUTOTUNE_CATALOG.json'
        reference = json.loads(reference_path.read_text())
        assert reference['mode'] == 'capture' and reference['records']
        reference = reference['records']

    def identity(self, local):
        function = self.base_fn
        if not function.__module__.startswith('fla.'):
            return None, None
        key = list(local['key']) if 'key' in local else None
        descriptor = {'function': function.__module__ + '.' + function.__qualname__, 'native_autotune_key': key}
        token = json.dumps(descriptor, sort_keys=True, separators=(',', ':'))
        return token, descriptor

    def pin(self, local):
        token, descriptor = identity(self, local)
        if token is None or reference is None:
            return
        if token not in reference:
            write(stage / case['name'] / 'MISSING_PINNED_CONFIG.json', {'kernel': descriptor,
                'scope': 'Stop before a new native configuration is benchmarked; do not infer a config for an unobserved key.', 'quality_gate_passed': False})
            raise RuntimeError('No captured native FLA autotune choice for this exact key')
        expected = reference[token]['config']
        candidates = [config for config in self.configs if config.all_kwargs() == expected]
        assert len(candidates) == 1, (descriptor, expected)
        # The original native run uses this exact cache key and configuration.
        # No kernel implementation, tensor argument or model parameter changes.
        self.cache[local['key']] = candidates[0]

    def observe(self, local):
        nonlocal total_calls
        token, descriptor = identity(self, local)
        if token is None:
            return
        config = self.best_config
        assert config.pre_hook is None, 'Preserve any unexpected per-config side effect before pinning'
        value = config.all_kwargs(); total_calls += 1
        if reference is not None:
            assert token in reference and value == reference[token]['config']
        if token in records:
            assert records[token]['config'] == value, 'Native config changed within one exact cache key'
            records[token]['calls'] += 1
        else:
            records[token] = {**descriptor, 'config': value, 'calls': 1,
                'source_file': inspect.getsourcefile(self.base_fn), 'source_sha256': sha(inspect.getsourcefile(self.base_fn)),
                'native_candidate_configs': len(self.configs),
                'first_call_tensor_layouts': {name: {'shape': list(arg.shape), 'stride': list(arg.stride()), 'dtype': str(arg.dtype)}
                    for name, arg in {**dict(zip(self.arg_names, local['args'])), **local['kwargs']}.items() if isinstance(arg, torch.Tensor)}}

    namespace = dict(original_run.__globals__); namespace.update(_pin_fla_config=pin, _observe_fla_config=observe)
    exec(compile(tree, str(Path(__file__)) + '::native_autotuner_observer', 'exec'), namespace)
    Autotuner.run = namespace['run']

    def finish():
        Autotuner.run = original_run
        assert records and total_calls
        write(stage / case['name'] / 'AUTOTUNE_CATALOG.json', {'at': datetime.now().astimezone().isoformat(),
            'mode': case['mode'], 'reference_case': case.get('reference_case'), 'records': records,
            'observed_calls': total_calls, 'native_autotuner_ast_preserved_except_declared_hooks': True,
            'pin_scope': 'Pin mode only preloads the exact native cache key with the previously selected member of the same native configuration list. Missing keys stop before autotuning. Capture mode observes after normal native selection.',
            'quality_gate_passed': False, 'notifications_enabled': False})

    return finish
