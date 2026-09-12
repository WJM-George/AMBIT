"""Preserve observed FLA choices and capture genuinely new native keys.

This only inserts the existing two observer hooks into the native Autotuner.run
AST. Known keys use their previously observed configuration. Unknown keys execute
the original benchmark/configuration selection; no configuration is inferred.
"""
from datetime import datetime
import inspect
import json
from pathlib import Path

from . import state
import probe_clap44_fla_autotune_replay_v1 as base

_native_install = base.install_autotune_observer


def install(case, stage):
    if case['mode'] != 'inherit_capture':
        return _native_install(case, stage)
    reference_path = Path(case['reference_case']) / 'AUTOTUNE_CATALOG.json'
    reference = state.read(reference_path)
    if not reference['records'] or reference['mode'] != 'capture':
        raise RuntimeError('Continuation requires actual captured native choices')
    inherited = reference['records']; observed = {}; calls = 0
    for value in inherited.values():
        if state.sha(value['source_file']) != value['source_sha256']:
            raise RuntimeError('An inherited native kernel implementation changed')
    original = base.Autotuner.run; tree = base.autotune_tree()
    output = Path(stage) / case['name']

    def snapshot(final=False):
        # Include unused but previously observed keys for subsequent recovery.
        records = {k: {**v, 'calls': 0, 'origin': 'inherited_not_called_in_this_segment'} for k, v in inherited.items()}
        records.update(observed)
        result = {'at': datetime.now().astimezone().isoformat(), 'mode': 'capture',
            'capture_policy': 'Pin previously observed keys, preserve native selection for new keys.',
            'inherited_catalog': str(reference_path), 'inherited_catalog_sha256': state.sha(reference_path),
            'records': records, 'observed_calls': calls,
            'native_autotuner_ast_preserved_except_declared_hooks': True,
            'newly_observed_keys': sorted(set(observed) - set(inherited)),
            'quality_gate_passed': False, 'notifications_enabled': False}
        state.write(output / ('AUTOTUNE_CATALOG.json' if final else 'LIVE_AUTOTUNE_CATALOG.json'), result, replace=not final)

    def identity(self, local):
        function = self.base_fn
        if not function.__module__.startswith('fla.'):
            return None, None
        descriptor = {'function': function.__module__ + '.' + function.__qualname__,
                      'native_autotune_key': list(local['key']) if 'key' in local else None}
        return json.dumps(descriptor, sort_keys=True, separators=(',', ':')), descriptor

    def pin(self, local):
        token, descriptor = identity(self, local)
        if token not in inherited:
            return
        expected = inherited[token]['config']
        configs = [config for config in self.configs if config.all_kwargs() == expected]
        if len(configs) != 1:
            raise RuntimeError(f'Inherited choice is not a unique native configuration: {descriptor}')
        self.cache[local['key']] = configs[0]

    def observe(self, local):
        nonlocal calls
        token, descriptor = identity(self, local)
        if token is None:
            return
        config = self.best_config
        if config.pre_hook is not None:
            raise RuntimeError('Unexpected per-configuration side effect')
        values = config.all_kwargs(); calls += 1
        if token in inherited and values != inherited[token]['config']:
            raise RuntimeError('Native continuation changed an inherited configuration')
        if token in observed:
            if observed[token]['config'] != values:
                raise RuntimeError('One native cache key selected different configurations')
            observed[token]['calls'] += 1
        else:
            observed[token] = {**descriptor, 'config': values, 'calls': 1,
                'source_file': inspect.getsourcefile(self.base_fn),
                'source_sha256': state.sha(inspect.getsourcefile(self.base_fn)),
                'native_candidate_configs': len(self.configs),
                'origin': 'inherited_observed_choice' if token in inherited else 'native_benchmark_on_actual_new_key',
                'first_call_tensor_layouts': {name: {'shape': list(arg.shape), 'stride': list(arg.stride()), 'dtype': str(arg.dtype)}
                    for name, arg in {**dict(zip(self.arg_names, local['args'])), **local['kwargs']}.items() if isinstance(arg, base.torch.Tensor)}}
            # Preserve every newly encountered choice even if this process fails.
            snapshot()

    namespace = dict(original.__globals__, _pin_fla_config=pin, _observe_fla_config=observe)
    exec(compile(tree, str(Path(__file__)) + '::native_observed_continuation', 'exec'), namespace)
    base.Autotuner.run = namespace['run']
    snapshot()

    def finish():
        base.Autotuner.run = original
        if not observed or not calls:
            raise RuntimeError('No actual native kernel calls were observed')
        snapshot(final=True)
    return finish
