"""Keep the V1 training method and bind deterministic native FA backward."""
import argparse
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.t2a.experiments.ar_source_grounding_v1 import runtime as original
from scripts.t2a.experiments.ar_source_grounding_v1.evidence import sha, write
from stable_audio_tools.models import transformer

RELATIVE_ROOT = 'scripts/t2a/experiments/ar_source_grounding_v2'
ORIGINAL_INVENTORY = original.source_inventory


def source_inventory():
    return {**ORIGINAL_INVENTORY(), **{f'{RELATIVE_ROOT}/{name}': sha(ROOT / RELATIVE_ROOT / name)
            for name in ['runtime.py', 'ORIGINS.json']}}


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--experiment-plan', type=Path, required=True)
    p.add_argument('--phase', required=True)
    args, _ = p.parse_known_args()
    plan = json.loads(args.experiment_plan.read_text())
    spec = json.loads(Path(plan['experiment_spec']).read_text())
    phase = spec['phases'][args.phase]
    cfg = json.loads(Path(phase['config']).read_text())
    policy = cfg['numerical_execution']
    assert policy['schema'] == 'editing_ar_flashattention_deterministic_backward_v1'
    assert policy['deterministic_backward'] is True
    for path, expected in policy['implementation_sha256'].items(): assert sha(path) == expected, path
    functions = {name: getattr(transformer, name) for name in ['flash_attn_func', 'flash_attn_varlen_func']}
    calls = {name: 0 for name in functions}
    def wrap(name, function):
        signature = inspect.signature(function)
        assert signature.parameters['deterministic'].default is False
        def invoke(*values, **keywords):
            bound = signature.bind_partial(*values, **keywords)
            assert 'deterministic' not in bound.arguments, 'Underlying native call contract changed'
            keywords['deterministic'] = True
            calls[name] += 1
            return function(*values, **keywords)
        return invoke
    prior_inventory = original.source_inventory
    for name, function in functions.items(): setattr(transformer, name, wrap(name, function))
    original.source_inventory = source_inventory
    try:
        original.main()
        assert sum(calls.values()) > 0
        write(Path(phase['artifacts']) / 'FLASH_BACKWARD_VERIFIED.json', {
            'policy': policy, 'calls': calls, 'all_observed_native_FA_calls_deterministic_backward': True,
            'process_local_keyword_override_only': True, 'native_core_source_files_unchanged': True,
            'quality_gate_passed': False})
    finally:
        original.source_inventory = prior_inventory
        for name, function in functions.items(): setattr(transformer, name, function)


if __name__ == '__main__':
    main()
