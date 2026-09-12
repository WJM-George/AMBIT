#!/usr/bin/env python3
"""Run the frozen natural-request evaluator with truthful test provenance.

No scoring function, matching rule, tolerance or threshold is changed. The old
generic helper's validation-only metadata is replaced on newly written files.
"""
import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluator-snapshot', type=Path, required=True)
    args, helper_args = parser.parse_known_args()
    path = args.evaluator_snapshot / 'scripts/t2a/diagnostics/evaluate_generation_ar_natural_candidates.py'
    sys.path.insert(0, str(args.evaluator_snapshot))
    spec = importlib.util.spec_from_file_location('frozen_request_evaluator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old_write = module.write
    adapter_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def write(output, value):
        if isinstance(value, dict):
            value = {**value, 'test_used': True, 'test_metadata_adapter_sha256': adapter_sha}
        return old_write(output, value)

    module.write = write
    # Reuse the helper's own argument surface through an isolated execution
    # of its CLI section, with its actual function objects and globals.
    source = path.read_text()
    cli = source.split("if __name__=='__main__':", 1)
    if len(cli) != 2:
        raise RuntimeError('Frozen evaluator CLI layout changed')
    import textwrap
    sys.argv = [str(path), *helper_args]
    exec(compile(textwrap.dedent(cli[1]), str(path), 'exec'), module.__dict__)


if __name__ == '__main__':
    main()
