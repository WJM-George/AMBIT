"""Bounded native4 versus2+2 proof for the fixed DDP bucket configuration."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.t2a.experiments.ar_instruction_t200_v3 import policy, resume_proof, review_resume
from scripts.t2a.experiments.ar_source_grounding_v1.evidence import sha, write


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    plan_path = parser.parse_args().plan
    plan = policy.read(plan_path)
    for path, expected in plan['source_sha256'].items(): assert sha(path) == expected, path
    try:
        for name in ('continuous4', 'split2', 'resumed2'):
            command = resume_proof.command(plan_path, name)
            index = command.index(str(ROOT / 'scripts/t2a/experiments/ar_instruction_t200_v3/runtime.py'))
            command[index] = str(ROOT / 'scripts/t2a/experiments/ar_instruction_t200_v3/fixed_ddp.py')
            print(json.dumps({'at': datetime.now().astimezone().isoformat(), 'phase': name, 'status': 'starting'}), flush=True)
            with (plan_path.parent / f'{name}.log').open('x') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(json.dumps({'at': datetime.now().astimezone().isoformat(), 'phase': name, 'status': 'complete'}), flush=True)
        result = review_resume.review(plan_path.parent)
        write(plan_path.parent / 'REVIEW.json', result)
        print(json.dumps({'three_rank_native_resume_exact': True}), flush=True)
    except Exception as error:
        write(plan_path.parent / 'FAILURE.json', {'at': datetime.now().astimezone().isoformat(), 'error': repr(error), 'full_AR_expansion_allowed': False})
        raise


if __name__ == '__main__': main()
