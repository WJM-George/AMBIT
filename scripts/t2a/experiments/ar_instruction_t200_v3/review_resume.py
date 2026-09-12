"""Review completed native restart evidence using its actual RF field name."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.t2a.experiments.ar_source_grounding_v1.evidence import sha, write


def review(stage):
    refs = {str(Path(__file__)): sha(__file__)}
    def read(path):
        refs[str(path)] = sha(path)
        return json.loads(path.read_text())
    def windows(path):
        refs[str(path)] = sha(path)
        return [json.loads(line) for line in path.read_text().splitlines()]
    spec = read(stage / 'SPEC.json')
    comparisons = []
    for rank in range(3):
        dirs = {name: Path(spec['phases'][name]['artifacts']) / f'rank{rank}' for name in ('continuous4', 'split2', 'resumed2')}
        for directory in dirs.values():
            done = read(directory / 'RUNTIME_COMPLETE.json')
            assert done['all_frozen_parameters_exact'] and done['RF_training_calls'] == 0
        for step, arm in ((5252, 'split2'), (5254, 'resumed2')):
            left = read(dirs['continuous4'] / f'PREFIX_FINGERPRINT_step{step}.json')
            right = read(dirs[arm] / f'PREFIX_FINGERPRINT_step{step}.json')
            assert left == right, (rank, step, left['sha256'], right['sha256'])
            comparisons.append({'rank': rank, 'step': step, 'complete_state_sha256': left['sha256']})
        assert windows(dirs['continuous4'] / 'training_windows.jsonl') == windows(dirs['split2'] / 'training_windows.jsonl') + windows(dirs['resumed2'] / 'training_windows.jsonl'), rank
        loaded = read(dirs['resumed2'] / 'NATIVE_RESUME_LOADED.json')
        assert loaded['native_loader_all_checks_enabled'] and loaded['checkpoint']['step'] == 5252
    return {'at': datetime.now().astimezone().isoformat(), 'schema': 'AR_three_rank_native_restart_review_v3',
        'three_rank_native_resume_exact': True, 'continuous_and_restarted_windows_exact': True,
        'compared_state': 'Every model and Adam tensor, scheduler, all-rank RNG, epoch and next batch.',
        'results': comparisons, 'source_sha256': refs,
        'reviewer_correction': 'Read the actual RF_training_calls field; no GPU output or bound training source changed.',
        'full_AR_expansion_allowed': False, 'AR_quality_gate_passed': False, 'independent_test_used': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', type=Path, required=True)
    stage = parser.parse_args().stage
    result = review(stage); write(stage / 'REVIEW.json', result)
    print(json.dumps({'three_rank_native_resume_exact': True, 'review': str(stage / 'REVIEW.json')}))
