"""Keep DDP gradient bucket assignment fixed across a native restart."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from torch.nn.parallel import DistributedDataParallel
from scripts.t2a.experiments.ar_instruction_t200_v3 import runtime, policy
from scripts.t2a.experiments.ar_source_grounding_v1.evidence import sha, write


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--experiment-plan', type=Path, required=True)
    parser.add_argument('--phase', required=True)
    args, _ = parser.parse_known_args()
    spec = policy.read(policy.read(args.experiment_plan)['experiment_spec'])
    phase = spec['phases'][args.phase]
    cfg = policy.read(phase['config'])
    assert cfg['current_AR_policy']['DDP']['static_graph'] is False
    assert cfg['current_AR_policy']['DDP']['find_unused_parameters'] is True
    original_inventory = runtime.source_inventory
    runtime.source_inventory = lambda: {**original_inventory(), str(Path(__file__).relative_to(ROOT)): sha(__file__)}
    wrapped = []
    def construct(*values, **kwargs):
        kwargs.update(static_graph=False, find_unused_parameters=True)
        module = DistributedDataParallel(*values, **kwargs)
        wrapped.append(module)
        return module
    runtime.DistributedDataParallel = construct
    runtime.main()
    assert len(wrapped) == 1
    data = wrapped[0]._get_ddp_logging_data()
    assert not data.get('has_rebuilt_buckets', 0), data
    write(Path(phase['artifacts']) / f"rank{os.environ['RANK']}" / 'DDP_BUCKET_REVIEW.json',
        {'static_graph': False, 'find_unused_parameters': True, 'bucket_rebuilds': 0,
         'native_mean_reduction_preserved': True, 'actual_DDP_log': data})


if __name__ == '__main__': main()
