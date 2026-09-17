"""Explicit rank-ordered FP32 means, independent of DDP bucket rebuilding.

Only gradient communication changes. The exact mean has the same mathematical
objective; its floating-point association is now recorded and reproducible.
"""
import argparse
import ast
import copy
import inspect
from pathlib import Path

import torch
from torch import distributed as dist
from scripts.t2a.experiments.clap_factual50k_v1 import runtime as base

REDUCTION = {'schema': 'clap44_rank_ordered_fp32_gradient_mean_v1', 'rank_order': [0, 1, 2],
    'operation': 'all_gather_each_bucket_then_rank0_add_rank1_add_rank2_divide3',
    'dtype': 'torch.float32', 'reason': 'Fix floating-point association across first-iteration and rebuilt DDP buckets.'}


def rank_ordered_mean(state, bucket):
    value = bucket.buffer()
    if value.dtype != torch.float32 or dist.get_world_size() != 3:
        raise RuntimeError('Rank-ordered gradient mean requires three FP32 ranks')
    parts = [torch.empty_like(value) for _ in range(3)]
    dist.all_gather(parts, value)
    result = parts[0].add_(parts[1]).add_(parts[2]).div_(3)
    future = torch.futures.Future()
    future.set_result(result)
    return future


def compiled_runtime():
    original = ast.parse(inspect.getsource(base.run)); count = 0
    added = ast.parse('model.register_comm_hook(state=None, hook=_rank_ordered_mean)').body[0]
    class Add(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal count
            if ast.unparse(node) == 'model = DistributedDataParallel(model, device_ids=[local])':
                count += 1
                return [node, copy.deepcopy(added)]
            return self.generic_visit(node)
    changed = Add().visit(copy.deepcopy(original)); assert count == 1
    class Remove(ast.NodeTransformer):
        def visit_Expr(self, node):
            if ast.dump(node, include_attributes=False) == ast.dump(added, include_attributes=False): return None
            return self.generic_visit(node)
    assert ast.dump(Remove().visit(copy.deepcopy(changed)), include_attributes=False) == ast.dump(original, include_attributes=False)
    namespace = dict(base.__dict__, _rank_ordered_mean=rank_ordered_mean)
    exec(compile(ast.fix_missing_locations(changed), __file__ + '::rank_ordered_DDP', 'exec'), namespace)
    return namespace['run']


def main(path, phase):
    plan = base.state.read(path); contract = base.state.read(plan['phases'][phase]['contract'])
    if contract.get('gradient_reduction') != REDUCTION:
        raise RuntimeError('Contract must explicitly declare the gradient association')
    compiled_runtime()(path, phase)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--plan', type=Path, required=True); p.add_argument('--phase', required=True)
    a = p.parse_args(); main(a.plan.resolve(strict=True), a.phase)
