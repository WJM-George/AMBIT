"""Profile and improve execution of the unchanged joint OPSD objective.

Keep global16+512, all loss weights, learning rates and the100-step optimizer.
Elide forced-token forwards only after exact-token checks. Select paired
microbatches by measured time and memory, and retain the declared native kernels.
"""
import argparse
from collections import defaultdict
import gc
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from scripts.t2a.rl.train_editing_opsd_stability import StabilityLearner, base
from scripts.t2a.rl import train_editing_opsd_spatial as spatial
from stable_audio_tools.training.transfusion_opsd.editing_stream import OrdinalStream, homogeneous_microbatches
from stable_audio_tools.training.transfusion_opsd.frozen_forward_cache import FrozenForwardCache
from stable_audio_tools.training.transfusion_opsd.native_greedy_fastpath import NativeGreedyFastpath

PERFORMANCE_DIRECTORY = None


def peek(stream, count):
    return OrdinalStream(stream.ordinals, seed=stream.seed, rank=stream.rank, world=stream.world,
                         state=stream.state_dict()).take(count)


@torch.no_grad()
def native_plans(adapter, observations, batch_cap):
    if batch_cap == 1:
        return {key: adapter.native_plan(obs) for key, obs in observations.items()}
    from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_pipeline import ScenePlanTransfusionEditingCLAP44Pipeline
    view = ScenePlanTransfusionEditingCLAP44Pipeline(diffusion=adapter.diffusion, editing_ar=adapter.ar,
                                                    codec=adapter.codec)
    buckets = defaultdict(list)
    for key, obs in observations.items():
        buckets[obs.source_foa_latent.shape[-1]].append((key, obs))
    result = {}
    for rows in buckets.values():
        for start in range(0, len(rows), batch_cap):
            chunk = rows[start:start + batch_cap]
            if len(chunk) == 1:
                key, obs = chunk[0]
                result[key] = adapter.native_plan(obs)
                continue
            plans, tokens = view.generate_new_sceneplans(
                torch.cat([obs.source_foa_latent for _, obs in chunk]),
                torch.cat([obs.source_attention_mask for _, obs in chunk]),
                [obs.request for _, obs in chunk],
                duration_sec=[obs.model_num_samples / 44100 for _, obs in chunk])
            for (key, _), plan, ids in zip(chunk, plans, tokens):
                # Scalar native_plan always names its sole output this way.
                plan['sample_id'] = 'edited_000000'
                result[key] = (plan, ids)
    return result


def same_plans(a, b):
    return set(a) == set(b) and all(a[k][0] == b[k][0] and torch.equal(a[k][1], b[k][1]) for k in a)


def choose_microbatch(records, *, memory_limit_MiB):
    feasible = [r for r in records if r['ok'] and r['peak_allocated_MiB'] <= memory_limit_MiB]
    baseline = next((r for r in feasible if r['microbatch'] == 48), None)
    if baseline is None:
        raise RuntimeError('The established48-row microbatch did not pass the memory/finite-gradient probe.')
    best = min(feasible, key=lambda r: r['seconds'])
    return best['microbatch'] if best['seconds'] < baseline['seconds'] * .97 else 48


class ThroughputLearner(StabilityLearner):
    def __init__(self, q, rank, world, out):
        super().__init__(q, rank, world, out)
        backbone = self.adapter.ar.instruction_conditioner.model
        if backbone is not self.reference.ar.instruction_conditioner.model or backbone is not self.teacher.conditioner.model:
            raise ValueError('Performance cache requires the same immutable Qwen backbone.')
        self.forward_cache = FrozenForwardCache(backbone)
        self.greedy_fastpaths = [NativeGreedyFastpath(model.ar) for model in (self.adapter, self.reference)]
        self.plan_batch_cap = 1
        self.pending_plans = {}
        self.performance_dir = PERFORMANCE_DIRECTORY

    def gather(self, value):
        values = [None] * self.world
        dist.all_gather_object(values, value)
        return values

    def resume(self, path, *, resize=False, state=None):
        if resize or state is None or state['step'] != 100 or self.world != 4:
            raise ValueError('This throughput calibration starts from the protected four-rank100-step recovery.')
        elapsed = super().resume(path, resize=False, state=state)
        base.write(self.out / f'STATUS_rank{self.rank}.json', dict(phase='PERFORMANCE_CALIBRATION',
            rank=self.rank, pid=os.getpid(), time=time.time(), step=self.step))
        selected_elision, plan_trials = self.profile_planning()
        selected_microbatch, paired_trials = self.profile_paired()
        self.forward_cache.clear()
        self.optimizer.zero_grad(set_to_none=True)
        # All calibration forwards/backwards are discarded. Restore model,
        # Adam, costs, sampler positions and every rank RNG before update101.
        super().resume(path, resize=False, state=state)
        self.plan_batch_cap = 1
        for fastpath in self.greedy_fastpaths:
            fastpath.enabled = selected_elision
        self.q['paired_microbatch'] = selected_microbatch
        observed_hash = base.tensor_digest(self.trainable)
        if observed_hash != state['model_sha256']:
            raise RuntimeError('Calibration changed the resumed100-step model.')
        gc.collect(); torch.cuda.empty_cache()
        report = dict(step=100, rank=self.rank, selected_native_plan_batch_cap=1,
            selected_singleton_forward_elision=selected_elision,
            selected_paired_microbatch=selected_microbatch, long_audio_microbatch_cap=40,
            paired_trials=paired_trials, planning_trials=plan_trials,
            restored_model_sha256=observed_hash, original_model_sha256=state['model_sha256'],
            optimizer_and_RNG_and_streams_restored=True, global_request_batch=16, global_paired_batch=512,
            objective_and_learning_rates_unchanged=True, cache=self.forward_cache.statistics(),
            numerical_scope='Native batch1 retained after batching failed exact-output checks in the first calibration. Forced-token forward elision is enabled only after exact-token/full-plan checks on all upcoming16 requests for student and reference. Microbatch regrouping, if selected, can change RF noise grouping and floating-point summation.')
        base.write(self.performance_dir / f'SELECTION_rank{self.rank}.json', report)
        return elapsed

    def profile_planning(self):
        self.adapter.eval()
        ordinals = peek(self.request_stream, self.q['request_rows_per_rank'])
        observations = {i: self.requests.observe(self.adapter, i)[1] for i in ordinals}
        trials, expected = [], None
        for elision in (False, True):
            for fastpath in self.greedy_fastpaths:
                fastpath.enabled = elision
            before = [x.statistics() for x in self.greedy_fastpaths]
            self.progress('PROFILE_FORCED_TOKEN_ELISION', enabled=elision)
            self.forward_cache.clear()
            torch.cuda.synchronize(self.device); dist.barrier()
            started_unix = time.time()
            start = time.perf_counter()
            error = None
            try:
                current = native_plans(self.adapter, observations, 1)
                reference = native_plans(self.reference, observations, 1)
                matches = expected is None or (same_plans(current, expected[0]) and same_plans(reference, expected[1]))
                if expected is None:
                    expected = (current, reference)
            except torch.cuda.OutOfMemoryError as exc:
                matches, error = False, repr(exc)
                gc.collect(); torch.cuda.empty_cache()
            except RuntimeError as exc:
                if 'No captured native FLA' not in str(exc) and 'did not emit EOS' not in str(exc):
                    raise
                matches, error = False, repr(exc)
            torch.cuda.synchronize(self.device)
            observations_by_rank = self.gather(dict(rank=self.rank, seconds=time.perf_counter() - start,
                                                   exact_tokens_and_plans=matches, error=error))
            if not elision and not all(r['exact_tokens_and_plans'] for r in observations_by_rank):
                raise RuntimeError('The scalar native-planning baseline failed.')
            counts = [{k: fastpath.statistics()[k] - previous[k] for k in previous}
                      for fastpath, previous in zip(self.greedy_fastpaths, before)]
            trial = dict(batch_cap=1, singleton_forward_elision=elision,
                         local_forward_counts=counts, seconds=max(r['seconds'] for r in observations_by_rank),
                         ok=all(r['exact_tokens_and_plans'] for r in observations_by_rank),
                         ranks=observations_by_rank, started_unix=started_unix, completed_unix=time.time())
            trials.append(trial)
            base.write(self.performance_dir / f'PLANNING_rank{self.rank}.json', trials)
        feasible = [r for r in trials if r['ok']]
        best = min(feasible, key=lambda r: r['seconds'])
        selected = bool(best['singleton_forward_elision'] and best['seconds'] < trials[0]['seconds'] * .95)
        for fastpath in self.greedy_fastpaths:
            fastpath.enabled = False
        return selected, trials

    def paired_probe(self, indices, rows, denominators):
        from scripts.t2a.experiments.ar_structured_v1 import data
        self.adapter.train()
        for j, ids in enumerate(indices):
            batch = data.collate([rows[i] for i in ids], pad_id=self.adapter.codec.pad_id, joint=True)
            loss, _, _, _ = self.native.batch_loss(self.adapter, self.teacher, batch, self.cfg, self.device,
                                                   self.step * 100 + j, self.rank, self.world, denominators)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite native objective in throughput probe.')
            loss.backward()
            del loss, batch
        # Native paired training must still reach every trainable parameter.
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.parameters):
            raise RuntimeError('Throughput probe missed parameters or produced nonfinite gradients.')

    def profile_paired(self):
        from scripts.t2a.experiments.ar_structured_v1 import data
        ids = peek(self.pair_stream, self.q['paired_rows_per_rank'])
        rows = {i: self.paired[i] for i in ids}
        bucket_for = lambda i: self.requests.row(i)['latent_bucket_frames']
        limits = {432: self.cfg['performance']['short_batch_size'], 648: self.cfg['performance']['long_batch_size']}
        denominators = torch.zeros(3, dtype=torch.float64, device=self.device)
        for group in homogeneous_microbatches(ids, bucket_for, 48, bucket_limits=limits):
            batch = data.collate([rows[i] for i in group], pad_id=self.adapter.codec.pad_id, joint=True)
            ar, target, metadata, mask = self.native._move_joint_batch(batch, self.device)
            denominators += denominators.new_tensor([(ar['plan_labels'] != -100).sum(), mask.sum() * 64, len(metadata)])
            del batch, ar, target, metadata, mask
        dist.all_reduce(denominators)
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**2
        limit = min(total - 4500, total * .91)
        records = []
        for microbatch in (48, 64, 56):
            timings, peaks, errors, windows = [], [], [], []
            indices = homogeneous_microbatches(ids, bucket_for, microbatch, bucket_limits=limits)
            for repeat in range(2):
                self.progress('PROFILE_PAIRED_MICROBATCH', microbatch=microbatch, repeat=repeat)
                self.forward_cache.clear()
                self.optimizer.zero_grad(set_to_none=True)
                gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(self.device)
                dist.barrier(); torch.cuda.synchronize(self.device)
                started_unix = time.time()
                start = time.perf_counter()
                error = None
                try:
                    self.paired_probe(indices, rows, denominators)
                except torch.cuda.OutOfMemoryError as exc:
                    error = repr(exc)
                torch.cuda.synchronize(self.device)
                times = self.gather(dict(rank=self.rank, seconds=time.perf_counter() - start,
                    peak_allocated_MiB=torch.cuda.max_memory_allocated(self.device) / 1024**2, error=error))
                timings.append(max(r['seconds'] for r in times))
                windows.append(dict(started_unix=started_unix, completed_unix=time.time()))
                peaks.append(max(r['peak_allocated_MiB'] for r in times))
                errors.extend(r['error'] for r in times if r['error'])
                self.optimizer.zero_grad(set_to_none=True)
                gc.collect(); torch.cuda.empty_cache()
                if errors:
                    break
            records.append(dict(microbatch=microbatch, ok=not errors, seconds=min(timings),
                peak_allocated_MiB=max(peaks), repeats_seconds=timings, errors=errors,
                local_microbatch_sizes=[len(x) for x in indices], memory_limit_MiB=limit))
            records[-1]['repeat_windows'] = windows
            base.write(self.performance_dir / f'PAIRED_rank{self.rank}.json', records)
        self.adapter.eval()
        return choose_microbatch(records, memory_limit_MiB=limit), records

    @torch.no_grad()
    def prepare_plans(self):
        if self.plan_batch_cap == 1:
            return
        self.adapter.eval()
        ordinals = peek(self.request_stream, self.q['request_rows_per_rank'])
        inputs = {i: self.requests.observe(self.adapter, i) for i in ordinals}
        observations = {i: value[1] for i, value in inputs.items()}
        self.progress('BATCHED_CURRENT_NATIVE_PLANS', ordinals=ordinals, batch_cap=self.plan_batch_cap)
        current = native_plans(self.adapter, observations, self.plan_batch_cap)
        need_reference = {}
        for ordinal, (row, obs) in inputs.items():
            plan, tokens = current[ordinal]
            facts = spatial.request_facts(row['request'], row['operation'])
            if spatial.propose_current_decision(self.adapter, obs, plan, tokens.tolist(), facts, step=self.step) is not None:
                need_reference[ordinal] = obs
        reference = native_plans(self.reference, need_reference, self.plan_batch_cap)
        self.pending_plans = {i: dict(inputs=inputs[i], current=current[i], reference=reference.get(i)) for i in ordinals}

    def collect(self, ordinal):
        cached = self.pending_plans.pop(ordinal, None)
        if cached is None:
            return super().collect(ordinal)
        original_plan, original_reference, original_observe = self.adapter.native_plan, self.reference.native_plan, self.requests.observe
        row, observation = cached['inputs']
        def current_plan(obs, **kwargs):
            return cached['current'] if obs is observation and not kwargs else original_plan(obs, **kwargs)
        def reference_plan(obs, **kwargs):
            if obs is observation and cached['reference'] is not None and not kwargs:
                return cached['reference']
            return original_reference(obs, **kwargs)
        def observe(adapter, requested):
            return (row, observation) if adapter is self.adapter and requested == ordinal else original_observe(adapter, requested)
        self.adapter.native_plan, self.reference.native_plan, self.requests.observe = current_plan, reference_plan, observe
        try:
            return super().collect(ordinal)
        finally:
            self.adapter.native_plan, self.reference.native_plan, self.requests.observe = original_plan, original_reference, original_observe

    def train_step(self):
        torch.cuda.synchronize(self.device)
        started_unix = time.time()
        started = time.perf_counter()
        before = self.forward_cache.statistics()
        before_greedy = [x.statistics() for x in self.greedy_fastpaths]
        self.prepare_plans()
        torch.cuda.synchronize(self.device)
        prepared = time.perf_counter() - started
        try:
            stats = super().train_step()
        finally:
            self.pending_plans.clear()
        stats['performance']['native_plan_precompute_seconds'] = prepared
        stats['performance']['collection_seconds'] += prepared
        stats['performance']['step_seconds'] = time.perf_counter() - started
        stats['performance']['started_unix'] = started_unix
        stats['performance']['finished_unix'] = time.time()
        stats['performance']['native_plan_batch_cap'] = self.plan_batch_cap
        stats['performance']['singleton_forward_elision'] = self.greedy_fastpaths[0].enabled
        stats['performance']['greedy_forward_counts'] = [
            {k: fastpath.statistics()[k] - previous[k] for k in previous}
            for fastpath, previous in zip(self.greedy_fastpaths, before_greedy)]
        after = self.forward_cache.statistics()
        stats['performance']['frozen_Qwen_cache'] = dict(after,
            step_hits=after['hits'] - before['hits'], step_misses=after['misses'] - before['misses'])
        return stats


def main():
    global PERFORMANCE_DIRECTORY
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--performance-directory', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    PERFORMANCE_DIRECTORY = args.performance_directory.resolve()
    PERFORMANCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], *remaining]
    base.Learner = ThroughputLearner
    base.main()


if __name__ == '__main__':
    main()
