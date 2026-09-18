"""Full original20k teacher-forced diagnostics, separate from audio quality."""
from collections import defaultdict
import hashlib
import time
import types

import torch
import torch.distributed as dist


@torch.no_grad()
def validate_native(learner, model, *, max_batches=0, label='full'):
    from scripts.t2a.experiments.ar_structured_v1 import data, model as native_model
    from scripts.t2a.rl.train_editing_opsd_stream import write
    rank, world, device = learner.rank, learner.world, learner.device
    started = time.monotonic()
    dataset = learner.validation
    if len(dataset) != 20000:
        raise ValueError('Native diagnostic must use the original20000 validation rows.')
    entries = dataset._db().execute(
        'SELECT pair_ordinal,latent_bucket_frames,target_domain FROM pairs ORDER BY pair_ordinal').fetchall()
    if [r[0] for r in entries] != list(range(20000)):
        raise ValueError('Incomplete original validation catalog.')
    buckets = defaultdict(list)
    speech = {}
    for ordinal, bucket, domain in entries:
        if ordinal % world == rank:
            buckets[bucket].append(ordinal)
            speech[ordinal] = domain in ('speech_only', 'speech_mixed')
    batches = [ids[start:start+32] for _, ids in sorted(buckets.items()) for start in range(0, len(ids), 32)]
    if max_batches:
        # Startup probes exercise both short and long native shapes.
        probes = [ids[:32] for _, ids in sorted(buckets.items())]
        batches = probes[:max_batches]
    totals = None
    denominators = torch.zeros(3, dtype=torch.float64, device=device)
    seen, speech_rows = [], 0
    was_training = model.training
    model.eval()
    # frozen_copy() clones AR/DiT modules, not the instance-level forward
    # installed on the trainable adapter. Bind the identical native joint
    # function explicitly to the selected student OR original40k reference.
    forward = types.MethodType(native_model.model_forward, model)
    try:
        for index, ids in enumerate(batches):
            learner.progress('NATIVE_VALIDATION_LOSS', completed_local_rows=len(seen),
                             total_rows=20000, step=learner.step, probe=bool(max_batches))
            batch = data.collate([dataset[i] for i in ids], pad_id=learner.adapter.codec.pad_id, joint=True)
            ar, target, metadata, mask = learner.native._move_joint_batch(batch, device)
            den = denominators.new_tensor([(ar['plan_labels'] != -100).sum(), mask.sum()*64, len(ids)])
            # Fixed generator index across checkpoints, not training step.
            loss, sums, names, outputs = learner.native.batch_loss(forward, learner.teacher, batch,
                learner.cfg, device, 500000 + index, rank, world, den)
            if not torch.isfinite(sums).all() or not torch.isfinite(loss):
                raise RuntimeError('Nonfinite full-split native validation.')
            if totals is None:
                totals = torch.zeros_like(sums)
            totals += sums
            denominators += den
            seen.extend(ids)
            speech_rows += sum(speech[i] for i in ids)
            del batch, ar, target, metadata, mask, loss, sums, outputs
    finally:
        model.train(was_training)
    if totals is None or len(seen) != len(set(seen)):
        raise ValueError('Empty or duplicate validation shard.')
    speech_count = torch.tensor(speech_rows, dtype=torch.int64, device=device)
    dist.all_reduce(totals)
    dist.all_reduce(denominators)
    dist.all_reduce(speech_count)
    if not max_batches and (int(denominators[2]) != 20000 or int(speech_count) != 11918):
        raise ValueError('Full validation coverage or speech population changed.')
    normalized = torch.cat((totals[:3]/denominators, totals[3:]/denominators[2]))
    metrics = dict(zip(('AR_CE', 'RF_MSE', 'structured_loss', *names), normalized.cpu().tolist()))
    receipt = dict(step=learner.step, rows=int(denominators[2]), speech_rows=int(speech_count),
        local_rows=len(seen), unique_local_rows=len(set(seen)), rank=rank, world=world,
        ordered_ordinal_sha256=hashlib.sha256(','.join(map(str, seen)).encode()).hexdigest(),
        metrics=metrics, seconds=time.monotonic()-started, probe=bool(max_batches),
        scope='Teacher-forced AR/RF/structured diagnostic. Not nine-metric generated audio quality.')
    directory = learner.out / 'native_validation'
    directory.mkdir(exist_ok=True)
    write(directory / f'{label}_step{learner.step:06d}_rank{rank}.json', receipt)
    if rank == 0:
        write(directory / f'{label}_step{learner.step:06d}.json', receipt)
    return receipt
