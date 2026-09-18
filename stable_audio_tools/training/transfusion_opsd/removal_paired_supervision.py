"""Explicit native paired correction for removal requests lacking a teacher.

This reads training labels only in the backward path. It is recorded as
paired supervision, never as an execution-qualified OPSD teacher.
"""
import hashlib
import math

import torch


def validate_removal_pair(row, batch):
    if row['operation'] != 'event_removal' or len(batch['metadata']) != 1:
        raise ValueError('Removal correction requires exactly one removal pair.')
    meta = batch['metadata'][0]
    pairs = [('pair_ordinal', 'pair_ordinal'), ('pair_id', 'pair_id'),
             ('operation', 'operation'), ('request', 'raw_edit_request'),
             ('model_num_samples', 'model_num_samples'),
             ('source_latent_tensor_sha256', 'source_foa_latent_tensor_sha256')]
    if meta.get('editing_split') != 'train' or any(row[a] != meta.get(b) for a, b in pairs):
        raise ValueError('Removal correction pair/source/instruction identity mismatch.')
    target_hash = meta.get('target_foa_latent_tensor_sha256')
    if not isinstance(target_hash, str) or len(target_hash) != 64:
        raise ValueError('Missing verified paired target latent identity.')
    return dict(ordinal=row['pair_ordinal'], pair_id=row['pair_id'],
                target_latent_sha256=target_hash,
                target_source_ids=[s['source_id'] for s in meta['model_sceneplan']['sources']])


def removal_loss(native, adapter, teacher, batch, cfg, device, *, row, step, scale, weight):
    if not all(math.isfinite(x) and x > 0 for x in (scale, weight)):
        raise ValueError('Removal correction requires finite positive weights.')
    provenance = validate_removal_pair(row, batch)
    ar, target, metadata, mask = native._move_joint_batch(batch, device)
    den = torch.stack(((ar['plan_labels'] != -100).sum(), mask.sum() * target.shape[1],
                       mask.new_tensor(len(metadata), dtype=torch.long))).double()
    if bool((den <= 0).any()):
        raise ValueError('Removal correction has no valid paired targets.')
    # Namespace RF randomness by sample and update without consuming either
    # sampler or changing the paired stream's original noise sequence.
    local_cfg = dict(cfg)
    key = f"{cfg['seed']}:{step}:{row['pair_id']}:removal-paired-v1".encode()
    local_cfg['seed'] = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'big') % (2**62)
    loss, sums, _, _ = native.batch_loss(adapter, teacher, batch, local_cfg, device, 0, 0, 1, den)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite removal paired objective.')
    # scale=1/request_rows_per_rank, then the existing gradient reducer
    # averages all ranks. Missing teachers never shrink the denominator.
    report = dict(enabled=True, role='paired_removal_correction', execution_teacher=False,
                  native_joint_loss=float(loss.detach()), weight=weight, local_request_scale=scale,
                  AR_CE=float(sums[0] / den[0]), RF_MSE=float(sums[1] / den[1]),
                  structured_loss=float(sums[2] / den[2]), **provenance)
    return loss * (weight * scale), report
