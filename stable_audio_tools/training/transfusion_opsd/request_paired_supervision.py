"""Training-label fallback when a request lacks joint AR/DiT execution supervision.

Labels are consumed only by the native paired backward pass. Neither the
rollout nor an execution reward may read them. Coverage is not edit success.
"""
import hashlib
import math

import torch

OPERATIONS = ('event_addition', 'event_removal', 'static_to_linear',
              'stationary_spatial_relocation', 'linear_to_static')
RECIPE = dict(version='all_operations_paired_fallback_v1', paired_native_weight=1.,
              trigger='missing_AR_or_DiT_execution_supervision', require_full_coverage=True)


def validate_request_pair(row, batch):
    if row['operation'] not in OPERATIONS or len(batch['metadata']) != 1:
        raise ValueError('Request correction requires one supported editing pair.')
    meta = batch['metadata'][0]
    identities = [('pair_ordinal', 'pair_ordinal'), ('pair_id', 'pair_id'),
                  ('operation', 'operation'), ('request', 'raw_edit_request'),
                  ('model_num_samples', 'model_num_samples'),
                  ('source_latent_tensor_sha256', 'source_foa_latent_tensor_sha256')]
    if meta.get('editing_split') != 'train' or any(row[a] != meta.get(b) for a, b in identities):
        raise ValueError('Request correction pair/source/instruction identity mismatch.')
    target_hash = meta.get('target_foa_latent_tensor_sha256')
    if (not isinstance(target_hash, str) or len(target_hash) != 64
            or any(c not in '0123456789abcdef' for c in target_hash)):
        raise ValueError('Missing verified paired target latent identity.')
    # The native dataset separately verifies tensor/scene hashes and checks
    # edited/unchanged source identities, actions and the real post-edit count.
    return dict(ordinal=row['pair_ordinal'], pair_id=row['pair_id'], operation=row['operation'],
                target_latent_sha256=target_hash,
                target_source_ids=[s['source_id'] for s in meta['model_sceneplan']['sources']],
                target_kinds=sorted({s['kind'] for s in meta['model_sceneplan']['sources'] if 'kind' in s}))


def request_loss(native, adapter, teacher, batch, cfg, device, *, row, step, scale, weight):
    if not all(math.isfinite(x) and x > 0 for x in (scale, weight)):
        raise ValueError('Request correction requires finite positive weights.')
    provenance = validate_request_pair(row, batch)
    ar, target, metadata, mask = native._move_joint_batch(batch, device)
    den = torch.stack(((ar['plan_labels'] != -100).sum(), mask.sum() * target.shape[1],
                       mask.new_tensor(len(metadata), dtype=torch.long))).double()
    if bool((den <= 0).any()):
        raise ValueError('Request correction has no valid paired targets.')
    local_cfg = dict(cfg)
    key = f"{cfg['seed']}:{step}:{row['pair_id']}:request-paired-fallback-v1".encode()
    local_cfg['seed'] = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'big') % (2**62)
    loss, sums, _, _ = native.batch_loss(adapter, teacher, batch, local_cfg, device, 0, 0, 1, den)
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(sums).all()):
        raise RuntimeError('Nonfinite request paired objective.')
    # Keep the denominator at all requests, not only those needing fallback.
    # scale=1/local_request_count, followed by the existing rank average.
    report = dict(enabled=True, role='paired_request_correction', execution_teacher=False,
                  native_joint_loss=float(loss.detach()), weight=weight, local_request_scale=scale,
                  AR_CE=float(sums[0] / den[0]), RF_MSE=float(sums[1] / den[1]),
                  structured_loss=float(sums[2] / den[2]), **provenance)
    return loss * (weight * scale), report


def execution_supervision(stats, weights):
    """Inspect the objectives actually used, after all terminal selection gates."""
    coefficients = stats.get('terminal_RF_coefficients', [])
    if any(not math.isfinite(c) or c < 0 for c in coefficients):
        raise ValueError('Invalid execution coefficients in request coverage.')
    selected = sum(c > 0 for c in coefficients)
    if selected != len(stats['terminal_RF']):
        raise ValueError('Selected executions and actual RF backward passes disagree.')
    ar = bool(stats['enabled'] and weights['ar_teacher_weight'] > 0)
    rf = bool(stats['enabled'] and weights['terminal_RF_weight'] > 0 and selected)
    return dict(execution_AR=ar, execution_RF=rf, execution_RF_mass=math.fsum(coefficients), execution_joint=ar and rf,
                fallback_reason=None if ar and rf else (
                    'no_qualified_execution_teacher' if not stats['enabled']
                    else 'missing_AR_or_DiT_execution_supervision'))


def covered_request(route, correction):
    if correction.get('recipe') == 'exclusive_request_branch_v1':
        from .branch_request_supervision import covered_request as branch_coverage
        return branch_coverage(route, correction)
    paired = bool(correction.get('enabled'))
    if paired:
        if correction.get('execution_teacher') is not False:
            raise ValueError('Paired correction cannot count as an execution teacher.')
        if not all(math.isfinite(correction[k]) for k in ('native_joint_loss', 'AR_CE', 'RF_MSE', 'structured_loss')):
            raise ValueError('A nonfinite paired correction cannot count as coverage.')
    result = dict(route, paired_correction=paired,
                  covered=bool(route['execution_joint'] or paired))
    if not result['covered']:
        raise RuntimeError('Request has neither joint execution supervision nor paired correction.')
    return result


def feedback_coverage(requests, *, require_complete=False):
    coverage = {}
    for request in requests:
        row = coverage.setdefault(request['requested_operation'], dict(requests=0, qualified_teachers=0,
            selected_RF_teachers=0, request_constraints=0, bound_requests=0, native_reference_prefixes=0,
            paired_removal_corrections=0, paired_request_corrections=0,
            joint_execution_supervision=0, covered_requests=0, uncovered_requests=0,
            execution_AR_requests=0, execution_RF_requests=0, GT_AR_requests=0, GT_RF_requests=0,
            covered_AR_requests=0, covered_RF_requests=0))
        row['requests'] += 1
        row['qualified_teachers'] += int(request['enabled'])
        row['selected_RF_teachers'] += (request.get('terminal_selection') or {}).get('selected', 0)
        row['request_constraints'] += bool(request['request_constraint_fields'])
        row['bound_requests'] += bool(request['binding']['available'])
        row['native_reference_prefixes'] += bool(request.get('reference_native_prefix'))
        row['paired_removal_corrections'] += bool(request.get('paired_removal_correction', {}).get('enabled'))
        correction = request.get('paired_request_correction', {})
        row['paired_request_corrections'] += bool(correction.get('enabled'))
        supervision = request.get('request_supervision')
        if require_complete:
            if supervision is None or covered_request(supervision, correction) != supervision:
                raise RuntimeError('Missing or inconsistent per-request supervision receipt.')
        if supervision is not None:
            row['joint_execution_supervision'] += bool(supervision['execution_joint'])
            row['covered_requests'] += bool(supervision['covered'])
            row['uncovered_requests'] += not supervision['covered']
            row['execution_AR_requests'] += bool(supervision['execution_AR'])
            row['execution_RF_requests'] += bool(supervision['execution_RF'])
            row['GT_AR_requests'] += bool(supervision.get('GT_AR', correction.get('enabled')))
            row['GT_RF_requests'] += bool(supervision.get('GT_RF', correction.get('enabled')))
            row['covered_AR_requests'] += bool(supervision.get('covered_AR', supervision['covered']))
            row['covered_RF_requests'] += bool(supervision.get('covered_RF', supervision['covered']))
    return coverage
