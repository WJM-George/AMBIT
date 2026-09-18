"""One chosen objective per request and branch, with an explicit fixed budget.

Execution quality gates run before this module. A missing branch uses that
request's verified training pair; labels never enter rollout or reward code.
The independent 512-pair native objective owns all structured auxiliary loss.
"""
import hashlib
import math

import torch
import torch.nn.functional as F

from .request_paired_supervision import validate_request_pair


RECIPE = dict(version='exclusive_request_branch_v1', paired_native_weight=1.,
              trigger='missing_execution_supervision_per_branch', require_full_coverage=True,
              AR_budget=1., RF_budget=1., normalize_selected_RF=True,
              request_structured_weight=0., independent_paired_weight=1.)


def active(q):
    return (q.get('request_paired_correction') or {}).get('version') == RECIPE['version']


def normalize_selected_coefficients(coefficients):
    if any(not math.isfinite(c) or c < 0 for c in coefficients):
        raise ValueError('Execution coefficients must be finite and nonnegative.')
    mass = math.fsum(coefficients)
    return ([c / mass for c in coefficients] if mass > 0 else list(coefficients)), mass


def request_seed(cfg, step, pair_id):
    key = f"{cfg['seed']}:{step}:{pair_id}:request-branch-fallback-v1".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'big') % (2**62)


def branch_objective(adapter, ar, target, metadata, rf_mask, *, use_AR, use_RF, seed):
    """Native CE/RF formulas, without computing an unused branch or auxiliary.

    This is checked against native.batch_loss with the same noise and masked
    weights. The RF conditioner sees the *paired target plan*, never a rollout.
    """
    if not (use_AR or use_RF):
        raise ValueError('An empty fallback must not run a forward pass.')
    ar_loss = rf_loss = None
    with adapter._amp():
        if use_AR:
            features = adapter.ar.source_clap_model.source_features(
                ar['source_foa_latent'], ar['source_attention_mask'])
            context, mask = adapter.ar.encode_edit_instructions(
                ar['raw_edit_requests'], device=target.device)
            logits = adapter.ar(ar['source_foa_latent'], ar['source_attention_mask'],
                ar['plan_input_ids'], ar['plan_attention_mask'], context, mask,
                source_clap_features=features)
            ar_loss = F.cross_entropy(logits.float().flatten(0, 1), ar['plan_labels'].flatten(),
                                      ignore_index=-100, reduction='sum') / (ar['plan_labels'] != -100).sum()
        if use_RF:
            generator = torch.Generator(device=target.device).manual_seed(seed + 31_000_001)
            noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
            times = torch.rand(len(target), device=target.device, generator=generator)
            noised = (1 - times[:, None, None]) * target + times[:, None, None] * noise
            conditioning = dict(adapter.diffusion.conditioner(metadata, target.device))
            conditioning['source_foa_latent'] = [ar['source_foa_latent'], None]
            inputs = adapter.diffusion.get_conditioning_inputs(conditioning)
            prediction = adapter.diffusion.model(noised, times, **inputs,
                cfg_dropout_prob=0., padding_mask=rf_mask)
            rf_loss = (((prediction.float() - (noise - target)).square()) * rf_mask[:, None]).sum()
            rf_loss = rf_loss / (rf_mask.sum() * target.shape[1])
    loss = sum(x for x in (ar_loss, rf_loss) if x is not None)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite request branch objective.')
    return loss, ar_loss, rf_loss


def request_loss(native, adapter, teacher, batch, cfg, device, *, row, step, scale, weight, route):
    if not math.isfinite(scale) or scale <= 0 or weight != RECIPE['paired_native_weight']:
        raise ValueError('Invalid fixed request budget.')
    provenance = validate_request_pair(row, batch)
    ar, target, metadata, mask = native._move_joint_batch(batch, device)
    if not bool((ar['plan_labels'] != -100).any()) or not bool(mask.any()):
        raise ValueError('Request correction has no valid paired targets.')
    use_ar, use_rf = not route['execution_AR'], not route['execution_RF']
    loss, ar_loss, rf_loss = branch_objective(adapter, ar, target, metadata, mask,
        use_AR=use_ar, use_RF=use_rf, seed=request_seed(cfg, step, row['pair_id']))
    report = dict(enabled=True, recipe=RECIPE['version'], role='paired_request_correction',
        execution_teacher=False, AR=use_ar, RF=use_rf, native_joint_loss=float(loss.detach()),
        AR_CE=None if ar_loss is None else float(ar_loss.detach()),
        RF_MSE=None if rf_loss is None else float(rf_loss.detach()), structured_loss=0.,
        weight=weight, local_request_scale=scale, **provenance)
    return loss * (weight * scale), report


def covered_request(route, correction):
    if correction.get('recipe') != RECIPE['version']:
        raise ValueError('Missing branch supervision recipe.')
    gt_ar, gt_rf = correction.get('AR', False), correction.get('RF', False)
    if bool(correction['enabled']) != bool(gt_ar or gt_rf):
        raise ValueError('Fallback receipt does not match its branches.')
    if correction['enabled']:
        if correction.get('execution_teacher') is not False or correction.get('structured_loss') != 0.:
            raise ValueError('Fallback cannot claim execution feedback or a variable structured loss.')
        values = [correction['native_joint_loss']]
        for branch, flag in [('AR_CE', gt_ar), ('RF_MSE', gt_rf)]:
            value = correction[branch]
            if flag:
                values.append(value)
            elif value is not None:
                raise ValueError('An unselected GT branch has a loss.')
        if any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
            raise ValueError('Nonfinite or missing branch correction loss.')
    ar_count, rf_count = int(route['execution_AR']) + int(gt_ar), int(route['execution_RF']) + int(gt_rf)
    if ar_count != 1 or rf_count != 1:
        raise RuntimeError('Each request branch must have exactly one execution or GT objective.')
    if route['execution_RF'] and not math.isclose(route['execution_RF_mass'], 1., abs_tol=1e-7):
        raise ValueError('Qualified execution RF did not receive its fixed branch budget.')
    return dict(route, recipe=RECIPE['version'], paired_correction=bool(correction['enabled']),
        GT_AR=bool(gt_ar), GT_RF=bool(gt_rf), covered_AR=True, covered_RF=True, covered=True,
        AR_budget=1., RF_budget=1.)


def gradient_parameters(correction):
    names = ['ar.editing_dit.transformer.layers.0.pre_norm.gamma']
    if correction.get('AR'):
        names.append('ar.plan_adapter.plan_head.weight')
    if correction.get('RF'):
        names.append('ar.editing_dit.postprocess_conv.weight')
    return names
