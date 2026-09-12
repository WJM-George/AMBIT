"""Zero-update evidence for joint gradients and the AR information boundary."""
import gc

import torch
import torch.nn.functional as F


def verify(module, batch, cfg, device, out):
    from . import runtime as rt
    ar, target, metadata, mask = rt._move_joint_batch(batch, device)
    generator = torch.Generator(device=device).manual_seed(cfg['seed'] + 17)
    noise = torch.randn(target.shape, device=device, generator=generator)
    times = torch.full((len(target),), .5, device=device)
    args = dict(source_foa_latent=ar['source_foa_latent'], source_attention_mask=ar['source_attention_mask'],
        plan_input_ids=ar['plan_input_ids'], plan_attention_mask=ar['plan_attention_mask'],
        raw_edit_requests=ar['raw_edit_requests'], metadata=metadata, noised_target=.5*(noise+target),
        timesteps=times, rf_padding_mask=mask)
    shared = next(module.ar.shared_transformer.layers.parameters())
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits, prediction, structure = module(**args)
        ar_loss = F.cross_entropy(logits.float().flatten(0,1), ar['plan_labels'].flatten(), ignore_index=-100)
        rf_loss = ((prediction.float() - (noise-target)).square() * mask[:,None]).sum() / (mask.sum()*64)
        expected = logits.detach().clone()
        expected_slots = {k: v.detach().clone() for k, v in structure['source'].items()}
        ar_gradient = torch.autograd.grad(ar_loss, shared, retain_graph=True)[0]
        rf_gradient = torch.autograd.grad(rf_loss, shared)[0]
        norms = [float(x.float().norm()) for x in (ar_gradient, rf_gradient)]
        if not all(bool(torch.isfinite(x).all()) for x in (ar_gradient, rf_gradient)) or min(norms) <= 0:
            raise RuntimeError('Each loss must supply its own finite nonzero shared-backbone gradient')
        del ar_gradient, rf_gradient, ar_loss, rf_loss, logits, prediction, structure
        # Perturb only information available to the RF training branch.
        # AR and the source readout must remain bitwise identical.
        args['noised_target'] = -args['noised_target']
        args['timesteps'] = torch.full_like(times, .7)
        changed, _, slots = module(**args)
        if not torch.equal(changed, expected) or any(not torch.equal(slots['source'][k], v) for k,v in expected_slots.items()):
            raise RuntimeError('RF-only training information changed AR or source-slot predictions')
        del changed, slots
    module.zero_grad(set_to_none=True)
    rt.write(out / 'SHARED_GRADIENT_AND_INPUT_PROBE.json', {
        'optimizer_updates': 0, 'same_Transformer_object': module.ar.shared_transformer is module.diffusion.model.model.transformer,
        'AR_shared_gradient_norm': norms[0], 'RF_shared_gradient_norm': norms[1],
        'AR_and_source_slots_unchanged_when_RF_inputs_change': True,
        'old_plan_input': False, 'source_caption_input': False, 'quality_gate_passed': False})
    gc.collect(); torch.cuda.empty_cache()
