"""Complete released EVENT planner sharing a resident, trainable P10 renderer.

EVENT decisions include learned inventory/spans and qualitative heads. The
legacy token-only OPSD trainer is intentionally not accepted by this adapter.
Scoped LoRA/precision closures must be rebuilt, never copied with deepcopy.
"""
from __future__ import annotations

import json
from pathlib import Path
from contextlib import nullcontext

import torch

from .adapters import GenerationObservation, TransfusionOPSDAdapter
from .provenance import sha256_file


class EventGenerationAdapter(TransfusionOPSDAdapter):
    contract = 'sceneplan_event_generation_shared_opsd_bundle_v3'
    planner_contract = 'event_decisions_with_learned_heads_v1'

    def __init__(self, *, ar, diffusion, codec, audio_autoencoder, pointer, inventory, qualitative, release_path):
        pretransform = diffusion.pretransform
        super().__init__(mode='generation', ar=ar, diffusion=diffusion, codec=codec,
            audio_autoencoder=audio_autoencoder, fork=False)
        self.__dict__['native_pretransform'] = pretransform
        self.copy_pointer, self.source_inventory, self.qualitative_head = pointer, inventory, qualitative
        self.release_path = str(release_path)
        # This projection is unused by the selected use_ar_query=False head.
        if not self.qualitative_head.use_ar_query:
            self.qualitative_head.query_projection.requires_grad_(False)
        self.ar.encode_requests = self.encode_event_requests
        self.eval()

    def sample_plan(self, *args, **kwargs):
        raise ValueError('EVENT requires actual planning decisions; token-only OPSD collection is incompatible')

    def encode_event_requests(self, texts, *, device):
        """Preserve selected BF16 request math with FP32 trainable master weights.

        A functional BF16 parameter view matches the released conditioner,
        while gradients reach its FP32 projections. Only Qwen stays detached.
        """
        if torch.device(device) != self.device:
            raise ValueError('EVENT encoder and planner device differ')
        tokenized = self.prompt_conditioner.tokenizer(texts, add_special_tokens=True,
            padding=True, truncation=False, return_tensors='pt')
        ids = torch.as_tensor(tokenized['input_ids'], dtype=torch.long)
        mask = torch.as_tensor(tokenized['attention_mask'], dtype=torch.bool)
        if not mask.any(-1).all() or (mask.sum(-1) > 512).any():
            raise ValueError('EVENT raw request exceeds the unchanged 512-token contract')
        zeros = torch.zeros_like(ids)
        rows = [dict(input_ids=ids[i], attention_mask=mask[i], event_source_ids=zeros[i],
                     speech_source_ids=zeros[i]) for i in range(len(texts))]
        if self.device.type == 'cuda':
            views = {name: value.to(torch.bfloat16) for name, value in self.prompt_conditioner.named_parameters()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                context, keep = torch.func.functional_call(self.prompt_conditioner, views, (rows, self.device))
                context = self.diffusion.model.model.to_cond_embed(context)
        else:
            context, keep = self.prompt_conditioner(rows, self.device)
            context = self.diffusion.model.model.to_cond_embed(context)
        return context.float(), keep.bool()

    @torch.no_grad()
    def generate_released(self, observations, *, max_tokens=512):
        """Exact selected decision-block decoder; no training annotations accepted."""
        from ...inference.sceneplan_generation_ar_decision_blocks import generate_with_learned_copy
        from ...models import sceneplan_generation_ar_copy_pointer as pointer_module
        from ...models import sceneplan_generation_ar_qualitative_head as qualitative_module
        from ...models import sceneplan_generation_ar_source_inventory as inventory_module
        if not observations or any(not isinstance(o, GenerationObservation) for o in observations):
            raise ValueError('EVENT generation requires raw Generation observations')
        tokens, traces = generate_with_learned_copy(self.ar, self.copy_pointer, pointer_module,
            [o.request for o in observations], self.codec, device=self.device, max_plan_tokens=max_tokens,
            execution_head=self.qualitative_head, execution_module=qualitative_module,
            inventory_head=self.source_inventory, inventory_module=inventory_module)
        return tokens, traces

    def planning_forward(self, observations, token_rows, traces):
        """Differentiable decisions on the actual model-produced plans and spans.

        Returns raw head distributions plus actual AR prefix states. Spans are
        sampled decisions held fixed during fitting, never teacher annotations.
        """
        from ...models.sceneplan_generation_ar_copy_pointer import encode_character_alignment, FIELD_TYPES
        if not (len(observations) == len(token_rows) == len(traces)):
            raise ValueError('EVENT decision batch alignment changed')
        texts = [o.request for o in observations]
        context, context_mask = self.encode_event_requests(texts, device=self.device)
        alignment = encode_character_alignment(self.prompt_conditioner.tokenizer, texts, context_mask, device=self.device)
        inventory = self.source_inventory(context, context_mask, alignment)
        length = max(map(len, token_rows))
        tokens = torch.full((len(texts), length), self.codec.pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        for i, values in enumerate(token_rows):
            tokens[i, :len(values)] = torch.as_tensor(values, device=self.device)
            mask[i, :len(values)] = True
        captured = {}
        hook = self.ar.ar_adapter.output_norm.register_forward_hook(lambda module, args, output: captured.update(hidden=output))
        try:
            token_logits = self.ar(tokens, mask, context, context_mask)
        finally:
            hook.remove()
        hidden = captured['hidden']
        source_events = [[e for e in row if e['field'] != 'transcript'] for row in traces]
        n_sources = max(map(len, source_events))
        if n_sources < 1:
            raise ValueError('EVENT trace has no source planning decision')
        fields = torch.zeros(len(texts), n_sources, dtype=torch.long, device=self.device)
        first = context_mask.long().argmax(-1, keepdim=True)
        starts, ends = first.expand(-1, n_sources).clone(), first.expand(-1, n_sources).clone()
        scopes = torch.stack((starts, ends), -1)
        queries, active = [], torch.zeros_like(fields, dtype=torch.bool)
        for i, events in enumerate(source_events):
            local = []
            for j in range(n_sources):
                if j >= len(events):
                    local.append(hidden[i, 0] * 0)
                    continue
                event = events[j]
                position = event['query_position']
                if not 0 <= position < len(token_rows[i]) or token_rows[i][position] != self.codec._tid('<text_begin>'):
                    raise ValueError('EVENT query must be an actual generated text-field prefix')
                if texts[i][event['start']:event['end'] + 1] != event['text']:
                    raise ValueError('EVENT character span is not from the original request')
                local.append(hidden[i, position])
                fields[i, j] = FIELD_TYPES[event['field']]
                starts[i, j] = alignment.token_indices[i, event['start']]
                ends[i, j] = alignment.token_indices[i, event['end']]
                scope = event['learned_event_span']
                scopes[i, j, 0] = min(int(starts[i, j]), int(alignment.token_indices[i, scope['start']]))
                scopes[i, j, 1] = max(int(ends[i, j]), int(alignment.token_indices[i, scope['end']]))
                active[i, j] = True
            queries.append(torch.stack(local))
        queries = torch.stack(queries)
        qualitative = self.qualitative_head(queries, fields, starts, ends, context, context_mask, source_spans=scopes)
        return dict(token_logits=token_logits, inventory=inventory, qualitative=qualitative,
                    source_queries=queries, source_mask=active, context=context, context_mask=context_mask)

    def frozen_copy(self):
        # ContextVar and closures in scoped LoRA/precision are not deepcopy-safe.
        snapshot, _ = load_event_generation(self.release_path, device=self.device,
            qwen_runtime=self.qwen_runtime, dit_runtime=self.dit_runtime)
        snapshot.load_state_dict(self.state_dict(), strict=True)
        return snapshot.eval().requires_grad_(False)

    def native_conditioning(self, condition, *, differentiable):
        """P10 autocast with FP32 role-embedding sums, as in its release.

        EVENT AR casts its entire prompt conditioner to BF16. Native P10 keeps
        FP32 role weights and rounds only after their sum. Unknown CFG roles
        expose this distinction; applying AR's functional view here is wrong.
        """
        enabled = nullcontext() if differentiable else torch.no_grad()
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if self.device.type == 'cuda' else nullcontext()
        with enabled, amp:
            positive = self.diffusion.conditioner(condition.positive, self.device)
            result = self.diffusion.get_conditioning_inputs(positive)
            if self.cfg_scale != 1:
                negative = self.diffusion.conditioner(condition.negative, self.device)
                result.update(self.diffusion.get_conditioning_inputs(negative, negative=True))
            dtype = next(self.diffusion.model.parameters()).dtype
            return {key: value.to(dtype) if isinstance(value, torch.Tensor) else value for key, value in result.items()}

    def velocity_function(self, condition, *, differentiable):
        inputs = self.native_conditioning(condition, differentiable=differentiable)
        def velocity(z, t):
            amp = torch.autocast('cuda', dtype=torch.bfloat16) if self.device.type == 'cuda' else nullcontext()
            with amp:
                # Keep the native output dtype: casting BF16 v before scalar-dt
                # multiplication would change the released Euler integrator.
                return self.diffusion.model(z, t, **inputs, cfg_scale=self.cfg_scale,
                    batch_cfg=True, rescale_cfg=True, scale_phi=self.cfg_rescale_phi,
                    apg_scale=0., cfg_dropout_prob=0., padding_mask=condition.mask)
        return velocity

    @torch.no_grad()
    def native_rollout(self, condition, *, seed, steps=100):
        from ...inference.sampling import sample_discrete_euler
        from .objectives import EulerTrace
        noise = torch.randn((1, 64, condition.mask.shape[-1]), dtype=torch.float32,
            generator=torch.Generator(device='cpu').manual_seed(seed)).to(self.device)
        times = self.schedule(steps, noise.shape[-1])
        states = []
        final = sample_discrete_euler(self.velocity_function(condition, differentiable=False), noise,
            torch.tensor(times, dtype=torch.float32, device=self.device), disable_tqdm=True,
            callback=lambda values: states.append(values['x'].detach().clone()))
        states.append(final.detach().clone())
        return EulerTrace(tuple(states), times, condition.mask.detach().clone())

    def decode_for_reward(self, latent, model_num_samples):
        """Native frozen VAE decoding, including its selected autocast policy."""
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if self.device.type == 'cuda' else nullcontext()
        with amp:
            waveform = self.native_pretransform.decode(latent.float()).float()
        if waveform.ndim != 3 or waveform.shape[1] != 4 or waveform.shape[-1] < model_num_samples:
            raise ValueError('native EVENT FOA decoder geometry changed')
        return waveform[..., :model_num_samples]


def load_event_generation(release_path, *, device='cpu', qwen_runtime='released_fast', dit_runtime='native_bf16'):
    """Strict full EVENT state + native P10 EMA/conditioners and canonical VAE."""
    from .event_runtime import pin_qwen_fla_kernel
    from ...configuration import load_config
    from ...data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    from ...models import create_model_from_config
    from ...models.utils import load_ckpt_state_dict
    from ..factory import create_training_wrapper_from_config
    from ...models.sceneplan_transfusion_generation_ar import ScenePlanTransfusionGenerationAR
    from ...models.sceneplan_generation_ar_lora import AdaptedGenerationAR
    from ...models.sceneplan_generation_ar_copy_pointer import LiteralCopyPointer
    from ...models.sceneplan_generation_ar_source_inventory import SourceInventoryHead
    from ...models.sceneplan_generation_ar_qualitative_head import QualitativeExecutionHead
    from ...inference.sceneplan_generation_ar_precision import configure_float32_ar
    from ...models.sceneplan_transfusion_editing_pipeline import FROZEN_VAE_CHECKPOINT, FROZEN_VAE_CHECKPOINT_SHA256
    from ...data.sceneplan_transfusion_generation_ar_contract import canonical_sha256

    if qwen_runtime not in ('released_fast', 'torch_reference'):
        raise ValueError('unknown explicit EVENT Qwen runtime')
    if dit_runtime not in ('native_bf16', 'fp32'):
        raise ValueError('unknown explicit EVENT DiT runtime')
    runtime_receipt = pin_qwen_fla_kernel() if qwen_runtime == 'released_fast' else None
    release_path = Path(release_path).resolve(strict=True)
    release = json.loads(release_path.read_text())
    if release.get('schema') != 'generation_ar_local_inference_bundle_v1':
        raise ValueError('EVENT initialization requires a released full Generation bundle')
    checkpoint = release['components']['checkpoint']
    if sha256_file(checkpoint['path']) != checkpoint['sha256']:
        raise ValueError('EVENT checkpoint bytes differ from the selected release')
    payload = torch.load(checkpoint['path'], map_location='cpu', weights_only=False, mmap=True)
    required = {'ar_adapter', 'ar_lora', 'copy_pointer', 'source_inventory', 'qualitative_head'}
    if (payload.get('bundle_schema') != 'generation_ar_with_learned_source_inventory_v1'
            or not required.issubset(payload)):
        raise ValueError('all current EVENT learned modules must be present')
    root = Path(__file__).resolve().parents[3]
    runtime = ['models/sceneplan_transfusion_generation_ar.py', 'models/sceneplan_generation_ar_lora.py',
               'models/sceneplan_generation_ar_source_binding.py', 'models/transformer.py', 'models/dit.py',
               'models/conditioners.py', 'inference/sceneplan_generation_ar_precision.py']
    for relative in runtime:
        live = root / 'stable_audio_tools' / relative
        frozen = Path(release['model_snapshot']) / 'stable_audio_tools' / relative
        if sha256_file(live) != sha256_file(frozen):
            raise ValueError('EVENT shared runtime differs from frozen release: ' + relative)
    heads = [('copy_pointer', LiteralCopyPointer, 'copy_pointer_contract', 'copy_module_sha256'),
             ('source_inventory', SourceInventoryHead, 'source_inventory_contract', 'module_sha256'),
             ('qualitative_head', QualitativeExecutionHead, 'qualitative_contract', 'module_sha256')]
    restored = {}
    for key, cls, contract_key, hash_key in heads:
        import inspect
        if sha256_file(inspect.getfile(cls)) != payload[contract_key][hash_key]:
            raise ValueError('EVENT head implementation changed: ' + key)
        module = cls(**payload[contract_key]['model'])
        module.load_state_dict(payload[key], strict=True)
        restored[key] = module.to(device=device, dtype=torch.float32)
    parent = payload['run_contract']['p10_load']
    for key in ('checkpoint', 'model_config'):
        if sha256_file(parent[key]) != parent[key + '_sha256']:
            raise ValueError('EVENT P10 parent changed: ' + key)
    config = load_config(Path(parent['model_config']))
    if canonical_sha256(config) != parent['resolved_config_sha256']:
        raise ValueError('resolved P10 config differs from EVENT parent')
    codec_path = payload['run_contract'].get('codec_path') or payload['run_contract'].get('codec')
    if not isinstance(codec_path, str):
        raise ValueError('EVENT run contract must name its frozen codec path')
    codec = ModelScenePlanCodecV4(codec_path)
    if codec.fingerprint != payload['run_contract']['codec_fingerprint']:
        raise ValueError('EVENT codec fingerprint changed')
    diffusion = create_model_from_config(config)
    if qwen_runtime == 'torch_reference':
        from .event_runtime import configure_qwen_torch_reference
        runtime_receipt = configure_qwen_torch_reference(diffusion.conditioner.conditioners['prompt'].model)
    wrapper = create_training_wrapper_from_config(config, diffusion)
    state, metadata = load_ckpt_state_dict(parent['checkpoint'], return_metadata=True)
    if (wrapper.diffusion_ema is None or wrapper.conditioner_ema is None or
            tuple(metadata.get('conditioner_ema_parameter_names') or ()) != tuple(wrapper.conditioner_ema.parameter_names)):
        raise ValueError('EVENT requires the complete paired P10/conditioner EMA')
    wrapper.load_state_dict(state, strict=True)
    wrapper.conditioner_ema.copy_to(diffusion.conditioner)
    diffusion.model = wrapper.diffusion_ema.ema_model
    if sha256_file(FROZEN_VAE_CHECKPOINT) != FROZEN_VAE_CHECKPOINT_SHA256:
        raise ValueError('canonical FOA VAE changed')
    if diffusion.pretransform is None or diffusion.pretransform.scale != 1:
        raise ValueError('EVENT requires the unscaled canonical FOA latent convention')
    diffusion.pretransform.load_state_dict(load_ckpt_state_dict(str(FROZEN_VAE_CHECKPOINT)), strict=True)
    vae = diffusion.pretransform.model.float().eval().requires_grad_(False).to(device)
    base = ScenePlanTransfusionGenerationAR(p10_dit=diffusion.model.model,
        prompt_conditioner=diffusion.conditioner.conditioners['prompt'], pad_id=codec.pad_id,
        vocab_size=codec.vocab_size, activation_checkpointing=False)
    base.load_trainable_state_dict(payload['ar_adapter'])
    ar = AdaptedGenerationAR(base, codec, rank=8, alpha=8., binding_strength=0.)
    ar.load_lora_state_dict(payload['ar_lora'])
    ar.float().to(device)
    configure_float32_ar(ar)
    adapter = EventGenerationAdapter(ar=ar, diffusion=diffusion, codec=codec, audio_autoencoder=vae,
        pointer=restored['copy_pointer'], inventory=restored['source_inventory'],
        qualitative=restored['qualitative_head'], release_path=release_path)
    adapter.to(device)
    adapter.qwen_runtime = qwen_runtime
    adapter.dit_runtime = dit_runtime
    adapter.qwen_fused_norm_runtime = (runtime_receipt or {}).get('fused_gated_norm', {}).get('pinned_spec')
    receipt = {'contract': adapter.contract, 'release_sha256': sha256_file(release_path),
        'checkpoint': checkpoint, 'components_restored': sorted(required), 'codec_fingerprint': codec.fingerprint,
        'shared_transformer_object': ar.shared_transformer is diffusion.model.model.transformer,
        'p10_parent': parent, 'runtime': runtime_receipt, 'initial_state_only': True}
    if dit_runtime == 'fp32':
        from .event_dit_precision import configure_event_dit_fp32
        receipt['dit_runtime'] = configure_event_dit_fp32(adapter.diffusion)
    return adapter, receipt
