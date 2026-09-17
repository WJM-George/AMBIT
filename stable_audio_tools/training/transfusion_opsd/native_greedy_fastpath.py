"""Elide scalar greedy forwards when the native grammar allows one token.

Every actual choice uses the original full-prefix forward with the same shape,
context, source features and sorted allowed IDs. Forced tokens are appended to
that prefix normally. Training forwards and batched generation stay native.
"""
from collections.abc import Sequence

import torch

from stable_audio_tools.data.sceneplan_transfusion_editing_plan import editing_ar_allowed_next_ids


class NativeGreedyFastpath:
    def __init__(self, model):
        self.model, self.original = model, model.generate_batch
        self.enabled = False
        self.skipped_forwards = self.choice_forwards = 0
        model.generate_batch = self.generate_batch

    @torch.no_grad()
    def generate_batch(self, source_foa_latent, source_attention_mask, instructions, *, codec,
                       max_plan_tokens=1024, fixed_duration_sec=None, source_m2d_audio_embedding=None):
        if (not self.enabled or self.model.training or source_foa_latent.ndim != 3 or
                source_foa_latent.shape[0] != 1 or len(instructions) != 1 or
                source_attention_mask.shape != (1, source_foa_latent.shape[-1]) or
                source_m2d_audio_embedding is not None or int(max_plan_tokens) <= 1):
            return self.original(source_foa_latent, source_attention_mask, instructions, codec=codec,
                max_plan_tokens=max_plan_tokens, fixed_duration_sec=fixed_duration_sec,
                source_m2d_audio_embedding=source_m2d_audio_embedding)
        durations = (list(fixed_duration_sec)
                     if isinstance(fixed_duration_sec, Sequence) and not isinstance(fixed_duration_sec, (str, bytes))
                     else [fixed_duration_sec])
        if len(durations) != 1:
            raise ValueError('fixed-duration count must match audio batch')
        context, context_mask = self.model.encode_edit_instructions(instructions, device=source_foa_latent.device)
        semantic = {}
        if getattr(self.model, 'source_clap_model', None) is not None:
            semantic['source_clap_features'] = self.model.source_clap_model.source_features(
                source_foa_latent, source_attention_mask.to(torch.bool))
        prefix = [int(codec.bos_id)]
        for _ in range(int(max_plan_tokens) - 1):
            allowed = sorted(int(v) for v in editing_ar_allowed_next_ids(codec, prefix,
                                                                       fixed_duration_sec=durations[0]))
            if not allowed:
                raise RuntimeError('Editing AR grammar produced no next token')
            if len(allowed) == 1:
                selected = allowed[0]
                self.skipped_forwards += 1
            else:
                ids = torch.tensor([prefix], device=source_foa_latent.device, dtype=torch.long)
                logits = self.model(source_foa_latent, source_attention_mask, ids,
                    ids.ne(int(self.model.pad_id)), context, context_mask, **semantic)
                allowed_tensor = torch.tensor(allowed, device=logits.device, dtype=torch.long)
                selected = int(allowed_tensor[logits[0, len(prefix) - 1, allowed_tensor].argmax()].item())
                self.choice_forwards += 1
            prefix.append(selected)
            if selected == int(codec.eos_id):
                return [torch.tensor(prefix, dtype=torch.long)]
        raise RuntimeError('Editing AR did not emit EOS within max_plan_tokens for rows [0]')

    def statistics(self):
        return dict(skipped_forwards=self.skipped_forwards, choice_forwards=self.choice_forwards)

    def close(self):
        self.model.generate_batch = self.original
