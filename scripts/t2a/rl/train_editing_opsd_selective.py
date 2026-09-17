"""Joint OPSD with native-prefix retention and selective same-plan terminals.

The two additions are independently configurable for a short matched test.
All arms still train the complete native AR, DiT and shared Transformer.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from scripts.t2a.rl.train_editing_opsd_complete import CompleteLearner, base
from stable_audio_tools.training.transfusion_opsd.editing_request_constraints import (
    parse_edit_request, bind_edit_target, field_balanced_reference_kl,
)
from stable_audio_tools.training.transfusion_opsd.reference_prefix_retention import reference_prefix_targets
from stable_audio_tools.training.transfusion_opsd.execution_teacher_selection import same_plan_improvement_weights


class SelectiveLearner(CompleteLearner):
    @torch.no_grad()
    def collect(self, ordinal):
        item = super().collect(ordinal)
        config = self.q['selective_recipe']
        if config['select_same_plan_improvements'] and item['metrics']:
            coefficients, report = same_plan_improvement_weights(
                item['metrics'], item['plan_weights'].detach().cpu(),
                **config['terminal_selection'],
                preservation_reward_weight=self.q['spatial_recipe']['preservation_reward_weight'])
            item['terminal_training_coefficients'] = coefficients
            item['terminal_selection'] = report
        if config['reference_native_prefix']:
            self.progress('REFERENCE_NATIVE_PREFIX_RETENTION', ordinal=ordinal)
            plan, tokens = self.reference.native_plan(item['obs'])
            self.count('reference_native_plans')
            logits = self.reference.student_logits(item['obs'], tokens[None, :-1])[0].float()
            facts = parse_edit_request(item['row']['request'], item['row']['operation'])
            binding = bind_edit_target(plan, facts)
            holds = reference_prefix_targets(self.adapter.codec, tokens.tolist(), plan, facts, binding,
                logits, lambda prefix: self.adapter.allowed_next_ids(item['obs'], prefix))
            item['reference_prefix'] = dict(tokens=tokens, **holds)
        return item

    def backward_self(self, item, *, scale=1.):
        stats = super().backward_self(item, scale=scale)
        stats['terminal_selection'] = item.get('terminal_selection')
        ref = item.get('reference_prefix')
        if ref is not None:
            config = self.q['selective_recipe']
            logits = self.adapter.student_logits(item['obs'], ref['tokens'][None, :-1])[0].float()
            structure = field_balanced_reference_kl(logits, ref['structure'])
            text = field_balanced_reference_kl(logits, ref['text'])
            (scale * (config['reference_prefix_structure_weight'] * structure
                      + config['reference_prefix_text_weight'] * text)).backward()
            stats['reference_native_prefix'] = dict(
                structure_KL=float(structure.detach()), text_KL=float(text.detach()),
                structure_positions=len(ref['structure']), text_positions=len(ref['text']),
                same_as_student_prefix=bool(torch.equal(ref['tokens'], item['tokens'])),
                excluded_request_fields=ref['excluded_request_fields'],
                excluded_removed_sources=ref['excluded_removed_sources'])
        return stats


base.Learner = SelectiveLearner

if __name__ == '__main__':
    base.main()
