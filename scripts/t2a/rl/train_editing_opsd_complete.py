"""Editing OPSD with request constraints and coverage of retained capabilities.

Same native AR/DiT and one joint optimizer. Request rollout never reads a
target plan/audio. Optional request correction is a separately logged native
paired backward pass, declared by a versioned recipe.
"""
from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from scripts.t2a.rl import train_editing_opsd_spatial as spatial
from stable_audio_tools.training.transfusion_opsd.editing_request_constraints import (
    parse_edit_request, bind_edit_target, request_field_targets, frozen_text_targets,
    field_balanced_reference_kl,
)
from stable_audio_tools.training.transfusion_opsd.editing_spatial_retention import execution_authorized_decision
from stable_audio_tools.training.transfusion_opsd.native_coarse_choice_retention import field_balanced_native_set_ce
from stable_audio_tools.training.transfusion_opsd.native_paired_rf import paired_rf_example, paired_rf_loss
from stable_audio_tools.training.transfusion_opsd.supported_teacher import preserve_reference_support_mass
from stable_audio_tools.training.transfusion_opsd.removal_retention import apply_removal_retention
from stable_audio_tools.training.transfusion_opsd.removal_paired_supervision import removal_loss
from stable_audio_tools.training.transfusion_opsd.request_paired_supervision import (
    request_loss as paired_request_loss, execution_supervision, covered_request,
)
from stable_audio_tools.training.transfusion_opsd import branch_request_supervision
from stable_audio_tools.training.transfusion_opsd.editing_binaural_retention import (
    FrozenKemarRenderer, binaural_features, binaural_distances, select_operation_rows,
)

base = spatial.base


class CompleteLearner(spatial.SpatialLearner):
    def __init__(self, q, rank, world, out):
        super().__init__(q, rank, world, out)
        self.renderer = FrozenKemarRenderer().to(self.device).eval().requires_grad_(False)
        self._visited = []
        from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_pipeline import ScenePlanTransfusionEditingCLAP44Pipeline
        from stable_audio_tools.models.sceneplan_editing_gain_adapter import install_pipeline_gain_metadata
        self.reference_pipeline = ScenePlanTransfusionEditingCLAP44Pipeline(
            diffusion=self.reference.diffusion, editing_ar=self.reference.ar, codec=self.reference.codec)
        install_pipeline_gain_metadata(self.reference_pipeline)
        initial = q.get('initialization_checkpoint')
        if initial:
            if base.sha(initial['path']) != initial['sha256']:
                raise ValueError('Initialization checkpoint changed.')
            state = torch.load(initial['path'], map_location='cpu', weights_only=False, mmap=True)
            if (state['step'] != initial['step'] or set(state['model']) != set(self.trainable)
                    or state['original_checkpoint']['sha256'] != q['base_checkpoint']['sha256']):
                raise ValueError('The new recipe requires an identified compatible joint model.')
            with torch.no_grad():
                for name, value in state['model'].items():
                    self.trainable[name].copy_(value.to(self.device))
            del state
            # Reference was copied before initialization and stays original40k.
            # This is a new optimization run, not an Adam/sampler resume.
        base.write(out / f'COMPLETE_RECIPE_rank{rank}.json', dict(
            initialization=initial or q['base_checkpoint'], optimizer='new_AdamW', update=0,
            frozen_reference=q['base_checkpoint'], renderer=self.renderer.identity,
            request_rollout_target_access=False, paired_objectives_separate=True,
            removal_paired_correction=q.get('removal_repair'),
            request_paired_correction=q.get('request_paired_correction'),
            removal_execution_teacher='Unavailable without reliable source-specific acoustic evidence. Removal anchors are released consistently; optional native paired correction is separately logged. No inferred count label.'))

    @torch.no_grad()
    def execute(self, obs, plan, seed, feature, facts, *, register_plan=None, capture=None):
        record = capture if capture is not None else ([] if not self._visited else None)
        result = super().execute(obs, plan, seed, feature, facts, register_plan=register_plan, capture=record)
        if not self._visited and record:
            self._visited.append(dict(record[0], plan=plan, seed=seed))
        return result

    @torch.no_grad()
    def collect(self, ordinal):
        self._visited = []
        previous_teachers = self.costs['execution_teachers']
        item = super().collect(ordinal)
        obs = item['obs']
        facts = parse_edit_request(item['row']['request'], item['row']['operation'])
        binding = bind_edit_target(item['plan'], facts)
        allowed = lambda prefix: self.adapter.allowed_next_ids(obs, prefix)
        constraints = request_field_targets(self.adapter.codec, item['tokens'].tolist(), item['plan'],
            facts, binding, allowed, **self.q['complete_recipe']['request_tolerances'])
        teacher = self.reference.student_logits(obs, item['tokens'][None, :-1])[0].float()
        # Relocation and motion edits retain the edited object's content too.
        # Addition/removal text has explicit positive/negative request scope.
        excluded = ([binding['source_id']] if binding['available'] and facts and
                    facts['operation'] in ('event_addition', 'event_removal') else [])
        item['text_reference'] = frozen_text_targets(self.adapter.codec, item['tokens'].tolist(), teacher,
                                                     allowed, excluded_sources=excluded)
        item['request_constraints'], item['binding'] = constraints, binding

        if item['terminals']:
            # The current model still defines preference rewards. Original40k
            # provides only a persistent content-retention floor, using its
            # own native plan and actual normal execution.
            reference_plan, _ = self.reference.native_plan(obs)
            descriptions = [s.get('description', s.get('transcript', '')) for s in reference_plan['sources']]
            if facts and not facts['removal']:
                descriptions.extend(facts['fields'].values())
            feature = self.text_feature(' '.join(descriptions))
            reference_scores = {}
            for seed in sorted({terminal['seed'] for terminal in item['terminals']}):
                self.progress('FIXED_REFERENCE_CONTENT_FLOOR', ordinal=ordinal, seed=seed)
                reference_z = self.reference_pipeline.sample_edited_latents(
                    obs.source_foa_latent, obs.source_attention_mask, [reference_plan],
                    model_num_samples=[obs.model_num_samples], steps=self.q['inference_steps'],
                    cfg_scale=1., seed=seed)
                reference_scores[seed] = float((self.clap(reference_z, obs.source_attention_mask)['semantic'] * feature).sum())
                self.count('fixed_reference_audio')
                del reference_z
            for j, (terminal, metric) in enumerate(zip(item['terminals'], item['metrics'])):
                value = float((self.clap(terminal['clean'], obs.source_attention_mask)['semantic'] * feature).sum())
                reference_semantic = reference_scores[terminal['seed']]
                metric['fixed_reference_semantic'] = reference_semantic
                metric['anchored_semantic'] = value
                metric['fixed_reference_seed'] = terminal['seed']
                if value < reference_semantic - self.q['complete_recipe']['fixed_semantic_tolerance']:
                    metric['qualification_failures'].append('fixed40k_content_floor')
                    metric['qualified_terminal'] = False
                    item['terminal_qualified'][j] = False
            self.refresh_teacher(item, teacher)
        # Count the final qualified teacher, after all persistent gates.
        self.costs['execution_teachers'] = previous_teachers + int(item['enabled'])

        # Explicitly requested fields outrank an incompatible reference value.
        # Unspecified fields remain anchored; discrete prefixes are not changed.
        authorized = {t['position'] for t in constraints['targets']}
        item['holds'] = [h for h in item['holds'] if h['position'] not in authorized]
        item['coarse'] = [h for h in item['coarse'] if h['position'] not in authorized]
        removal = apply_removal_retention(self.adapter.codec, item, facts, binding)
        if not removal['reference_velocity_allowed']:
            # A whole-scene velocity anchor conditioned on the student's
            # wrong plan would also retain the object the request removes.
            item['velocity_reference'] = None
            self._visited = []
            return item
        if not self._visited:
            # Includes overlapping/ambiguous non-removal cases lacking an
            # improvement teacher. They still receive a DiT retention target
            # at a state the student actually visited under its own plan.
            text = ' '.join(s.get('description', s.get('transcript', '')) for s in item['plan']['sources'])
            _, wave, _ = self.execute(obs, item['plan'],
                self.q['seed'] + self.step * 1009 + self.rank * 100, self.text_feature(text),
                spatial.request_facts(item['row']['request'], item['row']['operation']), capture=[])
            del wave
        visit = self._visited[0]
        condition = self.reference.render_condition(obs, visit['plan'])
        velocity = self.reference.velocity_function(condition, differentiable=False)(visit['state'], visit['time'])
        self.count('fixed_reference_velocity')
        item['velocity_reference'] = dict(state=visit['state'], time=visit['time'], target=velocity.detach(),
                                          condition=self.adapter.render_condition(obs, visit['plan']))
        self._visited = []
        return item

    @torch.no_grad()
    def refresh_teacher(self, item, teacher):
        threshold = self.q['spatial_recipe']
        rewards = []
        for k in range(2):
            values = []
            for metric in item['metrics']:
                if metric['plan_index'] != k:
                    continue
                keep = metric['unchanged_windows']
                penalty = min(1., keep['covariance_error']) if keep['available'] else 0.
                values.append(metric['reward'] - threshold['preservation_reward_weight'] * penalty
                              - float(not metric['qualified_terminal']))
            rewards.append(sum(values) / len(values))
        decision = item['decision']
        logits = self.adapter.student_logits(item['obs'], item['tokens'][None, :-1])[0].float()
        reference = logits[decision['position'] - 1, decision['legal_ids']].detach()
        support = [decision['legal_ids'].index(i) for i in decision['choice_ids']]
        conditional = (reference[support] + reference.new_tensor(rewards) / self.q['temperature']).softmax(-1)
        item['target'] = preserve_reference_support_mass(reference, support, conditional).probabilities.detach()
        item['plan_weights'] = ((1 - self.q['coverage_weight']) * conditional + self.q['coverage_weight'] * .5).detach()
        item['enabled'], item['rewards'] = any(item['terminal_qualified']), rewards
        self.set_reference_targets(item, teacher,
            decision=execution_authorized_decision(decision, item['metrics'], rewards))

    def backward_self(self, item, *, scale=1.):
        stats = super().backward_self(item, scale=scale)
        config = self.q['complete_recipe']
        logits = self.adapter.student_logits(item['obs'], item['tokens'][None, :-1])[0].float()
        request_loss = field_balanced_native_set_ce(logits, item['request_constraints']['targets'])
        text_loss = field_balanced_reference_kl(logits, item['text_reference'])
        (scale * (config['request_constraint_weight'] * request_loss
                  + config['retained_text_KL_weight'] * text_loss)).backward()
        visit = item['velocity_reference']
        retention = None
        if visit is not None:
            pred = self.adapter.velocity_function(visit['condition'], differentiable=True)(visit['state'], visit['time'])
            retention = paired_rf_loss(pred, visit['target'], item['obs'].source_attention_mask)
            (scale * config['reference_velocity_weight'] * retention).backward()
        correction = dict(enabled=False)
        route = execution_supervision(stats, self.q['spatial_recipe'])
        branch_recipe = branch_request_supervision.active(self.q)
        if branch_recipe:
            correction.update(recipe=branch_request_supervision.RECIPE['version'], AR=False, RF=False)
        request_repair = self.q.get('request_paired_correction')
        repair = request_repair or self.q.get('removal_repair')
        required = (not route['execution_joint'] if request_repair else
                    bool(repair and item['row']['operation'] == 'event_removal'))
        if required:
            from scripts.t2a.experiments.ar_structured_v1 import data
            self.progress('PAIRED_REQUEST_CORRECTION' if request_repair else 'PAIRED_REMOVAL_CORRECTION',
                          ordinal=item['row']['pair_ordinal'], operation=item['row']['operation'])
            batch = data.collate([self.paired[item['row']['pair_ordinal']]],
                                 pad_id=self.adapter.codec.pad_id, joint=True)
            training = self.adapter.training
            self.adapter.train()
            try:
                loss_fn = (branch_request_supervision.request_loss if branch_recipe else
                           paired_request_loss if request_repair else removal_loss)
                loss, correction = loss_fn(self.native, self.adapter, self.teacher, batch,
                    self.cfg, self.device, row=item['row'], step=self.step, scale=scale,
                    weight=repair['paired_native_weight'], **({'route': route} if branch_recipe else {}))
                probe, hooks = {}, []
                checked = getattr(self, '_correction_gradient_checked', set())
                probe_key = ((item['row']['operation'], correction['AR'], correction['RF'])
                             if branch_recipe else item['row']['operation'])
                if probe_key not in checked:
                    names = ['ar.plan_adapter.plan_head.weight',
                             'ar.editing_dit.postprocess_conv.weight',
                             'ar.editing_dit.transformer.layers.0.pre_norm.gamma']
                    expected_names = (branch_request_supervision.gradient_parameters(correction)
                                      if branch_recipe else names)
                    for name in names:
                        def record(gradient, name=name):
                            probe[name] = float(gradient.detach().float().norm())
                        hooks.append(self.trainable[name].register_hook(record))
                try:
                    loss.backward()
                finally:
                    for hook in hooks:
                        hook.remove()
                if hooks:
                    if (not set(expected_names) <= set(probe)
                            or any(not math.isfinite(probe[n]) or probe[n] <= 0 for n in expected_names)
                            or any(probe[n] != 0. for n in set(probe) - set(expected_names))):
                        raise RuntimeError('Request correction missed AR/DiT/shared gradients: ' + repr(probe))
                    self._correction_gradient_checked = checked | {probe_key}
                    correction['gradient_probe'] = probe
                if request_repair:
                    correction['fallback_reason'] = route['fallback_reason']
                    self.count('paired_request_corrections')
                if item['row']['operation'] == 'event_removal':
                    self.count('paired_removal_corrections')
            finally:
                self.adapter.train(training)
        supervision = covered_request(route, correction) if request_repair else None
        stats.update(request_constraint_CE=float(request_loss.detach()),
            retained_text_KL=float(text_loss.detach()),
            reference_velocity_MSE=None if retention is None else float(retention.detach()),
            reference_velocity_enabled=visit is not None,
            removal_retention=item['removal_retention'],
            paired_removal_correction=correction if item['row']['operation'] == 'event_removal' else dict(enabled=False),
            paired_request_correction=correction if request_repair else dict(enabled=False),
            request_supervision=supervision,
            evaluation_cache=item.get('evaluation_cache'),
            request_constraint_fields=[t['field'] for t in item['request_constraints']['targets']],
            request_constraint_unavailable=item['request_constraints']['unavailable'],
            retained_text_positions=len(item['text_reference']), binding=item['binding'],
            positive_OPSD_weights={key: self.q['spatial_recipe'][key]
                                   for key in ('ar_teacher_weight', 'terminal_RF_weight')},
            supervision_roles='request/retention objectives are not execution-derived improvements')
        return stats

    def backward_extra(self, batches):
        from stable_audio_tools.training.losses.sceneplan_editing_audio import _audio_features, _audio_distances
        config = self.q['complete_recipe']
        metadata = [batch['metadata'] for batch in batches]
        chosen = select_operation_rows(metadata, self.step, self.rank, config['decoded_rows_per_rank'])
        records = []
        self.progress('PAIRED_FOA_BINAURAL_BACKWARD', selected_rows=len(chosen))
        for batch_index, index in chosen:
            _, targets, metadata, masks = self.native._move_joint_batch(batches[batch_index], self.device)
            meta = metadata[index]
            n, ordinal = int(meta['model_num_samples']), int(meta['pair_ordinal'])
            source = meta['source_foa_latent'].to(self.device).float()
            if source.ndim == 2:
                source = source[None]
            mask = masks[index:index + 1]
            obs = self.adapter.observe_editing(sample_id=meta['pair_id'], request=meta['raw_edit_request'],
                source_foa_latent=source, source_attention_mask=mask, model_num_samples=n)
            condition = self.adapter.render_condition(obs, meta['model_sceneplan'])
            target = targets[index:index + 1].float()
            generator = torch.Generator(device='cpu').manual_seed(
                self.q['seed'] + 151_000_001 + self.step * 17 + self.rank + index * 100003)
            noise = torch.randn(target.shape, generator=generator).to(self.device)
            t = target.new_tensor([(.10, .20)[self.step % 2]])
            z, _ = paired_rf_example(target, noise, t, mask)
            self.adapter.eval()
            prediction = self.adapter.velocity_function(condition, differentiable=True)(z, t)
            estimate = (z.detach().float() - t[:, None, None] * prediction.float()) * mask[:, None]
            vae = self.pipeline.audio_autoencoder
            if any(p.requires_grad for p in vae.parameters()):
                raise RuntimeError('Paired decoded losses require a frozen VAE.')
            with torch.autocast('cuda', enabled=False):
                wave = vae.decode(estimate.float())[0, :, :n].float()
                with torch.no_grad():
                    truth = vae.decode(target.float())[0, :, :n].float()
                    source_wave = vae.decode(source.float())[0, :, :n].float()
                    truth_foa, source_foa = _audio_features(truth), _audio_features(source_wave)
                    truth_binaural = binaural_features(self.renderer(truth))
                spec, covariance, edited, _ = _audio_distances(_audio_features(wave), truth_foa, source_foa)
                edited = edited if edited is not None else covariance * 0
                binaural = binaural_distances(binaural_features(self.renderer(wave)), truth_binaural)
                foa_loss = covariance + .25 * edited + .1 * spec
                stereo_loss = binaural['cross_phase'] + .25 * binaural['interaural_level'] + .1 * binaural['log_spectral']
                loss = self.q['spatial_recipe']['decoded_audio_weight'] * foa_loss + config['binaural_weight'] * stereo_loss
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite paired FOA/binaural objective.')
            probe = None
            if self.step == 0:
                names = ['ar.editing_dit.postprocess_conv.weight', 'ar.editing_dit.transformer.layers.0.pre_norm.gamma']
                gradients = torch.autograd.grad(stereo_loss, [self.trainable[name] for name in names],
                                                retain_graph=True, allow_unused=True)
                probe = {name: None if g is None else float(g.norm()) for name, g in zip(names, gradients)}
                if any(value is None or not math.isfinite(value) or value <= 0 for value in probe.values()):
                    raise RuntimeError('Binaural objective missed DiT/shared parameters: ' + repr(probe))
            (loss / len(chosen)).backward()
            records.append(dict(ordinal=ordinal, operation=meta['operation'], loss=float(loss.detach()),
                covariance=float(covariance.detach()), edited_covariance=float(edited.detach()),
                w_spectral=float(spec.detach()), gradient_probe=probe,
                **{k: float(v.detach()) for k, v in binaural.items()}))
        return dict(decoded_audio_enabled=bool(records), selected_rows=len(records), rows=records,
                    target_source='own original paired latent decoded by frozen VAE',
                    exact_evaluation_metrics=False, renderer=self.renderer.identity['id'])


base.Learner = CompleteLearner

if __name__ == '__main__':
    base.main()
