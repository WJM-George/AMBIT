"""Full native joint OPSD with fixed-reference geometry and decoded FOA loss.

STE is disabled. The frozen retention reference never supplies an optimistic
execution reward. Actual current-model executions alone define AR preferences.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from scripts.t2a.rl import train_editing_opsd_stream as base
from stable_audio_tools.training.transfusion_opsd.editing_spatial_retention import (
    request_facts, propose_current_decision, request_spatial_measure,
    reference_field_targets, reference_kl, preserved_window_measure,
    execution_authorized_decision,
)
from stable_audio_tools.training.transfusion_opsd.native_coarse_choice_retention import field_balanced_native_set_ce
from stable_audio_tools.training.transfusion_opsd.native_prefix_supervision import field_balanced_native_ce
from stable_audio_tools.training.transfusion_opsd.native_paired_rf import paired_rf_example, paired_rf_loss
from stable_audio_tools.training.transfusion_opsd.supported_teacher import preserve_reference_support_mass


class SpatialLearner(base.Learner):
    def __init__(self, q, rank, world, out):
        if q['connected_credit'] or q.get('initial_overlay') is not None:
            raise ValueError('This recipe starts at original40k with STE disabled.')
        super().__init__(q, rank, world, out)
        self.reference = self.adapter.frozen_copy()
        self.reference.ar._forward_hooks.clear()
        self.reference.diffusion.model._forward_hooks.clear()
        reference_named = dict(self.reference.named_parameters())
        if any(reference_named[n].data_ptr() == p.data_ptr() for n, p in self.trainable.items()):
            raise RuntimeError('The frozen reference aliases a trainable student parameter.')
        if any(p.requires_grad for p in self.reference.parameters()):
            raise RuntimeError('Retention reference is not frozen.')
        self.aux_pending = None
        base.write(out / f'FIXED_REFERENCE_rank{rank}.json', dict(
            checkpoint=q['base_checkpoint'], copied_trainable_tensors=len(self.trainable),
            initial_overlay=None, independent_parameters=True, same_student_prefix=True,
            teacher_for_execution_preferences='Current student real execution, not reference potential',
            native_source_clap_shared_frozen=self.reference.ar.source_clap_model is self.adapter.ar.source_clap_model))

    def collect(self, ordinal):
        item = super().collect(ordinal)
        # Old transient probability targets are replaced before any backward.
        with torch.no_grad():
            self.progress('FROZEN_REFERENCE_CURRENT_PREFIX', ordinal=ordinal)
            teacher = self.reference.student_logits(item['obs'], item['tokens'][None, :-1])[0].float()
            item['reference_identity'] = self.q['base_checkpoint']['sha256']
            item['credit'] = []
            item['terminal_qualified'] = []
            if not item['metrics']:
                self.set_reference_targets(item, teacher, decision=None)
                return item
            facts = request_facts(item['row']['request'], item['row']['operation'])
            source_wave, _ = self.pipeline.decode_foa_latents(item['obs'].source_foa_latent,
                model_num_samples=[item['obs'].model_num_samples])
            source_wave = source_wave[..., :item['obs'].model_num_samples]
            semantic_reference = sum(m['semantic'] for m in item['metrics'] if m['plan_index'] == 0) / 2
            thresholds = self.q['spatial_recipe']
            for terminal, metric in zip(item['terminals'], item['metrics']):
                wave, _ = self.pipeline.decode_foa_latents(terminal['clean'],
                    model_num_samples=[item['obs'].model_num_samples])
                wave = wave[..., :item['obs'].model_num_samples]
                preservation = preserved_window_measure(wave, source_wave, item['plan'], facts)
                metric['unchanged_windows'] = preservation
                reasons = []
                if metric['semantic'] < semantic_reference - thresholds['semantic_tolerance']:
                    reasons.append('content_regression')
                if not metric['spatial']['available']:
                    reasons.append('unverified_target_space')
                elif metric['spatial']['mean_capped_angle_deg'] > thresholds['teacher_max_angle_deg']:
                    reasons.append('target_space_error')
                if metric['clipping_fraction'] >= .01:
                    reasons.append('clipping')
                if preservation['available'] and (preservation['covariance_error'] > thresholds['preservation_covariance_limit']
                        or preservation['missing_energy_fraction'] > .25):
                    reasons.append('unchanged_source_regression')
                metric['qualified_terminal'] = not reasons
                metric['qualification_failures'] = reasons
                item['terminal_qualified'].append(not reasons)
                del wave
            del source_wave
            decision = item['decision']
            # Reliability includes failed executions. The transcript/source
            # checks are gates; covariance penalties use the same source only
            # in temporally unedited windows, not hidden target audio.
            rewards = []
            for k in range(2):
                values = []
                for metric in item['metrics']:
                    if metric['plan_index'] != k:
                        continue
                    preserve = metric['unchanged_windows']
                    penalty = min(1., preserve['covariance_error']) if preserve['available'] else 0.
                    values.append(metric['reward'] - thresholds['preservation_reward_weight'] * penalty
                                  - float(not metric['qualified_terminal']))
                rewards.append(sum(values) / len(values))
            with self.adapter._amp():
                logits = self.adapter.student_logits(item['obs'], item['tokens'][None, :-1])[0].float()
            reference = logits[decision['position'] - 1, decision['legal_ids']].detach()
            support = [decision['legal_ids'].index(i) for i in decision['choice_ids']]
            conditional = (reference[support] + reference.new_tensor(rewards) / self.q['temperature']).softmax(-1)
            item['target'] = preserve_reference_support_mass(reference, support, conditional).probabilities.detach()
            item['plan_weights'] = ((1 - self.q['coverage_weight']) * conditional + self.q['coverage_weight'] * .5).detach()
            item['enabled'] = any(item['terminal_qualified'])
            item['rewards'] = rewards
            self.set_reference_targets(item, teacher, decision=execution_authorized_decision(
                item['decision'], item['metrics'], rewards))
        return item

    def set_reference_targets(self, item, teacher, *, decision):
        item['holds'], item['coarse'] = reference_field_targets(
            self.adapter.codec, item['tokens'].tolist(), item['plan'],
            lambda prefix: self.adapter.allowed_next_ids(item['obs'], prefix), teacher,
            decision=decision, radius_deg=self.q['spatial_recipe']['reference_cone_deg'])
        item['reference_exempt_field'] = None if decision is None else decision['field']

    def backward_self(self, item, *, scale=1.):
        self.adapter.eval()
        logits = self.adapter.student_logits(item['obs'], item['tokens'][None, :-1])[0].float()
        text = field_balanced_native_ce(logits, item['quoted']['targets'])
        coarse = field_balanced_native_set_ce(logits, item['coarse'])
        hold = reference_kl(logits, item['holds'])
        ar = logits.sum() * 0
        if item['enabled']:
            decision = item['decision']
            lp = logits[decision['position'] - 1, decision['legal_ids']].log_softmax(-1)
            target = item['target']
            ar = (target * (target.clamp_min(1e-30).log() - lp)).sum()
        weights = self.q['spatial_recipe']
        (scale * (weights['ar_teacher_weight'] * ar + text + weights['cone_weight'] * coarse
                  + weights['reference_KL_weight'] * hold)).backward()
        rf = []
        coefficients = item.get('terminal_training_coefficients')
        if coefficients is None:
            coefficients = [float(item['plan_weights'][terminal['plan_index']]) / 2
                            if item['enabled'] and item['terminal_qualified'][j] else 0.
                            for j, terminal in enumerate(item['terminals'])]
        if len(coefficients) != len(item['terminals']) or any(not math.isfinite(c) or c < 0 for c in coefficients):
            raise ValueError('Invalid stopped same-plan terminal coefficients.')
        for j, (terminal, coefficient) in enumerate(zip(item['terminals'], coefficients)):
            if coefficient == 0:
                continue
            if not item['enabled'] or not item['terminal_qualified'][j]:
                raise ValueError('An unqualified terminal cannot become a DiT target.')
            noise = torch.randn(terminal['clean'].shape, generator=torch.Generator(device='cpu').manual_seed(
                self.q['seed'] + 99_000_001 + self.step * 100 + self.rank * 10 + j)).to(self.device)
            t = torch.tensor([(.125, .375)[j % 2]], device=self.device)
            z, target = paired_rf_example(terminal['clean'], noise, t, item['obs'].source_attention_mask)
            pred = self.adapter.velocity_function(terminal['condition'], differentiable=True)(z, t)
            loss = paired_rf_loss(pred, target, item['obs'].source_attention_mask)
            (scale * weights['terminal_RF_weight'] * coefficient * loss).backward()
            rf.append(float(loss.detach()))
        return dict(AR_distillation=float(ar.detach()), request_text_CE=float(text.detach()),
            coarse_choice_CE=float(coarse.detach()), structure_KL=float(hold.detach()),
            frozen_reference=True, retained_native_decisions=len(item['holds']), terminal_RF=rf,
            terminal_RF_coefficients=coefficients,
            reference_exempt_field=item['reference_exempt_field'],
            qualified_terminals=sum(item['terminal_qualified']), proposed_terminals=len(item['terminals']),
            field=None if item['decision'] is None else item['decision']['field'], enabled=item['enabled'],
            execution_metrics=item['metrics'], requested_operation=item['row']['operation'],
            ordinal=item['row']['pair_ordinal'], connected_credit=False, credit=[])

    def backward_extra(self, batches):
        """One paired row per rank, after the large native batch is freed.

        Uses its own paired target latent/plan. Decoder parameters are frozen, but the
        clean-estimate gradient reaches DiT and the shared Transformer.
        """
        weight = self.q['spatial_recipe']['decoded_audio_weight']
        if weight <= 0:
            return dict(decoded_audio_enabled=False)
        from stable_audio_tools.training.losses.sceneplan_editing_audio import _audio_features, _audio_distances
        self.progress('PAIRED_DECODED_SPATIAL_BACKWARD')
        # Rotate over all native operations, including addition and removal.
        batch = batches[self.step % len(batches)]
        _, targets, metadata, masks = self.native._move_joint_batch(batch, self.device)
        index = self.step % len(metadata)
        meta = metadata[index]
        n, frames = int(meta['model_num_samples']), int(meta['latent_frames_valid'])
        ordinal = int(meta['pair_ordinal'])
        row = self.paired._db().execute('SELECT pair_id FROM pairs WHERE pair_ordinal=?', (ordinal,)).fetchone()
        if row[0] != meta['pair_id']:
            raise RuntimeError('Paired audio and latent identity differ.')
        source = meta['source_foa_latent'].to(self.device).float()
        if source.ndim == 2:
            source = source[None]
        mask = masks[index:index + 1]
        obs = self.adapter.observe_editing(sample_id=meta['pair_id'], request=meta['raw_edit_request'],
            source_foa_latent=source, source_attention_mask=mask, model_num_samples=n)
        condition = self.adapter.render_condition(obs, meta['model_sceneplan'])
        target = targets[index:index + 1].float()
        generator = torch.Generator(device='cpu').manual_seed(self.q['seed'] + 151_000_001 + self.step * 17 + self.rank)
        noise = torch.randn(target.shape, generator=generator).to(self.device)
        time_value = (.10, .20)[self.step % 2]
        t = target.new_tensor([time_value])
        z, _ = paired_rf_example(target, noise, t, mask)
        self.adapter.eval()
        prediction = self.adapter.velocity_function(condition, differentiable=True)(z, t)
        estimate = (z.detach().float() - t[:, None, None] * prediction.float()) * mask[:, None]
        vae = self.pipeline.audio_autoencoder
        if any(p.requires_grad for p in vae.parameters()):
            raise RuntimeError('FOA VAE must stay frozen.')
        with torch.autocast('cuda', enabled=False):
            wave = vae.decode(estimate.float())[0, :, :n].float()
            with torch.no_grad():
                # Full training materialization guarantees paired latents,
                # not a retained raw target WAV for every one of1M rows.
                target_wave = vae.decode(target.float())[0, :, :n].float()
                source_wave = vae.decode(source.float())[0, :, :n].float()
                target_features = _audio_features(target_wave)
                source_features = _audio_features(source_wave)
            spectral, spatial, edited, resolutions = _audio_distances(_audio_features(wave), target_features, source_features)
            edit_loss = edited if edited is not None else spatial * 0
            loss = spatial + .25 * edit_loss + .1 * spectral
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite decoded FOA auxiliary.')
        # Calibrate the effective parameter path at the first actual backward.
        probe = None
        if self.step == 0:
            names = ['ar.editing_dit.postprocess_conv.weight', 'ar.editing_dit.transformer.layers.0.pre_norm.gamma']
            values = torch.autograd.grad(loss, [self.trainable[name] for name in names], retain_graph=True, allow_unused=True)
            probe = {name: None if value is None else float(value.norm()) for name, value in zip(names, values)}
            if any(value is None or not math.isfinite(value) or value <= 0 for value in probe.values()):
                raise RuntimeError('Decoded spatial loss missed DiT/shared parameters.')
        (weight * loss).backward()
        return dict(decoded_audio_enabled=True, ordinal=ordinal, operation=meta['operation'],
            teacher_source='frozen VAE decode of original paired target latent; no request-only target access',
            loss=float(loss.detach()), covariance=float(spatial.detach()), edited_covariance=float(edit_loss.detach()),
            w_spectral=float(spectral.detach()), edit_resolutions=resolutions, timestep=time_value,
            weight=weight, gradient_probe=probe, vae_frozen=True)

    @torch.no_grad()
    def evaluate(self):
        reference = self.q.get('initial_evaluation_reference')
        if self.step == 0 and reference:
            # Reuse the already-measured original40k development baseline.
            # New trained candidates still execute normally on this panel.
            if self.rank == 0:
                for item in (reference['evaluation'], reference['configuration']):
                    if base.sha(item['path']) != item['sha256']:
                        raise ValueError('Initial evaluation reference changed.')
                old = base.read(reference['configuration']['path'])
                for key in ('base_checkpoint', 'initial_overlay', 'native_run_contract',
                            'validation_ordinals', 'evaluation_seeds', 'inference_steps'):
                    if old[key] != self.q[key]:
                        raise ValueError('Initial evaluation reference differs on ' + key)
                result = base.read(reference['evaluation']['path'])
                if result['step'] != 0:
                    raise ValueError('Only the original step0 may be reused as the baseline.')
                result['reused_evaluation_reference'] = reference
                self.record_evaluation(result)
            dist.barrier()
            return
        import gc
        import soundfile as sf
        from collections import Counter
        from scripts.t2a.experiments.ar_structured_v1 import data
        from stable_audio_tools.training.transfusion_opsd.editing_nine_metrics import EditingNineMetrics, aggregate_records
        self.adapter.eval()
        self.optimizer.zero_grad(set_to_none=True)
        gc.collect(); torch.cuda.empty_cache()
        scorer = None
        rows = []
        save_audio = self.q.get('evaluation_save_audio', True)
        store = None
        resumed_outputs = 0
        if not save_audio:
            from stable_audio_tools.training.transfusion_opsd.evaluation_case_store import EvaluationCaseStore
            from stable_audio_tools.training.transfusion_opsd import editing_nine_metrics
            store = EvaluationCaseStore(self.out / f'eval_cases_step{self.step:06d}', dict(
                step=self.step, model_sha256=base.tensor_digest(self.trainable),
                config_sha256=base.sha(self.q['config_path']), base_checkpoint=self.q['base_checkpoint'],
                native_contract_sha256=base.sha(self.q['native_run_contract']),
                validation_ordinals=self.q['validation_ordinals'], evaluation_seeds=self.q['evaluation_seeds'],
                inference_steps=self.q['inference_steps'], cfg_scale=1.,
                intervention=self.q.get('evaluation_intervention'),
                scorer_sha256=base.sha(editing_nine_metrics.__file__), evaluator_sha256=base.sha(__file__),
                audio_retained=False))
        try:
            for ordinal in self.q['validation_ordinals'][self.rank::self.world]:
                completed = {seed: store.get(ordinal, seed) if store else None
                             for seed in self.q['evaluation_seeds']}
                if all(record is not None for record in completed.values()):
                    rows.extend(completed.values())
                    resumed_outputs += len(completed)
                    continue
                if scorer is None:
                    self.progress('LOAD_NINE_METRIC_OBSERVERS')
                    scorer = EditingNineMetrics(self.device, self.out / 'validation_target_features')
                self.progress('EVALUATE_NATIVE_AUDIO', ordinal=ordinal)
                row, obs = self.validation_inputs.observe(self.adapter, ordinal)
                plan, tokens = self.adapter.native_plan(obs)
                # Validation labels become accessible only after native decode.
                batch = data.collate([self.validation[ordinal]], pad_id=self.adapter.codec.pad_id, joint=True)
                _, _, metadata, _ = self.native._move_joint_batch(batch, self.device)
                expected = metadata[0]['model_sceneplan']
                target_path = self.validation._db().execute('SELECT target_foa_path FROM pairs WHERE pair_ordinal=?', (ordinal,)).fetchone()[0]
                facts = request_facts(row['request'], row['operation'])
                for seed in self.q['evaluation_seeds']:
                    if completed[seed] is not None:
                        rows.append(completed[seed])
                        resumed_outputs += 1
                        continue
                    latent = self.pipeline.sample_edited_latents(obs.source_foa_latent, obs.source_attention_mask, [plan],
                        model_num_samples=[obs.model_num_samples], steps=self.q['inference_steps'], cfg_scale=1., seed=seed)
                    wave, _ = self.pipeline.decode_foa_latents(latent, model_num_samples=[obs.model_num_samples])
                    wave = wave[..., :obs.model_num_samples]
                    prefix = self.out / f'eval_step{self.step:06d}_rank{self.rank}_{ordinal}_{seed}'
                    audio_path = str(prefix) + '.wav' if save_audio else None
                    if save_audio:
                        sf.write(audio_path, wave[0].float().cpu().T.numpy(), 44100, subtype='FLOAT')
                    feature_path = Path(str(prefix) + '.npz')
                    temporary_features = Path(str(prefix) + '.partial.npz')
                    metric = scorer.score(wave, target_path, row['pair_id'], obs.model_num_samples, temporary_features)
                    temporary_features.replace(feature_path)
                    metric['features'] = str(feature_path)
                    record = dict(ordinal=ordinal, pair_id=row['pair_id'], operation=row['operation'], seed=seed,
                        request=row['request'], plan=plan, expected_evaluation_only=expected, audio=audio_path,
                        request_spatial=request_spatial_measure(wave, expected, facts), **metric)
                    if store:
                        store.put(record)
                    rows.append(record)
                    del latent, wave
                del batch
            base.write(self.out / f'eval_step{self.step:06d}_rank{self.rank}.json', dict(step=self.step, rows=rows,
                audio_retained=save_audio, resumed_outputs=resumed_outputs,
                scope='Fixed development validation; same stored source FOA latents, native20EulerCFG1, nine frozen paper metrics. Not COMMON1000.'))
        finally:
            scorer = None
            gc.collect(); torch.cuda.empty_cache()
        dist.barrier()
        if self.rank == 0:
            all_rows = []
            for rank in range(self.world):
                rank_result = base.read(self.out / f'eval_step{self.step:06d}_rank{rank}.json')
                all_rows.extend(rank_result['rows'])
            if len(all_rows) != len(self.q['validation_ordinals']) * len(self.q['evaluation_seeds']):
                raise RuntimeError('Incomplete development evaluation.')
            metrics = aggregate_records(all_rows)
            angles, trajectories = Counter(), Counter()
            for row in all_rows[::len(self.q['evaluation_seeds'])]:
                for source in row['plan']['sources']:
                    trajectory = source['trajectory']
                    trajectories[trajectory['type']] += 1
                    if trajectory['type'] == 'static':
                        angles[trajectory['position']['azimuth_deg']] += 1
            self.record_evaluation(dict(step=self.step,
                requests=len(self.q['validation_ordinals']), outputs=len(all_rows), metrics=metrics,
                predicted_trajectory_counts=dict(trajectories), predicted_static_angle_counts=dict(angles),
                audio_retained=save_audio,
                comparison='Use this same validation panel at step0; do not compare its absolute Fréchet values to COMMON1000.',
                target_audio_in_inference=False))
        dist.barrier()

    def record_evaluation(self, result):
        path = self.out / f'EVALUATION_step{self.step:06d}.json'
        base.write(path, result)
        policy = self.q.get('top_checkpoint_policy')
        if policy:
            from stable_audio_tools.training.transfusion_opsd.top_checkpoints import retain_top_checkpoints
            retain_top_checkpoints(self.out, path, self.q['config_path'], policy)


# The common runner owns sampling, one synchronized optimizer, atomic recovery
# and normal inference. These overrides are local to this executable process.
base.request_facts = request_facts
base.propose_current_decision = propose_current_decision
base.request_spatial_measure = request_spatial_measure
base.Learner = SpatialLearner


if __name__ == '__main__':
    base.main()
