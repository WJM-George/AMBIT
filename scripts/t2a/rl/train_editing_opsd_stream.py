"""Resumable Editing execution-feedback joint self-distillation.

Run with torchrun. All ranks use the same shared AR/DiT parameter objects and
one optimizer. Sampling/teachers stop gradients; fitting uses fresh forwards.
"""
from __future__ import annotations

import argparse
import copy
from datetime import timedelta
import faulthandler
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist

from stable_audio_tools.training.transfusion_opsd.editing_stream import (
    OrdinalStream, RequestInputs, propose_current_decision, request_facts,
    request_spatial_measure, training_partitions, homogeneous_microbatches, resize_stream_position)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def validate_resume_configuration(config_path, checkpoint_sha, *, checkpoint_step=None):
    """Permit an archived schedule or request-batch change at a saved boundary.

    Read files rather than the runtime config, which also contains batch-size
    execution overrides. New checkpoints bind directly to the new config.
    """
    if sha(config_path) == checkpoint_sha:
        return
    current = read(config_path)
    parent_path = current.get('resume_config_parent')
    if parent_path is None or sha(parent_path) != checkpoint_sha:
        raise ValueError('Resume requires the same configuration or its verified parent.')
    parent = read(parent_path)
    schedule_fields = {'maximum_updates', 'save_every', 'recovery_every_seconds',
                       'initial_gate', 'resume_config_parent'}
    batch_fields = {'request_rows_per_rank', 'global_request_batch'}
    batch_changed = any(parent.get(k) != current.get(k) for k in batch_fields)
    allowed = schedule_fields
    if batch_changed:
        # Batch size affects sampling per update, not the dataset partition,
        # sampler cursors, optimizer, loss weights or paired supervision.
        expected = dict(after_update=checkpoint_step,
                        previous_global_request_batch=parent.get('global_request_batch'),
                        global_request_batch=current.get('global_request_batch'))
        transition = current.get('request_batch_transition')
        if (type(checkpoint_step) is not int or checkpoint_step < 0
                or transition != expected):
            raise ValueError('Request-batch change requires its explicit saved update boundary.')
        for config in (parent, current):
            world = len(config['physical_gpus'])
            if (world < 1 or any(type(config.get(k)) is not int or config[k] < 1 for k in batch_fields)
                    or config['request_rows_per_rank'] * world != config['global_request_batch']):
                raise ValueError('Request batch must match the unchanged rank topology.')
        allowed = batch_fields | {'resume_config_parent', 'request_batch_transition'}
    if ({k:v for k,v in parent.items() if k not in allowed} !=
            {k:v for k,v in current.items() if k not in allowed}):
        raise ValueError('Resume revision cannot change the model, data, optimizer or comparison recipe.')
    if any(type(current[k]) is not int or current[k] < 1
           for k in ('maximum_updates', 'save_every', 'recovery_every_seconds')):
        raise ValueError('Resume schedule values must be positive integers.')


def is_milestone(step, q):
    return step > 0 and (step % q['save_every'] == 0 or step == q['maximum_updates'])


def update_events(step, q, *, since_save, since_evaluation, extra_evaluations=()):
    evaluate = (step == 1 or is_milestone(step, q) or step in extra_evaluations
                or since_evaluation >= q['evaluate_every_seconds'])
    # Every evaluation starts from a persisted full optimizer boundary.
    save = evaluate or since_save >= q['recovery_every_seconds']
    return save, evaluate


def synchronize_gradients(parameters, world):
    """Bucketed sum/mean, once after all native and self-teacher backwards.

    Custom adapter methods bypass nn.Module.forward, so a DDP wrapper alone
    would not cover them. Explicit reduction includes all trainable parameters.
    """
    bucket, elements = [], 0
    def reduce(values):
        flat = torch.cat([p.grad.reshape(-1) for p in values])
        if not torch.isfinite(flat).all():
            raise RuntimeError('Nonfinite gradient before distributed update.')
        dist.all_reduce(flat)
        flat.div_(world)
        offset = 0
        for p in values:
            p.grad.copy_(flat[offset:offset + p.numel()].view_as(p))
            offset += p.numel()
    for p in parameters:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        if bucket and elements + p.numel() > 16 * 1024 * 1024:
            reduce(bucket)
            bucket, elements = [], 0
        bucket.append(p)
        elements += p.numel()
    if bucket:
        reduce(bucket)


def tensor_digest(named):
    h = hashlib.sha256()
    for name, value in sorted(named.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class Learner:
    def __init__(self, q, rank, world, out):
        from scripts.t2a.experiments.ar_structured_v1 import runtime as native, initialization, model
        from scripts.t2a.experiments.ar_structured_v1.pipeline import load_candidate
        from stable_audio_tools.training.transfusion_opsd.editing_clap44_adapter import StructuredEditingOPSDAdapter
        from stable_audio_tools.training.transfusion_opsd.editing_native_numerics import install_editing_native_numerics
        from stable_audio_tools.training.transfusion_opsd.native_latent_clap import FrozenNativeLatentCLAP
        from safetensors.torch import load_file
        self.q, self.rank, self.world, self.out, self.native = q, rank, world, out, native
        self.device = torch.device('cuda', rank)
        self.cfg = copy.deepcopy(read(q['native_run_contract'])['recipe'])
        _, self.restore_flash = native.original_runtime.install_flash(self.cfg['base_AR_configuration']['numerical_execution'])
        self.runtime_dir = out/'runtime_attempts'/f'rank{rank}_{os.getpid()}_{time.time_ns()}'
        self.numerics, self.finish_numerics = install_editing_native_numerics(self.cfg, self.runtime_dir, reference_rank='0')
        self.pipeline, self.identity = load_candidate(q['base_checkpoint']['path'], device=str(self.device), load_audio_autoencoder=True)
        if self.identity['checkpoint_sha256'] != q['base_checkpoint']['sha256']:
            raise ValueError('Wrong original joint checkpoint.')
        self.adapter = StructuredEditingOPSDAdapter.from_editing_pipeline(self.pipeline, fork=False).eval()
        self.adapter.forward = types.MethodType(model.model_forward, self.adapter)
        self.adapter.ar.activation_checkpointing = True
        self.adapter.diffusion.model.model.activation_checkpointing = False
        named = dict(self.adapter.named_parameters())
        self.trainable = {n: p for n, p in named.items() if p.requires_grad}
        if q.get('initial_overlay') is not None:
            state = load_file(q['initial_overlay']['path'], device='cpu')
            if sha(q['initial_overlay']['path']) != q['initial_overlay']['sha256']:
                raise ValueError('Wrong E79 initial overlay.')
            if set(state) != set(self.trainable):
                raise ValueError('E79 overlay and full native trainable scope differ.')
            with torch.no_grad():
                for name, value in state.items():
                    named[name].copy_(value.to(named[name].device))
            del state
        self.cfg['learning_rates'] = q['learning_rates']
        self.groups, self.scope = initialization.optimizer_groups(self.adapter, self.cfg)
        self.parameters = [p for g in self.groups for p in g['params']]
        self.optimizer = torch.optim.AdamW(self.groups, betas=(.9, .95), weight_decay=.001, fused=True)
        self.teacher = native.make_teacher(self.adapter, self.cfg).to(self.device).eval().requires_grad_(False)
        self.clap = FrozenNativeLatentCLAP(self.adapter.ar.source_clap_model)
        self.requests = RequestInputs(self.cfg['data']['train'])
        self.validation_inputs = RequestInputs(self.cfg['data']['validation'])
        self.paired = native.build_dataset(self.cfg, self.adapter.codec, self.adapter.prompt_conditioner.tokenizer, 'train')
        self.validation = native.build_dataset(self.cfg, self.adapter.codec, self.adapter.prompt_conditioner.tokenizer, 'validation')
        a, b = training_partitions(len(self.paired), q['seed'], q['request_fraction'], q['development_ordinals'])
        self.request_stream = OrdinalStream(a, seed=q['seed'] + 1, rank=rank, world=world)
        self.pair_stream = OrdinalStream(b, seed=q['seed'] + 2, rank=rank, world=world)
        self.partition = dict(request_only=len(a), paired_only=len(b), rows=len(self.paired), overlap=0)
        self.step = 0
        self.asr = None
        self.costs = dict(rollout_audio=0, request_rows=0, paired_rows=0, execution_teachers=0, joint_updates=0)
        self.adapter.ar.register_forward_hook(lambda *unused: self.count('AR_forwards'))
        self.adapter.diffusion.model.register_forward_hook(lambda *unused: self.count('DiT_forwards'))

    def count(self, key):
        self.costs[key] = self.costs.get(key, 0) + 1

    def progress(self, phase, **details):
        write(self.out/f'PHASE_rank{self.rank}.json', dict(phase=phase, next_update=self.step+1,
              rank=self.rank, pid=os.getpid(), time=time.time(), **details))

    @torch.no_grad()
    def text_feature(self, text):
        hidden = self.teacher([text], self.device)
        return self.clap.text_features(hidden, hidden)['semantic']

    @torch.no_grad()
    def execute(self, obs, plan, seed, feature, facts, *, register_plan=None, capture=None):
        calls = [0]
        def hook(module, args, value):
            if calls[0] == self.q['credit_state_index']:
                capture.append(dict(state=args[0].detach().clone(), time=args[1].detach().clone(), velocity=value.detach().clone()))
            calls[0] += 1
        handle = self.adapter.diffusion.model.register_forward_hook(hook) if capture is not None else None
        try:
            z = self.pipeline.sample_edited_latents(obs.source_foa_latent, obs.source_attention_mask, [plan],
                model_num_samples=[obs.model_num_samples], steps=self.q['inference_steps'], cfg_scale=1., seed=seed)
        finally:
            if handle:
                handle.remove()
        if capture is not None and (len(capture) != 1 or calls[0] != self.q['inference_steps']):
            raise RuntimeError('Credit requires an actual native CFG1 visited state.')
        wave, _ = self.pipeline.decode_foa_latents(z, model_num_samples=[obs.model_num_samples])
        wave = wave[..., :obs.model_num_samples]
        semantic = float((self.clap(z, obs.source_attention_mask)['semantic'] * feature).sum())
        spatial = request_spatial_measure(wave, register_plan or plan, facts)
        clipping = float((wave.abs() >= 1.).float().mean())
        # E79-style spatial execution feedback. CLAP remains a content gate,
        # rather than silently introducing a new combined reward formula.
        reward = -spatial['mean_capped_angle_deg'] / 30. if spatial['available'] else 0.
        self.count('rollout_audio')
        return z.detach(), wave, dict(semantic=semantic, spatial=spatial, clipping_fraction=clipping, reward=reward)

    def collect(self, ordinal):
        from stable_audio_tools.training.transfusion_opsd.native_prefix_supervision import _sites
        from stable_audio_tools.training.transfusion_opsd.native_coarse_choice_retention import native_azimuth_cone_targets
        from stable_audio_tools.training.transfusion_opsd.request_grounded_text_retention import request_quoted_text_targets
        from stable_audio_tools.training.transfusion_opsd.supported_teacher import preserve_reference_support_mass
        self.adapter.eval()
        self.progress('READ_REQUEST', ordinal=ordinal)
        row, obs = self.requests.observe(self.adapter, ordinal)
        with torch.no_grad():
            self.progress('NATIVE_PLAN', ordinal=ordinal)
            plan, tokens = self.adapter.native_plan(obs)
            self.progress('CURRENT_PREFIX_TARGETS', ordinal=ordinal)
            canonical = self.adapter.codec.encode(plan)['input_ids'].tolist()
            if tokens.tolist() != canonical:
                # Keep evidence of legal noncanonical segmentations and any
                # true mismatch. Never substitute canonical ids for rollout ids.
                write(self.out / f'NATIVE_TOKEN_ALIGNMENT_step{self.step + 1:06d}_rank{self.rank}_ordinal{ordinal}.json',
                      dict(step=self.step + 1, rank=self.rank, ordinal=ordinal, plan=plan,
                           native_tokens=tokens.tolist(), canonical_tokens=canonical))
            facts = request_facts(row['request'], row['operation'])
            decision = propose_current_decision(self.adapter, obs, plan, tokens.tolist(), facts, step=self.step)
            logits = self.adapter.student_logits(obs, tokens[None, :-1])[0].float()
            coarse = native_azimuth_cone_targets(self.adapter.codec, tokens.tolist(), plan,
                lambda prefix: self.adapter.allowed_next_ids(obs, prefix), radius_deg=10.)
            quoted = request_quoted_text_targets(self.adapter.codec, tokens.tolist(), plan, row['request'],
                lambda prefix: self.adapter.allowed_next_ids(obs, prefix)) if facts else dict(targets=[], fields=[])
            atoms, _ = _sites(self.adapter.codec, tokens.tolist())
            holds = []
            for key, pos in atoms.items():
                if decision and pos == decision['position']:
                    continue
                ids = sorted(self.adapter.allowed_next_ids(obs, tokens[:pos].tolist()))
                if len(ids) > 1:
                    holds.append(dict(position=pos, ids=ids, p=logits[pos - 1, ids].softmax(-1).detach()))
            result = dict(row=row, obs=obs, plan=plan, tokens=tokens, coarse=coarse, quoted=quoted,
                          holds=holds, decision=decision, terminals=[], metrics=[], credit=[], enabled=False)
            self.count('request_rows')
            if decision is None:
                return result
            # Current student's predicted retained content plus explicitly
            # requested target text; no old/source annotation or paired label.
            texts = [s.get('description', s.get('transcript', '')) for s in plan['sources']]
            texts.extend(facts['fields'].values())
            feature = self.text_feature(' '.join(texts))
            for k, p in enumerate(decision['plans']):
                for j in range(2):
                    self.progress('EXECUTE_REQUEST', ordinal=ordinal, plan_index=k, noise_index=j)
                    seed = self.q['seed'] + self.step * 1009 + self.rank * 100 + j
                    captured = [] if k == 0 else None
                    z, wave, metric = self.execute(obs, p, seed, feature, facts, register_plan=plan, capture=captured)
                    result['terminals'].append(dict(plan_index=k, seed=seed, clean=z, condition=self.adapter.render_condition(obs, p)))
                    result['metrics'].append(dict(plan_index=k, seed=seed, **metric))
                    if captured:
                        state = captured[0]
                        alt_condition = self.adapter.render_condition(obs, decision['plans'][1])
                        alternate = self.adapter.velocity_function(alt_condition, differentiable=False)(state['state'], state['time'])
                        result['credit'].append(dict(**state, alternate=alternate.detach(), seed=seed))
                    del wave
            rewards = [sum(x['reward'] for x in result['metrics'] if x['plan_index'] == k)/2 for k in range(2)]
            means = [sum(x['semantic'] for x in result['metrics'] if x['plan_index'] == k)/2 for k in range(2)]
            reference = logits[decision['position'] - 1, decision['legal_ids']].detach()
            support = [decision['legal_ids'].index(t) for t in decision['choice_ids']]
            # A large content fall cannot be bought by an angular reward.
            if means[1] < means[0] - .04:
                rewards[1] = min(rewards[1], rewards[0] - .1)
            conditional = (reference[support] + reference.new_tensor(rewards)/self.q['temperature']).softmax(-1).detach()
            target = preserve_reference_support_mass(reference, support, conditional).probabilities.detach()
            weights = ((1 - self.q['coverage_weight']) * conditional + self.q['coverage_weight'] * .5).detach()
            available = all(m['spatial']['available'] for m in result['metrics'])
            available = available and all(m['clipping_fraction'] < .01 for m in result['metrics'])
            result.update(target=target, plan_weights=weights, rewards=rewards, enabled=available,
                          credit_eligible=available and rewards[1] > rewards[0] and means[1] >= means[0] - .04,
                          support_mass=float(reference.softmax(-1)[support].sum()))
            if available:
                self.count('execution_teachers')
            return result

    def backward_self(self, item, *, scale=1.):
        from stable_audio_tools.training.transfusion_opsd.native_prefix_supervision import field_balanced_native_ce
        from stable_audio_tools.training.transfusion_opsd.native_coarse_choice_retention import field_balanced_native_set_ce
        from stable_audio_tools.training.transfusion_opsd.native_paired_rf import paired_rf_example, paired_rf_loss
        from stable_audio_tools.training.transfusion_opsd.native_prediction_credit import hard_native_prediction
        self.adapter.eval()
        logits = self.adapter.student_logits(item['obs'], item['tokens'][None, :-1])[0].float()
        text = field_balanced_native_ce(logits, item['quoted']['targets'])
        coarse = field_balanced_native_set_ce(logits, item['coarse'])
        hold = torch.stack([(v['p'] * (v['p'].clamp_min(1e-30).log() - logits[v['position'] - 1, v['ids']].log_softmax(-1))).sum()
                            for v in item['holds']]).mean() if item['holds'] else logits.sum() * 0
        ar = logits.sum() * 0
        if item['enabled']:
            d, target = item['decision'], item['target']
            lp = logits[d['position'] - 1, d['legal_ids']].log_softmax(-1)
            ar = (target * (target.clamp_min(1e-30).log() - lp)).sum()
        (scale * (ar + text + coarse + 4 * hold)).backward()
        credit_rows = []
        for state in item['credit']:
            d = item['decision']
            prefix = torch.tensor([d['prefix']], device=self.device)
            choices = self.adapter.student_logits(item['obs'], prefix)[0, -1, d['choice_ids']].float()
            hard, alternate = state['velocity'].float(), state['alternate'].float()
            secant_scale = paired_rf_loss(hard, alternate, item['obs'].source_attention_mask).detach()
            if float(secant_scale) <= 1e-10:
                continue
            probe = hard.clone().requires_grad_(True)
            pred = hard_native_prediction(probe, [hard, alternate], choices, hard_index=0,
                connect_decisions=self.q['connected_credit'])
            if not torch.equal(pred.detach(), hard):
                raise RuntimeError('Discrete credit changed the hard forward.')
            credit = paired_rf_loss(pred, alternate, item['obs'].source_attention_mask)/secant_scale
            derivative = torch.autograd.grad(credit, choices, allow_unused=True, retain_graph=True)[0]
            if self.q['connected_credit'] and (derivative is None or not torch.isfinite(derivative).all()):
                raise RuntimeError('Connected native credit lacks its decision gradient.')
            if not self.q['connected_credit'] and derivative is not None:
                raise RuntimeError('Detached comparison leaked a decision gradient.')
            reached = None
            if self.step == 0 and not credit_rows:
                named = self.trainable
                probes = [named['ar.plan_adapter.plan_head.weight'], named['ar.editing_dit.transformer.layers.0.pre_norm.gamma'],
                          named['ar.editing_dit.postprocess_conv.weight']]
                values = torch.autograd.grad(credit, probes, allow_unused=True, retain_graph=True)
                reached = {key: None if value is None else float(value.detach().norm())
                           for key,value in zip(('native_AR_head','shared_Transformer','DiT_private'), values)}
                if self.q['connected_credit']:
                    if not all(reached[k] is not None and reached[k] > 0 for k in ('native_AR_head','shared_Transformer')):
                        raise RuntimeError('Native credit did not reach AR head and shared Transformer.')
                elif any(v is not None for v in reached.values()):
                    raise RuntimeError('Detached native credit reached model parameters.')
                if reached['DiT_private'] is not None:
                    raise RuntimeError('Planner secant must not relabel DiT predictions across plans.')
            weight = scale * self.q['credit_weight'] * item.get('support_mass', 0.) * float(item.get('credit_eligible', False))
            (weight * credit / len(item['credit'])).backward()
            credit_rows.append(dict(seed=state['seed'], weight=weight, MSE_scale=float(secant_scale),
                selected_logit_derivative=None if derivative is None else derivative.detach().tolist(),
                native_parameter_gradient_norms=reached,
                hard_forward_exact=True, target_scope='Planner-only stopped same-state secant; never a cross-plan DiT regression label.'))
        rf = []
        if item['enabled']:
            for j, terminal in enumerate(item['terminals']):
                noise = torch.randn(terminal['clean'].shape, generator=torch.Generator(device='cpu').manual_seed(
                    self.q['seed'] + 99_000_001 + self.step * 100 + self.rank * 10 + j)).to(self.device)
                t = torch.tensor([(.125, .375)[j % 2]], device=self.device)
                z, target = paired_rf_example(terminal['clean'], noise, t, item['obs'].source_attention_mask)
                pred = self.adapter.velocity_function(terminal['condition'], differentiable=True)(z, t)
                loss = paired_rf_loss(pred, target, item['obs'].source_attention_mask)
                (scale * item['plan_weights'][terminal['plan_index']] * loss / 2).backward()
                rf.append(float(loss.detach()))
        return dict(AR_distillation=float(ar.detach()), request_text_CE=float(text.detach()), coarse_choice_CE=float(coarse.detach()),
                    structure_KL=float(hold.detach()), terminal_RF=rf, actual_prefix=None if item['decision'] is None else item['decision']['prefix'],
                    field=None if item['decision'] is None else item['decision']['field'], enabled=item['enabled'],
                    execution_metrics=item['metrics'], requested_operation=item['row']['operation'], ordinal=item['row']['pair_ordinal'],
                    connected_credit=self.q['connected_credit'], credit=credit_rows)

    def train_step(self):
        from scripts.t2a.experiments.ar_structured_v1 import data
        schedule = self.q.get('execution_schedule')
        if schedule:
            selected = schedule['previous'] if self.step < schedule['after_update'] else schedule['steady']
            self.q.update(selected)
            self.q['paired_rows_per_rank'] = selected['global_paired_batch'] // self.world
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        count = self.q['request_rows_per_rank']
        items = []
        for j in range(count):
            warm_index = self.step * self.q['global_request_batch'] + self.rank * count + j
            ordinal = self.q['development_ordinals'][warm_index] if warm_index < len(self.q['development_ordinals']) else self.request_stream.take(1)[0]
            items.append(self.collect(ordinal))
        torch.cuda.synchronize(self.device)
        collected = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        self.progress('SELF_DISTILLATION_BACKWARD', ordinals=[x['row']['pair_ordinal'] for x in items])
        requests = [self.backward_self(item, scale=1/count) for item in items]
        stats = dict(request_updates=requests, credit=[c for r in requests for c in r['credit']],
                     enabled=any(r['enabled'] for r in requests), connected_credit=self.q['connected_credit'])
        torch.cuda.synchronize(self.device)
        self_distilled = time.perf_counter()
        pairs = self.pair_stream.take(self.q['paired_rows_per_rank'])
        self.progress('LOAD_PAIRED_BATCH', ordinals=pairs)
        # Native collation requires homogeneous432/648-frame buckets. Group
        # within the already-sampled update; do not drop or resample any row.
        indices = homogeneous_microbatches(pairs, lambda i:self.requests.row(i)['latent_bucket_frames'], self.q['paired_microbatch'],
            bucket_limits={432:self.cfg['performance']['short_batch_size'], 648:self.cfg['performance']['long_batch_size']})
        batches = [data.collate([self.paired[i] for i in ids], pad_id=self.adapter.codec.pad_id, joint=True) for ids in indices]
        # Denominators use all valid values in the complete global update.
        den = torch.zeros(3, dtype=torch.float64, device=self.device)
        for b in batches:
            ar, target, meta, mask = self.native._move_joint_batch(b, self.device)
            den += den.new_tensor([(ar['plan_labels'] != -100).sum(), mask.sum() * 64, len(meta)])
        self.progress('GLOBAL_DENOMINATORS')
        dist.all_reduce(den)
        paired_loss = 0.
        self.adapter.train()
        for j, b in enumerate(batches):
            self.progress('PAIRED_BACKWARD', microbatch=j, microbatches=len(batches))
            loss, _, _, _ = self.native.batch_loss(self.adapter, self.teacher, b, self.cfg, self.device,
                self.step * 100 + j, self.rank, self.world, den)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite native joint loss.')
            loss.backward()
            paired_loss += float(loss.detach())
        missing = [n for n,p in self.trainable.items() if p.grad is None]
        if missing:
            raise RuntimeError('Native full joint loss missed trainable parameters: ' + repr(missing))
        extra = self.backward_extra(batches)
        synchronize_gradients(self.parameters, self.world)
        norms = {g['group_name']: float(torch.nn.utils.clip_grad_norm_(g['params'], 1., error_if_nonfinite=True)) for g in self.groups}
        self.optimizer.step()
        self.step += 1
        self.costs['paired_rows'] += len(pairs)
        self.count('joint_updates')
        torch.cuda.synchronize(self.device)
        trained = time.perf_counter()
        self.adapter.eval()
        audit = self.step == 1 or self.step % self.q.get('native_plan_audit_every', 1) == 0
        with torch.no_grad():
            after = []
            for item in items if audit else []:
                after_plan, after_tokens = self.adapter.native_plan(item['obs'])
                after.append(dict(ordinal=item['row']['pair_ordinal'], plan_before=item['plan'], plan_after=after_plan, normal_tokens_after=after_tokens.tolist()))
        torch.cuda.synchronize(self.device)
        finished = time.perf_counter()
        stats.update(paired_loss=paired_loss, paired_ordinals=pairs, clip_norms=norms, extra_objectives=extra,
                     actual_native_outputs=after, native_plan_audit_performed=audit,
                     performance=dict(step_seconds=finished-started, collection_seconds=collected-started,
                         self_distillation_seconds=self_distilled-collected, paired_update_seconds=trained-self_distilled,
                         native_plan_audit_seconds=finished-trained,
                         peak_allocated_MiB=torch.cuda.max_memory_allocated(self.device)/1024**2,
                         peak_reserved_MiB=torch.cuda.max_memory_reserved(self.device)/1024**2,
                         paired_microbatch_sizes=[len(ids) for ids in indices],
                         request_rows_per_rank=count,
                         global_request_batch=count * self.world,
                         paired_rows_per_rank=self.q['paired_rows_per_rank'],
                         global_paired_batch=self.q['global_paired_batch']),
                     global_denominators=den.tolist(), step=self.step, costs=dict(self.costs))
        return stats

    def backward_extra(self, batches):
        """Optional auxiliary backward after native microbatch graphs are freed."""
        return {}

    def save(self, elapsed):
        rolling = self.q.get('checkpoint_retention', {}).get('rolling_recoveries', 2)
        if type(rolling) is not int or rolling not in (1, 2):
            raise ValueError('Keep one or two atomic rolling recovery states.')
        local = dict(request=self.request_stream.state_dict(), paired=self.pair_stream.state_dict(),
                     random=random.getstate(), numpy=np.random.get_state(), cpu_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state(self.device), costs=self.costs)
        all_states = [None] * self.world
        dist.all_gather_object(all_states, local)
        hashes = [None] * self.world
        digest = tensor_digest(self.trainable)
        dist.all_gather_object(hashes, digest)
        if len(set(hashes)) != 1:
            raise RuntimeError('Ranks have different trainable model parameters.')
        if self.rank == 0:
            state = dict(schema='editing_execution_self_distillation_stream_v1', config_sha256=sha(self.q['config_path']),
                         model={n:p.detach().cpu() for n,p in self.trainable.items()}, optimizer=self.optimizer.state_dict(),
                         step=self.step, rank_states=all_states, world_size=self.world, elapsed_seconds=elapsed,
                         model_sha256=digest, original_checkpoint=self.q['base_checkpoint'], initial_overlay=self.q['initial_overlay'],
                         execution={k:self.q[k] for k in ('request_rows_per_rank','global_request_batch',
                             'paired_rows_per_rank','paired_microbatch','global_paired_batch','native_plan_audit_every')},
                         execution_schedule=self.q.get('execution_schedule'),
                         training_source_sha256=sha(__file__))
            temp, latest, previous = self.out/'resume.tmp.pt', self.out/'resume_latest.pt', self.out/'resume_previous.pt'
            torch.save(state, temp)
            if latest.exists() and rolling == 2:
                previous_temp = self.out/'resume_previous.tmp.pt'
                previous_temp.unlink(missing_ok=True)
                os.link(latest, previous_temp)
                previous_temp.replace(previous)
            temp.replace(latest)
            if is_milestone(self.step, self.q):
                milestones = self.out/'checkpoints'
                milestones.mkdir(exist_ok=True)
                milestone = milestones/f'step-{self.step:08d}.pt'
                if not milestone.exists():
                    # Hardlink is safe: recovery rotation replaces directory
                    # entries and never rewrites an existing checkpoint inode.
                    os.link(latest, milestone)
            write(self.out/'RESUME.json', dict(path=str(latest), step=self.step, model_sha256=digest,
                  includes=['full_trainable_weights','Adam_moments','per_rank_sampler_cursors','per_rank_RNG','costs'],
                  elapsed_seconds=elapsed, original_checkpoint_required=True, execution=state['execution'], execution_schedule=state['execution_schedule']))
        dist.barrier()

    def resume(self, path, *, resize=False, state=None):
        if state is None:
            state = torch.load(path, map_location='cpu', weights_only=False)
        if not resize:
            validate_resume_configuration(self.q['config_path'], state['config_sha256'], checkpoint_step=state['step'])
            if state['world_size'] != self.world:
                raise ValueError('Resume requires the same world size.')
        if resize:
            parent = read(self.q['resize_parent_config'])
            ignored = {'physical_gpus','paired_rows_per_rank','request_rows_per_rank','resize_parent_config','config_path',
                       'paired_microbatch','global_paired_batch','native_plan_audit_every','execution_schedule'}
            if ({k:v for k,v in parent.items() if k not in ignored} != {k:v for k,v in self.q.items() if k not in ignored}
                    or state['config_sha256'] != sha(self.q['resize_parent_config'])
                    or (state['world_size'], self.world) not in ((2, 4), (4, 8))):
                raise ValueError('Only an explicit equivalent2→4 or4→8rank transition is supported.')
        with torch.no_grad():
            for n,p in self.trainable.items():
                p.copy_(state['model'][n].to(p.device))
        self.optimizer.load_state_dict(state['optimizer'])
        local = state['rank_states'][self.rank % state['world_size']]
        if resize:
            local = copy.deepcopy(local)
            for key in ('request','paired'):
                local[key] = resize_stream_position([r[key] for r in state['rank_states']], new_world=self.world, rank=self.rank)
        self.request_stream = OrdinalStream(self.request_stream.ordinals, seed=self.request_stream.seed, rank=self.rank, world=self.world, state=local['request'])
        self.pair_stream = OrdinalStream(self.pair_stream.ordinals, seed=self.pair_stream.seed, rank=self.rank, world=self.world, state=local['paired'])
        random.setstate(local['random']); np.random.set_state(local['numpy'])
        torch.set_rng_state(local['cpu_rng']); torch.cuda.set_rng_state(local['cuda_rng'], self.device)
        self.step, self.costs = state['step'], local['costs']
        if resize:
            # Rank count changes stochastic execution. Record a reproducible
            # new-rank seed, not a claim of bit-identical old-topology replay.
            seed = self.q['seed'] + self.step * 1009 + self.rank
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            self.costs = {k:sum(r['costs'].get(k,0) for r in state['rank_states'])/self.world for k in local['costs']}
            write(self.out/f'RESIZE_rank{self.rank}.json', dict(step=self.step, from_world=state['world_size'], to_world=self.world,
                request=self.request_stream.state_dict(), paired=self.pair_stream.state_dict(), new_rng_seed=seed,
                optimizer_preserved=True, effective_batch_preserved=True, bit_exact_old_topology_replay=False))
        return state['elapsed_seconds']

    @torch.no_grad()
    def evaluate(self):
        """Temporarily load observers after returning idle CUDA memory.

        CTranslate2 allocates outside PyTorch's caching allocator. A large
        training cache can exhaust its allocation even though PyTorch's live
        tensors fit comfortably. Gradients are no longer needed after Adam.
        Observers must also leave the GPU before the next large training batch.
        """
        def memory():
            free, total = torch.cuda.mem_get_info(self.device)
            return dict(allocated_MiB=torch.cuda.memory_allocated(self.device)/1024**2,
                        reserved_MiB=torch.cuda.memory_reserved(self.device)/1024**2,
                        free_MiB=free/1024**2, total_MiB=total/1024**2)
        self.progress('EVALUATION_RELEASE_TRAINING_CACHE')
        before = memory()
        self.optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        released = memory()
        torch.cuda.reset_peak_memory_stats(self.device)
        whisper = ctc = processor = None
        completed = False
        try:
            self.progress('EVALUATION_LOAD_OBSERVERS')
            from faster_whisper import WhisperModel
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            whisper = WhisperModel(self.q['whisper'], device='cuda', device_index=self.rank,
                compute_type='float16', local_files_only=True, cpu_threads=2)
            processor = Wav2Vec2Processor.from_pretrained(self.q['ctc'], local_files_only=True)
            with torch.random.fork_rng(devices=[self.rank]):
                ctc = Wav2Vec2ForCTC.from_pretrained(self.q['ctc'], local_files_only=True).to(self.device).float().eval().requires_grad_(False)
            self.asr = whisper, processor, ctc
            self.progress('EVALUATION_NORMAL_GENERATION')
            self._evaluate_rows(whisper, processor, ctc)
            completed = True
        finally:
            self.progress('EVALUATION_UNLOAD_OBSERVERS')
            if whisper is not None:
                whisper.model.unload_model()
            if ctc is not None:
                ctc.cpu()
            self.asr = None
            whisper = ctc = processor = None
            gc.collect()
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            after = memory()
            write(self.out/f'EVAL_MEMORY_step{self.step:06d}_rank{self.rank}.json',
                dict(step=self.step, completed=completed, before=before, after_training_cache_release=released,
                     after_observer_release=after, peak_pytorch_allocated_MiB=torch.cuda.max_memory_allocated(self.device)/1024**2,
                     observer_scope='Same GPU FP16 Whisper and FP32 CTC, loaded only during evaluation.',
                     parameters_and_optimizer_state_preserved=True))
        self.progress('EVALUATION_COMPLETE')

    def _evaluate_rows(self, whisper, processor, ctc):
        import soundfile as sf
        from scripts.t2a.experiments.ar_structured_v1 import data
        from stable_audio_tools.data.sceneplan_transfusion_editing_clap44 import _content_text
        from stable_audio_tools.training.transfusion_opsd.lexical_content_evidence import lexical_error_rates
        from stable_audio_tools.training.transfusion_opsd.speech_ctc_objective import wav2vec2_w_input
        self.adapter.eval()
        rows = []
        for ordinal in self.q['validation_ordinals'][self.rank::self.world]:
            row, obs = self.validation_inputs.observe(self.adapter, ordinal)
            plan, tokens = self.adapter.native_plan(obs)
            # Target labels are loaded only in this evaluation function,
            # after native planning; never used in request-side training.
            batch = data.collate([self.validation[ordinal]], pad_id=self.adapter.codec.pad_id, joint=True)
            _, _, metadata, _ = self.native._move_joint_batch(batch, self.device)
            expected = metadata[0]['model_sceneplan']
            feature = self.text_feature(f"{len(expected['sources'])} audible sources. " + ' '.join(sorted(_content_text(s)+'.' for s in expected['sources'])))
            facts = request_facts(row['request'], row['operation'])
            words = [s['transcript'] for s in expected['sources'] if s['kind'] == 'speech']
            for seed in self.q['evaluation_seeds']:
                z, wave, metric = self.execute(obs, plan, seed, feature, facts, register_plan=expected)
                audio = wav2vec2_w_input(wave).to(self.device)
                if len(words) == 1:
                    ctext = processor.batch_decode(ctc(audio).logits.argmax(-1))[0]
                    segments, _ = whisper.transcribe(audio[0].cpu().numpy(), language='en', beam_size=5)
                    wtext = ' '.join(s.text.strip() for s in segments)
                    metric.update(Whisper_text=wtext, CTC_text=ctext,
                        Whisper=lexical_error_rates(wtext, words[0]), CTC=lexical_error_rates(ctext, words[0]))
                record = dict(ordinal=ordinal, pair_id=row['pair_id'], operation=row['operation'], seed=seed,
                              request=row['request'], plan=plan, expected_evaluation_only=expected, metrics=metric)
                path = self.out/f'eval_step{self.step:06d}_rank{self.rank}_{ordinal}_{seed}.wav'
                sf.write(path, wave[0].float().cpu().T.numpy(), 44100, subtype='FLOAT')
                record['audio'] = str(path)
                rows.append(record)
        write(self.out/f'eval_step{self.step:06d}_rank{self.rank}.json', dict(step=self.step, rows=rows,
            scope='Reserved validation of this run, native20EulerCFG1. Direct-latentCLAP plus single-placementWhisper/CTC; this profile is distinct from historical posterior/placement-averaged E79 metrics.'))
        dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--limit-updates', type=int)
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--paired-rows-per-rank', type=int)
    parser.add_argument('--paired-microbatch', type=int)
    parser.add_argument('--plan-audit-every', type=int)
    parser.add_argument('--batch-change-after-update', type=int)
    parser.add_argument('--evaluate-at-updates', type=int, nargs='+', default=[],
                        help='Additional cumulative update boundaries for explicit evaluation verification.')
    args = parser.parse_args()
    q = read(args.config); q['config_path'] = str(args.config.resolve())
    if (Path(q['output']).parent/'USER_HOLD.json').exists():
        raise RuntimeError('User requested code preparation only. Wait for an explicit go-ahead before training or GPU startup.')
    if args.resize and not args.resume:
        raise ValueError('--resize requires an explicit saved checkpoint via --resume.')
    if args.limit_updates is not None and not 0 < args.limit_updates <= q['maximum_updates']:
        raise ValueError('--limit-updates must be positive and within the configured schedule.')
    if any(not 0 < step <= q['maximum_updates'] for step in args.evaluate_at_updates):
        raise ValueError('Additional evaluation updates must be within the training schedule.')
    rank, world = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    resume_state = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    execution = resume_state.get('execution', {}) if resume_state else {}
    for key, override in [('paired_rows_per_rank', args.paired_rows_per_rank),
                          ('paired_microbatch', args.paired_microbatch),
                          ('native_plan_audit_every', args.plan_audit_every)]:
        previous = execution.get(key, q.get(key, 1))
        if args.resize and key == 'paired_rows_per_rank' and execution:
            previous = execution['global_paired_batch'] // world
        value = previous if override is None else override
        if value < 1:
            raise ValueError(f'{key} must be positive.')
        q[key] = value
    q['global_paired_batch'] = q['paired_rows_per_rank'] * world
    if args.batch_change_after_update is not None:
        if resume_state is None or not execution or args.batch_change_after_update < 0:
            raise ValueError('A shared batch-change boundary requires a saved execution profile.')
        q['execution_schedule'] = dict(after_update=args.batch_change_after_update, previous=execution,
            steady={key:q[key] for key in ('paired_rows_per_rank','paired_microbatch','global_paired_batch','native_plan_audit_every')})
    elif resume_state and resume_state.get('execution_schedule'):
        q['execution_schedule'] = resume_state['execution_schedule']
    if os.environ['CUDA_VISIBLE_DEVICES'] != ','.join(map(str,q['physical_gpus'])) or world not in (2,4,8) or world != len(q['physical_gpus']):
        raise RuntimeError('Use the authorized GPU arm of the Editing comparison.')
    assert q['request_rows_per_rank'] * world == q['global_request_batch']
    assert q['paired_rows_per_rank'] * world == q['global_paired_batch']
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    torch.manual_seed(42); random.seed(q['seed'] + rank); np.random.seed(q['seed'] + rank)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    out = Path(q['output']); out.mkdir(parents=True, exist_ok=True)
    started = time.time(); elapsed = 0.
    def status(phase, **extra):
        value = dict(phase=phase, rank=rank, pid=os.getpid(), time=time.time(), elapsed_seconds=elapsed+time.time()-started, **extra)
        write(out/f'STATUS_rank{rank}.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)
    status('LOADING')
    learner = Learner(q, rank, world, out)
    if args.resume:
        elapsed = learner.resume(args.resume, resize=args.resize, state=resume_state)
        del resume_state
    elif (out/'resume_latest.pt').exists():
        raise RuntimeError('Existing run requires explicit --resume.')
    status('READY', step=learner.step, partition=learner.partition, full_native_parameter_tensors=len(learner.trainable),
           extra_evaluation_updates=args.evaluate_at_updates,
           execution={k:q[k] for k in ('request_rows_per_rank','global_request_batch',
               'paired_rows_per_rank','paired_microbatch','global_paired_batch','native_plan_audit_every')})
    if learner.step == 0:
        learner.save(elapsed+time.time()-started)
        learner.evaluate()
    last_eval = time.time()
    # Persist the first resumed update with the current schedule identity.
    last_save = 0. if args.resume else time.time()
    last_eval_step = learner.step if learner.step == 0 else -1
    cap = args.limit_updates if args.limit_updates is not None else q['maximum_updates']
    try:
        while learner.step < cap:
            stop = torch.tensor(int((out/'STOP').exists()), device=learner.device)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if int(stop):
                break
            status('COLLECTING_CURRENT_NATIVE_FEEDBACK', step=learner.step)
            # Dump a stalled worker's Python stack before NCCL times out on
            # the peers. This records diagnostics without changing training.
            faulthandler.dump_traceback_later(180, repeat=True)
            try:
                update = learner.train_step()
            finally:
                faulthandler.cancel_dump_traceback_later()
            with (out/f'UPDATES_rank{rank}.jsonl').open('a') as handle:
                handle.write(json.dumps(update, ensure_ascii=False, allow_nan=False)+'\n')
            status('JOINT_UPDATE_COMPLETE', step=learner.step, costs=learner.costs, execution_teacher=update['enabled'], performance=update['performance'])
            now = time.time()
            events = torch.tensor(update_events(learner.step, q, since_save=now-last_save,
                since_evaluation=now-last_eval, extra_evaluations=args.evaluate_at_updates), dtype=torch.int, device=learner.device)
            dist.all_reduce(events, op=dist.ReduceOp.MAX)
            if int(events[0]):
                status('SAVING', step=learner.step)
                learner.save(elapsed+time.time()-started)
                last_save=time.time()
            if int(events[1]):
                status('EVALUATING', step=learner.step)
                learner.evaluate(); last_eval = time.time(); last_eval_step = learner.step
        learner.save(elapsed+time.time()-started)
        if last_eval_step != learner.step:
            learner.evaluate()
        phase = 'TRAINING_COMPLETE' if learner.step >= q['maximum_updates'] else ('STARTUP_CHECK_COMPLETE' if args.limit_updates else 'PAUSED_AT_UPDATE_BOUNDARY')
        status(phase, step=learner.step, maximum_updates=q['maximum_updates'], costs=learner.costs)
    except BaseException as exc:
        # A worker can fail before the other ranks reach a collective.
        # NCCL destruction may then hang, hiding the original Python error
        # until a peer's watchdog expires. Report it and let torchrun stop
        # this process group; atomic recovery checkpoints remain intact.
        traceback.print_exc()
        status('FAILED', step=learner.step, error=repr(exc))
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    finally:
        try:
            learner.finish_numerics()
        finally:
            learner.restore_flash()
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
