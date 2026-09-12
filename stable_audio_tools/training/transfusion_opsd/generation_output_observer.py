"""Native FOA-latent semantic and independent lexical output evaluation.

V40 observer logic with exact protocol-bound checkpoint identity (no fixed
update count). Historical reports keep their original observer and checkpoint.
"""
import os
import gc
from pathlib import Path


class NativeGenerationObserver:
    def __init__(self, protocol, *, count, labels, cache_path=None, cache_destination=None):
        import torch
        from faster_whisper import WhisperModel
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        from scripts.t2a.experiments.clap_factual50k_v1.evaluation import load_factual_encoder, native_configuration_view
        from stable_audio_tools.models.sceneplan_transfusion_editing_pipeline import _load_frozen_foa_vae
        from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_text import FrozenCLAP44TextFeatures
        from stable_audio_tools.training.transfusion_opsd.event_experiment import state_fingerprint
        from stable_audio_tools.training.transfusion_opsd.native_latent_clap import FrozenNativeLatentCLAP
        from stable_audio_tools.training.transfusion_opsd.native_clap_level_content import LevelCanonicalNativeCLAPContentObserver
        from stable_audio_tools.training.transfusion_opsd.native_clap_text_cache import save_native_text_cache, load_native_semantic_cache
        from stable_audio_tools.training.transfusion_opsd.request_coarse_spatial_reward import CoarseSpatialConfig
        from stable_audio_tools.data.sceneplan_generation_ar_natural_constraints import COMPASS_HALF_WIDTH
        import json
        self.device = torch.device('cuda:0'); self.count = count
        clap, identity = load_factual_encoder(protocol['clap_checkpoint']['path'],
            expected_sha256=protocol['clap_checkpoint']['sha256'], device=self.device)
        for key in ('path', 'sha256', 'step', 'new_updates', 'initialization_step'):
            if identity[key] != protocol['clap_checkpoint'][key]:
                raise ValueError('The loaded native CLAP identity differs from the declared protocol: ' + key)
        view = native_configuration_view(identity)
        self.observer = FrozenNativeLatentCLAP(clap)
        self.vae, self.vae_identity = _load_frozen_foa_vae(self.device)
        for key in ('config', 'checkpoint'):
            assert view['frontend_files'][self.vae_identity[key]] == self.vae_identity[key + '_sha256']
        if cache_path is None:
            assert cache_destination is not None
            text = FrozenCLAP44TextFeatures(**view['config']['text']).to(self.device).eval().requires_grad_(False)
            frozen = state_fingerprint(text)
            with torch.no_grad():
                raw = text(labels, self.device)
                semantic = self.observer.text_features(raw, raw)['semantic']
            count('qwen_texts', len(labels))
            assert state_fingerprint(text) == frozen
            cache_path = save_native_text_cache(cache_destination, labels=labels, raw_features=raw,
                semantic_features=semantic, checkpoint=protocol['clap_checkpoint'],
                text_encoder_provenance=dict(config=view['config']['text'], frozen_state_fingerprint=frozen,
                    training_contract_sha256=identity['contract_sha256']))
            del text, raw, semantic
            gc.collect(); torch.cuda.empty_cache()
        self.cache_path = Path(cache_path)
        self.whisper = WhisperModel(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/models/faster-distil-whisper-large-v3",
            device='cuda', device_index=0, compute_type='float16', local_files_only=True, cpu_threads=4, num_workers=1)
        self.processor = Wav2Vec2Processor.from_pretrained(protocol['ctc_model_directory'], local_files_only=True)
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(42)
            self.ctc = Wav2Vec2ForCTC.from_pretrained(protocol['ctc_model_directory'], local_files_only=True).float().eval().requires_grad_(False).to(self.device)
        self.frozen = dict(clap=state_fingerprint(self.observer), vae=state_fingerprint(self.vae), ctc=state_fingerprint(self.ctc))
        assert self.frozen['ctc'] == json.loads(Path(protocol['ctc_baseline_evidence']['path']).read_text())['model_fingerprint']
        class BoundTextOnly(torch.nn.Module):
            def forward(self, *args, **kwargs):
                raise RuntimeError('Only checkpoint-bound frozen text features are allowed.')
        self.scorer = LevelCanonicalNativeCLAPContentObserver(observer=self.observer, text_encoder=BoundTextOnly().eval(),
            vae=self.vae, whisper=self.whisper, checkpoint_identity={k:identity[k] for k in
            ('path', 'sha256', 'step', 'new_updates', 'initialization_step')}, posterior_seeds=[42,43])
        self.scorer.text_cache = load_native_semantic_cache(self.cache_path, checkpoint=protocol['clap_checkpoint'],
            expected_labels=labels, device=self.device)
        self.observer.register_forward_hook(lambda *args: count('clap_forwards'))
        self.vae.encoder.register_forward_hook(lambda *args: count('vae_encodes'))
        self.config = CoarseSpatialConfig(**json.loads(Path(protocol['evaluation_protocol']['path']).read_text())['config'])
        assert COMPASS_HALF_WIDTH == protocol['frozen_angles']['nominal_half_width_deg']
        assert COMPASS_HALF_WIDTH + self.config.extra_angle_tolerance_deg == protocol['frozen_angles']['effective_half_width_deg']
        self.receipt = dict(content=self.scorer.receipt, vae=self.vae_identity, frozen_models=self.frozen)

    def measure(self, waveform, row):
        import torch
        from scripts.t2a.eval.score_sceneplan_dit_p10_speech import _error_rates
        from stable_audio_tools.training.transfusion_opsd.event_content_views import event_relative_content_view
        from stable_audio_tools.training.transfusion_opsd.speech_ctc_objective import wav2vec2_w_input
        from stable_audio_tools.training.transfusion_opsd.event_rewards import EventRequestReward
        from stable_audio_tools.training.transfusion_opsd.request_semantic_time_reward import RequestSemanticTimeCoarseReward
        content = self.scorer.measure(waveform, row['request'], row['requirements'])
        self.count('asr_calls', len(content['asr']['decodes']))
        expected = next((c['value'] for source in row['requirements']['sources']
                         for c in source['constraints'] if c['op']=='transcript'), None)
        ctc_views = []
        if expected is not None:
            canonical, _ = event_relative_content_view(waveform)
            for name, wave in [('whole_w', waveform), ('canonical_event_w', canonical)]:
                with torch.no_grad():
                    logits = self.ctc(wav2vec2_w_input(wave).to(self.device)).logits
                self.count('ctc_forwards')
                text = self.processor.batch_decode(logits.argmax(-1))[0]
                ctc_views.append(dict(view=name, text=text, error=_error_rates(text, expected)))
        coarse = RequestSemanticTimeCoarseReward(EventRequestReward(row['request'], row['requirements']), config=self.config).measure(waveform)
        from .lexical_content_evidence import lexical_observation
        return lexical_observation(dict(content=content, independent_ctc=ctc_views,
            coarse=coarse, costs=coarse['costs']), row['requirements'])

    def unchanged(self):
        from stable_audio_tools.training.transfusion_opsd.event_experiment import state_fingerprint
        return {name: state_fingerprint(model)==self.frozen[name]
                for name, model in [('clap', self.observer), ('vae', self.vae), ('ctc', self.ctc)]}


import statistics
from .lexical_content_evidence import compare_lexical_native_content


def protection(after, before, guards):
    from stable_audio_tools.training.transfusion_opsd.native_execution_identity import identity_aware_execution_protection
    if after.get('failure') or before.get('failure'):
        return dict(passed=False, failures=['missing_or_failed_output'], uncertain=[])
    content = compare_lexical_native_content(after['content'], before['content'],
        maximum_semantic_drop=guards['native_semantic_drop'])
    failures = list(content['failures'])
    for key in ('direction_unobservable_fraction', 'clipping', 'source_presence_failure',
                'requested_time_failure', 'motion_trend_failure'):
        if after['costs'][key] > before['costs'][key] + guards[key+'_increase'] + 1e-9:
            failures.append(key)
    if after['costs']['requested_sector_failure'] > before['costs']['requested_sector_failure'] + 1e-9:
        failures.append('coarse_requested_sector_failure')
    result = dict(passed=not failures and not content['uncertain'], failures=failures,
        uncertain=content['uncertain'], content=content, after_plan_admissible=after['plan_admissible'],
        before_plan_admissible=before['plan_admissible'])
    return identity_aware_execution_protection(after, before, result)


def metrics(record):
    if record.get('failure'):
        return dict(failure=record['failure'])
    content = record['content']; speech = content['asr']['required']
    return dict(**record['costs'], native_semantic=content['native_semantic']['mean'],
        excess_angle_deg=record['coarse']['mean_excess_angle_deg'],
        whisper_WER=statistics.mean(x['wer'] for x in content['asr']['observed_error_bounds']['errors']) if speech else None,
        ctc_WER=statistics.mean(x['error']['wer'] for x in record['independent_ctc']) if speech else None,
        all_five_exact=(content['asr']['observed_error_bounds']['all_requested_words_supported'] and
            all(x['error']['wer'] == 0 for x in record['independent_ctc'])) if speech else None)


def reward(record):
    if record.get('failure') or not record['plan_admissible']:
        return -4.
    value = metrics(record)
    word_cost = (min(2., value['whisper_WER']) + min(2., value['ctc_WER'])) / 2 if value['whisper_WER'] is not None else 0.
    coarse = sum(value[key] for key in ('requested_sector_failure', 'requested_time_failure',
        'motion_trend_failure', 'source_presence_failure', 'direction_unobservable_fraction'))
    return value['native_semantic'] - word_cost - .25*coarse


def summary(records):
    good = [metrics(row) for row in records if not row.get('failure')]
    keys = list(good[0]) if good else []
    return dict(total=len(records), completed=len(good), failed=len(records)-len(good),
        admissible=sum(row.get('plan_admissible', False) for row in records),
        means={key:statistics.mean(x[key] for x in good if x[key] is not None)
            if any(x[key] is not None for x in good) else None for key in keys},
        all_five_exact=sum(x.get('all_five_exact') is True for x in good),
        mean_training_scalar=statistics.mean(reward(row) for row in records) if records else None)


def protection_v3(after, before, guards):
    original = protection(after, before, guards)
    failures, uncertain = list(original['failures']), list(original['uncertain'])
    differences = {}
    if after.get('failure') or before.get('failure'):
        uncertain.append('missing_audio_for_independent_ctc')
    elif after['content']['asr']['required'] or before['content']['asr']['required']:
        for label in ('whole_w', 'canonical_event_w'):
            a = [v for v in after.get('independent_ctc', []) if v['view'] == label]
            b = [v for v in before.get('independent_ctc', []) if v['view'] == label]
            if len(a) != 1 or len(b) != 1:
                uncertain.append('missing_or_duplicate_ctc_' + label)
                continue
            differences[label] = a[0]['error']['wer'] - b[0]['error']['wer']
            if differences[label] > 1e-12:
                failures.append('independent_ctc_' + label)
    return dict(contract='lexical_relative_content_with_independent_ctc_v3',
        original_v2=original, passed=not failures and not uncertain,
        failures=failures, uncertain=uncertain, independent_ctc_wer_changes=differences,
        scope='These fixed observer views only; not a guarantee of acoustic transcript correctness.')
