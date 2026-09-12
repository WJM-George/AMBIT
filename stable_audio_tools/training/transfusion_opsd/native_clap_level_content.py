"""Ignore unrequested global level only in our native CLAP semantic view."""
import torch

from .native_clap_event_content import NativeCLAPEventContentObserver, compare_native_content_evidence
from .native_latent_clap import requested_semantic_text
from .event_content_views import event_relative_content_view


def canonical_semantic_foa(waveform):
    """Scale all WYZX channels together; preserve directions and relative level.

    This is explicit input normalization, not learned CLAP level invariance.
    Silence stays silence. Actual-output presence, clipping, time, direction,
    and ASR measurements must still receive the original waveform.
    """
    if waveform.ndim != 3 or waveform.shape[:2] != (1,4) or not torch.isfinite(waveform).all():
        raise ValueError('Expected finite full [1,4,samples] FOA')
    value=waveform.float()
    peak=value[:,0].abs().amax()
    gain=.1/peak.clamp_min(1e-8)
    return value*gain,dict(contract='full_foa_W_peak_point1_semantic_view_v1',
        target_W_peak=.1,original_W_peak=float(peak.detach()),applied_gain=float(gain.detach()))


class LevelCanonicalNativeCLAPContentObserver(NativeCLAPEventContentObserver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.receipt=dict(self.receipt,contract='level_canonical_native_foa_clap_content_v1',
            semantic_view='WholeFOA scaled together to W peak.1, then the unchanged native VAE/CLAP. Raw audio retained for other measures.',
            gain_policy='full_foa_W_peak_point1_semantic_view_v1')

    @torch.no_grad()
    def measure(self,waveform,request,requirements):
        if len(requirements['sources'])!=1 or requirements.get('relations'):
            raise ValueError('Declare an observable single foreground source')
        from stable_audio_tools.models.sceneplan_transfusion_editing_pipeline import ScenePlanTransfusionEditingDiTPipeline
        canonical,level=canonical_semantic_foa(waveform)
        text=requested_semantic_text(requirements)
        if text not in self.text_cache:
            hidden=self.text_encoder([text],self.device)
            self.text_cache[text]=self.observer.text_features(hidden,hidden)['semantic']
        observations=[]
        for seed in self.posterior_seeds:
            latent,mask=ScenePlanTransfusionEditingDiTPipeline.encode_source_foa(self,canonical,
                model_num_samples=[waveform.shape[-1]],vae_seeds=[seed])
            features=self.observer(latent,mask)
            observations.append(dict(posterior_seed=seed,score=float((features['semantic']*self.text_cache[text]).sum(-1)[0])))
        asr_wave,content_view=event_relative_content_view(waveform,config=self.config)
        return dict(contract=self.receipt['contract'],checkpoint=self.receipt['checkpoint'],semantic_level_view=level,
            native_semantic=dict(text=text,observations=observations,mean=sum(x['score'] for x in observations)/len(observations)),
            content_view=content_view,presence=self.presence_evidence(waveform,request,requirements),
            asr=self.asr_evidence(asr_wave,requirements))


def compare_level_canonical_native_content(after,before,*,maximum_semantic_drop):
    if after['contract']!='level_canonical_native_foa_clap_content_v1' or before['contract']!=after['contract']:
        raise ValueError('Compare the same level-canonical native observation')
    for value in (after,before):
        view=value['semantic_level_view']
        if view['contract']!='full_foa_W_peak_point1_semantic_view_v1' or view['target_W_peak']!=.1:
            raise ValueError('The declared semantic level view changed')
    # The underlying paired-semantic/ASR comparison is shared. These are local
    # dictionary views, never rewritten provenance or relabeled saved records.
    result=compare_native_content_evidence(
        {**after,'contract':'native_foa_clap_content_observer_v1'},
        {**before,'contract':'native_foa_clap_content_observer_v1'},maximum_semantic_drop=maximum_semantic_drop)
    return dict(result,contract='level_canonical_native_content_comparison_v1')
