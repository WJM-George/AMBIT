from types import SimpleNamespace

import pytest

from stable_audio_tools.training.transfusion_opsd.event_qwen_norm_runtime import (
    _restrict_kernel, pin_qwen_fused_norm,
)


def config(bt, warps):
    return SimpleNamespace(kwargs={'BT': bt}, num_warps=warps, num_stages=3)


def test_pin_discards_a_previously_autotuned_numerically_different_layout():
    good, different = config(16, 4), config(16, 16)
    tuner = SimpleNamespace(configs=[good, different], cache={'shape': different})
    wrapper = SimpleNamespace(fn=tuner)
    declared = {'kwargs': {'BT': 16}, 'num_warps': 4, 'num_stages': 3}
    assert _restrict_kernel(wrapper, declared) is tuner
    assert tuner.configs == [good]
    assert tuner.cache == {}
    tuner.cache['shape'] = good
    _restrict_kernel(wrapper, declared)
    assert tuner.configs == [good] and tuner.cache == {}


def test_missing_declared_layout_does_not_mutate_an_existing_runtime():
    existing = config(32, 4)
    tuner = SimpleNamespace(configs=[existing], cache={'shape': existing})
    with pytest.raises(ValueError, match='lacks'):
        _restrict_kernel(tuner, {'kwargs': {'BT': 16}, 'num_warps': 4, 'num_stages': 3})
    assert tuner.configs == [existing] and tuner.cache == {'shape': existing}


@pytest.mark.parametrize('spec', [None, {}, {'contract': 'auto', 'source_sha256': 'a'*64},
    {'contract': 'event_qwen_fused_norm_bt16_w4_s3_v1', 'source_sha256': 'unverified'}])
def test_invalid_pin_is_rejected_before_loading_gpu_libraries(spec):
    with pytest.raises(ValueError, match='declare'):
        pin_qwen_fused_norm(spec)


def test_restore_cannot_silently_drop_the_numerical_runtime(monkeypatch):
    from dataclasses import asdict
    from stable_audio_tools.training.transfusion_opsd.event_trainer import EventOPSDTrainer, EventFitConfig
    trainer = object.__new__(EventOPSDTrainer)
    trainer.config = EventFitConfig()
    trainer.identity = {'initialization': 'same', 'data': 'same', 'protocol': 'same'}
    trainer.policy = SimpleNamespace(contract='policy', runtime_contract=None,
        bundle=SimpleNamespace(qwen_runtime='torch_reference', dit_runtime='fp32', qwen_fused_norm_runtime=None))
    payload = dict(contract=trainer.checkpoint_contract, policy_contract='policy',
        policy_runtime_contract=None, identity=trainer.identity, config=asdict(trainer.config),
        qwen_runtime='torch_reference', dit_runtime='fp32',
        qwen_fused_norm_runtime={'contract': 'event_qwen_fused_norm_bt16_w4_s3_v1', 'source_sha256': 'a'*64})
    monkeypatch.setattr('torch.load', lambda *args, **kwargs: payload)
    with pytest.raises(ValueError, match='qwen_fused_norm_runtime'):
        trainer.restore_candidate('/unused/candidate.pt')
