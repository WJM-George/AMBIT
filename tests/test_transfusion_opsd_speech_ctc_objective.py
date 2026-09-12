import numpy as np
import json
import pytest
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer

from stable_audio_tools.training.transfusion_opsd.speech_ctc_objective import (
    requested_ctc_loss, requested_ctc_token_ids, wav2vec2_w_input,
)


@pytest.fixture
def ctc_tokenizer(tmp_path):
    vocabulary = {'<pad>':0, '<s>':1, '</s>':2, '<unk>':3, '|':4}
    vocabulary.update({c:i+5 for i,c in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')})
    path = tmp_path/'vocab.json'
    path.write_text(json.dumps(vocabulary))
    return Wav2Vec2CTCTokenizer(str(path), word_delimiter_token='|')


def test_special_word_delimiter_and_repeated_letters_are_valid_targets(ctc_tokenizer):
    ctc_tokenizer.add_special_tokens({'additional_special_tokens':['|']})
    assert ctc_tokenizer.word_delimiter_token_id in ctc_tokenizer.all_special_ids
    text = 'THE PRIEST PRIEST HESITATED STILL'
    ids = requested_ctc_token_ids(ctc_tokenizer, text)
    assert ids.count(ctc_tokenizer.word_delimiter_token_id) == 4
    assert ctc_tokenizer.decode(ids, group_tokens=False) == text


def test_unknown_requested_characters_are_not_silently_supervised(ctc_tokenizer):
    with pytest.raises(ValueError, match='unsupported'):
        requested_ctc_token_ids(ctc_tokenizer, 'THE PRIEST 123')


def test_ctc_waveform_path_matches_frozen_processor_and_reaches_only_w():
    generator = torch.Generator().manual_seed(42)
    waveform = torch.randn(1, 4, 4410, generator=generator).requires_grad_(True)
    result = wav2vec2_w_input(waveform)
    mono = waveform.detach()[:, 0]
    mono = mono / mono.abs().max() * (10 ** (-1 / 20))
    samples = torchaudio.functional.resample(mono, 44100, 16000)[0].numpy()
    processor = Wav2Vec2FeatureExtractor(sampling_rate=16000, do_normalize=True, return_attention_mask=False)
    expected = processor(samples, sampling_rate=16000, return_tensors='pt').input_values
    torch.testing.assert_close(result, expected, rtol=2e-6, atol=2e-6)
    (result * torch.linspace(-1, 1, result.shape[-1])).sum().backward()
    assert torch.isfinite(waveform.grad).all() and waveform.grad[:, 0].norm() > 0
    assert waveform.grad[:, 1:].count_nonzero() == 0


def test_ctc_silence_stays_finite_and_has_no_spurious_content():
    result = wav2vec2_w_input(torch.zeros(1, 4, 4410))
    assert torch.isfinite(result).all() and result.count_nonzero() == 0


def test_requested_tokens_get_an_actual_finite_logit_gradient():
    logits = torch.zeros(1, 6, 4, requires_grad=True)
    target = torch.tensor([1, 2], dtype=torch.long)
    loss = requested_ctc_loss(logits, target, blank_id=0)
    gradient, = torch.autograd.grad(loss, logits)
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    assert requested_ctc_loss(logits.detach() - .1 * gradient, target, blank_id=0) < loss


@pytest.mark.parametrize('target', [torch.tensor([], dtype=torch.long), torch.tensor([0]), torch.tensor([4]), torch.tensor([1.])])
def test_invalid_target_evidence_is_rejected(target):
    with pytest.raises(ValueError):
        requested_ctc_loss(torch.zeros(1, 3, 4), target, blank_id=0)


def test_repeated_target_needs_blank_separation_time():
    with pytest.raises(ValueError, match='too short'):
        requested_ctc_loss(torch.zeros(1, 2, 4), torch.tensor([1, 1]), blank_id=0)


def test_nonfinite_audio_and_logits_are_rejected():
    with pytest.raises(ValueError):
        wav2vec2_w_input(torch.full((1, 4, 10), np.nan))
    with pytest.raises(ValueError):
        requested_ctc_loss(torch.full((1, 3, 4), np.nan), torch.tensor([1]), blank_id=0)
