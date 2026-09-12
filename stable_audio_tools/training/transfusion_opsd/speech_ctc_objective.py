"""Differentiable frozen-CTC input/loss helpers; not an audio quality gate."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio


def requested_ctc_token_ids(tokenizer, text):
    """Encode requested words while preserving the CTC word delimiter.

    Wav2Vec2 may list its delimiter among all_special_ids. It is an actual
    acoustic label here; only the other special IDs are forbidden. Explicitly
    disable CTC collapse when validating repeated letters/words on round trip.
    """
    if not isinstance(text, str) or not text.strip() or text != ' '.join(text.split()):
        raise ValueError('Expected nonempty requested text with normalized whitespace.')
    delimiter = tokenizer.word_delimiter_token_id
    if delimiter is None or delimiter in (tokenizer.pad_token_id, tokenizer.unk_token_id):
        raise ValueError('A distinct CTC word delimiter is required.')
    ids = tokenizer(text, add_special_tokens=False).input_ids
    forbidden = set(tokenizer.all_special_ids) - {delimiter}
    if not ids or set(ids) & forbidden:
        raise ValueError('Requested CTC text contains unsupported or non-acoustic special tokens.')
    decoded = tokenizer.decode(ids, group_tokens=False)
    if ' '.join(decoded.upper().split()) != text.upper():
        raise ValueError('Requested text did not survive the tokenizer round trip.')
    return ids


def wav2vec2_w_input(waveform, *, source_sample_rate=44100):
    """Whole W channel, matching the declared Wav2Vec2 observer preprocessing.

    The model/request must not determine a crop. Keep autograd through the
    waveform, peak normalization, resampling and population normalization;
    passing numpy through a Processor here would silently lose that path.
    This contract is for the pinned 16 kHz, do_normalize=True CTC observer.
    """
    if (waveform.ndim != 3 or waveform.shape[:2] != (1, 4) or waveform.shape[-1] < 2
            or not waveform.is_floating_point() or not torch.isfinite(waveform).all()):
        raise ValueError('Expected finite floating whole FOA audio [1,4,samples].')
    if type(source_sample_rate) is not int or source_sample_rate <= 0:
        raise ValueError('Declare a positive integer sample rate.')
    # The original observer resamples on CPU. Preserve that numerical path;
    # device copies retain autograd back to the native CUDA decoder.
    destination = waveform.device
    mono = waveform[:, 0].float().to('cpu')
    mono = mono / mono.abs().amax(-1, keepdim=True).clamp_min(1e-8) * (10 ** (-1 / 20))
    samples = torchaudio.functional.resample(mono, source_sample_rate, 16000)
    centered = samples - samples.mean(-1, keepdim=True)
    normalized = centered / torch.sqrt(samples.var(-1, unbiased=False, keepdim=True) + 1e-7)
    return normalized.to(destination)


def requested_ctc_loss(logits, target_ids, *, blank_id):
    """Mean CTC loss for one unpadded request, with finite input gradients.

    Targets contain only explicit requested text tokens; blank, unknown and
    extra prompt tokens must be rejected by the caller's tokenizer contract.
    Infinite losses are errors, never silently changed to zero.
    """
    if (logits.ndim != 3 or logits.shape[0] != 1 or min(logits.shape[1:]) <= 0
            or not logits.is_floating_point() or not torch.isfinite(logits).all()):
        raise ValueError('Expected finite floating logits [1,frames,vocabulary].')
    if type(blank_id) is not int or not 0 <= blank_id < logits.shape[-1]:
        raise ValueError('Invalid CTC blank ID.')
    if (target_ids.ndim != 1 or target_ids.numel() == 0 or target_ids.dtype != torch.long
            or target_ids.device != logits.device or (target_ids < 0).any()
            or (target_ids >= logits.shape[-1]).any() or (target_ids == blank_id).any()):
        raise ValueError('Expected nonempty in-vocabulary integer targets without blanks.')
    minimum_frames = target_ids.numel() + int((target_ids[1:] == target_ids[:-1]).sum())
    if minimum_frames > logits.shape[1]:
        raise ValueError('CTC input is too short for the requested tokens and repeats.')
    # This installed CUDA CTC backward is nondeterministic. The small dynamic
    # program runs on CPU; differentiable copies retain gradients to CUDA
    # acoustic logits while the native model's determinism stays enabled.
    lengths = torch.tensor([logits.shape[1]], dtype=torch.long)
    target_lengths = torch.tensor([target_ids.numel()], dtype=torch.long)
    with torch.backends.cudnn.flags(enabled=False):
        loss = F.ctc_loss(logits.float().log_softmax(-1).transpose(0, 1).cpu(), target_ids.cpu(),
            lengths, target_lengths, blank=blank_id, reduction='mean', zero_infinity=False)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite CTC loss cannot supervise a repair.')
    return loss.to(logits.device)
