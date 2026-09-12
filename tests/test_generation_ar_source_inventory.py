import importlib.util
from pathlib import Path
import sys

import torch
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


copy = load('inventory_test_copy', 'stable_audio_tools/models/sceneplan_generation_ar_copy_pointer.py')
inventory = load('inventory_test_model', 'stable_audio_tools/models/sceneplan_generation_ar_source_inventory.py')


@pytest.mark.parametrize('include_event_span', [False, True])
def test_inventory_is_invariant_to_left_and_right_encoder_padding(include_event_span):
    torch.manual_seed(3)
    model = inventory.SourceInventoryHead(hidden_dim=16, width=16, heads=4, layers=2, include_event_span=include_event_span).eval()
    request = 'A café bell, then speech.'
    offsets = [(i, i + 1) for i in range(len(request))]
    context = torch.randn(1, len(request), 16)
    mask = torch.ones(1, len(request)).bool()
    alignment = copy.character_alignment([request], [offsets], mask.tolist())
    with torch.inference_mode():
        reference = model(context, mask, alignment)
        for left, right in ((0, 7), (5, 0), (3, 4)):
            padded = torch.randn(1, len(request) + left + right, 16) * 100
            padded[:, left:left + len(request)] = context
            active = torch.zeros(1, padded.shape[1]).bool()
            active[:, left:left + len(request)] = True
            padded_offsets = [(0, 0)] * left + offsets + [(0, 0)] * right
            mapped = copy.character_alignment([request], [padded_offsets], active.tolist())
            result = model(padded, active, mapped)
            for key in reference:
                torch.testing.assert_close(result[key], reference[key], atol=3e-6, rtol=3e-5)


def test_learned_count_controls_inventory_without_reference_count():
    request = 'bell and dog'
    width = len(request)
    start = torch.full((1, 4, 2, width), -100.)
    end = torch.full_like(start, -100.)
    for slot, (a, b) in enumerate(((0, 3), (9, 11), (0, 3), (9, 11))):
        start[0, slot, :, a] = 0
        end[0, slot, :, b] = 0
    logits = {'count': torch.tensor([[-9., 9., -9., -9.]]),
        'kind': torch.tensor([[[5., 0., 0., 9.]] * 4]), 'start': start, 'end': end}
    mask = torch.tensor([[not c.isspace() for c in request]])
    decoded = inventory.decode_inventory(logits, [request], mask, copy.best_ordered_span)
    assert decoded[0]['count'] == 2
    assert [s['identity']['text'] for s in decoded[0]['sources']] == ['bell', 'dog']
    assert all(s['kind'] == 'sound' for s in decoded[0]['sources'])
    logits['count'] = torch.tensor([[9., -9., -9., -9.]])
    assert len(inventory.decode_inventory(logits, [request], mask, copy.best_ordered_span)[0]['sources']) == 1


def test_attention_targets_remain_outside_forward_and_have_finite_gradients():
    torch.manual_seed(5)
    model = inventory.SourceInventoryHead(hidden_dim=16, width=16, heads=4, layers=2)
    request = 'dog barking'
    mask = torch.ones(1, len(request)).bool()
    alignment = copy.character_alignment([request], [[(i, i + 1) for i in range(len(request))]], mask.tolist())
    context = torch.randn(1, len(request), 16)
    normal = model(context, mask, alignment)
    logits, attention = model(context, mask, alignment, return_attention=True)
    for key in logits:
        torch.testing.assert_close(logits[key], normal[key], atol=0, rtol=0)
    loss = logits['count'].log_softmax(-1)[0, 0].neg() - attention[:, :, :3].sum(-1).log().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
