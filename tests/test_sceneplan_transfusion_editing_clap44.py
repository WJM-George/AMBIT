import copy
import json
from pathlib import Path

import pytest
import torch

from stable_audio_tools.data.sceneplan_transfusion_editing_clap44 import binding_counterfactuals, make_clap44_label
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44 import CLAP44Config, EditingCLAP44, EditingCLAP44SourceBridge, binding_negative_loss, clap44_objective, contrastive_relations, symmetric_multi_positive_loss


def plan():
    def source(i, description, azimuth, onset):
        return {"source_id": f"source_{i}", "kind": "sound", "description": description, "gain_db": 0.0, "activity": {"onset_sec": onset, "offset_sec": onset + 0.8}, "trajectory": {"type": "static", "position": {"azimuth_deg": azimuth, "elevation_deg": 0.0, "distance_m": 2.0}}}
    return {"sample_id": "fixture", "duration_sec": 4.0, "room": {"type": "moderate"}, "sources": [source(0, "a dog barking", -60.0, 0.5), source(1, "a bell ringing", 60.0, 2.0)]}


def label(p, role="source", pair_id="pair"):
    return make_clap44_label(p, [{"asset_id": "dog"}, {"asset_id": "bell"}], pair_id=pair_id, role=role, operation="stationary_spatial_relocation")


def tiny():
    return CLAP44Config(width=32, layers=1, heads=4, semantic_dim=16, scene_dim=8, text_dim=12, dropout=0.0)


def test_spatial_edit_positive_only_in_content_head():
    a = plan(); b = copy.deepcopy(a)
    b["sources"][0]["trajectory"]["position"]["azimuth_deg"] = 30.0
    labels = [label(a), label(b, "target")]
    semantic, _ = contrastive_relations(labels, "semantic")
    scene, allowed = contrastive_relations(labels, "scene")
    assert semantic.all() and not scene[0, 1] and allowed[0, 1]
    _, warmup_allowed = contrastive_relations(labels, "scene", include_edit_negatives=False)
    assert not warmup_allowed[0, 1]


def test_binding_and_count_are_not_bags_of_attributes():
    a = plan(); b = copy.deepcopy(a)
    b["sources"][0]["trajectory"], b["sources"][1]["trajectory"] = b["sources"][1]["trajectory"], b["sources"][0]["trajectory"]
    assert label(a)["semantic_key"] == label(b)["semantic_key"]
    assert label(a)["scene_key"] != label(b)["scene_key"]
    b = copy.deepcopy(a); b["sources"].pop()
    assert label(a)["semantic_key"] != label(b)["semantic_key"]
    b = copy.deepcopy(a); b["sample_id"] = "different"; b["sources"].reverse()
    for i, source in enumerate(b["sources"]):
        source["source_id"] = f"source_{i}"
    assert label(a)["semantic_key"] == label(b)["semantic_key"]
    assert label(a)["scene_key"] == label(b)["scene_key"]


def test_partial_shared_content_is_not_a_random_negative():
    a = label(plan()); b = copy.deepcopy(a)
    b.update(pair_id="other", semantic_key="different", scene_key="different")
    pos, allowed = contrastive_relations([a, b], "semantic")
    assert not pos[0, 1] and not allowed[0, 1]
    b.update(asset_ids=["unrelated"], content_ids=["unrelated"])
    assert contrastive_relations([a, b], "semantic")[1][0, 1]


def test_counterfactuals_preserve_content_and_reject_ambiguous_swaps():
    p = plan(); before = copy.deepcopy(p)
    negatives = binding_counterfactuals(p)
    assert {x["kind"] for x in negatives} == {"swapped_trajectory", "swapped_activity"}
    assert all("dog barking" in x["scene_text"] and "bell ringing" in x["scene_text"] for x in negatives)
    assert p == before
    p["sources"][1]["description"] = p["sources"][0]["description"]
    assert not binding_counterfactuals(p)


def test_encoder_native_rate_padding_and_audio_dependence():
    torch.manual_seed(7); model = EditingCLAP44(tiny()).eval()
    x = torch.randn(2, 64, 19); mask = torch.ones(2, 19, dtype=torch.bool)
    expected = model.encode_audio(x, mask)
    padded = torch.randn(2, 64, 648) * 100
    padded[:, :, :19] = x; extended = torch.arange(648)[None].expand(2, -1) < 19
    actual = model.encode_audio(padded, extended)
    for head in ("semantic", "scene"):
        torch.testing.assert_close(actual[head], expected[head], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual[head].norm(dim=-1), torch.ones(2))
        assert not torch.allclose(actual[head][0], actual[head][1])
    with pytest.raises(ValueError, match="resample"):
        model.encode_audio(x, mask, sample_rate=48000)
    with pytest.raises(ValueError, match="right-padded"):
        bad = mask.clone(); bad[:, 2] = False; model.encode_audio(x, bad)
    with pytest.raises(ValueError, match="right-padded"):
        model.encode_audio(x, torch.zeros_like(mask))


def test_bridge_starts_exactly_at_latent_route_and_zero_ablation_is_exact():
    cfg = tiny(); model = EditingCLAP44(cfg).eval(); bridge = EditingCLAP44SourceBridge(20, cfg)
    features = model.source_features(torch.randn(2, 64, 20), torch.ones(2, 20, dtype=torch.bool))
    hidden = torch.randn(2, 20, 20)
    assert torch.equal(bridge.inject(hidden, features), hidden)
    with torch.no_grad():
        bridge.global_projection.weight.fill_(0.1); bridge.global_projection.bias.fill_(0.5)
        bridge.sequence_projection.weight.fill_(0.1); bridge.sequence_projection.bias.fill_(0.5)
    assert not torch.equal(bridge.inject(hidden, features), hidden)
    assert torch.equal(bridge.inject(hidden, features, torch.zeros(2)), hidden)
    features = dict(features); features["stride"] = 0
    with pytest.raises(ValueError): bridge.inject(hidden, features)


def test_multi_positive_objective_and_real_gradients():
    torch.manual_seed(42)
    audio = torch.eye(4, requires_grad=True); text = torch.eye(4, requires_grad=True)
    positives = torch.eye(4, dtype=torch.bool); allowed = torch.ones(4, 4, dtype=torch.bool)
    scale = torch.tensor(2.0, requires_grad=True)
    good = symmetric_multi_positive_loss(audio, text, positives, allowed, scale)
    bad = symmetric_multi_positive_loss(audio, text.flip(0), positives, allowed, scale)
    assert good < bad
    good.backward()
    assert audio.grad.abs().sum() > 0 and text.grad.abs().sum() > 0 and scale.grad.abs() > 0
    duplicates = torch.tensor([[True, True], [True, True]])
    value = symmetric_multi_positive_loss(torch.eye(2), torch.eye(2), duplicates, duplicates, torch.tensor(0.0))
    assert torch.isfinite(value)
    with pytest.raises(ValueError):
        symmetric_multi_positive_loss(torch.eye(2), torch.eye(2), duplicates, torch.eye(2, dtype=torch.bool), torch.tensor(0.0))


def test_binding_negatives_only_affect_their_owner():
    audio = torch.eye(3, requires_grad=True); positive = torch.eye(3)
    negative = torch.tensor([[1.0, 0.0, 0.0]])
    loss = binding_negative_loss(audio, positive, negative, torch.tensor([1]))
    loss.backward()
    assert audio.grad[1].abs().sum() > 0
    assert audio.grad[0].abs().sum() == 0 and audio.grad[2].abs().sum() == 0
    assert binding_negative_loss(audio, positive, torch.empty(0, 3), torch.empty(0, dtype=torch.long)) == 0


def test_joint_head_training_reduces_tiny_fixed_batch_loss():
    torch.manual_seed(5); torch.set_num_threads(1)
    model = EditingCLAP44(tiny()); optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    latent = torch.randn(4, 64, 24); mask = torch.ones(4, 24, dtype=torch.bool)
    features = torch.randn(4, 12)
    labels = [{"pair_id": str(i), "role": "source", "semantic_key": str(i), "scene_key": str(i), "asset_ids": [str(i)], "content_ids": [str(i)]} for i in range(4)]
    values = []
    for _ in range(12):
        audio, text = model(latent, mask, features, features)
        loss = clap44_objective(model, audio, text, labels)["loss"]
        optimizer.zero_grad(); loss.backward(); optimizer.step(); values.append(float(loss.detach()))
    assert values[-1] < values[0] * 0.5


def _ddp_worker(rank, init_file, output_dir):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    class Towers(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.projection = torch.nn.Linear(3, 3, bias=False)
            self.logit_scale_semantic = torch.nn.Parameter(torch.tensor(1.0))
            self.logit_scale_scene = torch.nn.Parameter(torch.tensor(1.0))
        def forward(self, a, t):
            a, t = self.projection(a), self.projection(t)
            return {"semantic": a, "scene": a}, {"semantic": t, "scene": t}
    torch.set_num_threads(1); torch.manual_seed(9)
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=2)
    wrapped = DistributedDataParallel(Towers())
    a = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.]])
    t = a + 0.1
    labels = [{"pair_id": str(i), "role": "source", "semantic_key": str(i), "scene_key": str(i), "asset_ids": [str(i)], "content_ids": [str(i)]} for i in range(4)]
    audio, text = wrapped(a[2*rank:2*rank+2], t[2*rank:2*rank+2])
    clap44_objective(wrapped.module, audio, text, labels[2*rank:2*rank+2])["loss"].backward()
    observed = {k: v.grad.clone() for k, v in wrapped.module.named_parameters()}
    dist.destroy_process_group()
    torch.manual_seed(9); reference = Towers(); audio, text = reference(a, t)
    clap44_objective(reference, audio, text, labels)["loss"].backward()
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(observed[name], parameter.grad, atol=2e-6, rtol=2e-6)
    Path(output_dir, f"rank-{rank}.json").write_text(json.dumps({"gradient_matches_global_batch": True}))


def test_ddp_gradient_matches_single_global_batch(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_ddp_worker, args=(str(tmp_path / "gloo-init"), str(tmp_path)), nprocs=2, join=True)
    assert len(list(tmp_path.glob("rank-*.json"))) == 2
