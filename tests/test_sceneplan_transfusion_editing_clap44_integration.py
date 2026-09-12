import copy
from dataclasses import asdict
import importlib.util
import inspect
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from stable_audio_tools.models.sceneplan_transfusion_editing_ar import EDITING_AR_CLAP44_CONTRACT, EditingScenePlanAdapter, ScenePlanTransfusionEditingAR
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44 import CLAP44_CONTRACT, CLAP44Config, EditingCLAP44, EditingCLAP44SourceBridge
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_io import file_sha256, load_clap44_checkpoint
from stable_audio_tools.training.sceneplan_transfusion_editing_clap44_metrics import margin_summary, retrieval_metrics


def config():
    return CLAP44Config(width=32, heads=4, layers=1, semantic_dim=16, scene_dim=8, text_dim=12, dropout=0)


def test_retrieval_duplicates_chunking_and_collapse():
    features = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    keys = ["dog", "dog", "bell", "speech"]
    a = retrieval_metrics(features, features, keys, keys, chunk_size=1)
    b = retrieval_metrics(features, features, keys, keys, chunk_size=3)
    assert a == b and a["r_at_1"] == 1 and a["unique_candidate_keys"] == 3
    collapsed = retrieval_metrics(torch.ones(4, 3), torch.ones(4, 3), keys, keys)
    assert collapsed["r_at_1"] == 0 and collapsed["mean_rank"] == 3.5
    with pytest.raises(ValueError, match="positive"):
        retrieval_metrics(features, features, ["unseen"]*4, keys)
    assert margin_summary([])["accuracy"] is None
    assert margin_summary([1., 0., -1.])["accuracy"] == pytest.approx(1/3)


def test_checkpoint_loader_is_strict_and_does_not_promote_quality(tmp_path):
    model = EditingCLAP44(config())
    code = tmp_path / "code.py"; code.write_text("original")
    contract = {"schema": CLAP44_CONTRACT, "config": {"model": asdict(config())}, "m2d_used": False, "test_split_used_for_training_or_selection": False, "source_sha256": {str(code): file_sha256(code)}}
    (tmp_path / "TRAIN_CONTRACT.json").write_text(json.dumps(contract))
    checkpoint = tmp_path / "step-000001.pt"
    torch.save({"contract": contract, "model": model.state_dict(), "step": 1, "quality_gate_passed": True}, checkpoint)
    restored, evidence = load_clap44_checkpoint(checkpoint, expected_sha256=file_sha256(checkpoint))
    assert not restored.training and not any(x.requires_grad for x in restored.parameters())
    assert evidence["quality_gate_passed"] is False
    with pytest.raises(RuntimeError, match="SHA256"):
        load_clap44_checkpoint(checkpoint, expected_sha256="wrong")
    code.write_text("changed")
    with pytest.raises(RuntimeError, match="source changed"):
        load_clap44_checkpoint(checkpoint)


class PrefixHarness(nn.Module):
    def forward(self, hidden, *, context, padding_mask, **kwargs):
        # This harness exercises the actual AR adapters/forward, not 15 large
        # pretrained blocks. Attention causality has separate existing tests.
        prefix = hidden[:, :12].mean(1, keepdim=True)
        return hidden + prefix + context[:, :1]


def ar_harness():
    ar = ScenePlanTransfusionEditingAR.__new__(ScenePlanTransfusionEditingAR)
    nn.Module.__init__(ar)
    ar.editing_dit = nn.Module(); ar.editing_dit.transformer = PrefixHarness()
    ar.instruction_conditioner = nn.Identity()
    ar.activation_checkpointing = False; ar.pad_id = 0; ar.vocab_size = 16
    ar.source_audio_adapter = nn.Linear(64, 1024, bias=False)
    ar.source_audio_type_embedding = nn.Parameter(torch.zeros(1024))
    ar.plan_type_embedding = nn.Parameter(torch.zeros(1024))
    ar.plan_adapter = EditingScenePlanAdapter(vocab_size=16, hidden_dim=1024, pad_id=0)
    ar.source_clap_model = EditingCLAP44(config()).eval().requires_grad_(False)
    ar.source_semantic_bridge = EditingCLAP44SourceBridge(1024, config())
    return ar


def test_ar_clap_features_receive_audio_only_and_stay_frozen():
    torch.manual_seed(2); ar = ar_harness().train()
    assert not ar.source_clap_model.training and ar.ar_contract == EDITING_AR_CLAP44_CONTRACT
    x = torch.randn(2, 64, 12); mask = torch.ones(2, 12, dtype=torch.bool)
    ids = torch.tensor([[1, 3], [1, 4]]); pm = torch.ones_like(ids, dtype=torch.bool)
    context = torch.randn(2, 1, 1024); cm = torch.ones(2, 1, dtype=torch.bool)
    baseline, query = ar(x, mask, ids, pm, context, cm, return_source_contrastive_query=True)
    with torch.no_grad(): ar.source_semantic_bridge.global_projection.weight.normal_(std=.03)
    changed, query_a = ar(x, mask, ids, pm, context, cm, return_source_contrastive_query=True)
    _, query_b = ar(x, mask, ids.flip(1), pm, context + 8, cm, return_source_contrastive_query=True)
    torch.testing.assert_close(query_a, query_b)
    assert not torch.allclose(baseline, changed)
    zeroed = ar(x, mask, ids, pm, context, cm, source_clap_keep_mask=torch.zeros(2))
    torch.testing.assert_close(zeroed, baseline, rtol=0, atol=0)
    changed.square().mean().backward()
    assert ar.source_semantic_bridge.global_projection.weight.grad.abs().sum() > 0
    assert all(x.grad is None for x in ar.source_clap_model.parameters())
    with pytest.raises(ValueError, match="M2D"):
        ar(x, mask, ids, pm, context, cm, source_m2d_audio_embedding=torch.randn(2, 768))
    parameters = inspect.signature(EditingCLAP44.source_features).parameters
    assert set(parameters) == {"self", "latent", "mask"}


def test_ar_generation_encodes_clap_once_for_all_generated_tokens():
    class CountEncoder(EditingCLAP44):
        calls = 0
        def source_features(self, latent, mask):
            self.calls += 1
            return super().source_features(latent, mask)
    class StubAR(ScenePlanTransfusionEditingAR):
        def __init__(self):
            nn.Module.__init__(self); self.pad_id = 0; self.vocab_size = 16
            self.source_clap_model = CountEncoder(config()).eval().requires_grad_(False)
            self.feature_objects = []
        def encode_edit_instructions(self, instructions, *, device):
            return torch.zeros(1, 1, 1024), torch.ones(1, 1, dtype=torch.bool)
        def forward(self, x, mask, ids, pm, context, cm, *, source_clap_features):
            self.feature_objects.append(source_clap_features)
            return torch.zeros(1, ids.shape[1], 16)
    class Codec:
        bos_id = 1; eos_id = 2
        token_to_id = {"<source_begin>": 8, **{f"<source_slot_{i}>": 12+i for i in range(4)}}
        def allowed_next_ids(self, prefix, *, fixed_duration_sec=None): return {[1, 5, 7, 2][len(prefix)]}
    ar = StubAR()
    generated = ar.generate_batch(torch.randn(1, 64, 12), torch.ones(1, 12, dtype=torch.bool), ["move the dog"], codec=Codec(), max_plan_tokens=8)
    assert generated[0].tolist() == [1, 5, 7, 2]
    assert ar.source_clap_model.calls == 1 and len(ar.feature_objects) == 3
    assert all(x is ar.feature_objects[0] for x in ar.feature_objects)


def test_dropout_applies_after_projection_bias():
    encoder = EditingCLAP44(config()).eval()
    bridge = EditingCLAP44SourceBridge(20, config(), audio_feature_dropout=1).train()
    for p in (bridge.global_projection.bias, bridge.sequence_projection.bias):
        with torch.no_grad(): p.fill_(3)
    hidden = torch.randn(2, 12, 20)
    features = encoder.source_features(torch.randn(2, 64, 12), torch.ones(2, 12, dtype=torch.bool))
    assert torch.equal(bridge.inject(hidden, features), hidden)
    bridge.eval()
    assert not torch.equal(bridge.inject(hidden, features), hidden)


def test_training_dataset_refuses_test_split_metadata(tmp_path):
    import sqlite3
    from stable_audio_tools.data.sceneplan_transfusion_editing_clap44 import EditingCLAP44Dataset
    index = tmp_path / "fixture.sqlite"
    marker = {"schema": "sceneplan_transfusion_editing_training_index", "state": "materialized_complete_frozen", "index_sha256": "fixture"}
    with sqlite3.connect(index) as db:
        db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT)")
        db.executemany("INSERT INTO metadata VALUES (?,?)", [*[(k, marker[k]) for k in ("schema", "state")], ("split", "test"), ("rows", "1")])
    index.with_suffix(".sqlite.frozen.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="independent test"):
        EditingCLAP44Dataset(index, expected_rows=1)


def test_training_rng_payload_supports_weights_only_resume(tmp_path, monkeypatch):
    import random
    import numpy as np
    path = Path(__file__).resolve().parents[1] / "scripts/t2a/train/train_sceneplan_transfusion_editing_clap44.py"
    spec = importlib.util.spec_from_file_location("clap44_train_test", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.zeros(8, dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value: None)
    random.seed(5); np.random.seed(5); torch.manual_seed(5)
    saved = module.rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    p = tmp_path / "rng.pt"; torch.save(saved, p)
    module.restore_rng(torch.load(p, weights_only=True))
    observed = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert observed == expected
