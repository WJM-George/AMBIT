from __future__ import annotations

import copy
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import torch

from stable_audio_tools.data.model_sceneplan_codec import (
    ModelScenePlanCodecError,
    load_model_sceneplan_codec,
)
from stable_audio_tools.data.scene_sketch_v1 import (
    DeltaSceneSketchCodec,
    SceneSketchCodec,
    apply_reliable_lexical_authority_to_sketch,
    assemble_sceneplan,
    compile_delta_scene_sketch,
    compile_execution_state,
    compile_scene_sketch,
    execution_state_core,
    project_delta_thought_to_atomic_patch,
)
from stable_audio_tools.data.sceneplan_edit_patch import ScenePlanEditPatchCodec
from stable_audio_tools.data.sceneplan_p11_lexical_cache import (
    P11_LEXICAL_CACHE_SCHEMA,
    P11_LEXICAL_CACHE_VERSION,
    P11_LEXICAL_CONFIDENCE_CONTRACT,
    P11_LEXICAL_EVIDENCE_CONTRACT,
    P11LexicalEvidenceCache,
)
from stable_audio_tools.data.sceneplan_p11_single_turn import (
    AudioAwareEditPlanningBundle,
    P10_SEMANTIC_CAPTION_COMPILER_VERSION,
    P10_SEMANTIC_CAPTION_CONTRACT,
    P11_EDITING_CONTRACT,
    P11Task,
    ScenePlanExecutionBundle,
)
from stable_audio_tools.inference.sceneplan_cot import ScenePlanCoTPipeline
from stable_audio_tools.models.sceneplan_p11_v4 import (
    ScenePlanP11V4Planner,
    _finite_field_token_objective,
)


CODEC_PATH = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)


def _plan() -> dict:
    return {
        "sample_id": "p11_v4_test",
        "duration_sec": 2.0,
        "room": {"type": "dry"},
        "sources": [
            {
                "source_id": "source_0",
                "kind": "sound",
                "description": "a short metal bell",
                "activity": {"onset_sec": 0.25, "offset_sec": 1.5},
                "trajectory": {
                    "type": "linear",
                    "start": {
                        "azimuth_deg": -45.0,
                        "elevation_deg": 0.0,
                        "distance_m": 1.0,
                    },
                    "end": {
                        "azimuth_deg": 45.0,
                        "elevation_deg": 0.0,
                        "distance_m": 2.0,
                    },
                },
                "gain_db": 0.0,
            }
        ],
    }


@pytest.fixture(scope="module")
def codec():
    if not CODEC_PATH.exists():
        pytest.skip("canonical P11 codec is unavailable")
    return load_model_sceneplan_codec(CODEC_PATH)


def test_scene_sketch_execution_roundtrip_and_authority_split(codec) -> None:
    target = codec.project_plan(_plan())
    sketch = compile_scene_sketch(target, codec)
    execution = compile_execution_state(target, codec)
    assert assemble_sceneplan(
        sketch, execution, codec, require_target_hash=True
    ) == target

    sketch_tokens = SceneSketchCodec(codec).encode(sketch)["input_ids"]
    assert torch.equal(
        SceneSketchCodec(codec).canonicalize(sketch_tokens)["input_ids"],
        sketch_tokens,
    )

    numeric = copy.deepcopy(execution)
    numeric["sceneplan_sha256"] = None
    core = np.asarray(numeric["core_features"], dtype=np.float32)
    core[1, 5], core[1, 6] = 1.0, 0.0
    numeric["core_features"] = core.tolist()
    numeric_plan = assemble_sceneplan(sketch, numeric, codec)
    assert numeric_plan["sources"][0]["description"] == "a short metal bell"

    semantic = copy.deepcopy(sketch)
    semantic["sceneplan_sha256"] = None
    semantic["sources"][0]["description"] = "a bright hand bell"
    semantic_plan = assemble_sceneplan(semantic, execution, codec)
    assert np.array_equal(
        execution_state_core(compile_execution_state(semantic_plan, codec)),
        execution_state_core(execution),
    )


def test_scene_sketch_grammar_rejects_whitespace_only_first_piece(codec) -> None:
    sketch_codec = SceneSketchCodec(codec)
    prefix = [
        codec._tid("<plan_bos>"),
        codec._tid("<room>"),
        codec._tid("<room_dry>"),
        codec._tid("<num_sources>"),
        codec._tid("<num_sources_1>"),
        codec._tid("<source_begin>"),
        codec._tid("<source_slot_0>"),
        codec._tid("<kind>"),
        codec._tid("<kind_sound>"),
        codec._tid("<description>"),
        codec._tid("<text_begin>"),
    ]
    allowed = sketch_codec.allowed_next_ids(prefix)
    whitespace_only = {
        codec.text_offset + piece_id
        for piece_id in range(codec.text_vocab_size)
        if not codec.text_processor.decode([piece_id]).strip()
    }
    assert whitespace_only
    assert not (allowed & whitespace_only)
    assert allowed == sketch_codec.nonempty_first_text_ids


def test_reliable_asr_authority_changes_only_discrete_speech_semantics(codec) -> None:
    target = codec.project_plan(_plan())
    sketch = compile_scene_sketch(target, codec)
    execution = compile_execution_state(target, codec)
    forced = apply_reliable_lexical_authority_to_sketch(
        sketch,
        transcript="The frozen recognizer heard this sentence.",
        source_id="source_0",
        codec=codec,
    )
    assert forced["room_intent"] == sketch["room_intent"]
    assert forced["sources"][0] == {
        "source_id": "source_0",
        "kind": "speech",
        "speaker_description": "a short metal bell",
        "transcript": "The frozen recognizer heard this sentence.",
    }
    assert forced["sceneplan_sha256"] is None
    assembled = assemble_sceneplan(forced, execution, codec)
    assert np.array_equal(
        execution_state_core(compile_execution_state(assembled, codec)),
        execution_state_core(execution),
    )

    replaced = apply_reliable_lexical_authority_to_sketch(
        forced,
        transcript="A corrected input-only hypothesis.",
        source_id="source_0",
        codec=codec,
    )
    assert replaced["sources"][0]["speaker_description"] == "a short metal bell"
    assert replaced["sources"][0]["transcript"] == (
        "A corrected input-only hypothesis."
    )
    with pytest.raises(ModelScenePlanCodecError, match="absent source"):
        apply_reliable_lexical_authority_to_sketch(
            sketch,
            transcript="Never applied.",
            source_id="source_3",
            codec=codec,
        )


def test_delta_sketch_plus_delta_thought_produces_atomic_local_patch(codec) -> None:
    patch_codec = ScenePlanEditPatchCodec(codec)
    delta_codec = DeltaSceneSketchCodec(codec, patch_codec)
    current = codec.project_plan(_plan())
    spec = {
        "operation": "rotate_source",
        "source_id": "source_0",
        "delta_azimuth_deg": 45,
    }
    target = patch_codec.apply(current, spec)
    delta = compile_delta_scene_sketch(current, target, spec, codec)
    tokens = delta_codec.encode(delta, spec)["input_ids"]
    program = delta_codec.decode(tokens)
    predicted = (
        execution_state_core(compile_execution_state(target, codec))
        - execution_state_core(compile_execution_state(current, codec))
    )
    projected = project_delta_thought_to_atomic_patch(
        current,
        program,
        predicted,
        codec,
        patch_codec,
    )
    assert projected["patch_spec"] == spec
    assert codec.encode(projected["target_sceneplan"])["input_ids"].equal(
        codec.encode(target)["input_ids"]
    )
    assert (
        projected["target_sceneplan"]["sources"][0]["description"]
        == current["sources"][0]["description"]
    )


def test_diagnostic_discrete_override_accepts_only_canonical_grammar(codec) -> None:
    class _DiagnosticPlanner:
        _ids = staticmethod(ScenePlanP11V4Planner._ids)
        scene_sketch_codec = SceneSketchCodec(codec)
        patch_codec = ScenePlanEditPatchCodec(codec)
        delta_sketch_codec = DeltaSceneSketchCodec(codec, patch_codec)

    planner = _DiagnosticPlanner()
    target = codec.project_plan(_plan())
    sketch = compile_scene_sketch(target, codec)
    sketch_tokens = planner.scene_sketch_codec.encode(sketch)["input_ids"]
    accepted = ScenePlanP11V4Planner._validate_v4_diagnostic_discrete_tokens(
        planner,
        sketch_tokens,
        task=P11Task.GENERATION,
        device=torch.device("cpu"),
    )
    assert torch.equal(accepted, sketch_tokens)
    with pytest.raises((ModelScenePlanCodecError, ValueError)):
        ScenePlanP11V4Planner._validate_v4_diagnostic_discrete_tokens(
            planner,
            sketch_tokens[:-1],
            task=P11Task.UNDERSTANDING,
            device=torch.device("cpu"),
        )

    spec = {
        "operation": "rotate_source",
        "source_id": "source_0",
        "delta_azimuth_deg": 45,
    }
    edited = planner.patch_codec.apply(target, spec)
    delta = compile_delta_scene_sketch(target, edited, spec, codec)
    delta_tokens = planner.delta_sketch_codec.encode(delta, spec)["input_ids"]
    accepted_delta = (
        ScenePlanP11V4Planner._validate_v4_diagnostic_discrete_tokens(
            planner,
            delta_tokens,
            task=P11Task.EDITING,
            device=torch.device("cpu"),
        )
    )
    assert torch.equal(accepted_delta, delta_tokens)


def test_u_inventory_objective_reuses_finite_vocabulary_logits() -> None:
    logits = torch.zeros(2, 3, 12, requires_grad=True)
    labels = torch.tensor([[0, 5, 0], [0, 9, 0]])
    active = torch.tensor([[False, True, False], [False, True, False]])
    candidates = torch.tensor([2, 5, 9])
    with torch.no_grad():
        logits[0, 1, 5] = 4.0
        logits[1, 1, 9] = 4.0
    loss, count, accuracy = _finite_field_token_objective(
        logits, labels, active, candidates
    )
    assert int(count) == 2
    assert float(accuracy) == 1.0
    assert 0.0 < float(loss) < 0.1
    loss.backward()
    assert logits.grad is not None
    assert bool(logits.grad[active][:, candidates].ne(0).all())
    outside = torch.ones(12, dtype=torch.bool)
    outside[candidates] = False
    assert not bool(logits.grad[active][:, outside].ne(0).any())

    zero, zero_count, zero_accuracy = _finite_field_token_objective(
        logits.detach(), labels, torch.zeros_like(active), candidates
    )
    assert float(zero) == 0.0
    assert int(zero_count) == 0
    assert float(zero_accuracy) == 0.0


def test_frozen_asr_cache_rejects_target_transcript_access(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.sqlite"
    index = tmp_path / "index.sqlite"
    manifest.touch()
    index.touch()
    cache_path = tmp_path / "lexical.sqlite"
    connection = sqlite3.connect(cache_path)
    connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
    connection.executemany(
        "INSERT INTO metadata(key,value) VALUES (?,?)",
        [
            ("schema", P11_LEXICAL_CACHE_SCHEMA),
            ("schema_version", str(P11_LEXICAL_CACHE_VERSION)),
            ("contract", P11_LEXICAL_EVIDENCE_CONTRACT),
            ("source_manifest", str(manifest.resolve())),
            ("source_index", str(index.resolve())),
            ("encoder_revision", "asr-test-revision"),
            ("source", "input_foa_only"),
            ("target_transcript_access", "forbidden"),
            ("confidence_contract", P11_LEXICAL_CONFIDENCE_CONTRACT),
        ],
    )
    connection.execute(
        """
        CREATE TABLE hypotheses(
            ordinal INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            has_speech INTEGER NOT NULL,
            confidence REAL NOT NULL,
            language TEXT NOT NULL,
            language_probability REAL NOT NULL,
            mean_average_log_probability REAL,
            mean_no_speech_probability REAL,
            speech_seconds REAL NOT NULL,
            segment_count INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT INTO hypotheses VALUES "
        "(0, 'a frozen hypothesis', 1, 0.8, 'en', 0.95, -0.2, 0.1, 1.0, 1)"
    )
    connection.commit()
    connection.close()

    cache = P11LexicalEvidenceCache(
        cache_path,
        source_manifest=manifest,
        source_index=index,
        encoder_revision="asr-test-revision",
        expected_ordinals=[0],
    )
    row = cache.row(0)
    assert row["text"] == "a frozen hypothesis"
    assert row["target_transcript_access"] is False
    cache.close()

    connection = sqlite3.connect(cache_path)
    connection.execute(
        "UPDATE metadata SET value='allowed' WHERE key='target_transcript_access'"
    )
    connection.commit()
    connection.close()
    with pytest.raises(RuntimeError, match="target_transcript_access"):
        P11LexicalEvidenceCache(
            cache_path,
            source_manifest=manifest,
            source_index=index,
            encoder_revision="asr-test-revision",
            expected_ordinals=[0],
        )


def _bundle(task: P11Task) -> ScenePlanExecutionBundle:
    editing = task is P11Task.EDITING
    return ScenePlanExecutionBundle(
        task=task,
        sample_id="pipeline-test",
        plan_token_ids=torch.tensor([1], dtype=torch.long),
        sceneplan=_plan(),
        renderer_caption={},
        p10_metadata={
            "semantic_caption_compiler_version": (
                P10_SEMANTIC_CAPTION_COMPILER_VERSION
            ),
            "semantic_caption_contract": P10_SEMANTIC_CAPTION_CONTRACT,
        },
        model_num_samples=32,
        latent_frames_valid=1,
        editing_contract=P11_EDITING_CONTRACT if editing else None,
        edit_patch_token_ids=(
            torch.tensor([2], dtype=torch.long) if editing else None
        ),
        edit_patch={"operation": "no_op"} if editing else None,
    )


class _Planner(torch.nn.Module):
    transfusion_cot_enabled = True

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def encode_audio(self, waveform):
        assert tuple(waveform.shape[:2]) == (1, 4)
        return torch.zeros((1, 64, 1), device=waveform.device)

    def plan_generation(self, *args, **kwargs):
        return _bundle(P11Task.GENERATION)

    def plan_understanding(self, *args, **kwargs):
        self.understanding_kwargs = kwargs
        return _bundle(P11Task.UNDERSTANDING)

    def plan_editing_audio_aware(self, *args, **kwargs):
        self.editing_kwargs = kwargs
        plan = _plan()
        return AudioAwareEditPlanningBundle(
            sample_id="pipeline-test",
            observed_plan_token_ids=torch.tensor([1]),
            observed_sceneplan=plan,
            edit_patch_token_ids=torch.tensor([2]),
            edit_patch={"operation": "no_op"},
            revised_plan_token_ids=torch.tensor([1]),
            revised_sceneplan=plan,
            source_to_target_slot_map={"source_0": "source_0"},
            execution_bundle=_bundle(P11Task.EDITING),
        )


class _Executor:
    def __init__(self) -> None:
        self.seeds: list[int | None] = []

    def render(self, bundle, *, seed=None):
        self.seeds.append(seed)
        return torch.full((4, bundle.model_num_samples), float(seed or 0))


def test_unified_pipeline_enforces_same_seed_and_supports_u_cycle() -> None:
    planner = _Planner()
    executor = _Executor()
    pipeline = ScenePlanCoTPipeline(planner, executor=executor)
    generated = pipeline.generate("a bell", seed=73)
    semantic = torch.zeros(512)
    edited = pipeline.edit_from_result(
        "move it", current=generated, input_semantic=semantic
    )
    assert generated.foa_role == "generated"
    assert edited.foa_role == "edited_same_seed"
    assert generated.render_seed == edited.render_seed == 73
    assert executor.seeds[:2] == [73, 73]
    assert tuple(planner.editing_kwargs["input_foa_latent"].shape) == (64, 1)
    assert planner.editing_kwargs["input_semantic"] is semantic
    assert planner.editing_kwargs["input_sceneplan"] == generated.sceneplan
    assert edited.audio_aware_edit_bundle is not None

    lexical = {"input_ids": torch.ones(4, dtype=torch.long)}
    understood = pipeline.understand(
        torch.zeros(64, 1),
        input_lexical=lexical,
        render_closure=True,
        seed=19,
    )
    assert understood.task is P11Task.UNDERSTANDING
    assert understood.foa_role == "understanding_p10_cycle"
    assert understood.render_seed == 19
    assert planner.understanding_kwargs["input_lexical"] is lexical
    assert executor.seeds[-1] == 19
