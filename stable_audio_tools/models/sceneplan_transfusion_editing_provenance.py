"""Immutable external-runtime identities for Transfusion Editing.

The frozen Qwen text backbone is intentionally kept outside the parent
``nn.Module`` registration tree by ``QwenTextConditioner``.  Consequently it
is not serialized in Editing DiT or joint checkpoints and must be pinned as a
separate runtime dependency.
"""

from __future__ import annotations
import os

from functools import lru_cache
from pathlib import Path
from typing import Any

from stable_audio_tools.data.sceneplan_transfusion_editing_index import sha256_file


FROZEN_QWEN_ROOT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/Qwen/Qwen3.5-0.8B"
)
FROZEN_QWEN_RUNTIME_FILES: dict[str, tuple[int, str]] = {
    "chat_template.jinja": (
        7_755,
        "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    ),
    "config.json": (
        2_907,
        "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    ),
    "merges.txt": (
        3_353_259,
        "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    ),
    "model.safetensors-00001-of-00001.safetensors": (
        1_746_942_600,
        "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    ),
    "model.safetensors.index.json": (
        50_900,
        "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
    ),
    "tokenizer.json": (
        12_807_982,
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    ),
    "tokenizer_config.json": (
        16_709,
        "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
    ),
    "vocab.json": (
        6_722_759,
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    ),
}

# Keep the producer and every formal consumer on one source-identity contract.
# These are Editing-only paths; Generation training does not import this module.
JOINT_TRAINING_SOURCE_PATHS = (
    "pyproject.toml",
    "scripts/t2a/eval/select_sceneplan_transfusion_editing_dit_checkpoint.py",
    "scripts/t2a/train/run_sceneplan_transfusion_editing_ar_joint_full_5gpu.sh",
    "scripts/t2a/train/sceneplan_transfusion_editing_dit_run_contract.py",
    "scripts/t2a/train/sceneplan_transfusion_editing_joint_run_contract.py",
    "scripts/t2a/train/train_sceneplan_transfusion_editing_ar_joint_full.py",
    "stable_audio_tools/configuration.py",
    "stable_audio_tools/data/model_sceneplan.py",
    "stable_audio_tools/data/model_sceneplan_codec_v3.py",
    "stable_audio_tools/data/model_sceneplan_codec_v4.py",
    "stable_audio_tools/data/sceneplan_bucket_sampler.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing_ar_dataset.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing_plan.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing_dataset.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing_index.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing_joint_dataset.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing_m2d_clap.py",
    "stable_audio_tools/data/text_conditioning.py",
    "stable_audio_tools/inference/generation.py",
    "stable_audio_tools/inference/sampling.py",
    "stable_audio_tools/models/blocks.py",
    "stable_audio_tools/models/conditioners.py",
    "stable_audio_tools/models/diffusion.py",
    "stable_audio_tools/models/dit.py",
    "stable_audio_tools/models/factory.py",
    "stable_audio_tools/models/pretransforms.py",
    "stable_audio_tools/models/sceneplan_transfusion_editing_ar.py",
    "stable_audio_tools/models/sceneplan_transfusion_editing_m2d_clap.py",
    "stable_audio_tools/models/sceneplan_transfusion_editing_m2d_runtime.py",
    "stable_audio_tools/models/sceneplan_transfusion_editing_provenance.py",
    "stable_audio_tools/models/transformer.py",
    "stable_audio_tools/models/utils.py",
    "stable_audio_tools/training/diffusion.py",
    "stable_audio_tools/training/ema.py",
    "stable_audio_tools/training/factory.py",
    "stable_audio_tools/training/utils.py",
    "uv.lock",
)

JOINT_SELECTION_SOURCE_PATHS = (
    "scripts/t2a/eval/select_sceneplan_transfusion_editing_joint_checkpoint.py",
    "scripts/t2a/eval/run_sceneplan_transfusion_editing_joint_checkpoint_selection_5gpu.sh",
    "scripts/t2a/eval/select_sceneplan_transfusion_editing_dit_checkpoint.py",
    "scripts/t2a/train/run_sceneplan_transfusion_editing_ar_joint_full_5gpu.sh",
    "scripts/t2a/train/sceneplan_transfusion_editing_dit_run_contract.py",
    "scripts/t2a/train/sceneplan_transfusion_editing_joint_run_contract.py",
    "scripts/t2a/train/train_sceneplan_transfusion_editing_ar_joint_full.py",
    "stable_audio_tools/configuration.py",
    "stable_audio_tools/data/model_sceneplan.py",
    "stable_audio_tools/data/model_sceneplan_codec_v3.py",
    "stable_audio_tools/data/model_sceneplan_codec_v4.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing_ar_dataset.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing_plan.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing_dataset.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing_index.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing_joint_dataset.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing_m2d_clap.py",
    "stable_audio_tools/data/sceneplan_transfusion_generation_ar_evaluation.py",
    "stable_audio_tools/models/conditioners.py",
    "stable_audio_tools/models/diffusion.py",
    "stable_audio_tools/models/dit.py",
    "stable_audio_tools/models/factory.py",
    "stable_audio_tools/models/sceneplan_transfusion_editing_ar.py",
    "stable_audio_tools/models/sceneplan_transfusion_editing_m2d_clap.py",
    "stable_audio_tools/models/sceneplan_transfusion_editing_m2d_runtime.py",
    "stable_audio_tools/models/sceneplan_transfusion_editing_pipeline.py",
    "stable_audio_tools/models/sceneplan_transfusion_editing_provenance.py",
    "stable_audio_tools/models/transformer.py",
    "stable_audio_tools/models/utils.py",
    "stable_audio_tools/training/diffusion.py",
    "stable_audio_tools/training/ema.py",
    "stable_audio_tools/training/factory.py",
)


@lru_cache(maxsize=1)
def verify_frozen_qwen_runtime(
    model_path: str | Path = FROZEN_QWEN_ROOT,
) -> dict[str, Any]:
    """Hash every model/tokenizer file that affects Editing conditioning."""

    root = Path(model_path).expanduser().resolve(strict=True)
    if root != FROZEN_QWEN_ROOT.resolve(strict=True):
        raise RuntimeError(f"Editing Qwen runtime path changed: {root}")
    files: dict[str, dict[str, Any]] = {}
    for relative, (expected_bytes, expected_sha256) in sorted(
        FROZEN_QWEN_RUNTIME_FILES.items()
    ):
        path = (root / relative).resolve(strict=True)
        if path.parent != root:
            raise RuntimeError("Editing Qwen runtime escaped its frozen root")
        observed_bytes = path.stat().st_size
        observed_sha256 = sha256_file(path)
        if (
            observed_bytes != int(expected_bytes)
            or observed_sha256 != str(expected_sha256)
        ):
            raise RuntimeError(f"frozen Editing Qwen file changed: {path}")
        files[relative] = {
            "path": str(path),
            "bytes": observed_bytes,
            "sha256": observed_sha256,
        }
    return {
        "contract": "frozen_qwen35_0p8b_model_and_tokenizer_sha256_v1",
        "root": str(root),
        "files": files,
        "model_parameter_elements": 752_393_024,
        "registered_under_parent_module": False,
        "frozen": True,
    }


__all__ = [
    "FROZEN_QWEN_ROOT",
    "FROZEN_QWEN_RUNTIME_FILES",
    "JOINT_SELECTION_SOURCE_PATHS",
    "JOINT_TRAINING_SOURCE_PATHS",
    "verify_frozen_qwen_runtime",
]
