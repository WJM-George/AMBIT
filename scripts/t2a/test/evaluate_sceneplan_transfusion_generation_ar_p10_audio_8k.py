#!/usr/bin/env python3
"""Frozen-P10 FOA closure for the Generation-AR 8K plan evaluation.

This evaluator never reruns Generation AR.  It consumes the immutable output
of ``evaluate_sceneplan_transfusion_generation_ar_8k.py``, independently
reconstructs the P10 execution bundles, and measures the audio consequence of
the predicted ScenePlan against the test target ScenePlan under identical P10
noise.

Rows whose canonical predicted and target ScenePlan bytes are equal are not
rendered exhaustively: equality of independently finalized render inputs is a
complete functional-equivalence proof.  A deterministic, stratified panel of
up to 32 such rows is nevertheless rendered as target/repeated-target/
prediction triplets and must be bit exact.  Every non-equal row is rendered as
a same-seed pair on one GPU.  Only hashes and metrics are retained, never the
full waveforms.

This core closure deliberately excludes CLAP and Whisper.  It therefore
measures waveform/spectral, FOA intensity/DoA, and activity consequences, but
does not claim absolute semantic-audio or transcript accuracy.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import platform
from typing import Any, Iterable, Mapping, Sequence
import zlib

# Must precede torch CUDA initialization for deterministic GEMM.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.evaluate_sceneplan_p11_v4_p10_closure import (  # noqa: E402
    _fit_length,
    _pair_metrics,
)
from scripts.t2a.eval.sceneplan_44_eval_common import audio_qc  # noqa: E402
from scripts.t2a.eval.score_sceneplan_dit_p10_core import (  # noqa: E402
    _activity_metrics,
    _doa_metrics,
    _paired_doa_metrics,
)
from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.data.model_sceneplan_codec_v4 import (  # noqa: E402
    ModelScenePlanCodecV4,
)
from stable_audio_tools.data.foa_intensity import (  # noqa: E402
    foa_to_intensity_trajectory,
)
from stable_audio_tools.data.sceneplan_p11_single_turn import (  # noqa: E402
    P10_CANONICAL_CHECKPOINT,
    P10_CANONICAL_CHECKPOINT_SHA256,
    P10_CANONICAL_CHECKPOINT_STEP,
    P10_CANONICAL_EXECUTOR_FAMILY,
    P10_CANONICAL_MODEL_CONFIG,
    P10_CANONICAL_MODEL_CONFIG_SHA256,
    P11Task,
    finalize_sceneplan_for_p10,
)
from stable_audio_tools.data.sceneplan_transfusion_generation_ar_dataset import (  # noqa: E402
    GENERATION_AR_CODEC_FINGERPRINT,
    GENERATION_AR_SOURCE_INDEX_SHA256,
    manifest_summary,
)
from stable_audio_tools.data.sceneplan_transfusion_generation_ar_evaluation import (  # noqa: E402
    GENERATION_AR_EVALUATION_CONTRACT,
)
from stable_audio_tools.inference.sceneplan_cot import (  # noqa: E402
    P10ScenePlanDiTExecutor,
)
from stable_audio_tools.models.sceneplan_transfusion_generation_ar import (  # noqa: E402
    GENERATION_AR_CONTRACT,
    P10_V11_RESOLVED_CONFIG_SHA256,
)


AUDIO_CLOSURE_CONTRACT = (
    "p10v11_frozen_generation_ar_8k_foa_closure_v1"
)
AUDIO_CLOSURE_SCHEMA = (
    "stable_audio_tools.sceneplan_transfusion_generation_ar_p10_audio"
)
EXPECTED_ROWS = 8_000
EXPECTED_WORLD_SIZE = 3
EXPECTED_VISIBLE_DEVICES = "0,1,2"
COMPLETION_BARRIER_TIMEOUT_SECONDS = 60 * 60
CANONICAL_ROOT_SEED = 42
CANONICAL_P10_STEPS = 100
DEFAULT_EXACT_ANCHORS = 32
P10_RELEASE = REPO_ROOT / "artifacts/releases/P10_SCENEPLAN_DIT_V11_150K_RELEASE.json"
P10_RELEASE_SHA256 = (
    "4074715a06e208ced678701d2bf8ad042a6e7a1ca0903d60d13a87d5b874ea51"
)
TEST_MANIFEST = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/transfusion_shared_v1/"
    "generation_ar/test.sqlite"
)
CODEC_PATH = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)
QWEN_PATH = Path("/mnt/sdc/ckpts/pretrained/Qwen/Qwen3.5-0.8B")

CODEC_ARTIFACT_SHA256 = {
    "READY": "36f98e8d91be72dc7fb0f550436e610c4487300f13c89ad0b5afb2b6bc1f6aff",
    "codec.json": "b91b3d51555263fa5817e8ecaa73b5fd07d0a8564c99e0415751c6b1dc2627b0",
    "sentencepiece.model": "d973eed83802e4da60ca27c9cd9bf31864c2745c4618c532b7bab93e70ba8b22",
}
QWEN_CRITICAL_SHA256 = {
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "config.json": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    "model.safetensors-00001-of-00001.safetensors": (
        "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
    ),
    "model.safetensors.index.json": (
        "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e"
    ),
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": (
        "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c"
    ),
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
}

PLAN_PREDICTION_COLUMNS = (
    "ordinal",
    "sample_id",
    "template_id",
    "source_count",
    "target_token_count",
    "predicted_token_count",
    "status",
    "error",
    "target_sceneplan_sha256",
    "prediction_sceneplan_sha256",
    "prediction_sceneplan_zlib",
    "predicted_token_ids_u16le",
    "metrics_json",
    "generation_sec",
)

AUDIO_RESULT_COLUMNS = (
    "ordinal",
    "sample_id",
    "template_id",
    "source_count",
    "plan_exact",
    "exact_anchor",
    "status",
    "error",
    "target_sceneplan_sha256",
    "prediction_sceneplan_sha256",
    "target_bundle_sha256",
    "prediction_bundle_sha256",
    "target_render_input_sha256",
    "prediction_render_input_sha256",
    "render_seed",
    "length_group",
    "target_model_num_samples",
    "prediction_model_num_samples",
    "target_latent_frames",
    "prediction_latent_frames",
    "target_foa_sha256",
    "prediction_foa_sha256",
    "target_repeat_foa_sha256",
    "metrics_json",
    "render_sec",
)


@dataclass(frozen=True)
class PlanEvaluationRow:
    ordinal: int
    sample_id: str
    template_id: str
    source_count: int
    room_type: str
    motion_signature: str
    manifest_latent_frames: int
    target_sceneplan_sha256: str
    prediction_sceneplan_sha256: str
    target_sceneplan_bytes: bytes
    prediction_sceneplan_bytes: bytes
    target_sceneplan: Mapping[str, Any]
    prediction_sceneplan: Mapping[str, Any]
    target_token_ids: tuple[int, ...]
    prediction_token_ids: tuple[int, ...]

    @property
    def plan_exact(self) -> bool:
        return (
            self.target_sceneplan_sha256 == self.prediction_sceneplan_sha256
            and self.target_sceneplan_bytes == self.prediction_sceneplan_bytes
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, default=TEST_MANIFEST)
    parser.add_argument("--p10-release", type=Path, default=P10_RELEASE)
    parser.add_argument("--codec-path", type=Path, default=CODEC_PATH)
    parser.add_argument("--exact-anchors", type=int, default=DEFAULT_EXACT_ANCHORS)
    parser.add_argument("--seed", type=int, default=CANONICAL_ROOT_SEED)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Revalidate an already completed output without initializing CUDA or "
            "rewriting its SQLite shards."
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "Source snapshot whose relative source hashes are recorded in an "
            "existing RUN_CONTRACT; used only with --verify-only."
        ),
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical_json_bytes(list(tensor.shape)))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _hashable(value: Any) -> Any:
    """Represent nested render inputs without lossy tensor JSON conversion."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "__tensor__": True,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(
                tensor.view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {str(key): _hashable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hashable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("render-input fingerprint rejects non-finite floats")
        return value
    raise TypeError(f"unsupported fingerprint value: {type(value).__name__}")


def _bundle_fingerprints(bundle: Any) -> dict[str, str]:
    bundle_payload = {
        "task": bundle.task,
        "sample_id": bundle.sample_id,
        "plan_token_ids": bundle.plan_token_ids,
        "sceneplan": bundle.sceneplan,
        "renderer_caption": bundle.renderer_caption,
        "p10_metadata": bundle.p10_metadata,
        "model_num_samples": int(bundle.model_num_samples),
        "latent_frames_valid": int(bundle.latent_frames_valid),
        "execution_contract": bundle.execution_contract,
        "editing_contract": bundle.editing_contract,
        "preserves_unedited_waveform": bool(bundle.preserves_unedited_waveform),
        "localized_editing": bool(bundle.localized_editing),
    }
    # These are precisely the values read by P10ScenePlanDiTExecutor.render.
    render_payload = {
        "task": bundle.task,
        "p10_metadata": bundle.p10_metadata,
        "model_num_samples": int(bundle.model_num_samples),
        "latent_frames_valid": int(bundle.latent_frames_valid),
        "execution_contract": bundle.execution_contract,
    }
    return {
        "bundle_sha256": _canonical_sha256(_hashable(bundle_payload)),
        "render_input_sha256": _canonical_sha256(_hashable(render_payload)),
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _stable_seed(root_seed: int, sample_id: str) -> int:
    payload = (
        f"{AUDIO_CLOSURE_CONTRACT}\0{int(root_seed)}\0{sample_id}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def _stable_order_key(seed: int, *parts: Any) -> bytes:
    payload = "\0".join([str(seed), *(str(value) for value in parts)]).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).digest()


def _motion_family(signature: str) -> str:
    motions = tuple(item for item in str(signature).split(",") if item)
    return "all_static" if motions and set(motions) == {"static"} else "has_linear"


def _select_exact_anchors(
    rows: Sequence[PlanEvaluationRow], *, count: int, seed: int
) -> tuple[int, ...]:
    """Deterministic round-robin over source/template/room/motion strata."""

    if int(count) < 0:
        raise ValueError("exact anchor count must be non-negative")
    exact = [row for row in rows if row.plan_exact]
    wanted = min(int(count), len(exact))
    if wanted == 0:
        return ()
    strata: dict[tuple[Any, ...], list[PlanEvaluationRow]] = defaultdict(list)
    for row in exact:
        key = (
            int(row.source_count),
            str(row.template_id),
            str(row.room_type),
            _motion_family(row.motion_signature),
        )
        strata[key].append(row)
    for key, values in strata.items():
        values.sort(
            key=lambda row: _stable_order_key(
                seed, "row", *key, row.sample_id, row.ordinal
            )
        )
    keys = sorted(
        strata,
        key=lambda key: _stable_order_key(seed, "stratum", *key),
    )
    selected: list[int] = []
    while len(selected) < wanted:
        progressed = False
        for key in keys:
            if not strata[key]:
                continue
            selected.append(int(strata[key].pop(0).ordinal))
            progressed = True
            if len(selected) == wanted:
                break
        if not progressed:
            raise RuntimeError("exact anchor selector exhausted unexpectedly")
    return tuple(sorted(selected))


def _duration_group(
    *,
    target_samples: int,
    prediction_samples: int,
    target_frames: int,
    prediction_frames: int,
) -> str:
    if target_samples == prediction_samples and target_frames == prediction_frames:
        return "same_duration_and_latent_length"
    if target_frames == prediction_frames:
        return "same_latent_length_duration_mismatch"
    return "duration_and_latent_length_mismatch"


def _spatial_pair_metrics(
    prediction_foa: torch.Tensor,
    target_foa: torch.Tensor,
    *,
    hop: int = 1024,
    min_coherence: float = 0.1,
) -> dict[str, Any]:
    """Compare pairwise FOA active intensity, including diffuseness.

    This is the GPU-independent metric surface used by the older Spatial-CoT
    evaluator's ``_spatial_alignment_metrics``.  It lives here because that
    legacy script performs a GPU-driver assertion at import time.  The P10
    closure already runs on CPU tensors and must remain importable in CPU tests.
    """

    prediction = prediction_foa.detach().float().cpu()
    target = target_foa.detach().float().cpu()
    if (
        prediction.ndim != 2
        or target.ndim != 2
        or int(prediction.shape[0]) != 4
        or int(target.shape[0]) != 4
        or int(hop) < 1
    ):
        raise ValueError("spatial pair metrics require two FOA [4,N] tensors")
    frame_count = min(int(prediction.shape[-1]), int(target.shape[-1])) // int(hop)
    if frame_count < 1:
        raise ValueError("FOA pair is shorter than one spatial frame")
    samples = frame_count * int(hop)
    prediction = prediction[:, :samples]
    target = target[:, :samples]
    prediction_trajectory = foa_to_intensity_trajectory(prediction, hop=int(hop))
    target_trajectory = foa_to_intensity_trajectory(target, hop=int(hop))
    target_energy = target.reshape(4, frame_count, int(hop)).square().mean(dim=(0, 2))
    energy_floor = max(1.0e-12, float(target_energy.max()) * 1.0e-4)
    active = target_energy >= energy_floor
    if not bool(active.any()):
        raise ValueError("target FOA has no active spatial frames")
    prediction_coherence = (1.0 - prediction_trajectory[:, 3]).clamp(0.0, 1.0)
    target_coherence = (1.0 - target_trajectory[:, 3]).clamp(0.0, 1.0)
    prediction_direction = prediction_trajectory[:, :3]
    target_direction = target_trajectory[:, :3]
    prediction_norm = prediction_direction.norm(dim=-1)
    target_norm = target_direction.norm(dim=-1)
    valid = (
        active
        & prediction_coherence.ge(float(min_coherence))
        & target_coherence.ge(float(min_coherence))
        & prediction_norm.gt(1.0e-6)
        & target_norm.gt(1.0e-6)
    )
    result: dict[str, Any] = {
        "frame_count": frame_count,
        "active_frame_count": int(active.sum()),
        "valid_direction_frame_count": int(valid.sum()),
        "valid_direction_fraction": float(valid.sum() / active.sum()),
        "direction_cosine": None,
        "angular_error_mean_deg": None,
        "angular_error_median_deg": None,
        "angular_error_p90_deg": None,
        "diffuseness_mae": None,
        "prediction_mean_diffuseness": float(
            prediction_trajectory[active, 3].mean()
        ),
        "target_mean_diffuseness": float(target_trajectory[active, 3].mean()),
        "min_coherence": float(min_coherence),
        "hop": int(hop),
        "metric_origin": (
            "evaluate_spatial_cot_checkpoint._spatial_alignment_metrics"
        ),
    }
    weights = target_energy[active].clamp_min(1.0e-12)
    diffuseness_error = (
        prediction_trajectory[active, 3] - target_trajectory[active, 3]
    ).abs()
    result["diffuseness_mae"] = float(
        (diffuseness_error * weights).sum() / weights.sum()
    )
    if bool(valid.any()):
        prediction_unit = prediction_direction[valid] / prediction_norm[valid, None]
        target_unit = target_direction[valid] / target_norm[valid, None]
        cosine = (prediction_unit * target_unit).sum(dim=-1).clamp(-1.0, 1.0)
        angles = torch.rad2deg(torch.acos(cosine))
        direction_weights = (
            target_energy[valid] * target_coherence[valid]
        ).clamp_min(1.0e-12)
        result.update(
            {
                "direction_cosine": float(
                    (cosine * direction_weights).sum() / direction_weights.sum()
                ),
                "angular_error_mean_deg": float(
                    (angles * direction_weights).sum() / direction_weights.sum()
                ),
                "angular_error_median_deg": float(angles.median()),
                "angular_error_p90_deg": float(torch.quantile(angles, 0.9)),
            }
        )
    return result


def _distributed() -> tuple[int, int, int, torch.device, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").replace(" ", "")
    if visible != EXPECTED_VISIBLE_DEVICES:
        raise RuntimeError(
            "Generation AR P10 closure requires CUDA_VISIBLE_DEVICES=0,1,2"
        )
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != EXPECTED_WORLD_SIZE or not torch.cuda.is_available():
        raise RuntimeError("Generation AR P10 closure requires exactly three GPUs")
    if not 0 <= local_rank < EXPECTED_WORLD_SIZE:
        raise RuntimeError("invalid local rank for Generation AR P10 closure")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    # Long-tail shard completion must not block unfinished GPU render work on
    # an NCCL barrier.  Use a CPU coordination group for that one wait.
    completion_group = dist.new_group(
        backend="gloo",
        timeout=timedelta(seconds=COMPLETION_BARRIER_TIMEOUT_SECONDS),
    )
    return rank, local_rank, world_size, device, completion_group


def _configure_determinism(seed: int, *, rank: int) -> None:
    if int(seed) != CANONICAL_ROOT_SEED:
        raise ValueError(f"canonical P10 closure requires seed {CANONICAL_ROOT_SEED}")
    np.random.seed(int(seed) + int(rank))
    torch.manual_seed(int(seed) + int(rank))
    torch.cuda.manual_seed_all(int(seed) + int(rank))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _safe_repo_source(relative: str) -> Path:
    candidate = (REPO_ROOT / str(relative)).resolve(strict=True)
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"source hash escapes repository: {relative}") from exc
    if not candidate.is_file():
        raise RuntimeError(f"source hash is not a file: {relative}")
    return candidate


def _verify_plan_source_hashes(run_contract: Mapping[str, Any]) -> None:
    source_hashes = run_contract.get("source_sha256")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise RuntimeError("plan evaluation lacks source-code hashes")
    for relative, expected in source_hashes.items():
        source = _safe_repo_source(str(relative))
        if _sha256_file(source) != str(expected):
            raise RuntimeError(f"plan evaluator source changed: {relative}")


def _prediction_columns(connection: sqlite3.Connection) -> tuple[str, ...]:
    return tuple(
        str(row[1]) for row in connection.execute("PRAGMA table_info(predictions)")
    )


def _load_prediction_records(
    prediction_dir: Path,
    *,
    world_size: int,
    plan_contract_sha256: str,
) -> tuple[dict[int, tuple[Any, ...]], dict[str, str]]:
    records: dict[int, tuple[Any, ...]] = {}
    shard_hashes: dict[str, str] = {}
    expected_names = {f"rank_{rank:03d}.sqlite" for rank in range(world_size)}
    actual_names = {path.name for path in prediction_dir.glob("rank_*.sqlite")}
    if actual_names != expected_names:
        raise RuntimeError(
            f"plan prediction shard set mismatch: {sorted(actual_names)}"
        )
    for rank in range(world_size):
        path = (prediction_dir / f"rank_{rank:03d}.sqlite").resolve(strict=True)
        if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm")):
            raise RuntimeError(f"plan prediction shard has live SQLite sidecars: {path}")
        shard_hashes[str(path)] = _sha256_file(path)
        connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError(f"plan prediction shard failed integrity: {path}")
            if _prediction_columns(connection) != PLAN_PREDICTION_COLUMNS:
                raise RuntimeError(f"plan prediction schema changed: {path}")
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            expected = {
                "contract": GENERATION_AR_EVALUATION_CONTRACT,
                "rank": str(rank),
                "world_size": str(world_size),
                "run_contract_sha256": plan_contract_sha256,
                "status": "COMPLETE",
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise RuntimeError(
                        f"plan prediction shard metadata mismatch: {path}: {key}"
                    )
            rows = list(
                connection.execute(
                    "SELECT " + ",".join(PLAN_PREDICTION_COLUMNS) + " "
                    "FROM predictions ORDER BY ordinal"
                )
            )
            if int(metadata.get("rows", -1)) != len(rows):
                raise RuntimeError(f"plan prediction shard row count mismatch: {path}")
            for record in rows:
                ordinal = int(record[0])
                if ordinal in records:
                    raise RuntimeError(f"duplicate plan prediction ordinal {ordinal}")
                if ordinal % world_size != rank:
                    raise RuntimeError(f"plan prediction rank ownership mismatch {ordinal}")
                records[ordinal] = record
        finally:
            connection.close()
    return records, shard_hashes


def _validate_canonical_plan_bytes(
    compressed: bytes, expected_sha256: str, *, label: str, ordinal: int
) -> tuple[bytes, Mapping[str, Any]]:
    try:
        payload = zlib.decompress(compressed)
        plan = json.loads(payload)
    except Exception as exc:
        raise RuntimeError(f"invalid {label} ScenePlan at ordinal {ordinal}") from exc
    if not isinstance(plan, Mapping):
        raise RuntimeError(f"{label} ScenePlan is not an object at ordinal {ordinal}")
    if hashlib.sha256(payload).hexdigest() != str(expected_sha256):
        raise RuntimeError(f"{label} ScenePlan SHA256 mismatch at ordinal {ordinal}")
    if _canonical_json_bytes(plan) != payload:
        raise RuntimeError(f"{label} ScenePlan bytes are not canonical at ordinal {ordinal}")
    return payload, plan


def _load_verified_plan_rows(
    plan_evaluation_dir: Path,
    test_manifest: Path,
    *,
    expected_rows: int = EXPECTED_ROWS,
    verify_source_hashes: bool = True,
) -> tuple[list[PlanEvaluationRow], dict[str, Any]]:
    """Load and independently bind every plan prediction to the test source."""

    plan_dir = plan_evaluation_dir.expanduser().resolve(strict=True)
    manifest = test_manifest.expanduser().resolve(strict=True)
    run_path = (plan_dir / "RUN_CONTRACT.json").resolve(strict=True)
    summary_path = (plan_dir / "SUMMARY.json").resolve(strict=True)
    teacher_path = (plan_dir / "TEACHER_FORCED.json").resolve(strict=True)
    run = _read_json_object(run_path, label="plan evaluation run contract")
    summary = _read_json_object(summary_path, label="plan evaluation summary")
    teacher = _read_json_object(teacher_path, label="teacher-forced report")
    if (
        run.get("schema")
        != "stable_audio_tools.sceneplan_transfusion_generation_ar_evaluation_run"
        or int(run.get("schema_version", -1)) != 2
        or run.get("contract") != GENERATION_AR_EVALUATION_CONTRACT
        or run.get("generation_ar_contract") != GENERATION_AR_CONTRACT
        or run.get("evaluation_split") != "test"
        or int(run.get("row_limit", -1)) != int(expected_rows)
        or int(run.get("world_size", -1)) != EXPECTED_WORLD_SIZE
        or str(run.get("cuda_visible_devices", "")).replace(" ", "")
        != EXPECTED_VISIBLE_DEVICES
        or Path(
            str(run.get("evaluation_manifest", {}).get("path", ""))
        ).resolve()
        != manifest
    ):
        raise RuntimeError("plan evaluation run contract mismatch")
    manifest_sha256 = _sha256_file(manifest)
    if run.get("evaluation_manifest_sha256") != manifest_sha256:
        raise RuntimeError("plan evaluation/test manifest SHA256 mismatch")
    if (
        summary.get("status") != "PASS"
        or summary.get("contract") != GENERATION_AR_EVALUATION_CONTRACT
        or int(summary.get("rows", -1)) != int(expected_rows)
        or summary.get("evaluation_split") != "test"
        or summary.get("evaluation_manifest_sha256") != manifest_sha256
        or summary.get("checkpoint") != run.get("checkpoint")
        or summary.get("checkpoint_sha256") != run.get("checkpoint_sha256")
        or int(summary.get("checkpoint_step", -1))
        != int(run.get("checkpoint_step", -2))
        or summary.get("teacher_forced") != teacher
    ):
        raise RuntimeError("audio closure requires a passing complete plan evaluation")
    free_coverage = dict(summary.get("free_decode", {}).get("coverage") or {})
    status_counts = dict(summary.get("free_decode", {}).get("status_counts") or {})
    if (
        free_coverage.get("ordinal_coverage_exact") is not True
        or int(free_coverage.get("rows", -1)) != int(expected_rows)
        or status_counts != {"ok": int(expected_rows)}
    ):
        raise RuntimeError("plan evaluation does not contain 8K successful decodes")
    if verify_source_hashes:
        _verify_plan_source_hashes(run)
    checkpoint = Path(str(run.get("checkpoint", ""))).resolve(strict=True)
    if _sha256_file(checkpoint) != str(run.get("checkpoint_sha256")):
        raise RuntimeError("selected Generation AR checkpoint changed")

    plan_contract_sha256 = _canonical_sha256(run)
    predictions, shard_hashes = _load_prediction_records(
        plan_dir / "predictions",
        world_size=EXPECTED_WORLD_SIZE,
        plan_contract_sha256=plan_contract_sha256,
    )
    if sorted(predictions) != list(range(int(expected_rows))):
        raise RuntimeError("plan prediction shards do not cover exact requested ordinals")

    manifest_info = manifest_summary(manifest)
    metadata = dict(manifest_info["metadata"])
    if (
        metadata.get("split") != "test"
        or metadata.get("is_full_split") != "true"
        or metadata.get("seed") != str(CANONICAL_ROOT_SEED)
        or int(metadata.get("rows", -1)) != int(expected_rows)
        or metadata.get("codec_fingerprint") != GENERATION_AR_CODEC_FINGERPRINT
        or metadata.get("source_index_sha256")
        != GENERATION_AR_SOURCE_INDEX_SHA256["test"]
        or Path(str(metadata.get("codec_path", ""))).resolve() != CODEC_PATH.resolve()
        or Path(str(metadata.get("tokenizer_path", ""))).resolve()
        != QWEN_PATH.resolve()
    ):
        raise RuntimeError("test manifest contract mismatch for P10 closure")
    source_index = Path(str(metadata["source_index"])).resolve(strict=True)
    if _sha256_file(source_index) != str(metadata["source_index_sha256"]):
        raise RuntimeError("test source index SHA256 changed")

    connection = sqlite3.connect(f"file:{manifest}?mode=ro&immutable=1", uri=True)
    rows: list[PlanEvaluationRow] = []
    try:
        connection.execute("PRAGMA query_only=ON")
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("test manifest SQLite integrity check failed")
        cursor = connection.execute(
            """
            SELECT ordinal,sample_id,template_id,source_count,room_type,
                   motion_signature,latent_frames_valid,target_sceneplan_zlib,
                   target_sceneplan_sha256,target_token_ids_u16le,
                   target_token_count,target_tokens_sha256
            FROM rows ORDER BY ordinal
            """
        )
        for manifest_record in cursor:
            (
                ordinal,
                sample_id,
                template_id,
                source_count,
                room_type,
                motion_signature,
                latent_frames,
                target_zlib,
                target_sha,
                target_token_blob,
                target_token_count,
                target_tokens_sha,
            ) = manifest_record
            ordinal = int(ordinal)
            prediction = predictions[ordinal]
            record = dict(zip(PLAN_PREDICTION_COLUMNS, prediction))
            if (
                record["status"] != "ok"
                or record["error"] is not None
                or str(record["sample_id"]) != str(sample_id)
                or str(record["template_id"]) != str(template_id)
                or int(record["source_count"]) != int(source_count)
                or int(record["target_token_count"]) != int(target_token_count)
                or str(record["target_sceneplan_sha256"]) != str(target_sha)
                or record["prediction_sceneplan_zlib"] is None
                or record["predicted_token_ids_u16le"] is None
            ):
                raise RuntimeError(f"plan prediction/test identity mismatch {ordinal}")
            target_bytes, target_plan = _validate_canonical_plan_bytes(
                target_zlib, str(target_sha), label="target", ordinal=ordinal
            )
            prediction_bytes, prediction_plan = _validate_canonical_plan_bytes(
                record["prediction_sceneplan_zlib"],
                str(record["prediction_sceneplan_sha256"]),
                label="prediction",
                ordinal=ordinal,
            )
            if (
                str(target_plan.get("sample_id")) != str(sample_id)
                or str(prediction_plan.get("sample_id")) != str(sample_id)
            ):
                raise RuntimeError(f"ScenePlan sample_id mismatch at ordinal {ordinal}")
            target_blob = bytes(target_token_blob)
            if hashlib.sha256(target_blob).hexdigest() != str(target_tokens_sha):
                raise RuntimeError(f"target token SHA256 mismatch at ordinal {ordinal}")
            target_tokens = np.frombuffer(target_blob, dtype="<u2").astype(np.int64)
            prediction_tokens = np.frombuffer(
                record["predicted_token_ids_u16le"], dtype="<u2"
            ).astype(np.int64)
            if (
                len(target_tokens) != int(target_token_count)
                or len(prediction_tokens) != int(record["predicted_token_count"])
            ):
                raise RuntimeError(f"ScenePlan token count mismatch at ordinal {ordinal}")
            metrics = json.loads(str(record["metrics_json"]))
            if not isinstance(metrics, dict) or metrics.get("parse_rate") != 1.0:
                raise RuntimeError(f"plan prediction metrics invalid at ordinal {ordinal}")
            rows.append(
                PlanEvaluationRow(
                    ordinal=ordinal,
                    sample_id=str(sample_id),
                    template_id=str(template_id),
                    source_count=int(source_count),
                    room_type=str(room_type),
                    motion_signature=str(motion_signature),
                    manifest_latent_frames=int(latent_frames),
                    target_sceneplan_sha256=str(target_sha),
                    prediction_sceneplan_sha256=str(
                        record["prediction_sceneplan_sha256"]
                    ),
                    target_sceneplan_bytes=target_bytes,
                    prediction_sceneplan_bytes=prediction_bytes,
                    target_sceneplan=target_plan,
                    prediction_sceneplan=prediction_plan,
                    target_token_ids=tuple(int(value) for value in target_tokens),
                    prediction_token_ids=tuple(
                        int(value) for value in prediction_tokens
                    ),
                )
            )
    finally:
        connection.close()
    if len(rows) != int(expected_rows) or [row.ordinal for row in rows] != list(
        range(int(expected_rows))
    ):
        raise RuntimeError("test manifest does not cover exact requested ordinals")
    identity = {
        "path": str(plan_dir),
        "run_contract": str(run_path),
        "run_contract_sha256": _sha256_file(run_path),
        "run_contract_canonical_sha256": plan_contract_sha256,
        "summary": str(summary_path),
        "summary_sha256": _sha256_file(summary_path),
        "teacher_forced": str(teacher_path),
        "teacher_forced_sha256": _sha256_file(teacher_path),
        "prediction_shard_sha256": shard_hashes,
        "generation_ar_checkpoint": str(checkpoint),
        "generation_ar_checkpoint_sha256": str(run["checkpoint_sha256"]),
        "generation_ar_checkpoint_step": int(run["checkpoint_step"]),
        "test_manifest": str(manifest),
        "test_manifest_sha256": manifest_sha256,
        "test_source_index": str(source_index),
        "test_source_index_sha256": str(metadata["source_index_sha256"]),
        "codec_fingerprint": str(metadata["codec_fingerprint"]),
        "rows": int(expected_rows),
    }
    return rows, identity


def _validate_release_document(
    release: Mapping[str, Any], *, release_path: Path
) -> dict[str, Path]:
    expected_sampling = {
        "weights": "EMA DiT plus EMA trainable 4+4 conditioner",
        "sampler": "euler_rectified_flow",
        "steps": CANONICAL_P10_STEPS,
        "cfg_scale": 3.0,
        "cfg_rescale_phi": 0.4,
        "rescale_cfg": True,
        "apg_scale": 0.0,
        "negative_condition": "caption and structured controls both unknown",
        "raw_output": "float32 native WYZX/ACN/SN3D FOA",
    }
    if (
        release.get("schema") != "stable_audio_tools.p10_release_manifest"
        or int(release.get("schema_version", -1)) != 2
        or release.get("release_id") != "p10-sceneplan-dit-v11-step150000"
        or release.get("status") != "frozen_canonical"
        or release.get("immutable") is not True
        or release.get("executor_family") != P10_CANONICAL_EXECUTOR_FAMILY
        or release.get("canonical_inference") != expected_sampling
        or int(release.get("checkpoint", {}).get("step", -1))
        != P10_CANONICAL_CHECKPOINT_STEP
        or release.get("checkpoint", {}).get("sha256")
        != P10_CANONICAL_CHECKPOINT_SHA256
        or release.get("model_config", {}).get("file_sha256")
        != P10_CANONICAL_MODEL_CONFIG_SHA256
        or release.get("model_config", {}).get("resolved_sha256")
        != P10_V11_RESOLVED_CONFIG_SHA256
        or release.get("runtime_envelope", {}).get("audio_channels") != 4
        or release.get("runtime_envelope", {}).get("sample_rate_hz") != 44_100
        or release.get("runtime_envelope", {}).get("vae_hop_samples") != 1024
        or release.get("runtime_envelope", {}).get("max_latent_frames") != 648
        or release.get("runtime_envelope", {}).get("validated_motion_types")
        != ["static", "linear"]
    ):
        raise RuntimeError("P10 release document is not the frozen canonical contract")
    paths = {
        "release": release_path.resolve(strict=True),
        "model_config": Path(release["model_config"]["path"]).resolve(strict=True),
        "checkpoint": Path(release["checkpoint"]["path"]).resolve(strict=True),
        "vae_model_config": Path(
            release["pretransform"]["model_config"]["path"]
        ).resolve(strict=True),
        "vae_checkpoint": Path(
            release["pretransform"]["checkpoint"]["path"]
        ).resolve(strict=True),
        "capability_contract": Path(
            release["capability_contract"]["path"]
        ).resolve(strict=True),
        "training_freeze_manifest": Path(
            release["training_data_release"]["freeze_manifest"]
        ).resolve(strict=True),
    }
    if paths["model_config"] != Path(P10_CANONICAL_MODEL_CONFIG).resolve():
        raise RuntimeError("P10 release model-config path changed")
    if paths["checkpoint"] != Path(P10_CANONICAL_CHECKPOINT).resolve():
        raise RuntimeError("P10 release checkpoint path changed")
    return paths


def _assert_file(path: Path, expected_sha256: str, *, expected_bytes: Any = None) -> dict[str, Any]:
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != str(expected_sha256):
        raise RuntimeError(f"canonical file SHA256 mismatch: {path}")
    size = int(path.stat().st_size)
    if expected_bytes is not None and size != int(expected_bytes):
        raise RuntimeError(f"canonical file byte-size mismatch: {path}")
    return {"path": str(path), "bytes": size, "sha256": actual_sha256}


def _build_p10_identity(release_path: Path, codec_path: Path) -> dict[str, Any]:
    release_file = release_path.expanduser().resolve(strict=True)
    if _sha256_file(release_file) != P10_RELEASE_SHA256:
        raise RuntimeError("frozen P10 release manifest changed")
    release = _read_json_object(release_file, label="P10 release")
    paths = _validate_release_document(release, release_path=release_file)
    config = load_config(paths["model_config"])
    if _canonical_sha256(config) != P10_V11_RESOLVED_CONFIG_SHA256:
        raise RuntimeError("resolved canonical P10 config changed")
    prompt_configs = [
        value
        for value in config["model"]["conditioning"]["configs"]
        if value.get("id") == "prompt" and value.get("type") == "qwen_text"
    ]
    if len(prompt_configs) != 1:
        raise RuntimeError("canonical P10 prompt conditioner topology changed")
    qwen_path = Path(prompt_configs[0]["config"]["model_path"]).resolve(strict=True)
    if qwen_path != QWEN_PATH.resolve():
        raise RuntimeError("canonical P10 Qwen path changed")

    codec_root = codec_path.expanduser().resolve(strict=True)
    if codec_root != CODEC_PATH.resolve():
        raise RuntimeError("P10 closure codec path changed")
    codec = ModelScenePlanCodecV4(codec_root)
    if codec.fingerprint != GENERATION_AR_CODEC_FINGERPRINT:
        raise RuntimeError("P10 closure codec fingerprint changed")

    file_identities = {
        "release": _assert_file(paths["release"], P10_RELEASE_SHA256),
        "model_config": _assert_file(
            paths["model_config"], release["model_config"]["file_sha256"]
        ),
        "checkpoint": _assert_file(
            paths["checkpoint"],
            release["checkpoint"]["sha256"],
            expected_bytes=release["checkpoint"]["bytes"],
        ),
        "vae_model_config": _assert_file(
            paths["vae_model_config"],
            release["pretransform"]["model_config"]["sha256"],
        ),
        "vae_checkpoint": _assert_file(
            paths["vae_checkpoint"],
            release["pretransform"]["checkpoint"]["sha256"],
            expected_bytes=release["pretransform"]["checkpoint"]["bytes"],
        ),
        "capability_contract": _assert_file(
            paths["capability_contract"], release["capability_contract"]["sha256"]
        ),
        "training_freeze_manifest": _assert_file(
            paths["training_freeze_manifest"],
            release["training_data_release"]["freeze_manifest_sha256"],
        ),
    }
    codec_files = {
        name: _assert_file(codec_root / name, expected)
        for name, expected in CODEC_ARTIFACT_SHA256.items()
    }
    qwen_files = {
        name: _assert_file(qwen_path / name, expected)
        for name, expected in QWEN_CRITICAL_SHA256.items()
    }
    return {
        "release_id": str(release["release_id"]),
        "executor_family": str(release["executor_family"]),
        "resolved_model_config_sha256": P10_V11_RESOLVED_CONFIG_SHA256,
        "sampling": dict(release["canonical_inference"]),
        "runtime_envelope": dict(release["runtime_envelope"]),
        "files": file_identities,
        "codec": {
            "path": str(codec_root),
            "fingerprint": codec.fingerprint,
            "files": codec_files,
        },
        "qwen": {"path": str(qwen_path), "critical_files": qwen_files},
    }


def _source_identity() -> dict[str, dict[str, Any]]:
    sources = (
        Path(__file__).resolve(),
        REPO_ROOT / "scripts/t2a/eval/evaluate_sceneplan_p11_v4_p10_closure.py",
        REPO_ROOT / "scripts/t2a/eval/sceneplan_44_eval_common.py",
        REPO_ROOT / "scripts/t2a/eval/score_sceneplan_dit_p10_core.py",
        REPO_ROOT / "stable_audio_tools/configuration.py",
        REPO_ROOT / "stable_audio_tools/data/foa_intensity.py",
        REPO_ROOT / "stable_audio_tools/data/model_sceneplan.py",
        REPO_ROOT / "stable_audio_tools/data/model_sceneplan_codec.py",
        REPO_ROOT / "stable_audio_tools/data/model_sceneplan_codec_v3.py",
        REPO_ROOT / "stable_audio_tools/data/model_sceneplan_codec_v4.py",
        REPO_ROOT / "stable_audio_tools/data/sceneplan_p11_single_turn.py",
        REPO_ROOT
        / "stable_audio_tools/data/sceneplan_transfusion_generation_ar_dataset.py",
        REPO_ROOT
        / "stable_audio_tools/data/sceneplan_transfusion_generation_ar_evaluation.py",
        REPO_ROOT / "stable_audio_tools/inference/sceneplan_cot.py",
        REPO_ROOT / "stable_audio_tools/inference/sampling.py",
        REPO_ROOT / "stable_audio_tools/models/autoencoders.py",
        REPO_ROOT / "stable_audio_tools/models/autoencoders_4ch.py",
        REPO_ROOT / "stable_audio_tools/models/conditioners.py",
        REPO_ROOT / "stable_audio_tools/models/diffusion.py",
        REPO_ROOT / "stable_audio_tools/models/dit.py",
        REPO_ROOT / "stable_audio_tools/models/factory.py",
        REPO_ROOT / "stable_audio_tools/models/transformer.py",
        REPO_ROOT / "stable_audio_tools/models/utils.py",
        REPO_ROOT / "stable_audio_tools/training/factory.py",
    )
    return {
        str(path.relative_to(REPO_ROOT)): {
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
        for path in sources
    }


def _identity_file_map(
    run_contract: Mapping[str, Any], *, source_root: Path = REPO_ROOT
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    plan = run_contract["plan_evaluation"]
    for key in ("run_contract", "summary", "teacher_forced"):
        mapping[str(plan[key])] = str(plan[f"{key}_sha256"])
    mapping.update(
        {str(path): str(value) for path, value in plan["prediction_shard_sha256"].items()}
    )
    mapping[str(plan["generation_ar_checkpoint"])] = str(
        plan["generation_ar_checkpoint_sha256"]
    )
    mapping[str(plan["test_manifest"])] = str(plan["test_manifest_sha256"])
    mapping[str(plan["test_source_index"])] = str(plan["test_source_index_sha256"])
    p10 = run_contract["p10"]
    for identity in p10["files"].values():
        mapping[str(identity["path"])] = str(identity["sha256"])
    for identity in p10["codec"]["files"].values():
        mapping[str(identity["path"])] = str(identity["sha256"])
    for identity in p10["qwen"]["critical_files"].values():
        mapping[str(identity["path"])] = str(identity["sha256"])
    for relative, identity in run_contract["source"].items():
        mapping[str((source_root / relative).resolve())] = str(identity["sha256"])
    return mapping


def _verify_run_inputs_unchanged(
    run_contract: Mapping[str, Any], *, source_root: Path = REPO_ROOT
) -> dict[str, Any]:
    checked = 0
    for raw_path, expected in _identity_file_map(
        run_contract, source_root=source_root
    ).items():
        path = Path(raw_path).resolve(strict=True)
        if _sha256_file(path) != expected:
            raise RuntimeError(f"audio closure input changed during run: {path}")
        checked += 1
    return {"files_checked": checked, "all_sha256_unchanged": True}


def _open_audio_shard(
    path: Path, *, rank: int, world_size: int, run_contract_sha256: str
) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS audio_results (
            ordinal INTEGER PRIMARY KEY,
            sample_id TEXT NOT NULL,
            template_id TEXT NOT NULL,
            source_count INTEGER NOT NULL,
            plan_exact INTEGER NOT NULL,
            exact_anchor INTEGER NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            target_sceneplan_sha256 TEXT NOT NULL,
            prediction_sceneplan_sha256 TEXT NOT NULL,
            target_bundle_sha256 TEXT,
            prediction_bundle_sha256 TEXT,
            target_render_input_sha256 TEXT,
            prediction_render_input_sha256 TEXT,
            render_seed INTEGER NOT NULL,
            length_group TEXT,
            target_model_num_samples INTEGER,
            prediction_model_num_samples INTEGER,
            target_latent_frames INTEGER,
            prediction_latent_frames INTEGER,
            target_foa_sha256 TEXT,
            prediction_foa_sha256 TEXT,
            target_repeat_foa_sha256 TEXT,
            metrics_json TEXT NOT NULL,
            render_sec REAL NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS audio_attempt_history (
            attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ordinal INTEGER NOT NULL,
            sample_id TEXT NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            metrics_json TEXT NOT NULL,
            render_sec REAL NOT NULL,
            archived_unix REAL NOT NULL
        )
        """
    )
    columns = tuple(
        str(row[1]) for row in connection.execute("PRAGMA table_info(audio_results)")
    )
    if columns != AUDIO_RESULT_COLUMNS:
        raise RuntimeError(f"audio result schema mismatch: {path}")
    expected = {
        "contract": AUDIO_CLOSURE_CONTRACT,
        "rank": str(rank),
        "world_size": str(world_size),
        "run_contract_sha256": str(run_contract_sha256),
    }
    existing = dict(connection.execute("SELECT key,value FROM metadata"))
    if existing:
        for key, value in expected.items():
            if existing.get(key) != value:
                raise RuntimeError(f"audio shard contract mismatch: {path}: {key}")
    else:
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)", expected.items()
        )
        connection.execute(
            "INSERT INTO metadata(key,value) VALUES ('status','IN_PROGRESS')"
        )
    # Successful/proven rows are durable resume points.  A system_error may be
    # a transient CUDA/OOM/runtime fault, so preserve the failed attempt and
    # remove it from the completed set before resuming.
    connection.execute(
        """
        INSERT INTO audio_attempt_history(
            ordinal,sample_id,status,error,metrics_json,render_sec,archived_unix
        )
        SELECT ordinal,sample_id,status,error,metrics_json,render_sec,?
        FROM audio_results WHERE status = 'system_error'
        """,
        (time.time(),),
    )
    connection.execute("DELETE FROM audio_results WHERE status = 'system_error'")
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES ('status','IN_PROGRESS')"
    )
    connection.commit()
    return connection


def _finalize_bundle(row: PlanEvaluationRow, *, target: bool, codec, tokenizer):
    tokens = row.target_token_ids if target else row.prediction_token_ids
    bundle = finalize_sceneplan_for_p10(
        codec,
        tokens,
        tokenizer=tokenizer,
        task=P11Task.GENERATION,
        sample_id=row.sample_id,
    )
    expected = row.target_sceneplan_bytes if target else row.prediction_sceneplan_bytes
    if _canonical_json_bytes(bundle.sceneplan) != expected:
        role = "target" if target else "prediction"
        raise RuntimeError(
            f"independent P10 finalize changed {role} plan at ordinal {row.ordinal}"
        )
    bundle.assert_external_p10_boundary()
    return bundle


def _base_result(row: PlanEvaluationRow, *, exact_anchor: bool, seed: int) -> dict[str, Any]:
    return {
        "ordinal": int(row.ordinal),
        "sample_id": str(row.sample_id),
        "template_id": str(row.template_id),
        "source_count": int(row.source_count),
        "plan_exact": int(row.plan_exact),
        "exact_anchor": int(exact_anchor),
        "status": "system_error",
        "error": None,
        "target_sceneplan_sha256": str(row.target_sceneplan_sha256),
        "prediction_sceneplan_sha256": str(row.prediction_sceneplan_sha256),
        "target_bundle_sha256": None,
        "prediction_bundle_sha256": None,
        "target_render_input_sha256": None,
        "prediction_render_input_sha256": None,
        "render_seed": int(seed),
        "length_group": None,
        "target_model_num_samples": None,
        "prediction_model_num_samples": None,
        "target_latent_frames": None,
        "prediction_latent_frames": None,
        "target_foa_sha256": None,
        "prediction_foa_sha256": None,
        "target_repeat_foa_sha256": None,
        "metrics_json": "{}",
        "render_sec": 0.0,
    }


def _audio_integrity(audio: torch.Tensor, bundle: Any) -> dict[str, Any]:
    qc = audio_qc(audio)
    return {
        "qc": qc,
        "shape_exact": tuple(audio.shape) == (4, int(bundle.model_num_samples)),
        "finite_foa": qc.get("finite") is True and qc.get("channels") == 4,
    }


def _evaluate_row(
    row: PlanEvaluationRow,
    *,
    exact_anchor: bool,
    seed: int,
    codec,
    tokenizer,
    executor,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = _base_result(row, exact_anchor=exact_anchor, seed=seed)
    try:
        target_bundle = _finalize_bundle(row, target=True, codec=codec, tokenizer=tokenizer)
        prediction_bundle = _finalize_bundle(
            row, target=False, codec=codec, tokenizer=tokenizer
        )
        target_fp = _bundle_fingerprints(target_bundle)
        prediction_fp = _bundle_fingerprints(prediction_bundle)
        result.update(
            {
                "target_bundle_sha256": target_fp["bundle_sha256"],
                "prediction_bundle_sha256": prediction_fp["bundle_sha256"],
                "target_render_input_sha256": target_fp["render_input_sha256"],
                "prediction_render_input_sha256": prediction_fp["render_input_sha256"],
                "target_model_num_samples": int(target_bundle.model_num_samples),
                "prediction_model_num_samples": int(
                    prediction_bundle.model_num_samples
                ),
                "target_latent_frames": int(target_bundle.latent_frames_valid),
                "prediction_latent_frames": int(
                    prediction_bundle.latent_frames_valid
                ),
            }
        )
        result["length_group"] = _duration_group(
            target_samples=int(target_bundle.model_num_samples),
            prediction_samples=int(prediction_bundle.model_num_samples),
            target_frames=int(target_bundle.latent_frames_valid),
            prediction_frames=int(prediction_bundle.latent_frames_valid),
        )
        if int(target_bundle.latent_frames_valid) != int(row.manifest_latent_frames):
            raise RuntimeError("target finalize disagrees with manifest latent length")

        if row.plan_exact:
            if (
                target_fp["bundle_sha256"] != prediction_fp["bundle_sha256"]
                or target_fp["render_input_sha256"]
                != prediction_fp["render_input_sha256"]
            ):
                raise RuntimeError(
                    "canonical-equal plans produced different finalized P10 inputs"
                )
            metrics: dict[str, Any] = {
                "proof": "canonical_bytes_plus_independent_finalize_render_input_v1",
                "canonical_plan_bytes_equal": True,
                "bundle_fingerprint_equal": True,
                "render_input_fingerprint_equal": True,
                "audio_equivalence_inferred": not exact_anchor,
                "target_repeat_waveform_exact": None,
                "prediction_target_waveform_exact": None,
            }
            if exact_anchor:
                target_audio = executor.render(target_bundle, seed=seed)
                target_repeat = executor.render(target_bundle, seed=seed)
                prediction_audio = executor.render(prediction_bundle, seed=seed)
                result.update(
                    {
                        "target_foa_sha256": _tensor_sha256(target_audio),
                        "target_repeat_foa_sha256": _tensor_sha256(target_repeat),
                        "prediction_foa_sha256": _tensor_sha256(prediction_audio),
                    }
                )
                metrics.update(
                    {
                        "audio_equivalence_inferred": False,
                        "target": _audio_integrity(target_audio, target_bundle),
                        "target_repeat": _audio_integrity(
                            target_repeat, target_bundle
                        ),
                        "prediction": _audio_integrity(
                            prediction_audio, prediction_bundle
                        ),
                        "target_repeat_waveform_exact": bool(
                            torch.equal(target_audio, target_repeat)
                        ),
                        "prediction_target_waveform_exact": bool(
                            torch.equal(prediction_audio, target_audio)
                        ),
                        "predicted_target": _pair_metrics(
                            prediction_audio, target_audio
                        ),
                    }
                )
                if not all(
                    metrics[key]["finite_foa"] is True
                    and metrics[key]["shape_exact"] is True
                    for key in ("target", "target_repeat", "prediction")
                ):
                    raise RuntimeError(
                        "exact anchor P10 render violated finite/shape contract"
                    )
                result["status"] = "exact_anchor_rendered"
                del target_audio, target_repeat, prediction_audio
            else:
                result["status"] = "exact_input_equivalent"
            result["metrics_json"] = json.dumps(
                metrics,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            return result

        target_audio = executor.render(target_bundle, seed=seed)
        prediction_audio = executor.render(prediction_bundle, seed=seed)
        target_integrity = _audio_integrity(target_audio, target_bundle)
        prediction_integrity = _audio_integrity(
            prediction_audio, prediction_bundle
        )
        if not all(
            integrity["finite_foa"] is True and integrity["shape_exact"] is True
            for integrity in (target_integrity, prediction_integrity)
        ):
            raise RuntimeError("nonexact P10 render violated finite/shape contract")
        target_samples = int(target_bundle.model_num_samples)
        prediction_samples = int(prediction_bundle.model_num_samples)
        target_frames = int(target_bundle.latent_frames_valid)
        prediction_frames = int(prediction_bundle.latent_frames_valid)
        fitted_prediction = _fit_length(prediction_audio, target_samples)
        metrics = {
            "proof": "same_gpu_same_seed_canonical_p10_pair_v1",
            "target": target_integrity,
            "prediction": prediction_integrity,
            "predicted_target": _pair_metrics(prediction_audio, target_audio),
            "predicted_target_spatial": _spatial_pair_metrics(
                fitted_prediction, target_audio
            ),
            "predicted_target_foa": _paired_doa_metrics(
                fitted_prediction,
                target_audio,
                dict(target_bundle.sceneplan),
                model_num_samples=target_samples,
                latent_frames=target_frames,
            ),
            "prediction_plan_foa": _doa_metrics(
                prediction_audio,
                dict(prediction_bundle.sceneplan),
                model_num_samples=prediction_samples,
                latent_frames=prediction_frames,
            ),
            "target_plan_foa": _doa_metrics(
                target_audio,
                dict(target_bundle.sceneplan),
                model_num_samples=target_samples,
                latent_frames=target_frames,
            ),
            "prediction_plan_activity": _activity_metrics(
                prediction_audio,
                dict(prediction_bundle.sceneplan),
                model_num_samples=prediction_samples,
                latent_frames=prediction_frames,
            ),
            "target_plan_activity": _activity_metrics(
                target_audio,
                dict(target_bundle.sceneplan),
                model_num_samples=target_samples,
                latent_frames=target_frames,
            ),
        }
        result.update(
            {
                "status": "nonexact_pair_rendered",
                "target_foa_sha256": _tensor_sha256(target_audio),
                "prediction_foa_sha256": _tensor_sha256(prediction_audio),
                "metrics_json": json.dumps(
                    metrics,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            }
        )
        del target_audio, prediction_audio, fitted_prediction
        return result
    except Exception as exc:
        result["status"] = "system_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["metrics_json"] = json.dumps(
            {"system_error": result["error"]},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return result
    finally:
        result["render_sec"] = float(time.perf_counter() - started)


def _insert_audio_result(connection: sqlite3.Connection, result: Mapping[str, Any]) -> None:
    connection.execute(
        "INSERT INTO audio_results(" + ",".join(AUDIO_RESULT_COLUMNS) + ") "
        "VALUES (" + ",".join("?" for _ in AUDIO_RESULT_COLUMNS) + ")",
        tuple(result[column] for column in AUDIO_RESULT_COLUMNS),
    )


def _mean(values: Iterable[Any]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(sum(finite) / len(finite)) if finite else None


def _metric(metrics: Mapping[str, Any], *path: str) -> Any:
    value: Any = metrics
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _quality_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    measured = [row for row in rows if row["status"] == "nonexact_pair_rendered"]
    return {
        "rows": len(rows),
        "measured_nonexact_rows": len(measured),
        "waveform_mae": _mean(
            _metric(row["metrics"], "predicted_target", "waveform_mae")
            for row in measured
        ),
        "waveform_rmse": _mean(
            _metric(row["metrics"], "predicted_target", "waveform_rmse")
            for row in measured
        ),
        "waveform_cosine": _mean(
            _metric(row["metrics"], "predicted_target", "waveform_cosine")
            for row in measured
        ),
        "w_log_magnitude_l1": _mean(
            _metric(
                row["metrics"],
                "predicted_target",
                "spectral",
                "w_log_magnitude_l1",
            )
            for row in measured
        ),
        "w_magnitude_cosine": _mean(
            _metric(
                row["metrics"],
                "predicted_target",
                "spectral",
                "w_magnitude_cosine",
            )
            for row in measured
        ),
        "w_spectral_convergence": _mean(
            _metric(
                row["metrics"],
                "predicted_target",
                "spectral",
                "w_spectral_convergence",
            )
            for row in measured
        ),
        "paired_foa_spherical_error_mean_deg": _mean(
            _metric(
                row["metrics"],
                "predicted_target_foa",
                "spherical_error_mean_deg",
            )
            for row in measured
        ),
        "spatial_angular_error_mean_deg": _mean(
            _metric(
                row["metrics"],
                "predicted_target_spatial",
                "angular_error_mean_deg",
            )
            for row in measured
        ),
        "spatial_direction_cosine": _mean(
            _metric(
                row["metrics"],
                "predicted_target_spatial",
                "direction_cosine",
            )
            for row in measured
        ),
        "spatial_diffuseness_mae": _mean(
            _metric(
                row["metrics"],
                "predicted_target_spatial",
                "diffuseness_mae",
            )
            for row in measured
        ),
        "prediction_plan_spherical_error_mean_deg": _mean(
            _metric(
                row["metrics"],
                "prediction_plan_foa",
                "spherical_error_mean_deg",
            )
            for row in measured
        ),
        "target_plan_spherical_error_mean_deg": _mean(
            _metric(
                row["metrics"],
                "target_plan_foa",
                "spherical_error_mean_deg",
            )
            for row in measured
        ),
        "prediction_plan_activity_temporal_iou": _mean(
            _metric(
                row["metrics"],
                "prediction_plan_activity",
                "temporal_iou",
            )
            for row in measured
        ),
        "prediction_plan_activity_onset_abs_error_sec": _mean(
            _metric(
                row["metrics"],
                "prediction_plan_activity",
                "onset_abs_error_sec",
            )
            for row in measured
        ),
        "prediction_plan_activity_offset_abs_error_sec": _mean(
            _metric(
                row["metrics"],
                "prediction_plan_activity",
                "offset_abs_error_sec",
            )
            for row in measured
        ),
    }


def _summarize_audio_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_rows: int,
    expected_anchor_ordinals: Sequence[int],
    inputs_unchanged: bool,
) -> dict[str, Any]:
    ordinals = [int(row["ordinal"]) for row in records]
    coverage_exact = sorted(ordinals) == list(range(int(expected_rows))) and len(
        set(ordinals)
    ) == len(ordinals)
    anchor_set = set(int(value) for value in expected_anchor_ordinals)
    stored_anchors = {int(row["ordinal"]) for row in records if row["exact_anchor"]}
    status_counts = dict(sorted(Counter(str(row["status"]) for row in records).items()))
    exact = [row for row in records if row["plan_exact"]]
    nonexact = [row for row in records if not row["plan_exact"]]
    anchors = [row for row in records if row["exact_anchor"]]

    exact_inputs_equal = all(
        row["target_bundle_sha256"] == row["prediction_bundle_sha256"]
        and row["target_render_input_sha256"]
        == row["prediction_render_input_sha256"]
        for row in exact
    )
    anchors_repeat_exact = all(
        _metric(row["metrics"], "target_repeat_waveform_exact") is True
        for row in anchors
    )
    anchors_prediction_exact = all(
        _metric(row["metrics"], "prediction_target_waveform_exact") is True
        for row in anchors
    )
    anchor_hashes_exact = all(
        row["target_foa_sha256"] is not None
        and row["target_foa_sha256"] == row["target_repeat_foa_sha256"]
        and row["target_foa_sha256"] == row["prediction_foa_sha256"]
        for row in anchors
    )
    nonexact_integrity = all(
        row["status"] == "nonexact_pair_rendered"
        and _metric(row["metrics"], "target", "finite_foa") is True
        and _metric(row["metrics"], "prediction", "finite_foa") is True
        and _metric(row["metrics"], "target", "shape_exact") is True
        and _metric(row["metrics"], "prediction", "shape_exact") is True
        for row in nonexact
    )
    gates = {
        "exact_ordinal_coverage": coverage_exact,
        "no_system_error_rows": status_counts.get("system_error", 0) == 0,
        "all_row_statuses_match_plan_and_anchor_class": all(
            row["status"]
            == (
                "exact_anchor_rendered"
                if row["plan_exact"] and row["exact_anchor"]
                else (
                    "exact_input_equivalent"
                    if row["plan_exact"]
                    else "nonexact_pair_rendered"
                )
            )
            for row in records
        ),
        "all_exact_rows_have_equal_independent_finalize_inputs": exact_inputs_equal,
        "exact_anchor_set_complete": stored_anchors == anchor_set,
        "all_exact_anchors_are_exact_plan_rows": all(
            row["plan_exact"] for row in anchors
        ),
        "all_exact_anchor_target_repeats_bit_exact": anchors_repeat_exact,
        "all_exact_anchor_prediction_target_pairs_bit_exact": anchors_prediction_exact,
        "all_exact_anchor_tensor_hashes_bit_exact": anchor_hashes_exact,
        "all_nonexact_pairs_finite_four_channel_and_own_length": nonexact_integrity,
        "all_input_hashes_unchanged_after_render": bool(inputs_unchanged),
    }
    broad_groups = {
        "all_nonexact": nonexact,
        "same_latent_length": [
            row
            for row in nonexact
            if row["target_latent_frames"] == row["prediction_latent_frames"]
        ],
        "duration_mismatch": [
            row
            for row in nonexact
            if row["target_model_num_samples"]
            != row["prediction_model_num_samples"]
        ],
        "latent_length_mismatch": [
            row
            for row in nonexact
            if row["target_latent_frames"] != row["prediction_latent_frames"]
        ],
    }
    by_length_group = {
        group: _quality_aggregate(
            [row for row in nonexact if row["length_group"] == group]
        )
        for group in (
            "same_duration_and_latent_length",
            "same_latent_length_duration_mismatch",
            "duration_and_latent_length_mismatch",
        )
    }
    nonexact_waveform_exact = sum(
        _metric(row["metrics"], "predicted_target", "waveform_exact") is True
        for row in nonexact
    )

    def full_distance(*path: str) -> float | None:
        values = [_metric(row["metrics"], *path) for row in nonexact]
        if any(value is None or not math.isfinite(float(value)) for value in values):
            return None
        return float(sum(float(value) for value in values) / int(expected_rows))

    status = "PASS" if all(gates.values()) else "FAIL"
    return {
        "status": status,
        "integrity_gates": gates,
        "coverage": {
            "expected_rows": int(expected_rows),
            "rows": len(records),
            "exact_plan_rows": len(exact),
            "nonexact_plan_rows": len(nonexact),
            "exact_anchor_rows": len(anchors),
            "exact_inferred_rows": len(exact) - len(anchors),
            "status_counts": status_counts,
        },
        "exact_plan_audio_equivalence": {
            "proof_contract": (
                "canonical_bytes_plus_independent_finalize_render_input_plus_"
                "stratified_runtime_determinism_anchors_v1"
            ),
            "anchor_ordinals": sorted(anchor_set),
            "all_exact_rows_equivalent": bool(
                exact_inputs_equal and anchors_repeat_exact and anchors_prediction_exact
            ),
        },
        "quality_metrics": {
            name: _quality_aggregate(values)
            for name, values in broad_groups.items()
        },
        "quality_metrics_by_length_group": by_length_group,
        "full_test_pair_consequences": {
            "contract": (
                "exact-plan rows contribute proven zero pair distance; nonexact "
                "rows contribute measured same-seed P10 pair distance"
            ),
            "waveform_exact_rate": (
                (len(exact) + nonexact_waveform_exact) / int(expected_rows)
                if int(expected_rows) > 0
                else None
            ),
            "waveform_mae": full_distance("predicted_target", "waveform_mae"),
            "waveform_rmse": full_distance("predicted_target", "waveform_rmse"),
            "w_log_magnitude_l1": full_distance(
                "predicted_target", "spectral", "w_log_magnitude_l1"
            ),
            "w_spectral_convergence": full_distance(
                "predicted_target", "spectral", "w_spectral_convergence"
            ),
        },
    }


def _load_audio_records(
    output_dir: Path, *, world_size: int, expected_rows: int, run_contract_sha256: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rank in range(int(world_size)):
        path = (output_dir / "audio_metrics" / f"rank_{rank:03d}.sqlite").resolve(
            strict=True
        )
        if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm")):
            raise RuntimeError(f"audio result shard has live SQLite sidecars: {path}")
        connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        try:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError(f"audio result shard failed integrity: {path}")
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            expected = {
                "contract": AUDIO_CLOSURE_CONTRACT,
                "rank": str(rank),
                "world_size": str(world_size),
                "run_contract_sha256": run_contract_sha256,
                "status": "COMPLETE",
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise RuntimeError(f"audio result shard metadata mismatch: {path}")
            if _prediction_columns_for_audio(connection) != AUDIO_RESULT_COLUMNS:
                raise RuntimeError(f"audio result columns changed: {path}")
            shard_rows = 0
            for values in connection.execute(
                "SELECT " + ",".join(AUDIO_RESULT_COLUMNS) + " "
                "FROM audio_results ORDER BY ordinal"
            ):
                row = dict(zip(AUDIO_RESULT_COLUMNS, values))
                if int(row["ordinal"]) % int(world_size) != rank:
                    raise RuntimeError(
                        f"audio result rank ownership mismatch: {path}: "
                        f"ordinal {row['ordinal']}"
                    )
                row["plan_exact"] = bool(row["plan_exact"])
                row["exact_anchor"] = bool(row["exact_anchor"])
                row["metrics"] = json.loads(row.pop("metrics_json"))
                records.append(row)
                shard_rows += 1
            if int(metadata.get("rows", -1)) != shard_rows:
                raise RuntimeError(f"audio result shard row count mismatch: {path}")
        finally:
            connection.close()
    if len(records) != int(expected_rows):
        raise RuntimeError("audio result shards have incomplete coverage")
    return records


def _prediction_columns_for_audio(connection: sqlite3.Connection) -> tuple[str, ...]:
    return tuple(
        str(row[1]) for row in connection.execute("PRAGMA table_info(audio_results)")
    )


def validate_completed_audio_output(
    output_dir: Path,
    *,
    plan_evaluation_dir: Path,
    test_manifest: Path,
    expected_rows: int = EXPECTED_ROWS,
    expected_world_size: int = EXPECTED_WORLD_SIZE,
    source_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Re-prove a completed audio closure without mutating any artifact."""

    output = output_dir.expanduser().resolve(strict=True)
    plan_dir = plan_evaluation_dir.expanduser().resolve(strict=True)
    manifest = test_manifest.expanduser().resolve(strict=True)
    frozen_source_root = source_root.expanduser().resolve(strict=True)
    contract_path = (output / "RUN_CONTRACT.json").resolve(strict=True)
    summary_path = (output / "SUMMARY.json").resolve(strict=True)
    run_contract = _read_json_object(contract_path, label="audio run contract")
    report = _read_json_object(summary_path, label="audio summary")
    plan_identity = dict(run_contract.get("plan_evaluation") or {})
    anchor_ordinals = [
        int(value) for value in list(run_contract.get("exact_anchor_ordinals") or ())
    ]
    if (
        run_contract.get("schema") != AUDIO_CLOSURE_SCHEMA + ".run_contract"
        or int(run_contract.get("schema_version", -1)) != 1
        or run_contract.get("contract") != AUDIO_CLOSURE_CONTRACT
        or int(run_contract.get("rows", -1)) != int(expected_rows)
        or int(run_contract.get("world_size", -1)) != int(expected_world_size)
        or str(run_contract.get("cuda_visible_devices", "")).replace(" ", "")
        != EXPECTED_VISIBLE_DEVICES
        or int(run_contract.get("root_seed", -1)) != CANONICAL_ROOT_SEED
        or run_contract.get("same_seed_pairing") is not True
        or int(run_contract.get("requested_exact_anchors", -1))
        != DEFAULT_EXACT_ANCHORS
        or run_contract.get("retained_audio") is not False
        or Path(str(plan_identity.get("path", ""))).resolve() != plan_dir
        or Path(str(plan_identity.get("test_manifest", ""))).resolve() != manifest
        or plan_identity.get("test_manifest_sha256") != _sha256_file(manifest)
        or len(anchor_ordinals) != len(set(anchor_ordinals))
        or anchor_ordinals != sorted(anchor_ordinals)
        or any(value < 0 or value >= int(expected_rows) for value in anchor_ordinals)
    ):
        raise RuntimeError("completed audio run contract identity mismatch")

    contract_file_sha256 = _sha256_file(contract_path)
    contract_canonical_sha256 = _canonical_sha256(run_contract)
    checkpoint = Path(
        str(plan_identity.get("generation_ar_checkpoint", ""))
    ).resolve(strict=True)
    checkpoint_sha256 = str(
        plan_identity.get("generation_ar_checkpoint_sha256", "")
    )
    report_without_self = dict(report)
    claimed_report_sha256 = report_without_self.pop(
        "report_sha256_without_self", None
    )
    if (
        report.get("schema") != AUDIO_CLOSURE_SCHEMA
        or int(report.get("schema_version", -1)) != 1
        or report.get("status") != "PASS"
        or report.get("contract") != AUDIO_CLOSURE_CONTRACT
        or Path(str(report.get("run_contract", ""))).resolve() != contract_path
        or report.get("run_contract_sha256") != contract_file_sha256
        or report.get("run_contract_canonical_sha256")
        != contract_canonical_sha256
        or Path(str(report.get("checkpoint", ""))).resolve() != checkpoint
        or report.get("checkpoint_sha256") != checkpoint_sha256
        or int(report.get("checkpoint_step", -1))
        != int(plan_identity.get("generation_ar_checkpoint_step", -2))
        or _sha256_file(checkpoint) != checkpoint_sha256
        or report.get("p10_release_id")
        != dict(run_contract.get("p10") or {}).get("release_id")
        or report.get("sampling")
        != dict(run_contract.get("p10") or {}).get("sampling")
        or claimed_report_sha256 != _canonical_sha256(report_without_self)
    ):
        raise RuntimeError("completed audio summary identity mismatch")

    unchanged = _verify_run_inputs_unchanged(
        run_contract, source_root=frozen_source_root
    )
    records = _load_audio_records(
        output,
        world_size=int(expected_world_size),
        expected_rows=int(expected_rows),
        run_contract_sha256=contract_canonical_sha256,
    )
    exact_rows = sum(bool(row["plan_exact"]) for row in records)
    if len(anchor_ordinals) != min(DEFAULT_EXACT_ANCHORS, exact_rows):
        raise RuntimeError("completed audio exact-anchor count mismatch")
    recomputed = _summarize_audio_records(
        records,
        expected_rows=int(expected_rows),
        expected_anchor_ordinals=anchor_ordinals,
        inputs_unchanged=bool(unchanged["all_sha256_unchanged"]),
    )
    if any(report.get(key) != value for key, value in recomputed.items()):
        raise RuntimeError("completed audio summary does not match stored shards")
    if report.get("input_reverification") != unchanged:
        raise RuntimeError("completed audio input reverification mismatch")
    return report


def _executor_from_identity(identity: Mapping[str, Any], *, device: torch.device):
    files = identity["files"]
    sampling = identity["sampling"]
    return P10ScenePlanDiTExecutor.from_checkpoints(
        model_config_path=files["model_config"]["path"],
        checkpoint_path=files["checkpoint"]["path"],
        vae_checkpoint_path=files["vae_checkpoint"]["path"],
        device=device,
        steps=CANONICAL_P10_STEPS,
        cfg_scale=float(sampling["cfg_scale"]),
        rescale_cfg=bool(sampling["rescale_cfg"]),
        cfg_rescale_phi=float(sampling["cfg_rescale_phi"]),
        apg_scale=float(sampling["apg_scale"]),
    )


def main() -> int:
    args = _parse_args()
    if args.verify_only:
        source_root = (
            args.source_root
            if args.source_root is not None
            else REPO_ROOT
        )
        report = validate_completed_audio_output(
            args.output_dir,
            plan_evaluation_dir=args.plan_evaluation_dir,
            test_manifest=args.test_manifest,
            source_root=source_root,
        )
        print(
            json.dumps(
                {
                    "event": "p10_audio_evaluation_reverified",
                    "status": report["status"],
                    "summary": str(
                        args.output_dir.expanduser().resolve(strict=True)
                        / "SUMMARY.json"
                    ),
                    "coverage": report["coverage"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    if args.source_root is not None:
        raise ValueError("--source-root is only valid with --verify-only")
    if args.exact_anchors != DEFAULT_EXACT_ANCHORS:
        raise ValueError(
            f"frozen P10 closure requires exactly {DEFAULT_EXACT_ANCHORS} "
            "requested exact anchors (or all exact rows when fewer exist)"
        )
    if args.log_every <= 0:
        raise ValueError("log interval must be positive")
    if args.seed != CANONICAL_ROOT_SEED:
        raise ValueError("frozen P10 closure permits only root seed 42")
    rank, local_rank, world_size, device, completion_group = _distributed()
    _configure_determinism(args.seed, rank=rank)
    plan_dir = args.plan_evaluation_dir.expanduser().resolve(strict=True)
    test_manifest = args.test_manifest.expanduser().resolve(strict=True)
    release_path = args.p10_release.expanduser().resolve(strict=True)
    codec_path = args.codec_path.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)

    # Rank zero performs all expensive immutable-artifact hashing once.  Every
    # rank independently parses and verifies row-level bytes before rendering.
    initialization: list[Any] = [None]
    if rank == 0:
        rows_zero, plan_identity = _load_verified_plan_rows(
            plan_dir, test_manifest, expected_rows=EXPECTED_ROWS
        )
        anchors = _select_exact_anchors(
            rows_zero, count=int(args.exact_anchors), seed=int(args.seed)
        )
        p10_identity = _build_p10_identity(release_path, codec_path)
        source_identity = _source_identity()
        run_contract = {
            "schema": AUDIO_CLOSURE_SCHEMA + ".run_contract",
            "schema_version": 1,
            "contract": AUDIO_CLOSURE_CONTRACT,
            "rows": EXPECTED_ROWS,
            "world_size": world_size,
            "cuda_visible_devices": EXPECTED_VISIBLE_DEVICES,
            "root_seed": CANONICAL_ROOT_SEED,
            "same_seed_pairing": True,
            "exact_plan_policy": (
                "canonical_bytes_and_independent_finalize_render_input_equivalence"
            ),
            "exact_anchor_policy": (
                "deterministic_round_robin_source_template_room_motion_v1"
            ),
            "requested_exact_anchors": int(args.exact_anchors),
            "exact_anchor_ordinals": list(anchors),
            "plan_evaluation": plan_identity,
            "p10": p10_identity,
            "source": source_identity,
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "deterministic_algorithms": True,
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "cudnn_benchmark": False,
                "completion_barrier_backend": "gloo",
                "completion_barrier_timeout_seconds": (
                    COMPLETION_BARRIER_TIMEOUT_SECONDS
                ),
            },
            "retained_audio": False,
            "measured": [
                "waveform_pair",
                "w_channel_spectral_pair",
                "foa_active_intensity_pair",
                "plan_foa_doa",
                "plan_activity",
            ],
            "not_measured": [
                "absolute_semantic_audio_alignment_CLAP",
                "speech_transcript_accuracy_Whisper_ASR",
            ],
        }
        contract_sha = _canonical_sha256(run_contract)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "audio_metrics").mkdir(exist_ok=True)
        contract_path = output_dir / "RUN_CONTRACT.json"
        if contract_path.exists():
            existing = _read_json_object(contract_path, label="audio run contract")
            if existing != run_contract:
                raise RuntimeError("audio output directory contract mismatch")
        else:
            _atomic_json(contract_path, run_contract)
        initialization[0] = {
            "run_contract": run_contract,
            "run_contract_sha256": contract_sha,
        }
    dist.broadcast_object_list(initialization, src=0, device=device)
    payload = initialization[0]
    if not isinstance(payload, Mapping):
        raise RuntimeError("rank zero did not publish the audio run contract")
    run_contract = dict(payload["run_contract"])
    run_contract_sha256 = str(payload["run_contract_sha256"])
    if _canonical_sha256(run_contract) != run_contract_sha256:
        raise RuntimeError("broadcast audio run contract hash mismatch")
    dist.barrier()

    rows, local_plan_identity = _load_verified_plan_rows(
        plan_dir,
        test_manifest,
        expected_rows=EXPECTED_ROWS,
        verify_source_hashes=False,
    )
    for key in (
        "run_contract_sha256",
        "summary_sha256",
        "teacher_forced_sha256",
        "prediction_shard_sha256",
        "test_manifest_sha256",
        "test_source_index_sha256",
    ):
        if local_plan_identity[key] != run_contract["plan_evaluation"][key]:
            raise RuntimeError(f"rank-local plan evaluation identity mismatch: {key}")
    anchor_set = set(int(value) for value in run_contract["exact_anchor_ordinals"])

    codec = ModelScenePlanCodecV4(codec_path)
    if codec.fingerprint != run_contract["p10"]["codec"]["fingerprint"]:
        raise RuntimeError("rank-local codec fingerprint mismatch")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        run_contract["p10"]["qwen"]["path"],
        local_files_only=True,
        use_fast=True,
    )
    executor = _executor_from_identity(run_contract["p10"], device=device)

    assigned = [row for row in rows if row.ordinal % world_size == rank]
    shard_path = output_dir / "audio_metrics" / f"rank_{rank:03d}.sqlite"
    shard = _open_audio_shard(
        shard_path,
        rank=rank,
        world_size=world_size,
        run_contract_sha256=run_contract_sha256,
    )
    completed = {
        int(ordinal): (
            str(sample_id),
            str(target_sha),
            str(prediction_sha),
            bool(plan_exact),
            bool(exact_anchor),
        )
        for ordinal, sample_id, target_sha, prediction_sha, plan_exact, exact_anchor
        in shard.execute(
            """
            SELECT ordinal,sample_id,target_sceneplan_sha256,
                   prediction_sceneplan_sha256,plan_exact,exact_anchor
            FROM audio_results
            """
        )
    }
    started = time.perf_counter()
    newly_processed = 0
    for row in assigned:
        expected_identity = (
            row.sample_id,
            row.target_sceneplan_sha256,
            row.prediction_sceneplan_sha256,
            row.plan_exact,
            row.ordinal in anchor_set,
        )
        if row.ordinal in completed:
            if completed[row.ordinal] != expected_identity:
                raise RuntimeError(f"audio resume identity mismatch {row.ordinal}")
            continue
        result = _evaluate_row(
            row,
            exact_anchor=row.ordinal in anchor_set,
            seed=_stable_seed(args.seed, row.sample_id),
            codec=codec,
            tokenizer=tokenizer,
            executor=executor,
        )
        _insert_audio_result(shard, result)
        shard.commit()
        newly_processed += 1
        if newly_processed % int(args.log_every) == 0:
            stored = int(
                shard.execute("SELECT COUNT(*) FROM audio_results").fetchone()[0]
            )
            print(
                json.dumps(
                    {
                        "event": "p10_audio_progress",
                        "rank": rank,
                        "local_rank": local_rank,
                        "rows": stored,
                        "assigned_rows": len(assigned),
                        "elapsed_sec": round(time.perf_counter() - started, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if result["status"] == "system_error":
            torch.cuda.empty_cache()

    stored_rows = int(shard.execute("SELECT COUNT(*) FROM audio_results").fetchone()[0])
    if stored_rows != len(assigned):
        raise RuntimeError(
            f"rank {rank} audio coverage mismatch: {stored_rows} != {len(assigned)}"
        )
    shard.execute(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES ('rows',?)",
        (str(stored_rows),),
    )
    shard.execute(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES ('status','COMPLETE')"
    )
    shard.commit()
    shard.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    shard.close()
    del executor, tokenizer, codec, rows
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier(group=completion_group)

    passed: list[Any] = [False]
    if rank == 0:
        unchanged = _verify_run_inputs_unchanged(run_contract)
        records = _load_audio_records(
            output_dir,
            world_size=world_size,
            expected_rows=EXPECTED_ROWS,
            run_contract_sha256=run_contract_sha256,
        )
        summary_core = _summarize_audio_records(
            records,
            expected_rows=EXPECTED_ROWS,
            expected_anchor_ordinals=run_contract["exact_anchor_ordinals"],
            inputs_unchanged=unchanged["all_sha256_unchanged"],
        )
        report = {
            "schema": AUDIO_CLOSURE_SCHEMA,
            "schema_version": 1,
            "status": summary_core["status"],
            "contract": AUDIO_CLOSURE_CONTRACT,
            "run_contract": str(output_dir / "RUN_CONTRACT.json"),
            "run_contract_sha256": _sha256_file(
                output_dir / "RUN_CONTRACT.json"
            ),
            "run_contract_canonical_sha256": run_contract_sha256,
            "checkpoint": run_contract["plan_evaluation"][
                "generation_ar_checkpoint"
            ],
            "checkpoint_sha256": run_contract["plan_evaluation"][
                "generation_ar_checkpoint_sha256"
            ],
            "checkpoint_step": run_contract["plan_evaluation"][
                "generation_ar_checkpoint_step"
            ],
            "p10_release_id": run_contract["p10"]["release_id"],
            "sampling": run_contract["p10"]["sampling"],
            "input_reverification": unchanged,
            **summary_core,
            "attribution": {
                "target_path": "GT ScenePlan -> frozen P10-v11(seed_i) -> target FOA",
                "prediction_path": (
                    "raw user input -> Generation AR ScenePlan -> same frozen "
                    "P10-v11(seed_i) -> prediction FOA"
                ),
                "same_p10_seed_per_row": True,
                "p10_residual_quality_is_not_generation_ar_error": True,
                "pair_difference_is_generation_plan_consequence_under_fixed_p10": True,
            },
            "limitations": {
                "clap_semantic_audio_alignment_measured": False,
                "whisper_asr_measured": False,
                "absolute_semantic_audio_accuracy_claimed": False,
                "absolute_transcript_accuracy_claimed": False,
                "waveforms_retained": False,
            },
            "quality_decision": (
                "METRICS_REPORTED_WITHOUT_AR_QUALITY_THRESHOLD; status gates "
                "only evaluator and frozen-executor integrity"
            ),
        }
        report["report_sha256_without_self"] = _canonical_sha256(report)
        _atomic_json(output_dir / "SUMMARY.json", report)
        print(
            json.dumps(
                {
                    "event": "p10_audio_evaluation_complete",
                    "status": report["status"],
                    "summary": str(output_dir / "SUMMARY.json"),
                    "coverage": report["coverage"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        passed[0] = report["status"] == "PASS"
    dist.broadcast_object_list(passed, src=0, device=device)
    dist.destroy_process_group()
    return 0 if bool(passed[0]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
