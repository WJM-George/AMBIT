#!/usr/bin/env python3
"""Real-checkpoint gate for P10-v11 shared-block Generation AR."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.model_sceneplan_codec_v4 import (  # noqa: E402
    ModelScenePlanCodecV4,
)
from stable_audio_tools.models.sceneplan_transfusion_generation_ar import (  # noqa: E402
    GENERATION_AR_CONTRACT,
    load_p10v11_generation_ar,
)


CODEC_PATH = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)
TEST_MANIFEST = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/transfusion_shared_v1/"
    "generation_ar/test.sqlite"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts/sceneplan_p11/"
    "p10v11_shared_generation_ar_real_graph_gate_20260904.json"
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=TEST_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA real-graph gate requested without CUDA")
    codec = ModelScenePlanCodecV4(CODEC_PATH)
    manifest = args.manifest.expanduser().resolve(strict=True)
    connection = sqlite3.connect(
        f"file:{manifest}?mode=ro&immutable=1", uri=True
    )
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    row = connection.execute(
        """
        SELECT sample_id, raw_user_request, raw_request_qwen_tokens,
               target_token_ids_u16le, target_token_count
        FROM rows
        WHERE source_count = 4
        ORDER BY raw_request_qwen_tokens DESC, ordinal
        LIMIT 1
        """
    ).fetchone()
    connection.close()
    if row is None:
        raise RuntimeError("Generation test manifest has no four-source row")
    sample_id, request, request_tokens, token_blob, target_count = row
    target_ids = np.frombuffer(token_blob, dtype="<u2").astype(np.int64)
    if len(target_ids) != int(target_count) or len(target_ids) < 10:
        raise RuntimeError("Generation target token blob is malformed")

    torch.manual_seed(42)
    model, load_report = load_p10v11_generation_ar(
        pad_id=codec.pad_id,
        verify_sha256=True,
        activation_checkpointing=False,
    )
    if model.shared_transformer is not model.p10_dit.transformer:
        raise RuntimeError("AR and DiT do not own the same Transformer object")
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable or any(
        not name.startswith("ar_adapter.") for name in trainable
    ):
        raise RuntimeError(f"non-adapter parameter is trainable: {list(trainable)[:8]}")
    if any(parameter.requires_grad for parameter in model.p10_dit.parameters()):
        raise RuntimeError("canonical P10-v11 tensors are not frozen")

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        context, context_mask = model.encode_requests([str(request)], device=device)
        plan = torch.from_numpy(target_ids[:-1].copy()).to(
            device=device, dtype=torch.long
        )[None]
        plan_mask = torch.ones_like(plan, dtype=torch.bool)

        acoustic_input = torch.randn(
            1, 9, 320, device=device, dtype=dtype
        )
        acoustic_mask = torch.ones(1, 9, device=device, dtype=torch.bool)
        acoustic_before = model.shared_transformer(
            acoustic_input,
            context=context,
            context_mask=context_mask,
            padding_mask=acoustic_mask,
            use_checkpointing=False,
        )
        logits = model(plan, plan_mask, context, context_mask)
        if tuple(logits.shape) != (1, len(target_ids) - 1, 4096):
            raise RuntimeError(f"unexpected Generation AR logits {tuple(logits.shape)}")
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("Generation AR real graph produced non-finite logits")

        cached_steps = min(16, int(plan.shape[1]))
        decode_cache = model.prepare_decode_cache(
            context, context_mask, max_plan_tokens=1024
        )
        cached_logits = torch.stack(
            [
                model.decode_step(plan[:, position], decode_cache)
                for position in range(cached_steps)
            ],
            dim=1,
        )
        cached_decode_max_abs = float(
            (cached_logits - logits[:, :cached_steps]).abs().max().float().item()
        )
        cached_decode_top1_match = float(
            cached_logits.argmax(dim=-1)
            .eq(logits[:, :cached_steps].argmax(dim=-1))
            .float()
            .mean()
            .item()
        )
        legal_matches = []
        for position in range(cached_steps):
            prefix = target_ids[: position + 1].tolist()
            allowed = torch.tensor(
                sorted(codec.allowed_next_ids(prefix)),
                device=device,
                dtype=torch.long,
            )
            full_choice = int(
                allowed[logits[0, position].index_select(0, allowed).argmax()].item()
            )
            cached_choice = int(
                allowed[
                    cached_logits[0, position].index_select(0, allowed).argmax()
                ].item()
            )
            legal_matches.append(full_choice == cached_choice)
        cached_decode_legal_top1_match = sum(legal_matches) / len(legal_matches)
        if (
            cached_decode_max_abs > 0.125
            or cached_decode_legal_top1_match != 1.0
        ):
            raise RuntimeError(
                "cached decoding differs from causal full-sequence AR: "
                f"max_abs={cached_decode_max_abs}, "
                f"top1_match={cached_decode_top1_match}, "
                f"legal_top1_match={cached_decode_legal_top1_match}"
            )

        boundary = min(8, int(plan.shape[1]) - 2)
        changed_plan = plan.clone()
        changed_plan[:, boundary:] = (
            changed_plan[:, boundary:] + 137
        ) % 4096
        changed_logits = model(changed_plan, plan_mask, context, context_mask)
        causal_prefix_max_abs = float(
            (logits[:, :boundary] - changed_logits[:, :boundary])
            .abs()
            .max()
            .float()
            .item()
        )
        if causal_prefix_max_abs != 0.0:
            raise RuntimeError(
                "future ScenePlan tokens leaked into the causal prefix: "
                f"max_abs={causal_prefix_max_abs}"
            )

        acoustic_after = model.shared_transformer(
            acoustic_input,
            context=context,
            context_mask=context_mask,
            padding_mask=acoustic_mask,
            use_checkpointing=False,
        )
        p10_default_parity_max_abs = float(
            (acoustic_before - acoustic_after).abs().max().float().item()
        )
        if p10_default_parity_max_abs != 0.0:
            raise RuntimeError(
                "AR invocation mutated the P10 DiT path: "
                f"max_abs={p10_default_parity_max_abs}"
            )

    output = args.output.expanduser().resolve(strict=False)
    report = {
        "schema": "stable_audio_tools.sceneplan_transfusion_generation_ar_gate",
        "schema_version": 1,
        "status": "PASS",
        "contract": GENERATION_AR_CONTRACT,
        "device": str(device),
        "dtype": str(dtype),
        "manifest": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "manifest_source_index_sha256": metadata["source_index_sha256"],
        "sample_id": str(sample_id),
        "request_qwen_tokens": int(request_tokens),
        "target_plan_tokens": int(target_count),
        "logits_shape": list(logits.shape),
        "logits_finite": True,
        "cached_decode_steps": cached_steps,
        "cached_decode_max_abs": cached_decode_max_abs,
        "cached_decode_top1_match": cached_decode_top1_match,
        "cached_decode_legal_top1_match": cached_decode_legal_top1_match,
        "same_transformer_object": True,
        "self_attention_causal_prefix_max_abs": causal_prefix_max_abs,
        "p10_default_path_pre_post_ar_max_abs": p10_default_parity_max_abs,
        "p10_frozen_parameter_count": sum(
            int(parameter.numel()) for parameter in model.p10_dit.parameters()
        ),
        "trainable_parameter_count": sum(
            int(parameter.numel()) for parameter in trainable.values()
        ),
        "trainable_parameter_names": sorted(trainable),
        "p10_load": load_report.as_dict(),
    }
    _atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
