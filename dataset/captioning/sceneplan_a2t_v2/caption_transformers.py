#!/usr/bin/env python3
"""Batched, resumable Qwen3-Omni-Instruct source annotation via Transformers.

The current stock vLLM 0.23 V1 adapter can hear the audio but was empirically
shown to ignore the accompanying text instruction.  This runner follows the
model revision's official Transformers path, disables the unused talker, and
registers only the complete direct decoder response after whitespace
compaction.  It retains deterministic modulo sharding and resumability while
batching audio prompts for production throughput.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from description_contract import (
    PROMPT_TEMPLATE_VERSION,
    PROMPT_USER_TRIGGER,
    TARGET_MAX_WORDS,
    TARGET_MIN_WORDS,
    prompt_contract_sha256,
    validate_source_description,
)


DEFAULT_MODEL = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/Qwen3-Omni-30B-A3B-Instruct")
DEFAULT_MODEL_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
DEFAULT_PROMPT = Path(__file__).with_name("source_description_prompt_v4.txt")
DEFAULT_SAFETY_MAX_GENERATION_TOKENS = 256
SCHEMA = "stable_audio_tools.sceneplan_source_description_annotation"
SCHEMA_VERSION = 3


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_input(
    path: Path,
    *,
    shard: int,
    num_shards: int,
    done: set[str],
    limit: int | None,
) -> tuple[list[dict[str, Any]], int, int, bool]:
    """Load only this process's deterministic modulo shard.

    The frozen manifest/finalizer own global uniqueness.  A worker parses and
    resolves only its assigned rows, avoiding four copies of the 873k-row
    manifest and four full filesystem walks in the eight-GPU topology.
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    input_rows = 0
    assigned_rows = 0
    scan_complete = True
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            ordinal = input_rows
            input_rows += 1
            if ordinal % num_shards != shard:
                continue
            assigned_rows += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid input JSON at {path}:{line_number}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"input row is not an object at {path}:{line_number}")
            annotation_id = str(raw.get("id") or "")
            audio_value = raw.get("audio_path") or raw.get("audio") or raw.get("path")
            if not annotation_id or not audio_value:
                raise ValueError(f"missing id/audio_path at {path}:{line_number}")
            if annotation_id in seen:
                raise ValueError(f"duplicate input id: {annotation_id}")
            seen.add(annotation_id)
            if annotation_id in done:
                continue
            audio = Path(str(audio_value)).expanduser().resolve(strict=True)
            if not audio.is_file():
                raise ValueError(f"audio is not a file: {audio}")
            row: dict[str, Any] = {"id": annotation_id, "audio_path": str(audio)}
            for key in (
                "kind",
                "split",
                "source_audio_sha256",
                "asset_id",
                "source_dataset",
                "raw_label",
            ):
                if key in raw:
                    row[key] = raw[key]
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                scan_complete = False
                break
    if not input_rows:
        raise ValueError(f"no input rows: {path}")
    return rows, input_rows, assigned_rows, scan_complete


def output_for_shard(base: Path, shard: int, num_shards: int) -> Path:
    if num_shards == 1:
        return base
    width = max(3, len(str(num_shards - 1)))
    return base.with_name(
        f"{base.stem}.shard{shard:0{width}d}-of-{num_shards:0{width}d}{base.suffix}"
    )


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    raw_text = path.read_text(encoding="utf-8")
    lines = raw_text.splitlines()
    done: set[str] = set()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            annotation_id = str(row["id"])
        except (json.JSONDecodeError, KeyError) as error:
            incomplete_tail = index == len(lines) - 1 and not raw_text.endswith("\n")
            if incomplete_tail:
                logging.warning("ignoring incomplete final output line in %s", path)
                continue
            raise ValueError(f"invalid existing output line {index + 1}: {path}") from error
        if annotation_id in done:
            raise ValueError(f"duplicate existing output id {annotation_id}: {path}")
        done.add(annotation_id)
    return done


def build_messages(row: dict[str, Any], instruction: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": instruction}],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": row["audio_path"]},
                {"type": "text", "text": PROMPT_USER_TRIGGER},
            ],
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--last-token-logits-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "During generation, project only the final hidden position to the "
            "151k-token vocabulary. Generation never consumes prompt-position logits."
        ),
    )
    parser.add_argument(
        "--fsync-every-batches",
        type=int,
        default=1,
        help="Durably checkpoint after this many generated batches.",
    )
    parser.add_argument(
        "--safety-max-generation-tokens",
        type=int,
        default=DEFAULT_SAFETY_MAX_GENERATION_TOKENS,
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_shards <= 0 or not 0 <= args.shard < args.num_shards:
        raise ValueError("require num_shards > 0 and 0 <= shard < num_shards")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.fsync_every_batches <= 0:
        raise ValueError("--fsync-every-batches must be positive")
    if not 64 <= args.safety_max_generation_tokens <= 1024:
        raise ValueError("safety generation ceiling must be between 64 and 1024")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    validate_args(args)
    input_path = args.input_jsonl.expanduser().resolve(strict=True)
    model_path = args.model_path.expanduser().resolve(strict=True)
    prompt_path = args.prompt_path.expanduser().resolve(strict=True)
    instruction = prompt_path.read_text(encoding="utf-8")
    if not instruction.strip() or "\r" in instruction:
        raise ValueError(f"invalid frozen instruction text: {prompt_path}")
    prompt_file_sha256 = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    prompt_sha256 = prompt_contract_sha256(instruction)
    model_config_sha256 = hashlib.sha256(
        (model_path / "config.json").read_bytes()
    ).hexdigest()

    base_output = args.out.expanduser().resolve(strict=False)
    output = output_for_shard(base_output, args.shard, args.num_shards)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(output)
    todo, input_rows, assigned_rows, input_scan_complete = load_input(
        input_path,
        shard=args.shard,
        num_shards=args.num_shards,
        done=done,
        limit=args.limit,
    )
    logging.info(
        "shard=%d/%d input_scanned=%d assigned_scanned=%d scan_complete=%s "
        "already=%d todo=%d output=%s",
        args.shard,
        args.num_shards,
        input_rows,
        assigned_rows,
        input_scan_complete,
        len(done),
        len(todo),
        output,
    )
    if not todo:
        return 0

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    import transformers
    from qwen_omni_utils import process_mm_info
    from transformers import (
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeProcessor,
    )

    torch.manual_seed(args.seed)
    load_started = time.monotonic()
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.disable_talker()
    if args.last_token_logits_only:
        class LastTokenLogits(torch.nn.Module):
            """Drop generation-only prompt logits that Hugging Face never consumes."""

            def __init__(self, projection: torch.nn.Module) -> None:
                super().__init__()
                self.projection = projection

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                return self.projection(hidden_states[:, -1:, :])

        model.thinker.lm_head = LastTokenLogits(model.thinker.lm_head)
    model.eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    model_load_seconds = time.monotonic() - load_started
    logging.info(
        "loaded model in %.3fs device_map=%s attn=%s",
        model_load_seconds,
        args.device_map,
        args.attn_implementation,
    )

    completed = 0
    generated_tokens_total = 0
    capped = 0
    capped_retry_events = 0
    oom_retries = 0
    adaptive_batch_size = int(args.batch_size)
    reduced_batch_successes = 0
    singleton_retry_remaining = 0
    singleton_restore_batch_size: int | None = None
    inference_seconds = 0.0
    annotation_started = time.monotonic()
    cursor = 0
    batch_ordinal = 0
    initial_todo_rows = len(todo)
    with output.open("a", encoding="utf-8") as sink:
        while cursor < len(todo):
            batch_rows = todo[cursor : cursor + adaptive_batch_size]
            conversations = [build_messages(row, instruction) for row in batch_rows]
            texts = processor.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=True,
            )
            audios, images, videos = process_mm_info(
                conversations,
                use_audio_in_video=False,
            )
            inputs = processor(
                text=texts,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            )
            inputs = inputs.to(model.device).to(model.dtype)
            input_length = int(inputs["input_ids"].shape[1])
            started = time.monotonic()
            try:
                with torch.inference_mode():
                    generated, _ = model.generate(
                        **inputs,
                        return_audio=False,
                        thinker_return_dict_in_generate=True,
                        use_audio_in_video=False,
                        do_sample=False,
                        max_new_tokens=args.safety_max_generation_tokens,
                    )
            except torch.OutOfMemoryError:
                previous_batch_size = len(batch_rows)
                if previous_batch_size <= 1:
                    raise
                oom_retries += 1
                adaptive_batch_size = max(1, previous_batch_size // 2)
                reduced_batch_successes = 0
                del inputs, conversations, texts, audios, images, videos
                gc.collect()
                for device_index in range(torch.cuda.device_count()):
                    with torch.cuda.device(device_index):
                        torch.cuda.empty_cache()
                logging.warning(
                    "CUDA OOM at todo_cursor=%d batch=%d; retrying the same rows "
                    "with adaptive_batch_size=%d",
                    cursor,
                    previous_batch_size,
                    adaptive_batch_size,
                )
                continue
            elapsed = time.monotonic() - started
            inference_seconds += elapsed
            special_ids = {
                value
                for value in (
                    processor.tokenizer.eos_token_id,
                    processor.tokenizer.pad_token_id,
                )
                if value is not None
            }
            payloads: list[dict[str, Any]] = []
            capped_rows: list[dict[str, Any]] = []
            capped_source_rows: list[dict[str, Any]] = []
            batch_generated_tokens = 0
            for row_index, row in enumerate(batch_rows):
                sequence = generated.sequences[row_index, input_length:]
                token_ids = sequence.detach().cpu().tolist()
                effective_ids: list[int] = []
                stopped = False
                for token_id in token_ids:
                    if token_id in special_ids:
                        stopped = True
                        break
                    effective_ids.append(int(token_id))
                finish_reason = "stop" if stopped else "length"
                was_capped = finish_reason == "length"
                decoder_text_raw = processor.tokenizer.decode(
                    effective_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                qc = validate_source_description(decoder_text_raw)
                decoder_text = qc.text
                if not decoder_text:
                    raise RuntimeError(f"empty description for {row['id']}")
                if was_capped:
                    capped_source_rows.append(row)
                    capped_rows.append(
                        {
                            "schema": SCHEMA + ".failure",
                            "schema_version": SCHEMA_VERSION,
                            "id": row["id"],
                            "reason": "generation_safety_ceiling_reached",
                            "decoder_text": decoder_text,
                            "generated_tokens": len(effective_ids),
                            "safety_max_generation_tokens": (
                                args.safety_max_generation_tokens
                            ),
                            "prompt_sha256": prompt_sha256,
                            "model_revision": args.model_revision,
                        }
                    )
                    continue

                payload: dict[str, Any] = {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "id": row["id"],
                "audio_path": row["audio_path"],
                "source_description": decoder_text,
                "source_description_sha256": hashlib.sha256(
                    decoder_text.encode("utf-8")
                ).hexdigest(),
                "description_word_count": qc.word_count,
                "description_target_words": [TARGET_MIN_WORDS, TARGET_MAX_WORDS],
                "decoder_text": decoder_text,
                "decoder_text_raw_sha256": hashlib.sha256(
                    decoder_text_raw.encode("utf-8")
                ).hexdigest(),
                "decoder_text_sha256": hashlib.sha256(
                    decoder_text.encode("utf-8")
                ).hexdigest(),
                "description_sentence_count": qc.sentence_count,
                "description_is_complete_direct_response": True,
                "description_cleanup": "compact_whitespace_only",
                "description_hard_qc_flags": list(qc.hard_flags),
                "description_soft_qc_flags": list(qc.soft_flags),
                "description_informational_flags": list(qc.informational_flags),
                "model_path": str(model_path),
                "model_revision": args.model_revision,
                "model_config_sha256": model_config_sha256,
                "prompt_path": str(prompt_path),
                "prompt_file_sha256": prompt_file_sha256,
                "prompt_sha256": prompt_sha256,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "decoder_constraint": "none",
                "engine": "transformers",
                "engine_version": transformers.__version__,
                "device_map": args.device_map,
                "attn_implementation": args.attn_implementation,
                "talker_disabled": True,
                "temperature": 0.0,
                "seed": args.seed,
                "safety_max_generation_tokens": args.safety_max_generation_tokens,
                "generated_tokens": len(effective_ids),
                "finish_reason": finish_reason,
                "generation_capped": was_capped,
                "batch_size_requested": args.batch_size,
                "batch_size_actual": len(batch_rows),
                "shard": args.shard,
                "num_shards": args.num_shards,
            }
                for key in (
                    "kind",
                    "split",
                    "source_audio_sha256",
                    "asset_id",
                    "source_dataset",
                    "raw_label",
                ):
                    if key in row:
                        payload[key] = row[key]
                payloads.append(payload)
                batch_generated_tokens += len(effective_ids)

            if capped_rows:
                failure_path = output.with_suffix(".failures.jsonl")
                with failure_path.open("a", encoding="utf-8") as failure_sink:
                    for failure in capped_rows:
                        failure_sink.write(stable_json(failure) + "\n")
                    failure_sink.flush()
                    os.fsync(failure_sink.fileno())
                for payload in payloads:
                    sink.write(stable_json(payload) + "\n")
                sink.flush()
                os.fsync(sink.fileno())
                completed += len(payloads)
                generated_tokens_total += batch_generated_tokens
                if singleton_retry_remaining:
                    capped += len(capped_rows)
                    raise RuntimeError(
                        "generation safety ceiling persisted during direct "
                        f"batch-1 retry for {capped_rows[0]['id']}"
                    )
                capped_retry_events += len(capped_rows)
                original_batch_size = len(batch_rows)
                todo[cursor : cursor + original_batch_size] = capped_source_rows
                singleton_retry_remaining = len(capped_source_rows)
                singleton_restore_batch_size = adaptive_batch_size
                adaptive_batch_size = 1
                reduced_batch_successes = 0
                logging.warning(
                    "generation ceiling reached for %d/%d rows at todo_cursor=%d; "
                    "saved complete peers and retrying only capped rows with batch=1",
                    len(capped_source_rows),
                    original_batch_size,
                    cursor,
                )
                batch_ordinal += 1
                continue

            for payload in payloads:
                sink.write(stable_json(payload) + "\n")
            sink.flush()
            if (batch_ordinal + 1) % args.fsync_every_batches == 0:
                os.fsync(sink.fileno())
            completed += len(payloads)
            generated_tokens_total += batch_generated_tokens
            logging.info(
                "batch=%d size=%d completed=%d/%d inference=%.3fs tokens=%d",
                batch_ordinal,
                len(batch_rows),
                completed,
                initial_todo_rows,
                elapsed,
                batch_generated_tokens,
            )
            cursor += len(batch_rows)
            batch_ordinal += 1
            if singleton_retry_remaining:
                singleton_retry_remaining -= len(batch_rows)
                if singleton_retry_remaining == 0:
                    if singleton_restore_batch_size is None:
                        raise RuntimeError("missing batch size for capped-row recovery")
                    adaptive_batch_size = singleton_restore_batch_size
                    singleton_restore_batch_size = None
                    logging.info(
                        "direct batch-1 capped-row retry passed; restored "
                        "adaptive_batch_size=%d",
                        adaptive_batch_size,
                    )
            elif adaptive_batch_size < int(args.batch_size):
                reduced_batch_successes += 1
                if reduced_batch_successes >= 2:
                    adaptive_batch_size = int(args.batch_size)
                    reduced_batch_successes = 0
                    logging.info(
                        "two reduced batches passed after OOM; restored "
                        "adaptive_batch_size=%d",
                        adaptive_batch_size,
                    )
            else:
                reduced_batch_successes = 0
        sink.flush()
        os.fsync(sink.fileno())

    annotation_seconds = time.monotonic() - annotation_started

    summary = {
        "schema": "stable_audio_tools.sceneplan_a2t_transformers_run",
        "schema_version": 1,
        "input_jsonl": str(input_path),
        "output_jsonl": str(output),
        "model_path": str(model_path),
        "model_revision": args.model_revision,
        "model_config_sha256": model_config_sha256,
        "prompt_path": str(prompt_path),
        "prompt_file_sha256": prompt_file_sha256,
        "prompt_sha256": prompt_sha256,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "decoder_constraint": "none",
        "shard": args.shard,
        "num_shards": args.num_shards,
        "completed_this_run": completed,
        "input_rows_scanned": input_rows,
        "assigned_shard_rows_scanned": assigned_rows,
        "input_scan_complete": input_scan_complete,
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "clips_per_second": completed / inference_seconds if inference_seconds else None,
        "annotation_seconds": annotation_seconds,
        "end_to_end_clips_per_second_excluding_model_load": (
            completed / annotation_seconds if annotation_seconds else None
        ),
        "generated_tokens": generated_tokens_total,
        "generation_capped": capped,
        "capped_rows_retried_as_singletons": capped_retry_events,
        "safety_max_generation_tokens": args.safety_max_generation_tokens,
        "output_token_count_is_data_gate": False,
        "description_cleanup": "compact_whitespace_only",
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "batch_size": args.batch_size,
        "final_adaptive_batch_size": adaptive_batch_size,
        "oom_retries": oom_retries,
        "last_token_logits_only": args.last_token_logits_only,
        "fsync_every_batches": args.fsync_every_batches,
        "talker_disabled": True,
        "visible_gpu_count": torch.cuda.device_count(),
        "cuda_peak_memory_allocated_bytes": [
            int(torch.cuda.max_memory_allocated(device))
            for device in range(torch.cuda.device_count())
        ],
    }
    atomic_json(output.with_suffix(".summary.json"), summary)
    logging.info("done %s", stable_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
