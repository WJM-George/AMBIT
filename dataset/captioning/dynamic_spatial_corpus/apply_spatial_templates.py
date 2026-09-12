#!/usr/bin/env python3
"""Compile synthesis-manifest rows into staged spatial captions.

Stage 1 covers horizontal direction. Stage 2 adds elevation and retains the
legacy JSON-template ablation. Stages 3-7 use total-coverage structured
renderers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Iterator

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from spatial_template_corpus import (  # noqa: E402
    DEFAULT_CORPUS_PATH,
    STAGE_NAMES,
    TemplateSpec,
    load_corpus,
    render_for_row,
    render_stage_caption,
    select_template_for_row,
)

LOG = logging.getLogger("apply_spatial_templates")
UNIVERSAL_STAGES = frozenset({1, 3, 4, 5, 6, 7})


def iter_manifest(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if row.get("status", "ok") != "ok" or not row.get("foa_path"):
                continue
            if not row.get("id"):
                raise ValueError(f"manifest row lacks id at {path}:{line_number}")
            yield row


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                done.add(str(json.loads(line)["id"]))
            except (KeyError, json.JSONDecodeError):
                continue
    return done


def default_output_path(manifest: Path, stage: int) -> Path:
    return manifest.parent / f"captions_stage{stage}.jsonl"


def _row_rng(seed: int, stage: int, row_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{stage}:{row_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def apply_row(
    row: dict,
    corpus: list[TemplateSpec],
    *,
    stage: int,
    seed: int,
    fallback: str,
) -> dict | None:
    row_id = str(row["id"])
    rng = _row_rng(seed, stage, row_id)
    match = "template"
    try:
        if stage in UNIVERSAL_STAGES:
            caption, template_id = render_stage_caption(stage, row, rng=rng)
        else:
            template = select_template_for_row(
                row,
                corpus,
                stage=stage,
                rng=rng,
                exclude_tags=("paraphrase",),
            )
            caption = render_for_row(template, row)
            template_id = template.template_id
    except LookupError:
        if fallback == "skip":
            return None
        if fallback == "error":
            raise
        caption = str(row.get("spatial_caption") or "A sound in an unspecified space.").strip()
        template_id = None
        match = "fallback_spatial"

    result = {
        "id": row_id,
        "foa_path": row["foa_path"],
        "caption": caption,
        "draft": caption,
        "stage": stage,
        "template_id": template_id,
        "match": match,
        "mix_type": row.get("mix_type"),
        "n_sources": row.get("n_sources", len(row.get("sources") or [])),
    }
    return result


def _parse_stages(stage: int | None, stages: str | None) -> list[int]:
    values = [stage] if stage is not None else [
        int(part.strip()) for part in (stages or "").split(",") if part.strip()
    ]
    if not values:
        raise ValueError("provide --stage or --stages")
    if any(value not in range(1, 8) for value in values):
        raise ValueError(f"stages must be in 1..7, got {values}")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate stages: {values}")
    return values


def _inspect(path: Path, count: int) -> None:
    stats = Counter()
    samples: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            stats[row.get("match", "unknown")] += 1
            if len(samples) < count:
                samples.append(row)
    total = sum(stats.values())
    print(f"[inspect] {path} rows={total} matches={dict(stats)}")
    for row in samples:
        print(
            f"[{row['id']}] stage={row['stage']} template={row.get('template_id')} "
            f"{row['caption']}"
        )


def compile_manifest(
    manifest: Path,
    stages: list[int],
    *,
    corpus: list[TemplateSpec],
    output_paths: dict[int, Path],
    seed: int,
    fallback: str,
    overwrite: bool,
    limit: int | None,
    log_every: int,
) -> dict[int, Counter]:
    done = {
        stage: set() if overwrite else load_done(output_paths[stage])
        for stage in stages
    }
    stats = {stage: Counter() for stage in stages}
    write_paths = {
        stage: (
            output_paths[stage].with_name(f".{output_paths[stage].name}.tmp")
            if overwrite
            else output_paths[stage]
        )
        for stage in stages
    }

    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    modes = {stage: "w" if overwrite else "a" for stage in stages}
    completed = False
    with ExitStack() as stack:
        handles = {
            stage: stack.enter_context(
                write_paths[stage].open(modes[stage], encoding="utf-8")
            )
            for stage in stages
        }
        seen = 0
        for row in iter_manifest(manifest):
            if limit is not None and seen >= limit:
                break
            seen += 1
            for stage in stages:
                if str(row["id"]) in done[stage]:
                    stats[stage]["already_done"] += 1
                    continue
                result = apply_row(
                    row,
                    corpus,
                    stage=stage,
                    seed=seed,
                    fallback=fallback,
                )
                if result is None:
                    stats[stage]["skipped"] += 1
                    continue
                handles[stage].write(json.dumps(result, ensure_ascii=False) + "\n")
                stats[stage][result["match"]] += 1
            if log_every > 0 and seen % log_every == 0:
                LOG.info("processed manifest rows=%d", seen)
        completed = True

    if completed and overwrite:
        for stage in stages:
            write_paths[stage].replace(output_paths[stage])
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage", type=int)
    group.add_argument("--stages", help="comma-separated stages, for example 3,4,5,6")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--out", type=Path, help="single-stage output path")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fallback", choices=("spatial", "skip", "error"), default="spatial")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--inspect", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100_000)
    # Kept for compatibility with older launch commands. This workload is
    # streaming/string-bound; one process avoids serializing million-row lists.
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stages = _parse_stages(args.stage, args.stages)
    if args.out and len(stages) != 1:
        raise SystemExit("--out is valid only with one stage")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.workers != 1:
        LOG.warning("--workers is retained for compatibility; streaming compiler uses one process")

    output_paths = {}
    for stage in stages:
        if args.out:
            output = args.out
        elif args.out_dir:
            suffix = "1-2" if stage == 1 else str(stage)
            output = args.out_dir / f"captions_stage{suffix}.jsonl"
        else:
            output = default_output_path(args.manifest, stage)
        output_paths[stage] = output

    corpus = load_corpus(args.corpus)
    stats = compile_manifest(
        args.manifest,
        stages,
        corpus=corpus,
        output_paths=output_paths,
        seed=args.seed,
        fallback=args.fallback,
        overwrite=args.overwrite,
        limit=args.limit,
        log_every=args.log_every,
    )
    for stage in stages:
        LOG.info(
            "DONE stage=%d name=%s out=%s stats=%s",
            stage,
            STAGE_NAMES[stage],
            output_paths[stage],
            dict(stats[stage]),
        )
        if args.inspect:
            _inspect(output_paths[stage], args.inspect)


if __name__ == "__main__":
    main()
