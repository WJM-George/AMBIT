#!/usr/bin/env python3
"""Fail-closed audit for the Spatial-CoT semantic-caption template catalog."""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.spatial_caption_templates import (  # noqa: E402
    SEMANTIC_CAPTION_TEMPLATE_COUNT,
    SEMANTIC_CAPTION_TEMPLATE_VERSION,
    render_semantic_caption,
    validate_semantic_caption_metadata,
)


_SYNTHETIC_SOURCES = [
    {
        "source_id": "source_0",
        "event": {"label": "Speech"},
        "content": {"transcript": "Turn left at the next street."},
    },
    {
        "source_id": "source_1",
        "event": {
            "label": "A power drill starts, followed by a circular saw cutting wood."
        },
        "content": {},
    },
    {
        "source_id": "source_2",
        "event": {"label": "Rhythm and blues music"},
        "content": {},
    },
    {
        "source_id": "source_3",
        "event": {"label": "Birdsong with several short chirps"},
        "content": {},
    },
]


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(len(ordered) - 1, low + 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _metadata_records(root: Path, limit: int) -> Iterable[dict[str, Any]]:
    seen = 0
    for path in sorted((root / "metadata").glob("families-*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield json.loads(line)
                seen += 1
                if seen >= limit:
                    return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-root", type=Path)
    parser.add_argument("--sample-families", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--examples", type=int, default=24)
    args = parser.parse_args()

    failures: list[str] = []
    catalog_uniques: dict[str, int] = {}
    example_stride = max(1, SEMANTIC_CAPTION_TEMPLATE_COUNT // max(1, args.examples))
    examples: list[dict[str, Any]] = []
    for source_count in (1, 2, 4):
        texts: set[str] = set()
        sources = _SYNTHETIC_SOURCES[:source_count]
        expected_ids = [str(source["source_id"]) for source in sources]
        for template_id in range(SEMANTIC_CAPTION_TEMPLATE_COUNT):
            rendered = render_semantic_caption(sources, template_id=template_id)
            try:
                validate_semantic_caption_metadata(
                    rendered.text,
                    rendered.metadata(),
                    expected_source_ids=expected_ids,
                )
            except Exception as exc:  # fail-closed report, continue catalog audit
                failures.append(
                    f"template {template_id} source_count={source_count}: {exc}"
                )
            if re.search(r"[.!?][,;]", rendered.text):
                failures.append(
                    f"template {template_id} has doubled list punctuation: "
                    f"{rendered.text!r}"
                )
            texts.add(rendered.text)
            if source_count == 4 and template_id % example_stride == 0:
                examples.append(
                    {"template_id": template_id, "caption": rendered.text}
                )
        catalog_uniques[str(source_count)] = len(texts)
        if len(texts) != SEMANTIC_CAPTION_TEMPLATE_COUNT:
            failures.append(
                f"source_count={source_count} yielded {len(texts)} unique captions, "
                f"expected {SEMANTIC_CAPTION_TEMPLATE_COUNT}"
            )

    char_lengths: list[int] = []
    word_lengths: list[int] = []
    template_counts: Counter[int] = Counter()
    sampled_families = 0
    sampled_turns = 0
    if args.latent_root is not None:
        latent_root = args.latent_root.expanduser().resolve()
        if not (latent_root / "READY").is_file():
            failures.append(f"latent store is not READY: {latent_root}")
        else:
            for family in _metadata_records(latent_root, args.sample_families):
                sampled_families += 1
                family_id = str(family.get("family_id") or "")
                family_template_ids: set[int] = set()
                for turn in family.get("turns") or []:
                    sources = (
                        (((turn.get("after") or {}).get("scene_plan") or {}).get("scene") or {})
                        .get("sources")
                        or []
                    )
                    rendered = render_semantic_caption(
                        sources, template_key=family_id
                    )
                    expected_ids = [
                        str(source.get("source_id") or f"source_{index}")
                        for index, source in enumerate(sources)
                    ]
                    try:
                        validate_semantic_caption_metadata(
                            rendered.text,
                            rendered.metadata(),
                            expected_source_ids=expected_ids,
                        )
                    except Exception as exc:
                        failures.append(f"family {family_id}: {exc}")
                    sampled_turns += 1
                    family_template_ids.add(rendered.template_id)
                    template_counts[rendered.template_id] += 1
                    char_lengths.append(len(rendered.text))
                    word_lengths.append(len(rendered.text.split()))
                if len(family_template_ids) != 1:
                    failures.append(
                        f"family {family_id} changed template across turns: "
                        f"{sorted(family_template_ids)}"
                    )

    report = {
        "status": "PASS" if not failures else "FAIL",
        "schema": "stable_audio_tools.spatial_caption_template_audit",
        "schema_version": 1,
        "template_version": SEMANTIC_CAPTION_TEMPLATE_VERSION,
        "template_count": SEMANTIC_CAPTION_TEMPLATE_COUNT,
        "catalog_unique_by_source_count": catalog_uniques,
        "sampled_families": sampled_families,
        "sampled_turns": sampled_turns,
        "sampled_distinct_templates": len(template_counts),
        "caption_chars": {
            "mean": statistics.fmean(char_lengths) if char_lengths else 0.0,
            "p50": _percentile(char_lengths, 0.50),
            "p90": _percentile(char_lengths, 0.90),
            "p99": _percentile(char_lengths, 0.99),
            "max": max(char_lengths, default=0),
        },
        "caption_words": {
            "mean": statistics.fmean(word_lengths) if word_lengths else 0.0,
            "p50": _percentile(word_lengths, 0.50),
            "p90": _percentile(word_lengths, 0.90),
            "p99": _percentile(word_lengths, 0.99),
            "max": max(word_lengths, default=0),
        },
        "examples": examples[: args.examples],
        "failures": failures[:100],
        "failure_count": len(failures),
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(output)
    print(payload, end="")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
