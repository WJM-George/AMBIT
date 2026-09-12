#!/usr/bin/env python3
"""Validate or export the canonical spatial-caption template corpus."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from spatial_template_corpus import (  # noqa: E402
    DEFAULT_CORPUS_PATH,
    corpus_to_dict,
    generate_template_corpus,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--target-total", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    corpus = generate_template_corpus(args.target_total, args.seed)
    ids = [template.template_id for template in corpus]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate template ids")
    counts = Counter(template.stage for template in corpus)
    missing = set(range(1, 8)) - set(counts)
    if missing:
        raise SystemExit(f"corpus lacks stages: {sorted(missing)}")

    payload = corpus_to_dict(corpus)
    print(
        json.dumps(
            {
                "templates": len(corpus),
                "per_stage": dict(sorted(counts.items())),
                "source": str(DEFAULT_CORPUS_PATH),
                "output": str(args.out),
            },
            sort_keys=True,
        )
    )
    if args.check_only:
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)


if __name__ == "__main__":
    main()
