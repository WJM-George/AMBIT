#!/usr/bin/env python3
"""Classify the TTS-background safety flag from stored A2T descriptions.

This stage never edits a source description.  It emits one deterministic
boolean used only to prevent a formal TTS scene from also containing clearly
spoken language inside a music/sound donor. Singing and rap remain music.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any


DEFAULT_POLICY = Path(__file__).with_name("spoken_language_policy_v1.txt")
POLICY_VERSION = "spoken_language_background_v1"
CLASSIFIER_REVISION = "spoken_language_explicit_lexical_v2"
SPOKEN_EVIDENCE_RE = re.compile(
    r"\b("
    r"speaks?|speaking|spoken|says?|saying|said|talks?|talking|"
    r"conversation|conversing|dialogue|dialog|"
    r"narrat(?:e|es|ed|ing|ion|or)|"
    r"announc(?:e|es|ed|ing|ement|er)|"
    r"audiobook|podcast|interview|speech|"
    r"whispers?|whispering|asks?|asking|replies?|replying|"
    r"instructions?|commands?|"
    r"public[- ]address|radio host|newsreader"
    r")\b",
    flags=re.IGNORECASE,
)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from error
                annotation_id = str(raw.get("id") or raw.get("annotation_id") or "")
                description = str(raw.get("source_description") or "").strip()
                if not annotation_id or not description:
                    raise ValueError(f"missing id/description at {path}:{line_number}")
                if annotation_id in seen:
                    raise ValueError(f"duplicate annotation id: {annotation_id}")
                seen.add(annotation_id)
                rows.append(
                    {
                        "id": annotation_id,
                        "source_description": description,
                    }
                )
    if not rows:
        raise ValueError("no descriptions to classify")
    return rows


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                annotation_id = str(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"invalid existing output at {path}:{line_number}") from error
            if annotation_id in done:
                raise ValueError(f"duplicate existing output id: {annotation_id}")
            done.add(annotation_id)
    return done


def classify_description(text: str) -> tuple[bool, str]:
    match = SPOKEN_EVIDENCE_RE.search(text)
    if match:
        return True, f"true:explicit_term:{match.group(0).casefold()}"
    return False, "false:no_explicit_spoken_language_term"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Rebuild the complete label file into a temporary file and atomically "
            "replace the old output. Use this after an annotation retry changes a "
            "source-description hash."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    input_paths = [path.expanduser().resolve(strict=True) for path in args.input_jsonl]
    output = args.out.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    policy_path = args.policy_path.expanduser().resolve(strict=True)
    policy = policy_path.read_text(encoding="utf-8")
    policy_sha256 = hashlib.sha256(policy.encode("utf-8")).hexdigest()
    rows = load_rows(input_paths)
    done = set() if args.overwrite else load_done(output)
    todo = [row for row in rows if row["id"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    if not todo:
        return 0

    started = time.monotonic()
    completed = 0
    true_count = 0
    write_path = (
        output.with_name(output.name + f".tmp.{os.getpid()}")
        if args.overwrite
        else output
    )
    write_mode = "w" if args.overwrite else "a"
    with write_path.open(write_mode, encoding="utf-8") as sink:
        for row in todo:
            value, raw_response = classify_description(row["source_description"])
            payload = {
                "schema": "stable_audio_tools.sceneplan_spoken_language_background_label",
                "schema_version": 1,
                "id": row["id"],
                "source_description_sha256": hashlib.sha256(
                    row["source_description"].encode("utf-8")
                ).hexdigest(),
                "spoken_language_background": value,
                "policy_version": POLICY_VERSION,
                "policy_sha256": policy_sha256,
                "method": "text_classifier",
                "classifier_revision": CLASSIFIER_REVISION,
                "classifier_engine": "python_regex",
                "classifier_engine_version": "1",
                "raw_response": raw_response,
            }
            sink.write(stable_json(payload) + "\n")
            completed += 1
            true_count += int(value)
        sink.flush()
        os.fsync(sink.fileno())
    if args.overwrite:
        os.replace(write_path, output)
    elapsed = time.monotonic() - started
    summary = {
        "schema": "stable_audio_tools.sceneplan_spoken_language_classifier_run",
        "schema_version": 1,
        "input_jsonl": [str(path) for path in input_paths],
        "input_rows": len(rows),
        "completed_this_run": completed,
        "spoken_language_background_true_this_run": true_count,
        "policy_version": POLICY_VERSION,
        "policy_path": str(policy_path),
        "policy_sha256": policy_sha256,
        "classifier_revision": CLASSIFIER_REVISION,
        "engine": "python_regex",
        "engine_version": "1",
        "elapsed_seconds": elapsed,
        "descriptions_per_second": completed / elapsed if elapsed else None,
        "description_rewrite": False,
        "atomic_overwrite": bool(args.overwrite),
    }
    temporary = output.with_suffix(".summary.json.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, output.with_suffix(".summary.json"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
