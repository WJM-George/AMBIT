#!/usr/bin/env python3
"""Build a self-contained HTML table for worst-case audio-evaluation rows."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


def _number(value: str) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else math.inf
    except (TypeError, ValueError):
        return math.inf


def _audio_cell(value: str, page_dir: Path) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    try:
        source = path.resolve().relative_to(page_dir.resolve()).as_posix()
    except ValueError:
        source = path.resolve().as_uri() if path.exists() else value
    escaped = html.escape(source, quote=True)
    return f'<audio controls preload="none" src="{escaped}"></audio>'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--higher-is-worse",
        action="store_true",
        help="Sort descending; default assumes lower metric values are worse.",
    )
    parser.add_argument(
        "--audio-columns",
        default="source_foa,recon_foa,source,recon,audio_path,wav_path",
    )
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("CSV contains no rows")
    if args.metric not in rows[0]:
        raise SystemExit(f"metric column not found: {args.metric}")

    rows.sort(
        key=lambda row: _number(row.get(args.metric, "")),
        reverse=args.higher_is_worse,
    )
    rows = rows[: args.limit]
    columns = list(rows[0])
    audio_columns = set(filter(None, args.audio_columns.split(",")))

    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = str(row.get(column, ""))
            rendered = (
                _audio_cell(value, args.out.parent)
                if column in audio_columns and value
                else html.escape(value)
            )
            cells.append(f"<td>{rendered}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")

    document = f"""<!doctype html>
<html lang="en"><meta charset="utf-8">
<title>Worst cases: {html.escape(args.metric)}</title>
<style>
body{{font:14px system-ui;margin:24px;background:#fafafa;color:#222}}
table{{border-collapse:collapse;width:100%;background:white}}
th,td{{border:1px solid #ddd;padding:7px;vertical-align:top}}
th{{position:sticky;top:0;background:#eee}} audio{{width:260px}}
</style>
<h1>Worst cases: {html.escape(args.metric)}</h1>
<p>Source: {html.escape(str(args.csv))}; rows: {len(rows)}</p>
<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>
</html>
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(document, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
