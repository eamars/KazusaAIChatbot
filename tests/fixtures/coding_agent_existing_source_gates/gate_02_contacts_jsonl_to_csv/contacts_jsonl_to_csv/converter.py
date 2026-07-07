"""Convert JSONL objects into CSV rows."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


def convert_jsonl_to_csv(
    input_path: Path,
    output_path: Path,
    *,
    fields: list[str] | None = None,
) -> int:
    """Convert JSON object lines into a CSV file."""

    records: list[dict[str, Any]] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"malformed JSON skipped: {exc}", file=sys.stderr)
            continue
        if isinstance(record, dict):
            records.append(record)

    if not records:
        output_path.write_text("", encoding="utf-8")
        return 0

    if fields is None:
        fieldnames = list(records[0].keys())
    else:
        fieldnames = sorted(fields)

    with output_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row: dict[str, str] = {}
            for fieldname in fieldnames:
                row[fieldname] = str(record.get(fieldname))
            writer.writerow(row)

    converted_count = len(records)
    return converted_count
