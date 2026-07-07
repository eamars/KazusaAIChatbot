"""Command-line interface for JSONL to CSV conversion."""

from __future__ import annotations

import argparse
from pathlib import Path

from contacts_jsonl_to_csv.converter import convert_jsonl_to_csv


def _parse_fields(raw_fields: str | None) -> list[str] | None:
    if raw_fields is None:
        return None
    fields = [
        field.strip()
        for field in raw_fields.split(",")
        if field.strip()
    ]
    return fields


def main(argv: list[str] | None = None) -> int:
    """Run the converter CLI."""

    parser = argparse.ArgumentParser(description="Convert JSONL contacts to CSV.")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--fields")
    args = parser.parse_args(argv)

    fields = _parse_fields(args.fields)
    converted_count = convert_jsonl_to_csv(
        args.input_path,
        args.output_path,
        fields=fields,
    )
    print(f"converted {converted_count} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
