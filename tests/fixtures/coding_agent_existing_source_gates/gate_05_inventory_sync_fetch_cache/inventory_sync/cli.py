"""Command-line interface for inventory synchronization."""

from __future__ import annotations

import argparse
from pathlib import Path

from inventory_sync.csv_io import read_inventory, write_report
from inventory_sync.fetch import fetch_page
from inventory_sync.html_extract import extract_page_metadata
from inventory_sync.report import build_report_rows


def main(argv: list[str] | None = None) -> int:
    """Fetch vendor metadata and write a consolidated report."""

    parser = argparse.ArgumentParser(description="Build an inventory page report.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    inventory_rows = read_inventory(args.input)
    fetched_pages = {}
    for row in inventory_rows:
        html = fetch_page(row["url"])
        fetched_pages[row["url"]] = extract_page_metadata(html)

    report_rows = build_report_rows(inventory_rows, fetched_pages)
    write_report(args.output, report_rows)
    print(f"wrote {len(report_rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
