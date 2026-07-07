"""CSV input and output helpers for inventory reports."""

from __future__ import annotations

import csv
from pathlib import Path


REPORT_FIELDS = ["sku", "name", "url", "title", "h1"]


def read_inventory(path: Path) -> list[dict[str, str]]:
    """Read inventory rows from a CSV file."""

    with path.open(encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        rows = list(reader)
    return rows


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    """Write consolidated inventory report rows."""

    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
