"""Build consolidated inventory report rows."""

from __future__ import annotations


def build_report_rows(
    inventory_rows: list[dict[str, str]],
    fetched_pages: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Merge inventory rows with fetched page metadata."""

    report_rows: list[dict[str, str]] = []
    for row in inventory_rows:
        metadata = fetched_pages[row["url"]]
        report_rows.append({
            "sku": row["sku"],
            "name": row["name"],
            "url": row["url"],
            "title": metadata["title"],
            "h1": metadata["h1"],
        })

    return report_rows
