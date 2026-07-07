from inventory_sync.report import build_report_rows


def test_build_report_rows_merges_metadata() -> None:
    rows = [{
        "sku": "1",
        "name": "Widget",
        "url": "https://vendor.example/item",
    }]
    fetched_pages = {
        "https://vendor.example/item": {
            "title": "Vendor",
            "h1": "Widget Detail",
        },
    }

    report_rows = build_report_rows(rows, fetched_pages)

    assert report_rows == [{
        "sku": "1",
        "name": "Widget",
        "url": "https://vendor.example/item",
        "title": "Vendor",
        "h1": "Widget Detail",
    }]
