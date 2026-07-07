from inventory_sync.fetch import fetch_page


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return b"<html><title>Vendor</title></html>"


def test_fetch_page_decodes_html(monkeypatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url: FakeResponse(),
    )

    html = fetch_page("https://vendor.example/item")

    assert "<title>Vendor</title>" in html
