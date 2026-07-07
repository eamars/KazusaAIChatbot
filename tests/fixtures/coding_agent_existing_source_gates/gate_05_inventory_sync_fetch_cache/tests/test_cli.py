from pathlib import Path

from inventory_sync import cli


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return b"<html><title>Vendor</title><h1>Widget</h1></html>"


def test_cli_writes_report(tmp_path: Path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "inventory.csv"
    output_path = tmp_path / "report.csv"
    input_path.write_text(
        "sku,name,url\n1,Widget,https://vendor.example/item\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url: FakeResponse(),
    )

    status = cli.main(["--input", str(input_path), "--output", str(output_path)])

    captured = capsys.readouterr()
    assert status == 0
    assert "wrote 1 rows" in captured.out
    assert "Widget" in output_path.read_text(encoding="utf-8")
