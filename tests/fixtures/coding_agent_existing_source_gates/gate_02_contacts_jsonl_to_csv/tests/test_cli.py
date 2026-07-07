from pathlib import Path

from contacts_jsonl_to_csv import cli


def test_cli_converts_jsonl(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "contacts.jsonl"
    output_path = tmp_path / "contacts.csv"
    input_path.write_text('{"id": "1", "name": "Ada"}\n', encoding="utf-8")

    status = cli.main([str(input_path), str(output_path)])

    captured = capsys.readouterr()
    assert status == 0
    assert "converted 1 records" in captured.out
    assert output_path.read_text(encoding="utf-8").startswith("id,name")
