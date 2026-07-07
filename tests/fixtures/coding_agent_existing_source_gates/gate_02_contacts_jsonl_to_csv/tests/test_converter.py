import csv
from pathlib import Path

from contacts_jsonl_to_csv.converter import convert_jsonl_to_csv


def test_convert_jsonl_to_csv_writes_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "contacts.jsonl"
    output_path = tmp_path / "contacts.csv"
    input_path.write_text(
        '{"id": "1", "name": "Ada"}\n{"id": "2", "name": "Lin"}\n',
        encoding="utf-8",
    )

    converted_count = convert_jsonl_to_csv(input_path, output_path)

    rows = list(csv.DictReader(output_path.open(encoding="utf-8")))
    assert converted_count == 2
    assert rows[0]["name"] == "Ada"


def test_convert_jsonl_to_csv_accepts_fields(tmp_path: Path) -> None:
    input_path = tmp_path / "contacts.jsonl"
    output_path = tmp_path / "contacts.csv"
    input_path.write_text('{"name": "Ada", "id": "1"}\n', encoding="utf-8")

    convert_jsonl_to_csv(input_path, output_path, fields=["name", "id"])

    header = output_path.read_text(encoding="utf-8").splitlines()[0]
    assert header == "id,name"
