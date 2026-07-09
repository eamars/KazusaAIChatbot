from pathlib import Path

from counter_cli.cli import count_lines, render_text


def test_count_lines_counts_rows(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("one\ntwo\nthree\n", encoding="utf-8")

    assert count_lines(source) == 3


def test_render_text_includes_filename_and_count() -> None:
    assert render_text(Path("sample.txt"), 3) == "sample.txt: 3 lines"
