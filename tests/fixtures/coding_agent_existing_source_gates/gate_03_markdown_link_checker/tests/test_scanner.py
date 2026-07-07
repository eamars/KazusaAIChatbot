from pathlib import Path

from mdlinkcheck.scanner import check_file, find_markdown_links


def test_find_markdown_links() -> None:
    links = find_markdown_links("See [Guide](guide.md#intro).")

    assert links[0].target == "guide.md#intro"


def test_check_file_reports_broken_relative_link(tmp_path: Path) -> None:
    page = tmp_path / "index.md"
    page.write_text("[Missing](missing.md)\n", encoding="utf-8")

    problems = check_file(page, tmp_path)

    assert problems
    assert "missing target" in problems[0].message


def test_check_file_accepts_existing_anchor(tmp_path: Path) -> None:
    page = tmp_path / "index.md"
    guide = tmp_path / "guide.md"
    page.write_text("[Guide](guide.md#intro)\n", encoding="utf-8")
    guide.write_text("# Intro\n", encoding="utf-8")

    problems = check_file(page, tmp_path)

    assert problems == []
