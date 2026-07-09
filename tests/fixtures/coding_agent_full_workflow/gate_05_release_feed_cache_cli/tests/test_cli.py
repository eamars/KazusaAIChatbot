import json

from release_feed.cli import main


def test_cli_include_drafts_prints_draft_releases(tmp_path, capsys) -> None:
    source = tmp_path / "releases.json"
    source.write_text(
        json.dumps([
            {"title": "stable", "draft": False},
            {"title": "draft", "draft": True},
        ]),
        encoding="utf-8",
    )

    status = main([str(source), "--include-drafts"])

    assert status == 0
    assert capsys.readouterr().out.splitlines() == ["stable", "draft"]
