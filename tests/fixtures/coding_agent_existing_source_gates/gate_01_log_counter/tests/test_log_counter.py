from pathlib import Path

import log_counter


def test_count_severities_skips_malformed_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "app.log"
    log_path.write_text(
        "\n".join([
            "INFO started",
            "ERROR failed",
            "DEBUG details",
            "BROKEN",
        ]),
        encoding="utf-8",
    )

    counts, skipped = log_counter.count_severities(log_path)

    assert counts["INFO"] == 1
    assert counts["ERROR"] == 1
    assert counts["DEBUG"] == 1
    assert skipped == 1


def test_format_summary_lists_all_severities() -> None:
    counts = {severity: 0 for severity in log_counter.SEVERITIES}
    counts["WARNING"] = 2

    summary = log_counter.format_summary(counts, skipped=3)

    assert "DEBUG: 0" in summary
    assert "WARNING: 2" in summary
    assert "skipped: 3" in summary


def test_main_reports_missing_file(capsys) -> None:
    status = log_counter.main(["missing.log"])

    captured = capsys.readouterr()
    assert status == 2
    assert "missing input file" in captured.err
