"""Tests for the project artifact counting script."""

from __future__ import annotations

from pathlib import Path

from scripts import count_project_artifacts as count_module


def _write(path: Path, text: str) -> None:
    """Create a text file for artifact-count tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_artifact_report_counts_source_tests_and_documents(
    tmp_path: Path,
) -> None:
    """Artifact report should count the three requested project categories."""

    _write(tmp_path / "src" / "kazusa_ai_chatbot" / "service.py", "a\nb\n")
    _write(tmp_path / "src" / "adapters" / "debug_adapter.py", "a\n")
    _write(tmp_path / "tests" / "test_service.py", "a\nb\nc\n")
    _write(tmp_path / "README.md", "# Title\n\nBody\n")
    _write(tmp_path / "docs" / "HOWTO.md", "Run it\n")
    _write(tmp_path / "LICENSE", "license text\n")

    report = count_module.build_artifact_report(tmp_path)

    assert report["production_code"]["files"] == 2
    assert report["production_code"]["lines"] == 3
    assert report["unit_test_code"]["files"] == 1
    assert report["unit_test_code"]["lines"] == 3
    assert report["documents"]["files"] == 3
    assert report["documents"]["lines"] == 5


def test_build_artifact_report_ignores_generated_and_cache_directories(
    tmp_path: Path,
) -> None:
    """Artifact report should skip local outputs that are not source assets."""

    _write(tmp_path / "src" / "kazusa_ai_chatbot" / "service.py", "a\n")
    _write(tmp_path / "venv" / "Lib" / "site-packages" / "dep.py", "a\n")
    _write(tmp_path / ".pytest_cache" / "README.md", "cache\n")
    _write(tmp_path / "test_artifacts" / "trace.md", "generated\n")
    _write(tmp_path / "__pycache__" / "module.py", "cache\n")

    report = count_module.build_artifact_report(tmp_path)

    assert report["production_code"] == {"files": 1, "lines": 1}
    assert report["unit_test_code"] == {"files": 0, "lines": 0}
    assert report["documents"] == {"files": 0, "lines": 0}


def test_format_artifact_report_prints_stable_category_rows() -> None:
    """Formatted report should expose stable category names and totals."""

    report = {
        "production_code": {"files": 2, "lines": 30},
        "unit_test_code": {"files": 1, "lines": 12},
        "documents": {"files": 3, "lines": 44},
    }

    rendered = count_module.format_artifact_report(report)

    assert "production_code" in rendered
    assert "unit_test_code" in rendered
    assert "documents" in rendered
    assert "2" in rendered
    assert "44" in rendered
