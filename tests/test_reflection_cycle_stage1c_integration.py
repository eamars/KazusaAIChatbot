"""Integration-style static checks for production reflection cycle."""

from __future__ import annotations

from pathlib import Path

from kazusa_ai_chatbot import reflection_cycle


def test_public_stage1c_facades_are_exported() -> None:
    """Service and CLI should have package-level reflection facades."""

    for name in (
        "run_hourly_reflection_cycle",
        "run_daily_channel_reflection_cycle",
        "run_global_reflection_promotion",
        "build_promoted_reflection_context",
        "start_reflection_cycle_worker",
        "stop_reflection_cycle_worker",
    ):
        assert hasattr(reflection_cycle, name)
        assert name in reflection_cycle.__all__


def test_stage1c_writers_do_not_use_legacy_memory_path() -> None:
    """Reflection production writers must not use the legacy memory path."""

    roots = [
        Path("src/kazusa_ai_chatbot/reflection_cycle"),
        Path("src/scripts/run_reflection_cycle.py"),
    ]
    offenders: list[str] = []
    for path in _python_files(roots):
        source = path.read_text(encoding="utf-8")
        for token in ("build_memory_doc", "save_memory"):
            if token in source:
                offenders.append(f"{path}:{token}")
    assert offenders == []


def test_stage1c_reflection_package_respects_storage_boundary() -> None:
    """Reflection package and CLI must not contain raw storage calls."""

    roots = [
        Path("src/kazusa_ai_chatbot/reflection_cycle"),
        Path("src/scripts/run_reflection_cycle.py"),
    ]
    forbidden = (
        "get_db",
        ".find(",
        ".aggregate(",
        "insert_one(",
        "update_one(",
        "update_many(",
        "delete_one(",
        "delete_many(",
        "replace_one(",
        "count_documents(",
    )
    offenders: list[str] = []
    for path in _all_files(roots):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                offenders.append(f"{path}:{token}")
    assert offenders == []


def test_readme_documents_production_interface_controls() -> None:
    """The reflection ICD should describe production boundaries and flags."""

    readme = Path("src/kazusa_ai_chatbot/reflection_cycle/README.md").read_text(
        encoding="utf-8",
    )

    assert "Public Facades" in readme
    assert "DB Boundaries" in readme
    assert "Memory Boundary" in readme
    assert "Worker Schedule" in readme
    assert "REFLECTION_CYCLE_DISABLED=false" in readme
    assert "CONSOLIDATION_LLM_BASE_URL" in readme


def test_service_uses_reflection_package_facades() -> None:
    """Service wiring should depend only on package-level reflection exports."""

    source = Path("src/kazusa_ai_chatbot/service.py").read_text(
        encoding="utf-8",
    )

    forbidden_imports = (
        "kazusa_ai_chatbot.reflection_cycle.context",
        "kazusa_ai_chatbot.reflection_cycle.models",
        "kazusa_ai_chatbot.reflection_cycle.worker",
        "kazusa_ai_chatbot.reflection_cycle.repository",
        "kazusa_ai_chatbot.reflection_cycle.promotion",
    )
    offenders = [
        import_path
        for import_path in forbidden_imports
        if import_path in source
    ]
    assert offenders == []


def _python_files(roots: list[Path]) -> list[Path]:
    """Return Python files under mixed file and directory roots."""

    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            files.append(root)
        elif root.is_dir():
            files.extend(root.glob("*.py"))
    return files


def _all_files(roots: list[Path]) -> list[Path]:
    """Return all files under mixed file and directory roots."""

    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(path for path in root.iterdir() if path.is_file())
    return files
