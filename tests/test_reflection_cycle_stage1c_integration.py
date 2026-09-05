"""Integration-style static checks for production reflection cycle."""

from __future__ import annotations

from pathlib import Path









def test_readme_documents_production_interface_controls() -> None:
    """The reflection ICD should describe production boundaries and flags."""

    readme = Path("src/kazusa_ai_chatbot/reflection_cycle/README.md").read_text(
        encoding="utf-8",
    )

    assert "Public Facades" in readme
    assert "DB Boundaries" in readme
    assert "Memory Boundary" in readme
    assert "Worker Schedule" in readme
    assert "REFLECTION_CYCLE_ENABLED=true" in readme
    assert "CONSOLIDATION_LLM_BASE_URL" in readme






