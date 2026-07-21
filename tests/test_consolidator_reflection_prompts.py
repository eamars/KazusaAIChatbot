"""Tests for the V2 consolidation ownership boundary."""

from __future__ import annotations

from pathlib import Path

from kazusa_ai_chatbot.consolidation import reflection as reflection_module


def test_reflection_has_no_cognition_state_authoring_llms() -> None:
    """Consolidation cannot author relationship or character cognition state."""

    assert not hasattr(reflection_module, "global_state_updater")
    assert not hasattr(reflection_module, "relationship_recorder")
    assert not hasattr(reflection_module, "_global_state_updater_llm")
    assert not hasattr(reflection_module, "_relationship_recorder_llm")


def test_reflection_source_contains_no_removed_writer_contracts() -> None:
    """Removed prose-affect write lanes stay absent from implementation text."""

    source = Path(reflection_module.__file__).read_text(encoding="utf-8")

    assert "relationship_delta" not in source
    assert "vibe_check" not in source
    assert "character_reflection" not in source
