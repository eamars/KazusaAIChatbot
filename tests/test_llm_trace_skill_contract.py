from __future__ import annotations

from pathlib import Path


def test_llm_trace_debug_skill_points_to_export_scripts():
    skill_path = Path(".agents/skills/llm-trace-debug/SKILL.md")

    assert skill_path.exists()
    text = skill_path.read_text(encoding="utf-8")
    assert "scripts.export_llm_trace" in text
    assert "scripts.export_dialog_trace_review_input" in text
    assert "LLM_TRACE_CAPTURE_MODE=metadata" in text
