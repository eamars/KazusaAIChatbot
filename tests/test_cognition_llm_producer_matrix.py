"""Governance test for current LLM producer ownership."""

from __future__ import annotations

import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = REPOSITORY_ROOT / "tests" / "fixtures" / "cognition_llm_producer_matrix.json"


def test_producer_matrix_matches_current_source_owners() -> None:
    """The producer matrix retains live owners and removes retired routes."""

    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    rules = matrix["call_site_rules"]
    patterns = {row["pattern"] for row in rules}
    routes = set(matrix["required_semantic_routes"])

    assert "src/kazusa_ai_chatbot/background_work/**" in patterns
    assert "src/kazusa_ai_chatbot/media_inspection/**" in patterns
    assert "src/kazusa_ai_chatbot/complex_task_resolver/**" not in patterns
    assert "src/kazusa_ai_chatbot/coding_agent/**" not in patterns
    assert "dsh_task_resolution" in routes
    assert "coding_agent" not in routes
