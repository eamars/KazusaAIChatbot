"""Static tests for conversation progress module boundaries."""

from __future__ import annotations

import ast
from pathlib import Path


def test_production_code_imports_only_public_conversation_progress_facade() -> None:
    """Production callers must not import conversation_progress internals."""

    source_root = Path("src/kazusa_ai_chatbot")
    internal_modules = {
        "kazusa_ai_chatbot.conversation_progress.repository",
        "kazusa_ai_chatbot.conversation_progress.cache",
        "kazusa_ai_chatbot.conversation_progress.projection",
        "kazusa_ai_chatbot.conversation_progress.recorder",
    }
    violations = []
    for path in source_root.rglob("*.py"):
        if "conversation_progress" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in internal_modules:
                        violations.append(f"{path}:{node.lineno}:{alias.name}")
            if isinstance(node, ast.ImportFrom) and node.module in internal_modules:
                violations.append(f"{path}:{node.lineno}:{node.module}")

    assert violations == []
