"""Static tests for message-envelope module boundaries."""

from __future__ import annotations

import ast
from pathlib import Path


def test_production_code_does_not_import_envelope_implementation_packages() -> None:
    """Consumers must depend on public protocols, not concrete implementations."""

    source_root = Path("src/kazusa_ai_chatbot")
    internal_prefixes = (
        "kazusa_ai_chatbot.message_envelope.normalizers",
        "kazusa_ai_chatbot.message_envelope.attachment_handlers",
    )
    violations = []
    for path in source_root.rglob("*.py"):
        if "message_envelope" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(internal_prefixes):
                        violations.append(f"{path}:{node.lineno}:{alias.name}")
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith(internal_prefixes):
                    violations.append(f"{path}:{node.lineno}:{module}")

    assert violations == []


def test_message_envelope_package_has_no_platform_normalizer_modules() -> None:
    """Concrete platform normalizers belong to adapters, not the brain package."""

    normalizer_dir = Path("src/kazusa_ai_chatbot/message_envelope/normalizers")

    assert not normalizer_dir.exists()


def test_service_does_not_import_platform_envelope_implementations() -> None:
    """Service may use the public envelope API but not implementation packages."""

    path = Path("src/kazusa_ai_chatbot/service.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_prefixes = (
        "kazusa_ai_chatbot.message_envelope.normalizers",
        "kazusa_ai_chatbot.message_envelope.attachment_handlers",
    )
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(forbidden_prefixes):
                    violations.append(f"{path}:{node.lineno}:{alias.name}")
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith(forbidden_prefixes):
                violations.append(f"{path}:{node.lineno}:{module}")

    assert violations == []
