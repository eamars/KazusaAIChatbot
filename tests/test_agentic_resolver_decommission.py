"""Static clean-replacement and Brain non-impact tests."""

from __future__ import annotations

import importlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOLVER_ROOT = PROJECT_ROOT / "src" / "agentic_resolver"


def test_old_resolver_contracts_facades_and_aliases_are_absent() -> None:
    legacy_paths = (
        "context_budget.py", "integrations/__init__.py",
        "integrations/kazusa_tools.py", "integrations/llm_interface.py",
        "json_protocol.py", "loop.py",
        "model.py", "session.py", "skills.py", "streaming.py", "subagents.py",
        "tools.py", "resolver_skills",
    )
    assert all(not (RESOLVER_ROOT / path).exists() for path in legacy_paths)
    package = importlib.import_module("agentic_resolver")
    legacy_exports = {
        "AgentLoop", "AgentSession", "ToolRegistry", "SkillRegistry",
        "AgenticModelClient", "ModelStreamChunk",
    }
    assert not legacy_exports.intersection(package.__dict__)


def test_brain_task_resolution_rag_and_coding_paths_do_not_import_or_spawn_resolver() -> None:
    roots = (
        PROJECT_ROOT / "src" / "kazusa_ai_chatbot" / "brain_service.py",
        PROJECT_ROOT / "src" / "kazusa_ai_chatbot" / "task_resolution",
        PROJECT_ROOT / "src" / "kazusa_ai_chatbot" / "rag",
        PROJECT_ROOT / "src" / "kazusa_ai_chatbot" / "coding_agent",
    )
    for root in roots:
        paths = [root] if root.is_file() else list(root.rglob("*.py"))
        for path in paths:
            text = path.read_text(encoding="utf-8")
            assert "agentic_resolver" not in text
            assert "sidecars/dsh_resolution" not in text


def test_old_native_tool_stream_surface_is_absent() -> None:
    package = importlib.import_module("kazusa_ai_chatbot.llm_interface")
    assert "LLMStreamChunk" not in package.__dict__
    assert "LLMToolDefinition" not in package.__dict__
    assert "astream_tools" not in (
        PROJECT_ROOT / "src" / "kazusa_ai_chatbot" / "llm_interface" /
        "interface.py"
    ).read_text(encoding="utf-8")


def test_legacy_resolver_dependency_and_package_files_are_absent() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "PyYAML" not in pyproject
    assert not (PROJECT_ROOT / "tests" / "test_llm_interface_tool_stream.py").exists()
