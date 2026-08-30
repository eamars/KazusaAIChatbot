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


def test_brain_task_resolution_and_rag_paths_use_only_canonical_dsh_bridge() -> None:
    """Retained paths have no deleted executor or compatibility imports."""

    roots = (
        PROJECT_ROOT / "src" / "kazusa_ai_chatbot" / "brain_service",
        PROJECT_ROOT / "src" / "kazusa_ai_chatbot" / "task_resolution",
        PROJECT_ROOT / "src" / "kazusa_ai_chatbot" / "rag",
    )
    forbidden = (
        "task_resolution.orchestrator",
        "task_resolution.specialists",
        "complex_task_resolver",
        "sidecars/dsh_resolution",
    )
    for root in roots:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else list(root.rglob("*.py"))
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                assert marker not in text


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


def test_plan2_adds_only_interaction_bridge_and_does_not_cut_over_task_resolution() -> None:
    """The interaction bridge keeps existing task resolution callers intact."""

    task_resolution_root = PROJECT_ROOT / "src" / "kazusa_ai_chatbot" / "task_resolution"
    task_sources = [
        path.read_text(encoding="utf-8")
        for path in task_resolution_root.rglob("*.py")
    ]
    assert task_sources
    assert all("dsh_interaction" not in source for source in task_sources)
    service_source = (
        PROJECT_ROOT / "src" / "kazusa_ai_chatbot" / "service.py"
    ).read_text(encoding="utf-8")
    assert any("task_resolution_request" in source for source in task_sources)
    assert "accepted_task" in service_source
    interaction_source = (
        PROJECT_ROOT / "src" / "kazusa_ai_chatbot" / "dsh_interaction" / "service.py"
    ).read_text(encoding="utf-8")
    assert "BrainInteractionService" in interaction_source
