"""Standalone package, construction, and import-boundary tests."""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path

import pytest
from setuptools import find_packages

from agentic_resolver.contracts import AgenticResolverContractError
from agentic_resolver.integrations import create_kazusa_resolver_runtime
from agentic_resolver.runtime import AgenticResolverRuntime

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
CORE_ROOT = SOURCE_ROOT / "agentic_resolver"


def test_runtime_resolves_without_brain_service_or_cognition_imports() -> None:
    """A fresh process resolves directly while workflow modules stay unloaded."""

    script = r'''
import asyncio
import json
import sys

from agentic_resolver import (
    AgenticModelCapabilitiesV1,
    AgenticResolverRequestV1,
    AgenticResolverRuntime,
    ModelStreamChunk,
    ToolRegistry,
    discover_skills,
)
from agentic_resolver.model import ModelStreamFinish


class Model:
    capabilities = AgenticModelCapabilitiesV1(
        thinking_strategy="fresh_process_test",
        reasoning_replay_policy="fresh_process_test",
    )

    async def astream(self, messages, *, tools):
        del messages, tools
        arguments = json.dumps({
            "status": "resolved",
            "summary": "Fresh process resolved directly.",
            "evidence": [],
            "completed_tasks": ["Resolve directly."],
            "remaining_needs": [],
        })
        yield ModelStreamChunk(
            kind="block_start",
            block_index=0,
            block_type="reasoning",
        )
        yield ModelStreamChunk(
            kind="reasoning_delta",
            block_index=0,
            block_type="reasoning",
            reasoning_delta="opaque",
        )
        yield ModelStreamChunk(
            kind="block_end",
            block_index=0,
            block_type="reasoning",
            completed_block={"type": "reasoning"},
        )
        yield ModelStreamChunk(
            kind="block_start",
            block_index=1,
            block_type="tool_call",
        )
        yield ModelStreamChunk(
            kind="tool_call_delta",
            block_index=1,
            block_type="tool_call",
            tool_call_id="submit-1",
            tool_name="submit_result",
            tool_arguments_delta=arguments,
        )
        yield ModelStreamChunk(
            kind="block_end",
            block_index=1,
            block_type="tool_call",
            completed_block={"type": "tool_call"},
        )
        yield ModelStreamChunk(
            kind="finish",
            finish=ModelStreamFinish(reason="tool_calls"),
        )


async def main():
    runtime = AgenticResolverRuntime(
        model=Model(),
        tools=ToolRegistry(),
        skills=discover_skills([]),
    )
    result = await runtime.resolve(
        AgenticResolverRequestV1(objective="Resolve directly.")
    )
    assert result.status == "resolved"
    forbidden = [
        name
        for name in sys.modules
        if name.startswith("kazusa_ai_chatbot.brain_service")
        or name.startswith("kazusa_ai_chatbot.cognition")
    ]
    assert forbidden == [], forbidden


asyncio.run(main())
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_runtime_factory_requires_explicit_model_tools_and_skill_roots() -> None:
    """Direct and Kazusa factories expose required dependency injection."""

    runtime_signature = inspect.signature(AgenticResolverRuntime)
    integration_signature = inspect.signature(create_kazusa_resolver_runtime)

    assert runtime_signature.parameters["model"].default is inspect.Parameter.empty
    assert runtime_signature.parameters["tools"].default is inspect.Parameter.empty
    assert runtime_signature.parameters["skills"].default is inspect.Parameter.empty
    for name in (
        "llm_interface",
        "llm_config",
        "execution_context",
        "skill_roots",
    ):
        parameter = integration_signature.parameters[name]
        assert parameter.default is inspect.Parameter.empty
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_runtime_rejects_disabled_or_unsupported_thinking() -> None:
    """Runtime admission requires the immutable supported capability type."""

    class _UnsupportedModel:
        def __init__(self) -> None:
            self.capabilities = {
                "streaming": True,
                "thinking_enabled": False,
            }

        async def astream(self, messages, *, tools):
            del messages, tools
            if False:
                yield None

    with pytest.raises(
        AgenticResolverContractError,
        match="AgenticModelCapabilitiesV1",
    ):
        AgenticResolverRuntime(
            model=_UnsupportedModel(),
            tools=__import__("agentic_resolver").ToolRegistry(),
            skills=__import__("agentic_resolver").discover_skills([]),
        )


def test_current_workflow_sources_do_not_import_agentic_resolver() -> None:
    """Every retained live workflow remains disconnected in the first pass."""

    workflow_roots = (
        "brain_service",
        "cognition_resolver",
        "cognition_core_v3",
        "nodes",
        "task_resolution",
        "local_context_resolver",
        "complex_task_resolver",
        "accepted_task",
        "background_work",
    )
    matches: list[str] = []
    for relative_root in workflow_roots:
        root = SOURCE_ROOT / "kazusa_ai_chatbot" / relative_root
        for path in root.rglob("*.py"):
            if "agentic_resolver" in path.read_text(encoding="utf-8"):
                matches.append(str(path.relative_to(REPOSITORY_ROOT)))

    assert matches == []


def test_core_modules_keep_kazusa_imports_inside_integrations() -> None:
    """Core package imports remain provider- and application-neutral."""

    violations: list[str] = []
    for path in CORE_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            else:
                continue
            if any(name.startswith("kazusa_ai_chatbot") for name in imported):
                violations.append(str(path.relative_to(REPOSITORY_ROOT)))

    assert violations == []


def test_distribution_discovers_agentic_resolver_package() -> None:
    """Configured source package discovery includes core and integrations."""

    packages = set(find_packages(where=str(SOURCE_ROOT)))
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "agentic_resolver" in packages
    assert "agentic_resolver.integrations" in packages
    assert '"agentic_resolver*"' in pyproject
