"""Optional downward-only integrations for direct runtime composition."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from agentic_resolver.contracts import AgenticResolverLimitsV1
from agentic_resolver.integrations.kazusa_tools import (
    build_kazusa_tool_registry,
)
from agentic_resolver.integrations.llm_interface import LLInterfaceToolModel
from agentic_resolver.runtime import AgenticResolverRuntime
from agentic_resolver.skills import discover_skills
from kazusa_ai_chatbot.llm_interface import LLInterface, LLMCallConfig
from kazusa_ai_chatbot.task_resolution.contracts import (
    TaskResolutionExecutionContextV1,
)

__all__ = [
    "LLInterfaceToolModel",
    "build_kazusa_tool_registry",
    "create_kazusa_resolver_runtime",
]


def create_kazusa_resolver_runtime(
    *,
    llm_interface: LLInterface,
    llm_config: LLMCallConfig,
    execution_context: TaskResolutionExecutionContextV1,
    skill_roots: Sequence[str | Path],
    limits: AgenticResolverLimitsV1 | None = None,
) -> AgenticResolverRuntime:
    """Compose a direct-call resolver without application registration."""

    effective_limits = limits or AgenticResolverLimitsV1()
    if llm_config.context_window_tokens is not None:
        effective_context_window = min(
            effective_limits.context_window_tokens,
            llm_config.context_window_tokens,
        )
        effective_limits = replace(
            effective_limits,
            context_window_tokens=effective_context_window,
        )
    model = LLInterfaceToolModel(
        llm_interface=llm_interface,
        llm_config=llm_config,
    )
    tools = build_kazusa_tool_registry(execution_context)
    skills = discover_skills(skill_roots, limits=effective_limits)
    runtime = AgenticResolverRuntime(
        model=model,
        tools=tools,
        skills=skills,
        limits=effective_limits,
    )
    return runtime
