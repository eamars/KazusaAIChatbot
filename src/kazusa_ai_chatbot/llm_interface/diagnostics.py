"""Data-only diagnostics for configured chat LLM routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from kazusa_ai_chatbot.config import DEFAULT_LLM_MAX_COMPLETION_TOKENS
from kazusa_ai_chatbot.llm_interface.contracts import (
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.llm_interface.interface import LLInterface


@dataclass(frozen=True)
class RouteDiagnostic:
    """Sanitized diagnostic row for one configured LLM route."""

    route_name: str
    backend: str
    model: str
    normalized_base_url: str
    model_family: str
    thinking_strategy: str
    required: bool
    fallback_backed: bool


def build_route_diagnostics(
    route_configs: Iterable[LLMCallConfig],
    *,
    required_routes: set[str],
    fallback_backed_routes: set[str],
) -> tuple[RouteDiagnostic, ...]:
    """Describe configured routes without exposing provider secrets."""

    interface = LLInterface()
    rows: list[RouteDiagnostic] = []
    for config in route_configs:
        descriptor = interface.describe_backend(config=config)
        row = RouteDiagnostic(
            route_name=config.route_name,
            backend=descriptor.backend_kind,
            model=descriptor.model,
            normalized_base_url=descriptor.normalized_base_url,
            model_family=descriptor.model_family,
            thinking_strategy=descriptor.thinking_strategy,
            required=config.route_name in required_routes,
            fallback_backed=config.route_name in fallback_backed_routes,
        )
        rows.append(row)
    diagnostics = tuple(rows)
    return diagnostics


def default_max_completion_tokens() -> int:
    """Return the shared default chat completion budget."""

    return DEFAULT_LLM_MAX_COMPLETION_TOKENS


def disabled_thinking() -> LLMThinkingConfig:
    """Return the default thinking config for route diagnostics."""

    thinking = LLMThinkingConfig(enabled=False)
    return thinking
