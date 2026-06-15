"""Startup reporting for configured LLM routes."""

from __future__ import annotations

from collections.abc import Iterable

from kazusa_ai_chatbot import config as cfg
from kazusa_ai_chatbot.llm_interface import LLMCallConfig, LLMThinkingConfig
from kazusa_ai_chatbot.llm_interface.diagnostics import (
    RouteDiagnostic,
    build_route_diagnostics,
)

_REQUIRED_ROUTES = frozenset((
    "RELEVANCE_AGENT_LLM",
    "VISION_DESCRIPTOR_LLM",
    "MSG_DECONTEXTUALIZER_LLM",
    "RAG_PLANNER_LLM",
    "RAG_SUBAGENT_LLM",
    "WEB_SEARCH_LLM",
    "COGNITION_LLM",
    "BOUNDARY_CORE_LLM",
    "DIALOG_GENERATOR_LLM",
    "DIALOG_EVALUATOR_LLM",
    "CONSOLIDATION_LLM",
    "JSON_REPAIR_LLM",
))
_FALLBACK_BACKED_ROUTES = frozenset((
    "BACKGROUND_ARTIFACT_LLM",
    "BACKGROUND_WORK_LLM",
))


def _route_config(route_name: str) -> LLMCallConfig:
    """Build one sanitized diagnostic config from public route constants."""

    return LLMCallConfig(
        stage_name="llm_interface.route_report",
        route_name=route_name,
        base_url=getattr(cfg, f"{route_name}_BASE_URL"),
        api_key=getattr(cfg, f"{route_name}_API_KEY"),
        model=getattr(cfg, f"{route_name}_MODEL"),
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        max_completion_tokens=getattr(
            cfg,
            f"{route_name}_MAX_COMPLETION_TOKENS",
        ),
        presence_penalty=None,
        thinking=LLMThinkingConfig(
            enabled=getattr(cfg, f"{route_name}_THINKING_ENABLED"),
        ),
    )


def _configured_chat_routes() -> tuple[LLMCallConfig, ...]:
    """Return all chat routes shown in startup diagnostics."""

    route_names = (
        "RELEVANCE_AGENT_LLM",
        "VISION_DESCRIPTOR_LLM",
        "MSG_DECONTEXTUALIZER_LLM",
        "RAG_PLANNER_LLM",
        "RAG_SUBAGENT_LLM",
        "WEB_SEARCH_LLM",
        "COGNITION_LLM",
        "BOUNDARY_CORE_LLM",
        "DIALOG_GENERATOR_LLM",
        "DIALOG_EVALUATOR_LLM",
        "CONSOLIDATION_LLM",
        "JSON_REPAIR_LLM",
        "BACKGROUND_ARTIFACT_LLM",
        "BACKGROUND_WORK_LLM",
    )
    routes = tuple(_route_config(route_name) for route_name in route_names)
    return routes


def configured_route_diagnostics() -> tuple[RouteDiagnostic, ...]:
    """Return sanitized backend diagnostics for configured chat routes."""

    diagnostics = build_route_diagnostics(
        _configured_chat_routes(),
        required_routes=set(_REQUIRED_ROUTES),
        fallback_backed_routes=set(_FALLBACK_BACKED_ROUTES),
    )
    return diagnostics


def _embedding_row() -> dict[str, str]:
    """Return non-chat embedding route details for startup reporting."""

    return {
        "route_name": "EMBEDDING",
        "backend": "embedding",
        "model": cfg.EMBEDDING_MODEL,
        "normalized_base_url": cfg.EMBEDDING_BASE_URL.rstrip("/"),
        "model_family": "embedding",
        "thinking_strategy": "unsupported",
        "optional_feature": "-",
        "required": "yes",
        "fallback_backed": "no",
    }


def _text(value: bool) -> str:
    """Render booleans in a compact, stable table form."""

    if value:
        return_value = "yes"
        return return_value
    return_value = "no"
    return return_value


def _optional_feature_tags(diagnostic: RouteDiagnostic) -> str:
    """Render effective optional backend features as compact route tags."""

    tags: list[str] = []
    if diagnostic.thinking_strategy == "gemma4_enabled":
        tags.append("thinking_on")
    elif diagnostic.thinking_strategy == "ignored_unsupported_model":
        tags.append("thinking_ignored")

    if diagnostic.fallback_backed:
        tags.append("fallback_backed")

    if not tags:
        return_value = "-"
        return return_value

    return_value = " | ".join(tags)
    return return_value


def _table_rows(
    diagnostics: Iterable[RouteDiagnostic],
) -> tuple[dict[str, str], ...]:
    """Project diagnostics into render-only row dictionaries."""

    rows = [
        {
            "route_name": diagnostic.route_name,
            "backend": diagnostic.backend,
            "model": diagnostic.model,
            "normalized_base_url": diagnostic.normalized_base_url,
            "model_family": diagnostic.model_family,
            "thinking_strategy": diagnostic.thinking_strategy,
            "optional_feature": _optional_feature_tags(diagnostic),
            "required": _text(diagnostic.required),
            "fallback_backed": _text(diagnostic.fallback_backed),
        }
        for diagnostic in diagnostics
    ]
    rows.append(_embedding_row())
    return tuple(rows)


def render_llm_route_table() -> str:
    """Render sanitized LLM route diagnostics for startup logs."""

    rows = _table_rows(configured_route_diagnostics())
    columns = (
        ("route_name", "Route"),
        ("backend", "Backend"),
        ("model", "Model"),
        ("normalized_base_url", "Source"),
        ("model_family", "Family"),
        ("thinking_strategy", "Thinking"),
        ("optional_feature", "Optional Feature"),
        ("required", "Required"),
        ("fallback_backed", "Fallback"),
    )
    widths = {
        key: max(len(title), *(len(row[key]) for row in rows))
        for key, title in columns
    }
    header = "  ".join(
        f"{title:<{widths[key]}}"
        for key, title in columns
    )
    lines = ["Configured model routes:", header]
    for row in rows:
        lines.append("  ".join(
            f"{row[key]:<{widths[key]}}"
            for key, _title in columns
        ))
    table = "\n".join(lines)
    return table
