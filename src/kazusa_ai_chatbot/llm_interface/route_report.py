"""Startup reporting for configured LLM routes."""

from __future__ import annotations

from collections.abc import Iterable

from kazusa_ai_chatbot import config as cfg
from kazusa_ai_chatbot.llm_interface import LLMCallConfig, LLMThinkingConfig
from kazusa_ai_chatbot.llm_interface.diagnostics import (
    RouteDiagnostic,
    build_route_diagnostics,
)

_ROUTES_BEFORE_COGNITION_CORE = (
    "RELEVANCE_AGENT_LLM",
    "VISION_DESCRIPTOR_LLM",
    "MSG_DECONTEXTUALIZER_LLM",
    "RAG_PLANNER_LLM",
    "RAG_SUBAGENT_LLM",
    "WEB_SEARCH_LLM",
    "COGNITION_LLM",
    "COGNITION_LLM_CHARACTER_CARRYOVER",
)
_ROUTES_AFTER_COGNITION_CORE = (
    "DIALOG_GENERATOR_LLM",
    "CONSOLIDATION_LLM",
    "JSON_REPAIR_LLM",
)
_V3_COGNITION_ROUTE_NAMES = frozenset({"COGNITION_V3_CHAIN_LLM"})
_SHARED_REQUIRED_ROUTES = frozenset({
    *_ROUTES_BEFORE_COGNITION_CORE,
    *_ROUTES_AFTER_COGNITION_CORE,
})
_FALLBACK_BACKED_ROUTES = frozenset()


def _route_config(route_name: str) -> LLMCallConfig:
    """Build one sanitized diagnostic config from public route constants."""

    config = LLMCallConfig(
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
    return config


def _setting_route_config(
    route_name: str,
    setting: cfg.CognitionRouteSettingV1,
) -> LLMCallConfig:
    """Build one diagnostic config from an engine-selected route setting."""

    config = LLMCallConfig(
        stage_name="llm_interface.route_report",
        route_name=route_name,
        base_url=setting.base_url,
        api_key=setting.api_key,
        model=setting.model,
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        max_completion_tokens=setting.max_completion_tokens,
        presence_penalty=None,
        thinking=LLMThinkingConfig(enabled=setting.thinking_enabled),
        context_window_tokens=setting.context_window_tokens,
    )
    return config


def _selected_cognition_routes() -> tuple[LLMCallConfig, ...]:
    """Return the configured V3 chain binding."""

    settings_v3 = cfg.get_cognition_v3_route_settings()
    routes = (
        _setting_route_config(
            "COGNITION_V3_CHAIN_LLM",
            settings_v3.chain,
        ),
    )
    return routes


def _configured_chat_routes() -> tuple[LLMCallConfig, ...]:
    """Return shared routes plus the selected cognition core family."""

    routes = (
        *(
            _route_config(route_name)
            for route_name in _ROUTES_BEFORE_COGNITION_CORE
        ),
        *_selected_cognition_routes(),
        *(_route_config(route_name) for route_name in _ROUTES_AFTER_COGNITION_CORE),
    )
    return routes


def _required_routes() -> set[str]:
    """Return required shared routes plus the V3 chain route."""

    required_routes = set(_SHARED_REQUIRED_ROUTES)
    required_routes.add("COGNITION_V3_CHAIN_LLM")
    return required_routes


def configured_route_diagnostics() -> tuple[RouteDiagnostic, ...]:
    """Return sanitized backend diagnostics for configured chat routes."""

    required_routes = _required_routes()
    configured_routes = tuple(
        route
        for route in _configured_chat_routes()
        if route.route_name in required_routes
    )
    diagnostics = build_route_diagnostics(
        configured_routes,
        required_routes=required_routes,
        fallback_backed_routes=set(_FALLBACK_BACKED_ROUTES),
    )
    return diagnostics


def _embedding_row() -> dict[str, str]:
    """Return non-chat embedding route details for startup reporting."""

    return {
        "route_name": "EMBEDDING",
        "route_group": "embedding",
        "model": cfg.EMBEDDING_MODEL,
        "normalized_base_url": cfg.EMBEDDING_BASE_URL.rstrip("/"),
        "optional_feature": "-",
    }


def _optional_feature_tags(diagnostic: RouteDiagnostic) -> str:
    """Render effective optional backend features as compact route tags."""

    tags: list[str] = []
    if diagnostic.thinking_strategy in {"gemma4_enabled", "qwen3_enabled"}:
        tags.append("thinking_on")

    if not tags:
        return_value = "-"
        return return_value

    return_value = " | ".join(tags)
    return return_value


def _route_group(route_name: str) -> str:
    """Classify one route for selected-family operator diagnostics."""

    if route_name in _V3_COGNITION_ROUTE_NAMES:
        return_value = "reasoning"
        return return_value

    return_value = "shared_non_core"
    return return_value


def _table_rows(
    diagnostics: Iterable[RouteDiagnostic],
) -> tuple[dict[str, str], ...]:
    """Project diagnostics into render-only row dictionaries."""

    rows = [
        {
            "route_name": diagnostic.route_name,
            "route_group": _route_group(diagnostic.route_name),
            "model": diagnostic.model,
            "normalized_base_url": diagnostic.normalized_base_url,
            "optional_feature": _optional_feature_tags(diagnostic),
        }
        for diagnostic in diagnostics
    ]
    rows.append(_embedding_row())
    table_rows = tuple(rows)
    return table_rows


def render_llm_route_table() -> str:
    """Render sanitized LLM route diagnostics for startup logs."""

    rows = _table_rows(configured_route_diagnostics())
    columns = (
        ("route_name", "Route"),
        ("route_group", "Group"),
        ("model", "Model"),
        ("normalized_base_url", "Source"),
        ("optional_feature", "Optional Feature"),
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
