"""Brain LLM route projections for the control console."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import inspect
import re
from typing import Any

import httpx

from kazusa_ai_chatbot.llm_interface.detection import (
    detect_model_family,
    normalize_base_url,
)


MODEL_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:@+/-]{0,199}$"
MODEL_LIST_LIMIT = 200
MODEL_LIST_TIMEOUT_SECONDS = 4.0
RouteFieldName = str
ModelListTransport = Callable[[str, dict[str, str], float], dict[str, Any]]
_MODEL_LIST_TRANSPORT: ModelListTransport | httpx.AsyncBaseTransport | None = None


@dataclass(frozen=True)
class BrainModelRouteDescriptor:
    """One operator-configurable Brain chat model route."""

    route_key: str
    env_prefix: str
    label: str
    group: str
    required: bool
    fallback_backed: bool
    editable_fields: tuple[RouteFieldName, ...] = (
        "model",
        "max_completion_tokens",
        "thinking_enabled",
    )


_ROUTES: tuple[BrainModelRouteDescriptor, ...] = (
    BrainModelRouteDescriptor(
        route_key="RELEVANCE_AGENT_LLM",
        env_prefix="RELEVANCE_AGENT_LLM",
        label="Relevance agent",
        group="Intake",
        required=True,
        fallback_backed=False,
    ),
    BrainModelRouteDescriptor(
        route_key="VISION_DESCRIPTOR_LLM",
        env_prefix="VISION_DESCRIPTOR_LLM",
        label="Vision descriptor",
        group="Intake",
        required=True,
        fallback_backed=False,
    ),
    BrainModelRouteDescriptor(
        route_key="MSG_DECONTEXTUALIZER_LLM",
        env_prefix="MSG_DECONTEXTUALIZER_LLM",
        label="Message decontextualizer",
        group="Intake",
        required=True,
        fallback_backed=False,
    ),
    BrainModelRouteDescriptor(
        route_key="RAG_PLANNER_LLM",
        env_prefix="RAG_PLANNER_LLM",
        label="RAG planner",
        group="Retrieval",
        required=True,
        fallback_backed=False,
    ),
    BrainModelRouteDescriptor(
        route_key="RAG_SUBAGENT_LLM",
        env_prefix="RAG_SUBAGENT_LLM",
        label="RAG subagent",
        group="Retrieval",
        required=True,
        fallback_backed=False,
    ),
    BrainModelRouteDescriptor(
        route_key="WEB_SEARCH_LLM",
        env_prefix="WEB_SEARCH_LLM",
        label="Web search",
        group="Retrieval",
        required=True,
        fallback_backed=False,
    ),
    BrainModelRouteDescriptor(
        route_key="COGNITION_LLM",
        env_prefix="COGNITION_LLM",
        label="Cognition",
        group="Reasoning",
        required=True,
        fallback_backed=False,
    ),
    BrainModelRouteDescriptor(
        route_key="BOUNDARY_CORE_LLM",
        env_prefix="BOUNDARY_CORE_LLM",
        label="Boundary core",
        group="Reasoning",
        required=True,
        fallback_backed=False,
    ),
    BrainModelRouteDescriptor(
        route_key="DIALOG_GENERATOR_LLM",
        env_prefix="DIALOG_GENERATOR_LLM",
        label="Dialog generator",
        group="Surface",
        required=True,
        fallback_backed=False,
    ),
    BrainModelRouteDescriptor(
        route_key="CONSOLIDATION_LLM",
        env_prefix="CONSOLIDATION_LLM",
        label="Consolidation",
        group="Memory",
        required=True,
        fallback_backed=False,
    ),
    BrainModelRouteDescriptor(
        route_key="JSON_REPAIR_LLM",
        env_prefix="JSON_REPAIR_LLM",
        label="JSON repair",
        group="Utility",
        required=True,
        fallback_backed=False,
    ),
    BrainModelRouteDescriptor(
        route_key="BACKGROUND_ARTIFACT_LLM",
        env_prefix="BACKGROUND_ARTIFACT_LLM",
        label="Background artifact",
        group="Background",
        required=False,
        fallback_backed=True,
    ),
    BrainModelRouteDescriptor(
        route_key="BACKGROUND_WORK_LLM",
        env_prefix="BACKGROUND_WORK_LLM",
        label="Background work",
        group="Background",
        required=False,
        fallback_backed=True,
    ),
)
_ROUTE_BY_KEY = {route.route_key: route for route in _ROUTES}
_ROUTE_BY_UI_KEY = {route.env_prefix.lower(): route for route in _ROUTES}
_MODEL_ID_RE = re.compile(MODEL_ID_PATTERN)


def route_descriptors() -> tuple[BrainModelRouteDescriptor, ...]:
    """Return the bounded Brain chat route catalog."""

    return _ROUTES


def descriptor_for_route(route_key: str) -> BrainModelRouteDescriptor:
    """Return a route descriptor or raise ``KeyError``."""

    route = _ROUTE_BY_KEY.get(route_key) or _ROUTE_BY_UI_KEY[route_key]
    return route


def route_field_key(
    route: BrainModelRouteDescriptor,
    field_name: RouteFieldName,
) -> str:
    """Return the lowercase descriptor field key for one route field."""

    key = f"{route.env_prefix.lower()}_{field_name}"
    return key


def route_env_name(
    route: BrainModelRouteDescriptor,
    field_name: RouteFieldName,
) -> str:
    """Return the environment variable backing one route field."""

    if field_name == "model":
        suffix = "MODEL"
    elif field_name == "max_completion_tokens":
        suffix = "MAX_COMPLETION_TOKENS"
    elif field_name == "thinking_enabled":
        suffix = "THINKING_ENABLED"
    elif field_name == "base_url":
        suffix = "BASE_URL"
    elif field_name == "api_key":
        suffix = "API_KEY"
    else:
        raise KeyError(f"unknown route field: {field_name}")
    return f"{route.env_prefix}_{suffix}"


def route_default_fallback_env(
    route: BrainModelRouteDescriptor,
    field_name: RouteFieldName,
) -> list[str]:
    """Return fallback environment variables that mirror Brain config rules."""

    if field_name == "max_completion_tokens":
        return ["DEFAULT_LLM_MAX_COMPLETION_TOKENS"]
    if field_name == "thinking_enabled":
        return []
    if field_name not in {"model", "base_url", "api_key"}:
        return []

    if route.env_prefix == "BACKGROUND_ARTIFACT_LLM":
        return [f"COGNITION_LLM_{_env_suffix(field_name)}"]
    if route.env_prefix == "BACKGROUND_WORK_LLM":
        return [
            f"BACKGROUND_ARTIFACT_LLM_{_env_suffix(field_name)}",
            f"COGNITION_LLM_{_env_suffix(field_name)}",
        ]
    return []


def route_default_literal(field_name: RouteFieldName) -> str:
    """Return literal defaults that are not backed by a route env var."""

    if field_name == "max_completion_tokens":
        return "8192"
    if field_name == "thinking_enabled":
        return "false"
    return ""


def route_environment_value(
    route: BrainModelRouteDescriptor,
    field_name: RouteFieldName,
    environment: Mapping[str, str],
) -> tuple[str, str]:
    """Return a route value and the source used to resolve it."""

    env_name = route_env_name(route, field_name)
    raw_value = environment.get(env_name, "").strip()
    if raw_value:
        return raw_value, env_name

    for fallback_env in route_default_fallback_env(route, field_name):
        raw_value = environment.get(fallback_env, "").strip()
        if raw_value:
            return raw_value, fallback_env

    literal = route_default_literal(field_name)
    if literal:
        return literal, "literal"
    return "", ""


def project_brain_model_routes(
    snapshot: Any,
    environment: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Project the generic Brain config snapshot into route-focused rows."""

    field_by_key = {
        field.key: field
        for field in getattr(snapshot, "fields", [])
    }
    rows: list[dict[str, Any]] = []
    for route in _ROUTES:
        default_values: dict[str, Any] = {}
        override_values: dict[str, Any] = {}
        effective_values: dict[str, Any] = {}
        sources: dict[str, str] = {}
        for field_name in route.editable_fields:
            key = route_field_key(route, field_name)
            field = field_by_key.get(key)
            if field is None:
                continue
            default_values[field_name] = field.default_value
            override_values[field_name] = field.override_value
            effective_values[field_name] = field.effective_value
            if field.override_value is not None:
                sources[field_name] = "override"
            elif field.default_source:
                sources[field_name] = field.default_source
            else:
                sources[field_name] = "default"

        base_url, base_url_source = route_environment_value(
            route,
            "base_url",
            environment,
        )
        model = str(effective_values.get("model") or "")
        thinking_enabled = bool(effective_values.get("thinking_enabled"))
        rows.append({
            "route_key": route.env_prefix.lower(),
            "env_prefix": route.env_prefix,
            "label": route.label,
            "group": route.group,
            "required": route.required,
            "fallback_backed": route.fallback_backed,
            "editable_fields": list(route.editable_fields),
            "default": default_values,
            "override": override_values,
            "effective": {
                **effective_values,
                "source": _route_source(sources),
            },
            "sources": sources,
            "diagnostics": {
                "backend_kind": "openai_compatible",
                "base_url_label": _base_url_label(base_url),
                "base_url_source": base_url_source,
                "model_family": _model_family(model),
                "thinking_strategy": (
                    "enabled" if thinking_enabled else "disabled"
                ),
            },
            "available_models": {"status": "not_loaded", "count": 0},
        })
    return rows


async def fetch_available_models(
    base_url: str,
    api_key: str,
    *,
    transport: ModelListTransport | httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Fetch and sanitize provider model IDs for one route."""

    if not base_url.strip():
        return _unavailable_model_list("Route base URL is not configured.")
    if not api_key.strip():
        return _unavailable_model_list("Route API key is not configured.")

    headers = {"authorization": f"Bearer {api_key}"}
    try:
        selected_transport = transport or _MODEL_LIST_TRANSPORT
        if selected_transport is not None:
            if isinstance(selected_transport, httpx.AsyncBaseTransport):
                payload = await _httpx_model_list_transport(
                    base_url,
                    headers,
                    MODEL_LIST_TIMEOUT_SECONDS,
                    transport=selected_transport,
                )
            else:
                payload_result = selected_transport(
                    base_url,
                    headers,
                    MODEL_LIST_TIMEOUT_SECONDS,
                )
                if inspect.isawaitable(payload_result):
                    payload = await payload_result
                else:
                    payload = payload_result
        else:
            payload = await _httpx_model_list_transport(
                base_url,
                headers,
                MODEL_LIST_TIMEOUT_SECONDS,
            )
    except (httpx.HTTPError, ValueError, TypeError):
        return _unavailable_model_list("Provider model list unavailable.")

    models = _sanitize_model_payload(payload)
    if not models:
        return {
            "status": "empty",
            "models": [],
            "message": "Provider returned no valid model ids.",
        }

    return {
        "status": "available",
        "models": models,
        "message": None,
    }


def _env_suffix(field_name: RouteFieldName) -> str:
    """Return the uppercase environment suffix for a route field."""

    if field_name == "model":
        return "MODEL"
    if field_name == "base_url":
        return "BASE_URL"
    if field_name == "api_key":
        return "API_KEY"
    if field_name == "max_completion_tokens":
        return "MAX_COMPLETION_TOKENS"
    if field_name == "thinking_enabled":
        return "THINKING_ENABLED"
    raise KeyError(f"unknown route field: {field_name}")


def _route_source(sources: Mapping[str, str]) -> str:
    """Return a compact source label for a route row."""

    if "override" in sources.values():
        return "override"
    return "default"


def _base_url_label(base_url: str) -> str:
    """Return a redacted provider origin label."""

    normalized = normalize_base_url(base_url)
    if not normalized:
        return "not configured"
    return normalized


async def _httpx_model_list_transport(
    base_url: str,
    headers: dict[str, str],
    timeout_seconds: float,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Default model-list HTTP transport."""

    normalized_base_url = normalize_base_url(base_url)
    if not normalized_base_url:
        raise ValueError("base URL is empty")
    url = f"{normalized_base_url.rstrip('/')}/models"
    async with httpx.AsyncClient(
        transport=transport,
        timeout=timeout_seconds,
    ) as client:
        response = await client.get(url, headers=headers)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("model list payload must be an object")
    return payload


def _sanitize_model_payload(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return bounded model metadata from an OpenAI-compatible model payload."""

    raw_items = payload.get("data", [])
    if not isinstance(raw_items, list):
        return []
    seen_ids: set[str] = set()
    models: list[dict[str, str]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str):
            continue
        normalized_id = model_id.strip()
        if not _MODEL_ID_RE.fullmatch(normalized_id):
            continue
        if normalized_id in seen_ids:
            continue
        seen_ids.add(normalized_id)
        models.append({
            "id": normalized_id,
            "family": _model_family(normalized_id),
        })
        if len(models) >= MODEL_LIST_LIMIT:
            break
    models.sort(key=lambda model: model["id"].lower())
    return models


def _unavailable_model_list(message: str) -> dict[str, Any]:
    """Return a redacted unavailable model-list response."""

    return {
        "status": "unavailable",
        "models": [],
        "message": message,
    }


def _model_family(model: str) -> str:
    """Return only the browser-facing model family label."""

    family = detect_model_family(model)
    if isinstance(family, tuple):
        return str(family[0])
    return str(family)
