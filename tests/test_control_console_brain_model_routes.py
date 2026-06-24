"""Brain model-route configuration tests for the control console."""

from __future__ import annotations

import pytest


def _route_environment() -> dict[str, str]:
    """Return complete route defaults for deterministic model-route tests."""

    routes = [
        "RELEVANCE_AGENT_LLM",
        "VISION_DESCRIPTOR_LLM",
        "MSG_DECONTEXTUALIZER_LLM",
        "RAG_PLANNER_LLM",
        "RAG_SUBAGENT_LLM",
        "WEB_SEARCH_LLM",
        "COGNITION_LLM",
        "BOUNDARY_CORE_LLM",
        "DIALOG_GENERATOR_LLM",
        "CONSOLIDATION_LLM",
        "JSON_REPAIR_LLM",
    ]
    environment = {
        "DEFAULT_LLM_MAX_COMPLETION_TOKENS": "8192",
    }
    for route in routes:
        environment[f"{route}_BASE_URL"] = "http://localhost:1234/v1"
        environment[f"{route}_API_KEY"] = "test-key"
        environment[f"{route}_MODEL"] = f"{route.lower()}-qwen3"
    return environment


def test_route_catalog_matches_configured_chat_routes() -> None:
    """The Control Console catalog must cover every editable chat route."""

    from control_console.brain_model_routes import route_descriptors

    route_prefixes = [route.env_prefix for route in route_descriptors()]

    assert route_prefixes == [
        "RELEVANCE_AGENT_LLM",
        "VISION_DESCRIPTOR_LLM",
        "MSG_DECONTEXTUALIZER_LLM",
        "RAG_PLANNER_LLM",
        "RAG_SUBAGENT_LLM",
        "WEB_SEARCH_LLM",
        "COGNITION_LLM",
        "BOUNDARY_CORE_LLM",
        "DIALOG_GENERATOR_LLM",
        "CONSOLIDATION_LLM",
        "JSON_REPAIR_LLM",
        "BACKGROUND_ARTIFACT_LLM",
        "BACKGROUND_WORK_LLM",
    ]
    assert all(route.editable_fields == (
        "model",
        "max_completion_tokens",
        "thinking_enabled",
    ) for route in route_descriptors())


def test_brain_descriptor_projects_routes_and_environment_overlay() -> None:
    """Brain config uses descriptor fields and renders only approved env names."""

    from control_console.brain_model_routes import project_brain_model_routes
    from control_console.service_config import (
        ServiceConfigOverrideStore,
        build_default_service_config_registry,
    )

    registry = build_default_service_config_registry()
    overrides = ServiceConfigOverrideStore()
    environment = _route_environment()

    assert "brain" in registry.configurable_service_ids()
    snapshot = registry.snapshot_for_service(
        service_id="brain",
        environment=environment,
        overrides=overrides,
    )
    assert len(snapshot.fields) == 39

    routes = project_brain_model_routes(snapshot=snapshot, environment=environment)
    assert len(routes) == 13
    cognition = next(route for route in routes if route["route_key"] == "cognition_llm")
    assert cognition["effective"]["model"] == "cognition_llm-qwen3"
    assert cognition["effective"]["max_completion_tokens"] == 8192
    assert cognition["effective"]["thinking_enabled"] is False
    assert cognition["diagnostics"]["model_family"] == "qwen"
    assert cognition["diagnostics"]["base_url_label"] == "http://localhost:1234/v1"

    overrides.set_override(
        service_id="brain",
        values={
            "cognition_llm_model": "deepseek-v4-flash",
            "cognition_llm_max_completion_tokens": 4096,
            "cognition_llm_thinking_enabled": True,
        },
        registry=registry,
        environment=environment,
    )
    overlay = registry.render_environment_overlay(
        service_id="brain",
        environment=environment,
        overrides=overrides,
    )
    assert overlay == {
        "COGNITION_LLM_MODEL": "deepseek-v4-flash",
        "COGNITION_LLM_MAX_COMPLETION_TOKENS": "4096",
        "COGNITION_LLM_THINKING_ENABLED": "true",
    }
    napcat_overlay = registry.render_environment_overlay(
        service_id="adapter.napcat",
        environment={"NAPCAT_ACTIVE_GROUPS": "54369546"},
        overrides=ServiceConfigOverrideStore(),
    )
    assert napcat_overlay == {}


def test_brain_descriptor_validation_rejects_invalid_route_values() -> None:
    """Model ids, token budgets, and thinking flags are validated server-side."""

    from control_console.service_config import (
        ServiceConfigOverrideStore,
        ServiceConfigValidationError,
        build_default_service_config_registry,
    )

    registry = build_default_service_config_registry()
    overrides = ServiceConfigOverrideStore()
    environment = _route_environment()
    invalid_values = [
        {"cognition_llm_model": "../bad"},
        {"cognition_llm_model": "bad model with spaces"},
        {"cognition_llm_max_completion_tokens": 0},
        {"cognition_llm_max_completion_tokens": 65537},
        {"cognition_llm_thinking_enabled": "true"},
    ]

    for values in invalid_values:
        with pytest.raises(ServiceConfigValidationError):
            overrides.set_override(
                service_id="brain",
                values=values,
                registry=registry,
                environment=environment,
            )


@pytest.mark.asyncio
async def test_available_model_listing_is_bounded_and_redacted() -> None:
    """Available model discovery should not expose credentials or raw failures."""

    import httpx

    from control_console.brain_model_routes import fetch_available_models

    seen_authorization_headers: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_authorization_headers.append(request.headers["authorization"])
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "qwen3-32b"},
                    {"id": "deepseek-v4-flash"},
                    {"id": "qwen3-32b"},
                    {"id": "x" * 201},
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    result = await fetch_available_models(
        base_url="http://provider.local/v1",
        api_key="secret-token",
        transport=transport,
    )

    assert seen_authorization_headers == ["Bearer secret-token"]
    assert result == {
        "status": "available",
        "models": [
            {"id": "deepseek-v4-flash", "family": "deepseek"},
            {"id": "qwen3-32b", "family": "qwen"},
        ],
        "message": None,
    }

    unavailable_transport = httpx.MockTransport(
        lambda request: httpx.Response(500, text="secret-token leaked"),
    )
    unavailable = await fetch_available_models(
        base_url="http://provider.local/v1",
        api_key="secret-token",
        transport=unavailable_transport,
    )
    assert unavailable == {
        "status": "unavailable",
        "models": [],
        "message": "Provider model list unavailable.",
    }
