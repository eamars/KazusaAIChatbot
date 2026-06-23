"""Pure helpers for resolving code-writing LLM routes."""

from __future__ import annotations

from collections.abc import Mapping


def resolve_code_writing_llm_settings(
    values: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    """Resolve required PM, programmer, and synthesis LLM settings.

    Args:
        values: Environment-like mapping used by tests or diagnostics.

    Returns:
        Effective route settings. Synthesis intentionally reuses the same
        route identity and provider fields as PM.

    Raises:
        ValueError: When any required route identity field is missing.
    """

    pm_settings = _required_route_settings(values, "CODING_AGENT_PM_LLM")
    programmer_settings = _required_route_settings(
        values,
        "CODING_AGENT_PROGRAMMER_LLM",
    )
    settings = {
        "pm": pm_settings,
        "programmer": programmer_settings,
        "synthesis": dict(pm_settings),
    }
    return settings


def _required_route_settings(
    values: Mapping[str, str],
    route_name: str,
) -> dict[str, str]:
    settings = {
        "route_name": route_name,
        "base_url": _required_value(values, route_name, "BASE_URL"),
        "api_key": _required_value(values, route_name, "API_KEY"),
        "model": _required_value(values, route_name, "MODEL"),
    }
    return settings


def _optional_value(values: Mapping[str, str], key: str) -> str | None:
    value = values.get(key)
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped


def _required_value(
    values: Mapping[str, str],
    route_name: str,
    field_name: str,
) -> str:
    key = f"{route_name}_{field_name}"
    value = _optional_value(values, key)
    if value is None:
        raise ValueError(f"{key} is required for {route_name}")
    return value
