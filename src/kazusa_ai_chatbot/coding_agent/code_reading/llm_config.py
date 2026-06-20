"""Pure helpers for resolving the effective code-reading LLM route."""

from __future__ import annotations

from collections.abc import Mapping


def resolve_coding_agent_llm_settings(
    values: Mapping[str, str],
) -> dict[str, object]:
    """Resolve optional `CODING_AGENT_LLM` settings from explicit values.

    Args:
        values: Environment-like mapping used by tests or diagnostics.

    Returns:
        Effective route settings. The route identity remains
        `CODING_AGENT_LLM`; when no coding-agent identity is configured, its
        provider fields fall back to the background-work route values.

    Raises:
        ValueError: When only part of the three coding-agent identity fields is
            provided.
    """

    coding_base_url = _optional_value(values, "CODING_AGENT_LLM_BASE_URL")
    coding_api_key = _optional_value(values, "CODING_AGENT_LLM_API_KEY")
    coding_model = _optional_value(values, "CODING_AGENT_LLM_MODEL")
    coding_identity = (coding_base_url, coding_api_key, coding_model)
    present = [item is not None for item in coding_identity]
    if any(present) and not all(present):
        raise ValueError(
            "CODING_AGENT_LLM_BASE_URL, CODING_AGENT_LLM_API_KEY, and "
            "CODING_AGENT_LLM_MODEL must be configured together"
        )

    if all(present):
        settings = {
            "route_name": "CODING_AGENT_LLM",
            "base_url": coding_base_url,
            "api_key": coding_api_key,
            "model": coding_model,
            "fallback_route_name": None,
            "uses_fallback": False,
        }
        return settings

    settings = {
        "route_name": "CODING_AGENT_LLM",
        "base_url": _required_value(values, "BACKGROUND_WORK_LLM_BASE_URL"),
        "api_key": _required_value(values, "BACKGROUND_WORK_LLM_API_KEY"),
        "model": _required_value(values, "BACKGROUND_WORK_LLM_MODEL"),
        "fallback_route_name": "BACKGROUND_WORK_LLM",
        "uses_fallback": True,
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


def _required_value(values: Mapping[str, str], key: str) -> str:
    value = _optional_value(values, key)
    if value is None:
        raise ValueError(f"{key} is required for CODING_AGENT_LLM fallback")
    return value
