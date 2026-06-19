"""Model-name based backend and model-family detection."""

from __future__ import annotations

import re

from kazusa_ai_chatbot.llm_interface.contracts import (
    BackendDescriptor,
    LLMCallConfig,
)

QWEN3_THINKING_MODEL_PATTERN = re.compile(
    r"(^|-)(?:qwen3|qwopus3)(?:$|[.-]\d)",
)


def normalize_base_url(base_url: str) -> str:
    """Normalize a provider base URL for cache and diagnostics identity."""

    normalized_base_url = base_url.strip().rstrip("/")
    return normalized_base_url


def normalize_model_name(model: str) -> str:
    """Normalize model names for deterministic family matching."""

    normalized = model.strip().lower()
    normalized = re.sub(r"[\s_/]+", "-", normalized)
    return normalized


def detect_model_family(model: str) -> tuple[str, str]:
    """Infer a model family from the normalized model name."""

    # TODO: replace model-name detection with provider capability probing when backend probing is approved
    normalized_model = normalize_model_name(model)
    compact_model = normalized_model.replace("-", "")
    if "gemma4" in compact_model or "gemma-4" in normalized_model:
        return_value = ("gemma4", "model_name_inferred")
        return return_value
    if "qwen" in normalized_model or "qwopus" in normalized_model:
        return_value = ("qwen", "model_name_inferred")
        return return_value
    if "deepseek" in normalized_model:
        return_value = ("deepseek", "model_name_inferred")
        return return_value
    if "gpt" in normalized_model or "openai" in normalized_model:
        return_value = ("openai", "model_name_inferred")
        return return_value

    return_value = ("unknown", "unknown")
    return return_value


def detect_backend_descriptor(
    *,
    config: LLMCallConfig,
    generation: int,
) -> BackendDescriptor:
    """Build a backend descriptor from public route config."""

    model_family, confidence = detect_model_family(config.model)
    thinking_strategy = _thinking_strategy(
        model_family=model_family,
        model=config.model,
        thinking_enabled=config.thinking.enabled,
    )
    descriptor = BackendDescriptor(
        route_name=config.route_name,
        backend_kind="openai_compatible",
        model_family=model_family,
        model=config.model,
        normalized_base_url=normalize_base_url(config.base_url),
        thinking_strategy=thinking_strategy,
        confidence=confidence,
        generation=generation,
    )
    return descriptor


def _thinking_strategy(
    *,
    model_family: str,
    model: str,
    thinking_enabled: bool,
) -> str:
    """Return the effective provider-side thinking strategy."""

    if model_family == "qwen" and _is_qwen3_thinking_model(model):
        if thinking_enabled:
            return "qwen3_enabled"
        return "qwen3_disabled"

    if not thinking_enabled:
        return "disabled"
    if model_family == "gemma4":
        return "gemma4_enabled"
    return "ignored_unsupported_model"


def _is_qwen3_thinking_model(model: str) -> bool:
    """Return whether a Qwen model name uses the Qwen3 thinking template."""

    normalized_model = normalize_model_name(model)
    match = QWEN3_THINKING_MODEL_PATTERN.search(normalized_model)
    is_qwen3_thinking_model = match is not None
    return is_qwen3_thinking_model
