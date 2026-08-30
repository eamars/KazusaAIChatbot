"""Tests-first gate for DSH-only route configuration."""

from __future__ import annotations

import inspect

import pytest


def test_legacy_background_route_is_absent_while_dsh_rag_and_worker_settings_remain() -> None:
    """Configuration removes the retired route and retains live bundles."""

    try:
        from kazusa_ai_chatbot import config
    except ModuleNotFoundError as exc:
        pytest.fail(f"planned configuration owner is unavailable: {exc}")
    source = inspect.getsource(config)

    assert "AGENTIC_RESOLVER_LLM" in source
    assert "RAG_" in source
    assert "BACKGROUND_WORK_LLM" not in source
