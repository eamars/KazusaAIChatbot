"""Executable tests for the post-cutover LLM route report."""

from __future__ import annotations

import pytest


def _route(name: str):
    from kazusa_ai_chatbot.llm_interface import LLMCallConfig, LLMThinkingConfig

    return LLMCallConfig(
        stage_name="test.route_report",
        route_name=name,
        base_url="http://localhost/v1",
        api_key="route-test-key",
        model="qwen-test",
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        max_completion_tokens=128,
        presence_penalty=None,
        thinking=LLMThinkingConfig(enabled=False),
    )


def test_route_report_omits_decommissioned_routes_and_keeps_live_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route diagnostics execute the report builder and filter retired owners."""

    from kazusa_ai_chatbot.llm_interface import route_report

    monkeypatch.setattr(
        route_report,
        "_configured_chat_routes",
        lambda: (
            _route("COGNITION_V3_CHAIN_LLM"),
            _route("RAG_PLANNER_LLM"),
            _route("BACKGROUND_WORK_LLM"),
            _route("CODING_AGENT_PM_LLM"),
        ),
    )
    monkeypatch.setattr(
        route_report,
        "_required_routes",
        lambda: {"COGNITION_V3_CHAIN_LLM", "RAG_PLANNER_LLM"},
    )

    diagnostics = route_report.configured_route_diagnostics()
    names = {row.route_name for row in diagnostics}

    assert names == {"COGNITION_V3_CHAIN_LLM", "RAG_PLANNER_LLM"}
    assert all(row.route_name != "BACKGROUND_WORK_LLM" for row in diagnostics)
    assert all(row.route_name != "CODING_AGENT_PM_LLM" for row in diagnostics)
    rendered = route_report.render_llm_route_table()
    assert "COGNITION_V3_CHAIN_LLM" in rendered
    assert "RAG_PLANNER_LLM" in rendered
    assert "CODING_AGENT_PM_LLM" not in rendered
